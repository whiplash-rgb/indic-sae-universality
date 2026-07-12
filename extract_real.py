"""Extract layer-L residual-stream activations over the real corpus into an
on-disk fp16 memmap, dropping each window's first position.

  python extract_real.py --model hindi_54M --layer 3 --target-tokens 25000000 \
         --shuffle-stories --shuffle-seed 101 --hf-cache /workspace/cache

Outputs -> ARTIFACT_DIR/real/<model>_layer<L>/ (acts.f16, toks.i32, meta.json)
"""
import argparse
import gzip
import hashlib
import json
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download

import config as C
from slm import load_slm


def sha256_of(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()

HF_DATASET = {
    "hindi":   "TinyStories-Regional/hindi-generated_4o-mini_2M",
    "marathi": "TinyStories-Regional/marathi-generated_4o-mini_2M",
    "bengali": "TinyStories-Regional/beng-generated_4o-mini_2M",
}


def lang_of(model):
    for L in ("hindi", "marathi", "bengali"):
        if L in model.lower():
            return L
    raise ValueError(f"cannot infer language from model name '{model}'")


def pick_text_col(keys):
    keys = list(keys)
    for c in ("text", "story", "content", "Story", "output", "generated_story"):
        if c in keys:
            return c
    return keys[0]


def iter_texts(repo, cache_dir):
    """Yield story strings from a HF dataset repo. Downloads raw files and
    parses them directly (datasets' streaming JSON reader overflows int32
    block_size on this corpus's large JSON files)."""
    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset")
    parquet = sorted(f for f in files if f.endswith(".parquet"))
    jsonls = sorted(f for f in files if f.endswith((".jsonl", ".jsonl.gz")))
    jsons = sorted(f for f in files if f.endswith((".json", ".json.gz"))
                   and "config" not in os.path.basename(f).lower())
    chosen = parquet or jsonls or jsons
    if not chosen:
        raise RuntimeError(f"no data files found in {repo}; files={files}")
    print(f"[data] {repo}: {len(chosen)} file(s) -> {chosen[:3]}"
          f"{'...' if len(chosen) > 3 else ''}")

    for fn in chosen:
        path = hf_hub_download(repo, fn, repo_type="dataset", cache_dir=cache_dir)
        if fn.endswith(".parquet"):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            tc = pick_text_col(pf.schema_arrow.names)
            for rg in range(pf.num_row_groups):
                for v in pf.read_row_group(rg, columns=[tc]).column(0).to_pylist():
                    if v:
                        yield v
        else:
            opn = gzip.open if fn.endswith(".gz") else open
            with opn(path, "rt", encoding="utf-8") as fh:
                head = fh.read(1)
                fh.seek(0)
                if head == "[":
                    for row in json.load(fh):
                        tc = pick_text_col(row.keys())
                        if row.get(tc):
                            yield row[tc]
                else:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        tc = pick_text_col(row.keys())
                        if row.get(tc):
                            yield row[tc]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hindi_54M", choices=list(C.CHECKPOINTS))
    ap.add_argument("--layer", type=int, default=C.DEFAULT_LAYER)
    ap.add_argument("--target-tokens", type=int, default=25_000_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-stories", type=int, default=0,
                    help="skip the first N stories (disjoint probe slice)")
    ap.add_argument("--shuffle-stories", action="store_true",
                    help="shuffle story order (fixed seed) before extraction; "
                         "off by default so a first-slice run stays reproducible")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--hf-cache", default=None, help="where to download raw data files")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    lang = lang_of(args.model)
    device = C.DEVICE
    torch.manual_seed(C.SEED)

    model, cfg = load_slm(C.CHECKPOINTS[args.model], device=device)
    d_model = cfg.n_embd
    seq_len = min(args.seq_len, cfg.block_size)
    tok = AutoTokenizer.from_pretrained(C.TOKENIZER_NAME)

    out_dir = args.out_dir or os.path.join(C.ARTIFACT_DIR, "real",
                                           f"{args.model}_layer{args.layer}")
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = args.hf_cache or os.path.join(C.ARTIFACT_DIR, "hf_cache")

    N = args.target_tokens
    acts_mm = np.memmap(os.path.join(out_dir, "acts.f16"), dtype=np.float16,
                        mode="w+", shape=(N, d_model))
    toks_mm = np.memmap(os.path.join(out_dir, "toks.i32"), dtype=np.int32,
                        mode="w+", shape=(N,))
    sum_ = np.zeros(d_model, dtype=np.float64)
    sumsq = np.zeros(d_model, dtype=np.float64)
    written = 0

    def flush(seq_batch):
        nonlocal written
        if not seq_batch or written >= N:
            return
        batch = torch.tensor(seq_batch, dtype=torch.long, device=device)
        a = model.residual_at_layer(batch, args.layer)
        # drop each window's first position (no in-window left context)
        a = a[:, 1:, :].reshape(-1, d_model).float().cpu().numpy()
        take = min(a.shape[0], N - written)
        a = a[:take]
        acts_mm[written:written + take] = a.astype(np.float16)
        tk = np.array(seq_batch, dtype=np.int32)[:, 1:].reshape(-1)[:take]
        toks_mm[written:written + take] = tk
        sum_[:] += a.sum(0)
        sumsq[:] += (a.astype(np.float64) ** 2).sum(0)
        written += take
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    stories = iter_texts(HF_DATASET[lang], cache_dir)
    if args.shuffle_stories:
        stories = list(stories)
        random.Random(args.shuffle_seed).shuffle(stories)
        print(f"[extract_real] shuffled {len(stories):,} stories (seed {args.shuffle_seed}) "
              f"-> random sample, not first-slice")

    token_buf, seq_batch, seen = [], [], 0
    for text in stories:
        seen += 1
        if seen <= args.skip_stories:
            continue
        token_buf.extend(tok(text, add_special_tokens=False)["input_ids"])
        while len(token_buf) >= seq_len:
            seq_batch.append(token_buf[:seq_len])
            token_buf = token_buf[seq_len:]
            if len(seq_batch) == args.batch_size:
                flush(seq_batch)
                seq_batch = []
                if written % 1_000_000 < seq_len * args.batch_size:
                    print(f"  ... {written:,}/{N:,} tokens")
        if written >= N:
            break
    flush(seq_batch)

    acts_mm.flush()
    toks_mm.flush()

    mean = sum_ / max(written, 1)
    var = np.maximum(sumsq / max(written, 1) - mean ** 2, 0.0)
    scale = float(np.sqrt(d_model / (var.sum() + mean @ mean + 1e-8)))

    meta = {"model": args.model, "lang": lang, "layer": args.layer,
            "d_model": int(d_model), "n": int(written), "seq_len": seq_len,
            "dropped_bos": True, "act_mean": mean.tolist(), "scale": scale,
            "dataset": HF_DATASET[lang], "skip_stories": args.skip_stories,
            "tokenizer": C.TOKENIZER_NAME,
            "ckpt_sha256": sha256_of(C.CHECKPOINTS[args.model]),
            "shuffle_stories": args.shuffle_stories,
            "shuffle_seed": args.shuffle_seed if args.shuffle_stories else None}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    print(f"[extract_real] wrote {written:,} activation vectors ({d_model}-dim) -> {out_dir}")
    print(f"  standardization scale = {scale:.4f}")
    if written < N:
        print(f"  NOTE: dataset exhausted before target ({written:,} < {N:,}); "
              f"meta['n'] is the true count -- loaders slice to it.")


if __name__ == "__main__":
    main()
