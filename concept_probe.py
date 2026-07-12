"""Build a per-model concept-signature matrix on the shared, meaning-aligned
generation lexicon. Per concept: find occurrences of the concept's word in
this model's own corpus, mean-pool SAE feature activations over a fixed,
equalized number of token positions.

  python concept_probe.py --model hindi_54M --acts artifacts/real/hindi_54M_layer3 \
         --min-occ 60 --fixed-occ 60
  python concept_probe.py --model random_hindi --acts artifacts/real/random_hindi_layer3 \
         --tag random_hindi --min-occ 60 --fixed-occ 60

Saves SAE_CKPT_DIR/concept_sig_<tag>.npz (concept_ids, signatures, n_occ_used).
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer

import config as C
from featutil import load_sae
from extract_real import lang_of


def load_concepts(lex_dir):
    mr = json.load(open(os.path.join(lex_dir, "marathi_prompt-elements.json")))["nouns"]
    bn = json.load(open(os.path.join(lex_dir, "bangla_prompt-elements.json")))["nouns"]
    himap = json.load(open(os.path.join(lex_dir, "hindi_lex_map.json")))
    concepts = []
    for i in range(min(len(mr), len(bn))):
        words = {"marathi": mr[i], "bengali": bn[i]}
        if mr[i] in himap:
            words["hindi"] = himap[mr[i]]
        concepts.append((i, words))
    return concepts


def open_acts(acts_dir):
    meta = json.load(open(os.path.join(acts_dir, "meta.json")))
    d = meta["d_model"]
    ap = os.path.join(acts_dir, "acts.f16")
    rows = os.path.getsize(ap) // (d * 2)
    acts = np.memmap(ap, dtype=np.float16, mode="r", shape=(rows, d))
    toks = np.memmap(os.path.join(acts_dir, "toks.i32"), dtype=np.int32, mode="r", shape=(rows,))
    return acts, toks, meta


def find_occurrences(toks_np, seq, cap=5000):
    """Start positions where token subsequence `seq` occurs, and its length."""
    L = len(seq)
    first = np.flatnonzero(toks_np[:len(toks_np) - L + 1] == seq[0]) if L else np.array([], int)
    if L <= 1:
        return first[:cap], max(L, 1)
    keep = []
    for p in first:
        if np.array_equal(toks_np[p:p + L], seq):
            keep.append(p)
            if len(keep) >= cap:
                break
    return np.array(keep, dtype=np.int64), L


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--sae", default=None)
    ap.add_argument("--lex-dir", default="lexicons")
    ap.add_argument("--min-occ", type=int, default=30)
    ap.add_argument("--fixed-occ", type=int, default=None,
                    help="subsample every kept concept to exactly this many occurrences "
                         "before averaging (default: same as --min-occ); without "
                         "equalization, shared occurrence counts alone produce spurious "
                         "cross-model RSA")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--occ-seed", type=int, default=0)
    ap.add_argument("--raw", action="store_true",
                    help="raw-residual baseline: skip the SAE and mean-pool the raw "
                         "residual vectors; same .npz format; requires an explicit --tag")
    args = ap.parse_args()

    lang = lang_of(args.model)
    # Guards: refuse tag defaults that would silently overwrite the trained
    # model's signatures, and occ settings that reopen the equalization confound.
    if args.raw and args.tag is None:
        raise SystemExit(
            f"[concept_probe] --raw requires an explicit --tag (e.g. --tag {lang}_raw); "
            f"otherwise it would overwrite concept_sig_{lang}.npz.")
    if args.tag is None and not args.model.endswith("_54M"):
        raise SystemExit(
            f"[concept_probe] --model '{args.model}' is not a canonical '<lang>_54M' name, "
            f"so it needs an explicit --tag (e.g. --tag {args.model}) or it will overwrite "
            f"concept_sig_{lang}.npz. Refusing to guess.")
    tag = args.tag or lang
    fixed_occ = args.fixed_occ if args.fixed_occ is not None else args.min_occ
    if args.min_occ < fixed_occ:
        raise SystemExit(
            f"[concept_probe] --min-occ ({args.min_occ}) < --fixed-occ ({fixed_occ}): concepts "
            f"with raw count in [{args.min_occ}, {fixed_occ}) would be kept WITHOUT being "
            f"equalized, reopening the occurrence-count confound. Set --min-occ >= --fixed-occ.")
    device = C.DEVICE
    acts, toks, meta = open_acts(args.acts)
    if args.raw:
        sae, scale, d_hidden = None, meta["scale"], meta["d_model"]
    else:
        sae_path = args.sae or os.path.join(C.SAE_CKPT_DIR, f"{args.model}_layer3_batchtopk.pt")
        sae, blob = load_sae(sae_path, device)
        scale = blob["scale"]
        d_hidden = sae.d_hidden
    toks_np = np.asarray(toks[:meta["n"]])
    tok = AutoTokenizer.from_pretrained(C.TOKENIZER_NAME)
    concepts = load_concepts(args.lex_dir)

    ids, sigs, n_used = [], [], []
    rng = np.random.default_rng(args.occ_seed)
    for cid, words in concepts:
        if lang not in words:
            continue
        seq = np.array(tok(words[lang], add_special_tokens=False)["input_ids"], dtype=np.int64)
        if seq.size == 0:
            continue
        pos, L = find_occurrences(toks_np, seq)
        if len(pos) < args.min_occ:
            continue
        if len(pos) > fixed_occ:
            pos = np.sort(rng.choice(pos, size=fixed_occ, replace=False))
        rows = np.sort(np.concatenate([pos + j for j in range(L)]))
        rows = rows[rows < len(toks_np)]
        x = torch.from_numpy(np.asarray(acts[rows], dtype=np.float32)).to(device) * scale
        z = x if args.raw else sae.encode(x)
        sigs.append(z.mean(0).cpu().numpy())
        ids.append(cid)
        n_used.append(len(pos))

    ids = np.array(ids, dtype=np.int64)
    sigs = np.stack(sigs) if sigs else np.zeros((0, d_hidden), dtype=np.float32)
    out = os.path.join(C.SAE_CKPT_DIR, f"concept_sig_{tag}.npz")
    np.savez(out, concept_ids=ids, signatures=sigs.astype(np.float32),
             n_occ_used=np.array(n_used, dtype=np.int64), lang=lang, model=args.model,
             raw=args.raw)
    print(f"[concept_probe] {tag}: {len(ids)} concepts kept "
          f"(>= {args.min_occ} occ, equalized to {fixed_occ}"
          f"{', RAW residual' if args.raw else ''}) -> {out}")


if __name__ == "__main__":
    main()
