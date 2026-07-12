"""Cross-model feature matching between two SAEs sharing a tokenizer, via
per-token-id activation profiles over the shared vocabulary (greedy best
match + Hungarian assignment, permutation null, BH-FDR).

  python matching.py --hindi-acts artifacts/real/hindi_54M_layer3 \
      --marathi-acts artifacts/real/marathi_54M_layer3
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from transformers import AutoTokenizer

import config as C
from featutil import load_sae


def open_acts(acts_dir):
    with open(os.path.join(acts_dir, "meta.json")) as f:
        meta = json.load(f)
    d = meta["d_model"]
    ap = os.path.join(acts_dir, "acts.f16")
    rows = os.path.getsize(ap) // (d * 2)
    acts = np.memmap(ap, dtype=np.float16, mode="r", shape=(rows, d))
    toks = np.memmap(os.path.join(acts_dir, "toks.i32"), dtype=np.int32, mode="r", shape=(rows,))
    return acts, toks, meta


@torch.no_grad()
def token_profile(sae_ckpt, acts_dir, vocab_size, device, batch=16384):
    """(vocab x d_hidden) mean feature activation per token id, plus counts."""
    sae, blob = load_sae(sae_ckpt, device)
    acts, toks, meta = open_acts(acts_dir)
    n, scale, d = meta["n"], blob["scale"], sae.d_hidden
    sum_z = torch.zeros(vocab_size, d, device=device)
    cnt = torch.zeros(vocab_size, device=device)
    for i in range(0, n, batch):
        j = min(i + batch, n)
        x = torch.from_numpy(np.asarray(acts[i:j], dtype=np.float32)).to(device) * scale
        t = torch.from_numpy(np.asarray(toks[i:j], dtype=np.int64)).to(device)
        z = sae.encode(x)
        sum_z.index_add_(0, t, z)
        cnt.index_add_(0, t, torch.ones_like(t, dtype=torch.float))
    prof = sum_z / cnt.clamp(min=1).unsqueeze(1)
    return prof, cnt


def bh_reject(pvals, q=0.05):
    p = np.asarray(pvals)
    order = np.argsort(p)
    m = len(p)
    thresh_idx = -1
    for rank, idx in enumerate(order, start=1):
        if p[idx] <= rank / m * q:
            thresh_idx = rank
    reject = np.zeros(m, dtype=bool)
    if thresh_idx > 0:
        reject[order[:thresh_idx]] = True
    return reject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hindi-acts", default="artifacts/real/hindi_54M_layer3")
    ap.add_argument("--marathi-acts", default="artifacts/real/marathi_54M_layer3")
    ap.add_argument("--hindi-sae", default=None)
    ap.add_argument("--marathi-sae", default=None)
    ap.add_argument("--vocab-size", type=int, default=68096)
    ap.add_argument("--min-count", type=int, default=50,
                    help="a token must occur >= this in BOTH corpora to enter the profile")
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--label", default="hindi_marathi")
    args = ap.parse_args()

    device = C.DEVICE
    hsae = args.hindi_sae or os.path.join(C.SAE_CKPT_DIR, "hindi_54M_layer3_batchtopk.pt")
    msae = args.marathi_sae or os.path.join(C.SAE_CKPT_DIR, "marathi_54M_layer3_batchtopk.pt")

    print("[match] building token profiles ...")
    PH, cntH = token_profile(hsae, args.hindi_acts, args.vocab_size, device)
    PM, cntM = token_profile(msae, args.marathi_acts, args.vocab_size, device)

    keep = (cntH >= args.min_count) & (cntM >= args.min_count)
    T = int(keep.sum())
    PH, PM = PH[keep].T.contiguous(), PM[keep].T.contiguous()
    print(f"[match] shared tokens (>= {args.min_count} in both): {T}")

    def zscore(P):
        mu, sd = P.mean(1, keepdim=True), P.std(1, keepdim=True)
        return (P - mu) / (sd + 1e-6), (sd.squeeze(1) > 1e-8)
    ZH, aliveH = zscore(PH)
    ZM, aliveM = zscore(PM)
    ZH, ZM = ZH[aliveH], ZM[aliveM]
    nH, nM = ZH.size(0), ZM.size(0)
    print(f"[match] alive features: Hindi {nH}, Marathi {nM}")

    Cmat = (ZH @ ZM.T) / T
    obs_best, obs_arg = Cmat.max(dim=1)

    k = min(nH, nM)
    cost = (-Cmat).detach().cpu().numpy()
    ri, ci = linear_sum_assignment(cost)
    hungarian_corr = Cmat[ri, ci].detach().cpu().numpy()

    print(f"[match] permutation null ({args.perms} perms) ...")
    obs = obs_best.detach().cpu().numpy()
    null_ge = np.zeros(nH, dtype=np.int64)
    g = torch.Generator(device=device).manual_seed(0)
    for _ in range(args.perms):
        perm = torch.randperm(T, generator=g, device=device)
        Cn = (ZH @ ZM[:, perm].T) / T
        null_best = Cn.max(dim=1).values
        null_ge += (null_best >= obs_best).detach().cpu().numpy().astype(np.int64)
    pvals = (1 + null_ge) / (1 + args.perms)
    sig = bh_reject(pvals, q=args.q)

    n_sig = int(sig.sum())
    match_rate = n_sig / nH
    print("\n" + "=" * 56)
    print(f"CROSS-LINGUAL CONVERGENCE ({args.label}):")
    print(f"  alive Hindi features           : {nH}")
    print(f"  FDR-significant matches (q={args.q}) : {n_sig}  ({100*match_rate:.1f}%)")
    print(f"  median matched r (significant)  : "
          f"{np.median(obs[sig]) if n_sig else float('nan'):.3f}")
    print(f"  Hungarian one-to-one median r   : {np.median(hungarian_corr):.3f}")
    print("=" * 56)

    tok = AutoTokenizer.from_pretrained(C.TOKENIZER_NAME)
    keep_ids = torch.nonzero(keep, as_tuple=True)[0].cpu().numpy()
    aliveH_ids = torch.nonzero(aliveH, as_tuple=True)[0].cpu().numpy()
    aliveM_ids = torch.nonzero(aliveM, as_tuple=True)[0].cpu().numpy()
    order = np.argsort(-obs)
    print("\nTop matched features (Hindi feat <-> Marathi feat | r | shared top tokens):")
    for h in order[:15]:
        if not sig[h]:
            continue
        m = int(obs_arg[h])
        both = (ZH[h] + ZM[m]).topk(4).indices.cpu().numpy()
        toks_str = " ".join(repr(tok.decode([int(keep_ids[b])]).strip()) for b in both)
        print(f"  H{aliveH_ids[h]:>4} <-> M{aliveM_ids[m]:>4} | r={obs[h]:.2f} | {toks_str}")

    out = os.path.join(C.SAE_CKPT_DIR, f"matching_{args.label}.json")
    with open(out, "w") as f:
        json.dump({"n_alive_hindi": nH, "n_alive_marathi": nM, "shared_tokens": T,
                   "n_fdr_significant": n_sig, "match_rate": match_rate,
                   "hungarian_median_r": float(np.median(hungarian_corr)),
                   "perms": args.perms, "min_count": args.min_count}, f, indent=2)
    print(f"\n[match] saved -> {out}")


if __name__ == "__main__":
    main()
