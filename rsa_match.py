"""RSA between two models' concept signatures: Spearman correlation of the
concept x concept cosine-similarity matrices on the concept intersection,
with a concept-label permutation null.

  python rsa_match.py --a concept_sig_marathi.npz --b concept_sig_bengali.npz --label M-B
"""
import argparse
import os

import numpy as np
from scipy.stats import spearmanr

import config as C


def _resolve(path):
    return path if os.path.dirname(path) else os.path.join(C.SAE_CKPT_DIR, path)


def sim_matrix(sig):
    s = sig / (np.linalg.norm(sig, axis=1, keepdims=True) + 1e-8)
    return s @ s.T


def upper(m):
    return m[np.triu_indices_from(m, k=1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label", default="pair")
    ap.add_argument("--perms", type=int, default=5000)
    args = ap.parse_args()

    A = np.load(_resolve(args.a), allow_pickle=True)
    B = np.load(_resolve(args.b), allow_pickle=True)
    ida, idb = list(A["concept_ids"]), list(B["concept_ids"])
    common = sorted(set(ida) & set(idb))
    if len(common) < 10:
        print(f"[rsa] only {len(common)} shared concepts — too few to test.")
        return
    sa = A["signatures"][[ida.index(c) for c in common]]
    sb = B["signatures"][[idb.index(c) for c in common]]

    Ma, Mb = sim_matrix(sa), sim_matrix(sb)
    ua = upper(Ma)
    rho = spearmanr(ua, upper(Mb)).correlation

    rng = np.random.default_rng(0)
    n = len(common)
    null = np.empty(args.perms)
    for k in range(args.perms):
        perm = rng.permutation(n)
        null[k] = spearmanr(ua, upper(Mb[np.ix_(perm, perm)])).correlation
    p = (1 + int(np.sum(null >= rho))) / (1 + args.perms)

    print("=" * 56)
    print(f"RSA ({args.label}):  shared concepts = {n}")
    print(f"  Spearman rho      = {rho:.3f}")
    print(f"  permutation-null  = {null.mean():.3f} ± {null.std():.3f}")
    print(f"  p-value           = {p:.4f}")
    print(f"  VERDICT: {'SIGNIFICANT convergence' if p < 0.05 else 'NO significant convergence'}")
    print("=" * 56)

    out = os.path.join(C.SAE_CKPT_DIR, f"rsa_{args.label}.json")
    import json
    json.dump({"label": args.label, "n_concepts": n, "rho": float(rho),
               "null_mean": float(null.mean()), "null_sd": float(null.std()),
               "p": p}, open(out, "w"), indent=2)
    print(f"[rsa] saved -> {out}")


if __name__ == "__main__":
    main()
