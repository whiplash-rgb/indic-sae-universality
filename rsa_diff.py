"""Decision statistic: is a trained pair's RSA significantly above its own
random-init floor's RSA? Paired bootstrap over the shared concept set (same
resample for trained and floor each iteration), 95% percentile CI of the
difference, optional Bonferroni-corrected CI.

  python rsa_diff.py --ta concept_sig_hindi.npz --tb concept_sig_bengali.npz \
         --fa concept_sig_random_hindi.npz --fb concept_sig_random_bengali.npz \
         --label H-B --boots 10000 --bonferroni 3

Run only on signatures produced with min-occ == fixed-occ (fully equalized).
Concepts are intersected across all four files.
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

import config as C


def _resolve(p):
    return p if os.path.dirname(p) else os.path.join(C.SAE_CKPT_DIR, p)


def load(p):
    z = np.load(_resolve(p), allow_pickle=True)
    return list(z["concept_ids"]), z["signatures"]


def sim(sig):
    s = sig / (np.linalg.norm(sig, axis=1, keepdims=True) + 1e-8)
    return s @ s.T


def rho_of(sa, sb, idx):
    Ma, Mb = sim(sa[idx]), sim(sb[idx])
    iu = np.triu_indices(len(idx), k=1)
    return spearmanr(Ma[iu], Mb[iu]).correlation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ta", required=True, help="trained model A signatures (.npz)")
    ap.add_argument("--tb", required=True, help="trained model B signatures (.npz)")
    ap.add_argument("--fa", required=True, help="floor (random) model A signatures (.npz)")
    ap.add_argument("--fb", required=True, help="floor (random) model B signatures (.npz)")
    ap.add_argument("--label", default="pair")
    ap.add_argument("--boots", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bonferroni", type=int, default=1,
                    help="number of tests in the family; reports a corrected CI "
                         "alongside the 95%% CI")
    ap.add_argument("--common-with", nargs="*", default=[],
                    help="extra signature .npz files whose concept sets further "
                         "restrict the intersection (composition control)")
    args = ap.parse_args()

    (ita, sta), (itb, stb) = load(args.ta), load(args.tb)
    (ifa, sfa), (ifb, sfb) = load(args.fa), load(args.fb)
    inter = set(ita) & set(itb) & set(ifa) & set(ifb)
    for extra in args.common_with:
        ids_e, _ = load(extra)
        inter &= set(ids_e)
    common = sorted(inter)
    n = len(common)
    if n < 10:
        raise SystemExit(f"[rsa_diff] only {n} shared concepts across the 4 files -- too few.")
    A = sta[[ita.index(c) for c in common]]
    B = stb[[itb.index(c) for c in common]]
    FA = sfa[[ifa.index(c) for c in common]]
    FB = sfb[[ifb.index(c) for c in common]]

    full = np.arange(n)
    rho_t, rho_f = rho_of(A, B, full), rho_of(FA, FB, full)
    delta = rho_t - rho_f

    rng = np.random.default_rng(args.seed)
    d = np.empty(args.boots)
    for k in range(args.boots):
        idx = rng.integers(0, n, n)
        d[k] = rho_of(A, B, idx) - rho_of(FA, FB, idx)
    lo, hi = np.percentile(d, [2.5, 97.5])
    excludes = (lo > 0) or (hi < 0)
    m = max(args.bonferroni, 1)
    alpha_c = 0.05 / m
    lo_c, hi_c = np.percentile(d, [100 * alpha_c / 2, 100 * (1 - alpha_c / 2)])
    excludes_c = (lo_c > 0) or (hi_c < 0)

    print("=" * 60)
    print(f"RSA DIFFERENCE ({args.label}):  shared concepts = {n}")
    print(f"  trained rho          = {rho_t:.3f}")
    print(f"  floor   rho          = {rho_f:.3f}")
    print(f"  delta (trained-floor)= {delta:.3f}")
    print(f"  95% bootstrap CI     = [{lo:.3f}, {hi:.3f}]  ({args.boots} paired resamples)")
    if m > 1:
        print(f"  {100*(1-alpha_c):.2f}% CI (Bonferroni m={m}) = [{lo_c:.3f}, {hi_c:.3f}]")
    print(f"  VERDICT (uncorrected): {'CI EXCLUDES 0 -> convergence exceeds the surface floor' if excludes else 'CI includes 0 -> NOT distinguishable from the surface floor'}")
    if m > 1:
        print(f"  VERDICT (corrected)  : {'corrected CI EXCLUDES 0' if excludes_c else 'corrected CI includes 0'}")
    print("=" * 60)

    out = os.path.join(C.SAE_CKPT_DIR, f"rsa_diff_{args.label}.json")
    json.dump({"label": args.label, "n_concepts": n, "rho_trained": float(rho_t),
               "rho_floor": float(rho_f), "delta": float(delta),
               "ci_lo": float(lo), "ci_hi": float(hi), "boots": args.boots,
               "ci_excludes_zero": bool(excludes),
               "bonferroni_m": m, "ci_corrected_level": 1 - alpha_c,
               "ci_corrected_lo": float(lo_c), "ci_corrected_hi": float(hi_c),
               "ci_corrected_excludes_zero": bool(excludes_c),
               "common_with": args.common_with}, open(out, "w"), indent=2)
    print(f"[rsa_diff] saved -> {out}")


if __name__ == "__main__":
    main()
