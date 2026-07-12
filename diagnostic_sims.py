"""Zero-signal diagnostic simulations for Appendix A: (1) occurrence-frequency
leakage into cross-model RSA; (2) sparse-profile inflation of matching
statistics. Pure numpy/scipy; CPU, ~1 min.

  python diagnostic_sims.py
"""
import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment


def rsa(sig_a, sig_b):
    def sim(s):
        s = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-8)
        return s @ s.T
    iu = np.triu_indices(sig_a.shape[0], k=1)
    return spearmanr(sim(sig_a)[iu], sim(sig_b)[iu]).correlation


def make_signatures(counts, d, rng):
    sigs = np.empty((len(counts), d))
    for i, n in enumerate(counts):
        sigs[i] = np.maximum(rng.standard_normal((int(n), d)), 0.0).mean(0)
    return sigs


def sim1_occurrence_leakage(n_concepts=150, d=512, seeds=5):
    print("=" * 68)
    print("SIM 1 -- occurrence-frequency leakage (zero real signal)")
    print(f"  {n_concepts} concepts, d={d}, mean over n_c ReLU-Gaussian vectors,")
    print("  count profile SHARED across models (as in template-matched corpora)")
    results = {"unequalized": [], "partial (60<=n<100 kept raw)": [], "fully equalized (60)": []}
    for s in range(seeds):
        rng = np.random.default_rng(s)
        raw = np.clip(np.exp(rng.normal(5.5, 1.0, n_concepts)).astype(int), 60, 2000)
        a = make_signatures(raw, d, np.random.default_rng(1000 + s))
        b = make_signatures(raw, d, np.random.default_rng(2000 + s))
        results["unequalized"].append(rsa(a, b))
        part = np.minimum(raw, 100)
        a = make_signatures(part, d, np.random.default_rng(3000 + s))
        b = make_signatures(part, d, np.random.default_rng(4000 + s))
        results["partial (60<=n<100 kept raw)"].append(rsa(a, b))
        eq = np.full(n_concepts, 60)
        a = make_signatures(eq, d, np.random.default_rng(5000 + s))
        b = make_signatures(eq, d, np.random.default_rng(6000 + s))
        results["fully equalized (60)"].append(rsa(a, b))
    for k, v in results.items():
        v = np.array(v)
        print(f"  {k:32s} spurious RSA = {v.mean():+.3f} +/- {v.std():.3f}  ({seeds} seeds)")
    return results


def sim2_sparse_matching(P=1024, T=1164, k_hot=10, seeds=3):
    print("=" * 68)
    print("SIM 2 -- sparse-matching inflation (zero signal)")
    print(f"  {P} features x {T} shared token IDs; Gaussian vs {k_hot}-hot sparse profiles")
    for kind in ("gaussian", "sparse"):
        best, hung = [], []
        for s in range(seeds):
            rng = np.random.default_rng(10 * s + (0 if kind == "gaussian" else 5))
            def draw():
                if kind == "gaussian":
                    return rng.standard_normal((P, T))
                X = np.zeros((P, T))
                for i in range(P):
                    idx = rng.choice(T, k_hot, replace=False)
                    X[i, idx] = np.abs(rng.standard_normal(k_hot)) * np.sqrt(T / k_hot)
                return X
            A, B = draw(), draw()
            A = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-8)
            B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-8)
            R = (A @ B.T) / T
            best.append(float(np.median(R.max(1))))
            ri, ci = linear_sum_assignment(-R)
            hung.append(float(np.median(R[ri, ci])))
        print(f"  {kind:9s} profiles: median best-match r = {np.mean(best):.3f} +/- {np.std(best):.3f} | "
              f"Hungarian median r = {np.mean(hung):.3f} +/- {np.std(hung):.3f}  ({seeds} seeds)")


if __name__ == "__main__":
    sim1_occurrence_leakage()
    sim2_sparse_matching()
    print("=" * 68)
    print("These values correspond to the Appendix A.1/A.2 tables.")
