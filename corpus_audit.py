"""Pre-extraction sampling-representativeness audit: is the first ~25M-token
slice (file order) distributionally the same as the rest of that corpus?
Within-language FIRST vs REST via concept-frequency stability, a classifier
two-sample test (C2ST), and a topic-histogram shift; optional cross-lingual
positional-excess check. Needs only the raw datasets + a sentence embedder.

  python corpus_audit.py --langs hindi marathi bengali --hf-cache /workspace/cache
"""
import argparse
import itertools
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

import config as C
from extract_real import iter_texts, HF_DATASET, lang_of


class Reservoir:
    """Uniform random sample of size k from a stream of unknown length."""
    def __init__(self, k, rng):
        self.k, self.rng, self.buf, self.n = k, rng, [], 0

    def add(self, x):
        self.n += 1
        if len(self.buf) < self.k:
            self.buf.append(x)
        else:
            j = int(self.rng.integers(0, self.n))
            if j < self.k:
                self.buf[j] = x


def load_concept_words(lex_dir, lang):
    if lang in ("marathi", "bengali"):
        fn = {"marathi": "marathi_prompt-elements.json",
              "bengali": "bangla_prompt-elements.json"}[lang]
        d = json.load(open(os.path.join(lex_dir, fn)))
        words = []
        for cat in ("nouns", "verbs", "adjectives"):
            words += d.get(cat, [])
        return sorted(set(words))
    if lang == "hindi":
        himap = json.load(open(os.path.join(lex_dir, "hindi_lex_map.json")))
        return sorted(set(himap.values()))
    raise ValueError(lang)


def count_subseq(ids, seq):
    L = len(seq)
    if L == 0 or len(ids) < L:
        return 0
    m = ids[: len(ids) - L + 1] == seq[0]
    for j in range(1, L):
        m &= ids[j: len(ids) - L + 1 + j] == seq[j]
    return int(m.sum())


def concept_freq(texts, concept_words, tok):
    """Normalized concept-occurrence vector over the pool."""
    ids = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
    ids = np.asarray(ids, dtype=np.int64)
    seqs = [tok(w, add_special_tokens=False)["input_ids"] for w in concept_words]
    v = np.array([count_subseq(ids, s) for s in seqs], dtype=np.float64)
    return v / (v.sum() + 1e-9)


def c2st_auc(Xf, Xr):
    """Classifier two-sample test; AUC ~ 0.5 => indistinguishable."""
    X = np.vstack([Xf, Xr])
    y = np.r_[np.ones(len(Xf)), np.zeros(len(Xr))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    return float(auc.mean()), float(auc.std())


def topic_shift(Xf, Xr, k, rng, texts_first=None):
    """K-means topic histograms of FIRST vs REST: TV distance, permutation p,
    over/under-represented clusters with exemplars."""
    X = np.vstack([Xf, Xr])
    lab = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(X)
    nf = len(Xf)
    pf = np.bincount(lab[:nf], minlength=k) / nf
    pr = np.bincount(lab[nf:], minlength=k) / len(Xr)
    tv = 0.5 * np.abs(pf - pr).sum()
    y = np.r_[np.zeros(nf), np.ones(len(Xr))].astype(int)
    null = []
    for _ in range(200):
        yp = rng.permutation(y)
        a, b = lab[yp == 0], lab[yp == 1]
        null.append(0.5 * np.abs(np.bincount(a, minlength=k) / len(a)
                                 - np.bincount(b, minlength=k) / len(b)).sum())
    p = (1 + sum(n >= tv for n in null)) / (1 + len(null))
    over = np.argsort(pf - pr)[::-1][:3].tolist()
    under = np.argsort(pf - pr)[:3].tolist()
    exemplars = {}
    if texts_first is not None:
        labf = lab[:nf]
        for c in over:
            rows = np.flatnonzero(labf == c)[:2]
            exemplars[int(c)] = {
                "excess_pp": float(100 * (pf[c] - pr[c])),
                "stories": [texts_first[r][:220] for r in rows]}
    return float(tv), float(p), over, under, exemplars


def sample_first_rest(lang, target_tokens, n_sample, cache_dir, rng, tok, xling_bins=0):
    """Single pass with the exact extract_real.py boundary: FIRST until the
    cumulative token count reaches target_tokens, REST after; reservoir-sample both."""
    first, rest = Reservoir(n_sample, rng), Reservoir(n_sample, rng)
    xling = Reservoir(xling_bins * 400, rng) if xling_bins else None
    cum, M, idx = 0, None, 0
    for t in iter_texts(HF_DATASET[lang], cache_dir):
        if M is None:
            cum += len(tok(t, add_special_tokens=False)["input_ids"])
            first.add(t)
            if cum >= target_tokens:
                M = idx + 1
        else:
            rest.add(t)
        if xling is not None:
            xling.add((idx, t))
        idx += 1
    if M is None:
        M = idx
    return first.buf, rest.buf, idx, M, int(cum), (xling.buf if xling else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["hindi", "marathi", "bengali"])
    ap.add_argument("--target-tokens", type=float, default=25e6)
    ap.add_argument("--n-sample", type=int, default=15000, help="stories per group per language")
    ap.add_argument("--embedder", default="sentence-transformers/LaBSE")
    ap.add_argument("--k-topics", type=int, default=50)
    ap.add_argument("--lex-dir", default="lexicons")
    ap.add_argument("--hf-cache", default="/workspace/cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cross-lingual", action="store_true",
                    help="also run the secondary positional-excess test")
    ap.add_argument("--bins", type=int, default=20, help="position bins for --cross-lingual")
    ap.add_argument("--out", default="corpus_audit.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(C.TOKENIZER_NAME)
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(args.embedder)

    def embed(texts):
        return np.asarray(embed_model.encode(texts, batch_size=64, show_progress_bar=False,
                                             normalize_embeddings=True), dtype=np.float32)

    report, xling_store = {}, {}
    for lang in args.langs:
        print(f"\n================  {lang.upper()}  ================")
        firstT, restT, total, M, first_tokens, xl = sample_first_rest(
            lang, int(args.target_tokens), args.n_sample, args.hf_cache, rng, tok,
            xling_bins=args.bins if args.cross_lingual else 0)
        print(f"[{lang}] EXACT boundary: first-slice = first {M:,} stories "
              f"({first_tokens:,} tokens >= {int(args.target_tokens):,} target)")
        print(f"[{lang}] corpus = {total:,} stories | FIRST sample={len(firstT):,} REST sample={len(restT):,} "
              f"| first-slice is ~{100*M/max(total,1):.1f}% of corpus")

        cw = load_concept_words(args.lex_dir, lang)
        ff, fr = concept_freq(firstT, cw, tok), concept_freq(restT, cw, tok)
        rho = float(spearmanr(ff, fr).correlation)
        tv_concept = float(0.5 * np.abs(ff - fr).sum())

        Xf, Xr = embed(firstT), embed(restT)
        auc, auc_sd = c2st_auc(Xf, Xr)
        tv_topic, p_topic, over, under, exemplars = topic_shift(
            Xf, Xr, args.k_topics, rng, texts_first=firstT)
        if args.cross_lingual and xl is not None:
            xling_store[lang] = (np.array([i for i, _ in xl]), total, embed([t for _, t in xl]))

        # a-priori conservative effect-size thresholds
        passed = (auc <= 0.55) and (rho >= 0.90)
        report[lang] = {"boundary_story_M": M, "corpus_stories": total,
                        "concept_freq_spearman": rho, "concept_freq_tv": tv_concept,
                        "c2st_auc": auc, "c2st_auc_sd": auc_sd,
                        "topic_tv": tv_topic, "topic_perm_p": p_topic,
                        "topic_over_clusters": over, "topic_under_clusters": under,
                        "topic_exemplars": exemplars,
                        "verdict": "REPRESENTATIVE" if passed else "ESCALATE"}
        print(f"[{lang}] concept-freq: Spearman={rho:.3f}  TV={tv_concept:.3f}   (want rho>=0.90, TV small)")
        print(f"[{lang}] C2ST AUC   : {auc:.3f} +/- {auc_sd:.3f}                (want <=0.55; 0.5=identical)")
        print(f"[{lang}] topic TV   : {tv_topic:.3f}  perm-p={p_topic:.3f}       (want p not tiny)")
        print(f"[{lang}] VERDICT    : {report[lang]['verdict']}")
        if not passed:
            print(f"[{lang}] over-represented topics in FIRST (cluster: +excess pp, exemplar):")
            for c, info in exemplars.items():
                ex = info["stories"][0].replace("\n", " ") if info["stories"] else ""
                print(f"    #{c}: +{info['excess_pp']:.1f}pp  \"{ex[:120]}...\"")

    if args.cross_lingual and len(xling_store) >= 2:
        print("\n----------  SECONDARY: cross-lingual positional-excess  ----------")
        print("(raw cross-language similarity is expected by construction; only diagonal-offdiagonal is informative)")
        def bin_centroids(idxs, total, X, B):
            b = np.minimum((idxs * B // max(total, 1)), B - 1)
            return np.vstack([X[b == k].mean(0) if (b == k).any() else np.zeros(X.shape[1]) for k in range(B)])
        cents = {l: bin_centroids(i, t, X, args.bins) for l, (i, t, X) in xling_store.items()}
        report["cross_lingual"] = {}
        for a, bb in itertools.combinations(cents, 2):
            S = cents[a] @ cents[bb].T
            D = float(S.diagonal().mean() - S[~np.eye(args.bins, dtype=bool)].mean())
            null = []
            for _ in range(2000):
                p = rng.permutation(args.bins)
                Sp = cents[a] @ cents[bb][p].T
                null.append(Sp.diagonal().mean() - Sp[~np.eye(args.bins, dtype=bool)].mean())
            pval = (1 + sum(n >= D for n in null)) / (1 + len(null))
            report["cross_lingual"][f"{a}-{bb}"] = {"D_positional_excess": D, "perm_p": float(pval)}
            print(f"  {a}-{bb}: D(diag-offdiag) = {D:+.4f}  perm-p={pval:.3f}   (D~0 => no ordering structure beyond baseline)")

    escalate = [l for l in args.langs if report[l]["verdict"] == "ESCALATE"]
    print("\n==================  RECOMMENDATION  ==================")
    if not escalate:
        print("All languages REPRESENTATIVE (AUC<=0.55, concept-freq Spearman>=0.90).")
        print("-> Sequential first-slice extraction is unbiased; keep the first-slice protocol.")
    else:
        print(f"ESCALATE for: {escalate}. First-slice may be non-representative there.")
        print("-> Re-extract with --shuffle-stories (distinct --shuffle-seed per language),")
        print("   re-probe, recompute the floor; switch all models to the shuffled protocol")
        print("   if a floor moves > ~0.03.")
    print("(Thresholds are a-priori conservative effect-size criteria; the topic-shift")
    print(" perm-p is reported alongside as the significance companion.)")
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\n[corpus_audit] saved -> {args.out}")


if __name__ == "__main__":
    main()
