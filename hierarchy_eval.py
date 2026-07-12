"""Matryoshka hierarchy diagnostics per SAE: nested-prefix FVE curve,
firing-rate bands, decoder-cosine geometry.

  python hierarchy_eval.py --sae $SAE_CKPT_DIR/hindi_54M_layer3_batchtopk.pt \
      --acts $ARTIFACT_DIR/real/hindi_54M_layer3 --out $SAE_CKPT_DIR/hier_hindi.json
  python hierarchy_eval.py --plot hier_hindi.json hier_marathi.json hier_bengali.json \
      --pdf figures/hierarchy.pdf
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import config as C
from featutil import load_sae
from train_batchtopk import open_acts


@torch.no_grad()
def run_eval(sae_path, acts_dir, n_tokens, batch, seed=0):
    device = C.DEVICE
    sae, blob = load_sae(sae_path, device)
    mm, n_valid, meta = open_acts(acts_dir)
    scale, prefixes, d_hidden = blob["scale"], list(blob["prefixes"]), blob["d_hidden"]

    m = min(n_tokens, n_valid)
    idx = np.sort(np.random.default_rng(seed).integers(0, n_valid, size=m))
    mean = torch.tensor(np.asarray(meta["act_mean"], dtype=np.float32) * scale,
                        device=device)
    ss_res = {p: 0.0 for p in prefixes}
    ss_tot = 0.0
    fire = torch.zeros(d_hidden, device=device)
    for i in range(0, m, batch):
        rows = idx[i:i + batch]
        x = torch.from_numpy(np.asarray(mm[rows], dtype=np.float32)).to(device) * scale
        z = sae.encode(x)
        fire += (z > 0).float().sum(0)
        ss_tot += float(((x - mean) ** 2).sum())
        for p in prefixes:
            ss_res[p] += float(((sae.decode_prefix(z, p) - x) ** 2).sum())
    freq = (fire / m).cpu().numpy()
    fve = {str(p): 1.0 - ss_res[p] / (ss_tot + 1e-8) for p in prefixes}

    bands = [(0, prefixes[0])] + [(prefixes[j - 1], prefixes[j])
                                  for j in range(1, len(prefixes))]
    band_stats = []
    for lo, hi in bands:
        f = freq[lo:hi]
        band_stats.append({"band": f"[{lo},{hi})",
                           "mean_freq": float(f.mean()),
                           "median_freq": float(np.median(f)),
                           "dead_frac": float((f == 0).mean())})

    W = F.normalize(sae.W_dec.detach(), dim=1)
    G = (W @ W.T).abs().cpu().numpy()
    nb = len(bands)
    M = np.zeros((nb, nb))
    for a, (la, ha) in enumerate(bands):
        for b, (lb, hb) in enumerate(bands):
            blk = G[la:ha, lb:hb]
            if a == b:
                iu = np.triu_indices(ha - la, k=1)
                M[a, b] = float(blk[iu].mean())
            else:
                M[a, b] = float(blk.mean())
    within = float(np.mean(np.diag(M)))
    between = float(M[~np.eye(nb, dtype=bool)].mean())

    return {"model": meta["model"], "lang": meta["lang"], "layer": meta["layer"],
            "sae": os.path.basename(sae_path), "n_tokens": int(m),
            "prefixes": prefixes, "fve_per_prefix": fve,
            "band_stats": band_stats, "cos_within_mean": within,
            "cos_between_mean": between, "cos_band_matrix": M.tolist()}


def make_plot(json_paths, pdf_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = [json.load(open(p)) for p in json_paths]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.45))

    ax = axes[0]
    for r in reports:
        ps = r["prefixes"]
        ax.plot(ps, [100 * r["fve_per_prefix"][str(p)] for p in ps],
                marker="o", label=r["lang"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(reports[0]["prefixes"])
    ax.set_xticklabels(reports[0]["prefixes"])
    ax.set_xlabel("dictionary prefix $m$")
    ax.set_ylabel("FVE (%) with first $m$ features")
    ax.set_title("(a) nested-prefix reconstruction")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for r in reports:
        bs = r["band_stats"]
        ax.plot(range(len(bs)), [max(b["mean_freq"], 1e-7) for b in bs],
                marker="o", label=r["lang"])
    ax.set_yscale("log")
    ax.set_xticks(range(len(reports[0]["band_stats"])))
    ax.set_xticklabels([b["band"] for b in reports[0]["band_stats"]],
                       rotation=30, fontsize=7)
    ax.set_xlabel("prefix band")
    ax.set_ylabel("mean firing frequency")
    ax.set_title("(b) firing rate by band")

    ax = axes[2]
    x = np.arange(len(reports))
    ax.bar(x - 0.16, [r["cos_within_mean"] for r in reports], width=0.32,
           label="within band")
    ax.bar(x + 0.16, [r["cos_between_mean"] for r in reports], width=0.32,
           label="between bands")
    ax.set_xticks(x)
    ax.set_xticklabels([r["lang"] for r in reports])
    ax.set_ylabel(r"mean $|\cos|$, decoder dirs.")
    ax.set_title("(c) decoder geometry")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.30))

    fig.tight_layout()
    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[hierarchy_eval] figure -> {pdf_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", help="SAE checkpoint (.pt)")
    ap.add_argument("--acts", help="activation memmap dir (full or eval slice)")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--tokens", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", nargs="*", default=None,
                    help="plot mode: list of hier_*.json files to combine")
    ap.add_argument("--pdf", default="figures/hierarchy.pdf")
    args = ap.parse_args()

    if args.plot:
        make_plot(args.plot, args.pdf)
        return
    if not (args.sae and args.acts):
        raise SystemExit("eval mode needs --sae and --acts (or use --plot ...)")
    rep = run_eval(args.sae, args.acts, args.tokens, args.batch, args.seed)
    out = args.out or os.path.join(C.SAE_CKPT_DIR, f"hier_{rep['lang']}.json")
    json.dump(rep, open(out, "w"), indent=2)
    print(f"[hierarchy_eval] {rep['lang']}: "
          + "  ".join(f"FVE@{p}={100*rep['fve_per_prefix'][str(p)]:.1f}%"
                      for p in rep["prefixes"]))
    print(f"[hierarchy_eval] band mean-freq: "
          + "  ".join(f"{b['band']}={b['mean_freq']:.4f}" for b in rep["band_stats"]))
    print(f"[hierarchy_eval] decoder |cos| within={rep['cos_within_mean']:.4f} "
          f"between={rep['cos_between_mean']:.4f}")
    print(f"[hierarchy_eval] saved -> {out}")


if __name__ == "__main__":
    main()
