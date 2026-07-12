"""Compare the FIRST-SLICE (file-order) run against the SHUFFLED (random-
sample) run, everything else fixed. Pre-registered escalation criterion:
a floor moving > 0.03.

  python compare_runs.py --old <first-slice results dir> --new <shuffled results dir>

Reads rsa_{H-M,H-B,M-B}.json, rsa_floor_*.json, rsa_diff_*.json from each dir;
writes ordering_robustness.json into --new.
"""
import argparse
import json
import os

PAIRS = ["H-M", "H-B", "M-B"]


def grab(d, name):
    p = os.path.join(os.path.expanduser(d), name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="dir with the FIRST-SLICE rsa_*.json")
    ap.add_argument("--new", required=True, help="dir with the SHUFFLED rsa_*.json")
    ap.add_argument("--criterion", type=float, default=0.03,
                    help="pre-registered floor-movement escalation criterion")
    args = ap.parse_args()

    out = {}
    fmt = lambda a, b: (f"{a:6.3f} -> {b:6.3f}"
                        if (a is not None and b is not None) else "     (missing)")
    print(f"{'pair':6s} {'trained old->new':>22s} {'floor old->new':>22s} "
          f"{'delta old->new':>22s}")
    for pr in PAIRS:
        to, tn = grab(args.old, f"rsa_{pr}.json"), grab(args.new, f"rsa_{pr}.json")
        fo, fn = (grab(args.old, f"rsa_floor_{pr}.json"),
                  grab(args.new, f"rsa_floor_{pr}.json"))
        do, dn = (grab(args.old, f"rsa_diff_{pr}.json"),
                  grab(args.new, f"rsa_diff_{pr}.json"))
        row = {"trained_old": to and to["rho"], "trained_new": tn and tn["rho"],
               "floor_old": fo and fo["rho"], "floor_new": fn and fn["rho"],
               "delta_old": do and do["delta"], "delta_new": dn and dn["delta"],
               "ci95_new": dn and [dn["ci_lo"], dn["ci_hi"]],
               "ci_corrected_new": dn and [dn.get("ci_corrected_lo"),
                                           dn.get("ci_corrected_hi")]}
        out[pr] = row
        print(f"{pr:6s} {fmt(row['trained_old'], row['trained_new']):>22s} "
              f"{fmt(row['floor_old'], row['floor_new']):>22s} "
              f"{fmt(row['delta_old'], row['delta_new']):>22s}")

    moved = [p for p in PAIRS
             if out[p]["floor_old"] is not None and out[p]["floor_new"] is not None
             and abs(out[p]["floor_old"] - out[p]["floor_new"]) > args.criterion]
    print(f"\nfloors moving > {args.criterion} (pre-registered criterion): "
          f"{moved if moved else 'none'}")

    dst = os.path.join(os.path.expanduser(args.new), "ordering_robustness.json")
    json.dump({"criterion": args.criterion, "floors_moved": moved, "pairs": out},
              open(dst, "w"), indent=2)
    print(f"[compare_runs] saved -> {dst}")


if __name__ == "__main__":
    main()
