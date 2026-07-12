"""Copy a small prefix slice of an activation memmap so post-hoc SAE
evaluations survive deletion of the full memmap. Only valid as a random
sample when the source extraction was shuffled.

  python save_eval_slice.py --src $ARTIFACT_DIR/real/hindi_54M_layer3 \
      --out $ARTIFACT_DIR/eval/hindi_54M_layer3_eval2M
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir written by extract_real.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=2_000_000)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.src, "meta.json")))
    d = meta["d_model"]
    n = min(args.rows, meta["n"])
    if not meta.get("shuffle_stories"):
        print("[eval_slice] WARNING: source was NOT shuffled -- this prefix slice is a "
              "first-slice sample, not a random one.")
    os.makedirs(args.out, exist_ok=True)

    src_a = np.memmap(os.path.join(args.src, "acts.f16"), dtype=np.float16, mode="r",
                      shape=(meta["n"], d))
    dst_a = np.memmap(os.path.join(args.out, "acts.f16"), dtype=np.float16, mode="w+",
                      shape=(n, d))
    step = 500_000
    for i in range(0, n, step):
        j = min(i + step, n)
        dst_a[i:j] = src_a[i:j]
    dst_a.flush()

    src_t = np.memmap(os.path.join(args.src, "toks.i32"), dtype=np.int32, mode="r",
                      shape=(meta["n"],))
    dst_t = np.memmap(os.path.join(args.out, "toks.i32"), dtype=np.int32, mode="w+",
                      shape=(n,))
    dst_t[:] = src_t[:n]
    dst_t.flush()

    out_meta = dict(meta, n=int(n), eval_slice_of=os.path.abspath(args.src),
                    eval_slice_rows=int(n))
    json.dump(out_meta, open(os.path.join(args.out, "meta.json"), "w"))
    print(f"[eval_slice] {n:,} rows ({n * d * 2 / 1e9:.2f} GB) -> {args.out}")


if __name__ == "__main__":
    main()
