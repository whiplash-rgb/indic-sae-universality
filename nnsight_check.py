"""Cross-validate the direct residual-stream extraction path against NNsight
tracing of the same tensor (model.transformer.h[L].output). Expected max
absolute deviation: 0 (or float noise ~1e-6).

  python nnsight_check.py
  python nnsight_check.py --models hindi_54M random_hindi
"""
import argparse

import torch

import config as C
from slm import load_slm


def check_model(name, layer, batch_shape, seed, device):
    model, cfg = load_slm(C.CHECKPOINTS[name], device=device)
    model.eval()
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, cfg.vocab_size, batch_shape, generator=g).to(device)

    with torch.no_grad():
        ref = model.residual_at_layer(ids, layer)

    from nnsight import NNsight
    nn_model = NNsight(model)
    try:
        with nn_model.trace(ids, scan=False, validate=False):
            out = nn_model.transformer.h[layer].output.save()
    except TypeError:
        with nn_model.trace(ids):
            out = nn_model.transformer.h[layer].output.save()
    val = out.value if hasattr(out, "value") else out
    if isinstance(val, tuple):
        val = val[0]

    diff = (ref.float() - val.float()).abs().max().item()
    print(f"[nnsight_check] {name:16s} layer {layer}: max|ours - nnsight| = {diff:.3e} "
          f"{'OK' if diff < 1e-4 else '<-- MISMATCH, investigate'}")
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["hindi_54M", "marathi_54M", "bengali_54M"])
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = C.DEVICE
    worst = 0.0
    for m in args.models:
        worst = max(worst, check_model(m, args.layer, (args.batch, args.seq_len),
                                       args.seed, device))
    print(f"[nnsight_check] worst deviation across {len(args.models)} models: {worst:.3e}")
    print("[nnsight_check] if all OK: the direct extraction path and NNsight access "
          "read the identical layer-3 residual tensor.")


if __name__ == "__main__":
    main()
