"""Train a BatchTopK Matryoshka SAE on extracted activations and print the
health verdict (FVE >= 85%, L0 ~= k, dead < 10%, over >= 1M tokens).

  python train_batchtopk.py --acts artifacts/real/hindi_54M_layer3
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import config as C
from sae import BatchTopKMatryoshkaSAE


def open_acts(acts_dir):
    """Open an extract_real.py memmap read-only; return (memmap, n_valid, meta)."""
    with open(os.path.join(acts_dir, "meta.json")) as f:
        meta = json.load(f)
    d = meta["d_model"]
    path = os.path.join(acts_dir, "acts.f16")
    total_rows = os.path.getsize(path) // (d * 2)
    mm = np.memmap(path, dtype=np.float16, mode="r", shape=(total_rows, d))
    return mm, meta["n"], meta


def sample_rows(mm, n_valid, idx, scale, device):
    x = np.asarray(mm[idx], dtype=np.float32)
    return torch.from_numpy(x).to(device) * scale


@torch.no_grad()
def health_check(sae, mm, n_valid, scale, device, n_tokens=1_000_000, batch=8192):
    sae.eval()
    m = min(n_tokens, n_valid)
    idx = np.sort(np.random.default_rng(0).integers(0, n_valid, size=m))
    X = sample_rows(mm, n_valid, idx, scale, device)
    mean = X.mean(0, keepdim=True)
    ss_res = ss_tot = l0_sum = 0.0
    alive = torch.zeros(sae.d_hidden, dtype=torch.bool, device=device)
    for i in range(0, X.size(0), batch):
        x = X[i:i + batch]
        z = sae.encode(x)
        x_hat = sae.decode(z)
        ss_res += F.mse_loss(x_hat, x, reduction="sum").item()
        ss_tot += ((x - mean) ** 2).sum().item()
        l0_sum += (z > 0).float().sum().item()
        alive |= (z > 0).any(0)
    return {"fve": 1.0 - ss_res / (ss_tot + 1e-8),
            "l0": l0_sum / X.size(0),
            "dead_frac": 1.0 - alive.float().mean().item(),
            "n_health": X.size(0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True, help="dir written by extract_real.py")
    ap.add_argument("--dict", type=int, default=4096)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--prefixes", type=int, nargs="+",
                    default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--aux-k", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    assert args.prefixes[-1] == args.dict, "last prefix must equal dict size"
    device = C.DEVICE
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    mm, n_valid, meta = open_acts(args.acts)
    d_model, scale = meta["d_model"], meta["scale"]
    print(f"[train] {meta['model']} layer {meta['layer']} | {n_valid:,} tokens | "
          f"d_model={d_model} | scale={scale:.4f} | device={device}")

    sae = BatchTopKMatryoshkaSAE(d_model, args.dict, args.prefixes,
                                 k=args.k, aux_k=args.aux_k).to(device)

    warm_idx = np.sort(np.random.default_rng(1).integers(0, n_valid, size=min(200_000, n_valid)))
    sae.init_b_dec(sample_rows(mm, n_valid, warm_idx, scale, device).mean(0))
    sae.normalize_decoder()

    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        decay_start = int(0.8 * args.steps)
        if step < decay_start:
            return args.lr
        frac = (step - decay_start) / max(1, args.steps - decay_start)
        return args.lr * (1.0 - frac)

    sae.train()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx = torch.randint(0, n_valid, (args.batch,), generator=gen).numpy()
        idx.sort()
        x = sample_rows(mm, n_valid, idx, scale, device)
        _, _, loss = sae(x)
        opt.zero_grad()
        loss["total"].backward()
        opt.step()
        sae.normalize_decoder()

        if step % 250 == 0 or step == args.steps - 1:
            dead_now = float((sae.steps_since_fired > sae.dead_steps).float().mean())
            tag = "  (theta warming)" if step < args.warmup else ""
            print(f"  step {step:>5}/{args.steps}  recon={loss['recon']:.4f}  "
                  f"aux={loss['aux']:.4f}  dead~{dead_now*100:.0f}%{tag}")

    h = health_check(sae, mm, n_valid, scale, device)
    fve, l0, dead = h["fve"] * 100, h["l0"], h["dead_frac"] * 100
    ok = (fve >= 85) and (dead < 10)
    print("\n" + "=" * 52)
    print(f"GATE-1 HEALTH ({h['n_health']:,} tokens):")
    print(f"  variance explained : {fve:5.1f}%   (target >= 85%)")
    print(f"  L0 (active/token)  : {l0:5.1f}     (target ~= {args.k})")
    print(f"  dead features      : {dead:5.1f}%   (target < 10%)")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
    print("=" * 52)

    out = args.out or os.path.join(
        C.SAE_CKPT_DIR, f"{meta['model']}_layer{meta['layer']}_batchtopk.pt")
    torch.save({"type": "batchtopk_matryoshka", "d_model": d_model,
                "d_hidden": args.dict, "prefixes": args.prefixes,
                "k": args.k, "aux_k": args.aux_k, "state_dict": sae.state_dict(),
                "scale": scale, "act_mean": meta["act_mean"],
                "model": meta["model"], "lang": meta["lang"],
                "layer": meta["layer"], "health": h}, out)
    print(f"[train] saved -> {out}")


if __name__ == "__main__":
    main()
