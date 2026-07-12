"""Create a random-init checkpoint matching an existing model's architecture
(the random-init floor).

  python make_random_model.py --like hindi_54M --seed 1 --out $CKPT_DIR/random_hindi.pt
"""
import argparse

import torch

import config as C
from slm import GPT, GPTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", default="hindi_54M", help="existing model whose architecture to copy")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(C.CHECKPOINTS[args.like], map_location="cpu", weights_only=False)
    a = ckpt["model_args"]
    torch.manual_seed(args.seed)
    cfg = GPTConfig(block_size=a["block_size"], vocab_size=a["vocab_size"],
                    n_layer=a["n_layer"], n_head=a["n_head"], n_embd=a["n_embd"],
                    dropout=0.0, bias=a["bias"])
    model = GPT(cfg)
    torch.save({"model": model.state_dict(), "model_args": a,
                "seed": args.seed, "random_init": True, "like": args.like}, args.out)
    n = sum(p.numel() for p in model.parameters()) - model.transformer.wpe.weight.numel()
    print(f"[make_random] {n/1e6:.1f}M params, {args.like} architecture, seed {args.seed} -> {args.out}")


if __name__ == "__main__":
    main()
