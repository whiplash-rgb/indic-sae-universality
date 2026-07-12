"""Minimal nanoGPT SLM: checkpoint loading, generation, residual-stream capture."""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, last_only=False):
        x = self._embed_and_blocks(idx)
        x = self.transformer.ln_f(x)
        if last_only:
            return self.lm_head(x[:, [-1], :])
        return self.lm_head(x)

    def _embed_and_blocks(self, idx, stop_after_layer=None):
        device = idx.device
        t = idx.size(1)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for i, block in enumerate(self.transformer.h):
            x = block(x)
            if stop_after_layer is not None and i == stop_after_layer:
                return x
        return x

    @torch.no_grad()
    def residual_at_layer(self, idx, layer):
        """Residual stream after block `layer`, shape (B, T, d_model)."""
        return self._embed_and_blocks(idx, stop_after_layer=layer)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits = self(idx_cond, last_only=True)[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def load_slm(ckpt_path, device="cpu"):
    """Load a checkpoint into a GPT. Returns (model, cfg)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    a = ckpt["model_args"]
    cfg = GPTConfig(block_size=a["block_size"], vocab_size=a["vocab_size"],
                    n_layer=a["n_layer"], n_head=a["n_head"], n_embd=a["n_embd"],
                    dropout=0.0, bias=a["bias"])
    model = GPT(cfg)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) - model.transformer.wpe.weight.numel()
    print(f"[load_slm] {ckpt_path.split('/')[-1]}: {n_params/1e6:.1f}M params, "
          f"n_layer={cfg.n_layer}, d_model={cfg.n_embd}, vocab={cfg.vocab_size}")
    return model, cfg


SEED_PROMPTS = {
    "hindi": [
        "एक समय की बात है, एक छोटे से गाँव में",
        "बहुत समय पहले एक राजा था जो",
        "एक दिन एक नन्हा खरगोश जंगल में",
        "छोटी सी लड़की अपनी माँ के साथ",
    ],
    "marathi": [
        "एका गावात एक लहान मुलगा राहत होता",
        "फार पूर्वी एक राजा होता जो",
        "एके दिवशी एक ससा जंगलात",
        "एका लहान मुलीला तिच्या आईने",
    ],
}


@torch.no_grad()
def generate_corpus(model, tokenizer, lang, device,
                    num_sequences=96, gen_length=256, temperature=1.0, top_k=50):
    """Sample an on-distribution token corpus from the SLM.
    Returns LongTensor (num_sequences, seed_len + gen_length)."""
    seeds = SEED_PROMPTS[lang]
    per_seed = max(1, num_sequences // len(seeds))
    all_seqs = []
    for seed in seeds:
        ids = tokenizer(seed, return_tensors="pt")["input_ids"].to(device)
        ids = ids.repeat(per_seed, 1)
        out = model.generate(ids, max_new_tokens=gen_length,
                             temperature=temperature, top_k=top_k)
        all_seqs.append(out.cpu())
    maxlen = max(s.size(1) for s in all_seqs)
    padded = [F.pad(s, (0, maxlen - s.size(1)), value=0) for s in all_seqs]
    corpus = torch.cat(padded, dim=0)
    print(f"[generate_corpus] {corpus.size(0)} sequences x {corpus.size(1)} tokens "
          f"= {corpus.numel():,} tokens ({lang})")
    return corpus


@torch.no_grad()
def collect_residual_activations(model, corpus, layer, device, batch_size=16):
    """Residual stream after block `layer` at every position; (N, d_model)."""
    chunks = []
    for i in range(0, corpus.size(0), batch_size):
        batch = corpus[i:i + batch_size].to(device)
        a = model.residual_at_layer(batch, layer)
        chunks.append(a.reshape(-1, a.size(-1)).cpu())
        del a, batch
        if device == "mps":
            torch.mps.empty_cache()

    acts = torch.cat(chunks, dim=0)
    print(f"[collect] layer {layer}: {acts.shape[0]:,} activation vectors "
          f"of dim {acts.shape[1]}")
    return acts
