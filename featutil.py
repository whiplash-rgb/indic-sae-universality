"""Shared helpers for loading SAEs and interpreting features."""
import torch
import torch.nn.functional as F

from sae import SAE, MatryoshkaSAE, BatchTopKMatryoshkaSAE


def load_sae(path, device="cpu"):
    blob = torch.load(path, weights_only=False)
    if blob["type"] == "batchtopk_matryoshka":
        sae = BatchTopKMatryoshkaSAE(blob["d_model"], blob["d_hidden"],
                                     blob["prefixes"], k=blob.get("k", 32),
                                     aux_k=blob.get("aux_k", 512))
    elif blob["type"] == "matryoshka":
        sae = MatryoshkaSAE(blob["d_model"], blob["d_hidden"],
                            blob["prefixes"], l1_coeff=blob["l1_coeff"])
    else:
        sae = SAE(blob["d_model"], blob["d_hidden"], l1_coeff=blob["l1_coeff"])
    sae.load_state_dict(blob["state_dict"])
    sae.to(device).eval()
    return sae, blob


@torch.no_grad()
def feature_activations(sae, acts, scale, device, batch=8192):
    """Sparse feature activations z (N, d_hidden) for every token."""
    zs = []
    for i in range(0, acts.size(0), batch):
        x = (acts[i:i + batch] * scale).to(device)
        zs.append(sae.encode(x).cpu())
    return torch.cat(zs, dim=0)


def row_to_token(row, seq_len, corpus):
    seq, pos = row // seq_len, row % seq_len
    return int(corpus[seq, pos]), seq, pos


def top_tokens_per_feature(z, corpus, seq_len, feature_ids,
                           n_examples=24, n_tokens=6):
    """{feature_id: [(token_id, mean_activation, count), ...]}."""
    out = {}
    for f in feature_ids:
        col = z[:, f]
        if col.max() <= 0:
            out[f] = []
            continue
        top_rows = torch.topk(col, min(n_examples, (col > 0).sum().item())).indices
        counts, sums = {}, {}
        for r in top_rows.tolist():
            tid, _, _ = row_to_token(r, seq_len, corpus)
            counts[tid] = counts.get(tid, 0) + 1
            sums[tid] = sums.get(tid, 0.0) + float(col[r])
        ranked = sorted(counts, key=lambda t: counts[t], reverse=True)[:n_tokens]
        out[f] = [(t, sums[t] / counts[t], counts[t]) for t in ranked]
    return out


def feature_frequency(z):
    return (z > 0).float().mean(dim=0)


def context_string(row, seq_len, corpus, tokenizer, window=8):
    seq, pos = row // seq_len, row % seq_len
    start = max(0, pos - window + 1)
    ids = corpus[seq, start:pos + 1].tolist()
    return tokenizer.decode(ids).replace("\n", " ")
