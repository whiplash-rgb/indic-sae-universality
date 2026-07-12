"""Sparse autoencoders: ReLU+L1, Matryoshka, and BatchTopK Matryoshka."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    """ReLU + L1 sparse autoencoder.

        z     = ReLU((x - b_dec) @ W_enc + b_enc)
        x_hat = z @ W_dec + b_dec
    """

    def __init__(self, d_model: int, d_hidden: int, l1_coeff: float = 4e-3):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        W_dec = F.normalize(torch.randn(d_hidden, d_model), dim=1)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, self.loss(x, x_hat, z)

    def loss(self, x, x_hat, z):
        recon = F.mse_loss(x_hat, x)
        sparsity = z.abs().sum(dim=-1).mean()
        total = recon + self.l1_coeff * sparsity
        return {"total": total, "recon": recon, "sparsity": sparsity}

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    @torch.no_grad()
    def init_b_dec(self, x_mean):
        self.b_dec.data = x_mean.clone()


class MatryoshkaSAE(SAE):
    """SAE with reconstruction loss summed over nested dictionary prefixes
    (Bussmann et al. 2025). `prefixes` is ascending; last entry == d_hidden."""

    def __init__(self, d_model, d_hidden, prefixes, l1_coeff=4e-3):
        super().__init__(d_model, d_hidden, l1_coeff)
        assert prefixes[-1] == d_hidden, "last prefix must use the full dictionary"
        self.prefixes = list(prefixes)

    def decode_prefix(self, z, m):
        return z[:, :m] @ self.W_dec[:m] + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        x_hat_full = self.decode(z)
        return x_hat_full, z, self.loss(x, z)

    def loss(self, x, z):
        recon_terms = []
        for m in self.prefixes:
            x_hat_m = self.decode_prefix(z, m)
            recon_terms.append(F.mse_loss(x_hat_m, x))
        recon = torch.stack(recon_terms).mean()
        sparsity = z.abs().sum(dim=-1).mean()
        total = recon + self.l1_coeff * sparsity
        return {"total": total, "recon": recon, "sparsity": sparsity,
                "recon_per_prefix": [t.item() for t in recon_terms]}


class BatchTopKMatryoshkaSAE(nn.Module):
    """BatchTopK Matryoshka SAE (Bussmann et al. 2025 main recipe).

    No L1: of the (B * d_hidden) pre-activations per batch, only the top
    (k * B) survive, so average L0 == k. An auxiliary loss revives dead
    features. At eval, the per-batch threshold is replaced by a fixed
    JumpReLU threshold `theta` (EMA of the smallest kept pre-activation),
    saved with the checkpoint.
    """

    def __init__(self, d_model, d_hidden, prefixes, k=32,
                 aux_k=512, aux_coeff=1.0 / 32, dead_steps=1000):
        super().__init__()
        assert prefixes[-1] == d_hidden, "last prefix must use the full dictionary"
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.prefixes = list(prefixes)
        self.k = k
        self.aux_k = aux_k
        self.aux_coeff = aux_coeff
        self.dead_steps = dead_steps

        W_dec = F.normalize(torch.randn(d_hidden, d_model), dim=1)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        self.register_buffer("steps_since_fired", torch.zeros(d_hidden))
        self.register_buffer("theta", torch.tensor(0.0))
        self.theta_ema = 0.99

    def preacts(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def batch_topk(self, z_pre):
        B = z_pre.size(0)
        n_keep = self.k * B
        flat = z_pre.flatten()
        if n_keep < flat.numel():
            thresh = torch.topk(flat, n_keep, sorted=False).values.min()
        else:
            thresh = flat.min()
        z = torch.where(z_pre >= thresh, z_pre, torch.zeros_like(z_pre))
        return z, thresh

    def encode(self, x):
        """Eval-time encode: fixed JumpReLU threshold (batch-independent)."""
        z_pre = self.preacts(x)
        return torch.where(z_pre >= self.theta, z_pre, torch.zeros_like(z_pre))

    def decode_prefix(self, z, m):
        return self.b_dec + z[:, :m] @ self.W_dec[:m]

    def decode(self, z):
        return self.b_dec + z @ self.W_dec

    def forward(self, x):
        z_pre = self.preacts(x)
        z, thresh = self.batch_topk(z_pre)

        with torch.no_grad():
            self.theta.mul_(self.theta_ema).add_((1 - self.theta_ema) * thresh)
            fired = (z > 0).any(dim=0)
            self.steps_since_fired += 1
            self.steps_since_fired[fired] = 0

        recon_terms = [F.mse_loss(self.decode_prefix(z, m), x) for m in self.prefixes]
        recon = torch.stack(recon_terms).mean()

        x_hat = self.decode(z)
        aux = self._aux_loss(x, x_hat, z_pre)

        total = recon + self.aux_coeff * aux
        return x_hat, z, {"total": total, "recon": recon, "aux": aux,
                          "recon_per_prefix": [t.item() for t in recon_terms]}

    def _aux_loss(self, x, x_hat, z_pre):
        dead = self.steps_since_fired > self.dead_steps
        if int(dead.sum()) == 0:
            return x.new_tensor(0.0)
        residual = x - x_hat.detach()
        z_dead = torch.where(dead.unsqueeze(0), z_pre, torch.zeros_like(z_pre))
        kk = min(self.aux_k, int(dead.sum()))
        topv, topi = torch.topk(z_dead, kk, dim=-1)
        z_aux = torch.zeros_like(z_pre).scatter(-1, topi, topv)
        aux_hat = z_aux @ self.W_dec
        return F.mse_loss(aux_hat, residual)

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    @torch.no_grad()
    def init_b_dec(self, x_mean):
        self.b_dec.data = x_mean.clone()


@torch.no_grad()
def evaluate(sae: SAE, x: torch.Tensor):
    """Standard SAE health metrics on a batch x (N, d_model)."""
    z = sae.encode(x)
    x_hat = sae.decode(z)
    mse = F.mse_loss(x_hat, x).item()
    var = x.var(dim=0, unbiased=False).mean().item()
    l0 = (z > 0).float().sum(dim=-1).mean().item()
    alive = (z > 0).any(dim=0)
    dead_fraction = 1.0 - alive.float().mean().item()
    return {
        "l0": l0,
        "frac_variance_explained": 1.0 - mse / (var + 1e-8),
        "dead_fraction": dead_fraction,
        "recon_mse": mse,
    }
