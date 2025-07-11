import torch
from torch import nn
from torch import Tensor
import math


def vae_loss_dep(x, x_recon, mu, logvar, beta=1.0):
    """deprecated version of VAE loss function."""
    # Reconstruction loss (e.g., MSE)
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]

    return recon_loss, beta * kl_loss


def vae_loss_dep2(x, x_recon, mu, logvar, beta=1.0):
    """VAE loss function."""
    B = x.size(0)
    # per-sample reconstruction error (sum, then mean over batch)
    recon = nn.functional.mse_loss(x_recon, x, reduction="sum") / B

    # per-sample KL, then mean over batch
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=[1, 2, 3]).mean()

    return recon, beta * kl


def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    # per-pixel average MSE
    recon = nn.functional.mse_loss(x_recon, x, reduction="sum") / x.shape[0]

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=[1, 2, 3]).mean()

    return recon, beta * kl


class BetaScheduler:
    def __init__(self, start_beta=0.0, end_beta=1.0, anneal_steps=10000):
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.anneal_steps = anneal_steps

    def get_beta(self, step):
        progress = min(1.0, step / self.anneal_steps)
        return 1
        # return self.start_beta + progress * (self.end_beta - self.start_beta)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb
