import torch
from torch import nn
from torch import Tensor
import math


def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    # Reconstruction loss (e.g., MSE)
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]

    return recon_loss + beta * kl_loss


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
