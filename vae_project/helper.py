import torch
from torch import nn


def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    B = x.size(0)
    # per-sample reconstruction error (sum, then mean over batch)
    recon = nn.functional.mse_loss(x_recon, x, reduction="sum") / B

    # per-sample KL, then mean over batch
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=[1, 2, 3]).mean()

    return recon, beta * kl
