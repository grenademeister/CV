import torch
from torch import nn


def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    # Reconstruction loss (e.g., MSE)
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]

    return recon_loss + beta * kl_loss
