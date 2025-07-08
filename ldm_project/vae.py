import torch
import torch.nn as nn
from torch import Tensor

from typing import Optional, Tuple

from blocks import MidBlock, DownBlock, UpBlock


class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        # input shape: (B, C, H, W)
        self.downblocks = nn.ModuleList(
            [
                DownBlock(128, 256),
                DownBlock(256, 256),
                DownBlock(256, 256),
            ]
        )  # shape: (B, 256, H/8, W/8)
        self.midblocks = nn.ModuleList(
            [
                MidBlock(256, 256),
            ]
        )
        self.norm_out = nn.GroupNorm(32, 256)
        self.conv_out = nn.Conv2d(256, 2 * latent_dim, kernel_size=3, padding=1)
        self.pre_quant_conv = nn.Conv2d(2 * latent_dim, 2 * latent_dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, list[Tensor]]:
        assert x.dim() == 4, "Input x must be 4D (B, C, H, W)"
        x = self.conv_in(x)

        res_outs = []
        for block in self.downblocks:
            x, res_out = block(x)
            res_outs.append(res_out)
        for block in self.midblocks:
            x = block(x)

        x = self.norm_out(x)
        x = nn.SiLU()(x)
        x = self.conv_out(x)
        x = self.pre_quant_conv(x)

        # Split into mean and log variance
        mu, log_var = torch.chunk(x, 2, dim=1)
        std = torch.exp(0.5 * log_var)
        sample = mu + std * torch.randn_like(std).to(mu.device)
        # sample, x, res_outs shapes: (B, latent_dim, H/8, W/8), (B, 2*latent_dim, H/8, W/8), list of 3 tensors
        assert (
            sample.shape[1] == mu.shape[1]
        ), "Sample and mu must have same channel size"
        return sample, x, res_outs


class Decoder(nn.Module):
    def __init__(self, out_channels: int, latent_dim: int):
        super().__init__()
        # input shape: (B, latent_dim, H/8, W/8)
        self.post_quant_conv = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)
        self.conv_in = nn.Conv2d(latent_dim, 256, kernel_size=3, padding=1)
        self.midblocks = nn.ModuleList(
            [
                MidBlock(256, 256),
            ]
        )  # shape: (B, 128, H, W)
        self.upblocks = nn.ModuleList(
            [
                UpBlock(256, 256),
                UpBlock(256, 256),
                UpBlock(256, 256),
            ]
        )  # final shape: (B, 128, H*8, W*8)
        # TODO: should fix use skip issue
        self.norm_out = nn.GroupNorm(32, 256)
        self.conv_out = nn.Conv2d(256, out_channels, kernel_size=3, padding=1)

    def forward(self, z: Tensor, res_outs: list[Tensor]) -> Tensor:
        assert isinstance(z, torch.Tensor), "Input z must be a torch.Tensor"
        assert z.dim() == 4, "Input z must be 4D (B, latent_dim, H, W)"
        z = self.post_quant_conv(z)
        z = self.conv_in(z)

        for block in self.midblocks:
            z = block(z)
        for i, block in enumerate(self.upblocks):
            z = block(z, out_down=res_outs[-i - 1] if res_outs else None)

        z = self.norm_out(z)
        z = nn.SiLU()(z)
        z = self.conv_out(z)
        return z


class VAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor, list[Tensor]]:
        return self.encoder(x)

    def decode(self, z: Tensor, res_outs: Optional[list[Tensor]]) -> Tensor:
        return self.decoder(z, res_outs)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z, latent, res_outs = self.encode(x)
        reconstructed_x = self.decode(z, res_outs)
        return reconstructed_x, latent


if __name__ == "__main__":
    # Example usage
    x = torch.randn(1, 3, 64, 64)  # Example input tensor
    vae = VAE(in_channels=3, latent_dim=128)

    reconstructed_x, latent = vae(x)
    print(reconstructed_x.shape)  # Should be (1, 3, 64, 64)
    print(latent.shape)  # Should be (1, 256, 8, 8) for the latent representation

    x_latent = torch.randn(1, 128, 8, 8)  # Example latent tensor
    reconstructed_x_from_latent = vae.decode(x_latent, None)
    print(reconstructed_x_from_latent.shape)  # Should be (1, 3, 64, 64)
