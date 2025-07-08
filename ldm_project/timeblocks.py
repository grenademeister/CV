import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class TimeResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        time_injection: str = "Add",  # Options: "FiLM", "Add"
    ):
        super().__init__()
        assert time_injection in ["FiLM", "Add"], "Invalid time injection method"
        self.time_injection = time_injection
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        skip = self.skip_connection(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.SiLU()(x)

        # Add time embedding
        if self.time_injection == "FiLM":
            time_emb = self.time_mlp(time_emb).view(-1, x.size(1), 1, 1)
            x = x * (1 + time_emb)
        elif self.time_injection == "Add":
            time_emb = self.time_mlp(time_emb).view(-1, x.size(1), 1, 1)
            x = x + time_emb

        x = self.conv2(x)
        x = self.norm2(x)

        return x + skip


class TimeMidBlock(nn.Module):
    def __init__(self, channels: int, time_emb_dim: int, num_layers: int = 2):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [TimeResBlock(channels, channels, time_emb_dim) for _ in range(num_layers)]
        )
        # TODO: implement attention block here for better performance

    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        for block in self.res_blocks:
            x = block(x, time_emb)
        return x


class TimeDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [TimeResBlock(in_channels, out_channels, time_emb_dim)]
            + [
                TimeResBlock(out_channels, out_channels, time_emb_dim)
                for _ in range(num_layers - 1)
            ]
        )
        self.downsample = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: Tensor, time_emb: Tensor) -> tuple[Tensor, Tensor]:
        for block in self.res_blocks:
            x = block(x, time_emb)
        res_out = x
        x = self.downsample(x)
        return x, res_out


class TimeUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [TimeResBlock(in_channels + out_channels, out_channels, time_emb_dim)]
            + [
                TimeResBlock(out_channels, out_channels, time_emb_dim)
                for _ in range(num_layers - 1)
            ]
        )
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

    def forward(
        self, x: Tensor, time_emb: Tensor, skip: Optional[Tensor] = None
    ) -> Tensor:
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        for block in self.res_blocks:
            x = block(x, time_emb)
        return x
