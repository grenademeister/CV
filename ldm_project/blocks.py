import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

# TODO IMPORTANT: implement attention


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.activation = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(0.1)
        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        skip = self.skip_connection(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout(x)
        return x + skip


class MidBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.res_blocks = nn.ModuleList(
            [ResBlock(in_channels, out_channels)]
            + [ResBlock(out_channels, out_channels) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.res_blocks:
            x = block(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.res_blocks = nn.ModuleList(
            [ResBlock(in_channels, out_channels)]
            + [ResBlock(out_channels, out_channels) for _ in range(num_layers - 1)]
        )
        self.downsample = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        for block in self.res_blocks:
            x = block(x)
        res_out = x
        x = self.downsample(x)
        return x, res_out


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        use_skip: bool = True,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.use_skip = use_skip
        first_block_in_channels = out_channels * 2 if use_skip else out_channels
        self.res_blocks = nn.ModuleList(
            [ResBlock(first_block_in_channels, out_channels)]
            + [ResBlock(out_channels, out_channels) for _ in range(num_layers - 1)]
        )
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

    def forward(self, x: Tensor, out_down: Optional[Tensor] = None) -> Tensor:
        x = self.upsample(x)
        if out_down is not None and self.use_skip:
            x = torch.cat([x, out_down], dim=1)
        for block in self.res_blocks:
            x = block(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        skip = x

        x = self.norm(x)
        x = x.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)

        x, _ = self.attention(x, x, x)

        x = x.transpose(1, 2).view(B, C, H, W)
        return x + skip


if __name__ == "__main__":
    # Example usage
    x = torch.randn(1, 3, 64, 64)  # Example input tensor
    down_block = DownBlock(3, 64)
    up_block = UpBlock(128, 64)
    mid_block = MidBlock(64, 128)

    down_output, res = down_block(x)
    print(down_output.shape)  # Should be (1, 64, 32, 32) after downsampling
    print(res.shape)  # Should be (1, 64, 64, 64) for the residual connection
    mid_output = mid_block(down_output)
    print(mid_output.shape)  # Should be (1, 128, 32, 32) if downsampled correctly
    up_output = up_block(mid_output, res)
    print(up_output.shape)  # Should be (1, 64, 64, 64) after upsampling
