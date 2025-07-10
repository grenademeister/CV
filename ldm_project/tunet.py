import torch
import torch.nn as nn
from torch import Tensor

from helper import TimeEmbedding
from timeblocks import TimeDownBlock, TimeUpBlock, TimeMidBlock


class TimeUnet(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 64,
        num_pool_layers: int = 3,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.conv_in = nn.Conv2d(in_chans, chans, kernel_size=3, padding=1)

        # Build down blocks
        down_channels = [chans]
        for i in range(num_pool_layers):
            down_channels.append(chans * (2 ** (i + 1)))

        self.down_blocks = nn.ModuleList(
            [
                TimeDownBlock(down_channels[i], down_channels[i + 1], time_emb_dim, attention=True)
                for i in range(num_pool_layers)
            ]
        )

        # Middle block
        mid_channels = down_channels[-1]
        self.mid_block = TimeMidBlock(mid_channels, time_emb_dim, attention=True)

        # Build up blocks
        up_channels = list(reversed(down_channels))
        self.up_blocks = nn.ModuleList(
            [
                TimeUpBlock(up_channels[i], up_channels[i + 1], time_emb_dim, attention=True)
                for i in range(num_pool_layers)
            ]
        )

        self.norm_out = nn.GroupNorm(min(32, chans//4), chans)
        self.conv_out = nn.Conv2d(chans, out_chans, kernel_size=3, padding=1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        time_emb = self.time_embedding(t)

        x = self.conv_in(x)

        # Down path
        skip_connections = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, time_emb)
            skip_connections.append(skip)

        # Middle
        x = self.mid_block(x, time_emb)

        # Up path
        for up_block in self.up_blocks:
            skip = skip_connections.pop()
            x = up_block(x, time_emb, skip)

        x = self.norm_out(x)
        x = nn.SiLU()(x)
        x = self.conv_out(x)

        return x


if __name__ == "__main__":
    # Example usage for latent space (matching VAE encoder output)
    x = torch.randn(2, 128, 8, 8)  # Batch=2, latent_dim=128, H/8, W/8
    t = torch.randint(0, 1000, (2,)).float()  # Timesteps

    model = TimeUnet(
        in_chans=128, out_chans=128, chans=64, num_pool_layers=3, time_emb_dim=256
    )

    output = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Time embedding shape: {t.shape}")
