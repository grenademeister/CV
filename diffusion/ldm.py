import torch
import torch.nn as nn
from torch import Tensor

from typing import Tuple

from ddpm import Diffusion
from vae import Autoencoder, Encoder, Decoder
from misc import ModelConfig


class LDM(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, diffusion: Diffusion):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.diffusion = diffusion

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            z = self.encoder(x)
        pred_noise, noise = self.diffusion.forward(z)
        return pred_noise, noise


vae = Autoencoder(in_channels=3, out_channels=3, latent_dim=64)
vae.load_state_dict(torch.load("vae.pth"))
encoder = vae.encoder.eval()
decoder = vae.decoder.eval()

diffusion = Diffusion(
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    config=ModelConfig(),
)
ldm = LDM(encoder=encoder, decoder=decoder, diffusion=diffusion)
