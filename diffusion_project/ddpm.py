# minimal DDPM implementation

import torch
import torch.nn as nn
from torch import Tensor

from typing import Tuple
import math
import itertools

from tunet import TimeUnet as Unet


def beta_scheduling_linear(timesteps: int) -> Tensor:
    """Linear beta schedule"""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def beta_scheduling(timesteps: int) -> Tensor:
    """Cosine beta schedule"""
    s = 0.008  # small offset to prevent singularities
    t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize to start at 1

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999).to(dtype=torch.float32)


class Diffusion(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.network = Unet(
            in_chans=1,
            out_chans=1,
            chans=64,
            num_pool_layers=4,
            time_emb_dim=256,
        )
        self.timesteps: int = 1000
        # Register as buffers so they move with the model
        self.register_buffer("betas", beta_scheduling(timesteps=self.timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("a_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_a_cumprod", torch.sqrt(self.a_cumprod))
        self.register_buffer("sqrt_one_minus_a_cumprod", torch.sqrt(1 - self.a_cumprod))

    def forward(self, lab: Tensor) -> tuple[Tensor, Tensor]:
        device = lab.device
        if not (lab.dim() == 4):
            raise AssertionError("Tensors must be 4D")
        batch_size = lab.shape[0]

        t = torch.randint(
            low=0,
            high=self.timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )  # sample timesteps
        noise = torch.randn_like(lab).to(device)  # sample noise

        x_t = (
            self.sqrt_a_cumprod.gather(0, t).view(batch_size, 1, 1, 1) * lab
            + self.sqrt_one_minus_a_cumprod.gather(0, t).view(batch_size, 1, 1, 1)
            * noise
        ).type(torch.float32)

        target = noise
        output = self.network(x_t, t.type(torch.float32))
        return (output, target)

    @torch.inference_mode()
    def recon(self, out_size: torch.Size, interval: float = 50):
        device = next(self.parameters()).device

        batch_size = out_size[0]

        times = torch.arange(
            0, self.timesteps, interval, device=device, dtype=torch.long
        )
        if times[-1] != self.timesteps - 1:
            times = torch.cat([times, times.new_tensor([self.timesteps - 1])])
        times = list(reversed(times.tolist()))
        time_pairs = list(itertools.pairwise(times))

        x_t = torch.randn(out_size).to(device).type(torch.float32)

        for time, time_next in time_pairs:
            print(f"\rSampling at time {time} -> {time_next}", end="")
            t_batch = torch.full((batch_size,), time, device=device, dtype=torch.long)
            t_next_batch = torch.full(
                (batch_size,), time_next, device=device, dtype=torch.long
            )

            # network inference
            predicted_noise = self.network(x_t, t_batch.type(torch.float32))

            if interval == 1:
                a_cumprod_t = self.a_cumprod.gather(0, t_batch).view(
                    batch_size, 1, 1, 1
                )
                a_cumprod_t_next = self.a_cumprod.gather(0, t_next_batch).view(
                    batch_size, 1, 1, 1
                )
                alpha_t = self.alphas.gather(0, t_batch).view(batch_size, 1, 1, 1)
                random_noise = torch.randn_like(x_t)  # Renamed to avoid collision

                mu = (
                    x_t - (1 - alpha_t) / torch.sqrt(1 - a_cumprod_t) * predicted_noise
                ) / torch.sqrt(alpha_t)
                sigma = torch.sqrt(
                    (1 - alpha_t) * ((1 - a_cumprod_t_next) / (1 - a_cumprod_t))
                )
                x_t = mu + sigma * random_noise  # Direct assignment
            else:
                # DDIM sampling (works for arbitrary step sizes)
                a_cumprod_t = self.a_cumprod.gather(0, t_batch).view(
                    batch_size, 1, 1, 1
                )
                a_cumprod_t_next = self.a_cumprod.gather(0, t_next_batch).view(
                    batch_size, 1, 1, 1
                )

                # Predict x_0 from x_t and predicted noise
                pred_x0 = (
                    x_t - torch.sqrt(1 - a_cumprod_t) * predicted_noise
                ) / torch.sqrt(a_cumprod_t)

                # Compute x_{t-interval} using DDIM formula
                x_t = (
                    torch.sqrt(a_cumprod_t_next) * pred_x0
                    + torch.sqrt(1 - a_cumprod_t_next) * predicted_noise
                )

        return x_t

    @torch.inference_mode()
    def reconstruct(self, x_t: Tensor, t: Tensor | int) -> Tensor:
        """
        Reconstruct the original image x0 from a noised image x_t at arbitrary timestep t.
        """
        device = x_t.device
        batch_size = x_t.shape[0]
        # Prepare timestep tensor
        if isinstance(t, int):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        else:
            t_batch = t.to(device).long()
            if t_batch.dim() == 0:
                t_batch = t_batch.view(1).expand(batch_size)
        # Network prediction of noise
        predicted_noise = self.network(x_t, t_batch.type(torch.float32))
        # Gather cumulative product of alphas
        a_cumprod_t = self.a_cumprod.gather(0, t_batch).view(batch_size, 1, 1, 1)
        # Compute predicted x0
        pred_x0 = (x_t - torch.sqrt(1 - a_cumprod_t) * predicted_noise) / torch.sqrt(
            a_cumprod_t
        )
        return pred_x0

    @torch.inference_mode()
    def denoise_step(self, x_t: Tensor, t: Tensor | int, dt: int = 1) -> Tensor:
        """
        Perform dt-step denoising from x_t at timestep t:
        - If dt == 1, use the DDPM posterior.
        - If dt > 1, use the DDIM formula for arbitrary step sizes.
        """
        device = x_t.device
        batch_size = x_t.shape[0]
        # Prepare timestep tensor
        if isinstance(t, int):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        else:
            t_batch = t.to(device).long()
            if t_batch.dim() == 0:
                t_batch = t_batch.view(1).expand(batch_size)
        # Determine target timestep
        t_target = (t_batch - dt).clamp(min=0)
        # Predict noise
        predicted_noise = self.network(x_t, t_batch.type(torch.float32))
        # Gather cumulative prods
        a_cumprod_t = self.a_cumprod.gather(0, t_batch).view(batch_size, 1, 1, 1)
        a_cumprod_target = self.a_cumprod.gather(0, t_target).view(batch_size, 1, 1, 1)
        if dt == 1:
            # DDPM posterior
            alpha_t = self.alphas.gather(0, t_batch).view(batch_size, 1, 1, 1)
            mu = (
                x_t - (1 - alpha_t) / torch.sqrt(1 - a_cumprod_t) * predicted_noise
            ) / torch.sqrt(alpha_t)
            sigma = torch.sqrt(
                (1 - alpha_t) * ((1 - a_cumprod_target) / (1 - a_cumprod_t))
            )
            noise = torch.randn_like(x_t)
            return mu + sigma * noise
        else:
            # DDIM step
            pred_x0 = (
                x_t - torch.sqrt(1 - a_cumprod_t) * predicted_noise
            ) / torch.sqrt(a_cumprod_t)
            return (
                torch.sqrt(a_cumprod_target) * pred_x0
                + torch.sqrt(1 - a_cumprod_target) * predicted_noise
            )
