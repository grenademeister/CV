# longitudinal reconstruction demo

import torch
from torch import Tensor
import torch.nn.functional as F

from scipy.io import loadmat
import matplotlib.pyplot as plt

from ddpm import Diffusion

DEVICE = "cuda"


def load_model(debug=False):
    diff = Diffusion(device=DEVICE)
    if debug:
        print("Loading model...")
    state_dict = torch.load("./var/checkpoints/epoch_44.pth")
    # Remove 'module.' prefix if present
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    if debug:
        print("Loading model state dict...")
    diff.load_state_dict(new_state_dict)
    print("Model loaded successfully.")
    diff.to(DEVICE)
    if debug:
        print("Model moved to CUDA.")
    return diff


def load_data(visualize=False, debug=False):

    mat_data = loadmat("example.mat")
    mat_data_other = loadmat("example.mat")
    print("Loaded .mat file successfully.")
    if debug:
        print("Keys in .mat file:", mat_data.keys())
    data_prior = mat_data["img1_reg"][0]
    data = mat_data_other["img2"][0]
    if debug:
        print("Data shape:", data.shape)

    if visualize:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(data_prior, cmap="gray")
        plt.title("Prior Data")
        plt.subplot(1, 2, 2)
        plt.imshow(data, cmap="gray")
        plt.title("Data")
        plt.tight_layout()
        plt.savefig("./var/samples/data_visualization.png")

    lab_prior = (
        torch.tensor(data_prior, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )
    lab = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return lab_prior, lab


def denoising_showcase(
    lab_prior: Tensor, lab: Tensor, diff: Diffusion, visualize=False
):
    # Add noise to lab at t=10 and reconstruct back the data using the model
    t_val = 500
    t_recon = torch.tensor([t_val], device=DEVICE)
    noise_recon = torch.randn_like(lab_prior).to(DEVICE)
    x_t_recon = (
        diff.sqrt_a_cumprod.gather(0, t_recon).view(1, 1, 1, 1) * lab_prior
        + diff.sqrt_one_minus_a_cumprod.gather(0, t_recon).view(1, 1, 1, 1)
        * noise_recon
    ).type(torch.float32)

    # Reconstruct (denoise) using the model
    output_recon = diff.reconstruct(x_t_recon, t_recon.type(torch.float32))

    # Optionally, plot the noisy and reconstructed images
    if visualize:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(x_t_recon[0, 0].cpu().numpy(), cmap="gray")
        plt.title(f"Noisy Image (t={t_val})")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(output_recon[0, 0].detach().cpu().numpy(), cmap="gray")
        plt.title("Reconstructed Noise")
        plt.axis("off")
        plt.savefig("./var/samples/dc_reconstruction.png")

    loss = F.mse_loss(output_recon, lab_prior).item()
    print(f"Reconstruction loss: {loss:.4f}")


def A(x):
    """ "Forward operator A: FFT along spatial dimensions."""
    return torch.fft.fftn(x, dim=[2, 3])


def generate_mask(shape, R, center=0.1):
    """
    Generates an undersampling mask for MRI reconstruction.
    Args:
        shape (tuple): Shape of the mask (H, W).
        R (int): Undersampling rate (sample every R-th row).
        center (float, optional): Fraction of center region to fully sample. Default is 0.1.
    Returns:
        Tensor: Boolean mask tensor of shape [H, W].
    """
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[::R, ::] = True  # sample every R-th row

    cutoff = max(int(center * shape[0] / 2), 1)
    # print(f"Cutoff: {cutoff}")
    mask[:cutoff, :] = True  # center region
    mask[-cutoff:, :] = True  # center region
    return mask


def denoising_dc_showcase(
    lab_prior: Tensor,
    lab: Tensor,
    diff: Diffusion,
    t_val: int = 500,
    dt: int = 10,
    R: int = 2,
    center: float = 0.1,
    n_steps: int = 2,
    visualize=False,
):
    t_recon = torch.tensor([t_val], device=DEVICE)
    noise_recon = torch.randn_like(lab_prior).to(DEVICE)
    x_t_recon = (
        diff.sqrt_a_cumprod.gather(0, t_recon).view(1, 1, 1, 1) * lab_prior
        + diff.sqrt_one_minus_a_cumprod.gather(0, t_recon).view(1, 1, 1, 1)
        * noise_recon
    ).type(torch.float32)

    output_recon = x_t_recon

    y = A(x=lab).to(DEVICE)
    mask = generate_mask((512, 512), R=R, center=center).to(DEVICE)

    diff.network.eval()  # Ensure the model is in evaluation mode

    # Reconstruct (denoise) using the model
    for i in range(t_val, 0, -dt):
        print(f"\rReconstructing t={i+1}/{t_val}", end="")
        t_recon = torch.tensor([i], device="cuda")
        output_recon = diff.denoise_step(
            output_recon, t_recon.type(torch.float32), dt=dt
        )

        # apply data consistency step
        lr = 0.01

        y = y * mask  # undersampled k space data
        x_var = output_recon.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_var], lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()

            batch_size = x_var.shape[0]
            t_batch = t_recon.view(1).expand(batch_size)
            predicted_noise = diff.network(x_var, t_batch.type(torch.float32))
            a_cumprod_t = diff.a_cumprod.gather(0, t_batch).view(batch_size, 1, 1, 1)
            x0_pred = (
                x_var - torch.sqrt(1 - a_cumprod_t) * predicted_noise
            ) / torch.sqrt(a_cumprod_t)

            masked_Ax = A(x0_pred) * mask
            loss = F.mse_loss(masked_Ax.real, y.real) + F.mse_loss(
                masked_Ax.imag, y.imag
            )
            loss.backward()
            optimizer.step()
        output_recon = x_var.detach()

    if visualize:
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(lab_prior[0, 0].cpu().numpy(), cmap="gray")
        plt.title(f"Prior Image (noisy t={t_val})")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(output_recon[0, 0].detach().cpu().numpy(), cmap="gray")
        plt.title("Reconstructed Noise")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(lab[0, 0].detach().cpu().numpy(), cmap="gray")
        plt.title("Target Image")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig("./var/samples/dc_reconstruction.png")


if __name__ == "__main__":
    diff = load_model()
    lab_prior, lab = load_data(visualize=True)

    # Configurable parameters for testing
    t_val = 500
    dt = 10
    R = 2
    center = 0.1
    n_steps = 2

    denoising_showcase(lab_prior, lab, diff, visualize=True)
    denoising_dc_showcase(
        lab_prior,
        lab,
        diff,
        t_val=t_val,
        dt=dt,
        R=R,
        center=center,
        n_steps=n_steps,
        visualize=True,
    )
