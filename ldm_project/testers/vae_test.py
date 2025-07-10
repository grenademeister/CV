import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from ldm_project.model.vae import VAE

from scipy.io import loadmat


def load_model(self, model_path):
    """Load a pre-trained VAE model."""
    checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

    # Handle different checkpoint formats
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    # Remove 'module.' prefix if present (for DataParallel models)
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        new_state_dict[key] = v

    self.model.load_state_dict(new_state_dict)
    self.model.to(self.device)
    self.model.eval()
    print(f"✓ Model loaded from {model_path}")


def load_mat_file(file_path):
    """
    Load a .mat file and return its contents.

    Parameters:
    file_path (str): The path to the .mat file.

    Returns:
    dict: A dictionary containing the contents of the .mat file.
    """
    try:
        data = loadmat(file_path)
        return data
    except Exception as e:
        print(f"An error occurred while loading the .mat file: {e}")
        return None


def load_vae_model(model_path="best_1_vae.pth"):
    """Load a pre-trained VAE model."""
    model = VAE(in_channels=1, latent_dim=8)
    state_dict = torch.load(model_path, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        new_state_dict[key] = v
    model.load_state_dict(new_state_dict)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model


def test_vae_sampling():
    # Create a simple VAE model
    model = load_vae_model()

    # Sample multiple images from the latent space
    num_samples = 16
    plt.figure(figsize=(12, 12))

    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, 8, 64, 64).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            sampled_img = model.decode(z, None)
            sampled_img = sampled_img.view(512, 512).cpu().numpy()

            plt.subplot(4, 4, i + 1)
            plt.title(f"Sample {i+1}")
            plt.imshow(sampled_img, cmap="gray")
            plt.axis("off")

    plt.tight_layout()
    plt.savefig("vae_sampled_images_multiple.png")
    plt.show()


def test_vae_recon():
    """Test VAE reconstruction with clean visualization."""
    # Load model and data
    model = load_vae_model()
    mat_data = load_mat_file("example.mat")

    if mat_data is None:
        print("Failed to load the .mat file.")
        return

    # Get the image data
    img1 = mat_data["img1_reg"][0]  # Assuming first image
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare input tensor
    input_tensor = torch.Tensor(img1).view(1, 1, 512, 512).to(device)

    # Perform reconstruction
    with torch.no_grad():
        latent, _, _ = model.encode(input_tensor)
        reconstructed = model.decode(latent, None)

        # Convert to numpy for visualization
        original = img1
        recon = reconstructed.view(512, 512).cpu().numpy()

    # Visualization
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(recon, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("vae_reconstruction_comparison.png", dpi=150)
    plt.show()

    # Print reconstruction quality metrics
    mse = torch.nn.functional.mse_loss(
        torch.tensor(original), torch.tensor(recon)
    ).item()
    print(f"Reconstruction MSE: {mse:.6f}")


if __name__ == "__main__":
    # test_vae()
    print("VAE test completed successfully.")
    test_vae_sampling()
    test_vae_recon()
    print("VAE sampling test completed successfully.")
