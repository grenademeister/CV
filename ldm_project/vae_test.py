import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from vae import VAE


def load_model(self, model_path):
    """Load a pre-trained DDPM model."""
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


def test_vae_sampling():
    # Create a simple VAE model
    model = VAE(in_channels=1, latent_dim=8)  # Example dimensions for MNIST
    state_dict = torch.load(
        "var/checkpoints/20250709_171445/best_27.pth",
        weights_only=True,
    )
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        new_state_dict[key] = v
    model.load_state_dict(new_state_dict)  # Load the pre-trained model
    model = model.to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Move model to GPU if available
    model.eval()  # Set the model to evaluation mode

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


if __name__ == "__main__":
    # test_vae()
    # print("VAE test completed successfully.")
    test_vae_sampling()
    print("VAE sampling test completed successfully.")
