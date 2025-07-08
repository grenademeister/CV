import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from vae import VAE
from helper import vae_loss
from dataset import DataSet


def test_vae():
    # Create a simple VAE model
    model = VAE(in_channels=1, latent_dim=128)  # Example dimensions for MNIST
    model.load_state_dict(
        torch.load("var/checkpoints/20250708_001839/best_27.pth", weights_only=True)
    )  # Load the pre-trained model
    model = model.to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Move model to GPU if available
    model.eval()  # Set the model to evaluation mode

    # load data
    dataset = DataSet("mnist", split="test")

    # Test multiple examples
    num_examples = 8
    indices = [2, 5, 10, 15, 20, 25, 30, 35]  # Different test samples

    plt.figure(figsize=(16, 4))

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        dummy_input = img.view(1, 1, 32, 32).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Forward pass through the VAE
        with torch.no_grad():
            reconstructed, latent = model(dummy_input)
            mu, logvar = torch.chunk(latent, 2, dim=1)

        # Calculate the loss for first example
        if i == 0:
            loss_recon, loss_kl = vae_loss(reconstructed, dummy_input, mu, logvar)
            loss = loss_recon + loss_kl
            print("Test passed. Loss:", loss.item())
            print("mu shape:", mu.shape)
            print("logvar shape:", logvar.shape)

        # Visualize original image
        plt.subplot(2, num_examples, i + 1)
        plt.title(f"Original {idx}")
        plt.imshow(dummy_input.view(32, 32).cpu().numpy(), cmap="gray")
        plt.axis("off")

        # Visualize reconstructed image
        plt.subplot(2, num_examples, i + 1 + num_examples)
        plt.title(f"Reconstructed {idx}")
        plt.imshow(reconstructed.view(32, 32).cpu().numpy(), cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("vae_reconstruction_multiple.png")


def test_vae_sampling():
    # Create a simple VAE model
    model = VAE(in_channels=1, latent_dim=128)  # Example dimensions for MNIST
    model.load_state_dict(
        torch.load("var/checkpoints/20250708_001839/best_27.pth", weights_only=True)
    )  # Load the pre-trained model
    model = model.to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Move model to GPU if available
    model.eval()  # Set the model to evaluation mode

    # Sample multiple images from the latent space
    num_samples = 16
    plt.figure(figsize=(12, 12))

    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, 128, 4, 4).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            sampled_img = model.decode(z, None)
            sampled_img = sampled_img.view(32, 32).cpu().numpy()

            plt.subplot(4, 4, i + 1)
            plt.title(f"Sample {i+1}")
            plt.imshow(sampled_img, cmap="gray")
            plt.axis("off")

    plt.tight_layout()
    plt.savefig("vae_sampled_images_multiple.png")


if __name__ == "__main__":
    test_vae()
    print("VAE test completed successfully.")
    test_vae_sampling()
    print("VAE sampling test completed successfully.")
