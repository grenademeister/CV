import torch
import matplotlib.pyplot as plt
import numpy as np
from ddpm import Diffusion
import os


class DDPMSampler:
    def __init__(self, device="auto"):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        self.model = Diffusion(device=self.device)

    def load_model(self, model_path):
        """Load a pre-trained DDPM model."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        # Handle different checkpoint formats
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        # Remove 'module.' prefix if present (for DataParallel models)
        new_state_dict = {
            k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()
        }

        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def sample(self, num_samples=16, image_size=32, channels=1, interval=50):
        """Generate samples using the DDPM model."""
        out_size = torch.Size([num_samples, channels, image_size, image_size])
        with torch.no_grad():
            samples = self.model.recon(out_size=out_size, interval=interval)

        # Clamp values to valid range
        samples = torch.clamp(samples, 0, 1)
        return samples.cpu().numpy()

    def visualize_samples(self, samples, save_path="generated_samples.png", title=None):
        """Visualize generated samples in a grid."""
        num_samples = len(samples)

        # Determine grid size
        grid_size = int(np.ceil(np.sqrt(num_samples)))

        # Create figure with better aesthetics
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        if title:
            fig.suptitle(title, fontsize=16, fontweight="bold")

        axes = np.array(axes).reshape(-1)
        for i, ax in enumerate(axes):
            if i < num_samples:
                # Display sample
                img = samples[i, 0] if samples.shape[1] == 1 else samples[i]
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"Sample {i+1}", fontsize=10)
            else:
                # Empty subplot
                ax.set_visible(False)

            ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved visualization to {save_path}")

    def compare_intervals(self, intervals=[1, 10, 25, 50], num_samples=4):
        """Compare sampling quality across different intervals."""
        fig, axes = plt.subplots(
            len(intervals), num_samples, figsize=(15, 4 * len(intervals))
        )
        fig.suptitle("Sampling Quality vs Interval", fontsize=16, fontweight="bold")

        for i, interval in enumerate(intervals):
            samples = self.sample(num_samples=num_samples, interval=interval)
            for j in range(num_samples):
                ax = axes[i, j] if len(intervals) > 1 else axes[j]
                ax.imshow(samples[j, 0], cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"Interval={interval}, Sample {j+1}", fontsize=10)
                ax.axis("off")

        plt.tight_layout()
        plt.savefig("interval_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved interval comparison to interval_comparison.png")


def main():
    # Initialize sampler
    sampler = DDPMSampler()

    # Load the best model
    model_path = "./var/checkpoints/best.pth"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        checkpoint_dir = "./var/checkpoints/"
        if os.path.exists(checkpoint_dir):
            print("Available checkpoints:")
            for file in os.listdir(checkpoint_dir):
                if file.endswith(".pth"):
                    print(f"  - {file}")
        return

    sampler.load_model(model_path)

    # Generate and visualize samples
    print("DDPM Image Generation")
    samples = sampler.sample(num_samples=16, interval=50)
    sampler.visualize_samples(
        samples, save_path="ddpm_samples_grid.png", title="DDPM Generated Samples (16x)"
    )
    sampler.compare_intervals(intervals=[1, 10, 25, 50], num_samples=4)
    hq_samples = sampler.sample(num_samples=9, interval=1)
    sampler.visualize_samples(
        hq_samples,
        save_path="ddpm_hq_samples.png",
        title="High-Quality DDPM Samples (interval=1)",
    )
    print("Done. Check the generated images:")
    print("  - ddpm_samples_grid.png (16 samples)")
    print("  - interval_comparison.png (interval comparison)")
    print("  - ddpm_hq_samples.png (high-quality samples)")


if __name__ == "__main__":
    main()
