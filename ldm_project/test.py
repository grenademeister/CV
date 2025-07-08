import torch
import torch.nn.functional as F
from torch import nn

import yaml
import matplotlib.pyplot as plt

from ddpm import Diffusion
from vae import VAE
from dataset import DataSet


class DiffusionTester:
    def __init__(self, device, criterion, callbacks=None):
        self.callbacks = callbacks if callbacks is not None else []
        self.model = Diffusion(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = device
        self.criterion = criterion

    def load_model(self, model_path):
        """Load a pre-trained model from the specified path."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {model_path}.")

    def test_recon(self):
        self.load_model("best_0.pth")
        out_size = torch.Size([1, 1, 32, 32])
        output = self.model.recon(out_size=out_size, interval=0.5)
        print("Reconstruction completed.")
        # visualize results below
        plt.imshow(output[0, 0].cpu().numpy(), cmap="gray")
        plt.axis("off")
        plt.show()

    def test(self, test_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                preds, target = self.model(x)

                # Compute loss
                if hasattr(self.criterion, "__call__") and not isinstance(
                    self.criterion, type
                ):
                    # For nn.Module losses
                    loss = self.criterion(preds, target)
                else:
                    # For functional losses
                    loss = self.criterion(preds, target)

                total_loss += loss.item()

                # Update metrics if callback exists
                for cb in self.callbacks:
                    if hasattr(cb, "on_test_batch_end"):
                        cb.on_test_batch_end(self, preds, target)

        avg_loss = total_loss / len(test_loader)
        print(f"Test Loss: {avg_loss:.4f}")
        return avg_loss


class LDMTester:
    def __init__(self, device, criterion, callbacks=None, config=None):
        self.config = config if config is not None else yaml.safe_load("config.yaml")
        self.callbacks = callbacks if callbacks is not None else []
        self.vae = VAE(**self.config["vae"]["params"]).to(self.device)
        self.vae.load_state_dict(
            torch.load(self.config["vae"]["checkpoint_dir"], map_location=self.device)
        )
        self.diffusion = Diffusion(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = device
        self.criterion = criterion

    def load_models(self, vae_path, diffusion_path):
        """Load pre-trained VAE and diffusion models from specified paths."""
        # Load VAE
        vae_checkpoint = torch.load(vae_path, map_location=self.device)
        self.vae.load_state_dict(vae_checkpoint)
        self.vae.to(self.device)
        print(f"VAE model loaded from {vae_path}.")

        # Load Diffusion
        diffusion_checkpoint = torch.load(diffusion_path, map_location=self.device)
        self.diffusion.load_state_dict(diffusion_checkpoint)
        self.diffusion.to(self.device)
        print(f"Diffusion model loaded from {diffusion_path}.")

    def test_recon(self):
        self.load_models("vae_best.pth", "diffusion_best.pth")
        self.vae.eval()
        self.diffusion.eval()

        with torch.no_grad():
            # Generate in latent space
            latent_size = torch.Size([1, 4, 4, 4])  # Assuming latent dimensions
            latent_output = self.diffusion.recon(out_size=latent_size, interval=1)

            # Decode from latent space to image space
            output = self.vae.decode(latent_output, res_outs=None)

        print("LDM Reconstruction completed.")
        # visualize results below
        plt.imshow(output[0, 0].cpu().numpy(), cmap="gray")
        plt.axis("off")
        plt.show()

    def test(self, test_loader):
        self.vae.eval()
        self.diffusion.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Encode to latent space
                latent_x = self.vae.encode(x)

                # Apply diffusion in latent space
                preds, target = self.diffusion(latent_x)

                # Decode predictions back to image space for loss computation
                decoded_preds = self.vae.decode(preds, res_outs=None)
                decoded_target = self.vae.decode(target, res_outs=None)

                # Compute loss
                if hasattr(self.criterion, "__call__") and not isinstance(
                    self.criterion, type
                ):
                    # For nn.Module losses
                    loss = self.criterion(decoded_preds, decoded_target)
                else:
                    # For functional losses
                    loss = self.criterion(decoded_preds, decoded_target)

                total_loss += loss.item()

                # Update metrics if callback exists
                for cb in self.callbacks:
                    if hasattr(cb, "on_test_batch_end"):
                        cb.on_test_batch_end(self, decoded_preds, decoded_target)

        avg_loss = total_loss / len(test_loader)
        print(f"LDM Test Loss: {avg_loss:.4f}")
        return avg_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()  # Example loss function

    # Test DiffusionTester
    diffusion_tester = DiffusionTester(device, criterion)

    # Test LDMTester
    ldm_tester = LDMTester(device, criterion)

    # Assuming you have a DataLoader `test_loader` defined
    test_loader = DataSet(root="mnist", split="test")

    # For testing reconstruction
    diffusion_tester.test_recon()
    ldm_tester.test_recon()
