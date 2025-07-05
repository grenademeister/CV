import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from callback import SegmentationMetrics
from dataset import DataSet
from unet import UNet

import matplotlib.pyplot as plt
import numpy as np


class Tester:
    def __init__(self, will_visualize: bool = False, visualize_idx: int = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.test_loader = self._load_test_dataloader()
        self.metrics_callback = SegmentationMetrics(
            num_classes=3, class_names=["background", "unknown", "foreground"]
        )
        self.will_visualize = will_visualize
        self.visualize_idx = visualize_idx
        if self.will_visualize and visualize_idx is None:
            raise ValueError("visualize_idx must be provided if visualize is True")

    def _load_model(self):
        model_path = Path("var/checkpoints/")
        models = [f for f in model_path.glob("*.pth") if f.is_file()]
        state_dict = (
            torch.load(models[-1], map_location=self.device) if models else None
        )
        best_model = UNet(n_channels=3, n_classes=3)
        best_model.load_state_dict(state_dict) if state_dict else None
        best_model.eval()
        return best_model

    def _load_test_dataloader(self):
        dataset = DataSet(root="data_pet", split="test", image_size=(256, 256))
        return DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
        )

    def run(self):
        if not self.model:
            raise ValueError("No model found to test.")

        self.model.to(self.device)

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                print(
                    f"\rProcessing batch {i + 1}/{self.test_loader.__len__()}", end=""
                )
                images, masks = batch
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                self.metrics_callback.update(outputs, masks)

                if i == self.visualize_idx and self.will_visualize:
                    self.visualize(outputs, images)

                if self.device != torch.device("cuda"):
                    break

        metrics = self.metrics_callback.get_summary_string()
        print("\nTest Metrics:", metrics)

    def visualize(self, outputs=None, images=None):
        # Convert outputs to numpy for visualization
        outputs_np = outputs.cpu().numpy()
        # Unnormalize the images
        images_np = images.cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        images_np = images_np * std + mean

        # Visualize the first image and its output
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(images_np[2], (1, 2, 0)))
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(outputs_np[2].argmax(axis=0), cmap="gray")
        plt.title("Model Output")
        plt.axis("off")

        plt.show()


if __name__ == "__main__":
    tester = Tester(will_visualize=True, visualize_idx=0)
    tester.run()
