# dataset.py
from pathlib import Path
from typing import Optional
import struct

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch.nn.functional as F


class DataSet(Dataset):
    """MNIST dataset loader"""

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: tuple[int, int] = (28, 28),
        train_ratio: float = 0.8,
        transforms: Optional[T.Compose] = None,
    ):
        assert split in ("train", "val", "test")
        self.root = Path(root)
        self.image_size = image_size
        self.transforms = transforms or T.Compose(
            [
                T.Resize(self.image_size),
                T.ToTensor(),
            ]
        )

        # Load MNIST data
        if split in ("train", "val"):
            # Load full training dataset
            full_images = self._load_images(self.root / "train-images.idx3-ubyte")
            full_labels = self._load_labels(self.root / "train-labels.idx1-ubyte")

            # Split training data into train and validation using sklearn
            train_images, val_images, train_labels, val_labels = train_test_split(
                full_images,
                full_labels,
                train_size=train_ratio,
                random_state=42,
                stratify=full_labels,
            )

            if split == "train":
                self.images = train_images
                self.labels = train_labels
            else:  # val
                self.images = val_images
                self.labels = val_labels

        else:  # test
            self.images = self._load_images(self.root / "t10k-images.idx3-ubyte")
            self.labels = self._load_labels(self.root / "t10k-labels.idx1-ubyte")

    def _load_images(self, filepath: Path) -> np.ndarray:
        """Load MNIST images from idx3-ubyte file"""
        with open(filepath, "rb") as f:
            # Read header
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            assert magic == 2051, f"Invalid magic number: {magic}"

            # Read image data
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows, cols)

        return images

    def _load_labels(self, filepath: Path) -> np.ndarray:
        """Load MNIST labels from idx1-ubyte file"""
        with open(filepath, "rb") as f:
            # Read header
            magic, num_labels = struct.unpack(">II", f.read(8))
            assert magic == 2049, f"Invalid magic number: {magic}"

            # Read label data
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        return labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get image and label
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to PIL Image and apply transforms
        pil_image = Image.fromarray(image, mode="L")
        image_tensor = self.transforms(pil_image)

        # Ensure we have a tensor (ToTensor() in transforms guarantees this)
        assert isinstance(image_tensor, torch.Tensor)
        image_tensor = F.pad(
            image_tensor, (2, 2, 2, 2), mode="constant", value=0
        )  # Pad to 32x32

        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


if __name__ == "__main__":
    # Test the dataset
    dataset = DataSet(root="mnist", split="train", image_size=(28, 28))
    print(f"Training dataset size: {len(dataset)}")

    val_dataset = DataSet(root="mnist", split="val", image_size=(28, 28))
    print(f"Validation dataset size: {len(val_dataset)}")

    test_dataset = DataSet(root="mnist", split="test", image_size=(28, 28))
    print(f"Test dataset size: {len(test_dataset)}")

    # Test loading a sample
    img, label = dataset[0]
    print(f"Image shape: {img.shape}, Label: {label.item()}")
    print(f"Image dtype: {img.dtype}, Label dtype: {label.dtype}")
    print(f"Image range: {img.min():.3f} to {img.max():.3f}")

    # Simple visualization
    plt.figure(figsize=(6, 2))
    for i in range(5):
        img_sample, label_sample = dataset[i]
        plt.subplot(1, 5, i + 1)
        plt.imshow(img_sample.squeeze(), cmap="gray")
        plt.title(f"{label_sample.item()}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
