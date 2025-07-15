# dataset_mri.py
from pathlib import Path
from typing import Optional
from glob import glob
import os
from scipy.io import loadmat
import numpy as np
from numpy import ndarray
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import matplotlib.pyplot as plt
from PIL import Image


class DataSet(Dataset):
    """MRI dataset Loader"""

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: tuple[int, int] = (512, 512),
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
                # Normalize to [0, 1] range
                T.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
                # scale to [-1,1] range
                T.Lambda(lambda x: (x - 0.5) * 2),
            ]
        )

        self.image_refs = []  # List of (file_path, index) tuples
        full_data = glob(os.path.join(self.root, "*.mat"))
        # print(f"Found {len(full_data)} .mat files in {self.root}")
        for file in full_data:
            # Assume each file has 6 images as before
            for idx in range(2):
                self.image_refs.append((file, idx))

    def _load_mri_trimaps(self, file_path):
        try:
            data = loadmat(file_path)
            img1 = data.get("img1_reg")
            img2 = data.get("img2")
            if img1 is None or img2 is None:
                print(f"Missing keys in {file_path}")
                return None
            return [img1[1], img2[1]]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def __len__(self):
        return len(self.image_refs)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        file_path, img_idx = self.image_refs[idx]
        trimaps = self._load_mri_trimaps(file_path)
        if trimaps is None or img_idx >= len(trimaps):
            return torch.zeros(1, *self.image_size, dtype=torch.float32), 0

        image = trimaps[img_idx].astype(np.float32)
        if np.isnan(image).any() or np.max(image) < 1e-6:
            return torch.zeros(1, *self.image_size, dtype=torch.float32), 0

        # Remove manual normalization, just convert to PIL and apply transforms
        image = Image.fromarray(image)
        image = self.transforms(image)
        if not isinstance(image, torch.Tensor):
            raise TypeError(
                f"Expected image to be a torch.Tensor, got {type(image)} instead."
            )
        return image, 0


if __name__ == "__main__":
    dataset = DataSet(
        root="/fast_storage/hyeokgi/data_v2_slice_512/train", split="train"
    )
    print(f"dataset initialized with {len(dataset)} images.")
    # print(f"Number of images in dataset: {len(dataset)}")

    # Display the first image
    first_image, dummy = dataset[0]
    if isinstance(first_image, torch.Tensor):
        first_image = first_image.squeeze()  # Remove channel dimension
        print("First image shape:", first_image.shape)
    print("max, min, mean, std")
    print(
        f"max: {first_image.max()}, min: {first_image.min()}, mean: {first_image.mean()}, std: {first_image.std()}"
    )
    plt.imshow(first_image, cmap="gray")
    plt.axis("off")
    plt.savefig("example.png")
