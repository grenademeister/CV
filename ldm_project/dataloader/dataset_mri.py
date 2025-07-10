# dataset_mri.py
from pathlib import Path
from typing import Optional
import struct

from glob import glob
import os
from PIL import Image
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch.nn.functional as F


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
            ]
        )

        self.image_refs = []  # List of (file_path, index) tuples
        full_data = glob(os.path.join(self.root, "*.mat"))
        # print(f"Found {len(full_data)} .mat files in {self.root}")
        for file in full_data:
            # Assume each file has 6 images as before
            for idx in range(6):
                self.image_refs.append((file, idx))

    def _load_mri_trimaps(self, file_path):
        data = self._load_mat_file(file_path)
        if data is not None:
            img1 = data["img1"]
            img2 = data["img2"]
            return [img1[0], img1[1], img1[2], img2[0], img2[1], img2[2]]

    def _load_mat_file(self, file_path):
        try:
            data = loadmat(file_path)
            return data
        except Exception as e:
            print(f"An error occurred while loading the .mat file: {e}")
            return None

    def __len__(self):
        return len(self.image_refs)

    def __getitem__(self, idx):
        file_path, img_idx = self.image_refs[idx]
        trimaps = self._load_mri_trimaps(file_path)
        if trimaps is None:
            raise RuntimeError(f"Failed to load trimaps from {file_path}")
        image = trimaps[img_idx]
        image = Image.fromarray(image.astype(np.float32))
        if self.transforms:
            image = self.transforms(image)
        return image, 0


if __name__ == "__main__":
    dataset = DataSet(root="../data/data_v2_slice_512/train/", split="train")
    print(f"dataset initialized with {len(dataset)} images.")
    # print(f"Number of images in dataset: {len(dataset)}")

    # Display the first image
    first_image, dummy = dataset[0]
    if isinstance(first_image, torch.Tensor):
        first_image = first_image.squeeze().numpy()  # Remove channel dimension
        print("First image shape:", first_image.shape)
    plt.imshow(first_image, cmap="gray")
    plt.axis("off")
    plt.savefig("example.png")
