# dataset.py
from pathlib import Path
from typing import Optional, List

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from sklearn.model_selection import train_test_split


class DataSet(Dataset):
    """Oxford-IIIT Pet dataset loader"""

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: tuple[int, int] = (256, 256),
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
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        img_dir = self.root / "images"
        mask_dir = self.root / "annotations" / "trimaps"
        all_files: List[str] = [p.stem for p in img_dir.glob("*.jpg")]
        all_files.sort()

        # Split dataset into train and validation/test sets
        if split == "train":
            train_files, _ = train_test_split(
                all_files, train_size=train_ratio, random_state=42
            )
            self.names = train_files
        else:
            _, test_files = train_test_split(
                all_files, train_size=train_ratio, random_state=42
            )
            if split == "val":
                self.names = test_files[: len(test_files) // 2]
            else:
                self.names = test_files[len(test_files) // 2 :]

        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[idx]

        # -------- Load and preprocess input image --------
        img_path = self.img_dir / f"{name}.jpg"
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = self.transforms(img)  # Tensor [3,H,W], float32

        # -------- Load and preprocess trimap mask --------
        mask_path = self.mask_dir / f"{name}.png"
        with Image.open(mask_path) as m:
            m = m.resize(self.image_size, resample=Image.NEAREST)
            mask_np = np.array(m, dtype=np.int64) - 1  # {1,2,3} → {0,1,2}, shape [H, W]
            mask_tensor = torch.from_numpy(
                np.eye(3, dtype=np.float32)[mask_np]  # shape [H, W, 3]
            ).permute(
                2, 0, 1
            )  # shape [3, H, W]

        return img, mask_tensor


if __name__ == "__main__":
    dataset = DataSet(root="data_pet", split="train", image_size=(256, 256))
    print(f"Dataset size: {len(dataset)}")
    img, mask = dataset[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    print(f"Image dtype: {img.dtype}, Mask dtype: {mask.dtype}")

    # Check some properties
    print(img.min(), img.max())
    print(mask.min(), mask.max())

    # visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.title("Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(mask.permute(1, 2, 0).numpy())
    plt.title("Mask")
    plt.axis("off")
    plt.show()
