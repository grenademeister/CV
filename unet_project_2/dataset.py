# dataset.py

import os
from pathlib import Path
from typing import Optional, List

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class DataSet(Dataset):
    """
    Oxford-IIIT Pet dataset loader producing:
      - input: 3-channel RGB image (Tensor float32 [0,1])
      - target: 3-channel one-hot trimap mask (Tensor float32 {0,1})

    Assumes directory structure:
      root/
        images/      *.jpg
        annotations/trimaps/*.png

    Args:
        root:        Path to dataset root.
        split:       'train' or 'val'.
        image_size:  (H, W) to which both input and mask are resized.
        train_ratio: Fraction of files used for training; rest for validation.
        transforms:  Optional torchvision transforms applied to input images.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: tuple[int, int] = (256, 256),
        train_ratio: float = 0.8,
        transforms: Optional[T.Compose] = None,
    ):
        assert split in ("train", "val")
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
        split_idx = int(train_ratio * len(all_files))
        if split == "train":
            self.names = all_files[:split_idx]
        else:
            self.names = all_files[split_idx:]

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
            mask_np: np.ndarray = np.array(m, dtype=np.int64)
            # trimap values {1,2,3} → shift to {0,1,2}
            mask_np = mask_np - 1
            # one‐hot encoding: shape (H,W,3)
            mask_oh = np.eye(3, dtype=np.float32)[mask_np]
            # to (3,H,W)
            mask_tensor = torch.from_numpy(mask_oh).permute(2, 0, 1)

        return img, mask_tensor
