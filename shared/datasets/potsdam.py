"""
Component: Potsdam Dataset Loader
Location: shared/datasets/potsdam.py

Ref:
    - rules/STRUCTURE.md
    - rules/CONVENTIONS.md
"""
import os
import os.path as osp
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as albu

from shared.utils.cdga_utils import compute_distance_map, compute_modulation_mask

CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 204, 0], [255, 0, 0]]


class PotsdamDataset(Dataset):
    """Dataset class for ISPRS Potsdam segmentation dataset (pre-cropped 512x512 PNGs)."""

    def __init__(self, split: str, crop_size: Optional[int] = None, data_root: str = "data/potsdamRGB") -> None:
        self.split = split
        self.data_root = data_root
        self.crop_size = crop_size

        # Layout on local SSD / Google Drive after extraction
        self.img_dir = osp.join(data_root, "img_dir", split)
        self.mask_dir = osp.join(data_root, "ann_dir", split)

        self.img_ids = self.get_img_ids()

        # Define data augmentations and normalization
        if split == "train":
            self.transform = albu.Compose([
                albu.RandomRotate90(p=0.5),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.transform = albu.Compose([
                albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def get_img_ids(self) -> list[str]:
        """Load and return image IDs in the directory."""
        if not osp.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        img_filenames = os.listdir(self.img_dir)
        # Filter and sort to be deterministic
        img_filenames = [f for f in img_filenames if not f.startswith(".") and f.lower().endswith((".png", ".tif"))]
        img_filenames.sort()
        img_ids = [str(f.split('.')[0]) for f in img_filenames]
        return img_ids

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        img_id = self.img_ids[index]
        
        # Support both .png and .tif extensions
        img_path = osp.join(self.img_dir, f"{img_id}.png")
        if not osp.exists(img_path):
            img_path = osp.join(self.img_dir, f"{img_id}.tif")

        mask_path = osp.join(self.mask_dir, f"{img_id}.png")
        if not osp.exists(mask_path):
            mask_path = osp.join(self.mask_dir, f"{img_id}.tif")

        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        # The HuggingFace Geo_dataset (wsdwJohn1231) stores Potsdam labels as 1-indexed (1-6).
        # CrossEntropyLoss expects 0-indexed labels (0-4). Remap:
        #   Clutter (label 6) → 255 (ignore_index — excluded from metric)
        #   Labels 1-5 → 0-4 (valid classes)
        clutter = (mask == 6)
        valid = (mask != 255) & (~clutter)
        mask[valid] = mask[valid] - 1
        mask[clutter] = 255

        augmented = self.transform(image=img, mask=mask)
        img_aug = augmented['image']
        mask_aug = augmented['mask']

        # Format to PyTorch Tensors
        img_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_aug).long()

        results = {
            "img_id": img_id,
            "image": img_tensor,
            "label": mask_tensor
        }

        # If training, compute boundary mask for CDGA
        if self.split == "train":
            D_M = compute_distance_map(mask_aug)
            # Use default sigma=5.0 and decay_fn='gaussian'
            S_mask = compute_modulation_mask(D_M, sigma=5.0, decay_fn="gaussian")
            results["boundary_mask"] = torch.from_numpy(S_mask).unsqueeze(0).float()  # (1, H, W)

        return results

    def __len__(self) -> int:
        return len(self.img_ids)
