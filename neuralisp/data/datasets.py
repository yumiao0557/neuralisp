from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_image_srgb01(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr  # (H, W, 3) in [0,1]


class PatchTrainDataset(Dataset):
    """Random-crop patches from a folder of clean sRGB images.

    Returns clean sRGB patches (3,H,W) in [0,1]; degradation (mosaic + noise)
    is applied on-the-fly, batched, on GPU inside the training loop -- see
    neuralisp/data/degradation.py. This keeps noise/CCM/WB randomized fresh
    every epoch instead of baking a single realization to disk.
    """

    def __init__(self, root: str | Path, patch_size: int = 128, patches_per_image: int = 8):
        self.root = Path(root)
        self.paths = sorted([p for p in self.root.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        if len(self.paths) == 0:
            raise FileNotFoundError(f"no images found in {self.root}")
        assert patch_size % 2 == 0, "patch_size must be even (RGGB mosaic constraint)"
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image

    def __len__(self) -> int:
        return len(self.paths) * self.patches_per_image

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_idx = idx // self.patches_per_image
        path = self.paths[img_idx]
        arr = _load_image_srgb01(path)
        h, w, _ = arr.shape
        ps = self.patch_size

        if h < ps or w < ps:
            # pad small images by reflection
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            h, w, _ = arr.shape

        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        patch = arr[top : top + ps, left : left + ps, :]

        if random.random() < 0.5:
            patch = np.flip(patch, axis=1)
        if random.random() < 0.5:
            patch = np.flip(patch, axis=0)
        if random.random() < 0.5:
            patch = np.rot90(patch, k=1, axes=(0, 1))
            # rot90 on non-square patch would break shape; patch_size is square so it's fine

        patch = np.ascontiguousarray(patch)
        tensor = torch.from_numpy(patch).permute(2, 0, 1).float()  # (3, H, W)
        return tensor


class FullImageTestDataset(Dataset):
    """Deterministic, full-image (center-cropped to even dims) test set.

    Used for held-out evaluation on Kodak / CBSD68. No degradation applied
    here -- the eval script calls degrade() once with a fixed seed so
    reported numbers are reproducible across runs.
    """

    def __init__(self, root: str | Path, max_size: int | None = 512):
        self.root = Path(root)
        self.paths = sorted([p for p in self.root.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        if len(self.paths) == 0:
            raise FileNotFoundError(f"no images found in {self.root}")
        self.max_size = max_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.paths[idx]
        arr = _load_image_srgb01(path)
        h, w, _ = arr.shape

        if self.max_size is not None:
            h = min(h, self.max_size)
            w = min(w, self.max_size)

        # even-crop from center (RGGB mosaic needs even H, W)
        h -= h % 2
        w -= w % 2
        full_h, full_w, _ = arr.shape
        top = (full_h - h) // 2
        left = (full_w - w) // 2
        arr = arr[top : top + h, left : left + w, :]

        tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
        return tensor, path.stem
