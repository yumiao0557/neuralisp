from pathlib import Path

import torch

from neuralisp.data.datasets import FullImageTestDataset, PatchTrainDataset

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data_raw"


def test_patch_train_dataset():
    ds = PatchTrainDataset(DATA_RAW / "train" / "bsds500", patch_size=64, patches_per_image=4)
    assert len(ds) == len(ds.paths) * 4
    patch = ds[0]
    assert patch.shape == (3, 64, 64)
    assert patch.min() >= 0 and patch.max() <= 1


def test_full_image_test_dataset_kodak():
    ds = FullImageTestDataset(DATA_RAW / "test" / "kodak", max_size=256)
    assert len(ds) == 24
    img, name = ds[0]
    assert img.shape[0] == 3
    assert img.shape[1] % 2 == 0 and img.shape[2] % 2 == 0
    assert img.shape[1] <= 256 and img.shape[2] <= 256
    assert isinstance(name, str)


def test_full_image_test_dataset_cbsd68():
    ds = FullImageTestDataset(DATA_RAW / "test" / "cbsd68", max_size=None)
    assert len(ds) == 68
    img, _ = ds[0]
    assert img.shape[1] % 2 == 0 and img.shape[2] % 2 == 0
