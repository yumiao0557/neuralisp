"""
Fetch open-source ground-truth RGB image sets used as the "clean" side of
the synthetic-degradation pipeline (see neuralisp/data/degradation.py).

We don't need paired real-RAW datasets to get started -- the unprocessing
approach turns any clean, well-exposed sRGB image into a plausible noisy
Bayer RAW with a calibrated forward model. What matters is:
  - a large, diverse *training* pool of clean natural images (BSDS500,
    ~500 images), and
  - small, standard, held-out *test* sets that are commonly used to report
    demosaicing/denoising numbers (Kodak-24, CBSD68), so results are
    comparable to published baselines.

Layout after running:
  data_raw/train/bsds500/*.jpg        (~500 images, train+val split of BSDS500)
  data_raw/test/kodak/*.png           (24 images)
  data_raw/test/cbsd68/*.png          (68 images)
"""
from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data_raw"

KODAK_BASE = "https://r0k.us/graphics/kodak/kodak"
BSDS500_ZIP = "https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip"
CBSD68_ZIP = "https://github.com/clausmichele/CBSD68-dataset/archive/refs/heads/master.zip"


def download(url: str, timeout: int = 60) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def fetch_kodak():
    out_dir = DATA_RAW / "test" / "kodak"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.png"))
    if len(existing) == 24:
        print(f"[kodak] already have {len(existing)} images, skipping")
        return
    for i in range(1, 25):
        name = f"kodim{i:02d}.png"
        dest = out_dir / name
        if dest.exists():
            continue
        print(f"[kodak] downloading {name}")
        data = download(f"{KODAK_BASE}/{name}")
        dest.write_bytes(data)
    print(f"[kodak] done: {len(list(out_dir.glob('*.png')))} images")


def fetch_zip_dataset(
    url: str,
    out_dir: Path,
    image_suffix_filter=(".jpg", ".jpeg", ".png"),
    path_must_contain: str | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out_dir.glob("*") if p.suffix.lower() in image_suffix_filter]
    if len(existing) > 0:
        print(f"[{out_dir.name}] already have {len(existing)} images, skipping download")
        return
    print(f"[{out_dir.name}] downloading zip from {url}")
    content = download(url, timeout=180)
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        members = [
            m for m in zf.namelist()
            if Path(m).suffix.lower() in image_suffix_filter
            and not m.endswith("/")
            and (path_must_contain is None or path_must_contain in m)
        ]
        for m in members:
            data = zf.read(m)
            dest = out_dir / Path(m).name
            dest.write_bytes(data)
    print(f"[{out_dir.name}] done: {len(list(out_dir.glob('*')))} files")


def fetch_bsds500():
    out_dir = DATA_RAW / "train" / "bsds500"
    # BIDS/BSDS500 layout: BSDS500-master/BSDS500/data/images/{train,val,test}/*.jpg
    fetch_zip_dataset(BSDS500_ZIP, out_dir, image_suffix_filter=(".jpg",))


def fetch_cbsd68():
    out_dir = DATA_RAW / "test" / "cbsd68"
    # clausmichele/CBSD68-dataset layout has original/noisy* subfolders sharing
    # filenames (0000.png..0067.png) -- only pull the clean "original_png" ones.
    fetch_zip_dataset(
        CBSD68_ZIP, out_dir, image_suffix_filter=(".png",), path_must_contain="/original_png/"
    )


if __name__ == "__main__":
    fetch_kodak()
    fetch_cbsd68()
    fetch_bsds500()
    print("All downloads complete.")
    for split_dir in [DATA_RAW / "train" / "bsds500", DATA_RAW / "test" / "kodak", DATA_RAW / "test" / "cbsd68"]:
        n = len(list(split_dir.glob("*")))
        print(f"{split_dir}: {n} files")
