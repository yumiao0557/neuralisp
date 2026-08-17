"""
Standalone evaluation: load a trained checkpoint, run it against a held-out
test set (Kodak / CBSD68) across several simulated ISO/gain regimes, and
report PSNR/SSIM for the network vs. the classical bilinear-demosaic
baseline. Also dumps qualitative side-by-side PNGs (baseline | prediction |
ground truth) rendered to viewable sRGB.

This is deliberately separate from the quick in-training validate() in
train.py: that one checks progress on a fixed noise regime; this one is the
"how does this actually perform across the operating range" report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from neuralisp.data.datasets import FullImageTestDataset
from neuralisp.data.degradation import bilinear_demosaic, degrade, render_srgb
from neuralisp.metrics import psnr as psnr_metric
from neuralisp.metrics import ssim_metric
from neuralisp.models.unet import JointISPNet, demosaic_denoise

ROOT = Path(__file__).resolve().parent.parent

# (name, log10(shot_a) range) -- roughly low/mid/high ISO operating points
NOISE_REGIMES = [
    ("low_iso", (-4.5, -4.0)),
    ("mid_iso", (-3.0, -2.5)),
    ("high_iso", (-1.5, -1.0)),
]


def _save_tensor_png(tensor: torch.Tensor, path: Path) -> None:
    arr = tensor.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


def load_model(checkpoint_path: str, device) -> JointISPNet:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["cfg"]
    model = JointISPNet(
        in_channels=6,
        base_channels=cfg["base_channels"],
        channel_mults=tuple(cfg["channel_mults"]),
        blocks_per_level=cfg["blocks_per_level"],
        bottleneck_blocks=cfg["bottleneck_blocks"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded checkpoint from epoch {ckpt['epoch']} (best_psnr={ckpt.get('best_psnr', 'n/a')})")
    return model


@torch.no_grad()
def evaluate_regime(model, loader, device, gain_range, save_dir: Path | None, n_qualitative: int = 4):
    torch.manual_seed(42)
    pred_psnrs, pred_ssims, base_psnrs, base_ssims = [], [], [], []

    for i, (clean, name) in enumerate(loader):
        clean = clean.to(device)
        out = degrade(clean, gain_range=gain_range)
        baseline = bilinear_demosaic(out.packed_bayer)
        pred = demosaic_denoise(model, out.packed_bayer, out.noise_map, baseline)

        pred_psnrs.append(psnr_metric(pred, out.target_linear_rgb))
        pred_ssims.append(ssim_metric(pred, out.target_linear_rgb))
        base_psnrs.append(psnr_metric(baseline, out.target_linear_rgb))
        base_ssims.append(ssim_metric(baseline, out.target_linear_rgb))

        if save_dir is not None and i < n_qualitative:
            save_dir.mkdir(parents=True, exist_ok=True)
            gt_srgb = render_srgb(out.target_linear_rgb, out.wb_gains, out.ccm)
            base_srgb = render_srgb(baseline, out.wb_gains, out.ccm)
            pred_srgb = render_srgb(pred, out.wb_gains, out.ccm)
            grid = torch.cat([base_srgb[0], pred_srgb[0], gt_srgb[0]], dim=2)
            _save_tensor_png(grid, save_dir / f"{name[0]}.png")

    return {
        "pred_psnr": float(np.mean(pred_psnrs)),
        "pred_ssim": float(np.mean(pred_ssims)),
        "baseline_psnr": float(np.mean(base_psnrs)),
        "baseline_ssim": float(np.mean(base_ssims)),
        "n_images": len(pred_psnrs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="data_raw/test/kodak")
    p.add_argument("--max-size", type=int, default=512)
    p.add_argument("--output-dir", type=str, default="outputs/eval")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    ds = FullImageTestDataset(ROOT / args.dataset, max_size=args.max_size)
    if args.limit is not None:
        ds.paths = ds.paths[: args.limit]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    dataset_name = Path(args.dataset).name
    out_root = ROOT / args.output_dir / dataset_name
    results = {}
    for regime_name, gain_range in NOISE_REGIMES:
        r = evaluate_regime(model, loader, device, gain_range, out_root / regime_name)
        results[regime_name] = r
        gain = r["pred_psnr"] - r["baseline_psnr"]
        print(
            f"[{dataset_name}/{regime_name}] n={r['n_images']} | "
            f"net: PSNR={r['pred_psnr']:.2f}dB SSIM={r['pred_ssim']:.4f} | "
            f"bilinear: PSNR={r['baseline_psnr']:.2f}dB SSIM={r['baseline_ssim']:.4f} | "
            f"delta={gain:+.2f}dB"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved qualitative samples + results.json under {out_root}")


if __name__ == "__main__":
    main()
