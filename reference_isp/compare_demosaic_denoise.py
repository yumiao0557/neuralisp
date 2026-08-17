"""
6-panel breakdown, per test image PER NOISE REGIME (low/mid/high simulated
ISO), splitting "demosaic quality" from "end-to-end quality" so each noise
level gets both comparisons on the SAME noisy input:

  Row 1 -- demosaic comparison:
    [1] bilinear demosaic (ours, classical, no denoise)
    [2] Malvar demosaic (fast-openISP's CFA module, isolated -- no WB/CCM/
        gamma/denoise/sharpening, so it's directly comparable to [1])
    [3] JointISPNet
  Row 2 -- end-to-end comparison:
    [4] traditional ISP (fast-openISP's full pipeline: its own demosaic,
        NLM/BNF denoise, gamma, sharpening -- rendered with ITS OWN stages,
        because that's what that system actually outputs end-to-end)
    [5] JointISPNet (same output as [3] -- see note below)
    [6] ground truth

Panels [1], [2], [3], [6] are rendered through THIS PROJECT'S OWN
render_srgb (same WB/CCM/gamma) so demosaic differences aren't confounded
by a different tone curve. Panel [4] keeps fast-openISP's own rendering,
since that's what that system actually outputs.

Note on [3] vs [5]: JointISPNet does joint demosaic+denoise -- it has no
separate "demosaic-only" mode, so its output is identical in both rows.
That's not a bug, it's the point: unlike the traditional pipeline (a
classical demosaic step with a classical denoise step bolted on after),
JointISPNet's demosaic result IS its end-to-end result. bilinear/Malvar,
by contrast, differ a lot between the two rows once noise is high, because
the traditional ISP's row 2 adds real denoise stages (NLM/BNF) on top of
Malvar that row 1's isolated Malvar panel doesn't have.

Output layout: outputs/demosaic_denoise_breakdown/<dataset>/<regime>/<image>.png

Run (from repo root, with venv active):
    python reference_isp\\compare_demosaic_denoise.py --checkpoint checkpoints\\joint_isp_v1\\best.pt --dataset data_raw\\test\\kodak
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compare_traditional_isp as trad  # noqa: E402 (local module, adds ROOT/fast-openISP to sys.path)
from utils.yacs import Config  # noqa: E402 (vendored, fast-openISP)
from modules.cfa import CFA  # noqa: E402 (vendored, fast-openISP)

from neuralisp.data.datasets import FullImageTestDataset  # noqa: E402
from neuralisp.data.degradation import bilinear_demosaic, degrade, render_srgb, unpack_rggb  # noqa: E402
from neuralisp.evaluate import NOISE_REGIMES, load_model  # noqa: E402
from neuralisp.metrics import psnr as psnr_metric  # noqa: E402
from neuralisp.metrics import ssim_metric  # noqa: E402
from neuralisp.models.unet import demosaic_denoise  # noqa: E402

ROOT = trad.ROOT
RAW_MAX = trad.RAW_MAX

LABELS = [
    "Bilinear demosaic",
    "Malvar demosaic (openISP)",
    "JointISPNet",
    "Traditional ISP (full pipeline)",
    "JointISPNet (same as row 1)",
    "Ground truth",
]


def run_malvar_demosaic_only(bayer_01: np.ndarray) -> np.ndarray:
    """Isolated Malvar demosaic: the SAME pre-WB camera-native bayer mosaic
    that bilinear_demosaic() and JointISPNet see, no WB/CCM/gamma/denoise/
    sharpening applied. Returns (H, W, 3) float32 in [0,1], camera-native
    linear RGB (pre-WB) -- directly comparable to bilinear_demosaic()'s
    output before this project's render_srgb() is applied to both.
    """
    cfg = Config({
        "hardware": {"bayer_pattern": "rggb"},
        "cfa": {"mode": "malvar"},
        "saturation_values": {"hdr": RAW_MAX},
    })
    cfa = CFA(cfg)
    bayer_uint16 = np.round(np.clip(bayer_01, 0.0, 1.0) * RAW_MAX).astype(np.uint16)
    data = {"bayer": bayer_uint16}
    cfa.execute(data)
    rgb_uint16 = data["rgb_image"]  # (H, W, 3), camera-native, pre-WB, in [0, RAW_MAX]
    return rgb_uint16.astype(np.float32) / RAW_MAX


def _load_font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_labeled_grid(panels: list[np.ndarray], labels: list[str], title: str, cols: int = 3) -> Image.Image:
    h, w = panels[0].shape[:2]
    rows = -(-len(panels) // cols)
    label_h = 40
    title_h = 26
    pad = 4
    font = _load_font(15)
    title_font = _load_font(16)

    canvas_w = cols * w + (cols + 1) * pad
    canvas_h = title_h + rows * (h + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad + 4, 4), title, fill=(255, 210, 120), font=title_font)

    for idx, (panel, label) in enumerate(zip(panels, labels)):
        r, c = divmod(idx, cols)
        x = pad + c * (w + pad)
        y = title_h + pad + r * (h + label_h + pad)
        draw.text((x + 4, y + 4), label, fill=(255, 255, 255), font=font)
        canvas.paste(Image.fromarray(panel), (x, y + label_h))

    return canvas


def process_regime(model, device, clean, regime_name, gain_range):
    """Run one noise regime and return (panels, metrics_dict) for a single image."""
    deg = degrade(clean, gain_range=gain_range)

    bilinear_lin = bilinear_demosaic(deg.packed_bayer)
    full_bayer = unpack_rggb(deg.packed_bayer)[0, 0].cpu().numpy()
    malvar_lin_np = run_malvar_demosaic_only(full_bayer)
    malvar_lin = torch.from_numpy(malvar_lin_np).permute(2, 0, 1).unsqueeze(0).to(device)
    net_lin = demosaic_denoise(model, deg.packed_bayer, deg.noise_map, bilinear_lin)

    gt_srgb = render_srgb(deg.target_linear_rgb, deg.wb_gains, deg.ccm)
    bilinear_srgb = render_srgb(bilinear_lin, deg.wb_gains, deg.ccm)
    malvar_srgb = render_srgb(malvar_lin, deg.wb_gains, deg.ccm)
    net_srgb = render_srgb(net_lin, deg.wb_gains, deg.ccm)

    wb_np = deg.wb_gains[0].cpu().numpy()
    ccm_np = deg.ccm[0].cpu().numpy()
    trad_output = trad.run_traditional_isp(full_bayer, wb_np, ccm_np)  # uint8 HWC
    trad_tensor = trad._uint8_hwc_to_tensor01(trad_output, device)

    metrics = {
        "bilinear": (psnr_metric(bilinear_srgb, gt_srgb), ssim_metric(bilinear_srgb, gt_srgb)),
        "malvar": (psnr_metric(malvar_srgb, gt_srgb), ssim_metric(malvar_srgb, gt_srgb)),
        "traditional_isp": (psnr_metric(trad_tensor, gt_srgb), ssim_metric(trad_tensor, gt_srgb)),
        "net": (psnr_metric(net_srgb, gt_srgb), ssim_metric(net_srgb, gt_srgb)),
    }

    panels = [
        trad._tensor01_to_uint8_hwc(bilinear_srgb[0]),
        trad._tensor01_to_uint8_hwc(malvar_srgb[0]),
        trad._tensor01_to_uint8_hwc(net_srgb[0]),
        trad_output,
        trad._tensor01_to_uint8_hwc(net_srgb[0]),
        trad._tensor01_to_uint8_hwc(gt_srgb[0]),
    ]
    return panels, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="data_raw/test/kodak")
    p.add_argument("--max-size", type=int, default=384)
    p.add_argument("--output-dir", type=str, default="outputs/demosaic_denoise_breakdown")
    p.add_argument("--n-images", type=int, default=6,
                    help="how many test images to render grids for per regime (-1 = all)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    ds = FullImageTestDataset(ROOT / args.dataset, max_size=args.max_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    dataset_name = Path(args.dataset).name
    out_root = ROOT / args.output_dir / dataset_name
    n_save = len(ds) if args.n_images < 0 else args.n_images

    all_results = {}
    for regime_name, gain_range in NOISE_REGIMES:
        torch.manual_seed(42)
        regime_dir = out_root / regime_name
        acc = {k: {"psnr": [], "ssim": []} for k in ("bilinear", "malvar", "traditional_isp", "net")}

        for i, (clean, name) in enumerate(loader):
            clean = clean.to(device)
            panels, metrics = process_regime(model, device, clean, regime_name, gain_range)

            for k, (psnr_v, ssim_v) in metrics.items():
                acc[k]["psnr"].append(psnr_v)
                acc[k]["ssim"].append(ssim_v)

            if i < n_save:
                regime_dir.mkdir(parents=True, exist_ok=True)
                title = f"{name[0]}  |  regime: {regime_name}  (log10 shot_a in {gain_range})"
                grid = make_labeled_grid(panels, LABELS, title, cols=3)
                grid.save(regime_dir / f"{name[0]}.png")

        summary = {k: {"psnr": float(np.mean(v["psnr"])), "ssim": float(np.mean(v["ssim"])), "n_images": len(v["psnr"])}
                   for k, v in acc.items()}
        all_results[regime_name] = summary

        print(f"[{dataset_name}/{regime_name}] n={summary['bilinear']['n_images']}")
        for k in ("bilinear", "malvar", "traditional_isp", "net"):
            m = summary[k]
            print(f"  {k:<16} PSNR={m['psnr']:6.2f}dB  SSIM={m['ssim']:.4f}")

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsaved up to {n_save} labeled 6-panel grids per regime, and results.json, under {out_root}")


if __name__ == "__main__":
    main()
