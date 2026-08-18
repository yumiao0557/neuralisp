"""
Classical/traditional ISP reference baseline, using fast-openISP
(https://github.com/QiuJueqin/fast-openISP, vendored in ./fast-openISP as a
fork with local additions: an `nfc` chroma noise reduction module and an
`lsc` lens shading correction module, plus rewritten `bcc`/`hsc` stages).

Runs fast-openISP on the *exact same* synthetic noisy-Bayer test inputs
used by neuralisp.evaluate, in two scopes, so the comparison against
JointISPNet can be read two ways:

    1. bilinear demosaic          (ours, no denoise at all)
    2. classical_dd                (fast-openISP, SAME SCOPE as JointISPNet:
                                     Malvar demosaic + NLM/NFC/BNF denoise
                                     only -- no sharpening/contrast/hue/
                                     brightness. The fair, apples-to-apples
                                     "does joint beat cascaded" test.)
    3. traditional_isp_full        (fast-openISP, everything on: adds CEH/
                                     EEH/FCS/HSC/BCC -- what a real product
                                     classical ISP would actually ship.)
    4. JointISPNet                  (ours, learned demosaic+denoise)
    5. ground truth

The traditional pipeline is given the SAME per-image white-balance gains
and color-correction matrix that degrade() sampled for that image (see
build_openisp_config), so differences in the comparison reflect demosaic/
denoise/pipeline quality, not incidental color differences.

Caveat on traditional_isp_full specifically: it includes EEH (sharpening)
and CEH (contrast/CLAHE) stages that JointISPNet does not attempt. Those
push pixel values away from the flat, un-enhanced ground truth on purpose
(real, standard ISP stages that usually look better to a human), which can
make its PSNR/SSIM look worse than a same-scope comparison would -- despite
the output often looking perfectly reasonable. classical_dd is the number
to trust for "is the joint network better," not traditional_isp_full.

Run (from repo root, with venv active):
    python reference_isp\\compare_traditional_isp.py --checkpoint checkpoints\\joint_isp_v1\\best.pt --dataset data_raw\\test\\kodak
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAST_OPENISP_DIR = Path(__file__).resolve().parent / "fast-openISP"
if str(FAST_OPENISP_DIR) not in sys.path:
    sys.path.insert(0, str(FAST_OPENISP_DIR))

from pipeline import Pipeline  # noqa: E402  (vendored, fast-openISP)
from utils.yacs import Config  # noqa: E402  (vendored, fast-openISP)

from neuralisp.data.datasets import FullImageTestDataset  # noqa: E402
from neuralisp.data.degradation import bilinear_demosaic, degrade, render_srgb, unpack_rggb  # noqa: E402
from neuralisp.evaluate import NOISE_REGIMES, load_model  # noqa: E402
from neuralisp.metrics import psnr as psnr_metric  # noqa: E402
from neuralisp.metrics import ssim_metric  # noqa: E402
from neuralisp.models.unet import demosaic_denoise  # noqa: E402

RAW_BIT_DEPTH = 12
RAW_MAX = 2 ** RAW_BIT_DEPTH - 1  # 4095 -- matches degrade()'s ADC quantization


def build_openisp_config(
    height: int, width: int, wb_gains: np.ndarray, ccm: np.ndarray, full_pipeline: bool = True
) -> Config:
    """Config matched to this project's forward model: no black-level offset
    (our synthetic RAW is already post-BLC), and the SAME per-image WB gains
    / CCM sampled by degrade(), so the classical pipeline runs under
    identical color conditions to the network and the ground truth.

    full_pipeline=True: the complete product-style pipeline (adds CEH/EEH/
    FCS/HSC/BCC -- contrast, sharpening, hue/saturation, brightness -- none
    of which JointISPNet attempts, so this is *not* a same-scope comparison,
    just "what would a real classical ISP output").

    full_pipeline=False: stops right after BNF -- Malvar demosaic + NLM +
    NFC (chroma) + BNF denoise, nothing else. This is the actual same-scope
    comparison: both sides own exactly "noisy Bayer -> demosaiced, denoised
    linear-ish RGB" and nothing more, which is what JointISPNet was built to
    do as a module (see project README's pipeline-boundary diagram). EEH and
    FCS are disabled together because FCS depends on EEH's edge map
    (fast-openISP raises if FCS is enabled without EEH).
    """
    r_gain = int(round(float(wb_gains[0]) * 1024))
    b_gain = int(round(float(wb_gains[2]) * 1024))
    ccm_rows = [[int(round(float(v) * 1024)) for v in row] + [0] for row in ccm]

    cfg_dict = {
        "module_enable_status": {
            "dpc": False,  # our noise model has no dead/hot-pixel defects; DPC's
                           # fixed threshold=30 would misfire on ordinary heavy
                           # Gaussian noise at high simulated ISO
            "blc": True,
            "lsc": False,  # our forward model has no vignetting/lens-shading
                           # falloff to correct; applying a radial gain here
                           # would just introduce a distortion not present in
                           # the ground truth, same reasoning as disabling DPC
            "aaf": True, "awb": True, "cnf": True, "cfa": True,
            "ccm": True, "gac": True, "csc": True, "nlm": True,
            "nfc": True,  # chroma noise IS simulated by our forward model,
                          # so this one is a fair, relevant comparison
            "bnf": True,
            "ceh": full_pipeline, "eeh": full_pipeline, "fcs": full_pipeline,
            "hsc": full_pipeline, "bcc": full_pipeline,
            "scl": False,
        },
        "hardware": {
            "raw_width": width, "raw_height": height,
            "raw_bit_depth": RAW_BIT_DEPTH, "bayer_pattern": "rggb",
        },
        "dpc": {"diff_threshold": 30},
        "blc": {"bl_r": 0, "bl_gr": 0, "bl_gb": 0, "bl_b": 0, "alpha": 0, "beta": 0},
        "aaf": None,
        "awb": {"r_gain": r_gain, "gr_gain": 1024, "gb_gain": 1024, "b_gain": b_gain},
        "cnf": {"diff_threshold": 0, "r_gain": r_gain, "b_gain": b_gain},
        "cfa": {"mode": "malvar"},
        "ccm": {"ccm": ccm_rows},
        "gac": {"gain": 256, "gamma": 0.42},
        "csc": None,
        "nlm": {"search_window_size": 9, "patch_size": 3, "h": 10},
        "nfc": {"alpha": 0.3, "thresh": 2.5},
        "bnf": {"intensity_sigma": 0.8, "spatial_sigma": 0.8},
        "ceh": {"tiles": [4, 6], "clip_limit": 0.01},
        "eeh": {"edge_gain": 384, "flat_threshold": 4, "edge_threshold": 8, "delta_threshold": 64},
        "fcs": {"delta_min": 8, "delta_max": 32},
        "hsc": {"hue_offset": 0, "saturation_intensity": 1.0},
        "bcc": {"brightness_offset": 0, "new_max": 235, "new_min": 16},
    }
    return Config(cfg_dict)


def run_traditional_isp(
    bayer_01: np.ndarray, wb_gains: np.ndarray, ccm: np.ndarray, full_pipeline: bool = True
) -> np.ndarray:
    """bayer_01: (H, W) float in [0,1]. Returns (H, W, 3) uint8 sRGB-ish output."""
    height, width = bayer_01.shape
    cfg = build_openisp_config(height, width, wb_gains, ccm, full_pipeline=full_pipeline)
    pipeline = Pipeline(cfg)

    bayer_uint16 = np.round(np.clip(bayer_01, 0.0, 1.0) * RAW_MAX).astype(np.uint16)
    data, _ = pipeline.execute(bayer_uint16, verbose=False)
    return data["output"]  # (H, W, 3) uint8 RGB


def _uint8_hwc_to_tensor01(arr: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def _tensor01_to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    arr = t.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _save_strip(images_uint8_hwc: list[np.ndarray], path: Path) -> None:
    strip = np.concatenate(images_uint8_hwc, axis=1)
    Image.fromarray(strip).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="data_raw/test/kodak")
    p.add_argument("--max-size", type=int, default=384, help="kept modest -- traditional pipeline runs on CPU")
    p.add_argument("--output-dir", type=str, default="outputs/eval_traditional")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-qualitative", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    ds = FullImageTestDataset(ROOT / args.dataset, max_size=args.max_size)
    if args.limit is not None:
        ds.paths = ds.paths[: args.limit]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    dataset_name = Path(args.dataset).name
    out_root = ROOT / args.output_dir / dataset_name

    all_results = {}
    for regime_name, gain_range in NOISE_REGIMES:
        torch.manual_seed(42)  # match neuralisp.evaluate's seeding for comparable numbers

        method_names = ("bilinear", "classical_dd", "traditional_isp_full", "net")
        metrics = {k: {"psnr": [], "ssim": []} for k in method_names}
        regime_dir = out_root / regime_name

        for i, (clean, name) in enumerate(loader):
            clean = clean.to(device)
            deg = degrade(clean, gain_range=gain_range)

            baseline = bilinear_demosaic(deg.packed_bayer)
            pred = demosaic_denoise(model, deg.packed_bayer, deg.noise_map, baseline)
            gt_srgb = render_srgb(deg.target_linear_rgb, deg.wb_gains, deg.ccm)
            base_srgb = render_srgb(baseline, deg.wb_gains, deg.ccm)
            pred_srgb = render_srgb(pred, deg.wb_gains, deg.ccm)

            full_bayer = unpack_rggb(deg.packed_bayer)[0, 0].cpu().numpy()
            wb_gains_np = deg.wb_gains[0].cpu().numpy()
            ccm_np = deg.ccm[0].cpu().numpy()
            # classical_dd: same scope as JointISPNet (demosaic + denoise only, no
            # sharpening/contrast/hue/brightness). traditional_isp_full: the
            # complete product-style pipeline, kept for reference.
            classical_output = run_traditional_isp(full_bayer, wb_gains_np, ccm_np, full_pipeline=False)
            trad_output = run_traditional_isp(full_bayer, wb_gains_np, ccm_np, full_pipeline=True)
            classical_tensor = _uint8_hwc_to_tensor01(classical_output, device)
            trad_tensor = _uint8_hwc_to_tensor01(trad_output, device)

            metrics["bilinear"]["psnr"].append(psnr_metric(base_srgb, gt_srgb))
            metrics["bilinear"]["ssim"].append(ssim_metric(base_srgb, gt_srgb))
            metrics["net"]["psnr"].append(psnr_metric(pred_srgb, gt_srgb))
            metrics["net"]["ssim"].append(ssim_metric(pred_srgb, gt_srgb))
            metrics["classical_dd"]["psnr"].append(psnr_metric(classical_tensor, gt_srgb))
            metrics["classical_dd"]["ssim"].append(ssim_metric(classical_tensor, gt_srgb))
            metrics["traditional_isp_full"]["psnr"].append(psnr_metric(trad_tensor, gt_srgb))
            metrics["traditional_isp_full"]["ssim"].append(ssim_metric(trad_tensor, gt_srgb))

            if i < args.n_qualitative:
                regime_dir.mkdir(parents=True, exist_ok=True)
                strip = [
                    _tensor01_to_uint8_hwc(base_srgb[0]),
                    classical_output,
                    trad_output,
                    _tensor01_to_uint8_hwc(pred_srgb[0]),
                    _tensor01_to_uint8_hwc(gt_srgb[0]),
                ]
                _save_strip(strip, regime_dir / f"{name[0]}.png")

        summary = {method: {"psnr": float(np.mean(v["psnr"])), "ssim": float(np.mean(v["ssim"])),
                             "n_images": len(v["psnr"])}
                   for method, v in metrics.items()}
        all_results[regime_name] = summary

        print(f"[{dataset_name}/{regime_name}] n={summary['bilinear']['n_images']}")
        for method in method_names:
            m = summary[method]
            print(f"  {method:<22} PSNR={m['psnr']:6.2f}dB  SSIM={m['ssim']:.4f}")

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsaved qualitative strips (bilinear | classical demosaic+denoise | "
          f"traditional ISP full | ML | ground truth) and results.json under {out_root}")


if __name__ == "__main__":
    main()
