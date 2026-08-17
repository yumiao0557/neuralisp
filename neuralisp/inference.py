"""
Single-image inference.

Two modes:
  - Real RAW file (.dng/.arw/.cr2/.nef/.raf/...): read via rawpy, use its
    black-level/white-level/Bayer-pattern metadata to build a normalized
    RGGB packed input, run the network, render to sRGB with rawpy's
    daylight WB + color matrix (or --no-render to dump linear RGB).
    NOTE: this path is best-effort and untested end-to-end -- this project
    was built and validated entirely on synthetic (unprocessed-sRGB) data
    (see README "What's missing for production"). Treat real-RAW output as
    a starting point to debug against your specific sensor's metadata, not
    a validated result.
  - Regular image (.png/.jpg): treated as a clean sRGB reference and run
    through the same synthetic degradation used in training, so you can
    sanity-check the model on an arbitrary photo.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from neuralisp.data.degradation import bilinear_demosaic, degrade, pack_rggb, render_srgb
from neuralisp.evaluate import load_model

RAW_EXTENSIONS = {".dng", ".arw", ".cr2", ".cr3", ".nef", ".raf", ".rw2", ".orf", ".pef"}


def _save_tensor_png(tensor: torch.Tensor, path: Path) -> None:
    arr = tensor.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


def _run_real_raw(path: Path, model, device, out_path: Path, render: bool):
    import rawpy

    with rawpy.imread(str(path)) as raw:
        pattern = raw.raw_pattern
        desc = raw.color_desc.decode()
        # Only handle the common RGGB layout this project's mosaic assumes;
        # other CFA layouts (e.g. GRBG/BGGR) would need pack_rggb generalized.
        idx = {desc[pattern[i, j]]: (i, j) for i in range(2) for j in range(2)}
        if not ({"R", "G", "B"} <= set(idx.keys())) or idx.get("R") != (0, 0) or idx.get("B") != (1, 1):
            print(
                f"warning: CFA pattern {desc} at raw_pattern={pattern.tolist()} is not RGGB; "
                "results will be wrong. This project's mosaic/pack functions assume RGGB."
            )

        raw_img = raw.raw_image_visible.astype(np.float32)
        black = np.array(raw.black_level_per_channel, dtype=np.float32).mean()
        white = float(raw.white_level)
        normalized = np.clip((raw_img - black) / max(white - black, 1.0), 0.0, 1.0)

        h, w = normalized.shape
        h -= h % 2
        w -= w % 2
        normalized = normalized[:h, :w]

        bayer = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0).to(device)
        packed = pack_rggb(bayer)

        # No per-shot noise metadata available cheaply from rawpy; use a
        # conservative mid-range noise-level estimate. For real deployment
        # this should come from ISO -> calibrated (shot_a, read_b) lookup.
        b, _, h2, w2 = packed.shape
        a_map = torch.full((b, 1, h2, w2), 10 ** -2.5, device=device)
        b_map = torch.full((b, 1, h2, w2), 1e-4, device=device)
        noise_map = torch.cat([a_map, b_map], dim=1)

        baseline = bilinear_demosaic(packed)
        with torch.no_grad():
            from neuralisp.models.unet import demosaic_denoise

            pred = demosaic_denoise(model, packed, noise_map, baseline)

        if render:
            wb = np.array(raw.camera_whitebalance[:3], dtype=np.float32)
            wb = wb / wb[1]  # normalize to green=1
            wb_t = torch.tensor(wb, device=device).unsqueeze(0)
            ccm_np = np.array(raw.color_matrix[:3, :3], dtype=np.float32)
            if np.allclose(ccm_np, 0):
                ccm_np = np.eye(3, dtype=np.float32)
            ccm_t = torch.tensor(ccm_np, device=device).unsqueeze(0)
            out_img = render_srgb(pred, wb_t, ccm_t)[0]
        else:
            out_img = pred[0]

        _save_tensor_png(out_img, out_path)
        print(f"saved {out_path}")


def _run_synthetic_demo(path: Path, model, device, out_path: Path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    h -= h % 2
    w -= w % 2
    arr = arr[:h, :w]
    clean = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)

    out = degrade(clean)
    baseline = bilinear_demosaic(out.packed_bayer)
    with torch.no_grad():
        from neuralisp.models.unet import demosaic_denoise

        pred = demosaic_denoise(model, out.packed_bayer, out.noise_map, baseline)

    gt_srgb = render_srgb(out.target_linear_rgb, out.wb_gains, out.ccm)[0]
    base_srgb = render_srgb(baseline, out.wb_gains, out.ccm)[0]
    pred_srgb = render_srgb(pred, out.wb_gains, out.ccm)[0]
    grid = torch.cat([base_srgb, pred_srgb, gt_srgb], dim=2)
    _save_tensor_png(grid, out_path)
    print(f"saved comparison (bilinear | prediction | ground truth) to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--no-render", action="store_true", help="for real RAW: dump linear RGB, skip WB/CCM/gamma")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".neuralisp_out.png")

    if in_path.suffix.lower() in RAW_EXTENSIONS:
        _run_real_raw(in_path, model, device, out_path, render=not args.no_render)
    else:
        _run_synthetic_demo(in_path, model, device, out_path)


if __name__ == "__main__":
    main()
