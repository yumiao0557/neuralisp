"""Evaluation metrics. PSNR/SSIM are necessary-but-insufficient (see README) --
these are the automatable subset; visual/texture-acutance checks are a
separate, manual step for a real production harness."""
from __future__ import annotations

import torch

from neuralisp.losses import ssim as _ssim


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((pred - target) ** 2).clamp(min=1e-12)
    return (10 * torch.log10(max_val**2 / mse)).item()


@torch.no_grad()
def ssim_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    return _ssim(pred, target).item()
