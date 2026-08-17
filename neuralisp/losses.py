"""Losses for training the joint demosaic+denoise network.

L1 dominant, computed in a gamma-compressed (perceptual) space rather than
linear -- linear-space L1/L2 over-weights bright regions and under-weights
shadow detail relative to how errors are actually perceived. MS-SSIM adds
structural sensitivity that pure L1 misses (texture/edge preservation).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gamma_compress(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Simple monotonic compression for loss computation (not display gamma)."""
    return torch.log1p(x.clamp(min=0.0) / eps) / torch.log1p(torch.tensor(1.0 / eps, device=x.device))


def _gaussian_window(window_size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g.outer(g)
    return window_2d


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Single-scale SSIM, averaged over channels/batch. Inputs (B,C,H,W) in [0,1]."""
    device = img1.device
    channels = img1.shape[1]
    window = _gaussian_window(window_size, 1.5, device)
    window = window.expand(channels, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def ms_ssim(img1: torch.Tensor, img2: torch.Tensor, levels: int = 3) -> torch.Tensor:
    """Lightweight multi-scale SSIM (few levels, fine for small training patches)."""
    vals = []
    x1, x2 = img1, img2
    for i in range(levels):
        vals.append(ssim(x1, x2))
        if i < levels - 1:
            x1 = F.avg_pool2d(x1, 2)
            x2 = F.avg_pool2d(x2, 2)
    weights = torch.tensor([0.5, 0.3, 0.2][:levels], device=img1.device)
    weights = weights / weights.sum()
    return sum(w * v for w, v in zip(weights, vals))


class ISPLoss(nn.Module):
    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 0.15):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        pred_c = gamma_compress(pred)
        target_c = gamma_compress(target)

        l1 = F.l1_loss(pred_c, target_c)
        ssim_loss = 1.0 - ms_ssim(pred_c, target_c)

        total = self.l1_weight * l1 + self.ssim_weight * ssim_loss
        return {"total": total, "l1": l1.detach(), "ssim_loss": ssim_loss.detach()}
