"""
Joint demosaic + denoise network.

Input:  packed RGGB Bayer (4ch) + broadcast noise-level map (2ch: shot, read
        params) at (H/2, W/2) -- 6 channels total.
Output: linear RGB residual added on top of a bilinear-demosaic baseline,
        upsampled back to full (H, W) resolution via PixelShuffle.

Design notes:
  - Demosaic and denoise are estimation problems on the same underlying
    signal (see project README), so they're solved jointly by one network
    rather than as a classical cascade.
  - The network predicts a *residual* on top of a cheap bilinear-demosaic
    baseline. This stabilizes training (the net only has to learn the
    correction, not reinvent interpolation) and gives a sane fallback if
    the residual path is disabled.
  - Conditioning on explicit (shot, read) noise params (rather than
    estimating noise level from the image) lets one model generalize
    across the ISO/gain operating range instead of overfitting to a single
    noise regime -- this is what "one model, not one per ISO" means in
    production.
  - Color, white balance, and tone mapping are deliberately NOT learned
    here -- those stay classical/tunable downstream (see degradation.py
    render_srgb). This network only outputs camera-native linear RGB.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from neuralisp.models.blocks import DownBlock, ResidualBlock, UpBlock


class JointISPNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        base_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        blocks_per_level: int = 2,
        bottleneck_blocks: int = 4,
    ):
        super().__init__()
        chs = [base_channels * m for m in channel_mults]

        self.stem = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.enc_blocks.append(
                nn.Sequential(*[ResidualBlock(chs[i]) for _ in range(blocks_per_level)])
            )
            self.downs.append(DownBlock(chs[i], chs[i + 1]))

        self.bottleneck = nn.Sequential(*[ResidualBlock(chs[-1]) for _ in range(bottleneck_blocks)])

        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(len(chs) - 1)):
            self.ups.append(UpBlock(chs[i + 1], chs[i], chs[i]))
            self.dec_blocks.append(
                nn.Sequential(*[ResidualBlock(chs[i]) for _ in range(blocks_per_level)])
            )

        # final: -> 12 channels at (H/2, W/2), pixel-shuffle to 3ch @ (H, W)
        self.head = nn.Conv2d(chs[0], 3 * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, packed_bayer: torch.Tensor, noise_map: torch.Tensor) -> torch.Tensor:
        """Returns a full-resolution linear-RGB residual, (B, 3, H, W)."""
        x = torch.cat([packed_bayer, noise_map], dim=1)
        x = self.stem(x)

        skips = []
        for enc, down in zip(self.enc_blocks, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            x = up(x, skip)
            x = dec(x)

        x = self.head(x)
        residual = self.shuffle(x)
        return residual

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def demosaic_denoise(
    model: JointISPNet,
    packed_bayer: torch.Tensor,
    noise_map: torch.Tensor,
    baseline: torch.Tensor,
) -> torch.Tensor:
    """Full forward: baseline (bilinear demosaic) + predicted residual, clamped."""
    residual = model(packed_bayer, noise_map)
    return (baseline + residual).clamp(0.0, 1.0)
