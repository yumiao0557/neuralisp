import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return x + y


class DownBlock(nn.Module):
    """Strided conv downsample, channel doubling."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class UpBlock(nn.Module):
    """PixelShuffle-based 2x upsample, then fuse with skip connection."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.pre = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.pre(x))
        x = self.shuffle(x)
        x = torch.cat([x, skip], dim=1)
        return self.act(self.fuse(x))
