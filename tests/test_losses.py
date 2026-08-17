import torch

from neuralisp.losses import ISPLoss, ms_ssim
from neuralisp.metrics import psnr, ssim_metric


def test_identical_images_low_loss_high_metrics():
    x = torch.rand(2, 3, 32, 32).clamp(0.05, 0.95)
    loss_fn = ISPLoss()
    out = loss_fn(x, x)
    assert out["total"].item() < 1e-4
    assert psnr(x, x) > 80
    assert ssim_metric(x, x) > 0.99


def test_different_images_higher_loss():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 32, 32)
    y = torch.rand(2, 3, 32, 32)
    loss_fn = ISPLoss()
    same = loss_fn(x, x)["total"].item()
    diff = loss_fn(x, y)["total"].item()
    assert diff > same
    assert psnr(x, y) < psnr(x, x)


def test_ms_ssim_range():
    x = torch.rand(1, 3, 32, 32).clamp(0.05, 0.95)
    val = ms_ssim(x, x).item()
    assert 0.9 <= val <= 1.0001
