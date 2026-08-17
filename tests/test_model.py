import torch

from neuralisp.models.unet import JointISPNet, demosaic_denoise


def test_forward_shape():
    model = JointISPNet(base_channels=16, channel_mults=(1, 2, 4))
    packed = torch.rand(2, 4, 32, 32)  # H/2 x W/2 = 32x32 -> full res 64x64
    noise_map = torch.rand(2, 2, 32, 32)
    out = model(packed, noise_map)
    assert out.shape == (2, 3, 64, 64)


def test_demosaic_denoise_clamped():
    model = JointISPNet(base_channels=16, channel_mults=(1, 2, 4))
    packed = torch.rand(1, 4, 16, 16)
    noise_map = torch.rand(1, 2, 16, 16)
    baseline = torch.rand(1, 3, 32, 32)
    out = demosaic_denoise(model, packed, noise_map, baseline)
    assert out.shape == (1, 3, 32, 32)
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_param_count_reasonable():
    model = JointISPNet()
    n = model.num_params()
    assert 1e5 < n < 5e7, f"unexpected param count: {n}"


def test_divisible_by_downsample_factor():
    # 3-level encoder -> 4x downsample internally beyond the initial /2 packing
    model = JointISPNet(base_channels=8, channel_mults=(1, 2, 4, 8))
    packed = torch.rand(1, 4, 16, 16)
    noise_map = torch.rand(1, 2, 16, 16)
    out = model(packed, noise_map)
    assert out.shape == (1, 3, 32, 32)
