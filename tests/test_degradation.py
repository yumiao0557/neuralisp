import math

import torch

from neuralisp.data.degradation import (
    bilinear_demosaic,
    degrade,
    mosaic_rggb,
    pack_rggb,
    render_srgb,
    sample_noise_params,
    unpack_rggb,
)


def test_pack_unpack_roundtrip():
    bayer = torch.rand(2, 1, 16, 16)
    packed = pack_rggb(bayer)
    assert packed.shape == (2, 4, 8, 8)
    recon = unpack_rggb(packed)
    assert torch.allclose(bayer, recon)


def test_mosaic_shape():
    rgb = torch.rand(2, 3, 16, 16)
    bayer = mosaic_rggb(rgb)
    assert bayer.shape == (2, 1, 16, 16)
    # spot check RGGB pattern assignment
    assert torch.allclose(bayer[:, 0, 0, 0], rgb[:, 0, 0, 0])  # R
    assert torch.allclose(bayer[:, 0, 1, 1], rgb[:, 2, 1, 1])  # B


def test_degrade_shapes():
    clean = torch.rand(3, 3, 32, 32).clamp(0.01, 0.99)
    out = degrade(clean)
    assert out.packed_bayer.shape == (3, 4, 16, 16)
    assert out.noise_map.shape == (3, 2, 16, 16)
    assert out.target_linear_rgb.shape == (3, 3, 32, 32)
    assert torch.all(out.packed_bayer >= 0) and torch.all(out.packed_bayer <= 1)


def test_render_srgb_roundtrip_no_noise():
    torch.manual_seed(0)
    clean = torch.rand(2, 3, 32, 32).clamp(0.05, 0.95)
    out = degrade(clean, gain_range=(-8.0, -7.9))  # near-zero noise
    rendered = render_srgb(out.target_linear_rgb, out.wb_gains, out.ccm)
    err = (rendered - clean).abs().mean().item()
    assert err < 0.05, f"round trip error too high: {err}"


def test_bilinear_demosaic_shape():
    packed = torch.rand(2, 4, 8, 8)
    demosaiced = bilinear_demosaic(packed)
    assert demosaiced.shape == (2, 3, 16, 16)


def test_noise_skew_power_default_matches_uniform():
    torch.manual_seed(0)
    gain_range = (-4.0, -1.0)
    n = 20000
    unskewed = sample_noise_params(n, "cpu", gain_range=gain_range, skew_power=1.0)
    log10_a = torch.log(unskewed.shot_a) / math.log(10)
    # plain uniform over a 3.0-wide range: roughly even density everywhere,
    # so the top 1/6 of the range should hold roughly 1/6 of the mass.
    frac_in_high_iso_band = ((log10_a >= -1.5) & (log10_a <= -1.0)).float().mean().item()
    assert 0.12 < frac_in_high_iso_band < 0.22


def test_noise_skew_power_biases_toward_high_noise():
    torch.manual_seed(0)
    gain_range = (-4.0, -1.0)
    n = 20000
    skewed = sample_noise_params(n, "cpu", gain_range=gain_range, skew_power=0.5)
    log10_a = torch.log(skewed.shot_a) / math.log(10)
    frac_in_high_iso_band = ((log10_a >= -1.5) & (log10_a <= -1.0)).float().mean().item()
    # skew_power=0.5 should land roughly double the uniform density (~17%) in
    # the high-ISO band -- expect somewhere around 25-37%.
    assert 0.25 < frac_in_high_iso_band < 0.37
    # every sample must still stay within the requested range
    assert log10_a.min().item() >= gain_range[0] - 1e-4
    assert log10_a.max().item() <= gain_range[1] + 1e-4
