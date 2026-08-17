import torch

from neuralisp.data.degradation import (
    bilinear_demosaic,
    degrade,
    mosaic_rggb,
    pack_rggb,
    render_srgb,
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
