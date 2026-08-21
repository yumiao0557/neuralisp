"""
Forward degradation model: clean sRGB -> synthetic noisy Bayer RAW.

Since real (RAW, noise-free-ground-truth) pairs are scarce, we use the
"unprocessing" approach (Brooks et al., CVPR 2019, "Unprocessing Images for
Learned Raw Denoising"): take a clean, already-ISP'd sRGB image and invert
the classical pipeline steps (tone curve -> CCM -> white balance -> mosaic)
to land back in a plausible camera-native linear RAW domain, then add a
calibrated Poisson-Gaussian-equivalent noise model.

Pipeline boundary (matches the production architecture this project targets):

    clean sRGB image  --[this module, inverse direction]-->  noisy Bayer RAW
    noisy Bayer RAW    --[neural net, this project]-->        linear RGB (camera-native, pre-WB)
    linear RGB          --[classical, this module, forward dir]--> WB -> CCM -> gamma -> viewable sRGB

The network's supervision target is `cam_linear_noWB`: camera-native linear
RGB, BEFORE white balance and BEFORE color correction. WB/CCM/tone stay
classical and tunable, per the hybrid architecture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# A small bank of real-ish camera RGB -> sRGB color correction matrices
# (approximate values in the spirit of published camera calibration
# matrices, e.g. as used in Brooks et al. / DNG color matrices). We sample
# a random convex combination of these at train time for diversity, then
# use the matrix inverse to go sRGB -> camera-native RGB.
# ---------------------------------------------------------------------------
_CCM_BANK = torch.tensor(
    [
        [[1.8795, -0.8250, -0.0545], [-0.1119, 1.5720, -0.4601], [0.0699, -0.6051, 1.5352]],
        [[1.6083, -0.5599, -0.0484], [-0.1553, 1.5590, -0.4038], [0.0192, -0.4485, 1.4293]],
        [[2.0344, -0.9548, -0.0796], [-0.1968, 1.6903, -0.4936], [0.0292, -0.5232, 1.4940]],
    ],
    dtype=torch.float32,
)


@dataclass
class NoiseParams:
    """Signal-dependent noise: var(I) = shot_a * I + read_b, I in [0, 1]."""

    shot_a: torch.Tensor  # (B,) or scalar
    read_b: torch.Tensor  # (B,) or scalar


def sample_noise_params(batch_size: int, device, gain_range=(-4.0, -1.0), skew_power: float = 1.0) -> NoiseParams:
    """Sample plausible per-image (shot, read) noise params.

    Follows the log-linear shot/read relationship calibrated by Foi et al.
    and reused in Brooks et al.: higher shot noise correlates with higher
    read noise, with some scatter. gain_range is log10(shot_a) range,
    roughly spanning low-ISO to high-ISO operating points.

    skew_power biases sampling toward the high-noise end of gain_range,
    without a hard cutoff: draw u ~ Uniform(0,1), then use u**skew_power
    in place of u. skew_power=1.0 (default) is plain uniform, matching
    prior behavior exactly. skew_power<1.0 pushes u toward 1 (the high-
    noise end of gain_range), giving high-ISO conditions more training
    density -- e.g. skew_power=0.5 roughly doubles the probability mass
    landing in the top 1/6 of the range (~17% -> ~31%). Opt-in only: the
    default leaves every existing call site (evaluate.py, the reference_isp
    comparison scripts, which pass fixed per-regime ranges) unaffected.
    """
    u = torch.rand(batch_size, device=device)
    if skew_power != 1.0:
        u = u**skew_power
    log_a = (gain_range[0] + (gain_range[1] - gain_range[0]) * u) * math.log(10)
    a = torch.exp(log_a)
    log_b = 2.18 * log_a + 1.20 + torch.randn(batch_size, device=device) * 0.26
    b = torch.exp(log_b) * 1e-3  # scale into [0,1]-normalized intensity units
    return NoiseParams(shot_a=a, read_b=b)


def sample_ccm(batch_size: int, device) -> torch.Tensor:
    """Random convex combination of the CCM bank -> (B, 3, 3) sRGB<-cam matrices."""
    bank = _CCM_BANK.to(device)
    weights = torch.rand(batch_size, bank.shape[0], device=device)
    weights = weights / weights.sum(dim=1, keepdim=True)
    ccm = torch.einsum("bk,kij->bij", weights, bank)
    return ccm


def sample_wb_gains(batch_size: int, device) -> torch.Tensor:
    """Random per-channel WB gains (R, G, B), G fixed to 1.0."""
    r_gain = torch.empty(batch_size, device=device).uniform_(1.4, 2.2)
    b_gain = torch.empty(batch_size, device=device).uniform_(1.2, 2.4)
    g_gain = torch.ones(batch_size, device=device)
    return torch.stack([r_gain, g_gain, b_gain], dim=1)  # (B, 3)


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Inverse sRGB gamma (approx piecewise, using the standard formula)."""
    a = 0.055
    thresh = 0.04045
    lin_low = x / 12.92
    lin_high = ((x + a) / (1 + a)).clamp(min=1e-8) ** 2.4
    return torch.where(x <= thresh, lin_low, lin_high)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0.0, max=1.0)
    a = 0.055
    thresh = 0.0031308
    srgb_low = x * 12.92
    srgb_high = (1 + a) * x.clamp(min=1e-8) ** (1 / 2.4) - a
    return torch.where(x <= thresh, srgb_low, srgb_high)


def apply_ccm(rgb: torch.Tensor, ccm: torch.Tensor) -> torch.Tensor:
    """rgb: (B,3,H,W), ccm: (B,3,3) mapping src->dst channels."""
    b, c, h, w = rgb.shape
    flat = rgb.reshape(b, c, h * w)
    out = torch.bmm(ccm, flat)
    return out.reshape(b, c, h, w)


def mosaic_rggb(rgb: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) camera-native linear RGB -> (B,1,H,W) RGGB Bayer mosaic."""
    b, c, h, w = rgb.shape
    assert h % 2 == 0 and w % 2 == 0, "H and W must be even for RGGB mosaic"
    bayer = torch.zeros(b, 1, h, w, device=rgb.device, dtype=rgb.dtype)
    bayer[:, 0, 0::2, 0::2] = rgb[:, 0, 0::2, 0::2]  # R
    bayer[:, 0, 0::2, 1::2] = rgb[:, 1, 0::2, 1::2]  # Gr
    bayer[:, 0, 1::2, 0::2] = rgb[:, 1, 1::2, 0::2]  # Gb
    bayer[:, 0, 1::2, 1::2] = rgb[:, 2, 1::2, 1::2]  # B
    return bayer


def pack_rggb(bayer: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) Bayer -> (B,4,H/2,W/2) packed [R, Gr, Gb, B]."""
    r = bayer[:, :, 0::2, 0::2]
    gr = bayer[:, :, 0::2, 1::2]
    gb = bayer[:, :, 1::2, 0::2]
    b_ = bayer[:, :, 1::2, 1::2]
    return torch.cat([r, gr, gb, b_], dim=1)


def unpack_rggb(packed: torch.Tensor) -> torch.Tensor:
    """(B,4,H/2,W/2) -> (B,1,H,W) Bayer mosaic (inverse of pack_rggb)."""
    b, _, h2, w2 = packed.shape
    bayer = torch.zeros(b, 1, h2 * 2, w2 * 2, device=packed.device, dtype=packed.dtype)
    bayer[:, :, 0::2, 0::2] = packed[:, 0:1]
    bayer[:, :, 0::2, 1::2] = packed[:, 1:2]
    bayer[:, :, 1::2, 0::2] = packed[:, 2:3]
    bayer[:, :, 1::2, 1::2] = packed[:, 3:4]
    return bayer


def add_shot_read_noise(clean: torch.Tensor, noise: NoiseParams) -> torch.Tensor:
    """clean: (B,1,H,W) or (B,4,H,W) in [0,1]. Signal-dependent Gaussian noise."""
    a = noise.shot_a.view(-1, *([1] * (clean.dim() - 1)))
    b = noise.read_b.view(-1, *([1] * (clean.dim() - 1)))
    variance = (a * clean.clamp(min=0.0) + b).clamp(min=1e-8)
    sigma = torch.sqrt(variance)
    return clean + torch.randn_like(clean) * sigma


def bilinear_wb_gains(wb_gains: torch.Tensor) -> torch.Tensor:
    return wb_gains


@dataclass
class DegradationOutput:
    packed_bayer: torch.Tensor  # (B, 4, H/2, W/2) noisy, in [0,1]
    noise_map: torch.Tensor  # (B, 2, H/2, W/2) shot/read params broadcast
    target_linear_rgb: torch.Tensor  # (B, 3, H, W) camera-native, pre-WB, noise-free
    wb_gains: torch.Tensor  # (B, 3) for downstream classical WB
    ccm: torch.Tensor  # (B, 3, 3) cam->sRGB matrix for downstream classical CCM


def degrade(clean_srgb: torch.Tensor, gain_range=(-4.0, -1.0), noise_skew_power: float = 1.0) -> DegradationOutput:
    """Full forward degradation: clean sRGB (B,3,H,W) in [0,1] -> synthetic RAW.

    Returns everything the network needs as input/target, plus the WB/CCM
    parameters a classical post-stage would apply (kept separate, tunable).

    noise_skew_power: see sample_noise_params(). Default 1.0 (plain
    uniform) leaves every existing caller's behavior unchanged.
    """
    device = clean_srgb.device
    b = clean_srgb.shape[0]

    linear = srgb_to_linear(clean_srgb)  # linear, still WB'd + color-corrected (Rec.709)

    ccm = sample_ccm(b, device)  # sRGB <- cam  (i.e. cam -> sRGB)
    ccm_inv = torch.linalg.inv(ccm)  # sRGB -> cam
    cam_linear = apply_ccm(linear, ccm_inv)  # camera-native linear RGB, still WB'd

    wb_gains = sample_wb_gains(b, device)
    cam_linear_noWB = cam_linear / wb_gains.view(b, 3, 1, 1).clamp(min=1e-3)
    cam_linear_noWB = cam_linear_noWB.clamp(0.0, 1.0)

    bayer_clean = mosaic_rggb(cam_linear_noWB)
    noise = sample_noise_params(b, device, gain_range=gain_range, skew_power=noise_skew_power)
    bayer_noisy = add_shot_read_noise(bayer_clean, noise).clamp(0.0, 1.0)

    # simulate ADC quantization (~12-bit)
    levels = 4095.0
    bayer_noisy = torch.round(bayer_noisy * levels) / levels

    packed = pack_rggb(bayer_noisy)

    h2, w2 = packed.shape[-2:]
    a_map = noise.shot_a.view(b, 1, 1, 1).expand(b, 1, h2, w2)
    b_map = noise.read_b.view(b, 1, 1, 1).expand(b, 1, h2, w2)
    noise_map = torch.cat([a_map, b_map], dim=1)

    return DegradationOutput(
        packed_bayer=packed,
        noise_map=noise_map,
        target_linear_rgb=cam_linear_noWB,
        wb_gains=wb_gains,
        ccm=ccm,
    )


def render_srgb(linear_rgb: torch.Tensor, wb_gains: torch.Tensor, ccm: torch.Tensor) -> torch.Tensor:
    """Classical post-stage: camera-native linear RGB -> viewable sRGB.

    linear_rgb: (B,3,H,W) pre-WB camera-native linear (network output or GT)
    """
    b = linear_rgb.shape[0]
    wb = linear_rgb * wb_gains.view(b, 3, 1, 1)
    srgb_linear = apply_ccm(wb, ccm)
    return linear_to_srgb(srgb_linear)


def bilinear_demosaic(packed: torch.Tensor) -> torch.Tensor:
    """Simple bilinear-interpolation demosaic baseline from packed RGGB.

    Used both as the classical baseline for comparison, and as the base
    image the network predicts a residual on top of.
    """
    bayer = unpack_rggb(packed)  # (B,1,H,W)
    b, _, h, w = bayer.shape
    device = bayer.device

    r = torch.zeros(b, 1, h, w, device=device, dtype=bayer.dtype)
    g = torch.zeros(b, 1, h, w, device=device, dtype=bayer.dtype)
    bl = torch.zeros(b, 1, h, w, device=device, dtype=bayer.dtype)

    r[:, :, 0::2, 0::2] = bayer[:, :, 0::2, 0::2]
    g[:, :, 0::2, 1::2] = bayer[:, :, 0::2, 1::2]
    g[:, :, 1::2, 0::2] = bayer[:, :, 1::2, 0::2]
    bl[:, :, 1::2, 1::2] = bayer[:, :, 1::2, 1::2]

    def fill(channel: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        weight = kernel.to(device=device, dtype=channel.dtype).view(1, 1, *kernel.shape)
        pad = kernel.shape[0] // 2
        num = torch.nn.functional.conv2d(channel, weight, padding=pad)
        mask = (channel != 0).float()
        den = torch.nn.functional.conv2d(mask, weight, padding=pad).clamp(min=1e-6)
        filled = num / den
        return torch.where(channel != 0, channel, filled)

    g_kernel = torch.tensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=torch.float32) / 4
    rb_kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 4

    g = fill(g, g_kernel)
    r = fill(r, rb_kernel)
    bl = fill(bl, rb_kernel)

    return torch.cat([r, g, bl], dim=1).clamp(0.0, 1.0)
