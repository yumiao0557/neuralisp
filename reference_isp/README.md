# Traditional ISP reference baseline

A classical (non-ML) ISP pipeline, run on the *exact same* synthetic
noisy-Bayer test inputs as `neuralisp.evaluate`, so `JointISPNet` can be
compared against a real traditional pipeline — not just the plain-bilinear
strawman used elsewhere in this project.

## What's here

- **`fast-openISP/`** — a fork of
  [QiuJueqin/fast-openISP](https://github.com/QiuJueqin/fast-openISP)
  (MIT licensed; forked from commit `26fd824`, 2023-06-21) with local
  additions on top of the original pure-numpy reimplementation of
  [openISP](https://github.com/cruxopen/openISP)'s classical pipeline:
  DPC → BLC → **LSC** → AAF → AWB → CNF → CFA (Malvar demosaic) → CCM → GAC
  (gamma) → CSC → NLM → **NFC** → BNF → CEH → EEH → FCS → HSC → BCC, where
  **LSC** (lens shading correction) and **NFC** (chroma noise reduction, a
  local-neighborhood outlier filter on Cb/Cr) are additions not present
  upstream, and `bcc`/`hsc` were rewritten (min-max stretch contrast and
  floating-point hue/saturation, respectively, in place of the original
  fixed-point x256 gains). Full history lives at
  [yumiao0557/fast-openISP](https://github.com/yumiao0557/fast-openISP).
- **`compare_traditional_isp.py`** — the adapter. For each test image and
  noise regime, it reuses this project's own `degrade()` to produce a noisy
  Bayer mosaic, runs it through fast-openISP with a config matched to this
  project's forward model (see below), and produces a 4-way comparison
  against bilinear / `JointISPNet` / ground truth.
- **`compare_demosaic_denoise.py`** — splits that 4-way comparison into two
  controlled sub-comparisons (demosaic algorithm alone, and denoise alone),
  since the full-pipeline comparison above confounds several things at
  once. See "Demosaic vs. denoise, isolated" below.

## Matched conditions (why this is a fair comparison)

fast-openISP is given the **same per-image white-balance gains and CCM**
that `degrade()` sampled for that image, so its AWB/CCM stages start from
the same color truth as the network and the ground truth — differences in
the result reflect demosaic/denoise/pipeline quality, not incidental color
mismatch. DPC (dead-pixel correction) and LSC (lens shading correction) are
disabled: this project's noise/degradation model has no impulse/dead-pixel
defects and no vignetting/lens-shading falloff, so DPC's fixed threshold=30
would misfire on ordinary heavy Gaussian noise at high simulated ISO, and
LSC's radial gain would introduce a brightness distortion not present in
the ground truth — both would add an unintended extra effect rather than
correcting a real defect. NFC (chroma noise reduction) is left **enabled**,
since chroma noise *is* part of this project's noise model, making it a
fair, relevant comparison — see "Does chroma noise reduction help?" below.

## Run it

```powershell
venv\Scripts\python reference_isp\compare_traditional_isp.py --checkpoint checkpoints\joint_isp_v1\best.pt --dataset data_raw\test\kodak
venv\Scripts\python reference_isp\compare_traditional_isp.py --checkpoint checkpoints\joint_isp_v1\best.pt --dataset data_raw\test\cbsd68
```

Output: `outputs/eval_traditional/<dataset>/results.json` (PSNR/SSIM per
method per regime) and `<regime>/*.png` qualitative strips — each image is
`bilinear | traditional ISP | JointISPNet | ground truth`, left to right.

## Results (24 Kodak images, 68 CBSD68 images, matched conditions)

| Dataset | Regime | Bilinear | Traditional ISP | JointISPNet |
|---|---|---|---|---|
| Kodak | low ISO | 25.18 dB / 0.748 | 23.22 dB / 0.698 | **35.62 dB / 0.967** |
| Kodak | mid ISO | 22.97 dB / 0.539 | 23.09 dB / 0.644 | **32.17 dB / 0.911** |
| Kodak | high ISO | 13.97 dB / 0.143 | 18.90 dB / 0.333 | **22.11 dB / 0.491** |
| CBSD68 | low ISO | 24.19 dB / 0.738 | 20.98 dB / 0.674 | **35.25 dB / 0.973** |
| CBSD68 | mid ISO | 22.28 dB / 0.553 | 20.91 dB / 0.629 | **31.84 dB / 0.922** |
| CBSD68 | high ISO | 14.22 dB / 0.169 | 18.63 dB / 0.367 | **22.05 dB / 0.533** |

*(Traditional ISP now runs with `nfc`, the local chroma-noise-reduction
module, enabled — see "Does chroma noise reduction help?" below for why
these numbers are lower than a run without it, especially at low ISO.)*

**A genuinely interesting result, not just "ML wins": at low and mid ISO,
the traditional pipeline scores *below* plain bilinear**, despite using
Malvar demosaic (strictly better than bilinear on clean data) plus real
denoise stages. Visual inspection of the qualitative strips explains why:
at low noise, demosaic/denoise error is tiny for both, so the dominant
source of error becomes fast-openISP's own tone and denoise response —
its GAC module uses a fixed power-law gamma (0.42) rather than this
project's exact sRGB curve, its EEH (sharpening) and CEH (CLAHE contrast)
stages deliberately push pixel values away from a flat, un-enhanced
rendering, and (see below) `nfc` smooths chroma detail that isn't actually
noise yet at low ISO. All three are real, standard ISP behaviors that
often look better to a human, but they cost PSNR/SSIM against a literal
ground truth. Only at high ISO, once there's real denoising work to do,
does the traditional pipeline's Malvar+NLM+BNF+NFC combo pull ahead of
undenoised bilinear.

**At high ISO, `JointISPNet` still wins by a wide margin** (22.1dB vs
18.9dB, SSIM 0.49 vs 0.33) even though the traditional pipeline has real
denoising stages. Looking at the qualitative strips explains this too: NLM
and BNF in this pipeline only denoise the luma (Y) channel, and `nfc`'s
gain there is modest — chroma noise still passes through more than in the
joint (RGB, not YCbCr) network's output.

## Does chroma noise reduction help?

`nfc` is a local addition to fast-openISP: for each Cb/Cr pixel, it
compares the pixel to its 8-neighbor mean/std and blends toward the
neighborhood mean wherever the deviation exceeds `thresh` standard
deviations (config: `alpha=0.3`, `thresh=2.5`). Isolating its effect means
holding everything else fixed and only toggling `nfc` — comparing against
the pre-fork numbers above does *not* do that, since `bcc`/`hsc` were also
rewritten (see below), so here `nfc` is toggled on the current pipeline
with `bcc`/`hsc` held fixed at their current (rewritten) behavior:

| Dataset | Regime | Traditional ISP, nfc off | Traditional ISP, nfc on | Δ |
|---|---|---|---|---|
| Kodak | low ISO | 23.20 dB / 0.6972 | 23.22 dB / 0.698 | +0.02 dB |
| Kodak | mid ISO | 23.07 dB / 0.6415 | 23.09 dB / 0.644 | +0.02 dB |
| Kodak | high ISO | 18.80 dB / 0.3273 | 18.90 dB / 0.333 | +0.10 dB |
| CBSD68 | low ISO | 20.96 dB / 0.6733 | 20.98 dB / 0.674 | +0.02 dB |
| CBSD68 | mid ISO | 20.90 dB / 0.6275 | 20.91 dB / 0.629 | +0.01 dB |
| CBSD68 | high ISO | 18.56 dB / 0.3615 | 18.63 dB / 0.367 | +0.07 dB |

**Isolated, `nfc`'s effect is small and consistently non-negative** — a
few hundredths of a dB at low/mid ISO, growing to +0.07-0.10dB at high
ISO where there's more real chroma noise for it to catch. That's the
expected shape (more noise -> more genuine work for a denoiser to do), but
the magnitude is modest: NLM/BNF have already done the bulk of the
denoising by the time `nfc` runs, so there isn't much residual chroma
noise left for it to clean up in this pipeline position, and each pass
over 8-neighbor mean/std at `thresh=2.5` is a conservative, small-blend
(`alpha=0.3`) correction by design.

**The multi-dB drop from the pre-fork baseline (e.g. Kodak low ISO 25.04dB
-> 23.22dB, CBSD68 low ISO 24.58dB -> 20.98dB) is not `nfc` — it's the
`bcc` rewrite.** The original `bcc` module, at its config default
`contrast_gain=256`, is a mathematical no-op: `(y - median) * 256 >> 8 ==
y - median`, i.e. exactly identity at that gain. The rewritten `bcc`
always performs a per-image min-max stretch — it reads the image's actual
min/max luma and remaps that range onto `[new_min, new_max] = [16, 235]`
unconditionally, on every image, regardless of whether that image's
dynamic range needed stretching. That's a real, content-dependent
transform where the old default was a no-op, so nearly all of the
regression traces back to it, not to chroma denoising. (`hsc` at its
current defaults, `saturation_intensity=1.0, hue_offset=0`, is close to
identity by construction — `s * dist * cos(theta) == cb` algebraically —
so it isn't a meaningful contributor either.) This is the same lesson as
the GAC/EEH/CEH finding above: fast-openISP's non-demosaic, non-denoise
stages dominate the PSNR/SSIM delta against a literal ground truth far
more than the actual signal-processing stages being compared.

**Caveat**: this is a full, representative product ISP output (with
sharpening/contrast/hue stages `JointISPNet` doesn't attempt), not a
narrow demosaic+denoise-only ablation — that's deliberate, since the point
is "what would a real classical pipeline output," but it means the
PSNR/SSIM gap includes those extra cosmetic stages, not purely demosaic/
denoise fidelity. Read the qualitative strips, not just the table.

## Demosaic vs. end-to-end, isolated -- at every noise level

`compare_demosaic_denoise.py` produces a labeled 6-panel grid **per test
image, per noise regime** (low/mid/high simulated ISO), so both questions
("how good is the demosaic algorithm alone?" and "how good is the full
pipeline end-to-end?") get answered at every noise level on the *same*
noisy input, instead of picking one fixed noise level per question:

- **Row 1 — demosaic comparison**: bilinear | Malvar (fast-openISP's `CFA`
  module, called directly — no WB/CCM/gamma/denoise/sharpening) |
  JointISPNet. All rendered through this project's own `render_srgb()`
  with identical WB/CCM, so the only variable is the demosaic algorithm.
- **Row 2 — end-to-end comparison**: traditional ISP (fast-openISP's full
  pipeline: its own demosaic, NLM/BNF denoise, gamma, sharpening, its own
  rendering) | JointISPNet | ground truth.

**Why JointISPNet's panel is identical in both rows, and that's the
point**: it does joint demosaic+denoise, so it has no separate
"demosaic-only" mode — there's only one output per (image, noise level).
The traditional pipeline's two panels *do* differ once noise is real,
because row 2 adds NLM/BNF denoise on top of the same Malvar demosaic that
row 1 shows in isolation. That asymmetry — one path has a demosaic step
with denoise bolted on after, the other has neither step separately — is
the whole architectural difference this project is about, made visible.

```powershell
venv\Scripts\python reference_isp\compare_demosaic_denoise.py --checkpoint checkpoints\joint_isp_v1\best.pt --dataset data_raw\test\kodak
venv\Scripts\python reference_isp\compare_demosaic_denoise.py --checkpoint checkpoints\joint_isp_v1\best.pt --dataset data_raw\test\cbsd68
```

Output: `outputs/demosaic_denoise_breakdown/<dataset>/<regime>/*.png`
(labeled grids, up to `--n-images` per regime; pass `-1` for all) and
`outputs/demosaic_denoise_breakdown/<dataset>/results.json`.

### Results (full 24 Kodak + 68 CBSD68, all 4 methods vs. ground truth, every regime)

| Dataset | Regime | Bilinear | Malvar (openISP) | Traditional ISP (full) | JointISPNet |
|---|---|---|---|---|---|
| Kodak | low ISO | 25.18 dB / 0.748 | **26.05 dB / 0.845** | 23.22 dB / 0.698 | **35.62 dB / 0.967** |
| Kodak | mid ISO | 22.97 dB / 0.539 | 23.19 dB / 0.633 | 23.09 dB / 0.644 | **32.17 dB / 0.911** |
| Kodak | high ISO | **13.97 dB** / 0.143 | 13.70 dB / 0.197 | 18.90 dB / 0.333 | **22.11 dB / 0.491** |
| CBSD68 | low ISO | 24.19 dB / 0.738 | **25.12 dB / 0.844** | 20.98 dB / 0.674 | **35.25 dB / 0.973** |
| CBSD68 | mid ISO | 22.28 dB / 0.553 | 22.61 dB / 0.658 | 20.91 dB / 0.629 | **31.84 dB / 0.922** |
| CBSD68 | high ISO | **14.22 dB** / 0.169 | 14.12 dB / 0.237 | 18.63 dB / 0.367 | **22.05 dB / 0.533** |

*(Traditional ISP includes `nfc`, whose isolated contribution is small —
see "Does chroma noise reduction help?" above; most of its gap from the
Malvar-alone column traces to the rewritten `bcc` stage, not denoising.)*

Three findings, each visible only because demosaic and end-to-end are
compared at every regime instead of one fixed regime each:

**1. Malvar's advantage over bilinear shrinks and then reverses as noise
grows.** At low ISO Malvar clearly wins (26.05 vs 25.18dB) — its
gradient-corrected kernel reconstructs edges better than plain averaging,
exactly as demosaicing literature predicts. By high ISO, Malvar's PSNR
dips *below* bilinear's (13.70 vs 13.97dB) even though its SSIM stays
higher. The reason is the kernel itself: Malvar's weights include negative
side-lobes (edge-sharpening, visible in `modules/cfa.py`'s
`channel_indices_and_weights`), which amplify high-frequency noise right
along with edges. Bilinear's pure-averaging kernel has no such lobes, so
it incidentally low-pass-filters some noise away — a real, literature-known
sharpness/noise-amplification tradeoff, not a bug in either algorithm.

**2. The full traditional pipeline only clearly pulls ahead of undenoised
demosaic once there's real denoising work to do — and it takes until high
ISO to get there.** At low ISO the full pipeline (23.22dB) scores well
*below* isolated Malvar (26.05dB); at mid ISO it's still marginally below
on PSNR (23.09 vs 23.19dB), though ahead on SSIM (0.644 vs 0.633). Its own
tone curve, sharpening/contrast stages, and the `bcc` rewrite's always-on
per-image contrast stretch (see "Does chroma noise reduction help?" above)
cost more PSNR than its better-than-bilinear demosaic gains it, all the
way through mid ISO. Only at high ISO does NLM+BNF+NFC's actual denoising
finally outweigh that cost (18.90dB vs Malvar's 13.70dB).

**3. `JointISPNet` wins at every regime and every task, by a growing
margin at low noise and a shrinking-but-still-large margin at high
noise** — +9.6dB over the best classical demosaic at low ISO, +3.2dB over
the full traditional pipeline at high ISO. The gap doesn't come from a
better demosaic kernel alone (Malvar is a fine kernel); it comes from
denoising being joint, RGB, and learned, instead of being a separate
luma-only classical filter bolted on afterward.
