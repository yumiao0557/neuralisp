# Traditional ISP reference baseline

A classical (non-ML) ISP pipeline, run on the *exact same* synthetic
noisy-Bayer test inputs as `neuralisp.evaluate`, so `JointISPNet` can be
compared against a real traditional pipeline — not just the plain-bilinear
strawman used elsewhere in this project.

## What's here

- **`fast-openISP/`** — vendored, unmodified source of
  [QiuJueqin/fast-openISP](https://github.com/QiuJueqin/fast-openISP)
  (MIT licensed; commit `26fd824`, 2023-06-21). A pure-numpy reimplementation
  of [openISP](https://github.com/cruxopen/openISP)'s classical pipeline:
  DPC → BLC → AAF → AWB → CNF → CFA (Malvar demosaic) → CCM → GAC (gamma) →
  CSC → NLM → BNF → CEH → EEH → FCS → HSC → BCC. Its `.git` history was
  stripped since it's vendored as plain source, not a submodule.
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
mismatch. DPC (dead-pixel correction) is disabled: this project's noise
model has no impulse/dead-pixel defects, and DPC's fixed threshold=30
would misfire on ordinary heavy Gaussian noise at high simulated ISO,
muddying the comparison with an unintended extra denoising effect.

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
| Kodak | low ISO | 25.18 dB / 0.748 | 25.04 dB / 0.707 | **35.62 dB / 0.967** |
| Kodak | mid ISO | 22.97 dB / 0.539 | 24.52 dB / 0.646 | **32.17 dB / 0.911** |
| Kodak | high ISO | 13.97 dB / 0.143 | 18.80 dB / 0.324 | **22.11 dB / 0.491** |
| CBSD68 | low ISO | 24.19 dB / 0.738 | 24.58 dB / 0.703 | **35.25 dB / 0.973** |
| CBSD68 | mid ISO | 22.28 dB / 0.553 | 24.09 dB / 0.645 | **31.84 dB / 0.922** |
| CBSD68 | high ISO | 14.22 dB / 0.169 | 18.79 dB / 0.357 | **22.05 dB / 0.533** |

**A genuinely interesting result, not just "ML wins": at low ISO, the
traditional pipeline scores *below* plain bilinear**, despite using Malvar
demosaic (strictly better than bilinear on clean data) plus real denoise
stages. Visual inspection of the qualitative strips explains why: at low
noise, demosaic/denoise error is tiny for both, so the dominant source of
error becomes fast-openISP's own tone response — its GAC module uses a
fixed power-law gamma (0.42) rather than this project's exact sRGB curve,
and its EEH (sharpening) and CEH (CLAHE contrast) stages deliberately push
pixel values away from a flat, un-enhanced rendering. Both are real,
standard ISP behaviors that often look better to a human, but they cost
PSNR/SSIM against a literal ground truth. Only once real denoising work is
needed (mid/high ISO) does the traditional pipeline's Malvar+NLM+BNF combo
pull ahead of undenoised bilinear.

**At high ISO, `JointISPNet` still wins by a wide margin** (22.1dB vs
18.8dB, SSIM 0.49 vs 0.32) even though the traditional pipeline has real
denoising stages. Looking at the qualitative strips explains this too: NLM
and BNF in this pipeline only denoise the luma (Y) channel — chroma noise
passes through largely uncorrected, visible as color speckle in the
traditional-ISP output that the joint (RGB, not YCbCr) network doesn't
have.

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
| Kodak | low ISO | 25.18 dB / 0.748 | **26.05 dB / 0.845** | 25.04 dB / 0.707 | **35.62 dB / 0.967** |
| Kodak | mid ISO | 22.97 dB / 0.539 | 23.19 dB / 0.633 | 24.52 dB / 0.646 | **32.17 dB / 0.911** |
| Kodak | high ISO | **13.97 dB** / 0.143 | 13.70 dB / **0.197** | 18.80 dB / 0.324 | **22.11 dB / 0.491** |
| CBSD68 | low ISO | 24.19 dB / 0.738 | **25.12 dB / 0.844** | 24.58 dB / 0.703 | **35.25 dB / 0.973** |
| CBSD68 | mid ISO | 22.28 dB / 0.553 | 22.61 dB / 0.658 | 24.09 dB / 0.645 | **31.84 dB / 0.922** |
| CBSD68 | high ISO | **14.22 dB** / 0.169 | 14.12 dB / **0.237** | 18.79 dB / 0.357 | **22.05 dB / 0.533** |

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

**2. The full traditional pipeline only pulls ahead of undenoised
demosaic once there's real denoising work to do.** At low ISO the full
pipeline (25.04dB) scores *below* isolated Malvar (26.05dB) — its own
tone curve and sharpening/contrast stages cost more PSNR than its
better-than-bilinear demosaic gains it (see the caveat above). By mid/high
ISO, NLM+BNF's actual denoising more than pays for that tone-curve cost,
and the full pipeline clearly beats both isolated demosaic algorithms.

**3. `JointISPNet` wins at every regime and every task, by a growing
margin at low noise and a shrinking-but-still-large margin at high
noise** — +9.6dB over the best classical demosaic at low ISO, +3.3dB over
the full traditional pipeline at high ISO. The gap doesn't come from a
better demosaic kernel alone (Malvar is a fine kernel); it comes from
denoising being joint, RGB, and learned, instead of being a separate
luma-only classical filter bolted on afterward.
