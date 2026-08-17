# neuralisp

A from-scratch neural ISP: joint demosaic + denoise on synthetic Bayer RAW,
trained and evaluated entirely on open-source data. Built to answer a
concrete question — "how would you actually build one of these, and how far
can you get solo" — with working code, not just a diagram.

## Why this scope (demosaic+denoise only, not the whole ISP)

Production neural-ISP systems split into a part that's worth learning and a
part that isn't:

```
RAW (post-BLC/LSC/defect-correction, classical)
   |
   v
[ NEURAL: joint demosaic + denoise ]   <-- this project
   |
   v
linear RGB, camera-native, pre-WB
   |
   v
WB -> CCM -> tone curve -> LTM -> sharpen   (classical, tunable)
```

**Demosaic and denoise are learned, jointly, as one network.** They're
estimation problems on the same underlying signal: demosaic-first correlates
noise spatially/across channels before the denoiser sees it; denoise-first
destroys the high-frequency Bayer structure demosaic needs. A single network
that goes straight from packed Bayer to linear RGB avoids both failure
modes and consistently beats the classical cascade in the literature.

**WB, color correction, and tone mapping stay classical.** Those are product
knobs — "5% warmer," "less shadow contrast" — that need to change on a
product manager's schedule, not a retraining schedule. Baking them into
network weights means every tuning request becomes a data-collection +
retrain + full-regression cycle. Keeping them as parametric blocks fed by
(optionally learned) parameter *estimates* keeps them instantly tunable.

This mirrors why a neural-ISP company still runs an optics team even though
they don't design lenses: the network is trained to invert a specific,
measured optical/sensor degradation. The forward model in
[`neuralisp/data/degradation.py`](neuralisp/data/degradation.py) — noise
calibration, color matrices, white balance — is the thing that determines
whether synthetic training data is realistic. Get the forward model wrong
and the network learns to invert the wrong thing.

## Architecture

`neuralisp/models/unet.py` — `JointISPNet`:

- **Input**: packed RGGB Bayer (4ch, at H/2 x W/2) concatenated with a
  broadcast noise-level map (2ch: shot and read noise params) — 6 channels
  in. Conditioning on *known* noise parameters (derived from ISO/analog
  gain, not estimated from the noisy image) lets one model cover the full
  operating range instead of needing one model per ISO bucket.
- **Body**: U-Net, strided-conv encoder / PixelShuffle decoder, residual
  blocks at each scale, skip connections. Default: 4 levels, channel
  multipliers (1,2,4,8) on a 32-channel base — ~8.6M params.
- **Output**: the network predicts a *residual* on top of a cheap bilinear
  demosaic baseline, PixelShuffled back up to full resolution (H, W, 3).
  Predicting a residual instead of the raw image stabilizes training and
  gives a sane fallback (the baseline alone) if the residual path misbehaves.
- **Loss**: L1 (dominant) + MS-SSIM, computed in a gamma-compressed space —
  linear-space losses over-weight bright regions relative to how error is
  actually perceived. See `neuralisp/losses.py`.

## The forward degradation model (why there's no real paired dataset here)

Real (RAW, clean-ground-truth) pairs at scale are scarce and heavy
(SIDD, DND, HDR+ are tens-to-hundreds of GB with registration/licensing
overhead). Instead this project uses **unprocessing** (Brooks et al., CVPR
2019): take any clean, well-exposed sRGB photo and invert the classical ISP
steps to land back in a plausible sensor-RAW domain:

```
clean sRGB -> inverse gamma -> inverse CCM -> inverse WB -> mosaic (RGGB) -> + calibrated noise
```

Noise is signal-dependent: `var(I) = shot_a * I + read_b`, with `(shot_a,
read_b)` sampled from the log-linear relationship calibrated in Foi et al. /
Brooks et al., spanning roughly low-ISO to high-ISO operating points. CCM
and white-balance gains are randomized per training sample (sampled from a
small bank of real-ish camera color matrices) so the network doesn't overfit
to one sensor's color response.

This is applied **on-the-fly, batched, on GPU**, inside the training loop —
not baked to disk — so every epoch sees a fresh noise/color realization of
the same clean image. See `neuralisp/data/degradation.py`; round-trip and
shape correctness are covered in `tests/test_degradation.py`.

The network's supervision target is the camera-native linear RGB *before*
WB/CCM are re-applied, matching the pipeline boundary above.

## Data

All open-source, fetched by [`scripts/download_data.py`](scripts/download_data.py):

| Split | Source | Count | Role |
|---|---|---|---|
| train | [BSDS500](https://github.com/BIDS/BSDS500) | 502 images | patch source for training (random 128x128 crops, 8/image/epoch) |
| test | [Kodak](https://r0k.us/graphics/kodak/) | 24 images | primary held-out eval (standard demosaicing benchmark) |
| test | [CBSD68](https://github.com/clausmichele/CBSD68-dataset) | 68 images | secondary held-out eval (standard denoising benchmark) |

Re-run `python scripts/download_data.py` any time; it's idempotent (skips
what's already present).

## Project layout

```
neuralisp/
  data/
    degradation.py   forward model: sRGB -> synthetic noisy Bayer RAW
    datasets.py       PatchTrainDataset, FullImageTestDataset
  models/
    blocks.py         residual/down/up blocks
    unet.py           JointISPNet
  losses.py            gamma-space L1 + MS-SSIM
  metrics.py           PSNR, SSIM
  train.py             training loop, checkpointing, TensorBoard
  evaluate.py           held-out eval across simulated ISO regimes
scripts/
  download_data.py     fetch Kodak / CBSD68 / BSDS500
tests/                  unit tests for degradation math, model shapes, losses
data_raw/               downloaded datasets (gitignored)
checkpoints/<run>/      best.pt, latest.pt (gitignored)
runs/<run>/             TensorBoard logs (gitignored)
outputs/<run>/          qualitative PNG samples during training (gitignored)
```

## Usage

```powershell
# one-time setup
python -m venv venv
venv\Scripts\pip install -r requirements.txt
python scripts\download_data.py

# train
venv\Scripts\python -m neuralisp.train --epochs 100 --batch-size 16 --patch-size 128

# evaluate a checkpoint across low/mid/high-ISO regimes, on Kodak and CBSD68
venv\Scripts\python -m neuralisp.evaluate --checkpoint checkpoints\<run>\best.pt --dataset data_raw\test\kodak
venv\Scripts\python -m neuralisp.evaluate --checkpoint checkpoints\<run>\best.pt --dataset data_raw\test\cbsd68

# watch training
venv\Scripts\tensorboard --logdir runs
```

`train.py --max-steps N --limit-val M` runs a fast smoke test instead of a
full run — useful for verifying a change didn't break anything before
committing GPU time.

## Results

100 epochs, `JointISPNet` (8.6M params, base_channels=32), trained on BSDS500
patches, evaluated on two held-out sets it never saw during training, across
three simulated noise regimes (see `NOISE_REGIMES` in `evaluate.py`):

| Dataset | Regime | Net PSNR | Net SSIM | Bilinear PSNR | Bilinear SSIM | Δ PSNR |
|---|---|---|---|---|---|---|
| Kodak (24) | low ISO | 42.67 dB | 0.9888 | 33.79 dB | 0.9313 | +8.88 dB |
| Kodak (24) | mid ISO | 38.69 dB | 0.9662 | 32.37 dB | 0.8730 | +6.32 dB |
| Kodak (24) | high ISO | 29.93 dB | 0.7867 | 23.41 dB | 0.4465 | +6.52 dB |
| CBSD68 (68) | low ISO | 42.26 dB | 0.9905 | 32.87 dB | 0.9188 | +9.39 dB |
| CBSD68 (68) | mid ISO | 37.96 dB | 0.9692 | 31.37 dB | 0.8676 | +6.59 dB |
| CBSD68 (68) | high ISO | 28.97 dB | 0.7891 | 22.92 dB | 0.4755 | +6.06 dB |

The gap over bilinear demosaic grows sharply at high simulated ISO (SSIM
0.79 vs. 0.45-0.48) — qualitatively, the bilinear baseline is heavily
colored-noise-corrupted at that regime while the network output stays
close to ground truth (see `outputs/eval/*/high_iso/*.png`). Performance is
consistent between Kodak and CBSD68 despite CBSD68 being entirely unseen,
which is the generalization check that matters here: nothing about CBSD68
was used for training or hyperparameter selection.

Full per-regime numbers: `outputs/eval/<dataset>/results.json`. Qualitative
triplets (bilinear | prediction | ground truth, rendered to sRGB):
`outputs/eval/<dataset>/<regime>/*.png`.

**Caveat, stated plainly**: PSNR/SSIM on synthetic data validate that the
network correctly inverts *this project's* forward model — they are not a
substitute for real-sensor validation, dead-leaves/acutance texture metrics,
or a human A/B panel, all of which are necessary before anything like this
ships. See "What's missing for production" below.

## The retraining problem

The biggest operational difference from a classical, knob-tuned ISP: a
classical ISP has parameters you can nudge and rebuild in an afternoon; this
network has weights, and a bad result on some scene means **change the
data, retrain, and re-validate everything** — fixing skin tone can regress
foliage. This project's design choices exist specifically to blunt that:

- **Noise-level conditioning** (the 2-channel noise map) means one model
  covers an ISO range instead of needing per-ISO models, so most "it's
  wrong at high ISO" complaints are a data-coverage problem, not an
  architecture problem.
- **WB/CCM/tone kept classical** means color and contrast tuning requests
  never touch the network at all.
- **A fixed, versioned held-out set** (Kodak + CBSD68, evaluated identically
  every run via `evaluate.py`) is the minimum viable regression harness —
  every checkpoint gets the same numbers, so a "fix" that regresses another
  scene class is visible immediately instead of surfacing after ship.

In a real deployment, the missing piece this project doesn't attempt is the
**data-centric loop**: triage a real failure -> capture/synthesize more of
that failure class -> add to train set -> retrain -> re-run the regression
set. That loop, not the architecture, is what production neural-ISP teams
actually spend most of their time on.

## What's missing for production

This is a solo/portfolio-scoped system. Honest gaps between this and a
shippable product:

- **Real sensor data.** Everything here is synthetic (unprocessed sRGB).
  A real deployment finetunes on real (noisy-RAW, clean-reference) pairs
  from the target sensor — e.g. SIDD, or a captured burst-averaged set —
  because no synthetic forward model perfectly matches a real sensor's
  PRNU, defect pixels, or read-noise correlations.
  [`neuralisp/data/datasets.py`](neuralisp/data/datasets.py) is structured
  so a `RealPairedRawDataset` (reading real RAW via `rawpy`, already a
  project dependency) can be dropped in alongside `PatchTrainDataset`
  without touching the model or training loop.
- **Lens-specific degradation.** No PSF/CA/flare modeling here — the
  "optics team" role described above (field-varying MTF, per-unit
  calibration bounds) isn't simulated. Adding a measured, field-varying
  blur kernel to the forward model would be the direct next step and would
  turn this into denoise+demosaic+deblur, still one joint network.
- **Quantization / on-device deployment.** No INT8/QAT export or NPU op
  coverage check. Restoration networks are unusually sensitive to
  quantization because they operate on small residuals — production would
  need per-channel quantization and QAT before targeting a phone NPU.
- **Texture/acutance and perceptual eval.** Only PSNR/SSIM here. A real IQ
  harness needs dead-leaves acutance (catches oversmoothing that PSNR
  rewards), ISO 15739 visual noise, chart+scene SFR, and a blind A/B panel.
- **Video temporal consistency.** Single-frame only; no temporal loss or
  recurrent/burst input.

## Traditional ISP baseline

`reference_isp/` runs a real classical ISP ([fast-openISP](https://github.com/QiuJueqin/fast-openISP),
vendored, MIT licensed) — DPC/BLC/AAF/AWB/CNF/Malvar-demosaic/CCM/gamma/CSC/NLM/BNF/CEH/EEH/FCS/HSC/BCC —
on the exact same synthetic test inputs, for a fair 4-way comparison against
bilinear and `JointISPNet`. Headline result: at high simulated ISO the
traditional pipeline's real denoise stages (NLM/BNF) meaningfully beat plain
bilinear, but `JointISPNet` still wins by a wide margin (22.1dB vs 18.8dB
PSNR on Kodak) because NLM/BNF only denoise luma, leaving visible chroma
noise that the joint RGB network doesn't. See `reference_isp/README.md` for
the full numbers and a genuinely counterintuitive finding at low ISO (the
traditional pipeline scores *below* bilinear there — its own gamma curve and
sharpening/contrast stages, not demosaic quality, turn out to dominate the
error at low noise). `reference_isp/compare_demosaic_denoise.py` isolates
demosaic-only from end-to-end at every noise level (bilinear vs. Malvar vs.
`JointISPNet` demosaic; traditional-ISP vs. `JointISPNet` end-to-end),
confirming Malvar beats bilinear at low ISO once it's not confounded by
the full pipeline's tone curve — and reveals that this flips at high ISO,
where Malvar's edge-sharpening kernel amplifies noise more than bilinear's
plain averaging does.

## Tests

```powershell
venv\Scripts\python -m pytest tests -v
```

Covers: degradation math (mosaic/pack round-trip, noise-free round-trip
error, shape correctness), model forward-pass shapes across configs, and
loss/metric sanity (identical images -> near-zero loss / high PSNR).
