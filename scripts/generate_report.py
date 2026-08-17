"""
Generate a multi-page PDF explaining the neuralisp architecture and data
flow: which stages are classical/deterministic vs. learned (ML), how
training-time synthetic data flows through the system, how a real
deployment would flow at inference time, and the network's internal
architecture. Also includes actual results from this project's training
run.

Run: venv\\Scripts\\python scripts\\generate_report.py
Output: docs/neuralisp_architecture.pdf
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "docs" / "neuralisp_architecture.pdf"

# ---------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------
CLASSICAL = "#4A6FA5"
CLASSICAL_TXT = "#FFFFFF"
ML = "#D9662B"
ML_TXT = "#FFFFFF"
DATA = "#EDEDED"
DATA_TXT = "#222222"
DATA_BORDER = "#999999"
SYNTH = "#7A5CA3"
SYNTH_TXT = "#FFFFFF"
LOSS = "#B23A48"
LOSS_TXT = "#FFFFFF"
BG = "#FFFFFF"
INK = "#1A1A1A"
MUTED = "#6B6B6B"

FIGSIZE = (11.69, 8.27)  # A4 landscape
PT_PER_UNIT = 72 * FIGSIZE[0] / 100.0  # xlim is always 0..100


def new_page():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    return fig, ax


def _fit_fontsize(lines, w_units, h_units, max_fs, min_fs=6.0, char_w=0.56, line_h=1.35):
    """Largest fontsize (<= max_fs) that keeps `lines` inside a w x h (data-unit) box."""
    if not lines:
        return max_fs
    longest = max(len(l) for l in lines) or 1
    w_pt = w_units * PT_PER_UNIT * 0.86
    h_pt = h_units * PT_PER_UNIT * 0.80
    fs_w = w_pt / (longest * char_w)
    fs_h = h_pt / (len(lines) * line_h)
    return max(min_fs, min(max_fs, fs_w, fs_h))


def wrap_lines(text, width_chars):
    """Wrap text to a target character width, preserving explicit newlines as breaks."""
    out = []
    for para in text.split("\n"):
        if para.strip() == "":
            out.append("")
        else:
            out.extend(textwrap.wrap(para, width=width_chars) or [""])
    return out


def box(ax, x, y, w, h, text, fc, tc, fontsize=11, weight="bold", border=None, autofit=True):
    b = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.35",
        linewidth=1.2, edgecolor=border if border else fc, facecolor=fc,
    )
    ax.add_patch(b)
    lines = text.split("\n")
    fs = _fit_fontsize(lines, w, h, fontsize) if autofit else fontsize
    ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center",
             fontsize=fs, color=tc, weight=weight, linespacing=1.3)
    return b


def note(ax, cx, cy, w, text, color=INK, fontsize=8.4, weight="normal", ha="center", va="top"):
    """Free (non-boxed) text block, manually wrapped to a target width in data units."""
    char_w_pt = fontsize * 0.54
    width_chars = max(10, int((w * PT_PER_UNIT * 0.88) / char_w_pt))
    lines = wrap_lines(text, width_chars)
    ax.text(cx, cy, "\n".join(lines), ha=ha, va=va, fontsize=fontsize,
             color=color, weight=weight, linespacing=1.45)
    return len(lines)


def paragraph_box(ax, x, y, w, h, text, fc, tc, fontsize=10, title_fontsize=None, border=None):
    """Box containing wrapped paragraph text (title line optionally, blank line,
    then body). Unlike box(), this pre-wraps to the target width so long
    paragraphs never overflow the box."""
    b = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.35",
        linewidth=1.2, edgecolor=border if border else fc, facecolor=fc,
    )
    ax.add_patch(b)
    char_w_pt = fontsize * 0.52
    width_chars = max(10, int((w * PT_PER_UNIT * 0.88) / char_w_pt))
    lines = wrap_lines(text, width_chars)
    ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center",
             fontsize=fontsize, color=tc, weight="normal", linespacing=1.55)
    return b


def arrow(ax, xy_from, xy_to, color=INK, lw=1.6, style="-|>", connectionstyle="arc3,rad=0.0", ls="-"):
    a = FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=13,
        color=color, linewidth=lw, connectionstyle=connectionstyle, linestyle=ls,
    )
    ax.add_patch(a)


def header(ax, title, subtitle=None):
    ax.text(4, 95, title, fontsize=18, weight="bold", color=INK, ha="left", va="top")
    if subtitle:
        note(ax, 4, 90.5, 92, subtitle, color=MUTED, fontsize=10.5, ha="left")
    ax.plot([4, 96], [86.5, 86.5], color="#DDDDDD", lw=1)


def footer(ax, page_label):
    ax.text(96, 3, page_label, fontsize=8.5, color=MUTED, ha="right", va="bottom")
    ax.text(4, 3, "neuralisp", fontsize=8.5, color=MUTED, ha="left", va="bottom", style="italic")


def legend_grid(ax, x0, y0, items, cols=2, cell_w=44, cell_h=6.5, fontsize=10):
    for i, (label, color) in enumerate(items):
        col = i % cols
        row = i // cols
        cx = x0 + col * cell_w
        cy = y0 - row * cell_h
        ax.add_patch(FancyBboxPatch((cx, cy), 2.6, 2.6, boxstyle="round,pad=0.08",
                                     facecolor=color, edgecolor=color))
        ax.text(cx + 3.6, cy + 1.3, label, fontsize=fontsize, color=INK, ha="left", va="center")


# ---------------------------------------------------------------------------
# Page 1: title / overview
# ---------------------------------------------------------------------------
def page_title(pdf):
    fig, ax = new_page()
    ax.text(50, 74, "neuralisp", fontsize=44, weight="bold", color=INK, ha="center")
    ax.text(50, 65, "A Neural ISP: Architecture & Data Flow", fontsize=16.5, color=MUTED, ha="center")
    ax.plot([25, 75], [60, 60], color="#DDDDDD", lw=1)

    body = (
        "This document explains how neuralisp works: which pipeline stages are learned "
        "(a neural network) versus classical/deterministic, how data flows through the "
        "system during training and during inference, and the internal architecture of "
        "the network itself.\n\n"
        "Short version: only demosaic + denoise is learned. Everything else (white "
        "balance, color correction, tone mapping) stays classical and tunable, on "
        "purpose \u2014 see page 1 of the pipeline diagram."
    )
    note(ax, 50, 55, 66, body, color=INK, fontsize=11, ha="center", va="top")

    legend_grid(ax, 30, 24, [
        ("Classical / deterministic (tunable)", CLASSICAL),
        ("Learned (neural network)", ML),
        ("Synthetic-data generation (training only)", SYNTH),
        ("Tensor / data", DATA),
    ], cols=1, cell_w=0, cell_h=5.0, fontsize=9.5)

    ax.text(50, 3, "Repo: d:\\neuralisp   |   Open-source data: BSDS500, Kodak-24, CBSD68",
            fontsize=9.5, color=MUTED, ha="center")
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 2: production pipeline, classical vs ML
# ---------------------------------------------------------------------------
def page_pipeline(pdf):
    fig, ax = new_page()
    header(ax, "1. The Pipeline: What's Learned vs. What's Classical",
           "A production neural ISP is mostly classical. Only one stage is a neural network.")

    y = 60
    h = 15
    stages = [
        ("RAW\nsensor\ndata", DATA, DATA_TXT),
        ("BLC / LSC /\ndefect\ncorrection", CLASSICAL, CLASSICAL_TXT),
        ("JOINT\nDEMOSAIC +\nDENOISE\n(neural net)", ML, ML_TXT),
        ("linear RGB\n(camera-\nnative,\npre-WB)", DATA, DATA_TXT),
        ("White\nBalance", CLASSICAL, CLASSICAL_TXT),
        ("Color\nCorrection\n(CCM)", CLASSICAL, CLASSICAL_TXT),
        ("Tone Curve\n+ Local\nTone Map", CLASSICAL, CLASSICAL_TXT),
        ("Sharpen /\noutput\nsRGB", CLASSICAL, CLASSICAL_TXT),
    ]
    n = len(stages)
    w = 10.2
    gap = (94 - n * w) / (n - 1)
    xs = [3 + i * (w + gap) for i in range(n)]
    for (text, fc, tc), x in zip(stages, xs):
        box(ax, x, y, w, h, text, fc, tc, fontsize=10.5)
    for i in range(n - 1):
        arrow(ax, (xs[i] + w, y + h / 2), (xs[i + 1], y + h / 2))

    # why callouts, stacked vertically so they never collide horizontally
    ml_cx = xs[2] + w / 2
    note(ax, ml_cx, y - 3, 24,
         "why ML here: demosaic & denoise are estimation problems on the same signal; "
         "the classical order hurts one or the other.",
         color=ML, fontsize=8.4, weight="bold")

    cls_cx = (xs[4] + xs[7] + w) / 2
    note(ax, cls_cx, y - 3, 30,
         "why classical here: WB / color / tone are product decisions (\"5% warmer\") that "
         "must change on a PM's schedule, not a retrain-and-revalidate schedule.",
         color=CLASSICAL, fontsize=8.4, weight="bold")

    ax.text(50, 29, "Why not learn the whole pipeline end-to-end (RAW \u2192 sRGB in one network)?",
            fontsize=10.5, weight="bold", color=INK, ha="center")
    paragraph_box(ax, 3, 8, 94, 18,
        "Because color/tone controllability would live inside opaque weights. Every tuning request "
        "(\"warmer skin tones\", \"punchier shadows\") would require new training data and a full "
        "retrain-plus-regression cycle instead of changing a parameter. Keeping WB/CCM/tone classical "
        "keeps the ML surface area \u2014 and the retraining cost \u2014 limited to demosaic+denoise, "
        "where learning actually beats classical interpolation (see page 6 for measured numbers).",
        DATA, DATA_TXT, fontsize=10, border=DATA_BORDER)

    footer(ax, "1 / 6")
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 3: training-time data flow
# ---------------------------------------------------------------------------
def page_training_flow(pdf):
    fig, ax = new_page()
    header(ax, "2. Training-Time Data Flow",
           "No real paired RAW dataset is used — clean photos are synthetically degraded on the fly, on GPU.")

    box(ax, 3, 66, 17, 14, "Clean sRGB\nphoto\n(BSDS500)", DATA, DATA_TXT, fontsize=10)
    arrow(ax, (20, 73), (27, 73))

    box(ax, 27, 55, 25, 28,
        "degrade()\n\ninverse gamma\ninverse CCM\ninverse white bal.\nmosaic (RGGB)\n+ shot/read noise",
        SYNTH, SYNTH_TXT, fontsize=9.2)
    ax.text(39.5, 85, "synthetic RAW generation \u2014 training only", fontsize=8.6,
            color=SYNTH, ha="center", weight="bold")

    arrow(ax, (52, 76), (61, 76))
    box(ax, 61, 70, 19, 12, "noisy packed\nBayer (4ch) +\nnoise map (2ch)", DATA, DATA_TXT, fontsize=8.6)

    arrow(ax, (52, 60), (61, 53))
    box(ax, 61, 47, 19, 12, "target linear RGB\n(clean, pre-WB,\ncamera-native)", DATA, DATA_TXT, fontsize=8.3)

    arrow(ax, (70.5, 70), (70.5, 59))

    box(ax, 61, 28, 19, 14, "JointISPNet\n(demosaic +\ndenoise)", ML, ML_TXT, fontsize=10.5)
    arrow(ax, (70.5, 47), (70.5, 42))

    arrow(ax, (80, 35), (88, 35))
    box(ax, 88, 28, 9, 14, "pred", DATA, DATA_TXT, fontsize=9)

    box(ax, 61, 10, 19, 12, "Loss\ngamma-space L1\n+ MS-SSIM", LOSS, LOSS_TXT, fontsize=9)
    arrow(ax, (92, 28), (76, 22), connectionstyle="arc3,rad=-0.25")
    arrow(ax, (61, 47), (72, 22), connectionstyle="arc3,rad=0.35")
    note(ax, 54, 18, 15, "compares\npred vs.\ntarget", color=MUTED, fontsize=7.8)

    arrow(ax, (61, 14), (39.5, 14), color=LOSS, ls="--")
    arrow(ax, (39.5, 14), (39.5, 28), color=LOSS, ls="--")
    arrow(ax, (39.5, 28), (61, 34), color=LOSS, ls="--", connectionstyle="arc3,rad=-0.15")
    note(ax, 39.5, 9, 34,
         "backprop updates network weights only \u2014 everything upstream of the network is fixed math, not learned",
         color=LOSS, fontsize=8)

    footer(ax, "2 / 6")
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 4: inference-time data flow (real deployment)
# ---------------------------------------------------------------------------
def page_inference_flow(pdf):
    fig, ax = new_page()
    header(ax, "3. Inference-Time Data Flow (Real Deployment)",
           "At inference there's no ground truth and no synthetic-degradation step; real sensor RAW goes straight in.")

    y = 62
    h = 16
    w = 14.5
    stages = [
        ("Real camera\nRAW +\nISO/gain\nmetadata", DATA),
        ("BLC / LSC /\ndefect\ncorrection", CLASSICAL),
        ("pack RGGB +\nnoise map\n(ISO\u2192LUT)", CLASSICAL),
        ("JointISPNet\n(trained\nweights)", ML),
        ("linear RGB\n(camera-\nnative)", DATA),
        ("WB \u2192 CCM \u2192\ntone \u2192\nsharpen", CLASSICAL),
        ("sRGB\noutput", DATA),
    ]
    n = len(stages)
    gap = (94 - n * w) / (n - 1)
    xs = [3 + i * (w + gap) for i in range(n)]
    for (text, fc), x in zip(stages, xs):
        tc = ML_TXT if fc == ML else (CLASSICAL_TXT if fc == CLASSICAL else DATA_TXT)
        box(ax, x, y, w, h, text, fc, tc, fontsize=9.6)
    for i in range(n - 1):
        arrow(ax, (xs[i] + w, y + h / 2), (xs[i + 1], y + h / 2))

    paragraph_box(ax, 3, 30, 94, 24,
        "Key difference from training: noise-level conditioning at inference comes from a calibrated "
        "ISO/analog-gain \u2192 (shot_a, read_b) lookup table (built during sensor characterization), not "
        "randomly sampled synthetic parameters. This is the same reason the \"optics team\" matters in a "
        "real neural-ISP company: the network was trained to invert a specific, measured noise/optical "
        "model. If the deployed sensor's real noise characteristics drift from what was characterized "
        "(different module batch, different temperature), the network is being asked to invert a "
        "degradation it wasn't trained on, and quality silently degrades outside that operating range.",
        DATA, DATA_TXT, fontsize=9.6, border=DATA_BORDER)

    note(ax, 50, 24, 80,
         "This project's inference.py implements this path for real RAW files (via rawpy), flagged as "
         "best-effort / untested since no real sensor file was available to validate against.",
         color=MUTED, fontsize=8.8)

    footer(ax, "3 / 6")
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 5: network architecture (U-Net) -- encoder/bottleneck/decoder
# ---------------------------------------------------------------------------
def page_network(pdf):
    fig, ax = new_page()
    header(ax, "4. Network Architecture: JointISPNet",
           "U-Net: strided-conv encoder, residual blocks, PixelShuffle decoder, skip connections. 8.6M parameters.")

    bw, bh = 13, 12

    # input + stem, left edge, vertically centered on the encoder's top row
    box(ax, 3, 72, 12, 12, "input 6ch\n(4 Bayer +\n2 noise map)", DATA, DATA_TXT, fontsize=8)
    arrow(ax, (15, 78), (20, 78))
    box(ax, 20, 72, 9, 12, "stem\nconv", ML, ML_TXT, fontsize=8.5)

    # encoder: 3 levels, stepping down-right
    enc_x = [33, 47, 61]
    enc_y = [72, 58, 44]
    enc_labels = ["enc L0\nC=32, res\u00d72", "enc L1\nC=64, res\u00d72", "enc L2\nC=128, res\u00d72"]
    prev = (29, 78)
    enc_boxes = []
    for x, y, lbl in zip(enc_x, enc_y, enc_labels):
        box(ax, x, y, bw, bh, lbl, ML, ML_TXT, fontsize=8.2)
        arrow(ax, prev, (x, y + bh / 2))
        prev = (x + bw, y + bh / 2)
        enc_boxes.append((x, y))

    # bottleneck
    box(ax, 75, 30, bw, bh, "bottleneck\nC=256, res\u00d74", ML, ML_TXT, fontsize=8.2)
    arrow(ax, prev, (75, 36))

    # decoder: mirror back down-left along the bottom row, all at same y so
    # nothing goes off-page
    dec_x = [61, 47, 33]
    dec_y = 16
    dec_labels = ["dec L2\nup + skip, res\u00d72", "dec L1\nup + skip, res\u00d72", "dec L0\nup + skip, res\u00d72"]
    prevd = (75 + bw / 2, 30)
    for x, lbl in zip(dec_x, dec_labels):
        box(ax, x, dec_y, bw, bh, lbl, ML, ML_TXT, fontsize=8.0)
        arrow(ax, prevd, (x + bw / 2, dec_y + bh))
        prevd = (x + bw / 2, dec_y)

    # skip connections: encoder level i -> matching decoder level i (dashed)
    skip_style = dict(color=MUTED, ls="--", lw=1.1, connectionstyle="arc3,rad=0.2")
    arrow(ax, (enc_x[0] + bw / 2, enc_y[0]), (dec_x[2] + bw / 2, dec_y + bh), **skip_style)
    arrow(ax, (enc_x[1] + bw / 2, enc_y[1]), (dec_x[1] + bw / 2, dec_y + bh), **skip_style)
    arrow(ax, (enc_x[2] + bw / 2, enc_y[2]), (dec_x[0] + bw / 2, dec_y + bh), **skip_style)
    note(ax, 89, 78, 18, "dashed arrows = skip connections (channel-concat before the decoder's residual blocks)",
         color=MUTED, fontsize=8, weight="normal")

    # head + pixelshuffle, output row
    arrow(ax, (33, dec_y), (26, dec_y + bh / 2), connectionstyle="arc3,rad=-0.3")
    box(ax, 12, dec_y, 14, bh, "head conv\n\u2192 12ch @ H/2,W/2", ML, ML_TXT, fontsize=8)
    arrow(ax, (19, dec_y), (19, 6))
    box(ax, 12, -2, 14, 8, "PixelShuffle \u00d72\n\u2192 3ch @ H,W", ML, ML_TXT, fontsize=7.6)

    footer(ax, "4 / 6")
    pdf.savefig(fig)
    plt.close(fig)


def page_network_bottom(pdf):
    fig, ax = new_page()
    header(ax, "4b. Output Head: Residual-on-Baseline", "The network predicts a correction, not the image from scratch.")

    box(ax, 5, 64, 20, 14, "packed RGGB\nBayer (4ch)", DATA, DATA_TXT, fontsize=9.5)
    arrow(ax, (25, 71), (35, 71))
    box(ax, 35, 64, 22, 14, "bilinear_demosaic()\ncheap classical\ninterpolation", CLASSICAL, CLASSICAL_TXT, fontsize=8.8)
    arrow(ax, (57, 71), (67, 71))
    box(ax, 67, 64, 22, 14, "baseline linear RGB\n(soft, some noise)", DATA, DATA_TXT, fontsize=8.8)

    box(ax, 5, 42, 20, 14, "packed Bayer +\nnoise map (6ch)", DATA, DATA_TXT, fontsize=8.8)
    arrow(ax, (25, 49), (35, 49))
    box(ax, 35, 42, 22, 14, "JointISPNet\n(full U-Net,\npage 4)", ML, ML_TXT, fontsize=9.5)
    arrow(ax, (57, 49), (67, 49))
    box(ax, 67, 42, 22, 14, "predicted\nresidual (3ch)", DATA, DATA_TXT, fontsize=8.8)

    arrow(ax, (78, 64), (78, 56))
    arrow(ax, (78, 42), (78, 34))
    box(ax, 67, 20, 22, 12, "baseline + residual\n\u2192 clamp[0,1]", LOSS, LOSS_TXT, fontsize=9)
    note(ax, 78, 18, 40, "= final linear RGB output of the network", color=INK, fontsize=8.8)

    paragraph_box(ax, 4, 3, 92, 12,
        "Why a residual instead of predicting the image outright? It stabilizes training (the network "
        "only has to learn the correction on top of a sane interpolation) and gives a graceful fallback: "
        "if the residual path misbehaves, the output degrades toward the classical bilinear baseline, not garbage.",
        DATA, DATA_TXT, fontsize=9.4, border=DATA_BORDER)

    footer(ax, "5 / 6")
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 6: summary table + measured results
# ---------------------------------------------------------------------------
def page_summary(pdf):
    fig, ax = new_page()
    header(ax, "5. Summary: ML vs. Classical, and Measured Results")

    rows = [
        ("Stage", "Classical or ML?", "Why", True),
        ("BLC / LSC / defect correction", "Classical", "per-unit sensor calibration, not a learning problem", False),
        ("Demosaic + Denoise", "ML (JointISPNet)", "joint estimation problem; classical cascade order hurts either way", False),
        ("White balance", "Classical", "instant product/scene tuning, no retrain needed", False),
        ("Color correction (CCM)", "Classical", "camera color response is measured, not learned", False),
        ("Tone curve / local tone mapping", "Classical", "product-level look; must be a same-day knob", False),
        ("Sharpening", "Classical", "tunable per scene mode", False),
    ]
    y0 = 76
    rh = 6.2
    colx = [4, 44, 63]
    for i, (a, b, c, is_header) in enumerate(rows):
        y = y0 - i * rh
        fc = "#EFEFEF" if is_header else (BG if i % 2 else "#FAFAFA")
        ax.add_patch(FancyBboxPatch((3, y - rh + 2.0), 93, rh - 0.4, boxstyle="square,pad=0", facecolor=fc, edgecolor="none"))
        b_color = ML if ("ML" in b) else INK
        ax.text(colx[0], y, a, fontsize=9.2 if not is_header else 9.8, weight="bold" if is_header else "normal", color=INK, va="center")
        ax.text(colx[1], y, b, fontsize=9.2 if not is_header else 9.8, weight="bold", color=b_color, va="center")
        note(ax, colx[2], y, 33, c, color=MUTED if not is_header else INK, fontsize=8.2,
             weight="bold" if is_header else "normal", ha="left", va="center")
    ax.plot([3, 96], [y0 - rh + 2.0, y0 - rh + 2.0], color="#BBBBBB", lw=1)

    ax.text(4, 27, "Measured results (100 epochs, evaluated on data held out from training)", fontsize=11, weight="bold", color=INK)
    results_path = ROOT / "outputs" / "eval" / "kodak" / "results.json"
    cb_path = ROOT / "outputs" / "eval" / "cbsd68" / "results.json"
    lines = []
    for name, p in [("Kodak-24", results_path), ("CBSD68 (68, unseen)", cb_path)]:
        if p.exists():
            data = json.loads(p.read_text())
            for regime, r in data.items():
                delta = r["pred_psnr"] - r["baseline_psnr"]
                lines.append(
                    f"{name:<22}{regime:<11}net {r['pred_psnr']:5.2f} dB / {r['pred_ssim']:.3f} SSIM"
                    f"    bilinear {r['baseline_psnr']:5.2f} dB / {r['baseline_ssim']:.3f} SSIM"
                    f"    \u0394 {delta:+.2f} dB"
                )
    if not lines:
        lines = ["(run neuralisp.evaluate to populate outputs/eval/*/results.json)"]
    ax.text(4, 22, "\n".join(lines), fontsize=8.4, color=INK, family="monospace", va="top", linespacing=2.0)

    footer(ax, "6 / 6")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PATH) as pdf:
        page_title(pdf)
        page_pipeline(pdf)
        page_training_flow(pdf)
        page_inference_flow(pdf)
        page_network(pdf)
        page_network_bottom(pdf)
        page_summary(pdf)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
