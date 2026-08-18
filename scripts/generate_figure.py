"""
A single, publication-style "Figure 1" of JointISPNet, in the visual
grammar of papers like the original U-Net (Ronneberger et al. 2015):
box height encodes spatial resolution, box width/thickness encodes
channel depth, color-coded arrows mark each operation type, plain white
background for print/portfolio use.

Outputs (docs/): jointispnet_figure.svg, .png (300dpi), .pdf

Run: venv\\Scripts\\python scripts\\generate_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Patch
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs"

INK = "#1A1A1A"
MUTED = "#5A5A5A"
BLOCK = "#3B5B92"        # residual-block bars
BLOCK_EDGE = "#22334F"
BOTTLENECK = "#2E4772"
DOWN = "#B54A3F"          # downsample arrows
UP = "#2F8F5B"            # upsample arrows
SKIP = "#8A8A8A"          # skip-connection arcs
RESID = "#1A1A1A"         # additive-residual arc
BG = "#FFFFFF"

plt.rcParams["font.family"] = "DejaVu Sans"


def bar(ax, x, height, width, color, edgecolor, y0=0.0, alpha=1.0, hatch=None):
    """A vertical bar centered on x, sitting on baseline y0, going up by `height`."""
    rect = Rectangle((x - width / 2, y0), width, height,
                      facecolor=color, edgecolor=edgecolor, linewidth=1.4,
                      alpha=alpha, hatch=hatch, zorder=3)
    ax.add_patch(rect)
    return rect


def op_arrow(ax, x0, x1, y, color, lw=2.2, style="-|>"):
    a = FancyArrowPatch((x0, y), (x1, y), arrowstyle=style, mutation_scale=16,
                         color=color, linewidth=lw, zorder=4)
    ax.add_patch(a)


def skip_arc(ax, x0, x1, y_top, height, color, ls="--", lw=1.6, label=None, label_y=None):
    verts = [(x0, y_top), (x0, y_top + height), (x1, y_top + height), (x1, y_top)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
    path = MPath(verts, codes)
    patch = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=13,
                             color=color, linewidth=lw, linestyle=ls, zorder=2)
    ax.add_patch(patch)
    if label:
        ax.text((x0 + x1) / 2, label_y if label_y else y_top + height + 0.12, label,
                ha="center", va="bottom", fontsize=9.5, color=color, style="italic")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15.5, 7.2), dpi=300)
    ax.set_xlim(0, 15.2)
    ax.set_ylim(-2.9, 5.6)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ---- stage geometry: (x, height, width, label, channels, res_label, sub) ----
    stages = [
        (0.7,  1.55, 0.30, "input",     "6",   "H/2×W/2",  None),
        (2.15, 3.10, 0.55, "enc L0",    "32",  "H/2×W/2",  "res×2"),
        (3.75, 2.45, 0.75, "enc L1",    "64",  "H/4×W/4",  "res×2"),
        (5.35, 1.90, 1.00, "enc L2",    "128", "H/8×W/8",  "res×2"),
        (6.95, 1.40, 1.30, "bottleneck","256", "H/16×W/16","res×4"),
        (8.55, 1.90, 1.00, "dec L2",    "128", "H/8×W/8",  "res×2"),
        (10.15,2.45, 0.75, "dec L1",    "64",  "H/4×W/4",  "res×2"),
        (11.75,3.10, 0.55, "dec L0",    "32",  "H/2×W/2",  "res×2"),
        (13.35,3.75, 0.28, "output",    "3",   "H×W",      None),
    ]

    baseline = 0.0
    for i, (x, h, w, label, ch, res, sub) in enumerate(stages):
        is_end = label in ("input", "output")
        color = "none" if is_end else (BOTTLENECK if label == "bottleneck" else BLOCK)
        edge = MUTED if is_end else BLOCK_EDGE
        bar(ax, x, h, w, color, edge, y0=baseline)
        ax.text(x, baseline + h + 0.22, ch, ha="center", va="bottom",
                 fontsize=12.5, color=INK, fontweight="bold")
        ax.text(x, baseline - 0.20, res, ha="center", va="top",
                 fontsize=9.5, color=MUTED)
        ax.text(x, baseline - 0.46, label, ha="center", va="top",
                 fontsize=10.5, color=INK)
        if sub:
            ax.text(x, baseline - 0.68, sub, ha="center", va="top",
                     fontsize=8.5, color=MUTED, style="italic")

    # ---- main-flow arrows between stages ----
    arrow_specs = [
        (0, 1, "grey", "pack\n(pixel-unshuffle)"),
        (1, 2, DOWN, "↓2  stride-2 conv"),
        (2, 3, DOWN, "↓2  stride-2 conv"),
        (3, 4, DOWN, "↓2  stride-2 conv"),
        (4, 5, UP, "↑2  PixelShuffle"),
        (5, 6, UP, "↑2  PixelShuffle"),
        (6, 7, UP, "↑2  PixelShuffle"),
        (7, 8, UP, "↑2  PixelShuffle\n(head conv)"),
    ]
    y_arrow = 1.85
    for i0, i1, color, label in arrow_specs:
        x0 = stages[i0][0] + stages[i0][2] / 2 + 0.06
        x1 = stages[i1][0] - stages[i1][2] / 2 - 0.06
        c = MUTED if color == "grey" else color
        op_arrow(ax, x0, x1, y_arrow, c)
        ax.text((x0 + x1) / 2, y_arrow + 0.16, label, ha="center", va="bottom",
                 fontsize=8, color=c)

    # ---- skip connections (concat), arcs above ----
    skip_arc(ax, stages[1][0], stages[7][0], 3.55, 0.55, SKIP,
             label="skip: concat", label_y=4.28)
    skip_arc(ax, stages[2][0], stages[6][0], 2.90, 0.85, SKIP)
    skip_arc(ax, stages[3][0], stages[5][0], 2.35, 1.05, SKIP)

    # ---- additive residual (baseline), arc below ----
    verts = [(stages[0][0], -0.9), (stages[0][0], -1.85),
             (stages[8][0], -1.85), (stages[8][0], -0.55)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
    path = MPath(verts, codes)
    patch = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=15,
                             color=RESID, linewidth=1.8, linestyle=(0, (1, 3)), zorder=2)
    ax.add_patch(patch)
    ax.text((stages[0][0] + stages[8][0]) / 2, -2.05,
             "bilinear_demosaic(input)  —  additive baseline residual",
             ha="center", va="top", fontsize=9.5, color=RESID, style="italic")

    plus_x, plus_y = stages[8][0] - 0.55, -0.15
    circ = Circle((plus_x, plus_y), 0.16, facecolor="white", edgecolor=RESID, linewidth=1.6, zorder=5)
    ax.add_patch(circ)
    ax.text(plus_x, plus_y, "+", ha="center", va="center", fontsize=13, color=RESID,
             fontweight="bold", zorder=6)
    op_arrow(ax, plus_x + 0.16, stages[8][0] - stages[8][2] / 2 - 0.05, plus_y, RESID, lw=1.6)

    # ---- title + legend ----
    ax.text(7.75, 5.35, "JointISPNet", ha="center", va="top",
             fontsize=20, color=INK, fontweight="bold", fontfamily="DejaVu Sans")
    ax.text(7.75, 4.92, "joint demosaicking + denoising of Bayer RAW input",
             ha="center", va="top", fontsize=12, color=MUTED, style="italic")

    legend_handles = [
        Patch(facecolor=BLOCK, edgecolor=BLOCK_EDGE, label="residual block (conv 3×3, LeakyReLU ×2)"),
        Line2D([0], [0], color=DOWN, lw=2.4, marker=">", markersize=7, label="downsample — stride-2 conv"),
        Line2D([0], [0], color=UP, lw=2.4, marker=">", markersize=7, label="upsample — PixelShuffle ×2"),
        Line2D([0], [0], color=SKIP, lw=1.8, ls="--", label="skip connection — channel concat"),
        Line2D([0], [0], color=RESID, lw=1.8, ls=(0, (1, 2)), label="additive residual"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=5, frameon=False, fontsize=9.5, labelcolor=MUTED,
              handlelength=1.8, columnspacing=1.6, handletextpad=0.6)

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("svg", "pdf", "png"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(OUT_DIR / f"jointispnet_figure.{ext}", facecolor=BG,
                     bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'jointispnet_figure.svg'} (+ .png @300dpi, .pdf)")


if __name__ == "__main__":
    main()
