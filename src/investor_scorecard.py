"""Investor-facing performance scorecard for the diamond-grading-from-video system.

    python src/investor_scorecard.py

Renders a single clean dashboard: headline KPIs, per-attribute accuracy bars
(color-coded by readiness tier), the cut-geometry precision panel with this
cycle's gains, an honest "frontier" roadmap, and a strip of real dossier outputs.
Output: data/processed/demo/investor_scorecard.jpg
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.image as mpimg

OUT = Path("data/processed/demo")
DOSSIER = Path("data/processed/dossier")
INK = "#16263a"
GREEN = "#1f9d55"
TEAL = "#2a9d8f"
AMBER = "#e09f3e"
RED = "#c1503f"
GREY = "#9aa7b4"
BG = "#ffffff"
PANEL = "#f4f6f9"

# ---- benchmark numbers (held-out; see RESULTS) ----
ACC = [  # (label, value %, tier color, note)
    ("Shape", 99.3, GREEN, "9 cuts"),
    ("Eye-clean", 89.8, GREEN, ""),
    ("Color  (3-tier)", 87.0, GREEN, "D-F / G-H / I-J"),
    ("Inclusion recall", 91.0, TEAL, "flaw caught"),
    ("Clarity  (±1 grade ≈ 100%)", 69.5, AMBER, "3-class exact"),
    ("Fluorescence", 44.8, RED, "needs UV light"),
]
GEOM = [  # (label, old, new, unit)
    ("Depth %", 0.54, 0.36),
    ("Table %", 0.86, 0.61),
    ("Crown / pav angle", None, 0.34),
    ("L / W ratio", None, 0.008),
]
EXAMPLES = ["VSFEG016_dossier.jpg", "ZSMOU199_dossier.jpg", "ZSMRN010_dossier.jpg"]


def chip(ax, x, y, w, h, text, fc, tc="white", fs=12, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                linewidth=0, facecolor=fc, transform=ax.transAxes, clip_on=False))
    ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold" if bold else "normal")


def main():
    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.62, 1.15, 0.95],
                          width_ratios=[1.35, 1.0], hspace=0.34, wspace=0.16,
                          left=0.055, right=0.965, top=0.93, bottom=0.045)

    # ---------- header ----------
    hd = fig.add_subplot(gs[0, :]); hd.axis("off")
    hd.text(0.0, 0.74, "Diamond Grading from Video", transform=hd.transAxes,
            fontsize=30, fontweight="bold", color=INK)
    hd.text(0.0, 0.40, "A full GIA-style grade — shape, color, clarity, cut geometry, 3D & inclusions — "
            "read from a 360° video alone.", transform=hd.transAxes, fontsize=13.5, color="#4a5b6b")
    for i, (txt, fc) in enumerate([("10,497 certified stones", INK), ("GIA ground truth", TEAL),
                                   ("predictions from video only", GREEN), ("0 human input", "#3d5a73")]):
        chip(hd, 0.0 + i * 0.205, 0.0, 0.19, 0.24, txt, fc, fs=11.5)

    # ---------- accuracy bars ----------
    ab = fig.add_subplot(gs[1, 0]); ab.set_facecolor(BG)
    labels = [a[0] for a in ACC]
    ypos = list(range(len(ACC)))[::-1]
    for (lab, val, col, note), y in zip(ACC, ypos):
        ab.barh(y, val, color=col, height=0.6, zorder=3)
        ab.barh(y, 100, color=PANEL, height=0.6, zorder=1)
        ab.text(val + 1.5, y, f"{val:.1f}%", va="center", ha="left", fontsize=13,
                fontweight="bold", color=col)
        if note:                                   # note inside the bar, white
            ab.text(1.8, y, note, va="center", ha="left", fontsize=8.5,
                    color="white", style="italic", zorder=4)
    ab.axvline(85, color="#b9c4cf", lw=1.2, ls=(0, (5, 4)), zorder=2)
    ab.text(85, len(ACC) - 0.35, " deployable", color="#7c8a98", fontsize=9, va="bottom")
    ab.set_yticks(ypos); ab.set_yticklabels(labels, fontsize=12, color=INK)
    ab.set_xlim(0, 108); ab.set_ylim(-0.6, len(ACC) - 0.3)
    ab.set_xticks([0, 25, 50, 75, 100]); ab.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=9, color="#7c8a98")
    for s in ["top", "right", "left"]:
        ab.spines[s].set_visible(False)
    ab.tick_params(length=0)
    ab.set_title("Per-attribute accuracy on held-out stones", fontsize=13.5,
                 fontweight="bold", color=INK, loc="left", pad=10)

    # ---------- geometry precision panel ----------
    gp = fig.add_subplot(gs[1, 1]); gp.axis("off")
    gp.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.0,rounding_size=0.03",
                                facecolor=PANEL, edgecolor="none", transform=gp.transAxes))
    gp.text(0.06, 0.9, "Cut geometry — the hard 3D part", fontsize=13.5, fontweight="bold", color=INK)
    gp.text(0.06, 0.82, "average error vs GIA cert (lower is better)", fontsize=10, color="#7c8a98")
    y = 0.66
    for lab, old, new in GEOM:
        gp.text(0.06, y, lab, fontsize=11.5, color=INK, va="center")
        unit = "" if lab.startswith("L") else ("%" if "%" in lab else "°")
        gp.text(0.62, y, f"±{new:g}{unit}", fontsize=15, fontweight="bold", color=GREEN, va="center")
        if old is not None:
            gp.text(0.80, y, f"was ±{old:g}{unit}", fontsize=9.5, color="#9aa7b4", va="center")
        y -= 0.135
    chip(gp, 0.06, 0.04, 0.5, 0.11, "depth error  −33%  this cycle", GREEN, fs=11)
    gp.text(0.60, 0.095, "+3.4× training data", fontsize=10, color="#4a5b6b", va="center")

    # ---------- examples strip ----------
    for i, fname in enumerate(EXAMPLES):
        ax = fig.add_subplot(gs[2, :].subgridspec(1, 3, wspace=0.04)[0, i])
        p = DOSSIER / fname
        if p.exists():
            ax.imshow(mpimg.imread(p))
        ax.axis("off")
        if i == 1:
            ax.set_title("One video in  →  a complete digital cert out  (grade · 3D reconstruction · GIA proportions · clarity map)",
                         fontsize=12, color=INK, fontweight="bold", pad=8)

    fig.text(0.5, 0.012, "Frontier (next unlock): fluorescence · exact carat · micro-inclusions are gated on capture hardware "
             "— microscope + UV + scale — not on the models.", ha="center", fontsize=10.5, color="#7c8a98", style="italic")

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "investor_scorecard.jpg"
    fig.savefig(dest, dpi=140, facecolor=BG, bbox_inches="tight"); plt.close(fig)
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
