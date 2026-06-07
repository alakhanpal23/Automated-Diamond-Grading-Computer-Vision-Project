"""Blind-validation card: a stone graded from video alone, then revealed vs its GIA cert.

    python src/investor_blindtest.py

Investor proof point: the system was handed only a 360 video of stone XSBDF049
(no cert), produced a full grade, and we compare it to the real GIA cert.
Output: data/processed/demo/investor_blindtest.jpg
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("data/processed/demo")
FRAME = Path("data/processed/_dossier_tmp/test_video/frame_00.jpg")
INK = "#16263a"
GREEN = "#1f9d55"
AMBER = "#e09f3e"
RED = "#c1503f"
PANEL = "#f4f6f9"

# (attribute, AI prediction from video, GIA cert, status)  status: ok / partial / miss
ROWS = [
    ("Shape", "Round brilliant", "Round", "ok"),
    ("Color", "near-colorless (G-H)", "G", "ok"),
    ("Clarity", "SI", "SI1", "ok"),
    ("Cut — depth", "61.9%", "61.7%", "ok"),
    ("Cut — table", "60%", "59%", "ok"),
    ("Cut — crown angle", "35.0°", "34.5°", "ok"),
    ("Cut — pavilion angle", "43°", "43.5°", "ok"),
    ("Eye-clean", "No (included)", "SI1, feather", "ok"),
    ("Fluorescence", "Faint", "None", "miss"),
    ("Inclusions", "Feather ✓  (+ over-calls)", "TwWisp, Feather, Pinpt", "partial"),
]
COLOR = {"ok": GREEN, "partial": AMBER, "miss": RED}
MARK = {"ok": "✓", "partial": "≈", "miss": "✗"}


def main():
    fig = plt.figure(figsize=(13.5, 7.6), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[0.82, 1.25], wspace=0.08,
                          left=0.04, right=0.97, top=0.82, bottom=0.10)

    fig.text(0.04, 0.93, "Blind validation — a stone the system was never told the answer to",
             fontsize=20, fontweight="bold", color=INK)
    fig.text(0.04, 0.88, "Handed only a 360° video (no certificate). Predictions below were made from the video, "
             "then revealed against the real GIA cert.", fontsize=11.5, color="#4a5b6b")

    # ---- input video frame ----
    ax = fig.add_subplot(gs[0, 0])
    if FRAME.exists():
        ax.imshow(cv2.cvtColor(cv2.imread(str(FRAME)), cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title("INPUT:  360° video only", fontsize=12, color=INK, fontweight="bold", pad=8)
    ax.text(0.5, -0.06, "stone XSBDF049  ·  no cert provided to the model", transform=ax.transAxes,
            ha="center", va="top", fontsize=10, color="#7c8a98", style="italic")

    # ---- comparison table ----
    tb = fig.add_subplot(gs[0, 1]); tb.axis("off")
    tb.set_xlim(0, 1); tb.set_ylim(0, 1)
    tb.text(0.02, 1.00, "ATTRIBUTE", fontsize=10, fontweight="bold", color="#7c8a98")
    tb.text(0.40, 1.00, "AI  (from video)", fontsize=10, fontweight="bold", color=GREEN)
    tb.text(0.74, 1.00, "GIA cert", fontsize=10, fontweight="bold", color="#b06a00")
    n = len(ROWS); top = 0.95; row_h = top / n
    for i, (attr, pred, cert, st) in enumerate(ROWS):
        y = top - i * row_h - row_h * 0.5
        if i % 2 == 0:
            tb.add_patch(FancyBboxPatch((0.0, y - row_h * 0.42), 1.0, row_h * 0.84,
                                        boxstyle="round,pad=0,rounding_size=0.01",
                                        facecolor=PANEL, edgecolor="none"))
        tb.text(0.02, y, attr, fontsize=11, color=INK, va="center")
        tb.text(0.40, y, pred, fontsize=11, color="#1f7a3f", va="center", fontweight="bold")
        tb.text(0.74, y, cert, fontsize=10.5, color="#9a6500", va="center")
        tb.text(0.975, y, MARK[st], fontsize=14, color=COLOR[st], va="center", ha="right", fontweight="bold")

    # ---- scoreline ----
    chip = FancyBboxPatch((0.04, 0.015), 0.5, 0.062, boxstyle="round,pad=0.004,rounding_size=0.02",
                          facecolor=GREEN, edgecolor="none", transform=fig.transFigure)
    fig.add_artist(chip)
    fig.text(0.29, 0.046, "8 of 10 attributes correct from video alone", ha="center", va="center",
             fontsize=12.5, color="white", fontweight="bold")
    fig.text(0.57, 0.046, "cut geometry within 0.5°  ·  the 2 soft spots (fluorescence, exact inclusion list) "
             "need UV + microscope", ha="left", va="center", fontsize=10, color="#4a5b6b", style="italic")

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "investor_blindtest.jpg"
    fig.savefig(dest, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
