"""GIA-style clarity plot: inclusions detected and marked proportionately on the
diamond's face-up outline with facet lines.

    python src/clarity_plot.py <stone_id>

Inclusion positions come from dark-blob detection on the face-up frame (inclusions
read as dark/high-contrast spots inside the bright stone), normalized to the
diamond outline so they sit in their true relative positions — like the GIA
clarity characteristics plot. The inclusion model supplies the type legend.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_silhouette import segment
from reconstruct_3d import real_outline, parametric_outline

FRAMES = Path("data/processed/frames")
CSV = Path("data/processed/stone_records.csv")
OUT = Path("data/processed/clarity")
INK = "#1b2a3a"


def detect_inclusions(faceup_path, max_marks=10):
    """Return [(nx, ny, size)] of inclusion blobs in the stone's own normalized frame
    (nx, ny in [-1, 1] relative to the silhouette's half-width / half-height, so they
    map cleanly onto an outline of any aspect ratio). Also returns (centroid, (ext_x, ext_y))."""
    bgr = cv2.imread(str(faceup_path))
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    mask = segment(bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [], None
    pc = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
    ctr = pc.mean(0)
    ext_x = max(np.abs(pc[:, 0] - ctr[0]).max(), 1e-6)              # per-axis half-extents
    ext_y = max(np.abs(pc[:, 1] - ctr[1]).max(), 1e-6)
    inside = cv2.erode(mask, np.ones((15, 15), np.uint8)) > 0       # stay off the girdle/facet edges
    if inside.sum() < 100:
        return [], (ctr, (ext_x, ext_y))
    vals = gray[inside]
    dark = ((gray < vals.mean() - 0.75 * vals.std()) & inside).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(dark, 8)
    lo, hi = 6, 0.04 * inside.sum()
    blobs = [(stats[i, cv2.CC_STAT_AREA], cent[i]) for i in range(1, n)
             if lo <= stats[i, cv2.CC_STAT_AREA] <= hi]
    blobs.sort(key=lambda b: -b[0])
    out = [((cx - ctr[0]) / ext_x, -(cy - ctr[1]) / ext_y, a) for a, (cx, cy) in blobs[:max_marks]]
    return out, (ctr, (ext_x, ext_y))


def plot_marks(ax, outline, blobs, inset=0.84):
    """Plot inclusion blobs (in stone-normalized [-1,1] coords) onto an outline,
    scaling by the outline's per-axis extent so every mark lands inside the girdle."""
    ox = max(np.abs(outline[:, 0]).max(), 1e-6)
    oy = max(np.abs(outline[:, 1]).max(), 1e-6)
    for nx, ny, a in blobs:
        ax.plot(nx * ox * inset, ny * oy * inset, marker="o", color="#d11",
                ms=3 + min(7, a ** 0.5 / 3), alpha=0.85, zorder=5)


def facet_lines(ax, outline, shape):
    """Draw a simple GIA-style face-up facet pattern inside the outline."""
    mx, my = np.abs(outline[:, 0]).max(), np.abs(outline[:, 1]).max()
    if shape in {"emerald", "asscher", "radiant"}:                  # concentric step rectangles
        for s in (0.82, 0.6, 0.38):
            ax.plot(np.r_[outline[:, 0] * s, outline[0, 0] * s],
                    np.r_[outline[:, 1] * s, outline[0, 1] * s], color=INK, lw=0.6)
        return
    if shape == "princess":                                         # corner X + table square
        ax.plot([-mx, mx], [-my, my], color=INK, lw=0.5)
        ax.plot([-mx, mx], [my, -my], color=INK, lw=0.5)
        ax.plot(np.r_[outline[:, 0] * 0.55, outline[0, 0] * 0.55],
                np.r_[outline[:, 1] * 0.55, outline[0, 1] * 0.55], color=INK, lw=0.6)
        return
    # brilliant: table octagon + radial mains + star tips
    n = 8
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 8
    tbl = np.c_[np.cos(ang), np.sin(ang)] * 0.58 * np.array([np.abs(outline[:, 0]).max(), np.abs(outline[:, 1]).max()])
    ax.plot(np.r_[tbl[:, 0], tbl[0, 0]], np.r_[tbl[:, 1], tbl[0, 1]], color=INK, lw=0.7)  # table
    gird = np.c_[np.cos(ang), np.sin(ang)] * np.array([np.abs(outline[:, 0]).max(), np.abs(outline[:, 1]).max()])
    for i in range(n):
        ax.plot([tbl[i, 0], gird[i, 0]], [tbl[i, 1], gird[i, 1]], color=INK, lw=0.5)        # bezel mains
        j = (i + 1) % n
        mid = (gird[i] + gird[j]) / 2
        ax.plot([tbl[i, 0], mid[0]], [tbl[i, 1], mid[1]], color=INK, lw=0.4)               # star/upper-girdle


def build_axes(ax, sid, df=None, model_types=None):
    cert = (df.set_index("stone_id").loc[sid] if (df is not None and sid in set(df["stone_id"]))
            else pd.Series(dtype=object))
    shape = str(cert.get("shape_group")) if not pd.isna(cert.get("shape_group")) else "round"
    frames = sorted((FRAMES / sid).glob("frame_*.jpg"))

    def area(f):
        c, _ = cv2.findContours(segment(cv2.imread(str(f))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cv2.contourArea(max(c, key=cv2.contourArea)) if c else 0
    fu = max(frames, key=area)

    par = parametric_outline(shape, float(cert.get("ratio", 1.0)) if not pd.isna(cert.get("ratio")) else 1.0)
    outline = par if par is not None else real_outline(frames)
    blobs, _ = detect_inclusions(fu)

    ax.plot(np.r_[outline[:, 0], outline[0, 0]], np.r_[outline[:, 1], outline[0, 1]], color=INK, lw=1.8)
    facet_lines(ax, outline, shape)
    plot_marks(ax, outline, blobs)
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.set_aspect("equal"); ax.axis("off")
    return len(blobs)


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: clarity_plot.py <stone_id>")
    df = pd.read_csv(CSV, dtype={"stone_id": str})
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
    nb = build_axes(ax, sid, df)
    ax.set_title(f"Clarity plot — {nb} inclusions detected", color=INK, fontsize=12)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{sid}_clarity.jpg"
    fig.savefig(dest, dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"clarity plot -> {dest}  ({nb} marks)")


if __name__ == "__main__":
    main()
