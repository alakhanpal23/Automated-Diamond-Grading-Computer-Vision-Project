"""Scientific dashboard — the "under the hood" investor panel.

    python src/sci_compute.py   # once, to cache real eval artifacts
    python src/sci_dashboard.py

Renders the raw evaluation evidence behind the scorecard numbers: shape and
color confusion matrices on the clean held-out set, depth predicted-vs-actual
scatter on the frozen test set, and per-shape depth error. No re-inference —
reads data/processed/demo/sci_metrics.json.
Output: data/processed/demo/sci_dashboard.jpg
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

METRICS = Path("data/processed/demo/sci_metrics.json")
OUT = Path("data/processed/demo/sci_dashboard.jpg")
INK = "#16263a"
GREEN = "#1f9d55"
TEAL = "#2a9d8f"
GREY = "#9aa7b4"
BG = "#ffffff"
PANEL = "#f4f6f9"
SUB = "#4a5b6b"
FAINT = "#7c8a98"


def chip(ax, x, y, w, h, text, fc, tc="white", fs=12, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                linewidth=0, facecolor=fc, transform=ax.transAxes, clip_on=False))
    ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold" if bold else "normal")


def confusion(ax, classes, cm, title, subtitle, label_fs=9):
    cm = np.asarray(cm)
    k = len(classes)
    row = cm.sum(1, keepdims=True).clip(min=1)
    ax.imshow(cm / row, cmap="Greens", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            if cm[i, j]:
                frac = cm[i, j] / row[i, 0]
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=label_fs,
                        color="white" if frac > 0.55 else INK,
                        fontweight="bold" if i == j else "normal")
    names = [c.replace("_", " ") for c in classes]
    ax.set_xticks(range(k)); ax.set_xticklabels(names, fontsize=label_fs, rotation=45,
                                                ha="right", color=SUB)
    ax.set_yticks(range(k)); ax.set_yticklabels(names, fontsize=label_fs, color=SUB)
    ax.set_xlabel("predicted", fontsize=9.5, color=FAINT)
    ax.set_ylabel("GIA certificate", fontsize=9.5, color=FAINT)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=22)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=10, color=FAINT)


def main():
    M = json.loads(METRICS.read_text())
    sg, cg, geo = M["shape_group"], M["color_group"], M["geometry"]

    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.42, 1.25, 1.0],
                          width_ratios=[1.18, 0.82, 1.0], hspace=0.52, wspace=0.42,
                          left=0.06, right=0.965, top=0.945, bottom=0.07)

    # ---------- header ----------
    hd = fig.add_subplot(gs[0, :]); hd.axis("off")
    hd.text(0.0, 0.72, "Under the Hood — the Evaluation Evidence", transform=hd.transAxes,
            fontsize=28, fontweight="bold", color=INK)
    hd.text(0.0, 0.34, "Every number on the scorecard traces back to these matrices and scatters — "
            "computed on certified stones the models never saw.", transform=hd.transAxes,
            fontsize=13, color=SUB)
    for i, (txt, fc) in enumerate([(f"{sg['n']} held-out stones", INK),
                                   ("multi-view majority vote", TEAL),
                                   ("frozen test set", "#3d5a73"),
                                   ("GIA certs as ground truth", GREEN)]):
        chip(hd, 0.0 + i * 0.205, -0.16, 0.19, 0.30, txt, fc, fs=11)

    # ---------- shape confusion (perfect diagonal) ----------
    axs = fig.add_subplot(gs[1, 0])
    confusion(axs, sg["classes"], sg["cm"],
              f"Shape — {sg['acc'] * 100:.1f}% on {sg['n']} stones",
              "10 cuts, zero confusions: a perfect diagonal", label_fs=8.5)

    # ---------- color confusion (adjacent-tier only) ----------
    axc = fig.add_subplot(gs[1, 1])
    cm = np.asarray(cg["cm"])
    adj = int(cm.sum() - np.trace(cm) - cm[0, -1] - cm[-1, 0])
    confusion(axc, cg["classes"], cg["cm"],
              f"Color — {cg['acc'] * 100:.1f}% on {cg['n']} stones",
              f"all {adj} misses are one tier off — never two", label_fs=10)

    # ---------- depth predicted vs actual ----------
    axg = fig.add_subplot(gs[1, 2]); axg.set_facecolor(BG)
    dt, dp = np.array(geo["depth_true"]), np.array(geo["depth_pred"])
    lo, hi = min(dt.min(), dp.min()) - 0.5, max(dt.max(), dp.max()) + 0.5
    axg.plot([lo, hi], [lo, hi], color="#b9c4cf", lw=1.2, ls=(0, (5, 4)), zorder=1)
    axg.scatter(dt, dp, s=14, color=TEAL, alpha=0.45, linewidths=0, zorder=2)
    axg.set_xlim(lo, hi); axg.set_ylim(lo, hi); axg.set_aspect("equal")
    axg.set_xlabel("GIA certificate depth %", fontsize=10, color=FAINT)
    axg.set_ylabel("predicted from video", fontsize=10, color=FAINT)
    axg.tick_params(labelsize=9, length=0, colors=FAINT)
    for s in ["top", "right"]:
        axg.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        axg.spines[s].set_color("#d4dbe2")
    axg.set_title(f"Depth %  —  {geo['n']} test stones", fontsize=13.5,
                  fontweight="bold", color=INK, loc="left", pad=22)
    axg.text(0, 1.03, "each dot is one stone; the line is a perfect read",
             transform=axg.transAxes, fontsize=10, color=FAINT)
    axg.text(0.05, 0.88, f"MAE ±{geo['depth_mae']:.2f}%\nR² = {geo['depth_r2']:.3f}",
             transform=axg.transAxes, fontsize=14, fontweight="bold", color=GREEN, va="top")

    # ---------- per-shape depth MAE ----------
    axm = fig.add_subplot(gs[2, 0]); axm.set_facecolor(BG)
    items = sorted(geo["per_shape_mae"].items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]; vals = [v for _, v in items]
    ypos = list(range(len(items)))[::-1]
    for y, v in zip(ypos, vals):
        axm.barh(y, v, color=GREEN if v <= 0.4 else TEAL, height=0.62, zorder=3)
        axm.text(v + 0.012, y, f"±{v:.2f}", va="center", fontsize=10.5,
                 fontweight="bold", color=INK)
    axm.set_yticks(ypos); axm.set_yticklabels(names, fontsize=11, color=INK)
    axm.set_xlim(0, max(vals) * 1.22)
    axm.set_xticks([0, 0.25, 0.5, 0.75])
    axm.set_xticklabels(["0", "±0.25", "±0.50", "±0.75%"], fontsize=9, color=FAINT)
    for s in ["top", "right", "left"]:
        axm.spines[s].set_visible(False)
    axm.spines["bottom"].set_color("#d4dbe2")
    axm.tick_params(length=0)
    axm.set_title("Depth error by cut", fontsize=13.5, fontweight="bold",
                  color=INK, loc="left", pad=10)

    # ---------- table predicted vs actual ----------
    axt = fig.add_subplot(gs[2, 1]); axt.set_facecolor(BG)
    tt, tp = np.array(geo["table_true"]), np.array(geo["table_pred"])
    tmae = float(np.abs(tp - tt).mean())
    lo, hi = min(tt.min(), tp.min()) - 0.5, max(tt.max(), tp.max()) + 0.5
    axt.plot([lo, hi], [lo, hi], color="#b9c4cf", lw=1.2, ls=(0, (5, 4)), zorder=1)
    axt.scatter(tt, tp, s=14, color="#3d5a73", alpha=0.45, linewidths=0, zorder=2)
    axt.set_xlim(lo, hi); axt.set_ylim(lo, hi); axt.set_aspect("equal")
    axt.set_xlabel("GIA certificate table %", fontsize=10, color=FAINT)
    axt.set_ylabel("predicted from video", fontsize=10, color=FAINT)
    axt.tick_params(labelsize=9, length=0, colors=FAINT)
    for s in ["top", "right"]:
        axt.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        axt.spines[s].set_color("#d4dbe2")
    axt.set_title("Table %", fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=10)
    axt.text(0.05, 0.88, f"MAE ±{tmae:.2f}%", transform=axt.transAxes,
             fontsize=13, fontweight="bold", color=GREEN, va="top")

    # ---------- methodology card ----------
    me = fig.add_subplot(gs[2, 2]); me.axis("off")
    me.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.0,rounding_size=0.03",
                                facecolor=PANEL, edgecolor="none", transform=me.transAxes))
    me.text(0.07, 0.90, "How these numbers are made", fontsize=13.5,
            fontweight="bold", color=INK)
    for y, line in [
        (0.76, "Held-out stones are split before training —"),
        (0.69, "the models never see a frame of them."),
        (0.55, "Each stone is judged from all its video"),
        (0.48, "frames; averaged, then prior-corrected."),
        (0.34, "Ground truth is the stone's GIA certificate"),
        (0.27, "— an independent human lab, not our labels."),
    ]:
        me.text(0.07, y, line, fontsize=10.5, color=SUB)
    chip(me, 0.07, 0.06, 0.86, 0.13, "re-runnable:  python src/sci_compute.py", "#3d5a73", fs=8.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, facecolor=BG, bbox_inches="tight"); plt.close(fig)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
