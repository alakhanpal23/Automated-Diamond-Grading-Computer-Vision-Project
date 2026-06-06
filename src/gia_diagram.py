"""GIA-style proportions diagram from the model's predicted proportions.

    python src/gia_diagram.py <stone_id>

Renders the classic GIA side-profile cross-section (table, crown, girdle,
pavilion, culet) with dimension annotations — table %, depth %, crown angle,
pavilion angle — plus a data block, styled like the GIA report's PROPORTIONS
panel. Values are model-predicted (vs cert shown in the data block).
Output: data/processed/gia/<stone>_gia.jpg
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch

GEOM = Path("data/models/geometry_resnet18")
FRAMES = Path("data/processed/frames")
CSV = Path("data/processed/stone_records.csv")
OUT = Path("data/processed/gia")
INK = "#1b2a3a"


def predict(sid):
    import torch
    from torch import nn
    from torchvision import models, transforms
    from PIL import Image
    frames = sorted((FRAMES / sid).glob("frame_*.jpg"))
    norm = json.loads((GEOM / "norm.json").read_text())
    tg, gm, gs, size = norm["targets"], np.array(norm["mean"]), np.array(norm["std"]), norm["img_size"]
    m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, len(tg))
    m.load_state_dict(torch.load(GEOM / "best.pt", map_location="cpu")); m.eval()
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    x = torch.stack([tf(Image.open(f).convert("RGB")) for f in frames])
    with torch.no_grad():
        p = (m(x).numpy() * gs + gm).mean(0)
    return dict(zip(tg, p))


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: gia_diagram.py <stone_id>")
    prop = predict(sid)
    cert = pd.read_csv(CSV, dtype={"stone_id": str}).set_index("stone_id").loc[sid]

    R = 1.0
    table_r = prop["table_pct"] / 100.0 * R
    depth = prop["depth_pct"] / 100.0 * 2 * R
    crown_deg = prop["crown_angle"] if prop["crown_angle"] > 5 else math.degrees(math.atan2(depth * 0.30, R - table_r))
    crown_h = (R - table_r) * math.tan(math.radians(crown_deg))
    gt = depth * 0.03
    pav_h = max(depth - crown_h - gt, depth * 0.4)
    pav_deg = math.degrees(math.atan2(pav_h, R))

    # ---- profile outline (shape-aware: step cut vs brilliant) ----
    is_step = str(cert.get("shape_group")) in {"emerald", "asscher"}
    yt, yg0, yg1, yc = crown_h, 0.0, -gt, -(gt + pav_h)
    keel_w = 0.22 * R if is_step else 0.0
    if is_step:                                                       # terraced crown + pavilion, keel
        mx, mh = table_r + 0.5 * (R - table_r), yt * 0.5
        px, pm = keel_w / 2 + 0.5 * (R - keel_w / 2), yg1 - pav_h * 0.5
        right = [(table_r, yt), (mx, yt), (mx, mh), (R, mh), (R, yg0), (R, yg1),
                 (px, yg1), (px, pm), (keel_w / 2, pm), (keel_w / 2, yc)]
    else:                                                            # brilliant: smooth crown + point culet
        right = [(table_r, yt), (R, yg0), (R, yg1), (0.0, yc)]
    pts = right + [(-x, y) for x, y in reversed(right)] + [right[0]]

    fig, (axd, axt) = plt.subplots(1, 2, figsize=(10.5, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("white")

    # ---- diagram ----
    axd.plot([p[0] for p in pts], [p[1] for p in pts], color=INK, lw=2.2)
    axd.plot([-table_r, table_r], [yt, yt], color=INK, lw=2.2)        # table
    axd.plot([0, 0], [yt, yc], color=INK, lw=0.7, ls=(0, (4, 4)))     # center axis
    axd.plot([-R, R], [yg0, yg0], color=INK, lw=0.7)                  # girdle top
    axd.plot([-R, R], [yg1, yg1], color=INK, lw=0.7)                  # girdle bottom
    if not is_step:                                                  # brilliant facet hint lines
        for sgn in (-1, 1):
            axd.plot([sgn * table_r, sgn * (table_r + (R - table_r) * 0.5)],
                     [yt, yt * 0.5], color=INK, lw=0.6)               # star facet
            axd.plot([sgn * R, sgn * R * 0.42], [yg1, yc * 0.55], color=INK, lw=0.6)  # lower-girdle

    def dim_h(x0, x1, y, label):
        axd.annotate("", (x0, y), (x1, y), arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
        axd.text((x0 + x1) / 2, y + 0.05, label, ha="center", va="bottom", color=INK, fontsize=10)

    def dim_v(x, y0, y1, label):
        axd.annotate("", (x, y0), (x, y1), arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
        axd.text(x + 0.06, (y0 + y1) / 2, label, ha="left", va="center", color=INK, fontsize=10, rotation=90)

    dim_h(-table_r, table_r, yt + 0.16, f"Table {prop['table_pct']:.0f}%")
    dim_v(R + 0.30, yt, yc, f"Depth {prop['depth_pct']:.1f}%")
    axd.add_patch(Arc((R, yg0), 0.7, 0.7, angle=0, theta1=180 - crown_deg, theta2=180, color=INK, lw=1.2))
    axd.text(R - 0.55, yg0 + 0.12, f"{crown_deg:.1f}°", color=INK, fontsize=9)
    axd.add_patch(Arc((R, yg0), 0.7, 0.7, angle=0, theta1=180, theta2=180 + pav_deg, color=INK, lw=1.2))
    axd.text(R - 0.62, yg0 - 0.22, f"{pav_deg:.1f}°", color=INK, fontsize=9)
    axd.text(0, yc - 0.12, "Keel" if is_step else "Culet: None", ha="center", va="top", color=INK, fontsize=9)
    axd.set_xlim(-R - 0.8, R + 1.1); axd.set_ylim(yc - 0.5, yt + 0.5)
    axd.set_aspect("equal"); axd.axis("off")
    axd.set_title("PROPORTIONS", color=INK, fontsize=12, fontweight="bold", loc="left")

    # ---- data block (predicted vs cert) ----
    axt.axis("off")
    def cv(c):
        v = cert.get(c)
        return "-" if pd.isna(v) else (f"{float(v):.2f}" if isinstance(v, float) else str(v))
    rows = [
        ("GIA-style report (predicted)", "", ""),
        ("Shape", str(cert.get("shape_raw")), ""),
        ("Measurements (mm)", f"{cv('mes_length')}x{cv('mes_width')}x{cv('mes_depth')}", ""),
        ("Carat (weighed)", cv("weight_ct"), ""),
        ("", "predicted", "cert"),
        ("Table %", f"{prop['table_pct']:.0f}", cv("table_pct")),
        ("Depth %", f"{prop['depth_pct']:.1f}", cv("depth_pct")),
        ("Crown angle", f"{crown_deg:.1f}deg", cv("crown_angle")),
        ("Pavilion depth %", f"{prop['pavilion_depth']:.1f}", cv("pavilion_depth")),
        ("L/W ratio", f"{prop['ratio']:.2f}", cv("ratio")),
        ("Color / Clarity", f"{cert.get('color_raw')} / {cert.get('clarity_raw')}", ""),
    ]
    y = 0.96
    for a, b, c in rows:
        bold = a.startswith("GIA")
        axt.text(0.0, y, a, fontsize=11 if bold else 10, fontweight="bold" if bold else "normal", color=INK)
        axt.text(0.62, y, b, fontsize=10, color="#1f7a3f")
        axt.text(0.86, y, c, fontsize=10, color="#b06a00")
        y -= 0.084

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{sid}_gia.jpg"
    fig.suptitle(f"GIA Diamond Dossier (model-reconstructed)   {sid}", color=INK, fontsize=13, fontweight="bold")
    fig.savefig(dest, dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"GIA-style diagram -> {dest}")
    print("  proportions:", {k: round(float(v), 1) for k, v in prop.items()})


if __name__ == "__main__":
    main()
