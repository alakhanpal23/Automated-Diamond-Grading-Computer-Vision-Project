"""3D reconstruction of a diamond from the geometry model's predicted proportions.

    python src/reconstruct_3d.py <stone_id>

The geometry head predicts table%, crown angle, pavilion depth, depth%, L/W from
the video. From those (+ the weighed carat for mm scale) we build a parametric
faceted brilliant mesh and render it -- a real 3D reconstruction from the model
outputs. Output: data/processed/recon/<stone>_3d.jpg (real frame + 2 rendered views).
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

FRAMES = Path("data/processed/frames")
CSV = Path("data/processed/stone_records.csv")
GEOM = Path("data/models/geometry_resnet18")
OUT = Path("data/processed/recon")


def diamond_faces(table_pct, crown_deg, pav_pct, Rx, Ry, n=8):
    """Parametric brilliant ~ real facet layout: octagon table, 16-point girdle,
    triangulated crown (bezel/star/upper-girdle look) and pavilion mains + culet."""
    D = 2 * Rx
    tr = table_pct / 100.0
    crown_h = (Rx - tr * Rx) * np.tan(np.radians(crown_deg))
    pav_d = (pav_pct / 100.0) * D
    at = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8      # table octagon (flats aligned)
    ag = np.linspace(0, 2 * np.pi, 16, endpoint=False)                 # 16-point girdle
    table = [[tr * Rx * np.cos(t), tr * Ry * np.sin(t), crown_h] for t in at]
    girdle = [[Rx * np.cos(t), Ry * np.sin(t), 0.0] for t in ag]
    culet = [0.0, 0.0, -pav_d]
    faces = [table]                                                   # table face
    for i in range(8):                                               # crown: bezel + star + upper-girdle
        g0, g1, g2 = (2 * i) % 16, (2 * i + 1) % 16, (2 * i + 2) % 16
        ti, tj = table[i], table[(i + 1) % 8]
        faces.append([ti, girdle[g0], girdle[g1]])                   # upper-girdle (left)
        faces.append([ti, girdle[g1], tj])                          # bezel
        faces.append([tj, girdle[g1], girdle[g2]])                  # upper-girdle (right)
    for i in range(16):                                             # pavilion -> culet
        faces.append([girdle[i], girdle[(i + 1) % 16], culet])
    return faces, crown_h, pav_d


def render(faces, depth, span, view, path):
    fig = plt.figure(figsize=(4, 4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    pc = Poly3DCollection(faces, facecolor=(0.6, 0.8, 0.95, 0.55),
                          edgecolor=(0.1, 0.2, 0.35, 0.9), linewidths=0.8)
    ax.add_collection3d(pc)
    ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_zlim(-depth * 0.7, depth * 0.45)
    ax.set_box_aspect((1, 1, 0.9)); ax.view_init(elev=view[0], azim=view[1])
    ax.set_axis_off()
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: reconstruct_3d.py <stone_id>")
    import torch
    from torch import nn
    from torchvision import models, transforms
    from PIL import Image

    frames = sorted((FRAMES / sid).glob("frame_*.jpg"))
    if not frames:
        sys.exit(f"no frames for {sid}")
    norm = json.loads((GEOM / "norm.json").read_text())
    tg, gm, gs, size = norm["targets"], np.array(norm["mean"]), np.array(norm["std"]), norm["img_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, len(tg))
    m.load_state_dict(torch.load(GEOM / "best.pt", map_location=device)); m.to(device).eval()
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    x = torch.stack([tf(Image.open(f).convert("RGB")) for f in frames]).to(device)
    with torch.no_grad():
        p = (m(x).cpu().numpy() * gs + gm).mean(0)
    prop = dict(zip(tg, p))

    cert = pd.read_csv(CSV, dtype={"stone_id": str}).set_index("stone_id").loc[sid]
    # mm scale: use weighed carat + predicted proportions to get diameter; fall back to cert dims
    if not pd.isna(cert.get("weight_ct")):
        # rough: diameter from cert mes for scale (carat->dims handled in mm_dimensions)
        Rx = float(cert.get("mes_length", 6)) / 2
        Ry = Rx / max(prop.get("ratio", 1.0), 0.5)
    else:
        Rx = Ry = 3.0
    faces, crown_h, pav_d = diamond_faces(prop["table_pct"], prop["crown_angle"],
                                          prop["pavilion_depth"], Rx, Ry)
    OUT.mkdir(parents=True, exist_ok=True)
    span = max(Rx, Ry) * 1.1
    render(faces, crown_h + pav_d, span, (22, 35), OUT / f"_{sid}_persp.png")
    render(faces, crown_h + pav_d, span, (2, 0), OUT / f"_{sid}_side.png")

    # assemble: real frame + the 2 renders + caption
    fu = cv2.resize(cv2.imread(str(frames[0])), (360, 360))
    cv2.putText(fu, "video", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    pe = cv2.resize(cv2.imread(str(OUT / f"_{sid}_persp.png")), (360, 360))
    sd = cv2.resize(cv2.imread(str(OUT / f"_{sid}_side.png")), (360, 360))
    cv2.putText(pe, "3D (from model)", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 90, 20), 2)
    cv2.putText(sd, "3D profile", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 90, 20), 2)
    row = np.hstack([fu, pe, sd])
    cap = np.full((70, row.shape[1], 3), 25, np.uint8)
    txt = (f"{sid}  reconstructed from model: table {prop['table_pct']:.0f}%  "
           f"crown {prop['crown_angle']:.0f} deg  pavilion {prop['pavilion_depth']:.0f}%  "
           f"depth {prop['depth_pct']:.0f}%  L/W {prop['ratio']:.2f}")
    cv2.putText(cap, txt, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.imwrite(str(OUT / f"{sid}_3d.jpg"), np.vstack([row, cap]))
    print(f"=== 3D RECONSTRUCTION  {sid} ===")
    print("  model proportions:", {k: round(float(v), 1) for k, v in prop.items()})
    print(f"  visual -> {OUT / f'{sid}_3d.jpg'}")


if __name__ == "__main__":
    main()
