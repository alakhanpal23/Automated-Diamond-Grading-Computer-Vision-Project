"""3D reconstruction of a diamond from the geometry model + the stone's real outline.

    python src/reconstruct_3d.py <stone_id>

Improvements:
  #1 facet detail   : star-length% and lower-half% subdivide the crown/pavilion.
  #2 shape-specific : brilliant faceting for round/oval/pear/...; STEP faceting
                      (concentric terraces, keel) for emerald/asscher.
  #3 per-stone form : the GIRDLE OUTLINE is the stone's ACTUAL silhouette (not an
                      idealized ellipse), so each stone's real shape is reconstructed;
                      depth/angles come from the geometry model.
Output: data/processed/recon/<stone>_3d.jpg (real frame + 2 rendered views).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_silhouette import segment

FRAMES = Path("data/processed/frames")
CSV = Path("data/processed/stone_records.csv")
GEOM = Path("data/models/geometry_resnet18")
OUT = Path("data/processed/recon")
STEP_SHAPES = {"emerald", "asscher"}


def real_outline(frames, n=28):
    """The stone's actual girdle outline from its largest (face-up) silhouette,
    resampled to n points by angle and normalized so the max radius = 1."""
    best_area, best = 0, None
    for f in frames:
        m = segment(cv2.imread(str(f)))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > best_area:
            best_area, best = cv2.contourArea(c), c
    if best is None:
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.c_[np.cos(a), np.sin(a)]
    pts = best[:, 0, :].astype(float)
    pts -= pts.mean(0)
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    out = []
    for k in range(n):
        ta = -np.pi + 2 * np.pi * k / n
        d = np.abs((ang - ta + np.pi) % (2 * np.pi) - np.pi)
        out.append(pts[d.argmin()])
    out = np.array(out)
    # light circular smoothing to remove silhouette jitter
    k = np.array([0.25, 0.5, 0.25])
    out = np.c_[np.convolve(np.r_[out[-1, 0], out[:, 0], out[0, 0]], k, "valid"),
                np.convolve(np.r_[out[-1, 1], out[:, 1], out[0, 1]], k, "valid")]
    out /= np.abs(out).max()
    return out


def _edges(corners, per=5):
    """Densify a polygon's edges, keeping the corners sharp."""
    pts = []
    for i in range(len(corners)):
        a, b = np.array(corners[i]), np.array(corners[(i + 1) % len(corners)])
        for t in np.linspace(0, 1, per, endpoint=False):
            pts.append(a + (b - a) * t)
    return np.array(pts)


def parametric_outline(shape, ratio):
    """Clean, crisp-cornered outline for shapes the silhouette rounds off.
    Returns None for smooth shapes (use the real silhouette instead)."""
    wx, wy = 1.0, 1.0 / max(ratio, 0.5)
    if shape in {"emerald", "radiant"}:                              # cut-corner rectangle (sharp)
        c = 0.10
        cor = [(wx, wy - c * wy), (wx - c * wx, wy), (-(wx - c * wx), wy), (-wx, wy - c * wy),
               (-wx, -(wy - c * wy)), (-(wx - c * wx), -wy), (wx - c * wx, -wy), (wx, -(wy - c * wy))]
        return _edges(cor, 7)
    if shape == "asscher":                                           # cut-corner square
        c = 0.18
        cor = [(1, 1 - c), (1 - c, 1), (-(1 - c), 1), (-1, 1 - c),
               (-1, -(1 - c)), (-(1 - c), -1), (1 - c, -1), (1, -(1 - c))]
        return _edges(cor, 7)
    if shape == "princess":                                          # sharp square
        return _edges([(wx, wy), (-wx, wy), (-wx, -wy), (wx, -wy)], 7)
    if shape == "cushion":                                           # superellipse (rounded square)
        a = np.linspace(0, 2 * np.pi, 28, endpoint=False)
        p = 0.42
        return np.c_[np.sign(np.cos(a)) * np.abs(np.cos(a)) ** p * wx,
                     np.sign(np.sin(a)) * np.abs(np.sin(a)) ** p * wy]
    return None


def chevron_pavilion(outline, pav_d):
    """Princess pavilion: inverted pyramid from the square girdle to the culet
    (the corner-to-culet edges form the chevron / X)."""
    n = len(outline)
    girdle = ring(outline, 1.0, 0.0)
    mid = ring(outline, 0.5, -pav_d * 0.5)
    culet = [0.0, 0.0, -pav_d]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([girdle[i], girdle[j], mid[j], mid[i]])
        faces.append([mid[i], mid[j], culet])
    return faces, girdle


def ring(outline, scale, z):
    return [[p[0] * scale, p[1] * scale, z] for p in outline]


def brilliant_mesh(outline, table_f, crown_h, pav_d, star_f, lower_f):
    n = len(outline)
    girdle = ring(outline, 1.0, 0.0)
    star_z = crown_h * (1 - star_f)                                   # star tips sit below the table
    star = ring(outline, table_f + (1 - table_f) * (1 - star_f), star_z)
    table = ring(outline, table_f, crown_h)
    lower = ring(outline, 1 - lower_f, -pav_d * lower_f)              # lower-girdle ring
    culet = [0.0, 0.0, -pav_d]
    faces = [table]
    for i in range(n):
        j = (i + 1) % n
        faces.append([table[i], star[i], star[j], table[j]])         # crown upper (table->star)
        faces.append([star[i], girdle[i], girdle[j], star[j]])       # crown lower (star->girdle)
        faces.append([girdle[i], lower[i], lower[j], girdle[j]])     # pavilion upper (girdle->lower)
        faces.append([lower[i], culet, lower[j]])                    # pavilion main (lower->culet)
    return faces


def step_mesh(outline, table_f, crown_h, pav_d):
    """Step cut: concentric terraces, ends in a keel line (not a point)."""
    n = len(outline)
    levels_up = [(table_f, crown_h), (0.78, crown_h * 0.45), (1.0, 0.0)]        # table -> girdle
    levels_dn = [(1.0, 0.0), (0.6, -pav_d * 0.55), (0.22, -pav_d)]              # girdle -> keel
    rings = [ring(outline, s, z) for s, z in levels_up] + [ring(outline, s, z) for s, z in levels_dn[1:]]
    faces = [rings[0]]                                                # table
    for r in range(len(rings) - 1):
        a, b = rings[r], rings[r + 1]
        for i in range(n):
            j = (i + 1) % n
            faces.append([a[i], a[j], b[j], b[i]])                    # terrace wall
    return faces


def princess_mesh(outline, table_f, crown_h, pav_d, star_f):
    """Square outline, brilliant-style crown, chevron (inverted-pyramid) pavilion."""
    n = len(outline)
    girdle = ring(outline, 1.0, 0.0)
    table = ring(outline, table_f, crown_h)
    star = ring(outline, table_f + (1 - table_f) * (1 - star_f), crown_h * (1 - star_f))
    faces = [table]
    for i in range(n):
        j = (i + 1) % n
        faces.append([table[i], star[i], star[j], table[j]])
        faces.append([star[i], girdle[i], girdle[j], star[j]])
    pav, _ = chevron_pavilion(outline, pav_d)
    return faces + pav


def render(faces, span, depth, view, path):
    fig = plt.figure(figsize=(4, 4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(faces, facecolor=(0.62, 0.80, 0.95, 0.55),
                                         edgecolor=(0.10, 0.20, 0.35, 0.9), linewidths=0.6))
    ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_zlim(-depth * 0.72, depth * 0.42)
    ax.set_box_aspect((1, 1, 0.9)); ax.view_init(elev=view[0], azim=view[1]); ax.set_axis_off()
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
    shape = str(cert.get("shape_group"))
    Rx = float(cert.get("mes_length", 6)) / 2 or 3.0
    # crisp parametric outline for cornered shapes; real silhouette for smooth ones
    par = parametric_outline(shape, float(cert.get("ratio", 1.0)) if not pd.isna(cert.get("ratio")) else 1.0)
    outline = par if par is not None else real_outline(frames)
    # scale outline to mm so x:y matches the real L/W
    sx = Rx / max(np.abs(outline[:, 0]).max(), 1e-6)
    sy = (Rx / max(prop.get("ratio", 1.0), 0.5)) / max(np.abs(outline[:, 1]).max(), 1e-6)
    outline = outline * [sx, sy]

    Ry = Rx / max(prop.get("ratio", 1.0), 0.5)
    table_f = prop["table_pct"] / 100.0
    total_depth = prop["depth_pct"] / 100.0 * 2 * Ry                  # depth% = depth/width; always reported
    if prop["crown_angle"] > 5:                                       # round brilliants report a real crown angle
        crown_h = (Rx - table_f * Rx) * np.tan(np.radians(prop["crown_angle"]))
    else:                                                             # fancy shapes: split depth by typical proportions
        crown_h = total_depth * 0.30
    pav_d = max(total_depth - crown_h - total_depth * 0.04, total_depth * 0.4)
    # #1 facet detail from cert star/lower-half (typical defaults when not reported)
    def _facet(col, default):
        v = cert.get(col)
        return (float(v) if (not pd.isna(v) and float(v) > 1) else default) / 100.0
    star_f, lower_f = _facet("star_length_pct", 50), _facet("lower_half_pct", 75)

    if shape in STEP_SHAPES:
        faces = step_mesh(outline, table_f, crown_h or 0.3 * Rx, pav_d)
        style = "step cut"
    elif shape == "princess":
        faces = princess_mesh(outline, table_f, crown_h or 0.3 * Rx, pav_d, star_f)
        style = "princess"
    else:
        faces = brilliant_mesh(outline, table_f, crown_h, pav_d, star_f, lower_f)
        style = "brilliant"

    OUT.mkdir(parents=True, exist_ok=True)
    span = np.abs(outline).max() * 1.1
    render(faces, span, crown_h + pav_d, (22, 35), OUT / f"_{sid}_p.png")
    render(faces, span, crown_h + pav_d, (2, 0), OUT / f"_{sid}_s.png")

    def frame_area(f):
        cnts, _ = cv2.findContours(segment(cv2.imread(str(f))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cv2.contourArea(max(cnts, key=cv2.contourArea)) if cnts else 0
    faceup_frame = max(frames, key=frame_area)
    fu = cv2.resize(cv2.imread(str(faceup_frame)), (360, 360))
    cv2.putText(fu, "video", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    pe = cv2.resize(cv2.imread(str(OUT / f"_{sid}_p.png")), (360, 360))
    sd = cv2.resize(cv2.imread(str(OUT / f"_{sid}_s.png")), (360, 360))
    cv2.putText(pe, f"3D ({style})", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 90, 20), 2)
    cv2.putText(sd, "3D profile", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 90, 20), 2)
    row = np.hstack([fu, pe, sd])
    cap = np.full((70, row.shape[1], 3), 25, np.uint8)
    txt = (f"{sid} ({shape}, {style})  table {prop['table_pct']:.0f}%  crown {prop['crown_angle']:.0f}deg  "
           f"pavilion {prop['pavilion_depth']:.0f}%  depth {prop['depth_pct']:.0f}%  L/W {prop['ratio']:.2f}  "
           f"star {star_f*100:.0f}% lower {lower_f*100:.0f}%")
    cv2.putText(cap, txt, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    cv2.imwrite(str(OUT / f"{sid}_3d.jpg"), np.vstack([row, cap]))
    print(f"=== 3D  {sid}  ({shape}, {style}) ===")
    print("  proportions:", {k: round(float(v), 1) for k, v in prop.items()},
          "star%", round(star_f * 100), "lower%", round(lower_f * 100))
    print(f"  -> {OUT / f'{sid}_3d.jpg'}")


if __name__ == "__main__":
    main()
