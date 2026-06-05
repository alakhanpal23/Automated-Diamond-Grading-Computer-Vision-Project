"""Deterministic proportions from the PROFILE silhouette (no ML, not dataset-bound).

    python src/geometry_3d.py <stone_id>            # one stone, with a visual
    python src/geometry_3d.py --benchmark ROUND 40  # MAE vs cert over N round stones

Finds the frame closest to a clean edge-on profile (max silhouette aspect),
straightens it so the girdle is horizontal, and measures geometry directly from
the outline:
    depth %   = vertical extent (table->culet) / girdle diameter
    table %   = flat-top width / girdle diameter
    crown ang = slope girdle-edge -> table-edge
    pav  ang  = slope girdle-edge -> culet
This is exactly what a controlled turntable in the grading machine would do
(perfectly), but run on the free-tumble marketing videos it is only as good as
how close the tumble gets to edge-on -- so it is a feasibility probe.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_silhouette import segment

FRAMES = Path("data/processed/frames")
CSV = Path("data/processed/stone_records.csv")
OUT = Path("data/processed/geometry")


def straightened_mask(bgr):
    m = segment(bgr)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    aspect = max(w, h) / max(min(w, h), 1)
    # rotate so the long axis (girdle) is horizontal
    if w < h:
        ang += 90
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rot = cv2.warpAffine(m, M, (m.shape[1], m.shape[0]))
    ys, xs = np.where(rot > 0)
    if len(xs) == 0:
        return None, aspect
    return rot[ys.min():ys.max() + 1, xs.min():xs.max() + 1], aspect


def measure_profile(mask):
    """mask: straightened profile (girdle horizontal). Returns proportions in %."""
    H, W = mask.shape
    widths = (mask > 0).sum(1)                 # width per row
    girdle = widths.max()
    # table is the flatter, wider end; culet tapers to a point -> orient table up
    if widths[:3].mean() < widths[-3:].mean():
        mask = mask[::-1]; widths = widths[::-1]
    table_w = widths[:max(2, H // 20)].mean()  # width along the top (table)
    girdle_row = int(widths.argmax())
    depth_pct = 100.0 * H / girdle
    table_pct = 100.0 * table_w / girdle
    # crown angle: rise = girdle_row, run = (girdle - table_w)/2
    crown = math.degrees(math.atan2(girdle_row, max((girdle - table_w) / 2, 1)))
    pav_h = H - girdle_row
    pav = math.degrees(math.atan2(pav_h, max(girdle / 2, 1)))
    return {"depth_pct": depth_pct, "table_pct": table_pct,
            "crown_angle": crown, "pavilion_depth": 100.0 * pav_h / girdle}


def measure_stone(sid):
    frames = sorted((FRAMES / sid).glob("frame_*.jpg"))
    best, best_asp = None, 0
    for f in frames:
        m, asp = straightened_mask(cv2.imread(str(f)))
        if m is not None and asp > best_asp:
            best, best_asp = m, asp
    if best is None:
        return None, 0, None
    return measure_profile(best), best_asp, best


def main():
    cert = pd.read_csv(CSV, dtype={"stone_id": str}).set_index("stone_id")
    if sys.argv[1] == "--benchmark":
        shape = sys.argv[2] if len(sys.argv) > 2 else "ROUND"
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
        import os
        frames = set(os.listdir(FRAMES))
        df = cert[(cert["shape_raw"] == shape)].dropna(subset=["depth_pct", "table_pct"])
        ids = [s for s in df.index if s in frames]
        rng = np.random.default_rng(0); rng.shuffle(ids); ids = ids[:n]
        errs = {"depth_pct": [], "table_pct": []}
        for sid in ids:
            meas, asp, _ = measure_stone(sid)
            if meas is None:
                continue
            for k in errs:
                errs[k].append(abs(meas[k] - float(cert.loc[sid, k])))
        print(f"=== DETERMINISTIC profile measurement vs cert ({shape}, {len(errs['depth_pct'])} stones) ===")
        for k, v in errs.items():
            print(f"  {k:<12} MAE {np.mean(v):.2f}  (median {np.median(v):.2f})")
        return

    sid = sys.argv[1]
    meas, asp, mask = measure_stone(sid)
    if meas is None:
        sys.exit(f"no usable profile for {sid}")
    print(f"=== DETERMINISTIC GEOMETRY  {sid}  (profile aspect {asp:.2f}) ===")
    for k in ("depth_pct", "table_pct", "crown_angle", "pavilion_depth"):
        c = cert.loc[sid, k] if sid in cert.index else None
        cs = f"   cert {float(c):.1f}" if c is not None and not pd.isna(c) else ""
        print(f"  {k:<14} measured {meas[k]:5.1f}{cs}")
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{sid}_profile.jpg"), mask)
    print(f"  straightened profile -> {OUT / f'{sid}_profile.jpg'}")


if __name__ == "__main__":
    main()
