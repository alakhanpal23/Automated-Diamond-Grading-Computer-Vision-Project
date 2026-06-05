"""Silhouette geometry from the 360° frames + a visual montage.

    python src/geometry_silhouette.py <stone_id>

Segments the stone from each frame (uniform background), fits its outline, and
measures scale-free geometry across the rotation:
  * L/W ratio    -- from the face-up frame (largest silhouette), via min-area rect
  * outline area / aspect across the rotation (how the silhouette changes)
Writes a montage (data/processed/geometry/<stone>_montage.jpg) with the contour +
fitted box drawn on every frame, and prints the measured L/W vs the GIA cert.

This is the deterministic half of geometry estimation; the ML head predicts the
harder ratios (depth %, table %, angles) that need the 3D profile.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FRAMES = Path("data/processed/frames")
OUT = Path("data/processed/geometry")
CSV = Path("data/processed/stone_records.csv")


def segment(bgr: np.ndarray) -> np.ndarray:
    """Largest blob differing from the uniform border background."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    border = np.concatenate([g[:8].ravel(), g[-8:].ravel(), g[:, :8].ravel(), g[:, -8:].ravel()])
    bg = np.median(border)
    mask = (np.abs(g.astype(int) - bg) > 16).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == big).astype(np.uint8)


def main() -> None:
    sid = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: geometry_silhouette.py <stone_id>")
    frames = sorted((FRAMES / sid).glob("frame_*.jpg"))
    if not frames:
        sys.exit(f"no frames for {sid}")

    panels, metrics = [], []
    for f in frames:
        bgr = cv2.imread(str(f))
        m = segment(bgr)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = bgr.copy()
        long_s = short_s = area = 0.0
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            (cx, cy), (w, h), ang = cv2.minAreaRect(c)
            long_s, short_s = max(w, h), max(min(w, h), 1)
            box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(int)
            cv2.drawContours(vis, [c], -1, (0, 255, 0), 2)
            cv2.drawContours(vis, [box], -1, (0, 180, 255), 2)
        metrics.append((area, long_s, short_s))
        panels.append(cv2.resize(vis, (256, 256)))

    # montage 3x4
    rows = [np.hstack(panels[i:i + 4]) for i in range(0, 12, 4)]
    montage = np.vstack(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{sid}_montage.jpg"
    cv2.imwrite(str(dest), montage)

    areas = np.array([m[0] for m in metrics])
    faceup = int(areas.argmax())                       # largest silhouette = most face-up
    lw = metrics[faceup][1] / metrics[faceup][2]
    aspects = [m[1] / m[2] for m in metrics]

    cert = pd.read_csv(CSV, dtype={"stone_id": str}).set_index("stone_id")
    cert_ratio = float(cert.loc[sid, "ratio"]) if sid in cert.index else None

    print(f"=== SILHOUETTE GEOMETRY  {sid} ===")
    print(f"  face-up frame: {faceup:02d}  (largest silhouette)")
    print(f"  silhouette L/W ratio: {lw:.3f}" + (f"   cert ratio: {cert_ratio:.3f}" if cert_ratio else ""))
    print(f"  aspect across rotation: min {min(aspects):.2f}  max {max(aspects):.2f}")
    print(f"  montage -> {dest}")


if __name__ == "__main__":
    main()
