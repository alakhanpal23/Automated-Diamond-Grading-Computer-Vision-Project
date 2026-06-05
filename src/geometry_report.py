"""Paired geometry report for one stone: silhouette (L/W) + ML (proportions).

    python src/geometry_report.py <stone_id>

Combines the deterministic silhouette L/W ratio with the ML-predicted depth %,
table %, crown angle and pavilion depth, and renders an annotated image
(data/processed/geometry/<stone>_report.jpg) comparing both to the GIA cert.
Carat is shown as "scale" because in the machine it is physically weighed -- and
that weight, with these proportions, fixes the absolute mm dimensions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_silhouette import segment

FRAMES = Path("data/processed/frames")
OUT = Path("data/processed/geometry")
CSV = Path("data/processed/stone_records.csv")
GEOM = Path("data/models/geometry_resnet18")


def main() -> None:
    sid = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: geometry_report.py <stone_id>")
    import torch
    from torch import nn
    from torchvision import models, transforms
    from PIL import Image

    frames = sorted((FRAMES / sid).glob("frame_*.jpg"))
    if not frames:
        sys.exit(f"no frames for {sid}")
    bgr = [cv2.imread(str(f)) for f in frames]

    # --- silhouette L/W from the face-up (largest) frame ---
    areas, lws = [], []
    for im in bgr:
        m = segment(im)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            areas.append(0); lws.append(1.0); continue
        c = max(cnts, key=cv2.contourArea)
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        areas.append(cv2.contourArea(c)); lws.append(max(w, h) / max(min(w, h), 1))
    faceup = int(np.argmax(areas))
    profile = int(np.argmax(lws))
    lw_sil = lws[faceup]

    # --- ML proportions (averaged across views) ---
    norm = json.loads((GEOM / "norm.json").read_text())
    targets, mean, std, size = norm["targets"], np.array(norm["mean"]), np.array(norm["std"]), norm["img_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(weights=None); model.fc = nn.Linear(model.fc.in_features, len(targets))
    model.load_state_dict(torch.load(GEOM / "best.pt", map_location=device)); model.to(device).eval()
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    x = torch.stack([tf(Image.open(f).convert("RGB")) for f in frames]).to(device)
    with torch.no_grad():
        pred = (model(x).cpu().numpy() * std + mean).mean(0)
    ml = dict(zip(targets, pred))

    cert = pd.read_csv(CSV, dtype={"stone_id": str}).set_index("stone_id").loc[sid]

    # --- render annotated visual: face-up + profile frame, with a metrics panel ---
    fu = cv2.resize(bgr[faceup], (360, 360)); pr = cv2.resize(bgr[profile], (360, 360))
    cv2.putText(fu, "face-up", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(pr, "profile", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
    panel = np.full((360, 460, 3), 30, np.uint8)
    rows = [("metric", "predicted", "cert"),
            ("L/W (silhouette)", f"{lw_sil:.3f}", f"{float(cert['ratio']):.3f}"),
            ("depth %  (ML)", f"{ml['depth_pct']:.1f}", f"{float(cert['depth_pct']):.1f}"),
            ("table %  (ML)", f"{ml['table_pct']:.1f}", f"{float(cert['table_pct']):.1f}"),
            ("crown ang (ML)", f"{ml['crown_angle']:.1f}", f"{float(cert['crown_angle']):.1f}"),
            ("pavilion (ML)", f"{ml['pavilion_depth']:.1f}", f"{float(cert['pavilion_depth']):.1f}"),
            ("carat", "scale", f"{float(cert['weight_ct']):.2f}")]
    for r, (a, b, c) in enumerate(rows):
        y = 40 + r * 46
        col = (200, 200, 200) if r == 0 else (255, 255, 255)
        cv2.putText(panel, a, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
        cv2.putText(panel, b, (250, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 120), 1)
        cv2.putText(panel, c, (360, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 200, 255), 1)
    out = np.hstack([fu, pr, panel])
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{sid}_report.jpg"
    cv2.imwrite(str(dest), out)

    print(f"=== GEOMETRY REPORT  {sid} ===")
    print(f"  L/W (silhouette): {lw_sil:.3f}   cert {float(cert['ratio']):.3f}")
    for t in targets:
        print(f"  {t:<16} ML {ml[t]:5.1f}   cert {float(cert[t]):5.1f}   |err {abs(ml[t]-float(cert[t])):.1f}|")
    print(f"  carat: weighed in machine (cert {float(cert['weight_ct']):.2f}); proportions + weight -> mm dims")
    print(f"  visual -> {dest}")


if __name__ == "__main__":
    main()
