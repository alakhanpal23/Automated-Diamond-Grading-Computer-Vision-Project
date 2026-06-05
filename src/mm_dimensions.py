"""Real mm dimensions from predicted proportions + weighed carat.

    python src/mm_dimensions.py [--n 400]

Pipeline (the machine's actual geometry output):
    weight (scale)  -> stone volume V = ct * 56.818 mm^3
    shape (head)    -> per-shape fill factor C = V / (L*W*D), learned from data
    depth% (ML)     -> D = depth%/100 * W
    L/W ratio (silhouette) -> L = ratio * W
    solve:  W = ( V / (C * ratio * depth%/100) ) ** (1/3)

Evaluates on held-out stones: predicted L/W/D vs the cert's mes_* (mm).
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
CSV = Path("data/processed/stone_records.csv")
GEOM = Path("data/models/geometry_resnet18")
HELD = Path("data/processed/heldout_stones.txt")


def silhouette_ratio(frames):
    best_area, lw = 0, 1.0
    for f in frames:
        m = segment(cv2.imread(str(f)))
        c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not c:
            continue
        cc = max(c, key=cv2.contourArea)
        a = cv2.contourArea(cc)
        if a > best_area:
            (_, _), (w, h), _ = cv2.minAreaRect(cc)
            best_area, lw = a, max(w, h) / max(min(w, h), 1)
    return lw


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=400); args = ap.parse_args()
    import torch
    from torch import nn
    from torchvision import models, transforms
    from PIL import Image

    df = pd.read_csv(CSV, dtype={"stone_id": str})
    df = df.dropna(subset=["weight_ct", "mes_length", "mes_width", "mes_depth", "depth_pct", "ratio"])
    df = df[(df["mes_width"] > 0) & (df["mes_depth"] > 0)]
    # per-shape fill factor C from the whole dataset (shape known from the shape head)
    V_all = df["weight_ct"] * 56.818
    C = (V_all / (df["mes_length"] * df["mes_width"] * df["mes_depth"])).groupby(df["shape_group"]).median()

    norm = json.loads((GEOM / "norm.json").read_text())
    targets, mean, std, size = norm["targets"], np.array(norm["mean"]), np.array(norm["std"]), norm["img_size"]
    di = targets.index("depth_pct")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(weights=None); model.fc = nn.Linear(model.fc.in_features, len(targets))
    model.load_state_dict(torch.load(GEOM / "best.pt", map_location=device)); model.to(device).eval()
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    cert = df.set_index("stone_id")
    held = [s for s in (l.strip() for l in HELD.read_text().splitlines()) if s in cert.index and (FRAMES / s).exists()]
    rng = np.random.default_rng(0); rng.shuffle(held); held = held[:args.n]

    smooth = {"round", "oval", "pear", "marquise", "heart"}
    err = {"length": [], "width": [], "depth": []}
    err_smooth = {"length": [], "width": [], "depth": []}
    with torch.no_grad():
        for sid in held:
            frames = sorted((FRAMES / sid).glob("frame_*.jpg"))
            r = cert.loc[sid]
            ratio = silhouette_ratio(frames)
            x = torch.stack([tf(Image.open(f).convert("RGB")) for f in frames]).to(device)
            depth_pct = float((model(x).cpu().numpy() * std + mean).mean(0)[di])
            V = float(r["weight_ct"]) * 56.818
            c = float(C.get(r["shape_group"], C.median()))
            W = (V / (c * ratio * depth_pct / 100)) ** (1 / 3)
            L, D = ratio * W, depth_pct / 100 * W
            e = {"length": abs(L - r["mes_length"]), "width": abs(W - r["mes_width"]), "depth": abs(D - r["mes_depth"])}
            for k in err:
                err[k].append(e[k])
                if r["shape_group"] in smooth:
                    err_smooth[k].append(e[k])

    print(f"=== mm DIMENSIONS from (predicted proportions + weight) vs cert  [{len(held)} held-out stones] ===")
    for k in ("length", "width", "depth"):
        print(f"  {k:<7} MAE {np.mean(err[k]):.3f} mm   (median {np.median(err[k]):.3f})")
    print(f"  -- smooth shapes only (round/oval/pear/marquise/heart, n={len(err_smooth['length'])}; silhouette L/W is exact there):")
    for k in ("length", "width", "depth"):
        print(f"     {k:<7} MAE {np.mean(err_smooth[k]):.3f} mm")


if __name__ == "__main__":
    main()
