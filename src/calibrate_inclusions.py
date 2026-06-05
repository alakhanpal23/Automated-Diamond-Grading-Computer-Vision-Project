"""Calibrate per-type decision thresholds for the inclusion model (no retraining).

    python src/calibrate_inclusions.py

The model outputs a probability per inclusion type; a fixed 0.5 threshold over-
flags (low precision). This finds, per type, the threshold that maximizes F1 on
the validation split, then reports test micro precision/recall with fixed-0.5 vs
calibrated thresholds, and saves the thresholds into the model's classes.json.
Rebuild the inclusion manifest first (same stones/seed as training).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MDIR = Path("data/models/inclusion_resnet18")
MANIFEST = Path("data/processed")


def stone_probs(model, df, types, size, device, batch=32):
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_inclusion import InclusionDS
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    dl = DataLoader(InclusionDS(df, tf, types), batch_size=batch, shuffle=False, num_workers=4)
    model.eval(); pr = []
    with torch.no_grad():
        for x, _ in dl:
            pr.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    pr = np.concatenate(pr); d = df.copy()
    for k in range(len(types)):
        d[f"_q{k}"] = pr[:, k]
    P, Y = [], []
    for sid, g in d.groupby("stone_id"):
        s = g["inclusion_types"].iloc[0]; gt = set(str(s).split("|")) if isinstance(s, str) else set()
        P.append([g[f"_q{k}"].max() for k in range(len(types))])
        Y.append([1.0 if types[k] in gt else 0.0 for k in range(len(types))])
    return np.array(P), np.array(Y)


def micro(P, Y, th):
    pred = P >= th[None, :]
    tp = (pred & (Y > 0)).sum(); fp = (pred & (Y == 0)).sum(); fn = ((~pred) & (Y > 0)).sum()
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return pr, rc, 2 * pr * rc / max(pr + rc, 1e-9)


def main():
    import torch
    from torch import nn
    from torchvision import models
    meta = json.loads((MDIR / "classes.json").read_text())
    types, size = meta["types"], meta.get("img_size", 512)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(weights=None); model.fc = nn.Linear(model.fc.in_features, len(types))
    model.load_state_dict(torch.load(MDIR / "best.pt", map_location=device)); model.to(device)

    val = pd.read_csv(MANIFEST / "manifest_val.csv", dtype={"stone_id": str, "inclusion_types": str})
    test = pd.read_csv(MANIFEST / "manifest_test.csv", dtype={"stone_id": str, "inclusion_types": str})
    Pv, Yv = stone_probs(model, val, types, size, device)
    Pt, Yt = stone_probs(model, test, types, size, device)

    thr = []
    for k in range(len(types)):
        bt, bf = 0.5, -1.0
        for t in np.arange(0.1, 0.905, 0.05):
            p = Pv[:, k] >= t
            tp = (p & (Yv[:, k] > 0)).sum(); fp = (p & (Yv[:, k] == 0)).sum(); fn = ((~p) & (Yv[:, k] > 0)).sum()
            pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); f = 2 * pr * rc / max(pr + rc, 1e-9)
            if f > bf:
                bf, bt = f, float(t)
        thr.append(round(bt, 2))
    thr = np.array(thr)

    p0, r0, f0 = micro(Pt, Yt, np.full(len(types), 0.5))
    pc, rc, fc = micro(Pt, Yt, thr)
    print("=== inclusion thresholds (test micro, per-stone multi-view) ===")
    print(f"  fixed 0.5  : precision {p0:.2f}  recall {r0:.2f}  F1 {f0:.2f}")
    print(f"  calibrated : precision {pc:.2f}  recall {rc:.2f}  F1 {fc:.2f}")
    print("  per-type thresholds:", dict(zip(types, thr.tolist())))
    meta["thresholds"] = thr.tolist()
    (MDIR / "classes.json").write_text(json.dumps(meta, indent=2))
    print(f"  saved per-type thresholds -> {MDIR / 'classes.json'}")


if __name__ == "__main__":
    main()
