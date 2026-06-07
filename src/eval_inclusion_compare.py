"""Evaluate an inclusion model on the CURRENT frozen val/test manifests with
per-type val-calibrated thresholds, so two models can be compared apples-to-apples
on the identical test stones.

    python src/eval_inclusion_compare.py <model_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_inclusions import stone_probs, micro

MANIFEST = Path("data/processed")


def main():
    mdir = Path(sys.argv[1])
    meta = json.loads((mdir / "classes.json").read_text())
    types, size = meta["types"], meta.get("img_size", 512)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(types))
    m.load_state_dict(torch.load(mdir / "best.pt", map_location=dev))
    m.to(dev)

    val = pd.read_csv(MANIFEST / "manifest_val.csv", dtype={"stone_id": str, "inclusion_types": str})
    test = pd.read_csv(MANIFEST / "manifest_test.csv", dtype={"stone_id": str, "inclusion_types": str})
    Pv, Yv = stone_probs(m, val, types, size, dev)
    Pt, Yt = stone_probs(m, test, types, size, dev)

    # per-type threshold that maximizes val F1
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

    pred = Pt >= thr[None, :]
    f1s = []
    print(f"model={mdir.name}  types={len(types)}  (test stones={len(Yt)})")
    for k, t in enumerate(types):
        tp = (pred[:, k] & (Yt[:, k] > 0)).sum(); fp = (pred[:, k] & (Yt[:, k] == 0)).sum(); fn = ((~pred[:, k]) & (Yt[:, k] > 0)).sum()
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); f = 2 * pr * rc / max(pr + rc, 1e-9)
        f1s.append(f)
        print(f"  {t:18s} thr={thr[k]:.2f}  P={pr:.2f} R={rc:.2f} F1={f:.2f}  (n={int(tp+fn)})")
    pmi, rmi, fmi = micro(Pt, Yt, thr)
    print(f"  macro-F1={np.mean(f1s):.3f}   micro P={pmi:.2f} R={rmi:.2f} F1={fmi:.2f}")
    # also save calibrated thresholds back into this model's classes.json
    meta["thresholds"] = thr.tolist()
    (mdir / "classes.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
