"""Recall-targeted threshold calibration for the inclusion model.

    python src/calibrate_recall.py [--target-recall 0.92]

For grading, missing an inclusion is worse than over-calling one, so instead of
maximizing F1 we pick, per type, the HIGHEST (most precise) threshold that still
reaches the target recall on val; if no threshold reaches it (rare/subtle types),
we fall back to the lowest threshold = maximum recall. Saves thresholds into the
production model's classes.json and reports the resulting test P/R/F1.
"""
from __future__ import annotations

import argparse
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

MDIR = Path("data/models/inclusion_resnet18")
MANIFEST = Path("data/processed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument("--min-precision", type=float, default=0.25,
                    help="If a type can't reach target recall above this precision, it's "
                         "noise at high recall -> fall back to its max-F1 threshold instead "
                         "of flooding every stone with it.")
    args = ap.parse_args()

    meta = json.loads((MDIR / "classes.json").read_text())
    types, size = meta["types"], meta.get("img_size", 512)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(types))
    m.load_state_dict(torch.load(MDIR / "best.pt", map_location=dev))
    m.to(dev)

    val = pd.read_csv(MANIFEST / "manifest_val.csv", dtype={"stone_id": str, "inclusion_types": str})
    test = pd.read_csv(MANIFEST / "manifest_test.csv", dtype={"stone_id": str, "inclusion_types": str})
    Pv, Yv = stone_probs(m, val, types, size, dev)
    Pt, Yt = stone_probs(m, test, types, size, dev)

    grid = np.arange(0.1, 0.905, 0.05)
    thr, modes = [], []
    for k in range(len(types)):
        recall_t, recall_p, f1_t, f1_best = None, -1.0, 0.5, -1.0
        for t in grid:
            p = Pv[:, k] >= t
            tp = (p & (Yv[:, k] > 0)).sum(); fp = (p & (Yv[:, k] == 0)).sum(); fn = ((~p) & (Yv[:, k] > 0)).sum()
            rc = tp / max(tp + fn, 1); pr = tp / max(tp + fp, 1); f = 2 * pr * rc / max(pr + rc, 1e-9)
            if f > f1_best:                                   # track max-F1 threshold (fallback)
                f1_best, f1_t = f, float(t)
            if rc >= args.target_recall and pr > recall_p:    # most precise threshold that hits target recall
                recall_p, recall_t = pr, float(t)
        # high recall only if it's not pure noise; else fall back to max-F1
        if recall_t is not None and recall_p >= args.min_precision:
            thr.append(round(recall_t, 2)); modes.append("recall")
        else:
            thr.append(round(f1_t, 2)); modes.append("F1")
    thr = np.array(thr)

    pred = Pt >= thr[None, :]
    f1s = []
    print(f"recall-targeted thresholds (target recall {args.target_recall}, min-precision {args.min_precision}):")
    for k, t in enumerate(types):
        tp = (pred[:, k] & (Yt[:, k] > 0)).sum(); fp = (pred[:, k] & (Yt[:, k] == 0)).sum(); fn = ((~pred[:, k]) & (Yt[:, k] > 0)).sum()
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); f = 2 * pr * rc / max(pr + rc, 1e-9)
        f1s.append(f)
        print(f"  {t:18s} thr={thr[k]:.2f} [{modes[k]:6s}] P={pr:.2f} R={rc:.2f} F1={f:.2f}  (n={int(tp+fn)})")
    pmi, rmi, fmi = micro(Pt, Yt, thr)
    print(f"  macro-F1={np.mean(f1s):.3f}   micro P={pmi:.2f} R={rmi:.2f} F1={fmi:.2f}")

    meta["thresholds"] = thr.tolist()
    (MDIR / "classes.json").write_text(json.dumps(meta, indent=2))
    print(f"saved -> {MDIR / 'classes.json'}")


if __name__ == "__main__":
    main()
