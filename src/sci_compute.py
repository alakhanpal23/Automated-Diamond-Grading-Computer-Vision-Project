"""Compute real evaluation artifacts for the scientific dashboard and cache them.

    python src/sci_compute.py

- Shape & Color confusion matrices on the clean heldout set (prior-corrected,
  multi-view majority vote).
- Geometry depth/table predicted-vs-actual (per stone) + per-shape MAE on the
  frozen test set (the retrained model's clean held-out).
Saves -> data/processed/demo/sci_metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FRAMES = Path("data/processed/frames")
MODELS = Path("data/models")
CSV = Path("data/processed/stone_records.csv")
HELDOUT = Path("data/processed/heldout_stones.txt")
OUT = Path("data/processed/demo/sci_metrics.json")


def main():
    import torch
    from torch import nn
    from torchvision import models, transforms
    from PIL import Image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev)
    df = pd.read_csv(CSV, dtype={"stone_id": str})
    cert = df.set_index("stone_id")
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def tf(sz):
        return transforms.Compose([transforms.Resize((sz, sz)), transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])

    def load_cls(name):
        d = MODELS / name
        meta = json.loads((d / "classes.json").read_text())
        m = models.resnet50(weights=None) if meta.get("backbone") == "resnet50" else models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(meta["classes"]))
        m.load_state_dict(torch.load(d / "best.pt", map_location=dev)); m.to(dev).eval()
        return m, meta["classes"], meta["img_size"]

    out = {}

    # ---------- classification confusion matrices on heldout ----------
    ids = [s.strip() for s in HELDOUT.read_text().splitlines() if s.strip()]
    ids = [s for s in ids if (FRAMES / s).exists()][:700]
    for head, name in [("shape_group", "shape_group_resnet18"), ("color_group", "color_group_resnet18")]:
        m, classes, sz = load_cls(name)
        vc = df[head].astype(str).value_counts()
        logp = torch.tensor(np.log(np.array([vc.get(c, 1) for c in classes], float) / vc.sum()),
                            dtype=torch.float32, device=dev)
        cm = np.zeros((len(classes), len(classes)), int)
        n = 0
        with torch.no_grad():
            for sid in ids:
                cv = cert.loc[sid].get(head) if sid in cert.index else None
                if cv is None or pd.isna(cv) or str(cv) not in classes:
                    continue
                frames = sorted((FRAMES / sid).glob("frame_*.jpg"))
                if not frames:
                    continue
                xb = torch.stack([tf(sz)(Image.open(f).convert("RGB")) for f in frames]).to(dev)
                logits = m(xb).mean(0) + logp
                pred = classes[int(logits.argmax())]
                cm[classes.index(str(cv)), classes.index(pred)] += 1
                n += 1
        acc = float(np.trace(cm) / max(cm.sum(), 1))
        out[head] = {"classes": classes, "cm": cm.tolist(), "acc": acc, "n": n}
        print(f"{head}: acc={acc:.3f} n={n}")

    # ---------- geometry predicted vs actual on frozen test ----------
    gd = MODELS / "geometry_resnet18"
    norm = json.loads((gd / "norm.json").read_text())
    tg, gm, gs, sz = norm["targets"], np.array(norm["mean"]), np.array(norm["std"]), norm["img_size"]
    m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, len(tg))
    m.load_state_dict(torch.load(gd / "best.pt", map_location=dev)); m.to(dev).eval()
    test = pd.read_csv("data/processed/manifest_test.csv", dtype={"stone_id": str})
    test_ids = sorted(test.stone_id.unique())
    di, ti = tg.index("depth_pct"), tg.index("table_pct")
    rows = []
    with torch.no_grad():
        for sid in test_ids:
            if sid not in cert.index:
                continue
            frames = sorted((FRAMES / sid).glob("frame_*.jpg"))
            if not frames:
                continue
            xb = torch.stack([tf(sz)(Image.open(f).convert("RGB")) for f in frames]).to(dev)
            p = (m(xb).cpu().numpy() * gs + gm).mean(0)
            row = cert.loc[sid]
            rows.append((str(row.get("shape_group")), float(row.get("depth_pct")), float(p[di]),
                         float(row.get("table_pct")), float(p[ti])))
    arr = pd.DataFrame(rows, columns=["shape", "depth_t", "depth_p", "table_t", "table_p"]).dropna()
    out["geometry"] = {
        "depth_true": arr.depth_t.tolist(), "depth_pred": arr.depth_p.tolist(),
        "table_true": arr.table_t.tolist(), "table_pred": arr.table_p.tolist(),
        "shapes": arr["shape"].tolist(),
        "per_shape_mae": {s: float(np.abs(g.depth_p - g.depth_t).mean())
                          for s, g in arr.groupby("shape") if len(g) >= 8},
        "depth_mae": float(np.abs(arr.depth_p - arr.depth_t).mean()),
        "depth_r2": float(1 - ((arr.depth_p - arr.depth_t) ** 2).sum() / ((arr.depth_t - arr.depth_t.mean()) ** 2).sum()),
        "n": len(arr),
    }
    print(f"geometry: depth MAE={out['geometry']['depth_mae']:.3f} R2={out['geometry']['depth_r2']:.3f} n={out['geometry']['n']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
