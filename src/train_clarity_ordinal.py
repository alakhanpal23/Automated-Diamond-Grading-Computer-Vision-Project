"""Ordinal clarity: regress the clarity grade index (I<SI<VS<VVS<IF) from frames.

    python src/train_clarity_ordinal.py [--epochs 8] [--img-size 384]

Clarity is an ORDERED scale, so treating it as regression (not flat classes)
respects that predicting VS when the truth is VVS is a near-miss, not as wrong as
predicting I. We report two held-out metrics: exact-grade accuracy and
within-one-grade accuracy (the practically meaningful one). Balanced random
sample across the 5 grades; per-stone = mean of per-frame predictions.

Outputs to data/models/clarity_ordinal_resnet18/: best.pt, meta.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

CSV = Path("data/processed/stone_records.csv")
FRAMES = Path("data/processed/frames")
MANIFEST = Path("data/processed")
OUT = Path("data/models/clarity_ordinal_resnet18")
ORDER = ["I", "SI", "VS", "VVS", "IF"]            # ascending clarity
IDX = {g: i for i, g in enumerate(ORDER)}


class ClarityDS:
    def __init__(self, df, tf):
        self.paths = df["frame_path"].tolist()
        self.tf = tf
        import numpy as np
        self.y = df["clarity_group"].map(IDX).values.astype("float32")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        return self.tf(Image.open(self.paths[i]).convert("RGB")), torch.tensor(self.y[i])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import models, transforms

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO: device={device}")

    def load(split):
        df = pd.read_csv(MANIFEST / f"manifest_{split}.csv", dtype={"stone_id": str})
        return df[df["clarity_group"].isin(ORDER)].reset_index(drop=True)
    tr, va, te = load("train"), load("val"), load("test")
    print(f"INFO: frames train={len(tr)} val={len(va)} test={len(te)}")

    m_, s_ = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    ttf = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                              transforms.RandomHorizontalFlip(0.5),
                              transforms.ToTensor(), transforms.Normalize(m_, s_)])
    etf = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                              transforms.ToTensor(), transforms.Normalize(m_, s_)])

    def L(df, tf, sh):
        return DataLoader(ClarityDS(df, tf), batch_size=args.batch_size, shuffle=sh,
                          num_workers=args.workers, persistent_workers=args.workers > 0,
                          pin_memory=device == "cuda")
    tl, vl, tel = L(tr, ttf, True), L(va, etf, False), L(te, etf, False)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.SmoothL1Loss()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "meta.json").write_text(json.dumps({"order": ORDER, "img_size": args.img_size}, indent=2))

    def evaluate(df, ldr, name):
        model.eval(); preds = []
        with torch.no_grad():
            for x, _ in ldr:
                preds.append(model(x.to(device)).squeeze(1).cpu().numpy())
        df = df.copy(); df["_p"] = np.concatenate(preds)
        agg = df.groupby("stone_id").agg(pred=("_p", "mean"), true=("clarity_group", "first"))
        agg["ti"] = agg["true"].map(IDX); agg["pi"] = agg["pred"].round().clip(0, 4)
        exact = (agg["pi"] == agg["ti"]).mean()
        within1 = ((agg["pi"] - agg["ti"]).abs() <= 1).mean()
        mae = (agg["pred"] - agg["ti"]).abs().mean()
        print(f"  [{name}] exact={exact:.1%}  within-1-grade={within1:.1%}  MAE(grades)={mae:.2f}  ({len(agg)} stones)")
        return within1

    best = 0
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); ls = n = 0
        for x, y in tl:
            x, y = x.to(device), y.to(device).float()
            opt.zero_grad(); loss = crit(model(x).squeeze(1), y); loss.backward(); opt.step()
            ls += loss.item() * x.size(0); n += x.size(0)
        sched.step()
        print(f"EPOCH {ep}/{args.epochs} loss={ls/n:.4f} ({time.time()-t0:.0f}s)")
        w = evaluate(va, vl, "val")
        torch.save(model.state_dict(), OUT / "last.pt")
        if w > best:
            best = w; torch.save(model.state_dict(), OUT / "best.pt"); print("  saved best")

    model.load_state_dict(torch.load(OUT / "best.pt", map_location=device))
    print("\nFINAL TEST:")
    evaluate(te, tel, "test")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
