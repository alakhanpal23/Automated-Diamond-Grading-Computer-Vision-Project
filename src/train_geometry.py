"""ML regression of scale-free PROPORTIONS from the 360° frames.

    python src/train_geometry.py [--epochs 8] [--img-size 256] [--workers 4]

Predicts depth %, table %, crown angle, pavilion depth -- the proportions that
need the 3D profile (the silhouette already nails L/W ratio deterministically,
see geometry_silhouette.py). Targets come from stone_records (GIA cert), joined
to the manifest by stone_id. Per-frame training; per-stone inference averages
predictions across the 12 views. Reports held-out MAE per proportion.

Outputs to data/models/geometry_resnet18/: best.pt, norm.json (targets + mean/std).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

MANIFEST = Path("data/processed")
CSV = Path("data/processed/stone_records.csv")
OUT = Path("data/models/geometry_resnet18")
TARGETS = ["depth_pct", "table_pct", "crown_angle", "pavilion_depth", "ratio"]


# crown_angle & pavilion_depth are reported (>1) ONLY for round brilliants; they
# are stored as 0 for every fancy shape (~83% of stones). Training the shared
# backbone to output that constant 0 is pure noise that competes with depth/table,
# so we MASK those two targets wherever they're 0 and supervise them on rounds only.
MASKED = {"crown_angle", "pavilion_depth"}


class GeomDS:
    def __init__(self, df, tf, mean, std):
        self.paths = df["frame_path"].tolist()
        self.tf = tf
        import numpy as np
        raw = df[TARGETS].values.astype("float32")
        self.Y = ((raw - mean) / std).astype("float32")
        M = np.ones_like(raw, dtype="float32")
        for j, t in enumerate(TARGETS):
            if t in MASKED:
                M[:, j] = (raw[:, j] > 1.0).astype("float32")
        self.M = M

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        return (self.tf(Image.open(self.paths[i]).convert("RGB")),
                torch.from_numpy(self.Y[i]), torch.from_numpy(self.M[i]))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
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

    cert = pd.read_csv(CSV, dtype={"stone_id": str}).set_index("stone_id")[TARGETS]

    def load(split):
        df = pd.read_csv(MANIFEST / f"manifest_{split}.csv", dtype={"stone_id": str})
        df = df.join(cert, on="stone_id").dropna(subset=TARGETS)
        return df.reset_index(drop=True)

    tr, va, te = load("train"), load("val"), load("test")
    mean = tr[TARGETS].mean().values
    std = tr[TARGETS].std().values
    print(f"INFO: frames train={len(tr)} val={len(va)} test={len(te)}")
    print(f"INFO: target means {dict(zip(TARGETS, mean.round(1)))}")

    m_, s_ = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                   transforms.RandomHorizontalFlip(0.5),
                                   transforms.ToTensor(), transforms.Normalize(m_, s_)])
    eval_tf = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                  transforms.ToTensor(), transforms.Normalize(m_, s_)])

    def loader(df, tf, sh):
        return DataLoader(GeomDS(df, tf, mean, std), batch_size=args.batch_size, shuffle=sh,
                          num_workers=args.workers, persistent_workers=args.workers > 0,
                          pin_memory=device == "cuda")
    tl, vl, tel = loader(tr, train_tf, True), loader(va, eval_tf, False), loader(te, eval_tf, False)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(TARGETS))
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.SmoothL1Loss(reduction="none")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "norm.json").write_text(json.dumps(
        {"targets": TARGETS, "mean": mean.tolist(), "std": std.tolist(), "img_size": args.img_size}, indent=2))

    def evaluate(df, ldr, name):
        model.eval()
        preds = []
        with torch.no_grad():
            for x, _, _ in ldr:
                preds.append(model(x.to(device)).cpu().numpy())
        preds = np.concatenate(preds) * std + mean
        d = df.copy()
        for k, t in enumerate(TARGETS):
            d[f"_p_{k}"] = preds[:, k]
        # per-stone: average frame predictions
        agg = d.groupby("stone_id").agg({**{f"_p_{k}": "mean" for k in range(len(TARGETS))},
                                         **{t: "first" for t in TARGETS}})
        # depth/table/ratio over all stones; crown/pavilion only where reported (rounds)
        mae = {}
        for k, t in enumerate(TARGETS):
            err = np.abs(agg[f"_p_{k}"] - agg[t])
            sel = agg[t] > 1 if t in MASKED else slice(None)
            mae[t] = float(err[sel].mean()) if (t not in MASKED or (agg[t] > 1).any()) else 0.0
        print(f"  [{name}] per-stone MAE: " + "  ".join(f"{t}={mae[t]:.2f}" for t in TARGETS))
        return mae, agg

    best = 1e9
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); ls = 0; n = 0
        for x, y, m in tl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            loss = (crit(model(x), y) * m).sum() / m.sum().clamp(min=1)
            loss.backward(); opt.step()
            ls += loss.item() * x.size(0); n += x.size(0)
        sched.step()
        print(f"EPOCH {ep}/{args.epochs} loss={ls/n:.4f} ({time.time()-t0:.0f}s)")
        mae, _ = evaluate(va, vl, "val")
        # select on the universal targets we care about (depth/table/ratio), not the round-only ones
        score = mae["depth_pct"] + mae["table_pct"] + 10 * mae["ratio"]
        torch.save(model.state_dict(), OUT / "last.pt")
        if score < best:
            best = score; torch.save(model.state_dict(), OUT / "best.pt"); print("  saved best")

    model.load_state_dict(torch.load(OUT / "best.pt", map_location=device))
    print("\nFINAL TEST (per-stone, multi-view):")
    mae, agg = evaluate(te, tel, "test")
    (OUT / "eval_test.json").write_text(json.dumps(mae, indent=2))
    # reference: stddev of each target (what MAE to beat = predicting the mean)
    print("  (baseline MAE if you just guessed the mean = the target's stddev:)")
    print("   " + "  ".join(f"{t}={te[t].std():.2f}" for t in TARGETS))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
