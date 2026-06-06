"""Multi-label inclusion-type model: predict which inclusion types are visible
in a stone's video, and aggregate across the 360° views.

    python src/train_inclusion.py [--epochs 6] [--img-size 384] [--workers 4]
                                  [--min-positives 30] [--out-dir ...]

Why this differs from the single-label heads:
* Multi-label -- a stone has several inclusion types at once (Feather|Crystal|
  Needle...), so we use a sigmoid + BCE head, one logit per type, not softmax.
* Multi-view aggregation -- an inclusion hidden behind a facet in one frame is
  visible in another. We train per-frame (each frame carries its stone's type
  set) but EVALUATE per stone by taking the MAX probability across the stone's
  frames: "present if visible in ANY view". Per-frame voting would wrongly let
  frames that hide an inclusion dilute the signal.

Labels come from stone_records.inclusion_types (validated 100% vs the GIA cert
plot by src/05_extract_cert_inclusions.py). Types with too few positive stones
in train are dropped (too sparse to learn).

Outputs to data/models/inclusion_resnet18/: best.pt, classes.json (the type
list + decision thresholds), history.json, eval_test.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

MANIFEST_DIR = Path("data/processed")
MODELS_ROOT  = Path("data/models")


class InclusionDS:
    """Per-frame dataset returning (image, multi-hot type vector). Module-level
    so Windows 'spawn' DataLoader workers can pickle it."""
    def __init__(self, df, tf, types):
        self.paths = df["frame_path"].tolist()
        self.tf = tf
        self.types = types
        tset = [set(str(s).split("|")) if isinstance(s, str) else set()
                for s in df["inclusion_types"]]
        import numpy as np
        self.Y = np.array([[1.0 if t in s else 0.0 for t in types] for s in tset],
                          dtype="float32")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), torch.from_numpy(self.Y[i])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-label inclusion-type model")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--min-positives", type=int, default=30,
                   help="Drop inclusion types with fewer than N positive TRAIN stones")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out-dir", type=Path, default=MODELS_ROOT / "inclusion_resnet18")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import models, transforms
    except ImportError as e:
        sys.exit(f"ERROR: missing dependency: {e}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"INFO: device={device}")

    def load(split):
        path = MANIFEST_DIR / f"manifest_{split}.csv"
        if not path.exists():
            sys.exit(f"ERROR: {path} not found; run src/04_build_manifest.py first")
        df = pd.read_csv(path, dtype={"stone_id": str, "inclusion_types": str})
        return df.reset_index(drop=True)

    train_df, val_df, test_df = load("train"), load("val"), load("test")

    # Derive the type vocabulary from TRAIN: keep types with enough positive stones.
    stone_types = train_df.drop_duplicates("stone_id")["inclusion_types"]
    cnt = Counter()
    for s in stone_types:
        if isinstance(s, str):
            for t in s.split("|"):
                if t:
                    cnt[t] += 1
    types = sorted([t for t, c in cnt.items() if c >= args.min_positives])
    if not types:
        sys.exit("ERROR: no inclusion types meet --min-positives")
    print(f"INFO: {len(types)} types (>= {args.min_positives} train stones): {types}")
    print(f"INFO: frames train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])

    def loader(df, tf, shuffle):
        return DataLoader(InclusionDS(df, tf, types), batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=args.workers,
                          persistent_workers=args.workers > 0, pin_memory=use_amp)

    train_loader = loader(train_df, train_tf, True)
    val_loader   = loader(val_df, eval_tf, False)
    test_loader  = loader(test_df, eval_tf, False)

    # pos_weight per type = #neg / #pos on train frames -> counter class imbalance
    ds_tr = train_loader.dataset
    pos = ds_tr.Y.sum(0)
    neg = len(ds_tr) - pos
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 20.0),
                              dtype=torch.float32, device=device)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(types))
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(df, ldr, name):
        """Per-stone multi-view eval: max prob across a stone's frames per type."""
        model.eval()
        probs = []
        with torch.no_grad():
            for x, _ in ldr:
                x = x.to(device)
                probs.append(torch.sigmoid(model(x)).cpu().numpy())
        probs = np.concatenate(probs) if probs else np.zeros((0, len(types)))
        d = df.copy()
        for k, t in enumerate(types):
            d[f"_p_{k}"] = probs[:, k]
        # aggregate per stone: max prob, and ground-truth presence
        rows = []
        for sid, g in d.groupby("stone_id"):
            gt = set(str(g["inclusion_types"].iloc[0]).split("|")) if isinstance(g["inclusion_types"].iloc[0], str) else set()
            pred_p = [g[f"_p_{k}"].max() for k in range(len(types))]
            rows.append((gt, pred_p))
        # per-type P/R/F1 at threshold
        tp = np.zeros(len(types)); fp = np.zeros(len(types)); fn = np.zeros(len(types))
        for gt, pred_p in rows:
            for k, t in enumerate(types):
                pred = pred_p[k] >= args.threshold
                truth = t in gt
                if pred and truth: tp[k] += 1
                elif pred and not truth: fp[k] += 1
                elif not pred and truth: fn[k] += 1
        prec = tp / np.maximum(tp + fp, 1)
        rec  = tp / np.maximum(tp + fn, 1)
        f1   = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        macro_f1 = float(f1.mean())
        per_type = {types[k]: {"precision": round(float(prec[k]), 3),
                               "recall": round(float(rec[k]), 3),
                               "f1": round(float(f1[k]), 3),
                               "n_pos": int(tp[k] + fn[k])} for k in range(len(types))}
        print(f"  [{name}] {len(rows)} stones  macro-F1={macro_f1:.3f}")
        for t in types:
            m = per_type[t]
            print(f"      {t:<18} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} (n={m['n_pos']})")
        return {"macro_f1": macro_f1, "per_type": per_type, "n_stones": len(rows)}

    if args.quick_test:
        xb, yb = next(iter(train_loader))
        out = model(xb.to(device))
        loss = criterion(out, yb.to(device))
        loss.backward()
        print(f"OK quick-test: batch={tuple(xb.shape)} out={tuple(out.shape)} loss={loss.item():.4f}")
        return

    (args.out_dir / "classes.json").write_text(
        json.dumps({"types": types, "threshold": args.threshold, "img_size": args.img_size},
                   indent=2), encoding="utf-8")

    history, best_f1 = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            loss_sum += loss.item() * x.size(0); n += x.size(0)
        sched.step()
        print(f"EPOCH {epoch}/{args.epochs} train_loss={loss_sum/max(n,1):.4f} ({time.time()-t0:.0f}s)")
        val = evaluate(val_df, val_loader, "val")
        history.append({"epoch": epoch, "train_loss": loss_sum / max(n, 1), **val})
        torch.save(model.state_dict(), args.out_dir / "last.pt")
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            print(f"  saved best (val macro-F1={best_f1:.3f})")

    model.load_state_dict(torch.load(args.out_dir / "best.pt", map_location=device))
    print("\nFINAL TEST (per-stone, multi-view max-agg):")
    test_metrics = evaluate(test_df, test_loader, "test")
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
    (args.out_dir / "eval_test.json").write_text(json.dumps(test_metrics, indent=2, default=str), encoding="utf-8")
    print(f"\nDONE. best val macro-F1={best_f1:.3f}, test macro-F1={test_metrics['macro_f1']:.3f}")
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
