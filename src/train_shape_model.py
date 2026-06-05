"""Train a baseline shape classifier from frame JPEGs.

Run from the repo root once src/04_build_manifest.py has produced the manifests:

    python src/train_shape_model.py [--epochs 10] [--batch-size 64] [--lr 3e-4]
                                    [--device auto] [--workers 4]
                                    [--img-size 224] [--quick-test]
                                    [--exclude-other]

Outputs (under data/models/<label-col>_resnet18/, e.g. shape_group_resnet18,
eye_clean_resnet18 -- so heads never overwrite each other; override with --out-dir):
    best.pt  last.pt  history.json  eval_val.json  eval_test.json

Defaults:
- ResNet-18 ImageNet-pretrained
- 224x224 inputs (JPEGs are stored at 512; resized on load)
- Class-weighted cross-entropy with weights = 1 / sqrt(class_freq)
- AdamW + cosine LR schedule
- Mixed precision when CUDA is available
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

# Heavy deps are imported lazily so --help works without torch installed.


MANIFEST_DIR = Path("data/processed")
MODELS_ROOT  = Path("data/models")


class FrameDS:
    """Map-style dataset of (transformed RGB image, class index).

    Defined at module level (not inside main) so it is picklable by Windows /
    Python 3.12+ 'spawn' DataLoader workers, which re-import this module and
    rebuild the dataset in each worker. PIL is imported lazily inside
    __getitem__ so --help still works without torch/torchvision/pillow.
    """
    def __init__(self, df, tf, class_to_idx, label_col="shape_group"):
        self.paths  = df["frame_path"].tolist()
        self.labels = [class_to_idx[c] for c in df[label_col]]
        self.tf     = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train shape classifier")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--label-col", default="shape_group",
                   help="Manifest column to use as the target label, e.g. shape_group, "
                        "shape_raw, eye_clean, clarity_group, color_group, "
                        "fluorescence_group, inclusion_weak_label (default: shape_group)")
    p.add_argument("--quick-test", action="store_true",
                   help="One batch fwd/back per split; verifies wiring then exits")
    p.add_argument("--exclude-other", action="store_true",
                   help="Drop 'other' shape_group rows (rare, noisy)")
    p.add_argument("--no-color-jitter", action="store_true",
                   help="Disable brightness/contrast augmentation. Use for the color "
                        "and clarity heads, where jitter destroys the target signal.")
    p.add_argument("--backbone", default="resnet18", choices=("resnet18", "resnet50"),
                   help="Backbone architecture (resnet50 has more capacity for hard heads)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write checkpoints (default: data/models/<label-col>_resnet18, "
                        "so different heads never overwrite each other)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Per-head output dir so eye_clean / clarity / shape models never clobber each other.
    MODEL_DIR = args.out_dir or (MODELS_ROOT / f"{args.label_col}_resnet18")

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import models, transforms
        from PIL import Image
    except ImportError as e:
        sys.exit(f"ERROR: missing dependency: {e}. "
                 "Install with: pip install torch torchvision pillow")

    # Reproducibility
    torch.manual_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda"
    print(f"INFO: device={device}  amp={use_amp}")

    # Manifests
    def load(split):
        path = MANIFEST_DIR / f"manifest_{split}.csv"
        if not path.exists():
            sys.exit(f"ERROR: {path} not found; run src/04_build_manifest.py first")
        df = pd.read_csv(path)
        if args.exclude_other:
            df = df[df["shape_group"] != "other"]
        # Drop rows with a missing/blank label for the chosen target column so
        # arbitrary heads (eye_clean, clarity_group, ...) don't train on NaNs.
        before = len(df)
        df = df[df[args.label_col].notna()]
        df = df[df[args.label_col].astype(str).str.strip().ne("")]
        dropped = before - len(df)
        if dropped:
            print(f"INFO: {split}: dropped {dropped} rows with blank {args.label_col}")
        return df.reset_index(drop=True)

    train_df = load("train")
    val_df   = load("val")
    test_df  = load("test")
    print(f"INFO: frames train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    classes = sorted(train_df[args.label_col].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"INFO: label_col={args.label_col}  {len(classes)} classes: {classes}")

    # Class weights = 1 / sqrt(freq)
    train_counts = Counter(train_df[args.label_col])
    weights = torch.tensor(
        [1.0 / math.sqrt(train_counts[c]) for c in classes],
        dtype=torch.float32, device=device)
    weights = weights * (len(classes) / weights.sum())
    print(f"INFO: class weights: {dict(zip(classes, weights.cpu().tolist()))}")

    # Transforms
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    aug = [transforms.Resize((args.img_size, args.img_size)),
           transforms.RandomHorizontalFlip(p=0.5)]
    # Brightness/contrast jitter corrupts the very signal for the color and
    # clarity heads, so it is opt-out via --no-color-jitter.
    if not args.no_color_jitter:
        aug.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
    aug += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    train_tf = transforms.Compose(aug)
    print(f"INFO: train augmentation: {'flip' if args.no_color_jitter else 'flip + color-jitter'}")
    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_loader = DataLoader(FrameDS(train_df, train_tf, class_to_idx, args.label_col),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=use_amp,
                              persistent_workers=args.workers > 0)
    val_loader   = DataLoader(FrameDS(val_df, eval_tf, class_to_idx, args.label_col),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=use_amp,
                              persistent_workers=args.workers > 0)
    test_loader  = DataLoader(FrameDS(test_df, eval_tf, class_to_idx, args.label_col),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=use_amp,
                              persistent_workers=args.workers > 0)

    # Model
    if args.backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Persist the class order so inference (predict_stone.py) can map model output
    # indices back to labels for this head.
    (MODEL_DIR / "classes.json").write_text(
        json.dumps({"classes": [str(c) for c in classes], "img_size": args.img_size,
                    "backbone": args.backbone}, indent=2), encoding="utf-8")

    def run_eval(loader, name):
        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        per_class_correct = Counter()
        per_class_total   = Counter()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)
                loss_sum += loss.item() * x.size(0)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += x.size(0)
                for p, t in zip(pred.cpu().tolist(), y.cpu().tolist()):
                    per_class_total[t] += 1
                    if p == t:
                        per_class_correct[t] += 1
        acc = correct / max(total, 1)
        avg_loss = loss_sum / max(total, 1)
        # str() the keys: non-string label columns (eye_clean -> numpy bool,
        # clarity ints) are not JSON-serialisable as dict keys otherwise.
        per_class_recall = {
            str(classes[c]): per_class_correct[c] / per_class_total[c]
            for c in per_class_total
        }
        print(f"  [{name}] loss={avg_loss:.4f}  acc={acc:.4f}  "
              f"per-class recall: {per_class_recall}")
        return {"loss": avg_loss, "acc": acc, "per_class_recall": per_class_recall}

    # Quick wiring test: one fwd+back per loader, then evaluate, then exit.
    if args.quick_test:
        print("INFO: --quick-test (one batch per split, no real training)")
        for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            xb, yb = next(iter(loader))
            xb, yb = xb.to(device), yb.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            if name == "train":
                scaler.scale(loss).backward()
                scaler.step(optim); scaler.update(); optim.zero_grad()
            print(f"  {name}: batch={xb.shape}  loss={loss.item():.4f}")
        print("OK: model + dataloaders wired correctly")
        return

    # Real training
    history = []
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        train_loss = loss_sum / max(n, 1)
        elapsed = time.time() - t0
        print(f"EPOCH {epoch:>2}/{args.epochs}  train_loss={train_loss:.4f}  ({elapsed:.1f}s)")
        val_metrics = run_eval(val_loader, "val")
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        torch.save(model.state_dict(), MODEL_DIR / "last.pt")
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(model.state_dict(), MODEL_DIR / "best.pt")
            print(f"  saved new best (val_acc={best_val_acc:.4f})")

    # Final test eval using best checkpoint
    model.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location=device))
    print("\nFINAL EVAL on test split (using best.pt):")
    test_metrics = run_eval(test_loader, "test")

    (MODEL_DIR / "history.json").write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
    (MODEL_DIR / "eval_val.json").write_text(json.dumps(history[-1], indent=2, default=str), encoding="utf-8")
    (MODEL_DIR / "eval_test.json").write_text(json.dumps(test_metrics, indent=2, default=str), encoding="utf-8")
    print(f"\nDONE. Best val_acc={best_val_acc:.4f}, test_acc={test_metrics['acc']:.4f}")
    print(f"checkpoints + history under {MODEL_DIR}")


if __name__ == "__main__":
    main()
