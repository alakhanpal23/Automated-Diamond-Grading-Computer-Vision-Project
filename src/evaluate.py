"""Per-frame AND per-stone (majority-vote) evaluation of a trained head.

Run from the repo root after train_shape_model.py has produced a checkpoint:

    python src/evaluate.py [--label-col shape_group] [--split test]
                           [--ckpt PATH] [--img-size 224] [--workers 0]

Per-frame accuracy is the training-time view. PER-STONE accuracy -- a majority
vote over each stone's frames -- is the metric that matters for tagging
inventory, and is usually higher because per-frame errors at awkward rotation
angles get out-voted by the stone's other frames.

Classes and their index order are derived from manifest_train.csv exactly as in
training, so the loaded checkpoint's output indices line up.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

MANIFEST_DIR = Path("data/processed")
MODELS_ROOT  = Path("data/models")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-frame + per-stone evaluation")
    p.add_argument("--label-col", default="shape_group")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--ckpt", type=Path, default=None,
                   help="Checkpoint (default: data/models/<label-col>_resnet18/best.pt)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import models, transforms
    except ImportError as e:
        sys.exit(f"ERROR: missing dependency: {e}")
    # Reuse the exact dataset class used in training (module-level, picklable).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_shape_model import FrameDS

    ckpt = args.ckpt or (MODELS_ROOT / f"{args.label_col}_resnet18" / "best.pt")
    if not ckpt.exists():
        sys.exit(f"ERROR: checkpoint not found: {ckpt} (train this head first)")

    def load(split: str) -> pd.DataFrame:
        path = MANIFEST_DIR / f"manifest_{split}.csv"
        if not path.exists():
            sys.exit(f"ERROR: {path} not found; run src/04_build_manifest.py first")
        df = pd.read_csv(path)
        df = df[df[args.label_col].notna()]
        df = df[df[args.label_col].astype(str).str.strip().ne("")]
        return df.reset_index(drop=True)

    # Class order MUST match training: sorted(unique(train labels)).
    classes = sorted(load("train")[args.label_col].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    df = load(args.split)
    print(f"INFO: ckpt={ckpt}")
    print(f"INFO: label={args.label_col} split={args.split} | "
          f"{len(df)} frames, {df['stone_id'].nunique()} stones, {len(classes)} classes")

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    # shuffle=False -> batch order follows df row order, so preds align with df.
    loader = DataLoader(FrameDS(df, tf, class_to_idx, args.label_col),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    preds: list[int] = []
    with torch.no_grad():
        for x, _ in loader:
            preds.extend(model(x).argmax(1).tolist())

    df = df.copy()
    df["pred_idx"] = preds
    df["true_idx"] = [class_to_idx[c] for c in df[args.label_col]]

    frame_acc = float((df["pred_idx"] == df["true_idx"]).mean())

    # Per-stone majority vote
    stone_correct = stone_total = 0
    pc_correct, pc_total = Counter(), Counter()
    for _, g in df.groupby("stone_id"):
        true = int(g["true_idx"].iloc[0])
        vote = Counter(g["pred_idx"]).most_common(1)[0][0]
        stone_total += 1
        pc_total[true] += 1
        if vote == true:
            stone_correct += 1
            pc_correct[true] += 1
    stone_acc = stone_correct / max(stone_total, 1)

    print(f"\nPER-FRAME acc: {frame_acc:.4f}   ({len(df)} frames)")
    print(f"PER-STONE  acc: {stone_acc:.4f}   ({stone_total} stones, majority vote)")
    print("per-stone recall by class:")
    for i in sorted(pc_total):
        print(f"  {str(idx_to_class[i]):<18} {pc_correct[i] / pc_total[i]:.4f}  (n={pc_total[i]})")


if __name__ == "__main__":
    main()
