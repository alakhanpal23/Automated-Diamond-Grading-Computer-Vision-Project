"""QC: flag stones where the video-predicted SHAPE disagrees with the GIA cert.

    python src/qc_report.py [--n 1000] [--seed 1]

Shape is ~99% accurate, so a video-vs-cert shape disagreement (excluding the rare
'other' shapes the head was not trained on) is a high-precision candidate for
human review: a mislabeled stone, a swapped stone, or the wrong video attached to
a record. This is the concrete, useful application of the grader on an inventory
that is already certified.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

FRAMES = Path("data/processed/frames")
MDIR = Path("data/models/shape_group_resnet18")
CSV = Path("data/processed/stone_records.csv")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-conf", type=float, default=0.9, help="flag only confident mismatches")
    args = ap.parse_args()

    import numpy as np
    import torch
    from torch import nn
    from torchvision import models, transforms
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((MDIR / "classes.json").read_text())
    classes, size = meta["classes"], meta["img_size"]
    model = models.resnet18(weights=None); model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(MDIR / "best.pt", map_location=device)); model.to(device).eval()

    df = pd.read_csv(CSV, dtype={"stone_id": str}).set_index("stone_id")
    rng = np.random.default_rng(args.seed)
    ids = [d for d in (p.name for p in FRAMES.iterdir()) if d in df.index]
    rng.shuffle(ids); ids = ids[:args.n]

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize(mean, std)])

    checked = 0
    flags = []
    with torch.no_grad():
        for sid in ids:
            frames = sorted((FRAMES / sid).glob("frame_*.jpg"))
            if not frames:
                continue
            cert = str(df.loc[sid, "shape_group"])
            if cert == "other":          # head can't predict 'other' -> skip
                continue
            x = torch.stack([tf(Image.open(f).convert("RGB")) for f in frames]).to(device)
            prob = model(x).softmax(1).mean(0)
            idx = int(prob.argmax())
            pred, conf = classes[idx], float(prob[idx])
            checked += 1
            if pred != cert and conf >= args.min_conf:
                flags.append((sid, cert, pred, round(conf, 3)))

    print(f"=== SHAPE QC over {checked} certified stones (conf>={args.min_conf}) ===")
    print(f"flagged (video shape != cert shape): {len(flags)}  "
          f"= {len(flags)/max(checked,1):.2%} of inventory")
    for sid, cert, pred, conf in sorted(flags, key=lambda r: -r[3])[:25]:
        print(f"  {sid:<12} cert={cert:<10} video={pred:<10} conf={conf:.2f}")
    print("\nThese are candidates for human review (mislabel / swapped stone / wrong video).")


if __name__ == "__main__":
    main()
