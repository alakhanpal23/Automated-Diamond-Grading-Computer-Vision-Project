"""End-to-end grading accuracy over stones held out of EVERY head's training set.

    python src/evaluate_grade.py [--n 600] [--seed 42]

For each sampled held-out stone, run every head on its frames (each at its own
resolution), majority-vote the single-label heads, max-aggregate the inclusion
head, and compare to the GIA cert. Reports per-attribute per-stone accuracy and
inclusion micro precision/recall -- the POC's headline numbers on unseen stones.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

FRAMES_DIR = Path("data/processed/frames")
MODELS = Path("data/models")
CSV = Path("data/processed/stone_records.csv")
HELDOUT = Path("data/processed/heldout_stones.txt")
SINGLE = {
    "shape_group": "shape_group_resnet18", "color_group": "color_group_resnet18",
    "clarity_group": "clarity_group_resnet18", "fluorescence_group": "fluorescence_group_resnet18",
    "eye_clean": "eye_clean_resnet18",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import numpy as np
    import torch
    from torch import nn
    from torchvision import models, transforms
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO: device={device}")
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(CSV, dtype={"stone_id": str})
    cert = df.set_index("stone_id")
    ids = [s.strip() for s in HELDOUT.read_text().splitlines() if s.strip()]
    ids = [s for s in ids if (FRAMES_DIR / s).exists()]
    rng.shuffle(ids)
    ids = ids[:args.n]
    print(f"INFO: evaluating {len(ids)} held-out stones")

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    def tf(size):
        return transforms.Compose([transforms.Resize((size, size)),
                                   transforms.ToTensor(), transforms.Normalize(mean, std)])

    # Load all heads once
    heads = {}
    for head, dname in SINGLE.items():
        mdir = MODELS / dname
        if not (mdir / "best.pt").exists():
            continue
        meta = json.loads((mdir / "classes.json").read_text())
        classes, size = meta["classes"], meta["img_size"]
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, len(classes))
        m.load_state_dict(torch.load(mdir / "best.pt", map_location=device)); m.to(device).eval()
        heads[head] = (m, classes, size)
    inc = None
    idir = MODELS / "inclusion_resnet18"
    if (idir / "best.pt").exists():
        meta = json.loads((idir / "classes.json").read_text())
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, len(meta["types"]))
        m.load_state_dict(torch.load(idir / "best.pt", map_location=device)); m.to(device).eval()
        inc = (m, meta["types"], meta.get("threshold", 0.5), meta["img_size"])

    correct = Counter(); total = Counter()
    inc_tp = inc_fp = inc_fn = 0
    sizes = sorted({s for _, _, s in heads.values()} | ({inc[3]} if inc else set()))

    with torch.no_grad():
        for n_done, sid in enumerate(ids, 1):
            frames = sorted((FRAMES_DIR / sid).glob("frame_*.jpg"))
            if not frames or sid not in cert.index:
                continue
            pil = [Image.open(f).convert("RGB") for f in frames]
            batches = {s: torch.stack([tf(s)(im) for im in pil]).to(device) for s in sizes}
            row = cert.loc[sid]
            for head, (m, classes, size) in heads.items():
                preds = m(batches[size]).argmax(1).tolist()
                vote = classes[Counter(preds).most_common(1)[0][0]]
                cv = row.get(head)
                if pd.isna(cv):
                    continue
                total[head] += 1
                correct[head] += int(vote == str(cv))
            if inc is not None:
                m, types, thr, size = inc
                p = torch.sigmoid(m(batches[size])).cpu().numpy().max(0)
                pred = {types[k] for k in range(len(types)) if p[k] >= thr}
                truth = set(str(row.get("inclusion_types") or "").split("|")) - {""}
                truth &= set(types)  # only score types the model knows
                inc_tp += len(pred & truth); inc_fp += len(pred - truth); inc_fn += len(truth - pred)
            if n_done % 100 == 0:
                print(f"  ...{n_done}/{len(ids)}")

    print("\n=== END-TO-END GRADING ON HELD-OUT STONES (per-stone vs GIA cert) ===")
    for head in SINGLE:
        if total[head]:
            print(f"  {head:<20} {correct[head]/total[head]:.1%}   ({correct[head]}/{total[head]})")
    if inc is not None:
        prec = inc_tp / max(inc_tp + inc_fp, 1)
        rec = inc_tp / max(inc_tp + inc_fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"  {'inclusions (micro)':<20} P={prec:.2f} R={rec:.2f} F1={f1:.2f}")


if __name__ == "__main__":
    main()
