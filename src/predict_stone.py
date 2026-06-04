"""Grade one diamond from its 360° video frames -- the POC end deliverable.

    python src/predict_stone.py <stone_id> [--img-size 224] [--json-only]

Runs every trained head (data/models/<label>_resnet18/best.pt), majority-votes
each head across the stone's frames, and prints a grading report. If the stone
is in stone_records.csv it also shows the GIA cert value per attribute and flags
agreement (a lightweight QC / mismatch check), plus the cert's inclusion types.

Heads with no trained model yet are skipped, so this works incrementally as the
training chain fills in heads.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

FRAMES_DIR  = Path("data/processed/frames")
MODELS_ROOT = Path("data/models")
INPUT_CSV   = Path("data/processed/stone_records.csv")
CERT_JSONL  = Path("data/processed/cert_inclusions.jsonl")

# Attribute heads that make up a "grade", in report order.
HEADS = ["shape_group", "color_group", "clarity_group", "fluorescence_group", "eye_clean"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grade one diamond from its video frames")
    p.add_argument("stone_id")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    p.add_argument("--json-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from torch import nn
        from torchvision import models, transforms
        from PIL import Image
    except ImportError as e:
        sys.exit(f"ERROR: missing dependency: {e}")

    sid = args.stone_id
    fdir = args.frames_dir / sid
    frames = sorted(fdir.glob("frame_*.jpg"))
    if not frames:
        sys.exit(f"ERROR: no frames for {sid} in {fdir} (extract frames first)")

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    batch = torch.stack([tf(Image.open(f).convert("RGB")) for f in frames])

    report: dict = {"stone_id": sid, "n_frames": len(frames), "predictions": {}}

    for head in HEADS:
        mdir = MODELS_ROOT / f"{head}_resnet18"
        ckpt, cls_path = mdir / "best.pt", mdir / "classes.json"
        if not (ckpt.exists() and cls_path.exists()):
            continue  # head not trained yet
        classes = json.loads(cls_path.read_text())
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            logits = model(batch)
            probs = logits.softmax(1)
            preds = logits.argmax(1).tolist()
        vote_idx, n_votes = Counter(preds).most_common(1)[0]
        report["predictions"][head] = {
            "label": classes[vote_idx],
            "vote_frac": round(n_votes / len(preds), 3),     # frame agreement on the winner
            "mean_prob": round(float(probs[:, vote_idx].mean()), 3),
        }

    # QC vs GIA cert + inclusion context
    if INPUT_CSV.exists():
        df = pd.read_csv(INPUT_CSV, dtype={"stone_id": str})
        row = df[df["stone_id"] == sid]
        if not row.empty:
            r = row.iloc[0]
            for head, pred in report["predictions"].items():
                cert_val = "" if pd.isna(r.get(head)) else str(r.get(head))
                pred["cert"] = cert_val
                pred["match"] = (pred["label"] == cert_val) if cert_val else None
            report["cert_clarity_grade"] = "" if pd.isna(r.get("clarity_raw")) else str(r.get("clarity_raw"))
            report["cert_inclusion_types"] = "" if pd.isna(r.get("inclusion_types")) else str(r.get("inclusion_types"))

    # Inclusion positions from the cert extractor, if available
    if CERT_JSONL.exists():
        for line in CERT_JSONL.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("stone_id") == sid:
                report["cert_inclusion_marks"] = obj.get("red_mark_clusters")
                report["cert_inclusion_positions"] = obj.get("positions")
                break

    if args.json_only:
        print(json.dumps(report, indent=2, default=str))
        return

    print(json.dumps(report, indent=2, default=str))
    print(f"\n=== GRADING REPORT  {sid}  ({len(frames)} frames) ===")
    if not report["predictions"]:
        print("  (no trained heads found yet under data/models/*_resnet18/)")
    for head, p in report["predictions"].items():
        cert = p.get("cert") or "?"
        flag = "" if p.get("match") is None else ("  ✓ match" if p["match"] else "  ✗ MISMATCH")
        print(f"  {head:<20} {p['label']:<16} "
              f"(vote {p['vote_frac']:.0%}, p={p['mean_prob']:.2f})   cert={cert}{flag}")
    if report.get("cert_inclusion_types"):
        print(f"  inclusions (cert):   {report['cert_inclusion_types']}")
    if "cert_inclusion_marks" in report:
        print(f"  inclusion marks plotted (cert): {report['cert_inclusion_marks']}")


if __name__ == "__main__":
    main()
