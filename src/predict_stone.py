"""Grade one diamond from its 360° video frames -- the POC end deliverable.

    python src/predict_stone.py <stone_id> [--json-only]

Runs every trained head on the stone's frames -- each at the resolution it was
trained at -- majority-votes the single-label heads and max-aggregates the
multi-label inclusion head across the 360° views, then prints a grade report.
If the stone is in stone_records.csv it shows the GIA cert value per attribute
and flags agreement (lightweight QC), plus inclusion type agreement.
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

# single-label head -> its model dir name + the cert column to compare against
SINGLE_HEADS = {
    "shape_group":        "shape_group_resnet18",
    "color_group":        "color_group_resnet18",
    "clarity_group":      "clarity_group_resnet18",
    "fluorescence_group": "fluorescence_group_resnet18",
    "eye_clean":          "eye_clean_resnet18",
}
INCLUSION_DIR = "inclusion_resnet18"


def head_meta(mdir: Path) -> dict:
    """Read classes.json, tolerating the old plain-list format."""
    obj = json.loads((mdir / "classes.json").read_text())
    if isinstance(obj, list):
        return {"classes": obj, "img_size": 224}
    return obj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grade one diamond from its video frames")
    p.add_argument("stone_id")
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sid = args.stone_id
    frames = sorted((args.frames_dir / sid).glob("frame_*.jpg"))
    if not frames:
        sys.exit(f"ERROR: no frames for {sid} in {args.frames_dir / sid}")
    pil = [Image.open(f).convert("RGB") for f in frames]
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def batch_at(size):
        tf = transforms.Compose([transforms.Resize((size, size)),
                                 transforms.ToTensor(), transforms.Normalize(mean, std)])
        return torch.stack([tf(im) for im in pil]).to(device)

    report: dict = {"stone_id": sid, "n_frames": len(frames), "grade": {}, "inclusions": {}}

    # natural class priors for prior correction (matches evaluate_grade.py)
    import numpy as np
    prior_df = pd.read_csv(INPUT_CSV, dtype={"stone_id": str}) if INPUT_CSV.exists() else None

    # ---- single-label attribute heads (aggregate logits across views + prior) ----
    for head, dname in SINGLE_HEADS.items():
        mdir = MODELS_ROOT / dname
        if not (mdir / "best.pt").exists():
            continue
        meta = head_meta(mdir)
        classes, size = meta["classes"], meta.get("img_size", 224)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(torch.load(mdir / "best.pt", map_location=device))
        model.to(device).eval()
        with torch.no_grad():
            logits = model(batch_at(size)).mean(0)         # aggregate the 360 views
        log_prior = torch.zeros(len(classes), device=device)
        if prior_df is not None:
            vc = prior_df[head].astype(str).value_counts()
            freq = np.array([vc.get(c, 1) for c in classes], dtype=float)
            log_prior = torch.tensor(np.log(freq / freq.sum()), dtype=torch.float32, device=device)
        adj = logits + log_prior
        idx = int(adj.argmax())
        report["grade"][head] = {"label": classes[idx],
                                 "prob": round(float(adj.softmax(0)[idx]), 3)}

    # ---- multi-label inclusion head (max prob across frames per type) ----
    mdir = MODELS_ROOT / INCLUSION_DIR
    if (mdir / "best.pt").exists():
        meta = head_meta(mdir)
        types = meta["types"]
        thr = meta.get("threshold", 0.5)
        size = meta.get("img_size", 512)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(types))
        model.load_state_dict(torch.load(mdir / "best.pt", map_location=device))
        model.to(device).eval()
        with torch.no_grad():
            p = torch.sigmoid(model(batch_at(size))).cpu().numpy()
        maxp = p.max(0)  # multi-view: present if seen in ANY frame
        present = [types[k] for k in range(len(types)) if maxp[k] >= thr]
        report["inclusions"] = {"predicted": present,
                                "scores": {types[k]: round(float(maxp[k]), 2) for k in range(len(types))}}

    # ---- QC against the GIA cert ----
    if INPUT_CSV.exists():
        df = pd.read_csv(INPUT_CSV, dtype={"stone_id": str})
        row = df[df["stone_id"] == sid]
        if not row.empty:
            r = row.iloc[0]
            for head, pred in report["grade"].items():
                cert = "" if pd.isna(r.get(head)) else str(r.get(head))
                pred["cert"] = cert
                pred["match"] = (pred["label"] == cert) if cert else None
            cert_inc = "" if pd.isna(r.get("inclusion_types")) else str(r.get("inclusion_types"))
            report["inclusions"]["cert"] = cert_inc
            if report["inclusions"].get("predicted") is not None:
                pset, cset = set(report["inclusions"]["predicted"]), set(t for t in cert_inc.split("|") if t)
                report["inclusions"]["match_types"] = sorted(pset & cset)
                report["inclusions"]["missed"] = sorted(cset - pset)
                report["inclusions"]["extra"] = sorted(pset - cset)

    if args.json_only:
        print(json.dumps(report, indent=2, default=str)); return

    print(json.dumps(report, indent=2, default=str))
    print(f"\n=== GRADE REPORT  {sid}  ({len(frames)} frames) ===")
    for head, p in report["grade"].items():
        cert = p.get("cert") or "?"
        flag = "" if p.get("match") is None else ("  OK" if p["match"] else "  MISMATCH")
        print(f"  {head:<20} {str(p['label']):<16} (p={p['prob']:.2f})   cert={cert}{flag}")
    if report["inclusions"].get("predicted") is not None:
        print(f"  inclusions  predicted: {report['inclusions']['predicted']}")
        print(f"              cert:      {report['inclusions'].get('cert','?')}")


if __name__ == "__main__":
    main()
