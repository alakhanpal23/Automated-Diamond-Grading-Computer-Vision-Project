"""Localize inclusions with Grad-CAM: show WHERE the inclusion model is looking.

    python src/localize_inclusions.py <stone_id> [--top 3]

For the stone's most-confidently-predicted inclusion types, finds the frame where
each is most visible, runs Grad-CAM on the inclusion model's last conv block, and
writes a heatmap overlay to data/processed/localize/<stone>/<type>.jpg. Also
counts heatmap hotspots and compares to the GIA plot red-mark count (if a
data/processed/cert_inclusions.jsonl entry exists) as a sanity cross-check.

This is weakly-supervised localization: no box labels needed -- the heatmap shows
where the evidence for each inclusion type sits in the real frame. The GIA plot is
an approximate schematic, so this answers "where in the video" not "GIA-exact".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FRAMES_DIR = Path("data/processed/frames")
MODEL_DIR = Path("data/models/inclusion_resnet18")
CERT_JSONL = Path("data/processed/cert_inclusions.jsonl")
OUT_DIR = Path("data/processed/localize")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grad-CAM inclusion localization")
    p.add_argument("stone_id")
    p.add_argument("--top", type=int, default=3, help="Localize the top-N predicted types")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import cv2
        import torch
        from torch import nn
        from torchvision import models, transforms
        from PIL import Image
    except ImportError as e:
        sys.exit(f"ERROR: missing dependency: {e}")

    if not (MODEL_DIR / "best.pt").exists():
        sys.exit("ERROR: inclusion model not found; train it first")
    meta = json.loads((MODEL_DIR / "classes.json").read_text())
    types, size = meta["types"], meta.get("img_size", 512)

    sid = args.stone_id
    frames = sorted((FRAMES_DIR / sid).glob("frame_*.jpg"))
    if not frames:
        sys.exit(f"ERROR: no frames for {sid}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(types))
    model.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location=device))
    model.to(device).eval()

    # Grad-CAM hooks on the last conv block (layer4)
    store = {}
    model.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    model.layer4.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tf = transforms.Compose([transforms.Resize((size, size)),
                             transforms.ToTensor(), transforms.Normalize(mean, std)])
    pil = [Image.open(f).convert("RGB") for f in frames]
    x = torch.stack([tf(im) for im in pil]).to(device)

    # per-frame per-type probabilities -> pick top types + their best frame
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).cpu().numpy()       # (F, T)
    type_max = probs.max(0)                                 # best prob per type across frames
    order = np.argsort(-type_max)
    chosen = [t for t in order if type_max[t] >= 0.5][:args.top] or list(order[:args.top])

    out = OUT_DIR / sid
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for t in chosen:
        f = int(probs[:, t].argmax())                       # frame where this type is clearest
        xi = x[f:f + 1].clone().requires_grad_(True)
        model.zero_grad()
        score = model(xi)[0, t]
        score.backward()
        A = store["act"][0]                                 # (C, h, w)
        G = store["grad"][0]                                # (C, h, w)
        w = G.mean(dim=(1, 2))
        cam = torch.relu((w[:, None, None] * A).sum(0)).detach().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (pil[f].width, pil[f].height))

        # overlay heatmap on the original frame
        base = cv2.cvtColor(np.array(pil[f]), cv2.COLOR_RGB2BGR)
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(base, 0.6, heat, 0.4, 0)
        dest = out / f"{types[t]}_frame{f:02d}.jpg"
        cv2.imwrite(str(dest), overlay)

        # count hotspots (connected components above a high CAM threshold)
        hot = (cam >= 0.6).astype(np.uint8)
        n_hot = max(0, cv2.connectedComponents(hot)[0] - 1)
        results.append((types[t], float(type_max[t]), f, n_hot, str(dest)))

    # GIA cross-check
    gia_marks = None
    if CERT_JSONL.exists():
        for line in CERT_JSONL.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("stone_id") == sid:
                gia_marks = o.get("red_mark_clusters")
                break

    print(f"=== INCLUSION LOCALIZATION  {sid} ===")
    for typ, p, f, n_hot, dest in results:
        print(f"  {typ:<16} conf={p:.2f}  best frame={f:02d}  hotspots={n_hot}  -> {dest}")
    if gia_marks is not None:
        print(f"  GIA plot red-mark clusters (cross-check): {gia_marks}")
    print(f"  heatmap overlays written under {out}/")


if __name__ == "__main__":
    main()
