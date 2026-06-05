"""Unified 'digital cert' for one stone -- the capstone deliverable.

    python src/digital_cert.py <stone_id>

Runs the whole stack on a stone's 360 frames and produces ONE artifact:
  * grade: shape, color, clarity (ordinal), eye-clean, fluorescence  (prior-corrected)
  * geometry: L/W, depth%, table%, crown, pavilion  + reconstructed L x W x D (mm)
  * inclusions: predicted types + a Grad-CAM location heatmap
  * QC: every field compared to the GIA cert (match / MISS)
Writes data/processed/cert/<stone>_cert.jpg (visual) and <stone>_cert.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_silhouette import segment

FRAMES = Path("data/processed/frames")
MODELS = Path("data/models")
CSV = Path("data/processed/stone_records.csv")
OUT = Path("data/processed/cert")
SINGLE = {"shape_group": "shape_group_resnet18", "color_group": "color_group_resnet18",
          "fluorescence_group": "fluorescence_group_resnet18", "eye_clean": "eye_clean_resnet18"}


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: digital_cert.py <stone_id>")
    import torch
    from torch import nn
    from torchvision import models, transforms
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = sorted((FRAMES / sid).glob("frame_*.jpg"))
    if not frames:
        sys.exit(f"no frames for {sid}")
    bgr = [cv2.imread(str(f)) for f in frames]
    pil = [Image.open(f).convert("RGB") for f in frames]
    df = pd.read_csv(CSV, dtype={"stone_id": str})
    cert = df.set_index("stone_id").loc[sid]
    mean_n, std_n = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def batch(size):
        tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                 transforms.Normalize(mean_n, std_n)])
        return torch.stack([tf(im) for im in pil]).to(device)

    def resnet(n):
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, n)
        return m

    rep = {"stone_id": sid, "grade": {}, "geometry": {}, "inclusions": {}}

    # ---- single-label heads (prior-corrected) ----
    for head, dn in SINGLE.items():
        d = MODELS / dn
        if not (d / "best.pt").exists():
            continue
        meta = json.loads((d / "classes.json").read_text())
        cls, size = meta["classes"], meta["img_size"]
        m = resnet(len(cls)); m.load_state_dict(torch.load(d / "best.pt", map_location=device)); m.to(device).eval()
        with torch.no_grad():
            lg = m(batch(size)).mean(0)
        vc = df[head].astype(str).value_counts()
        lp = torch.tensor(np.log(np.array([vc.get(c, 1) for c in cls], float) / vc.sum()), device=device)
        i = int((lg + lp).argmax())
        cv = "" if pd.isna(cert.get(head)) else str(cert.get(head))
        rep["grade"][head] = {"pred": cls[i], "cert": cv, "match": cls[i] == cv if cv else None}

    # ---- clarity (ordinal) ----
    cd = MODELS / "clarity_ordinal_resnet18"
    if (cd / "best.pt").exists():
        meta = json.loads((cd / "meta.json").read_text()); order, size = meta["order"], meta["img_size"]
        m = resnet(1); m.load_state_dict(torch.load(cd / "best.pt", map_location=device)); m.to(device).eval()
        with torch.no_grad():
            val = float(m(batch(size)).squeeze(1).mean())
        gi = int(round(min(max(val, 0), len(order) - 1)))
        cv = "" if pd.isna(cert.get("clarity_group")) else str(cert.get("clarity_group"))
        rep["grade"]["clarity"] = {"pred": order[gi], "cert": cv,
                                   "match": order[gi] == cv if cv else None,
                                   "within1": (abs(gi - order.index(cv)) <= 1) if cv in order else None}

    # ---- geometry proportions ----
    gd = MODELS / "geometry_resnet18"
    norm = json.loads((gd / "norm.json").read_text())
    tg, gm, gs, size = norm["targets"], np.array(norm["mean"]), np.array(norm["std"]), norm["img_size"]
    m = resnet(len(tg)); m.load_state_dict(torch.load(gd / "best.pt", map_location=device)); m.to(device).eval()
    with torch.no_grad():
        gp = (m(batch(size)).cpu().numpy() * gs + gm).mean(0)
    prop = dict(zip(tg, gp))

    # silhouette L/W (+ pick face-up / profile frames)
    areas, lws = [], []
    for im in bgr:
        msk = segment(im); c, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not c:
            areas.append(0); lws.append(1.0); continue
        cc = max(c, key=cv2.contourArea); (_, _), (w, h), _ = cv2.minAreaRect(cc)
        areas.append(cv2.contourArea(cc)); lws.append(max(w, h) / max(min(w, h), 1))
    faceup, profile = int(np.argmax(areas)), int(np.argmax(lws))
    ratio = lws[faceup]

    # mm dimensions (proportions + weighed carat)
    sub = df.dropna(subset=["weight_ct", "mes_length", "mes_width", "mes_depth"])
    sub = sub[(sub.mes_width > 0) & (sub.mes_depth > 0)]
    C = ((sub.weight_ct * 56.818) / (sub.mes_length * sub.mes_width * sub.mes_depth)).groupby(sub.shape_group).median()
    dims = None
    if not pd.isna(cert.get("weight_ct")):
        V = float(cert["weight_ct"]) * 56.818
        c = float(C.get(cert["shape_group"], C.median()))
        W = (V / (c * ratio * prop["depth_pct"] / 100)) ** (1 / 3)
        dims = {"length": ratio * W, "width": W, "depth": prop["depth_pct"] / 100 * W}
    rep["geometry"] = {"lw_ratio": round(ratio, 3), **{k: round(float(v), 1) for k, v in prop.items()},
                       "dims_mm": {k: round(v, 2) for k, v in dims.items()} if dims else None,
                       "cert_ratio": float(cert.get("ratio")) if not pd.isna(cert.get("ratio")) else None,
                       "cert_dims_mm": [float(cert.get("mes_length")), float(cert.get("mes_width")), float(cert.get("mes_depth"))]
                       if not pd.isna(cert.get("mes_depth")) else None,
                       "carat_weighed": float(cert.get("weight_ct")) if not pd.isna(cert.get("weight_ct")) else None}

    # ---- inclusions + Grad-CAM for the top type ----
    incd = MODELS / "inclusion_resnet18"
    heat = None
    if (incd / "best.pt").exists():
        meta = json.loads((incd / "classes.json").read_text()); types, size = meta["types"], meta["img_size"]
        thr = np.array(meta.get("thresholds", [meta.get("threshold", 0.5)] * len(types)))
        m = resnet(len(types)); m.load_state_dict(torch.load(incd / "best.pt", map_location=device)); m.to(device).eval()
        store = {}
        m.layer4.register_forward_hook(lambda a, b, o: store.__setitem__("A", o))
        m.layer4.register_full_backward_hook(lambda a, gi, go: store.__setitem__("G", go[0]))
        xb = batch(size)
        with torch.no_grad():
            p = torch.sigmoid(m(xb)).cpu().numpy()
        maxp = p.max(0)
        present = [types[k] for k in range(len(types)) if maxp[k] >= thr[k]]
        cset = {t for t in str(cert.get("inclusion_types") or "").split("|") if t}
        rep["inclusions"] = {"pred": present, "cert": sorted(cset),
                             "matched": sorted(set(present) & cset)}
        # Grad-CAM on the top predicted type, on the frame where it's clearest
        t = int(maxp.argmax()); f = int(p[:, t].argmax())
        xi = xb[f:f + 1].clone().requires_grad_(True)
        m.zero_grad(); m(xi)[0, t].backward()
        cam = torch.relu((store["G"][0].mean((1, 2))[:, None, None] * store["A"][0]).sum(0)).detach().cpu().numpy()
        cam = cv2.resize(cam / (cam.max() + 1e-8), (pil[f].width, pil[f].height))
        base = cv2.cvtColor(np.array(pil[f]), cv2.COLOR_RGB2BGR)
        heat = (cv2.addWeighted(base, 0.6, cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.4, 0), types[t])

    # ---- render the digital cert ----
    def panel(img, label):
        p = cv2.resize(img, (340, 340)); cv2.putText(p, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return p
    top = [panel(bgr[faceup], "face-up"), panel(bgr[profile], "profile")]
    if heat is not None:
        top.append(panel(heat[0], f"incl: {heat[1]}"))
    else:
        top.append(np.full((340, 340, 3), 30, np.uint8))
    toprow = np.hstack(top)

    txt = np.full((420, toprow.shape[1], 3), 22, np.uint8)
    def line(y, a, b, c, ok=None):
        cv2.putText(txt, a, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1)
        cv2.putText(txt, str(b), (430, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130, 255, 130), 1)
        cv2.putText(txt, str(c), (660, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130, 200, 255), 1)
        if ok is not None:
            cv2.putText(txt, "OK" if ok else "MISS", (880, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (120, 255, 120) if ok else (120, 120, 255), 2)
    line(28, f"DIGITAL CERT  {sid}", "predicted", "cert", None)
    y = 64
    for k in ("shape_group", "color_group", "eye_clean", "fluorescence_group"):
        if k in rep["grade"]:
            g = rep["grade"][k]; line(y, k.replace("_group", ""), g["pred"], g["cert"] or "-", g["match"]); y += 30
    if "clarity" in rep["grade"]:
        g = rep["grade"]["clarity"]; line(y, "clarity (ordinal)", g["pred"], g["cert"] or "-", g["match"]); y += 30
    gg = rep["geometry"]
    line(y, "L/W ratio", gg["lw_ratio"], gg["cert_ratio"]); y += 30
    line(y, "depth% / table%", f"{gg['depth_pct']}/{gg['table_pct']}",
         f"{cert.get('depth_pct')}/{cert.get('table_pct')}"); y += 30
    if gg["dims_mm"]:
        d = gg["dims_mm"]; cd_ = gg["cert_dims_mm"]
        line(y, "dims LxWxD (mm)", f"{d['length']}x{d['width']}x{d['depth']}",
             f"{cd_[0]}x{cd_[1]}x{cd_[2]}" if cd_ else "-"); y += 30
    line(y, "carat (weighed)", gg["carat_weighed"], gg["carat_weighed"]); y += 30
    if rep["inclusions"]:
        line(y, "inclusions", ",".join(rep["inclusions"]["pred"])[:16] or "-",
             ",".join(rep["inclusions"]["cert"])[:16] or "-"); y += 30

    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{sid}_cert.jpg"), np.vstack([toprow, txt]))
    (OUT / f"{sid}_cert.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"=== DIGITAL CERT  {sid} ===")
    print(json.dumps(rep, indent=2, default=str))
    print(f"visual -> {OUT / f'{sid}_cert.jpg'}")


if __name__ == "__main__":
    main()
