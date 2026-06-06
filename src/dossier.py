"""The capstone: a single unified "Diamond Dossier" per stone.

    python src/dossier.py <stone_id>

One cohesive page pairing EVERYTHING the system produces, each element once:
  - the real video frame
  - the 3D reconstruction (shape-aware, from predicted proportions)
  - the GIA-style proportions diagram (cross-section + dimensions)
  - the inclusion Grad-CAM heatmap (where the inclusions are)
  - the full grade table: shape/color/clarity/eye-clean/fluorescence + geometry +
    mm dimensions + inclusions, each compared to the GIA cert (✓/✗).

Output: data/processed/dossier/<stone>_dossier.jpg
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_silhouette import segment
from reconstruct_3d import (parametric_outline, real_outline, brilliant_mesh,
                            step_mesh, princess_mesh, STEP_SHAPES)

MODELS = Path("data/models")
FRAMES = Path("data/processed/frames")
CSV = Path("data/processed/stone_records.csv")
OUT = Path("data/processed/dossier")
INK = "#1b2a3a"
SINGLE = {"shape_group": "shape_group_resnet18", "color_group": "color_group_resnet18",
          "fluorescence_group": "fluorescence_group_resnet18", "eye_clean": "eye_clean_resnet18"}


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: dossier.py <stone_id>")
    import torch
    from torch import nn
    from torchvision import models, transforms
    from PIL import Image
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    frames = sorted((FRAMES / sid).glob("frame_*.jpg"))
    if not frames:
        sys.exit(f"no frames for {sid}")
    pil = [Image.open(f).convert("RGB") for f in frames]
    df = pd.read_csv(CSV, dtype={"stone_id": str}); cert = df.set_index("stone_id").loc[sid]
    shape = str(cert.get("shape_group"))
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def batch(sz):
        tf = transforms.Compose([transforms.Resize((sz, sz)), transforms.ToTensor(), transforms.Normalize(mean, std)])
        return torch.stack([tf(im) for im in pil]).to(dev)

    def resnet(n, bb="resnet18"):
        m = models.resnet50(weights=None) if bb == "resnet50" else models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n); return m

    grade = {}
    # single-label heads (prior-corrected)
    for head, dn in SINGLE.items():
        d = MODELS / dn
        if not (d / "best.pt").exists():
            continue
        meta = json.loads((d / "classes.json").read_text()); cls = meta["classes"]
        m = resnet(len(cls), meta.get("backbone", "resnet18"))
        m.load_state_dict(torch.load(d / "best.pt", map_location=dev)); m.to(dev).eval()
        with torch.no_grad():
            lg = m(batch(meta["img_size"])).mean(0)
        vc = df[head].astype(str).value_counts()
        lp = torch.tensor(np.log(np.array([vc.get(c, 1) for c in cls], float) / vc.sum()), device=dev)
        i = int((lg + lp).argmax()); cvv = "" if pd.isna(cert.get(head)) else str(cert.get(head))
        grade[head] = (cls[i], cvv, cls[i] == cvv if cvv else None)
    # clarity ordinal
    cd = MODELS / "clarity_ordinal_resnet18"
    if (cd / "best.pt").exists():
        meta = json.loads((cd / "meta.json").read_text()); order = meta["order"]
        m = resnet(1); m.load_state_dict(torch.load(cd / "best.pt", map_location=dev)); m.to(dev).eval()
        with torch.no_grad():
            gi = int(round(min(max(float(m(batch(meta["img_size"])).squeeze(1).mean()), 0), len(order) - 1)))
        cvv = "" if pd.isna(cert.get("clarity_group")) else str(cert.get("clarity_group"))
        grade["clarity"] = (order[gi], cvv, order[gi] == cvv if cvv else None)
    # geometry
    gn = json.loads((MODELS / "geometry_resnet18" / "norm.json").read_text())
    gm = resnet(len(gn["targets"])); gm.load_state_dict(torch.load(MODELS / "geometry_resnet18" / "best.pt", map_location=dev)); gm.to(dev).eval()
    with torch.no_grad():
        prop = dict(zip(gn["targets"], (gm(batch(gn["img_size"])).cpu().numpy() * np.array(gn["std"]) + np.array(gn["mean"])).mean(0)))
    # inclusion + gradcam
    incl_pred, heat = [], None
    idir = MODELS / "inclusion_resnet18"
    if (idir / "best.pt").exists():
        meta = json.loads((idir / "classes.json").read_text()); types = meta["types"]
        thr = np.array(meta.get("thresholds", [meta.get("threshold", 0.5)] * len(types)))
        im_ = resnet(len(types)); im_.load_state_dict(torch.load(idir / "best.pt", map_location=dev)); im_.to(dev).eval()
        store = {}
        im_.layer4.register_forward_hook(lambda a, b, o: store.__setitem__("A", o))
        im_.layer4.register_full_backward_hook(lambda a, gi_, go: store.__setitem__("G", go[0]))
        xb = batch(meta["img_size"])
        with torch.no_grad():
            pr = torch.sigmoid(im_(xb)).cpu().numpy()
        mx = pr.max(0); incl_pred = [types[k] for k in range(len(types)) if mx[k] >= thr[k]]
        t = int(mx.argmax()); f = int(pr[:, t].argmax())
        xi = xb[f:f + 1].clone().requires_grad_(True); im_.zero_grad(); im_(xi)[0, t].backward()
        cam = torch.relu((store["G"][0].mean((1, 2))[:, None, None] * store["A"][0]).sum(0)).detach().cpu().numpy()
        cam = cv2.resize(cam / (cam.max() + 1e-8), (pil[f].width, pil[f].height))
        base = cv2.cvtColor(np.array(pil[f]), cv2.COLOR_RGB2BGR)
        heat = cv2.addWeighted(base, 0.6, cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.4, 0)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    # face-up frame + silhouette L/W
    def area(f):
        c, _ = cv2.findContours(segment(cv2.imread(str(f))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cv2.contourArea(max(c, key=cv2.contourArea)) if c else 0
    fu = max(frames, key=area)
    cnts, _ = cv2.findContours(segment(cv2.imread(str(fu))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, _), (w, h), _ = cv2.minAreaRect(max(cnts, key=cv2.contourArea)); lw = max(w, h) / max(min(w, h), 1)

    # mm dimensions
    sub = df.dropna(subset=["weight_ct", "mes_length", "mes_width", "mes_depth"])
    sub = sub[(sub.mes_width > 0) & (sub.mes_depth > 0)]
    C = ((sub.weight_ct * 56.818) / (sub.mes_length * sub.mes_width * sub.mes_depth)).groupby(sub.shape_group).median()
    dims = None
    if not pd.isna(cert.get("weight_ct")):
        V = float(cert["weight_ct"]) * 56.818; cc = float(C.get(shape, C.median()))
        W = (V / (cc * lw * prop["depth_pct"] / 100)) ** (1 / 3)
        dims = (lw * W, W, prop["depth_pct"] / 100 * W)

    # ---- build 3D mesh ----
    Rx = float(cert.get("mes_length", 6)) / 2 or 3.0
    Ry = Rx / max(prop.get("ratio", 1.0), 0.5)
    par = parametric_outline(shape, float(cert.get("ratio", 1.0)) if not pd.isna(cert.get("ratio")) else 1.0)
    outline = par if par is not None else real_outline(frames)
    outline = outline * [Rx / max(np.abs(outline[:, 0]).max(), 1e-6), Ry / max(np.abs(outline[:, 1]).max(), 1e-6)]
    tf_ = prop["table_pct"] / 100.0
    total_d = prop["depth_pct"] / 100.0 * 2 * Ry
    crown_h = (Rx - tf_ * Rx) * math.tan(math.radians(prop["crown_angle"])) if prop["crown_angle"] > 5 else total_d * 0.30
    pav_d = max(total_d - crown_h - total_d * 0.04, total_d * 0.4)
    star_f = (float(cert.get("star_length_pct")) if (not pd.isna(cert.get("star_length_pct")) and float(cert.get("star_length_pct")) > 1) else 50) / 100
    lower_f = (float(cert.get("lower_half_pct")) if (not pd.isna(cert.get("lower_half_pct")) and float(cert.get("lower_half_pct")) > 1) else 75) / 100
    if shape in STEP_SHAPES:
        faces, style = step_mesh(outline, tf_, crown_h, pav_d), "step"
    elif shape == "princess":
        faces, style = princess_mesh(outline, tf_, crown_h, pav_d, star_f), "princess"
    else:
        faces, style = brilliant_mesh(outline, tf_, crown_h, pav_d, star_f, lower_f), "brilliant"

    # =================== compose the dossier ===================
    fig = plt.figure(figsize=(15, 9), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.18, wspace=0.18)
    fig.suptitle(f"DIAMOND DOSSIER  (model-reconstructed from video)   —   {sid}",
                 fontsize=16, fontweight="bold", color=INK, y=0.97)

    ax = fig.add_subplot(gs[0, 0]); ax.imshow(cv2.cvtColor(cv2.imread(str(fu)), cv2.COLOR_BGR2RGB))
    ax.set_title("360° video", color=INK, fontsize=11); ax.axis("off")

    ax3 = fig.add_subplot(gs[0, 1], projection="3d")
    ax3.add_collection3d(Poly3DCollection(faces, facecolor=(0.62, 0.80, 0.95, 0.55), edgecolor=(0.1, 0.2, 0.35, 0.85), lw=0.5))
    sp = np.abs(outline).max() * 1.1
    ax3.set_xlim(-sp, sp); ax3.set_ylim(-sp, sp); ax3.set_zlim(-(crown_h + pav_d) * 0.72, (crown_h + pav_d) * 0.42)
    ax3.set_box_aspect((1, 1, 0.9)); ax3.view_init(24, 35); ax3.axis("off")
    ax3.set_title(f"3D reconstruction ({style})", color=INK, fontsize=11)

    # GIA proportions diagram
    axg = fig.add_subplot(gs[0, 2]); R = 1.0
    tr = prop["table_pct"] / 100 * R
    cdg = prop["crown_angle"] if prop["crown_angle"] > 5 else math.degrees(math.atan2((prop["depth_pct"]/100*2*R) * 0.30, R - tr))
    ch = (R - tr) * math.tan(math.radians(cdg)); gtt = prop["depth_pct"]/100*2*R * 0.03
    ph = max(prop["depth_pct"]/100*2*R - ch - gtt, prop["depth_pct"]/100*2*R*0.4); pdg = math.degrees(math.atan2(ph, R))
    yt, yg0, yg1, yc = ch, 0.0, -gtt, -(gtt+ph); is_step = shape in STEP_SHAPES; kw = 0.22*R if is_step else 0
    if is_step:
        mx, mh = tr+0.5*(R-tr), yt*0.5; px, pm = kw/2+0.5*(R-kw/2), yg1-ph*0.5
        rt = [(tr,yt),(mx,yt),(mx,mh),(R,mh),(R,yg0),(R,yg1),(px,yg1),(px,pm),(kw/2,pm),(kw/2,yc)]
    else:
        rt = [(tr,yt),(R,yg0),(R,yg1),(0,yc)]
    pp = rt + [(-x,y) for x,y in reversed(rt)] + [rt[0]]
    axg.plot([p[0] for p in pp],[p[1] for p in pp], color=INK, lw=2); axg.plot([-tr,tr],[yt,yt], color=INK, lw=2)
    axg.plot([-R,R],[yg0,yg0], color=INK, lw=0.6); axg.plot([-R,R],[yg1,yg1], color=INK, lw=0.6)
    axg.annotate("",(-tr,yt+0.14),(tr,yt+0.14),arrowprops=dict(arrowstyle="<->",color=INK,lw=1))
    axg.text(0,yt+0.2,f"Table {prop['table_pct']:.0f}%",ha="center",color=INK,fontsize=9)
    axg.annotate("",(R+0.28,yt),(R+0.28,yc),arrowprops=dict(arrowstyle="<->",color=INK,lw=1))
    axg.text(R+0.34,(yt+yc)/2,f"Depth {prop['depth_pct']:.1f}%",rotation=90,va="center",color=INK,fontsize=9)
    axg.add_patch(Arc((R,yg0),0.6,0.6,theta1=180-cdg,theta2=180,color=INK,lw=1)); axg.text(R-0.5,yg0+0.1,f"{cdg:.0f}deg",fontsize=8,color=INK)
    axg.set_xlim(-R-0.7,R+1.0); axg.set_ylim(yc-0.4,yt+0.45); axg.set_aspect("equal"); axg.axis("off")
    axg.set_title("GIA-style proportions", color=INK, fontsize=11)

    axh = fig.add_subplot(gs[1, 0])
    if heat is not None:
        axh.imshow(heat); axh.set_title("inclusion heatmap (Grad-CAM)", color=INK, fontsize=11)
    axh.axis("off")

    axt = fig.add_subplot(gs[1, 1:3]); axt.axis("off")
    def g(k): return grade.get(k, ("-", "", None))
    rows = [("", "predicted", "GIA cert", "")]
    for key, lab in [("shape_group","Shape"),("color_group","Color"),("clarity","Clarity"),
                     ("eye_clean","Eye-clean"),("fluorescence_group","Fluorescence")]:
        p_, c_, mt = g(key)
        rows.append((lab, str(p_), c_ or "-", "" if mt is None else ("OK" if mt else "MISS")))
    rows.append(("Geometry depth/table", f"{prop['depth_pct']:.1f}% / {prop['table_pct']:.0f}%",
                 f"{cert.get('depth_pct')}% / {cert.get('table_pct')}%", ""))
    if dims:
        cd_ = [cert.get('mes_length'), cert.get('mes_width'), cert.get('mes_depth')]
        rows.append(("Dimensions (mm)", f"{dims[0]:.2f}x{dims[1]:.2f}x{dims[2]:.2f}",
                     f"{cd_[0]}x{cd_[1]}x{cd_[2]}", ""))
    rows.append(("Carat", "weighed", f"{cert.get('weight_ct')}", ""))
    rows.append(("Inclusions", ", ".join(incl_pred)[:30] or "-",
                 str(cert.get("inclusion_types") or "-")[:26], ""))
    y = 0.98
    for a, b, c, mk in rows:
        bold = (b == "predicted")
        axt.text(0.0, y, a, fontsize=11, fontweight="bold" if bold else "normal", color=INK)
        axt.text(0.42, y, b, fontsize=10.5, color="#1f7a3f")
        axt.text(0.72, y, c, fontsize=10.5, color="#b06a00")
        axt.text(0.95, y, mk, fontsize=10.5, fontweight="bold",
                 color=("#1f7a3f" if mk == "OK" else "#b03030"))
        y -= 0.092
    axt.set_title("Predicted grade  vs  GIA cert", color=INK, fontsize=11, loc="left")

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{sid}_dossier.jpg"
    fig.savefig(dest, dpi=125, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"DOSSIER -> {dest}")


if __name__ == "__main__":
    main()
