"""3D reconstruction of a diamond from the geometry model + the stone's real outline.

    python src/reconstruct_3d.py <stone_id>

Improvements:
  #1 facet detail   : star-length% and lower-half% subdivide the crown/pavilion.
  #2 shape-specific : brilliant faceting for round/oval/pear/...; STEP faceting
                      (concentric terraces, keel) for emerald/asscher.
  #3 per-stone form : the GIRDLE OUTLINE is the stone's ACTUAL silhouette (not an
                      idealized ellipse), so each stone's real shape is reconstructed;
                      depth/angles come from the geometry model.
  #4 all-frames 3D  : an orthographic VISUAL HULL is carved from EVERY frame's
                      silhouette at its tumble angle — the 360° video pitches the
                      stone, so the views cover all elevations. The girdle outline
                      is the hull's widest slice and the crown/pavilion split comes
                      from the hull profile: the 3D form is pieced from all frames,
                      never from one image.
Output: data/processed/recon/<stone>_3d.jpg (real frame + carved hull + 2 rendered views).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_silhouette import segment

FRAMES = Path("data/processed/frames")
CSV = Path("data/processed/stone_records.csv")
GEOM = Path("data/models/geometry_resnet18")
OUT = Path("data/processed/recon")
STEP_SHAPES = {"emerald", "asscher"}


def _ring_pts(pts, n):
    """Resample a closed contour to n points by angle around the origin."""
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    out = []
    for k in range(n):
        ta = -np.pi + 2 * np.pi * k / n
        d = np.abs((ang - ta + np.pi) % (2 * np.pi) - np.pi)
        out.append(pts[d.argmin()])
    return np.array(out)


def _smooth_norm(out):
    """Light circular smoothing to remove jitter, then normalize max radius = 1."""
    k = np.array([0.25, 0.5, 0.25])
    out = np.c_[np.convolve(np.r_[out[-1, 0], out[:, 0], out[0, 0]], k, "valid"),
                np.convolve(np.r_[out[-1, 1], out[:, 1], out[0, 1]], k, "valid")]
    return out / np.abs(out).max()


def _masks(frames):
    """(filled silhouette, centroid, max radius, area) for every frame."""
    out = []
    for f in frames:
        m = segment(cv2.imread(str(f)))
        # kill the static pedestal bar: rows that reach both image borders
        bar = m[:, :4].any(1) & m[:, -4:].any(1)
        m[bar] = 0
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 100:
            continue
        fill = np.zeros(m.shape, np.uint8)
        cv2.drawContours(fill, [c], -1, 1, -1)
        mom = cv2.moments(c)
        cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
        r = float(np.sqrt(((c[:, 0, :] - [cx, cy]) ** 2).sum(1)).max())
        # open with a stone-sized kernel: drops the thin static holder stub,
        # which would otherwise survive every view and grow a fin on the hull
        k = max(3, int(r / 30) | 1)
        fill = cv2.morphologyEx(fill, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cnts, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        # cut diamonds are CONVEX bodies, so every true silhouette is convex:
        # convexifying repairs the bites a transparent stone leaves when it
        # melts into the background (solidity, kept from the raw contour,
        # still tells _good_views how damaged the frame was)
        solidity = cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1)
        fill = np.zeros(m.shape, np.uint8)
        cv2.drawContours(fill, [cv2.convexHull(c)], -1, 1, -1)
        out.append((fill, cx, cy, r, float(cv2.contourArea(c)), solidity))
    return out


def _good_views(ms):
    """Indices of trustworthy silhouettes. A transparent stone sometimes melts
    into the background: ragged masks (low solidity) or masks far smaller than
    their angular neighbors would bite chunks out of the hull — drop them.
    Both tests are RELATIVE: rigs differ in how clean masks can get, and the
    legitimate profile views are area minima that must not be dropped, so area
    is judged against the adjacent frames (smooth tumble => gentle dips)."""
    areas = np.array([m[4] for m in ms])
    sol = np.array([m[5] for m in ms])
    adj = np.array([min(areas[k - 1], areas[(k + 1) % len(ms)]) for k in range(len(ms))])
    good = (sol > sol.max() - 0.25) & (areas > 0.75 * adj)
    return np.where(good)[0] if good.sum() >= 4 else np.arange(len(ms))


def _largest_component(occ):
    """Largest 6-connected 3D component (flood fill from the centroid voxel)."""
    idx = np.argwhere(occ)
    if not len(idx):
        return occ
    c = idx.mean(0)
    seed = np.zeros_like(occ)
    seed[tuple(idx[np.abs(idx - c).sum(1).argmin()])] = True
    while True:
        grown = seed.copy()
        grown[1:] |= seed[:-1]; grown[:-1] |= seed[1:]
        grown[:, 1:] |= seed[:, :-1]; grown[:, :-1] |= seed[:, 1:]
        grown[:, :, 1:] |= seed[:, :, :-1]; grown[:, :, :-1] |= seed[:, :, 1:]
        grown &= occ
        if (grown == seed).all():
            return grown
        seed = grown


def _carve(ms, view_ids, k_up, R, lin, axis, slack=0):
    """Carve the chosen frames' silhouette cones (orthographic) into a voxel
    cube — strict intersection, since untrustworthy silhouettes are already
    dropped by _good_views. Tumble angles use each frame's ORIGINAL index —
    dropped bad frames leave a gap, not a shifted rotation."""
    grid = len(lin)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    x, y, z = X.ravel(), Y.ravel(), Z.ravel()
    cnt = np.zeros(x.shape, np.int16)
    for k in view_ids:
        mask, cx, cy = ms[k][0], ms[k][1], ms[k][2]
        th = 2 * np.pi * (k - k_up) / len(ms)            # this frame's tumble angle
        c_, s_ = np.cos(th), np.sin(th)
        if axis == "x":                                  # stone pitches about the horizontal image axis
            u, v = x, y * c_ - z * s_
        else:                                            # stone yaws about the vertical image axis
            u, v = x * c_ - z * s_, y
        col = np.clip((cx + u * R).astype(np.int32), 0, mask.shape[1] - 1)
        row = np.clip((cy + v * R).astype(np.int32), 0, mask.shape[0] - 1)
        cnt += mask[row, col] > 0
    return (cnt >= len(view_ids) - slack).reshape(grid, grid, grid)


def _detect_axis(ms):
    """The rotation axis is the image direction whose silhouette extent stays
    constant across the tumble — lengths along the axis are invariant, while
    the perpendicular extent shrinks and grows as the stone turns."""
    ws = np.array([float(m.any(0).sum()) for m, *_ in ms])   # horizontal extents
    hs = np.array([float(m.any(1).sum()) for m, *_ in ms])   # vertical extents
    return "x" if ws.std() / ws.mean() < hs.std() / hs.mean() else "y"


def visual_hull(frames, grid=96):
    """Carve the stone's 3D form from EVERY frame's silhouette (visual hull).

    The 360° video tumbles the stone, so each frame is a view from a different
    angle; intersecting all the silhouette cones pieces the true shape together
    from all of them — no single frame decides it. The rotation axis is detected
    from the invariant silhouette extent, and the hull is flipped so +z is the
    crown (shallow) side. Returns (occupancy[grid^3], voxel centers) or None.
    """
    ms = _masks(frames)
    if len(ms) < 4:
        return None
    good = _good_views(ms)
    k_up = int(good[np.argmax([ms[k][4] for k in good])])
    R = max(ms[k][3] for k in good) * 1.02
    lin = np.linspace(-1.02, 1.02, grid, dtype=np.float32)
    axis = _detect_axis([ms[k] for k in good])
    occ = _despindle(_carve(ms, good, k_up, R, lin, axis))
    occ = _open3d(_smooth3d(_largest_component(occ)), r=2)
    occ = _smooth3d(_align_to_symmetry(occ))
    rad = _slice_radii(occ)
    zs = np.where(rad > 0)[0]
    if len(zs) < 5:
        return None
    zi = int(rad.argmax())                               # girdle = widest slice
    if (zs.max() - zi) > (zi - zs.min()):                # deeper (pavilion) side on top -> flip
        occ = occ[:, :, ::-1]
    return occ, lin, len(good)


def _align_to_symmetry(occ):
    """Rotate the hull into the stone's own axes — the stone is mounted tilted,
    so the carve frame's z is NOT the table-culet axis. PCA of the occupied
    voxels gives the axes: symmetry (depth, smallest extent) -> z, long girdle
    axis -> y, so slices are girdle-parallel and the outline is upright."""
    idx = np.argwhere(occ).astype(np.float32)
    if len(idx) < 32:
        return occ
    ctr = idx.mean(0)
    evec = np.linalg.eigh(np.cov((idx - ctr).T))[1]      # eigenvalues ascending
    B = evec[:, [1, 2, 0]]                               # x = mid, y = long, z = depth axis
    if np.linalg.det(B) < 0:
        B[:, 0] *= -1
    g = occ.shape[0]
    cs = np.arange(g, dtype=np.float32) - (g - 1) / 2
    X, Y, Z = np.meshgrid(cs, cs, cs, indexing="ij")
    src = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1) @ B.T + ctr
    si = np.round(src).astype(np.int32)
    ok = ((si >= 0) & (si < g)).all(1)
    out = np.zeros(occ.size, bool)
    out[ok] = occ[si[ok, 0], si[ok, 1], si[ok, 2]]
    return out.reshape(g, g, g)


def _despindle(occ):
    """Remove the never-carved spindle along the rotation axis (carve-frame y):
    voxels on the axis project to the same image column in every view, so no
    silhouette can cut them. Per y-slice, keep the largest blob and drop slices
    with a tiny cross-section (the spindle poking past the stone's ends)."""
    areas = occ.sum((0, 2))
    amax = areas.max()
    for y in range(occ.shape[1]):
        if not areas[y]:
            continue
        if areas[y] < 0.03 * amax:
            occ[:, y, :] = False
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(occ[:, y, :].astype(np.uint8), 8)
        if n > 2:
            occ[:, y, :] = lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return occ


def _dilate6(occ):
    out = occ.copy()
    out[1:] |= occ[:-1]; out[:-1] |= occ[1:]
    out[:, 1:] |= occ[:, :-1]; out[:, :-1] |= occ[:, 1:]
    out[:, :, 1:] |= occ[:, :, :-1]; out[:, :, :-1] |= occ[:, :, 1:]
    return out


def _open3d(occ, r=3):
    """3D opening: removes protrusions thinner than ~2r voxels — in particular
    the spindle of never-carved voxels along the rotation axis (every view
    projects them onto the same image column, so silhouettes can't cut them)."""
    out = occ
    for _ in range(r):
        out = ~_dilate6(~out)                            # erode
    for _ in range(r):
        out = _dilate6(out)                              # dilate back
    return out & occ                                     # never grow past the hull


def _smooth3d(occ, thresh=14):
    """One 3x3x3 majority pass: kills voxel jaggies and thin hull-wedge flaps."""
    cnt = np.zeros(occ.shape, np.int8)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cnt += np.roll(occ, (dx, dy, dz), (0, 1, 2))
    return cnt >= thresh


def _slice_radii(occ):
    g = occ.shape[0]
    c = (g - 1) / 2
    ii, jj = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    rr = np.sqrt((ii - c) ** 2 + (jj - c) ** 2)
    return np.array([rr[occ[:, :, z]].max() if occ[:, :, z].any() else 0.0
                     for z in range(occ.shape[2])])


def _clip_ring(r, factor=1.18):
    """Clip narrow radial outliers (hull warts) to a circular median baseline;
    real outline features (lobes, tips) are angularly wide and pass through."""
    rad = np.sqrt((r ** 2).sum(1))
    pad = np.r_[rad[-2:], rad, rad[:2]]
    med = np.array([np.median(pad[i:i + 5]) for i in range(len(rad))])
    return r * np.minimum(1.0, factor * med / np.maximum(rad, 1e-9))[:, None]


def _slice_ring(sl, lin, n):
    """One z-slice's outline as n world-coordinate points (None if degenerate)."""
    cnts, _ = cv2.findContours(sl.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 6:
        return None
    # slice is indexed [ix, iy]; cv2 contour points are (col=iy, row=ix)
    return _clip_ring(_ring_pts(np.c_[lin[c[:, 0, 1]], lin[c[:, 0, 0]]], n))


def hull_outline(hv, n=28):
    """Girdle outline from the hull's widest slice — pieced from all frames."""
    occ, lin = hv[0], hv[1]
    zi = int(_slice_radii(occ).argmax())
    ring = _slice_ring(occ[:, :, zi], lin, n)
    return _smooth_norm(ring - ring.mean(0)) if ring is not None else None


def hull_crown_frac(hv):
    """Fraction of total depth above the girdle (crown height / total depth)."""
    occ, lin = hv[0], hv[1]
    rad = _slice_radii(occ)
    zs = np.where(rad > 0)[0]
    zi = int(rad.argmax())
    return (zs.max() - zi) / max(zs.max() - zs.min(), 1)


def hull_mesh(hv, n=28, levels=20):
    """Display mesh of the carved hull: stacked slice rings (girdle at z=0)."""
    occ, lin = hv[0], hv[1]
    zs = np.where(occ.any((0, 1)))[0]
    zi = int(_slice_radii(occ).argmax())
    sel = np.unique(np.linspace(zs.min(), zs.max(), levels).astype(int))
    k = np.array([0.25, 0.5, 0.25])
    rings = []
    for z in sel:
        r = _slice_ring(occ[:, :, z], lin, n)
        if r is None:
            continue
        r = np.c_[np.convolve(np.r_[r[-1, 0], r[:, 0], r[0, 0]], k, "valid"),   # circular smoothing
                  np.convolve(np.r_[r[-1, 1], r[:, 1], r[0, 1]], k, "valid")]
        rings.append(np.c_[r, np.full(n, lin[z] - lin[zi])])
    if len(rings) < 3:
        return None
    faces = [rings[-1][::-1].tolist()]                   # table cap
    for a, b in zip(rings, rings[1:]):
        for i in range(n):
            j = (i + 1) % n
            faces.append([a[i].tolist(), a[j].tolist(), b[j].tolist(), b[i].tolist()])
    faces.append(rings[0].tolist())                      # culet cap
    return faces


def real_outline(frames, n=28):
    """The stone's girdle outline pieced from ALL frames (visual-hull widest
    slice); falls back to the single largest face-up silhouette if carving fails."""
    hv = visual_hull(frames)
    if hv is not None:
        out = hull_outline(hv, n)
        if out is not None:
            return out
    return _faceup_outline(frames, n)


def _faceup_outline(frames, n=28):
    """Fallback: outline from the largest single (face-up) silhouette only."""
    best_area, best = 0, None
    for f in frames:
        m = segment(cv2.imread(str(f)))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > best_area:
            best_area, best = cv2.contourArea(c), c
    if best is None:
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.c_[np.cos(a), np.sin(a)]
    pts = best[:, 0, :].astype(float)
    return _smooth_norm(_ring_pts(pts - pts.mean(0), n))


def _edges(corners, per=5):
    """Densify a polygon's edges, keeping the corners sharp."""
    pts = []
    for i in range(len(corners)):
        a, b = np.array(corners[i]), np.array(corners[(i + 1) % len(corners)])
        for t in np.linspace(0, 1, per, endpoint=False):
            pts.append(a + (b - a) * t)
    return np.array(pts)


def parametric_outline(shape, ratio):
    """Clean, crisp-cornered outline for shapes the silhouette rounds off.
    Returns None for smooth shapes (use the real silhouette instead)."""
    # GIA display orientation: LONG axis vertical (y), short axis horizontal (x).
    wx, wy = 1.0 / max(ratio, 0.5), 1.0
    if shape == "round":                                             # perfect circle
        a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        return np.c_[np.cos(a), np.sin(a)]
    if shape == "oval":                                              # clean ellipse from L/W
        a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        return np.c_[np.cos(a) * wx, np.sin(a) * wy]
    if shape == "marquise":                                          # navette: two arcs meeting at sharp tips
        a_, b_ = wy, wx                                              # a_ = half-length (y), b_ = half-width (x)
        xc = (b_ ** 2 - a_ ** 2) / (2 * b_)                          # arc centre on x-axis (xc < 0)
        r2 = a_ ** 2 + xc ** 2
        ys = np.linspace(-a_, a_, 26)
        xr = xc + np.sqrt(np.maximum(r2 - ys ** 2, 0))               # right arc; tips at (0, ±a_)
        return np.vstack([np.c_[xr, ys], np.c_[-xr[::-1], ys[::-1]]])
    if shape in {"emerald", "radiant"}:                              # cut-corner rectangle (sharp)
        c = 0.10
        cor = [(wx, wy - c * wy), (wx - c * wx, wy), (-(wx - c * wx), wy), (-wx, wy - c * wy),
               (-wx, -(wy - c * wy)), (-(wx - c * wx), -wy), (wx - c * wx, -wy), (wx, -(wy - c * wy))]
        return _edges(cor, 7)
    if shape == "asscher":                                           # cut-corner square
        c = 0.18
        cor = [(1, 1 - c), (1 - c, 1), (-(1 - c), 1), (-1, 1 - c),
               (-1, -(1 - c)), (-(1 - c), -1), (1 - c, -1), (1, -(1 - c))]
        return _edges(cor, 7)
    if shape == "princess":                                          # sharp square
        return _edges([(wx, wy), (-wx, wy), (-wx, -wy), (wx, -wy)], 7)
    if shape == "cushion":                                           # superellipse (rounded square)
        a = np.linspace(0, 2 * np.pi, 28, endpoint=False)
        p = 0.42
        return np.c_[np.sign(np.cos(a)) * np.abs(np.cos(a)) ** p * wx,
                     np.sign(np.sin(a)) * np.abs(np.sin(a)) ** p * wy]
    return None


def chevron_pavilion(outline, pav_d):
    """Princess pavilion: inverted pyramid from the square girdle to the culet
    (the corner-to-culet edges form the chevron / X)."""
    n = len(outline)
    girdle = ring(outline, 1.0, 0.0)
    mid = ring(outline, 0.5, -pav_d * 0.5)
    culet = [0.0, 0.0, -pav_d]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([girdle[i], girdle[j], mid[j], mid[i]])
        faces.append([mid[i], mid[j], culet])
    return faces, girdle


def ring(outline, scale, z):
    return [[p[0] * scale, p[1] * scale, z] for p in outline]


def brilliant_mesh(outline, table_f, crown_h, pav_d, star_f, lower_f):
    n = len(outline)
    girdle = ring(outline, 1.0, 0.0)
    star_z = crown_h * (1 - star_f)                                   # star tips sit below the table
    star = ring(outline, table_f + (1 - table_f) * (1 - star_f), star_z)
    table = ring(outline, table_f, crown_h)
    lower = ring(outline, 1 - lower_f, -pav_d * lower_f)              # lower-girdle ring
    culet = [0.0, 0.0, -pav_d]
    faces = [table]
    for i in range(n):
        j = (i + 1) % n
        faces.append([table[i], star[i], star[j], table[j]])         # crown upper (table->star)
        faces.append([star[i], girdle[i], girdle[j], star[j]])       # crown lower (star->girdle)
        faces.append([girdle[i], lower[i], lower[j], girdle[j]])     # pavilion upper (girdle->lower)
        faces.append([lower[i], culet, lower[j]])                    # pavilion main (lower->culet)
    return faces


def step_mesh(outline, table_f, crown_h, pav_d):
    """Step cut: concentric terraces, ends in a keel line (not a point)."""
    n = len(outline)
    levels_up = [(table_f, crown_h), (0.78, crown_h * 0.45), (1.0, 0.0)]        # table -> girdle
    levels_dn = [(1.0, 0.0), (0.6, -pav_d * 0.55), (0.22, -pav_d)]              # girdle -> keel
    rings = [ring(outline, s, z) for s, z in levels_up] + [ring(outline, s, z) for s, z in levels_dn[1:]]
    faces = [rings[0]]                                                # table
    for r in range(len(rings) - 1):
        a, b = rings[r], rings[r + 1]
        for i in range(n):
            j = (i + 1) % n
            faces.append([a[i], a[j], b[j], b[i]])                    # terrace wall
    return faces


def princess_mesh(outline, table_f, crown_h, pav_d, star_f):
    """Square outline, brilliant-style crown, chevron (inverted-pyramid) pavilion."""
    n = len(outline)
    girdle = ring(outline, 1.0, 0.0)
    table = ring(outline, table_f, crown_h)
    star = ring(outline, table_f + (1 - table_f) * (1 - star_f), crown_h * (1 - star_f))
    faces = [table]
    for i in range(n):
        j = (i + 1) % n
        faces.append([table[i], star[i], star[j], table[j]])
        faces.append([star[i], girdle[i], girdle[j], star[j]])
    pav, _ = chevron_pavilion(outline, pav_d)
    return faces + pav


def render(faces, span, depth, view, path):
    fig = plt.figure(figsize=(4, 4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(faces, facecolor=(0.62, 0.80, 0.95, 0.55),
                                         edgecolor=(0.10, 0.20, 0.35, 0.9), linewidths=0.6))
    ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_zlim(-depth * 0.72, depth * 0.42)
    ax.set_box_aspect((1, 1, 0.9)); ax.view_init(elev=view[0], azim=view[1]); ax.set_axis_off()
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: reconstruct_3d.py <stone_id>")
    import torch
    from torch import nn
    from torchvision import models, transforms
    from PIL import Image

    frames = sorted((FRAMES / sid).glob("frame_*.jpg"))
    if not frames:
        sys.exit(f"no frames for {sid}")
    norm = json.loads((GEOM / "norm.json").read_text())
    tg, gm, gs, size = norm["targets"], np.array(norm["mean"]), np.array(norm["std"]), norm["img_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, len(tg))
    m.load_state_dict(torch.load(GEOM / "best.pt", map_location=device)); m.to(device).eval()
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    x = torch.stack([tf(Image.open(f).convert("RGB")) for f in frames]).to(device)
    with torch.no_grad():
        p = (m(x).cpu().numpy() * gs + gm).mean(0)
    prop = dict(zip(tg, p))

    cert = pd.read_csv(CSV, dtype={"stone_id": str}).set_index("stone_id").loc[sid]
    shape = str(cert.get("shape_group"))
    Rx = float(cert.get("mes_length", 6)) / 2 or 3.0
    # piece the true 3D form together from ALL frames (visual hull)
    hv = visual_hull(frames)
    # crisp parametric outline for cornered shapes; hull girdle slice for smooth ones
    par = parametric_outline(shape, float(cert.get("ratio", 1.0)) if not pd.isna(cert.get("ratio")) else 1.0)
    outline = par
    if outline is None and hv is not None:
        outline = hull_outline(hv)
    if outline is None:
        outline = _faceup_outline(frames)
    # scale outline to mm so x:y matches the real L/W (long axis = y / length = Rx)
    sx = (Rx / max(prop.get("ratio", 1.0), 0.5)) / max(np.abs(outline[:, 0]).max(), 1e-6)  # short -> x
    sy = Rx / max(np.abs(outline[:, 1]).max(), 1e-6)                                       # long  -> y
    outline = outline * [sx, sy]

    Ry = Rx / max(prop.get("ratio", 1.0), 0.5)
    table_f = prop["table_pct"] / 100.0
    total_depth = prop["depth_pct"] / 100.0 * 2 * Ry                  # depth% = depth/width; always reported
    if prop["crown_angle"] > 5:                                       # round brilliants report a real crown angle
        crown_h = (Rx - table_f * Rx) * np.tan(np.radians(prop["crown_angle"]))
    else:                                                             # fancy shapes: crown/pavilion split measured
        cf = hull_crown_frac(hv) if hv is not None else 0.30          # from the all-frames hull profile
        crown_h = total_depth * float(np.clip(cf, 0.20, 0.42))
    pav_d = max(total_depth - crown_h - total_depth * 0.04, total_depth * 0.4)
    # #1 facet detail from cert star/lower-half (typical defaults when not reported)
    def _facet(col, default):
        v = cert.get(col)
        return (float(v) if (not pd.isna(v) and float(v) > 1) else default) / 100.0
    star_f, lower_f = _facet("star_length_pct", 50), _facet("lower_half_pct", 75)

    if shape in STEP_SHAPES:
        faces = step_mesh(outline, table_f, crown_h or 0.3 * Rx, pav_d)
        style = "step cut"
    elif shape == "princess":
        faces = princess_mesh(outline, table_f, crown_h or 0.3 * Rx, pav_d, star_f)
        style = "princess"
    else:
        faces = brilliant_mesh(outline, table_f, crown_h, pav_d, star_f, lower_f)
        style = "brilliant"

    OUT.mkdir(parents=True, exist_ok=True)
    span = np.abs(outline).max() * 1.1
    render(faces, span, crown_h + pav_d, (22, 35), OUT / f"_{sid}_p.png")
    render(faces, span, crown_h + pav_d, (2, 0), OUT / f"_{sid}_s.png")
    hfaces = hull_mesh(hv) if hv is not None else None
    if hfaces:
        hz = np.array([p[2] for f in hfaces for p in f])
        render(hfaces, 1.08, hz.max() - hz.min(), (22, 35), OUT / f"_{sid}_h.png")

    def frame_area(f):
        cnts, _ = cv2.findContours(segment(cv2.imread(str(f))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cv2.contourArea(max(cnts, key=cv2.contourArea)) if cnts else 0
    faceup_frame = max(frames, key=frame_area)
    fu = cv2.resize(cv2.imread(str(faceup_frame)), (360, 360))
    cv2.putText(fu, "video", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    tiles = [fu]
    if hfaces:
        hu = cv2.resize(cv2.imread(str(OUT / f"_{sid}_h.png")), (360, 360))
        cv2.putText(hu, f"carved hull ({hv[2]}/{len(frames)} views)", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 90, 20), 2)
        tiles.append(hu)
    pe = cv2.resize(cv2.imread(str(OUT / f"_{sid}_p.png")), (360, 360))
    sd = cv2.resize(cv2.imread(str(OUT / f"_{sid}_s.png")), (360, 360))
    cv2.putText(pe, f"3D ({style})", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 90, 20), 2)
    cv2.putText(sd, "3D profile", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 90, 20), 2)
    row = np.hstack(tiles + [pe, sd])
    cap = np.full((70, row.shape[1], 3), 25, np.uint8)
    txt = (f"{sid} ({shape}, {style})  table {prop['table_pct']:.0f}%  crown {prop['crown_angle']:.0f}deg  "
           f"pavilion {prop['pavilion_depth']:.0f}%  depth {prop['depth_pct']:.0f}%  L/W {prop['ratio']:.2f}  "
           f"star {star_f*100:.0f}% lower {lower_f*100:.0f}%"
           + (f"  |  hull carved from {hv[2]}/{len(frames)} views" if hv is not None else ""))
    cv2.putText(cap, txt, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    cv2.imwrite(str(OUT / f"{sid}_3d.jpg"), np.vstack([row, cap]))
    print(f"=== 3D  {sid}  ({shape}, {style}) ===")
    print("  proportions:", {k: round(float(v), 1) for k, v in prop.items()},
          "star%", round(star_f * 100), "lower%", round(lower_f * 100))
    print(f"  -> {OUT / f'{sid}_3d.jpg'}")


if __name__ == "__main__":
    main()
