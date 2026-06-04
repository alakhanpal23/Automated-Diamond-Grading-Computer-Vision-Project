"""Build train/val/test manifests for the shape model (and friends).

Run from the repo root once 03_extract_frames.py has populated data/processed/frames/:

    python src/04_build_manifest.py [--frames-dir data/processed/frames]
                                    [--ratios 0.70 0.15 0.15]
                                    [--seed 42]
                                    [--exclude-shape-other]

Outputs (in data/processed/):
    manifest_train.csv
    manifest_val.csv
    manifest_test.csv
    manifest_split_summary.json

Splits are stratified by shape_group AND grouped by stone_id, so every frame
of a single stone lands in exactly one split (no leakage).

Manifest columns:
    frame_path, stone_id, shape_raw, shape_group, color_group,
    clarity_group, fluorescence_group, eye_clean, inclusion_weak_label,
    inclusion_types
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


INPUT_CSV   = Path("data/processed/stone_records.csv")
FRAMES_DIR  = Path("data/processed/frames")
OUT_DIR     = Path("data/processed")

MANIFEST_COLS = [
    "frame_path", "stone_id", "shape_raw", "shape_group",
    "color_group", "clarity_group", "fluorescence_group",
    "eye_clean", "inclusion_weak_label", "inclusion_types",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build train/val/test manifests")
    p.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    p.add_argument("--min-frames", type=int, default=1,
                   help="Only include stones whose frame folder holds at least N "
                        "frame_*.jpg files. Set this to --frames-per-video so the "
                        "manifest is built only from fully/successfully extracted "
                        "folders (default: 1)")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-shape-other", action="store_true",
                   help="Drop stones in the 'other' shape group from the manifests")
    return p.parse_args()


def stratified_grouped_split(stone_ids_by_group: dict[str, list[str]],
                             ratios: tuple[float, float, float],
                             rng: random.Random) -> dict[str, str]:
    """Return {stone_id: split_name} where split_name in {train, val, test}.

    Within each shape_group, shuffle the stones and assign by ratio. This keeps
    every frame of a stone in one split and balances rare shapes across splits.
    """
    train_r, val_r, test_r = ratios
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "ratios must sum to 1.0"

    assignment: dict[str, str] = {}
    for group, stones in stone_ids_by_group.items():
        ids = stones[:]
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_r))
        n_val   = int(round(n * val_r))
        # Test gets the remainder so rounding never overshoots.
        n_test  = n - n_train - n_val
        if n_test < 0:
            n_val += n_test
            n_test = 0
        for sid in ids[:n_train]:
            assignment[sid] = "train"
        for sid in ids[n_train:n_train + n_val]:
            assignment[sid] = "val"
        for sid in ids[n_train + n_val:]:
            assignment[sid] = "test"
    return assignment


def main() -> None:
    args = parse_args()

    if not INPUT_CSV.exists():
        sys.exit(f"ERROR: {INPUT_CSV} not found")
    if not args.frames_dir.exists() or not any(args.frames_dir.iterdir()):
        sys.exit(f"ERROR: no frame directories under {args.frames_dir}")

    df = pd.read_csv(INPUT_CSV, dtype={"stone_id": str})
    print(f"INFO: loaded {len(df)} stones from {INPUT_CSV}")

    # Build frame inventory from disk. Only folders with at least --min-frames
    # JPEGs count as "successfully extracted"; partial folders are skipped so
    # the manifest never references an incompletely-extracted stone.
    frames_by_stone: dict[str, list[Path]] = defaultdict(list)
    partial = 0
    for stone_dir in args.frames_dir.iterdir():
        if not stone_dir.is_dir():
            continue
        jpegs = sorted(stone_dir.glob("frame_*.jpg"))
        if len(jpegs) >= args.min_frames:
            frames_by_stone[stone_dir.name] = jpegs
        elif jpegs:
            partial += 1
    print(f"INFO: found frames for {len(frames_by_stone)} stones on disk "
          f"(>= {args.min_frames} frames each; skipped {partial} partial folders)")

    df = df[df["stone_id"].isin(frames_by_stone.keys())].copy()
    if args.exclude_shape_other:
        before = len(df)
        df = df[df["shape_group"] != "other"]
        print(f"INFO: --exclude-shape-other dropped {before - len(df)} stones")

    if df.empty:
        sys.exit("ERROR: no stones remain after joining CSV with frame inventory")

    # Group stone_ids by shape_group for stratification
    stones_by_group: dict[str, list[str]] = defaultdict(list)
    for sid, grp in zip(df["stone_id"], df["shape_group"]):
        stones_by_group[grp].append(sid)

    rng = random.Random(args.seed)
    assignment = stratified_grouped_split(stones_by_group, tuple(args.ratios), rng)

    # Build per-split rowsets
    split_rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    per_split_shape_counts = {s: Counter() for s in split_rows}

    df_indexed = df.set_index("stone_id")
    for sid, frame_paths in frames_by_stone.items():
        if sid not in assignment:
            continue
        split = assignment[sid]
        row = df_indexed.loc[sid]
        per_split_shape_counts[split][row["shape_group"]] += 1
        for fp in frame_paths:
            # Use forward-slash relative paths so manifests are portable.
            rel = fp.relative_to(Path.cwd()) if fp.is_absolute() else fp
            split_rows[split].append({
                "frame_path":           str(rel).replace("\\", "/"),
                "stone_id":             sid,
                "shape_raw":            row["shape_raw"],
                "shape_group":          row["shape_group"],
                "color_group":          row["color_group"],
                "clarity_group":        row["clarity_group"],
                "fluorescence_group":   row["fluorescence_group"],
                "eye_clean":            row["eye_clean"],
                "inclusion_weak_label": row["inclusion_weak_label"],
                "inclusion_types":      row["inclusion_types"],
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": args.seed,
        "ratios": list(args.ratios),
        "exclude_shape_other": args.exclude_shape_other,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        rows = split_rows[split]
        out = OUT_DIR / f"manifest_{split}.csv"
        pd.DataFrame(rows, columns=MANIFEST_COLS).to_csv(out, index=False, encoding="utf-8")
        summary["splits"][split] = {
            "frames": len(rows),
            "stones": len({r["stone_id"] for r in rows}),
            "shape_group_stone_counts": dict(per_split_shape_counts[split].most_common()),
        }
        print(f"WROTE {out} ({len(rows)} frames, "
              f"{summary['splits'][split]['stones']} stones)")

    (OUT_DIR / "manifest_split_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"WROTE {OUT_DIR / 'manifest_split_summary.json'}")


if __name__ == "__main__":
    main()
