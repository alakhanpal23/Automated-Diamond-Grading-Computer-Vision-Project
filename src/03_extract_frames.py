"""Extract evenly-spaced frames from the downloaded mp4 videos.

Run from the repo root after src/02_download_media.py has populated data/raw/videos/:

    python src/03_extract_frames.py [--limit-stones N] [--workers 4]
                                    [--frames-per-video 12] [--size 224|384]
                                    [--quality 85] [--overwrite false]

For each video, sample N frames at uniform indices, centre-crop to a square and
resize to --size, then write JPEGs to:

    data/processed/frames/{stone_id}/frame_{i:02d}.jpg

A run is resumable: if a stone's directory already contains the expected number
of JPEGs (>=1 byte each), it is skipped unless --overwrite is set.

A per-video line is appended to data/processed/frame_extraction_log.csv:
    stone_id,status,frames_written,src_frames,src_fps,src_w,src_h,err
"""
from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_CSV  = Path("data/processed/stone_records.csv")
VIDEO_DIR  = Path("data/raw/videos")
FRAME_DIR  = Path("data/processed/frames")
LOG_CSV    = Path("data/processed/frame_extraction_log.csv")

DEFAULT_FRAMES  = 24
DEFAULT_SIZE    = 512
DEFAULT_QUALITY = 90
DEFAULT_WORKERS = 4   # OpenCV's decoder is CPU-bound; ~cores/2 is a safe default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class ExtractResult:
    stone_id: str
    status: str          # "ok" | "skipped" | "error"
    frames_written: int
    src_frames: int
    src_fps: float
    src_w: int
    src_h: int
    err: str | None


def centre_square_crop(img):
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side]


def extract_one(stone_id: str, video_path: Path, *,
                n_frames: int, size: int, quality: int,
                overwrite: bool) -> ExtractResult:
    out_dir = FRAME_DIR / stone_id

    expected_jpegs = [out_dir / f"frame_{i:02d}.jpg" for i in range(n_frames)]
    if (not overwrite
            and out_dir.exists()
            and all(p.exists() and p.stat().st_size > 0 for p in expected_jpegs)):
        return ExtractResult(stone_id, "skipped", n_frames, 0, 0.0, 0, 0, None)

    if not video_path.exists():
        return ExtractResult(stone_id, "error", 0, 0, 0.0, 0, 0,
                             f"video not on disk: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return ExtractResult(stone_id, "error", 0, 0, 0.0, 0, 0, "cv2 could not open")

    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps    = float(cap.get(cv2.CAP_PROP_FPS))
    src_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if src_frames <= 0:
        cap.release()
        return ExtractResult(stone_id, "error", 0, src_frames, src_fps, src_w, src_h,
                             "src has 0 frames")

    # Evenly-spaced indices including endpoints but avoiding the last frame
    # (some encoders return -1 reads at exactly count-1).
    if src_frames >= n_frames:
        step = src_frames / n_frames
        indices = [int(step * i) for i in range(n_frames)]
    else:
        # Fewer frames than requested -- repeat the last one (rare for these videos)
        indices = list(range(src_frames)) + [src_frames - 1] * (n_frames - src_frames)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    last_err = None
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            last_err = f"failed read at idx {idx}"
            continue
        sq = centre_square_crop(frame)
        if sq.shape[0] != size:
            sq = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
        dest = out_dir / f"frame_{i:02d}.jpg"
        ok = cv2.imwrite(str(dest), sq, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            written += 1
        else:
            last_err = f"imwrite failed for {dest}"

    cap.release()
    status = "ok" if written == n_frames else "error"
    return ExtractResult(stone_id, status, written, src_frames, src_fps, src_w, src_h,
                         last_err if status == "error" else None)


def str2bool(v: str | bool) -> bool:
    """Parse a CLI boolean so ``--overwrite false`` / ``--overwrite true`` both work."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "t", "yes", "y", "1"):
        return True
    if s in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {v!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract evenly-spaced frames from videos")
    p.add_argument("--limit-stones", "--limit", dest="limit", type=int, default=None,
                   help="Process only the first N stones (default: all on disk)")
    p.add_argument("--stone-list", type=Path, default=None,
                   help="Path to a text file of stone_ids (one per line); restrict "
                        "extraction to these (applied before --limit-stones)")
    p.add_argument("--workers",  type=int, default=DEFAULT_WORKERS)
    p.add_argument("--frames-per-video", "--frames", dest="frames",
                   type=int, default=DEFAULT_FRAMES,
                   help=f"Frames to sample per video (default: {DEFAULT_FRAMES})")
    p.add_argument("--size",     type=int, default=DEFAULT_SIZE,
                   help=f"Output square side, e.g. 224 or 384 (default: {DEFAULT_SIZE})")
    p.add_argument("--quality",  type=int, default=DEFAULT_QUALITY,
                   help=f"JPEG quality 1-100 (default: {DEFAULT_QUALITY})")
    p.add_argument("--overwrite", type=str2bool, nargs="?", const=True, default=False,
                   help="Re-extract even if expected JPEGs already exist (default: false)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not INPUT_CSV.exists():
        sys.exit(f"ERROR: {INPUT_CSV} not found; run src/01_ingest_excel.py first")
    if not VIDEO_DIR.exists() or not any(VIDEO_DIR.iterdir()):
        sys.exit(f"ERROR: no videos in {VIDEO_DIR}; run src/02_download_media.py first")

    df = pd.read_csv(INPUT_CSV, dtype={"stone_id": str})
    df = df[df["has_media"] == True]  # noqa: E712
    if args.stone_list is not None:
        ids = [ln.strip() for ln in args.stone_list.read_text(encoding="utf-8").splitlines()
               if ln.strip()]
        df = df[df["stone_id"].isin(ids)]
        print(f"INFO: --stone-list {args.stone_list} -> {len(df)} of {len(ids)} stones matched")
    if args.limit is not None:
        df = df.head(args.limit)

    # Filter to stones whose video file is actually on disk
    on_disk = []
    for sid in df["stone_id"]:
        p = VIDEO_DIR / f"{sid}.mp4"
        if p.exists() and p.stat().st_size > 0:
            on_disk.append((sid, p))
    print(f"INFO: {len(on_disk)} videos on disk (of {len(df)} CSV stones with media)")
    if not on_disk:
        sys.exit("ERROR: no usable videos found on disk")

    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    log_lock = threading.Lock()
    log_fh   = LOG_CSV.open("w", newline="", encoding="utf-8")
    writer   = csv.writer(log_fh)
    writer.writerow(["stone_id", "status", "frames_written",
                     "src_frames", "src_fps", "src_w", "src_h", "err"])

    counts = {"ok": 0, "skipped": 0, "error": 0}
    t0 = time.time()
    progress_every = max(50, len(on_disk) // 50)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(extract_one, sid, p,
                            n_frames=args.frames, size=args.size,
                            quality=args.quality, overwrite=args.overwrite): sid
                for sid, p in on_disk}
        for i, fut in enumerate(as_completed(futs), start=1):
            res: ExtractResult = fut.result()
            counts[res.status] += 1
            with log_lock:
                writer.writerow([res.stone_id, res.status, res.frames_written,
                                 res.src_frames, f"{res.src_fps:.2f}",
                                 res.src_w, res.src_h, res.err or ""])
                log_fh.flush()
            if i % progress_every == 0 or i == len(on_disk):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  [{i:>6}/{len(on_disk)}]  ok={counts['ok']} "
                      f"skipped={counts['skipped']} err={counts['error']}  "
                      f"{rate:.1f} vids/s")

    log_fh.close()
    print(f"\nDONE in {time.time() - t0:.1f}s | ok={counts['ok']} "
          f"skipped={counts['skipped']} error={counts['error']}")
    print(f"per-video log: {LOG_CSV}")
    if counts["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
