"""Download mp4 videos + GIA cert PDFs for stones listed in stone_records.csv.

Run from the repo root, after src/01_ingest_excel.py has produced the CSV:

    python src/02_download_media.py [--limit N] [--kind videos|both|pdfs]
                                    [--workers N] [--shuffle] [--dry-run]

PDF download is OFF by default (--kind defaults to "videos"); videos alone are
enough for the first CV POC. Pass --kind both or --kind pdfs to fetch GIA PDFs.

Files are written under data/raw/{videos,pdfs}/{stone_id}.{mp4,pdf}. A run is
resumable -- already-on-disk files with non-zero size are skipped. Each task is
written atomically (.tmp + rename) so an interrupted run never leaves a
partially-downloaded file in place.

A per-task line is appended to data/processed/download_log.csv:
    stone_id,kind,status,bytes,attempts,err
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_CSV   = Path("data/processed/stone_records.csv")
VIDEO_DIR   = Path("data/raw/videos")
PDF_DIR     = Path("data/raw/pdfs")
LOG_CSV     = Path("data/processed/download_log.csv")
DOTENV_PATH = Path("config/.env")

MP4_URL_TPL = "https://assets.mydiamonds.info/mp4/{ref}?u={token}"

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
DEFAULT_WORKERS = 8
TIMEOUT_S       = 30
MAX_RETRIES     = 4
BACKOFF_BASE    = 1.5
CHUNK_SIZE      = 64 * 1024
RETRY_STATUSES  = {408, 425, 429, 500, 502, 503, 504}
MIN_OK_BYTES    = 1024  # below this, treat the file as a stub/error page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_dotenv_key(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            v = v.strip().strip('"').strip("'")
            return v or None
    return None


@dataclass
class Task:
    stone_id: str
    kind: str       # "video" or "pdf"
    url: str
    dest: Path


@dataclass
class Result:
    stone_id: str
    kind: str
    status: str     # "ok" | "skipped" | "error"
    bytes: int
    attempts: int
    err: str | None


def fetch(task: Task, session: requests.Session) -> Result:
    """Download a single task with retries and atomic rename."""
    if task.dest.exists() and task.dest.stat().st_size >= MIN_OK_BYTES:
        return Result(task.stone_id, task.kind, "skipped",
                      task.dest.stat().st_size, 0, None)

    tmp = task.dest.with_suffix(task.dest.suffix + ".tmp")
    task.dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with session.get(task.url, stream=True, timeout=TIMEOUT_S) as r:
                if r.status_code in RETRY_STATUSES:
                    last_err = f"http {r.status_code}"
                    time.sleep(BACKOFF_BASE ** attempt)
                    continue
                if r.status_code != 200:
                    return Result(task.stone_id, task.kind, "error", 0, attempt,
                                  f"http {r.status_code}")
                bytes_written = 0
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            bytes_written += len(chunk)
                if bytes_written < MIN_OK_BYTES:
                    tmp.unlink(missing_ok=True)
                    last_err = f"short body ({bytes_written} bytes)"
                    time.sleep(BACKOFF_BASE ** attempt)
                    continue
                tmp.replace(task.dest)
                return Result(task.stone_id, task.kind, "ok", bytes_written, attempt, None)
        except requests.RequestException as e:
            last_err = type(e).__name__ + ": " + str(e)[:120]
            time.sleep(BACKOFF_BASE ** attempt)

    # All retries exhausted
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    return Result(task.stone_id, task.kind, "error", 0, MAX_RETRIES, last_err)


def build_worklist(df: pd.DataFrame, kind: str, token: str) -> list[Task]:
    tasks: list[Task] = []
    if kind in ("videos", "both"):
        for r in df[df["has_media"] == True].itertuples(index=False):  # noqa: E712
            ref = r.media_ref
            if not isinstance(ref, str) or not ref:
                continue
            tasks.append(Task(
                stone_id=r.stone_id, kind="video",
                url=MP4_URL_TPL.format(ref=ref, token=token),
                dest=VIDEO_DIR / f"{r.stone_id}.mp4",
            ))
    if kind in ("pdfs", "both"):
        for r in df[df["has_pdf"] == True].itertuples(index=False):  # noqa: E712
            link = r.pdf_link
            if not isinstance(link, str) or not link:
                continue
            tasks.append(Task(
                stone_id=r.stone_id, kind="pdf",
                url=link,
                dest=PDF_DIR / f"{r.stone_id}.pdf",
            ))
    return tasks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download mp4 videos + GIA cert PDFs.")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit to the first N STONES (each stone = 1 video + 1 pdf when --kind=both)")
    p.add_argument("--kind", choices=("both", "videos", "pdfs"), default="videos",
                   help="What to download. PDFs are OFF by default; pass "
                        "--kind both or --kind pdfs to fetch GIA cert PDFs.")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle the worklist (useful to spread load across the inventory)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the first 5 planned URLs and totals, then exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    token = (os.environ.get("AARIN_MEDIA_TOKEN")
             or _read_dotenv_key(DOTENV_PATH, "AARIN_MEDIA_TOKEN"))
    if not token:
        sys.exit("ERROR: AARIN_MEDIA_TOKEN required (set env var or fill config/.env)")
    if not INPUT_CSV.exists():
        sys.exit(f"ERROR: {INPUT_CSV} not found; run src/01_ingest_excel.py first")

    df = pd.read_csv(INPUT_CSV, dtype={"stone_id": str, "media_ref": str, "pdf_link": str})
    print(f"INFO: loaded {len(df)} stones from {INPUT_CSV}")

    if args.limit is not None:
        df = df.head(args.limit)
        print(f"INFO: --limit {args.limit} -> {len(df)} stones")

    tasks = build_worklist(df, args.kind, token)
    print(f"INFO: built worklist of {len(tasks)} tasks (kind={args.kind})")

    if args.shuffle:
        random.shuffle(tasks)
        print("INFO: --shuffle applied")

    if args.dry_run:
        print("INFO: --dry-run; first 5 tasks:")
        for t in tasks[:5]:
            print(f"  {t.kind:<5}  {t.stone_id}  ->  {t.dest}")
            print(f"          {t.url[:120]}{'...' if len(t.url) > 120 else ''}")
        return

    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    log_lock = threading.Lock()
    log_fh = LOG_CSV.open("w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_fh)
    log_writer.writerow(["stone_id", "kind", "status", "bytes", "attempts", "err"])

    counts = {"ok": 0, "skipped": 0, "error": 0}
    total_bytes = 0
    t0 = time.time()
    progress_every = max(50, len(tasks) // 50)

    with requests.Session() as session:
        session.headers.update({"User-Agent": "aarin-poc/0.1"})
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch, t, session): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), start=1):
                res: Result = fut.result()
                counts[res.status] += 1
                total_bytes += res.bytes
                with log_lock:
                    log_writer.writerow([res.stone_id, res.kind, res.status,
                                         res.bytes, res.attempts, res.err or ""])
                    log_fh.flush()
                if i % progress_every == 0 or i == len(tasks):
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    mb = total_bytes / (1024 * 1024)
                    print(f"  [{i:>6}/{len(tasks)}]  ok={counts['ok']} "
                          f"skipped={counts['skipped']} err={counts['error']}  "
                          f"{mb:.0f} MB  {rate:.1f} req/s")

    log_fh.close()
    elapsed = time.time() - t0
    mb = total_bytes / (1024 * 1024)
    print(f"\nDONE in {elapsed:.1f}s | ok={counts['ok']} skipped={counts['skipped']} "
          f"error={counts['error']} | {mb:.1f} MB written")
    print(f"per-task log: {LOG_CSV}")

    if counts["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
