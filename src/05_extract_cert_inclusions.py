"""Extract clarity / inclusion ground truth from GIA certificate PDFs.

Run from the repo root (PDFs are fetched on demand from pdf_link in the CSV):

    python src/05_extract_cert_inclusions.py [--limit N] [--stone-list FILE]
                                             [--workers 8] [--download true]
                                             [--keep-pdf false]

For each stone it pulls two complementary signals from the GIA report:

  1. Inclusion TYPES -- read from the authoritative "KEY TO SYMBOLS" text block
     (the symbols actually plotted on the clarity diagram). This is cleaner than
     parsing the spreadsheet's free-text Comment, and we cross-check the two.

  2. Inclusion POSITIONS / COUNT -- detected from the red marks on the native
     clarity-characteristics diagram image (red = internal characteristics per
     GIA's key; green/black = external blemishes). Positions are normalised to
     the diagram box so they are comparable across stones.

GIA caveats that bound what is recoverable (printed on every report): the plot is
an "approximate representation", "all clarity characteristics may not be shown",
and tiny features (e.g. additional pinpoints / graining) are often omitted. So
positions are approximate and counts are a lower bound -- documented, not hidden.

Outputs (data/processed/):
    cert_inclusions.csv   one row per stone (types, counts, agreement flag)
    cert_inclusions.jsonl one JSON object per stone (adds per-mark positions)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import requests

INPUT_CSV = Path("data/processed/stone_records.csv")
PDF_DIR   = Path("data/raw/pdfs")
OUT_CSV   = Path("data/processed/cert_inclusions.csv")
OUT_JSONL = Path("data/processed/cert_inclusions.jsonl")

TIMEOUT_S    = 30
MIN_OK_BYTES = 2048
# Inclusion (internal) symbols GIA plots in red; we only enumerate the canonical
# set so an unexpected token in the key is surfaced, not silently dropped.
KNOWN_SYMBOLS = {
    "Feather", "Crystal", "Needle", "Pinpoint", "Cloud", "Indented Natural",
    "Twinning Wisp", "Natural", "Cavity", "Knot", "Etch Channel", "Extra Facet",
    "Chip", "Bruise", "Laser Drill Hole", "Crystal Surface", "Surface Graining",
    "Internal Graining", "Pit", "Nick", "Scratch", "Abrasion", "Cleavage",
}


@dataclass
class CertResult:
    stone_id: str
    status: str                      # ok | skipped | error
    report_no: str | None = None
    shape: str | None = None
    color_grade: str | None = None
    clarity_grade: str | None = None
    comments: str | None = None
    key_symbols: list[str] = field(default_factory=list)
    unknown_symbols: list[str] = field(default_factory=list)
    red_mark_blobs: int = 0          # raw red connected components
    red_mark_clusters: int = 0       # blobs merged within a small radius
    positions: list[tuple[float, float]] = field(default_factory=list)  # normalised cluster centres
    csv_types: str | None = None     # spreadsheet inclusion_types for cross-check
    types_match: bool | None = None
    err: str | None = None


# ---------------------------------------------------------------------------
# PDF parsing helpers
# ---------------------------------------------------------------------------
def _field(txt: str, label: str) -> str | None:
    m = re.search(re.escape(label) + r"\s*\.*\s*(.+)", txt)
    return m.group(1).strip() if m else None


def parse_text(txt: str) -> dict:
    """Pull report fields, the KEY TO SYMBOLS list, and Comments from page text."""
    m = re.search(r"KEY TO SYMBOLS\*?\s*(.*?)\s*\*\s*Red symbols", txt, re.S)
    raw_syms = [ln.strip() for ln in (m.group(1).splitlines() if m else []) if ln.strip()]
    known, unknown = [], []
    for s in raw_syms:
        (known if s in KNOWN_SYMBOLS else unknown).append(s)

    cm = re.search(r"Comments?:\s*(.*?)(?:\n[A-Z]|\Z)", txt, re.S)
    comments = None
    if cm:
        comments = re.sub(r"\s+", " ", cm.group(1)).strip()
        # Strip a trailing bare report number that the layout appends.
        comments = re.sub(r"\s*\b\d{8,}\b\s*$", "", comments).strip() or None

    return {
        "report_no":     _field(txt, "GIA Report Number"),
        "shape":         _field(txt, "Shape and Cutting Style"),
        "color_grade":   _field(txt, "Color Grade"),
        "clarity_grade": _field(txt, "Clarity Grade"),
        "comments":      comments,
        "key_symbols":   known,
        "unknown_symbols": unknown,
    }


def find_diagram_image(pg) -> int | None:
    """xref of the clarity-characteristics diagram (widest image in the band
    between the CLARITY CHARACTERISTICS header and the KEY TO SYMBOLS label)."""
    cc = pg.search_for("CLARITY CHARACTERISTICS")
    ks = pg.search_for("KEY TO SYMBOLS")
    if not cc or not ks:
        return None
    y_top, y_bot = cc[0].y1, ks[0].y0
    best, best_w = None, 0
    for im in pg.get_image_info(xrefs=True):
        x0, y0, x1, y1 = im["bbox"]
        cy = (y0 + y1) / 2
        w = x1 - x0
        if y_top <= cy <= y_bot and w > 150 and w > best_w:
            best, best_w = im["xref"], w
    return best


def detect_marks(bgr: np.ndarray, color: str) -> list[tuple[int, float, float]]:
    """Return [(area, norm_x, norm_y)] for red (inclusion) or green (blemish) marks."""
    B, G, R = (bgr[:, :, i].astype(int) for i in range(3))
    if color == "red":      # red and pale-red/pink: R dominant over both G and B
        mask = (R > 140) & (R - G > 35) & (R - B > 35)
    else:                   # green: G dominant
        mask = (G > 110) & (G - R > 30) & (G - B > 20)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    H, W = bgr.shape[:2]
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= 2:
            out.append((a, cent[i, 0] / W, cent[i, 1] / H))
    return out


def merge_clusters(marks: list[tuple[int, float, float]], radius: float = 0.035):
    """Merge blobs whose centres lie within `radius` (normalised) into one cluster
    centre, so a single multi-stroke symbol (e.g. a feather) counts once."""
    centres: list[list[float]] = []
    for _, x, y in sorted(marks, key=lambda m: -m[0]):
        for c in centres:
            if abs(c[0] - x) <= radius and abs(c[1] - y) <= radius:
                break
        else:
            centres.append([round(x, 4), round(y, 4)])
    return centres


# ---------------------------------------------------------------------------
# Per-stone driver
# ---------------------------------------------------------------------------
def ensure_pdf(stone_id: str, pdf_link: str | None, *, download: bool,
               session: requests.Session) -> tuple[Path | None, str | None]:
    dest = PDF_DIR / f"{stone_id}.pdf"
    if dest.exists() and dest.stat().st_size >= MIN_OK_BYTES:
        return dest, None
    if not download:
        return None, "pdf not on disk and --download false"
    if not isinstance(pdf_link, str) or not pdf_link:
        return None, "no pdf_link in CSV"
    try:
        r = session.get(pdf_link, timeout=TIMEOUT_S)
        if r.status_code != 200 or len(r.content) < MIN_OK_BYTES:
            return None, f"http {r.status_code} ({len(r.content)} bytes)"
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".pdf.tmp")
        tmp.write_bytes(r.content)
        tmp.replace(dest)
        return dest, None
    except requests.RequestException as e:
        # The PDF link may carry a token; requests errors can echo the full URL.
        return None, type(e).__name__


def process(stone_id: str, pdf_link: str | None, csv_types: str | None, *,
            download: bool, keep_pdf: bool, session: requests.Session) -> CertResult:
    csv_types = csv_types if isinstance(csv_types, str) else ""   # NaN -> "" for clean stones
    pdf_link  = pdf_link if isinstance(pdf_link, str) else None
    res = CertResult(stone_id, "error", csv_types=csv_types)
    path, err = ensure_pdf(stone_id, pdf_link, download=download, session=session)
    if path is None:
        res.err = err
        return res
    try:
        doc = fitz.open(path)
        pg = doc[0]
        info = parse_text(pg.get_text())
        res.report_no     = info["report_no"]
        res.shape         = info["shape"]
        res.color_grade   = info["color_grade"]
        res.clarity_grade = info["clarity_grade"]
        res.comments      = info["comments"]
        res.key_symbols   = info["key_symbols"]
        res.unknown_symbols = info["unknown_symbols"]

        xref = find_diagram_image(pg)
        if xref:
            base = doc.extract_image(xref)
            bgr = cv2.imdecode(np.frombuffer(base["image"], np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                reds = detect_marks(bgr, "red")
                clusters = merge_clusters(reds)
                res.red_mark_blobs = len(reds)
                res.red_mark_clusters = len(clusters)
                res.positions = [(c[0], c[1]) for c in clusters]
        doc.close()

        # Cross-check plotted types vs spreadsheet inclusion_types
        csv_set = {t.strip() for t in csv_types.split("|") if t.strip()}
        res.types_match = (set(res.key_symbols) == csv_set) if (res.key_symbols or csv_set) else None
        res.status = "ok"
        if not keep_pdf and download:
            path.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001 -- one bad PDF must not kill the run
        res.err = f"{type(e).__name__}: {str(e)[:120]}"
    return res


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "t", "yes", "y", "1"):
        return True
    if s in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract inclusion ground truth from GIA cert PDFs")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--stone-list", type=Path, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--download", type=str2bool, nargs="?", const=True, default=True,
                   help="Fetch the PDF from pdf_link if not already on disk (default: true)")
    p.add_argument("--keep-pdf", type=str2bool, nargs="?", const=True, default=False,
                   help="Keep downloaded PDFs on disk (default: false -> delete after parse)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not INPUT_CSV.exists():
        sys.exit(f"ERROR: {INPUT_CSV} not found; run src/01_ingest_excel.py first")

    df = pd.read_csv(INPUT_CSV, dtype={"stone_id": str, "pdf_link": str, "inclusion_types": str})
    df = df[df["has_pdf"] == True]  # noqa: E712
    if args.stone_list is not None:
        ids = [ln.strip() for ln in args.stone_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
        df = df[df["stone_id"].isin(ids)]
    if args.limit is not None:
        df = df.head(args.limit)
    work = list(df[["stone_id", "pdf_link", "inclusion_types"]].itertuples(index=False, name=None))
    print(f"INFO: {len(work)} stones to process (download={args.download}, keep_pdf={args.keep_pdf})")
    if not work:
        sys.exit("ERROR: no stones with a PDF to process")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["stone_id", "status", "report_no", "shape", "color_grade", "clarity_grade",
            "key_symbols", "unknown_symbols", "csv_types", "types_match",
            "red_mark_blobs", "red_mark_clusters", "comments", "err"]
    counts = {"ok": 0, "skipped": 0, "error": 0}
    match_ok = match_total = 0
    t0 = time.time()

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fcsv, \
         OUT_JSONL.open("w", encoding="utf-8") as fjson:
        writer = csv.writer(fcsv)
        writer.writerow(cols)
        lock = threading.Lock()
        with requests.Session() as session, ThreadPoolExecutor(max_workers=args.workers) as pool:
            session.headers.update({"User-Agent": "aarin-poc/0.1"})
            futs = {pool.submit(process, sid, link, ctypes,
                                download=args.download, keep_pdf=args.keep_pdf,
                                session=session): sid
                    for sid, link, ctypes in work}
            for i, fut in enumerate(as_completed(futs), 1):
                r: CertResult = fut.result()
                counts[r.status] += 1
                if r.types_match is not None:
                    match_total += 1
                    match_ok += int(r.types_match)
                with lock:
                    writer.writerow([r.stone_id, r.status, r.report_no, r.shape,
                                     r.color_grade, r.clarity_grade,
                                     "|".join(r.key_symbols), "|".join(r.unknown_symbols),
                                     r.csv_types, r.types_match,
                                     r.red_mark_blobs, r.red_mark_clusters,
                                     r.comments or "", r.err or ""])
                    fcsv.flush()
                    fjson.write(json.dumps({
                        "stone_id": r.stone_id, "status": r.status,
                        "clarity_grade": r.clarity_grade, "key_symbols": r.key_symbols,
                        "red_mark_clusters": r.red_mark_clusters, "positions": r.positions,
                    }) + "\n")
                if i % 25 == 0 or i == len(work):
                    print(f"  [{i}/{len(work)}] ok={counts['ok']} err={counts['error']}")

    dt = time.time() - t0
    print(f"\nDONE in {dt:.1f}s | ok={counts['ok']} skipped={counts['skipped']} error={counts['error']}")
    if match_total:
        print(f"types_match vs spreadsheet: {match_ok}/{match_total} = {match_ok/match_total:.1%}")
    print(f"wrote {OUT_CSV} and {OUT_JSONL}")


if __name__ == "__main__":
    main()
