"""Packetize stones into Kara Data Platform "capture packets" and upload to S3.

This is Phase 1 of the karatraining -> Kara Data Platform bridge. Each stone
becomes a packet of frame JPEGs + a single metadata.json under the prefix layout
the platform's ingest Lambda expects:

    s3://<bucket>/raw/parcels/{parcel_id}/{run_id}/{stone_id}/
        brightfield_0.jpg, brightfield_1.jpg, ..., darkfield_8.jpg, ...
        metadata.json   <- the COMMIT MARKER, uploaded LAST

The platform's ingest triggers on ObjectCreated for `metadata.json` and resolves
every media[].s3_key immediately. So metadata.json MUST land only after every
frame upload is confirmed -- otherwise ingest races its own media. This script
structurally enforces that: build_metadata() takes the list of *already-confirmed*
frame media entries as input, and the upload path uploads all frames (with retry +
confirm) before it ever constructs or sends metadata.json.

Two-command flow:

    # 1. Dry run -- write packets to local disk exactly as they'd land in S3,
    #    so you can diff against the platform's reference packet before any AWS.
    python src/packetizer.py --sample 3 --seed 42 --out-dir /tmp/packets --darkfield-tail 4

    # 2. Real upload (test against MinIO first, then real S3 -- see README).
    python src/packetizer.py --sample 3 --seed 42 --bucket my-bucket [--endpoint-url ...]

Run `python src/packetizer.py --check` for a no-data self-test of the metadata
builder (required fields present, empty optionals omitted, correct platform keys).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_CSV  = Path("data/processed/stone_records.csv")
FRAMES_DIR = Path("data/processed/frames")
LOG_CSV    = Path("data/processed/packetizer_log.csv")

# Defaults for packet identity (honest: this is the video bridge, not a rig).
DEFAULT_MACHINE_ID = "KARA-BRIDGE-01"
DEFAULT_PARCEL_ID  = "PARCEL_AARIN01"
DEFAULT_RUN_ID     = "RUN001"
DEFAULT_DARKFIELD_TAIL = 4
DEFAULT_WORKERS = 8
UPLOAD_RETRIES = 3

LOG_COLS = ["stone_id", "n_frames", "n_darkfield", "bytes_uploaded",
            "dest", "status", "error", "timestamp"]

# Platform measurement keys (LEFT) <- this repo's CSV column (RIGHT).
# VERIFIED against the platform's ingest code: keys not in this exact set are
# SILENTLY DROPPED at ingest (no error). Note `star_length` and `lower_half`
# have NO `_pct` suffix on the platform side even though our columns do.
MEASUREMENT_MAP = {
    "mes1_length_mm": "mes_length",
    "mes2_width_mm":  "mes_width",
    "mes3_depth_mm":  "mes_depth",
    "depth_pct":      "depth_pct",
    "table_pct":      "table_pct",
    "crown_angle":    "crown_angle",
    "pavilion_depth": "pavilion_depth",
    "ratio":          "ratio",
    "star_length":    "star_length_pct",   # platform key has NO _pct suffix
    "lower_half":     "lower_half_pct",     # platform key has NO _pct suffix
}

# Frames are evenly spaced over 360 degrees; idx is the 0-based sorted position.
FRAME_GLOB = "frame_*.jpg"
FRAME_NUM_RE = re.compile(r"frame_(\d+)\.jpg$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Value cleaning helpers (never send nulls / NaN; coerce numpy -> native)
# ---------------------------------------------------------------------------
def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def clean_number(v):
    """Return a native float for a present numeric value, else None.

    0.0 is a real value, not missing -- only None/NaN are dropped (per contract)."""
    if _is_missing(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def clean_str(v):
    """Return a stripped string for a present value, else None."""
    if _is_missing(v):
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    return s


def clean_id(v):
    """Stringify an identifier without a trailing `.0` from float parsing."""
    if _is_missing(v):
        return None
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return clean_str(v)


# ---------------------------------------------------------------------------
# Frame discovery
# ---------------------------------------------------------------------------
def discover_frames(stone_id: str, frames_dir: Path) -> list[Path]:
    """Return this stone's frame JPEGs sorted NUMERICALLY (frame_2 before frame_10).

    On-disk filenames are zero-padded (frame_00.jpg ... frame_11.jpg) so a plain
    sort already works, but we sort by the parsed integer to be padding-agnostic."""
    stone_path = frames_dir / stone_id
    if not stone_path.is_dir():
        return []
    frames = list(stone_path.glob(FRAME_GLOB))

    def frame_num(p: Path) -> int:
        m = FRAME_NUM_RE.search(p.name)
        return int(m.group(1)) if m else 0

    return sorted(frames, key=frame_num)


def build_frame_plan(stone_id: str, frames: list[Path], darkfield_tail: int,
                     parcel_id: str, run_id: str) -> list[dict]:
    """Build the per-frame upload plan + media entries (no metadata yet).

    Each entry: {local_path, s3_key, media (the metadata media dict)}.

    `lighting` is "brightfield" for all frames except the LAST `darkfield_tail`,
    which we tag "darkfield". DEMO AFFORDANCE ONLY: our frames are white-light
    video stills, but the platform's inclusion-box labeling UI activates only on
    frames whose lighting is exactly "darkfield"/"polarized", so we relabel a tail
    subset to make that UI exercisable in the demo.
    """
    n = len(frames)
    df_start = n - darkfield_tail if darkfield_tail > 0 else n
    plan = []
    for idx, _path in enumerate(frames):
        lighting = "darkfield" if idx >= df_start else "brightfield"
        angle = int(round(idx * 360 / n)) if n else 0
        tag = "df" if lighting == "darkfield" else "bf"
        # Object key uses the lighting label + un-padded idx, NOT the source
        # frame filename -- this is the name the packet lands under in S3.
        s3_key = f"raw/parcels/{parcel_id}/{run_id}/{stone_id}/{lighting}_{idx}.jpg"
        plan.append({
            "local_path": _path,
            "s3_key": s3_key,
            "media": {
                "media_id": f"{stone_id}-{tag}-{idx}",  # unique within & across packets
                "lighting": lighting,
                "angle": angle,
                "s3_key": s3_key,
                "media_kind": "image",
            },
        })
    return plan


# ---------------------------------------------------------------------------
# Metadata assembly
# ---------------------------------------------------------------------------
def build_measurements(row) -> dict:
    """measurements dict using verified platform keys; omit any missing values."""
    out = {}
    for plat_key, csv_col in MEASUREMENT_MAP.items():
        val = clean_number(row.get(csv_col)) if csv_col in row else None
        if val is not None:
            out[plat_key] = val
    return out


def build_business(row) -> dict:
    """business dict; omit missing values. cert fields are stored but export as
    "N/A" under the default kara_native source -- correct for blind grading."""
    out = {}
    weight = clean_number(row.get("weight_ct"))
    if weight is not None:
        out["size_label"] = f"{weight:.2f} CTS"
    rap_rate = clean_number(row.get("rap_rate"))
    if rap_rate is not None:
        out["rap_rate"] = rap_rate
    cert_lab = clean_str(row.get("cert_lab"))
    if cert_lab is not None:
        out["cert_lab"] = cert_lab
    cert_no = clean_id(row.get("certificate_no"))
    if cert_no is not None:
        out["certificate_no"] = cert_no
    # Intentionally NOT sent: rap_total, days_in_stock (ingest never reads them;
    # both are derived at export time). Also NOT sent: grade labels (shape/color/
    # clarity/...) -- those are what the platform's graders produce; sending them
    # would contaminate the labeling demo.
    return out


def build_metadata(stone_id: str, row, media_entries: list[dict],
                   *, parcel_id: str, run_id: str, machine_id: str,
                   captured_at: str) -> dict:
    """Construct the metadata.json dict from CONFIRMED frame media entries.

    Taking `media_entries` as input is deliberate: in the upload path these are
    the frames already uploaded-and-confirmed, so metadata (the commit marker)
    can only ever describe media that exists. Do not call this before frames are
    confirmed on the destination.
    """
    meta = {
        "stone_id": stone_id,
        "parcel_id": parcel_id,
        "run_id": run_id,
        "machine_id": machine_id,
        "capture_status": "complete",
        "captured_at": captured_at,
        "media_type": "image",
        "media": media_entries,
    }
    weight = clean_number(row.get("weight_ct"))
    if weight is not None:
        meta["weight_carats"] = weight
    measurements = build_measurements(row)
    if measurements:
        meta["measurements"] = measurements
    business = build_business(row)
    if business:
        meta["business"] = business
    return meta


def now_iso_z() -> str:
    """Current UTC time, ISO8601 with trailing Z, second precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Schema validation (optional, degrades gracefully)
# ---------------------------------------------------------------------------
def load_validator(schema_path: Path | None):
    """Return a function(meta)->None that raises on invalid, or None if no schema."""
    if schema_path is None:
        return None
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        sys.exit("ERROR: --schema given but `jsonschema` is not installed. "
                 "Install it (pip install jsonschema) or omit --schema.")
    if not schema_path.exists():
        sys.exit(f"ERROR: --schema file not found: {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    # Auto-detect the schema's draft from its $schema so we honor whatever the
    # platform repo actually ships (Draft 7, 2020-12, ...).
    from jsonschema.validators import validator_for
    validator_cls = validator_for(schema)
    validator_cls.check_schema(schema)
    validator = validator_cls(schema)

    def _validate(meta: dict) -> None:
        errors = sorted(validator.iter_errors(meta), key=lambda e: e.path)
        if errors:
            msgs = "; ".join(f"{list(e.path)}: {e.message}" for e in errors[:5])
            raise ValueError(f"schema validation failed: {msgs}")

    return _validate


# ---------------------------------------------------------------------------
# Output: dry-run (local disk) and S3 upload
# ---------------------------------------------------------------------------
def dest_metadata_path_local(out_dir: Path, parcel_id: str, run_id: str,
                             stone_id: str) -> Path:
    return out_dir / "raw" / "parcels" / parcel_id / run_id / stone_id / "metadata.json"


def write_packet_local(out_dir: Path, plan: list[dict], meta: dict) -> int:
    """Write a packet to local disk exactly as it would land in S3. Returns bytes.

    Frames first, metadata.json last -- mirrors the commit-marker ordering even on
    disk so the dry-run tree is a faithful preview of the S3 object set."""
    import shutil
    stone_dir = (out_dir / "raw" / "parcels" / meta["parcel_id"]
                 / meta["run_id"] / meta["stone_id"])
    stone_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for item in plan:
        dst = out_dir / item["s3_key"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["local_path"], dst)
        total += dst.stat().st_size
    # metadata.json LAST.
    meta_bytes = json.dumps(meta, indent=2).encode("utf-8")
    (stone_dir / "metadata.json").write_bytes(meta_bytes)
    total += len(meta_bytes)
    return total


def _put_frame(s3, bucket: str, item: dict) -> int:
    """Upload one frame with retries; return bytes on success, raise on failure."""
    data = item["local_path"].read_bytes()
    last_err = None
    for attempt in range(1, UPLOAD_RETRIES + 1):
        try:
            resp = s3.put_object(Bucket=bucket, Key=item["s3_key"], Body=data,
                                 ContentType="image/jpeg")
            status = resp.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status == 200:
                return len(data)
            last_err = f"HTTP {status}"
        except Exception as exc:  # noqa: BLE001 -- log and retry any S3 error
            last_err = str(exc)
        time.sleep(0.5 * attempt)
    raise RuntimeError(f"frame {item['s3_key']} failed after {UPLOAD_RETRIES} "
                       f"attempts: {last_err}")


def upload_packet_s3(s3, bucket: str, stone_id: str, row, plan: list[dict],
                     *, parcel_id: str, run_id: str, machine_id: str,
                     captured_at: str, validate, workers: int) -> int:
    """Upload all frames (confirmed) THEN metadata.json. Returns bytes uploaded.

    If any frame fails after retries, raises WITHOUT uploading metadata.json, so
    the platform never sees a commit marker for an incomplete packet."""
    total = 0
    # Frames first, in parallel, each confirmed by a 200 put response.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_put_frame, s3, bucket, item): item for item in plan}
        for fut in as_completed(futures):
            total += fut.result()  # raises -> caller logs stone as failed

    # Every frame confirmed -> build metadata from the confirmed media entries.
    media_entries = [item["media"] for item in plan]
    meta = build_metadata(stone_id, row, media_entries, parcel_id=parcel_id,
                          run_id=run_id, machine_id=machine_id, captured_at=captured_at)
    if validate is not None:
        validate(meta)  # fail loudly before sending the commit marker

    meta_key = f"raw/parcels/{parcel_id}/{run_id}/{stone_id}/metadata.json"
    meta_bytes = json.dumps(meta, indent=2).encode("utf-8")
    resp = s3.put_object(Bucket=bucket, Key=meta_key, Body=meta_bytes,
                         ContentType="application/json")
    if resp.get("ResponseMetadata", {}).get("HTTPStatusCode") != 200:
        raise RuntimeError(f"metadata.json upload failed for {stone_id}")
    total += len(meta_bytes)
    return total


def metadata_exists_s3(s3, bucket: str, parcel_id: str, run_id: str,
                       stone_id: str) -> bool:
    from botocore.exceptions import ClientError
    key = f"raw/parcels/{parcel_id}/{run_id}/{stone_id}/metadata.json"
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


# ---------------------------------------------------------------------------
# Stone selection
# ---------------------------------------------------------------------------
def eligible_stones(df: pd.DataFrame, frames_dir: Path) -> list[str]:
    """Stones that BOTH appear in the CSV AND have >=1 frame jpg on disk."""
    csv_ids = set(df["stone_id"].astype(str))
    out = []
    for sid in csv_ids:
        if discover_frames(sid, frames_dir):
            out.append(sid)
    return sorted(out)


def select_stones(args, df: pd.DataFrame, frames_dir: Path) -> list[str]:
    if args.stone_list is not None:
        ids = []
        for ln in args.stone_list.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#", 1)[0].strip()
            if ln:
                ids.append(ln)
        # Preserve list order but drop ids with no frames / not in CSV (warn).
        eligible = set(eligible_stones(df, frames_dir))
        kept, dropped = [], []
        seen = set()
        for sid in ids:
            if sid in seen:
                continue
            seen.add(sid)
            (kept if sid in eligible else dropped).append(sid)
        if dropped:
            print(f"WARN: {len(dropped)} stone(s) from --stone-list have no frames "
                  f"or are not in the CSV and were skipped: {dropped[:5]}"
                  f"{' ...' if len(dropped) > 5 else ''}")
        return kept

    # --sample N: RANDOM sample (never first-N/alphabetical) from eligible stones.
    pool = eligible_stones(df, frames_dir)
    rng = random.Random(args.seed)
    if args.stratify_col:
        if args.stratify_col not in df.columns:
            sys.exit(f"ERROR: --stratify-col {args.stratify_col!r} not in CSV columns")
        by_group: dict[str, list[str]] = {}
        col = df.set_index("stone_id")[args.stratify_col]
        for sid in pool:
            grp = col.get(sid)
            grp = "NA" if _is_missing(grp) else str(grp)
            by_group.setdefault(grp, []).append(sid)
        # Proportional allocation across groups, remainder by largest group.
        chosen: list[str] = []
        groups = sorted(by_group)
        for grp in groups:
            ids = by_group[grp][:]
            rng.shuffle(ids)
            take = int(math.floor(args.sample * len(ids) / len(pool)))
            chosen.extend(ids[:take])
        # Top up to exactly N from the leftover, shuffled.
        if len(chosen) < args.sample:
            leftover = [s for s in pool if s not in set(chosen)]
            rng.shuffle(leftover)
            chosen.extend(leftover[: args.sample - len(chosen)])
        rng.shuffle(chosen)
        return chosen[: args.sample]

    sample = pool[:]
    rng.shuffle(sample)
    if args.sample > len(sample):
        print(f"WARN: requested --sample {args.sample} but only {len(sample)} "
              f"eligible stones exist; using all of them.")
    return sample[: args.sample]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def append_log(rows: list[dict]) -> None:
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOG_CSV.exists()
    with LOG_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLS)
        if new_file:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# Self-check (--check)
# ---------------------------------------------------------------------------
def run_self_check() -> None:
    """Assert the metadata builder behaves on a synthetic record."""
    print("Running packetizer self-check...")
    # Synthetic record: some values present, some missing (None/NaN).
    row = {
        "weight_ct": 1.01,
        "mes_length": 6.0, "mes_width": 6.01, "mes_depth": 3.7,
        "depth_pct": 61.5, "table_pct": 57.0,
        "crown_angle": 34.5, "pavilion_depth": 43.0, "ratio": 1.0,
        "star_length_pct": 50.0, "lower_half_pct": 75.0,
        "rap_rate": 4200.0, "cert_lab": "GIA", "certificate_no": 1234567890.0,
        # Missing -> must be omitted:
        "rap_total": float("nan"), "days_in_stock": None,
    }
    plan = build_frame_plan("ST0101", [Path(f"frame_{i:02d}.jpg") for i in range(12)],
                            darkfield_tail=4, parcel_id="PARCEL_KT01", run_id="RUN001")
    media = [item["media"] for item in plan]
    meta = build_metadata("ST0101", row, media, parcel_id="PARCEL_KT01",
                          run_id="RUN001", machine_id="KARA-M1",
                          captured_at="2026-06-06T10:00:00Z")

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # Required top-level fields.
    for key in ("stone_id", "parcel_id", "run_id", "machine_id",
                "capture_status", "captured_at", "media_type", "media"):
        check(key in meta, f"missing required field {key!r}")
    check(meta["capture_status"] == "complete", "capture_status must be 'complete'")
    check(meta["media_type"] == "image", "media_type must be 'image'")
    check(meta["captured_at"].endswith("Z"), "captured_at must end with Z")

    # Media entries.
    check(len(meta["media"]) == 12, "expected 12 media entries")
    df_count = sum(1 for m in meta["media"] if m["lighting"] == "darkfield")
    check(df_count == 4, f"expected 4 darkfield entries, got {df_count}")
    check([m["angle"] for m in meta["media"]][:4] == [0, 30, 60, 90],
          "angles not evenly spaced over 360")
    check(meta["media"][0]["media_id"] == "ST0101-bf-0", "bf media_id format wrong")
    check(meta["media"][-1]["media_id"] == "ST0101-df-11", "df media_id format wrong")
    check(len({m["media_id"] for m in meta["media"]}) == 12, "media_id not unique")
    for m in meta["media"]:
        check(set(m) == {"media_id", "lighting", "angle", "s3_key", "media_kind"},
              f"media entry has wrong keys: {set(m)}")

    # Correct platform measurement keys (the silent-drop trap).
    check("star_length" in meta["measurements"], "missing platform key star_length")
    check("lower_half" in meta["measurements"], "missing platform key lower_half")
    check("star_length_pct" not in meta["measurements"], "leaked _pct key star_length_pct")
    check("lower_half_pct" not in meta["measurements"], "leaked _pct key lower_half_pct")
    check(meta["measurements"]["mes1_length_mm"] == 6.0, "mes1_length_mm wrong")

    # Optionals: present ones in, missing ones omitted.
    check(meta["weight_carats"] == 1.01, "weight_carats missing")
    check(meta["business"]["size_label"] == "1.01 CTS", "size_label format wrong")
    check(meta["business"]["certificate_no"] == "1234567890", "cert_no not int-stringified")
    check("rap_total" not in meta.get("business", {}), "rap_total must not be sent")
    check("days_in_stock" not in meta.get("business", {}), "days_in_stock must not be sent")

    # No nulls anywhere; no _expected_label.
    def has_null(o):
        if isinstance(o, dict):
            return any(v is None or has_null(v) for v in o.values())
        if isinstance(o, list):
            return any(has_null(v) for v in o)
        return False
    check(not has_null(meta), "metadata contains a null value")
    check("_expected_label" not in json.dumps(meta), "_expected_label must not be present")

    if failures:
        print("SELF-CHECK FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("SELF-CHECK PASSED (all assertions ok).")
    print(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Packetize stones into Kara Data Platform capture packets.")
    # Self-check.
    p.add_argument("--check", action="store_true",
                   help="Run the metadata-builder self-check and exit (no data needed).")
    # Stone selection (mutually exclusive).
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--stone-list", type=Path,
                     help="Text file of stone ids, one per line (# comments ok).")
    sel.add_argument("--sample", type=int,
                     help="Random sample of N eligible stones (use with --seed).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for --sample.")
    p.add_argument("--stratify-col", default=None,
                   help="Optional CSV column to stratify the --sample by (e.g. shape_group).")
    # Packet identity.
    p.add_argument("--parcel-id", default=DEFAULT_PARCEL_ID)
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument("--machine-id", default=DEFAULT_MACHINE_ID)
    p.add_argument("--darkfield-tail", type=int, default=DEFAULT_DARKFIELD_TAIL,
                   help="Tag the LAST N frames as 'darkfield' so the platform's "
                        "inclusion-box UI is exercisable (demo affordance).")
    # Output modes (mutually exclusive).
    out = p.add_mutually_exclusive_group()
    out.add_argument("--out-dir", type=Path,
                     help="Dry run: write packets to this local dir as they'd land in S3.")
    out.add_argument("--bucket", help="Real upload: target S3 bucket name.")
    p.add_argument("--endpoint-url", default=None,
                   help="S3 endpoint URL (e.g. http://localhost:9000 for MinIO).")
    p.add_argument("--profile", default=None, help="boto3 session profile name.")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help="Parallel frame-upload workers (default 8).")
    p.add_argument("--force", action="store_true",
                   help="Re-send even if the destination metadata.json already exists.")
    p.add_argument("--schema", type=Path, default=None,
                   help="Path to platform metadata.schema.json; validate every "
                        "metadata.json with jsonschema before writing/uploading.")
    p.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    p.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.check:
        run_self_check()
        return

    if args.stone_list is None and args.sample is None:
        sys.exit("ERROR: choose stones with either --stone-list <file> or "
                 "--sample N [--seed S]. See --help.")
    if args.out_dir is None and args.bucket is None:
        sys.exit("ERROR: choose an output mode: --out-dir <path> (dry run) or "
                 "--bucket <name> (upload). See --help.")

    if not args.input_csv.exists():
        sys.exit(f"ERROR: {args.input_csv} not found")
    if not args.frames_dir.exists():
        sys.exit(f"ERROR: frames dir {args.frames_dir} not found")

    df = pd.read_csv(args.input_csv, dtype={"stone_id": str})
    print(f"INFO: loaded {len(df)} stones from {args.input_csv}")

    stones = select_stones(args, df, args.frames_dir)
    if not stones:
        sys.exit("ERROR: no eligible stones selected.")
    print(f"INFO: selected {len(stones)} stone(s): {stones}")

    validate = load_validator(args.schema)

    # S3 client (only for upload mode).
    s3 = None
    if args.bucket is not None:
        try:
            import boto3
        except ImportError:
            sys.exit("ERROR: --bucket requires boto3 (pip install boto3).")
        session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
        s3 = session.client("s3", endpoint_url=args.endpoint_url)
        print(f"INFO: uploading to s3://{args.bucket} "
              f"(endpoint={args.endpoint_url or 'AWS default'})")

    df_indexed = df.set_index("stone_id")
    log_rows = []
    n_ok = n_skipped = n_failed = 0

    for sid in stones:
        captured_at = now_iso_z()
        frames = discover_frames(sid, args.frames_dir)
        n_frames = len(frames)
        plan = build_frame_plan(sid, frames, args.darkfield_tail,
                                args.parcel_id, args.run_id)
        n_darkfield = sum(1 for it in plan if it["media"]["lighting"] == "darkfield")
        row = df_indexed.loc[sid].to_dict()
        status = error = ""
        bytes_uploaded = 0
        dest = ""

        try:
            if args.out_dir is not None:
                dest = str(dest_metadata_path_local(
                    args.out_dir, args.parcel_id, args.run_id, sid).parent)
                meta_path = dest_metadata_path_local(
                    args.out_dir, args.parcel_id, args.run_id, sid)
                if meta_path.exists() and not args.force:
                    status, n_skipped = "skipped", n_skipped + 1
                    print(f"SKIP {sid}: metadata.json already at {dest} (use --force)")
                else:
                    # Build metadata from the (local, known) media entries.
                    media = [it["media"] for it in plan]
                    meta = build_metadata(sid, row, media, parcel_id=args.parcel_id,
                                          run_id=args.run_id, machine_id=args.machine_id,
                                          captured_at=captured_at)
                    if validate is not None:
                        validate(meta)
                    bytes_uploaded = write_packet_local(args.out_dir, plan, meta)
                    status, n_ok = "ok", n_ok + 1
                    print(f"OK   {sid}: {n_frames} frames ({n_darkfield} darkfield) "
                          f"-> {dest} ({bytes_uploaded} bytes)")
            else:
                dest = f"s3://{args.bucket}/raw/parcels/{args.parcel_id}/{args.run_id}/{sid}"
                if (not args.force
                        and metadata_exists_s3(s3, args.bucket, args.parcel_id,
                                               args.run_id, sid)):
                    status, n_skipped = "skipped", n_skipped + 1
                    print(f"SKIP {sid}: metadata.json already at {dest} (use --force)")
                else:
                    bytes_uploaded = upload_packet_s3(
                        s3, args.bucket, sid, row, plan,
                        parcel_id=args.parcel_id, run_id=args.run_id,
                        machine_id=args.machine_id, captured_at=captured_at,
                        validate=validate, workers=args.workers)
                    status, n_ok = "ok", n_ok + 1
                    print(f"OK   {sid}: {n_frames} frames ({n_darkfield} darkfield) "
                          f"-> {dest} ({bytes_uploaded} bytes)")
        except Exception as exc:  # noqa: BLE001 -- log and continue with next stone
            status, error, n_failed = "failed", str(exc), n_failed + 1
            print(f"FAIL {sid}: {exc}")

        log_rows.append({
            "stone_id": sid, "n_frames": n_frames, "n_darkfield": n_darkfield,
            "bytes_uploaded": bytes_uploaded, "dest": dest, "status": status,
            "error": error, "timestamp": captured_at,
        })

    append_log(log_rows)

    # Summary table.
    print("\n" + "=" * 52)
    print(f"{'SUMMARY':<20}{'count':>10}")
    print("-" * 52)
    print(f"{'ok':<20}{n_ok:>10}")
    print(f"{'skipped':<20}{n_skipped:>10}")
    print(f"{'failed':<20}{n_failed:>10}")
    print(f"{'total':<20}{len(stones):>10}")
    print("=" * 52)
    print(f"log appended -> {LOG_CSV}")
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
