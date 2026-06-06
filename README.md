# aarin-stones-poc — diamond grading from video

Computer-vision system that reads a diamond's GIA-style grade from its 360° video,
over **10,497 real GIA-certified stones**. It predicts shape, color, clarity,
eye-clean, fluorescence and inclusions, reconstructs the cut geometry and mm
dimensions, and assembles a one-page "digital cert" with a QC comparison to the
real cert. See **`OVERVIEW.md`** (advisor one-pager), **`RESULTS.md`** (full
numbers), **`CAPTURE_RIG.md`** (the hardware path).

## Definitive results — locked test set (1,200 stones, leak-free)
| Attribute | Result |
|---|:---:|
| Shape | **99.3%** |
| Color | **87.2%** |
| Eye-clean | **89.4%** |
| Clarity (3-class; within-1 ~100%) | **70.3%** |
| Geometry (depth% / table% / L-W) | **±0.5 / ±0.9 / ±0.01** |
| mm dimensions (L×W×D) | **±0.1 mm** |
| Inclusions | detect + localize, **P0.55 R0.84** |
| Fluorescence | no signal — needs UV (physics) |

A fixed 1,500-stone test set is held out up front; every head is retrained on the
rest (0 overlap, verified). Carat is not predictable from video (no scale) and is
*weighed* in the machine. Honest limits are capture limits, not algorithm limits.

## Pipeline
| Step | Script |
|---|---|
| 1. Ingest spreadsheet → CSV | `01_ingest_excel.py` |
| 2. Download videos | `02_download_media.py` |
| 3. Extract frames (512px) | `03_extract_frames.py` |
| 4. Build train/val/test manifest | `04_build_manifest.py` |
| 5. Cert inclusion ground truth | `05_extract_cert_inclusions.py` |

## Models & tools
| Script | What it does |
|---|---|
| `train_shape_model.py` | any single-label head (shape/color/clarity/eye-clean/fluorescence); `--label-col`, `--backbone`, `--no-color-jitter` |
| `train_inclusion.py` | multi-label inclusion model (multi-view max-agg) |
| `calibrate_inclusions.py` | per-type decision thresholds (precision) |
| `train_clarity_ordinal.py` | clarity as an ordered scale (within-1 metric) |
| `train_geometry.py` | proportion regressor (depth%/table%/crown/pavilion/ratio) |
| `geometry_silhouette.py` | deterministic L/W + outline montage |
| `geometry_3d.py` | deterministic profile geometry (exact with a turntable; fails on free-tumble video) |
| `mm_dimensions.py` | proportions + weighed carat → real mm dimensions |
| `reconstruct_3d.py` | parametric 3D diamond rendered from predicted proportions |
| `localize_inclusions.py` | Grad-CAM "where are the inclusions" heatmaps |
| `predict_stone.py` | full grade for one stone (prior-corrected) + cert QC |
| `digital_cert.py` | one-page visual report (grade + geometry + inclusion heatmap + QC) |
| `evaluate_grade.py` | end-to-end held-out scorecard (prior correction, TTA, backbones) |
| `qc_report.py` | flag mislabeled / swapped certified stones (shape, 99%) |
| `evaluate.py` | per-frame + per-stone accuracy for a single head |

## Quick start
```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt          # numpy, opencv, pymupdf, matplotlib, ...
# put the media token in config/.env (see config/.env.example)
python src/01_ingest_excel.py
python src/02_download_media.py --kind videos --workers 8 --shuffle
python src/03_extract_frames.py --frames-per-video 12 --size 512 --quality 95 --workers 4
# grade one stone (needs trained models in data/models/):
python src/digital_cert.py <stone_id>
```

## Platform bridge (Phase 1 packetizer)
`src/packetizer.py` turns stones into **capture packets** for the Kara Data
Platform and uploads them to S3 under the prefix layout its ingest expects:
`s3://<bucket>/raw/parcels/{parcel_id}/{run_id}/{stone_id}/` — frame JPEGs +
one `metadata.json`. Two-command flow:
```bash
# 1. Dry run — write packets to local disk exactly as they'd land in S3, so you
#    can diff metadata.json against the platform's reference packet before AWS.
python src/packetizer.py --sample 3 --seed 42 --out-dir /tmp/packets --darkfield-tail 4

# 2a. Upload to a local MinIO (platform docker-compose) — test before real S3:
AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin \
python src/packetizer.py --sample 3 --seed 42 --darkfield-tail 4 \
    --bucket kara-captures --endpoint-url http://localhost:9000

# 2b. Upload to real S3 (drop --endpoint-url; use a profile or the default chain):
python src/packetizer.py --sample 3 --seed 42 --darkfield-tail 4 \
    --bucket <your-bucket> --profile <aws-profile>
```
The upload path (boto3) is verified end-to-end against an S3-compatible server
(moto/MinIO): frames upload in parallel, each confirmed by a 2xx, then
metadata.json. `--schema <metadata.schema.json>` validates each packet first.
**Commit-marker rule (hard requirement):** `metadata.json` is the platform's
commit marker — its ingest Lambda triggers on `metadata.json` and resolves every
`media[].s3_key` immediately. The packetizer therefore uploads **all frames
first, confirms each, and writes `metadata.json` last**; if any frame fails after
retries, no `metadata.json` is written for that stone (it's logged failed and the
run continues). Re-runs are idempotent (skip if `metadata.json` already exists,
unless `--force`). Self-test the metadata builder with `python src/packetizer.py
--check`; validate against the platform schema with `--schema <metadata.schema.json>`.
Full setup (local MinIO test + AWS account/bucket/IAM) is in **`PLATFORM_BRIDGE.md`**.

## Key lessons (the honest ones)
- **Random (not first-N) sampling** for balanced subsets — alphabetical sampling leaked vendor/batch cues and inflated numbers.
- **Prior correction** at inference — balanced training gives a flat prior; add log(real class frequency).
- **Lock a fixed test set up front** — we caught our own leakage and re-measured cleanly.
- **The ceiling is the data:** 620px video caps clarity/inclusions; fluorescence needs UV; carat needs a scale. More training won't break these — the **capture rig** will.

## Layout / gitignore
`config/.env` (token), `data/raw/`, `data/processed/`, `data/models/`, `*.pt/*.mp4/*.jpg`
are gitignored. Only code + docs are tracked. GPU: trains on CUDA automatically (`device=auto`).
