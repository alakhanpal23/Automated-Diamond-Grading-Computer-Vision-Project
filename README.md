# aarin-stones-poc

Computer-vision proof-of-concept for diamonds/gemstones. Each stone has a 360°
`mp4` video and a GIA certificate PDF. The pipeline turns videos into labelled
still frames and trains image models to predict a stone's attributes (shape
first; clarity / eye-clean / colour / inclusions are the harder follow-ons).

Inventory: **10,537 stones** (10,498 with media) from `Aarin_Complied_Stones.xlsx`.

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1. Ingest | `src/01_ingest_excel.py` | `data/processed/stone_records.csv` (+ parquet, QA report) |
| 2. Download media | `src/02_download_media.py` | `data/raw/videos/{id}.mp4` (PDFs optional, **off by default**) |
| 3. Extract frames | `src/03_extract_frames.py` | `data/processed/frames/{id}/frame_NN.jpg` |
| 4. Build manifest | `src/04_build_manifest.py` | `manifest_{train,val,test}.csv` |
| 5. Train a head | `src/train_shape_model.py` | `data/models/<label>_resnet18/best.pt` |
| 6. Evaluate | `src/evaluate.py` | per-frame **and** per-stone accuracy |
| (aux) Cert inclusions | `src/05_extract_cert_inclusions.py` | `cert_inclusions.{csv,jsonl}` |

### Quick start (POC)

```bash
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# one-time: put the media token in config/.env  (see config/.env.example)
python src/01_ingest_excel.py
python src/02_download_media.py --kind videos --workers 8 --shuffle
python src/03_extract_frames.py --stone-list <ids.txt> --frames-per-video 12 --size 224 --quality 85 --workers 4
python src/04_build_manifest.py --min-frames 12 --strat-col shape_group
python src/train_shape_model.py --label-col shape_group --epochs 8 --workers 4
python src/evaluate.py --label-col shape_group --split test
```

The extractor is resumable (skips folders that already hold the expected frame
count) and writes `frame_extraction_log.csv`. `--stone-list` restricts work to a
chosen set of stone ids — use it to build **shape/label-balanced** subsets
(`head(N)` is round-heavy and not representative).

## Results so far (ResNet-18, ImageNet-pretrained, CPU)

| Head | Setup | Test accuracy |
|------|-------|---------------|
| Shape (5-class `shape_raw`) | balanced 600 stones | **99.4%** (per-frame) |
| Eye-clean (binary) | balanced 700, shape-controlled | **75.0%** (per-frame; recall 74/76) |

Shape is an easy task (distinct outlines, clean 360° views). Eye-clean is a
genuinely hard naked-eye judgement — 75% is well above the 50% shape-controlled
baseline but shows the grading-level frontier. Run `evaluate.py` for the
per-stone (majority-vote) numbers, which is the metric that matters in practice.

## Inclusions from GIA PDFs (`src/05_extract_cert_inclusions.py`)

The spreadsheet lists inclusion **types** (`inclusion_types`) but **not their
positions**. The GIA PDF's "Clarity Characteristics" plot does: red symbols mark
each inclusion's type, position and approximate size. The extractor pulls:

* **Types** from the authoritative "KEY TO SYMBOLS" text — validated 100% match
  to the spreadsheet on a 30-stone IF→I sample.
* **Positions / count** from red-mark detection on the native diagram image —
  counts track clarity grade (I1≈13, SI≈10–17, VS≈4, VVS≈2–3 marks).

Caveat (printed on every report): the plot is an *approximate schematic*, "all
clarity characteristics may not be shown". It is **not** registered to the video
frame, so true inclusion *localisation on the video* needs a schematic→frame
registration step or manual annotation — a separate project. Type/presence,
clarity grade and eye-clean are trainable directly from frames today.

## Storage note

Videos for the full set are ~23 GB; **all frames are only ~0.9 GB** (224px). On a
disk-constrained machine, prefer a stream-and-delete flow (download → extract →
delete the video) — frames are the only artefact training needs.

## Layout / gitignore

`config/.env` (media token), `data/raw/`, `data/processed/`, `data/models/`,
and `*.pt/*.mp4/*.jpg` are gitignored. Only code is tracked.
