# Automated Diamond Grading from Images — POC Overview

**Thesis:** A diamond's GIA grade can be read from images by computer vision. We
built the software core of an automated grading station and benchmarked exactly
what's recoverable from video vs. what needs better capture.

## What we built (this POC)
- End-to-end pipeline over **10,497 real GIA-certified stones** (360° videos + cert data).
- **10+ trained models** (shape, color, clarity, eye-clean, fluorescence, inclusions, geometry) + a **"digital cert"** that grades any stone from its video.
- GPU-trained, reproducible, version-controlled. Tools: `predict_stone.py` (video→grade),
  `digital_cert.py` (one-page report), `qc_report.py` (mislabel detector).

## Benchmarks — held-out, unseen stones (the honest metric)
Evaluated on stones in **no** model's training set, vs. the GIA cert.

| Attribute | Result | Note |
|---|:---:|---|
| **Cut geometry** (depth%, table%, angles) | **±0.5–0.8%, ±0.2°** | near-GIA precision |
| **L/W ratio** | **±0.01** | all shapes |
| **Physical dimensions** (L×W×D mm) | **±0.1 mm** | proportions + weighed carat |
| **Shape** | **99%** | 10 classes |
| **Color** (3 groups) | **87%** | D–J body color |
| **Eye-clean** | **92%** | naked-eye clarity |
| **Clarity** | **within-1-grade 100%** | exact-grade 58% (5-class) |
| **Inclusions** | detect + **localize** (Grad-CAM) | precision 0.51, recall 0.62 |

**Honest limits (physics, not model):** fluorescence needs **UV light**; carat needs a
**scale** — neither is recoverable from white-light video. These are *capture* gaps,
not algorithm gaps.

## Why it's credible
- **Leak-free held-out evaluation** (training/test stones never overlap — verified 0 overlap).
- Labels are **real GIA cert data**; inputs are **real video frames**.
- We **caught and fixed our own inflated numbers** (a sampling bug) — the reported
  figures are the post-correction, deployment-honest ones.

## What it proves / use cases
1. **Cut geometry & dimensions are reliably readable from images** — to near-cert accuracy.
2. **QC / verification works today:** flags mislabeled or swapped certified stones
   (85% catch rate, 0.2% false-positive on shape).
3. A clear **feasibility map** for a low-cost grading machine.

## Roadmap — the machine
The software stack plugs directly into a capture rig that breaks the physics ceilings:
**microscope + darkfield** (clarity/inclusions), **UV** (fluorescence), **calibrated
turntable + scale** (exact 3D geometry + carat). Same models, sharper inputs → full grade.
