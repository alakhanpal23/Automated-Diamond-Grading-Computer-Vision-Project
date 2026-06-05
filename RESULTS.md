# POC results — diamond grading from 360° video

All models: ResNet-18 (ImageNet-pretrained), trained on GPU from 512px frames,
12 frames/stone, stone-grouped leak-free splits.

## Two ways to read accuracy

**Balanced test** (equal classes) measures *discriminative ability*.
**Held-out @ natural distribution** (stones in NO head's training set, real
inventory class mix) measures *deployment accuracy* — this is the honest number.

Held-out numbers are prior-corrected (log of real class frequency added at
inference). The "rescued" column uses **random** balanced sampling instead of
first-N-alphabetical.

| Attribute | Held-out (alphabetical) | **Held-out (random)** | Majority baseline | Verdict |
|---|:---:|:---:|:---:|---|
| Shape (10-cls) | 99.5% | **99.2%** | ~21% | ✅ deployable |
| Eye-clean (2-cls) | 53.5% | **91.0%** | 72% | ✅ deployable |
| Clarity (VVS/VS/SI) | 6.8% | **68.9%** | 37% | ✅ usable (3-class) |
| Color (3-cls) | 67.3% | **63.6%** | ~40% | ✅ adds value |
| Inclusions (multi-label) | P=0.49 R=0.64 | P=0.48 R=0.62 | — | 🟡 useful screen (common types) |
| Fluorescence (4-cls) | 10.7% | 62.5%* | 58% | ❌ *=majority baseline; no real signal* |

## The key lesson
The original collapse was a **sampling bug, not an impossible task**. First-N
**alphabetical** balanced subsets drew training stones from a narrow slice of the
inventory (shared vendor/batch/lighting cues), so the subtle heads did not
transfer to other stones. **Random sampling fixed it**: clarity 6.8%→68.9%,
eye-clean 53.5%→91.0% on held-out, unseen inventory.

Two methodology points baked in:
1. **Random sampling** for balanced subsets (never first-N).
2. **Prior correction** at inference — add log(real class frequency) so the
   balanced-trained flat prior matches the natural inventory distribution.

The one genuine physical limit is **fluorescence**: even random sampling leaves
it at chance (raw 30.8%); the 62.5% is just prior correction predicting "none".
It needs a UV light source.

## What genuinely works from video
- **Shape** (99.5%) and **color** (67%) generalize to unseen inventory.
- **Inclusion presence** detects common types (Crystal/Needle/Feather) as a screen.

## Geometry / proportions (the strongest result)
Proportions are scale-free, so they ARE recoverable from video. Paired approach:
- **Silhouette (deterministic, `geometry_silhouette.py`)** — L/W ratio to ~0.001
  for smooth shapes (round/oval/pear); approximate for cornered (emerald/cushion).
- **ML regressor (`train_geometry.py`, 4500 stones, 384px)** — per-stone held-out MAE:

| Proportion | MAE | "guess the mean" baseline |
|---|:---:|:---:|
| depth % | **±0.51** | 3.84 |
| table % | **±0.77** | 4.05 |
| crown angle | ±0.20° | 13.4 |
| pavilion depth | ±0.18 | 16.6 |

Near-GIA precision. Example (round): depth 62.8 vs 62.7, table 57.9 vs 58.0,
crown 35.6 vs 36.0, pavilion 43.3 vs 43.0. **Carat is NOT predicted from video**
(no scale reference — pixel footprint vs carat correlation = -0.07). In the
machine it is **weighed**; weight + these proportions = exact mm dimensions.
`geometry_report.py` renders the annotated face-up+profile visual.

## What needs the grading machine, not the video
- **Clarity & inclusions** → microscope + darkfield (10–40×).
- **Fluorescence** → UV light source (white-light video lacks the signal).
- **Size / proportions** → calibrated silhouette/structured-light 3D (not learnable from video — no scale reference).
- All heads → **random, larger sampling** + calibration.
