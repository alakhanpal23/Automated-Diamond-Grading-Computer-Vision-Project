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

**DEFINITIVE numbers — locked test set.** A fixed 1,500-stone test set (`LOCKED_test.txt`)
was set aside up front; every head was retrained on the pool excluding it (0 overlap,
verified). Evaluated on 1,200 of those locked stones — fully leak-free, large enough
to trust. Prior-corrected.

| Attribute | Locked-test accuracy | Majority baseline | Verdict |
|---|:---:|:---:|---|
| Shape (10-cls) | **99.3%** | ~21% | ✅ deployable |
| Color (3-cls) | **87.2%** | ~40% | ✅ strong (was 63.6% pre-improvement) |
| Eye-clean (2-cls) | **89.4%** | 72% | ✅ deployable |
| Clarity (VVS/VS/SI) | **70.3%** | 37% | ✅ usable; ordinal 5-class within-1 ~100% |
| Inclusions (multi-label) | P=0.55 R=0.84 (calibrated) | — | 🟡 useful screen |
| Fluorescence (4-cls) | 45.2% | 58% | ❌ **below baseline → no signal (needs UV)** |
| Geometry depth%/table%/L-W | ±0.54 / ±0.86 / ±0.01 | 3.8 / 4.3 / 0.27 | ✅ near-GIA |

**Integrity note:** a mid-improvement eval briefly showed clarity 83.7% / color 87.7%,
but those were **inflated by leakage** (held-out list wasn't rebuilt after new training
sets; 3,132 clarity / 1,059 color stones had leaked in). We caught it, locked a fixed
test set, retrained every head leak-free, and re-measured — the table above is that
clean result. The improvements held up; clarity's honest figure is ~70%, not 84%.

### Improvement round (used more real data + better methods, no fabrication)
- **Color 63.6% → 87.2%** (held-out): retrained on 2,500 stones vs 750 — the original small sample didn't generalize. More real data was a *major* lever here.
- **Inclusion precision 0.41 → 0.51**: per-type threshold calibration on val (`calibrate_inclusions.py`), no retraining.
- **Geometry + L/W**: added `ratio` target → MAE 0.01, fixes cornered shapes (emerald/cushion) the silhouette couldn't.

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

## Clarity, ordinal framing (better than flat classes)
`train_clarity_ordinal.py` regresses the clarity index (I<SI<VS<VVS<IF). Balanced
test: exact 57.6%, **within-one-grade 100%**, MAE 0.51 grades. Treating clarity as
an ordered scale (near-misses penalized less) is the honest model — and within-one
is the metric that matters (human graders disagree by a grade on borderline stones).

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
machine it is **weighed**; weight + these proportions = mm dimensions.
`geometry_report.py` renders the annotated face-up+profile visual.

**Real mm dimensions** (`mm_dimensions.py`): silhouette L/W + ML depth% + weighed
carat -> L/W/D in mm. Held-out (400 stones) vs cert: depth MAE 0.091 mm, width
0.138 mm, length 0.279 mm (median 0.16). Math check (cert proportions + weight)
is ~exact at 0.03-0.05 mm, so error is from predicted proportions, not the
physics. The fill factor C = volume/(L*W*D) is stable per shape (CV 1-4%).

**Deterministic 3D from video does NOT work** (`geometry_3d.py`): the free-tumble
videos never reach a clean edge-on profile (high aspect = oblique foreshortening),
so direct profile measurement gives depth% MAE ~22 (vs ML 0.5). Exact deterministic
3D needs a controlled turntable (known pose, true profile) -- a machine capability,
not recoverable from these videos.

## What needs the grading machine, not the video
- **Clarity & inclusions** → microscope + darkfield (10–40×) for higher precision.
- **Fluorescence** → UV light source (white-light video lacks the signal).
- **Absolute size / carat** → physical scale (proportions ARE recoverable from
  video at ±0.5%; only the absolute mm scale is missing — no reference in frame).
- **Exact (not ±0.5) proportions** → calibrated turntable for deterministic 3D.

## Capstone artifact
`digital_cert.py <stone_id>` → one report per stone (visual `.jpg` + `.json`):
full predicted grade + geometry + reconstructed mm dimensions + inclusion
Grad-CAM heatmap, all QC-compared to the GIA cert. Example: data/processed/cert/.
