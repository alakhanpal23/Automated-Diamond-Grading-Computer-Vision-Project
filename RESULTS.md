# POC results — diamond grading from 360° video

All models: ResNet-18 (ImageNet-pretrained), trained on GPU from 512px frames,
12 frames/stone, stone-grouped leak-free splits.

## Two ways to read accuracy

**Balanced test** (equal classes) measures *discriminative ability*.
**Held-out @ natural distribution** (stones in NO head's training set, real
inventory class mix) measures *deployment accuracy* — this is the honest number.

| Attribute | Balanced test | Held-out (natural) | Majority baseline | Verdict |
|---|:---:|:---:|:---:|---|
| Shape (10-cls) | 100% | **99.5%** | ~21% | ✅ deployable |
| Color (3-cls) | 76% | **67.3%** | ~40% | ✅ adds value |
| Inclusions (multi-label) | macro-F1 0.48 | P=0.49 R=0.64 | — | 🟡 useful screen (common types) |
| Clarity (5-cls) | 69% | 6.8% | 37% | ❌ below baseline on unseen data |
| Fluorescence (4-cls) | 57% | 10.7% | 58% | ❌ below baseline |
| Eye-clean (2-cls) | 85% | 53.5% | 72% | ❌ below baseline |

## The key lesson
The balanced-test numbers were **optimistic**. The held-out-of-everything eval
(`src/evaluate_grade.py`) exposed that **clarity / fluorescence / eye-clean do
not generalize** from low-res marketing video — they score *below* the trivial
majority-class baseline on unseen stones. Causes:

1. **Flat-prior mismatch** — balanced training treats rare grades (IF, strong
   fluor) as common; partly corrected with log-prior adjustment at inference.
2. **Weak generalization** — small, non-randomly-sampled balanced subsets (first-N
   alphabetical → shared vendor/batch cues) made the within-pool test split not
   truly independent. The subtle heads latched onto subset-specific cues.

## What genuinely works from video
- **Shape** (99.5%) and **color** (67%) generalize to unseen inventory.
- **Inclusion presence** detects common types (Crystal/Needle/Feather) as a screen.

## What needs the grading machine, not the video
- **Clarity & inclusions** → microscope + darkfield (10–40×).
- **Fluorescence** → UV light source (white-light video lacks the signal).
- **Size / proportions** → calibrated silhouette/structured-light 3D (not learnable from video — no scale reference).
- All heads → **random, larger sampling** + calibration.
