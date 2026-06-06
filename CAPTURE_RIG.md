# Capture-Rig Spec — the hardware that breaks the video ceilings

The software POC proved what's recoverable from a 620px marketing video and, more
usefully, **what isn't** — and exactly *why*. Every remaining gap is a **capture**
limit, not an algorithm limit. This rig closes them. Our existing models plug in
unchanged; they just receive far better inputs.

## The four ceilings the video hit, and the fix
| Attribute | Why video fails | Hardware fix |
|---|---|---|
| Clarity / inclusions | 620px native; small inclusions need 10–40× | **Microscope optics + darkfield lighting** |
| Fluorescence | white light only; no UV | **365 nm UV LED** |
| Carat / absolute mm | no scale reference in frame | **Precision scale** (+ calibrated optics) |
| Exact 3D proportions | free tumble, no clean profile/pose | **Motorized turntable** (known angles) + **backlight** |
| Color | uncontrolled white balance | **D65 lightbox** + master-stone reference |

## Components (production rig)
1. **Imaging:** machine-vision camera (e.g., 12–20 MP, global shutter) + **macro/microscope lens** giving 10–40× on the stone. Motorized focus for **focus-stacking** (inclusions sit at different depths).
2. **Darkfield illumination:** a ring of LEDs at a grazing angle against a dark background — makes inclusions light up bright against the stone (how gemologists view them). The single biggest lever for clarity/inclusions.
3. **UV source:** 365 nm LED, light-tight enclosure → fluorescence none/faint/medium/strong becomes directly measurable.
4. **Motorized turntable:** stone on a clear/known mount, rotated in known angular steps (e.g., 1–2°). Gives clean profile views + **known pose** → deterministic silhouette 3D.
5. **Backlight panel:** for crisp silhouettes (girdle outline, profile) at each turntable angle.
6. **Calibrated scale:** carat to 0.001 ct → with the 3D shape gives exact mm dimensions.
7. **Color station:** D65 (daylight) diffuse lighting + a set of master stones (D–J) in frame or via sequential comparison.
8. **Controller:** triggers the sequence (rotate → image darkfield → image backlit → UV → color) and a PC/GPU running the models.

## Capture protocol (per stone, ~30–60 s)
1. **Weigh** → carat (scale).
2. **Silhouette sweep:** rotate 360° on the turntable against backlight, capture outline at known angles → space-carve the 3D hull → exact proportions + (with weight) exact mm.
3. **Darkfield microscope sweep:** focus-stacked, magnified images at several angles → inclusion detection/localization + clarity.
4. **UV shot:** under 365 nm → fluorescence.
5. **Color shot:** D65 + master reference → color grade.

## How the existing software plugs in
| Tool we built | Rig input it consumes |
|---|---|
| `geometry_3d.py` (deterministic) | turntable silhouettes → **exact** geometry (it failed on video *only* for lack of clean pose) |
| `mm_dimensions.py` | silhouette 3D + weighed carat → exact mm |
| `train_inclusion.py` / `localize_inclusions.py` | darkfield microscope images → retrain → far higher precision + real localization |
| `train_shape_model.py` heads (color, clarity, eye-clean) | controlled images → retrain → higher accuracy |
| (new) fluorescence head | UV images → now has real signal |
| `predict_stone.py` / `digital_cert.py` | assembles all of it into the grade report |

## Validate cheap first — a $150–400 bench POC
Before building anything precise, prove the lift with off-the-shelf parts:
- **USB digital microscope** (10–200×, ~$50–150) + a cheap **darkfield/ring light**.
- **Manual or hobby turntable** + a phone/USB cam + a backlight panel.
- **365 nm UV flashlight** (~$15).
- **0.001 ct jeweler's scale** (~$30).
- Capture ~50 stones you have certs for; retrain the inclusion/clarity heads on these
  images and re-measure. **Expected:** clarity exact-grade and inclusion precision jump
  well past the video numbers (the inclusions are finally in-focus and high-contrast);
  fluorescence becomes gradeable; geometry becomes exact.

## Expected outcome
- **Geometry/carat/dimensions:** exact (deterministic, not ML) — the rig's strongest output.
- **Clarity/inclusions:** the big lift — magnification + darkfield is the difference between
  "screen" and "grade".
- **Fluorescence:** from *no signal* to gradeable.
- **Color:** controlled lighting closes the remaining gap.

The software is done and waiting. The next dollar goes to **capture**, not compute.
