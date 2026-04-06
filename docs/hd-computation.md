# Head Direction Computation

How head direction (HD) is computed from DLC pose tracking keypoints
in the hm2p pipeline (Stage 3 — Kinematics).

## Overview

HD is computed by fusing up to three independent angular estimates from
the 5 head keypoints, combined per-frame via circular mean. The system
falls back gracefully when keypoints are occluded (e.g. nose hidden
behind the 2P implant).

## Head keypoints

```
                  nose_tip
                    ●
                   / \
                  /   \
     left_ear ● ── ● ── ● right_ear
                   |
              implant_base_rear
                   |
                   ● neck
```

| Keypoint | Role in HD |
|---|---|
| `nose_tip` | Front of head — used in estimates 2 and 3 |
| `left_ear` | Primary HD estimate (ear perpendicular) |
| `right_ear` | Primary HD estimate (ear perpendicular) |
| `implant_base_rear` | Rigid skull reference — used in estimate 2 |
| `neck` | Head-body junction — used in estimate 3 |

## Three HD estimates

### Estimate 1: Ear perpendicular (primary)

The direction perpendicular to the line connecting the two ears.

```
            nose (facing up)
               ↑
               |  ← HD direction (perpendicular to ear line)
               |
   L_ear ●─────┼─────● R_ear
               |
              body
```

**Formula:** `atan2(left_ear_x - right_ear_x, left_ear_y - right_ear_y)`

This is the classic method used in most freely-moving HD studies. It works
well when both ears are clearly visible. Fails when one ear is occluded
(e.g. mouse against a wall, or ear hidden under the implant wiring).

### Estimate 2: Nose → implant axis

The direction from the implant (rear of skull) to the nose (front of head).
This is the head midline.

```
         nose_tip
            ●
            ↑  ← HD direction
            |
            ● implant_base_rear
```

**Formula:** `atan2(nose_x - implant_x, nose_y - implant_y)`

The implant is rigidly fixed to the skull and always visible (high-contrast
metal/ceramic). This estimate is robust when ears are ambiguous but fails
when the nose is occluded behind the implant (certain head-down poses).

### Estimate 3: Nose → neck axis

The direction from the neck (base of skull) to the nose.

```
         nose_tip
            ●
            ↑  ← HD direction
            |
            ● neck
```

**Formula:** `atan2(nose_x - neck_x, nose_y - neck_y)`

Fallback when the implant keypoint is unreliable. Less precise than
estimate 2 because the neck is softer tissue (slight movement relative
to skull), but still useful when other estimates are unavailable.

## Fusion: circular mean

At each frame, all non-NaN estimates are combined via **circular mean**:

1. Convert each available estimate to a unit vector: `(cos θ, sin θ)`
2. Sum the unit vectors
3. Take the angle of the resultant: `atan2(Σ sin θ, Σ cos θ)`

```
    Estimate 1 (ears): 45°   →  (0.71, 0.71)
    Estimate 2 (nose→implant): 47°  →  (0.68, 0.73)
    Estimate 3 (nose→neck): 44°    →  (0.72, 0.69)
    ─────────────────────────────────────
    Sum: (2.11, 2.13)
    Fused HD: atan2(2.13, 2.11) = 45.3°
```

When estimates agree (normal case), the fused result is close to all of
them. When one estimate is noisy or wrong, the other two pull the result
toward the correct direction. When a keypoint is NaN (below confidence
threshold), that estimate is simply excluded from the mean.

## Fallback chain

```
All 5 head keypoints available:
  → Circular mean of 3 estimates (most robust)

Nose occluded (behind implant):
  → Ear perpendicular only (estimate 1)

One ear occluded (against wall):
  → Nose→implant + nose→neck (estimates 2 + 3)

Only ears available (legacy 5-bodypart pose data):
  → Ear perpendicular only (backwards compatible)

Both ears + implant missing (rare):
  → Nose→neck only (estimate 3)

All head keypoints NaN:
  → NaN for that frame
```

## Post-processing

After computing the fused wrapped angle (0–360°) per frame:

1. **Interpolate NaN gaps** — short gaps (≤5 frames) are linearly
   interpolated so that unwrapping works cleanly.

2. **Unwrap** — remove 360° discontinuities to produce a continuous
   signal. Uses `numpy.unwrap` with π discontinuity threshold.

3. **Median filter** — 5-sample rolling median on the unwrapped signal
   to smooth frame-to-frame jitter. This is the only `scipy` call in
   the kinematics pipeline (movement's xarray-based filters don't
   handle 1D scalar signals).

4. **Restore NaN** — frames that were NaN before interpolation are
   set back to NaN in the final output.

## Camera rotation correction

Before HD computation, all keypoint coordinates are rotated by the
per-session `orientation` angle from `experiments.csv`. This corrects
for camera placement variation between sessions so that 0° means the
same real-world direction across all sessions.

## Output

The final HD signal is stored as:
- `kinematics.h5:/hd_deg` — unwrapped, in degrees, float32
- `sync.h5:/hd_deg` — same, aligned to imaging frames

## Implementation

Source: `src/hm2p/kinematics/compute.py`

Key functions:
- `_ear_perpendicular_angle()` — estimate 1
- `_vector_angle_deg()` — estimates 2 and 3
- `_fused_hd_wrapped()` — circular mean fusion
- `_unwrap_and_smooth()` — unwrap + median filter
- `compute_head_direction()` — top-level function (reads Dataset, dispatches)

Tests: `tests/kinematics/test_compute.py` (42 tests covering all functions)
