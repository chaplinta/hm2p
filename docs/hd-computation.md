# Head Direction Computation

How head direction (HD) is computed from DLC pose tracking keypoints
in the hm2p pipeline (Stage 3 — Kinematics).

## Overview

HD is computed by fusing up to five independent angular estimates from
the 5 head keypoints, combined per-frame via **confidence-weighted
circular mean**. Each estimate is weighted by the minimum DLC confidence
of its constituent keypoints. The system falls back gracefully when
keypoints are occluded (e.g. nose hidden behind the 2P implant).

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

## Five HD estimates

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

### Estimate 4: Ear midpoint → nose

Direction from the midpoint of the two ears to the nose.

```
         nose_tip
            ●
            ↑  ← HD direction
            |
    L_ear ●─●─● R_ear
         midpoint
```

**Formula:** `atan2(nose_x - mid_x, nose_y - mid_y) - 90°`

Combines ear and nose information. Useful when one ear is slightly
off but both still above threshold — the midpoint averages out
single-ear noise.

### Estimate 5: Neck → implant axis

Direction from neck to implant (head axis without requiring nose).

```
    implant_base_rear
            ●
            ↑  ← HD direction
            |
            ● neck
```

**Formula:** `atan2(implant_x - neck_x, implant_y - neck_y) - 90°`

Works when the nose is completely hidden (head-down poses, grooming).
The implant and neck are both rigid/semi-rigid landmarks that are
rarely occluded simultaneously.

## Fusion: confidence-weighted circular mean

Each estimate is weighted by the **minimum DLC confidence** of the
keypoints involved (e.g. estimate 2 uses min(conf_nose, conf_implant)).

1. For each estimate, compute weight = min(confidence of keypoints)
2. Convert each to a weighted unit vector: `w × (cos θ, sin θ)`
3. Sum the weighted vectors
4. Take the angle of the resultant: `atan2(Σ w·sin θ, Σ w·cos θ)`

```
    Est 1 (ears, conf=0.98):        0.98 × (0.71, 0.71)
    Est 2 (implant→nose, conf=0.95): 0.95 × (0.68, 0.73)
    Est 3 (neck→nose, conf=0.60):    0.60 × (0.72, 0.69)
    Est 4 (earmid→nose, conf=0.95):  0.95 × (0.70, 0.71)
    Est 5 (neck→implant, conf=0.60): 0.60 × (0.69, 0.72)
    ──────────────────────────────────────────
    Weighted sum → Fused HD ≈ 45.1°
```

High-confidence estimates dominate. When the nose is behind the implant
(low nose confidence), estimates 2, 3, and 4 get low weight and the
fusion naturally relies on the ears (estimate 1) and neck→implant
(estimate 5). When no confidence data is available (legacy pose files),
all estimates get equal weight (w=1).

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

---

## Head centre

The head centre position is computed as the **confidence-weighted mean**
of all available head keypoints (nose_tip, left_ear, right_ear,
implant_base_rear, neck).

```
         nose_tip (conf=0.95)
            ●
           / \
  L_ear ● ─ ● ─ ● R_ear
  (0.98)   |×|   (0.97)     ← × = head centre
        implant (0.99)
            |
         neck (0.80)
```

Each keypoint contributes proportionally to its DLC confidence. NaN
keypoints (below threshold) get zero weight. Falls back to ear midpoint
if only ears are available.

**Output:** `kinematics.h5:/head_centre_x`, `/head_centre_y` — in pixels.

**Implementation:** `compute_head_centre(ds)` in `compute.py`.

---

## Posture angles

### Head-body angle

The signed angular difference between head direction and body direction.

```
              nose
               ↑ HD
     head-body  ╲
     angle = θ   ╲
                  ↑ body direction
              mid_back
                  |
              tail_base
```

- **0°** = head aligned with body (moving straight)
- **+** = head turned left relative to body
- **−** = head turned right relative to body

Body direction: `tail_base → mid_back` (same angular convention as HD).

Useful for detecting head scanning at maze junctions — mice often
turn their head to look down an arm before committing to a path.

**Output:** `kinematics.h5:/head_body_angle` — degrees, range (-180, 180].

**Implementation:** `compute_head_body_angle(ds)` in `compute.py`.

### Neck angle

The angle at the neck keypoint between the head axis and body axis.

```
         implant
            ●
             \
     neck ● ─── angle = 180° (straight)
             /
            ●
         mid_back
```

- **180°** = head and body in a straight line
- **<180°** = head flexed forward/down
- **>180°** = head extended back/up

Head axis: `neck → implant_base_rear` (or `neck → ear_midpoint`).
Body axis: `neck → mid_back`.

Useful for detecting rearing, grooming, or head-dip postures.

**Output:** `kinematics.h5:/neck_angle` — degrees.

**Implementation:** `compute_neck_angle(ds)` in `compute.py`.

---

---

## Speed computation

### Locomotion speed (body)

Confidence-weighted mean of per-keypoint speeds from body keypoints.

```
                    mid_back ● ─── speed₁ (conf=0.97)
                             |
              mouse_center ● ─── speed₂ (conf=0.95)
                             |
                  tail_base ● ─── speed₃ (conf=0.98)
                             
    Locomotion speed = Σ(conf × speed) / Σ(conf)
```

Each body keypoint's speed is computed independently using windowed
linear regression (0.2s window, matching legacy pipeline). The per-keypoint
speeds are then combined via confidence-weighted mean.

Why not just use a single centroid speed? Because:
- A noisy keypoint with low confidence gets down-weighted
- Multiple independent estimates average out tracking jitter
- If one keypoint is NaN (below threshold), it's excluded automatically

**Keypoints used:** `mid_back`, `mouse_center`, `tail_base`
**Output:** `kinematics.h5:/speed_cm_s` — cm/s, float32

### Head translation speed

Same method but using head keypoints — captures head movement
independent of body translation. Useful for detecting head scanning
at maze junctions (head moves but body stays still).

```
         nose_tip ● ─── speed₁ (conf=0.92)
         left_ear ● ─── speed₂ (conf=0.98)
        right_ear ● ─── speed₃ (conf=0.97)
  implant_base_rear ● ─── speed₄ (conf=0.99)
             neck ● ─── speed₅ (conf=0.85)

    Head speed = Σ(conf × speed) / Σ(conf)
```

**Keypoints used:** `nose_tip`, `left_ear`, `right_ear`, `implant_base_rear`, `neck`
**Output:** `kinematics.h5:/head_speed_cm_s` — cm/s, float32

### Why separate head and body speed?

```
    Scenario 1: Walking straight
    ────────────────────────────
    head speed ≈ body speed ≈ 10 cm/s
    head-body angle ≈ 0°

    Scenario 2: Head scanning at junction
    ──────────────────────────────────────
    head speed ≈ 5 cm/s (head moving)
    body speed ≈ 0.5 cm/s (body stationary)
    head-body angle oscillating ±30°

    Scenario 3: Sharp turn
    ─────────────────────
    head speed > body speed (head leads)
    head-body angle increasing then returning to 0°
```

The ratio `head_speed / body_speed` and the `head_body_angle` together
characterise the mouse's movement strategy at each moment.

## Angular head velocity (AHV)

AHV is the time derivative of the **fused HD signal** — so it benefits
from all the confidence-weighted fusion described above. A cleaner HD
signal (less frame-to-frame jitter) produces a cleaner AHV.

**Formula:** windowed linear regression on unwrapped HD, same 0.2s window.

```
    HD (fused, unwrapped):  ...  45.2°  45.8°  46.1°  46.9°  47.3°  ...
                                 \___________ window ___________/
                                        slope = AHV (°/s)
```

**Output:** `kinematics.h5:/ahv_deg_s` — degrees/second, float32.
Positive = leftward rotation, negative = rightward.

### Movement library integration

The `movement` library provides `compute_speed()` and `compute_velocity()`
which operate per-keypoint on xarray Datasets. These are used for QC and
validation. The pipeline's production speed computation uses windowed
linear regression (matching the legacy pipeline) rather than movement's
central-difference method, because the windowed approach is more robust
to frame-to-frame jitter at ~30 fps.

For future work, movement's `compute_kinetic_energy(decompose=True)` could
provide an alternative locomotion speed estimate based on centre-of-mass
translational kinetic energy.

---

## Implementation

Source: `src/hm2p/kinematics/compute.py`

Key functions:
- `_ear_perpendicular_angle()` — HD estimate 1
- `_vector_angle_deg()` — HD estimates 2–5 (with -90° convention correction)
- `_fused_hd_wrapped()` — confidence-weighted circular mean fusion
- `_unwrap_and_smooth()` — unwrap + median filter
- `compute_head_direction()` — fused HD (top-level)
- `compute_head_centre()` — confidence-weighted head position
- `compute_head_body_angle()` — head vs body direction difference
- `compute_neck_angle()` — neck flexion angle
- `compute_locomotion_speed()` — body keypoint confidence-weighted speed
- `compute_head_speed()` — head keypoint confidence-weighted speed
- `compute_multipoint_speed()` — generic multi-keypoint speed (used by both)

Tests: `tests/kinematics/test_compute.py`
