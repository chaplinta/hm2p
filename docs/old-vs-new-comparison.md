# Old Pipeline vs New Pipeline — Critical Comparison

Systematic comparison of the legacy pipeline (`old-pipeline/`) with the new
implementation (`src/hm2p/`). Flagging differences that could affect scientific
conclusions.

**Date:** 2026-03-14

---

## Summary of Critical Issues

| # | Issue | Severity | New Code Location |
|---|-------|----------|-------------------|
| 1 | `run.py` gaussian_sigma default 5.0s, should be 10.0s | **FIXED** | `calcium/run.py:56` |
| 2 | No median filtering on HD or body positions | **FIXED** | `kinematics/compute.py` |
| 3 | Active threshold 2.0 cm/s vs old 0.5 cm/s | **FIXED** | `kinematics/compute.py` |
| 4 | Speed from body centroid vs head position | **MEDIUM** | `kinematics/compute.py` |
| 5 | No `bad_2p_frames` exclusion | **MEDIUM** | `calcium/run.py` |
| 6 | Single Suite2p run vs dual soma/dend runs | **MEDIUM** | `extraction/suite2p.py` |
| 7 | Speed algorithm (central diff vs windowed regression) | **FIXED** | `kinematics/compute.py` |
| 8 | Mean-resampling vs linear interpolation for sync | **LOW** | `sync/align.py` |
| 9 | dF/F0 safety floor + clip (new) vs raw division (old) | **LOW** | `calcium/dff.py` |

---

## 1. dF/F0 Computation

### Old pipeline (`S2PData.py`, `get_f0`)
- Neuropil: `F_corr = F - ops["neucoeff"] * Fneu` (Suite2p's stored coefficient, typically 0.7)
- Baseline F0: 3-step rolling filter (Gaussian smooth → min → max)
- Parameters from Suite2p ops: `win_baseline` (60s), `sig_baseline` (varies)
- Gaussian filter: `sigma=[0., sigma]` (2D, axis=1)
- dF/F0 = `(F_corr - F0) / F0` — **no safety floor on F0**

### New pipeline (`dff.py`, `run.py`, `neuropil.py`)
- Neuropil: `F_corr = F - 0.7 * Fneu` (hardcoded default)
- Same 3-step baseline filter
- Default sigma: `compute_baseline()` defaults to 10.0s, but **`run.py` overrides to 5.0s**
- Safety floor: `safe_F0 = max(F0, max(median(F0)*0.1, 1.0))`
- Hard clip: `dff = np.clip(dff, -1.0, 20.0)`

### Critical issues
1. **Sigma mismatch**: ~~`run.py` has `dff_gaussian_sigma_s=5.0` contradicting the intended 10.0s.~~
   **FIXED** — Default changed from 5.0 to 10.0 in `run.py`, matching the old pipeline.
2. **Safety floor**: New code prevents extreme values from near-zero F0. Old code has no
   protection. This is an improvement, not a bug.
3. **dtype**: Old uses float64; new uses float32 output (small rounding differences).

---

## 2. Event Detection (Voigts & Harnett)

### Old pipeline (`utils/ca.py`)
- Constants: `SMOOTH_SIGMA=3`, `PRC_MEAN=40`, `PRC_LOW=10`, `PRC_HIGH=90`,
  `PROB_ONSET=0.2`, `PROB_OFFSET=0.7`, `ALPHA=1`
- Noise estimation: rectify → smooth → normalize via `trace/max(trace)` then `(trace-min)/(max-min)`
- Offset search: `noise_prob[i_onset:-1]` (excludes last frame)

### New pipeline (`events.py`)
- Same constants and algorithm
- Noise estimation: rectify → smooth → `(trace-min)/(max-min)` (single normalization)
- Offset search: `noise_prob[i_onset:]` (includes last frame)

### Analysis
The double normalization in the old code (`/max` then `(x-min)/(max-min)`) is mathematically
equivalent to single min-max normalization for non-negative inputs, so **no actual discrepancy**.
The offset search slice difference (`:-1` vs `:`) only affects events at the very last frame.

---

## 3. Head Direction Computation

### Old pipeline (`behave.py`)
```python
absolute_head_angle = np.arctan2((left_x - right_x), (left_y - right_y))
absolute_head_angle = 180 + absolute_head_angle * 180 / np.pi
```
- Uses **5-frame rolling median filtered** ear positions
- Unwraps, then applies **5-frame rolling median** to unwrapped HD
- Wraps back to [0, 360)

### New pipeline (`compute.py`)
```python
angle_rad = np.arctan2(ear_left_x - ear_right_x, ear_left_y - ear_right_y)
angle_deg = 180.0 + np.degrees(angle_rad)
```
- **Identical formula** for raw HD
- NaN interpolation before unwrapping (handles missing keypoints)
- **No median filter** on positions or HD

### Critical issue — FIXED
~~**No median filtering**~~: **FIXED** — Added `_median_filter_1d` (window=5) applied to
both ear positions AND unwrapped HD in `_compute_hd_deg`, matching the old pipeline's
two-stage median filtering. This reduces noise propagation to HD tuning curves, AHV, and
significance testing.

---

## 4. Speed Computation

### Old pipeline
- Multiple measures: `SPEED_INST`, `SPEED_GRAD`, `SPEED_FILT_GRAD`, `LOCO_SPEED_FILT_GRAD`
- Primary: windowed linear regression over 0.2s sliding window
- Head speed: from mean ear position
- Locomotion speed: from back/body position
- Units: cm/s

### New pipeline
- Single measure: `np.gradient(x_mm, frame_times)` → central difference
- From body centroid (mean of mid_back, mouse_center, tail_base)
- Units: cm/s

### Issues
1. **Body part**: Old uses head for "head speed" and back for "locomotion speed". New uses
   body centroid only.
2. **Algorithm**: ~~Windowed regression vs central difference~~ **FIXED** — Added
   `_windowed_gradient` and `_windowed_speed` using 0.2s sliding window linear regression,
   replacing `np.gradient` central difference. Now matches the old pipeline's temporal
   smoothing approach.
3. **No head speed**: The old pipeline's `SPEED_FILT_GRAD` was used for most analyses.

---

## 5. Active/Inactive Classification

### Old pipeline (`db.py`)
- Multi-criterion: speed > 0.5 cm/s OR AHV > 10 deg/s OR locomotion > 1.5 cm/s
- Very permissive — most frames classified as active

### New pipeline
- Simple: speed > 2.0 cm/s
- Much more restrictive — many more frames classified as inactive

### Critical issue — FIXED
~~**Threshold difference**: 2.0 cm/s vs 0.5 cm/s~~: **FIXED** — Default changed from 2.0
to 0.5 cm/s, matching the old pipeline. The threshold remains configurable via the
`speed_threshold` parameter.

---

## 6. Angular Head Velocity (AHV)

### Old pipeline
- `AHV_FILT_GRAD`: windowed regression of median-filtered unwrapped HD × fps
- Double smoothing: median filter on HD + gradient over 0.2s window

### New pipeline
- `np.gradient(hd_deg, frame_times)` — central difference on unfiltered HD
- No smoothing at all

### Issue — FIXED
~~Much noisier AHV in new pipeline.~~ **FIXED** — AHV now uses `_windowed_gradient` (0.2s
sliding window linear regression) on median-filtered unwrapped HD, matching the old
pipeline's two-stage smoothing (median filter + windowed regression).

---

## 7. ROI Classification (Soma vs Dendrite)

### Old pipeline
- Runs Suite2p **twice** with different classifiers (soma, dendrite)
- Different detection parameters (`crop_soma=False` for dendrites)
- Gets different ROI populations from each run

### New pipeline
- Single Suite2p run
- Applies both classifiers **post-hoc** to same ROIs
- Fallback heuristic: aspect_ratio > 2.5 → dendrite

### Issue
Different ROI populations. The old pipeline's dendrite run with `crop_soma=False` could
detect elongated dendrite ROIs that the single-run approach misses entirely.

---

## 8. Synchronization / Resampling

### Old pipeline
- For each imaging frame, finds all camera frames in the interval
- Takes **mean** of camera metrics within the interval
- HD: means unwrapped HD (avoids 0/360 discontinuity), then re-wraps

### New pipeline
- `np.interp()` — linear interpolation at imaging timestamps
- For booleans: nearest-neighbour via `searchsorted`

### Issue
At ~100 Hz camera / ~9.6 Hz imaging (~10 camera frames per imaging frame), mean provides
temporal averaging while interpolation gives instantaneous values. For slowly-varying
signals this difference is minor.

---

## 9. Light Cycle Detection

### Old pipeline
- Inserts sentinel `0` at start of `light_off_times`
- For each frame: finds nearest on/off event, picks whichever is closer

### New pipeline
- Uses `searchsorted` to find most recent on/off event
- Explicit edge case handling

### Analysis
Logically equivalent, different implementation. Should produce identical results.

---

## 10. Features in Old Pipeline Not Yet Ported

1. **Joint soma-dendrite event detection** (`get_joint_ca_events`): Complex algorithm
   classifying events as somatic, dendritic, or joint. Not in new pipeline.
2. **Multiple speed measures** (head speed, locomotion speed, gradient-based)
3. **Allocentric and egocentric heading** from velocity vectors
4. **Acceleration** from speed gradient
5. **Cumulative distance moved**
6. **DLC jump detection**: Old code interpolated positions with large frame-to-frame jumps
   (`thresh_diff=20` pixels) in addition to low-likelihood filtering

---

## 11. Improvements in New Pipeline (Not in Old)

1. **F0 safety floor and clip**: Prevents extreme dF/F0 from near-zero baselines
2. **NaN-aware HD computation**: Handles missing keypoints gracefully
3. **Multiple decoders**: PVA + template matching (old had none)
4. **Multi-signal analysis**: Systematic comparison of dF/F0 vs deconv vs events
5. **Automated significance testing**: Circular shuffle with configurable parameters
6. **Population-level statistics**: Cross-session pooling
7. **Comprehensive frontend**: 43+ pages for visualization and QC

---

## Recommended Priority Fixes

1. **~~Fix sigma in `run.py`~~**: DONE — Changed `dff_gaussian_sigma_s` default from 5.0 to 10.0 in `run.py`.
2. **~~Add HD median filter~~**: DONE — Added `_median_filter_1d` (win=5) on ear positions AND unwrapped HD in `_compute_hd_deg`.
3. **~~Lower active threshold~~**: DONE — Changed default from 2.0 to 0.5 cm/s in `compute.py`.
4. **~~Add windowed speed/AHV~~**: DONE — Added `_windowed_gradient` and `_windowed_speed` using 0.2s sliding window linear regression, replacing `np.gradient` central difference.
5. **Document dual vs single Suite2p run trade-off**: Decide if dendrite detection matters
