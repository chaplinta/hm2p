# Plan: Perspective Correction for Keypoint Positions

## Problem

The overhead camera is not at the center of the maze and the mouse has
height above the maze floor (~2-5cm body, ~4-5cm with implant). This
causes **parallax displacement** — keypoints project outward from the
camera axis, appearing further from center than their true floor position.

Near maze walls, this pushes keypoints **outside the maze boundary**.
The current fix (clip to maze polygon) distorts trajectories.

## Geometry

```
Camera at height H = 700mm above maze floor
Camera optical center in cropped frame: ~(532, 251) px
Maze center in cropped frame: ~(453, 291) px
Scale: 0.811 mm/px
Maze: ~500mm × 347mm
Mouse body height: 2-5cm (varies by bodypart + implant)
```

The camera is ~64mm right and ~32mm above the maze center (in physical
space).

## Correction Formula

For a keypoint at pixel position `(px, py)` with the bodypart at height
`h` above the maze floor:

```
# Camera optical center in pixels (from uncropped frame center → crop offset)
cx, cy = camera_center_px  # per-session from meta.txt

# Apparent position relative to camera axis
dx = px - cx
dy = py - cy

# True floor position (corrected for parallax)
correction = H / (H - h)
px_corrected = cx + dx / correction
py_corrected = cy + dy / correction
```

This pushes the apparent position **toward the camera axis** by the
ratio `H / (H - h)`. The correction is larger for:
- Higher bodyparts (implant, ears during rearing)
- Positions further from the camera axis (maze edges)

## Per-Bodypart Height Estimates

| Bodypart | Typical height (mm) | With implant (mm) |
|----------|--------------------|--------------------|
| tail_base | 10 | 10 |
| mid_back | 20 | 20 |
| mouse_center | 20 | 20 |
| left_ear | 25 | 40 |
| right_ear | 25 | 40 |
| nose | 20 | 35 |

Note: head keypoints are higher with the 2P implant + headstage.
These are approximate — exact values would need calibration object or
side-camera measurement. The correction is not very sensitive to ±5mm
errors in height (at H=700mm, ±5mm changes the correction by < 1%).

## Implementation

### New function: `src/hm2p/kinematics/perspective.py`

```python
def correct_perspective(
    x_px: np.ndarray,       # (N,) keypoint x in pixels
    y_px: np.ndarray,       # (N,) keypoint y in pixels
    camera_center_px: tuple[float, float],  # (cx, cy) optical center
    camera_height_mm: float,  # H, default 700
    bodypart_height_mm: float,  # h, default 25
    scale_mm_per_px: float,   # from meta.txt
) -> tuple[np.ndarray, np.ndarray]:
    """Correct parallax displacement due to mouse body height."""
```

### Integration point: `src/hm2p/kinematics/compute.py`

In `compute_head_direction()` and `compute_position_mm()`, apply
perspective correction to each keypoint's pixel coordinates BEFORE
computing HD or converting to mm.

The correction happens in pixel space (before the px→mm scale
conversion), using the camera center from meta.txt.

### Per-session parameters

Each session has its own `meta.txt` with:
- Crop offset → camera center in cropped coords
- Scale (mm/px)
- ROI corners (maze boundary)

The camera height (700mm) and bodypart heights are constant across
sessions (same rig, same mice).

### Steps

1. Create `src/hm2p/kinematics/perspective.py` with `correct_perspective()`
2. Create `tests/kinematics/test_perspective.py` with unit tests:
   - Center pixel unchanged
   - Edge pixels pushed inward
   - h=0 gives no correction
   - Correction magnitude scales with distance from center
3. Integrate into `compute.py:run()` — apply after loading pose Dataset,
   before computing HD/position
4. Remove the `_clip_to_maze_polygon()` hack (or make it a final safety net)
5. Re-run kinematics for all 26 sessions (after DLC re-run finishes)

### Camera center estimation

For each session, the camera optical center = uncropped frame center
mapped to cropped coordinates:

```python
# From meta.txt [crop] section
crop_x, crop_y = 108, 261      # crop offset
uncrop_w, uncrop_h = 1280, 1024  # original frame (from movie-frame.tif)

# Camera center in cropped coordinates
cx = uncrop_w / 2 - crop_x  # = 532
cy = uncrop_h / 2 - crop_y  # = 251
```

This assumes the camera optical center = frame center (true after lens
distortion correction, which has already been applied).

## Expected Impact

At the maze edge (~250mm from center), with bodypart height 25mm:
- Displacement: 250 × 25/700 = **8.9mm** (~11 pixels)
- After correction: position is shifted 9mm inward toward camera axis

For head keypoints with implant (~40mm height):
- Displacement: 250 × 40/700 = **14.3mm** (~18 pixels)

This is significant — 14mm is a full maze corridor width offset.

## What This Does NOT Fix

- Detector failures (FasterRCNN losing the mouse) — fixed by max_individuals=1
- Jitter from low confidence — fixed by median filtering
- The implant occluding bodyparts — needs manual labelling + fine-tuning
