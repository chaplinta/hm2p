# Plan: Perspective Correction for Keypoint Positions

## Problem

The overhead camera is not centered over the maze and the mouse has
height above the maze floor (~2-5cm body, ~4-5cm with 2P implant). This
causes **parallax displacement** — keypoints project outward from the
camera axis, appearing further from center than their true floor position.

Near maze walls, this pushes keypoints **outside the maze boundary**.
The current fix (clip to maze polygon) distorts trajectories and loses
information about wall-adjacent positions.

The goal is to **project each keypoint vertically down to the ground
plane**, removing the parallax caused by the bodypart's height.

## Geometry

```
Camera: Basler acA1300-200um (1/2" sensor, 4.8µm pixels, 1280×1024)
Lenses: f4mm (sessions 1-19) or f6mm (sessions 20-26)
Camera height: H = 700mm above maze floor
Camera position: off-center (~64mm right, ~32mm above maze center)
Camera optical center in cropped frame: ~(532, 251) px

Maze: ~500 × 347 mm (7×5 grid, 23 accessible cells)
Wall height: 120 mm (12 cm), 9mm clear acrylic
Corridor width: ~71 mm

Scale: 0.811 mm/px (f4mm), ~0.56 mm/px (f6mm)
```

## Correction Formula

For a keypoint at pixel position `(px, py)` with the bodypart at height
`h` above the maze floor:

```python
# Camera optical center in pixels
cx, cy = camera_center_px  # per-session from meta.txt

# Apparent position relative to camera axis
dx = px - cx
dy = py - cy

# True floor position (project vertically down to ground plane)
scale = H / (H - h)      # > 1 when h > 0
px_corrected = cx + dx / scale
py_corrected = cy + dy / scale
```

This pushes the apparent position **toward the camera axis** by
`H / (H - h)`. Equivalently, it projects the 3D point (x, y, h) straight
down to (x_floor, y_floor, 0) as seen by the off-axis camera.

## Maze Wall Dimensions (from laser-cut SVG)

| Parameter | Value |
|-----------|-------|
| Wall height above floor | 120 mm (12 cm) |
| Material thickness | 9 mm (acrylic) |
| Tab (slots into floor) | 10 mm |
| Total cut height | 130 mm |

Source: `Tristan-maze-walls-large-2.cdr` (CorelDRAW laser-cut file).

## Per-Bodypart Height Estimates

| Bodypart | Walking (mm) | With 2P implant (mm) |
|----------|-------------|---------------------|
| tail_base | 10 | 10 |
| mid_back | 20 | 20 |
| mouse_center | 20 | 20 |
| left_ear | 25 | 40 |
| right_ear | 25 | 40 |
| nose | 20 | 35 |

Max possible height: 120 mm (mouse rearing to wall top).

The 2P miniscope + headstage raises head keypoints ~15mm above normal.
Height estimates are approximate — exact values would require side-camera
measurement. The correction is not very sensitive to ±5mm errors
(at H=700mm, ±5mm changes the correction by < 1%).

## Parallax Displacement Table

At 250mm from camera axis (maze edge), camera at 700mm:

| Height | Displacement | Pixels (f4mm) |
|--------|-------------|--------------|
| 20 mm (walking) | 7.4 mm | 9 px |
| 40 mm (implant) | 15.2 mm | 19 px |
| 80 mm (rearing) | 32.3 mm | 40 px |
| 120 mm (wall top) | 51.7 mm | 64 px |

15mm displacement with implant ≈ 20% of corridor width.

## Implementation

### Step 1: New module `src/hm2p/kinematics/perspective.py`

```python
# Per-bodypart default heights (mm above maze floor)
BODYPART_HEIGHTS = {
    "tail_base": 10, "mid_back": 20, "mouse_center": 20,
    "left_ear": 25, "right_ear": 25, "nose": 20,
}
BODYPART_HEIGHTS_IMPLANT = {
    "tail_base": 10, "mid_back": 20, "mouse_center": 20,
    "left_ear": 40, "right_ear": 40, "nose": 35,
}
DEFAULT_CAMERA_HEIGHT = 700.0  # mm

def correct_perspective(
    x_px: np.ndarray,
    y_px: np.ndarray,
    camera_center_px: tuple[float, float],
    camera_height_mm: float = 700.0,
    bodypart_height_mm: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project keypoint from (x, y, h) to floor (x, y, 0)."""

def correct_dataset_perspective(
    ds: xr.Dataset,
    camera_center_px: tuple[float, float],
    camera_height_mm: float = 700.0,
    bodypart_heights: dict[str, float] | None = None,
) -> xr.Dataset:
    """Apply per-bodypart perspective correction to a movement Dataset.

    Iterates over keypoints, applies correct_perspective with the
    appropriate height for each bodypart.
    """

def estimate_camera_center(
    crop_x: int, crop_y: int,
    uncrop_w: int = 1280, uncrop_h: int = 1024,
) -> tuple[float, float]:
    """Camera optical center in cropped frame coordinates."""
    return (uncrop_w / 2 - crop_x, uncrop_h / 2 - crop_y)

def load_camera_params(meta_txt_path: Path) -> dict:
    """Parse meta.txt for crop offset, scale, ROI corners.
    Returns dict with camera_center_px, scale_mm_per_px, roi_corners.
    """
```

### Step 2: Tests `tests/kinematics/test_perspective.py`

- `test_center_pixel_unchanged`: point at camera center → no correction
- `test_edge_pushed_inward`: point far from center → moved toward center
- `test_zero_height_no_correction`: h=0 → identical output
- `test_correction_scales_with_distance`: further from center → larger correction
- `test_correction_scales_with_height`: higher bodypart → larger correction
- `test_per_bodypart_heights`: different keypoints get different corrections
- `test_dataset_correction`: movement Dataset input/output shapes preserved
- `test_estimate_camera_center`: known crop offset → correct center
- hypothesis: corrected point always between original and camera center

### Step 3: Integrate into `compute.py:run()`

In the pipeline `run()` function, after loading the pose Dataset and
applying confidence filtering + interpolation + median filter:

```python
# Load camera params from meta.txt (S3 or local)
cam_params = load_camera_params(meta_txt_path)

# Apply perspective correction (per-bodypart heights)
ds = correct_dataset_perspective(
    ds,
    camera_center_px=cam_params["camera_center_px"],
    camera_height_mm=700.0,
    bodypart_heights=BODYPART_HEIGHTS_IMPLANT,  # all mice have 2P implant
)
```

This goes AFTER median filtering (we want to correct the filtered
positions, not filter the corrected ones — order matters because the
correction is position-dependent).

Pipeline order:
1. Load pose (movement)
2. Rotate (orientation correction)
3. Filter by confidence (movement)
4. Interpolate gaps (movement)
5. Median filter (movement)
6. **Perspective correction** ← NEW
7. Compute HD, position, speed, etc.

### Step 4: Update `run()` to load meta.txt from S3

The `run()` function currently receives `scale_mm_per_px` and
`maze_corners_px` as parameters. Add `crop_offset` parameter (or load
meta.txt directly) to compute the camera center.

meta.txt is at: `s3://hm2p-rawdata/rawdata/{sub}/{ses}/behav/meta.txt`
Parsing: configparser INI format with sections [crop], [scale], [roi].

### Step 5: Demote clip-to-polygon to safety net

The current `_clip_to_maze_polygon()` clips out-of-bounds positions to
the nearest point on the maze boundary. After perspective correction,
most positions should be within bounds. Keep the clip as a final safety
net for extreme outliers (rearing, lost detections) but log a warning
when it activates — if it fires frequently, the height estimates need
adjusting.

### Step 6: Re-run kinematics for all 26 sessions

After DLC re-run completes (i-09f0a6f47e5834fac, ~8-12h):

```bash
python scripts/run_stage3_kinematics.py --all
python scripts/launch_kpms_ec2.py          # MoSeq (reads DLC directly)
python scripts/run_stage5_sync.py --all
python scripts/run_stage6_analysis.py --all
```

### Step 7: Validate correction

- Compare pre/post correction: how many points were outside maze before
  vs after
- Check HD computation: perspective correction on ear coordinates should
  give more stable HD near maze walls
- Visual QC: render labelled videos with corrected positions, compare
  with uncorrected

## Camera Center Per Session

Each session has its own crop region in meta.txt. The camera center in
cropped coordinates:

```python
cx = 1280/2 - crop_x   # uncropped frame center → cropped coords
cy = 1024/2 - crop_y
```

This must be computed per session (crop region varies slightly).

## What This Fixes

- Keypoints appearing outside maze walls (the main complaint)
- Systematic position bias near maze edges
- HD computation errors from parallax-displaced ear positions
- Speed/position artifacts near walls

## What This Does NOT Fix

- Detector failures (FasterRCNN losing the mouse) → fixed by max_individuals=1
- Jitter from low confidence → fixed by median filtering
- Implant occluding bodyparts → needs manual labelling + fine-tuning
- Height variation during locomotion (bobbing) → would need per-frame height
  estimation, probably not worth the complexity

## Dependencies

- Requires DLC re-run to complete first (new pose data)
- meta.txt must be accessible (on S3 in rawdata bucket)
- All downstream stages (sync, analysis, MoSeq) must re-run after

## Execution Order

```
1. Wait for DLC re-run (i-09f0a6f47e5834fac)
2. Implement perspective.py + tests
3. Update compute.py:run() to integrate correction
4. Re-run Stage 3 (kinematics) for all 26 sessions
5. Re-run Stage 3b (MoSeq) — reads DLC directly, not affected
6. Re-run Stage 5 (sync)
7. Re-run Stage 6 (analysis)
8. Re-render DLC labelled videos with corrected positions
9. Re-run hypothesis tests
10. Delete pipeline_rerun.json marker
```
