"""Stage 3 — behavioural kinematics via movement.

Loads pose output (any tracker) via movement.io.load_dataset(), applies
per-session camera rotation correction, filters low-confidence detections,
computes HD, position, speed, AHV, movement state, light epoch alignment,
and maze-coordinate positions. Writes kinematics.h5.

All body keypoints: ear-left, ear-right, back-upper, back-middle, back-tail.
HD = forward vector from ear-left → ear-right, unwrapped (degrees).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

# Maze is a 7×5 unit rose-maze grid.
# This shapely Polygon clips out-of-bounds positions.
MAZE_POLYGON_COORDS: list[tuple[int, int]] = [
    (0, 0),
    (3, 0),
    (3, 1),
    (2, 1),
    (2, 2),
    (5, 2),
    (5, 1),
    (4, 1),
    (4, 0),
    (7, 0),
    (7, 1),
    (6, 1),
    (6, 4),
    (7, 4),
    (7, 5),
    (4, 5),
    (4, 4),
    (5, 4),
    (5, 3),
    (4, 3),
    (4, 5),
    (3, 5),
    (3, 3),
    (2, 3),
    (2, 4),
    (3, 4),
    (3, 5),
    (0, 5),
    (0, 4),
    (1, 4),
    (1, 1),
    (0, 1),
]


def load_pose_dataset(pose_path: Path, tracker: str) -> xr.Dataset:
    """Load tracker-native pose file into a unified movement xarray Dataset.

    Args:
        pose_path: Path to the tracker-native output file (.h5 for DLC/SLEAP, .csv for LP).
        tracker: Tracker identifier ('dlc', 'sleap', 'lp').

    Returns:
        xarray.Dataset with dimensions (time, individuals, keypoints, space)
        and a 'confidence' DataArray.
    """
    raise NotImplementedError


def apply_orientation_rotation(ds: xr.Dataset, angle_deg: float) -> xr.Dataset:
    """Rotate all keypoint (x, y) coordinates by angle_deg around the frame centre.

    Applied to correct for per-session camera placement variation. The rotation
    angle is stored in experiments.csv orientation column.

    Args:
        ds: movement Dataset with position DataArray (..., space) where space=[x, y].
        angle_deg: Clockwise rotation angle in degrees.

    Returns:
        Dataset with rotated position coordinates.
    """
    raise NotImplementedError


def filter_low_confidence(
    ds: xr.Dataset,
    threshold: float = 0.9,
) -> xr.Dataset:
    """Set position to NaN for keypoints with confidence below threshold.

    Args:
        ds: movement Dataset.
        threshold: Likelihood threshold (default 0.9).

    Returns:
        Dataset with low-confidence detections replaced by NaN.
    """
    raise NotImplementedError


def interpolate_gaps(ds: xr.Dataset, max_gap_frames: int = 5) -> xr.Dataset:
    """Linearly interpolate NaN gaps of up to max_gap_frames consecutive frames.

    Args:
        ds: movement Dataset (after filter_low_confidence).
        max_gap_frames: Maximum gap length to interpolate over.

    Returns:
        Dataset with short NaN gaps filled.
    """
    raise NotImplementedError


def compute_head_direction(ds: xr.Dataset) -> np.ndarray:
    """Compute unwrapped head direction from ear-left → ear-right forward vector.

    Args:
        ds: movement Dataset (filtered + interpolated).

    Returns:
        (N,) float32 — HD in degrees, unwrapped, referenced to camera frame.
    """
    raise NotImplementedError


def compute_position_mm(
    ds: xr.Dataset,
    scale_mm_per_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute body centroid position in mm.

    Centroid is the mean of back-upper, back-middle, back-tail keypoints.

    Args:
        ds: movement Dataset.
        scale_mm_per_px: Pixel → mm scale factor from meta.txt.

    Returns:
        Tuple of (x_mm, y_mm), each (N,) float32.
    """
    raise NotImplementedError


def compute_maze_coords(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    maze_corners_px: np.ndarray,
    scale_mm_per_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map mm positions to rose-maze coordinate units (0–7 × 0–5).

    Out-of-bounds positions are clipped to the nearest point on the maze
    boundary polygon (MAZE_POLYGON_COORDS).

    Args:
        x_mm: (N,) float32 — x position in mm.
        y_mm: (N,) float32 — y position in mm.
        maze_corners_px: (4, 2) pixel coordinates of maze corners from meta.txt.
        scale_mm_per_px: Pixel → mm scale factor.

    Returns:
        Tuple of (x_maze, y_maze), each (N,) float32, clipped to maze polygon.
    """
    raise NotImplementedError


def compute_light_on(
    frame_times: np.ndarray,
    light_on_times: np.ndarray,
    light_off_times: np.ndarray,
) -> np.ndarray:
    """Compute per-frame light_on boolean from DAQ light pulse timestamps.

    Uses searchsorted to assign each camera frame to its lighting state.
    Light follows a periodic 1 min on / 1 min off cycle.

    Args:
        frame_times: (N,) float64 — camera frame timestamps in seconds.
        light_on_times: (L,) float64 — timestamps of light-on transitions.
        light_off_times: (L,) float64 — timestamps of light-off transitions.

    Returns:
        (N,) bool — True when overhead lights are on.
    """
    # Index of the last on/off event at or before each frame (-1 if none yet)
    i_on = np.searchsorted(light_on_times, frame_times, side="right") - 1
    i_off = np.searchsorted(light_off_times, frame_times, side="right") - 1

    has_on = i_on >= 0
    has_off = i_off >= 0

    result = np.zeros(len(frame_times), dtype=bool)

    # A light-on event exists but no light-off yet → lights on
    result[has_on & ~has_off] = True

    # Both events exist → whichever is more recent determines state
    both = has_on & has_off
    if both.any():
        dist_on = frame_times[both] - light_on_times[i_on[both]]
        dist_off = frame_times[both] - light_off_times[i_off[both]]
        result[both] = dist_on < dist_off

    return result


def compute_bad_behav_mask(
    frame_times: np.ndarray,
    bad_behav_intervals: list[tuple[float, float]],
) -> np.ndarray:
    """Build per-frame boolean mask for head-mount stuck artefact periods.

    Args:
        frame_times: (N,) float64 — camera frame timestamps in seconds.
        bad_behav_intervals: List of (start_s, end_s) from parse_bad_behav_times().

    Returns:
        (N,) bool — True during artefact (bad behaviour) periods.
    """
    mask = np.zeros(len(frame_times), dtype=bool)
    for start, end in bad_behav_intervals:
        mask |= (frame_times >= start) & (frame_times <= end)
    return mask


def run(
    pose_path: Path,
    timestamps_h5: Path,
    session_id: str,
    tracker: str,
    orientation_deg: float,
    scale_mm_per_px: float,
    maze_corners_px: np.ndarray,
    bad_behav_intervals: list[tuple[float, float]],
    output_path: Path,
    confidence_threshold: float = 0.9,
    gap_fill_frames: int = 5,
    speed_active_threshold: float = 2.0,
) -> None:
    """End-to-end Stage 3: pose file → kinematics.h5.

    Args:
        pose_path: Tracker-native pose output file.
        timestamps_h5: Stage 0 timestamps file.
        session_id: Canonical session identifier.
        tracker: Tracker backend name for movement.io.load_dataset().
        orientation_deg: Camera rotation from experiments.csv.
        scale_mm_per_px: Pixel → mm conversion from meta.txt.
        maze_corners_px: (4, 2) maze corner pixel coordinates.
        bad_behav_intervals: Stuck-fibre periods as (start_s, end_s) tuples.
        output_path: Destination kinematics.h5 file path.
        confidence_threshold: DLC/SLEAP likelihood cutoff.
        gap_fill_frames: Max frames to interpolate over.
        speed_active_threshold: cm/s threshold for active/inactive state.
    """
    raise NotImplementedError
