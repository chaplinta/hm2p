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

# movement source_software names keyed by tracker shorthand
_TRACKER_MAP: dict[str, str] = {
    "dlc": "DeepLabCut",
    "sleap": "SLEAP",
    "lp": "LightningPose",
}

# Keypoints used for body centroid position
_BODY_KEYPOINTS: tuple[str, ...] = ("back-upper", "back-middle", "back-tail")


# ---------------------------------------------------------------------------
# Pure helper functions (no I/O — fully unit-testable)
# ---------------------------------------------------------------------------


def _compute_hd_deg(
    ear_left_x: np.ndarray,
    ear_left_y: np.ndarray,
    ear_right_x: np.ndarray,
    ear_right_y: np.ndarray,
) -> np.ndarray:
    """Compute unwrapped head direction from ear vectors.

    Formula: arctan2(left_x - right_x, left_y - right_y) → 180 + degrees.
    NaN frames are temporarily filled by interpolation so that np.unwrap
    sees a continuous signal; NaN is restored afterward.

    Args:
        ear_left_x: (N,) x coordinate of ear-left keypoint.
        ear_left_y: (N,) y coordinate of ear-left keypoint.
        ear_right_x: (N,) x coordinate of ear-right keypoint.
        ear_right_y: (N,) y coordinate of ear-right keypoint.

    Returns:
        (N,) float32 — HD in degrees, unwrapped.
    """
    angle_rad = np.arctan2(ear_left_x - ear_right_x, ear_left_y - ear_right_y)
    angle_deg = 180.0 + np.degrees(angle_rad)

    nan_mask = np.isnan(angle_deg)
    if nan_mask.all():
        return np.full(len(angle_deg), np.nan, dtype=np.float32)

    # Fill NaN with linear interpolation so unwrap works cleanly
    angle_filled = angle_deg.copy()
    if nan_mask.any():
        indices = np.arange(len(angle_deg), dtype=float)
        valid = ~nan_mask
        angle_filled[nan_mask] = np.interp(indices[nan_mask], indices[valid], angle_deg[valid])

    rad_unwrapped = np.unwrap(np.deg2rad(angle_filled), discont=np.pi)
    deg_unwrapped = np.degrees(rad_unwrapped)
    deg_unwrapped[nan_mask] = np.nan
    return deg_unwrapped.astype(np.float32)


def _rotate_xy(
    x: np.ndarray,
    y: np.ndarray,
    angle_deg: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate (x, y) coordinates clockwise by angle_deg around (cx, cy).

    Args:
        x: x coordinates.
        y: y coordinates.
        angle_deg: Clockwise rotation angle in degrees.
        cx: Rotation centre x.
        cy: Rotation centre y.

    Returns:
        Tuple of (x_rot, y_rot).
    """
    rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    dx = x - cx
    dy = y - cy
    x_rot = cx + dx * cos_a + dy * sin_a
    y_rot = cy - dx * sin_a + dy * cos_a
    return x_rot, y_rot


def _maze_linear_transform(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    x1_mm: float,
    y1_mm: float,
    width_mm: float,
    height_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map mm positions to maze units (0–7 × 0–5) via linear scaling.

    Args:
        x_mm: (N,) x positions in mm.
        y_mm: (N,) y positions in mm.
        x1_mm: Maze top-left corner x in mm.
        y1_mm: Maze top-left corner y in mm.
        width_mm: Maze width in mm (x-span).
        height_mm: Maze height in mm (y-span).

    Returns:
        Tuple of (x_maze, y_maze), each (N,) float32.
    """
    x_maze = ((x_mm - x1_mm) / width_mm) * 7.0
    y_maze = ((y_mm - y1_mm) / height_mm) * 5.0
    return x_maze.astype(np.float32), y_maze.astype(np.float32)


def _clip_to_maze_polygon(
    x_maze: np.ndarray,
    y_maze: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip maze-unit positions to the rose-maze boundary polygon.

    Points outside the polygon are moved to their nearest point on the
    polygon boundary. NaN positions are preserved unchanged.

    Args:
        x_maze: (N,) x positions in maze units.
        y_maze: (N,) y positions in maze units.

    Returns:
        Tuple of (x_clipped, y_clipped), each (N,) float32.
    """
    from shapely import make_valid
    from shapely.geometry import Point, Polygon
    from shapely.ops import nearest_points

    maze_poly = make_valid(Polygon(MAZE_POLYGON_COORDS))

    x_out = x_maze.copy()
    y_out = y_maze.copy()

    for i in range(len(x_maze)):
        if np.isnan(x_maze[i]) or np.isnan(y_maze[i]):
            continue
        pt = Point(x_maze[i], y_maze[i])
        if not maze_poly.contains(pt):
            nearest = nearest_points(maze_poly, pt)[0]
            x_out[i] = nearest.x
            y_out[i] = nearest.y

    return x_out.astype(np.float32), y_out.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset-level functions
# ---------------------------------------------------------------------------


def load_pose_dataset(pose_path: Path, tracker: str) -> xr.Dataset:
    """Load tracker-native pose file into a unified movement xarray Dataset.

    Args:
        pose_path: Path to the tracker-native output file (.h5 for DLC/SLEAP, .csv for LP).
        tracker: Tracker identifier ('dlc', 'sleap', 'lp').

    Returns:
        xarray.Dataset with dimensions (time, individuals, keypoints, space)
        and a 'confidence' DataArray.
    """
    from movement.io import load_dataset

    if tracker not in _TRACKER_MAP:
        raise ValueError(f"Unknown tracker '{tracker}'. Known trackers: {list(_TRACKER_MAP)}")
    source_software = _TRACKER_MAP[tracker]
    return load_dataset(file=pose_path, source_software=source_software)


def apply_orientation_rotation(ds: xr.Dataset, angle_deg: float) -> xr.Dataset:
    """Rotate all keypoint (x, y) coordinates by angle_deg around the frame centre.

    Applied to correct for per-session camera placement variation. The rotation
    angle is stored in experiments.csv orientation column.

    Args:
        ds: movement Dataset with position DataArray.
        angle_deg: Clockwise rotation angle in degrees.

    Returns:
        Dataset with rotated position coordinates (copy).
    """
    if angle_deg == 0.0:
        return ds

    pos = ds.position  # (time, space, keypoints, individuals)
    x = pos.sel(space="x").values  # (time, keypoints, individuals)
    y = pos.sel(space="y").values

    # Rotate around mean of all keypoints (ignoring NaN)
    cx = float(np.nanmean(x))
    cy = float(np.nanmean(y))

    x_rot, y_rot = _rotate_xy(x, y, angle_deg, cx, cy)

    new_pos = pos.copy(data=np.stack([x_rot, y_rot], axis=pos.dims.index("space")))
    return ds.assign(position=new_pos)


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
    from movement.filtering import filter_by_confidence

    filtered_pos = filter_by_confidence(
        data=ds.position,
        confidence=ds.confidence,
        threshold=threshold,
    )
    return ds.assign(position=filtered_pos)


def interpolate_gaps(ds: xr.Dataset, max_gap_frames: int = 5) -> xr.Dataset:
    """Linearly interpolate NaN gaps of up to max_gap_frames consecutive frames.

    Args:
        ds: movement Dataset (after filter_low_confidence).
        max_gap_frames: Maximum gap length to interpolate over.

    Returns:
        Dataset with short NaN gaps filled.
    """
    from movement.filtering import interpolate_over_time

    interp_pos = interpolate_over_time(
        data=ds.position,
        method="linear",
        max_gap=max_gap_frames,
    )
    return ds.assign(position=interp_pos)


def compute_head_direction(ds: xr.Dataset) -> np.ndarray:
    """Compute unwrapped head direction from ear-left → ear-right forward vector.

    Args:
        ds: movement Dataset (filtered + interpolated).

    Returns:
        (N,) float32 — HD in degrees, unwrapped, referenced to camera frame.
    """
    pos = ds.position.isel(individuals=0)  # (time, space, keypoints)
    ear_left = pos.sel(keypoints="ear-left")
    ear_right = pos.sel(keypoints="ear-right")

    return _compute_hd_deg(
        ear_left_x=ear_left.sel(space="x").values,
        ear_left_y=ear_left.sel(space="y").values,
        ear_right_x=ear_right.sel(space="x").values,
        ear_right_y=ear_right.sel(space="y").values,
    )


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
    pos = ds.position.isel(individuals=0)  # (time, space, keypoints)
    back = pos.sel(keypoints=list(_BODY_KEYPOINTS))  # (time, space, keypoints_subset)

    x_px = float(scale_mm_per_px) * back.sel(space="x").mean(dim="keypoints").values
    y_px = float(scale_mm_per_px) * back.sel(space="y").mean(dim="keypoints").values

    return x_px.astype(np.float32), y_px.astype(np.float32)


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
            Ordered [top-left, top-right, bottom-right, bottom-left].
        scale_mm_per_px: Pixel → mm scale factor.

    Returns:
        Tuple of (x_maze, y_maze), each (N,) float32, clipped to maze polygon.
    """
    # Convert corners to mm
    corners_mm = maze_corners_px * scale_mm_per_px  # (4, 2)
    x1_mm = float(corners_mm[0, 0])
    y1_mm = float(corners_mm[0, 1])
    # Width: span from TL corner to TR corner; height: span from TL to BL corner
    width_mm = float(corners_mm[2, 0] - corners_mm[0, 0])
    height_mm = float(corners_mm[2, 1] - corners_mm[0, 1])

    x_maze, y_maze = _maze_linear_transform(x_mm, y_mm, x1_mm, y1_mm, width_mm, height_mm)
    return _clip_to_maze_polygon(x_maze, y_maze)


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
    from hm2p.io.hdf5 import read_h5, write_h5

    # --- Load timestamps ---
    ts = read_h5(timestamps_h5)
    frame_times = ts["frame_times_camera"]  # (N,) float64

    # --- Pose processing ---
    ds = load_pose_dataset(pose_path, tracker)
    ds = apply_orientation_rotation(ds, orientation_deg)
    ds = filter_low_confidence(ds, threshold=confidence_threshold)
    ds = interpolate_gaps(ds, max_gap_frames=gap_fill_frames)

    # --- Kinematics ---
    hd_deg = compute_head_direction(ds)  # (N,) float32
    x_mm, y_mm = compute_position_mm(ds, scale_mm_per_px)  # (N,) float32
    x_maze, y_maze = compute_maze_coords(x_mm, y_mm, maze_corners_px, scale_mm_per_px)

    # Speed (cm/s): first-order finite difference of mm position, converted to cm/s
    dx = np.gradient(x_mm, frame_times)
    dy = np.gradient(y_mm, frame_times)
    speed_cm_s = (np.sqrt(dx**2 + dy**2) / 10.0).astype(np.float32)  # mm/s → cm/s

    # Angular head velocity (deg/s)
    ahv_deg_s = np.gradient(hd_deg, frame_times).astype(np.float32)

    # Active/inactive state
    active = (speed_cm_s >= speed_active_threshold).astype(bool)

    # Light epoch and bad behaviour
    light_on_times = ts.get("light_on_times", np.empty(0, dtype=np.float64))
    light_off_times = ts.get("light_off_times", np.empty(0, dtype=np.float64))
    light_on = compute_light_on(frame_times, light_on_times, light_off_times)
    bad_behav = compute_bad_behav_mask(frame_times, bad_behav_intervals)

    # --- Write ---
    datasets = {
        "frame_times": frame_times,
        "hd_deg": hd_deg,
        "x_mm": x_mm,
        "y_mm": y_mm,
        "x_maze": x_maze,
        "y_maze": y_maze,
        "speed_cm_s": speed_cm_s,
        "ahv_deg_s": ahv_deg_s,
        "active": active,
        "light_on": light_on,
        "bad_behav": bad_behav,
    }
    attrs: dict[str, object] = {
        "session_id": session_id,
        "tracker": tracker,
        "confidence_threshold": confidence_threshold,
        "gap_fill_frames": gap_fill_frames,
        "scale_mm_per_px": scale_mm_per_px,
        "orientation_deg": orientation_deg,
        "speed_active_threshold_cm_s": speed_active_threshold,
    }
    write_h5(output_path, datasets, attrs=attrs)
