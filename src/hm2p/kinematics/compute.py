"""Stage 3 — behavioural kinematics via movement.

Loads pose output (any tracker) via movement.io.load_dataset(), applies
per-session camera rotation correction, filters low-confidence detections,
computes HD, position, speed, AHV, movement state, light epoch alignment,
and maze-coordinate positions. Writes kinematics.h5.

Keypoints used: nose_tip, left_ear, right_ear, head_midpoint, neck,
mid_back, mouse_center, tail_base.

HD is computed by fusing 4 independent estimates from the head keypoints,
weighted by mean DLC confidence of the constituent keypoints:
  1. Ears: perpendicular to left_ear → right_ear.
  2. Nose → head_midpoint axis: direction from head_midpoint to nose_tip.
  3. Nose → neck axis: direction from neck to nose_tip.
  4. Head_midpoint → neck axis: direction from neck to head_midpoint.
Falls back gracefully when keypoints are occluded (e.g. nose behind the
2P implant). Backwards-compatible: works with ears-only pose data.

Individual HD estimates are stored alongside the fused signal for QC.

Position is separated into:
  - Head position: confidence-weighted average of 3 head-keypoint estimates.
  - Body position: confidence-weighted centroid of mid_back, mouse_center,
    tail_base (body-axis keypoints; head rotates independently).

Speed uses np.gradient on median-filtered position (3-point window).
AHV uses np.gradient on unwrapped, median-filtered fused HD.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hm2p.constants import SPEED_ACTIVE_THRESHOLD

if TYPE_CHECKING:
    import xarray as xr

_log = logging.getLogger("hm2p.kinematics")

# Maze is a 7×5 unit q-rose maze grid.
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

# Keypoint names
_EAR_LEFT: str = "left_ear"
_EAR_RIGHT: str = "right_ear"
_NOSE: str = "nose_tip"
_IMPLANT: str = "head_midpoint"
_NECK: str = "neck"

# Keypoints used for body centroid position
_BODY_KEYPOINTS: tuple[str, ...] = ("mid_back", "mouse_center", "tail_base")


# ---------------------------------------------------------------------------
# Pure helper functions (no I/O — fully unit-testable)
# ---------------------------------------------------------------------------


def _median_filter_1d(arr: np.ndarray, win: int = 5) -> np.ndarray:
    """Apply rolling median filter to a 1D numpy signal, preserving NaN.

    Used only for post-HD-unwrap smoothing (a scalar 1D signal that
    movement's xarray-based rolling_filter cannot handle). For position
    data, use ``movement.filtering.rolling_filter`` on the Dataset instead.

    Args:
        arr: (N,) input signal (may contain NaN).
        win: Window size (default 5, must be odd).

    Returns:
        (N,) float64 — median-filtered signal with NaN preserved.
    """
    from scipy.ndimage import median_filter

    if win <= 1:
        return arr.copy()
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return arr.copy()
    filled = arr.copy()
    if nan_mask.any():
        idx = np.arange(len(arr), dtype=float)
        valid = ~nan_mask
        filled[nan_mask] = np.interp(idx[nan_mask], idx[valid], arr[valid])
    out = median_filter(filled, size=win, mode="nearest")
    out[nan_mask] = np.nan
    return out


def _savgol_filter_1d(
    arr: np.ndarray, window: int, polyorder: int = 2
) -> np.ndarray:
    """Apply a Savitzky-Golay smoother to a 1D signal, preserving NaN.

    NaN values are linearly interpolated before filtering and restored
    afterwards so the smoother sees a continuous signal but does not leak
    smoothed values into frames that were missing in the input.

    Used on the unwrapped HD timeseries to reduce high-frequency tracking
    jitter while preserving the curvature of fast head turns better than a
    plain median filter (Schafer 2011 — "What is a Savitzky-Golay filter?",
    IEEE Signal Process. Mag. 28(4):111-117. doi:10.1109/MSP.2011.941097).

    Args:
        arr: (N,) input signal (may contain NaN).
        window: Filter window size. Must be odd and > polyorder. Values
            ``<= 1`` disable the filter and return ``arr`` unchanged.
        polyorder: Polynomial order (default 2). Must be < window.

    Returns:
        (N,) float64 — smoothed signal with NaN preserved. Returns ``arr``
        unchanged if it has fewer than ``window`` valid samples.
    """
    from scipy.signal import savgol_filter

    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        raise ValueError(f"savgol window must be odd, got {window}")
    if polyorder >= window:
        raise ValueError(
            f"polyorder ({polyorder}) must be < window ({window})"
        )

    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return arr.copy()
    n_valid = int((~nan_mask).sum())
    if n_valid < window:
        return arr.copy()

    filled = arr.copy().astype(np.float64)
    if nan_mask.any():
        idx = np.arange(len(arr), dtype=float)
        valid = ~nan_mask
        filled[nan_mask] = np.interp(idx[nan_mask], idx[valid], arr[valid])

    out = savgol_filter(filled, window_length=window, polyorder=polyorder)
    out[nan_mask] = np.nan
    return out


def median_filter_dataset(ds: xr.Dataset, window: int = 3) -> xr.Dataset:
    """Apply movement's rolling median filter to the position DataArray.

    This replaces per-coordinate ``_median_filter_1d`` calls with
    movement's native xarray-based filter for all keypoints at once.

    The default window of 3 frames at 30fps gives ~100ms temporal
    smoothing.  The old pipeline used 5 frames at 100fps = 50ms.
    If the DLC inference frame rate changes, adjust this window to
    maintain approximately 100ms smoothing (window = round(0.1 * fps)).

    Args:
        ds: movement Dataset with a ``position`` DataArray.
        window: Rolling window size (default 3; ~100ms at 30fps).

    Returns:
        Dataset with median-filtered positions.
    """
    from movement.filtering import rolling_filter

    import logging as _logging

    try:
        filtered = rolling_filter(ds.position, window=window, statistic="median")
        if filtered.shape == ds.position.shape:
            ds["position"].values[:] = filtered.values
    except Exception as exc:
        _logging.getLogger("hm2p.kinematics").warning(
            "Median filter failed — using unfiltered positions: %s", exc
        )
    return ds


def _vector_angle_deg(
    from_x: np.ndarray,
    from_y: np.ndarray,
    to_x: np.ndarray,
    to_y: np.ndarray,
) -> np.ndarray:
    """Compute per-frame heading angle (degrees) of the vector from→to.

    Applies a -90° correction so the output matches the angular convention
    used by ``_ear_perpendicular_angle``. The ear perpendicular formula
    defines the HD coordinate system for the pipeline; all other estimates
    must be rotated to match.

    Returns (N,) float64. NaN where either point is NaN.
    """
    dx = to_x - from_x
    dy = to_y - from_y
    # Raw atan2 angle, then -90° to match ear perpendicular convention.
    raw = 180.0 + np.degrees(np.arctan2(dx, dy))
    return raw - 90.0


def _ear_perpendicular_angle(
    ear_left_x: np.ndarray,
    ear_left_y: np.ndarray,
    ear_right_x: np.ndarray,
    ear_right_y: np.ndarray,
) -> np.ndarray:
    """HD from perpendicular to ear-ear line (original method).

    Returns (N,) float64. NaN where either ear is NaN.
    """
    angle_rad = np.arctan2(
        ear_left_x - ear_right_x, ear_left_y - ear_right_y
    )
    return 180.0 + np.degrees(angle_rad)


def _unwrap_and_smooth(
    angle_deg: np.ndarray,
    median_filter_win: int = 5,
    savgol_window: int = 5,
    savgol_polyorder: int = 2,
) -> np.ndarray:
    """Unwrap a wrapped angle timeseries and apply two-pass smoothing.

    Pass 1 (median filter, ``median_filter_win`` frames) removes single-frame
    impulses from tracking errors.

    Pass 2 (Savitzky-Golay filter, ``savgol_window`` frames, polynomial of
    order ``savgol_polyorder``) smooths residual high-frequency jitter while
    preserving the curvature of fast head turns. At 30 fps the default
    5-frame window corresponds to ~165 ms of smoothing — short relative to
    the typical HD bin-dwell time, so HD tuning curves are not blurred.

    Args:
        angle_deg: (N,) wrapped angle in degrees (may contain NaN).
        median_filter_win: Window for post-unwrap median filter. ``<= 1``
            disables the median pass.
        savgol_window: Window for the Savitzky-Golay smoother. ``<= 1``
            disables the SG pass.
        savgol_polyorder: Polynomial order for SG. Must be < window.

    Returns:
        (N,) float32 — unwrapped, smoothed HD in degrees.
    """
    nan_mask = np.isnan(angle_deg)
    if nan_mask.all():
        return np.full(len(angle_deg), np.nan, dtype=np.float32)

    angle_filled = angle_deg.copy()
    if nan_mask.any():
        idx = np.arange(len(angle_deg), dtype=float)
        valid = ~nan_mask
        angle_filled[nan_mask] = np.interp(
            idx[nan_mask], idx[valid], angle_deg[valid]
        )

    rad_unwrapped = np.unwrap(np.deg2rad(angle_filled), discont=np.pi)
    deg_unwrapped = np.degrees(rad_unwrapped)

    # Pass 1: median filter — removes single-frame tracking impulses.
    deg_unwrapped = _median_filter_1d(deg_unwrapped, median_filter_win)
    # Pass 2: Savitzky-Golay — removes residual high-frequency jitter
    # without introducing the phase distortion of cumulative median filters.
    deg_unwrapped = _savgol_filter_1d(
        deg_unwrapped, window=savgol_window, polyorder=savgol_polyorder
    )

    deg_unwrapped[nan_mask] = np.nan
    return deg_unwrapped.astype(np.float32)


def _compute_hd_deg(
    ear_left_x: np.ndarray,
    ear_left_y: np.ndarray,
    ear_right_x: np.ndarray,
    ear_right_y: np.ndarray,
    median_filter_win: int = 5,
) -> np.ndarray:
    """Compute unwrapped HD from ear vectors only (legacy interface).

    Kept for backwards compatibility and unit tests.
    """
    angle_deg = _ear_perpendicular_angle(
        ear_left_x, ear_left_y, ear_right_x, ear_right_y
    )
    return _unwrap_and_smooth(angle_deg, median_filter_win)


def _fused_hd_wrapped(
    ear_left_x: np.ndarray,
    ear_left_y: np.ndarray,
    ear_right_x: np.ndarray,
    ear_right_y: np.ndarray,
    nose_x: np.ndarray | None = None,
    nose_y: np.ndarray | None = None,
    implant_x: np.ndarray | None = None,
    implant_y: np.ndarray | None = None,
    neck_x: np.ndarray | None = None,
    neck_y: np.ndarray | None = None,
    conf_ear_left: np.ndarray | None = None,
    conf_ear_right: np.ndarray | None = None,
    conf_nose: np.ndarray | None = None,
    conf_implant: np.ndarray | None = None,
    conf_neck: np.ndarray | None = None,
) -> np.ndarray:
    """Fuse multiple HD estimates via confidence-weighted circular mean.

    Five estimates are computed (when keypoints are available and not NaN):
      1. Ear perpendicular: perpendicular to left_ear→right_ear.
      2. Head midpoint→nose: direction from head_midpoint to nose_tip.
      3. Neck→nose: direction from neck to nose_tip.
      4. Ear midpoint→nose: direction from mean(ears) to nose_tip.
      5. Neck→head midpoint: direction from neck to head_midpoint.

    Each estimate is weighted by the minimum confidence of the keypoints
    involved. If no confidence arrays are provided, equal weights are used.

    At each frame, weighted circular mean of all non-NaN estimates is taken.
    If only ears are available (legacy pose data), this reduces to the
    original ear-only method.

    Args:
        ear_left_x/y: (N,) ear positions (required).
        nose_x/y: (N,) nose positions (optional).
        implant_x/y: (N,) implant positions (optional).
        neck_x/y: (N,) neck positions (optional).
        conf_*: (N,) DLC confidence per keypoint (optional, 0-1).

    Returns:
        (N,) float64 — fused wrapped HD in [0, 360) degrees. NaN where
        no estimate is available.
    """
    n = len(ear_left_x)

    def _min_conf(*arrays: np.ndarray | None) -> np.ndarray:
        """Per-frame minimum confidence across keypoints, default 1.0."""
        valid = [a for a in arrays if a is not None]
        if not valid:
            return np.ones(n, dtype=np.float64)
        return np.minimum.reduce(valid)

    # Estimate 1: ear perpendicular (always available)
    est_ear = _ear_perpendicular_angle(
        ear_left_x, ear_left_y, ear_right_x, ear_right_y
    )
    w_ear = _min_conf(conf_ear_left, conf_ear_right)

    # Estimate 2: implant → nose
    est_implant_nose = np.full(n, np.nan)
    w_implant_nose = np.ones(n)
    if nose_x is not None and implant_x is not None:
        est_implant_nose = _vector_angle_deg(
            implant_x, implant_y, nose_x, nose_y
        )
        w_implant_nose = _min_conf(conf_nose, conf_implant)

    # Estimate 3: neck → nose
    est_neck_nose = np.full(n, np.nan)
    w_neck_nose = np.ones(n)
    if nose_x is not None and neck_x is not None:
        est_neck_nose = _vector_angle_deg(neck_x, neck_y, nose_x, nose_y)
        w_neck_nose = _min_conf(conf_nose, conf_neck)

    # Estimate 4: ear midpoint → nose
    est_earmid_nose = np.full(n, np.nan)
    w_earmid_nose = np.ones(n)
    if nose_x is not None:
        mid_x = (ear_left_x + ear_right_x) / 2.0
        mid_y = (ear_left_y + ear_right_y) / 2.0
        est_earmid_nose = _vector_angle_deg(mid_x, mid_y, nose_x, nose_y)
        w_earmid_nose = _min_conf(conf_ear_left, conf_ear_right, conf_nose)

    # Estimate 5: neck → implant (head axis without nose)
    est_neck_implant = np.full(n, np.nan)
    w_neck_implant = np.ones(n)
    if implant_x is not None and neck_x is not None:
        est_neck_implant = _vector_angle_deg(
            neck_x, neck_y, implant_x, implant_y
        )
        w_neck_implant = _min_conf(conf_neck, conf_implant)

    # Confidence-weighted circular mean of all available estimates.
    estimates = [
        est_ear, est_implant_nose, est_neck_nose,
        est_earmid_nose, est_neck_implant,
    ]
    weights = [
        w_ear, w_implant_nose, w_neck_nose,
        w_earmid_nose, w_neck_implant,
    ]

    sin_sum = np.zeros(n, dtype=np.float64)
    cos_sum = np.zeros(n, dtype=np.float64)
    w_sum = np.zeros(n, dtype=np.float64)

    for est, w in zip(estimates, weights):
        valid = ~np.isnan(est)
        rad = np.deg2rad(est)
        sin_sum[valid] += w[valid] * np.sin(rad[valid])
        cos_sum[valid] += w[valid] * np.cos(rad[valid])
        w_sum[valid] += w[valid]

    fused = np.full(n, np.nan)
    has_any = w_sum > 0
    mean_angle = np.degrees(np.arctan2(sin_sum[has_any], cos_sum[has_any]))
    fused[has_any] = mean_angle % 360.0
    return fused


def compute_head_centre(ds: "xr.Dataset") -> tuple[np.ndarray, np.ndarray]:
    """Compute head centre position from available head keypoints.

    Uses confidence-weighted mean of all available head keypoints:
    nose_tip, left_ear, right_ear, head_midpoint, neck.
    Falls back to ear midpoint if only ears are available.

    Args:
        ds: movement Dataset (filtered + interpolated) with confidence.

    Returns:
        Tuple of (x, y), each (N,) float32 — head centre in pixels.
    """
    pos = ds.position.isel(individuals=0)
    available_kps = list(pos.coords["keypoints"].values)

    # Collect head keypoint positions and confidences
    head_kp_names = [_NOSE, _EAR_LEFT, _EAR_RIGHT, _IMPLANT, _NECK]
    xs = []
    ys = []
    confs = []

    has_conf = "confidence" in ds
    conf_da = ds.confidence.isel(individuals=0) if has_conf else None

    for name in head_kp_names:
        if name not in available_kps:
            continue
        kp = pos.sel(keypoints=name)
        x = kp.sel(space="x").values.astype(np.float64)
        y = kp.sel(space="y").values.astype(np.float64)
        if has_conf and name in list(conf_da.coords["keypoints"].values):
            c = conf_da.sel(keypoints=name).values.astype(np.float64)
        else:
            c = np.where(np.isnan(x), 0.0, 1.0)
        # NaN positions get zero weight
        c = np.where(np.isnan(x) | np.isnan(y), 0.0, c)
        xs.append(x)
        ys.append(y)
        confs.append(c)

    if not xs:
        n = len(pos.coords["time"])
        return np.full(n, np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32)

    xs_arr = np.stack(xs, axis=1)       # (N, K)
    ys_arr = np.stack(ys, axis=1)       # (N, K)
    confs_arr = np.stack(confs, axis=1)  # (N, K)

    w_sum = confs_arr.sum(axis=1, keepdims=True)  # (N, 1)
    w_sum = np.where(w_sum == 0, np.nan, w_sum)

    # Replace NaN positions with 0 for weighted sum (weight is already 0)
    xs_safe = np.nan_to_num(xs_arr, nan=0.0)
    ys_safe = np.nan_to_num(ys_arr, nan=0.0)

    cx = (confs_arr * xs_safe).sum(axis=1) / w_sum.squeeze()
    cy = (confs_arr * ys_safe).sum(axis=1) / w_sum.squeeze()

    return cx.astype(np.float32), cy.astype(np.float32)


def compute_head_body_angle(ds: "xr.Dataset") -> np.ndarray:
    """Compute the angle between head direction and body direction (degrees).

    Head direction: from implant/neck to nose (or ear perpendicular).
    Body direction: from tail_base to mid_back.
    Head-body angle: signed difference, positive = head turned left.

    This measures postural state — whether the mouse is looking straight
    ahead (0°), turning its head left (+) or right (-) relative to its
    body axis. Useful for detecting head scanning at maze junctions.

    Args:
        ds: movement Dataset (filtered + interpolated).

    Returns:
        (N,) float32 — head-body angle in degrees, range (-180, 180].
        NaN where either direction is unavailable.
    """
    pos = ds.position.isel(individuals=0)
    available_kps = list(pos.coords["keypoints"].values)

    # Head direction (use ear perpendicular — always available)
    ear_left = _get_keypoint_xy(pos, _EAR_LEFT)
    ear_right = _get_keypoint_xy(pos, _EAR_RIGHT)
    if ear_left is None or ear_right is None:
        n = len(pos.coords["time"])
        return np.full(n, np.nan, dtype=np.float32)

    hd = _ear_perpendicular_angle(
        ear_left[0], ear_left[1], ear_right[0], ear_right[1]
    )

    # Body direction: tail_base → mid_back
    tail = _get_keypoint_xy(pos, "tail_base")
    back = _get_keypoint_xy(pos, "mid_back")
    if tail is None or back is None:
        n = len(pos.coords["time"])
        return np.full(n, np.nan, dtype=np.float32)

    # Use same convention as ear perpendicular (atan2(dx, dy) + 180)
    body_dir = 180.0 + np.degrees(
        np.arctan2(back[0] - tail[0], back[1] - tail[1])
    )

    # Signed angular difference
    diff = hd - body_dir
    # Wrap to (-180, 180]
    diff = (diff + 180.0) % 360.0 - 180.0

    return diff.astype(np.float32)


def compute_neck_angle(ds: "xr.Dataset") -> np.ndarray:
    """Compute the neck flexion angle (degrees).

    Angle at the neck keypoint between the head axis (neck→implant or
    neck→ear_midpoint) and the body axis (neck→mid_back). 180° = straight,
    <180° = head flexed down/forward, >180° = head extended up/back.

    Args:
        ds: movement Dataset (filtered + interpolated).

    Returns:
        (N,) float32 — neck angle in degrees. NaN where keypoints missing.
    """
    pos = ds.position.isel(individuals=0)

    neck = _get_keypoint_xy(pos, _NECK)
    back = _get_keypoint_xy(pos, "mid_back")
    if neck is None or back is None:
        n = len(pos.coords["time"])
        return np.full(n, np.nan, dtype=np.float32)

    # Head end: prefer implant, fallback to ear midpoint
    implant = _get_keypoint_xy(pos, _IMPLANT)
    ear_left = _get_keypoint_xy(pos, _EAR_LEFT)
    ear_right = _get_keypoint_xy(pos, _EAR_RIGHT)

    if implant is not None:
        head_x, head_y = implant
    elif ear_left is not None and ear_right is not None:
        head_x = (ear_left[0] + ear_right[0]) / 2.0
        head_y = (ear_left[1] + ear_right[1]) / 2.0
    else:
        n = len(pos.coords["time"])
        return np.full(n, np.nan, dtype=np.float32)

    # Vector from neck to head
    dx_head = head_x - neck[0]
    dy_head = head_y - neck[1]
    # Vector from neck to back
    dx_back = back[0] - neck[0]
    dy_back = back[1] - neck[1]

    # Angle between vectors via atan2 of cross and dot products
    cross = dx_head * dy_back - dy_head * dx_back
    dot = dx_head * dx_back + dy_head * dy_back
    angle = np.degrees(np.arctan2(cross, dot))

    # Convert to 0-360 range where 180 = straight
    angle = 180.0 - angle

    return angle.astype(np.float32)


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
    """Clip maze-unit positions to the q-rose maze boundary polygon.

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
        if not (np.isfinite(x_maze[i]) and np.isfinite(y_maze[i])):
            # Skip NaN and inf — shapely's nearest_points raises GEOSException
            # on non-finite coordinates. Preserve as NaN in output.
            x_out[i] = np.nan
            y_out[i] = np.nan
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
    from movement.io import load_poses

    if tracker not in _TRACKER_MAP:
        raise ValueError(f"Unknown tracker '{tracker}'. Known trackers: {list(_TRACKER_MAP)}")
    source_software = _TRACKER_MAP[tracker]
    # movement ≥0.1.0 renamed 'file' → 'file_path'. Support both.
    import inspect
    sig = inspect.signature(load_poses.from_file)
    if "file_path" in sig.parameters:
        return load_poses.from_file(file_path=pose_path, source_software=source_software)
    return load_poses.from_file(file=pose_path, source_software=source_software)


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


def filter_by_keypoint_quantile(
    ds: xr.Dataset,
    quantile: float = 0.25,
    floor: float = 0.0,
) -> tuple[xr.Dataset, dict[str, float]]:
    """Set position to NaN below a per-keypoint confidence quantile threshold.

    For each keypoint, the threshold is the per-keypoint *quantile* of its
    confidence distribution within this session. Frames whose confidence is
    below their keypoint's threshold are NaN'd. This is the recommended
    filter for DLC 3.x PyTorch outputs, whose confidence values are
    uncalibrated and sit in a low absolute range — a fixed scalar threshold
    either drops nothing or drops everything, while a per-keypoint quantile
    consistently drops the worst-tracked frames per keypoint.

    The returned dict maps keypoint name → applied threshold so callers can
    log it (and unit-test it). NaN-only keypoints get threshold ``floor``
    and are passed through unchanged.

    Args:
        ds: movement Dataset with ``position`` and ``confidence`` data
            variables.
        quantile: Quantile in [0, 1] used as the per-keypoint cutoff.
            Default 0.25 (drops the bottom quartile of each keypoint).
        floor: Minimum threshold value. If a keypoint's quantile lies below
            this value, ``floor`` is used instead. Useful when even the
            quantile is implausibly low (default 0.0 — no floor).

    Returns:
        Tuple of (filtered Dataset, dict mapping keypoint → applied
        threshold).

    Notes:
        Implemented with xarray's ``DataArray.where`` so it broadcasts
        correctly across the (time, keypoints) dimensions. The xarray-based
        path matches movement's own ``filter_by_confidence`` semantics
        (data is set to NaN where confidence < threshold) but supports a
        per-keypoint threshold which the upstream API does not.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")

    conf = ds.confidence
    # Compute one threshold per keypoint over (time, individuals).
    reduce_dims = [d for d in conf.dims if d != "keypoints"]
    thresholds = conf.quantile(quantile, dim=reduce_dims, skipna=True)
    thresholds = thresholds.where(~thresholds.isnull(), other=floor)
    if floor > 0.0:
        thresholds = thresholds.where(thresholds >= floor, other=floor)
    # Drop quantile coord introduced by .quantile() if present.
    if "quantile" in thresholds.coords:
        thresholds = thresholds.drop_vars("quantile")

    keep = conf >= thresholds
    filtered_pos = ds.position.where(keep)

    threshold_map: dict[str, float] = {}
    for kp_name in conf.coords["keypoints"].values.tolist():
        threshold_map[str(kp_name)] = float(
            thresholds.sel(keypoints=kp_name).values
        )

    return ds.assign(position=filtered_pos), threshold_map


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


def _get_keypoint_xy(
    pos: "xr.DataArray", name: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract (x, y) arrays for a keypoint, or None if not in the dataset."""
    kps = list(pos.coords["keypoints"].values)
    if name not in kps:
        return None
    kp = pos.sel(keypoints=name)
    return kp.sel(space="x").values, kp.sel(space="y").values


def _get_keypoint_conf(
    ds: "xr.Dataset", name: str
) -> np.ndarray | None:
    """Extract confidence array for a keypoint, or None if unavailable."""
    if "confidence" not in ds:
        return None
    conf = ds.confidence.isel(individuals=0)
    kps = list(conf.coords["keypoints"].values)
    if name not in kps:
        return None
    return conf.sel(keypoints=name).values.astype(np.float64)


def compute_head_direction(ds: xr.Dataset) -> np.ndarray:
    """Compute unwrapped head direction by fusing available head keypoints.

    Uses confidence-weighted circular mean of up to five independent estimates:
      1. Ear perpendicular (left_ear, right_ear) — primary.
      2. Head midpoint→nose axis (head_midpoint → nose_tip).
      3. Neck→nose axis (neck → nose_tip).
      4. Ear midpoint→nose (mean(ears) → nose_tip).
      5. Neck→head midpoint axis (neck → head_midpoint).

    Each estimate is weighted by the minimum DLC confidence of its
    constituent keypoints. Falls back to ears-only if other keypoints
    are absent (backwards compatible with legacy 5-bodypart pose data).

    Args:
        ds: movement Dataset (filtered + interpolated).

    Returns:
        (N,) float32 — HD in degrees, unwrapped, referenced to camera frame.
    """
    pos = ds.position.isel(individuals=0)
    available_kps = list(pos.coords["keypoints"].values)

    ear_left = _get_keypoint_xy(pos, _EAR_LEFT)
    ear_right = _get_keypoint_xy(pos, _EAR_RIGHT)
    if ear_left is None or ear_right is None:
        raise ValueError(
            f"Ears required for HD. Available keypoints: {available_kps}"
        )

    nose = _get_keypoint_xy(pos, _NOSE)
    implant = _get_keypoint_xy(pos, _IMPLANT)
    neck = _get_keypoint_xy(pos, _NECK)

    # Get confidence arrays for weighting
    conf_el = _get_keypoint_conf(ds, _EAR_LEFT)
    conf_er = _get_keypoint_conf(ds, _EAR_RIGHT)
    conf_nose = _get_keypoint_conf(ds, _NOSE)
    conf_implant = _get_keypoint_conf(ds, _IMPLANT)
    conf_neck = _get_keypoint_conf(ds, _NECK)

    # Log which estimates will be used
    methods = ["ear_perpendicular"]
    if nose is not None and implant is not None:
        methods.append("implant_nose")
    if nose is not None and neck is not None:
        methods.append("neck_nose")
    if nose is not None:
        methods.append("earmid_nose")
    if implant is not None and neck is not None:
        methods.append("neck_implant")
    has_conf = conf_el is not None
    _log.info(
        "HD fusion: %d estimates %s, confidence-weighted=%s",
        len(methods), methods, has_conf,
    )

    fused_wrapped = _fused_hd_wrapped(
        ear_left_x=ear_left[0],
        ear_left_y=ear_left[1],
        ear_right_x=ear_right[0],
        ear_right_y=ear_right[1],
        nose_x=nose[0] if nose else None,
        nose_y=nose[1] if nose else None,
        implant_x=implant[0] if implant else None,
        implant_y=implant[1] if implant else None,
        neck_x=neck[0] if neck else None,
        neck_y=neck[1] if neck else None,
        conf_ear_left=conf_el,
        conf_ear_right=conf_er,
        conf_nose=conf_nose,
        conf_implant=conf_implant,
        conf_neck=conf_neck,
    )

    return _unwrap_and_smooth(fused_wrapped)


def compute_position_mm(
    ds: xr.Dataset,
    scale_mm_per_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute body centroid position in mm.

    Centroid is the mean of mid_back, mouse_center, tail_base keypoints.

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
    """Map mm positions to q-rose maze coordinate units (0–7 × 0–5).

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
    # Convert corners to mm.
    corners_mm = maze_corners_px * scale_mm_per_px  # (4, 2)
    origin = corners_mm[0]
    # Basis vectors along the maze edges (TL→TR for x, TL→BL for y).
    # Using vectors instead of x/y differences makes the transform correct
    # for any rotation of the corners — required because corners may be
    # rotated by orientation_deg before reaching this function. Axis-aligned
    # corners give the same result as the previous scalar-diff form.
    dx_vec = corners_mm[1] - origin
    dy_vec = corners_mm[3] - origin
    dx_sq = float(dx_vec @ dx_vec)
    dy_sq = float(dy_vec @ dy_vec)

    delta_x = x_mm.astype(np.float64) - float(origin[0])
    delta_y = y_mm.astype(np.float64) - float(origin[1])
    x_maze = (delta_x * float(dx_vec[0]) + delta_y * float(dx_vec[1])) / dx_sq * 7.0
    y_maze = (delta_x * float(dy_vec[0]) + delta_y * float(dy_vec[1])) / dy_sq * 5.0
    return _clip_to_maze_polygon(x_maze.astype(np.float32), y_maze.astype(np.float32))


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


def _windowed_gradient(
    signal: np.ndarray,
    frame_times: np.ndarray,
    window_s: float = 0.2,
) -> np.ndarray:
    """Compute gradient using windowed linear regression (matching legacy pipeline).

    For each frame, fits a linear regression over a symmetric window of ±window_s/2
    seconds. The slope of the fit gives the smoothed derivative. Falls back to
    np.gradient for edge frames where the window extends beyond the data.

    Args:
        signal: (N,) input signal (e.g., unwrapped HD in degrees, position in mm).
        frame_times: (N,) timestamps in seconds.
        window_s: Window duration in seconds (default 0.2, matching legacy).

    Returns:
        (N,) float64 — windowed gradient (units of signal per second).
    """
    n = len(signal)
    fps = n / (frame_times[-1] - frame_times[0]) if n > 1 else 30.0
    half_win = max(1, int(round(window_s * fps / 2)))
    # Make window odd for symmetry
    win = 2 * half_win + 1

    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half_win)
        hi = min(n, i + half_win + 1)
        t_local = frame_times[lo:hi]
        s_local = signal[lo:hi]
        valid = np.isfinite(s_local) & np.isfinite(t_local)
        if valid.sum() >= 2:
            # Linear regression slope = cov(t, s) / var(t)
            t_v = t_local[valid]
            s_v = s_local[valid]
            t_mean = t_v.mean()
            s_mean = s_v.mean()
            dt = t_v - t_mean
            denom = (dt * dt).sum()
            if denom > 0:
                result[i] = ((dt * (s_v - s_mean)).sum()) / denom

    # Fill any remaining NaN at edges with simple central difference.
    # Only use np.gradient where signal is finite (it propagates NaN).
    nan_mask = np.isnan(result)
    if nan_mask.any():
        finite = np.isfinite(signal)
        if finite.all():
            simple = np.gradient(signal, frame_times)
            result[nan_mask] = simple[nan_mask]
        # If signal has NaN, leave those gradient values as NaN

    return result


def _windowed_speed(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    frame_times: np.ndarray,
    window_s: float = 0.2,
) -> np.ndarray:
    """Compute speed (cm/s) using windowed linear regression on position.

    Matches the legacy pipeline's SPEED_FILT_GRAD: fits a line to x(t) and y(t)
    over a sliding window, then computes speed as sqrt(dx_dt^2 + dy_dt^2) / 10.

    Args:
        x_mm: (N,) x position in mm.
        y_mm: (N,) y position in mm.
        frame_times: (N,) timestamps in seconds.
        window_s: Window duration in seconds (default 0.2).

    Returns:
        (N,) float64 — speed in cm/s.
    """
    dx_dt = _windowed_gradient(x_mm, frame_times, window_s)
    dy_dt = _windowed_gradient(y_mm, frame_times, window_s)
    return np.sqrt(dx_dt**2 + dy_dt**2) / 10.0  # mm/s → cm/s


def compute_multipoint_speed(
    ds: "xr.Dataset",
    keypoints: list[str],
    frame_times: np.ndarray,
    scale_mm_per_px: float,
    window_s: float = 0.2,
) -> np.ndarray:
    """Compute speed (cm/s) as confidence-weighted mean of per-keypoint speeds.

    For each keypoint, computes speed independently using windowed linear
    regression, then combines via confidence-weighted mean. This is more
    robust than computing speed from a single centroid because:
    - A noisy keypoint with low confidence contributes less
    - Multiple independent speed estimates average out tracking jitter
    - NaN keypoints are automatically excluded

    Args:
        ds: movement Dataset (filtered + interpolated) with confidence.
        keypoints: List of keypoint names to use.
        frame_times: (N,) timestamps in seconds.
        scale_mm_per_px: Pixel → mm conversion factor.
        window_s: Speed smoothing window in seconds.

    Returns:
        (N,) float32 — confidence-weighted speed in cm/s.
    """
    pos = ds.position.isel(individuals=0)
    available_kps = list(pos.coords["keypoints"].values)
    has_conf = "confidence" in ds
    conf_da = ds.confidence.isel(individuals=0) if has_conf else None

    n = len(frame_times)
    speed_sum = np.zeros(n, dtype=np.float64)
    weight_sum = np.zeros(n, dtype=np.float64)

    for kp_name in keypoints:
        if kp_name not in available_kps:
            continue
        kp = pos.sel(keypoints=kp_name)
        x_mm = scale_mm_per_px * kp.sel(space="x").values.astype(np.float64)
        y_mm = scale_mm_per_px * kp.sel(space="y").values.astype(np.float64)

        spd = _windowed_speed(x_mm, y_mm, frame_times, window_s)

        if has_conf and kp_name in list(conf_da.coords["keypoints"].values):
            w = conf_da.sel(keypoints=kp_name).values.astype(np.float64)
        else:
            w = np.where(np.isnan(x_mm), 0.0, 1.0)

        valid = np.isfinite(spd) & (w > 0)
        speed_sum[valid] += w[valid] * spd[valid]
        weight_sum[valid] += w[valid]

    result = np.full(n, np.nan, dtype=np.float64)
    has_weight = weight_sum > 0
    result[has_weight] = speed_sum[has_weight] / weight_sum[has_weight]
    return result.astype(np.float32)


def compute_locomotion_speed(
    ds: "xr.Dataset",
    frame_times: np.ndarray,
    scale_mm_per_px: float,
    window_s: float = 0.2,
) -> np.ndarray:
    """Compute locomotion speed from body keypoints, confidence-weighted.

    Uses mid_back, mouse_center, tail_base — body keypoints that reflect
    whole-body translation rather than head movements.

    Args:
        ds: movement Dataset.
        frame_times: (N,) timestamps in seconds.
        scale_mm_per_px: Pixel → mm scale.
        window_s: Smoothing window in seconds.

    Returns:
        (N,) float32 — locomotion speed in cm/s.
    """
    return compute_multipoint_speed(
        ds, list(_BODY_KEYPOINTS), frame_times, scale_mm_per_px, window_s,
    )


def compute_head_speed(
    ds: "xr.Dataset",
    frame_times: np.ndarray,
    scale_mm_per_px: float,
    window_s: float = 0.2,
) -> np.ndarray:
    """Compute head translation speed from head keypoints, confidence-weighted.

    Uses nose_tip, left_ear, right_ear, head_midpoint, neck — head
    keypoints that capture head translation independent of body movement.
    Useful for distinguishing head bobbing/scanning from locomotion.

    Args:
        ds: movement Dataset.
        frame_times: (N,) timestamps in seconds.
        scale_mm_per_px: Pixel → mm scale.
        window_s: Smoothing window in seconds.

    Returns:
        (N,) float32 — head translation speed in cm/s.
    """
    head_kps = [_NOSE, _EAR_LEFT, _EAR_RIGHT, _IMPLANT, _NECK]
    return compute_multipoint_speed(
        ds, head_kps, frame_times, scale_mm_per_px, window_s,
    )


# ---------------------------------------------------------------------------
# New primary kinematics functions (4-estimate HD, head/body position, AHV)
# ---------------------------------------------------------------------------


def compute_hd_multi(ds: xr.Dataset, scale_mm_per_px: float) -> dict[str, np.ndarray]:
    """Compute HD from 4 independent estimates via confidence-weighted circular mean.

    Estimates:
      1. **hd_ears**: perpendicular to left_ear → right_ear vector.
      2. **hd_nose_head**: direction from head_midpoint to nose_tip.
      3. **hd_nose_neck**: direction from neck to nose_tip.
      4. **hd_head_neck**: direction from neck to head_midpoint.

    Each estimate is weighted by the mean DLC confidence of the keypoints
    used in that estimate (not the minimum, so a single low-confidence keypoint
    does not dominate). The fused HD is the confidence-weighted circular mean:

        HD = atan2(sum(w * sin(θ)), sum(w * cos(θ)))

    NaN estimates (keypoints absent or below confidence threshold) are skipped
    and weights are renormalised automatically. If no estimate is available at a
    frame, the fused HD is NaN. The fused HD is then unwrapped (so AHV can be
    computed via gradient) and the wrapped individual estimates are also returned
    for QC.

    The ``scale_mm_per_px`` argument is accepted for API symmetry with
    ``compute_head_position`` and ``compute_body_position`` but is not used
    internally (HD is a pure angle, independent of scale).

    Parameters
    ----------
    ds : xr.Dataset
        movement Dataset (filtered + interpolated). Must contain ``position``
        with dims (time, space, keypoints, individuals) and optionally
        ``confidence`` with dims (time, keypoints, individuals).
    scale_mm_per_px : float
        Pixel → mm scale factor (not used for HD; included for API consistency).

    Returns
    -------
    dict with keys:
        hd_deg : (N,) float32
            Fused, unwrapped HD in degrees.
        hd_ears : (N,) float32
            Wrapped HD from ear perpendicular method, degrees.
        hd_nose_head : (N,) float32
            Wrapped HD from nose→head_midpoint axis, degrees. NaN if keypoints absent.
        hd_nose_neck : (N,) float32
            Wrapped HD from nose→neck axis, degrees. NaN if keypoints absent.
        hd_head_neck : (N,) float32
            Wrapped HD from head_midpoint→neck axis, degrees. NaN if keypoints absent.
        hd_confidence : (N,) float32
            Sum of weights used in the circular mean (proxy for fusion quality).
    """
    pos = ds.position.isel(individuals=0)
    available_kps = list(pos.coords["keypoints"].values)

    def _mean_conf(*names: str) -> np.ndarray:
        """Per-frame mean confidence across the named keypoints.

        Returns an array of 1.0 where confidence is unavailable. Returns 0.0
        for any frame where any constituent keypoint position is NaN (since
        that estimate is unusable regardless of reported confidence).
        """
        arrays = []
        for name in names:
            c = _get_keypoint_conf(ds, name)
            if c is None:
                # No confidence data → treat as full confidence
                kp_xy = _get_keypoint_xy(pos, name)
                if kp_xy is not None:
                    c = np.where(
                        np.isnan(kp_xy[0]) | np.isnan(kp_xy[1]), 0.0, 1.0
                    ).astype(np.float64)
                else:
                    c = np.zeros(pos.sizes["time"], dtype=np.float64)
            arrays.append(c)
        if not arrays:
            return np.zeros(pos.sizes["time"], dtype=np.float64)
        return np.mean(np.stack(arrays, axis=0), axis=0)

    n = pos.sizes["time"]

    # --- Estimate 1: ear perpendicular (required) ---
    ear_left = _get_keypoint_xy(pos, _EAR_LEFT)
    ear_right = _get_keypoint_xy(pos, _EAR_RIGHT)
    if ear_left is None or ear_right is None:
        raise ValueError(
            f"left_ear and right_ear required for HD. Available: {available_kps}"
        )
    hd_ears = _ear_perpendicular_angle(
        ear_left[0], ear_left[1], ear_right[0], ear_right[1]
    ).astype(np.float32)
    w_ears = _mean_conf(_EAR_LEFT, _EAR_RIGHT)
    # Zero weight where angle is NaN (both ears missing)
    w_ears = np.where(np.isnan(hd_ears), 0.0, w_ears)

    # --- Estimate 2: head_midpoint → nose_tip ---
    nose = _get_keypoint_xy(pos, _NOSE)
    implant = _get_keypoint_xy(pos, _IMPLANT)
    if nose is not None and implant is not None:
        hd_nose_head = _vector_angle_deg(
            implant[0], implant[1], nose[0], nose[1]
        ).astype(np.float32)
        w_nose_head = _mean_conf(_NOSE, _IMPLANT)
        w_nose_head = np.where(np.isnan(hd_nose_head), 0.0, w_nose_head)
    else:
        hd_nose_head = np.full(n, np.nan, dtype=np.float32)
        w_nose_head = np.zeros(n, dtype=np.float64)

    # --- Estimate 3: neck → nose_tip ---
    neck = _get_keypoint_xy(pos, _NECK)
    if nose is not None and neck is not None:
        hd_nose_neck = _vector_angle_deg(
            neck[0], neck[1], nose[0], nose[1]
        ).astype(np.float32)
        w_nose_neck = _mean_conf(_NOSE, _NECK)
        w_nose_neck = np.where(np.isnan(hd_nose_neck), 0.0, w_nose_neck)
    else:
        hd_nose_neck = np.full(n, np.nan, dtype=np.float32)
        w_nose_neck = np.zeros(n, dtype=np.float64)

    # --- Estimate 4: neck → head_midpoint ---
    if implant is not None and neck is not None:
        hd_head_neck = _vector_angle_deg(
            neck[0], neck[1], implant[0], implant[1]
        ).astype(np.float32)
        w_head_neck = _mean_conf(_IMPLANT, _NECK)
        w_head_neck = np.where(np.isnan(hd_head_neck), 0.0, w_head_neck)
    else:
        hd_head_neck = np.full(n, np.nan, dtype=np.float32)
        w_head_neck = np.zeros(n, dtype=np.float64)

    # --- Confidence-weighted circular mean ---
    estimates = [hd_ears, hd_nose_head, hd_nose_neck, hd_head_neck]
    weights = [w_ears, w_nose_head, w_nose_neck, w_head_neck]

    sin_sum = np.zeros(n, dtype=np.float64)
    cos_sum = np.zeros(n, dtype=np.float64)
    w_sum = np.zeros(n, dtype=np.float64)

    for est, w in zip(estimates, weights, strict=True):
        valid = ~np.isnan(est) & (w > 0)
        rad = np.deg2rad(est.astype(np.float64))
        sin_sum[valid] += w[valid] * np.sin(rad[valid])
        cos_sum[valid] += w[valid] * np.cos(rad[valid])
        w_sum[valid] += w[valid]

    fused_wrapped = np.full(n, np.nan, dtype=np.float64)
    has_any = w_sum > 0
    fused_wrapped[has_any] = (
        np.degrees(np.arctan2(sin_sum[has_any], cos_sum[has_any])) % 360.0
    )

    # Unwrap fused HD so AHV can be computed via gradient
    hd_fused = _unwrap_and_smooth(fused_wrapped, median_filter_win=3)

    n_methods = sum(1 for w in weights if np.any(w > 0))
    _log.info(
        "HD fusion: %d/4 estimates available, confidence-weighted circular mean",
        n_methods,
    )

    return {
        "hd_deg": hd_fused,
        "hd_ears": hd_ears,
        "hd_nose_head": hd_nose_head,
        "hd_nose_neck": hd_nose_neck,
        "hd_head_neck": hd_head_neck,
        "hd_confidence": w_sum.astype(np.float32),
    }


def compute_head_position(
    ds: xr.Dataset, scale_mm_per_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Confidence-weighted head position from 3 independent estimates.

    Estimates:
      1. head_midpoint (x, y) — weight = head_midpoint confidence.
      2. Centroid of (nose_tip, left_ear, right_ear) — weight = mean of their confidences.
      3. Midpoint of (nose_tip, neck) — weight = mean of their confidences.

    NaN estimates (keypoints absent or below threshold) are skipped and weights
    are renormalised automatically. Returns NaN where no estimate is available.

    Parameters
    ----------
    ds : xr.Dataset
        movement Dataset (filtered + interpolated) with confidence.
    scale_mm_per_px : float
        Pixel → mm scale factor.

    Returns
    -------
    tuple of (x_mm, y_mm), each (N,) float32 — head position in mm.
    """
    pos = ds.position.isel(individuals=0)
    n = pos.sizes["time"]

    def _kp_xy_mm(name: str) -> tuple[np.ndarray, np.ndarray] | None:
        xy = _get_keypoint_xy(pos, name)
        if xy is None:
            return None
        return (
            (xy[0] * scale_mm_per_px).astype(np.float64),
            (xy[1] * scale_mm_per_px).astype(np.float64),
        )

    def _kp_conf(name: str) -> np.ndarray:
        c = _get_keypoint_conf(ds, name)
        if c is None:
            xy = _get_keypoint_xy(pos, name)
            if xy is None:
                return np.zeros(n, dtype=np.float64)
            return np.where(np.isnan(xy[0]) | np.isnan(xy[1]), 0.0, 1.0)
        return c.astype(np.float64)

    x_sum = np.zeros(n, dtype=np.float64)
    y_sum = np.zeros(n, dtype=np.float64)
    w_sum = np.zeros(n, dtype=np.float64)

    # Estimate 1: head_midpoint
    implant_mm = _kp_xy_mm(_IMPLANT)
    if implant_mm is not None:
        w1 = _kp_conf(_IMPLANT)
        valid = ~np.isnan(implant_mm[0]) & ~np.isnan(implant_mm[1]) & (w1 > 0)
        x_sum[valid] += w1[valid] * implant_mm[0][valid]
        y_sum[valid] += w1[valid] * implant_mm[1][valid]
        w_sum[valid] += w1[valid]

    # Estimate 2: centroid of nose_tip, left_ear, right_ear
    nose_mm = _kp_xy_mm(_NOSE)
    ear_l_mm = _kp_xy_mm(_EAR_LEFT)
    ear_r_mm = _kp_xy_mm(_EAR_RIGHT)
    if nose_mm is not None and ear_l_mm is not None and ear_r_mm is not None:
        cx2 = (nose_mm[0] + ear_l_mm[0] + ear_r_mm[0]) / 3.0
        cy2 = (nose_mm[1] + ear_l_mm[1] + ear_r_mm[1]) / 3.0
        w2 = (
            _kp_conf(_NOSE) + _kp_conf(_EAR_LEFT) + _kp_conf(_EAR_RIGHT)
        ) / 3.0
        valid = ~np.isnan(cx2) & ~np.isnan(cy2) & (w2 > 0)
        x_sum[valid] += w2[valid] * cx2[valid]
        y_sum[valid] += w2[valid] * cy2[valid]
        w_sum[valid] += w2[valid]

    # Estimate 3: midpoint of nose_tip and neck
    neck_mm = _kp_xy_mm(_NECK)
    if nose_mm is not None and neck_mm is not None:
        cx3 = (nose_mm[0] + neck_mm[0]) / 2.0
        cy3 = (nose_mm[1] + neck_mm[1]) / 2.0
        w3 = (_kp_conf(_NOSE) + _kp_conf(_NECK)) / 2.0
        valid = ~np.isnan(cx3) & ~np.isnan(cy3) & (w3 > 0)
        x_sum[valid] += w3[valid] * cx3[valid]
        y_sum[valid] += w3[valid] * cy3[valid]
        w_sum[valid] += w3[valid]

    x_mm_out = np.full(n, np.nan, dtype=np.float64)
    y_mm_out = np.full(n, np.nan, dtype=np.float64)
    has_w = w_sum > 0
    x_mm_out[has_w] = x_sum[has_w] / w_sum[has_w]
    y_mm_out[has_w] = y_sum[has_w] / w_sum[has_w]

    return x_mm_out.astype(np.float32), y_mm_out.astype(np.float32)


def compute_body_position_unweighted(
    ds: xr.Dataset, scale_mm_per_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Simple unweighted mean of body keypoints — no confidence weighting.

    Used to produce the "raw" position fields displayed by the maze
    animation page when the user wants the unfiltered DLC pose. Unlike
    ``compute_body_position`` this does not weight by DLC confidence
    (so low-confidence keypoints contribute equally) and does not skip
    them. NaN keypoints are still excluded from the mean (via skipna)
    so a single missing keypoint does not propagate to the centroid.

    Parameters
    ----------
    ds : xr.Dataset
        movement Dataset, typically from raw (unfiltered) DLC output.
    scale_mm_per_px : float
        Pixel → mm scale factor.

    Returns
    -------
    tuple of (x_mm, y_mm), each (N,) float32 — unweighted body centroid.
    """
    pos = ds.position.isel(individuals=0)
    available = list(pos.coords["keypoints"].values)
    body_kps = [k for k in _BODY_KEYPOINTS if k in available]
    n = pos.sizes["time"]
    if not body_kps:
        return np.full(n, np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32)
    sub = pos.sel(keypoints=body_kps)
    x_px = sub.sel(space="x").mean(dim="keypoints", skipna=True).values
    y_px = sub.sel(space="y").mean(dim="keypoints", skipna=True).values
    return (x_px * scale_mm_per_px).astype(np.float32), (y_px * scale_mm_per_px).astype(np.float32)


def compute_body_position(
    ds: xr.Dataset, scale_mm_per_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Confidence-weighted body centroid from mid_back, mouse_center, tail_base.

    These keypoints lie along the body axis. Because the head rotates
    independently, head-axis keypoints (ears, nose, implant) are excluded.

    Each keypoint is weighted by its DLC confidence. Keypoints with NaN
    position or zero confidence are excluded from the weighted average.
    Returns NaN where no body keypoint is available.

    Parameters
    ----------
    ds : xr.Dataset
        movement Dataset (filtered + interpolated) with confidence.
    scale_mm_per_px : float
        Pixel → mm scale factor.

    Returns
    -------
    tuple of (x_mm, y_mm), each (N,) float32 — body centroid in mm.
    """
    pos = ds.position.isel(individuals=0)
    n = pos.sizes["time"]
    available_kps = list(pos.coords["keypoints"].values)

    has_conf = "confidence" in ds
    conf_da = ds.confidence.isel(individuals=0) if has_conf else None

    x_sum = np.zeros(n, dtype=np.float64)
    y_sum = np.zeros(n, dtype=np.float64)
    w_sum = np.zeros(n, dtype=np.float64)

    for kp_name in _BODY_KEYPOINTS:
        if kp_name not in available_kps:
            continue
        kp = pos.sel(keypoints=kp_name)
        x_px = kp.sel(space="x").values.astype(np.float64)
        y_px = kp.sel(space="y").values.astype(np.float64)
        x_kp = x_px * scale_mm_per_px
        y_kp = y_px * scale_mm_per_px

        if has_conf and kp_name in list(conf_da.coords["keypoints"].values):
            w = conf_da.sel(keypoints=kp_name).values.astype(np.float64)
        else:
            w = np.where(np.isnan(x_px), 0.0, 1.0)

        # Zero weight where position is NaN
        valid = ~np.isnan(x_kp) & ~np.isnan(y_kp) & (w > 0)
        x_sum[valid] += w[valid] * x_kp[valid]
        y_sum[valid] += w[valid] * y_kp[valid]
        w_sum[valid] += w[valid]

    x_out = np.full(n, np.nan, dtype=np.float64)
    y_out = np.full(n, np.nan, dtype=np.float64)
    has_w = w_sum > 0
    x_out[has_w] = x_sum[has_w] / w_sum[has_w]
    y_out[has_w] = y_sum[has_w] / w_sum[has_w]

    return x_out.astype(np.float32), y_out.astype(np.float32)


def compute_speed_from_position(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    frame_times: np.ndarray,
    median_window: int = 3,
) -> np.ndarray:
    """Compute translational speed from position using median filter + gradient.

    Applies a 3-point (or ``median_window``-point) median filter to x and y
    independently, then computes the gradient with ``np.gradient`` using the
    actual frame timestamps as spacing. Speed is the Euclidean norm of (dx/dt,
    dy/dt) converted from mm/s to cm/s.

    For signals containing NaN (e.g., keypoints lost during tracking), NaN
    frames are linearly interpolated before filtering, the gradient is
    computed, and then NaN is restored at the original positions. This ensures
    that gradient values at the edges of NaN regions are not artificially
    inflated.

    Parameters
    ----------
    x_mm : np.ndarray
        (N,) x position in mm. May contain NaN.
    y_mm : np.ndarray
        (N,) y position in mm. May contain NaN.
    frame_times : np.ndarray
        (N,) frame timestamps in seconds. Must be strictly increasing.
    median_window : int
        Kernel size for the median filter (default 3).

    Returns
    -------
    (N,) float32 — speed in cm/s. NaN where position is NaN.
    """
    nan_mask = np.isnan(x_mm) | np.isnan(y_mm)

    x_filt = _median_filter_1d(x_mm, win=median_window)
    y_filt = _median_filter_1d(y_mm, win=median_window)

    dx_dt = np.gradient(x_filt, frame_times)
    dy_dt = np.gradient(y_filt, frame_times)

    speed = np.sqrt(dx_dt**2 + dy_dt**2) / 10.0  # mm/s → cm/s
    speed[nan_mask] = np.nan

    return speed.astype(np.float32)


def compute_ahv(
    hd_deg: np.ndarray,
    frame_times: np.ndarray,
    median_window: int = 3,
    savgol_window: int = 5,
    savgol_polyorder: int = 2,
) -> np.ndarray:
    """Compute angular head velocity from smoothed HD via gradient.

    Unwraps ``hd_deg`` to a continuous angle, applies a median filter
    (``median_window``) to remove single-frame impulses, then a
    Savitzky-Golay smoother (``savgol_window`` / ``savgol_polyorder``) to
    suppress high-frequency jitter, and finally computes ``np.gradient``
    using the actual frame timestamps. Returns deg/s.

    If ``hd_deg`` is already unwrapped (as returned by ``compute_hd_multi``),
    the unwrap step is a no-op for well-behaved data.

    Parameters
    ----------
    hd_deg : np.ndarray
        (N,) head direction in degrees. May be wrapped or unwrapped. May
        contain NaN.
    frame_times : np.ndarray
        (N,) frame timestamps in seconds. Must be strictly increasing.
    median_window : int
        Kernel size for the median filter (default 3).
    savgol_window : int
        Window length for Savitzky-Golay smoothing (default 5; ~165 ms at
        30 fps). ``<= 1`` disables.
    savgol_polyorder : int
        Polynomial order for the Savitzky-Golay smoother (default 2).

    Returns
    -------
    (N,) float32 — angular head velocity in deg/s. NaN where HD is NaN.
    """
    nan_mask = np.isnan(hd_deg)

    # Ensure the signal is unwrapped before gradient
    hd_unwrapped = _unwrap_and_smooth(
        hd_deg,
        median_filter_win=median_window,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
    )
    ahv = np.gradient(hd_unwrapped, frame_times)
    ahv[nan_mask] = np.nan

    return ahv.astype(np.float32)


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
    confidence_threshold: float | str = "quantile:0.25",
    gap_fill_frames: int = 5,
    speed_active_threshold: float = SPEED_ACTIVE_THRESHOLD,
    camera_center_px: tuple[float, float] | None = None,
    camera_height_mm: float = 700.0,
    dlc_model_name: str = "unknown",
    dlc_snapshot: str = "unknown",
    dlc_champion_id: str = "unknown",
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
        confidence_threshold: DLC/SLEAP confidence cutoff. Either:

            - ``float`` — a fixed scalar applied uniformly to every
              keypoint (movement's ``filter_by_confidence`` path).
            - ``"quantile:Q"`` — a string of the form ``"quantile:0.25"``,
              which uses the per-keypoint quantile filter
              (:func:`filter_by_keypoint_quantile`). This is the
              recommended setting for DLC 3.x PyTorch outputs whose
              absolute confidence values are uncalibrated.
        gap_fill_frames: Max frames to interpolate over.
        speed_active_threshold: cm/s threshold for active/inactive state.
        camera_center_px: Camera optical centre in cropped-frame pixels for
            perspective correction. If None, perspective correction is skipped.
        camera_height_mm: Camera-to-floor distance in mm (default 700).
        dlc_model_name: DLC project name or ``"superanimal_topviewmouse"`` for
            the SuperAnimal baseline. Stored as HDF5 attribute for provenance.
        dlc_snapshot: Snapshot iteration number as a string, or
            ``"superanimal"`` for the baseline model. Stored as HDF5 attribute.
        dlc_champion_id: Project-wide champion identifier from
            ``dlc-champion.json``. Stored as HDF5 attribute so the frontend
            can refuse to display sessions whose id does not match the
            current champion. ``"unknown"`` means the session was processed
            before the champion system existed (treated as stale).
    """
    from hm2p.io.hdf5 import read_h5, write_h5
    from hm2p.kinematics.perspective import (
        BODYPART_HEIGHTS_IMPLANT,
        correct_dataset_perspective,
    )

    # --- Load timestamps ---
    ts = read_h5(timestamps_h5)
    frame_times = ts["frame_times_camera"]  # (N,) float64

    # --- Pose processing ---
    ds = load_pose_dataset(pose_path, tracker)

    # DLC may have been run on subsampled video (e.g., 100fps → 30fps).
    # Subsample frame_times to match pose data length.
    n_pose = ds.sizes["time"]
    n_cam = len(frame_times)
    if n_cam != n_pose and n_cam > n_pose:
        indices = np.round(np.linspace(0, n_cam - 1, n_pose)).astype(int)
        frame_times = frame_times[indices]
    ds = apply_orientation_rotation(ds, orientation_deg)
    # Snapshot raw (unfiltered) pose for the "raw" frontend display.
    # Orientation rotation is geometric and applied for coordinate alignment;
    # confidence filtering / gap interpolation / median smoothing are skipped.
    ds_raw = ds.copy(deep=True)
    applied_thresholds: dict[str, float] | None = None
    if isinstance(confidence_threshold, str) and confidence_threshold.startswith(
        "quantile:"
    ):
        q = float(confidence_threshold.split(":", 1)[1])
        ds, applied_thresholds = filter_by_keypoint_quantile(ds, quantile=q)
        _log.info(
            "Per-keypoint quantile=%g confidence filter applied: %s",
            q,
            ", ".join(
                f"{k}={v:.3f}" for k, v in sorted(applied_thresholds.items())
            ),
        )
    else:
        ds = filter_low_confidence(ds, threshold=float(confidence_threshold))
    ds = interpolate_gaps(ds, max_gap_frames=gap_fill_frames)
    # 3 frames at 30fps ≈ 100ms (old pipeline: 5 frames at 100fps = 50ms)
    ds = median_filter_dataset(ds, window=3)

    # Perspective correction: project bodypart heights to ground plane.
    # Applied after filtering so corrected positions are based on clean data.
    if camera_center_px is not None:
        ds = correct_dataset_perspective(
            ds,
            camera_center_px=camera_center_px,
            camera_height_mm=camera_height_mm,
            bodypart_heights=BODYPART_HEIGHTS_IMPLANT,
        )
        ds_raw = correct_dataset_perspective(
            ds_raw,
            camera_center_px=camera_center_px,
            camera_height_mm=camera_height_mm,
            bodypart_heights=BODYPART_HEIGHTS_IMPLANT,
        )

    # --- Kinematics ---

    # Head direction: 4-estimate confidence-weighted circular mean.
    hd_result = compute_hd_multi(ds, scale_mm_per_px)
    hd_deg = hd_result["hd_deg"]  # (N,) float32, unwrapped

    # Head position: confidence-weighted average of 3 head-keypoint estimates.
    x_head_mm, y_head_mm = compute_head_position(ds, scale_mm_per_px)

    # Body position: confidence-weighted centroid of mid_back, mouse_center, tail_base.
    x_body_mm, y_body_mm = compute_body_position(ds, scale_mm_per_px)

    # Rotate maze corners by the same orientation angle as the keypoints,
    # using the SAME rotation centre (mean of all keypoint positions).
    if orientation_deg != 0.0:
        pos = ds.position.isel(individuals=0)
        all_x = pos.sel(space="x").values
        all_y = pos.sel(space="y").values
        cx = float(np.nanmean(all_x))
        cy = float(np.nanmean(all_y))
        rot_x, rot_y = _rotate_xy(
            maze_corners_px[:, 0].astype(float),
            maze_corners_px[:, 1].astype(float),
            orientation_deg, cx, cy,
        )
        maze_corners_px = np.column_stack([rot_x, rot_y])

    # Maze coords use body position (body centroid, not head).
    x_maze, y_maze = compute_maze_coords(
        x_body_mm, y_body_mm, maze_corners_px, scale_mm_per_px
    )

    # Per-bodypart positions in mm and maze coords for skeleton visualisation
    pos = ds.position.isel(individuals=0)
    available_kps = pos.coords["keypoints"].values.tolist()
    bp_positions_maze: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kp in available_kps:
        kp_x_px = pos.sel(keypoints=kp, space="x").values
        kp_y_px = pos.sel(keypoints=kp, space="y").values
        kp_x_mm = (kp_x_px * scale_mm_per_px).astype(np.float32)
        kp_y_mm = (kp_y_px * scale_mm_per_px).astype(np.float32)
        kp_x_mz, kp_y_mz = compute_maze_coords(
            kp_x_mm, kp_y_mm, maze_corners_px, scale_mm_per_px,
        )
        bp_positions_maze[kp] = (kp_x_mz, kp_y_mz)

    # Raw (unfiltered) body position and per-bodypart maze coords.
    # No confidence filter, no gap interpolation, no median smoothing,
    # no confidence weighting — the user-facing "raw" view of the pose.
    x_body_raw_mm, y_body_raw_mm = compute_body_position_unweighted(
        ds_raw, scale_mm_per_px
    )
    x_maze_raw, y_maze_raw = compute_maze_coords(
        x_body_raw_mm, y_body_raw_mm, maze_corners_px, scale_mm_per_px,
    )
    pos_raw = ds_raw.position.isel(individuals=0)
    bp_positions_maze_raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kp in available_kps:
        kp_x_px = pos_raw.sel(keypoints=kp, space="x").values
        kp_y_px = pos_raw.sel(keypoints=kp, space="y").values
        kp_x_mm = (kp_x_px * scale_mm_per_px).astype(np.float32)
        kp_y_mm = (kp_y_px * scale_mm_per_px).astype(np.float32)
        kp_x_mz, kp_y_mz = compute_maze_coords(
            kp_x_mm, kp_y_mm, maze_corners_px, scale_mm_per_px,
        )
        bp_positions_maze_raw[kp] = (kp_x_mz, kp_y_mz)

    # Head translational speed: 3-point median filter on head position + gradient.
    speed_head_cm_s = compute_speed_from_position(
        x_head_mm, y_head_mm, frame_times, median_window=3
    )

    # Locomotion speed: 3-point median filter on body position + gradient.
    speed_body_cm_s = compute_speed_from_position(
        x_body_mm, y_body_mm, frame_times, median_window=3
    )

    # Angular head velocity: unwrap → 3-point median filter → gradient.
    ahv_deg_s = compute_ahv(hd_deg, frame_times, median_window=3)

    # Active/inactive state — based on body (locomotion) speed.
    active = (speed_body_cm_s >= speed_active_threshold).astype(bool)

    # Light epoch and bad behaviour
    light_on_times = ts.get("light_on_times", np.empty(0, dtype=np.float64))
    light_off_times = ts.get("light_off_times", np.empty(0, dtype=np.float64))
    light_on = compute_light_on(frame_times, light_on_times, light_off_times)
    bad_behav = compute_bad_behav_mask(frame_times, bad_behav_intervals)

    # --- Write ---
    datasets: dict[str, np.ndarray] = {
        "frame_times": frame_times,
        # Fused HD and per-method estimates for QC
        "hd_deg": hd_deg,
        "hd_ears": hd_result["hd_ears"],
        "hd_nose_head": hd_result["hd_nose_head"],
        "hd_nose_neck": hd_result["hd_nose_neck"],
        "hd_head_neck": hd_result["hd_head_neck"],
        "hd_confidence": hd_result["hd_confidence"],
        # Head position
        "x_head_mm": x_head_mm,
        "y_head_mm": y_head_mm,
        "speed_head_cm_s": speed_head_cm_s,
        # Body position and locomotion speed
        "x_body_mm": x_body_mm,
        "y_body_mm": y_body_mm,
        "speed_body_cm_s": speed_body_cm_s,
        # Maze coordinates (body-based)
        "x_maze": x_maze,
        "y_maze": y_maze,
        # Raw maze coordinates (no confidence filter, no interpolation,
        # no median smoothing, unweighted body centroid). Used by the
        # frontend "Raw" display mode for honest unfiltered pose.
        "x_maze_raw": x_maze_raw,
        "y_maze_raw": y_maze_raw,
        # AHV
        "ahv_deg_s": ahv_deg_s,
        # Movement state and experimental flags
        "active": active,
        "light_on": light_on,
        "bad_behav": bad_behav,
        # Backward-compatible aliases (point to body position/speed)
        "x_mm": x_body_mm,
        "y_mm": y_body_mm,
        "speed_cm_s": speed_body_cm_s,
    }
    # Per-bodypart maze coordinates for skeleton visualisation
    for kp, (kp_x, kp_y) in bp_positions_maze.items():
        datasets[f"bp_{kp}_x_maze"] = kp_x
        datasets[f"bp_{kp}_y_maze"] = kp_y
    # Raw per-bodypart maze coordinates (parallel to bp_*_x_maze).
    for kp, (kp_x, kp_y) in bp_positions_maze_raw.items():
        datasets[f"bp_{kp}_x_maze_raw"] = kp_x
        datasets[f"bp_{kp}_y_maze_raw"] = kp_y
    attrs: dict[str, object] = {
        "session_id": session_id,
        "tracker": tracker,
        "confidence_threshold": str(confidence_threshold),
        "gap_fill_frames": gap_fill_frames,
        "scale_mm_per_px": scale_mm_per_px,
        "orientation_deg": orientation_deg,
        "speed_active_threshold_cm_s": speed_active_threshold,
        "dlc_model_name": dlc_model_name,
        "dlc_snapshot": dlc_snapshot,
        "dlc_champion_id": dlc_champion_id,
    }
    if applied_thresholds is not None:
        for kp_name, thr in applied_thresholds.items():
            attrs[f"confidence_threshold_{kp_name}"] = float(thr)
    write_h5(output_path, datasets, attrs=attrs)
