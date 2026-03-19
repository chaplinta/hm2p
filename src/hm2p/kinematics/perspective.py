"""Perspective correction for overhead camera keypoints.

Projects keypoint pixel positions from their true 3D height above the maze
floor down to the ground plane, removing parallax displacement caused by the
off-axis camera geometry.

The overhead Basler acA1300-200um camera is mounted ~700 mm above the maze
floor but is not centred over it. Keypoints at height *h* above the floor
appear displaced radially outward from the camera axis by a factor of
``H / (H - h)``. This module corrects that displacement per bodypart using
estimated heights (walking mouse with 2P miniscope).

References
----------
Standard pinhole camera geometry / collinearity equations. No specific paper
— this is textbook perspective projection (Hartley & Zisserman, *Multiple
View Geometry*, 2nd ed., Cambridge University Press, 2003).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

# ── Camera constants ─────────────────────────────────────────────────────
DEFAULT_CAMERA_HEIGHT_MM: float = 700.0
"""Distance from camera sensor to maze floor (mm)."""

UNCROPPED_WIDTH: int = 1280
"""Full sensor width (pixels) for the Basler acA1300-200um."""

UNCROPPED_HEIGHT: int = 1024
"""Full sensor height (pixels)."""

# ── Per-bodypart height estimates (mm above maze floor) ──────────────────
BODYPART_HEIGHTS: dict[str, float] = {
    "tail_base": 10.0,
    "mid_back": 20.0,
    "mouse_center": 20.0,
    "left_ear": 25.0,
    "right_ear": 25.0,
    "nose": 20.0,
}
"""Height estimates for a walking mouse without 2P implant."""

BODYPART_HEIGHTS_IMPLANT: dict[str, float] = {
    "tail_base": 10.0,
    "mid_back": 20.0,
    "mouse_center": 20.0,
    "left_ear": 40.0,
    "right_ear": 40.0,
    "nose": 35.0,
}
"""Height estimates for a mouse with 2P miniscope + headstage (~15 mm added
to head keypoints)."""


# ── Core functions ───────────────────────────────────────────────────────


def correct_perspective(
    x_px: np.ndarray,
    y_px: np.ndarray,
    camera_center_px: tuple[float, float],
    camera_height_mm: float = DEFAULT_CAMERA_HEIGHT_MM,
    bodypart_height_mm: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project keypoint positions from height *h* to the ground plane.

    For each pixel position ``(px, py)``, computes the true floor position
    by removing the radial displacement caused by the bodypart being at
    height ``h`` above the floor while the camera is at height ``H``.

    .. math::

        \\text{scale} = H / (H - h)
        px_{\\text{corrected}} = cx + (px - cx) / \\text{scale}
        py_{\\text{corrected}} = cy + (py - cy) / \\text{scale}

    Args:
        x_px: (...) x pixel coordinates (any shape, NaN-safe).
        y_px: (...) y pixel coordinates (same shape as *x_px*).
        camera_center_px: ``(cx, cy)`` optical centre in cropped-frame pixels.
        camera_height_mm: Camera-to-floor distance in mm (default 700).
        bodypart_height_mm: Bodypart height above the floor in mm.

    Returns:
        Tuple of ``(x_corrected, y_corrected)`` with the same shape and
        dtype as the inputs. Points at the camera centre are unchanged.

    Raises:
        ValueError: If *bodypart_height_mm* ≥ *camera_height_mm* (would
            place the bodypart at or above the camera).
    """
    if bodypart_height_mm >= camera_height_mm:
        raise ValueError(
            f"bodypart_height_mm ({bodypart_height_mm}) must be < "
            f"camera_height_mm ({camera_height_mm})"
        )
    if bodypart_height_mm == 0.0:
        return x_px.copy(), y_px.copy()

    cx, cy = camera_center_px
    scale = camera_height_mm / (camera_height_mm - bodypart_height_mm)

    x_corrected = cx + (x_px - cx) / scale
    y_corrected = cy + (y_px - cy) / scale
    return x_corrected, y_corrected


def correct_dataset_perspective(
    ds: xr.Dataset,
    camera_center_px: tuple[float, float],
    camera_height_mm: float = DEFAULT_CAMERA_HEIGHT_MM,
    bodypart_heights: dict[str, float] | None = None,
) -> xr.Dataset:
    """Apply per-bodypart perspective correction to a movement Dataset.

    Iterates over each keypoint in the Dataset's ``position`` DataArray,
    looks up its estimated height, and applies :func:`correct_perspective`.
    Keypoints not listed in *bodypart_heights* are left unchanged.

    Args:
        ds: movement xarray.Dataset with ``position`` DataArray having
            dimensions ``(time, space, keypoints, individuals)``.
        camera_center_px: ``(cx, cy)`` optical centre in cropped pixels.
        camera_height_mm: Camera-to-floor distance in mm.
        bodypart_heights: Mapping of keypoint name → height (mm). Defaults
            to :data:`BODYPART_HEIGHTS_IMPLANT` (all mice have 2P implant).

    Returns:
        A copy of *ds* with perspective-corrected positions.
    """
    if bodypart_heights is None:
        bodypart_heights = BODYPART_HEIGHTS_IMPLANT

    pos = ds.position.copy()
    keypoint_names = list(pos.coords["keypoints"].values)

    for kp in keypoint_names:
        h = bodypart_heights.get(kp, 0.0)
        if h == 0.0:
            continue

        kp_pos = pos.sel(keypoints=kp)  # (time, space, individuals)
        x = kp_pos.sel(space="x").values  # (time, individuals) or (time,)
        y = kp_pos.sel(space="y").values

        x_corr, y_corr = correct_perspective(
            x, y, camera_center_px, camera_height_mm, h
        )

        # Write back into the position array
        pos.loc[dict(keypoints=kp, space="x")] = x_corr
        pos.loc[dict(keypoints=kp, space="y")] = y_corr

    return ds.assign(position=pos)


def estimate_camera_center(
    crop_x: int,
    crop_y: int,
    uncrop_w: int = UNCROPPED_WIDTH,
    uncrop_h: int = UNCROPPED_HEIGHT,
) -> tuple[float, float]:
    """Compute the camera optical centre in cropped-frame coordinates.

    The optical centre of the full (uncropped) sensor is at
    ``(uncrop_w / 2, uncrop_h / 2)``. After cropping with top-left offset
    ``(crop_x, crop_y)``, the centre moves to::

        cx = uncrop_w / 2 - crop_x
        cy = uncrop_h / 2 - crop_y

    Args:
        crop_x: Top-left x offset of the crop window.
        crop_y: Top-left y offset of the crop window.
        uncrop_w: Full sensor width (default 1280).
        uncrop_h: Full sensor height (default 1024).

    Returns:
        ``(cx, cy)`` — optical centre in cropped-frame pixel coordinates.
    """
    return (uncrop_w / 2.0 - crop_x, uncrop_h / 2.0 - crop_y)


def load_camera_params(meta_txt_path) -> dict:
    """Parse meta.txt and return camera parameters for perspective correction.

    Reads the meta.txt produced by the legacy video preprocessing pipeline
    (``mov_crop.py``). Extracts the crop offset to compute the camera optical
    centre, plus the pixel scale and maze corner coordinates.

    Args:
        meta_txt_path: Path to the session ``meta.txt`` file (typically in
            ``rawdata/.../behav/meta/meta.txt``).

    Returns:
        Dict with keys:

        - ``camera_center_px`` — ``(cx, cy)`` in cropped-frame pixels
        - ``crop_offset`` — ``(crop_x, crop_y)`` int tuple
        - ``scale_mm_per_px`` — float
        - ``maze_corners`` — ``(4, 2)`` float64 ndarray
    """
    from hm2p.pose.preprocess import load_meta

    meta = load_meta(meta_txt_path)
    crop_x, crop_y = meta["roi"][0], meta["roi"][1]
    cx, cy = estimate_camera_center(crop_x, crop_y)

    return {
        "camera_center_px": (cx, cy),
        "crop_offset": (crop_x, crop_y),
        "scale_mm_per_px": meta["scale_mm_per_px"],
        "maze_corners": meta["maze_corners"],
    }
