"""Stage 2 — video pre-processing common to all pose trackers.

Steps applied before tracker inference:
    1. Lens undistortion using camera-specific calibration (.npz)
    2. Crop to maze ROI (from meta/meta.txt)

The side camera (_side_left) is NEVER used — overhead camera only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_calibration(npz_path: Path) -> dict[str, np.ndarray]:
    """Load lens calibration arrays from a .npz file.

    Args:
        npz_path: Path to camera calibration .npz (camera_matrix, dist_coeffs).

    Returns:
        Dict with keys 'camera_matrix' (3×3) and 'dist_coeffs' (1×5).
    """
    raise NotImplementedError


def undistort_frame(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Apply lens undistortion to a single video frame.

    Args:
        frame: (H, W, 3) uint8 BGR image.
        camera_matrix: (3, 3) float64 intrinsics matrix.
        dist_coeffs: (1, 5) float64 distortion coefficients.

    Returns:
        Undistorted frame, same shape as input.
    """
    raise NotImplementedError


def crop_to_maze_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a frame to the maze region of interest.

    Args:
        frame: (H, W, 3) uint8 image.
        roi: (x, y, width, height) bounding box in pixel coordinates.

    Returns:
        Cropped frame (height, width, 3).
    """
    raise NotImplementedError


def load_meta(meta_txt_path: Path) -> dict[str, object]:
    """Parse meta/meta.txt for crop ROI, pixel scale (mm/px), maze corners.

    Args:
        meta_txt_path: Path to the session meta.txt file.

    Returns:
        Dict with keys: 'roi' (x,y,w,h), 'scale_mm_per_px' (float),
        'maze_corners' (4×2 ndarray).
    """
    raise NotImplementedError
