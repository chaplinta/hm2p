"""Stage 2 — video pre-processing utilities for pose trackers.

**All 26 sessions already have pre-processed videos** (undistorted then
cropped by the legacy pipeline).  The `.mp4` files uploaded to S3 /
stored in ``rawdata/.../behav/`` are ready for direct pose-tracker
inference — no additional preprocessing step is needed at runtime.

The functions below (``load_calibration``, ``undistort_frame``,
``crop_to_maze_roi``) are retained for reference and potential future
use (e.g. new sessions recorded after the legacy pipeline is
decommissioned).  They are **not** called by the current Snakemake
pipeline.

``load_meta`` **is** used by Stage 3 (kinematics) to read the crop
ROI, pixel-scale, and maze-corner metadata that accompanies each
cropped video.

The side camera (_side_left) is NEVER used — overhead camera only.

meta.txt format (INI, configparser-compatible)
----------------------------------------------
[crop]
x = 108          # top-left x of crop window in undistorted frame
y = 261          # top-left y of crop window
width = 832
height = 608

[scale]
mm_per_pix = 0.811...   # pixel → mm conversion at maze plane

[roi]
x1 = 149.0   # maze corner 1 x (in cropped frame coordinates)
y1 = 72.0    # maze corner 1 y
x2 = 764.0   # corner 2 (clockwise, top-right)
y2 = 82.0
x3 = 757.0   # corner 3 (bottom-right)
y3 = 509.0
x4 = 143.0   # corner 4 (bottom-left)
y4 = 500.0
...

Calibration .npz keys
---------------------
mtx   : (3, 3) float64 — camera intrinsics matrix
dist  : (1, 5) float64 — radial + tangential distortion coefficients
"""

from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np


def load_calibration(npz_path: Path) -> dict[str, np.ndarray]:
    """Load lens calibration arrays from a .npz file.

    Args:
        npz_path: Path to camera calibration .npz (mtx, dist keys).

    Returns:
        Dict with keys 'camera_matrix' (3×3 float64) and
        'dist_coeffs' (1×5 float64).

    Raises:
        FileNotFoundError: If npz_path does not exist.
        KeyError: If required keys 'mtx' or 'dist' are absent.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {npz_path}")
    data = np.load(npz_path)
    if "mtx" not in data or "dist" not in data:
        raise KeyError(f"Calibration .npz must contain 'mtx' and 'dist'; got: {list(data.keys())}")
    return {
        "camera_matrix": data["mtx"].astype(np.float64),
        "dist_coeffs": data["dist"].astype(np.float64),
    }


def undistort_frame(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Apply lens undistortion to a single video frame.

    Uses cv2.undistort() — requires OpenCV to be installed.

    Args:
        frame: (H, W, 3) uint8 BGR image.
        camera_matrix: (3, 3) float64 intrinsics matrix.
        dist_coeffs: (1, 5) float64 distortion coefficients.

    Returns:
        Undistorted frame, same shape as input.
    """
    import cv2  # optional heavy dep — only needed for Stage 2 GPU jobs

    return cv2.undistort(frame, camera_matrix, dist_coeffs)


def crop_to_maze_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a frame to the maze region of interest.

    Args:
        frame: (H, W, 3) uint8 image.
        roi: (x, y, width, height) bounding box in pixel coordinates.

    Returns:
        Cropped frame (height, width, 3).
    """
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def load_meta(meta_txt_path: Path) -> dict[str, object]:
    """Parse meta/meta.txt for crop ROI, pixel scale (mm/px), maze corners.

    Reads the meta.txt written by the legacy video processing pipeline
    (mov_crop.py). The file is in INI format with [crop], [scale], [roi]
    sections.

    Args:
        meta_txt_path: Path to the session meta.txt file (in the behav/
            directory alongside the cropped video).

    Returns:
        Dict with keys:
            'roi'             : (x, y, w, h) int tuple — crop bounding box
            'scale_mm_per_px' : float — mm per pixel at maze plane
            'maze_corners'    : (4, 2) float64 ndarray — corner pixel
                                coordinates in the cropped frame, ordered
                                [top-left, top-right, bottom-right, bottom-left]

    Raises:
        FileNotFoundError: If meta_txt_path does not exist.
        KeyError: If required sections or keys are absent.
    """
    if not meta_txt_path.exists():
        raise FileNotFoundError(f"meta.txt not found: {meta_txt_path}")

    cfg = configparser.ConfigParser()
    cfg.read(meta_txt_path)

    try:
        roi = (
            int(cfg["crop"]["x"]),
            int(cfg["crop"]["y"]),
            int(cfg["crop"]["width"]),
            int(cfg["crop"]["height"]),
        )
        scale_mm_per_px = float(cfg["scale"]["mm_per_pix"])
        maze_corners = np.array(
            [
                [float(cfg["roi"]["x1"]), float(cfg["roi"]["y1"])],
                [float(cfg["roi"]["x2"]), float(cfg["roi"]["y2"])],
                [float(cfg["roi"]["x3"]), float(cfg["roi"]["y3"])],
                [float(cfg["roi"]["x4"]), float(cfg["roi"]["y4"])],
            ],
            dtype=np.float64,
        )
    except KeyError as exc:
        raise KeyError(f"Required key missing in {meta_txt_path}: {exc}") from exc

    return {
        "roi": roi,
        "scale_mm_per_px": scale_mm_per_px,
        "maze_corners": maze_corners,
    }
