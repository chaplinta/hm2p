"""Image-based duplicate frame detection for DLC training data.

Compares frames at full resolution using pixel difference. Two frames
are considered duplicates if fewer than MIN_CHANGED_PCT of their pixels
differ by more than PIXEL_NOISE intensity units.

This catches genuinely identical frames (mouse didn't move, consecutive
video frames) without false-positiving on frames where the mouse is in
a similar but distinguishable position.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# Thresholds calibrated against manual review (2026-04-17).
# PIXEL_NOISE=15 absorbs sensor noise and compression artefacts.
# MIN_CHANGED_PCT=1.0 means <1% of pixels changed = duplicate.
PIXEL_NOISE: int = 15
MIN_CHANGED_PCT: float = 1.0


def load_frame_gray(path: Path) -> np.ndarray | None:
    """Load an image as grayscale, following symlinks."""
    real = path.resolve() if path.is_symlink() else path
    img = cv2.imread(str(real), cv2.IMREAD_GRAYSCALE)
    return img


def is_duplicate(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    pixel_noise: int = PIXEL_NOISE,
    min_changed_pct: float = MIN_CHANGED_PCT,
) -> tuple[bool, float]:
    """Check if two grayscale frames are near-identical.

    Returns
    -------
    tuple[bool, float]
        (is_dup, pct_changed). is_dup is True if fewer than
        min_changed_pct of pixels differ by more than pixel_noise.
    """
    if frame_a.shape != frame_b.shape:
        return False, 100.0
    diff = cv2.absdiff(frame_a, frame_b)
    pct_changed = 100.0 * float(np.mean(diff > pixel_noise))
    return pct_changed < min_changed_pct, pct_changed


def filter_duplicates_against_existing(
    video_path: str | Path,
    candidate_indices: list[int],
    existing_dir: Path,
    pixel_noise: int = PIXEL_NOISE,
    min_changed_pct: float = MIN_CHANGED_PCT,
) -> list[int]:
    """Filter candidate frame indices, removing those that are duplicates
    of existing PNGs on disk or of each other.

    Parameters
    ----------
    video_path : str or Path
        Path to the video file to extract candidates from.
    candidate_indices : list[int]
        Video frame indices to consider.
    existing_dir : Path
        Directory containing existing PNGs (e.g. labeled-data session dir
        or retrain_frames session dir). All *.png files are loaded.
    pixel_noise : int
        Per-pixel noise threshold.
    min_changed_pct : float
        Minimum percentage of pixels that must differ.

    Returns
    -------
    list[int]
        Filtered indices with duplicates removed.
    """
    # Load existing frames from disk
    existing_imgs: list[np.ndarray] = []
    for p in sorted(existing_dir.glob("*.png")):
        img = load_frame_gray(p)
        if img is not None:
            existing_imgs.append(img)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return candidate_indices

    kept: list[int] = []
    kept_imgs: list[np.ndarray] = []

    for idx in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check against existing frames on disk
        dup = False
        for existing in existing_imgs:
            is_dup, _ = is_duplicate(gray, existing, pixel_noise, min_changed_pct)
            if is_dup:
                dup = True
                break

        # Check against already-kept candidates from this batch
        if not dup:
            for kept_img in kept_imgs:
                is_dup, _ = is_duplicate(gray, kept_img, pixel_noise, min_changed_pct)
                if is_dup:
                    dup = True
                    break

        if not dup:
            kept.append(idx)
            kept_imgs.append(gray)

    cap.release()
    return kept
