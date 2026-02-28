"""Stage 4b — baseline estimation and dF/F0 computation.

Baseline F0 is estimated using a sliding-window minimum of a Gaussian-smoothed
trace (Suite2p method). This is robust to slow drift.
"""

from __future__ import annotations

import numpy as np


def compute_baseline(
    F: np.ndarray,
    fps: float,
    window_s: float = 60.0,
    gaussian_sigma_s: float = 5.0,
) -> np.ndarray:
    """Estimate baseline F0 via sliding window minimum of Gaussian-smoothed trace.

    Args:
        F: (n_rois, n_frames) float32 — neuropil-corrected fluorescence.
        fps: Imaging frame rate (Hz).
        window_s: Sliding minimum window length (seconds, default 60 s).
        gaussian_sigma_s: Gaussian smoothing sigma (seconds, default 5 s).

    Returns:
        (n_rois, n_frames) float32 — estimated baseline F0.
    """
    raise NotImplementedError


def compute_dff(F: np.ndarray, F0: np.ndarray) -> np.ndarray:
    """Compute dF/F0 = (F - F0) / F0.

    Args:
        F: (n_rois, n_frames) float32 — neuropil-corrected fluorescence.
        F0: (n_rois, n_frames) float32 — estimated baseline.

    Returns:
        (n_rois, n_frames) float32 — dF/F0.

    Raises:
        ValueError: If F and F0 shapes do not match.
    """
    if F.shape != F0.shape:
        raise ValueError(f"F shape {F.shape} != F0 shape {F0.shape}")
    return (F - F0) / np.where(F0 == 0, np.finfo(float).eps, F0)
