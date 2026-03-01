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

    Mirrors the Suite2p baseline estimator:
      1. Gaussian-smooth each trace to attenuate transients.
      2. Sliding-window minimum to track slow drift.

    Args:
        F: (n_rois, n_frames) float32 — neuropil-corrected fluorescence.
        fps: Imaging frame rate (Hz).
        window_s: Sliding minimum window length (seconds, default 60 s).
        gaussian_sigma_s: Gaussian smoothing sigma (seconds, default 5 s).

    Returns:
        (n_rois, n_frames) float32 — estimated baseline F0.
    """
    from scipy.ndimage import gaussian_filter1d, minimum_filter1d

    sigma_frames = gaussian_sigma_s * fps
    window_frames = max(1, int(window_s * fps))

    F_smooth = gaussian_filter1d(F.astype(np.float64), sigma=sigma_frames, axis=1)
    F0 = minimum_filter1d(F_smooth, size=window_frames, axis=1)
    return F0.astype(np.float32)


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
