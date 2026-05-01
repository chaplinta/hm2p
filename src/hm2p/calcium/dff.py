"""Stage 4b — baseline estimation and dF/F0 computation.

Two baseline methods are available (configured via config/pipeline.yaml f0_method):

    rolling    — Gaussian-smooth → rolling-min → rolling-max (Suite2p method;
                 default). Tracks the very bottom of the trace; may underestimate
                 F0 for highly active cells, biasing dF/F upward.

    percentile — Sliding-window percentile (Jia et al. 2011). More robust for
                 active cells; stored alongside rolling baseline for sensitivity
                 comparison.

Both baselines are always computed and stored in ca.h5 as ``F0_rolling`` and
``F0_percentile``. The primary ``dff`` array is computed from the method
selected by ``f0_method`` in pipeline.yaml (default ``rolling``).

References:
    Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
    two-photon microscopy." doi:10.1101/061507
    https://github.com/MouseLand/suite2p/blob/main/suite2p/extraction/dcnv.py

    Jia H, Rochefort NL, Chen X, Bhatt DL, Bhatt DL, Bhatt DL, Konnerth A. 2011.
    "In vivo two-photon imaging of sensory-evoked dendritic calcium signals in
    cortical neurons." Nature Protocols 6:28-35. doi:10.1038/nprot.2010.169
"""

from __future__ import annotations

import numpy as np


def compute_baseline(
    F: np.ndarray,
    fps: float,
    window_s: float = 60.0,
    gaussian_sigma_s: float = 10.0,
) -> np.ndarray:
    """Estimate baseline F0 via rolling min–max of Gaussian-smoothed trace.

    Implements Suite2p's ``dcnv.preprocess`` baseline algorithm
    (Pachitariu et al. 2017, doi:10.1101/061507):

      1. Gaussian-smooth each trace (sigma = ``gaussian_sigma_s``) to
         attenuate fast calcium transients.
      2. Rolling minimum (window = ``window_s``) to find the lower
         envelope of the smoothed trace.
      3. Rolling maximum (same window) to smooth sharp dips and prevent
         the baseline from dropping into noise troughs.

    Args:
        F: (n_rois, n_frames) float32 — neuropil-corrected fluorescence.
        fps: Imaging frame rate (Hz).
        window_s: Rolling min/max window length (seconds, default 60 s).
        gaussian_sigma_s: Gaussian smoothing sigma (seconds, default 10 s).

    Returns:
        (n_rois, n_frames) float32 — estimated baseline F0.
    """
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d

    sigma_frames = gaussian_sigma_s * fps
    window_frames = max(1, int(window_s * fps))

    # Step 1: Gaussian smooth to suppress transients
    F_smooth = gaussian_filter1d(F.astype(np.float64), sigma=sigma_frames, axis=1)

    # Step 2: Rolling minimum — lower envelope
    F_min = minimum_filter1d(F_smooth, size=window_frames, axis=1)

    # Step 3: Rolling maximum — smooth sharp dips in the minimum trace
    F0 = maximum_filter1d(F_min, size=window_frames, axis=1)

    return F0.astype(np.float32)


def compute_baseline_percentile(
    F: np.ndarray,
    fps: float,
    window_s: float = 60.0,
    percentile: float = 8.0,
) -> np.ndarray:
    """Estimate baseline F0 via sliding-window percentile.

    Computes the ``percentile``-th percentile of fluorescence within a sliding
    window of length ``window_s`` seconds. This method is more robust than the
    rolling-min approach for highly active cells, since it does not track the
    absolute minimum but rather a stable lower quantile of the signal. A
    window of 60 s with the 8th percentile is the standard parameterisation
    from Jia et al. (2011).

    Parameters
    ----------
    F : np.ndarray
        (n_rois, n_frames) float32 — neuropil-corrected fluorescence.
    fps : float
        Imaging frame rate (Hz).
    window_s : float
        Sliding window length in seconds (default 60 s).
    percentile : float
        Percentile to extract within each window (default 8.0, as in Jia et al.
        2011). Lower values track closer to the noise floor; higher values give
        a higher, more conservative baseline.

    Returns
    -------
    np.ndarray
        (n_rois, n_frames) float32 — estimated baseline F0.

    References
    ----------
    Jia H, Rochefort NL, Chen X, Konnerth A. 2011. "In vivo two-photon imaging
    of sensory-evoked dendritic calcium signals in cortical neurons."
    Nature Protocols 6:28-35. doi:10.1038/nprot.2010.169
    """
    half_w = max(0, int(window_s * fps / 2))
    n_frames = F.shape[1]
    F64 = F.astype(np.float64)
    F0 = np.empty_like(F64)

    for t in range(n_frames):
        lo = max(0, t - half_w)
        hi = min(n_frames, t + half_w + 1)
        F0[:, t] = np.percentile(F64[:, lo:hi], percentile, axis=1)

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
    # Per-ROI floor: prevent near-zero denominators after neuropil subtraction
    f0_median = np.median(F0, axis=1, keepdims=True)
    f0_floor = np.maximum(f0_median * 0.1, 1.0)
    safe_F0 = np.maximum(F0, f0_floor)
    dff = ((F - F0) / safe_F0).astype(np.float32)
    # Hard clip: GCaMP rarely exceeds 10x baseline
    dff = np.clip(dff, -1.0, 20.0)
    return dff
