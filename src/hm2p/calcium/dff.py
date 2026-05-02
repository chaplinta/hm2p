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

import logging

import numpy as np

log = logging.getLogger(__name__)


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

    # ``np.nanpercentile`` rather than ``np.percentile``: motion-correction
    # NaNs or Suite2p-rejected frames in F propagate through plain
    # percentile. The rolling-min/max baseline path masks NaN via the
    # filter implementation, so the two methods previously disagreed on
    # NaN handling. NaN-aware percentile here brings them into line.
    # When every sample in a window is NaN, np.nanpercentile returns NaN —
    # callers should pre-fill or post-fill those windows.
    with np.errstate(invalid="ignore"):
        for t in range(n_frames):
            lo = max(0, t - half_w)
            hi = min(n_frames, t + half_w + 1)
            F0[:, t] = np.nanpercentile(F64[:, lo:hi], percentile, axis=1)

    return F0.astype(np.float32)


# Hard saturation bounds for dF/F0. The lower bound prevents
# F0-overestimation pathologies from producing < -100 % spikes that
# masquerade as activity at the 3·MAD threshold used elsewhere. The
# upper bound guards against single-frame ringing — for GCaMP6s/8 a
# 2000 % transient is never seen in soma traces, but FISSA
# decompositions can occasionally produce one while converging. The
# threshold below this fraction of saturated samples is informational;
# above it ``compute_dff`` logs a per-ROI warning so the user knows F0
# estimation is suspect for that ROI.
DFF_CLIP_LOW: float = -1.0
DFF_CLIP_HIGH: float = 20.0
DFF_CLIP_WARN_FRACTION: float = 0.001

# Lower bound for the dF/F0 denominator. F0 below this is replaced
# elementwise. Suite2p F is in photon counts; baselines are ≳ 50, so a
# constant floor of 1.0 sits well below typical F0 while preventing
# divide-by-zero. Previously this module also applied a per-ROI
# 10 %-of-median floor on top of the constant; that biased dF/F toward
# zero in exactly the windows where F0 is most uncertain (post-bleach
# tails, edges) and is removed here in favour of the constant floor
# alone. See Pachitariu et al. 2017 (doi:10.1101/061507) for the F0
# lower-envelope guarantee that this floor backstops.
DFF_F0_FLOOR: float = 1.0


def compute_dff(F: np.ndarray, F0: np.ndarray) -> np.ndarray:
    """Compute dF/F0 = (F - F0) / F0 with a small constant denominator floor.

    The denominator is ``max(F0, DFF_F0_FLOOR)`` per element, so a
    badly-estimated F0 (negative or very near zero) cannot blow up the
    output. The result is hard-clipped to ``[DFF_CLIP_LOW, DFF_CLIP_HIGH]``;
    when more than :data:`DFF_CLIP_WARN_FRACTION` of an ROI's samples hit
    a clip bound, a warning is logged so the user knows F0 estimation is
    suspect for that ROI. Use :func:`compute_dff_with_clip_counts` when
    the per-ROI clip counts are needed downstream (e.g. to persist as QC
    in ca.h5).

    Parameters
    ----------
    F : np.ndarray
        (n_rois, n_frames) float32 — neuropil-corrected fluorescence.
    F0 : np.ndarray
        (n_rois, n_frames) float32 — estimated baseline.

    Returns
    -------
    np.ndarray
        (n_rois, n_frames) float32 — saturation-clipped dF/F0.

    Raises
    ------
    ValueError
        If F and F0 shapes do not match.

    References
    ----------
    Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
    two-photon microscopy." doi:10.1101/061507 — F0 estimation algorithm.
    """
    dff, _ = compute_dff_with_clip_counts(F, F0)
    return dff


def compute_dff_with_clip_counts(F: np.ndarray, F0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute dF/F0 and return the per-ROI count of clipped samples.

    Identical to :func:`compute_dff` but exposes ``n_clipped`` so callers
    can persist per-ROI saturation counts as QC in ca.h5.

    Parameters
    ----------
    F : np.ndarray
        (n_rois, n_frames) float32 — neuropil-corrected fluorescence.
    F0 : np.ndarray
        (n_rois, n_frames) float32 — estimated baseline.

    Returns
    -------
    dff : np.ndarray
        (n_rois, n_frames) float32 — saturation-clipped dF/F0.
    n_clipped : np.ndarray
        (n_rois,) int32 — per-ROI count of samples that hit either clip
        bound. ``np.where(n_clipped > 0)`` flags ROIs whose F0 estimation
        is questionable.

    Raises
    ------
    ValueError
        If F and F0 shapes do not match.
    """
    if F.shape != F0.shape:
        raise ValueError(f"F shape {F.shape} != F0 shape {F0.shape}")
    safe_F0 = np.maximum(F0, DFF_F0_FLOOR)
    dff_raw = (F - F0) / safe_F0
    clipped_mask = (dff_raw < DFF_CLIP_LOW) | (dff_raw > DFF_CLIP_HIGH)
    dff = np.clip(dff_raw, DFF_CLIP_LOW, DFF_CLIP_HIGH).astype(np.float32)
    n_clipped = clipped_mask.sum(axis=1).astype(np.int32)

    # Warn loudly when an ROI's clip rate exceeds the threshold — that is
    # a strong indicator F0 was over- or under-estimated for that ROI.
    n_frames = max(1, F.shape[1])
    high_clip_rois = np.flatnonzero(n_clipped > DFF_CLIP_WARN_FRACTION * n_frames)
    if high_clip_rois.size:
        log.warning(
            "compute_dff: %d ROI(s) have > %.2f%% of samples at the "
            "saturation boundary [%.1f, %.1f]: ROIs %s. F0 estimation "
            "may be biased for these ROIs.",
            high_clip_rois.size,
            DFF_CLIP_WARN_FRACTION * 100.0,
            DFF_CLIP_LOW,
            DFF_CLIP_HIGH,
            high_clip_rois.tolist()[:20],
        )
    return dff, n_clipped
