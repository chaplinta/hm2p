"""Per-ROI feature extraction for soma/dendrite/artefact classification.

Builds a single :class:`pandas.DataFrame` with one row per ROI containing
both Suite2p shape statistics and activity features derived from the raw
fluorescence and neuropil traces.  The resulting feature table is consumed
by :mod:`hm2p.extraction.soma_classifier`.

Feature set
-----------
**Shape features** (from each Suite2p ``stat[i]`` dict):

- ``radius`` — equivalent disk radius of the ROI footprint (px)
- ``compact`` — compactness (4π·area / perimeter²; ranges over (0, 1])
- ``aspect_ratio`` — major-axis / minor-axis ratio (≥ 1 in practice)
- ``npix`` — number of pixels in the ROI footprint
- ``npix_norm`` — Suite2p's pixel-count z-score
- ``skew`` — skewness of the trace
- ``std`` — standard deviation of the trace (population estimate)

**Activity features** (computed from the per-ROI traces):

- ``peak_to_noise_dff`` — peak event amplitude divided by a robust noise
  estimate of the dF/F trace.  The robust noise is ``MAD * 1.4826``.
- ``autocorr_halfwidth_s`` — time (in seconds) for the autocorrelation of
  the dF/F trace to drop to 0.5.  Wider half-widths indicate slower
  kinetics, which tend to be characteristic of dendritic processes.
- ``fneu_corr`` — Spearman rank correlation between the ROI's dF/F trace
  and the mean ``Fneu`` trace across all ROIs.  High positive values
  indicate residual neuropil contamination.

The dF/F trace is computed inline here (eighth-percentile baseline; same
parameterisation as :mod:`hm2p.calcium.dff.compute_baseline_percentile`)
so that the feature extractor can be applied to raw Suite2p arrays without
re-running the full Stage 4 pipeline.

References
----------
Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
two-photon microscopy." bioRxiv. doi:10.1101/061507.
https://github.com/MouseLand/suite2p

Pedregosa et al. 2011. "Scikit-learn: Machine Learning in Python."
Journal of Machine Learning Research 12:2825–2830.
https://scikit-learn.org

Jia H, Rochefort NL, Chen X, Konnerth A. 2011. "In vivo two-photon imaging
of sensory-evoked dendritic calcium signals in cortical neurons." Nature
Protocols 6:28-35. doi:10.1038/nprot.2010.169
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)


# Columns produced by ``extract_soma_features`` — the canonical feature order
# expected by downstream classifiers.  Tests import this list to verify the
# feature schema.
FEATURE_COLUMNS: tuple[str, ...] = (
    # Shape features (from Suite2p stat[i])
    "radius",
    "compact",
    "aspect_ratio",
    "npix",
    "npix_norm",
    "skew",
    "std",
    # Activity features
    "peak_to_noise_dff",
    "autocorr_halfwidth_s",
    "fneu_corr",
)


# Default fallback values for missing stat keys.  These match the defaults
# used by the legacy heuristic classifier so that ROIs with incomplete stats
# still receive reasonable feature values.
_STAT_DEFAULTS: dict[str, float] = {
    "radius": 5.0,
    "compact": 0.5,
    "aspect_ratio": 1.0,
    "npix": 0.0,
    "npix_norm": 0.0,
    "skew": 0.0,
    "std": 0.0,
}


def _quick_dff(F: np.ndarray) -> np.ndarray:
    """Return a quick-and-dirty per-ROI dF/F0 trace.

    Uses the 8th-percentile of each ROI's raw fluorescence as F0 (Jia et al.
    2011).  This is a global percentile rather than a sliding window — it is
    intentionally cheaper than :func:`hm2p.calcium.dff.compute_baseline_percentile`
    because feature extraction can run on tens of thousands of ROIs and a
    full sliding-window baseline would dominate runtime.  For classifier
    feature ranking the global 8th-percentile baseline is sufficient.

    Parameters
    ----------
    F : np.ndarray
        ``(n_rois, n_frames)`` raw fluorescence (or neuropil-corrected F).

    Returns
    -------
    np.ndarray
        ``(n_rois, n_frames)`` float32 dF/F0.
    """
    F64 = F.astype(np.float64)
    f0 = np.percentile(F64, 8.0, axis=1, keepdims=True)
    # Per-ROI floor matching ``compute_dff`` (avoids near-zero denominators).
    f0_floor = np.maximum(f0, 1.0)
    f0_safe = np.where(f0 > f0_floor, f0, f0_floor)
    dff = (F64 - f0_safe) / f0_safe
    return dff.astype(np.float32)


def _peak_to_noise(dff_trace: np.ndarray) -> float:
    """Return ``max(dff) / (1.4826 * MAD(dff))``.

    Uses the median absolute deviation as a robust noise estimate.  The
    1.4826 factor converts MAD to a Gaussian-equivalent standard deviation.
    Returns NaN when the trace is constant (MAD == 0) or empty.

    Parameters
    ----------
    dff_trace : np.ndarray
        ``(n_frames,)`` float dF/F0 for one ROI.

    Returns
    -------
    float
        Peak-to-noise ratio, or NaN if undefined.
    """
    if dff_trace.size == 0:
        return float("nan")
    med = float(np.median(dff_trace))
    mad = float(np.median(np.abs(dff_trace - med)))
    sigma = 1.4826 * mad
    if sigma <= 0.0:
        return float("nan")
    peak = float(np.max(dff_trace))
    return (peak - med) / sigma


def _autocorr_halfwidth_s(
    dff_trace: np.ndarray,
    fps: float,
    max_lag_s: float = 5.0,
) -> float:
    """Return the lag (in seconds) where the autocorrelation drops below 0.5.

    The autocorrelation is normalised so that lag 0 equals 1.  Lags are
    searched in the range ``[1, n_lag_max]`` where ``n_lag_max`` is the
    smaller of ``max_lag_s * fps`` and ``len(trace) - 1``.  Returns NaN if
    the trace has insufficient samples or is constant; returns ``max_lag_s``
    if the autocorrelation never crosses 0.5 within the search window
    (effectively right-censored).

    Parameters
    ----------
    dff_trace : np.ndarray
        ``(n_frames,)`` float dF/F0 for one ROI.
    fps : float
        Imaging frame rate (Hz).
    max_lag_s : float
        Maximum lag to search, in seconds.  Default 5 s.

    Returns
    -------
    float
        Half-width of the autocorrelation in seconds.
    """
    n = dff_trace.size
    if n < 4 or fps <= 0:
        return float("nan")
    x = dff_trace - np.mean(dff_trace)
    var = float(np.dot(x, x))
    if var <= 0.0:
        return float("nan")
    n_lag_max = min(int(round(max_lag_s * fps)), n - 1)
    if n_lag_max < 1:
        return float("nan")
    # Compute a few short lags directly (cheaper than full FFT autocorr).
    for lag in range(1, n_lag_max + 1):
        # Pearson autocorrelation at this lag, normalised by lag-0 variance.
        num = float(np.dot(x[:-lag], x[lag:]))
        ac = num / var
        if ac < 0.5:
            # Linear interpolation between lag-1 and lag for a sub-frame estimate.
            if lag == 1:
                return lag / fps
            num_prev = float(np.dot(x[: -(lag - 1)], x[lag - 1 :]))
            ac_prev = num_prev / var
            if ac_prev == ac:
                return lag / fps
            frac = (ac_prev - 0.5) / (ac_prev - ac)
            return ((lag - 1) + frac) / fps
    # Autocorrelation stays above 0.5 within the search window — right-censored.
    return float(max_lag_s)


def _fneu_corr(dff_trace: np.ndarray, mean_fneu: np.ndarray) -> float:
    """Return the Spearman rank correlation between an ROI's dF/F and mean Fneu.

    Returns NaN when either trace is constant or shorter than 10 samples.

    Parameters
    ----------
    dff_trace : np.ndarray
        ``(n_frames,)`` ROI dF/F0.
    mean_fneu : np.ndarray
        ``(n_frames,)`` mean neuropil trace (averaged across all ROIs).

    Returns
    -------
    float
        Spearman r in ``[-1, 1]`` or NaN if undefined.
    """
    n = dff_trace.size
    if n < 10 or mean_fneu.size != n:
        return float("nan")
    if np.std(dff_trace) == 0.0 or np.std(mean_fneu) == 0.0:
        return float("nan")
    res = stats.spearmanr(dff_trace, mean_fneu)
    val = float(res.statistic)
    if not np.isfinite(val):
        return float("nan")
    return val


def extract_soma_features(
    stat: list[dict],  # type: ignore[type-arg]
    F: np.ndarray,
    Fneu: np.ndarray,
    fps: float,
) -> pd.DataFrame:
    """Build a per-ROI feature table for soma/dendrite/artefact classification.

    Parameters
    ----------
    stat : list of dict
        Suite2p ``stat.npy`` contents — one dict per ROI containing at least
        ``radius``, ``compact``, ``aspect_ratio``, ``npix``, ``npix_norm``,
        ``skew``, and ``std``.  Missing keys fall back to the same defaults
        used by the legacy heuristic classifier.
    F : np.ndarray
        ``(n_rois, n_frames)`` raw fluorescence traces.  Used to compute
        the activity features.  Must satisfy ``len(stat) == F.shape[0]``.
    Fneu : np.ndarray
        ``(n_rois, n_frames)`` neuropil traces.  The mean across ROIs is
        used as the reference signal for ``fneu_corr``.
    fps : float
        Imaging frame rate (Hz).  Required for converting autocorrelation
        lags to seconds.

    Returns
    -------
    pandas.DataFrame
        ``(n_rois, len(FEATURE_COLUMNS))`` with one row per ROI.  All values
        are floats.  NaN values are returned for activity features when a
        trace is constant or too short — downstream classifiers must handle
        NaNs explicitly.

    Raises
    ------
    ValueError
        If ``len(stat)`` does not match ``F.shape[0]`` or ``Fneu.shape[0]``,
        or if ``F`` and ``Fneu`` shapes disagree.

    References
    ----------
    Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
    two-photon microscopy." bioRxiv. doi:10.1101/061507.

    Pedregosa et al. 2011. "Scikit-learn: Machine Learning in Python."
    Journal of Machine Learning Research 12:2825–2830.
    """
    if F.shape != Fneu.shape:
        raise ValueError(f"F shape {F.shape} != Fneu shape {Fneu.shape}")
    n_rois, n_frames = F.shape
    if len(stat) != n_rois:
        raise ValueError(f"len(stat)={len(stat)} does not match F.shape[0]={n_rois}")

    if n_rois == 0:
        return pd.DataFrame({col: pd.Series(dtype="float64") for col in FEATURE_COLUMNS})

    # Compute dF/F0 once for the full population — global 8th-percentile baseline.
    dff = _quick_dff(F)
    mean_fneu = np.mean(Fneu.astype(np.float64), axis=0)

    rows: list[dict[str, float]] = []
    for i in range(n_rois):
        s = stat[i]
        row: dict[str, float] = {}
        # Shape features
        for key, default in _STAT_DEFAULTS.items():
            row[key] = float(s.get(key, default))

        # Activity features
        trace = dff[i].astype(np.float64)
        row["peak_to_noise_dff"] = _peak_to_noise(trace)
        row["autocorr_halfwidth_s"] = _autocorr_halfwidth_s(trace, fps)
        row["fneu_corr"] = _fneu_corr(trace, mean_fneu)

        rows.append(row)

    df = pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))
    log.info(
        "Extracted soma features for %d ROIs (n_frames=%d, fps=%.2f Hz)",
        n_rois,
        n_frames,
        fps,
    )
    return df
