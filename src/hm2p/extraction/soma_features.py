"""Per-ROI feature extraction for soma/dendrite/artefact classification.

Builds a single :class:`pandas.DataFrame` with one row per ROI containing
both Suite2p shape statistics and activity features derived from the
neuropil-corrected fluorescence traces.  The resulting feature table is
consumed by :mod:`hm2p.extraction.soma_classifier`.

Feature set
-----------
**Shape features** (from each Suite2p ``stat[i]`` dict):

- ``radius`` — equivalent disk radius of the ROI footprint (px)
- ``compact`` — ratio of observed mean radial spread to expected spread
  for a disk of equal area.  1.0 = perfect disk; >1 = less compact.
- ``aspect_ratio`` — major-axis / minor-axis ratio (>= 1 in practice)
- ``npix`` — number of pixels in the ROI footprint
- ``npix_norm`` — Suite2p's pixel-count z-score (after soma crop)
- ``skew`` — skewness of the neuropil-corrected trace (from Suite2p)
- ``std`` — standard deviation of the neuropil-corrected trace
- ``solidity`` — area / convex hull area.  Compact somas ~1.0.

**Derived shape features** (computed from ``stat[i]``):

- ``soma_crop_fraction`` — ``npix_soma / npix``.
- ``npix_norm_ratio`` — ``npix_norm / npix_norm_no_crop``.
- ``overlap_fraction`` — fraction of ROI pixels shared with other ROIs.
- ``eccentricity`` — from eigenvalues of the 2D pixel coordinate
  covariance matrix.  0 = circle, approaching 1 = elongated.
- ``lam_cv`` — coefficient of variation of pixel weights (``lam``).
  Uniform weighting (soma) vs concentrated (dendrite tips).
- ``n_connected_components`` — number of connected components in the
  2D pixel mask.  Branching dendrites may split into >1 component.
- ``n_branch_points`` — number of skeleton branch points (junctions
  where >2 skeleton pixels meet).  Branching dendrites have these;
  somas and thin fragments do not.
- ``boundary_roughness`` — perimeter / (2 * sqrt(pi * area)).  Spiny
  dendrites have rough boundaries (>1.0); smooth somas are closer to
  1.0.

**Activity features** (computed from neuropil-corrected traces):

- ``peak_to_noise_dff`` — 99th-percentile dF/F / robust noise (MAD).
- ``autocorr_halfwidth_s`` — lag where autocorrelation drops to 0.5.
- ``fneu_corr`` — Spearman correlation with per-ROI neuropil ring.
- ``kurtosis`` — excess kurtosis of the dF/F trace.
- ``signal_to_background`` — ``mean(F[i]) / mean(Fneu[i])``.  Bright
  somas have high S/B; artefact blobs are close to neuropil intensity.
- ``event_rate`` — threshold crossings per minute on dF/F (2*MAD).
- ``derivative_skew`` — skewness of ``diff(dff)``.  Fast-rise/slow-decay
  calcium transients produce positive derivative skewness.
- ``trace_sparsity`` — fraction of frames where dF/F > 2*MAD.
  Real neurons are active ~5-15% of frames; artefacts are not.
- ``power_slope`` — log-log slope of the power spectral density.
  Real neurons have flatter spectra (transient broadband); artefacts
  are dominated by slow drift (steep negative slope).
- ``max_pairwise_corr`` — highest Pearson correlation with any other
  ROI in the session.  Dendrites correlate with their parent soma.

References
----------
Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
two-photon microscopy." bioRxiv. doi:10.1101/061507.
https://github.com/MouseLand/suite2p

Jia H, Rochefort NL, Chen X, Konnerth A. 2011. "In vivo two-photon imaging
of sensory-evoked dendritic calcium signals in cortical neurons." Nature
Protocols 6:28-35. doi:10.1038/nprot.2010.169
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

log = logging.getLogger(__name__)


FEATURE_COLUMNS: tuple[str, ...] = (
    # Shape features (from Suite2p stat[i])
    "radius",
    "compact",
    "aspect_ratio",
    "npix",
    "npix_norm",
    "skew",
    "std",
    "solidity",
    # Derived shape features
    "soma_crop_fraction",
    "npix_norm_ratio",
    "overlap_fraction",
    "eccentricity",
    "lam_cv",
    # Mask topology features (from 2D pixel mask reconstruction)
    "n_connected_components",
    "n_branch_points",
    "boundary_roughness",
    # Activity features
    "peak_to_noise_dff",
    "autocorr_halfwidth_s",
    "fneu_corr",
    "kurtosis",
    "signal_to_background",
    "event_rate",
    "derivative_skew",
    "trace_sparsity",
    "power_slope",
    "max_pairwise_corr",
)


_STAT_DEFAULTS: dict[str, float] = {
    "radius": 5.0,
    "compact": 1.0,
    "aspect_ratio": 1.0,
    "npix": 1,
    "npix_norm": 1.0,
    "skew": 0.0,
    "std": 0.0,
    "solidity": 1.0,
    "npix_soma": 1,
    "npix_norm_no_crop": 1.0,
}

_STAT_FEATURE_KEYS: tuple[str, ...] = (
    "radius",
    "compact",
    "aspect_ratio",
    "npix",
    "npix_norm",
    "skew",
    "std",
    "solidity",
)


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------


def _quick_dff(F_corr: np.ndarray) -> np.ndarray:
    """Return a quick per-ROI dF/F0 from neuropil-corrected fluorescence."""
    from hm2p.calcium.dff import DFF_F0_FLOOR

    F64 = F_corr.astype(np.float64)
    f0 = np.percentile(F64, 8.0, axis=1, keepdims=True)
    f0_safe = np.maximum(f0, DFF_F0_FLOOR)
    dff = (F64 - f0_safe) / f0_safe
    return dff.astype(np.float32)


def _robust_noise(trace: np.ndarray) -> float:
    """MAD-based noise estimate: 1.4826 * median(|x - median(x)|)."""
    if trace.size == 0:
        return float("nan")
    med = float(np.median(trace))
    mad = float(np.median(np.abs(trace - med)))
    return 1.4826 * mad


# ---------------------------------------------------------------------------
# Activity feature functions
# ---------------------------------------------------------------------------


def _peak_to_noise(dff_trace: np.ndarray) -> float:
    """99th-percentile dF/F divided by MAD-based noise."""
    if dff_trace.size == 0:
        return float("nan")
    sigma = _robust_noise(dff_trace)
    if sigma <= 0.0:
        return float("nan")
    med = float(np.median(dff_trace))
    peak99 = float(np.percentile(dff_trace, 99.0))
    return (peak99 - med) / sigma


def _autocorr_halfwidth_s(
    dff_trace: np.ndarray,
    fps: float,
    max_lag_s: float = 5.0,
) -> float:
    """Lag (seconds) where autocorrelation drops below 0.5.

    Uses FFT-based autocorrelation for speed (O(n log n) instead of
    O(n * n_lags) for the direct dot-product loop).
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

    # FFT-based autocorrelation (zero-padded to avoid circular correlation)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    xf = np.fft.rfft(x, n=nfft)
    ac_full = np.fft.irfft(xf * np.conj(xf), n=nfft)[: n_lag_max + 1]
    ac_full = ac_full / var  # normalize so lag-0 = 1

    # Find first lag where ac drops below 0.5
    for lag in range(1, n_lag_max + 1):
        if ac_full[lag] < 0.5:
            if lag == 1:
                return lag / fps
            ac_prev = ac_full[lag - 1]
            ac_curr = ac_full[lag]
            if ac_prev == ac_curr:
                return lag / fps
            frac = (ac_prev - 0.5) / (ac_prev - ac_curr)
            return ((lag - 1) + frac) / fps
    return float(max_lag_s)


def _fneu_corr_batch(dff: np.ndarray, Fneu: np.ndarray) -> np.ndarray:
    """Pearson correlation between each ROI's dF/F and its neuropil ring.

    Vectorized over all ROIs at once.  Uses Pearson instead of Spearman
    for speed — avoids per-ROI rank sorting (the bottleneck of Spearman
    on long traces).  Pearson is adequate here because we only need a
    relative contamination score, not a p-value.

    Parameters
    ----------
    dff : (n_rois, n_frames) float64
    Fneu : (n_rois, n_frames) float64

    Returns
    -------
    (n_rois,) float64 — Pearson r per ROI, NaN for constant traces.
    """
    n_rois, n_frames = dff.shape
    if n_frames < 10:
        return np.full(n_rois, np.nan)

    # Center both
    dff_c = dff - dff.mean(axis=1, keepdims=True)
    fneu_c = Fneu - Fneu.mean(axis=1, keepdims=True)

    # Norms
    dff_norm = np.sqrt(np.sum(dff_c**2, axis=1))
    fneu_norm = np.sqrt(np.sum(fneu_c**2, axis=1))

    # Avoid division by zero for constant traces
    valid = (dff_norm > 0) & (fneu_norm > 0)
    result = np.full(n_rois, np.nan)

    if valid.any():
        # Row-wise dot product
        numerator = np.sum(dff_c[valid] * fneu_c[valid], axis=1)
        result[valid] = numerator / (dff_norm[valid] * fneu_norm[valid])

    return result


def _max_pairwise_corr_batch(dff: np.ndarray) -> np.ndarray:
    """Max Pearson correlation with any other ROI, vectorized.

    Parameters
    ----------
    dff : (n_rois, n_frames) float64

    Returns
    -------
    (n_rois,) float64 — max correlation per ROI, NaN for constant traces.
    """
    n_rois, n_frames = dff.shape
    if n_rois < 2 or n_frames < 10:
        return np.full(n_rois, np.nan)

    # Center and normalize rows
    dff_c = dff - dff.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(dff_c**2, axis=1))
    valid = norms > 0
    result = np.full(n_rois, np.nan)

    if valid.sum() < 2:
        return result

    # Work only with non-constant ROIs
    dff_normed = np.zeros_like(dff_c)
    dff_normed[valid] = dff_c[valid] / norms[valid, np.newaxis]

    # Full correlation matrix via matrix multiply
    corr = dff_normed[valid] @ dff_normed[valid].T
    np.fill_diagonal(corr, -np.inf)  # exclude self

    # Max per row
    max_corrs = np.max(corr, axis=1)

    # Map back to full ROI indices
    valid_idx = np.where(valid)[0]
    result[valid_idx] = max_corrs

    return result


def _kurtosis(dff_trace: np.ndarray) -> float:
    """Excess kurtosis (Fisher definition: normal = 0)."""
    if dff_trace.size < 4 or np.std(dff_trace) == 0.0:
        return float("nan")
    return float(sp_stats.kurtosis(dff_trace, fisher=True))


def _signal_to_background(F_roi: np.ndarray, Fneu_roi: np.ndarray) -> float:
    """mean(F) / mean(Fneu) for one ROI. Raw traces, not dF/F."""
    if F_roi.size == 0:
        return float("nan")
    mean_f = float(np.mean(F_roi))
    mean_fneu = float(np.mean(Fneu_roi))
    if mean_fneu <= 0.0:
        return float("nan")
    return mean_f / mean_fneu


def _event_rate(dff_trace: np.ndarray, fps: float) -> float:
    """Threshold crossings per minute. Threshold = 2 * MAD noise."""
    if dff_trace.size < 10 or fps <= 0:
        return float("nan")
    sigma = _robust_noise(dff_trace)
    if sigma <= 0.0:
        return float("nan")
    med = float(np.median(dff_trace))
    threshold = med + 2.0 * sigma
    above = dff_trace > threshold
    # Count rising edges (0->1 transitions)
    crossings = int(np.sum(np.diff(above.astype(np.int8)) == 1))
    duration_min = dff_trace.size / fps / 60.0
    if duration_min <= 0.0:
        return float("nan")
    return crossings / duration_min


def _derivative_skew(dff_trace: np.ndarray) -> float:
    """Skewness of diff(dF/F). Fast rise + slow decay -> positive skew."""
    if dff_trace.size < 10:
        return float("nan")
    d = np.diff(dff_trace)
    if np.std(d) == 0.0:
        return float("nan")
    return float(sp_stats.skew(d))


def _trace_sparsity(dff_trace: np.ndarray) -> float:
    """Fraction of frames where dF/F exceeds 2*MAD above median."""
    if dff_trace.size < 10:
        return float("nan")
    sigma = _robust_noise(dff_trace)
    if sigma <= 0.0:
        return float("nan")
    med = float(np.median(dff_trace))
    return float(np.mean(dff_trace > med + 2.0 * sigma))


def _power_slope(dff_trace: np.ndarray, fps: float) -> float:
    """Log-log slope of the power spectral density.

    Fit a line to log(PSD) vs log(freq) in the range [0.1 Hz, fps/2].
    Steep negative slope = slow drift (artefact); flatter = broadband
    transients (real neuron).
    """
    n = dff_trace.size
    if n < 32 or fps <= 0:
        return float("nan")
    if np.std(dff_trace) == 0.0:
        return float("nan")

    # Compute PSD via FFT
    fft_vals = np.fft.rfft(dff_trace - np.mean(dff_trace))
    psd = np.abs(fft_vals) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)

    # Select frequency range [0.1, nyquist]
    mask = freqs >= 0.1
    if mask.sum() < 5:
        return float("nan")
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask] + 1e-30)  # avoid log(0)

    # Linear fit
    slope, _, _, _, _ = sp_stats.linregress(log_f, log_p)
    return float(slope) if np.isfinite(slope) else float("nan")


# ---------------------------------------------------------------------------
# Shape feature functions
# ---------------------------------------------------------------------------


def _eccentricity(stat_entry: dict) -> float:
    """Eccentricity from eigenvalues of pixel coordinate covariance."""
    xpix = stat_entry.get("xpix")
    ypix = stat_entry.get("ypix")
    if xpix is None or ypix is None:
        return float("nan")
    if len(xpix) < 3:
        return float("nan")
    coords = np.stack([xpix.astype(np.float64), ypix.astype(np.float64)], axis=0)
    cov = np.cov(coords)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]  # descending
    if eigvals[0] <= 0:
        return 0.0
    return float(np.sqrt(1.0 - eigvals[1] / eigvals[0]))


def _mask_topology(stat_entry: dict) -> tuple[int, int, float]:
    """Compute topology features from the 2D pixel mask.

    Reconstructs the binary mask from xpix/ypix, then computes:
    1. Number of connected components (branching dendrites may split)
    2. Number of skeleton branch points (junctions in the thinned mask)
    3. Boundary roughness: perimeter / (2 * sqrt(pi * area)).
       Smooth soma ~1.0; spiny/rough dendrite > 1.0.

    Returns (n_components, n_branch_points, boundary_roughness).
    All returned as (0, 0, NaN) if xpix/ypix are missing.
    """
    from scipy import ndimage

    xpix = stat_entry.get("xpix")
    ypix = stat_entry.get("ypix")
    if xpix is None or ypix is None or len(xpix) < 3:
        return (0, 0, float("nan"))

    # Build minimal bounding-box mask (avoids allocating full FOV image)
    x = xpix.astype(np.int64)
    y = ypix.astype(np.int64)
    x0, y0 = x.min(), y.min()
    x -= x0
    y -= y0
    h, w = x.max() + 3, y.max() + 3  # +3 for 1px border (needed for perimeter)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[x, y] = 1

    # Connected components
    labeled, n_components = ndimage.label(mask)

    # Skeleton branch points
    # Skeletonize via hit-or-miss with 3x3 structuring elements.
    # A branch point has >2 neighbors in the skeleton.
    try:
        from skimage.morphology import skeletonize

        skel = skeletonize(mask > 0)
        # Count pixels with >2 neighbors (branch points)
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        neighbor_count = ndimage.convolve(skel.astype(np.uint8), kernel, mode="constant")
        n_branch_points = int(np.sum((skel > 0) & (neighbor_count > 2)))
    except ImportError:
        # skimage not available — skip branch points
        n_branch_points = 0

    # Boundary roughness: perimeter / (2 * sqrt(pi * area))
    area = float(mask.sum())
    if area < 1:
        return (n_components, n_branch_points, float("nan"))
    # Perimeter via binary erosion: boundary = mask - eroded_mask
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
    perimeter = float((mask.astype(np.int8) - eroded.astype(np.int8)).sum())
    expected_perimeter = 2.0 * np.sqrt(np.pi * area)
    if expected_perimeter <= 0:
        return (n_components, n_branch_points, float("nan"))
    roughness = perimeter / expected_perimeter

    return (n_components, n_branch_points, roughness)


def _lam_cv(stat_entry: dict) -> float:
    """Coefficient of variation of pixel intensity weights (lam)."""
    lam = stat_entry.get("lam")
    if lam is None or len(lam) < 2:
        return float("nan")
    mean_lam = float(np.mean(lam))
    if mean_lam <= 0:
        return float("nan")
    return float(np.std(lam) / mean_lam)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def extract_soma_features(
    stat: list[dict],  # type: ignore[type-arg]
    F: np.ndarray,
    Fneu: np.ndarray,
    fps: float,
    neucoeff: float = 0.7,
) -> pd.DataFrame:
    """Build a per-ROI feature table for soma/dendrite/artefact classification.

    Parameters
    ----------
    stat : list of dict
        Suite2p ``stat.npy`` contents — one dict per ROI.
    F : np.ndarray
        ``(n_rois, n_frames)`` raw fluorescence traces.
    Fneu : np.ndarray
        ``(n_rois, n_frames)`` neuropil traces (per-ROI neuropil rings).
    fps : float
        Imaging frame rate (Hz).
    neucoeff : float
        Neuropil subtraction coefficient.  Default 0.7.

    Returns
    -------
    pandas.DataFrame
        ``(n_rois, len(FEATURE_COLUMNS))`` with one row per ROI.

    References
    ----------
    Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
    two-photon microscopy." bioRxiv. doi:10.1101/061507.
    """
    if F.shape != Fneu.shape:
        raise ValueError(f"F shape {F.shape} != Fneu shape {Fneu.shape}")
    n_rois, n_frames = F.shape
    if len(stat) != n_rois:
        raise ValueError(f"len(stat)={len(stat)} does not match F.shape[0]={n_rois}")

    if n_rois == 0:
        return pd.DataFrame({col: pd.Series(dtype="float64") for col in FEATURE_COLUMNS})

    # Neuropil-correct then compute dF/F0 for the full population.
    F_f64 = F.astype(np.float64)
    Fneu_f64 = Fneu.astype(np.float64)
    F_corr = F_f64 - neucoeff * Fneu_f64
    dff = _quick_dff(F_corr.astype(np.float32))

    dff_f64 = dff.astype(np.float64)

    # Batch-compute vectorized correlation features (all ROIs at once).
    fneu_corr_all = _fneu_corr_batch(dff_f64, Fneu_f64)
    max_pw_corr_all = _max_pairwise_corr_batch(dff_f64)

    rows: list[dict[str, float]] = []
    for i in range(n_rois):
        s = stat[i]
        row: dict[str, float] = {}

        # Shape features (direct from stat)
        for key in _STAT_FEATURE_KEYS:
            row[key] = float(s.get(key, _STAT_DEFAULTS[key]))

        # Derived shape features
        npix = max(int(s.get("npix", _STAT_DEFAULTS["npix"])), 1)
        npix_soma = int(s.get("npix_soma", _STAT_DEFAULTS["npix_soma"]))
        row["soma_crop_fraction"] = npix_soma / npix

        npix_norm = float(s.get("npix_norm", _STAT_DEFAULTS["npix_norm"]))
        npix_norm_no_crop = float(s.get("npix_norm_no_crop", _STAT_DEFAULTS["npix_norm_no_crop"]))
        denom = max(abs(npix_norm_no_crop), 1e-6)
        row["npix_norm_ratio"] = npix_norm / denom

        overlap = s.get("overlap", None)
        if overlap is not None and hasattr(overlap, "__len__") and len(overlap) > 0:
            row["overlap_fraction"] = float(np.sum(overlap)) / len(overlap)
        else:
            row["overlap_fraction"] = 0.0

        row["eccentricity"] = _eccentricity(s)
        row["lam_cv"] = _lam_cv(s)

        n_comp, n_branch, roughness = _mask_topology(s)
        row["n_connected_components"] = float(n_comp)
        row["n_branch_points"] = float(n_branch)
        row["boundary_roughness"] = roughness

        # Activity features (per-ROI)
        trace = dff_f64[i]
        row["peak_to_noise_dff"] = _peak_to_noise(trace)
        row["autocorr_halfwidth_s"] = _autocorr_halfwidth_s(trace, fps)
        row["fneu_corr"] = float(fneu_corr_all[i])
        row["kurtosis"] = _kurtosis(trace)
        row["signal_to_background"] = _signal_to_background(F_f64[i], Fneu_f64[i])
        row["event_rate"] = _event_rate(trace, fps)
        row["derivative_skew"] = _derivative_skew(trace)
        row["trace_sparsity"] = _trace_sparsity(trace)
        row["power_slope"] = _power_slope(trace, fps)
        row["max_pairwise_corr"] = float(max_pw_corr_all[i])

        rows.append(row)

    df = pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))
    log.info(
        "Extracted soma features for %d ROIs (n_frames=%d, fps=%.2f Hz, %d features)",
        n_rois,
        n_frames,
        fps,
        len(FEATURE_COLUMNS),
    )
    return df
