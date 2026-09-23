"""Population coupling and noise-correlation analysis for RSP populations.

Quantifies how strongly each ROI is coupled to the rest of the simultaneously
imaged population ("soloist" vs "chorister" axis), how residual (noise)
correlations depend on anatomical distance, and whether a cell leads or lags
the population in time.

Motivation (H9): Penk+ RSP neurons release enkephalin, which acts on delta/mu
opioid receptors expressed by SST/PV interneurons and therefore disinhibits
local circuitry. If that pathway is active, Penk+ cells and Penk-CamKII+ cells
may differ in how tightly their activity is locked to the local population.

All functions are pure numpy/scipy/pandas — no I/O.

References
----------
Okun et al. 2015. "Diverse coupling of neurons to populations in sensory
cortex." Nature 521:511-515. doi:10.1038/nature14273

Notes
-----
Population coupling is reported as a descriptive correlation coefficient; no
parametric significance test is derived from it. Correlation-vs-distance uses
Spearman rank correlation, and residual (noise) correlations reuse the
rank-based :func:`hm2p.analysis.population.pairwise_correlations`.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

from hm2p.analysis.population import pairwise_correlations

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _safe_pearson(a: npt.NDArray[np.floating], b: npt.NDArray[np.floating]) -> float:
    """Pearson correlation that returns NaN instead of warning on degenerate input.

    Parameters
    ----------
    a, b : (n,) float
        Paired samples. Non-finite entries are dropped pairwise.

    Returns
    -------
    float
        Correlation coefficient in [-1, 1], or NaN if either input is constant,
        has fewer than two usable samples, or the lengths differ.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 2:
        return float("nan")

    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 2:
        return float("nan")

    a = a[finite] - a[finite].mean()
    b = b[finite] - b[finite].mean()
    norm_a = float(np.sqrt(a @ a))
    norm_b = float(np.sqrt(b @ b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return float("nan")
    return float(np.clip((a @ b) / (norm_a * norm_b), -1.0, 1.0))


def _zscore_rows(signals: npt.NDArray[np.floating]) -> np.ndarray:
    """Z-score each row (ROI) across time.

    Parameters
    ----------
    signals : (n_rois, n_frames) float

    Returns
    -------
    (n_rois, n_frames) float
        Z-scored traces. Rows with zero or non-finite standard deviation are
        returned as all-zero rows (no division warning is emitted).
    """
    x = np.asarray(signals, dtype=np.float64)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    good = np.isfinite(std) & (std > 0.0)
    out = np.zeros_like(x)
    np.divide(x - mean, np.where(good, std, 1.0), out=out, where=good)
    return out


def _hd_bin_index(hd_deg: npt.NDArray[np.floating], n_bins: int) -> np.ndarray:
    """Map head-direction values onto tuning-curve bin indices.

    Uses the same binning convention as
    :func:`hm2p.analysis.tuning.compute_hd_tuning_curve` so that tuning-curve
    entries can be looked up per frame.

    Parameters
    ----------
    hd_deg : (n,) float
        Head direction in degrees (may be unwrapped).
    n_bins : int
        Number of angular bins spanning [0, 360).

    Returns
    -------
    (n,) int
        Bin index in [0, n_bins).

    Raises
    ------
    ValueError
        If ``n_bins`` is not positive.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}")
    hd_mod = np.mod(np.asarray(hd_deg, dtype=np.float64), 360.0)
    bin_edges = np.linspace(0.0, 360.0, n_bins + 1)
    idx = np.digitize(hd_mod, bin_edges) - 1
    return np.clip(idx, 0, n_bins - 1).astype(int)


# ---------------------------------------------------------------------------
# Population coupling
# ---------------------------------------------------------------------------


def population_coupling(
    signals: npt.NDArray[np.floating],
    exclude_self: bool = True,
) -> np.ndarray:
    """Per-ROI coupling between an ROI trace and the population mean trace.

    Implements the population-coupling measure of Okun et al. (2015): each
    z-scored single-cell trace is correlated with the mean z-scored trace of
    the remaining cells. High values identify "choristers" (strongly locked to
    the local population), low values "soloists".

    Parameters
    ----------
    signals : (n_rois, n_frames) float
        Neural signals (dF/F, deconvolved rate, or events).
    exclude_self : bool
        If True (default) the ROI is excluded from the population mean, which
        removes the trivial self-contribution. If False, the mean over all ROIs
        is used.

    Returns
    -------
    coupling : (n_rois,) float
        Correlation coefficient per ROI. NaN for constant traces and for every
        ROI when fewer than two ROIs or fewer than two frames are supplied.

    Raises
    ------
    ValueError
        If ``signals`` is not 2-D.

    References
    ----------
    Okun et al. 2015. "Diverse coupling of neurons to populations in sensory
    cortex." Nature 521:511-515. doi:10.1038/nature14273
    """
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"signals must be 2-D (n_rois, n_frames); got shape {x.shape}")

    n_rois, n_frames = x.shape
    coupling = np.full(n_rois, np.nan, dtype=np.float64)
    if n_rois < 2 or n_frames < 2:
        return coupling

    z = _zscore_rows(x)
    total = z.sum(axis=0)
    for i in range(n_rois):
        if exclude_self:
            pop = (total - z[i]) / (n_rois - 1)
        else:
            pop = total / n_rois
        coupling[i] = _safe_pearson(z[i], pop)
    return coupling


def population_coupling_by_condition(
    signals: npt.NDArray[np.floating],
    condition_masks: dict[str, npt.NDArray[np.bool_]],
    exclude_self: bool = True,
) -> dict[str, np.ndarray]:
    """Population coupling computed separately within each condition.

    Parameters
    ----------
    signals : (n_rois, n_frames) float
    condition_masks : dict of str to (n_frames,) bool
        Frame masks, e.g. ``{"light": ..., "dark": ...}``.
    exclude_self : bool
        Passed to :func:`population_coupling`.

    Returns
    -------
    dict of str to (n_rois,) float
        Coupling per ROI for each condition. Conditions with fewer than two
        frames yield all-NaN arrays.

    Raises
    ------
    ValueError
        If a mask length does not match the number of frames.
    """
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"signals must be 2-D (n_rois, n_frames); got shape {x.shape}")
    n_rois, n_frames = x.shape

    out: dict[str, np.ndarray] = {}
    for name, mask in condition_masks.items():
        m = np.asarray(mask, dtype=bool)
        if m.shape != (n_frames,):
            raise ValueError(
                f"condition mask '{name}' has shape {m.shape}; expected ({n_frames},)"
            )
        if int(m.sum()) < 2:
            out[name] = np.full(n_rois, np.nan, dtype=np.float64)
        else:
            out[name] = population_coupling(x[:, m], exclude_self=exclude_self)
    return out


# ---------------------------------------------------------------------------
# Noise correlations
# ---------------------------------------------------------------------------


def noise_correlations(
    signals: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_bins: int = 36,
) -> np.ndarray:
    """Pairwise correlations of activity residuals after removing HD tuning.

    Each cell's head-direction tuning curve is evaluated at every frame's HD
    bin and subtracted from its trace; the remaining residual is the
    trial-to-trial variability that is not explained by head direction. Pairwise
    rank correlations of these residuals are the noise correlations.

    Parameters
    ----------
    signals : (n_rois, n_frames) float
    hd_deg : (n_frames,) float
        Head direction in degrees.
    mask : (n_frames,) bool
        Valid frames (e.g. active, not ``bad_behav``).
    n_bins : int
        Number of HD bins used for the tuning-curve prediction.

    Returns
    -------
    noise_corr : (n_rois, n_rois) float
        Symmetric matrix of residual rank correlations with a NaN diagonal.
        Rows/columns for constant residuals are NaN.

    Raises
    ------
    ValueError
        If array shapes are inconsistent.
    """
    from hm2p.analysis.tuning import compute_hd_tuning_curve

    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"signals must be 2-D (n_rois, n_frames); got shape {x.shape}")
    hd = np.asarray(hd_deg, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    n_rois, n_frames = x.shape
    if hd.shape != (n_frames,) or m.shape != (n_frames,):
        raise ValueError(
            "hd_deg and mask must have shape (n_frames,); "
            f"got {hd.shape}, {m.shape} for n_frames={n_frames}"
        )

    out = np.full((n_rois, n_rois), np.nan, dtype=np.float64)
    if n_rois < 2 or int(m.sum()) < 3:
        return out

    bin_idx = _hd_bin_index(hd[m], n_bins)
    residuals = np.zeros((n_rois, int(m.sum())), dtype=np.float64)
    for i in range(n_rois):
        trace = x[i, m]
        tuning_curve, _ = compute_hd_tuning_curve(x[i], hd, m, n_bins=n_bins)
        prediction = tuning_curve[bin_idx]
        # Unoccupied/unsmoothable bins are NaN; fall back to the cell mean so
        # that a single empty bin cannot void the whole residual trace.
        fallback = float(np.nanmean(trace)) if np.isfinite(trace).any() else 0.0
        prediction = np.where(np.isfinite(prediction), prediction, fallback)
        residuals[i] = trace - prediction

    # scipy's spearmanr collapses to a scalar NaN if any input row is constant,
    # so correlate only the usable rows and scatter the result back.
    constant = ~np.isfinite(residuals).all(axis=1) | (residuals.std(axis=1) <= 0.0)
    usable = np.where(~constant)[0]
    if usable.size < 2:
        return out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.asarray(pairwise_correlations(residuals[usable]), dtype=np.float64)

    out[np.ix_(usable, usable)] = corr
    np.fill_diagonal(out, np.nan)
    return out


# ---------------------------------------------------------------------------
# Correlation vs anatomical distance
# ---------------------------------------------------------------------------


def pairwise_distances_um(
    centroids_xy: npt.NDArray[np.floating],
    um_per_pixel: float,
) -> np.ndarray:
    """Euclidean pairwise distances between ROI centroids, in micrometres.

    Parameters
    ----------
    centroids_xy : (n_rois, 2) float
        ROI centroid coordinates in pixels.
    um_per_pixel : float
        Field-of-view scale factor (micrometres per pixel). Must be positive.

    Returns
    -------
    dist : (n_rois, n_rois) float
        Symmetric distance matrix in micrometres with a zero diagonal.

    Raises
    ------
    ValueError
        If ``centroids_xy`` is not (n_rois, 2) or ``um_per_pixel`` is not
        positive and finite.
    """
    xy = np.asarray(centroids_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"centroids_xy must have shape (n_rois, 2); got {xy.shape}")
    if not np.isfinite(um_per_pixel) or um_per_pixel <= 0:
        raise ValueError(f"um_per_pixel must be positive and finite; got {um_per_pixel}")

    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1)) * float(um_per_pixel)
    # Enforce exact symmetry and zero diagonal against floating-point drift.
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


def correlation_vs_distance(
    corr_matrix: npt.NDArray[np.floating],
    dist_matrix: npt.NDArray[np.floating],
    distance_bins_um: npt.NDArray[np.floating],
) -> dict:
    """Relate pairwise correlation to anatomical pair distance.

    Parameters
    ----------
    corr_matrix : (n_rois, n_rois) float
        Pairwise correlations (e.g. output of :func:`noise_correlations`).
    dist_matrix : (n_rois, n_rois) float
        Pairwise distances in micrometres.
    distance_bins_um : (n_edges,) float
        Monotonically increasing bin edges in micrometres.

    Returns
    -------
    dict
        ``"bin_edges_um"`` — the supplied bin edges.
        ``"bin_centers_um"`` — (n_edges - 1,) bin centres.
        ``"mean_corr"`` — (n_edges - 1,) mean correlation per bin (NaN if empty).
        ``"median_corr"`` — (n_edges - 1,) median correlation per bin.
        ``"n_pairs_per_bin"`` — (n_edges - 1,) int pair counts.
        ``"distances_um"`` — (n_pairs,) distances of the usable pairs.
        ``"correlations"`` — (n_pairs,) correlations of the usable pairs.
        ``"n_pairs"`` — number of usable upper-triangle pairs.
        ``"spearman_rho"`` — Spearman rank correlation across all pairs.
        ``"spearman_p"`` — two-sided p-value for ``spearman_rho``.

    Raises
    ------
    ValueError
        If the matrices are not square and of equal shape, or fewer than two
        bin edges are given.
    """
    from scipy.stats import spearmanr

    corr = np.asarray(corr_matrix, dtype=np.float64)
    dist = np.asarray(dist_matrix, dtype=np.float64)
    edges = np.asarray(distance_bins_um, dtype=np.float64)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"corr_matrix must be square; got {corr.shape}")
    if dist.shape != corr.shape:
        raise ValueError(f"dist_matrix shape {dist.shape} != corr_matrix shape {corr.shape}")
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("distance_bins_um must be a 1-D array of at least two edges")

    n_bins = edges.size - 1
    centers = 0.5 * (edges[:-1] + edges[1:])

    triu = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    usable = triu & np.isfinite(corr) & np.isfinite(dist)
    pair_corr = corr[usable]
    pair_dist = dist[usable]

    mean_corr = np.full(n_bins, np.nan, dtype=np.float64)
    median_corr = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=int)

    if pair_corr.size > 0:
        bin_idx = np.digitize(pair_dist, edges) - 1
        for b in range(n_bins):
            sel = bin_idx == b
            counts[b] = int(sel.sum())
            if counts[b] > 0:
                mean_corr[b] = float(np.mean(pair_corr[sel]))
                median_corr[b] = float(np.median(pair_corr[sel]))

    if pair_corr.size >= 3 and np.std(pair_corr) > 0 and np.std(pair_dist) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, p_value = spearmanr(pair_dist, pair_corr)
        rho = float(rho)
        p_value = float(p_value)
    else:
        rho = float("nan")
        p_value = float("nan")

    return {
        "bin_edges_um": edges,
        "bin_centers_um": centers,
        "mean_corr": mean_corr,
        "median_corr": median_corr,
        "n_pairs_per_bin": counts,
        "distances_um": pair_dist,
        "correlations": pair_corr,
        "n_pairs": int(pair_corr.size),
        "spearman_rho": rho,
        "spearman_p": p_value,
    }


# ---------------------------------------------------------------------------
# Lagged cross-correlation with the population
# ---------------------------------------------------------------------------


def lagged_population_crosscorr(
    signals: npt.NDArray[np.floating],
    max_lag_frames: int = 10,
) -> dict:
    """Cross-correlation between each ROI and the mean of the other ROIs.

    The cross-correlation at lag ``l`` is the correlation between the ROI at
    frame ``t`` and the population mean at frame ``t + l``. A positive peak lag
    therefore means the population follows the cell (the cell leads).

    Parameters
    ----------
    signals : (n_rois, n_frames) float
    max_lag_frames : int
        Maximum absolute lag, in frames. Must be non-negative.

    Returns
    -------
    dict
        ``"lags"`` — (2 * max_lag + 1,) int lags in frames.
        ``"xcorr"`` — (n_rois, 2 * max_lag + 1) correlation per ROI and lag.
        ``"peak_lag"`` — (n_rois,) float lag of the maximum correlation (NaN if
        undefined).
        ``"asymmetry"`` — (n_rois,) float sum of positive-lag minus sum of
        negative-lag correlations; positive means the cell leads the population.

    Raises
    ------
    ValueError
        If ``signals`` is not 2-D or ``max_lag_frames`` is negative.
    """
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"signals must be 2-D (n_rois, n_frames); got shape {x.shape}")
    if max_lag_frames < 0:
        raise ValueError(f"max_lag_frames must be >= 0; got {max_lag_frames}")

    n_rois, n_frames = x.shape
    lags = np.arange(-max_lag_frames, max_lag_frames + 1, dtype=int)
    xcorr = np.full((n_rois, lags.size), np.nan, dtype=np.float64)
    peak_lag = np.full(n_rois, np.nan, dtype=np.float64)
    asymmetry = np.full(n_rois, np.nan, dtype=np.float64)

    if n_rois < 2 or n_frames < 2:
        return {"lags": lags, "xcorr": xcorr, "peak_lag": peak_lag, "asymmetry": asymmetry}

    z = _zscore_rows(x)
    total = z.sum(axis=0)

    for i in range(n_rois):
        pop = (total - z[i]) / (n_rois - 1)
        for j, lag in enumerate(lags):
            if lag >= 0:
                a = z[i, : n_frames - lag]
                b = pop[lag:]
            else:
                a = z[i, -lag:]
                b = pop[: n_frames + lag]
            xcorr[i, j] = _safe_pearson(a, b)

        row = xcorr[i]
        if np.isfinite(row).any():
            peak_lag[i] = float(lags[int(np.nanargmax(row))])
            pos = row[lags > 0]
            neg = row[lags < 0]
            asymmetry[i] = float(np.nansum(pos) - np.nansum(neg))

    return {"lags": lags, "xcorr": xcorr, "peak_lag": peak_lag, "asymmetry": asymmetry}


# ---------------------------------------------------------------------------
# Session-level summary
# ---------------------------------------------------------------------------


def coupling_summary(
    signals: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    light_on: npt.NDArray[np.bool_],
    max_lag_frames: int = 10,
    n_bins: int = 36,
) -> pd.DataFrame:
    """Per-ROI coupling metrics for one session.

    Parameters
    ----------
    signals : (n_rois, n_frames) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
        Valid frames (e.g. active and not ``bad_behav``).
    light_on : (n_frames,) bool
        Room-light state per frame.
    max_lag_frames : int
        Maximum absolute lag for the cross-correlation.
    n_bins : int
        HD bins for the noise-correlation tuning-curve prediction.

    Returns
    -------
    pandas.DataFrame
        One row per ROI (index named ``"roi"``) with columns
        ``pop_coupling_all``, ``pop_coupling_light``, ``pop_coupling_dark``,
        ``mean_noise_corr``, ``xcorr_peak_lag``, ``xcorr_asymmetry``.

    Raises
    ------
    ValueError
        If array shapes are inconsistent.
    """
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"signals must be 2-D (n_rois, n_frames); got shape {x.shape}")
    n_rois, n_frames = x.shape
    m = np.asarray(mask, dtype=bool)
    light = np.asarray(light_on, dtype=bool)
    hd = np.asarray(hd_deg, dtype=np.float64)
    if m.shape != (n_frames,) or light.shape != (n_frames,) or hd.shape != (n_frames,):
        raise ValueError(
            "hd_deg, mask, and light_on must have shape (n_frames,); "
            f"got {hd.shape}, {m.shape}, {light.shape} for n_frames={n_frames}"
        )

    coupling = population_coupling_by_condition(
        x,
        {"all": m, "light": m & light, "dark": m & ~light},
    )

    noise = noise_correlations(x, hd, m, n_bins=n_bins)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_noise = np.nanmean(noise, axis=1) if n_rois > 0 else np.zeros(0)
    mean_noise = np.asarray(mean_noise, dtype=np.float64)

    xc = lagged_population_crosscorr(x[:, m], max_lag_frames=max_lag_frames)

    frame = pd.DataFrame(
        {
            "pop_coupling_all": coupling["all"],
            "pop_coupling_light": coupling["light"],
            "pop_coupling_dark": coupling["dark"],
            "mean_noise_corr": mean_noise,
            "xcorr_peak_lag": xc["peak_lag"],
            "xcorr_asymmetry": xc["asymmetry"],
        }
    )
    frame.index = pd.RangeIndex(n_rois, name="roi")
    return frame
