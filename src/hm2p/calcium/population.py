"""Population-level calcium analysis without ROI detection.

Analyses the full imaging movie or Suite2p spatial components to extract
population activity signals, correlate with behaviour, and compare
CASCADE spike inference with raw fluorescence.

Methods:
1. Population signals from SVD/PCA of the imaging movie
2. Frame-to-frame correlation as a brain-state proxy
3. Movement regression on pixel/component time series
4. CASCADE vs dF/F comparison (Rupprecht et al. 2021)

References:
    Stringer C et al. 2026. "Extracting large-scale neural activity with
    Suite2p." (PCA of motion-corrected movie for spatial components)

    Rupprecht P et al. 2021. "A database and deep learning toolbox for
    noise-optimized, generalized spike inference from calcium imaging."
    Nature Neuroscience 24:1324-1337. doi:10.1038/s41593-021-00895-5
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

# ── 1. Population signals from SVD/PCA ─────────────────────────────────


def compute_population_signals(
    F: np.ndarray,
    n_components: int = 10,
) -> dict:
    """Extract population activity signals via PCA on the ROI matrix.

    Operates on the fluorescence matrix F (n_rois × n_frames) without
    knowing which pixels are cells. The top PCs capture dominant
    co-activation patterns.

    Args:
        F: (n_rois, n_frames) raw or corrected fluorescence.
        n_components: Number of PCs to extract.

    Returns:
        Dict with:
            components — (n_components, n_frames) PC time courses
            explained_variance_ratio — (n_components,) fraction of variance
            mean_activity — (n_frames,) mean fluorescence across all ROIs
    """
    from sklearn.decomposition import PCA

    mean_activity = np.nanmean(F, axis=0)

    # Centre each ROI
    F_centered = F - np.nanmean(F, axis=1, keepdims=True)

    # Handle NaN
    nan_mask = np.isnan(F_centered)
    if nan_mask.any():
        F_centered = np.nan_to_num(F_centered, nan=0.0)

    n_comp = min(n_components, F.shape[0], F.shape[1])
    pca = PCA(n_components=n_comp)
    components = pca.fit_transform(F_centered.T).T  # (n_comp, n_frames)

    return {
        "components": components.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "mean_activity": mean_activity.astype(np.float32),
    }


# ── 2. Frame-to-frame correlation ──────────────────────────────────────


def frame_correlation(
    F: np.ndarray,
    lag: int = 1,
) -> np.ndarray:
    """Compute frame-to-frame Pearson correlation as a brain-state proxy.

    High correlation between adjacent frames indicates stable population
    state; drops in correlation indicate state transitions.

    Args:
        F: (n_rois, n_frames) fluorescence matrix.
        lag: Frame lag (default 1 = consecutive frames).

    Returns:
        (n_frames - lag,) correlation values.
    """
    n_frames = F.shape[1]
    corrs = np.empty(n_frames - lag, dtype=np.float32)

    for i in range(n_frames - lag):
        a = F[:, i]
        b = F[:, i + lag]
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 3 or np.std(a[valid]) < 1e-10 or np.std(b[valid]) < 1e-10:
            corrs[i] = np.nan
        else:
            corrs[i] = float(spearmanr(a[valid], b[valid])[0])

    return corrs


# ── 3. Movement regression on neural signals ───────────────────────────


def regress_movement(
    signals: np.ndarray,
    speed: np.ndarray,
    ahv: np.ndarray | None = None,
    acceleration: np.ndarray | None = None,
) -> dict:
    """Regress movement variables against neural signals.

    Tests how much variance in each signal (ROI or PC) is explained by
    movement, following Zagha et al. 2022 recommendations.

    Args:
        signals: (n_signals, n_frames) neural signals (ROIs, PCs, or pixels).
        speed: (n_frames,) locomotion speed.
        ahv: (n_frames,) angular head velocity, optional.
        acceleration: (n_frames,) acceleration, optional.

    Returns:
        Dict with:
            r_squared — (n_signals,) R² of movement model per signal
            speed_corr — (n_signals,) Spearman correlation with speed
            ahv_corr — (n_signals,) Spearman correlation with AHV (or None)
            mean_r_squared — float, mean R² across all signals
    """
    n_signals, n_frames = signals.shape

    # Build regressor matrix
    regressors = [speed]
    if ahv is not None:
        regressors.append(np.abs(ahv))  # absolute AHV (direction-invariant)
    if acceleration is not None:
        regressors.append(acceleration)

    X = np.column_stack(regressors)  # (n_frames, n_regressors)

    # Mask NaN
    valid = np.all(np.isfinite(X), axis=1)

    r_squared = np.full(n_signals, np.nan)
    speed_corr = np.full(n_signals, np.nan)
    ahv_corr = np.full(n_signals, np.nan) if ahv is not None else None

    for i in range(n_signals):
        sig = signals[i]
        v = valid & np.isfinite(sig)
        if v.sum() < 10:
            continue

        # R² from OLS
        Xv = X[v]
        yv = sig[v]
        Xv_aug = np.column_stack([Xv, np.ones(v.sum())])  # add intercept
        try:
            beta, _, _, _ = np.linalg.lstsq(Xv_aug, yv, rcond=None)
            y_pred = Xv_aug @ beta
            ss_res = np.sum((yv - y_pred) ** 2)
            ss_tot = np.sum((yv - yv.mean()) ** 2)
            r_squared[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except np.linalg.LinAlgError:
            pass

        # Spearman correlations
        if np.std(sig[v]) > 0 and np.std(speed[v]) > 0:
            speed_corr[i] = float(spearmanr(sig[v], speed[v])[0])
        if ahv is not None and ahv_corr is not None and np.std(np.abs(ahv[v])) > 0:
            ahv_corr[i] = float(spearmanr(sig[v], np.abs(ahv[v]))[0])

    return {
        "r_squared": r_squared,
        "speed_corr": speed_corr,
        "ahv_corr": ahv_corr,
        "mean_r_squared": float(np.nanmean(r_squared)),
    }


# ── 4. CASCADE vs dF/F comparison (Rupprecht et al. 2021) ─────────────


def compare_spikes_to_fluorescence(
    dff: np.ndarray,
    spikes: np.ndarray,
    deconv_norm: np.ndarray | None = None,
    fps: float = 9.8,
) -> dict:
    """Compare CASCADE spike rates with dF/F and deconvolved signals.

    Metrics following Rupprecht et al. 2021 (CASCADE paper):
    - Per-ROI Spearman correlation between spikes and dF/F
    - Per-ROI Spearman correlation between spikes and deconv
    - Temporal offset at peak cross-correlation (spike timing vs dF/F)
    - Event-triggered average dF/F aligned to spike onsets

    Args:
        dff: (n_rois, n_frames) dF/F0 traces.
        spikes: (n_rois, n_frames) CASCADE spike rates.
        deconv_norm: (n_rois, n_frames) normalized deconv, optional.
        fps: Imaging frame rate.

    Returns:
        Dict with per-ROI and population-level comparison metrics.
    """
    n_rois, n_frames = dff.shape

    corr_dff = np.full(n_rois, np.nan)
    corr_deconv = np.full(n_rois, np.nan)
    peak_lag_frames = np.full(n_rois, np.nan)

    for i in range(n_rois):
        d = dff[i]
        s = spikes[i]

        valid = np.isfinite(d) & np.isfinite(s)
        if valid.sum() < 10 or np.std(s[valid]) < 1e-10:
            continue

        # Spearman correlation
        if np.std(d[valid]) > 0:
            corr_dff[i] = float(spearmanr(d[valid], s[valid])[0])

        # Deconv correlation
        if deconv_norm is not None:
            dc = deconv_norm[i]
            v2 = valid & np.isfinite(dc)
            if v2.sum() > 10 and np.std(dc[v2]) > 0:
                corr_deconv[i] = float(spearmanr(dc[v2], s[v2])[0])

        # Cross-correlation to find temporal offset
        # (spikes should lead dF/F due to indicator dynamics)
        d_norm = (d[valid] - np.mean(d[valid])) / (np.std(d[valid]) + 1e-10)
        s_norm = (s[valid] - np.mean(s[valid])) / (np.std(s[valid]) + 1e-10)
        max_lag = min(int(2.0 * fps), len(d_norm) // 4)  # ±2 seconds
        if max_lag > 0:
            xcorr = np.correlate(d_norm, s_norm, mode="full")
            mid = len(d_norm) - 1
            xcorr_window = xcorr[mid - max_lag : mid + max_lag + 1]
            peak_idx = np.argmax(xcorr_window)
            peak_lag_frames[i] = float(peak_idx - max_lag)

    # Event-triggered average: average dF/F aligned to spike peaks
    eta_window = int(3.0 * fps)  # ±3 seconds
    eta_traces = []
    for i in range(min(n_rois, 50)):  # sample up to 50 ROIs
        s = spikes[i]
        d = dff[i]
        if np.nanmax(s) < 1e-10:
            continue
        # Find spike peaks (local maxima above 50th percentile)
        threshold = np.nanpercentile(s[s > 0], 50) if (s > 0).any() else 0
        peaks = []
        for j in range(1, n_frames - 1):
            if (
                s[j] > threshold
                and s[j] > s[j - 1]
                and s[j] > s[j + 1]
                and eta_window <= j < n_frames - eta_window
            ):
                peaks.append(j)
        if len(peaks) < 3:
            continue
        # Average dF/F around peaks
        snippets = np.array([d[p - eta_window : p + eta_window + 1] for p in peaks[:100]])
        eta_traces.append(np.nanmean(snippets, axis=0))

    eta_mean = np.nanmean(np.array(eta_traces), axis=0) if eta_traces else None
    eta_time = np.arange(-eta_window, eta_window + 1) / fps if eta_mean is not None else None

    return {
        "corr_dff_spikes": corr_dff,
        "corr_deconv_spikes": corr_deconv,
        "peak_lag_frames": peak_lag_frames,
        "peak_lag_seconds": peak_lag_frames / fps,
        "mean_corr_dff": float(np.nanmean(corr_dff)),
        "mean_corr_deconv": float(np.nanmean(corr_deconv)),
        "mean_lag_s": float(np.nanmean(peak_lag_frames / fps)),
        "event_triggered_avg": eta_mean,
        "event_triggered_time": eta_time,
    }
