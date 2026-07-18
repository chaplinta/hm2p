"""Neuropil signal analysis — local network input from Fneu.

The neuropil signal (Fneu) represents aggregate fluorescence from axonal
and dendritic processes surrounding each ROI. It reflects local network
input activity (Kerr et al. 2005, PNAS) rather than individual cell output.

Analysing the neuropil signal reveals brain-state modulation (arousal,
movement, visual input) at the population level, independent of single-cell
HD tuning.

References:
    Kerr et al. 2005. PNAS 102(39):14063-14068.
    Dipoppa et al. 2018. Neuron 98(3):602-615.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr


def compute_mean_neuropil(
    Fneu: np.ndarray,
    cell_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Mean neuropil signal across all (or accepted) ROIs.

    Args:
        Fneu: (n_rois, n_frames) neuropil fluorescence.
        cell_mask: (n_rois,) bool — if provided, average only accepted ROIs.

    Returns:
        (n_frames,) mean neuropil signal.
    """
    if cell_mask is not None:
        Fneu = Fneu[cell_mask]
    return np.nanmean(Fneu, axis=0).astype(np.float32)


def compute_neuropil_ratio(
    F: np.ndarray,
    Fneu: np.ndarray,
) -> np.ndarray:
    """Neuropil-to-soma fluorescence ratio per ROI.

    Higher ratio = more neuropil contamination relative to soma signal.

    Args:
        F: (n_rois, n_frames) raw fluorescence.
        Fneu: (n_rois, n_frames) neuropil fluorescence.

    Returns:
        (n_rois,) ratio of mean(Fneu) / mean(F).
    """
    mean_f = np.nanmean(F, axis=1)
    mean_fneu = np.nanmean(Fneu, axis=1)
    ratio = np.where(mean_f > 0, mean_fneu / mean_f, np.nan)
    return ratio.astype(np.float32)


def neuropil_behaviour_correlation(
    mean_fneu: np.ndarray,
    speed: np.ndarray,
    ahv: np.ndarray | None = None,
    light_on: np.ndarray | None = None,
    active_mask: np.ndarray | None = None,
) -> dict:
    """Correlate mean neuropil signal with behavioural variables.

    Args:
        mean_fneu: (n_frames,) mean neuropil signal.
        speed: (n_frames,) speed in cm/s.
        ahv: (n_frames,) angular head velocity, optional.
        light_on: (n_frames,) bool, optional.
        active_mask: (n_frames,) bool — valid frames.

    Returns:
        Dict with Spearman correlations and condition means.
    """
    n = min(len(mean_fneu), len(speed))
    fneu = mean_fneu[:n]
    spd = speed[:n]

    if active_mask is not None:
        mask = active_mask[:n]
    else:
        mask = np.ones(n, dtype=bool)

    result = {}

    # Speed correlation
    v = mask & np.isfinite(fneu) & np.isfinite(spd)
    if v.sum() > 10:
        r, p = spearmanr(fneu[v], spd[v])
        result["speed_corr"] = float(r)
        result["speed_p"] = float(p)

    # AHV correlation
    if ahv is not None:
        ahv_abs = np.abs(ahv[:n])
        v2 = mask & np.isfinite(fneu) & np.isfinite(ahv_abs)
        if v2.sum() > 10:
            r, p = spearmanr(fneu[v2], ahv_abs[v2])
            result["ahv_corr"] = float(r)
            result["ahv_p"] = float(p)

    # Light vs dark
    if light_on is not None:
        lt = light_on[:n].astype(bool) & mask
        dk = ~light_on[:n].astype(bool) & mask
        if lt.sum() > 10 and dk.sum() > 10:
            result["mean_fneu_light"] = float(np.nanmean(fneu[lt]))
            result["mean_fneu_dark"] = float(np.nanmean(fneu[dk]))
            U, p = mannwhitneyu(fneu[lt], fneu[dk], alternative="two-sided")
            result["light_dark_p"] = float(p)
            denom = result["mean_fneu_light"] + result["mean_fneu_dark"]
            result["light_mod_index"] = (
                (result["mean_fneu_light"] - result["mean_fneu_dark"]) / denom
                if denom > 0
                else 0.0
            )

    # Moving vs stationary
    moving = mask & (spd >= 2.5)
    stationary = mask & (spd < 2.5)
    if moving.sum() > 10 and stationary.sum() > 10:
        result["mean_fneu_moving"] = float(np.nanmean(fneu[moving]))
        result["mean_fneu_stationary"] = float(np.nanmean(fneu[stationary]))
        denom = result["mean_fneu_moving"] + result["mean_fneu_stationary"]
        result["movement_mod_index"] = (
            (result["mean_fneu_moving"] - result["mean_fneu_stationary"]) / denom
            if denom > 0
            else 0.0
        )

    return result


def neuropil_soma_correlation(
    dff: np.ndarray,
    Fneu: np.ndarray,
) -> np.ndarray:
    """Per-ROI Spearman correlation between neuropil and somatic dF/F.

    High correlation = somatic signal dominated by shared input.
    Low correlation = cell has independent activity.

    Args:
        dff: (n_rois, n_frames) somatic dF/F0.
        Fneu: (n_rois, n_frames) neuropil fluorescence.

    Returns:
        (n_rois,) Spearman correlation values.
    """
    n_rois = dff.shape[0]
    corrs = np.full(n_rois, np.nan, dtype=np.float32)

    for i in range(n_rois):
        d = dff[i]
        f = Fneu[i]
        v = np.isfinite(d) & np.isfinite(f)
        if v.sum() > 10 and np.std(d[v]) > 0 and np.std(f[v]) > 0:
            r, _ = spearmanr(d[v], f[v])
            corrs[i] = float(r)

    return corrs
