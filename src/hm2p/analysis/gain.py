"""Gain modulation analysis for HD cells.

Compares response amplitude, dynamic range, and peak firing between
conditions (light vs dark). Gain changes indicate that visual input
modulates the amplitude of HD tuning, distinct from PD drift.

All functions pure numpy — no I/O.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length


def gain_modulation_index(
    signal: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    light_on: npt.NDArray[np.bool_],
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
) -> dict:
    """Compute gain modulation between light and dark.

    Gain modulation index = (peak_light - peak_dark) / (peak_light + peak_dark).
    Positive = higher gain in light. Near zero = no gain change.

    Parameters
    ----------
    signal : (n_frames,) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
    light_on : (n_frames,) bool
    n_bins : int
    smoothing_sigma_deg : float

    Returns
    -------
    dict
        ``"gain_index"`` — gain modulation index [-1, 1].
        ``"peak_light"`` — peak tuning curve value in light.
        ``"peak_dark"`` — peak tuning curve value in dark.
        ``"dynamic_range_light"`` — max - min of tuning curve in light.
        ``"dynamic_range_dark"`` — max - min of tuning curve in dark.
        ``"mean_rate_light"`` — mean signal in light.
        ``"mean_rate_dark"`` — mean signal in dark.
    """
    mask_light = mask & light_on
    mask_dark = mask & ~light_on

    tc_light, _ = compute_hd_tuning_curve(
        signal, hd_deg, mask_light, n_bins=n_bins,
        smoothing_sigma_deg=smoothing_sigma_deg,
    )
    tc_dark, _ = compute_hd_tuning_curve(
        signal, hd_deg, mask_dark, n_bins=n_bins,
        smoothing_sigma_deg=smoothing_sigma_deg,
    )

    # Handle NaN tuning curves (no data in condition)
    peak_light = float(np.nanmax(tc_light)) if not np.all(np.isnan(tc_light)) else 0.0
    peak_dark = float(np.nanmax(tc_dark)) if not np.all(np.isnan(tc_dark)) else 0.0

    denom = peak_light + peak_dark
    gain_index = (peak_light - peak_dark) / denom if denom > 0 else 0.0

    dr_light = float(np.nanmax(tc_light) - np.nanmin(tc_light)) if not np.all(np.isnan(tc_light)) else 0.0
    dr_dark = float(np.nanmax(tc_dark) - np.nanmin(tc_dark)) if not np.all(np.isnan(tc_dark)) else 0.0

    mean_light = float(np.mean(signal[mask_light])) if mask_light.any() else 0.0
    mean_dark = float(np.mean(signal[mask_dark])) if mask_dark.any() else 0.0

    return {
        "gain_index": float(gain_index),
        "peak_light": peak_light,
        "peak_dark": peak_dark,
        "dynamic_range_light": dr_light,
        "dynamic_range_dark": dr_dark,
        "mean_rate_light": mean_light,
        "mean_rate_dark": mean_dark,
    }


def population_gain_modulation(
    signals: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    light_on: npt.NDArray[np.bool_],
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
) -> list[dict]:
    """Compute gain modulation for all cells.

    Parameters
    ----------
    signals : (n_cells, n_frames) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
    light_on : (n_frames,) bool
    n_bins : int
    smoothing_sigma_deg : float

    Returns
    -------
    list of dict
        Per-cell gain modulation results from :func:`gain_modulation_index`.
    """
    n_cells = signals.shape[0]
    results = []
    for i in range(n_cells):
        result = gain_modulation_index(
            signals[i], hd_deg, mask, light_on,
            n_bins=n_bins, smoothing_sigma_deg=smoothing_sigma_deg,
        )
        results.append(result)
    return results


def epoch_gain_tracking(
    signal: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    light_on: npt.NDArray[np.bool_],
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
) -> dict:
    """Track peak response and dynamic range across light/dark epochs.

    Parameters
    ----------
    signal : (n_frames,) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
    light_on : (n_frames,) bool
    n_bins : int
    smoothing_sigma_deg : float

    Returns
    -------
    dict
        ``"epoch_centers"`` — frame index at centre of each epoch.
        ``"epoch_peaks"`` — peak tuning curve value per epoch.
        ``"epoch_dynamic_ranges"`` — dynamic range per epoch.
        ``"epoch_mvls"`` — MVL per epoch.
        ``"epoch_is_light"`` — bool per epoch.
        ``"n_epochs"`` — number of epochs.
    """
    n = len(signal)
    changes = np.where(np.diff(light_on.astype(int)) != 0)[0] + 1
    boundaries = np.concatenate([[0], changes, [n]])

    centers = []
    peaks = []
    drs = []
    mvls = []
    is_light = []

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end - start < n_bins:
            continue

        epoch_mask = np.zeros_like(mask)
        epoch_mask[start:end] = mask[start:end]

        if epoch_mask.sum() < n_bins:
            continue

        tc, bc = compute_hd_tuning_curve(
            signal, hd_deg, epoch_mask, n_bins=n_bins,
            smoothing_sigma_deg=smoothing_sigma_deg,
        )
        centers.append((start + end) // 2)
        peaks.append(float(np.nanmax(tc)))
        drs.append(float(np.nanmax(tc) - np.nanmin(tc)))
        mvls.append(mean_vector_length(tc, bc))
        is_light.append(bool(light_on[start]))

    return {
        "epoch_centers": np.array(centers, dtype=int),
        "epoch_peaks": np.array(peaks, dtype=np.float64),
        "epoch_dynamic_ranges": np.array(drs, dtype=np.float64),
        "epoch_mvls": np.array(mvls, dtype=np.float64),
        "epoch_is_light": np.array(is_light, dtype=bool),
        "n_epochs": len(centers),
    }
