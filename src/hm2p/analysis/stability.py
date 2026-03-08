"""Temporal stability of HD tuning.

Tests whether head direction tuning is stable across time periods within
a session. Key analysis for light/dark manipulation: does HD tuning drift
when visual cues are removed?

All functions are pure numpy — no I/O.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hm2p.analysis.comparison import tuning_curve_correlation
from hm2p.analysis.tuning import (
    compute_hd_tuning_curve,
    mean_vector_length,
    preferred_direction,
)


def split_temporal_halves(
    signal: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
) -> dict:
    """Compare HD tuning between first and second half of session.

    Parameters
    ----------
    signal : (n_frames,) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
    n_bins : int
    smoothing_sigma_deg : float

    Returns
    -------
    dict
        ``"correlation"`` — Pearson r between half tuning curves.
        ``"pd_shift_deg"`` — preferred direction shift in degrees.
        ``"mvl_half1"``, ``"mvl_half2"`` — MVL of each half.
        ``"tuning_curve_1"``, ``"tuning_curve_2"`` — the two tuning curves.
        ``"bin_centers"`` — bin centres.
    """
    n = len(signal)
    mid = n // 2

    mask1 = mask.copy()
    mask1[mid:] = False
    mask2 = mask.copy()
    mask2[:mid] = False

    tc1, bc = compute_hd_tuning_curve(
        signal, hd_deg, mask1, n_bins=n_bins,
        smoothing_sigma_deg=smoothing_sigma_deg,
    )
    tc2, _ = compute_hd_tuning_curve(
        signal, hd_deg, mask2, n_bins=n_bins,
        smoothing_sigma_deg=smoothing_sigma_deg,
    )

    corr = tuning_curve_correlation(tc1, tc2)
    pd1 = preferred_direction(tc1, bc)
    pd2 = preferred_direction(tc2, bc)
    pd_shift = ((pd2 - pd1 + 180) % 360) - 180

    return {
        "correlation": corr,
        "pd_shift_deg": float(pd_shift),
        "mvl_half1": mean_vector_length(tc1, bc),
        "mvl_half2": mean_vector_length(tc2, bc),
        "tuning_curve_1": tc1,
        "tuning_curve_2": tc2,
        "bin_centers": bc,
    }


def sliding_window_stability(
    signal: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    window_frames: int = 1000,
    step_frames: int = 200,
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
) -> dict:
    """Compute HD tuning metrics over sliding time windows.

    Parameters
    ----------
    signal : (n_frames,) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
    window_frames : int
    step_frames : int
    n_bins : int
    smoothing_sigma_deg : float

    Returns
    -------
    dict
        ``"window_centers"`` — frame indices of window centres.
        ``"mvls"`` — MVL per window.
        ``"preferred_dirs"`` — preferred direction per window.
        ``"n_windows"`` — number of windows.
    """
    n = len(signal)
    centers = []
    mvls = []
    pds = []

    for start in range(0, n - window_frames + 1, step_frames):
        end = start + window_frames
        win_mask = np.zeros_like(mask)
        win_mask[start:end] = mask[start:end]

        if win_mask.sum() < n_bins:
            continue

        tc, bc = compute_hd_tuning_curve(
            signal, hd_deg, win_mask, n_bins=n_bins,
            smoothing_sigma_deg=smoothing_sigma_deg,
        )
        centers.append(start + window_frames // 2)
        mvls.append(mean_vector_length(tc, bc))
        pds.append(preferred_direction(tc, bc))

    return {
        "window_centers": np.array(centers, dtype=int),
        "mvls": np.array(mvls, dtype=np.float64),
        "preferred_dirs": np.array(pds, dtype=np.float64),
        "n_windows": len(centers),
    }


def light_dark_stability(
    signal: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    light_on: npt.NDArray[np.bool_],
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
) -> dict:
    """Compare HD tuning between light-on and light-off epochs.

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
        ``"correlation"`` — Pearson r between light/dark tuning curves.
        ``"pd_shift_deg"`` — preferred direction shift (dark - light).
        ``"mvl_light"``, ``"mvl_dark"`` — MVL in each condition.
        ``"tuning_curve_light"``, ``"tuning_curve_dark"`` — tuning curves.
        ``"bin_centers"`` — bin centres.
    """
    mask_light = mask & light_on
    mask_dark = mask & ~light_on

    tc_light, bc = compute_hd_tuning_curve(
        signal, hd_deg, mask_light, n_bins=n_bins,
        smoothing_sigma_deg=smoothing_sigma_deg,
    )
    tc_dark, _ = compute_hd_tuning_curve(
        signal, hd_deg, mask_dark, n_bins=n_bins,
        smoothing_sigma_deg=smoothing_sigma_deg,
    )

    corr = tuning_curve_correlation(tc_light, tc_dark)
    pd_light = preferred_direction(tc_light, bc)
    pd_dark = preferred_direction(tc_dark, bc)
    pd_shift = ((pd_dark - pd_light + 180) % 360) - 180

    return {
        "correlation": corr,
        "pd_shift_deg": float(pd_shift),
        "mvl_light": mean_vector_length(tc_light, bc),
        "mvl_dark": mean_vector_length(tc_dark, bc),
        "tuning_curve_light": tc_light,
        "tuning_curve_dark": tc_dark,
        "bin_centers": bc,
    }


def drift_per_epoch(
    signal: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    light_on: npt.NDArray[np.bool_],
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
) -> dict:
    """Measure PD drift across sequential light/dark epochs.

    Identifies individual light and dark epochs (contiguous blocks),
    computes tuning curve per epoch, and tracks PD and MVL over time.

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
        ``"epoch_pds"`` — preferred direction per epoch.
        ``"epoch_mvls"`` — MVL per epoch.
        ``"epoch_is_light"`` — bool per epoch.
        ``"cumulative_drift"`` — cumulative PD drift from first epoch.
        ``"n_epochs"`` — number of epochs.
    """
    n = len(signal)
    # Find epoch boundaries
    changes = np.where(np.diff(light_on.astype(int)) != 0)[0] + 1
    boundaries = np.concatenate([[0], changes, [n]])

    centers = []
    pds = []
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
        pds.append(preferred_direction(tc, bc))
        mvls.append(mean_vector_length(tc, bc))
        is_light.append(bool(light_on[start]))

    pds_arr = np.array(pds, dtype=np.float64)
    # Cumulative drift from first epoch
    if len(pds_arr) > 0:
        diffs = np.diff(pds_arr)
        diffs = ((diffs + 180) % 360) - 180  # wrap
        cum_drift = np.concatenate([[0.0], np.cumsum(diffs)])
    else:
        cum_drift = np.array([], dtype=np.float64)

    return {
        "epoch_centers": np.array(centers, dtype=int),
        "epoch_pds": pds_arr,
        "epoch_mvls": np.array(mvls, dtype=np.float64),
        "epoch_is_light": np.array(is_light, dtype=bool),
        "cumulative_drift": cum_drift,
        "n_epochs": len(centers),
    }


def dark_drift_rate(
    signal: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    light_on: npt.NDArray[np.bool_],
    fps: float = 30.0,
    window_frames: int = 300,
    step_frames: int = 100,
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
) -> dict:
    """Estimate drift rate during dark epochs.

    Uses sliding windows within dark epochs to compute PD change per second.

    Parameters
    ----------
    signal : (n_frames,) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
    light_on : (n_frames,) bool
    fps : float
    window_frames : int
    step_frames : int
    n_bins : int
    smoothing_sigma_deg : float

    Returns
    -------
    dict
        ``"dark_drift_deg_per_s"`` — mean absolute drift rate in dark.
        ``"light_drift_deg_per_s"`` — mean absolute drift rate in light.
        ``"dark_pds"`` — PDs during dark windows.
        ``"light_pds"`` — PDs during light windows.
        ``"dark_times_s"`` — time of each dark window centre.
        ``"light_times_s"`` — time of each light window centre.
    """
    n = len(signal)
    dark_centers = []
    dark_pds = []
    light_centers = []
    light_pds = []

    for start in range(0, n - window_frames + 1, step_frames):
        end = start + window_frames
        win_mask = np.zeros_like(mask)
        win_mask[start:end] = mask[start:end]

        if win_mask.sum() < n_bins:
            continue

        # Determine if mostly light or dark
        light_frac = light_on[start:end].mean()

        tc, bc = compute_hd_tuning_curve(
            signal, hd_deg, win_mask, n_bins=n_bins,
            smoothing_sigma_deg=smoothing_sigma_deg,
        )
        pd = preferred_direction(tc, bc)
        center = (start + end) // 2

        if light_frac > 0.5:
            light_centers.append(center)
            light_pds.append(pd)
        else:
            dark_centers.append(center)
            dark_pds.append(pd)

    def _drift_rate(pds_list, centers_list):
        if len(pds_list) < 2:
            return 0.0
        pds = np.array(pds_list)
        times = np.array(centers_list) / fps
        diffs = np.diff(pds)
        diffs = ((diffs + 180) % 360) - 180
        dt = np.diff(times)
        dt[dt == 0] = 1e-10
        rates = np.abs(diffs) / dt
        return float(np.mean(rates))

    return {
        "dark_drift_deg_per_s": _drift_rate(dark_pds, dark_centers),
        "light_drift_deg_per_s": _drift_rate(light_pds, light_centers),
        "dark_pds": np.array(dark_pds, dtype=np.float64),
        "light_pds": np.array(light_pds, dtype=np.float64),
        "dark_times_s": np.array(dark_centers, dtype=np.float64) / fps,
        "light_times_s": np.array(light_centers, dtype=np.float64) / fps,
    }
