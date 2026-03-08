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
