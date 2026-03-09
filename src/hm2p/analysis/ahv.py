"""Angular head velocity (AHV) analysis for HD cells.

Computes AHV tuning curves and statistics. HD cells in RSP can show
AHV modulation (anticipatory time delay, AHV tuning). Taube et al. (1990).

All functions are pure numpy — no I/O.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_ahv(
    hd_deg: npt.NDArray[np.floating],
    fps: float = 30.0,
    smoothing_frames: int = 3,
) -> np.ndarray:
    """Compute angular head velocity from head direction time series.

    Parameters
    ----------
    hd_deg : (n_frames,) float
        Head direction in degrees (may be unwrapped).
    fps : float
        Sampling rate.
    smoothing_frames : int
        Gaussian smoothing window for HD before differentiation.

    Returns
    -------
    ahv : (n_frames,) float
        Angular head velocity in deg/s. Positive = CW, negative = CCW.
        First and last frames are set to 0.
    """
    hd = np.asarray(hd_deg, dtype=np.float64)

    # Smooth HD
    if smoothing_frames > 1:
        from scipy.ndimage import gaussian_filter1d
        hd = gaussian_filter1d(hd, sigma=smoothing_frames / 3)

    # Compute angular difference (handle wrapping)
    diff = np.diff(hd)
    diff = ((diff + 180) % 360) - 180
    ahv = diff * fps

    # Pad to match input length
    ahv = np.concatenate([[0], ahv])
    return ahv


def ahv_tuning_curve(
    signal: npt.NDArray[np.floating],
    ahv_deg_s: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_bins: int = 30,
    max_ahv: float = 600.0,
    smoothing_sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute AHV tuning curve (signal vs angular head velocity).

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal.
    ahv_deg_s : (n_frames,) float
        Angular head velocity in deg/s.
    mask : (n_frames,) bool
        Valid frames.
    n_bins : int
        Number of AHV bins (spanning -max_ahv to +max_ahv).
    max_ahv : float
        Maximum AHV to include (deg/s).
    smoothing_sigma : float
        Gaussian smoothing sigma in bins.

    Returns
    -------
    tuning_curve : (n_bins,) float
        Mean signal per AHV bin.
    bin_centers : (n_bins,) float
        AHV bin centres in deg/s.
    """
    signal = np.asarray(signal, dtype=np.float64)
    ahv = np.asarray(ahv_deg_s, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    # Filter by max AHV
    valid = mask & (np.abs(ahv) <= max_ahv)

    bin_edges = np.linspace(-max_ahv, max_ahv, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_idx = np.clip(np.digitize(ahv[valid], bin_edges) - 1, 0, n_bins - 1)

    signal_sum = np.zeros(n_bins, dtype=np.float64)
    occupancy = np.zeros(n_bins, dtype=np.float64)

    np.add.at(signal_sum, bin_idx, signal[valid])
    np.add.at(occupancy, bin_idx, 1.0)

    tuning_curve = np.full(n_bins, np.nan, dtype=np.float64)
    occupied = occupancy > 0
    tuning_curve[occupied] = signal_sum[occupied] / occupancy[occupied]

    if smoothing_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        tc_filled = np.where(np.isnan(tuning_curve), 0, tuning_curve)
        weight = np.where(np.isnan(tuning_curve), 0, 1.0)
        tc_s = gaussian_filter1d(tc_filled, sigma=smoothing_sigma)
        w_s = gaussian_filter1d(weight, sigma=smoothing_sigma)
        result = np.full_like(tuning_curve, np.nan)
        valid_bins = w_s > 1e-12
        result[valid_bins] = tc_s[valid_bins] / w_s[valid_bins]
        tuning_curve = result

    return tuning_curve, bin_centers


def ahv_modulation_index(
    tuning_curve: npt.NDArray[np.floating],
    bin_centers: npt.NDArray[np.floating],
) -> dict:
    """Compute AHV modulation index and asymmetry.

    Parameters
    ----------
    tuning_curve : (n_bins,) float
    bin_centers : (n_bins,) float

    Returns
    -------
    dict
        ``"cw_mean"`` — mean rate during CW rotation.
        ``"ccw_mean"`` — mean rate during CCW rotation.
        ``"asymmetry_index"`` — (CW - CCW) / (CW + CCW), in [-1, 1].
        ``"modulation_depth"`` — max - min of tuning curve.
        ``"preferred_ahv"`` — AHV with peak activity.
    """
    tc = tuning_curve.copy()
    valid = ~np.isnan(tc)

    cw_mask = valid & (bin_centers > 0)
    ccw_mask = valid & (bin_centers < 0)

    cw_mean = float(np.mean(tc[cw_mask])) if cw_mask.any() else 0.0
    ccw_mean = float(np.mean(tc[ccw_mask])) if ccw_mask.any() else 0.0

    denom = cw_mean + ccw_mean
    asym = (cw_mean - ccw_mean) / denom if denom > 0 else 0.0

    tc_valid = tc[valid]
    mod_depth = float(np.max(tc_valid) - np.min(tc_valid)) if len(tc_valid) > 0 else 0.0
    preferred = float(bin_centers[valid][np.argmax(tc_valid)]) if len(tc_valid) > 0 else 0.0

    return {
        "cw_mean": cw_mean,
        "ccw_mean": ccw_mean,
        "asymmetry_index": float(asym),
        "modulation_depth": mod_depth,
        "preferred_ahv": preferred,
    }


def anticipatory_time_delay(
    signal: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    max_lag_frames: int = 10,
    fps: float = 30.0,
    n_bins: int = 36,
) -> dict:
    """Estimate anticipatory time delay (ATD) of HD tuning.

    Computes HD tuning curve at different time lags between neural activity
    and head direction. The lag that maximises MVL indicates whether the
    cell anticipates (positive lag) or follows (negative lag) head direction.

    Parameters
    ----------
    signal : (n_frames,) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
    max_lag_frames : int
        Maximum lag to test (in frames).
    fps : float
        Sampling rate.
    n_bins : int

    Returns
    -------
    dict
        ``"lags_ms"`` — tested lag values in ms.
        ``"mvls"`` — MVL at each lag.
        ``"best_lag_ms"`` — lag with highest MVL.
        ``"best_mvl"`` — highest MVL.
    """
    from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length

    lags = range(-max_lag_frames, max_lag_frames + 1)
    lag_ms = [l * 1000.0 / fps for l in lags]
    mvls = []

    n = len(signal)
    for lag in lags:
        if lag >= 0:
            sig_slice = signal[:n - lag] if lag > 0 else signal
            hd_slice = hd_deg[lag:] if lag > 0 else hd_deg
            mask_slice = mask[:n - lag] if lag > 0 else mask
        else:
            sig_slice = signal[-lag:]
            hd_slice = hd_deg[:n + lag]
            mask_slice = mask[-lag:]

        tc, bc = compute_hd_tuning_curve(
            sig_slice, hd_slice, mask_slice, n_bins=n_bins,
        )
        mvls.append(mean_vector_length(tc, bc))

    mvls = np.array(mvls, dtype=np.float64)
    best_idx = int(np.argmax(mvls))

    return {
        "lags_ms": np.array(lag_ms, dtype=np.float64),
        "mvls": mvls,
        "best_lag_ms": float(lag_ms[best_idx]),
        "best_mvl": float(mvls[best_idx]),
    }
