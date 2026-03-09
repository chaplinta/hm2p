"""Speed modulation analysis for HD cells.

Examines how running speed affects HD tuning sharpness and gain.
Important control: speed-related gain must be separated from
light-related gain to properly attribute visual anchoring effects.

All functions pure numpy — no I/O.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hm2p.analysis.tuning import (
    compute_hd_tuning_curve,
    mean_vector_length,
    preferred_direction,
)


def speed_tuning_curve(
    signal: npt.NDArray[np.floating],
    speed: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_bins: int = 20,
    max_speed: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean signal as a function of running speed.

    Parameters
    ----------
    signal : (n_frames,) float
    speed : (n_frames,) float
        Running speed (cm/s or arbitrary units).
    mask : (n_frames,) bool
    n_bins : int
    max_speed : float or None
        Maximum speed for binning. If None, uses 95th percentile.

    Returns
    -------
    tuning_curve : (n_bins,) float
        Mean signal per speed bin.
    bin_centers : (n_bins,) float
        Speed bin centres.
    """
    sig = signal[mask]
    spd = speed[mask]

    if max_speed is None:
        max_speed = float(np.percentile(spd, 95))

    edges = np.linspace(0, max_speed, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    tc = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        in_bin = (spd >= edges[i]) & (spd < edges[i + 1])
        if i == n_bins - 1:
            in_bin = in_bin | (spd >= edges[i + 1])
        if in_bin.sum() > 0:
            tc[i] = np.mean(sig[in_bin])
        else:
            tc[i] = np.nan

    return tc, centers


def speed_modulation_index(
    signal: npt.NDArray[np.floating],
    speed: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    speed_threshold: float | None = None,
) -> dict:
    """Compute speed modulation: compare activity at high vs low speeds.

    Parameters
    ----------
    signal : (n_frames,) float
    speed : (n_frames,) float
    mask : (n_frames,) bool
    speed_threshold : float or None
        Threshold to split high/low speed. If None, uses median.

    Returns
    -------
    dict
        ``"speed_modulation_index"`` — (high - low) / (high + low).
        ``"mean_signal_fast"`` — mean signal at high speed.
        ``"mean_signal_slow"`` — mean signal at low speed.
        ``"speed_correlation"`` — Pearson r between speed and signal.
    """
    sig = signal[mask]
    spd = speed[mask]

    if speed_threshold is None:
        speed_threshold = float(np.median(spd))

    fast = spd >= speed_threshold
    slow = ~fast

    mean_fast = float(np.mean(sig[fast])) if fast.any() else 0.0
    mean_slow = float(np.mean(sig[slow])) if slow.any() else 0.0

    denom = mean_fast + mean_slow
    smi = (mean_fast - mean_slow) / denom if denom > 0 else 0.0

    # Correlation
    if len(sig) > 2 and np.std(spd) > 0 and np.std(sig) > 0:
        corr = float(np.corrcoef(spd, sig)[0, 1])
    else:
        corr = 0.0

    return {
        "speed_modulation_index": float(smi),
        "mean_signal_fast": mean_fast,
        "mean_signal_slow": mean_slow,
        "speed_correlation": corr,
    }


def hd_tuning_by_speed(
    signal: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    speed: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    speed_quantiles: tuple[float, ...] = (0.33, 0.67),
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
) -> dict:
    """Compare HD tuning curves across speed terciles.

    Parameters
    ----------
    signal : (n_frames,) float
    hd_deg : (n_frames,) float
    speed : (n_frames,) float
    mask : (n_frames,) bool
    speed_quantiles : tuple of float
        Quantiles for splitting speed into groups.
    n_bins : int
    smoothing_sigma_deg : float

    Returns
    -------
    dict
        ``"tuning_curves"`` — list of (n_bins,) arrays per speed group.
        ``"bin_centers"`` — (n_bins,) bin centres.
        ``"mvls"`` — MVL per speed group.
        ``"pds"`` — preferred direction per speed group.
        ``"speed_labels"`` — descriptive labels per group.
        ``"speed_thresholds"`` — actual quantile values used.
    """
    spd = speed[mask]
    thresholds = [float(np.quantile(spd, q)) for q in speed_quantiles]

    groups = []
    labels = []

    # Build masks for each speed group
    low_mask = mask & (speed < thresholds[0])
    groups.append(low_mask)
    labels.append(f"Slow (<{thresholds[0]:.1f})")

    for i in range(len(thresholds) - 1):
        mid_mask = mask & (speed >= thresholds[i]) & (speed < thresholds[i + 1])
        groups.append(mid_mask)
        labels.append(f"Medium ({thresholds[i]:.1f}-{thresholds[i+1]:.1f})")

    high_mask = mask & (speed >= thresholds[-1])
    groups.append(high_mask)
    labels.append(f"Fast (>{thresholds[-1]:.1f})")

    tcs = []
    mvls = []
    pds = []
    bc = None

    for group_mask in groups:
        if group_mask.sum() < n_bins:
            tcs.append(np.full(n_bins, np.nan))
            mvls.append(np.nan)
            pds.append(np.nan)
            continue

        tc, bc_i = compute_hd_tuning_curve(
            signal, hd_deg, group_mask, n_bins=n_bins,
            smoothing_sigma_deg=smoothing_sigma_deg,
        )
        if bc is None:
            bc = bc_i
        tcs.append(tc)
        mvls.append(mean_vector_length(tc, bc_i))
        pds.append(preferred_direction(tc, bc_i))

    if bc is None:
        bc = np.linspace(5, 355, n_bins)

    return {
        "tuning_curves": tcs,
        "bin_centers": bc,
        "mvls": mvls,
        "pds": pds,
        "speed_labels": labels,
        "speed_thresholds": thresholds,
    }
