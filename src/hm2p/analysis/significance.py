"""Bootstrap significance testing for HD and place tuning via circular time shift.

All functions are pure numpy — no I/O, no classes.

Circular shift method follows:
    Muller et al. 1987. "The effects of changes in the environment on the
    spatial firing of hippocampal complex-spike cells." J Neurosci
    7(7):1951-1968. doi:10.1523/JNEUROSCI.07-07-01951.1987
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

from hm2p.analysis.tuning import (
    compute_hd_tuning_curve,
    compute_place_rate_map,
    mean_vector_length,
    peak_to_trough_ratio,
    spatial_information,
)


def circular_shift_significance(
    signal: npt.NDArray[np.floating],
    tuning_fn: Callable[[npt.NDArray[np.floating]], float],
    observed_metric: float,
    n_shuffles: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generic significance test via circular time shift.

    For each shuffle, the *signal* array is circularly rolled by a random
    offset and the *tuning_fn* is recomputed.  The p-value is the fraction
    of shuffle metrics that equal or exceed the observed metric, with a
    conservative +1 / +1 correction.

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal to shuffle.
    tuning_fn : callable
        Takes a (n_frames,) signal array, returns a float metric.
    observed_metric : float
        The actual observed metric value.
    n_shuffles : int
        Number of shuffles to perform.
    rng : numpy.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    dict
        ``"observed"`` — the observed metric value.
        ``"p_value"`` — conservative p-value.
        ``"shuffle_distribution"`` — (n_shuffles,) array of shuffled metrics.
        ``"n_shuffles"`` — number of shuffles performed.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_frames = len(signal)
    shuffle_metrics = np.empty(n_shuffles, dtype=np.float64)

    for i in range(n_shuffles):
        offset = rng.integers(1, n_frames)
        shifted = np.roll(signal, offset)
        shuffle_metrics[i] = tuning_fn(shifted)

    # Conservative p-value: (count >= observed + 1) / (n_shuffles + 1)
    n_ge = np.sum(shuffle_metrics >= observed_metric)
    p_value = float((n_ge + 1) / (n_shuffles + 1))

    return {
        "observed": observed_metric,
        "p_value": p_value,
        "shuffle_distribution": shuffle_metrics,
        "n_shuffles": n_shuffles,
    }


def hd_tuning_significance(
    signal: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_shuffles: int = 1000,
    metric: str = "mvl",
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Significance test for HD tuning via circular shift.

    Computes the observed HD tuning curve and metric, then runs
    :func:`circular_shift_significance` to assess significance.

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal (e.g. dF/F).
    hd_deg : (n_frames,) float
        Head direction in degrees.
    mask : (n_frames,) bool
        Valid frames.
    n_shuffles : int
        Number of shuffles.
    metric : ``"mvl"`` or ``"peak_to_trough"``
        Which metric to test.
    n_bins : int
        Number of angular bins.
    smoothing_sigma_deg : float
        Gaussian smoothing sigma in degrees.
    rng : numpy.random.Generator or None
        For reproducibility.

    Returns
    -------
    dict
        Same keys as :func:`circular_shift_significance` plus
        ``"tuning_curve"`` and ``"bin_centers"``.
    """
    if metric not in ("mvl", "peak_to_trough"):
        raise ValueError(f"metric must be 'mvl' or 'peak_to_trough', got {metric!r}")

    # Compute observed tuning curve and metric
    tuning_curve, bin_centers = compute_hd_tuning_curve(
        signal, hd_deg, mask, n_bins=n_bins, smoothing_sigma_deg=smoothing_sigma_deg
    )

    if metric == "mvl":
        observed = mean_vector_length(tuning_curve, bin_centers)
    else:
        observed = peak_to_trough_ratio(tuning_curve)

    def _tuning_fn(shifted_signal: npt.NDArray[np.floating]) -> float:
        tc, bc = compute_hd_tuning_curve(
            shifted_signal,
            hd_deg,
            mask,
            n_bins=n_bins,
            smoothing_sigma_deg=smoothing_sigma_deg,
        )
        if metric == "mvl":
            return mean_vector_length(tc, bc)
        return peak_to_trough_ratio(tc)

    result = circular_shift_significance(
        signal, _tuning_fn, observed, n_shuffles=n_shuffles, rng=rng
    )
    result["tuning_curve"] = tuning_curve
    result["bin_centers"] = bin_centers
    return result


def place_tuning_significance(
    signal: npt.NDArray[np.floating],
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_shuffles: int = 1000,
    metric: str = "spatial_info",
    bin_size: float = 2.5,
    smoothing_sigma: float = 3.0,
    min_occupancy_s: float = 0.5,
    fps: float = 9.8,
    rng: np.random.Generator | None = None,
) -> dict:
    """Significance test for place tuning via circular shift.

    Computes the observed spatial rate map and spatial information,
    then shuffles the signal to build a null distribution.

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal.
    x, y : (n_frames,) float
        Position coordinates.
    mask : (n_frames,) bool
        Valid frames.
    n_shuffles : int
        Number of shuffles.
    metric : str
        Currently only ``"spatial_info"`` is supported.
    bin_size : float
        Spatial bin size.
    smoothing_sigma : float
        Gaussian smoothing sigma in bins.
    min_occupancy_s : float
        Minimum occupancy in seconds.
    fps : float
        Sampling rate.
    rng : numpy.random.Generator or None
        For reproducibility.

    Returns
    -------
    dict
        Same keys as :func:`circular_shift_significance` plus
        ``"rate_map"`` and ``"occupancy_map"``.
    """
    # Compute observed rate map
    rate_map, occupancy_map, _, _ = compute_place_rate_map(
        signal,
        x,
        y,
        mask,
        bin_size=bin_size,
        smoothing_sigma=smoothing_sigma,
        min_occupancy_s=min_occupancy_s,
        fps=fps,
    )
    observed = spatial_information(rate_map, occupancy_map)

    def _tuning_fn(shifted_signal: npt.NDArray[np.floating]) -> float:
        rm, om, _, _ = compute_place_rate_map(
            shifted_signal,
            x,
            y,
            mask,
            bin_size=bin_size,
            smoothing_sigma=smoothing_sigma,
            min_occupancy_s=min_occupancy_s,
            fps=fps,
        )
        return spatial_information(rm, om)

    result = circular_shift_significance(
        signal, _tuning_fn, observed, n_shuffles=n_shuffles, rng=rng
    )
    result["rate_map"] = rate_map
    result["occupancy_map"] = occupancy_map
    return result
