"""Tuning curve comparison between conditions (e.g. light vs dark).

Pure numpy functions for comparing HD tuning curves and spatial rate maps
across experimental conditions, plus split-half reliability and Rayleigh test.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hm2p.analysis.tuning import (
    compute_hd_tuning_curve,
    mean_vector_length,
    preferred_direction,
    spatial_information,
)


def tuning_curve_correlation(
    curve_a: npt.NDArray[np.floating],
    curve_b: npt.NDArray[np.floating],
) -> float:
    """Pearson correlation between two tuning curves.

    Parameters
    ----------
    curve_a, curve_b : (n_bins,) float
        Tuning curves to compare.  May contain NaN.

    Returns
    -------
    float
        Pearson r.  Returns NaN if fewer than 3 valid (non-NaN) bins
        in common.
    """
    valid = ~np.isnan(curve_a) & ~np.isnan(curve_b)
    if valid.sum() < 3:
        return float("nan")

    a = curve_a[valid]
    b = curve_b[valid]

    # Handle constant arrays (zero or near-zero std)
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def preferred_direction_shift(
    curve_a: npt.NDArray[np.floating],
    curve_b: npt.NDArray[np.floating],
    bin_centers_deg: npt.NDArray[np.floating],
) -> float:
    """Angular difference between preferred directions of two HD tuning curves.

    Parameters
    ----------
    curve_a, curve_b : (n_bins,) float
        HD tuning curves.
    bin_centers_deg : (n_bins,) float
        Bin centre angles in degrees.

    Returns
    -------
    float
        Angular shift in degrees, in [-180, 180].
    """
    pd_a = preferred_direction(curve_a, bin_centers_deg)
    pd_b = preferred_direction(curve_b, bin_centers_deg)

    diff = pd_b - pd_a
    # Wrap to [-180, 180]
    diff = ((diff + 180.0) % 360.0) - 180.0
    return float(diff)


def rate_map_correlation(
    map_a: npt.NDArray[np.floating],
    map_b: npt.NDArray[np.floating],
) -> float:
    """Pearson correlation between two 2D rate maps.

    Parameters
    ----------
    map_a, map_b : 2D float arrays
        Spatial rate maps.  May contain NaN.

    Returns
    -------
    float
        Pearson r.  Returns NaN if fewer than 3 valid bins in common.
    """
    a = map_a.ravel()
    b = map_b.ravel()

    # Handle mismatched sizes (different spatial bin counts between halves)
    if len(a) != len(b):
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]

    valid = ~np.isnan(a) & ~np.isnan(b)

    if valid.sum() < 3:
        return float("nan")

    av = a[valid]
    bv = b[valid]

    if np.std(av) < 1e-15 or np.std(bv) < 1e-15:
        return float("nan")

    return float(np.corrcoef(av, bv)[0, 1])


def mvl_ratio(
    curve_a: npt.NDArray[np.floating],
    curve_b: npt.NDArray[np.floating],
    bin_centers_deg: npt.NDArray[np.floating],
) -> float:
    """Ratio of mean vector lengths: MVL(curve_b) / MVL(curve_a).

    Parameters
    ----------
    curve_a, curve_b : (n_bins,) float
        HD tuning curves.
    bin_centers_deg : (n_bins,) float
        Bin centre angles in degrees.

    Returns
    -------
    float
        MVL ratio.  Returns NaN if MVL of curve_a is zero.
    """
    mvl_a = mean_vector_length(curve_a, bin_centers_deg)
    mvl_b = mean_vector_length(curve_b, bin_centers_deg)

    if mvl_a == 0.0:
        return float("nan")

    return float(mvl_b / mvl_a)


def si_ratio(
    map_a: npt.NDArray[np.floating],
    occ_a: npt.NDArray[np.floating],
    map_b: npt.NDArray[np.floating],
    occ_b: npt.NDArray[np.floating],
) -> float:
    """Ratio of spatial information: SI(map_b) / SI(map_a).

    Parameters
    ----------
    map_a : 2D float array
        Rate map for condition A.
    occ_a : 2D float array
        Occupancy map for condition A (seconds).
    map_b : 2D float array
        Rate map for condition B.
    occ_b : 2D float array
        Occupancy map for condition B (seconds).

    Returns
    -------
    float
        SI ratio.  Returns NaN if SI of map_a is zero.
    """
    si_a = spatial_information(map_a, occ_a)
    si_b = spatial_information(map_b, occ_b)

    if si_a == 0.0:
        return float("nan")

    return float(si_b / si_a)


# ---------------------------------------------------------------------------
# Split-half reliability
# ---------------------------------------------------------------------------


def split_half_reliability(
    signal: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
) -> dict:
    """Compute split-half reliability of HD tuning.

    Splits valid frames into odd/even halves, computes tuning curves for
    each half, and returns the Pearson correlation between them.

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
        ``"correlation"`` — Pearson r between half-curves.
        ``"mvl_half1"``, ``"mvl_half2"`` — MVL of each half.
        ``"pd_shift"`` — angular difference in preferred direction (degrees).
    """
    valid_idx = np.where(mask)[0]
    n_valid = len(valid_idx)

    mask_odd = np.zeros_like(mask)
    mask_even = np.zeros_like(mask)
    mask_odd[valid_idx[0::2]] = True
    mask_even[valid_idx[1::2]] = True

    tc1, bc1 = compute_hd_tuning_curve(
        signal, hd_deg, mask_odd, n_bins=n_bins,
        smoothing_sigma_deg=smoothing_sigma_deg,
    )
    tc2, bc2 = compute_hd_tuning_curve(
        signal, hd_deg, mask_even, n_bins=n_bins,
        smoothing_sigma_deg=smoothing_sigma_deg,
    )

    corr = tuning_curve_correlation(tc1, tc2)
    pd_shift = preferred_direction_shift(tc1, tc2, bc1)

    return {
        "correlation": corr,
        "mvl_half1": mean_vector_length(tc1, bc1),
        "mvl_half2": mean_vector_length(tc2, bc2),
        "pd_shift": pd_shift,
    }


# ---------------------------------------------------------------------------
# Rayleigh test for circular uniformity
# ---------------------------------------------------------------------------


def rayleigh_test(
    angles_deg: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating] | None = None,
) -> dict:
    """Rayleigh test for non-uniformity of circular data.

    Tests H0: the population is uniformly distributed around the circle.

    Parameters
    ----------
    angles_deg : (n,) float
        Angles in degrees.
    weights : (n,) float or None
        Optional weights (e.g. firing rates). If None, uniform weights.

    Returns
    -------
    dict
        ``"z"`` — Rayleigh's Z statistic (n * R^2).
        ``"p_value"`` — approximate p-value.
        ``"mean_resultant_length"`` — R (mean resultant length).
        ``"mean_direction_deg"`` — circular mean direction in degrees [0, 360).
    """
    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()  # Normalise to sum to 1
        C = np.sum(w * np.cos(angles_rad))
        S = np.sum(w * np.sin(angles_rad))
        n = np.sum(w > 0)  # effective sample size
        R = np.sqrt(C**2 + S**2)
    else:
        n = len(angles_rad)
        C = np.mean(np.cos(angles_rad))
        S = np.mean(np.sin(angles_rad))
        R = np.sqrt(C**2 + S**2)

    n_eff = float(n)
    Z = n_eff * R**2

    # Approximate p-value (Mardia & Jupp, 2000)
    p = np.exp(-Z)
    # Correction for small samples
    if n_eff < 50:
        p = p * (1 + (2 * Z - Z**2) / (4 * n_eff) -
                 (24 * Z - 132 * Z**2 + 76 * Z**3 - 9 * Z**4) / (288 * n_eff**2))
    p = max(0.0, min(1.0, float(p)))

    mean_dir = float(np.rad2deg(np.arctan2(S, C))) % 360.0

    return {
        "z": float(Z),
        "p_value": p,
        "mean_resultant_length": float(R),
        "mean_direction_deg": mean_dir,
    }
