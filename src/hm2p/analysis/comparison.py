"""Tuning curve comparison between conditions (e.g. light vs dark).

Pure numpy functions for comparing HD tuning curves and spatial rate maps
across experimental conditions.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hm2p.analysis.tuning import (
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

    # Handle constant arrays (zero std)
    if np.std(a) == 0 or np.std(b) == 0:
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
    valid = ~np.isnan(a) & ~np.isnan(b)

    if valid.sum() < 3:
        return float("nan")

    av = a[valid]
    bv = b[valid]

    if np.std(av) == 0 or np.std(bv) == 0:
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
