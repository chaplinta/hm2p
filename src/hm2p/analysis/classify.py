"""Cell classification — automated HD cell identification.

Combines multiple metrics (MVL, Rayleigh test, split-half reliability,
information content) to classify cells as HD-tuned or non-HD.
All functions pure numpy — no I/O, no classes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hm2p.analysis.comparison import rayleigh_test, split_half_reliability
from hm2p.analysis.information import mutual_information_binned
from hm2p.analysis.significance import hd_tuning_significance
from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length


def classify_single_cell(
    signal: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    mvl_threshold: float = 0.15,
    p_threshold: float = 0.05,
    reliability_threshold: float = 0.5,
    n_shuffles: int = 500,
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Classify a single cell as HD-tuned or not.

    Criteria for HD classification (all must pass):
    1. MVL > mvl_threshold
    2. Shuffle test p-value < p_threshold
    3. Split-half reliability > reliability_threshold

    Parameters
    ----------
    signal : (n_frames,) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
    mvl_threshold : float
        Minimum MVL to qualify as HD cell.
    p_threshold : float
        Maximum p-value from shuffle test.
    reliability_threshold : float
        Minimum split-half correlation.
    n_shuffles : int
        Number of shuffles for significance test.
    n_bins : int
    smoothing_sigma_deg : float
    rng : Generator or None

    Returns
    -------
    dict
        ``"is_hd"`` — bool, True if cell passes all criteria.
        ``"mvl"`` — float, mean vector length.
        ``"p_value"`` — float, shuffle test p-value.
        ``"reliability"`` — float, split-half correlation.
        ``"mi"`` — float, mutual information (bits).
        ``"preferred_direction"`` — float, PD in degrees.
        ``"criteria_passed"`` — dict of bool per criterion.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Tuning curve and MVL
    tc, bc = compute_hd_tuning_curve(
        signal, hd_deg, mask, n_bins=n_bins,
        smoothing_sigma_deg=smoothing_sigma_deg,
    )
    mvl = mean_vector_length(tc, bc)

    # Preferred direction via circular mean weighted by tuning curve
    bc_rad = np.deg2rad(bc)
    pd_deg = float(np.rad2deg(np.arctan2(
        np.sum(tc * np.sin(bc_rad)),
        np.sum(tc * np.cos(bc_rad)),
    ))) % 360.0

    # Shuffle significance
    sig_result = hd_tuning_significance(
        signal, hd_deg, mask,
        n_shuffles=n_shuffles, metric="mvl",
        n_bins=n_bins, smoothing_sigma_deg=smoothing_sigma_deg,
        rng=rng,
    )
    p_value = sig_result["p_value"]

    # Split-half reliability
    rel = split_half_reliability(
        signal, hd_deg, mask,
        n_bins=n_bins, smoothing_sigma_deg=smoothing_sigma_deg,
    )
    reliability = rel["correlation"]

    # Mutual information
    mi = mutual_information_binned(signal, hd_deg, mask)

    # Classification
    criteria = {
        "mvl": mvl >= mvl_threshold,
        "significance": p_value < p_threshold,
        "reliability": not np.isnan(reliability) and reliability >= reliability_threshold,
    }
    is_hd = all(criteria.values())

    return {
        "is_hd": is_hd,
        "mvl": mvl,
        "p_value": p_value,
        "reliability": reliability,
        "mi": mi,
        "preferred_direction": pd_deg,
        "criteria_passed": criteria,
    }


def classify_population(
    signals: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    mvl_threshold: float = 0.15,
    p_threshold: float = 0.05,
    reliability_threshold: float = 0.5,
    n_shuffles: int = 500,
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Classify all cells in a population.

    Parameters
    ----------
    signals : (n_cells, n_frames) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
    mvl_threshold, p_threshold, reliability_threshold : float
        Thresholds for HD classification.
    n_shuffles : int
    n_bins : int
    smoothing_sigma_deg : float
    rng : Generator or None

    Returns
    -------
    dict
        ``"cells"`` — list of per-cell dicts from classify_single_cell.
        ``"n_hd"`` — int, number of HD cells.
        ``"n_non_hd"`` — int, number of non-HD cells.
        ``"fraction_hd"`` — float, fraction of cells classified as HD.
        ``"hd_indices"`` — list of int, indices of HD cells.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_cells = signals.shape[0]
    cells = []
    for i in range(n_cells):
        result = classify_single_cell(
            signals[i], hd_deg, mask,
            mvl_threshold=mvl_threshold,
            p_threshold=p_threshold,
            reliability_threshold=reliability_threshold,
            n_shuffles=n_shuffles,
            n_bins=n_bins,
            smoothing_sigma_deg=smoothing_sigma_deg,
            rng=rng,
        )
        cells.append(result)

    hd_indices = [i for i, c in enumerate(cells) if c["is_hd"]]
    n_hd = len(hd_indices)

    return {
        "cells": cells,
        "n_hd": n_hd,
        "n_non_hd": n_cells - n_hd,
        "fraction_hd": n_hd / n_cells if n_cells > 0 else 0.0,
        "hd_indices": hd_indices,
    }


def classification_summary_table(
    pop_result: dict,
) -> list[dict]:
    """Convert population classification result to a table-friendly format.

    Parameters
    ----------
    pop_result : dict
        Output from :func:`classify_population`.

    Returns
    -------
    list of dict
        One row per cell with columns: cell, is_hd, mvl, p_value,
        reliability, mi, preferred_direction, grade.
    """
    rows = []
    for i, cell in enumerate(pop_result["cells"]):
        # Grade: A (strong HD), B (moderate), C (weak), D (non-HD)
        if cell["is_hd"]:
            if cell["mvl"] >= 0.4 and cell["reliability"] >= 0.8:
                grade = "A"
            elif cell["mvl"] >= 0.25:
                grade = "B"
            else:
                grade = "C"
        else:
            grade = "D"

        rows.append({
            "cell": i,
            "is_hd": cell["is_hd"],
            "mvl": cell["mvl"],
            "p_value": cell["p_value"],
            "reliability": cell["reliability"],
            "mi": cell["mi"],
            "preferred_direction": cell["preferred_direction"],
            "grade": grade,
        })
    return rows
