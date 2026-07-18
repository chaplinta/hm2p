"""Celltype-specific population dynamics — Penk+ vs Penk⁻CamKII+.

Compares population-level 2P imaging dynamics between cell types,
controlling for movement state and light condition. Analyses operate
on dF/F or spike rates without requiring individual HD tuning.

All statistical tests are non-parametric (Mann-Whitney U for unpaired,
Wilcoxon signed-rank for paired comparisons).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def population_rate_by_condition(
    dff: np.ndarray,
    speed: np.ndarray,
    light_on: np.ndarray,
    active_mask: np.ndarray,
    speed_threshold: float = 2.5,
) -> dict:
    """Compute mean population activity in each condition.

    Conditions: 2×2 factorial of (moving/stationary) × (light/dark).

    Args:
        dff: (n_rois, n_frames) signal (dF/F, spikes, deconv_norm).
        speed: (n_frames,) speed in cm/s.
        light_on: (n_frames,) bool.
        active_mask: (n_frames,) bool — valid frames (not bad_behav).
        speed_threshold: cm/s threshold for moving vs stationary.

    Returns:
        Dict with per-condition mean rates (n_rois,) and frame counts.
    """
    n_rois, n_frames = dff.shape
    n = min(n_frames, len(speed), len(light_on), len(active_mask))

    moving = (speed[:n] >= speed_threshold) & active_mask[:n]
    stationary = (speed[:n] < speed_threshold) & active_mask[:n]
    light = light_on[:n].astype(bool)
    dark = ~light

    conditions = {
        "moving_light": moving & light,
        "moving_dark": moving & dark,
        "stationary_light": stationary & light,
        "stationary_dark": stationary & dark,
    }

    result = {}
    for name, mask in conditions.items():
        n_frames_cond = mask.sum()
        if n_frames_cond > 0:
            mean_rates = np.nanmean(dff[:, :n][:, mask], axis=1)
        else:
            mean_rates = np.full(n_rois, np.nan)
        result[name] = {
            "mean_rate": mean_rates,
            "n_frames": int(n_frames_cond),
        }

    return result


def compare_celltypes(
    penk_rates: np.ndarray,
    nonpenk_rates: np.ndarray,
) -> dict:
    """Compare a metric between Penk+ and Penk⁻CamKII+ using Mann-Whitney U.

    Args:
        penk_rates: (n_penk,) values for Penk+ cells.
        nonpenk_rates: (n_nonpenk,) values for non-Penk cells.

    Returns:
        Dict with statistic, p_value, effect size (rank-biserial r).
    """
    penk_valid = penk_rates[np.isfinite(penk_rates)]
    nonpenk_valid = nonpenk_rates[np.isfinite(nonpenk_rates)]

    if len(penk_valid) < 2 or len(nonpenk_valid) < 2:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "n_penk": len(penk_valid),
            "n_nonpenk": len(nonpenk_valid),
        }

    U, p = mannwhitneyu(penk_valid, nonpenk_valid, alternative="two-sided")

    # Rank-biserial correlation as effect size
    n1, n2 = len(penk_valid), len(nonpenk_valid)
    r = 1.0 - (2.0 * U) / (n1 * n2)

    return {
        "statistic": float(U),
        "p_value": float(p),
        "effect_size": float(r),
        "n_penk": n1,
        "n_nonpenk": n2,
        "penk_median": float(np.median(penk_valid)),
        "nonpenk_median": float(np.median(nonpenk_valid)),
    }


def celltype_dynamics_summary(
    sessions: list[dict],
    signal_key: str = "dff",
    speed_threshold: float = 2.5,
) -> dict:
    """Compare population dynamics between Penk+ and Penk⁻CamKII+ across sessions.

    For each condition (moving/stationary × light/dark), computes per-cell
    mean activity, then compares distributions between cell types.

    Args:
        sessions: List of session dicts from load_all_sync_data(). Each must
            have: dff, speed_cm_s, light_on, active, bad_behav, celltype.
        signal_key: Key for the signal to analyse ("dff", "spikes", "deconv_norm",
            "event_masks", "event_masks_sd").
        speed_threshold: cm/s for moving/stationary split.

    Returns:
        Dict with per-condition celltype comparisons and summary stats.
    """
    penk_cells = []  # list of (n_rois,) per-cell rates per condition
    nonpenk_cells = []
    conditions = ["moving_light", "moving_dark", "stationary_light", "stationary_dark"]

    penk_by_cond = {c: [] for c in conditions}
    nonpenk_by_cond = {c: [] for c in conditions}

    for ses in sessions:
        signal = ses.get(signal_key)
        if signal is None:
            signal = ses.get("dff")
        if signal is None:
            continue

        speed = ses.get("speed_cm_s")
        light_on = ses.get("light_on")
        bad_behav = ses.get("bad_behav")
        if speed is None or light_on is None:
            continue

        active_mask = ~bad_behav if bad_behav is not None else np.ones(signal.shape[1], dtype=bool)
        celltype = ses.get("celltype", "unknown")

        cond_rates = population_rate_by_condition(
            np.nan_to_num(signal),
            speed,
            light_on,
            active_mask,
            speed_threshold,
        )

        for cond in conditions:
            rates = cond_rates[cond]["mean_rate"]
            if celltype == "penk":
                penk_by_cond[cond].append(rates)
            elif celltype == "nonpenk":
                nonpenk_by_cond[cond].append(rates)

    # Pool across sessions
    comparisons = {}
    for cond in conditions:
        penk_all = np.concatenate(penk_by_cond[cond]) if penk_by_cond[cond] else np.array([])
        nonpenk_all = (
            np.concatenate(nonpenk_by_cond[cond]) if nonpenk_by_cond[cond] else np.array([])
        )
        comparisons[cond] = compare_celltypes(penk_all, nonpenk_all)

    # Overall (all conditions pooled)
    penk_overall = (
        np.concatenate([np.concatenate(penk_by_cond[c]) for c in conditions if penk_by_cond[c]])
        if any(penk_by_cond[c] for c in conditions)
        else np.array([])
    )
    nonpenk_overall = (
        np.concatenate(
            [np.concatenate(nonpenk_by_cond[c]) for c in conditions if nonpenk_by_cond[c]]
        )
        if any(nonpenk_by_cond[c] for c in conditions)
        else np.array([])
    )
    comparisons["overall"] = compare_celltypes(penk_overall, nonpenk_overall)

    return {
        "comparisons": comparisons,
        "conditions": conditions,
        "n_penk_sessions": sum(1 for ses in sessions if ses.get("celltype") == "penk"),
        "n_nonpenk_sessions": sum(1 for ses in sessions if ses.get("celltype") == "nonpenk"),
    }
