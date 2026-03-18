"""Non-parametric statistical tests accounting for animal-level nesting.

Implements the 3-approach framework from ``docs/stats-strategy.md``:

1. **Animal-level summary** — collapse to animal means, Mann-Whitney U.
2. **Cluster permutation** — shuffle group labels at animal level.
3. **Within-cell paired** — Wilcoxon signed-rank for repeated measures.

Plus helpers for interaction contrasts, confound checks, and FDR correction.

All primary tests are non-parametric (no normality assumptions).

References
----------
Aarts et al. 2014. "A solution to dependency: using multilevel analysis to
    accommodate nested data." Nature Neuroscience 17, 491-496.
    doi:10.1038/nn.3648
Benjamini & Hochberg 1995. "Controlling the false discovery rate: a practical
    and powerful approach to multiple testing." JRSS-B 57(1), 289-300.
    doi:10.1111/j.2517-6161.1995.tb02031.x
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ============================================================================
# Approach 1: Animal-level summary statistics
# ============================================================================


def animal_summary_test(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str = "celltype",
    animal_col: str = "animal_id",
) -> dict:
    """Collapse to animal means, then compare groups with Mann-Whitney U.

    This is the simplest correct approach to account for animal-level nesting.
    Each animal contributes one value (the mean of its cells), so observations
    are independent.

    Parameters
    ----------
    df : DataFrame
        Must contain *animal_col*, *group_col*, and *metric_col*.
    metric_col : str
        Numeric column to compare.
    group_col : str
        Column defining the two groups (default ``"celltype"``).
    animal_col : str
        Column identifying the animal (default ``"animal_id"``).

    Returns
    -------
    dict
        Keys: ``statistic``, ``p_value``, ``n_penk``, ``n_nonpenk``,
        ``penk_mean``, ``nonpenk_mean``, ``effect_size`` (CLES).

    Raises
    ------
    ValueError
        If required columns are missing or fewer than 2 groups are present.
    """
    _validate_columns(df, [metric_col, group_col, animal_col])

    animal_means = (
        df.groupby([animal_col, group_col])[metric_col].mean().reset_index()
    )
    penk = animal_means.loc[
        animal_means[group_col] == "penk", metric_col
    ].values
    nonpenk = animal_means.loc[
        animal_means[group_col] == "nonpenk", metric_col
    ].values

    if len(penk) < 1 or len(nonpenk) < 1:
        raise ValueError(
            f"Need at least 1 animal per group, got penk={len(penk)}, "
            f"nonpenk={len(nonpenk)}"
        )

    u_stat, p_val = stats.mannwhitneyu(penk, nonpenk, alternative="two-sided")

    # Common language effect size (CLES): P(penk > nonpenk)
    cles = float(u_stat) / (len(penk) * len(nonpenk))

    return {
        "statistic": float(u_stat),
        "p_value": float(p_val),
        "n_penk": len(penk),
        "n_nonpenk": len(nonpenk),
        "penk_mean": float(np.mean(penk)),
        "nonpenk_mean": float(np.mean(nonpenk)),
        "effect_size": cles,
    }


# ============================================================================
# Approach 3: Cluster permutation test
# ============================================================================


def cluster_permutation_test(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str = "celltype",
    cluster_col: str = "animal_id",
    n_perms: int = 10000,
    seed: int = 42,
) -> dict:
    """Permutation test that shuffles group labels at the cluster (animal) level.

    All cells from a given animal stay together; only the animal's group
    assignment is permuted. This respects the nesting structure exactly.

    Parameters
    ----------
    df : DataFrame
        Must contain *cluster_col*, *group_col*, and *metric_col*.
    metric_col : str
        Numeric column to test.
    group_col : str
        Column defining the two groups (default ``"celltype"``).
    cluster_col : str
        Column identifying the cluster unit (default ``"animal_id"``).
    n_perms : int
        Number of permutations (default 10000).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    dict
        Keys: ``observed``, ``p_value``, ``null_mean``, ``null_std``.

    Raises
    ------
    ValueError
        If required columns are missing or fewer than 2 groups are present.
    """
    _validate_columns(df, [metric_col, group_col, cluster_col])

    # Observed statistic: difference in group means (penk - nonpenk)
    penk_vals = df.loc[df[group_col] == "penk", metric_col].dropna().values
    nonpenk_vals = df.loc[df[group_col] == "nonpenk", metric_col].dropna().values

    if len(penk_vals) < 1 or len(nonpenk_vals) < 1:
        raise ValueError(
            "Need at least 1 observation per group for permutation test"
        )

    observed = float(np.mean(penk_vals) - np.mean(nonpenk_vals))

    # Cluster-level assignments
    cluster_groups = df.groupby(cluster_col)[group_col].first()
    cluster_ids = cluster_groups.index.values
    n_nonpenk = int((cluster_groups == "nonpenk").sum())

    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_perms)

    for i in range(n_perms):
        perm_idx = rng.choice(len(cluster_ids), size=n_nonpenk, replace=False)
        perm_nonpenk = set(cluster_ids[perm_idx])

        a_vals = df.loc[
            ~df[cluster_col].isin(perm_nonpenk), metric_col
        ].dropna().values
        b_vals = df.loc[
            df[cluster_col].isin(perm_nonpenk), metric_col
        ].dropna().values

        if len(a_vals) == 0 or len(b_vals) == 0:
            null_stats[i] = 0.0
        else:
            null_stats[i] = float(np.mean(a_vals) - np.mean(b_vals))

    # Two-sided p-value with continuity correction (+1)
    p_value = float(
        (np.sum(np.abs(null_stats) >= np.abs(observed)) + 1) / (n_perms + 1)
    )

    return {
        "observed": observed,
        "p_value": p_value,
        "null_mean": float(np.mean(null_stats)),
        "null_std": float(np.std(null_stats)),
    }


# ============================================================================
# Within-cell paired test
# ============================================================================


def within_cell_test(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> dict:
    """Paired Wilcoxon signed-rank test on two conditions within the same cells.

    Computes the difference ``col_a - col_b`` for each row and tests whether
    the median difference is significantly different from zero.

    Parameters
    ----------
    df : DataFrame
        Must contain *col_a* and *col_b* columns.
    col_a : str
        First condition column.
    col_b : str
        Second condition column.

    Returns
    -------
    dict
        Keys: ``statistic``, ``p_value``, ``n_cells``, ``mean_diff``,
        ``median_diff``.

    Raises
    ------
    ValueError
        If columns are missing or fewer than 2 valid pairs remain.
    """
    _validate_columns(df, [col_a, col_b])

    subset = df[[col_a, col_b]].dropna()
    diffs = (subset[col_a] - subset[col_b]).values

    if len(diffs) < 2:
        raise ValueError(
            f"Need at least 2 valid pairs, got {len(diffs)}"
        )

    # Remove zero differences (Wilcoxon discards them)
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 1:
        return {
            "statistic": np.nan,
            "p_value": 1.0,
            "n_cells": len(diffs),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
        }

    w_stat, p_val = stats.wilcoxon(nonzero, alternative="two-sided")

    return {
        "statistic": float(w_stat),
        "p_value": float(p_val),
        "n_cells": len(diffs),
        "mean_diff": float(np.mean(diffs)),
        "median_diff": float(np.median(diffs)),
    }


# ============================================================================
# Interaction contrast
# ============================================================================


def interaction_contrast(
    df: pd.DataFrame,
    cols_list: list[str],
) -> pd.Series:
    """Compute a 2x2 factorial interaction contrast per row.

    For a design with factors A (e.g. movement state) and B (e.g. light
    condition), the interaction is:

        (A1B1 - A2B1) - (A1B2 - A2B2)

    where ``cols_list = [A1B1, A2B1, A1B2, A2B2]``.

    Parameters
    ----------
    df : DataFrame
        Must contain all four columns in *cols_list*.
    cols_list : list of str
        Exactly 4 column names: [moving_light, stationary_light,
        moving_dark, stationary_dark].

    Returns
    -------
    pd.Series
        Per-row interaction contrast values.

    Raises
    ------
    ValueError
        If *cols_list* does not have exactly 4 elements or columns are missing.
    """
    if len(cols_list) != 4:
        raise ValueError(
            f"interaction_contrast requires exactly 4 columns, got {len(cols_list)}"
        )
    _validate_columns(df, cols_list)

    a1b1, a2b1, a1b2, a2b2 = cols_list
    return (df[a1b1] - df[a2b1]) - (df[a1b2] - df[a2b2])


# ============================================================================
# Confound check
# ============================================================================


def confound_check(
    df: pd.DataFrame,
    metric_col: str,
    confound_cols: list[str],
) -> list[dict]:
    """Check Spearman correlations between a metric and potential confounds.

    Parameters
    ----------
    df : DataFrame
        Must contain *metric_col* and all *confound_cols*.
    metric_col : str
        Primary metric column.
    confound_cols : list of str
        Columns to check for confounding.

    Returns
    -------
    list of dict
        One dict per confound with keys: ``confound``, ``rho``, ``p_value``,
        ``flagged`` (True if |rho| > 0.3).

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    _validate_columns(df, [metric_col] + confound_cols)

    results = []
    for col in confound_cols:
        subset = df[[metric_col, col]].dropna()
        if len(subset) < 3:
            results.append({
                "confound": col,
                "rho": np.nan,
                "p_value": np.nan,
                "flagged": False,
            })
        else:
            rho, p_val = stats.spearmanr(subset[metric_col], subset[col])
            results.append({
                "confound": col,
                "rho": float(rho),
                "p_value": float(p_val),
                "flagged": bool(abs(rho) > 0.3),
            })

    return results


# ============================================================================
# Orchestrator
# ============================================================================


def run_between_group_test(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str = "celltype",
    animal_col: str = "animal_id",
    n_perms: int = 10000,
    seed: int = 42,
) -> dict:
    """Run both animal-summary and cluster-permutation tests, return combined result.

    Parameters
    ----------
    df : DataFrame
        Must contain *metric_col*, *group_col*, and *animal_col*.
    metric_col : str
        Numeric column to compare.
    group_col : str
        Column defining the two groups (default ``"celltype"``).
    animal_col : str
        Column identifying the animal (default ``"animal_id"``).
    n_perms : int
        Number of permutations for cluster test (default 10000).
    seed : int
        Random seed (default 42).

    Returns
    -------
    dict
        Combined results with keys prefixed by ``summary_`` and ``perm_``,
        plus ``verdict`` (``"supported"``, ``"inconsistent"``, or
        ``"not_supported"``).
    """
    summary = animal_summary_test(
        df, metric_col, group_col=group_col, animal_col=animal_col
    )
    perm = cluster_permutation_test(
        df, metric_col, group_col=group_col, cluster_col=animal_col,
        n_perms=n_perms, seed=seed,
    )

    # Determine verdict
    alpha = 0.05
    summary_sig = summary["p_value"] < alpha
    perm_sig = perm["p_value"] < alpha

    if summary_sig and perm_sig:
        verdict = "supported"
    elif summary_sig != perm_sig:
        verdict = "inconsistent"
    else:
        verdict = "not_supported"

    combined: dict = {"metric": metric_col}
    for k, v in summary.items():
        combined[f"summary_{k}"] = v
    for k, v in perm.items():
        combined[f"perm_{k}"] = v
    combined["verdict"] = verdict

    return combined


# ============================================================================
# FDR correction
# ============================================================================


def fdr_correct(
    results: list[dict],
    alpha: float = 0.05,
) -> list[dict]:
    """Apply Benjamini-Hochberg FDR correction to a list of test results.

    Corrects the ``perm_p_value`` field (cluster permutation p-value) from
    each result dict. If ``perm_p_value`` is not present, falls back to
    ``p_value``.

    Parameters
    ----------
    results : list of dict
        Each dict must contain a ``perm_p_value`` or ``p_value`` key.
    alpha : float
        Significance threshold (default 0.05).

    Returns
    -------
    list of dict
        Input dicts augmented with ``p_fdr`` and ``significant_fdr`` fields.

    Raises
    ------
    ValueError
        If *results* is empty.
    """
    if not results:
        raise ValueError("results must not be empty")

    # Extract p-values
    p_key = "perm_p_value" if "perm_p_value" in results[0] else "p_value"
    p_vals = np.array([r.get(p_key, np.nan) for r in results], dtype=float)
    valid = ~np.isnan(p_vals)

    p_fdr = np.full(len(p_vals), np.nan)
    sig_fdr = np.full(len(p_vals), False)

    if valid.any():
        reject, corrected, _, _ = multipletests(
            p_vals[valid], alpha=alpha, method="fdr_bh"
        )
        p_fdr[valid] = corrected
        sig_fdr[valid] = reject

    out = []
    for i, r in enumerate(results):
        augmented = dict(r)
        augmented["p_fdr"] = float(p_fdr[i]) if not np.isnan(p_fdr[i]) else np.nan
        augmented["significant_fdr"] = bool(sig_fdr[i])
        out.append(augmented)

    return out


# ============================================================================
# Internal helpers
# ============================================================================


def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError if any required columns are missing from *df*."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")
