"""Population-level statistics comparing Penk+ vs non-Penk cell types.

Reimplements the summary statistics and group comparisons from
``sumTableNum.m`` in the legacy MATLAB pipeline.

Provides:
- Per-group descriptive statistics (mean, median, SEM, etc.)
- Mann-Whitney U tests with Benjamini-Hochberg FDR correction
- Spearman correlation (pairwise and matrix)
- CSV export of results
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ============================================================================
# Descriptive statistics
# ============================================================================


def compute_summary_stats(
    df: pd.DataFrame,
    metric_cols: list[str],
    group_col: str = "cell_type",
) -> pd.DataFrame:
    """Compute per-group descriptive statistics for each metric.

    Parameters
    ----------
    df : DataFrame
        Population metrics table with a group column and numeric metric columns.
    metric_cols : list of str
        Column names to summarise.
    group_col : str
        Column that defines the two groups (default ``"cell_type"``).

    Returns
    -------
    DataFrame
        One row per metric.  Columns are prefixed by group name, e.g.
        ``penk_n``, ``penk_mean``, ``nonpenk_median``, etc.

    Raises
    ------
    ValueError
        If *metric_cols* is empty or *group_col* is not in *df*.
    """
    if group_col not in df.columns:
        raise ValueError(f"group_col {group_col!r} not found in DataFrame columns")
    if not metric_cols:
        raise ValueError("metric_cols must not be empty")

    groups = sorted(df[group_col].dropna().unique())
    rows: list[dict[str, object]] = []

    for metric in metric_cols:
        row: dict[str, object] = {"metric": metric}
        for grp in groups:
            vals = df.loc[df[group_col] == grp, metric].dropna()
            prefix = str(grp)
            row[f"{prefix}_n"] = len(vals)
            if len(vals) == 0:
                for stat in ("mean", "median", "std", "sem", "min", "max"):
                    row[f"{prefix}_{stat}"] = np.nan
            else:
                row[f"{prefix}_mean"] = float(vals.mean())
                row[f"{prefix}_median"] = float(vals.median())
                row[f"{prefix}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
                row[f"{prefix}_sem"] = (
                    float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
                )
                row[f"{prefix}_min"] = float(vals.min())
                row[f"{prefix}_max"] = float(vals.max())
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# Group comparisons
# ============================================================================


def mann_whitney_comparison(
    df: pd.DataFrame,
    metric_cols: list[str],
    group_col: str = "cell_type",
) -> pd.DataFrame:
    """Run Mann-Whitney U tests between two groups with FDR correction.

    Parameters
    ----------
    df : DataFrame
        Population metrics table.
    metric_cols : list of str
        Numeric columns to compare.
    group_col : str
        Column defining the two groups (must contain exactly two unique
        non-NaN values).

    Returns
    -------
    DataFrame
        Columns: ``metric``, ``statistic``, ``p_value``, ``p_fdr``,
        ``significant`` (bool, at alpha=0.05 after FDR correction).

    Raises
    ------
    ValueError
        If *group_col* does not have exactly two groups, *metric_cols* is
        empty, or *group_col* is missing.
    """
    if group_col not in df.columns:
        raise ValueError(f"group_col {group_col!r} not found in DataFrame columns")
    if not metric_cols:
        raise ValueError("metric_cols must not be empty")

    groups = sorted(df[group_col].dropna().unique())
    if len(groups) != 2:
        raise ValueError(
            f"Mann-Whitney requires exactly 2 groups in {group_col!r}, "
            f"found {len(groups)}: {groups}"
        )

    g1, g2 = groups
    rows: list[dict[str, object]] = []

    for metric in metric_cols:
        a = df.loc[df[group_col] == g1, metric].dropna().values
        b = df.loc[df[group_col] == g2, metric].dropna().values

        if len(a) < 1 or len(b) < 1:
            rows.append(
                {"metric": metric, "statistic": np.nan, "p_value": np.nan}
            )
        else:
            u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append(
                {"metric": metric, "statistic": float(u_stat), "p_value": float(p_val)}
            )

    result = pd.DataFrame(rows)

    # FDR correction — only on non-NaN p-values
    p_vals = result["p_value"].values.astype(float)
    valid_mask = ~np.isnan(p_vals)

    p_fdr = np.full(len(p_vals), np.nan)
    significant = np.full(len(p_vals), False)

    if valid_mask.any():
        reject, corrected, _, _ = multipletests(
            p_vals[valid_mask], alpha=0.05, method="fdr_bh"
        )
        p_fdr[valid_mask] = corrected
        significant[valid_mask] = reject

    result["p_fdr"] = p_fdr
    result["significant"] = significant

    return result


# ============================================================================
# Correlation
# ============================================================================


def spearman_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> tuple[float, float]:
    """Compute Spearman rank correlation between two columns.

    NaN pairs are dropped before computation.

    Parameters
    ----------
    df : DataFrame
        Data table.
    x_col, y_col : str
        Column names.

    Returns
    -------
    rho : float
        Spearman rho.
    p_value : float
        Two-sided p-value.

    Raises
    ------
    ValueError
        If fewer than 3 valid pairs remain after dropping NaN.
    """
    subset = df[[x_col, y_col]].dropna()
    if len(subset) < 3:
        raise ValueError(
            f"Need at least 3 valid pairs for Spearman correlation, "
            f"got {len(subset)}"
        )
    rho, p_val = stats.spearmanr(subset[x_col], subset[y_col])
    return float(rho), float(p_val)


def correlation_matrix(
    df: pd.DataFrame,
    metric_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute all pairwise Spearman correlations.

    Parameters
    ----------
    df : DataFrame
        Data table.
    metric_cols : list of str
        Columns to correlate.

    Returns
    -------
    rho_matrix : DataFrame
        Spearman rho values (metric x metric).
    p_matrix : DataFrame
        Two-sided p-values (metric x metric).
    """
    n = len(metric_cols)
    rho_arr = np.full((n, n), np.nan)
    p_arr = np.full((n, n), np.nan)

    for i in range(n):
        rho_arr[i, i] = 1.0
        p_arr[i, i] = 0.0
        for j in range(i + 1, n):
            subset = df[[metric_cols[i], metric_cols[j]]].dropna()
            if len(subset) >= 3:
                rho, p_val = stats.spearmanr(
                    subset[metric_cols[i]], subset[metric_cols[j]]
                )
                rho_arr[i, j] = rho_arr[j, i] = float(rho)
                p_arr[i, j] = p_arr[j, i] = float(p_val)

    rho_df = pd.DataFrame(rho_arr, index=metric_cols, columns=metric_cols)
    p_df = pd.DataFrame(p_arr, index=metric_cols, columns=metric_cols)
    return rho_df, p_df


# ============================================================================
# Export
# ============================================================================


def save_stats_summary(
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    path: Path,
) -> None:
    """Save summary statistics and comparison results to CSV files.

    Creates two files:
    - ``{path}_summary.csv`` — descriptive statistics
    - ``{path}_comparison.csv`` — Mann-Whitney results

    Parameters
    ----------
    summary_df : DataFrame
        Output of :func:`compute_summary_stats`.
    comparison_df : DataFrame
        Output of :func:`mann_whitney_comparison`.
    path : Path
        Base path (without extension).  E.g. ``Path("results/stats")``
        produces ``results/stats_summary.csv`` and
        ``results/stats_comparison.csv``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary_path = path.parent / f"{path.name}_summary.csv"
    comparison_path = path.parent / f"{path.name}_comparison.csv"

    summary_df.to_csv(summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
