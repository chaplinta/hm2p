"""Population-level statistics comparing Penk+ vs non-Penk cell types.

Reimplements the summary statistics and group comparisons from
``sumTableNum.m`` in the legacy MATLAB pipeline.

Provides:
- Per-group descriptive statistics (mean, median, SEM, etc.)
- Mann-Whitney U tests with Benjamini-Hochberg FDR correction
- Cluster (animal-aware) permutation comparisons and animal-level paired tests
- Non-parametric effect sizes (common-language effect size, Cliff's delta)
- Spearman correlation (pairwise and matrix)
- CSV export of results

Cells are nested within animals, so cell-level tests pseudoreplicate. The
animal-aware tests here follow the framework in ``docs/stats-strategy.md``:
cell-level Mann-Whitney is descriptive only, the cluster permutation test and
the animal-level paired test are the primary (non-parametric) tests, and the
linear mixed model is supplementary (ICC reporting).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Preferred ordering of the two patching cell-type labels. Differences are
# always reported as ``penkpos - penkneg``.
PREFERRED_GROUP_ORDER: tuple[str, str] = ("penkpos", "penkneg")


# ============================================================================
# Multiple-comparison correction (statsmodels optional)
# ============================================================================


def _benjamini_hochberg(
    p_values: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up FDR correction (dependency-free fallback).

    Used when ``statsmodels`` is not installed. Equivalent to
    ``statsmodels.stats.multitest.multipletests(..., method="fdr_bh")``.

    Parameters
    ----------
    p_values : ndarray
        Finite p-values, shape (n,).
    alpha : float
        Target false discovery rate.

    Returns
    -------
    reject : ndarray of bool
        True where the adjusted p-value is below *alpha*.
    p_adjusted : ndarray of float
        BH-adjusted p-values, clipped to 1.0.

    References
    ----------
    Benjamini & Hochberg 1995. "Controlling the False Discovery Rate: A
    Practical and Powerful Approach to Multiple Testing." Journal of the Royal
    Statistical Society B 57:289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)

    order = np.argsort(p)
    ranked = p[order]
    factors = n / np.arange(1, n + 1, dtype=float)
    adjusted_sorted = np.minimum.accumulate((ranked * factors)[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    p_adjusted = np.empty(n, dtype=float)
    p_adjusted[order] = adjusted_sorted
    return p_adjusted < alpha, p_adjusted


def _multipletests_fn() -> Callable[..., Any] | None:
    """Return ``statsmodels`` ``multipletests`` if importable, else ``None``.

    Imported lazily so this module works without ``statsmodels`` installed.

    Returns
    -------
    callable or None
        The ``multipletests`` function, or ``None`` when statsmodels is absent.
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        return None
    return multipletests


def fdr_correct(p_values: Iterable[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction that tolerates NaN p-values.

    NaN entries are passed through unchanged (adjusted p NaN, reject False) and
    excluded from the correction family.

    Parameters
    ----------
    p_values : iterable of float
        Raw p-values; may contain NaN.
    alpha : float
        Target false discovery rate.

    Returns
    -------
    reject : ndarray of bool
        Significance flags aligned with *p_values*.
    p_adjusted : ndarray of float
        BH-adjusted p-values aligned with *p_values*.
    """
    p = np.asarray(list(p_values), dtype=float)
    reject = np.zeros(p.size, dtype=bool)
    adjusted = np.full(p.size, np.nan, dtype=float)

    valid = np.isfinite(p)
    if not valid.any():
        return reject, adjusted

    multipletests = _multipletests_fn()
    if multipletests is None:
        rej, adj = _benjamini_hochberg(p[valid], alpha=alpha)
    else:
        rej, adj, _, _ = multipletests(p[valid], alpha=alpha, method="fdr_bh")
        rej = np.asarray(rej, dtype=bool)
        adj = np.asarray(adj, dtype=float)

    reject[valid] = rej
    adjusted[valid] = adj
    return reject, adjusted


def ordered_groups(
    values: Iterable[object],
    preferred: Sequence[str] = PREFERRED_GROUP_ORDER,
) -> list[object]:
    """Order group labels, putting *preferred* labels first when present.

    Keeps the sign of reported differences stable (``penkpos - penkneg``)
    regardless of the order labels happen to appear in a table.

    Parameters
    ----------
    values : iterable
        Group labels (NaN entries are dropped).
    preferred : sequence of str
        Labels to place first, in this order.

    Returns
    -------
    list
        Unique labels: preferred ones first, then the remainder sorted.
    """
    uniq = pd.Series(list(values)).dropna().unique().tolist()
    head = [g for g in preferred if g in uniq]
    tail = sorted((g for g in uniq if g not in head), key=str)
    return head + tail


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
            rows.append({"metric": metric, "statistic": np.nan, "p_value": np.nan})
        else:
            u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append({"metric": metric, "statistic": float(u_stat), "p_value": float(p_val)})

    result = pd.DataFrame(rows)

    # FDR correction — only on non-NaN p-values
    significant, p_fdr = fdr_correct(result["p_value"].to_numpy(dtype=float), alpha=0.05)
    result["p_fdr"] = p_fdr
    result["significant"] = significant

    return result


# ============================================================================
# Mixed model — account for animal as random effect
# ============================================================================


def mixed_model_comparison(
    df: pd.DataFrame,
    metric_cols: list[str],
    group_col: str = "cell_type",
    random_col: str = "animal_id",
) -> pd.DataFrame:
    """Supplementary LMM: animal as random intercept for ICC estimation.

    For each metric, fits::

        metric ~ cell_type + (1 | animal_id)

    using restricted maximum likelihood (REML).  This disentangles true
    cell-type differences from animal-to-animal variability (e.g. prep
    quality, fill quality, slice health).

    **This is a supplementary check only** — the primary cell-type comparison
    must use a non-parametric test (Mann–Whitney U or permutation). The LMM
    p-value (``lmm_p_supplementary``) quantifies the fixed effect but should
    not be reported as the primary significance test.

    Parameters
    ----------
    df : DataFrame
        Must contain *group_col*, *random_col*, and metric columns.
    metric_cols : list of str
        Numeric columns to test.
    group_col : str
        Fixed effect column (default ``"cell_type"``).
    random_col : str
        Random intercept grouping (default ``"animal_id"``).

    Returns
    -------
    DataFrame
        Columns: ``metric``, ``beta`` (fixed-effect coefficient for group),
        ``se``, ``z``, ``lmm_p_supplementary``, ``lmm_p_fdr``, ``lmm_significant``,
        ``icc`` (intraclass correlation — proportion of variance due to animal),
        ``n_groups`` (number of animals), ``converged``.
    """
    import warnings

    import statsmodels.formula.api as smf

    rows: list[dict[str, object]] = []

    for metric in metric_cols:
        sub = df[[metric, group_col, random_col]].dropna()
        groups = sub[group_col].unique()
        n_random = sub[random_col].nunique()

        # Need ≥2 cell types and ≥2 animals to fit a mixed model
        if len(groups) < 2 or n_random < 2 or len(sub) < 5:
            rows.append(
                {
                    "metric": metric,
                    "beta": np.nan,
                    "se": np.nan,
                    "z": np.nan,
                    "lmm_p_supplementary": np.nan,
                    "icc": np.nan,
                    "n_groups": n_random,
                    "converged": False,
                }
            )
            continue

        # Rename metric to safe formula name
        safe = "y"
        tmp = sub.rename(columns={metric: safe})

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    f"{safe} ~ C({group_col})",
                    data=tmp,
                    groups=tmp[random_col],
                )
                result = model.fit(reml=True, method="lbfgs")

            # Extract fixed effect for cell_type (the non-intercept term)
            fe_names = [n for n in result.fe_params.index if n != "Intercept"]
            if fe_names:
                beta = float(result.fe_params[fe_names[0]])
                se = float(result.bse_fe[fe_names[0]])
                z = float(result.tvalues[fe_names[0]])
                p = float(result.pvalues[fe_names[0]])
            else:
                beta = se = z = p = np.nan

            # ICC = var(animal) / (var(animal) + var(residual))
            var_animal = float(result.cov_re.iloc[0, 0])
            var_resid = float(result.scale)
            icc = var_animal / (var_animal + var_resid) if (var_animal + var_resid) > 0 else 0.0

            rows.append(
                {
                    "metric": metric,
                    "beta": beta,
                    "se": se,
                    "z": z,
                    "lmm_p_supplementary": p,
                    "icc": icc,
                    "n_groups": n_random,
                    "converged": result.converged,
                }
            )
        except Exception:
            rows.append(
                {
                    "metric": metric,
                    "beta": np.nan,
                    "se": np.nan,
                    "z": np.nan,
                    "lmm_p_supplementary": np.nan,
                    "icc": np.nan,
                    "n_groups": n_random,
                    "converged": False,
                }
            )

    result_df = pd.DataFrame(rows)

    # FDR correction
    sig, p_fdr = fdr_correct(
        result_df["lmm_p_supplementary"].to_numpy(dtype=float),
        alpha=0.05,
    )
    result_df["lmm_p_fdr"] = p_fdr
    result_df["lmm_significant"] = sig

    return result_df


# ============================================================================
# Non-parametric effect sizes
# ============================================================================


def cles(a: Iterable[float], b: Iterable[float]) -> float:
    """Common-language effect size: P(X > Y) + 0.5 * P(X = Y).

    The probability that a randomly drawn value from *a* exceeds a randomly
    drawn value from *b*, with ties counted as half. 0.5 means no effect.
    Computed from the Mann-Whitney U statistic, so ties are handled exactly.

    Parameters
    ----------
    a, b : iterable of float
        Samples to compare. NaN values are dropped.

    Returns
    -------
    float
        CLES in [0, 1], or NaN if either sample is empty after dropping NaN.

    References
    ----------
    McGraw & Wong 1992. "A common language effect size statistic."
    Psychological Bulletin 111:361-365. doi:10.1037/0033-2909.111.2.361
    """
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(u_stat) / float(x.size * y.size)


def cliffs_delta(a: Iterable[float], b: Iterable[float]) -> float:
    """Cliff's delta: P(X > Y) - P(X < Y), a rank-based effect size.

    Equal to ``2 * cles(a, b) - 1``. Ranges from -1 (all of *a* below *b*) to
    +1 (all of *a* above *b*); 0 means complete overlap.

    Parameters
    ----------
    a, b : iterable of float
        Samples to compare. NaN values are dropped.

    Returns
    -------
    float
        Cliff's delta in [-1, 1], or NaN if either sample is empty.

    References
    ----------
    Cliff 1993. "Dominance statistics: Ordinal analyses to answer ordinal
    questions." Psychological Bulletin 114:494-509.
    doi:10.1037/0033-2909.114.3.494
    """
    return 2.0 * cles(a, b) - 1.0


# ============================================================================
# Animal-aware (cluster) comparisons
# ============================================================================


def _median_difference(values: np.ndarray, is_first: np.ndarray) -> float:
    """Median of the first group minus median of the second group.

    Parameters
    ----------
    values : ndarray
        Metric values, shape (n,), finite.
    is_first : ndarray of bool
        Group membership mask, shape (n,).

    Returns
    -------
    float
        ``median(values[is_first]) - median(values[~is_first])``, NaN if either
        side is empty.
    """
    if not is_first.any() or is_first.all():
        return float("nan")
    return float(np.median(values[is_first]) - np.median(values[~is_first]))


def _within_animal_shuffle(
    is_first: np.ndarray,
    animal_index_groups: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute group labels within each animal, preserving per-animal counts.

    Parameters
    ----------
    is_first : ndarray of bool
        Observed group membership, shape (n,).
    animal_index_groups : list of ndarray
        Row indices belonging to each animal.
    rng : numpy Generator
        Random source.

    Returns
    -------
    ndarray of bool
        Permuted membership mask, shape (n,).
    """
    permuted = is_first.copy()
    for idx in animal_index_groups:
        permuted[idx] = rng.permutation(is_first[idx])
    return permuted


def _animal_level_shuffle(
    animal_codes: np.ndarray,
    animal_is_first: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute group labels at the animal level (all cells move together).

    Parameters
    ----------
    animal_codes : ndarray of int
        Per-row animal index into *animal_is_first*, shape (n,).
    animal_is_first : ndarray of bool
        Per-animal group membership, shape (n_animals,).
    rng : numpy Generator
        Random source.

    Returns
    -------
    ndarray of bool
        Permuted per-row membership mask, shape (n,).
    """
    shuffled = rng.permutation(animal_is_first)
    return shuffled[animal_codes]


def cluster_permutation_comparison(
    df: pd.DataFrame,
    metric_cols: Sequence[str],
    group_col: str = "cell_type",
    animal_col: str = "animal_id",
    n_perms: int = 5000,
    rng: np.random.Generator | int | None = None,
    within_animal: bool = True,
) -> pd.DataFrame:
    """Permutation test on group median differences, respecting animal nesting.

    Both cell types are recorded in every animal of the patching dataset, so
    the exchangeability unit is the cell *within* an animal: labels are
    permuted within each animal, which preserves each animal's cell count per
    group and removes any animal-level offset from the null. With
    ``within_animal=False`` the labels are instead permuted at the animal level
    (all cells of an animal move together), which is the correct null when a
    given animal contributes only one cell type.

    Primary non-parametric test in the framework of ``docs/stats-strategy.md``
    (cluster permutation; cell-level Mann-Whitney is descriptive only).

    Parameters
    ----------
    df : DataFrame
        Cell-level table containing *group_col*, *animal_col* and the metrics.
    metric_cols : sequence of str
        Numeric columns to test.
    group_col : str
        Column defining exactly two groups.
    animal_col : str
        Clustering column (animal identity).
    n_perms : int
        Number of permutations.
    rng : numpy Generator, int or None
        Random source or seed.
    within_animal : bool
        Permute labels within animal (True) or at the animal level (False).

    Returns
    -------
    DataFrame
        One row per metric with columns ``metric``, ``n_<group>`` and
        ``median_<group>`` for each group, ``observed_diff`` (first group minus
        second), ``cliffs_delta``, ``cles``, ``n_animals``, ``n_perms``,
        ``p_perm``, ``p_fdr``, ``significant``, ``null_median``, ``null_sd``.

    Raises
    ------
    ValueError
        If *metric_cols* is empty, a required column is missing, *n_perms* is
        not positive, or *group_col* does not contain exactly two groups.

    References
    ----------
    Benjamini & Hochberg 1995. "Controlling the False Discovery Rate: A
    Practical and Powerful Approach to Multiple Testing." Journal of the Royal
    Statistical Society B 57:289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
    """
    metric_cols = list(metric_cols)
    if not metric_cols:
        raise ValueError("metric_cols must not be empty")
    for col in (group_col, animal_col):
        if col not in df.columns:
            raise ValueError(f"column {col!r} not found in DataFrame columns")
    missing = [c for c in metric_cols if c not in df.columns]
    if missing:
        raise ValueError(f"metric columns not found in DataFrame: {missing}")
    if n_perms < 1:
        raise ValueError(f"n_perms must be >= 1, got {n_perms}")

    groups = ordered_groups(df[group_col])
    if len(groups) != 2:
        raise ValueError(
            f"cluster permutation requires exactly 2 groups in {group_col!r}, "
            f"found {len(groups)}: {groups}"
        )
    g1, g2 = groups
    generator = np.random.default_rng(rng)

    rows: list[dict[str, object]] = []
    for metric in metric_cols:
        sub = df[[metric, group_col, animal_col]].dropna()
        values = sub[metric].to_numpy(dtype=float)
        is_g1 = (sub[group_col] == g1).to_numpy()
        animal_labels = sub[animal_col].to_numpy()
        animal_names, animal_codes = np.unique(animal_labels, return_inverse=True)

        row: dict[str, object] = {
            "metric": metric,
            f"n_{g1}": int(is_g1.sum()),
            f"n_{g2}": int((~is_g1).sum()),
            f"median_{g1}": float(np.median(values[is_g1])) if is_g1.any() else np.nan,
            f"median_{g2}": float(np.median(values[~is_g1])) if (~is_g1).any() else np.nan,
            "n_animals": int(animal_names.size),
            "n_perms": int(n_perms),
            "within_animal": bool(within_animal),
        }
        observed = _median_difference(values, is_g1)
        row["observed_diff"] = observed
        row["cliffs_delta"] = cliffs_delta(values[is_g1], values[~is_g1])
        row["cles"] = cles(values[is_g1], values[~is_g1])

        if not np.isfinite(observed):
            row.update({"p_perm": np.nan, "null_median": np.nan, "null_sd": np.nan})
            rows.append(row)
            continue

        if within_animal:
            index_groups = [np.flatnonzero(animal_codes == c) for c in range(animal_names.size)]
            null = np.array(
                [
                    _median_difference(
                        values, _within_animal_shuffle(is_g1, index_groups, generator)
                    )
                    for _ in range(n_perms)
                ]
            )
        else:
            # One label per animal: take the animal's first observed label.
            animal_is_g1 = np.array(
                [bool(is_g1[animal_codes == c][0]) for c in range(animal_names.size)]
            )
            null = np.array(
                [
                    _median_difference(
                        values, _animal_level_shuffle(animal_codes, animal_is_g1, generator)
                    )
                    for _ in range(n_perms)
                ]
            )

        finite_null = null[np.isfinite(null)]
        if finite_null.size == 0:
            row.update({"p_perm": np.nan, "null_median": np.nan, "null_sd": np.nan})
        else:
            n_extreme = int(np.sum(np.abs(finite_null) >= abs(observed)))
            row["p_perm"] = (n_extreme + 1) / (finite_null.size + 1)
            row["null_median"] = float(np.median(finite_null))
            row["null_sd"] = float(np.std(finite_null))
        rows.append(row)

    result = pd.DataFrame(rows)
    significant, p_fdr = fdr_correct(result["p_perm"].to_numpy(dtype=float), alpha=0.05)
    result["p_fdr"] = p_fdr
    result["significant"] = significant
    return result


def per_animal_differences(
    df: pd.DataFrame,
    metric: str,
    group_col: str = "cell_type",
    animal_col: str = "animal_id",
) -> pd.DataFrame:
    """Within-animal median difference between the two groups, per animal.

    Parameters
    ----------
    df : DataFrame
        Cell-level table.
    metric : str
        Metric column.
    group_col : str
        Column defining exactly two groups (difference is first minus second
        in :func:`ordered_groups` order).
    animal_col : str
        Animal identity column.

    Returns
    -------
    DataFrame
        Columns ``animal_id`` (named after *animal_col*), ``n_<group>``,
        ``median_<group>`` and ``diff``. Animals lacking either group have
        ``diff`` NaN.

    Raises
    ------
    ValueError
        If a required column is missing or *group_col* has other than 2 groups.
    """
    for col in (metric, group_col, animal_col):
        if col not in df.columns:
            raise ValueError(f"column {col!r} not found in DataFrame columns")

    groups = ordered_groups(df[group_col])
    if len(groups) != 2:
        raise ValueError(
            f"per-animal differences require exactly 2 groups in {group_col!r}, "
            f"found {len(groups)}: {groups}"
        )
    g1, g2 = groups

    sub = df[[metric, group_col, animal_col]].dropna()
    rows: list[dict[str, object]] = []
    for animal, chunk in sub.groupby(animal_col, sort=True):
        v1 = chunk.loc[chunk[group_col] == g1, metric].to_numpy(dtype=float)
        v2 = chunk.loc[chunk[group_col] == g2, metric].to_numpy(dtype=float)
        m1 = float(np.median(v1)) if v1.size else np.nan
        m2 = float(np.median(v2)) if v2.size else np.nan
        rows.append(
            {
                animal_col: animal,
                f"n_{g1}": int(v1.size),
                f"n_{g2}": int(v2.size),
                f"median_{g1}": m1,
                f"median_{g2}": m2,
                "diff": m1 - m2,
            }
        )
    columns = [animal_col, f"n_{g1}", f"n_{g2}", f"median_{g1}", f"median_{g2}", "diff"]
    return pd.DataFrame(rows, columns=columns)


def animal_level_comparison(
    df: pd.DataFrame,
    metric_cols: Sequence[str],
    group_col: str = "cell_type",
    animal_col: str = "animal_id",
    min_animals: int = 5,
) -> pd.DataFrame:
    """Paired within-animal comparison: Wilcoxon signed-rank across animals.

    Each animal contributes one value per metric — the within-animal difference
    of group medians (first group minus second). Because both cell types are
    sampled in each animal, the animals are paired, so the test across animals
    is a Wilcoxon signed-rank test against zero. This removes the
    pseudoreplication of the cell-level test at the cost of power (one
    observation per animal).

    With fewer than *min_animals* usable animals the signed-rank test cannot
    reach p < 0.05 two-sided, so the p-value is reported as NaN and only the
    sign counts are informative.

    Parameters
    ----------
    df : DataFrame
        Cell-level table.
    metric_cols : sequence of str
        Numeric columns to test.
    group_col : str
        Column defining the two groups.
    animal_col : str
        Animal identity column.
    min_animals : int
        Minimum number of animals with a defined difference for a p-value to be
        reported (default 5).

    Returns
    -------
    DataFrame
        Columns ``metric``, ``n_animals``, ``median_diff``, ``mean_diff``,
        ``n_positive``, ``n_negative``, ``n_zero``, ``statistic``, ``p_value``,
        ``p_fdr``, ``significant``.

    Raises
    ------
    ValueError
        If *metric_cols* is empty or a required column is missing.
    """
    metric_cols = list(metric_cols)
    if not metric_cols:
        raise ValueError("metric_cols must not be empty")

    rows: list[dict[str, object]] = []
    for metric in metric_cols:
        diffs_df = per_animal_differences(df, metric, group_col=group_col, animal_col=animal_col)
        diffs = diffs_df["diff"].to_numpy(dtype=float)
        diffs = diffs[np.isfinite(diffs)]

        row: dict[str, object] = {
            "metric": metric,
            "n_animals": int(diffs.size),
            "median_diff": float(np.median(diffs)) if diffs.size else np.nan,
            "mean_diff": float(np.mean(diffs)) if diffs.size else np.nan,
            "n_positive": int(np.sum(diffs > 0)),
            "n_negative": int(np.sum(diffs < 0)),
            "n_zero": int(np.sum(diffs == 0)),
        }
        if diffs.size >= min_animals and np.any(diffs != 0):
            w_stat, p_val = stats.wilcoxon(diffs)
            row["statistic"] = float(w_stat)
            row["p_value"] = float(p_val)
        else:
            row["statistic"] = np.nan
            row["p_value"] = np.nan
        rows.append(row)

    result = pd.DataFrame(rows)
    significant, p_fdr = fdr_correct(result["p_value"].to_numpy(dtype=float), alpha=0.05)
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
            f"Need at least 3 valid pairs for Spearman correlation, got {len(subset)}"
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
                rho, p_val = stats.spearmanr(subset[metric_cols[i]], subset[metric_cols[j]])
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
