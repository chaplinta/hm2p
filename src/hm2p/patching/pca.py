"""PCA analysis for the patching pipeline.

Provides dimensionality reduction on electrophysiology and morphology
metrics, with predefined metric subsets for common analyses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAResult:
    """Result of a PCA analysis.

    Attributes
    ----------
    scores : np.ndarray
        Projected data, shape (n_samples, n_components).
    loadings : np.ndarray
        Principal component loadings, shape (n_components, n_features).
    explained_variance : np.ndarray
        Fraction of variance explained per component, shape (n_components,).
    feature_names : list[str]
        Names of the features used.
    n_samples : int
        Number of samples after dropping NaN rows.
    n_components : int
        Number of components retained.
    """

    scores: np.ndarray  # (n_samples, n_components)
    loadings: np.ndarray  # (n_components, n_features)
    explained_variance: np.ndarray  # (n_components,) fraction
    feature_names: list[str]
    n_samples: int
    n_components: int


def run_pca(
    df: pd.DataFrame,
    metric_cols: list[str],
    n_components: int = 5,
) -> PCAResult:
    """Run PCA on selected metric columns of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Rows with any NaN in ``metric_cols`` are dropped.
    metric_cols : list[str]
        Column names to include in the PCA.
    n_components : int, optional
        Maximum number of principal components to retain. Clamped to
        ``min(n_components, n_samples, n_features)``.

    Returns
    -------
    PCAResult
        Dataclass with scores, loadings, explained variance, and metadata.

    Raises
    ------
    ValueError
        If no rows remain after dropping NaNs or if ``metric_cols`` is empty.
    """
    if not metric_cols:
        raise ValueError("metric_cols must not be empty")

    sub = df[metric_cols].dropna()
    n_samples, n_features = sub.shape

    if n_samples == 0:
        raise ValueError("No rows remain after dropping NaN values")

    n_components = min(n_components, n_samples, n_features)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(sub.values)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(scaled)

    return PCAResult(
        scores=scores,
        loadings=pca.components_,
        explained_variance=pca.explained_variance_ratio_,
        feature_names=list(metric_cols),
        n_samples=n_samples,
        n_components=n_components,
    )


_PASSIVE_EPHYS = [
    "rmp",
    "rheobase",
    "input_resistance",
    "tau",
    "input_capacitance",
    "sag_ratio",
]

_ACTIVE_EPHYS = [
    "max_spike_rate",
    "min_vm",
    "peak_vm",
    "max_vm_slope",
    "half_vm",
    "amplitude",
    "max_ahp",
    "half_width",
]

_APICAL_MORPH = [
    "ap_length",
    "ap_max_path_length",
    "ap_branch_points",
    "ap_mean_eucl",
    "ap_max_branch_order",
    "ap_mean_branch_length",
    "ap_mean_path_length",
    "ap_mean_branch_order",
    "ap_width",
    "ap_height",
    "ap_depth",
    "ap_wh_ratio",
    "ap_wd_ratio",
    "ap_sholl_peak_crossings",
    "ap_sholl_peak_distance",
]

_BASAL_MORPH = [
    "ba_length",
    "ba_max_path_length",
    "ba_branch_points",
    "ba_mean_eucl",
    "ba_max_branch_order",
    "ba_mean_branch_length",
    "ba_mean_path_length",
    "ba_mean_branch_order",
    "ba_width",
    "ba_height",
    "ba_depth",
    "ba_wh_ratio",
    "ba_wd_ratio",
    "ba_sholl_peak_crossings",
    "ba_sholl_peak_distance",
    "n_basal_trees",
]

_DEFAULT_EXCLUDE = [
    "ap_depth",
    "ba_depth",
    "ap_wd_ratio",
    "ba_wd_ratio",
    "ap_height",
    "ba_height",
    "ap_wh_ratio",
    "ba_wh_ratio",
]


def get_metric_subsets() -> dict[str, list[str]]:
    """Return predefined metric subsets for PCA analysis.

    Returns
    -------
    dict[str, list[str]]
        Mapping from subset name to list of column names.
    """
    all_ephys = _PASSIVE_EPHYS + _ACTIVE_EPHYS
    all_morph = _APICAL_MORPH + _BASAL_MORPH
    return {
        "passive_ephys": list(_PASSIVE_EPHYS),
        "active_ephys": list(_ACTIVE_EPHYS),
        "all_ephys": all_ephys,
        "apical_morph": list(_APICAL_MORPH),
        "basal_morph": list(_BASAL_MORPH),
        "all_morph": all_morph,
        "combined": all_ephys + all_morph,
    }


def filter_exclude_cols(
    cols: list[str],
    exclude: list[str] | None = None,
) -> list[str]:
    """Remove excluded columns from a list.

    Parameters
    ----------
    cols : list[str]
        Input column names.
    exclude : list[str] | None, optional
        Columns to remove. Defaults to depth-dependent morphology columns:
        ``["ap_depth", "ba_depth", "ap_wd_ratio", "ba_wd_ratio",
        "ap_height", "ba_height", "ap_wh_ratio", "ba_wh_ratio"]``.

    Returns
    -------
    list[str]
        Filtered column names with excluded columns removed.
    """
    if exclude is None:
        exclude = list(_DEFAULT_EXCLUDE)
    exclude_set = set(exclude)
    return [c for c in cols if c not in exclude_set]
