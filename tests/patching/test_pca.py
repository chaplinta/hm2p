"""Tests for hm2p.patching.pca module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hm2p.patching.pca import (
    PCAResult,
    filter_exclude_cols,
    get_metric_subsets,
    run_pca,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n_rows: int, cols: list[str], rng: np.random.Generator | None = None):
    """Create a DataFrame with random data."""
    if rng is None:
        rng = np.random.default_rng(42)
    data = rng.standard_normal((n_rows, len(cols)))
    return pd.DataFrame(data, columns=cols)


# ---------------------------------------------------------------------------
# run_pca tests
# ---------------------------------------------------------------------------


class TestRunPCA:
    """Tests for run_pca."""

    def test_basic_shapes(self):
        """Scores and loadings have correct shapes."""
        cols = ["a", "b", "c"]
        df = _make_df(20, cols)
        result = run_pca(df, cols, n_components=2)

        assert result.scores.shape == (20, 2)
        assert result.loadings.shape == (2, 3)
        assert result.explained_variance.shape == (2,)
        assert result.n_samples == 20
        assert result.n_components == 2
        assert result.feature_names == cols

    def test_perfectly_correlated_data(self):
        """PC1 explains ~100% for perfectly correlated columns."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(50)
        df = pd.DataFrame({"a": x, "b": x * 2 + 1})
        result = run_pca(df, ["a", "b"], n_components=2)

        assert result.explained_variance[0] > 0.99

    def test_orthogonal_data_equal_variance(self):
        """Orthogonal features give roughly equal variance explained."""
        n = 1000
        rng = np.random.default_rng(7)
        # Create uncorrelated data with same variance
        data = rng.standard_normal((n, 3))
        df = pd.DataFrame(data, columns=["x", "y", "z"])
        result = run_pca(df, ["x", "y", "z"], n_components=3)

        # Each component should explain roughly 1/3
        for v in result.explained_variance:
            assert 0.2 < v < 0.5

    def test_explained_variance_sums_lte_one(self):
        """Explained variance fractions sum to at most 1.0."""
        df = _make_df(30, ["a", "b", "c", "d"])
        result = run_pca(df, ["a", "b", "c", "d"], n_components=4)

        assert result.explained_variance.sum() <= 1.0 + 1e-10

    def test_nan_rows_dropped(self):
        """Rows with NaN in metric columns are excluded."""
        df = _make_df(10, ["a", "b"])
        df.loc[0, "a"] = np.nan
        df.loc[3, "b"] = np.nan
        result = run_pca(df, ["a", "b"], n_components=2)

        assert result.n_samples == 8
        assert result.scores.shape[0] == 8

    def test_extra_columns_ignored(self):
        """Columns not in metric_cols are ignored."""
        df = _make_df(15, ["a", "b", "c", "extra"])
        result = run_pca(df, ["a", "b", "c"], n_components=2)

        assert result.feature_names == ["a", "b", "c"]
        assert result.loadings.shape[1] == 3

    def test_n_components_clamped_by_features(self):
        """n_components is clamped to n_features when fewer features."""
        df = _make_df(50, ["a", "b"])
        result = run_pca(df, ["a", "b"], n_components=10)

        assert result.n_components == 2
        assert result.scores.shape[1] == 2

    def test_n_components_clamped_by_samples(self):
        """n_components is clamped to n_samples when fewer samples."""
        df = _make_df(3, ["a", "b", "c", "d", "e"])
        result = run_pca(df, ["a", "b", "c", "d", "e"], n_components=5)

        assert result.n_components == 3
        assert result.scores.shape == (3, 3)

    def test_more_features_than_samples(self):
        """Works when n_features > n_samples."""
        cols = [f"f{i}" for i in range(20)]
        df = _make_df(5, cols)
        result = run_pca(df, cols, n_components=10)

        assert result.n_components == 5
        assert result.scores.shape == (5, 5)
        assert result.loadings.shape == (5, 20)

    def test_single_component(self):
        """n_components=1 works."""
        df = _make_df(10, ["a", "b", "c"])
        result = run_pca(df, ["a", "b", "c"], n_components=1)

        assert result.n_components == 1
        assert result.scores.shape == (10, 1)

    def test_empty_metric_cols_raises(self):
        """Empty metric_cols raises ValueError."""
        df = _make_df(10, ["a"])
        with pytest.raises(ValueError, match="metric_cols must not be empty"):
            run_pca(df, [], n_components=1)

    def test_all_nan_raises(self):
        """All-NaN columns cause no rows to remain."""
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        with pytest.raises(ValueError, match="No rows remain"):
            run_pca(df, ["a", "b"])

    def test_default_n_components(self):
        """Default n_components=5 works for wide data."""
        cols = [f"f{i}" for i in range(10)]
        df = _make_df(20, cols)
        result = run_pca(df, cols)

        assert result.n_components == 5

    def test_pca_result_dataclass(self):
        """PCAResult is a proper dataclass."""
        result = PCAResult(
            scores=np.zeros((2, 1)),
            loadings=np.zeros((1, 3)),
            explained_variance=np.array([1.0]),
            feature_names=["a", "b", "c"],
            n_samples=2,
            n_components=1,
        )
        assert result.n_samples == 2
        assert result.n_components == 1


# ---------------------------------------------------------------------------
# get_metric_subsets tests
# ---------------------------------------------------------------------------


class TestGetMetricSubsets:
    """Tests for get_metric_subsets."""

    def test_expected_keys(self):
        """All expected subset names are present."""
        subsets = get_metric_subsets()
        expected = {
            "passive_ephys",
            "active_ephys",
            "all_ephys",
            "apical_morph",
            "basal_morph",
            "all_morph",
            "combined",
        }
        assert set(subsets.keys()) == expected

    def test_all_ephys_is_union(self):
        """all_ephys = passive_ephys + active_ephys."""
        subsets = get_metric_subsets()
        assert subsets["all_ephys"] == subsets["passive_ephys"] + subsets["active_ephys"]

    def test_all_morph_is_union(self):
        """all_morph = apical_morph + basal_morph."""
        subsets = get_metric_subsets()
        assert subsets["all_morph"] == subsets["apical_morph"] + subsets["basal_morph"]

    def test_combined_is_union(self):
        """combined = all_ephys + all_morph."""
        subsets = get_metric_subsets()
        assert subsets["combined"] == subsets["all_ephys"] + subsets["all_morph"]

    def test_passive_ephys_columns(self):
        """passive_ephys has the expected columns."""
        subsets = get_metric_subsets()
        assert subsets["passive_ephys"] == [
            "rmp",
            "rheobase",
            "input_resistance",
            "tau",
            "input_capacitance",
            "sag_ratio",
        ]

    def test_active_ephys_columns(self):
        """active_ephys has the expected columns."""
        subsets = get_metric_subsets()
        assert subsets["active_ephys"] == [
            "max_spike_rate",
            "min_vm",
            "peak_vm",
            "max_vm_slope",
            "half_vm",
            "amplitude",
            "max_ahp",
            "half_width",
        ]

    def test_basal_morph_has_n_basal_trees(self):
        """basal_morph includes n_basal_trees."""
        subsets = get_metric_subsets()
        assert "n_basal_trees" in subsets["basal_morph"]

    def test_returns_copies(self):
        """Returned lists are independent copies."""
        s1 = get_metric_subsets()
        s2 = get_metric_subsets()
        s1["passive_ephys"].append("extra")
        assert "extra" not in s2["passive_ephys"]


# ---------------------------------------------------------------------------
# filter_exclude_cols tests
# ---------------------------------------------------------------------------


class TestFilterExcludeCols:
    """Tests for filter_exclude_cols."""

    def test_default_exclude(self):
        """Default excludes depth-dependent columns."""
        cols = ["ap_depth", "ba_depth", "ap_length", "ba_length", "ap_wh_ratio"]
        result = filter_exclude_cols(cols)
        assert result == ["ap_length", "ba_length"]

    def test_custom_exclude(self):
        """Custom exclude list is respected."""
        cols = ["a", "b", "c", "d"]
        result = filter_exclude_cols(cols, exclude=["b", "d"])
        assert result == ["a", "c"]

    def test_empty_exclude(self):
        """Empty exclude list keeps all columns."""
        cols = ["ap_depth", "ba_depth", "ap_length"]
        result = filter_exclude_cols(cols, exclude=[])
        assert result == cols

    def test_no_overlap(self):
        """If no columns match exclude, all are kept."""
        cols = ["x", "y", "z"]
        result = filter_exclude_cols(cols)
        assert result == ["x", "y", "z"]

    def test_preserves_order(self):
        """Output order matches input order."""
        cols = ["c", "a", "b"]
        result = filter_exclude_cols(cols, exclude=["a"])
        assert result == ["c", "b"]

    def test_all_excluded(self):
        """If all columns are excluded, return empty list."""
        cols = ["ap_depth", "ba_depth"]
        result = filter_exclude_cols(cols)
        assert result == []

    def test_default_exclude_complete(self):
        """All 8 default exclusions are applied."""
        defaults = [
            "ap_depth",
            "ba_depth",
            "ap_wd_ratio",
            "ba_wd_ratio",
            "ap_height",
            "ba_height",
            "ap_wh_ratio",
            "ba_wh_ratio",
        ]
        # All defaults should be removed
        result = filter_exclude_cols(defaults)
        assert result == []

        # Mixed with non-excluded
        mixed = defaults + ["ap_length", "ba_length"]
        result = filter_exclude_cols(mixed)
        assert result == ["ap_length", "ba_length"]
