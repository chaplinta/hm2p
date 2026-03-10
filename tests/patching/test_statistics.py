"""Tests for hm2p.patching.statistics — population-level stats and comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hm2p.patching.statistics import (
    compute_summary_stats,
    correlation_matrix,
    mann_whitney_comparison,
    save_stats_summary,
    spearman_correlation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_group_df() -> pd.DataFrame:
    """DataFrame with two groups and known values."""
    rng = np.random.default_rng(42)
    n_per_group = 20
    return pd.DataFrame(
        {
            "cell_type": ["penk"] * n_per_group + ["nonpenk"] * n_per_group,
            "metric_a": np.concatenate(
                [rng.normal(10, 1, n_per_group), rng.normal(10, 1, n_per_group)]
            ),
            "metric_b": np.concatenate(
                [rng.normal(0, 1, n_per_group), rng.normal(5, 1, n_per_group)]
            ),
            "metric_c": np.concatenate(
                [rng.normal(50, 5, n_per_group), rng.normal(50, 5, n_per_group)]
            ),
        }
    )


@pytest.fixture()
def correlated_df() -> pd.DataFrame:
    """DataFrame with perfectly correlated columns."""
    x = np.arange(1.0, 21.0)
    return pd.DataFrame({"x": x, "y": 2 * x + 3, "z": -x, "rand": np.random.default_rng(0).random(20)})


# ---------------------------------------------------------------------------
# compute_summary_stats
# ---------------------------------------------------------------------------


class TestComputeSummaryStats:
    def test_basic_shape(self, two_group_df: pd.DataFrame) -> None:
        cols = ["metric_a", "metric_b"]
        result = compute_summary_stats(two_group_df, cols)
        assert len(result) == 2
        assert "metric" in result.columns
        # Each group should have 7 stat columns (n, mean, median, std, sem, min, max)
        for grp in ("nonpenk", "penk"):
            for stat in ("n", "mean", "median", "std", "sem", "min", "max"):
                assert f"{grp}_{stat}" in result.columns

    def test_correct_values(self) -> None:
        df = pd.DataFrame(
            {
                "cell_type": ["A", "A", "A", "B", "B"],
                "val": [1.0, 2.0, 3.0, 10.0, 20.0],
            }
        )
        result = compute_summary_stats(df, ["val"])
        row = result.iloc[0]
        assert row["A_n"] == 3
        assert row["A_mean"] == pytest.approx(2.0)
        assert row["A_median"] == pytest.approx(2.0)
        assert row["A_min"] == pytest.approx(1.0)
        assert row["A_max"] == pytest.approx(3.0)
        assert row["B_n"] == 2
        assert row["B_mean"] == pytest.approx(15.0)

    def test_nan_values_skipped(self) -> None:
        df = pd.DataFrame(
            {
                "cell_type": ["A", "A", "B", "B"],
                "val": [1.0, np.nan, 5.0, 6.0],
            }
        )
        result = compute_summary_stats(df, ["val"])
        assert result.iloc[0]["A_n"] == 1
        assert result.iloc[0]["B_n"] == 2

    def test_single_value_per_group(self) -> None:
        df = pd.DataFrame({"cell_type": ["A", "B"], "val": [3.0, 7.0]})
        result = compute_summary_stats(df, ["val"])
        assert result.iloc[0]["A_n"] == 1
        assert result.iloc[0]["A_mean"] == pytest.approx(3.0)
        # std/sem undefined for n=1
        assert np.isnan(result.iloc[0]["A_std"])
        assert np.isnan(result.iloc[0]["A_sem"])

    def test_missing_group_col_raises(self) -> None:
        df = pd.DataFrame({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="group_col"):
            compute_summary_stats(df, ["val"])

    def test_empty_metric_cols_raises(self, two_group_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="metric_cols"):
            compute_summary_stats(two_group_df, [])

    def test_all_nan_column(self) -> None:
        df = pd.DataFrame(
            {"cell_type": ["A", "A", "B", "B"], "val": [np.nan, np.nan, np.nan, np.nan]}
        )
        result = compute_summary_stats(df, ["val"])
        assert result.iloc[0]["A_n"] == 0
        assert np.isnan(result.iloc[0]["A_mean"])


# ---------------------------------------------------------------------------
# mann_whitney_comparison
# ---------------------------------------------------------------------------


class TestMannWhitneyComparison:
    def test_identical_distributions_high_p(self) -> None:
        rng = np.random.default_rng(99)
        df = pd.DataFrame(
            {
                "cell_type": ["A"] * 50 + ["B"] * 50,
                "val": rng.normal(0, 1, 100),
            }
        )
        result = mann_whitney_comparison(df, ["val"])
        assert result.iloc[0]["p_value"] > 0.05

    def test_very_different_distributions_low_p(self) -> None:
        df = pd.DataFrame(
            {
                "cell_type": ["A"] * 30 + ["B"] * 30,
                "val": np.concatenate([np.zeros(30), np.ones(30) * 100]),
            }
        )
        result = mann_whitney_comparison(df, ["val"])
        assert result.iloc[0]["p_value"] < 0.001
        assert result.iloc[0]["significant"] is True or result.iloc[0]["significant"] == True  # noqa: E712

    def test_fdr_correction_geq_raw_p(self, two_group_df: pd.DataFrame) -> None:
        cols = ["metric_a", "metric_b", "metric_c"]
        result = mann_whitney_comparison(two_group_df, cols)
        for _, row in result.iterrows():
            if not np.isnan(row["p_value"]):
                assert row["p_fdr"] >= row["p_value"] - 1e-15  # allow float rounding

    def test_output_columns(self, two_group_df: pd.DataFrame) -> None:
        result = mann_whitney_comparison(two_group_df, ["metric_a"])
        assert set(result.columns) == {"metric", "statistic", "p_value", "p_fdr", "significant"}

    def test_nan_metric_handled(self) -> None:
        df = pd.DataFrame(
            {
                "cell_type": ["A", "A", "B", "B"],
                "val": [1.0, np.nan, np.nan, np.nan],
            }
        )
        result = mann_whitney_comparison(df, ["val"])
        # Only 1 valid in A and 0 in B → NaN
        assert np.isnan(result.iloc[0]["p_value"])
        assert np.isnan(result.iloc[0]["statistic"])

    def test_single_group_raises(self) -> None:
        df = pd.DataFrame({"cell_type": ["A", "A"], "val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="exactly 2 groups"):
            mann_whitney_comparison(df, ["val"])

    def test_three_groups_raises(self) -> None:
        df = pd.DataFrame(
            {"cell_type": ["A", "B", "C"], "val": [1.0, 2.0, 3.0]}
        )
        with pytest.raises(ValueError, match="exactly 2 groups"):
            mann_whitney_comparison(df, ["val"])

    def test_empty_metric_cols_raises(self, two_group_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="metric_cols"):
            mann_whitney_comparison(two_group_df, [])

    def test_missing_group_col_raises(self) -> None:
        df = pd.DataFrame({"val": [1.0, 2.0]})
        with pytest.raises(ValueError, match="group_col"):
            mann_whitney_comparison(df, ["val"])

    def test_multiple_metrics_fdr(self) -> None:
        """With many metrics, FDR should correct at least some p-values upward."""
        rng = np.random.default_rng(7)
        n = 30
        data = {"cell_type": ["A"] * n + ["B"] * n}
        metric_cols = []
        for i in range(10):
            col = f"m{i}"
            metric_cols.append(col)
            data[col] = rng.normal(0, 1, 2 * n)
        df = pd.DataFrame(data)
        result = mann_whitney_comparison(df, metric_cols)
        assert len(result) == 10
        # All p_fdr should be >= p_value
        for _, row in result.iterrows():
            assert row["p_fdr"] >= row["p_value"] - 1e-15


# ---------------------------------------------------------------------------
# spearman_correlation
# ---------------------------------------------------------------------------


class TestSpearmanCorrelation:
    def test_perfect_positive(self, correlated_df: pd.DataFrame) -> None:
        rho, p = spearman_correlation(correlated_df, "x", "y")
        assert rho == pytest.approx(1.0)
        assert p < 0.001

    def test_perfect_negative(self, correlated_df: pd.DataFrame) -> None:
        rho, p = spearman_correlation(correlated_df, "x", "z")
        assert rho == pytest.approx(-1.0)
        assert p < 0.001

    def test_nan_pairs_dropped(self) -> None:
        df = pd.DataFrame(
            {"x": [1, 2, 3, 4, 5, np.nan], "y": [2, 4, 6, 8, 10, 999]}
        )
        rho, p = spearman_correlation(df, "x", "y")
        assert rho == pytest.approx(1.0)

    def test_too_few_pairs_raises(self) -> None:
        df = pd.DataFrame({"x": [1.0, np.nan], "y": [2.0, 3.0]})
        with pytest.raises(ValueError, match="at least 3"):
            spearman_correlation(df, "x", "y")

    def test_return_types(self, correlated_df: pd.DataFrame) -> None:
        rho, p = spearman_correlation(correlated_df, "x", "y")
        assert isinstance(rho, float)
        assert isinstance(p, float)


# ---------------------------------------------------------------------------
# correlation_matrix
# ---------------------------------------------------------------------------


class TestCorrelationMatrix:
    def test_shape(self, correlated_df: pd.DataFrame) -> None:
        cols = ["x", "y", "z"]
        rho_m, p_m = correlation_matrix(correlated_df, cols)
        assert rho_m.shape == (3, 3)
        assert p_m.shape == (3, 3)
        assert list(rho_m.index) == cols
        assert list(rho_m.columns) == cols

    def test_diagonal_is_one(self, correlated_df: pd.DataFrame) -> None:
        cols = ["x", "y", "z"]
        rho_m, p_m = correlation_matrix(correlated_df, cols)
        for c in cols:
            assert rho_m.loc[c, c] == pytest.approx(1.0)
            assert p_m.loc[c, c] == pytest.approx(0.0)

    def test_symmetry(self, correlated_df: pd.DataFrame) -> None:
        cols = ["x", "y", "z"]
        rho_m, p_m = correlation_matrix(correlated_df, cols)
        pd.testing.assert_frame_equal(rho_m, rho_m.T)
        pd.testing.assert_frame_equal(p_m, p_m.T)

    def test_known_values(self, correlated_df: pd.DataFrame) -> None:
        cols = ["x", "y"]
        rho_m, _ = correlation_matrix(correlated_df, cols)
        assert rho_m.loc["x", "y"] == pytest.approx(1.0)

    def test_insufficient_data_gives_nan(self) -> None:
        df = pd.DataFrame(
            {"a": [1.0, np.nan, np.nan], "b": [np.nan, 2.0, np.nan]}
        )
        rho_m, p_m = correlation_matrix(df, ["a", "b"])
        assert np.isnan(rho_m.loc["a", "b"])
        assert np.isnan(p_m.loc["a", "b"])

    def test_single_column(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        rho_m, p_m = correlation_matrix(df, ["x"])
        assert rho_m.shape == (1, 1)
        assert rho_m.iloc[0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# save_stats_summary
# ---------------------------------------------------------------------------


class TestSaveStatsSummary:
    def test_creates_csv_files(self, tmp_path: Path) -> None:
        summary = pd.DataFrame({"metric": ["a"], "penk_mean": [1.0]})
        comparison = pd.DataFrame(
            {"metric": ["a"], "statistic": [10.0], "p_value": [0.05]}
        )
        base = tmp_path / "output" / "stats"
        save_stats_summary(summary, comparison, base)

        summary_path = tmp_path / "output" / "stats_summary.csv"
        comparison_path = tmp_path / "output" / "stats_comparison.csv"
        assert summary_path.exists()
        assert comparison_path.exists()

        loaded_summary = pd.read_csv(summary_path)
        assert len(loaded_summary) == 1
        assert loaded_summary.iloc[0]["penk_mean"] == pytest.approx(1.0)

        loaded_comp = pd.read_csv(comparison_path)
        assert len(loaded_comp) == 1

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        base = tmp_path / "deep" / "nested" / "dir" / "stats"
        save_stats_summary(pd.DataFrame(), pd.DataFrame(), base)
        assert (tmp_path / "deep" / "nested" / "dir" / "stats_summary.csv").exists()

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        base = tmp_path / "stats"
        df1 = pd.DataFrame({"metric": ["a"]})
        df2 = pd.DataFrame({"metric": ["a", "b"]})
        save_stats_summary(df1, df1, base)
        save_stats_summary(df2, df2, base)
        loaded = pd.read_csv(tmp_path / "stats_summary.csv")
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# Integration-style: end-to-end workflow
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_workflow(self, two_group_df: pd.DataFrame, tmp_path: Path) -> None:
        cols = ["metric_a", "metric_b", "metric_c"]
        summary = compute_summary_stats(two_group_df, cols)
        comparison = mann_whitney_comparison(two_group_df, cols)

        # metric_b should be significant (groups differ by ~5 std)
        mb_row = comparison[comparison["metric"] == "metric_b"].iloc[0]
        assert mb_row["p_value"] < 0.001

        # Save and reload
        base = tmp_path / "results"
        save_stats_summary(summary, comparison, base)
        reloaded = pd.read_csv(tmp_path / "results_summary.csv")
        assert len(reloaded) == 3

    def test_correlation_workflow(self, correlated_df: pd.DataFrame) -> None:
        rho, p = spearman_correlation(correlated_df, "x", "y")
        assert rho == pytest.approx(1.0)

        rho_m, p_m = correlation_matrix(correlated_df, ["x", "y", "z"])
        assert rho_m.loc["x", "z"] == pytest.approx(-1.0)
