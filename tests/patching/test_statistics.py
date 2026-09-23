"""Tests for hm2p.patching.statistics — population-level stats and comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hm2p.patching import statistics as stats_mod
from hm2p.patching.statistics import (
    _animal_level_shuffle,
    _benjamini_hochberg,
    _median_difference,
    _multipletests_fn,
    _within_animal_shuffle,
    animal_level_comparison,
    cles,
    cliffs_delta,
    cluster_permutation_comparison,
    compute_summary_stats,
    correlation_matrix,
    fdr_correct,
    mann_whitney_comparison,
    ordered_groups,
    per_animal_differences,
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
    return pd.DataFrame(
        {"x": x, "y": 2 * x + 3, "z": -x, "rand": np.random.default_rng(0).random(20)}
    )


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
        df = pd.DataFrame({"cell_type": ["A", "B", "C"], "val": [1.0, 2.0, 3.0]})
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
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, np.nan], "y": [2, 4, 6, 8, 10, 999]})
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
        df = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [np.nan, 2.0, np.nan]})
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
        comparison = pd.DataFrame({"metric": ["a"], "statistic": [10.0], "p_value": [0.05]})
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


# ---------------------------------------------------------------------------
# FDR correction (statsmodels optional)
# ---------------------------------------------------------------------------


def _make_nested_df(
    seed: int = 0,
    n_animals: int = 6,
    n_per_type: int = 3,
    effect: float = 0.0,
    animal_sd: float = 5.0,
) -> pd.DataFrame:
    """Synthetic nested table: both cell types sampled in every animal."""
    rng = np.random.default_rng(seed)
    rows = []
    for animal in range(n_animals):
        offset = rng.normal(0.0, animal_sd)
        for cell_type in ("penkpos", "penkneg"):
            shift = effect if cell_type == "penkpos" else 0.0
            for _ in range(n_per_type):
                rows.append(
                    {
                        "animal_id": f"a{animal}",
                        "cell_type": cell_type,
                        "metric": offset + shift + rng.normal(0, 1),
                        "noise": rng.normal(0, 1),
                    }
                )
    return pd.DataFrame(rows)


class TestBenjaminiHochberg:
    def test_known_values(self) -> None:
        p = np.array([0.01, 0.02, 0.03, 0.04])
        reject, adjusted = _benjamini_hochberg(p, alpha=0.05)
        # BH: p * n / rank, enforced monotone
        assert adjusted == pytest.approx([0.04, 0.04, 0.04, 0.04])
        assert reject.all()

    def test_monotone_and_clipped(self) -> None:
        p = np.array([0.5, 0.9, 0.99])
        _, adjusted = _benjamini_hochberg(p)
        assert np.all(adjusted <= 1.0)
        assert np.all(np.diff(adjusted[np.argsort(p)]) >= -1e-12)

    def test_empty_input(self) -> None:
        reject, adjusted = _benjamini_hochberg(np.array([]))
        assert reject.size == 0
        assert adjusted.size == 0

    def test_single_value_unchanged(self) -> None:
        _, adjusted = _benjamini_hochberg(np.array([0.03]))
        assert adjusted[0] == pytest.approx(0.03)


class TestMultipletestsFn:
    def test_returns_callable_or_none(self) -> None:
        fn = _multipletests_fn()
        assert fn is None or callable(fn)


class TestFdrCorrect:
    def test_nan_passthrough(self) -> None:
        reject, adjusted = fdr_correct([0.001, np.nan, 0.5])
        assert np.isnan(adjusted[1])
        assert not reject[1]
        assert reject[0]

    def test_all_nan(self) -> None:
        reject, adjusted = fdr_correct([np.nan, np.nan])
        assert not reject.any()
        assert np.isnan(adjusted).all()

    def test_matches_internal_fallback(self) -> None:
        p = [0.001, 0.02, 0.4]
        _, adjusted = fdr_correct(p)
        _, expected = _benjamini_hochberg(np.array(p))
        assert adjusted == pytest.approx(expected)

    def test_fallback_used_when_statsmodels_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(stats_mod, "_multipletests_fn", lambda: None)
        reject, adjusted = fdr_correct([0.001, 0.5])
        assert adjusted == pytest.approx([0.002, 0.5])
        assert list(reject) == [True, False]

    def test_statsmodels_branch_used_when_available(self, monkeypatch) -> None:
        calls = {}

        def fake_multipletests(p, alpha, method):
            calls["method"] = method
            return np.array([True, False]), np.array([0.5, 0.75]), None, None

        monkeypatch.setattr(stats_mod, "_multipletests_fn", lambda: fake_multipletests)
        reject, adjusted = fdr_correct([0.25, 0.75])
        assert calls["method"] == "fdr_bh"
        assert adjusted == pytest.approx([0.5, 0.75])
        assert list(reject) == [True, False]

    def test_real_statsmodels_agrees_with_fallback(self) -> None:
        pytest.importorskip("statsmodels")
        from statsmodels.stats.multitest import multipletests

        p = np.array([0.001, 0.02, 0.4, 0.9])
        _, expected, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
        _, adjusted = _benjamini_hochberg(p)
        assert adjusted == pytest.approx(expected)


class TestOrderedGroups:
    def test_preferred_first(self) -> None:
        assert ordered_groups(["penkneg", "penkpos", "penkneg"]) == ["penkpos", "penkneg"]

    def test_unknown_labels_sorted(self) -> None:
        assert ordered_groups(["b", "a"]) == ["a", "b"]

    def test_mixed_preferred_and_other(self) -> None:
        assert ordered_groups(["zz", "penkneg"]) == ["penkneg", "zz"]

    def test_nan_dropped(self) -> None:
        assert ordered_groups(["penkpos", np.nan]) == ["penkpos"]

    def test_custom_preferred(self) -> None:
        assert ordered_groups(["a", "b"], preferred=("b",)) == ["b", "a"]


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


class TestEffectSizes:
    def test_cles_complete_separation(self) -> None:
        assert cles([3.0, 4.0], [1.0, 2.0]) == pytest.approx(1.0)
        assert cles([1.0, 2.0], [3.0, 4.0]) == pytest.approx(0.0)

    def test_cles_identical_samples_is_half(self) -> None:
        assert cles([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.5)

    def test_cles_ties_count_as_half(self) -> None:
        assert cles([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.5)

    def test_cles_nan_dropped(self) -> None:
        assert cles([3.0, np.nan], [1.0]) == pytest.approx(1.0)

    def test_cles_empty_is_nan(self) -> None:
        assert np.isnan(cles([], [1.0]))
        assert np.isnan(cles([1.0], []))

    def test_cliffs_delta_range(self) -> None:
        assert cliffs_delta([3.0, 4.0], [1.0, 2.0]) == pytest.approx(1.0)
        assert cliffs_delta([1.0, 2.0], [3.0, 4.0]) == pytest.approx(-1.0)
        assert cliffs_delta([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)

    def test_cliffs_delta_is_rescaled_cles(self) -> None:
        a = [1.0, 5.0, 3.0]
        b = [2.0, 4.0]
        assert cliffs_delta(a, b) == pytest.approx(2 * cles(a, b) - 1)

    def test_cliffs_delta_empty_is_nan(self) -> None:
        assert np.isnan(cliffs_delta([], [1.0]))


# ---------------------------------------------------------------------------
# Permutation helpers
# ---------------------------------------------------------------------------


class TestPermutationHelpers:
    def test_median_difference(self) -> None:
        values = np.array([1.0, 3.0, 10.0, 12.0])
        is_first = np.array([True, True, False, False])
        assert _median_difference(values, is_first) == pytest.approx(-9.0)

    def test_median_difference_single_group_is_nan(self) -> None:
        values = np.array([1.0, 2.0])
        assert np.isnan(_median_difference(values, np.array([True, True])))
        assert np.isnan(_median_difference(values, np.array([False, False])))

    def test_within_animal_shuffle_preserves_counts(self) -> None:
        is_first = np.array([True, False, True, False, False, False])
        groups = [np.array([0, 1, 2]), np.array([3, 4, 5])]
        rng = np.random.default_rng(0)
        permuted = _within_animal_shuffle(is_first, groups, rng)
        assert permuted[:3].sum() == 2
        assert permuted[3:].sum() == 0
        assert permuted.sum() == is_first.sum()

    def test_within_animal_shuffle_can_reorder(self) -> None:
        is_first = np.array([True, False, False, True])
        groups = [np.arange(4)]
        rng = np.random.default_rng(1)
        seen = {tuple(_within_animal_shuffle(is_first, groups, rng)) for _ in range(50)}
        assert len(seen) > 1

    def test_animal_level_shuffle_keeps_animals_together(self) -> None:
        animal_codes = np.array([0, 0, 1, 1, 2, 2])
        animal_is_first = np.array([True, False, True])
        rng = np.random.default_rng(0)
        permuted = _animal_level_shuffle(animal_codes, animal_is_first, rng)
        for code in (0, 1, 2):
            assert len(set(permuted[animal_codes == code])) == 1
        assert permuted.sum() == 4


# ---------------------------------------------------------------------------
# cluster_permutation_comparison
# ---------------------------------------------------------------------------


class TestClusterPermutationComparison:
    def test_detects_within_animal_effect(self) -> None:
        df = _make_nested_df(seed=0, effect=4.0, n_per_type=6)
        out = cluster_permutation_comparison(
            df, ["metric"], n_perms=500, rng=0, within_animal=True
        )
        row = out.iloc[0]
        assert row["observed_diff"] > 2.0
        assert row["p_perm"] < 0.01
        assert bool(row["significant"])
        assert row["n_animals"] == 6
        assert row["cliffs_delta"] > 0.5
        assert row["cles"] > 0.75

    def test_null_effect_gives_large_p(self) -> None:
        df = _make_nested_df(seed=1, effect=0.0, n_per_type=6)
        out = cluster_permutation_comparison(df, ["metric"], n_perms=500, rng=0)
        assert out.iloc[0]["p_perm"] > 0.05
        assert not bool(out.iloc[0]["significant"])

    def test_animal_level_shuffle_mode(self) -> None:
        # Each animal contributes one cell type only — animal-level null applies.
        rng = np.random.default_rng(2)
        rows = []
        for animal in range(6):
            cell_type = "penkpos" if animal < 3 else "penkneg"
            for _ in range(4):
                rows.append(
                    {
                        "animal_id": f"a{animal}",
                        "cell_type": cell_type,
                        "metric": (5.0 if cell_type == "penkpos" else 0.0) + rng.normal(0, 1),
                    }
                )
        df = pd.DataFrame(rows)
        out = cluster_permutation_comparison(
            df, ["metric"], n_perms=500, rng=0, within_animal=False
        )
        assert out.iloc[0]["observed_diff"] > 3.0
        assert out.iloc[0]["p_perm"] < 0.15  # only C(6,3)=20 distinct animal splits
        assert not bool(out.iloc[0]["within_animal"])

    def test_columns_present(self) -> None:
        df = _make_nested_df(seed=0)
        out = cluster_permutation_comparison(df, ["metric", "noise"], n_perms=50, rng=0)
        for col in (
            "metric",
            "n_penkpos",
            "n_penkneg",
            "median_penkpos",
            "median_penkneg",
            "observed_diff",
            "p_perm",
            "p_fdr",
            "significant",
            "null_median",
            "null_sd",
            "cles",
            "cliffs_delta",
        ):
            assert col in out.columns
        assert len(out) == 2

    def test_reproducible_with_seed(self) -> None:
        df = _make_nested_df(seed=0, effect=2.0)
        a = cluster_permutation_comparison(df, ["metric"], n_perms=200, rng=3)
        b = cluster_permutation_comparison(df, ["metric"], n_perms=200, rng=3)
        assert a.iloc[0]["p_perm"] == b.iloc[0]["p_perm"]

    def test_all_nan_metric_gives_nan_p(self) -> None:
        df = _make_nested_df(seed=0)
        df["metric"] = np.nan
        out = cluster_permutation_comparison(df, ["metric"], n_perms=50, rng=0)
        assert np.isnan(out.iloc[0]["p_perm"])
        assert np.isnan(out.iloc[0]["observed_diff"])

    def test_single_group_metric_gives_nan_p(self) -> None:
        df = _make_nested_df(seed=0)
        df.loc[df["cell_type"] == "penkneg", "metric"] = np.nan
        out = cluster_permutation_comparison(df, ["metric"], n_perms=50, rng=0)
        assert np.isnan(out.iloc[0]["p_perm"])

    def test_empty_metric_cols_raises(self) -> None:
        with pytest.raises(ValueError, match="metric_cols"):
            cluster_permutation_comparison(_make_nested_df(), [])

    def test_missing_group_col_raises(self) -> None:
        df = _make_nested_df().drop(columns=["cell_type"])
        with pytest.raises(ValueError, match="cell_type"):
            cluster_permutation_comparison(df, ["metric"])

    def test_missing_metric_col_raises(self) -> None:
        with pytest.raises(ValueError, match="metric columns not found"):
            cluster_permutation_comparison(_make_nested_df(), ["absent"])

    def test_bad_n_perms_raises(self) -> None:
        with pytest.raises(ValueError, match="n_perms"):
            cluster_permutation_comparison(_make_nested_df(), ["metric"], n_perms=0)

    def test_three_groups_raises(self) -> None:
        df = _make_nested_df()
        df.loc[0, "cell_type"] = "other"
        with pytest.raises(ValueError, match="exactly 2 groups"):
            cluster_permutation_comparison(df, ["metric"], n_perms=10)


# ---------------------------------------------------------------------------
# per_animal_differences / animal_level_comparison
# ---------------------------------------------------------------------------


class TestPerAnimalDifferences:
    def test_known_differences(self) -> None:
        df = pd.DataFrame(
            {
                "animal_id": ["a", "a", "b", "b"],
                "cell_type": ["penkpos", "penkneg", "penkpos", "penkneg"],
                "metric": [5.0, 1.0, 8.0, 2.0],
            }
        )
        out = per_animal_differences(df, "metric")
        assert list(out["animal_id"]) == ["a", "b"]
        assert out["diff"].to_list() == pytest.approx([4.0, 6.0])
        assert list(out["n_penkpos"]) == [1, 1]

    def test_animal_missing_a_group_gives_nan(self) -> None:
        df = pd.DataFrame(
            {
                "animal_id": ["a", "a", "b"],
                "cell_type": ["penkpos", "penkneg", "penkpos"],
                "metric": [5.0, 1.0, 8.0],
            }
        )
        out = per_animal_differences(df, "metric")
        assert np.isnan(out.loc[out["animal_id"] == "b", "diff"].iloc[0])

    def test_missing_column_raises(self) -> None:
        with pytest.raises(ValueError, match="absent"):
            per_animal_differences(_make_nested_df(), "absent")

    def test_three_groups_raises(self) -> None:
        df = _make_nested_df()
        df.loc[0, "cell_type"] = "other"
        with pytest.raises(ValueError, match="exactly 2 groups"):
            per_animal_differences(df, "metric")


class TestAnimalLevelComparison:
    def test_consistent_effect_detected(self) -> None:
        df = _make_nested_df(seed=0, effect=5.0, n_per_type=4, n_animals=8)
        out = animal_level_comparison(df, ["metric"])
        row = out.iloc[0]
        assert row["n_animals"] == 8
        assert row["median_diff"] > 2.0
        assert row["n_positive"] == 8
        assert row["n_negative"] == 0
        assert row["p_value"] < 0.05

    def test_null_effect_not_significant(self) -> None:
        df = _make_nested_df(seed=5, effect=0.0, n_per_type=4, n_animals=8)
        out = animal_level_comparison(df, ["metric"])
        assert out.iloc[0]["p_value"] > 0.05

    def test_too_few_animals_gives_nan_p(self) -> None:
        df = _make_nested_df(seed=0, effect=5.0, n_animals=4)
        out = animal_level_comparison(df, ["metric"])
        assert np.isnan(out.iloc[0]["p_value"])
        assert out.iloc[0]["n_positive"] == 4

    def test_min_animals_parameter(self) -> None:
        df = _make_nested_df(seed=0, effect=5.0, n_animals=4)
        out = animal_level_comparison(df, ["metric"], min_animals=3)
        assert np.isfinite(out.iloc[0]["p_value"])

    def test_all_zero_differences_gives_nan_p(self) -> None:
        rows = []
        for animal in range(6):
            for cell_type in ("penkpos", "penkneg"):
                rows.append({"animal_id": f"a{animal}", "cell_type": cell_type, "metric": 1.0})
        out = animal_level_comparison(pd.DataFrame(rows), ["metric"])
        assert np.isnan(out.iloc[0]["p_value"])
        assert out.iloc[0]["n_zero"] == 6

    def test_columns_and_multiple_metrics(self) -> None:
        df = _make_nested_df(seed=0, effect=1.0)
        out = animal_level_comparison(df, ["metric", "noise"])
        assert len(out) == 2
        for col in (
            "metric",
            "n_animals",
            "median_diff",
            "mean_diff",
            "n_positive",
            "n_negative",
            "n_zero",
            "statistic",
            "p_value",
            "p_fdr",
            "significant",
        ):
            assert col in out.columns

    def test_no_usable_animals_gives_nan(self) -> None:
        df = _make_nested_df(seed=0)
        df["metric"] = np.nan
        out = animal_level_comparison(df, ["metric"])
        assert out.iloc[0]["n_animals"] == 0
        assert np.isnan(out.iloc[0]["median_diff"])

    def test_empty_metric_cols_raises(self) -> None:
        with pytest.raises(ValueError, match="metric_cols"):
            animal_level_comparison(_make_nested_df(), [])


class TestClusterPermutationDegenerateNull:
    def test_all_nan_null_gives_nan_p(self, monkeypatch) -> None:
        """Defensive path: if no permutation yields a finite statistic, p is NaN."""
        df = _make_nested_df(seed=0, effect=2.0)
        monkeypatch.setattr(
            stats_mod,
            "_within_animal_shuffle",
            lambda is_first, groups, rng: np.ones_like(is_first, dtype=bool),
        )
        out = cluster_permutation_comparison(df, ["metric"], n_perms=10, rng=0)
        assert np.isfinite(out.iloc[0]["observed_diff"])
        assert np.isnan(out.iloc[0]["p_perm"])
        assert np.isnan(out.iloc[0]["null_sd"])
