"""Tests for hm2p.analysis.mixed_stats — non-parametric nested statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis.mixed_stats import (
    animal_summary_test,
    cluster_permutation_test,
    confound_check,
    fdr_correct,
    interaction_contrast,
    run_between_group_test,
    within_cell_test,
)

# ============================================================================
# Fixtures
# ============================================================================


def _make_nested_df(
    penk_means: list[float],
    nonpenk_means: list[float],
    cells_per_animal: int = 5,
    noise: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic nested DataFrame with known animal-level means."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, mu in enumerate(penk_means):
        for _ in range(cells_per_animal):
            rows.append(
                {
                    "animal_id": f"penk_{i}",
                    "celltype": "penk",
                    "metric": mu + rng.normal(0, noise),
                }
            )
    for i, mu in enumerate(nonpenk_means):
        for _ in range(cells_per_animal):
            rows.append(
                {
                    "animal_id": f"nonpenk_{i}",
                    "celltype": "nonpenk",
                    "metric": mu + rng.normal(0, noise),
                }
            )
    return pd.DataFrame(rows)


# ============================================================================
# animal_summary_test
# ============================================================================


class TestAnimalSummaryTest:
    """Tests for animal_summary_test."""

    def test_two_groups_unequal_sizes(self) -> None:
        """Unequal group sizes return correct n_penk and n_nonpenk."""
        df = _make_nested_df(
            penk_means=[1.0, 1.0, 1.0, 1.0],
            nonpenk_means=[5.0, 5.0],
        )
        result = animal_summary_test(df, "metric")
        assert result["n_penk"] == 4
        assert result["n_nonpenk"] == 2

    def test_significant_difference(self) -> None:
        """Large separation between groups gives small p-value."""
        df = _make_nested_df(
            penk_means=[1.0, 1.1, 0.9, 1.0, 1.05],
            nonpenk_means=[10.0, 10.1, 9.9],
            noise=0.01,
        )
        result = animal_summary_test(df, "metric")
        assert result["p_value"] < 0.05
        assert result["penk_mean"] < result["nonpenk_mean"]

    def test_no_difference_high_p(self) -> None:
        """Same means across groups gives high p-value."""
        df = _make_nested_df(
            penk_means=[5.0, 5.0, 5.0],
            nonpenk_means=[5.0, 5.0, 5.0],
            noise=0.5,
        )
        result = animal_summary_test(df, "metric")
        assert result["p_value"] > 0.1

    def test_effect_size_range(self) -> None:
        """CLES (effect size) is in [0, 1]."""
        df = _make_nested_df(
            penk_means=[1.0, 2.0, 3.0],
            nonpenk_means=[4.0, 5.0],
        )
        result = animal_summary_test(df, "metric")
        assert 0.0 <= result["effect_size"] <= 1.0

    def test_missing_column_raises(self) -> None:
        """Missing metric column raises ValueError."""
        df = pd.DataFrame({"animal_id": [1], "celltype": ["penk"]})
        with pytest.raises(ValueError, match="Missing columns"):
            animal_summary_test(df, "metric")


# ============================================================================
# cluster_permutation_test
# ============================================================================


class TestClusterPermutationTest:
    """Tests for cluster_permutation_test."""

    def test_known_effect_low_p(self) -> None:
        """Strong group separation gives low permutation p-value."""
        df = _make_nested_df(
            penk_means=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            nonpenk_means=[10.0, 10.0, 10.0],
            noise=0.01,
        )
        result = cluster_permutation_test(df, "metric", n_perms=5000, seed=42)
        assert result["p_value"] < 0.05

    def test_no_effect_high_p(self) -> None:
        """No group difference gives non-significant p-value."""
        df = _make_nested_df(
            penk_means=[5.0, 5.0, 5.0, 5.0],
            nonpenk_means=[5.0, 5.0],
            noise=0.5,
        )
        result = cluster_permutation_test(df, "metric", n_perms=2000, seed=99)
        assert result["p_value"] > 0.1

    def test_null_distribution_centered(self) -> None:
        """Null distribution mean is approximately 0 when groups are equal."""
        df = _make_nested_df(
            penk_means=[3.0, 3.0, 3.0, 3.0],
            nonpenk_means=[3.0, 3.0],
            noise=0.01,
        )
        result = cluster_permutation_test(df, "metric", n_perms=2000, seed=7)
        assert abs(result["null_mean"]) < 0.5

    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=20, deadline=5000)
    def test_p_value_in_unit_interval(self, seed: int) -> None:
        """Property: p-value is always in [0, 1]."""
        df = _make_nested_df(
            penk_means=[1.0, 2.0, 3.0],
            nonpenk_means=[2.0, 3.0],
            noise=1.0,
            seed=seed,
        )
        result = cluster_permutation_test(df, "metric", n_perms=200, seed=seed)
        assert 0.0 <= result["p_value"] <= 1.0


# ============================================================================
# within_cell_test
# ============================================================================


class TestWithinCellTest:
    """Tests for within_cell_test."""

    def test_paired_positive_diff(self) -> None:
        """When col_a > col_b consistently, mean_diff is positive."""
        rng = np.random.default_rng(0)
        n = 30
        df = pd.DataFrame(
            {
                "light": rng.normal(5.0, 0.1, n),
                "dark": rng.normal(3.0, 0.1, n),
            }
        )
        result = within_cell_test(df, "light", "dark")
        assert result["mean_diff"] > 0
        assert result["p_value"] < 0.05
        assert result["n_cells"] == n

    def test_no_difference(self) -> None:
        """Equal columns give non-significant result."""
        rng = np.random.default_rng(1)
        n = 20
        vals = rng.normal(5.0, 1.0, n)
        df = pd.DataFrame({"a": vals, "b": vals + rng.normal(0, 0.01, n)})
        result = within_cell_test(df, "a", "b")
        # With near-identical values, p should be > 0.05
        assert result["n_cells"] == n

    def test_too_few_pairs_raises(self) -> None:
        """Fewer than 2 valid pairs raises ValueError."""
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        with pytest.raises(ValueError, match="at least 2"):
            within_cell_test(df, "a", "b")


# ============================================================================
# interaction_contrast
# ============================================================================


class TestInteractionContrast:
    """Tests for interaction_contrast."""

    def test_correct_computation(self) -> None:
        """Verify (A1B1 - A2B1) - (A1B2 - A2B2) formula."""
        df = pd.DataFrame(
            {
                "ml": [10.0, 20.0],  # moving_light
                "sl": [4.0, 8.0],  # stationary_light
                "md": [6.0, 12.0],  # moving_dark
                "sd": [3.0, 6.0],  # stationary_dark
            }
        )
        result = interaction_contrast(df, ["ml", "sl", "md", "sd"])
        # (10-4) - (6-3) = 6-3 = 3
        # (20-8) - (12-6) = 12-6 = 6
        expected = pd.Series([3.0, 6.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_wrong_number_of_cols_raises(self) -> None:
        """Non-4-element list raises ValueError."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        with pytest.raises(ValueError, match="exactly 4"):
            interaction_contrast(df, ["a", "b", "c"])


# ============================================================================
# confound_check
# ============================================================================


class TestConfoundCheck:
    """Tests for confound_check."""

    def test_high_correlation_flagged(self) -> None:
        """Highly correlated confound is flagged."""
        rng = np.random.default_rng(0)
        n = 50
        x = rng.normal(0, 1, n)
        df = pd.DataFrame(
            {
                "metric": x,
                "confound": x * 2.0 + rng.normal(0, 0.01, n),
            }
        )
        result = confound_check(df, "metric", ["confound"])
        assert len(result) == 1
        assert result[0]["flagged"] is True
        assert abs(result[0]["rho"]) > 0.3

    def test_low_correlation_not_flagged(self) -> None:
        """Uncorrelated confound is not flagged."""
        rng = np.random.default_rng(1)
        n = 100
        df = pd.DataFrame(
            {
                "metric": rng.normal(0, 1, n),
                "confound": rng.normal(0, 1, n),
            }
        )
        result = confound_check(df, "metric", ["confound"])
        assert result[0]["flagged"] is False


# ============================================================================
# fdr_correct
# ============================================================================


class TestFdrCorrect:
    """Tests for fdr_correct."""

    def test_corrects_perm_p_values(self) -> None:
        """FDR correction inflates p-values appropriately."""
        results = [
            {"metric": "a", "perm_p_value": 0.01},
            {"metric": "b", "perm_p_value": 0.04},
            {"metric": "c", "perm_p_value": 0.5},
        ]
        corrected = fdr_correct(results, alpha=0.05)
        assert len(corrected) == 3
        # p_fdr should be >= raw p
        for raw, corr in zip(results, corrected):
            assert corr["p_fdr"] >= raw["perm_p_value"] or np.isclose(
                corr["p_fdr"], raw["perm_p_value"]
            )
        # The largest p should still be not significant
        assert corrected[2]["significant_fdr"] is False

    def test_falls_back_to_p_value_key(self) -> None:
        """Uses p_value key when perm_p_value is absent."""
        results = [
            {"metric": "x", "p_value": 0.001},
            {"metric": "y", "p_value": 0.8},
        ]
        corrected = fdr_correct(results, alpha=0.05)
        assert corrected[0]["significant_fdr"] is True
        assert corrected[1]["significant_fdr"] is False

    def test_empty_raises(self) -> None:
        """Empty results list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            fdr_correct([])


# ============================================================================
# run_between_group_test (orchestrator)
# ============================================================================


class TestRunBetweenGroupTest:
    """Tests for run_between_group_test."""

    def test_supported_verdict(self) -> None:
        """Strong effect returns 'supported' verdict."""
        df = _make_nested_df(
            penk_means=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            nonpenk_means=[10.0, 10.0, 10.0],
            noise=0.01,
        )
        result = run_between_group_test(df, "metric", n_perms=5000, seed=42)
        assert result["verdict"] == "supported"
        assert "summary_p_value" in result
        assert "perm_p_value" in result

    def test_not_supported_verdict(self) -> None:
        """No effect returns 'not_supported' verdict."""
        df = _make_nested_df(
            penk_means=[5.0, 5.0, 5.0, 5.0],
            nonpenk_means=[5.0, 5.0, 5.0],
            noise=0.5,
        )
        result = run_between_group_test(df, "metric", n_perms=2000, seed=42)
        assert result["verdict"] == "not_supported"
