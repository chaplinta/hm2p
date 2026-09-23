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
        for raw, corr in zip(results, corrected, strict=False):
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


# ============================================================================
# Additions for the Penk+ vs Penk⁻CamKII+ programme
# ============================================================================


from hm2p.analysis.mixed_stats import (  # noqa: E402
    _bh_fdr,
    equipment_matched_subset,
    lmm_celltype_test,
    loao_between_group,
)


class TestBhFdr:
    """Tests for the Benjamini-Hochberg helper and its numpy fallback."""

    def test_fallback_matches_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without statsmodels the numpy path gives the textbook BH values."""
        import builtins

        real_import = builtins.__import__

        def _no_statsmodels(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name.startswith("statsmodels"):
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_statsmodels)
        p = np.array([0.01, 0.04, 0.03, 0.20])
        corrected, reject = _bh_fdr(p, alpha=0.05)
        # sorted p 0.01, 0.03, 0.04, 0.20 -> 0.04, 0.06, 0.0533, 0.20
        # -> monotone from the top: 0.04, 0.0533, 0.0533, 0.20
        assert np.allclose(corrected, [0.04, 0.0533333, 0.0533333, 0.20], atol=1e-6)
        assert reject.tolist() == [True, False, False, False]

    def test_statsmodels_path_when_available(self) -> None:
        """When statsmodels is installed both paths agree."""
        pytest.importorskip("statsmodels")
        p = np.array([0.001, 0.5, 0.02])
        corrected, reject = _bh_fdr(p)
        assert corrected.shape == (3,)
        assert reject[0] and not reject[1]

    def test_all_ones(self) -> None:
        corrected, reject = _bh_fdr(np.array([1.0, 1.0]))
        assert np.all(corrected == 1.0)
        assert not reject.any()

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=20))
    @settings(max_examples=30, deadline=None)
    def test_corrected_within_unit_interval(self, p: list[float]) -> None:
        corrected, _ = _bh_fdr(np.array(p))
        assert np.all(corrected >= 0.0) and np.all(corrected <= 1.0)
        assert np.all(corrected >= np.array(p) - 1e-12)


class TestLmmCelltypeTest:
    """Tests for the supplementary LMM."""

    def test_unavailable_returns_nan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__

        def _no_statsmodels(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name.startswith("statsmodels"):
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_statsmodels)
        df = _make_nested_df([1.0, 1.1, 0.9], [2.0, 2.1, 1.9])
        res = lmm_celltype_test(df, "metric")
        assert res["available"] is False
        assert np.isnan(res["p_value"])
        assert res["n_animals"] == 6

    def test_too_few_animals(self) -> None:
        df = _make_nested_df([1.0], [2.0])
        res = lmm_celltype_test(df, "metric")
        assert res["available"] is False
        assert res["n_animals"] == 2

    def test_recovers_effect(self) -> None:
        pytest.importorskip("statsmodels")
        df = _make_nested_df([1.0, 1.1, 0.9, 1.05], [3.0, 3.1, 2.9], noise=0.05)
        res = lmm_celltype_test(df, "metric")
        assert res["available"] is True
        # levels sorted: ["nonpenk", "penk"]; coef = penk - nonpenk < 0
        assert res["coef"] < 0
        assert res["p_value"] < 0.05
        assert 0.0 <= res["icc"] <= 1.0

    def test_missing_column_raises(self) -> None:
        df = _make_nested_df([1.0, 2.0], [1.0, 2.0])
        with pytest.raises(ValueError, match="Missing columns"):
            lmm_celltype_test(df, "nope")


class TestLoaoBetweenGroup:
    """Tests for the leave-one-animal-out direction check."""

    def test_stable_direction(self) -> None:
        df = _make_nested_df(
            penk_means=[1.0, 1.0, 1.0, 1.0, 1.0],
            nonpenk_means=[5.0, 5.0, 5.0],
            noise=0.01,
        )
        res = loao_between_group(df, "metric", n_perms=200)
        assert res["drop_group"] == "nonpenk"
        assert res["n_drops"] == 3
        assert res["direction_stable"] is True
        # sorted levels are ["nonpenk", "penk"]; direction = penk - nonpenk < 0
        assert res["full_direction"] == -1
        assert all(d["dropped"].startswith("nonpenk") for d in res["drops"])

    def test_unstable_when_one_animal_drives_effect(self) -> None:
        df = _make_nested_df(
            penk_means=[1.0, 1.0, 1.0, 1.0],
            nonpenk_means=[1.0, 1.0, 20.0],
            noise=0.01,
        )
        res = loao_between_group(df, "metric", n_perms=100)
        assert res["direction_stable"] is False
        assert len(res["drops"]) == 3

    def test_explicit_drop_group(self) -> None:
        df = _make_nested_df([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], noise=0.01)
        res = loao_between_group(df, "metric", drop_group="penk", n_perms=50)
        assert res["drop_group"] == "penk"
        assert res["n_drops"] == 3

    def test_bad_group_raises(self) -> None:
        df = _make_nested_df([1.0, 1.0], [2.0, 2.0])
        with pytest.raises(ValueError, match="not in"):
            loao_between_group(df, "metric", drop_group="other", n_perms=10)

    def test_requires_two_groups(self) -> None:
        df = _make_nested_df([1.0, 1.0], [2.0, 2.0])
        df = df[df["celltype"] == "penk"]
        with pytest.raises(ValueError, match="exactly 2 groups"):
            loao_between_group(df, "metric", n_perms=10)


class TestEquipmentMatchedSubset:
    """Tests for the equipment-matched subset helper."""

    @staticmethod
    def _df() -> pd.DataFrame:
        rows = [
            ("a1", "penk", "SFB", "f4mm"),
            ("a2", "penk", "TFB", "f6mm"),
            ("a3", "penk", "TFB", "f4mm"),
            ("b1", "nonpenk", "TFB", "f6mm"),
            ("b2", "nonpenk", "SFB", "f4mm"),
        ]
        return pd.DataFrame(
            [
                {"animal_id": a, "celltype": c, "fibre": f, "lens": lens, "metric": 1.0}
                for a, c, f, lens in rows
            ]
        )

    def test_keeps_shared_configs_only(self) -> None:
        out = equipment_matched_subset(self._df())
        assert set(out["equipment"]) == {"SFB/f4mm", "TFB/f6mm"}
        assert "a3" not in set(out["animal_id"])
        assert len(out) == 4

    def test_empty_when_nothing_shared(self) -> None:
        df = self._df()
        df.loc[df["celltype"] == "nonpenk", "fibre"] = "XXX"
        out = equipment_matched_subset(df)
        assert out.empty

    def test_single_group_returns_empty(self) -> None:
        df = self._df()
        df = df[df["celltype"] == "penk"]
        assert equipment_matched_subset(df).empty

    def test_custom_columns(self) -> None:
        df = self._df()
        out = equipment_matched_subset(df, equipment_cols=["lens"])
        assert set(out["equipment"]) == {"f4mm", "f6mm"}
        assert len(out) == 5

    def test_missing_columns_raise(self) -> None:
        with pytest.raises(ValueError, match="Missing columns"):
            equipment_matched_subset(self._df().drop(columns=["lens"]))


class TestVarianceRatioTest:
    """Tests for the dispersion wrapper around heterogeneity.variance_ratio_permutation."""

    def test_wider_group_detected(self) -> None:
        from hm2p.analysis.mixed_stats import variance_ratio_test

        rng = np.random.default_rng(0)
        rows = []
        for i in range(6):
            sd = 3.0 if i < 3 else 0.2
            ct = "nonpenk" if i < 3 else "penk"
            for _ in range(8):
                rows.append({"animal_id": f"a{i}", "celltype": ct, "m": rng.normal(0, sd)})
        df = pd.DataFrame(rows)
        res = variance_ratio_test(df, "m", n_perms=200, seed=1)
        assert res["metric"] == "m"
        assert res["observed"] > 1.0
        assert 0.0 < res["p_value"] <= 1.0

    def test_missing_column_raises(self) -> None:
        from hm2p.analysis.mixed_stats import variance_ratio_test

        with pytest.raises(ValueError, match="Missing columns"):
            variance_ratio_test(_make_nested_df([1.0, 2.0], [1.0, 2.0]), "nope")
