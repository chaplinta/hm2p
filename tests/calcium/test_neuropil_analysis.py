"""Tests for hm2p.calcium.neuropil_analysis.

Covers:
- compute_mean_neuropil: mean across ROIs with optional cell_mask
- compute_neuropil_ratio: per-ROI Fneu/F ratio
- neuropil_behaviour_correlation: Spearman + Mann-Whitney U against behaviour
- neuropil_soma_correlation: per-ROI Spearman between Fneu and dF/F
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.calcium.neuropil_analysis import (
    compute_mean_neuropil,
    compute_neuropil_ratio,
    neuropil_behaviour_correlation,
    neuropil_soma_correlation,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_Fneu(n_rois: int = 20, n_frames: int = 300, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_rois, n_frames)) + 50.0).astype(np.float32)


def _make_F(n_rois: int = 20, n_frames: int = 300, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    return (rng.standard_normal((n_rois, n_frames)) + 200.0).astype(np.float32)


# ===========================================================================
# compute_mean_neuropil
# ===========================================================================


class TestComputeMeanNeuropil:
    def test_output_shape(self) -> None:
        Fneu = _make_Fneu(20, 300)
        result = compute_mean_neuropil(Fneu)
        assert result.shape == (300,)

    def test_output_dtype_float32(self) -> None:
        Fneu = _make_Fneu(20, 300)
        result = compute_mean_neuropil(Fneu)
        assert result.dtype == np.float32

    def test_no_cell_mask_averages_all_rois(self) -> None:
        Fneu = _make_Fneu(10, 100)
        result = compute_mean_neuropil(Fneu, cell_mask=None)
        expected = np.nanmean(Fneu, axis=0).astype(np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_cell_mask_selects_subset(self) -> None:
        rng = np.random.default_rng(0)
        Fneu = rng.standard_normal((10, 100)).astype(np.float32) + 50.0
        mask = np.array([True, True, False, True, False, True, True, False, False, True])
        result = compute_mean_neuropil(Fneu, cell_mask=mask)
        expected = np.nanmean(Fneu[mask], axis=0).astype(np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_single_roi(self) -> None:
        Fneu = _make_Fneu(n_rois=1, n_frames=50)
        result = compute_mean_neuropil(Fneu)
        assert result.shape == (50,)
        np.testing.assert_allclose(result, Fneu[0].astype(np.float32), rtol=1e-5)

    def test_all_mask_false_returns_nan(self) -> None:
        """All-False mask selects nothing → nanmean of empty → NaN."""
        Fneu = _make_Fneu(5, 50)
        mask = np.zeros(5, dtype=bool)
        result = compute_mean_neuropil(Fneu, cell_mask=mask)
        assert np.all(np.isnan(result))

    def test_with_nan_values(self) -> None:
        Fneu = _make_Fneu(10, 80)
        Fneu[2, 10:15] = np.nan
        result = compute_mean_neuropil(Fneu)
        assert result.shape == (80,)
        assert np.isfinite(result[12])  # other rows contributed

    def test_constant_input(self) -> None:
        Fneu = np.full((5, 60), 42.0, dtype=np.float32)
        result = compute_mean_neuropil(Fneu)
        np.testing.assert_allclose(result, 42.0, atol=1e-5)

    @given(
        n_rois=st.integers(min_value=1, max_value=30),
        n_frames=st.integers(min_value=5, max_value=100),
    )
    @settings(max_examples=25, deadline=None)
    def test_output_shape_property(self, n_rois: int, n_frames: int) -> None:
        rng = np.random.default_rng(0)
        Fneu = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
        result = compute_mean_neuropil(Fneu)
        assert result.shape == (n_frames,)


# ===========================================================================
# compute_neuropil_ratio
# ===========================================================================


class TestComputeNeuropilRatio:
    def test_output_shape(self) -> None:
        F = _make_F(20, 200)
        Fneu = _make_Fneu(20, 200)
        result = compute_neuropil_ratio(F, Fneu)
        assert result.shape == (20,)

    def test_output_dtype_float32(self) -> None:
        F = _make_F(10, 100)
        Fneu = _make_Fneu(10, 100)
        result = compute_neuropil_ratio(F, Fneu)
        assert result.dtype == np.float32

    def test_equal_arrays_gives_ratio_one(self) -> None:
        """When F == Fneu, ratio should be 1 everywhere."""
        rng = np.random.default_rng(0)
        arr = np.abs(rng.standard_normal((8, 100)).astype(np.float32)) + 10.0
        result = compute_neuropil_ratio(arr, arr)
        np.testing.assert_allclose(result, 1.0, rtol=1e-5)

    def test_fneu_double_f_gives_ratio_two(self) -> None:
        F = np.full((5, 50), 100.0, dtype=np.float32)
        Fneu = np.full((5, 50), 200.0, dtype=np.float32)
        result = compute_neuropil_ratio(F, Fneu)
        np.testing.assert_allclose(result, 2.0, rtol=1e-5)

    def test_zero_mean_f_gives_nan(self) -> None:
        """mean(F) = 0 → ratio is NaN (not inf)."""
        F = np.zeros((4, 50), dtype=np.float32)
        Fneu = np.ones((4, 50), dtype=np.float32) * 10.0
        result = compute_neuropil_ratio(F, Fneu)
        assert np.all(np.isnan(result))

    def test_ratio_positive_for_positive_inputs(self) -> None:
        F = _make_F(10, 100)
        Fneu = _make_Fneu(10, 100)
        result = compute_neuropil_ratio(F, Fneu)
        valid = result[np.isfinite(result)]
        assert np.all(valid > 0)

    def test_with_nan_in_F(self) -> None:
        F = _make_F(8, 100)
        Fneu = _make_Fneu(8, 100)
        F[2, 20:30] = np.nan
        result = compute_neuropil_ratio(F, Fneu)
        assert result.shape == (8,)

    def test_single_roi(self) -> None:
        F = np.full((1, 50), 150.0, dtype=np.float32)
        Fneu = np.full((1, 50), 30.0, dtype=np.float32)
        result = compute_neuropil_ratio(F, Fneu)
        np.testing.assert_allclose(result, 0.2, rtol=1e-5)

    @given(
        n_rois=st.integers(min_value=1, max_value=30),
        n_frames=st.integers(min_value=5, max_value=100),
    )
    @settings(max_examples=25, deadline=None)
    def test_output_shape_property(self, n_rois: int, n_frames: int) -> None:
        rng = np.random.default_rng(0)
        F = np.abs(rng.standard_normal((n_rois, n_frames)).astype(np.float32)) + 1.0
        Fneu = np.abs(rng.standard_normal((n_rois, n_frames)).astype(np.float32)) + 1.0
        result = compute_neuropil_ratio(F, Fneu)
        assert result.shape == (n_rois,)


# ===========================================================================
# neuropil_behaviour_correlation
# ===========================================================================


class TestNeuropilBehaviourCorrelation:
    def _make_inputs(self, n: int = 300, seed: int = 42):
        rng = np.random.default_rng(seed)
        mean_fneu = (rng.standard_normal(n) * 5.0 + 100.0).astype(np.float32)
        speed = np.abs(rng.standard_normal(n)).astype(np.float32) * 10.0
        ahv = rng.standard_normal(n).astype(np.float32) * 30.0
        light_on = (rng.uniform(size=n) > 0.5).astype(bool)
        active_mask = np.ones(n, dtype=bool)
        return mean_fneu, speed, ahv, light_on, active_mask

    def test_speed_corr_present(self) -> None:
        mean_fneu, speed, _, _, _ = self._make_inputs()
        result = neuropil_behaviour_correlation(mean_fneu, speed)
        assert "speed_corr" in result
        assert "speed_p" in result

    def test_speed_corr_in_valid_range(self) -> None:
        mean_fneu, speed, _, _, _ = self._make_inputs()
        result = neuropil_behaviour_correlation(mean_fneu, speed)
        assert abs(result["speed_corr"]) <= 1.0 + 1e-6

    def test_ahv_corr_absent_when_not_provided(self) -> None:
        mean_fneu, speed, _, _, _ = self._make_inputs()
        result = neuropil_behaviour_correlation(mean_fneu, speed)
        assert "ahv_corr" not in result

    def test_ahv_corr_present_when_provided(self) -> None:
        mean_fneu, speed, ahv, _, _ = self._make_inputs()
        result = neuropil_behaviour_correlation(mean_fneu, speed, ahv=ahv)
        assert "ahv_corr" in result
        assert abs(result["ahv_corr"]) <= 1.0 + 1e-6

    def test_light_dark_comparison_keys(self) -> None:
        mean_fneu, speed, _, light_on, active_mask = self._make_inputs()
        result = neuropil_behaviour_correlation(
            mean_fneu, speed, light_on=light_on, active_mask=active_mask
        )
        assert "mean_fneu_light" in result
        assert "mean_fneu_dark" in result
        assert "light_dark_p" in result
        assert "light_mod_index" in result

    def test_uses_mann_whitney_not_ttest(self) -> None:
        """Regression test: light/dark comparison must use Mann-Whitney U
        (non-parametric), never t-test."""
        # We verify by checking the p-value is consistent with MWU output
        from scipy.stats import mannwhitneyu

        rng = np.random.default_rng(0)
        n = 300
        mean_fneu = rng.standard_normal(n).astype(np.float32) + 100.0
        speed = np.abs(rng.standard_normal(n)).astype(np.float32)
        light_on = np.zeros(n, dtype=bool)
        light_on[:150] = True
        active_mask = np.ones(n, dtype=bool)

        result = neuropil_behaviour_correlation(
            mean_fneu, speed, light_on=light_on, active_mask=active_mask
        )
        # Recompute expected p manually
        lt = light_on & active_mask
        dk = ~light_on & active_mask
        _, expected_p = mannwhitneyu(mean_fneu[lt], mean_fneu[dk], alternative="two-sided")
        assert abs(result["light_dark_p"] - float(expected_p)) < 1e-10

    def test_movement_mod_keys_present(self) -> None:
        mean_fneu, speed, _, _, _ = self._make_inputs()
        result = neuropil_behaviour_correlation(mean_fneu, speed)
        assert "mean_fneu_moving" in result
        assert "mean_fneu_stationary" in result
        assert "movement_mod_index" in result

    def test_light_mod_index_range(self) -> None:
        mean_fneu, speed, _, light_on, active_mask = self._make_inputs()
        result = neuropil_behaviour_correlation(
            mean_fneu, speed, light_on=light_on, active_mask=active_mask
        )
        idx = result.get("light_mod_index", 0.0)
        assert -1.0 <= idx <= 1.0 + 1e-6

    def test_active_mask_filters_frames(self) -> None:
        """With active_mask=all False and fewer than 10 valid frames,
        speed_corr key should be absent."""
        rng = np.random.default_rng(1)
        n = 300
        mean_fneu = rng.standard_normal(n).astype(np.float32)
        speed = np.abs(rng.standard_normal(n)).astype(np.float32)
        active_mask = np.zeros(n, dtype=bool)
        result = neuropil_behaviour_correlation(mean_fneu, speed, active_mask=active_mask)
        assert "speed_corr" not in result

    def test_shorter_speed_than_fneu(self) -> None:
        """Function should handle mismatched lengths by taking min length."""
        rng = np.random.default_rng(2)
        mean_fneu = rng.standard_normal(300).astype(np.float32)
        speed = np.abs(rng.standard_normal(250)).astype(np.float32)
        result = neuropil_behaviour_correlation(mean_fneu, speed)
        assert "speed_corr" in result

    def test_all_nan_fneu_no_speed_corr_key(self) -> None:
        n = 200
        mean_fneu = np.full(n, np.nan, dtype=np.float32)
        speed = np.ones(n, dtype=np.float32) * 5.0
        result = neuropil_behaviour_correlation(mean_fneu, speed)
        assert "speed_corr" not in result

    @given(n=st.integers(min_value=30, max_value=500))
    @settings(max_examples=20, deadline=None)
    def test_speed_corr_always_valid_range(self, n: int) -> None:
        rng = np.random.default_rng(0)
        mean_fneu = rng.standard_normal(n).astype(np.float32) + 100.0
        speed = np.abs(rng.standard_normal(n)).astype(np.float32)
        result = neuropil_behaviour_correlation(mean_fneu, speed)
        if "speed_corr" in result:
            assert abs(result["speed_corr"]) <= 1.0 + 1e-6


# ===========================================================================
# neuropil_soma_correlation
# ===========================================================================


class TestNeuropilSomaCorrelation:
    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        n_rois, n_frames = 15, 300
        dff = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
        Fneu = rng.standard_normal((n_rois, n_frames)).astype(np.float32) + 50.0
        result = neuropil_soma_correlation(dff, Fneu)
        assert result.shape == (n_rois,)

    def test_output_dtype_float32(self) -> None:
        rng = np.random.default_rng(1)
        dff = rng.standard_normal((8, 200)).astype(np.float32)
        Fneu = rng.standard_normal((8, 200)).astype(np.float32)
        result = neuropil_soma_correlation(dff, Fneu)
        assert result.dtype == np.float32

    def test_values_in_valid_range(self) -> None:
        rng = np.random.default_rng(2)
        dff = rng.standard_normal((10, 300)).astype(np.float32)
        Fneu = rng.standard_normal((10, 300)).astype(np.float32)
        result = neuropil_soma_correlation(dff, Fneu)
        valid = result[np.isfinite(result)]
        assert np.all(np.abs(valid) <= 1.0 + 1e-6)

    def test_identical_arrays_give_corr_one(self) -> None:
        rng = np.random.default_rng(3)
        arr = rng.standard_normal((6, 200)).astype(np.float32)
        result = neuropil_soma_correlation(arr, arr)
        valid = result[np.isfinite(result)]
        np.testing.assert_allclose(valid, 1.0, atol=1e-4)

    def test_constant_signal_returns_nan(self) -> None:
        """Zero std → Spearman undefined → NaN."""
        dff = np.ones((5, 100), dtype=np.float32)
        Fneu = np.ones((5, 100), dtype=np.float32) * 2.0
        result = neuropil_soma_correlation(dff, Fneu)
        assert np.all(np.isnan(result))

    def test_nan_rows_return_nan(self) -> None:
        rng = np.random.default_rng(4)
        dff = rng.standard_normal((5, 100)).astype(np.float32)
        Fneu = rng.standard_normal((5, 100)).astype(np.float32)
        dff[2, :] = np.nan
        result = neuropil_soma_correlation(dff, Fneu)
        assert np.isnan(result[2])
        # Other ROIs should still have valid values
        assert np.isfinite(result[0])

    def test_anti_correlated_gives_negative_value(self) -> None:
        """Negatively correlated signals should give r < 0."""
        rng = np.random.default_rng(5)
        n_frames = 400
        base = rng.standard_normal((1, n_frames)).astype(np.float32)
        dff = base.copy()
        Fneu = -base + rng.standard_normal((1, n_frames)).astype(np.float32) * 0.01
        result = neuropil_soma_correlation(dff, Fneu)
        assert result[0] < 0

    def test_single_roi(self) -> None:
        rng = np.random.default_rng(6)
        dff = rng.standard_normal((1, 200)).astype(np.float32)
        Fneu = rng.standard_normal((1, 200)).astype(np.float32) + 50.0
        result = neuropil_soma_correlation(dff, Fneu)
        assert result.shape == (1,)

    def test_uses_spearman_not_pearson(self) -> None:
        """Verify non-parametric Spearman is used.

        For a monotone non-linear relationship, Spearman ≈ 1 while Pearson < 1.
        """
        from scipy.stats import pearsonr

        n_frames = 500
        x = np.arange(1, n_frames + 1, dtype=np.float32)
        # Exponential relationship: Spearman = 1, Pearson < 1
        dff = (np.exp(x / n_frames))[None, :]
        Fneu = x[None, :]
        result = neuropil_soma_correlation(dff, Fneu)
        r = result[0]
        assert np.isfinite(r)
        assert r > 0.99  # Spearman should be very close to 1

        # Also confirm Pearson would differ
        pr, _ = pearsonr(dff[0], Fneu[0])
        assert pr < 0.999  # Pearson is less than Spearman for non-linear case

    def test_insufficient_valid_frames_returns_nan(self) -> None:
        rng = np.random.default_rng(7)
        n_rois, n_frames = 3, 200
        dff = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
        Fneu = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
        # Make most frames NaN — only 5 valid per ROI
        dff[:, 10:] = np.nan
        result = neuropil_soma_correlation(dff, Fneu)
        assert np.all(np.isnan(result))

    @given(
        n_rois=st.integers(min_value=1, max_value=30),
        n_frames=st.integers(min_value=20, max_value=200),
    )
    @settings(max_examples=25, deadline=None)
    def test_output_shape_property(self, n_rois: int, n_frames: int) -> None:
        rng = np.random.default_rng(0)
        dff = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
        Fneu = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
        result = neuropil_soma_correlation(dff, Fneu)
        assert result.shape == (n_rois,)
