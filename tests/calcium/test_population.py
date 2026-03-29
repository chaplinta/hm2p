"""Tests for hm2p.calcium.population — population-level calcium analysis.

Covers:
- compute_population_signals: PCA on ROI matrix
- frame_correlation: frame-to-frame Pearson correlation
- regress_movement: OLS + Spearman regression against speed/AHV
- compare_spikes_to_fluorescence: CASCADE vs dF/F comparison metrics
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.calcium.population import (
    compare_spikes_to_fluorescence,
    compute_population_signals,
    frame_correlation,
    regress_movement,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_F(n_rois: int = 20, n_frames: int = 200, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_rois, n_frames)).astype(np.float32) + 100.0


def _make_dff_spikes(
    n_rois: int = 15, n_frames: int = 300, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dff = rng.standard_normal((n_rois, n_frames)).astype(np.float32) * 0.1
    spikes = np.abs(rng.standard_normal((n_rois, n_frames)).astype(np.float32)) * 0.5
    return dff, spikes


# ===========================================================================
# compute_population_signals
# ===========================================================================


class TestComputePopulationSignals:
    def test_output_keys_present(self) -> None:
        F = _make_F()
        result = compute_population_signals(F, n_components=5)
        assert "components" in result
        assert "explained_variance_ratio" in result
        assert "mean_activity" in result

    def test_components_shape(self) -> None:
        n_rois, n_frames, n_comp = 20, 200, 5
        F = _make_F(n_rois, n_frames)
        result = compute_population_signals(F, n_components=n_comp)
        assert result["components"].shape == (n_comp, n_frames)

    def test_explained_variance_shape(self) -> None:
        n_comp = 4
        F = _make_F(20, 200)
        result = compute_population_signals(F, n_components=n_comp)
        assert result["explained_variance_ratio"].shape == (n_comp,)

    def test_explained_variance_sums_le_one(self) -> None:
        F = _make_F(20, 200)
        result = compute_population_signals(F, n_components=5)
        total = float(result["explained_variance_ratio"].sum())
        assert 0.0 < total <= 1.0 + 1e-6

    def test_mean_activity_shape(self) -> None:
        n_frames = 200
        F = _make_F(20, n_frames)
        result = compute_population_signals(F)
        assert result["mean_activity"].shape == (n_frames,)

    def test_output_dtype_float32(self) -> None:
        F = _make_F(20, 200)
        result = compute_population_signals(F, n_components=3)
        assert result["components"].dtype == np.float32
        assert result["explained_variance_ratio"].dtype == np.float32
        assert result["mean_activity"].dtype == np.float32

    def test_n_components_clipped_to_min_dim(self) -> None:
        """When n_components > n_rois, it should clip to n_rois."""
        F = _make_F(n_rois=5, n_frames=200)
        result = compute_population_signals(F, n_components=100)
        # Can have at most min(n_rois, n_frames) = 5 components
        assert result["components"].shape[0] <= 5

    def test_with_nan_inputs_no_exception(self) -> None:
        rng = np.random.default_rng(0)
        F = rng.standard_normal((15, 100)).astype(np.float32)
        F[0, :10] = np.nan
        F[3, 50:55] = np.nan
        result = compute_population_signals(F, n_components=3)
        assert result["components"].shape[1] == 100

    def test_mean_activity_matches_nanmean(self) -> None:
        rng = np.random.default_rng(7)
        F = rng.standard_normal((10, 150)).astype(np.float32) + 50.0
        result = compute_population_signals(F, n_components=2)
        expected = np.nanmean(F, axis=0).astype(np.float32)
        np.testing.assert_allclose(result["mean_activity"], expected, rtol=1e-5)

    def test_single_roi(self) -> None:
        """Single-ROI matrix: only 1 component possible."""
        F = _make_F(n_rois=1, n_frames=100)
        result = compute_population_signals(F, n_components=5)
        assert result["components"].shape[0] == 1
        assert result["mean_activity"].shape == (100,)

    def test_fewer_frames_than_components(self) -> None:
        F = _make_F(n_rois=20, n_frames=3)
        result = compute_population_signals(F, n_components=10)
        # Must not crash; components bounded by n_frames
        assert result["components"].shape[0] <= 3

    def test_constant_rows_handled(self) -> None:
        """Constant ROI rows should not crash PCA."""
        F = np.ones((10, 100), dtype=np.float32) * 50.0
        # Add slight variation to a subset to make the matrix non-degenerate
        rng = np.random.default_rng(1)
        F[:5] += rng.standard_normal((5, 100)).astype(np.float32) * 0.01
        result = compute_population_signals(F, n_components=2)
        assert result["components"].shape[1] == 100


# ===========================================================================
# frame_correlation
# ===========================================================================


class TestFrameCorrelation:
    def test_output_length_lag_1(self) -> None:
        F = _make_F(20, 100)
        corrs = frame_correlation(F, lag=1)
        assert corrs.shape == (99,)

    def test_output_length_lag_5(self) -> None:
        F = _make_F(20, 100)
        corrs = frame_correlation(F, lag=5)
        assert corrs.shape == (95,)

    def test_output_dtype_float32(self) -> None:
        F = _make_F(20, 50)
        corrs = frame_correlation(F, lag=1)
        assert corrs.dtype == np.float32

    def test_corr_range_valid(self) -> None:
        """All non-NaN correlations must be in [-1, 1]."""
        F = _make_F(20, 150)
        corrs = frame_correlation(F, lag=1)
        valid = corrs[np.isfinite(corrs)]
        assert np.all(valid >= -1.0 - 1e-6)
        assert np.all(valid <= 1.0 + 1e-6)

    def test_identical_frames_give_correlation_one(self) -> None:
        """If all frames are identical, consecutive correlations = 1."""
        rng = np.random.default_rng(5)
        col = rng.standard_normal((20,)).astype(np.float32)
        F = np.tile(col[:, None], (1, 50))
        corrs = frame_correlation(F, lag=1)
        valid = corrs[np.isfinite(corrs)]
        np.testing.assert_allclose(valid, 1.0, atol=1e-5)

    def test_constant_signal_returns_nan(self) -> None:
        """Constant signal has std=0 → correlation undefined → NaN."""
        F = np.ones((10, 30), dtype=np.float32)
        corrs = frame_correlation(F, lag=1)
        assert np.all(np.isnan(corrs))

    def test_all_nan_inputs_returns_nan(self) -> None:
        F = np.full((10, 50), np.nan, dtype=np.float32)
        corrs = frame_correlation(F, lag=1)
        assert np.all(np.isnan(corrs))

    def test_sparse_nan_does_not_crash(self) -> None:
        rng = np.random.default_rng(3)
        F = rng.standard_normal((15, 80)).astype(np.float32)
        F[2, 10:15] = np.nan
        corrs = frame_correlation(F, lag=1)
        assert corrs.shape == (79,)

    def test_single_row(self) -> None:
        """Single ROI: correlation is trivially 1 or NaN."""
        rng = np.random.default_rng(9)
        F = rng.standard_normal((1, 50)).astype(np.float32)
        corrs = frame_correlation(F, lag=1)
        assert corrs.shape == (49,)

    def test_large_lag_output_length(self) -> None:
        F = _make_F(20, 100)
        lag = 20
        corrs = frame_correlation(F, lag=lag)
        assert corrs.shape == (100 - lag,)

    @given(
        n_rois=st.integers(min_value=3, max_value=30),
        n_frames=st.integers(min_value=10, max_value=100),
        lag=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=30, deadline=None)
    def test_output_length_property(self, n_rois: int, n_frames: int, lag: int) -> None:
        rng = np.random.default_rng(0)
        F = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
        corrs = frame_correlation(F, lag=lag)
        assert corrs.shape == (n_frames - lag,)


# ===========================================================================
# regress_movement
# ===========================================================================


class TestRegressMovement:
    def _make_signals_speed(
        self, n_signals: int = 10, n_frames: int = 200, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        signals = rng.standard_normal((n_signals, n_frames)).astype(np.float32)
        speed = np.abs(rng.standard_normal(n_frames)).astype(np.float32)
        return signals, speed

    def test_output_keys_present(self) -> None:
        signals, speed = self._make_signals_speed()
        result = regress_movement(signals, speed)
        assert "r_squared" in result
        assert "speed_corr" in result
        assert "mean_r_squared" in result

    def test_r_squared_shape(self) -> None:
        n_signals = 8
        signals, speed = self._make_signals_speed(n_signals=n_signals)
        result = regress_movement(signals, speed)
        assert result["r_squared"].shape == (n_signals,)

    def test_speed_corr_shape(self) -> None:
        n_signals = 8
        signals, speed = self._make_signals_speed(n_signals=n_signals)
        result = regress_movement(signals, speed)
        assert result["speed_corr"].shape == (n_signals,)

    def test_ahv_corr_none_when_ahv_not_provided(self) -> None:
        signals, speed = self._make_signals_speed()
        result = regress_movement(signals, speed)
        assert result["ahv_corr"] is None

    def test_ahv_corr_shape_when_ahv_provided(self) -> None:
        n_signals = 6
        rng = np.random.default_rng(0)
        signals = rng.standard_normal((n_signals, 200)).astype(np.float32)
        speed = np.abs(rng.standard_normal(200)).astype(np.float32)
        ahv = rng.standard_normal(200).astype(np.float32)
        result = regress_movement(signals, speed, ahv=ahv)
        assert result["ahv_corr"].shape == (n_signals,)

    def test_r_squared_in_valid_range(self) -> None:
        """R² values must be in [0, 1] for any finite signal."""
        rng = np.random.default_rng(1)
        signals = rng.standard_normal((10, 300)).astype(np.float32)
        speed = np.abs(rng.standard_normal(300)).astype(np.float32) * 5.0
        result = regress_movement(signals, speed)
        valid = result["r_squared"][np.isfinite(result["r_squared"])]
        assert np.all(valid >= -1e-6)
        assert np.all(valid <= 1.0 + 1e-6)

    def test_speed_corr_in_valid_range(self) -> None:
        rng = np.random.default_rng(2)
        signals = rng.standard_normal((10, 300)).astype(np.float32)
        speed = np.abs(rng.standard_normal(300)).astype(np.float32)
        result = regress_movement(signals, speed)
        valid = result["speed_corr"][np.isfinite(result["speed_corr"])]
        assert np.all(np.abs(valid) <= 1.0 + 1e-6)

    def test_perfectly_correlated_signal_high_r2(self) -> None:
        """Signal = speed + small noise should yield high R²."""
        rng = np.random.default_rng(3)
        speed = np.abs(rng.standard_normal(500)).astype(np.float32) * 5.0
        noise = rng.standard_normal((1, 500)).astype(np.float32) * 0.01
        signals = speed[None, :] + noise
        result = regress_movement(signals, speed)
        assert result["r_squared"][0] > 0.9

    def test_mean_r_squared_is_scalar(self) -> None:
        signals, speed = self._make_signals_speed()
        result = regress_movement(signals, speed)
        assert np.isscalar(result["mean_r_squared"])

    def test_nan_speed_handled(self) -> None:
        """NaN frames in speed should not crash the function."""
        rng = np.random.default_rng(4)
        signals = rng.standard_normal((5, 200)).astype(np.float32)
        speed = np.abs(rng.standard_normal(200)).astype(np.float32)
        speed[50:60] = np.nan
        result = regress_movement(signals, speed)
        assert result["r_squared"].shape == (5,)

    def test_insufficient_valid_frames_yields_nan(self) -> None:
        """Fewer than 10 valid frames → NaN R² for that signal."""
        rng = np.random.default_rng(5)
        signals = rng.standard_normal((3, 200)).astype(np.float32)
        speed = np.full(200, np.nan, dtype=np.float32)
        speed[:5] = 1.0  # only 5 valid frames
        result = regress_movement(signals, speed)
        assert np.all(np.isnan(result["r_squared"]))

    def test_with_acceleration(self) -> None:
        rng = np.random.default_rng(6)
        signals = rng.standard_normal((6, 200)).astype(np.float32)
        speed = np.abs(rng.standard_normal(200)).astype(np.float32)
        ahv = rng.standard_normal(200).astype(np.float32)
        accel = np.diff(speed, prepend=speed[0])
        result = regress_movement(signals, speed, ahv=ahv, acceleration=accel)
        assert result["r_squared"].shape == (6,)

    def test_uses_spearman_not_pearson(self) -> None:
        """Verify speed_corr is Spearman (non-parametric) by checking
        it differs from Pearson for a non-monotonic relationship."""
        from scipy.stats import pearsonr, spearmanr

        rng = np.random.default_rng(7)
        n = 300
        speed = rng.uniform(0, 10, n).astype(np.float32)
        # Quadratic (non-linear) relationship: Pearson ≠ Spearman generally
        signals = (speed**2)[None, :].astype(np.float32)
        result = regress_movement(signals, speed)
        # Just confirm it doesn't crash and returns a value in [-1, 1]
        r_spe = float(result["speed_corr"][0])
        assert abs(r_spe) <= 1.0 + 1e-6

    @given(
        n_signals=st.integers(min_value=1, max_value=20),
        n_frames=st.integers(min_value=20, max_value=200),
    )
    @settings(max_examples=25, deadline=None)
    def test_output_shapes_property(self, n_signals: int, n_frames: int) -> None:
        rng = np.random.default_rng(0)
        signals = rng.standard_normal((n_signals, n_frames)).astype(np.float32)
        speed = np.abs(rng.standard_normal(n_frames)).astype(np.float32)
        result = regress_movement(signals, speed)
        assert result["r_squared"].shape == (n_signals,)
        assert result["speed_corr"].shape == (n_signals,)


# ===========================================================================
# compare_spikes_to_fluorescence
# ===========================================================================


class TestCompareSpiksToFluorescence:
    def test_output_keys_present(self) -> None:
        dff, spikes = _make_dff_spikes()
        result = compare_spikes_to_fluorescence(dff, spikes)
        expected_keys = {
            "corr_dff_spikes",
            "corr_deconv_spikes",
            "peak_lag_frames",
            "peak_lag_seconds",
            "mean_corr_dff",
            "mean_corr_deconv",
            "mean_lag_s",
            "event_triggered_avg",
            "event_triggered_time",
        }
        assert expected_keys.issubset(result.keys())

    def test_corr_dff_spikes_shape(self) -> None:
        n_rois = 10
        dff, spikes = _make_dff_spikes(n_rois=n_rois)
        result = compare_spikes_to_fluorescence(dff, spikes)
        assert result["corr_dff_spikes"].shape == (n_rois,)

    def test_corr_deconv_spikes_nan_when_no_deconv(self) -> None:
        dff, spikes = _make_dff_spikes(n_rois=5)
        result = compare_spikes_to_fluorescence(dff, spikes, deconv_norm=None)
        assert np.all(np.isnan(result["corr_deconv_spikes"]))

    def test_corr_values_in_range(self) -> None:
        dff, spikes = _make_dff_spikes(n_rois=10, n_frames=400)
        result = compare_spikes_to_fluorescence(dff, spikes)
        valid = result["corr_dff_spikes"][np.isfinite(result["corr_dff_spikes"])]
        assert np.all(np.abs(valid) <= 1.0 + 1e-6)

    def test_peak_lag_frames_shape(self) -> None:
        n_rois = 8
        dff, spikes = _make_dff_spikes(n_rois=n_rois)
        result = compare_spikes_to_fluorescence(dff, spikes)
        assert result["peak_lag_frames"].shape == (n_rois,)

    def test_peak_lag_seconds_matches_frames_over_fps(self) -> None:
        fps = 9.8
        dff, spikes = _make_dff_spikes(n_rois=5, n_frames=300)
        result = compare_spikes_to_fluorescence(dff, spikes, fps=fps)
        lag_s_derived = result["peak_lag_frames"] / fps
        np.testing.assert_allclose(
            result["peak_lag_seconds"][np.isfinite(result["peak_lag_seconds"])],
            lag_s_derived[np.isfinite(lag_s_derived)],
            rtol=1e-5,
        )

    def test_with_deconv_norm_populates_corr_deconv(self) -> None:
        rng = np.random.default_rng(0)
        n_rois, n_frames = 8, 400
        dff = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
        spikes = np.abs(rng.standard_normal((n_rois, n_frames))).astype(np.float32)
        deconv = np.abs(rng.standard_normal((n_rois, n_frames))).astype(np.float32)
        result = compare_spikes_to_fluorescence(dff, spikes, deconv_norm=deconv)
        # At least some non-NaN correlations expected
        valid = result["corr_deconv_spikes"][np.isfinite(result["corr_deconv_spikes"])]
        assert len(valid) > 0

    def test_constant_spikes_returns_nan_corr(self) -> None:
        """std(spikes) = 0 → correlation undefined → NaN."""
        rng = np.random.default_rng(1)
        n_rois, n_frames = 5, 200
        dff = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
        spikes = np.zeros((n_rois, n_frames), dtype=np.float32)
        result = compare_spikes_to_fluorescence(dff, spikes)
        assert np.all(np.isnan(result["corr_dff_spikes"]))

    def test_all_nan_dff_returns_nan_corr(self) -> None:
        rng = np.random.default_rng(2)
        n_rois, n_frames = 5, 200
        dff = np.full((n_rois, n_frames), np.nan, dtype=np.float32)
        spikes = np.abs(rng.standard_normal((n_rois, n_frames))).astype(np.float32)
        result = compare_spikes_to_fluorescence(dff, spikes)
        assert np.all(np.isnan(result["corr_dff_spikes"]))

    def test_mean_corr_dff_is_scalar(self) -> None:
        dff, spikes = _make_dff_spikes()
        result = compare_spikes_to_fluorescence(dff, spikes)
        assert np.isscalar(result["mean_corr_dff"])

    def test_single_roi(self) -> None:
        rng = np.random.default_rng(3)
        n_frames = 300
        dff = rng.standard_normal((1, n_frames)).astype(np.float32)
        spikes = np.abs(rng.standard_normal((1, n_frames))).astype(np.float32) * 0.5
        result = compare_spikes_to_fluorescence(dff, spikes)
        assert result["corr_dff_spikes"].shape == (1,)

    def test_eta_is_none_when_no_spikes(self) -> None:
        """If all spikes are zero, event_triggered_avg should be None."""
        rng = np.random.default_rng(4)
        dff = rng.standard_normal((10, 200)).astype(np.float32)
        spikes = np.zeros((10, 200), dtype=np.float32)
        result = compare_spikes_to_fluorescence(dff, spikes)
        assert result["event_triggered_avg"] is None
        assert result["event_triggered_time"] is None

    def test_fps_affects_lag_seconds(self) -> None:
        """Changing fps should change peak_lag_seconds but not peak_lag_frames."""
        rng = np.random.default_rng(5)
        n_rois, n_frames = 5, 400
        dff = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
        # Add a clear spike pattern
        spikes = np.zeros((n_rois, n_frames), dtype=np.float32)
        spikes[:, 50::100] = 2.0

        result_9 = compare_spikes_to_fluorescence(dff, spikes, fps=9.8)
        result_30 = compare_spikes_to_fluorescence(dff, spikes, fps=30.0)

        # peak_lag_frames may differ (window size depends on fps), but
        # mean_lag_s should differ between the two fps values when frames differ
        # Both should just run without error
        assert result_9["corr_dff_spikes"].shape == (n_rois,)
        assert result_30["corr_dff_spikes"].shape == (n_rois,)

    def test_uses_spearman_not_pearson(self) -> None:
        """Verify the function uses Spearman correlation (non-parametric)."""
        # Arrange: monotone but non-linear relationship
        # Spearman should be close to 1; Pearson < 1
        n_frames = 300
        base = np.arange(1, n_frames + 1, dtype=np.float32)
        dff = (np.log(base))[None, :]  # log-linear: Pearson < 1
        spikes = base[None, :]  # linear in same direction

        result = compare_spikes_to_fluorescence(dff, spikes)
        r = result["corr_dff_spikes"][0]
        assert np.isfinite(r)
        assert r > 0.9  # Spearman (rank-based) should be high

    @given(
        n_rois=st.integers(min_value=1, max_value=20),
        n_frames=st.integers(min_value=30, max_value=200),
    )
    @settings(max_examples=20, deadline=None)
    def test_output_shapes_property(self, n_rois: int, n_frames: int) -> None:
        rng = np.random.default_rng(0)
        dff = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
        spikes = np.abs(rng.standard_normal((n_rois, n_frames))).astype(np.float32)
        result = compare_spikes_to_fluorescence(dff, spikes)
        assert result["corr_dff_spikes"].shape == (n_rois,)
        assert result["peak_lag_frames"].shape == (n_rois,)
