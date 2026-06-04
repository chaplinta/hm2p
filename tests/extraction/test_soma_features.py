"""Tests for hm2p.extraction.soma_features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hm2p.extraction.soma_features import (
    FEATURE_COLUMNS,
    _autocorr_halfwidth_s,
    _derivative_skew,
    _event_rate,
    _fneu_corr,
    _kurtosis,
    _peak_to_noise,
    _power_slope,
    _quick_dff,
    _signal_to_background,
    _trace_sparsity,
    extract_soma_features,
)

# ---------------------------------------------------------------------------
# _quick_dff
# ---------------------------------------------------------------------------


class TestQuickDff:
    def test_shape_and_dtype(self) -> None:
        rng = np.random.default_rng(0)
        F = rng.uniform(100, 500, size=(5, 200)).astype(np.float32)
        out = _quick_dff(F)
        assert out.shape == F.shape
        assert out.dtype == np.float32

    def test_constant_trace_yields_finite(self) -> None:
        F = np.full((3, 100), 200.0, dtype=np.float32)
        out = _quick_dff(F)
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 0.0, atol=1e-3)

    def test_step_increase_yields_positive_dff(self) -> None:
        F = np.full((1, 200), 100.0, dtype=np.float32)
        F[0, 100:] = 200.0
        out = _quick_dff(F)
        assert out[0, -1] > 0.5


# ---------------------------------------------------------------------------
# _peak_to_noise
# ---------------------------------------------------------------------------


class TestPeakToNoise:
    def test_constant_returns_nan(self) -> None:
        assert np.isnan(_peak_to_noise(np.zeros(100)))

    def test_empty_returns_nan(self) -> None:
        assert np.isnan(_peak_to_noise(np.array([])))

    def test_signal_with_peak_returns_positive(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, size=200)
        x[50] = 15.0
        x[100] = 20.0
        x[150] = 18.0
        ratio = _peak_to_noise(x)
        assert ratio > 5.0

    def test_uses_99th_percentile_not_max(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, size=1000)
        base_ratio = _peak_to_noise(x.copy())
        x_outlier = x.copy()
        x_outlier[500] = 500.0
        outlier_ratio = _peak_to_noise(x_outlier)
        assert abs(outlier_ratio - base_ratio) / max(base_ratio, 1e-6) < 0.5


# ---------------------------------------------------------------------------
# _autocorr_halfwidth_s
# ---------------------------------------------------------------------------


class TestAutocorrHalfwidth:
    def test_constant_returns_nan(self) -> None:
        assert np.isnan(_autocorr_halfwidth_s(np.zeros(100), fps=10.0))

    def test_short_returns_nan(self) -> None:
        assert np.isnan(_autocorr_halfwidth_s(np.array([1.0, 2.0]), fps=10.0))

    def test_white_noise_short_halfwidth(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, size=2000)
        hw = _autocorr_halfwidth_s(x, fps=10.0)
        assert np.isfinite(hw)
        assert hw < 1.0

    def test_smooth_signal_has_longer_halfwidth(self) -> None:
        from scipy.ndimage import gaussian_filter1d
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, size=2000)
        x_smooth = gaussian_filter1d(x, sigma=20.0)
        hw_white = _autocorr_halfwidth_s(x, fps=10.0)
        hw_smooth = _autocorr_halfwidth_s(x_smooth, fps=10.0)
        assert hw_smooth > hw_white


# ---------------------------------------------------------------------------
# _fneu_corr
# ---------------------------------------------------------------------------


class TestFneuCorr:
    def test_short_returns_nan(self) -> None:
        assert np.isnan(_fneu_corr(np.zeros(5), np.zeros(5)))

    def test_constant_returns_nan(self) -> None:
        assert np.isnan(_fneu_corr(np.zeros(50), np.linspace(0, 1, 50)))

    def test_perfect_correlation(self) -> None:
        rng = np.random.default_rng(4)
        x = rng.normal(0, 1, size=200)
        assert _fneu_corr(x, x.copy()) == pytest.approx(1.0)

    def test_anti_correlation(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.normal(0, 1, size=200)
        assert _fneu_corr(x, -x) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# _kurtosis
# ---------------------------------------------------------------------------


class TestKurtosis:
    def test_constant_returns_nan(self) -> None:
        assert np.isnan(_kurtosis(np.zeros(100)))

    def test_short_returns_nan(self) -> None:
        assert np.isnan(_kurtosis(np.array([1.0, 2.0, 3.0])))

    def test_normal_near_zero(self) -> None:
        rng = np.random.default_rng(20)
        x = rng.normal(0, 1, size=10000)
        assert abs(_kurtosis(x)) < 0.3

    def test_leptokurtic_positive(self) -> None:
        rng = np.random.default_rng(21)
        x = rng.normal(0, 1, size=2000)
        for idx in range(0, 2000, 100):
            x[idx] = 15.0
        assert _kurtosis(x) > 5.0


# ---------------------------------------------------------------------------
# New activity features
# ---------------------------------------------------------------------------


class TestSignalToBackground:
    def test_basic(self) -> None:
        F = np.full(100, 200.0)
        Fneu = np.full(100, 100.0)
        assert _signal_to_background(F, Fneu) == pytest.approx(2.0)

    def test_zero_fneu_returns_nan(self) -> None:
        assert np.isnan(_signal_to_background(np.ones(10), np.zeros(10)))

    def test_empty_returns_nan(self) -> None:
        assert np.isnan(_signal_to_background(np.array([]), np.array([])))


class TestEventRate:
    def test_constant_returns_nan(self) -> None:
        assert np.isnan(_event_rate(np.zeros(100), fps=10.0))

    def test_signal_with_transients(self) -> None:
        rng = np.random.default_rng(40)
        x = rng.normal(0, 0.1, size=6000)  # 10 min at 10 Hz
        # Add 20 transients
        for i in range(20):
            x[i * 300] = 5.0
        rate = _event_rate(x, fps=10.0)
        assert rate > 1.0  # should detect most of the 20 transients in 10 min

    def test_short_returns_nan(self) -> None:
        assert np.isnan(_event_rate(np.array([1.0, 2.0]), fps=10.0))


class TestDerivativeSkew:
    def test_constant_returns_nan(self) -> None:
        assert np.isnan(_derivative_skew(np.zeros(100)))

    def test_short_returns_nan(self) -> None:
        assert np.isnan(_derivative_skew(np.array([1.0, 2.0])))

    def test_sawtooth_positive_skew(self) -> None:
        """Fast rise + slow decay should give positive derivative skew."""
        x = np.zeros(1000)
        for i in range(10):
            start = i * 100
            x[start] = 5.0  # instant rise
            for j in range(1, 50):  # slow decay
                x[start + j] = 5.0 * np.exp(-j / 10.0)
        assert _derivative_skew(x) > 0


class TestTraceSparsity:
    def test_constant_returns_nan(self) -> None:
        assert np.isnan(_trace_sparsity(np.zeros(100)))

    def test_sparse_signal(self) -> None:
        rng = np.random.default_rng(50)
        x = rng.normal(0, 1, size=1000)
        x[::100] = 20.0  # 1% of frames are active
        sp = _trace_sparsity(x)
        assert 0.0 < sp < 0.1

    def test_dense_signal_higher(self) -> None:
        rng = np.random.default_rng(51)
        sparse = rng.normal(0, 1, size=1000)
        sparse[0] = 20.0
        dense = rng.normal(0, 1, size=1000)
        dense[::5] = 20.0
        assert _trace_sparsity(dense) > _trace_sparsity(sparse)


class TestPowerSlope:
    def test_constant_returns_nan(self) -> None:
        assert np.isnan(_power_slope(np.zeros(100), fps=10.0))

    def test_short_returns_nan(self) -> None:
        assert np.isnan(_power_slope(np.array([1.0, 2.0]), fps=10.0))

    def test_white_noise_near_flat(self) -> None:
        rng = np.random.default_rng(60)
        x = rng.normal(0, 1, size=2000)
        slope = _power_slope(x, fps=10.0)
        assert abs(slope) < 1.0  # white noise has ~flat PSD

    def test_smooth_signal_steeper(self) -> None:
        from scipy.ndimage import gaussian_filter1d
        rng = np.random.default_rng(61)
        white = rng.normal(0, 1, size=2000)
        smooth = gaussian_filter1d(white, sigma=30.0)
        assert _power_slope(smooth, fps=10.0) < _power_slope(white, fps=10.0)


# ---------------------------------------------------------------------------
# extract_soma_features
# ---------------------------------------------------------------------------


def _make_stat(n: int, rng: np.random.Generator) -> list[dict]:
    """Build synthetic stat dicts with realistic value ranges."""
    results = []
    for _ in range(n):
        npix = int(rng.integers(50, 500))
        xpix = rng.integers(0, 200, size=npix).astype(np.int64)
        ypix = rng.integers(0, 200, size=npix).astype(np.int64)
        results.append({
            "radius": float(rng.uniform(3.0, 10.0)),
            "compact": float(rng.uniform(0.95, 2.0)),
            "aspect_ratio": float(rng.uniform(1.0, 3.0)),
            "npix": npix,
            "npix_norm": float(rng.uniform(0.5, 3.0)),
            "skew": float(rng.uniform(-1.0, 5.0)),
            "std": float(rng.uniform(0.1, 5.0)),
            "solidity": float(rng.uniform(0.5, 1.2)),
            "npix_soma": int(rng.integers(30, 400)),
            "npix_norm_no_crop": float(rng.uniform(0.4, 3.0)),
            "overlap": rng.choice([True, False], size=npix),
            "xpix": xpix,
            "ypix": ypix,
            "lam": rng.uniform(0.1, 5.0, size=npix).astype(np.float32),
        })
    return results


class TestExtractSomaFeatures:
    def test_columns_match_canonical_order(self) -> None:
        rng = np.random.default_rng(6)
        stat = _make_stat(4, rng)
        F = rng.uniform(100, 500, size=(4, 200)).astype(np.float32)
        Fneu = rng.uniform(50, 200, size=(4, 200)).astype(np.float32)
        df = extract_soma_features(stat, F, Fneu, fps=10.0)
        assert list(df.columns) == list(FEATURE_COLUMNS)
        assert len(df) == 4

    def test_shape_matches_input(self) -> None:
        rng = np.random.default_rng(7)
        n = 12
        stat = _make_stat(n, rng)
        F = rng.uniform(100, 500, size=(n, 300)).astype(np.float32)
        Fneu = rng.uniform(50, 200, size=(n, 300)).astype(np.float32)
        df = extract_soma_features(stat, F, Fneu, fps=10.0)
        assert df.shape == (n, len(FEATURE_COLUMNS))

    def test_single_roi(self) -> None:
        rng = np.random.default_rng(8)
        stat = _make_stat(1, rng)
        F = rng.uniform(100, 500, size=(1, 200)).astype(np.float32)
        Fneu = rng.uniform(50, 200, size=(1, 200)).astype(np.float32)
        df = extract_soma_features(stat, F, Fneu, fps=10.0)
        assert len(df) == 1

    def test_zero_rois_returns_empty(self) -> None:
        df = extract_soma_features([], np.zeros((0, 100)), np.zeros((0, 100)), fps=10.0)
        assert df.shape == (0, len(FEATURE_COLUMNS))

    def test_constant_traces_yield_nan_activity(self) -> None:
        stat = _make_stat(2, np.random.default_rng(10))
        F = np.full((2, 200), 100.0, dtype=np.float32)
        Fneu = np.full((2, 200), 80.0, dtype=np.float32)
        df = extract_soma_features(stat, F, Fneu, fps=10.0)
        assert df["peak_to_noise_dff"].isna().all()
        assert df["kurtosis"].isna().all()

    def test_missing_stat_keys_use_defaults(self) -> None:
        stat = [{}, {"radius": 7.0}]
        F = np.random.default_rng(11).uniform(100, 500, size=(2, 200)).astype(np.float32)
        Fneu = np.random.default_rng(12).uniform(50, 200, size=(2, 200)).astype(np.float32)
        df = extract_soma_features(stat, F, Fneu, fps=10.0)
        assert df.iloc[0]["radius"] == pytest.approx(5.0)
        assert df.iloc[0]["compact"] == pytest.approx(1.0)
        assert df.iloc[1]["radius"] == pytest.approx(7.0)

    def test_new_features_present(self) -> None:
        rng = np.random.default_rng(70)
        stat = _make_stat(5, rng)
        F = rng.uniform(100, 500, size=(5, 500)).astype(np.float32)
        Fneu = rng.uniform(50, 200, size=(5, 500)).astype(np.float32)
        df = extract_soma_features(stat, F, Fneu, fps=10.0)
        for col in ["eccentricity", "lam_cv", "signal_to_background",
                     "event_rate", "derivative_skew", "trace_sparsity",
                     "power_slope", "max_pairwise_corr"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_neucoeff_affects_activity_features(self) -> None:
        rng = np.random.default_rng(35)
        stat = _make_stat(3, rng)
        F = rng.uniform(100, 500, size=(3, 500)).astype(np.float32)
        Fneu = rng.uniform(50, 200, size=(3, 500)).astype(np.float32)
        df_07 = extract_soma_features(stat, F, Fneu, fps=10.0, neucoeff=0.7)
        df_00 = extract_soma_features(stat, F, Fneu, fps=10.0, neucoeff=0.0)
        assert df_07["compact"].equals(df_00["compact"])
        assert not np.allclose(
            df_07["peak_to_noise_dff"].values,
            df_00["peak_to_noise_dff"].values,
            equal_nan=True,
        )

    def test_shape_mismatch_raises(self) -> None:
        rng = np.random.default_rng(13)
        F = rng.uniform(100, 500, size=(3, 200)).astype(np.float32)
        Fneu = rng.uniform(50, 200, size=(2, 200)).astype(np.float32)
        with pytest.raises(ValueError, match="F shape"):
            extract_soma_features(_make_stat(3, rng), F, Fneu, fps=10.0)

    def test_stat_length_mismatch_raises(self) -> None:
        rng = np.random.default_rng(14)
        F = rng.uniform(100, 500, size=(3, 200)).astype(np.float32)
        Fneu = rng.uniform(50, 200, size=(3, 200)).astype(np.float32)
        with pytest.raises(ValueError, match="len\\(stat\\)"):
            extract_soma_features(_make_stat(2, rng), F, Fneu, fps=10.0)

    @settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_rois=st.integers(min_value=1, max_value=8),
        n_frames=st.integers(min_value=20, max_value=300),
    )
    def test_returns_dataframe_with_correct_shape(self, n_rois: int, n_frames: int) -> None:
        rng = np.random.default_rng(15)
        stat = _make_stat(n_rois, rng)
        F = rng.uniform(100, 500, size=(n_rois, n_frames)).astype(np.float32)
        Fneu = rng.uniform(50, 200, size=(n_rois, n_frames)).astype(np.float32)
        df = extract_soma_features(stat, F, Fneu, fps=10.0)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (n_rois, len(FEATURE_COLUMNS))
