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
    _fneu_corr,
    _peak_to_noise,
    _quick_dff,
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
        # All near zero (constant signal → ~0 dF/F).
        assert np.allclose(out, 0.0, atol=1e-3)

    def test_step_increase_yields_positive_dff(self) -> None:
        F = np.full((1, 200), 100.0, dtype=np.float32)
        F[0, 100:] = 200.0
        out = _quick_dff(F)
        # The post-step samples should be well above zero.
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
        x[100] = 20.0  # huge transient
        ratio = _peak_to_noise(x)
        assert ratio > 5.0


# ---------------------------------------------------------------------------
# _autocorr_halfwidth_s
# ---------------------------------------------------------------------------


class TestAutocorrHalfwidth:
    def test_constant_returns_nan(self) -> None:
        assert np.isnan(_autocorr_halfwidth_s(np.zeros(100), fps=10.0))

    def test_short_returns_nan(self) -> None:
        assert np.isnan(_autocorr_halfwidth_s(np.array([1.0, 2.0]), fps=10.0))

    def test_white_noise_short_halfwidth(self) -> None:
        """White noise drops below 0.5 within one or two lags."""
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, size=2000)
        hw = _autocorr_halfwidth_s(x, fps=10.0)
        assert np.isfinite(hw)
        assert hw < 1.0  # well under 1 s

    def test_smooth_signal_has_longer_halfwidth(self) -> None:
        """A heavily filtered signal has a wider autocorrelation."""
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
        x = np.zeros(50)
        y = np.linspace(0, 1, 50)
        assert np.isnan(_fneu_corr(x, y))

    def test_perfect_correlation(self) -> None:
        rng = np.random.default_rng(4)
        x = rng.normal(0, 1, size=200)
        # Spearman is monotonic-rank invariant — copying the trace gives r ≈ 1.
        assert _fneu_corr(x, x.copy()) == pytest.approx(1.0)

    def test_anti_correlation(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.normal(0, 1, size=200)
        assert _fneu_corr(x, -x) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# extract_soma_features
# ---------------------------------------------------------------------------


def _make_stat(n: int, rng: np.random.Generator) -> list[dict]:
    return [
        {
            "radius": float(rng.uniform(3.0, 10.0)),
            "compact": float(rng.uniform(0.3, 0.9)),
            "aspect_ratio": float(rng.uniform(1.0, 3.0)),
            "npix": int(rng.integers(50, 500)),
            "npix_norm": float(rng.uniform(0.5, 3.0)),
            "skew": float(rng.uniform(-1.0, 5.0)),
            "std": float(rng.uniform(0.1, 5.0)),
        }
        for _ in range(n)
    ]


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

    def test_short_trace_handles_gracefully(self) -> None:
        rng = np.random.default_rng(9)
        stat = _make_stat(2, rng)
        F = rng.uniform(100, 500, size=(2, 5)).astype(np.float32)
        Fneu = rng.uniform(50, 200, size=(2, 5)).astype(np.float32)
        df = extract_soma_features(stat, F, Fneu, fps=10.0)
        # Activity features may be NaN for very short traces — that is OK.
        assert df.shape == (2, len(FEATURE_COLUMNS))

    def test_constant_traces_yield_nan_activity(self) -> None:
        stat = _make_stat(2, np.random.default_rng(10))
        F = np.full((2, 200), 100.0, dtype=np.float32)
        Fneu = np.full((2, 200), 80.0, dtype=np.float32)
        df = extract_soma_features(stat, F, Fneu, fps=10.0)
        # Constant signals → NaN peak_to_noise / autocorr / fneu_corr.
        assert df["peak_to_noise_dff"].isna().all()
        assert df["autocorr_halfwidth_s"].isna().all()
        assert df["fneu_corr"].isna().all()

    def test_missing_stat_keys_use_defaults(self) -> None:
        stat = [{}, {"radius": 7.0}]  # ROI 0 fully empty, ROI 1 partial
        F = np.random.default_rng(11).uniform(100, 500, size=(2, 200)).astype(np.float32)
        Fneu = np.random.default_rng(12).uniform(50, 200, size=(2, 200)).astype(np.float32)
        df = extract_soma_features(stat, F, Fneu, fps=10.0)
        # Default radius is 5.0 (from soma_features._STAT_DEFAULTS).
        assert df.iloc[0]["radius"] == pytest.approx(5.0)
        assert df.iloc[1]["radius"] == pytest.approx(7.0)

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
