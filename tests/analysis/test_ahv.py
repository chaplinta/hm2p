"""Tests for hm2p.analysis.ahv — angular head velocity analysis."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from hm2p.analysis.ahv import (
    ahv_modulation_index,
    ahv_tuning_curve,
    anticipatory_time_delay,
    compute_ahv,
)


class TestComputeAHV:
    """Tests for compute_ahv."""

    def test_stationary_zero_ahv(self):
        """Constant HD should give zero AHV."""
        hd = np.full(100, 90.0)
        ahv = compute_ahv(hd, fps=30.0, smoothing_frames=1)
        assert np.allclose(ahv[1:], 0.0, atol=1e-10)

    def test_constant_rotation(self):
        """Linear HD change should give constant AHV."""
        hd = np.linspace(0, 360, 101)  # 360° in 100 steps
        ahv = compute_ahv(hd, fps=10.0, smoothing_frames=1)
        # Each step is 3.6°, at 10 fps = 36 deg/s
        expected = 3.6 * 10.0
        # Exclude first (0) and last (wrapping) frames
        assert np.allclose(ahv[1:-1], expected, atol=1.0)

    def test_output_length(self):
        hd = np.random.default_rng(42).uniform(0, 360, 500)
        ahv = compute_ahv(hd)
        assert len(ahv) == len(hd)

    def test_wrapping_handled(self):
        """AHV should handle 359° → 1° correctly."""
        hd = np.array([350, 355, 0, 5, 10], dtype=float)
        ahv = compute_ahv(hd, fps=1.0, smoothing_frames=1)
        # All transitions are +5°
        assert np.allclose(ahv[1:], 5.0, atol=0.5)


class TestAHVTuningCurve:
    """Tests for ahv_tuning_curve."""

    def test_output_shapes(self):
        signal = np.random.default_rng(42).normal(0, 1, 1000)
        ahv = np.random.default_rng(42).normal(0, 100, 1000)
        mask = np.ones(1000, dtype=bool)
        tc, bc = ahv_tuning_curve(signal, ahv, mask, n_bins=20)
        assert tc.shape == (20,)
        assert bc.shape == (20,)

    def test_bin_centers_symmetric(self):
        signal = np.ones(500)
        ahv = np.zeros(500)
        mask = np.ones(500, dtype=bool)
        _, bc = ahv_tuning_curve(signal, ahv, mask, n_bins=20, max_ahv=600)
        # Should be symmetric around 0
        assert np.isclose(bc[0], -bc[-1], atol=bc[1] - bc[0])

    def test_cw_biased_signal(self):
        """Signal correlated with CW rotation should peak at positive AHV."""
        rng = np.random.default_rng(42)
        ahv = rng.normal(0, 200, 3000)
        # Signal proportional to CW rotation
        signal = np.clip(ahv / 200 + rng.normal(0, 0.3, 3000), 0, None)
        mask = np.ones(3000, dtype=bool)
        tc, bc = ahv_tuning_curve(signal, ahv, mask, n_bins=30, smoothing_sigma=0)
        # Peak should be at positive AHV
        valid = ~np.isnan(tc)
        peak_ahv = bc[valid][np.argmax(tc[valid])]
        assert peak_ahv > 0


class TestAHVModulationIndex:
    """Tests for ahv_modulation_index."""

    def test_symmetric_tuning(self):
        """Symmetric tuning curve should have ~0 asymmetry."""
        n = 30
        bc = np.linspace(-600, 600, n)
        tc = np.exp(-bc**2 / (2 * 200**2))
        result = ahv_modulation_index(tc, bc)
        assert abs(result["asymmetry_index"]) < 0.1

    def test_cw_biased(self):
        """CW-biased tuning should have positive asymmetry index."""
        bc = np.linspace(-600, 600, 30)
        tc = np.exp(-(bc - 200)**2 / (2 * 100**2))  # Peak at +200
        result = ahv_modulation_index(tc, bc)
        assert result["asymmetry_index"] > 0
        assert result["preferred_ahv"] > 0

    def test_modulation_depth(self):
        bc = np.linspace(-600, 600, 30)
        tc = np.array([0.1 + 0.9 * np.exp(-(b - 200)**2 / (2 * 100**2)) for b in bc])
        result = ahv_modulation_index(tc, bc)
        assert result["modulation_depth"] > 0
        assert result["modulation_depth"] <= 1.0

    def test_nan_handling(self):
        bc = np.linspace(-600, 600, 20)
        tc = np.full(20, np.nan)
        tc[5:15] = 1.0  # Only middle bins valid
        result = ahv_modulation_index(tc, bc)
        assert "asymmetry_index" in result


class TestAnticipatoryTimeDelay:
    """Tests for anticipatory_time_delay."""

    def test_output_keys(self):
        rng = np.random.default_rng(42)
        n = 2000
        hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
        signal = np.exp(3 * np.cos(np.deg2rad(hd) - np.deg2rad(90)))
        signal /= signal.max()
        signal += rng.normal(0, 0.1, n)
        mask = np.ones(n, dtype=bool)
        result = anticipatory_time_delay(signal, hd, mask, max_lag_frames=5)
        assert "lags_ms" in result
        assert "mvls" in result
        assert "best_lag_ms" in result
        assert "best_mvl" in result

    def test_zero_lag_for_synchronous(self):
        """Signal perfectly aligned with HD should have ~0 lag."""
        rng = np.random.default_rng(42)
        n = 3000
        hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
        signal = np.exp(5 * np.cos(np.deg2rad(hd)))
        signal /= signal.max()
        mask = np.ones(n, dtype=bool)
        result = anticipatory_time_delay(signal, hd, mask, max_lag_frames=5, fps=30)
        # Best lag should be near 0
        assert abs(result["best_lag_ms"]) < 100  # Within ~3 frames at 30Hz

    def test_mvl_values_valid(self):
        rng = np.random.default_rng(42)
        n = 1000
        hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
        signal = rng.normal(0, 1, n)
        signal = np.clip(signal, 0, None)
        mask = np.ones(n, dtype=bool)
        result = anticipatory_time_delay(signal, hd, mask, max_lag_frames=3)
        assert np.all(result["mvls"] >= 0)
        assert np.all(result["mvls"] <= 1)
        assert len(result["lags_ms"]) == 7  # -3 to +3


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------


class TestHypothesisAHV:
    @given(
        n=st.integers(min_value=10, max_value=2000),
        fps=st.floats(min_value=1.0, max_value=100.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_ahv_handles_wraparound(self, n, fps):
        """AHV should handle 0/360 boundary without huge spikes.

        A smoothly increasing HD that wraps from 359 -> 0 should not produce
        AHV values much larger than the true rotation rate.
        """
        # Create HD that wraps smoothly: 5 deg/frame at given fps
        step = 5.0
        hd = np.arange(n, dtype=np.float64) * step % 360.0
        ahv = compute_ahv(hd, fps=fps, smoothing_frames=1)
        # True AHV should be ~step * fps everywhere (except edges)
        expected = step * fps
        # Allow some tolerance for edge frames and small numerical drift
        # Key property: no huge spikes from wrapping
        inner = ahv[1:-1]
        if len(inner) > 0:
            assert np.all(np.abs(inner) < expected * 3), (
                f"AHV spike detected: max={np.abs(inner).max():.1f}, "
                f"expected ~{expected:.1f}"
            )

    @given(
        n=st.integers(min_value=5, max_value=1000),
    )
    @settings(max_examples=50, deadline=None)
    def test_ahv_output_shape_matches_input(self, n):
        """compute_ahv output length always matches input length."""
        rng = np.random.default_rng(42)
        hd = rng.uniform(0, 360, n)
        ahv = compute_ahv(hd, fps=30.0, smoothing_frames=1)
        assert ahv.shape == (n,), f"Expected shape ({n},), got {ahv.shape}"
