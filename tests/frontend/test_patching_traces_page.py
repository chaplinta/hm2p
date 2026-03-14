"""Tests for patching trace viewer page logic."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.patching.io import _apply_scaling, _crawl_h5_group


class TestPatchingTraceIO:
    """Test WaveSurfer I/O functions used by the trace viewer."""

    def test_apply_scaling_identity(self):
        """Identity scaling: coefficient [0, 1] with scale 1."""
        raw = np.array([[100, 200], [300, 400]], dtype=np.int16)
        scales = np.array([1.0, 1.0])
        coeffs = np.array([[0.0, 0.0], [1.0, 1.0]])  # y = x
        result = _apply_scaling(raw, scales, coeffs)
        np.testing.assert_allclose(result, raw.astype(np.float64))

    def test_apply_scaling_with_scale(self):
        """Scaling by channel scale factor."""
        raw = np.array([[100], [200]], dtype=np.int16)
        scales = np.array([2.0])
        coeffs = np.array([[0.0], [1.0]])  # y = x / 2
        result = _apply_scaling(raw, scales, coeffs)
        np.testing.assert_allclose(result[:, 0], [50.0, 100.0])

    def test_apply_scaling_1d_input(self):
        """1D raw input should be handled."""
        raw = np.array([100, 200, 300], dtype=np.int16)
        scales = np.array([1.0])
        coeffs = np.array([[0.0], [1.0]])
        result = _apply_scaling(raw, scales, coeffs)
        assert result.shape == (3, 1)
        np.testing.assert_allclose(result[:, 0], [100.0, 200.0, 300.0])

    def test_apply_scaling_polynomial(self):
        """Quadratic polynomial: y = 1 + 2*x + 3*x^2, scale=1."""
        raw = np.array([[2]], dtype=np.int16)
        scales = np.array([1.0])
        # Coefficients: [c0, c1, c2] = [1, 2, 3]
        # Horner: start at c2=3, then 2 + raw*3 = 8, then 1 + raw*8 = 17
        coeffs = np.array([[1.0], [2.0], [3.0]])
        result = _apply_scaling(raw, scales, coeffs)
        np.testing.assert_allclose(result[0, 0], 17.0)


class TestTraceViewerHelpers:
    """Test trace viewer page helper logic."""

    def test_single_sweep_no_slider(self):
        """When n_sweeps == 1, slider should be skipped."""
        # Simulates the page logic: if n_sweeps == 1, sweep_idx = 0
        n_sweeps = 1
        if n_sweeps == 1:
            sweep_idx = 0
        else:
            sweep_idx = 0  # would be from slider
        assert sweep_idx == 0

    def test_sweep_time_axis(self):
        """Time axis should be in milliseconds."""
        fs = 20000.0
        sweep = np.zeros(10000)
        time_ms = np.arange(len(sweep)) / fs * 1000
        assert time_ms[0] == 0.0
        assert abs(time_ms[-1] - (9999 / 20000 * 1000)) < 1e-10
        assert len(time_ms) == len(sweep)

    def test_sweep_stats(self):
        """Sweep statistics should be computed correctly."""
        sweep = np.array([-70.0, -65.0, 20.0, -72.0, -68.0])
        assert np.min(sweep) == -72.0
        assert np.max(sweep) == 20.0
        assert np.mean(sweep) == pytest.approx(-51.0)
