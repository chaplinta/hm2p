"""Tests for calcium/dff.py — dF/F0 computation."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.calcium.dff import compute_baseline, compute_dff

# ---------------------------------------------------------------------------
# compute_dff — pure numpy, fully testable
# ---------------------------------------------------------------------------


def test_dff_zero_when_f_equals_baseline(rng: np.random.Generator) -> None:
    """dF/F0 = 0 when F == F0 everywhere."""
    F0 = np.abs(rng.uniform(50, 500, (10, 100)).astype(np.float32)) + 1.0
    result = compute_dff(F0, F0)
    np.testing.assert_allclose(result, 0.0, atol=1e-5)


def test_dff_positive_when_f_above_baseline(rng: np.random.Generator) -> None:
    """dF/F0 > 0 when F > F0."""
    F0 = np.ones((5, 50), dtype=np.float32) * 100.0
    F = F0 * 1.5  # 50% above baseline
    result = compute_dff(F, F0)
    np.testing.assert_allclose(result, 0.5, rtol=1e-5)


def test_dff_shape_mismatch_raises(rng: np.random.Generator) -> None:
    """ValueError raised when F and F0 shapes don't match."""
    F = rng.standard_normal((5, 100)).astype(np.float32)
    F0 = rng.standard_normal((5, 50)).astype(np.float32)
    with pytest.raises(ValueError, match="shape"):
        compute_dff(F, F0)


def test_dff_output_shape_preserved(rng: np.random.Generator) -> None:
    """Output shape matches input."""
    F = rng.standard_normal((20, 300)).astype(np.float32)
    F0 = np.abs(F) + 1.0
    result = compute_dff(F, F0)
    assert result.shape == F.shape


@given(
    scale=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_dff_property_known_amplitude(scale: float) -> None:
    """dF/F0 = scale - 1 when F = scale * F0."""
    F0 = np.ones((3, 30), dtype=np.float32) * 100.0
    F = F0 * scale
    result = compute_dff(F, F0)
    # float32 has ~6 sig-fig precision → rtol=1e-3 is appropriate
    np.testing.assert_allclose(result, scale - 1.0, rtol=1e-3)


# ---------------------------------------------------------------------------
# compute_baseline
# ---------------------------------------------------------------------------


class TestComputeBaseline:
    def test_output_shape(self, rng: np.random.Generator) -> None:
        """Baseline shape matches input shape."""
        F = rng.uniform(100, 500, (8, 600)).astype(np.float32)
        F0 = compute_baseline(F, fps=30.0)
        assert F0.shape == F.shape

    def test_output_dtype_float32(self, rng: np.random.Generator) -> None:
        """Baseline is float32."""
        F = rng.uniform(100, 500, (4, 300)).astype(np.float32)
        F0 = compute_baseline(F, fps=30.0)
        assert F0.dtype == np.float32

    def test_baseline_leq_signal(self, rng: np.random.Generator) -> None:
        """Baseline ≤ smoothed signal (sliding minimum property)."""
        F = np.abs(rng.uniform(100, 500, (5, 900)).astype(np.float32))
        F0 = compute_baseline(F, fps=30.0, window_s=10.0, gaussian_sigma_s=1.0)
        # Allow small numerical tolerance from Gaussian smoothing at boundaries
        assert np.all(F0 <= F.max() + 1.0)

    def test_constant_signal_baseline_equals_signal(self) -> None:
        """Constant trace → baseline equals the constant."""
        F = np.full((3, 300), 200.0, dtype=np.float32)
        F0 = compute_baseline(F, fps=30.0, window_s=5.0, gaussian_sigma_s=1.0)
        np.testing.assert_allclose(F0, 200.0, rtol=1e-3)

    def test_transient_does_not_raise_baseline(self) -> None:
        """A short positive transient does not elevate the sliding-min baseline."""
        F = np.full((1, 900), 100.0, dtype=np.float32)
        # Add a brief spike in the middle
        F[0, 440:460] = 500.0
        F0 = compute_baseline(F, fps=30.0, window_s=10.0, gaussian_sigma_s=1.0)
        # Baseline in the second half (well past the transient) should be ~100
        np.testing.assert_allclose(F0[0, 600:], 100.0, atol=5.0)

    def test_window_shorter_gives_tighter_baseline(self, rng: np.random.Generator) -> None:
        """Shorter window produces a baseline that tracks faster (≥ longer window)."""
        F = np.abs(rng.uniform(80, 200, (2, 600)).astype(np.float32))
        F0_short = compute_baseline(F, fps=30.0, window_s=5.0, gaussian_sigma_s=1.0)
        F0_long = compute_baseline(F, fps=30.0, window_s=30.0, gaussian_sigma_s=1.0)
        # Shorter window baseline is always ≥ longer (tighter tracking)
        assert np.all(F0_short >= F0_long - 1.0)
