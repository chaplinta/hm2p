"""Tests for calcium/dff.py — dF/F0 computation."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.calcium.dff import compute_dff

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
