"""Tests for calcium/neuropil.py — neuropil subtraction."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.calcium.neuropil import subtract_fissa, subtract_fixed_coefficient

# ---------------------------------------------------------------------------
# subtract_fixed_coefficient — pure numpy, fully testable
# ---------------------------------------------------------------------------


def test_fixed_coefficient_default(rng: np.random.Generator) -> None:
    """Default coefficient (0.7) is applied correctly."""
    n_rois, n_frames = 10, 500
    F = rng.uniform(100, 1000, (n_rois, n_frames)).astype(np.float32)
    Fneu = rng.uniform(50, 500, (n_rois, n_frames)).astype(np.float32)
    result = subtract_fixed_coefficient(F, Fneu)
    np.testing.assert_allclose(result, F - 0.7 * Fneu, rtol=1e-5)


def test_fixed_coefficient_custom(rng: np.random.Generator) -> None:
    """Custom coefficient is applied correctly."""
    F = rng.uniform(0, 1, (5, 100)).astype(np.float32)
    Fneu = rng.uniform(0, 1, (5, 100)).astype(np.float32)
    result = subtract_fixed_coefficient(F, Fneu, coefficient=0.5)
    np.testing.assert_allclose(result, F - 0.5 * Fneu, rtol=1e-5)


def test_fixed_coefficient_output_shape(rng: np.random.Generator) -> None:
    """Output shape matches input shape."""
    F = rng.standard_normal((20, 300)).astype(np.float32)
    Fneu = rng.standard_normal((20, 300)).astype(np.float32)
    result = subtract_fixed_coefficient(F, Fneu)
    assert result.shape == F.shape


@given(
    coefficient=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_fixed_coefficient_property_range(coefficient: float) -> None:
    """subtract_fixed_coefficient produces finite outputs for any valid coefficient."""
    rng = np.random.default_rng(0)
    F = rng.uniform(0, 1000, (5, 50)).astype(np.float32)
    Fneu = rng.uniform(0, 500, (5, 50)).astype(np.float32)
    result = subtract_fixed_coefficient(F, Fneu, coefficient=coefficient)
    assert np.all(np.isfinite(result))


def test_fixed_coefficient_zero(rng: np.random.Generator) -> None:
    """Coefficient of 0 returns F unchanged."""
    F = rng.uniform(100, 500, (5, 100)).astype(np.float32)
    Fneu = rng.uniform(50, 200, (5, 100)).astype(np.float32)
    result = subtract_fixed_coefficient(F, Fneu, coefficient=0.0)
    np.testing.assert_array_equal(result, F)


def test_fixed_coefficient_one(rng: np.random.Generator) -> None:
    """Coefficient of 1 subtracts full neuropil."""
    F = rng.uniform(100, 500, (5, 100)).astype(np.float32)
    Fneu = rng.uniform(50, 200, (5, 100)).astype(np.float32)
    result = subtract_fixed_coefficient(F, Fneu, coefficient=1.0)
    np.testing.assert_allclose(result, F - Fneu, rtol=1e-5)


def test_fixed_coefficient_can_go_negative() -> None:
    """Result can be negative when Fneu > F."""
    F = np.full((1, 10), 100.0, dtype=np.float32)
    Fneu = np.full((1, 10), 200.0, dtype=np.float32)
    result = subtract_fixed_coefficient(F, Fneu, coefficient=0.7)
    assert np.all(result < 0)


def test_fissa_not_implemented() -> None:
    """subtract_fissa raises NotImplementedError (deferred)."""
    F = np.ones((5, 100), dtype=np.float32)
    masks = np.zeros((5, 64, 64), dtype=bool)
    with pytest.raises(NotImplementedError):
        subtract_fissa(F, masks)
