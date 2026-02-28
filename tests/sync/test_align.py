"""Tests for sync/align.py — neural–behavioural synchronisation."""

from __future__ import annotations

import numpy as np

from hm2p.sync.align import resample_bool_to_imaging_rate, resample_to_imaging_rate


def test_resample_constant_signal() -> None:
    """Resampling a constant signal returns the same constant."""
    src_times = np.linspace(0, 10, 1000)
    dst_times = np.linspace(0, 10, 300)
    values = np.full(1000, 42.0)
    result = resample_to_imaging_rate(values, src_times, dst_times)
    np.testing.assert_allclose(result, 42.0, rtol=1e-5)


def test_resample_linear_signal() -> None:
    """Resampling a linear ramp preserves values at interpolation points."""
    src_times = np.linspace(0.0, 1.0, 1000)
    dst_times = np.linspace(0.0, 1.0, 100)
    values = src_times.copy()  # identity ramp
    result = resample_to_imaging_rate(values, src_times, dst_times)
    np.testing.assert_allclose(result, dst_times, atol=1e-3)


def test_resample_output_shape() -> None:
    """Output length equals len(dst_times)."""
    src_times = np.linspace(0, 60, 6000)
    dst_times = np.linspace(0, 60, 1800)
    values = np.random.default_rng(0).standard_normal(6000)
    result = resample_to_imaging_rate(values, src_times, dst_times)
    assert result.shape == (1800,)


def test_resample_bool_preserves_dtype() -> None:
    """resample_bool_to_imaging_rate returns bool array."""
    src_times = np.linspace(0, 60, 6000)
    dst_times = np.linspace(0, 60, 1800)
    mask = np.zeros(6000, dtype=bool)
    mask[2000:4000] = True
    result = resample_bool_to_imaging_rate(mask, src_times, dst_times)
    assert result.dtype == bool
    assert result.shape == (1800,)
