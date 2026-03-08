"""Tests for AHV Analysis page logic."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.ahv import (
    ahv_modulation_index,
    ahv_tuning_curve,
    anticipatory_time_delay,
    compute_ahv,
)


def _make_ahv_cell(n=5000, kappa=3.0, ahv_gain=0.5, seed=42):
    """Generate cell with HD tuning and AHV modulation."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
    theta = np.deg2rad(hd)
    signal = 0.1 + np.exp(kappa * np.cos(theta))
    signal /= signal.max()
    # Add AHV modulation
    ahv = compute_ahv(hd, fps=30.0)
    signal += ahv_gain * np.abs(ahv) / np.max(np.abs(ahv) + 1e-10)
    signal += rng.normal(0, 0.1, n)
    signal = np.clip(signal, 0, None)
    mask = np.ones(n, dtype=bool)
    return signal, hd, mask


class TestAHVPageWorkflow:
    """Test AHV analysis workflow as used in page."""

    def test_ahv_computation(self):
        rng = np.random.default_rng(42)
        hd = np.cumsum(rng.normal(0, 5, 3000)) % 360.0
        ahv = compute_ahv(hd, fps=30.0)
        assert ahv.shape == (3000,)
        # Should have both positive and negative values (CW and CCW)
        assert np.any(ahv > 0)
        assert np.any(ahv < 0)

    def test_ahv_tuning_curve(self):
        signal, hd, mask = _make_ahv_cell()
        ahv = compute_ahv(hd, fps=30.0)
        tc, bc = ahv_tuning_curve(signal, ahv, mask)
        assert len(tc) > 0
        assert len(bc) == len(tc)

    def test_modulation_index(self):
        signal, hd, mask = _make_ahv_cell()
        ahv = compute_ahv(hd, fps=30.0)
        tc, bc = ahv_tuning_curve(signal, ahv, mask)
        result = ahv_modulation_index(tc, bc)
        assert "asymmetry_index" in result
        assert "modulation_depth" in result
        assert -1 <= result["asymmetry_index"] <= 1

    def test_anticipatory_time_delay(self):
        signal, hd, mask = _make_ahv_cell()
        result = anticipatory_time_delay(signal, hd, mask)
        assert "best_lag_ms" in result
        assert "best_mvl" in result
        assert len(result["lags_ms"]) == len(result["mvls"])

    def test_atd_best_mvl_positive(self):
        signal, hd, mask = _make_ahv_cell(kappa=4.0)
        result = anticipatory_time_delay(signal, hd, mask)
        assert result["best_mvl"] > 0
