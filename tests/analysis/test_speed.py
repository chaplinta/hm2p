"""Tests for hm2p.analysis.speed — speed modulation analysis."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.speed import (
    hd_tuning_by_speed,
    speed_modulation_index,
    speed_tuning_curve,
)


def _make_speed_cell(n=5000, pref=90.0, kappa=3.0, speed_gain=0.5,
                     noise=0.15, seed=42):
    """Cell with HD tuning and speed modulation."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
    speed = np.abs(rng.normal(10, 5, n))
    signal = 0.1 + np.exp(kappa * np.cos(np.deg2rad(hd) - np.deg2rad(pref)))
    signal /= signal.max()
    # Add speed modulation
    signal *= (1 + speed_gain * speed / np.max(speed))
    signal += rng.normal(0, noise, n)
    signal = np.clip(signal, 0, None)
    mask = np.ones(n, dtype=bool)
    return signal, hd, speed, mask


class TestSpeedTuningCurve:
    """Tests for speed_tuning_curve."""

    def test_output_shapes(self):
        signal, _, speed, mask = _make_speed_cell()
        tc, bc = speed_tuning_curve(signal, speed, mask, n_bins=15)
        assert tc.shape == (15,)
        assert bc.shape == (15,)

    def test_speed_modulated_cell_positive_trend(self):
        """Speed-modulated cell should have higher activity at higher speeds."""
        signal, _, speed, mask = _make_speed_cell(speed_gain=1.0)
        tc, bc = speed_tuning_curve(signal, speed, mask, n_bins=10)
        valid = ~np.isnan(tc)
        if valid.sum() >= 3:
            corr = np.corrcoef(bc[valid], tc[valid])[0, 1]
            assert corr > 0  # Positive speed-activity relationship

    def test_max_speed_parameter(self):
        signal, _, speed, mask = _make_speed_cell()
        tc, bc = speed_tuning_curve(signal, speed, mask, max_speed=20.0)
        assert bc[-1] < 20.5


class TestSpeedModulationIndex:
    """Tests for speed_modulation_index."""

    def test_output_keys(self):
        signal, _, speed, mask = _make_speed_cell()
        result = speed_modulation_index(signal, speed, mask)
        expected = {"speed_modulation_index", "mean_signal_fast",
                    "mean_signal_slow", "speed_correlation"}
        assert set(result.keys()) == expected

    def test_smi_bounded(self):
        signal, _, speed, mask = _make_speed_cell()
        result = speed_modulation_index(signal, speed, mask)
        assert -1 <= result["speed_modulation_index"] <= 1

    def test_speed_modulated_positive_smi(self):
        """Cell with positive speed modulation should have positive SMI."""
        signal, _, speed, mask = _make_speed_cell(speed_gain=1.0)
        result = speed_modulation_index(signal, speed, mask)
        assert result["speed_modulation_index"] > 0
        assert result["mean_signal_fast"] > result["mean_signal_slow"]

    def test_positive_correlation(self):
        """Speed-modulated cell should have positive speed-signal correlation."""
        signal, _, speed, mask = _make_speed_cell(speed_gain=1.0)
        result = speed_modulation_index(signal, speed, mask)
        assert result["speed_correlation"] > 0

    def test_custom_threshold(self):
        signal, _, speed, mask = _make_speed_cell()
        result = speed_modulation_index(signal, speed, mask, speed_threshold=15.0)
        assert "speed_modulation_index" in result


class TestHDTuningBySpeed:
    """Tests for hd_tuning_by_speed."""

    def test_output_keys(self):
        signal, hd, speed, mask = _make_speed_cell()
        result = hd_tuning_by_speed(signal, hd, speed, mask)
        expected = {"tuning_curves", "bin_centers", "mvls", "pds",
                    "speed_labels", "speed_thresholds"}
        assert set(result.keys()) == expected

    def test_three_speed_groups(self):
        """Default (0.33, 0.67) should give 3 groups."""
        signal, hd, speed, mask = _make_speed_cell()
        result = hd_tuning_by_speed(signal, hd, speed, mask)
        assert len(result["tuning_curves"]) == 3
        assert len(result["mvls"]) == 3
        assert len(result["pds"]) == 3

    def test_mvls_valid(self):
        signal, hd, speed, mask = _make_speed_cell()
        result = hd_tuning_by_speed(signal, hd, speed, mask)
        for mvl in result["mvls"]:
            if np.isfinite(mvl):
                assert 0 <= mvl <= 1

    def test_pds_in_range(self):
        signal, hd, speed, mask = _make_speed_cell()
        result = hd_tuning_by_speed(signal, hd, speed, mask)
        for pd in result["pds"]:
            if np.isfinite(pd):
                assert 0 <= pd < 360

    def test_custom_quantiles(self):
        signal, hd, speed, mask = _make_speed_cell()
        result = hd_tuning_by_speed(signal, hd, speed, mask,
                                    speed_quantiles=(0.25, 0.5, 0.75))
        assert len(result["tuning_curves"]) == 4  # 4 groups from 3 quantiles
