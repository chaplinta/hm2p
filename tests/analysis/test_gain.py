"""Tests for hm2p.analysis.gain — gain modulation analysis."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.gain import (
    epoch_gain_tracking,
    gain_modulation_index,
    population_gain_modulation,
)


def _make_cell(n=6000, pref=90.0, kappa=3.0, noise=0.15, seed=42):
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
    signal = 0.1 + np.exp(kappa * np.cos(np.deg2rad(hd) - np.deg2rad(pref)))
    signal /= signal.max()
    signal += rng.normal(0, noise, n)
    signal = np.clip(signal, 0, None)
    mask = np.ones(n, dtype=bool)
    return signal, hd, mask


def _make_light_on(n, cycle=1800):
    light_on = np.zeros(n, dtype=bool)
    for start in range(0, n, 2 * cycle):
        light_on[start:min(start + cycle, n)] = True
    return light_on


class TestGainModulationIndex:
    """Tests for gain_modulation_index."""

    def test_equal_gain_near_zero(self):
        """Same cell in light/dark should have gain index near 0."""
        signal, hd, mask = _make_cell()
        light_on = _make_light_on(len(signal))
        result = gain_modulation_index(signal, hd, mask, light_on)
        assert abs(result["gain_index"]) < 0.3

    def test_output_keys(self):
        signal, hd, mask = _make_cell()
        light_on = _make_light_on(len(signal))
        result = gain_modulation_index(signal, hd, mask, light_on)
        expected = {"gain_index", "peak_light", "peak_dark",
                    "dynamic_range_light", "dynamic_range_dark",
                    "mean_rate_light", "mean_rate_dark"}
        assert set(result.keys()) == expected

    def test_gain_index_bounded(self):
        signal, hd, mask = _make_cell()
        light_on = _make_light_on(len(signal))
        result = gain_modulation_index(signal, hd, mask, light_on)
        assert -1 <= result["gain_index"] <= 1

    def test_peaks_positive(self):
        signal, hd, mask = _make_cell()
        light_on = _make_light_on(len(signal))
        result = gain_modulation_index(signal, hd, mask, light_on)
        assert result["peak_light"] > 0
        assert result["peak_dark"] > 0

    def test_dynamic_range_positive(self):
        signal, hd, mask = _make_cell()
        light_on = _make_light_on(len(signal))
        result = gain_modulation_index(signal, hd, mask, light_on)
        assert result["dynamic_range_light"] >= 0
        assert result["dynamic_range_dark"] >= 0

    def test_all_light_no_dark_data(self):
        signal, hd, mask = _make_cell(n=3000)
        light_on = np.ones(3000, dtype=bool)
        result = gain_modulation_index(signal, hd, mask, light_on)
        assert result["peak_dark"] == 0.0
        assert result["mean_rate_dark"] == 0.0

    def test_scaled_signal_changes_gain(self):
        """Scaling signal in dark should change gain index."""
        rng = np.random.default_rng(42)
        n = 6000
        hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
        signal = 0.1 + np.exp(3.0 * np.cos(np.deg2rad(hd)))
        signal /= signal.max()
        light_on = _make_light_on(n)
        # Reduce signal in dark
        signal_mod = signal.copy()
        signal_mod[~light_on] *= 0.3
        signal_mod = np.clip(signal_mod + rng.normal(0, 0.05, n), 0, None)
        mask = np.ones(n, dtype=bool)
        result = gain_modulation_index(signal_mod, hd, mask, light_on)
        assert result["gain_index"] > 0.1  # Light should be stronger


class TestPopulationGainModulation:
    """Tests for population_gain_modulation."""

    def test_output_length(self):
        signal, hd, mask = _make_cell()
        signals = np.vstack([signal, signal, signal])
        light_on = _make_light_on(len(signal))
        results = population_gain_modulation(signals, hd, mask, light_on)
        assert len(results) == 3

    def test_all_cells_have_keys(self):
        signal, hd, mask = _make_cell()
        signals = np.vstack([signal, signal])
        light_on = _make_light_on(len(signal))
        results = population_gain_modulation(signals, hd, mask, light_on)
        for r in results:
            assert "gain_index" in r


class TestEpochGainTracking:
    """Tests for epoch_gain_tracking."""

    def test_output_keys(self):
        signal, hd, mask = _make_cell()
        light_on = _make_light_on(len(signal))
        result = epoch_gain_tracking(signal, hd, mask, light_on)
        expected = {"epoch_centers", "epoch_peaks", "epoch_dynamic_ranges",
                    "epoch_mvls", "epoch_is_light", "n_epochs"}
        assert set(result.keys()) == expected

    def test_epochs_found(self):
        signal, hd, mask = _make_cell()
        light_on = _make_light_on(len(signal))
        result = epoch_gain_tracking(signal, hd, mask, light_on)
        assert result["n_epochs"] > 0

    def test_peaks_positive(self):
        signal, hd, mask = _make_cell()
        light_on = _make_light_on(len(signal))
        result = epoch_gain_tracking(signal, hd, mask, light_on)
        assert np.all(result["epoch_peaks"] > 0)

    def test_mvls_bounded(self):
        signal, hd, mask = _make_cell()
        light_on = _make_light_on(len(signal))
        result = epoch_gain_tracking(signal, hd, mask, light_on)
        assert np.all(result["epoch_mvls"] >= 0)
        assert np.all(result["epoch_mvls"] <= 1)

    def test_alternating_light_dark(self):
        signal, hd, mask = _make_cell()
        light_on = _make_light_on(len(signal))
        result = epoch_gain_tracking(signal, hd, mask, light_on)
        if result["n_epochs"] >= 2:
            assert True in result["epoch_is_light"]
            assert False in result["epoch_is_light"]
