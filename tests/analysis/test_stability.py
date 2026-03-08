"""Tests for hm2p.analysis.stability — HD tuning temporal stability."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.stability import (
    light_dark_stability,
    sliding_window_stability,
    split_temporal_halves,
)


def _make_stable_cell(n=5000, pref=90.0, kappa=3.0, seed=42):
    """HD cell with stable tuning throughout."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
    theta = np.deg2rad(hd)
    signal = 0.1 + np.exp(kappa * np.cos(theta - np.deg2rad(pref)))
    signal /= signal.max()
    signal += rng.normal(0, 0.1, n)
    signal = np.clip(signal, 0, None)
    mask = np.ones(n, dtype=bool)
    return signal, hd, mask


def _make_drifting_cell(n=6000, pref_start=0.0, pref_end=180.0, kappa=3.0, seed=42):
    """HD cell whose preferred direction drifts from start to end."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
    theta = np.deg2rad(hd)
    # Linearly interpolate preferred direction
    prefs = np.linspace(pref_start, pref_end, n)
    signal = np.zeros(n)
    for i in range(n):
        signal[i] = 0.1 + np.exp(kappa * np.cos(theta[i] - np.deg2rad(prefs[i])))
    signal /= signal.max()
    signal += rng.normal(0, 0.1, n)
    signal = np.clip(signal, 0, None)
    mask = np.ones(n, dtype=bool)
    return signal, hd, mask


class TestSplitTemporalHalves:
    """Tests for split_temporal_halves."""

    def test_stable_cell_high_correlation(self):
        signal, hd, mask = _make_stable_cell()
        result = split_temporal_halves(signal, hd, mask)
        assert result["correlation"] > 0.7

    def test_stable_cell_small_pd_shift(self):
        signal, hd, mask = _make_stable_cell()
        result = split_temporal_halves(signal, hd, mask)
        assert abs(result["pd_shift_deg"]) < 30

    def test_drifting_cell_lower_correlation(self):
        signal, hd, mask = _make_drifting_cell()
        result = split_temporal_halves(signal, hd, mask)
        # Drifting cell should have lower correlation between halves
        stable_result = split_temporal_halves(*_make_stable_cell())
        assert result["correlation"] < stable_result["correlation"]

    def test_output_keys(self):
        signal, hd, mask = _make_stable_cell()
        result = split_temporal_halves(signal, hd, mask)
        expected = {"correlation", "pd_shift_deg", "mvl_half1", "mvl_half2",
                    "tuning_curve_1", "tuning_curve_2", "bin_centers"}
        assert set(result.keys()) == expected

    def test_tuning_curve_shapes(self):
        signal, hd, mask = _make_stable_cell()
        result = split_temporal_halves(signal, hd, mask, n_bins=36)
        assert result["tuning_curve_1"].shape == (36,)
        assert result["tuning_curve_2"].shape == (36,)
        assert result["bin_centers"].shape == (36,)


class TestSlidingWindowStability:
    """Tests for sliding_window_stability."""

    def test_stable_cell_consistent_mvl(self):
        signal, hd, mask = _make_stable_cell(n=5000)
        result = sliding_window_stability(
            signal, hd, mask, window_frames=1000, step_frames=500,
        )
        assert result["n_windows"] > 0
        # MVL should be relatively consistent
        mvl_std = np.std(result["mvls"])
        assert mvl_std < 0.25

    def test_stable_cell_consistent_pd(self):
        signal, hd, mask = _make_stable_cell(n=10000, pref=90.0)
        result = sliding_window_stability(
            signal, hd, mask, window_frames=2000, step_frames=1000,
        )
        # Preferred directions should cluster near 90° (skip first unstable window)
        pds = result["preferred_dirs"]
        if len(pds) > 2:
            later_pds = pds[1:]  # Skip first window (poor HD coverage)
            diffs = np.abs(((later_pds - 90 + 180) % 360) - 180)
            assert np.median(diffs) < 40

    def test_window_centers_spacing(self):
        signal, hd, mask = _make_stable_cell(n=3000)
        result = sliding_window_stability(
            signal, hd, mask, window_frames=500, step_frames=200,
        )
        if result["n_windows"] > 1:
            diffs = np.diff(result["window_centers"])
            assert np.all(diffs == 200)

    def test_short_signal_few_windows(self):
        signal, hd, mask = _make_stable_cell(n=500)
        result = sliding_window_stability(
            signal, hd, mask, window_frames=400, step_frames=200,
        )
        assert result["n_windows"] <= 2


class TestLightDarkStability:
    """Tests for light_dark_stability."""

    def test_stable_cell_light_dark_similar(self):
        """Stable cell should tune similarly in light and dark."""
        signal, hd, mask = _make_stable_cell(n=6000)
        # Alternating 1-min cycles at 30 Hz
        light_on = np.zeros(6000, dtype=bool)
        cycle = 30 * 60  # 1800 frames per cycle
        for start in range(0, 6000, 2 * cycle):
            light_on[start:min(start + cycle, 6000)] = True
        result = light_dark_stability(signal, hd, mask, light_on)
        assert result["correlation"] > 0.5

    def test_output_keys(self):
        signal, hd, mask = _make_stable_cell()
        light_on = np.ones(len(signal), dtype=bool)
        light_on[len(signal) // 2:] = False
        result = light_dark_stability(signal, hd, mask, light_on)
        expected = {"correlation", "pd_shift_deg", "mvl_light", "mvl_dark",
                    "tuning_curve_light", "tuning_curve_dark", "bin_centers"}
        assert set(result.keys()) == expected

    def test_mvl_values_valid(self):
        signal, hd, mask = _make_stable_cell()
        light_on = np.ones(len(signal), dtype=bool)
        light_on[len(signal) // 2:] = False
        result = light_dark_stability(signal, hd, mask, light_on)
        assert 0 <= result["mvl_light"] <= 1
        assert 0 <= result["mvl_dark"] <= 1

    def test_all_light_on(self):
        """If all frames are light-on, dark tuning should be mostly NaN."""
        signal, hd, mask = _make_stable_cell(n=1000)
        light_on = np.ones(1000, dtype=bool)
        result = light_dark_stability(signal, hd, mask, light_on)
        # Dark tuning curve should be all NaN
        assert np.all(np.isnan(result["tuning_curve_dark"]))
