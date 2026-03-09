"""Tests for Drift Analysis page logic."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.stability import (
    dark_drift_rate,
    drift_per_epoch,
    light_dark_stability,
    sliding_window_stability,
)


def _make_light_on(n, cycle=1800):
    light_on = np.zeros(n, dtype=bool)
    for start in range(0, n, 2 * cycle):
        light_on[start:min(start + cycle, n)] = True
    return light_on


def _make_drifting_cell(n=9000, pref=90.0, kappa=3.0, drift_deg=30.0,
                        cycle=1800, noise=0.15, seed=42):
    """Cell that drifts in dark, snaps back in light."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
    light_on = _make_light_on(n, cycle)
    current_pref = pref
    drift_per_frame = drift_deg / cycle
    signal = np.zeros(n)
    for i in range(n):
        if not light_on[i]:
            current_pref += drift_per_frame
        else:
            current_pref = pref
        signal[i] = 0.1 + np.exp(kappa * np.cos(np.deg2rad(hd[i]) - np.deg2rad(current_pref)))
    signal /= signal.max()
    signal += rng.normal(0, noise, n)
    signal = np.clip(signal, 0, None)
    mask = np.ones(n, dtype=bool)
    return signal, hd, mask, light_on


class TestDriftPageWorkflow:
    """Test the full drift analysis workflow as used by the page."""

    def test_epoch_tracking_pipeline(self):
        """Full pipeline: make cell -> drift_per_epoch -> cumulative drift."""
        signal, hd, mask, light_on = _make_drifting_cell()
        result = drift_per_epoch(signal, hd, mask, light_on)
        assert result["n_epochs"] > 0
        assert len(result["cumulative_drift"]) == result["n_epochs"]
        assert result["cumulative_drift"][0] == 0.0

    def test_drift_rate_comparison(self):
        """Dark drift rate should be computable."""
        signal, hd, mask, light_on = _make_drifting_cell()
        dr = dark_drift_rate(signal, hd, mask, light_on, fps=30.0)
        assert dr["dark_drift_deg_per_s"] >= 0
        assert dr["light_drift_deg_per_s"] >= 0

    def test_light_dark_overlay(self):
        """Light/dark tuning curve overlay data should be valid."""
        signal, hd, mask, light_on = _make_drifting_cell()
        ld = light_dark_stability(signal, hd, mask, light_on)
        assert np.isfinite(ld["correlation"])
        assert not np.all(np.isnan(ld["tuning_curve_light"]))

    def test_sliding_window_provides_time_series(self):
        """Sliding window should give time series for MVL/PD plots."""
        signal, hd, mask, _ = _make_drifting_cell()
        sw = sliding_window_stability(signal, hd, mask,
                                       window_frames=1000, step_frames=200)
        assert sw["n_windows"] > 0
        assert len(sw["mvls"]) == sw["n_windows"]
        assert len(sw["preferred_dirs"]) == sw["n_windows"]

    def test_drifting_cell_has_drift(self):
        """Drifting cell should accumulate non-trivial PD drift."""
        signal_d, hd_d, mask_d, lo_d = _make_drifting_cell(drift_deg=45.0)
        result_d = drift_per_epoch(signal_d, hd_d, mask_d, lo_d)
        if result_d["n_epochs"] > 1:
            max_drift = np.max(np.abs(result_d["cumulative_drift"]))
            assert max_drift > 5  # Should show some drift


class TestDriftPageLightCycles:
    """Test light cycle shading data generation."""

    def test_light_on_mask_alternates(self):
        light_on = _make_light_on(9000, cycle=1800)
        # First 1800 frames should be light
        assert light_on[0]
        assert light_on[1799]
        # Next 1800 should be dark
        assert not light_on[1800]
        assert not light_on[3599]
        # Then light again
        assert light_on[3600]

    def test_transition_count(self):
        light_on = _make_light_on(9000, cycle=1800)
        transitions = np.diff(light_on.astype(int))
        n_transitions = np.sum(np.abs(transitions))
        assert n_transitions >= 4  # At least 2 full cycles
