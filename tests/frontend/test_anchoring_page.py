"""Tests for Cue Anchoring page logic."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.anchoring import (
    anchoring_speed,
    anchoring_time_course,
    find_transitions,
)


def _make_light_on(n, cycle=1800):
    light_on = np.zeros(n, dtype=bool)
    for start in range(0, n, 2 * cycle):
        light_on[start:min(start + cycle, n)] = True
    return light_on


def _make_anchored_cell(n=9000, pref=90.0, kappa=3.0, drift=30.0,
                        cycle=1800, noise=0.15, seed=42):
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
    light_on = _make_light_on(n, cycle)
    current_pref = pref
    drift_per_frame = drift / cycle
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


class TestAnchoringPageWorkflow:
    """Test the full anchoring analysis as used by the page."""

    def test_time_course_and_speed(self):
        """Full pipeline: time_course -> anchoring_speed."""
        signal, hd, mask, light_on = _make_anchored_cell()
        result = anchoring_time_course(signal, hd, mask, light_on)
        assert result["n_transitions"] > 0

        speed = anchoring_speed(result["pd_deviations"], result["time_offsets_s"])
        assert "anchoring_strength" in speed
        assert "half_time_s" in speed

    def test_multi_cell_comparison(self):
        """Multiple cells with different drift rates."""
        drifts = [5.0, 30.0, 60.0]
        strengths = []
        for i, d in enumerate(drifts):
            signal, hd, mask, light_on = _make_anchored_cell(
                drift=d, seed=i + 42,
            )
            result = anchoring_time_course(signal, hd, mask, light_on)
            if result["n_transitions"] > 0:
                sp = anchoring_speed(result["pd_deviations"], result["time_offsets_s"])
                strengths.append(sp["anchoring_strength"])
        assert len(strengths) >= 2

    def test_transition_finding(self):
        """Transitions should be correctly identified."""
        light_on = _make_light_on(9000)
        trans = find_transitions(light_on)
        # Each dark→light transition should have light=False before and True at
        for idx in trans["dark_to_light"]:
            assert not light_on[idx - 1]
            assert light_on[idx]

    def test_pre_post_time_span(self):
        """Time offsets should span before and after transition."""
        signal, hd, mask, light_on = _make_anchored_cell()
        result = anchoring_time_course(
            signal, hd, mask, light_on,
            pre_transition_s=15.0, post_transition_s=30.0,
        )
        if result["n_transitions"] > 0:
            assert np.min(result["time_offsets_s"]) < 0
            assert np.max(result["time_offsets_s"]) > 0
