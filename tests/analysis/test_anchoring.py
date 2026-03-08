"""Tests for hm2p.analysis.anchoring — cue anchoring analysis."""

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
    """Cell that drifts in dark but re-anchors in light."""
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
            # Snap back to anchored PD
            current_pref = pref
        signal[i] = 0.1 + np.exp(kappa * np.cos(np.deg2rad(hd[i]) - np.deg2rad(current_pref)))

    signal /= signal.max()
    signal += rng.normal(0, noise, n)
    signal = np.clip(signal, 0, None)
    mask = np.ones(n, dtype=bool)
    return signal, hd, mask, light_on


class TestFindTransitions:
    """Tests for find_transitions."""

    def test_alternating_cycles(self):
        light_on = _make_light_on(9000, cycle=1800)
        result = find_transitions(light_on)
        assert len(result["dark_to_light"]) > 0
        assert len(result["light_to_dark"]) > 0

    def test_all_light(self):
        light_on = np.ones(1000, dtype=bool)
        result = find_transitions(light_on)
        assert len(result["dark_to_light"]) == 0
        assert len(result["light_to_dark"]) == 0

    def test_single_transition(self):
        light_on = np.zeros(1000, dtype=bool)
        light_on[500:] = True
        result = find_transitions(light_on)
        assert len(result["dark_to_light"]) == 1
        assert result["dark_to_light"][0] == 500

    def test_transition_indices_valid(self):
        light_on = _make_light_on(9000)
        result = find_transitions(light_on)
        for idx in result["dark_to_light"]:
            assert 0 < idx < 9000
            assert not light_on[idx - 1]
            assert light_on[idx]


class TestAnchoringTimeCourse:
    """Tests for anchoring_time_course."""

    def test_output_keys(self):
        signal, hd, mask, light_on = _make_anchored_cell()
        result = anchoring_time_course(signal, hd, mask, light_on)
        expected = {"time_offsets_s", "pd_deviations", "mvls",
                    "reference_pd", "n_transitions"}
        assert set(result.keys()) == expected

    def test_transitions_found(self):
        signal, hd, mask, light_on = _make_anchored_cell()
        result = anchoring_time_course(signal, hd, mask, light_on)
        assert result["n_transitions"] > 0

    def test_time_offsets_span_zero(self):
        signal, hd, mask, light_on = _make_anchored_cell()
        result = anchoring_time_course(signal, hd, mask, light_on)
        assert np.any(result["time_offsets_s"] < 0)
        assert np.any(result["time_offsets_s"] > 0)

    def test_reference_pd_in_range(self):
        signal, hd, mask, light_on = _make_anchored_cell()
        result = anchoring_time_course(signal, hd, mask, light_on)
        assert 0 <= result["reference_pd"] < 360

    def test_no_transitions(self):
        """All light → no transitions → empty result."""
        rng = np.random.default_rng(42)
        n = 3000
        hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
        signal = np.abs(rng.normal(1, 0.5, n))
        mask = np.ones(n, dtype=bool)
        light_on = np.ones(n, dtype=bool)
        result = anchoring_time_course(signal, hd, mask, light_on)
        assert result["n_transitions"] == 0

    def test_custom_reference_pd(self):
        signal, hd, mask, light_on = _make_anchored_cell()
        result = anchoring_time_course(
            signal, hd, mask, light_on, reference_pd=45.0,
        )
        assert result["reference_pd"] == 45.0


class TestAnchoringSpeed:
    """Tests for anchoring_speed."""

    def test_output_keys(self):
        time_offsets = np.linspace(-10, 30, 40)
        deviations = np.concatenate([
            np.ones(10) * 30,  # pre
            np.linspace(30, 5, 30),  # post: decaying
        ])
        result = anchoring_speed(deviations, time_offsets)
        expected = {"pre_deviation", "post_deviation", "half_time_s", "anchoring_strength"}
        assert set(result.keys()) == expected

    def test_re_anchoring_detected(self):
        """Deviation that decreases after transition should show re-anchoring."""
        time_offsets = np.linspace(-10, 30, 40)
        deviations = np.concatenate([
            np.ones(10) * 30,  # pre
            np.linspace(30, 2, 30),  # post: decaying
        ])
        result = anchoring_speed(deviations, time_offsets)
        assert result["pre_deviation"] > result["post_deviation"]
        assert result["anchoring_strength"] > 0.5

    def test_no_anchoring(self):
        """Constant deviation → no anchoring."""
        time_offsets = np.linspace(-10, 30, 40)
        deviations = np.ones(40) * 30
        result = anchoring_speed(deviations, time_offsets)
        assert abs(result["anchoring_strength"]) < 0.1

    def test_all_nan(self):
        time_offsets = np.linspace(-10, 30, 10)
        deviations = np.full(10, np.nan)
        result = anchoring_speed(deviations, time_offsets)
        assert np.isnan(result["anchoring_strength"])
