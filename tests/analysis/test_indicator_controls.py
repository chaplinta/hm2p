"""Tests for hm2p.analysis.indicator_controls (synthetic traces only)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis.indicator_controls import (
    _event_peak,
    baseline_fluorescence,
    event_decay_time,
    event_fwhm,
    isolated_event_mask,
    isolated_small_event_kinetics,
)

FPS = 10.0


def _exp_event(n: int, onset: int, amp: float, tau_s: float, fps: float = FPS) -> np.ndarray:
    x = np.zeros(n)
    t = np.arange(n - onset) / fps
    x[onset:] = amp * np.exp(-t / tau_s)
    return x


def test_event_peak() -> None:
    x = np.array([0, 1, 5, 2, 0.0])
    assert _event_peak(x, 0, 4) == 2
    assert _event_peak(x, 3, 3) == 3


class TestIsolation:
    def test_isolated_and_crowded(self) -> None:
        # events at 200, 260 (crowded), 600 (isolated); n=1000, gap=100 frames
        on = [200, 260, 600]
        off = [210, 270, 610]
        out = isolated_event_mask(on, off, 1000, FPS, isolation_s=10.0)
        assert out.tolist() == [False, False, True]

    def test_edge_events_not_isolated(self) -> None:
        out = isolated_event_mask([50, 500, 950], [60, 510, 960], 1000, FPS, 10.0)
        assert out.tolist() == [False, True, False]

    def test_empty(self) -> None:
        assert isolated_event_mask([], [], 100, FPS).size == 0


class TestDecayAndWidth:
    def test_decay_recovers_tau(self) -> None:
        x = _exp_event(400, 100, 2.0, tau_s=1.5)
        d = event_decay_time(x, 100, 120, FPS)
        assert d == pytest.approx(1.5, abs=0.11)

    def test_decay_nan_when_not_reached(self) -> None:
        x = _exp_event(400, 100, 2.0, tau_s=50.0)
        assert np.isnan(event_decay_time(x, 100, 120, FPS, max_window_s=5.0))

    def test_decay_nan_for_nonpositive_peak(self) -> None:
        assert np.isnan(event_decay_time(-np.ones(50), 10, 20, FPS))

    def test_fwhm_gaussian(self) -> None:
        t = np.arange(400) / FPS
        sigma = 0.5
        x = np.exp(-((t - 20.0) ** 2) / (2 * sigma**2))
        w = event_fwhm(x, 190, 210, FPS)
        assert w == pytest.approx(2.355 * sigma, abs=0.15)

    def test_fwhm_nan_cases(self) -> None:
        assert np.isnan(event_fwhm(-np.ones(50), 10, 20, FPS))
        x = np.ones(50)  # never drops below half max
        assert np.isnan(event_fwhm(x, 10, 20, FPS))


class TestIsolatedSmallKinetics:
    def _trace(self, tau_s: float) -> tuple[np.ndarray, list[int], list[int], list[float]]:
        n = 6000
        x = np.zeros(n)
        on, off, amp = [], [], []
        for k, start in enumerate(range(300, 5600, 400)):
            a = 1.0 if k % 2 == 0 else 3.0
            x += _exp_event(n, start, a, tau_s)
            on.append(start)
            off.append(start + int(2 * tau_s * FPS))
            amp.append(a)
        return x, on, off, amp

    def test_distinguishes_slow_from_fast_indicator(self) -> None:
        fast = isolated_small_event_kinetics(*self._trace(0.8), FPS, isolation_s=5.0)
        slow = isolated_small_event_kinetics(*self._trace(2.0), FPS, isolation_s=5.0)
        assert fast["iso_n_events"] >= 3 and slow["iso_n_events"] >= 3
        assert slow["iso_decay_s"] > 2 * fast["iso_decay_s"]
        assert fast["iso_amplitude"] == pytest.approx(1.0)
        assert slow["iso_fwhm_s"] > fast["iso_fwhm_s"]

    def test_too_few_events_gives_nan(self) -> None:
        x = _exp_event(2000, 500, 1.0, 1.0)
        out = isolated_small_event_kinetics(x, [500], [520], [1.0], FPS, isolation_s=5.0)
        assert out["iso_n_events"] == 1.0 and np.isnan(out["iso_decay_s"])

    def test_no_events(self) -> None:
        out = isolated_small_event_kinetics(np.zeros(100), [], [], [], FPS)
        assert out["iso_n_events"] == 0.0 and np.isnan(out["iso_fwhm_s"])


class TestBaseline:
    def test_low_percentile(self) -> None:
        f = np.concatenate([np.full(900, 100.0), np.full(100, 500.0)])
        assert baseline_fluorescence(f) == pytest.approx(100.0)

    def test_nan_handling(self) -> None:
        assert np.isnan(baseline_fluorescence(np.array([np.nan, np.nan])))
        assert baseline_fluorescence(np.array([np.nan, 5.0, 5.0])) == 5.0

    @given(st.lists(st.floats(0, 1e4), min_size=1, max_size=50))
    @settings(max_examples=40, deadline=None)
    def test_within_range(self, vals: list[float]) -> None:
        b = baseline_fluorescence(np.array(vals))
        assert min(vals) - 1e-9 <= b <= max(vals) + 1e-9
