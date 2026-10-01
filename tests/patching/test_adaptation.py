"""Tests for hm2p.patching.adaptation (synthetic voltage traces)."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.patching.adaptation import (
    _early_late_rate_ratio,
    cell_adaptation,
    sweep_adaptation,
    sweep_spike_times,
)

SR = 20000.0


def _trace(spike_times_s: list[float], dur_s: float = 1.0) -> np.ndarray:
    v = np.full(int(dur_s * SR), -70.0)
    for t in spike_times_s:
        i = int(t * SR)
        v[i : i + 20] = 30.0  # 1 ms spike
    return v


def _regular(n: int, isi: float, start: float = 0.05) -> list[float]:
    return [start + k * isi for k in range(n)]


def _adapting(n: int, first_isi: float, growth: float, start: float = 0.05) -> list[float]:
    t, isi, out = start, first_isi, []
    for _ in range(n):
        out.append(t)
        t += isi
        isi *= growth
    return out


def test_sweep_spike_times() -> None:
    times = _regular(5, 0.1)
    got = sweep_spike_times(_trace(times), SR)
    assert got.size == 5
    np.testing.assert_allclose(got, times, atol=2e-3)
    assert sweep_spike_times(np.full(1000, -70.0), SR).size == 0
    assert sweep_spike_times(np.array([]), SR).size == 0


def test_regular_vs_adapting() -> None:
    reg = sweep_adaptation(np.array(_regular(10, 0.05)))
    ada = sweep_adaptation(np.array(_adapting(10, 0.02, 1.3)))
    assert abs(reg["adaptation_index"]) < 1e-9 and reg["sfa_ratio"] == pytest.approx(1.0)
    assert ada["adaptation_index"] > 0.1 and ada["sfa_ratio"] > 5
    assert ada["early_late_rate_ratio"] > reg["early_late_rate_ratio"]
    assert reg["isi_cv"] < 1e-9 and reg["first_isi_ms"] == pytest.approx(50.0)


def test_sweep_adaptation_few_spikes() -> None:
    out = sweep_adaptation(np.array([0.1, 0.2]))
    assert np.isnan(out["adaptation_index"]) and out["first_isi_ms"] == pytest.approx(100.0)
    assert np.isnan(sweep_adaptation(np.zeros(0))["isi_cv"])


def test_early_late_ratio_edge_cases() -> None:
    assert np.isnan(_early_late_rate_ratio(np.array([0.1, 0.2, 0.3])))
    assert np.isnan(_early_late_rate_ratio(np.full(8, 0.5)))


def test_cell_adaptation_selects_sweeps() -> None:
    stim = np.array([0.0, 50.0, 100.0, 150.0])
    trains = [[], _regular(3, 0.1), _regular(7, 0.08), _adapting(12, 0.02, 1.2)]
    traces = np.column_stack([_trace(t) for t in trains])
    out = cell_adaptation(traces, stim, SR, target_spikes=6)
    assert out["max_n_spikes"] == 12 and out["max_adaptation_index"] > 0.05
    assert out["matched_current"] == 100.0 and out["matched_n_spikes"] == 7
    assert out["fi_slope_spikes_per_pa"] > 0


def test_cell_adaptation_no_spikes_and_bad_shape() -> None:
    traces = np.column_stack([_trace([]) for _ in range(3)])
    out = cell_adaptation(traces, np.array([0.0, 10.0, 20.0]), SR)
    assert np.isnan(out["matched_current"]) and np.isnan(out["max_adaptation_index"])
    assert np.isnan(out["fi_slope_spikes_per_pa"])
    with pytest.raises(ValueError):
        cell_adaptation(traces, np.array([0.0]), SR)
