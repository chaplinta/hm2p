"""Tests for hm2p.analysis.running_coding (synthetic data)."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.running_coding import (
    bout_duration_scaling,
    bout_time_course,
    run_bouts,
    speed_bin_means,
    speed_shape,
    speed_shape_significance,
)

FPS = 10.0


def _speed(n: int = 8000, seed: int = 0) -> np.ndarray:
    """Irregular bouts of running (5-25 cm/s) and stillness."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(20, 150, size=n)
    state = np.repeat(np.arange(lengths.size) % 2, lengths)[:n]
    level = np.repeat(rng.uniform(5, 25, size=lengths.size), lengths)[:n]
    return np.where(state == 1, level + rng.normal(0, 1, n), rng.uniform(0, 1.5, n))


def test_speed_bin_means_and_counts() -> None:
    s = np.array([0.5] * 40 + [3.0] * 40 + [40.0] * 40)
    x = np.array([1.0] * 40 + [2.0] * 40 + [5.0] * 40)
    means, centres, counts = speed_bin_means(x, s, np.ones(120, bool), min_frames=30)
    assert means[0] == 1.0 and means[1] == 2.0 and means[-1] == 5.0
    assert counts[0] == 40 and np.isnan(means[2])
    assert centres[-1] > 30.0


def test_speed_shape_step_vs_graded() -> None:
    centres = np.array([1.0, 4.0, 7.5, 12.5, 17.5, 25.0, 35.0])
    step = np.array([0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    graded = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    s1, s2 = speed_shape(step, centres), speed_shape(graded, centres)
    assert s1["step_index"] > 0.7 and np.isnan(s1["graded_rho"])
    assert s2["graded_rho"] == pytest.approx(1.0)
    assert np.isnan(speed_shape(np.full(7, np.nan), centres)["step_index"])


def test_significance_detects_graded_speed_cell() -> None:
    s = _speed()
    x = s / 20 + np.random.default_rng(1).normal(0, 0.05, s.size)
    res = speed_shape_significance(x, s, np.ones(s.size, bool), FPS, n_shuffles=60)
    assert res["graded_rho"] > 0.8 and res["graded_rho_p"] < 0.05
    assert res["step_index"] > 0 and res["step_index_p"] < 0.05
    assert len(res["speed_curve"]) == 7


def test_significance_noise_cell() -> None:
    s = _speed()
    x = np.random.default_rng(2).random(s.size)
    res = speed_shape_significance(x, s, np.ones(s.size, bool), FPS, n_shuffles=60)
    assert res["step_index_p"] > 0.05


def test_significance_short_recording() -> None:
    res = speed_shape_significance(np.ones(100), np.ones(100) * 5, np.ones(100, bool), FPS)
    assert np.isnan(res["step_index_p"])


def test_run_bouts() -> None:
    s = np.array([0.0] * 10 + [5.0] * 30 + [0.0] * 10 + [5.0] * 5 + [0.0] * 10)
    on, off = run_bouts(s, np.ones(s.size, bool), FPS, min_s=1.0)
    assert on.tolist() == [10] and off.tolist() == [40]


def test_bout_time_course_front_loaded_vs_sustained() -> None:
    on = np.arange(0, 2000, 100)
    off = on + 50
    front = np.zeros(2000)
    sust = np.zeros(2000)
    for a, b in zip(on, off, strict=True):
        front[a : a + 10] = 1.0
        sust[a:b] = 1.0
    f = bout_time_course(front, on, off, FPS)
    s = bout_time_course(sust, on, off, FPS)
    assert f["bout_adaptation_index"] == pytest.approx(1.0)
    assert s["bout_adaptation_index"] == pytest.approx(0.0)
    assert f["n_long_bouts"] == 20
    assert np.isnan(bout_time_course(sust, on[:2], off[:2], FPS)["bout_early"])


def test_bout_duration_scaling() -> None:
    rng = np.random.default_rng(3)
    on = np.cumsum(rng.integers(60, 120, size=20))
    dur = rng.integers(10, 50, size=20)
    off = on + dur
    x = np.zeros(int(off.max()) + 10)
    for a, b in zip(on, off, strict=True):
        x[a:b] = 1.0
    res = bout_duration_scaling(x, on, off)
    assert res["bout_dur_rho_sum"] > 0.95 and np.isnan(res["bout_dur_rho_mean"])
    assert bout_duration_scaling(x, on[:3], off[:3])["n_bouts"] == 3
