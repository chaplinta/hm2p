"""Tests for hm2p.analysis.location_running (synthetic data)."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.location_running import (
    CORRIDOR,
    DEAD_END,
    JUNCTION,
    _index,
    contrast_frames,
    contrast_significance,
    contrast_statistics,
    location_labels,
)

FPS = 10.0


def _behaviour(n: int = 6000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Blocks of 50 frames alternating corridor/junction; speed higher in corridors."""
    rng = np.random.default_rng(seed)
    # irregular block lengths so circular shifts cannot re-align the pattern
    lengths = rng.integers(20, 120, size=n)
    labels = np.repeat(np.arange(lengths.size) % 2, lengths)[:n]
    loc = np.where(labels == 0, CORRIDOR, JUNCTION)
    speed = np.where(loc == CORRIDOR, rng.uniform(0, 15, n), rng.uniform(0, 8, n))
    return speed, loc


def test_location_labels() -> None:
    ch = {
        "node_corridor": np.array([1.0, 0, 0, np.nan]),
        "node_junction": np.array([0.0, 1, 0, np.nan]),
        "node_dead_end": np.array([0.0, 0, 1, np.nan]),
    }
    assert location_labels(ch).tolist() == [CORRIDOR, JUNCTION, DEAD_END, -1]


def test_index() -> None:
    assert _index(3.0, 1.0) == 0.5
    assert np.isnan(_index(0.0, 0.0)) and np.isnan(_index(np.nan, 1.0))


def test_frames_are_speed_matched() -> None:
    speed, loc = _behaviour()
    fr = contrast_frames(speed, loc, np.ones(speed.size, bool), run_threshold=2.5)
    a, b = fr["corr_run_matched"], fr["junc_run_matched"]
    assert a.size > 100 and b.size > 100
    assert abs(np.median(speed[a]) - np.median(speed[b])) < 0.6
    assert np.all(loc[fr["corr_still"]] == CORRIDOR) and np.all(speed[fr["corr_still"]] < 2.5)


def test_frames_too_few_are_empty() -> None:
    speed, loc = _behaviour(n=200)
    fr = contrast_frames(speed, loc, np.ones(200, bool), min_frames=500)
    assert all(v.size == 0 for v in fr.values())


def test_running_cell_is_not_a_location_cell() -> None:
    speed, loc = _behaviour()
    signal = speed / 10.0  # driven by speed only
    fr = contrast_frames(speed, loc, np.ones(speed.size, bool))
    res = contrast_significance(signal, fr, FPS, n_shuffles=100, rng=np.random.default_rng(1))
    assert res["run_index_corr"] > 0.5 and res["run_index_corr_p"] < 0.05
    assert abs(res["loc_index"]) < 0.1


def test_location_cell_detected_at_matched_speed() -> None:
    speed, loc = _behaviour()
    rng = np.random.default_rng(2)
    signal = np.where(loc == CORRIDOR, 1.0, 0.2) + rng.normal(0, 0.05, speed.size)
    fr = contrast_frames(speed, loc, np.ones(speed.size, bool))
    res = contrast_significance(signal, fr, FPS, n_shuffles=100, rng=np.random.default_rng(3))
    assert res["loc_index"] > 0.5 and res["loc_index_p"] < 0.05
    assert abs(res["run_index_corr"]) < 0.05


def test_noise_cell_not_significant() -> None:
    speed, loc = _behaviour()
    signal = np.random.default_rng(4).random(speed.size)
    fr = contrast_frames(speed, loc, np.ones(speed.size, bool))
    res = contrast_significance(signal, fr, FPS, n_shuffles=100, rng=np.random.default_rng(5))
    assert res["loc_index_p"] > 0.05 and res["run_index_corr_p"] > 0.05


def test_statistics_with_empty_sets() -> None:
    empty = np.zeros(0, dtype=np.int64)
    fr = {
        k: empty
        for k in (
            "corr_run",
            "corr_still",
            "junc_run",
            "junc_still",
            "corr_run_matched",
            "junc_run_matched",
        )
    }
    st = contrast_statistics(np.ones(10), fr)
    assert np.isnan(st["loc_index"]) and np.isnan(st["rate_corr_run_matched"])
    res = contrast_significance(np.ones(1000), fr, FPS, n_shuffles=20)
    assert np.isnan(res["loc_index_p"])


def test_short_recording_gives_nan() -> None:
    speed, loc = _behaviour(n=150)
    fr = contrast_frames(speed, loc, np.ones(150, bool), min_frames=5)
    res = contrast_significance(speed, fr, FPS, n_shuffles=10, min_shift_s=10.0)
    assert np.isnan(res["run_index_corr_z"])
    assert res["run_index_corr"] == pytest.approx(contrast_statistics(speed, fr)["run_index_corr"])
