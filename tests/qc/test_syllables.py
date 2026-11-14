"""Tests for hm2p.qc.syllables (synthetic label sequences only)."""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from hm2p.qc.syllables import (
    bout_transitions,
    criteria_check,
    n_to_cover,
    overview_row,
    summarise_syllables,
    usage_entropy_ratio,
)


def test_entropy_ratio_uniform_and_degenerate():
    assert usage_entropy_ratio([10, 10, 10, 10]) == pytest.approx(1.0)
    assert usage_entropy_ratio([100, 1]) < 0.2
    assert np.isnan(usage_entropy_ratio([5]))


def test_n_to_cover():
    assert n_to_cover([80, 10, 10], 0.8) == 1
    assert n_to_cover([25, 25, 25, 25], 0.8) == 4
    assert n_to_cover([0, 0], 0.8) == 0


def test_bout_transitions_no_diagonal():
    t = bout_transitions([1, 2, 1, 2, 3])
    assert [1, 2, 2] in t and all(a != b for a, b, _ in t)
    assert bout_transitions([1]) == []


def _sequence(n_syll: int = 100, bout: int = 12, reps: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = 1.0 / (np.arange(n_syll) + 1.0) ** 1.1  # skewed usage, as real syllables are
    ids = rng.choice(n_syll, reps * n_syll, p=w / w.sum())
    ids = ids[np.r_[True, ids[1:] != ids[:-1]]]  # no repeated consecutive ids
    return np.repeat(ids, bout)


def test_summary_well_formed_sequence_passes_criteria():
    sid = _sequence()
    n = sid.size
    s = summarise_syllables(
        sid, fps=30.0, speed_cm_s=np.ones(n), ahv_deg_s=np.ones(n), light_on=np.arange(n) % 2 == 0
    )
    json.dumps(s)
    assert s["median_bout_ms"] == pytest.approx(400.0)
    assert s["single_frame_frac"] == 0.0
    assert all(s["criteria"].values())
    k = next(iter(s["per_syllable"]))
    assert s["per_syllable"][k]["speed_med"] == 1.0
    assert s["per_syllable"][k]["frac_light"] is not None
    lens = np.frombuffer(base64.b64decode(s["ethogram"]["lens"]), dtype="<u2")
    assert lens.sum() == n
    assert overview_row(s)["n_pass"] == 4


def test_summary_degenerate_sequence_fails_criteria():
    sid = np.r_[np.zeros(900, int), np.tile([1, 2], 50)]
    s = summarise_syllables(sid, fps=30.0, speed_cm_s=np.ones(5))  # misaligned speed ignored
    assert s["criteria"]["single_frame_frac"] is False
    assert s["criteria"]["n_syll_80pct"] is False
    assert "speed_med" not in s["per_syllable"]["0"]


def test_unassigned_and_empty():
    s = summarise_syllables([-1, -1, 0, 0, 0], fps=30.0)
    assert s["frac_unassigned"] == pytest.approx(0.4)
    assert s["n_bouts"] == 1
    with pytest.raises(ValueError):
        summarise_syllables([], fps=30.0)


def test_criteria_check_none_values():
    c = criteria_check(
        {
            "median_bout_ms": None,
            "n_syll_80pct": 25,
            "entropy_ratio": 0.7,
            "single_frame_frac": None,
        }
    )
    assert c["median_bout_ms"] is None and c["single_frame_frac"] is None and c["n_syll_80pct"]


def test_long_runs_are_split_in_ethogram():
    s = summarise_syllables(np.zeros(70000, int), fps=30.0)
    lens = np.frombuffer(base64.b64decode(s["ethogram"]["lens"]), dtype="<u2")
    assert lens.sum() == 70000 and s["ethogram"]["n_runs"] == 2
