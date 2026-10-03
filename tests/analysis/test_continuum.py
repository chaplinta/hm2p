"""Tests for ``hm2p.analysis.continuum`` (synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis import continuum as c


def test_count_modes_unimodal_and_bimodal() -> None:
    rng = np.random.default_rng(0)
    uni = rng.normal(0, 1, 300)
    bi = np.concatenate([rng.normal(-4, 0.5, 150), rng.normal(4, 0.5, 150)])
    assert c.count_modes(uni, 0.8) == 1
    assert c.count_modes(bi, 0.5) == 2
    assert c.count_modes([], 1.0) == 0


def test_critical_bandwidth_monotone_in_k() -> None:
    rng = np.random.default_rng(1)
    bi = np.concatenate([rng.normal(-3, 0.5, 100), rng.normal(3, 0.5, 100)])
    h1, h2 = c.critical_bandwidth(bi, 1), c.critical_bandwidth(bi, 2)
    assert h1 > h2 > 0
    assert c.count_modes(bi, h1 * 1.01) <= 1 and c.count_modes(bi, h1 * 0.9) > 1
    assert c.critical_bandwidth([1.0, 1.0]) == 0.0


def test_silverman_detects_bimodality_not_unimodal() -> None:
    rng = np.random.default_rng(2)
    bi = np.concatenate([rng.normal(-3, 0.6, 120), rng.normal(3, 0.6, 120)])
    uni = rng.normal(0, 1, 240)
    assert c.silverman_mode_test(bi, n_boot=200)["p"] < 0.05
    assert c.silverman_mode_test(uni, n_boot=200)["p"] > 0.1
    assert np.isnan(c.silverman_mode_test([1.0, 2.0])["p"])


def test_shift_function_uniform_shift() -> None:
    rng = np.random.default_rng(3)
    b = rng.normal(0, 1, 400)
    a = rng.normal(1, 1, 400)
    sf = c.shift_function(a, b, n_boot=200)
    assert np.allclose(sf["diff"], 1.0, atol=0.35)
    assert all(lo <= d <= hi for d, lo, hi in zip(sf["diff"], sf["lo"], sf["hi"], strict=True))
    assert all(lo > 0 for lo in sf["lo"])


def test_shift_function_tail_effect_and_groups() -> None:
    rng = np.random.default_rng(4)
    b = rng.normal(0, 1, 500)
    a = np.concatenate([rng.normal(0, 1, 400), rng.normal(4, 0.3, 100)])  # extra upper tail
    ga = np.repeat(np.arange(5), 100)
    gb = np.repeat(np.arange(5), 100)
    sf = c.shift_function(a, b, ga, gb, n_boot=100)
    assert sf["diff"][-1] > 1.5 and abs(sf["diff"][0]) < 0.4
    empty = c.shift_function([], b)
    assert all(np.isnan(empty["diff"]))


def test_rank_axis() -> None:
    r = c.rank_axis([[1, 2, 3], [3, 2, np.nan]], [1, -1])
    # col1 ranks 0, .5, 1 ; col2 negated: -3,-2 -> ranks 0,1 for first two
    assert r.tolist() == pytest.approx([0.0, 0.75, 1.0])
    assert np.isnan(c.rank_axis([[np.nan, 1.0]], [1])[0])
    with pytest.raises(ValueError):
        c.rank_axis([[1, 2]], [1, 1])


@settings(max_examples=25, deadline=None)
@given(st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=2, max_size=30))
def test_rank_axis_bounds(xs: list[float]) -> None:
    r = c.rank_axis([xs], [1])
    assert np.all((r >= 0) & (r <= 1))


def _two_types(separated: bool, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, lab, ani = [], [], []
    for a in range(8):
        t = "p" if a < 5 else "n"
        centre = (3.0 if t == "p" else -3.0) if separated else 0.0
        X.append(rng.normal(centre, 1, (15, 2)))
        lab += [t] * 15
        ani += [a] * 15
    return np.vstack(X), np.array(lab), np.array(ani)


def test_knn_label_mixing_separated_vs_mixed() -> None:
    X, lab, ani = _two_types(True)
    sep = c.knn_label_mixing(X, lab, ani, k=5)
    assert sep["n_assignments"] == 56
    assert sep["labels"]["p"]["observed"] > 0.95 and sep["labels"]["p"]["p_one_sided"] < 0.05
    X, lab, ani = _two_types(False, seed=1)
    mixed = c.knn_label_mixing(X, lab, ani, k=5)
    assert mixed["labels"]["n"]["p_one_sided"] > 0.05


def test_knn_label_mixing_subsamples_and_rejects_three_labels() -> None:
    X, lab, ani = _two_types(True)
    out = c.knn_label_mixing(X, lab, ani, k=5, max_perms=10, exclude_same_animal=False)
    assert out["n_assignments"] == 10
    with pytest.raises(ValueError):
        c.knn_label_mixing(X, np.where(ani == 0, "x", lab), ani)
