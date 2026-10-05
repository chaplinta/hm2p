"""Tests for ``hm2p.analysis.behaviour_mi`` (synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis import behaviour_mi as bm


def test_quantile_bins() -> None:
    b = bm.quantile_bins(np.arange(8.0), 4)
    assert b.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert bm.quantile_bins([np.nan, 1.0], 2).tolist()[0] == -1
    assert (bm.quantile_bins([np.nan, np.nan]) == -1).all()
    sparse = bm.quantile_bins(np.r_[np.zeros(90), np.arange(1, 11.0)], 4)
    assert sparse.max() <= 3 and (sparse[:90] == sparse[0]).all()


def test_plugin_mi_known_values() -> None:
    x = np.array([0, 1] * 50)
    assert bm.plugin_mi(x, x) == pytest.approx(1.0)
    assert bm.plugin_mi(x, np.zeros(100, int)) == pytest.approx(0.0)
    assert bm.plugin_mi([0], [0]) == 0.0
    assert bm.plugin_mi([0, 1, -1], [0, 1, 1]) == pytest.approx(1.0)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 10_000))
def test_plugin_mi_bounds(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 4, 200), rng.integers(0, 3, 200)
    mi = bm.plugin_mi(a, b)
    assert 0.0 <= mi <= np.log2(3) + 1e-9


def test_conditional_mi_removes_shared_variable() -> None:
    rng = np.random.default_rng(0)
    cond = rng.integers(0, 4, 4000)
    label = (cond >= 2).astype(int)  # label fully determined by condition
    act = cond.copy()  # activity tracks condition only
    assert bm.plugin_mi(act, label) > 0.9
    assert bm.conditional_mi(act, label, cond) == pytest.approx(0.0, abs=1e-9)
    assert bm.conditional_mi(act, label, -np.ones(4000, int)) == 0.0


def test_cell_label_mi_detects_label_coding() -> None:
    rng = np.random.default_rng(1)
    n = 20_000
    lab = np.where((np.arange(n) // 200) % 2 == 0, 0, 1)
    lab[::7] = -1
    sig = rng.poisson(0.5, (2, n)).astype(float)
    sig[0, lab == 1] += 3.0
    shifts = rng.integers(1000, n - 1000, 40)
    t = bm.cell_label_mi(sig, lab, shifts)
    assert t.loc[0, "mi_debiased"] > 0.2 and t.loc[0, "p"] < 0.05
    assert abs(t.loc[1, "mi_debiased"]) < 0.02
    tc = bm.cell_label_mi(sig, lab, shifts, cond=np.zeros(n, int))
    assert tc.loc[0, "mi"] == pytest.approx(t.loc[0, "mi"])
    t0 = bm.cell_label_mi(sig, lab, np.array([], int))
    assert np.isnan(t0.p).all()


def test_frames_from_windows_and_hd_sectors() -> None:
    f = bm.frames_from_windows(10, [1, 6], [3, 12], [1, 0])
    assert f.tolist() == [-1, 1, 1, -1, -1, -1, 0, 0, 0, 0]
    assert bm.hd_sectors([0, 44, 46, 359, np.nan, -10], 8).tolist() == [0, 0, 1, 7, -1, 7]
