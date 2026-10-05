"""Tests for ``hm2p.analysis.inout_auroc`` (synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis import inout_auroc as ia


def test_auroc_basic_cases() -> None:
    assert ia.auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert ia.auroc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert ia.auroc([1, 1, 1, 1], [0, 0, 1, 1]) == 0.5
    assert np.isnan(ia.auroc([1, 2], [1, 1]))
    assert ia.auroc([1, np.nan, 3, 4], [0, 0, 1, 1]) == 1.0


@settings(max_examples=30, deadline=None)
@given(
    st.lists(st.floats(-100, 100, allow_nan=False), min_size=4, max_size=40),
    st.integers(0, 10_000),
)
def test_auroc_symmetry(xs: list[float], seed: int) -> None:
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, 2, len(xs))
    if lab.min() == lab.max():
        lab[0] = 1 - lab[0]
    a, b = ia.auroc(xs, lab), ia.auroc(xs, 1 - lab)
    assert 0.0 <= a <= 1.0 and a + b == pytest.approx(1.0)


def test_remove_hd_removes_hd_component() -> None:
    rng = np.random.default_rng(0)
    ang = rng.uniform(0, 2 * np.pi, 200)
    act = np.column_stack([3 * np.sin(ang) + rng.normal(0, 0.01, 200), rng.normal(size=200)])
    res = ia.remove_hd(act, np.sin(ang), np.cos(ang))
    assert np.std(res[:, 0]) < 0.05 and abs(np.corrcoef(res[:, 1], act[:, 1])[0, 1]) > 0.95


def _runs(n_frames: int = 20_000, n_runs: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.choice(np.arange(0, n_frames - 30, 40), n_runs, replace=False))
    ends = starts + 15
    labels = rng.integers(0, 2, n_runs)
    sig = rng.poisson(0.3, (3, n_frames)).astype(float)
    hd = rng.uniform(0, 360, n_frames)
    for s, e, lab in zip(starts, ends, labels, strict=True):
        if lab == 1:
            sig[0, s:e] += 2.0  # cell 0 prefers inbound
        else:
            sig[1, s:e] += 2.0  # cell 1 prefers outbound
        hd[s:e] = 90.0 if lab == 1 else 270.0  # heading confound for the HD test
    sig[2] += 2.0 * np.sin(np.deg2rad(hd))  # cell 2 is a pure HD cell
    return sig, starts, ends, labels, hd


def test_cell_inout_auroc_detects_both_directions_and_hd() -> None:
    sig, st_, en, lab, hd = _runs()
    shifts = np.random.default_rng(1).integers(500, 19_000, 80)
    t, null = ia.cell_inout_auroc(sig, st_, en, lab, hd, shifts, n_null_keep=10)
    assert t.loc[0, "auc"] > 0.9 and t.loc[0, "auc_p"] < 0.05
    assert t.loc[1, "auc"] < 0.1 and t.loc[1, "auc_p"] < 0.05
    # the HD cell separates the classes raw (heading confound) but not after HD removal
    assert t.loc[2, "auc"] > 0.9 and abs(t.loc[2, "auc_hd"] - 0.5) < 0.2
    assert null["raw"].size == 30 and np.all((null["raw"] >= 0) & (null["raw"] <= 1))


def test_cell_inout_auroc_without_shifts() -> None:
    sig, st_, en, lab, hd = _runs(n_runs=40)
    t, null = ia.cell_inout_auroc(sig, st_, en, lab, hd, np.array([], dtype=int))
    assert np.isnan(t.auc_p).all() and null["raw"].size == 0
