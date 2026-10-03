"""Tests for ``scripts/make_penk_cell_level.py`` (synthetic tables only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import make_penk_cell_level as cl  # noqa: E402


def _table(tmp_path: Path, effect: float = 1.0, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(6):
        ct = "penk" if a < 4 else "nonpenk"
        for r in range(10):
            v = (effect if ct == "penk" else 0.0) + rng.normal(0, 0.3)
            for ch in ("speed_cm_s", "active"):
                rows.append(
                    {
                        "animal_id": a,
                        "exp_id": f"s{a}",
                        "roi_idx": r,
                        "celltype": ct,
                        "channel": ch,
                        "z": v,
                        "p": 0.01 if v > 0.5 else 0.5,
                    }
                )
    rows.append(
        {
            "animal_id": 9,
            "exp_id": "x",
            "roi_idx": 0,
            "celltype": "other",
            "channel": "active",
            "z": 0.0,
            "p": 0.5,
        }
    )
    path = tmp_path / "t" / "cells.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _m(**kw) -> cl.Measure:
    base = dict(
        key="k",
        section="running",
        label="L",
        path="t/cells.csv",
        column="z",
        signal="s",
        centred=True,
        row_filter=("channel", "active"),
        sig_column="p",
        sig_sign=1,
    )
    base.update(kw)
    return cl.Measure(**base)


def test_load_measure_filters_and_sig(tmp_path: Path) -> None:
    _table(tmp_path)
    df = cl.load_measure(_m(), tmp_path)
    assert df is not None and len(df) == 60 and set(df.celltype) == {"penk", "nonpenk"}
    assert (
        df.loc[df.celltype == "penk", "sig"].mean()
        > df.loc[df.celltype == "nonpenk", "sig"].mean()
    )
    neg = cl.load_measure(_m(sig_sign=-1), tmp_path)
    assert neg is not None and neg["sig"].sum() == 0
    assert cl.load_measure(_m(column="zz"), tmp_path) is None
    assert cl.load_measure(_m(path="none.csv"), tmp_path) is None


def test_cliffs_delta() -> None:
    assert cl.cliffs_delta(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert cl.cliffs_delta(np.array([1.0]), np.array([1.0])) == 0.0
    assert np.isnan(cl.cliffs_delta(np.array([]), np.array([1.0])))


def test_within_test_and_quartiles() -> None:
    assert cl.within_test(np.arange(1.0, 11.0)) < 0.01
    assert np.isnan(cl.within_test(np.array([0.0, 1.0])))
    q = cl.quartiles(np.arange(5.0))
    assert q["median"] == 2.0 and q["q25"] == 1.0


def test_fraction_test() -> None:
    r = cl.fraction_test(np.array([1, 1, 1, 0]), np.array([0, 0, 0, np.nan]))
    assert r["frac_penk"] == 0.75 and r["n_nonpenk"] == 3 and 0 < r["fisher_p"] <= 1
    assert cl.fraction_test(np.array([]), np.array([1.0])) == {}


def test_bh_fdr_matches_reference() -> None:
    p = [0.01, 0.04, 0.03, np.nan, 0.5]
    adj = cl.bh_fdr(p)
    assert adj[0] == pytest.approx(0.04) and adj[2] == pytest.approx(0.04 * 4 / 3)
    assert adj[1] == pytest.approx(0.04 * 4 / 3) and np.isnan(adj[3]) and adj[4] == 0.5
    assert all(np.isnan(cl.bh_fdr([np.nan])))


def test_build_detects_effect(tmp_path: Path) -> None:
    _table(tmp_path)
    out = cl.build(tmp_path, (_m(), _m(key="k2", row_filter=("channel", "speed_cm_s"))))
    assert [r["key"] for r in out["measures"]] == ["k", "k2"]
    r = out["measures"][0]
    assert r["mwu_p"] < 0.001 and r["cliffs_delta"] > 0.8 and r["mwu_fdr"] is not None
    assert r["n_animals_penk"] == 4 and r["within_p"]["penk"] < 0.01
    assert len(r["values"]["penk"]) == 40 and r["fraction_sig"]["frac_penk"] > 0.5
    assert out["running_scatter"] is None


def test_running_scatter(tmp_path: Path) -> None:
    p = tmp_path / "celltype_programme_runshape/runshape"
    p.mkdir(parents=True)
    pd.DataFrame(
        {
            "step_index": [0.5, 0.1, np.nan],
            "graded_rho": [0.0, 0.1, 0.2],
            "step_index_p": [0.01, 0.5, 0.01],
            "celltype": ["penk", "nonpenk", "penk"],
            "animal_id": [1, 2, 1],
        }
    ).to_csv(p / "cells.csv", index=False)
    s = cl._running_scatter(tmp_path)
    assert s["penk"]["step_sig"] == [True] and s["nonpenk"]["step_index"] == [0.1]


def test_clean() -> None:
    assert cl._clean({"a": [float("nan"), np.float64(1.0)]}) == {"a": [None, 1.0]}
