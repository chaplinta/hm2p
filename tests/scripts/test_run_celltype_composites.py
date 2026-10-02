"""Tests for ``scripts/run_celltype_composites.py`` (synthetic tables)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import run_celltype_composites as rc  # noqa: E402


def _cells(effect: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(8):
        ct = "penk" if a < 5 else "nonpenk"
        for _ in range(4):
            rows.append(
                {
                    "animal_id": str(a),
                    "celltype": ct,
                    "m": (effect if ct == "penk" else 0.0) + rng.normal(0, 0.1),
                    "channel": "speed_cm_s" if rng.random() < 0.5 else "active",
                }
            )
    return pd.DataFrame(rows)


def test_animal_measure_sign_and_filter() -> None:
    df = _cells(1.0)
    m = rc.animal_measure(df, "m", sign=-1)
    assert len(m) == 8 and (m.loc[m.celltype == "penk", "value"] < 0).all()
    f = rc.animal_measure(df, "m", row_filter=("channel", "speed_cm_s"))
    assert len(f) <= 8


def test_composite_scores_mean_rank() -> None:
    a = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0], "celltype": ["x", "y", "y"]}, index=["a", "b", "c"]
    )
    b = pd.DataFrame({"value": [3.0, 2.0], "celltype": ["x", "y"]}, index=["a", "b"])
    t = rc.composite_scores({"A": a, "B": b})
    assert t.loc["a", "composite"] == pytest.approx(1.5)  # ranks 1 and 2
    assert t.loc["c", "n_measures"] == 1 and t.loc["a", "celltype"] == "x"


def test_exact_permutation_p() -> None:
    vals = pd.Series([5.0, 6.0, 7.0, 1.0, 2.0])
    labs = pd.Series(["g", "g", "g", "h", "h"])
    obs, p, n = rc.exact_permutation_p(vals, labs, "g")
    assert n == 10 and p == pytest.approx(0.1) and obs > 0
    assert np.isnan(rc.exact_permutation_p(vals, pd.Series(["g"] * 5), "g")[1])


def test_test_composite_detects_effect() -> None:
    t = rc.composite_scores({"m": rc.animal_measure(_cells(1.0), "m")})
    res = rc.test_composite(t, "penk")
    assert res["n_higher"] == 5 and res["n_lower"] == 3
    assert res["perm_p_one_sided"] == pytest.approx(1 / 56) and res["cles"] == 1.0
    assert res["loao_direction_stable"] is True


def test_test_composite_null() -> None:
    t = rc.composite_scores({"m": rc.animal_measure(_cells(0.0, seed=3), "m")})
    assert rc.test_composite(t, "penk")["perm_p_one_sided"] > 0.05


def test_measure_agreement() -> None:
    t = rc.composite_scores(
        {"m": rc.animal_measure(_cells(1.0), "m"), "n": rc.animal_measure(_cells(-1.0), "m")}
    )
    ag = rc.measure_agreement(t, ["m", "n"], "penk")
    assert ag["agrees"].tolist() == [True, False]


def test_load_measures(tmp_path: Path) -> None:
    (tmp_path / "x").mkdir()
    _cells(1.0).to_csv(tmp_path / "x" / "cells.csv", index=False)
    spec = [
        ("ok", "x/cells.csv", "m", 1, None),
        ("nofile", "x/none.csv", "m", 1, None),
        ("nocol", "x/cells.csv", "zz", 1, None),
    ]
    out = rc.load_measures(spec, root=tmp_path)
    assert list(out) == ["ok"]
