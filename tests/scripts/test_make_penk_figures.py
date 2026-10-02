"""Tests for the pure helpers in ``scripts/make_penk_figures.py`` (synthetic data only)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import make_penk_figures as mpf  # noqa: E402


def _cells() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "celltype": ["penk"] * 5 + ["nonpenk"] * 4,
            "animal_id": [1, 1, 1, 2, 2, 3, 3, 4, 4],
            "x": [1.0, 2.0, 30.0, 4.0, np.nan, 10.0, 12.0, 20.0, 22.0],
            "p": [0.01, 0.2, 0.03, np.nan, 0.5, 0.04, 0.06, 0.001, 0.002],
        }
    )


# ── animal_medians ────────────────────────────────────────────────────────


def test_animal_medians_values_and_counts() -> None:
    out = mpf.animal_medians(_cells(), "x")
    assert list(out.columns) == ["celltype", "animal_id", "value", "n"]
    got = dict(zip(out["animal_id"], out["value"], strict=True))
    assert got == {1: 2.0, 2: 4.0, 3: 11.0, 4: 21.0}
    assert dict(zip(out["animal_id"], out["n"], strict=True))[2] == 1


def test_animal_medians_mean_agg() -> None:
    out = mpf.animal_medians(_cells(), "x", agg="mean")
    assert out.loc[out.animal_id == 1, "value"].item() == pytest.approx(11.0)


def test_animal_medians_drops_all_nan_animal() -> None:
    df = _cells()
    df.loc[df.animal_id == 2, "x"] = np.nan
    out = mpf.animal_medians(df, "x")
    assert 2 not in set(out["animal_id"])


def test_animal_medians_coerces_strings() -> None:
    df = pd.DataFrame({"celltype": ["penk", "penk"], "animal_id": [1, 1], "x": ["1", "bad"]})
    out = mpf.animal_medians(df, "x")
    assert out["value"].tolist() == [1.0]


# ── fraction_significant ─────────────────────────────────────────────────


def test_fraction_significant_per_animal() -> None:
    out = mpf.fraction_significant(_cells(), "p")
    got = dict(zip(out["animal_id"], out["value"], strict=True))
    assert got[1] == pytest.approx(2 / 3)
    assert got[2] == 0.0  # NaN p dropped, remaining 0.5 not significant
    assert got[3] == 0.5
    assert got[4] == 1.0


def test_fraction_significant_alpha() -> None:
    out = mpf.fraction_significant(_cells(), "p", alpha=0.0015)
    assert out.loc[out.animal_id == 4, "value"].item() == 0.5


# ── cles ──────────────────────────────────────────────────────────────────


def test_cles_complete_separation() -> None:
    assert mpf.cles([5, 6], [1, 2, 3]) == 1.0
    assert mpf.cles([1, 2, 3], [5, 6]) == 0.0


def test_cles_ties_and_nan() -> None:
    assert mpf.cles([1.0, np.nan], [1.0]) == 0.5
    assert math.isnan(mpf.cles([], [1.0]))
    assert math.isnan(mpf.cles([np.nan], [1.0]))


def test_cles_symmetry() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=7), rng.normal(size=4)
    assert mpf.cles(a, b) + mpf.cles(b, a) == pytest.approx(1.0)


# ── group_test / format_stats ────────────────────────────────────────────


def test_group_test_separated_groups() -> None:
    animals = pd.DataFrame(
        {"celltype": ["penk"] * 4 + ["nonpenk"] * 3, "value": [5, 6, 7, 8, 1, 2, 3]}
    )
    res = mpf.group_test(animals)
    assert res["n_penk"] == 4 and res["n_nonpenk"] == 3
    assert res["cles_penk_gt_nonpenk"] == 1.0
    assert res["median_penk"] == 6.5 and res["median_nonpenk"] == 2.0
    # exact two-sided MWU with complete separation: 2 / C(7, 3) = 2 / 35
    assert res["mwu_p_two_sided"] == pytest.approx(2 / 35)


def test_group_test_empty_group() -> None:
    animals = pd.DataFrame({"celltype": ["penk", "penk"], "value": [1.0, 2.0]})
    res = mpf.group_test(animals)
    assert res["n_nonpenk"] == 0
    assert math.isnan(res["mwu_p_two_sided"])
    assert math.isnan(res["cles_penk_gt_nonpenk"])


def test_format_stats() -> None:
    s = mpf.format_stats(
        {"cles_penk_gt_nonpenk": 0.8181, "mwu_p_two_sided": 0.0397, "n_penk": 11, "n_nonpenk": 4}
    )
    assert s == "CLES 0.82, p = 0.040 (11 vs 4 animals)"


# ── jitter / cell_quartiles ──────────────────────────────────────────────


def test_jitter_bounds_and_determinism() -> None:
    a = mpf.jitter(50, width=0.2, seed=3)
    assert a.shape == (50,)
    assert np.all(np.abs(a) <= 0.2)
    np.testing.assert_array_equal(a, mpf.jitter(50, width=0.2, seed=3))
    assert mpf.jitter(0).size == 0


def test_cell_quartiles() -> None:
    q = mpf.cell_quartiles(_cells(), "x")
    assert q["penk"]["n_cells"] == 4
    assert q["penk"]["median"] == pytest.approx(3.0)
    assert q["nonpenk"]["q25"] <= q["nonpenk"]["median"] <= q["nonpenk"]["q75"]


def test_cell_quartiles_skips_empty_group() -> None:
    df = _cells()
    df.loc[df.celltype == "nonpenk", "x"] = np.nan
    assert "nonpenk" not in mpf.cell_quartiles(df, "x")


# ── time courses ─────────────────────────────────────────────────────────


def test_timecourse_by_group() -> None:
    tc = {
        "time_s": [-1.0, 0.0, 1.0],
        "speed": {
            "penk": {"mean": [1, 2, 3], "n_sessions": 5},
            "nonpenk": {"mean": [3, 2, 1], "n_sessions": 2},
        },
    }
    t, means, ns = mpf.timecourse_by_group(tc, "speed")
    np.testing.assert_array_equal(t, [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(means["nonpenk"], [3, 2, 1])
    assert ns == {"penk": 5, "nonpenk": 2}


def test_timecourse_by_group_length_mismatch() -> None:
    tc = {"time_s": [0.0, 1.0], "speed": {"penk": {"mean": [1.0], "n_sessions": 1}}}
    with pytest.raises(ValueError, match="trace length"):
        mpf.timecourse_by_group(tc, "speed")


def test_has_group_traces() -> None:
    assert mpf.has_group_traces({"ltd": {"penk": [0.0]}}, "ltd")
    assert not mpf.has_group_traces({"ltd": [0.0, 1.0]}, "ltd")
    assert not mpf.has_group_traces({}, "ltd")


# ── JSON export helpers ──────────────────────────────────────────────────


def test_animals_to_records() -> None:
    animals = pd.DataFrame(
        {
            "celltype": ["penk", "nonpenk"],
            "animal_id": [11, 22],
            "value": [1.5, np.nan],
            "n": [3, 2],
        }
    )
    recs = mpf.animals_to_records(animals)
    assert recs[0] == {"animal_id": "11", "celltype": "penk", "value": 1.5, "n": 3}
    assert recs[1]["value"] is None


def test_clean_makes_json_serialisable() -> None:
    obj = {
        "a": np.float64(np.nan),
        "b": np.arange(3),
        "c": (np.int64(2), np.bool_(True)),
        "d": {1: float("inf")},
    }
    out = mpf._clean(obj)
    assert out == {"a": None, "b": [0, 1, 2], "c": [2, True], "d": {"1": None}}
    json.dumps(out)  # must not raise
