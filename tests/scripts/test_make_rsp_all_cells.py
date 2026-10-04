"""Tests for ``scripts/make_rsp_all_cells.py`` (synthetic tables only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import make_rsp_all_cells as ac  # noqa: E402


def _cells(n_animals: int = 6, n: int = 20) -> pd.DataFrame:
    rows = []
    for a in range(n_animals):
        for r in range(n):
            rows.append({"exp_id": f"s{a}", "roi_idx": r, "animal_id": a})
    return pd.DataFrame(rows)


def _write(root: Path, rel: str, df: pd.DataFrame) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def test_load_var_p_sign_and_bool(tmp_path: Path) -> None:
    c = _cells()
    rng = np.random.default_rng(0)
    c["p"] = rng.uniform(0, 1, len(c))
    c.loc[:9, "p"] = 0.001
    c["z"] = np.where(np.arange(len(c)) < 5, -3.0, 2.0)
    _write(tmp_path, "t/a.csv", c)
    v = ac.Var("x", "X", "locomotion", "t/a.csv", p_col="p", z_col="z", sign=1)
    d = ac.load_var(v, tmp_path)
    assert d is not None and d.sig.iloc[:5].sum() == 0 and d.sig.iloc[5:10].sum() == 5
    b = _cells()
    b["flag"] = ["True", "False"] * (len(b) // 2)
    b = b.rename(columns={"roi_idx": "roi", "animal_id": "animal"})
    _write(tmp_path, "t/b.csv", b)
    vb = ac.Var(
        "y",
        "Y",
        "direction",
        "t/b.csv",
        sig_col="flag",
        rename={"roi": "roi_idx", "animal": "animal_id"},
    )
    db = ac.load_var(vb, tmp_path)
    assert db is not None and db.sig.mean() == 0.5
    assert ac.load_var(ac.Var("z", "Z", "x", "missing.csv", p_col="p"), tmp_path) is None
    assert ac.load_var(ac.Var("z", "Z", "x", "t/a.csv", p_col="nope"), tmp_path) is None


def test_fraction_summary_detects_excess() -> None:
    d = _cells()
    d["sig"] = (np.arange(len(d)) % 3 == 0).astype(float)
    s = ac.fraction_summary(d)
    assert s["n_cells"] == 120 and abs(s["frac"] - 1 / 3) < 0.01
    assert s["binom_p"] < 1e-6 and s["animal_p"] < 0.05 and s["n_animals"] == 6


def test_cooccurrence_and_z_correlation() -> None:
    rng = np.random.default_rng(1)
    a = rng.random(200) < 0.3
    t = pd.DataFrame(
        {"a": a.astype(float), "b": a.astype(float), "c": (rng.random(200) < 0.3).astype(float)}
    )
    rows = {(r["a"], r["b"]): r for r in ac.cooccurrence(t, ["a", "b", "c"])}
    assert rows[("a", "b")]["log_or"] > 3 and rows[("a", "b")]["fisher_p"] < 1e-10
    assert abs(rows[("a", "c")]["log_or"]) < 1
    zt = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0) * 2, "c": rng.normal(size=50)})
    zr = {(r["a"], r["b"]): r for r in ac.z_correlations(zt, ["a", "b", "c"])}
    assert abs(zr[("a", "b")]["rho"] - 1.0) < 1e-9
    assert ac.cooccurrence(t.iloc[:5], ["a", "b"]) == []


def test_mixed_selectivity_detects_clustering() -> None:
    rng = np.random.default_rng(2)
    n = 200
    hub = rng.random(n) < 0.3
    t = pd.DataFrame({"exp_id": np.repeat(["s1", "s2"], n // 2)})
    for k in "abc":
        t[k] = (hub & (rng.random(n) < 0.9)).astype(float)
    m = ac.mixed_selectivity(t, ["a", "b", "c"], n_perm=200)
    assert m["n_cells"] == n and m["p_more_multi"] < 0.01 and m["p_more_none"] < 0.01
    assert len(m["observed"]) == 4
    assert ac.mixed_selectivity(t.iloc[:0], ["a"]) == {}


def test_build_and_clean(tmp_path: Path) -> None:
    c = _cells()
    c["step_index_p"] = np.where(np.arange(len(c)) % 2 == 0, 0.01, 0.5)
    c["step_index_z"] = 2.0
    c["graded_rho_p"] = 0.5
    c["graded_rho_z"] = 0.0
    _write(tmp_path, "celltype_programme_runshape/runshape/cells.csv", c)
    out = ac.build(tmp_path, n_perm=20)
    keys = [v["key"] for v in out["variables"]]
    assert keys == ["run_state", "graded_speed"] and out["n_cells_total"] == 120
    assert out["variables"][0]["frac"] == 0.5 and out["popdec"] == []
    assert ac._clean({"a": [float("nan")]}) == {"a": [None]}


def test_celltype_specificity_matched() -> None:
    rng = np.random.default_rng(3)
    c = _cells(6, 40)
    c["celltype"] = np.where(c.animal_id < 4, "penk", "nonpenk")
    c["f_baseline"] = rng.lognormal(3, 0.5, len(c))
    d = c[ac.KEYS + ["animal_id"]].copy()
    d["animal_id"] = d["animal_id"].astype(str)
    d["sig"] = np.where(
        c.celltype == "penk", rng.random(len(c)) < 0.6, rng.random(len(c)) < 0.2
    ).astype(float)
    d["z"] = np.nan
    v = ac.Var("x", "X", "locomotion", "unused.csv", p_col="p")
    rows = ac.celltype_specificity({"x": (v, d)}, c, n_iter=30)
    r = rows[0]
    assert r["frac_penk"] > r["frac_nonpenk"] and r["fisher_p"] < 1e-4
    assert r["matched_lo"] > 0 and r["matched_lo"] <= r["matched_diff"] <= r["matched_hi"]
