"""Tests for ``scripts/make_penk_continuum.py`` (synthetic tables only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import make_penk_continuum as mc  # noqa: E402


def _write_sources(root: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    cells = []
    for a in range(7):
        ct = "penk" if a < 4 else "nonpenk"
        for r in range(12):
            cells.append({"exp_id": f"s{a}", "roi_idx": r, "animal_id": a, "celltype": ct})
    base = pd.DataFrame(cells)
    shift = np.where(base.celltype == "penk", 1.0, 0.0)
    for name, (path, col, filt) in mc.SOURCES.items():
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        df = base.copy()
        df[col] = rng.normal(shift, 1.0) if "light" not in name else rng.normal(-shift, 1.0)
        if col.endswith("_s") or col == "f_baseline":
            df[col] = np.abs(df[col]) + 0.1
        if filt is not None:
            df[filt[0]] = filt[1]
            if p.exists():
                old = pd.read_csv(p)
                df = pd.concat([old, df], ignore_index=True)
        elif p.exists():
            old = pd.read_csv(p)
            df = old.merge(df[["exp_id", "roi_idx", col]], on=["exp_id", "roi_idx"], how="left")
        df.to_csv(p, index=False)


def test_load_table_and_axes(tmp_path: Path) -> None:
    _write_sources(tmp_path)
    df = mc.load_table(tmp_path)
    assert len(df) == 84 and set(df.celltype) == {"penk", "nonpenk"}
    assert {"speed_during_z", "light_during_z", "step_index"} <= set(df.columns)
    df = mc.add_axes(df)
    assert df.running_axis.between(0, 1).all()
    assert (
        df.loc[df.celltype == "penk", "running_axis"].median()
        > df.loc[df.celltype == "nonpenk", "running_axis"].median()
    )
    assert mc.load_table(tmp_path / "missing").empty


def test_build(tmp_path: Path) -> None:
    _write_sources(tmp_path)
    out = mc.build(tmp_path, n_boot_shift=50, n_boot_modes=20)
    assert out["n_cells"] == {"penk": 48, "nonpenk": 36}
    assert out["shift"]["running_axis"]["diff"][4] > 0
    assert {r["measure"] for r in out["modes"]} >= {"running_axis", "ev_mean_iei_s"}
    assert out["mixing"]["axes"]["n_assignments"] == 35
    assert "features" in out["mixing"] and out["axis_correlation"]["all"]["n"] == 84
    assert set(out["brightness"]) == {"penk", "nonpenk"}


def test_spearman_small_and_clean() -> None:
    r = mc._spearman(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
    assert np.isnan(r["rho"]) and r["n"] == 2
    assert mc._clean({1: [float("nan")]}) == {"1": [None]}
