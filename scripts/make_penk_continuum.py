#!/usr/bin/env python3
"""Discrete type or continuum? Cell-level analyses for the Penk results page.

Uses the per-cell programme tables (no new imaging analysis) and
:mod:`hm2p.analysis.continuum` to ask whether Penk+ cells form a distinct
group or a shifted part of a shared distribution:

1. Shift functions (Penk+ minus Penk-CamKII+ at each decile) with an
   animal-then-cell bootstrap band. A continuum shift moves every decile; a
   discrete sub-population mainly moves one tail.
2. Silverman mode tests within each cell type: a hidden discrete
   sub-population tends to make a measure multimodal.
3. A running axis and a light axis per cell (mean percentile rank of the
   measures used for the animal-level composites, here per cell), their
   correlation, and k-nearest-neighbour label mixing in that plane and in a
   larger amplitude-independent feature space, against an animal-level label
   permutation (cells from other animals only).
4. Within each cell type, the relation of the running axis to baseline
   fluorescence (an expression proxy).

Writes ``docs/figures/penk/data/continuum.json``. Citations are in
:mod:`hm2p.analysis.continuum`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from hm2p.analysis.continuum import (
    knn_label_mixing,
    rank_axis,
    shift_function,
    silverman_mode_test,
)

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"
OUT = REPO / "docs" / "figures" / "penk" / "data" / "continuum.json"
GROUPS = ("penk", "nonpenk")
KEYS = ["exp_id", "roi_idx"]

# (name, table, column, row filter)
SOURCES: dict[str, tuple[str, str, tuple[str, str] | None]] = {
    "step_index": ("celltype_programme_runshape/runshape/cells.csv", "step_index", None),
    "speed_during_z": (
        "celltype_programme_etb/etb/cells.csv",
        "during_z",
        ("channel", "speed_cm_s"),
    ),
    "active_during_z": ("celltype_programme_etb/etb/cells.csv", "during_z", ("channel", "active")),
    "act_movement_modulation": (
        "celltype_programme/h2/cells.csv",
        "act_movement_modulation",
        None,
    ),
    "xc_peak_r_z": (
        "celltype_programme_mazerun_ec2/spikes/mazerun/cells.csv",
        "xc_peak_r_z",
        None,
    ),
    "act_light_modulation": ("celltype_programme/h2/cells.csv", "act_light_modulation", None),
    "light_during_z": (
        "celltype_programme_etb/etb/cells.csv",
        "during_z",
        ("channel", "light_on"),
    ),
    "dtl_late_amplitude": ("celltype_programme/h5/cells.csv", "dtl_late_amplitude", None),
    "ltd_early_amplitude": ("celltype_programme/h5/cells.csv", "ltd_early_amplitude", None),
    "part_light": ("celltype_programme/h8/cells.csv", "part_light", None),
    "ev_mean_duration_s": ("celltype_programme_ctl/ctl/cells.csv", "ev_mean_duration_s", None),
    "ev_mean_iei_s": ("celltype_programme_ctl/ctl/cells.csv", "ev_mean_iei_s", None),
    "iso_decay_s": ("celltype_programme_ctl/ctl/cells.csv", "iso_decay_s", None),
    "f_baseline": ("celltype_programme_ctl/ctl/cells.csv", "f_baseline", None),
}

RUNNING_AXIS = (
    ("step_index", 1),
    ("speed_during_z", 1),
    ("active_during_z", 1),
    ("act_movement_modulation", 1),
    ("xc_peak_r_z", 1),
)
LIGHT_AXIS = (
    ("act_light_modulation", 1),
    ("light_during_z", 1),
    ("dtl_late_amplitude", 1),
    ("ltd_early_amplitude", -1),
    ("part_light", 1),
)
# measures that do not scale with fluorescence brightness
FEATURE_SPACE = (
    "speed_during_z",
    "active_during_z",
    "xc_peak_r_z",
    "light_during_z",
    "act_light_modulation",
    "ev_mean_duration_s",
    "ev_mean_iei_s",
    "iso_decay_s",
)
LABELS = {
    "running_axis": "Running axis (mean rank)",
    "light_axis": "Light axis (mean rank)",
    "step_index": "Run vs still step index",
    "speed_during_z": "Speed during events (z)",
    "xc_peak_r_z": "Coupling to running (z)",
    "act_light_modulation": "Light modulation index",
    "light_during_z": "Lights on during events (z)",
    "ev_mean_duration_s": "Event duration (s)",
    "ev_mean_iei_s": "Inter-event interval (s)",
    "iso_decay_s": "Isolated small-event decay (s)",
}
SHIFT_MEASURES = tuple(LABELS)
LOG_MEASURES = ("ev_mean_duration_s", "ev_mean_iei_s", "iso_decay_s")


def load_table(root: Path = RES) -> pd.DataFrame:
    """One row per cell with every available source measure as a column."""
    base: pd.DataFrame | None = None
    for name, (path, col, filt) in SOURCES.items():
        p = root / path
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if col not in df.columns:
            continue
        if filt is not None:
            df = df[df[filt[0]] == filt[1]]
        part = df[KEYS + ["animal_id", "celltype", col]].rename(columns={col: name})
        part = part.drop_duplicates(KEYS)
        if base is None:
            base = part
        else:
            base = base.merge(part[KEYS + [name]], on=KEYS, how="outer")
            fill = part.set_index(KEYS)[["animal_id", "celltype"]]
            base = base.set_index(KEYS)
            base[["animal_id", "celltype"]] = base[["animal_id", "celltype"]].fillna(fill)
            base = base.reset_index()
    if base is None:
        return pd.DataFrame(columns=KEYS + ["animal_id", "celltype"])
    base["animal_id"] = base["animal_id"].astype(str)
    return base[base.celltype.isin(GROUPS)].reset_index(drop=True)


def add_axes(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``running_axis`` and ``light_axis`` (mean percentile rank across all cells)."""
    out = df.copy()
    for axis, spec in (("running_axis", RUNNING_AXIS), ("light_axis", LIGHT_AXIS)):
        cols = [(k, s) for k, s in spec if k in out.columns]
        out[axis] = (
            rank_axis([out[k].to_numpy() for k, _ in cols], [s for _, s in cols])
            if cols
            else np.nan
        )
    return out


def _spearman(a: pd.Series, b: pd.Series) -> dict[str, float]:
    ok = a.notna() & b.notna()
    if ok.sum() < 5:
        return {"rho": float("nan"), "p": float("nan"), "n": int(ok.sum())}
    r = stats.spearmanr(a[ok], b[ok])
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(ok.sum())}


def shift_functions(df: pd.DataFrame, n_boot: int = 1000) -> dict[str, Any]:
    """Shift function per measure, Penk+ minus Penk-CamKII+, animal-then-cell bootstrap."""
    out = {}
    for m in SHIFT_MEASURES:
        if m not in df.columns:
            continue
        a = df[df.celltype == "penk"]
        b = df[df.celltype == "nonpenk"]
        xa, xb = a[m].to_numpy(float), b[m].to_numpy(float)
        if m in LOG_MEASURES:  # timescales: a uniform ratio is a uniform shift in log10
            xa = np.log10(np.where(xa > 0, xa, np.nan))
            xb = np.log10(np.where(xb > 0, xb, np.nan))
        sf = shift_function(xa, xb, a["animal_id"], b["animal_id"], n_boot=n_boot, seed=0)
        sf["log10"] = m in LOG_MEASURES
        sf["label"] = LABELS[m] + (" [log10 ratio]" if m in LOG_MEASURES else "")
        sf["median_penk"] = float(a[m].median())
        sf["median_nonpenk"] = float(b[m].median())
        out[m] = sf
    return out


def mode_tests(df: pd.DataFrame, n_boot: int = 300) -> list[dict[str, Any]]:
    """Silverman test of unimodality for each measure within each cell type."""
    rows = []
    for m in SHIFT_MEASURES:
        if m not in df.columns:
            continue
        row: dict[str, Any] = {"measure": m, "label": LABELS[m]}
        for g in GROUPS:
            x = df.loc[df.celltype == g, m].to_numpy(float)
            if m in LOG_MEASURES:
                x = np.log(x[x > 0])  # timescales are right-skewed; test on log scale
            r = silverman_mode_test(x, k=1, n_boot=n_boot, seed=1)
            row[g] = {"p": r["p"], "n": r["n"]}
        rows.append(row)
    return rows


def mixing(df: pd.DataFrame, k: int = 10) -> dict[str, Any]:
    """kNN label mixing in the axis plane and in the brightness-independent feature space."""
    out = {}
    plane = df.dropna(subset=["running_axis", "light_axis"])
    out["axes"] = knn_label_mixing(
        plane[["running_axis", "light_axis"]].to_numpy(),
        plane["celltype"].to_numpy(),
        plane["animal_id"].to_numpy(),
        k=k,
    )
    cols = [c for c in FEATURE_SPACE if c in df.columns]
    full = df.dropna(subset=cols)
    if len(full) > 2 * k and full.celltype.nunique() == 2:
        X = full[cols].to_numpy(float)
        for j, c in enumerate(cols):
            if c in LOG_MEASURES:
                X[:, j] = np.log(np.clip(X[:, j], 1e-6, None))
        out["features"] = knn_label_mixing(
            X, full["celltype"].to_numpy(), full["animal_id"].to_numpy(), k=k
        )
        out["features"]["columns"] = cols
    return out


def build(root: Path = RES, n_boot_shift: int = 1000, n_boot_modes: int = 300) -> dict[str, Any]:
    """All continuum results for the page."""
    df = add_axes(load_table(root))
    cells = {
        g: {
            "running_axis": df.loc[df.celltype == g, "running_axis"].round(4).tolist(),
            "light_axis": df.loc[df.celltype == g, "light_axis"].round(4).tolist(),
            "animal_id": df.loc[df.celltype == g, "animal_id"].tolist(),
        }
        for g in GROUPS
    }
    axis_corr = {"all": _spearman(df.running_axis, df.light_axis)}
    brightness = {}
    for g in GROUPS:
        d = df[df.celltype == g]
        axis_corr[g] = _spearman(d.running_axis, d.light_axis)
        if "f_baseline" in d.columns:
            brightness[g] = {
                "running_axis": _spearman(d.running_axis, d.f_baseline),
                "light_axis": _spearman(d.light_axis, d.f_baseline),
            }
    return {
        "n_cells": {g: int((df.celltype == g).sum()) for g in GROUPS},
        "axes": {
            "running": [k for k, _ in RUNNING_AXIS],
            "light": [f"{'-' if s < 0 else ''}{k}" for k, s in LIGHT_AXIS],
        },
        "cells": cells,
        "axis_correlation": axis_corr,
        "brightness": brightness,
        "shift": shift_functions(df, n_boot_shift),
        "modes": mode_tests(df, n_boot_modes),
        "mixing": mixing(df),
    }


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    return obj


def main() -> None:  # pragma: no cover - file I/O entry point
    data = _clean(build())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT}")


if __name__ == "__main__":  # pragma: no cover
    main()
