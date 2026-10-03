#!/usr/bin/env python3
"""Maze-context running results for the Penk+ vs Penk-CamKII+ results page.

Reads the ``mazerun`` runner outputs (``hm2p.analysis.maze_running``; EC2 run,
downloaded to ``results/celltype_programme_mazerun_ec2/<signal>/mazerun/``)
and the event durations from the ``ctl`` runner, and writes
``docs/figures/penk/data/maze_running.json``.

Each per-cell metric is summarised as its z score against the cell's own
shuffle null (circular shift of activity against behaviour, or route-label
permutation for route reliability), so 0 means no effect beyond chance for
every metric. Statistics (non-parametric): Wilcoxon signed-rank of z against 0
on cells and on animal medians; fraction of cells with shuffle p < 0.05;
Mann-Whitney U between cell types on cells and on animal medians.

Maze topology: Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a
labyrinth show rapid learning, sudden insight, and efficient exploration."
eLife 10:e66175. doi:10.7554/eLife.66175
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"
OUT = REPO / "docs" / "figures" / "penk" / "data" / "maze_running.json"
GROUPS = ("penk", "nonpenk")

METRICS: tuple[tuple[str, str, str], ...] = (
    ("approach_return", "Into vs out of dead ends", "higher on runs into dead ends"),
    ("novel_repeat", "New vs visited cells", "higher on runs into cells new this session"),
    (
        "edge_decay_rho",
        "Change with repeated traversals",
        "rises with repeats of the same corridor",
    ),
    ("route_reliability", "Route specificity", "consistent per route across traversals"),
    ("light_dark", "Light vs dark runs", "higher on runs in the light"),
    ("xc_peak_r", "Coupling to running", "correlated with the run indicator"),
    ("xc_lead_lag_asym", "Leads vs lags running", "leads running onset"),
)


def _wilcoxon(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x != 0)]
    return float(stats.wilcoxon(x).pvalue) if x.size >= 3 else float("nan")


def _mwu(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)[np.isfinite(a)]
    b = np.asarray(b, float)[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def metric_summary(cells: pd.DataFrame, key: str) -> dict[str, Any]:
    """Per-group z values and tests for one mazerun metric."""
    zcol, pcol = f"{key}_z", f"{key}_p"
    out: dict[str, Any] = {"groups": {}}
    animal_medians: dict[str, np.ndarray] = {}
    for g in GROUPS:
        d = cells[(cells.celltype == g) & np.isfinite(cells[zcol])]
        am = d.groupby("animal_id")[zcol].median()
        animal_medians[g] = am.to_numpy()
        p = d[pcol].dropna()
        out["groups"][g] = {
            "z": [round(float(v), 4) for v in d[zcol]],
            "animal_id": d["animal_id"].astype(str).tolist(),
            "n_cells": int(len(d)),
            "n_animals": int(am.size),
            "median_z": float(d[zcol].median()) if len(d) else float("nan"),
            "cell_p": _wilcoxon(d[zcol].to_numpy()),
            "animal_median_z": float(am.median()) if am.size else float("nan"),
            "animal_p": _wilcoxon(am.to_numpy()),
            "frac_sig": float((p < 0.05).mean()) if len(p) else float("nan"),
        }
    out["between_cell_p"] = _mwu(
        cells.loc[cells.celltype == "penk", zcol].to_numpy(),
        cells.loc[cells.celltype == "nonpenk", zcol].to_numpy(),
    )
    out["between_animal_p"] = _mwu(animal_medians["penk"], animal_medians["nonpenk"])
    return out


def run_behaviour(runs: pd.DataFrame) -> dict[str, Any]:
    """Run counts by type and the run-duration distribution."""
    dur = runs["duration_s"].to_numpy(float)
    dur = dur[np.isfinite(dur)]
    return {
        "n_runs": int(len(runs)),
        "n_sessions": int(runs["exp_id"].nunique()),
        "by_type": {str(k): int(v) for k, v in runs["run_type"].value_counts().items()},
        "duration_s": [round(float(v), 3) for v in dur],
        "duration_quartiles": [float(q) for q in np.percentile(dur, [25, 50, 75, 90])],
    }


def event_durations(ctl: pd.DataFrame) -> dict[str, list[float]]:
    """Mean calcium event duration per cell, by cell type."""
    return {
        g: [round(float(v), 3) for v in ctl.loc[ctl.celltype == g, "ev_mean_duration_s"].dropna()]
        for g in GROUPS
    }


def build(root: Path = RES, signal: str = "spikes") -> dict[str, Any]:
    """Assemble the page data from the mazerun and ctl outputs."""
    base = root / "celltype_programme_mazerun_ec2" / signal / "mazerun"
    cells = pd.read_csv(base / "cells.csv")
    runs = pd.read_csv(base / "runs.csv")
    ctl = pd.read_csv(root / "celltype_programme_ctl" / "ctl" / "cells.csv")
    metrics = []
    for key, label, meaning in METRICS:
        if f"{key}_z" not in cells.columns:
            continue
        metrics.append(
            {"key": key, "label": label, "meaning": meaning, **metric_summary(cells, key)}
        )
    return {
        "signal": signal,
        "runs": run_behaviour(runs),
        "event_duration_s": event_durations(ctl),
        "metrics": metrics,
    }


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
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
