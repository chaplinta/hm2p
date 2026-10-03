"""Tests for ``scripts/make_penk_maze.py`` (synthetic tables only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import make_penk_maze as mm  # noqa: E402


def _cells(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(6):
        ct = "penk" if a < 4 else "nonpenk"
        for r in range(8):
            row = {"animal_id": a, "celltype": ct, "roi_idx": r}
            for key, _, _ in mm.METRICS:
                z = rng.normal(2.0 if (key == "xc_peak_r" and ct == "penk") else 0.0, 1.0)
                row[f"{key}_z"] = z
                row[f"{key}_p"] = 0.01 if z > 1.96 else 0.5
            rows.append(row)
    return pd.DataFrame(rows)


def test_metric_summary_detects_effect() -> None:
    c = _cells()
    s = mm.metric_summary(c, "xc_peak_r")
    assert s["groups"]["penk"]["median_z"] > 1 and s["groups"]["penk"]["cell_p"] < 0.01
    assert s["groups"]["penk"]["n_animals"] == 4 and s["groups"]["nonpenk"]["n_cells"] == 16
    assert s["between_cell_p"] < 0.01 and 0 < s["groups"]["penk"]["frac_sig"] <= 1
    null = mm.metric_summary(c, "approach_return")
    assert null["groups"]["nonpenk"]["cell_p"] > 0.01


def test_wilcoxon_and_mwu_edge_cases() -> None:
    assert np.isnan(mm._wilcoxon(np.array([0.0, np.nan])))
    assert np.isnan(mm._mwu(np.array([]), np.array([1.0])))


def test_run_behaviour_and_event_durations() -> None:
    runs = pd.DataFrame(
        {
            "duration_s": [1.0, 2.0, 3.0, np.nan],
            "exp_id": ["a", "a", "b", "b"],
            "run_type": [
                "into_dead_end",
                "into_dead_end",
                "junction_to_junction",
                "no_transition",
            ],
        }
    )
    rb = mm.run_behaviour(runs)
    assert rb["n_runs"] == 4 and rb["n_sessions"] == 2 and rb["by_type"]["into_dead_end"] == 2
    assert len(rb["duration_s"]) == 3 and rb["duration_quartiles"][1] == 2.0
    ctl = pd.DataFrame(
        {"celltype": ["penk", "nonpenk", "penk"], "ev_mean_duration_s": [3.0, 2.0, np.nan]}
    )
    assert mm.event_durations(ctl) == {"penk": [3.0], "nonpenk": [2.0]}


def test_build(tmp_path: Path) -> None:
    base = tmp_path / "celltype_programme_mazerun_ec2" / "spikes" / "mazerun"
    base.mkdir(parents=True)
    _cells().to_csv(base / "cells.csv", index=False)
    pd.DataFrame({"duration_s": [1.0, 2.0], "exp_id": ["a", "b"], "run_type": ["x", "y"]}).to_csv(
        base / "runs.csv", index=False
    )
    (tmp_path / "celltype_programme_ctl" / "ctl").mkdir(parents=True)
    pd.DataFrame({"celltype": ["penk"], "ev_mean_duration_s": [3.0]}).to_csv(
        tmp_path / "celltype_programme_ctl" / "ctl" / "cells.csv", index=False
    )
    out = mm.build(tmp_path)
    assert [m["key"] for m in out["metrics"]] == [k for k, _, _ in mm.METRICS]
    assert out["runs"]["n_runs"] == 2 and out["event_duration_s"]["penk"] == [3.0]


def test_clean() -> None:
    assert mm._clean({"a": (float("inf"), np.int64(2))}) == {"a": [None, 2]}
