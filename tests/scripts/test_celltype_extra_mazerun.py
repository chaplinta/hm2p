"""Tests for ``scripts/celltype_extra_mazerun.py`` (synthetic sessions, no S3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import celltype_extra_mazerun as mz  # noqa: E402

FPS = 9.6


def _arrays(seed: int, n_steps: int = 500, n_rois: int = 3) -> dict[str, Any]:
    """Random walk on the maze graph: runs of several cells separated by pauses."""
    from hm2p.maze.topology import build_rose_maze

    maze = build_rose_maze()
    rng = np.random.default_rng(seed)
    cell = maze.cell_list[0]
    xs, ys, spd = [], [], []
    for step in range(n_steps):
        moving = (step // 4) % 2 == 0
        k = int(rng.integers(4, 8))
        if moving:
            nbs = maze.adj[cell]
            cell = nbs[int(rng.integers(len(nbs)))]
        xs += [cell[0] + 0.5] * k
        ys += [cell[1] + 0.5] * k
        spd += list(rng.uniform(8, 20, k) if moving else rng.uniform(0, 1, k))
    n = len(xs)
    speed = np.asarray(spd)
    sig = rng.random((n_rois, n)) * 0.1
    sig[0] += (speed > 2.5).astype(float)
    light = (np.arange(n) // 600) % 2 == 0
    return {
        "dff": sig,
        "x_maze": np.asarray(xs),
        "y_maze": np.asarray(ys),
        "speed_cm_s": speed,
        "light_on": light,
        "mask": speed > 2.5,
        "bad_behav": np.zeros(n, bool),
        "fps": FPS,
        "roi_idx": np.arange(n_rois),
    }


def _row(exp: str, animal: str, ct: str) -> pd.Series:
    return pd.Series(
        {
            "exp_id": exp,
            "animal_id": animal,
            "celltype": ct,
            "fibre": "SFB",
            "lens": "f4mm",
            "virus_id": "x",
            "gcamp": "7f",
        }
    )


def _args() -> argparse.Namespace:
    return argparse.Namespace(signal="dff", n_perms=50, seed=0, n_shuffles_evt=30)


def test_session_tables() -> None:
    res = mz.session_tables(_arrays(0), _args())
    assert res is not None
    cells, runs, summary = res
    assert len(cells) == 3
    assert set(mz.TESTED) <= set(cells.columns)
    assert {f"{k}_p" for k in mz.TESTED} <= set(cells.columns)
    assert len(runs) > 20 and runs["run_type"].isin(["into_dead_end", "out_of_dead_end"]).any()
    assert summary["n_bouts"].sum() == len(runs)
    assert cells.loc[0, "xc_peak_r"] > 0.5


def test_session_tables_without_maze_or_bad_behav() -> None:
    arr = _arrays(1)
    arr["x_maze"] = None
    assert mz.session_tables(arr, _args()) is None
    arr2 = _arrays(1)
    arr2.pop("bad_behav")
    res = mz.session_tables(arr2, _args())
    assert res is not None and len(res[0]) == 3


def test_within_group() -> None:
    vals = [0.2, 0.3, 0.1, 0.4, 0.2, 0.5]
    cells = pd.DataFrame(
        {
            "celltype": ["penk"] * 6,
            "animal_id": list("aabbcc"),
            **{k: vals for k in mz.TESTED + mz.DESCRIPTIVE},
            **{f"{k}_p": [0.01, 0.2, 0.03, 0.5, 0.01, 0.9] for k in mz.TESTED},
        }
    )
    wg = mz.within_group(cells)
    assert len(wg) == len(mz.TESTED) + len(mz.DESCRIPTIVE)
    row = wg[wg.measure == "approach_return"].iloc[0]
    assert row["n_sig"] == 3 and np.isfinite(row["cell_wilcoxon_p"])
    assert np.isfinite(row["animal_wilcoxon_p"]) and row["n_animals"] == 3
    assert np.isnan(wg[wg.measure == "xc_peak_lag_s"].iloc[0].get("n_sig", np.nan))


def test_run_mazerun(tmp_path: Path) -> None:
    specs = [("p1", "penk"), ("p2", "penk"), ("p3", "penk"), ("n1", "nonpenk"), ("n2", "nonpenk")]
    sessions = [(_row(f"e{i}", a, ct), _arrays(i)) for i, (a, ct) in enumerate(specs)]
    res = mz.run_mazerun(_args(), sessions, tmp_path)
    assert res["n_sessions"] == 5 and res["n_cells"] == 15
    for name in ("cells", "runs", "run_summary", "within_group", "between_group_report"):
        assert (tmp_path / f"{name}.csv").exists()


def test_run_mazerun_empty_and_skips(tmp_path: Path) -> None:
    assert mz.run_mazerun(_args(), [], tmp_path)["n_sessions"] == 0
    arr = _arrays(0)
    arr["y_maze"] = None
    assert mz.run_mazerun(_args(), [(_row("e0", "p1", "penk"), arr)], tmp_path)["n_sessions"] == 0


def test_registration() -> None:
    import run_celltype_programme as rcp

    assert mz.RUNNER_KEY == "mazerun" and mz.RUNNER is mz.run_mazerun
    assert mz._get(argparse.Namespace(x=None), "x", 2.0) == 2.0
    assert "mazerun" in rcp.HYPOTHESES
    assert rcp.load_extra_runners()["mazerun"] is mz.run_mazerun
