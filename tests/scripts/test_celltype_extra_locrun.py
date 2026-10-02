"""Tests for ``scripts/celltype_extra_locrun.py`` (synthetic sessions, no S3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import celltype_extra_locrun as lr  # noqa: E402

FPS = 9.6


def _arrays(seed: int, n: int = 4000, n_rois: int = 4) -> dict[str, Any]:
    """Animal walking along maze cells; speed higher between cell centres."""
    from hm2p.maze.topology import build_rose_maze

    rng = np.random.default_rng(seed)
    cells = np.array(build_rose_maze().cell_list, dtype=float)
    lengths = rng.integers(15, 60, size=n)
    idx = np.repeat(rng.integers(0, len(cells), size=lengths.size), lengths)[:n]
    xy = cells[idx] + rng.uniform(0.2, 0.8, (n, 2))
    speed = rng.uniform(0, 15, n)
    sig = rng.random((n_rois, n)) * 0.1
    sig[0] += speed / 10.0  # running cell
    return {
        "dff": sig,
        "x_maze": xy[:, 0],
        "y_maze": xy[:, 1],
        "speed_cm_s": speed,
        "mask": np.ones(n, bool),
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
    return argparse.Namespace(signal="dff", n_perms=50, seed=0, n_shuffles_evt=40)


def test_session_cells_detects_running_cell() -> None:
    df = lr.session_cells(_arrays(0), _args())
    assert df is not None and len(df) == 4
    assert {"loc_index", "run_index_corr_p", "n_frames_corr_run_matched"} <= set(df.columns)
    assert df.loc[0, "run_index_corr"] > 0.3


def test_session_cells_without_maze() -> None:
    arr = _arrays(0)
    arr["x_maze"] = None
    assert lr.session_cells(arr, _args()) is None


def test_within_group() -> None:
    cells = pd.DataFrame(
        {
            "celltype": ["penk"] * 6,
            "animal_id": ["a", "a", "b", "b", "c", "c"],
            **{k: [0.2, 0.3, 0.1, 0.4, 0.2, 0.5] for k in lr.INDICES},
            **{f"{k}_p": [0.01, 0.2, 0.03, 0.5, 0.01, 0.9] for k in lr.INDICES},
        }
    )
    wg = lr.within_group(cells)
    assert len(wg) == 3 and (wg["n_sig"] == 3).all()
    assert wg["frac_sig"].iloc[0] == 0.5 and np.isfinite(wg["wilcoxon_p"]).all()


def test_run_locrun(tmp_path: Path) -> None:
    specs = [("p1", "penk"), ("p2", "penk"), ("p3", "penk"), ("n1", "nonpenk"), ("n2", "nonpenk")]
    sessions = [(_row(f"e{i}", a, ct), _arrays(i)) for i, (a, ct) in enumerate(specs)]
    res = lr.run_locrun(_args(), sessions, tmp_path)
    assert res["n_sessions"] == 5 and res["n_cells"] == 20
    for name in ("cells.csv", "within_group.csv", "between_group_report.csv"):
        assert (tmp_path / name).exists()


def test_run_locrun_empty(tmp_path: Path) -> None:
    assert lr.run_locrun(_args(), [], tmp_path)["n_sessions"] == 0


def test_registration_constants() -> None:
    assert lr.RUNNER_KEY == "locrun" and lr.RUNNER is lr.run_locrun
    assert lr._get(argparse.Namespace(x=None), "x", 3.0) == 3.0
