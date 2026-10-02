"""Tests for ``scripts/celltype_extra_runshape.py`` (synthetic sessions, no S3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import celltype_extra_runshape as rs  # noqa: E402

FPS = 9.6


def _arrays(seed: int, n: int = 5000, n_rois: int = 3) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(20, 120, size=n)
    state = np.repeat(np.arange(lengths.size) % 2, lengths)[:n]
    level = np.repeat(rng.uniform(5, 25, size=lengths.size), lengths)[:n]
    speed = np.where(state == 1, level, rng.uniform(0, 1.5, n))
    sig = rng.random((n_rois, n)) * 0.1
    sig[0] += speed / 20.0
    return {
        "dff": sig,
        "speed_cm_s": speed,
        "mask": speed > 1.0,
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
    return argparse.Namespace(signal="dff", n_perms=50, seed=0, n_shuffles_evt=40)


def test_session_cells() -> None:
    df = rs.session_cells(_arrays(0), _args())
    assert len(df) == 3
    assert {"step_index_p", "graded_rho", "bout_adaptation_index", "bout_dur_rho_sum"} <= set(df)
    assert df.loc[0, "graded_rho"] > 0.8 and df.loc[0, "step_index_p"] < 0.05
    assert "speed_curve" not in df.columns


def test_session_cells_without_bad_behav_uses_mask() -> None:
    arr = _arrays(1)
    arr.pop("bad_behav")
    assert len(rs.session_cells(arr, _args())) == 3


def test_within_group() -> None:
    cells = pd.DataFrame(
        {
            "celltype": ["penk"] * 6,
            "animal_id": list("aabbcc"),
            **{k: [0.2, 0.3, 0.1, 0.4, 0.2, 0.5] for k in rs.TESTED + rs.DESCRIPTIVE},
            **{f"{k}_p": [0.01, 0.2, 0.03, 0.5, 0.01, 0.9] for k in rs.TESTED},
        }
    )
    wg = rs.within_group(cells)
    assert len(wg) == len(rs.TESTED) + len(rs.DESCRIPTIVE)
    assert wg.loc[wg.measure == "step_index", "n_sig"].item() == 3


def test_run_runshape(tmp_path: Path) -> None:
    specs = [("p1", "penk"), ("p2", "penk"), ("p3", "penk"), ("n1", "nonpenk"), ("n2", "nonpenk")]
    sessions = [(_row(f"e{i}", a, ct), _arrays(i)) for i, (a, ct) in enumerate(specs)]
    res = rs.run_runshape(_args(), sessions, tmp_path)
    assert res["n_cells"] == 15
    for name in ("cells.csv", "within_group.csv", "between_group_report.csv"):
        assert (tmp_path / name).exists()


def test_empty_and_constants(tmp_path: Path) -> None:
    assert rs.run_runshape(_args(), [], tmp_path)["n_sessions"] == 0
    assert rs.RUNNER_KEY == "runshape" and rs.RUNNER is rs.run_runshape
    assert rs._get(argparse.Namespace(x=None), "x", 2.0) == 2.0
