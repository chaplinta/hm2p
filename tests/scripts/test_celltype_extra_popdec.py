"""Tests for ``scripts/celltype_extra_popdec.py`` (synthetic sessions, no S3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import celltype_extra_popdec as pdx  # noqa: E402

FPS = 9.6


def _arrays(seed: int, n_target: int = 2500, n_rois: int = 4) -> dict[str, Any]:
    """Random walk on the maze graph (runs separated by pauses) with place-tuned cells."""
    from hm2p.maze.discretize import discretize_position_fast
    from hm2p.maze.topology import build_rose_maze

    maze = build_rose_maze()
    rng = np.random.default_rng(seed)
    cell, prev = maze.cell_list[0], None
    xs: list[float] = []
    ys: list[float] = []
    spd: list[float] = []
    hd: list[float] = []
    while len(xs) < n_target:
        for _ in range(int(rng.integers(2, 7))):
            nbs = [c for c in maze.adj[cell] if c != prev] or maze.adj[cell]
            nxt = nbs[int(rng.integers(len(nbs)))]
            k = int(rng.integers(3, 6))
            t = np.arange(k) / k
            xs += list(cell[0] + 0.5 + t * (nxt[0] - cell[0]))
            ys += list(cell[1] + 0.5 + t * (nxt[1] - cell[1]))
            spd += list(rng.uniform(6, 20, k))
            hd += list(np.full(k, np.degrees(np.arctan2(nxt[1] - cell[1], nxt[0] - cell[0]))))
            prev, cell = cell, nxt
        k = int(rng.integers(10, 25))
        xs += [cell[0] + 0.5] * k
        ys += [cell[1] + 0.5] * k
        spd += list(rng.uniform(0, 1, k))
        hd += [hd[-1]] * k
    x, y = np.asarray(xs), np.asarray(ys)
    n = x.size
    idx = discretize_position_fast(x, y, maze)
    sig = rng.random((n_rois, n)) * 0.1
    sig[0] += (idx == idx[0]).astype(float)
    return {
        "dff": sig,
        "spikes": sig * 2,
        "x_maze": x,
        "y_maze": y,
        "speed_cm_s": np.asarray(spd),
        "hd_deg": np.asarray(hd),
        "ahv_deg_s": rng.normal(0, 30, n),
        "light_on": (np.arange(n) // 576) % 2 == 0,
        "mask": np.asarray(spd) > 2.5,
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


def _args(signal: str = "dff") -> argparse.Namespace:
    return argparse.Namespace(signal=signal, seed=0, popdec_shuffles=4, n_jobs=1, n_perms=10)


def test_params_from_args() -> None:
    p = pdx.params_from_args(_args())
    assert p.n_shuffles == 4 and p.n_jobs == 1 and p.seed == 0
    d = pdx.params_from_args(argparse.Namespace(popdec_shuffles=None))
    assert d.n_shuffles == pdx.DEFAULT_SHUFFLES and d.n_jobs == 1


def test_signal_used_and_valid_frames() -> None:
    arr = _arrays(0, 600)
    assert pdx.signal_used(arr, "spikes") == "spikes"
    assert pdx.signal_used(arr, "dff") == "dff"
    assert pdx.signal_used({"spikes": None}, "spikes") == "dff"
    arr["bad_behav"][:5] = True
    v = pdx.valid_frames(arr, 10)
    assert v.size == 10 and not v[:5].any() and v[5:].all()
    arr.pop("bad_behav")
    assert np.array_equal(pdx.valid_frames(arr, 10), arr["mask"][:10])


def test_session_tables() -> None:
    res = pdx.session_tables(_arrays(1), _args("spikes"))
    assert res is not None
    dec, sub, counts = res
    assert (dec["signal"] == "spikes").all() and (sub["signal"] == "spikes").all()
    assert set(dec["decoder"]) == {"inout", "position", "edge_direction", "junction_exit"}
    assert (dec["n_cells"] == 4).all()
    assert counts["signal"] == "spikes" and counts["n_cells"] == 4
    arr = _arrays(1, 600)
    arr["x_maze"] = None
    assert pdx.session_tables(arr, _args()) is None


def test_run_popdec(tmp_path: Path) -> None:
    specs = [("p1", "penk"), ("p2", "penk"), ("p3", "nonpenk")]
    sessions = [(_row(f"e{i}", a, ct), _arrays(i)) for i, (a, ct) in enumerate(specs)]
    nomaze = _arrays(9, 600)
    nomaze["y_maze"] = None
    sessions.append((_row("e9", "p9", "penk"), nomaze))
    res = pdx.run_popdec(_args(), sessions, tmp_path)
    assert res["n_sessions"] == 3 and res["n_animals"] == 3
    for name in (
        "sessions",
        "subset_curve",
        "subset_curve_summary",
        "behaviour_counts",
        "summary",
        "comparisons",
    ):
        assert (tmp_path / f"{name}.csv").exists()
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["params"]["n_shuffles"] == 4
    counts = pd.read_csv(tmp_path / "behaviour_counts.csv")
    assert len(counts) == 4 and (counts["reason"] == "no maze").sum() == 1
    sess = pd.read_csv(tmp_path / "sessions.csv")
    assert {"celltype", "animal_id", "exp_id"} <= set(sess.columns)


def test_run_popdec_empty(tmp_path: Path) -> None:
    assert pdx.run_popdec(_args(), [], tmp_path)["n_sessions"] == 0
    arr = _arrays(0, 600)
    arr["x_maze"] = None
    assert pdx.run_popdec(_args(), [(_row("e0", "p1", "penk"), arr)], tmp_path)["n_sessions"] == 0


def test_registration() -> None:
    import run_celltype_programme as rcp

    assert pdx.RUNNER_KEY == "popdec" and pdx.RUNNER is pdx.run_popdec
    assert pdx._get(argparse.Namespace(x=None), "x", 2.0) == 2.0
    assert "popdec" in rcp.HYPOTHESES
    assert rcp.load_extra_runners()["popdec"] is pdx.run_popdec
