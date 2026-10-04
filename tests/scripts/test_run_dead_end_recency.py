"""Tests for ``scripts/run_dead_end_recency.py`` (synthetic random walk on the maze)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import run_dead_end_recency as rr  # noqa: E402

from hm2p.maze.topology import build_rose_maze  # noqa: E402


def _walk_session(n_frames: int = 12_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    maze = build_rose_maze()
    cur = maze.cell_list[0]
    xs, ys = [], []
    while len(xs) < n_frames:
        stay = int(rng.integers(5, 15))
        xs += [cur[0] + 0.5] * stay
        ys += [cur[1] + 0.5] * stay
        nb = maze.adj[cur]
        cur = nb[int(rng.integers(len(nb)))]
    n = n_frames
    return {
        "spikes": rng.poisson(0.2, (3, n)).astype(float),
        "speed_cm_s": rng.normal(5, 1, n),
        "ahv_deg_s": rng.normal(0, 40, n),
        "light_on": ((np.arange(n) // 600) % 2 == 0).astype(float),
        "bad_behav": np.zeros(n, bool),
        "x_maze": np.asarray(xs[:n]),
        "y_maze": np.asarray(ys[:n]),
        "frame_times": np.arange(n) / 10.0,
    }


def test_session_result_on_random_walk() -> None:
    res = rr.session_result(
        _walk_session(), {"exp_id": "s1", "animal_id": 7}, "spikes", 1.0, 20, 0
    )
    assert res is not None
    cells, ents = res
    assert set(cells.cond) >= {"all", "light", "dark"} and (cells.exp_id == "s1").all()
    assert (ents.recency_s > 0).all() and ents.animal_id.iloc[0] == "7"


def test_session_result_skips_unusable() -> None:
    d = _walk_session(2000)
    assert (
        rr.session_result(
            {k: v for k, v in d.items() if k != "x_maze"}, {"exp_id": "s"}, "spikes", 1.0, 5, 0
        )
        is None
    )
    d["x_maze"][:] = np.nan
    assert rr.session_result(d, {"exp_id": "s", "animal_id": 1}, "spikes", 1.0, 5, 0) is None


def test_session_result_dff_rise_signal() -> None:
    d = _walk_session()
    d["dff"] = np.cumsum(np.random.default_rng(1).normal(0, 0.1, d["spikes"].shape), axis=1)
    res = rr.session_result(d, {"exp_id": "s1", "animal_id": 7}, "dff_rise", 1.0, 10, 0)
    assert res is not None and len(res[0]) > 0
