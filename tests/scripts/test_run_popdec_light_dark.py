"""Tests for ``scripts/run_popdec_light_dark.py`` (synthetic random walk)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import run_popdec_light_dark as pl  # noqa: E402

from hm2p.analysis.population_maze import PopDecParams  # noqa: E402
from hm2p.maze.topology import build_rose_maze  # noqa: E402


def _walk(n: int = 6000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    maze = build_rose_maze()
    cur, xs, ys = maze.cell_list[0], [], []
    while len(xs) < n:
        stay = int(rng.integers(5, 15))
        xs += [cur[0] + 0.5] * stay
        ys += [cur[1] + 0.5] * stay
        nb = maze.adj[cur]
        cur = nb[int(rng.integers(len(nb)))]
    return {
        "spikes": rng.poisson(0.2, (3, n)).astype(float),
        "speed_cm_s": np.abs(rng.normal(5, 3, n)),
        "ahv_deg_s": rng.normal(0, 40, n),
        "hd_deg": rng.uniform(0, 360, n),
        "light_on": ((np.arange(n) // 600) % 2 == 0).astype(float),
        "bad_behav": np.zeros(n, bool),
        "x_maze": np.asarray(xs[:n]),
        "y_maze": np.asarray(ys[:n]),
        "fps": 10.0,
        "meta": {"exp_id": "s", "animal_id": 3},
    }


def test_session_rows_light_and_dark() -> None:
    rows = pl.session_rows(_walk(), PopDecParams(n_shuffles=3, subset_sizes=(), n_blocks=3))
    assert set(rows.condition) == {"light", "dark"} and (rows.animal_id == "3").all()
    assert {"inout", "position"} <= set(rows.decoder)
