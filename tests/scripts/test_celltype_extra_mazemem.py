"""Tests for ``scripts/celltype_extra_mazemem.py`` (synthetic sessions, no S3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import celltype_extra_mazemem as mmx  # noqa: E402

FPS = 9.6


def _arrays(seed: int, n_target: int = 3000, n_rois: int = 5) -> dict[str, Any]:
    """Walk on the maze graph with dead-end dwells and random activity."""
    from hm2p.maze.topology import build_rose_maze

    maze = build_rose_maze()
    rng = np.random.default_rng(seed)
    cell, prev = maze.cell_list[0], None
    xs: list[float] = []
    ys: list[float] = []
    spd: list[float] = []
    hd: list[float] = []
    while len(xs) < n_target:
        nbs = [c for c in maze.adj[cell] if c != prev] or maze.adj[cell]
        nxt = nbs[int(rng.integers(len(nbs)))]
        k = int(rng.integers(3, 6))
        t = np.arange(k) / k
        xs += list(cell[0] + 0.5 + t * (nxt[0] - cell[0]))
        ys += list(cell[1] + 0.5 + t * (nxt[1] - cell[1]))
        spd += list(rng.uniform(6, 20, k))
        hd += list(np.full(k, np.degrees(np.arctan2(nxt[1] - cell[1], nxt[0] - cell[0]))))
        prev, cell = cell, nxt
        if len(maze.adj[cell]) == 1:
            k = int(rng.integers(6, 15))
            xs += [cell[0] + 0.5] * k
            ys += [cell[1] + 0.5] * k
            spd += list(rng.uniform(0, 1, k))
            hd += [hd[-1]] * k
    n = n_target
    sig = rng.random((n_rois, n))
    return {
        "dff": sig,
        "spikes": sig * 2,
        "x_maze": np.asarray(xs[:n]),
        "y_maze": np.asarray(ys[:n]),
        "speed_cm_s": np.asarray(spd[:n]),
        "hd_deg": np.asarray(hd[:n]),
        "ahv_deg_s": rng.normal(0, 30, n),
        "light_on": (np.arange(n) // 576) % 2 == 0,
        "mask": np.asarray(spd[:n]) > 2.5,
        "bad_behav": np.zeros(n, bool),
        "syllable_id": rng.integers(0, 10, n).astype(np.int16),
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
    return argparse.Namespace(
        signal=signal, seed=0, mazemem_shuffles=3, n_jobs=1, mazemem_recent_k=2, mazemem_min_n=3
    )


def test_params_from_args() -> None:
    p = mmx.params_from_args(_args())
    assert p.n_shuffles == 3 and p.recent_k == 2 and p.min_n == 3 and p.seed == 0
    d = mmx.params_from_args(argparse.Namespace(mazemem_shuffles=None))
    assert d.n_shuffles == mmx.DEFAULT_SHUFFLES and d.n_jobs == 1 and d.recent_k == 3


def test_signal_used_and_valid_frames() -> None:
    arr = _arrays(0, 600)
    assert mmx.signal_used(arr, "spikes") == "spikes"
    assert mmx.signal_used(arr, "dff") == "dff"
    assert mmx.signal_used({"spikes": None}, "spikes") == "dff"
    arr["bad_behav"][:5] = True
    v = mmx.valid_frames(arr, 10)
    assert v.size == 10 and not v[:5].any() and v[5:].all()
    arr.pop("bad_behav")
    assert np.array_equal(mmx.valid_frames(arr, 10), arr["mask"][:10])


def test_session_tables() -> None:
    res = mmx.session_tables(_arrays(1), _args("spikes"))
    assert res is not None
    assert (res.sessions["signal"] == "spikes").all()
    assert (res.behaviour["signal"] == "spikes").all()
    assert set(res.sessions["analysis"]) == {"splitter", "distance", "planning"}
    assert (res.sessions["n_cells"] == 5).all()
    assert res.counts["signal"] == "spikes" and res.counts["n_cells"] == 5
    assert res.counts["n_top_syllables"] == 8
    assert "behaviour_plus_syllable" in set(res.sessions["feature_set"])
    arr = _arrays(1, 600)
    arr["x_maze"] = None
    assert mmx.session_tables(arr, _args()) is None


def test_run_mazemem(tmp_path: Path) -> None:
    specs = [("p1", "penk"), ("p2", "penk"), ("p3", "nonpenk")]
    sessions = [(_row(f"e{i}", a, ct), _arrays(i)) for i, (a, ct) in enumerate(specs)]
    nomaze = _arrays(9, 600)
    nomaze["y_maze"] = None
    sessions.append((_row("e9", "p9", "penk"), nomaze))
    res = mmx.run_mazemem(_args(), sessions, tmp_path)
    assert res["n_sessions"] == 3 and res["n_animals"] == 3
    for name in (
        "sessions",
        "behaviour",
        "sample_counts",
        "subset_curve",
        "subset_curve_summary",
        "summary",
        "comparisons",
    ):
        assert (tmp_path / f"{name}.csv").exists()
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["params"]["n_shuffles"] == 3 and info["params"]["recent_k"] == 2
    counts = pd.read_csv(tmp_path / "sample_counts.csv")
    assert len(counts) == 4 and (counts["reason"] == "no maze").sum() == 1
    sess = pd.read_csv(tmp_path / "sessions.csv")
    assert {"celltype", "animal_id", "exp_id", "signal"} <= set(sess.columns)
    summ = pd.read_csv(tmp_path / "summary.csv")
    assert "novelty_behaviour" in set(summ["analysis"])
    comp = pd.read_csv(tmp_path / "comparisons.csv")
    assert "light_vs_dark" in set(comp["comparison_type"])


def test_run_mazemem_empty(tmp_path: Path) -> None:
    assert mmx.run_mazemem(_args(), [], tmp_path)["n_sessions"] == 0
    arr = _arrays(0, 600)
    arr["x_maze"] = None
    out = mmx.run_mazemem(_args(), [(_row("e0", "p1", "penk"), arr)], tmp_path)
    assert out["n_sessions"] == 0


def test_registration() -> None:
    import run_celltype_programme as rcp

    assert mmx.RUNNER_KEY == "mazemem" and mmx.RUNNER is mmx.run_mazemem
    assert mmx._get(argparse.Namespace(x=None), "x", 2.0) == 2.0
    assert "mazemem" in rcp.HYPOTHESES
    assert rcp.load_extra_runners()["mazemem"] is mmx.run_mazemem
