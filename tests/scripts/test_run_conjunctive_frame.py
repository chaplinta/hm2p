"""Tests for ``scripts/run_conjunctive_frame.py`` (synthetic random walk)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

import run_conjunctive_frame as rc  # noqa: E402
from test_conjunctive_frame import smooth_walk  # noqa: E402


def _session(seed: int = 0) -> dict:
    w = smooth_walk(1500, 6, seed)
    n = w["x"].size
    rng = np.random.default_rng(seed)
    return {
        "spikes": rng.poisson(0.3 + 0.8 * (w["dirs"] == 0), (3, n)).astype(float),
        "speed_cm_s": np.abs(rng.normal(6, 2, n)),
        "hd_deg": w["dirs"] * 90.0 + rng.normal(0, 20, n),
        "light_on": ((np.arange(n) // 600) % 2 == 0),
        "bad_behav": np.zeros(n, bool),
        "x_maze": w["x"],
        "y_maze": w["y"],
        "fps": 10.0,
        "meta": {"exp_id": f"s{seed}", "animal_id": seed, "celltype": "penk"},
    }


def test_run_session_and_summary(tmp_path: Path) -> None:
    res = []
    for seed in range(3):
        f = tmp_path / f"s{seed}.pkl"
        with open(f, "wb") as fh:
            pickle.dump(_session(seed), fh)
        res.append(rc.run_session(f, n_shuffles=4, n_frame_shuffles=5, n_map_shuffles=3))
    r = res[0]
    assert set(r) == {"info", "mi", "frames", "stability"}
    assert set(r["mi"].variant) == {v[0] for v in rc.MI_VARIANTS}
    assert set(r["mi"].condition) == {"all", "light", "dark"}
    assert set(r["frames"].frame) == {"allo", "root", "deadend", "ego_side", "ego_turn"}
    assert set(r["frames"].signal) == {"raw", "spd_acc_removed"}
    assert set(r["stability"]["map"]) == {"full", "direction_residual"}
    info = r["info"].iloc[0]
    assert info.hd_td_R_same > 0.7 and info.n_cells == 3
    # the travel-direction cell (+x) is allocentric
    fr = r["frames"]
    allo = fr[(fr.frame == "allo") & (fr.condition == "all") & (fr.signal == "raw")]
    assert allo.score.iloc[0] > 0.5
    cat = {k: rc.pd.concat([x[k] for x in res], ignore_index=True) for k in r}
    s = rc.summarise(cat["mi"], cat["frames"], cat["stability"])
    assert {"test", "condition", "family", "p", "p_fdr", "rank_biserial"} <= set(s.columns)
    assert (s.n <= 3).all()
