"""Tests for ``hm2p.analysis.conjunctive_frame`` (synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis import behaviour_mi as bm
from hm2p.analysis import conjunctive_frame as cf
from hm2p.maze.topology import build_rose_maze

MAZE = build_rose_maze()
IDX = MAZE.cell_to_idx


def smooth_walk(n_moves: int = 3000, frames_per_move: int = 6, seed: int = 0) -> dict:
    """Random walk on the maze graph without reversals except at dead ends.

    Position moves linearly between cell centres; returns per-frame x, y,
    true cell index and true travel direction.
    """
    rng = np.random.default_rng(seed)
    prev, cur = None, MAZE.cell_list[IDX[(3, 2)]]
    xs, ys, cells, dirs = [], [], [], []
    for _ in range(n_moves):
        nb = [c for c in MAZE.adj[cur] if c != prev] or [prev]
        nxt = nb[int(rng.integers(len(nb)))]
        d = cf.DIRS.index((nxt[0] - cur[0], nxt[1] - cur[1]))
        for f in range(frames_per_move):
            t = f / frames_per_move
            x = cur[0] + 0.5 + t * (nxt[0] - cur[0])
            y = cur[1] + 0.5 + t * (nxt[1] - cur[1])
            xs.append(x)
            ys.append(y)
            cells.append(IDX[cur] if t < 0.5 else IDX[nxt])
            dirs.append(d)
        prev, cur = cur, nxt
    return {
        "x": np.asarray(xs),
        "y": np.asarray(ys),
        "cells": np.asarray(cells),
        "dirs": np.asarray(dirs),
    }


@pytest.fixture(scope="module")
def walk() -> dict:
    return smooth_walk()


# --------------------------------------------------------------------- MI


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 10_000))
def test_cond_mi_rows_matches_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a, y, c = rng.integers(0, 4, 500), rng.integers(0, 5, 500), rng.integers(0, 3, 500)
    v = cf.cond_mi_rows(np.stack([a, a[::-1]]), y, c)
    assert v[0] == pytest.approx(bm.conditional_mi(a, y, c), abs=1e-10)
    assert v[1] == pytest.approx(bm.conditional_mi(a[::-1], y, c), abs=1e-10)
    assert cf.cond_mi_rows(a[None], y, None)[0] == pytest.approx(bm.plugin_mi(a, y), abs=1e-10)


def test_cond_mi_rows_empty_and_mi_with_null() -> None:
    assert cf.cond_mi_rows(np.zeros((2, 0), int), np.zeros(0, int), None).tolist() == [0, 0]
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 500)
    sig = rng.poisson(0.5, 1000).astype(float) + 3 * y
    used = np.ones(1000, bool)
    b = cf.shifted_bins(sig, used, rng.integers(200, 800, 30))
    assert b.shape == (31, 1000)
    r = cf.mi_with_null(b, y, None)
    assert r["mi_debiased"] > 0.3 and r["p"] < 0.05
    assert np.isnan(cf.mi_with_null(b[:1], y, None)["p"])


def test_compact_joint_tertiles() -> None:
    assert cf.compact([5, -1, 9, 5]).tolist() == [0, -1, 1, 0]
    assert cf.joint([0, 1, 2, -1], [1, 0, 1, 0]).tolist() == [1, 2, 5, -1]
    t = cf.tertiles(np.arange(9.0), np.ones(9, bool))
    assert t.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert (cf.tertiles([1.0, 2.0], [True, True]) == -1).all()


def test_acceleration_of_ramp() -> None:
    sp = np.arange(200.0) * 0.5  # 0.5 units per frame at 10 fps -> 5 units/s^2
    sp[50] = np.nan
    a = cf.acceleration(sp, 10.0, 0.2)
    assert np.isnan(a[50])
    assert np.nanmedian(a[20:180]) == pytest.approx(5.0, rel=1e-3)


# ---------------------------------------------------------- travel/frames


def test_travel_direction_recovers_moves(walk: dict) -> None:
    td = cf.travel_direction(walk["x"], walk["y"], walk["cells"], MAZE, fps=10.0, sigma_s=0.1)
    ok = td >= 0
    assert ok.mean() > 0.8
    assert (td[ok] == walk["dirs"][ok]).mean() > 0.9
    nan_x = walk["x"].copy()
    nan_x[:5] = np.nan
    assert (cf.travel_direction(nan_x, walk["y"], walk["cells"], MAZE, 10.0)[:5] == -1).all()


def test_allowed_dirs_and_root() -> None:
    al = cf.allowed_dirs(MAZE)
    assert al[IDX[(1, 1)]].tolist() == [False, True, False, True]  # N-S corridor
    assert al[IDX[(1, 0)]].all()  # T-junction: E, W, N plus S when arriving from N
    assert MAZE.cell_list[cf.maze_root(MAZE)] == (3, 2)


def test_turn_code() -> None:
    assert cf.turn_code(1, 1) == 2
    assert cf.turn_code(1, 2) == 1  # heading +y, turn to -x = left
    assert cf.turn_code(1, 0) == 0
    assert cf.turn_code(1, 3) == -1


def test_state_labels_known_states() -> None:
    sl = cf.state_labels(MAZE)
    n, s = 1, 3
    assert sl.root_out[IDX[(1, 1)], s] == 1 and sl.root_out[IDX[(1, 1)], n] == 0
    assert sl.root_out[IDX[(1, 3)], n] == 1
    assert sl.toward_deadend[IDX[(1, 1)], s] == 1
    # heading N up (1,1): junction (1,2) has straight (N) and right (E) -> side arm right
    assert sl.ego_side[IDX[(1, 1)], n] == 0
    # heading S down (1,3): straight (S) and E, which is left of S
    assert sl.ego_side[IDX[(1, 3)], s] == 1
    # heading S down (3,3) into (3,2): T-end (E and W only) -> undefined
    assert sl.ego_side[IDX[(3, 3)], s] == -1
    # heading toward a dead end: undefined
    assert sl.ego_side[IDX[(1, 0)], 2] == -1


def test_upcoming_turn_labels_approach() -> None:
    seq = [(3, 4), (3, 3), (3, 2), (2, 2), (1, 2), (1, 1)]
    cells = np.repeat([IDX[c] for c in seq], 3)
    t = cf.upcoming_turn(cells, MAZE)
    # (3,3)->(3,2)->(2,2): heading S, leaving W = right of S -> 0
    assert t[3:9].tolist() == [0] * 6
    # (2,2)->(1,2)->(1,1): heading W, leaving S = left of W -> 1
    assert t[9:15].tolist() == [1] * 6
    assert (t[:3] == -1).all() and (t[15:] == -1).all()


def test_axis_and_turn_locations(walk: dict) -> None:
    n = walk["x"].size
    loc, lab = cf.axis_locations(walk["cells"], walk["dirs"], np.ones(n, bool))
    assert ((loc >= 0) == (lab >= 0)).all() and set(np.unique(lab)) == {0, 1}
    assert (loc[loc >= 0] // 2 == walk["cells"][loc >= 0]).all()
    loc0, _ = cf.axis_locations(walk["cells"], walk["dirs"], np.zeros(n, bool))
    assert (loc0 == -1).all()
    turn = cf.upcoming_turn(walk["cells"], MAZE)
    tl, tb = cf.turn_locations(walk["cells"], walk["dirs"], turn, np.ones(n, bool))
    assert set(np.unique(tb[tb >= 0])) == {0, 1}
    assert (tl[tl >= 0] % 4 == walk["dirs"][tl >= 0]).all()


def test_frame_signs() -> None:
    fs = cf.frame_signs(MAZE)
    assert set(fs) == {"allo", "root", "deadend", "ego_side"}
    allo_s, allo_g = fs["allo"]
    assert int((allo_s != 0).sum()) == 30 and set(allo_g) == {0, 1}
    assert int((fs["root"][0] != 0).sum()) == 23
    assert int((fs["deadend"][0] != 0).sum()) == 19
    assert int((fs["ego_side"][0] != 0).sum()) == 0
    # (1,1) N-S corridor: +y (north) is toward the root -> sigma -1
    assert fs["root"][0][IDX[(1, 1)] * 2 + 1] == -1.0


def test_lolo_score() -> None:
    sigma = np.array([1.0, -1.0, 1.0, 0.0])
    group = np.zeros(4, int)
    valid = np.ones(4, bool)
    p = np.array([[0.5, -0.5, 0.5, 0.2], [0.5, 0.5, 0.5, 0.0]])
    sc = cf.lolo_score(p, valid, sigma, group)
    assert sc[0] == pytest.approx(1.5 / 1.7)
    # cell 1: sigma*p = [.5, -.5, .5]; locations 0 and 2 tie (prediction 0),
    # location 1 is predicted wrong -> -0.5 / 1.5
    assert sc[1] == pytest.approx(-1 / 3)
    assert np.isnan(cf.lolo_score(np.zeros((1, 4)), valid, sigma, group))[0]


# ---------------------------------------------------------- consistency


def test_passes() -> None:
    pid, pl, pb = cf.passes([0, 0, 0, -1, 0, 1, 1], [1, 1, 0, -1, 0, 0, 0])
    assert pid.tolist() == [0, 0, 1, -1, 2, 3, 3]
    assert pl.tolist() == [0, 0, 0, 1] and pb.tolist() == [1, 0, 0, 0]


def _coders(walk: dict, rng: np.random.Generator) -> tuple[np.ndarray, tuple, tuple]:
    """Poisson cells: root-outward, +x (allo), inbound (toward dead end), upcoming-left, none."""
    n = walk["x"].size
    turn = cf.upcoming_turn(walk["cells"], MAZE)
    sl = cf.state_labels(MAZE)
    outward = sl.root_out[walk["cells"], walk["dirs"]] == 1
    inbound = sl.toward_deadend[walk["cells"], walk["dirs"]] == 1
    east = walk["dirs"] == 0
    rate = np.stack(
        [
            0.3 + 1.2 * outward,
            0.3 + 1.2 * east,
            0.3 + 1.2 * inbound,
            0.3 + 1.2 * (turn == 1),
            np.full(n, 0.6),
        ]
    )
    m = np.ones(n, bool)
    return (
        rng.poisson(rate).astype(float),
        cf.axis_locations(walk["cells"], walk["dirs"], m),
        cf.turn_locations(walk["cells"], walk["dirs"], turn, m),
    )


def test_frame_scores_identify_frame(walk: dict) -> None:
    rng = np.random.default_rng(3)
    sig, (loc, lab), (tl, tb) = _coders(walk, rng)
    t = cf.frame_scores(sig, loc, lab, cf.frame_signs(MAZE), 100, rng)
    sc = t.pivot(index="roi", columns="frame", values="score")
    pv = t.pivot(index="roi", columns="frame", values="score_p")
    # root-outward cell: root wins clearly over allo
    assert pv.loc[0, "root"] < 0.02 and sc.loc[0, "root"] > sc.loc[0, "allo"] + 0.3
    # allocentric cell: allo wins clearly over root
    assert pv.loc[1, "allo"] < 0.02 and sc.loc[1, "allo"] > sc.loc[1, "root"] + 0.3
    # inbound cell: detected in the dead-end frame, beats allo
    assert pv.loc[2, "deadend"] < 0.02 and sc.loc[2, "deadend"] > sc.loc[2, "allo"] + 0.3
    assert t[(t.roi == 2) & (t.frame == "deadend")].mean_signed.iloc[0] > 0
    # null cell
    assert (pv.loc[4] > 0.02).all()
    assert (t[t.frame == "ego_side"].n_def == 0).all()
    tt = cf.frame_scores(
        sig,
        tl,
        tb,
        {"ego_turn": (np.ones(4 * MAZE.n_cells), np.zeros(4 * MAZE.n_cells, int))},
        100,
        rng,
    )
    assert tt.score_p[3] < 0.02 and tt.score_p[4] > 0.02 and tt.score_p[0] > 0.02


def test_frame_scores_degenerate() -> None:
    rng = np.random.default_rng(0)
    fs = {"x": (np.ones(3), np.zeros(3, int))}
    out = cf.frame_scores(np.ones((2, 10)), -np.ones(10, int), -np.ones(10, int), fs, 5, rng)
    assert (out.n_loc == 0).all() and "score" not in out
    loc = np.r_[np.zeros(10, int), np.ones(10, int)]
    lab = np.tile(np.repeat([0, 1], 5), 2)
    sig = np.random.default_rng(1).poisson(1.0, (2, 20)).astype(float)
    out = cf.frame_scores(sig, loc, lab, fs, 0, rng, min_passes=1, min_frames=1)
    assert out.n_loc.eq(2).all() and out.score_p.isna().all() and out.score.notna().all()


def test_residualise() -> None:
    sig = np.array([[1.0, 3.0, 10.0, 20.0, 5.0]])
    r = cf.residualise(sig, [0, 0, 1, 1, -1])
    assert r.tolist() == [[-1.0, 1.0, -5.0, 5.0, 5.0]]


# ------------------------------------------------------------ stability


def test_state_map_and_residual() -> None:
    sig = np.array([[1.0, 3.0, 5.0, 7.0, 100.0]])
    st_ = np.array([0, 0, 1, 1, 2])
    m = cf.state_map(sig, st_, np.ones(5, bool), 8, min_frames=2)
    assert m[0, :2].tolist() == [2.0, 6.0] and np.isnan(m[0, 2:]).all()
    r = cf.direction_residual(m)
    assert r[0, :2].tolist() == [-2.0, 2.0] and np.isnan(r[0, 4:]).all()


def test_rowwise_spearman() -> None:
    a = np.array([[1.0, 2, 3, 4, 5, np.nan], [1, 1, 1, 1, 1, 1]])
    b = np.array([[2.0, 4, 6, 8, 10, 1], [1, 2, 3, 4, 5, 6]])
    r = cf.rowwise_spearman(a, b)
    assert r[0] == pytest.approx(1.0) and np.isnan(r[1])


def test_epoch_parity() -> None:
    lt = np.repeat([1, 0, 1, 0, 1, 0], 2).astype(bool)
    assert cf.epoch_parity(lt).tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    assert cf.epoch_parity([]).size == 0


def test_wilcoxon_and_animal_test() -> None:
    w = cf.wilcoxon_summary([1.0, 2, 3, 4, 5, 6, 7, np.nan])
    assert w["n"] == 7 and w["rank_biserial"] == pytest.approx(1.0) and w["p"] < 0.05
    assert np.isnan(cf.wilcoxon_summary([0.0, 1.0])["p"])
    df = pd.DataFrame({"animal_id": list("aabbccdd"), "v": [1, 2, 3, 4, -1, 5, 6, 7.0]})
    r = cf.animal_test(df, "v")
    assert r["n"] == 4 and r["n_cells"] == 8 and r["frac_cells_pos"] == pytest.approx(7 / 8)
