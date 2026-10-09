"""Tests for ``hm2p.analysis.maze_dynamics`` (small synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis import maze_dynamics as md
from hm2p.maze.topology import build_rose_maze

MAZE = build_rose_maze()
FPS = 10.0
IDX = MAZE.cell_to_idx
DEPTH = md.tree_depth(MAZE)


# ---------------------------------------------------------------------------
# Synthetic session: random walk on the maze graph with a position and a
# travel-direction code; the position code degrades with time in the dark
# (``drift``) and the direction code leads behaviour by ``lead_s``.
# ---------------------------------------------------------------------------


def _session(
    n_frames: int = 7200,
    drift: float = 0.0,
    lead_s: float = 0.0,
    seed: int = 0,
    n_per_cell: int = 2,
    n_dir: int = 12,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    cells: list[int] = []
    hd: list[float] = []
    spd: list[float] = []
    bdir: list[int] = []
    cur, prev = IDX[(3, 2)], None
    while len(cells) < n_frames:
        nbs = [IDX[c] for c in MAZE.adj[MAZE.cell_list[cur]]]
        fwd = [c for c in nbs if c != prev]
        nxt = int(rng.choice(fwd)) if fwd and rng.random() > 0.15 else int(rng.choice(nbs))
        k = int(rng.integers(10, 17))
        a = MAZE.cell_list[prev] if prev is not None else MAZE.cell_list[cur]
        b, c = MAZE.cell_list[cur], MAZE.cell_list[nxt]
        h_in = np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])) if prev is not None else 0.0
        h_out = np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0]))
        d_in = int(DEPTH[cur] > DEPTH[prev]) if prev is not None else int(DEPTH[nxt] > DEPTH[cur])
        d_out = int(DEPTH[nxt] > DEPTH[cur])
        half = k // 2
        cells += [cur] * k
        hd += [h_in] * half + [h_out] * (k - half)
        bdir += [d_in] * half + [d_out] * (k - half)
        s = np.full(k, 12.0)
        if nxt == prev or rng.random() < 0.3:
            s[half - 1 : half + 1] = 1.0
        spd += list(s)
        prev, cur = cur, nxt
    cells_a = np.asarray(cells[:n_frames])
    centres = np.array([(c[0] + 0.5, c[1] + 0.5) for c in MAZE.cell_list])
    xy = centres[cells_a] + rng.normal(0, 0.05, (n_frames, 2))
    t = np.arange(n_frames) / FPS
    light = (np.floor(t / 60.0) % 2) == 1
    t_ep = t % 60.0
    rep = cells_a.copy()
    wrong = (~light) & (rng.random(n_frames) < drift * t_ep / 60.0)
    rep[wrong] = rng.integers(0, MAZE.n_cells, wrong.sum())
    bdir_a = np.asarray(bdir[:n_frames])
    lead = int(round(lead_s * FPS))
    ndir = np.concatenate([bdir_a[lead:], np.repeat(bdir_a[-1], lead)]) if lead else bdir_a
    n_cells = MAZE.n_cells * n_per_cell + 2 * n_dir
    sig = rng.normal(0, 0.3, (n_cells, n_frames))
    for k in range(MAZE.n_cells):
        sig[k * n_per_cell : (k + 1) * n_per_cell] += 2.0 * (rep == k)
    base = MAZE.n_cells * n_per_cell
    sig[base : base + n_dir] += 1.5 * (ndir == 1)
    sig[base + n_dir :] += 1.5 * (ndir == 0)
    return {
        "signal": sig,
        "x": xy[:, 0],
        "y": xy[:, 1],
        "speed": np.asarray(spd[:n_frames]),
        "hd": np.asarray(hd[:n_frames]) + rng.normal(0, 5, n_frames),
        "ahv": rng.normal(0, 30, n_frames),
        "light": light,
        "valid": np.ones(n_frames, dtype=bool),
        "cells": cells_a,
    }


def _args(s: dict[str, np.ndarray]) -> tuple:
    return (
        s["signal"],
        s["x"],
        s["y"],
        s["speed"],
        s["hd"],
        s["ahv"],
        s["light"],
        s["valid"],
        FPS,
    )


@pytest.fixture(scope="module")
def drift_table() -> pd.DataFrame:
    return md.frame_table(
        *_args(_session(drift=0.9, seed=1)), params=md.DynParams(n_blocks=5, C=0.1, max_iter=100)
    )


@pytest.fixture(scope="module")
def flat_table() -> pd.DataFrame:
    return md.frame_table(
        *_args(_session(drift=0.0, seed=2)), params=md.DynParams(n_blocks=5, C=0.1, max_iter=100)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_boxcar_centred_and_causal() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.allclose(md.boxcar(x, 3), [0.5, 1.0, 2.0, 3.0, 3.5])
    assert np.allclose(md.boxcar(x, 3, causal=True), [0.0, 0.5, 1.0, 2.0, 3.0])


def test_boxcar_nan_and_2d() -> None:
    x = np.array([[np.nan, np.nan, 2.0], [1.0, 1.0, 1.0]])
    out = md.boxcar(x, 1)
    assert np.isnan(out[0, 0]) and out[0, 2] == 2.0
    assert md.boxcar(x, 3).shape == (2, 3)


@given(st.lists(st.floats(-100, 100), min_size=1, max_size=40), st.integers(1, 10))
@settings(max_examples=40, deadline=None)
def test_boxcar_bounded(vals: list[float], w: int) -> None:
    x = np.asarray(vals)
    for causal in (False, True):
        out = md.boxcar(x, w, causal)
        assert np.all(out >= x.min() - 1e-9) and np.all(out <= x.max() + 1e-9)


def test_tree_depth_root_is_centre() -> None:
    assert DEPTH[IDX[(3, 2)]] == 0
    assert DEPTH.max() == 5


def test_time_in_epoch() -> None:
    lo = np.array([0, 0, 1, 1, 1, 0], dtype=bool)
    t, e = md.time_in_epoch(lo, 1.0)
    assert np.allclose(t, [0, 1, 0, 1, 2, 0])
    assert list(e) == [0, 0, 1, 1, 1, 2]


def test_frame_directions_outward_inward_and_uturn() -> None:
    a, b, c = IDX[(3, 2)], IDX[(4, 2)], IDX[(5, 2)]
    cells = np.array([a] * 3 + [b] * 3 + [c] * 3 + [b] * 3 + [a] * 3)
    d, arr, lea = md.frame_directions(cells, MAZE)
    assert list(d[3:6]) == [1, 1, 1]  # a -> b -> c outward
    assert list(d[6:9]) == [-1, -1, -1]  # U-turn at c
    assert list(d[9:12]) == [0, 0, 0]  # inward
    assert arr[6] == 1 and lea[6] == 0


def test_behaviour_dir_frames_shape() -> None:
    out = md.behaviour_dir_frames(
        np.array([0.0, 90.0]), np.ones(2), np.ones(2), np.array([1, -1]), 23
    )
    assert out.shape == (48, 2)
    assert out[2, 0] == 0.0 and out[3, 0] == 1.0 and np.all(out[:46, 1] == 0)


def test_running_mask_requires_bout() -> None:
    s = np.r_[np.zeros(20), np.full(20, 10.0), np.zeros(5), np.full(3, 10.0)]
    m = md.running_mask(s, np.ones(s.size, bool), 10.0)
    assert m[20:40].all() and not m[45:].any()


def test_frame_blocks() -> None:
    b = md.frame_blocks(10, 5)
    assert list(b) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_cv_proba_separable() -> None:
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 100)
    rng.shuffle(y)
    x = np.eye(3)[y] * 3 + rng.normal(0, 0.3, (300, 3))
    pr, cl = md.cv_proba(x, y, np.ones(300, bool), n_blocks=5)
    assert list(cl) == [0, 1, 2]
    assert (cl[pr.argmax(1)] == y).mean() > 0.95
    assert np.allclose(pr.sum(1), 1)


def test_cv_proba_single_class_returns_nan() -> None:
    pr, cl = md.cv_proba(np.ones((10, 2)), np.zeros(10, int), np.ones(10, bool))
    assert cl.size == 1 and np.isnan(pr).all()


def test_position_metrics() -> None:
    dist = MAZE.dist.astype(float)
    cl = np.array([0, 1])
    pr = np.array([[0.9, 0.1], [0.2, 0.8], [np.nan, np.nan]])
    m = md.position_metrics(pr, cl, np.array([0, 0, 0]), dist)
    assert m["err"][0] == 0 and m["err"][1] == dist[0, 1]
    assert m["ptrue"][1] == pytest.approx(0.2)
    assert np.isnan(m["err"][2])


def test_cell_demean() -> None:
    out = md.cell_demean([1.0, 3.0, 5.0], [0, 0, 1], [True, True, True])
    assert np.allclose(out, [-1, 1, 0])


def test_pv_similarity_matching_template() -> None:
    rng = np.random.default_rng(0)
    cells = np.tile(np.repeat([0, 1, 2], 10), 20)
    z = np.eye(3)[cells] + rng.normal(0, 0.1, (cells.size, 3))
    r, spec = md.pv_similarity(z, cells, np.ones(cells.size, bool), 4, 0)
    assert np.nanmean(r) > 0.8 and np.nanmean(spec) > 0.8


# ---------------------------------------------------------------------------
# Question 1
# ---------------------------------------------------------------------------


def test_drift_positive_slope_in_dark_only(drift_table: pd.DataFrame) -> None:
    sl = md.drift_slopes(drift_table).set_index(["condition", "metric"])
    assert sl.loc[("dark", "neural_pos_err"), "prho"] > 0.15
    assert abs(sl.loc[("light", "neural_pos_err"), "prho"]) < 0.08
    assert sl.loc[("dark", "pv_r"), "prho"] < -0.15


def test_no_drift_flat_slopes(flat_table: pd.DataFrame) -> None:
    sl = md.drift_slopes(flat_table).set_index(["condition", "metric"])
    assert abs(sl.loc[("dark", "neural_pos_err"), "prho"]) < 0.08


def test_epoch_profile_rises_in_dark(drift_table: pd.DataFrame) -> None:
    prof = md.epoch_profile(drift_table, ("neural_pos_err",), 10.0, 60.0)
    dk = prof[prof["condition"] == "dark"].set_index("bin")["neural_pos_err"]
    assert dk.loc[50.0] > dk.loc[0.0] + 0.5


def test_switch_correction_after_lights_on(drift_table: pd.DataFrame) -> None:
    sw = md.switch_table(drift_table, ("neural_pos_err",), FPS)
    summ = md.switch_summary(sw, ("neural_pos_err",)).set_index("switch")
    assert summ.loc["dark_to_light", "median_correction"] > 0.5
    assert summ.loc["light_to_dark", "median_correction"] < 0.3


def test_switch_table_windows_disjoint() -> None:
    n = 1200
    df = pd.DataFrame(
        {
            "running": np.ones(n, bool),
            "epoch": np.repeat([0, 1], 600),
            "light": np.repeat([False, True], 600),
            "t_epoch": np.tile(np.arange(600) / 10.0, 2),
            "m": np.r_[np.arange(600) / 600.0, np.zeros(600)],
        }
    )
    sw = md.switch_table(df, ("m",), 10.0)
    assert len(sw) == 1 and sw["switch"].iloc[0] == "dark_to_light"
    assert sw["m__pre_late"].iloc[0] > sw["m__pre_mid"].iloc[0] > sw["m__pre_early"].iloc[0]
    assert sw["m__post_early"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Question 2
# ---------------------------------------------------------------------------


def test_head_turn_frame() -> None:
    hd = np.r_[np.zeros(5), np.full(5, 180.0)]
    assert md.head_turn_frame(hd, 0, 10, 0.0, 180.0) == 5
    assert md.head_turn_frame(np.zeros(10), 0, 10, 0.0, 180.0) == -1


def test_turn_events_reversal_and_pass() -> None:
    a, b, c = IDX[(3, 2)], IDX[(4, 2)], IDX[(5, 2)]
    cells = np.array([a] * 15 + [b] * 10 + [a] * 15 + [b] * 10 + [c] * 15)
    hd = np.r_[np.zeros(30), np.full(15, 180.0), np.zeros(25)]
    hd = np.r_[np.zeros(20), np.full(20, 180.0), np.zeros(25)][: cells.size]
    spd = np.full(cells.size, 10.0)
    spd[18:21] = 0.5
    ev = md.turn_events(cells, hd, spd, np.zeros(cells.size), np.zeros(cells.size), 10.0, MAZE)
    kinds = list(ev["kind"])
    assert "reversal" in kinds and "pass" in kinds
    r = ev[ev["kind"] == "reversal"].iloc[0]
    assert r["dir_pre"] == 1 and r["dir_post"] == 0 and r["t_headturn"] == 20


def test_match_passes_one_to_one() -> None:
    ev = pd.DataFrame(
        {
            "kind": ["reversal", "pass", "pass", "reversal"],
            "min_speed": [1, 1.1, 5, 4.9],
            "decel": [1, 1, 1, 1],
            "peak_abs_ahv": [1, 1, 1, 1],
            "dur_s": [1, 1, 1, 1],
            "light_frac": [0, 0, 0, 0],
        }
    )
    r = md.match_passes(ev)
    assert list(r["match"]) == [1, 2]


def test_aligned_and_p_new() -> None:
    x = np.arange(10, dtype=float) / 10
    a = md.aligned(x, np.array([0, 5]), 2)
    assert np.isnan(a[0, 0]) and a[1, 2] == 0.5
    pn = md.p_new(x, np.array([5]), np.array([0]), 1)
    assert np.allclose(pn, [[0.6, 0.5, 0.4]])


def test_half_rise_time_sigmoid() -> None:
    t = np.arange(-30, 31) / 10
    y = 1 / (1 + np.exp(-(t + 0.5) * 8))  # half-rise at -0.5 s
    assert md.half_rise_time(y, t) == pytest.approx(-0.5, abs=0.05)
    assert np.isnan(md.half_rise_time(np.zeros_like(t), t))


def test_lagged_xcorr_peak() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    b = np.roll(a, 3)
    assert md.lagged_xcorr_peak(a, b, np.ones(500, bool), 6) == 3


@pytest.mark.parametrize("lead_s", [0.0, 0.5])
def test_reversal_lead_recovered(lead_s: float) -> None:
    s = _session(n_frames=9000, lead_s=lead_s, seed=3, n_dir=15)
    p = md.DynParams(n_blocks=5, rev_smooth_s=0.1, C=0.1, max_iter=100)
    res = md.reversal_session(*_args(s), params=p)
    summ = res["summary"]
    assert summ["n_reversals"] >= 20
    # Head-turn-aligned half-rise of the neural decoded direction: about -lead.
    assert summ["neural_t_half_headturn"] == pytest.approx(-lead_s, abs=0.2)
    assert summ["behaviour_t_half_headturn"] == pytest.approx(0.0, abs=0.2)
    if lead_s > 0:
        assert summ["lead_vs_behaviour_headturn"] == pytest.approx(lead_s, abs=0.25)
        assert summ["neural_anticip"] > 0.1


# ---------------------------------------------------------------------------
# Question 3
# ---------------------------------------------------------------------------


def test_junction_departures_labels() -> None:
    j, a, b = IDX[(3, 2)], IDX[(2, 2)], IDX[(4, 2)]
    a2 = IDX[(1, 2)]  # T-junction beyond (2,2)
    b2 = IDX[(5, 2)]
    seq = [a2, a, j, b, b2, b, j, b, b2]
    cells = np.repeat(seq, 5)
    dep = md.junction_departures(cells, MAZE, 10.0, n_back=4)
    assert list(dep["junction"]) == [j, b2, j]  # (5, 2) is also a T-junction
    assert dep["uturn"].iloc[1] and dep["repeat"].iloc[1] == 1
    dep = dep[dep["junction"] == j]
    assert dep["repeat"].iloc[0] == 0  # first visit to b2 is new
    assert dep["uturn"].iloc[1] and dep["repeat"].iloc[1] == 1


def test_departure_features_and_effects() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame(
        {
            "neural_pos_ptrue": rng.random(n),
            "neural_pos_err": rng.random(n),
            "neural_pos_excess": rng.random(n),
            "behaviour_pos_ptrue": rng.random(n),
            "speed": rng.random(n),
            "abs_ahv": rng.random(n),
            "light": np.repeat([False, True], n // 2),
        }
    )
    rep = np.tile([0, 1], 30)
    departs = np.r_[np.linspace(50, 1900, 30), np.linspace(2100, 3900, 30)].astype(int)
    dep = pd.DataFrame({"junction": np.tile([1, 1, 2, 2], 15), "depart": departs, "repeat": rep})
    for k, d in enumerate(departs):
        df.loc[d - 20 : d - 1, "neural_pos_ptrue"] = 0.9 if rep[k] else 0.1
    feat = md.departure_features(dep, df, 10.0)
    eff = md.departure_effects(feat).set_index(["condition", "metric"])
    assert eff.loc[("dark", "neural_pos_ptrue"), "diff"] == pytest.approx(0.8)
    assert eff.loc[("all", "neural_pos_ptrue"), "prho"] > 0.8


# ---------------------------------------------------------------------------
# Question 4
# ---------------------------------------------------------------------------


def test_epoch_exploration_and_correlations(flat_table: pd.DataFrame) -> None:
    ep = md.epoch_exploration(flat_table, MAZE, FPS, np.random.default_rng(0), n_sims=50)
    assert len(ep) == 12
    assert ep["coverage_z"].notna().all()
    assert (ep["n_unique"] <= 23).all()
    cor = md.epoch_correlations(ep, min_epochs=4)
    assert set(cor["condition"]) == {"light", "dark", "all"}
    assert cor["prho"].abs().max() <= 1.0


def test_epoch_correlation_detects_association() -> None:
    rng = np.random.default_rng(0)
    n = 30
    z = rng.normal(size=n)
    ep = pd.DataFrame(
        {
            "light": np.tile([False, True], n // 2),
            "coverage_z": z,
            "speed": rng.random(n),
            "frac_running": rng.random(n),
            "t_start_s": np.arange(n) * 60.0,
            "neural_pos_err": -z + rng.normal(0, 0.1, n),
            "neural_pos_excess": rng.random(n),
            "neural_dir_ptrue": rng.random(n),
            "pv_spec": rng.random(n),
        }
    )
    cor = md.epoch_correlations(ep).set_index(["condition", "metric"])
    assert cor.loc[("all", "neural_pos_err"), "prho"] < -0.8


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_wilcoxon_summary() -> None:
    out = md.wilcoxon_summary(np.arange(1, 11, dtype=float))
    assert out["n"] == 10 and out["r_rb"] == 1.0 and out["p"] < 0.01
    assert np.isnan(md.wilcoxon_summary([1.0, np.nan])["p"])


def test_animal_test_uses_animal_medians() -> None:
    df = pd.DataFrame({"animal_id": list("aabbccdd"), "v": [1, 3, 2, 2, 5, 5, -1, 7.0]})
    out = md.animal_test(df, "v")
    assert out["animal_n"] == 4 and out["session_n"] == 8
    assert out["animal_median"] == pytest.approx(2.5)


def test_bh_fdr() -> None:
    adj = md.bh_fdr([0.01, 0.04, np.nan, 0.03])
    assert np.isnan(adj[2])
    assert adj[0] == pytest.approx(0.03) and adj[1] == pytest.approx(0.04)


def test_position_shift_null_observed_above_null() -> None:
    s = _session(n_frames=3000, seed=5)
    cells = md.session_cells(s["x"], s["y"], s["valid"], MAZE)
    tr = md.running_mask(s["speed"], s["valid"], FPS) & (cells >= 0)
    vals = md.position_shift_null(
        s["signal"], cells, tr, [0, 1000, 2000], 5, 5, 20, C=0.1, max_iter=100
    )
    assert vals[0] > 0.5 and vals[1:].max() < 0.2
