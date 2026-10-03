"""Tests for ``hm2p.analysis.maze_running`` (small synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis import maze_running as mr
from hm2p.maze.topology import build_rose_maze

FPS = 10.0
MAZE = build_rose_maze()
IDX = MAZE.cell_to_idx


def _xy(cells: list[tuple[int, int]], frames_per_cell: int) -> tuple[np.ndarray, np.ndarray]:
    xy = np.repeat(np.array(cells, dtype=float) + 0.5, frames_per_cell, axis=0)
    return xy[:, 0], xy[:, 1]


# ---------------------------------------------------------------------------
# Maze helpers
# ---------------------------------------------------------------------------


def test_dwell_filter_removes_short_visits() -> None:
    c = np.array([1, 1, 1, 2, 1, 1, 3, 3, 3])
    out = mr.dwell_filtered_cells(c, 2)
    assert out.tolist() == [1, 1, 1, -1, 1, 1, 3, 3, 3]
    assert mr.dwell_filtered_cells(c, 1).tolist() == c.tolist()
    assert mr.dwell_filtered_cells(np.zeros(0, int), 3).size == 0


def test_arm_ids_and_groups() -> None:
    arms = mr.arm_ids(MAZE)
    assert arms[IDX[(3, 3)]] == arms[IDX[(3, 4)]]
    assert arms[IDX[(1, 1)]] != arms[IDX[(1, 0)]]
    junction_arms = [arms[IDX[c]] for c in MAZE.junctions]
    assert len(set(junction_arms)) == len(junction_arms)
    groups = mr.node_groups(MAZE)
    assert groups[IDX[(0, 0)]] == "dead_end" and groups[IDX[(1, 0)]] == "junction"


def test_dead_end_distance() -> None:
    d = mr.dead_end_distance(MAZE)
    assert d[IDX[(0, 0)]] == 0 and d[IDX[(1, 0)]] == 1 and d[IDX[(1, 1)]] == 2


@pytest.mark.parametrize(
    ("s", "e", "moved", "expected"),
    [
        ("junction", "dead_end", True, "into_dead_end"),
        ("dead_end", "corridor", True, "out_of_dead_end"),
        ("junction", "corridor", True, "junction_to_junction"),
        ("dead_end", "dead_end", True, "dead_end_to_dead_end"),
        ("junction", "junction", False, "no_transition"),
        (None, "junction", True, "unknown"),
    ],
)
def test_classify_run(s: str | None, e: str, moved: bool, expected: str) -> None:
    assert mr.classify_run(s, e, moved) == expected


def test_prior_edge_counts() -> None:
    a, b, c = IDX[(1, 0)], IDX[(1, 1)], IDX[(5, 4)]
    out = mr.prior_edge_counts([a, b, a, b, c], MAZE)
    assert out[:3].tolist() == [0, 0, 1]
    assert np.isnan(out[3])  # non-adjacent jump


def test_light_epoch_starts() -> None:
    lo = np.array([1, 1, 0, 0, 0, 1], bool)
    assert mr.light_epoch_starts(lo).tolist() == [0, 0, 2, 2, 2, 5]
    assert mr.light_epoch_starts(np.zeros(0, bool)).size == 0


def test_light_condition() -> None:
    assert mr.light_condition(1.0) == "light"
    assert mr.light_condition(0.0) == "dark"
    assert mr.light_condition(0.5) == "mixed"


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def _session() -> tuple[np.ndarray, ...]:
    # junction (1,0) -> dead end (0,0) -> back (1,0) -> corridor (1,1) -> (1,2)
    path = [(1, 0), (0, 0), (1, 0), (1, 1), (1, 2)]
    x, y = _xy(path, 10)
    n = x.size
    speed = np.full(n, 10.0)
    light = np.ones(n, bool)
    light[25:] = False
    return x, y, speed, light


def test_annotate_bouts_into_and_out() -> None:
    x, y, speed, light = _session()
    on = np.array([5, 15, 25])
    off = np.array([15, 25, 45])
    b = mr.annotate_bouts(on, off, x, y, speed, light, FPS, MAZE, min_dwell_frames=1)
    assert b["run_type"].tolist() == ["into_dead_end", "out_of_dead_end", "junction_to_junction"]
    assert b.loc[0, "novel"] and b.loc[0, "n_novel"] == 1
    assert b.loc[1, "repeat"] and not b.loc[1, "novel"]
    assert b.loc[2, "path_cells"] == 2 and b.loc[2, "novel_frac"] == 1.0
    assert b.loc[0, "delta_dead_end"] == -1 and b.loc[1, "delta_dead_end"] == 1
    assert b.loc[0, "light_cond"] == "light" and b.loc[2, "light_cond"] == "dark"
    assert b.loc[0, "route"] == f"{IDX[(1, 0)]}->{IDX[(0, 0)]}"
    assert b.loc[0, "duration_s"] == pytest.approx(1.0)
    # first edge traversal (1,0)->(0,0) has no prior; (1,0)->(1,1) in bout 3 none either
    assert b.loc[0, "edge_prior_traversals"] == 0


def test_annotate_first_arm_entry_resets_with_light_epoch() -> None:
    path = [(1, 0), (0, 0), (1, 0), (0, 0)]
    x, y = _xy(path, 10)
    n = x.size
    light = np.ones(n, bool)
    light[20:] = False  # dark epoch begins before the second entry
    b = mr.annotate_bouts(
        [5, 25], [15, 35], x, y, np.full(n, 10.0), light, FPS, MAZE, min_dwell_frames=1
    )
    assert b["first_arm_entry"].tolist() == [True, True]
    light[:] = True
    b2 = mr.annotate_bouts(
        [5, 25], [15, 35], x, y, np.full(n, 10.0), light, FPS, MAZE, min_dwell_frames=1
    )
    assert b2["first_arm_entry"].tolist() == [True, False]


def test_annotate_unknown_and_no_transition() -> None:
    x = np.full(30, np.nan)
    b = mr.annotate_bouts([0], [10], x, x, np.ones(30), np.ones(30, bool), FPS, MAZE)
    assert b.loc[0, "run_type"] == "unknown"
    x2, y2 = _xy([(1, 0)], 30)
    b2 = mr.annotate_bouts([0], [10], x2, y2, np.ones(30), np.ones(30, bool), FPS, MAZE)
    assert b2.loc[0, "run_type"] == "no_transition" and b2.loc[0, "route"] == ""


def test_summarise_runs() -> None:
    x, y, speed, light = _session()
    b = mr.annotate_bouts([5, 15, 25], [15, 25, 45], x, y, speed, light, FPS, MAZE, 1)
    s = mr.summarise_runs(b)
    assert s["n_bouts"].sum() == 3
    assert {"light_cond", "run_type", "n_novel", "median_path_cells"} <= set(s.columns)
    assert mr.summarise_runs(b.iloc[:0]).empty


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def test_rank_residualiser_removes_covariate() -> None:
    rng = np.random.default_rng(0)
    cov = rng.random(50)
    m = mr.rank_residualiser(cov)
    r = mr._norm_ranks(cov)
    assert np.allclose(m @ r, 0, atol=1e-10)
    assert np.allclose(m @ np.ones(50), 0, atol=1e-10)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 99), st.integers(1, 5))
def test_shifted_bout_means_matches_roll(shift: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.random(100)
    on, off = np.array([0, 20, 90]), np.array([10, 35, 100])
    got = mr.shifted_bout_means(x, on, off, [shift])[0]
    xr = np.roll(x, shift)
    exp = [xr[a:b].mean() for a, b in zip(on, off, strict=True)]
    assert np.allclose(got, exp)


def test_shifted_bout_means_nan_handling() -> None:
    x = np.arange(20, dtype=float)
    x[0:5] = np.nan
    got = mr.shifted_bout_means(x, [0, 5], [5, 10], [0])[0]
    assert got[1] == pytest.approx(7.0) and got[0] == pytest.approx(7.0)


def test_split_halves_and_reliability() -> None:
    lab = np.array([0, 1, 0, -1, 1, 0])
    assert mr._split_halves(lab).tolist() == [0, 0, 1, -1, 1, 0]
    labels = np.repeat(np.arange(6), 4)
    halves = mr._split_halves(labels)
    vals = labels.astype(float) + np.tile([0.0, 0.1, 0.0, 0.1], 6)
    assert mr.split_half_reliability(vals, labels, halves) == pytest.approx(1.0)
    assert np.isnan(mr.split_half_reliability(vals, labels, halves, min_labels=10))
    assert np.isnan(mr.split_half_reliability(vals, -np.ones(24, int), halves))
    assert np.isnan(mr.split_half_reliability(np.ones(24), labels, halves))


def test_permute_within_strata_keeps_strata() -> None:
    rng = np.random.default_rng(0)
    lab = np.array([0, 1, 2, 3, -1, 4, 5])
    strata = np.array([0, 0, 0, 1, 1, 1, 1])
    out = mr.permute_within_strata(lab, strata, rng)
    assert sorted(out[:3]) == [0, 1, 2] and sorted(out[[3, 5, 6]]) == [3, 4, 5]
    assert out[4] == -1


def test_run_indicator() -> None:
    assert mr.run_indicator([1, 5], [3, 6], 7).tolist() == [0, 1, 1, 0, 0, 1, 0]


def test_circular_xcorr_lead_sign() -> None:
    n = 200
    r = np.zeros(n)
    r[50:60] = 1
    x = np.roll(r, -3)  # activity 3 frames before running
    c = mr.circular_xcorr(x, r, n)
    lags = np.arange(-10, 11)
    curve = c[lags % n]
    assert lags[np.argmax(curve)] == 3
    s = mr.xcorr_summary(curve, lags, FPS)
    assert s["xc_peak_lag_s"][0] == pytest.approx(0.3) and s["xc_lead_lag_asym"][0] > 0
    # shifting x by s maps c[k] -> c[k + s]
    c2 = mr.circular_xcorr(np.roll(x, 7), r, n)
    assert np.allclose(c2, np.roll(c, -7))


# ---------------------------------------------------------------------------
# Design + cell statistics
# ---------------------------------------------------------------------------


def _synthetic_session(seed: int = 0, n_bouts: int = 120) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Bout table built directly (no positions) with balanced categories."""
    rng = np.random.default_rng(seed)
    gap, rows, t = 40, [], 50
    types = ["into_dead_end", "out_of_dead_end", "junction_to_junction"]
    routes = [f"{a}->{b}" for a, b in [(1, 2), (2, 1), (3, 4), (4, 3), (5, 6), (6, 5)]]
    for i in range(n_bouts):
        dur = int(rng.integers(10, 40))
        rows.append(
            {
                "onset": t,
                "offset": t + dur,
                "onset_s": t / FPS,
                "duration_s": dur / FPS,
                "mean_speed": float(rng.uniform(3, 20)),
                "light_cond": "light" if (t // 600) % 2 == 0 else "dark",
                "run_type": types[i % 3],
                "novel": i < 10,
                "repeat": i >= 10,
                "first_arm_entry": bool(i % 4 == 0),
                "route": routes[i % 6],
                "edge_prior_traversals": float(rng.integers(0, 20)),
            }
        )
        t += dur + gap
    return pd.DataFrame(rows), np.ones(t + 50, bool), t + 50


def test_build_design_groups_and_routes() -> None:
    bouts, valid, n = _synthetic_session()
    d = mr.build_design(bouts, valid, FPS, n, n_route_perms=20)
    assert d.contrasts["approach_return"][0].size == 40
    assert d.contrasts["novel_repeat"][2] == "time"
    assert np.unique(d.route_labels[d.route_labels >= 0]).size == 6
    assert d.route_null_labels.shape == (20, len(bouts))
    assert d.prior_resid is not None and d.lags.size == 101
    small = mr.build_design(bouts.iloc[:4], valid, FPS, n, min_group=5)
    assert small.contrasts["approach_return"][0].size == 0


def test_cell_statistics_generic_run_signal_has_no_context_effects() -> None:
    bouts, valid, n = _synthetic_session()
    d = mr.build_design(bouts, valid, FPS, n, n_route_perms=50)
    rng = np.random.default_rng(1)
    sig = rng.random(n) * 0.1 + mr.run_indicator(d.onsets, d.offsets, n)
    res = mr.cell_statistics(sig, d, n_shuffles=50, rng=rng)
    for k in mr.TESTED:
        assert k in res and f"{k}_p" in res
    assert abs(res["approach_return"]) < 0.2
    assert res["xc_peak_r"] > 0.5 and res["xc_peak_r_p"] < 0.05
    assert abs(res["xc_peak_lag_s"]) <= 0.2
    assert res["n_runs"] == len(bouts) and res["n_routes"] == 6


def test_cell_statistics_detects_approach_and_route() -> None:
    bouts, valid, n = _synthetic_session()
    d = mr.build_design(bouts, valid, FPS, n, n_route_perms=100)
    rng = np.random.default_rng(2)
    sig = rng.random(n) * 0.05
    route_gain = {r: g for g, r in enumerate(sorted(bouts["route"].unique()))}
    for _, b in bouts.iterrows():
        level = 1.0 + (1.0 if b["run_type"] == "into_dead_end" else 0.0)
        level += 0.3 * route_gain[b["route"]]
        sig[b["onset"] : b["offset"]] += level
    res = mr.cell_statistics(sig, d, n_shuffles=100, rng=rng)
    assert res["approach_return"] > 0.2 and res["approach_return_p"] < 0.05
    assert res["rate_into_dead_end"] > res["rate_out_of_dead_end"]
    assert res["route_reliability"] > 0.5 and res["route_reliability_p"] < 0.05


def test_cell_statistics_detects_lead_and_decay() -> None:
    bouts, valid, n = _synthetic_session()
    d = mr.build_design(bouts, valid, FPS, n, n_route_perms=20)
    rng = np.random.default_rng(3)
    run = mr.run_indicator(d.onsets, d.offsets, n).astype(float)
    sig = np.roll(run, -10) + rng.random(n) * 0.1  # leads by 1 s
    res = mr.cell_statistics(sig, d, n_shuffles=100, rng=rng)
    assert res["xc_peak_lag_s"] == pytest.approx(1.0)
    assert res["xc_lead_lag_asym"] > 0 and res["xc_lead_lag_asym_p"] < 0.05
    # activity decaying with prior traversals of the same edge
    sig2 = rng.random(n) * 0.05
    for _, b in bouts.iterrows():
        sig2[b["onset"] : b["offset"]] += 3.0 - 0.1 * b["edge_prior_traversals"]
    res2 = mr.cell_statistics(sig2, d, n_shuffles=100, rng=rng)
    assert res2["edge_decay_rho"] < -0.8 and res2["edge_decay_rho_p"] < 0.05


def test_cell_statistics_without_shuffles_and_empty() -> None:
    bouts, valid, n = _synthetic_session()
    d = mr.build_design(bouts, valid, FPS, n, n_route_perms=0)
    res = mr.cell_statistics(np.random.default_rng(0).random(n), d, n_shuffles=0)
    assert np.isnan(res["approach_return_p"]) and np.isnan(res["xc_peak_r_p"])
    assert np.isnan(res["route_reliability_p"])
    empty = mr.build_design(bouts.iloc[:0], valid, FPS, n)
    r2 = mr.cell_statistics(np.ones(n), empty, n_shuffles=20)
    assert np.isnan(r2["approach_return"]) and r2["n_runs"] == 0


def test_bout_statistics_no_bouts() -> None:
    bouts, valid, n = _synthetic_session()
    d = mr.build_design(bouts.iloc[:0], valid, FPS, n)
    out = mr._bout_statistics(np.zeros((3, 0)), d)
    assert all(np.isnan(v).all() for v in out.values())


def test_shift_p() -> None:
    null = np.random.default_rng(0).normal(0, 1, 200)
    z, p = mr._shift_p(5.0, null)
    assert z > 3 and p < 0.01
    assert np.isnan(mr._shift_p(np.nan, null)[1])
    assert np.isnan(mr._shift_p(1.0, null[:5])[1])
