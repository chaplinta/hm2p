"""Tests for ``hm2p.analysis.population_maze`` (small synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis import population_maze as pm
from hm2p.maze.topology import build_rose_maze

MAZE = build_rose_maze()
FPS = 9.6


# ---------------------------------------------------------------------------
# Synthetic session: random walk on the maze graph, runs separated by pauses
# ---------------------------------------------------------------------------


def _walk(n_target: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    cell, prev = MAZE.cell_list[0], None
    xs: list[float] = []
    ys: list[float] = []
    spd: list[float] = []
    hd: list[float] = []
    while len(xs) < n_target:
        for _ in range(int(rng.integers(2, 7))):
            nbs = [c for c in MAZE.adj[cell] if c != prev] or MAZE.adj[cell]
            nxt = nbs[int(rng.integers(len(nbs)))]
            k = int(rng.integers(3, 6))
            t = np.arange(k) / k
            xs += list(cell[0] + 0.5 + t * (nxt[0] - cell[0]))
            ys += list(cell[1] + 0.5 + t * (nxt[1] - cell[1]))
            spd += list(rng.uniform(6, 20, k))
            ang = np.degrees(np.arctan2(nxt[1] - cell[1], nxt[0] - cell[0]))
            hd += list(ang + rng.normal(0, 10, k))
            prev, cell = cell, nxt
        k = int(rng.integers(10, 25))
        xs += [cell[0] + 0.5] * k
        ys += [cell[1] + 0.5] * k
        spd += list(rng.uniform(0, 1, k))
        hd += list(hd[-1] + rng.normal(0, 20, k))
    x, y = np.asarray(xs), np.asarray(ys)
    hdv = np.asarray(hd)
    n = x.size
    return {
        "x": x,
        "y": y,
        "speed": np.asarray(spd),
        "hd": hdv,
        "ahv": np.degrees(np.gradient(np.unwrap(np.radians(hdv)))) * FPS,
        "light": (np.arange(n) // 576) % 2 == 0,
        "valid": np.ones(n, dtype=bool),
    }


def _session(n_target: int = 5000, n_cells: int = 12, seed: int = 0) -> dict[str, np.ndarray]:
    """Walk plus place-tuned cells (each fires near one preferred maze cell)."""
    w = _walk(n_target, seed)
    rng = np.random.default_rng(seed + 1)
    idx = pm.discretize_position_fast(w["x"], w["y"], MAZE)
    pref = rng.integers(0, MAZE.n_cells, n_cells)
    sig = rng.poisson(0.2, (n_cells, idx.size)).astype(float)
    sig += 1.5 * (MAZE.dist[idx][:, pref].T <= 0)
    sig[0, :10] = np.nan  # spike inference leaves NaN at the edges
    w["signal"] = sig
    return w


def _decode(s: dict[str, np.ndarray], **kw: object) -> pm.SessionResult:
    params = pm.PopDecParams(
        n_shuffles=kw.pop("n_shuffles", 12),  # type: ignore[arg-type]
        n_subsets=2,
        n_subset_shuffles=3,
        subset_sizes=(5,),
        **kw,  # type: ignore[arg-type]
    )
    return pm.decode_session(
        s["signal"],
        s["x"],
        s["y"],
        s["speed"],
        s["hd"],
        s["ahv"],
        s["light"],
        s["valid"],
        FPS,
        params=params,
        maze=MAZE,
    )


@pytest.fixture(scope="module")
def session_result() -> pm.SessionResult:
    return _decode(_session())


# ---------------------------------------------------------------------------
# WindowMeans and feature helpers
# ---------------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(5, 40),
    shift=st.integers(-100, 100),
    seed=st.integers(0, 10_000),
)
def test_window_means_matches_roll(n: int, shift: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(3, n))
    x[rng.random((3, n)) < 0.2] = np.nan
    starts = rng.integers(0, n, 6)
    ends = np.minimum(starts + rng.integers(0, n, 6), n)
    got = pm.WindowMeans(x).means(starts, ends, shift)
    rolled = np.roll(x, shift, axis=1)
    for i, (a, b) in enumerate(zip(starts, ends, strict=True)):
        seg = rolled[:, a:b]
        for c in range(3):
            fin = seg[c][np.isfinite(seg[c])]
            if fin.size:
                assert got[i, c] == pytest.approx(fin.mean())
            else:
                assert np.isnan(got[i, c])


def test_window_means_1d_and_bad_windows() -> None:
    wm = pm.WindowMeans(np.arange(6.0))
    assert wm.n_channels == 1
    assert wm.means([0, 4], [2, 6])[:, 0].tolist() == [0.5, 4.5]
    with pytest.raises(ValueError):
        wm.means([3], [1])
    with pytest.raises(ValueError):
        wm.means([0], [7])


def test_impute_columns() -> None:
    x = np.array([[1.0, np.nan], [3.0, np.nan], [np.nan, np.nan]])
    out = pm.impute_columns(x)
    assert out[:, 0].tolist() == [1.0, 3.0, 2.0]
    assert np.all(out[:, 1] == 0)
    assert np.isnan(x[2, 0])  # input unchanged
    assert pm.impute_columns(np.zeros((0, 3))).shape == (0, 3)


@settings(max_examples=40, deadline=None)
@given(st.lists(st.floats(-720, 720), min_size=3, max_size=30))
def test_behaviour_features_unit_hd(hd: list[float]) -> None:
    n = len(hd)
    b = pm.behaviour_matrix(hd, np.ones(n), -np.ones(n) * 2)
    assert b.shape == (4, n) and np.all(b[3] == 2)
    f = pm.behaviour_features(pm.WindowMeans(b), [0], [n])
    r = np.hypot(f[0, 0], f[0, 1])
    assert np.isnan(r) or r == pytest.approx(1.0)
    assert f[0, 2] == pytest.approx(1.0)


def test_behaviour_features_circular_mean_and_zero_resultant() -> None:
    b = pm.behaviour_matrix([80.0, 100.0, 0.0, 180.0], np.zeros(4), np.zeros(4))
    f = pm.behaviour_features(pm.WindowMeans(b), [0, 2], [2, 4])
    assert f[0, 0] == pytest.approx(1.0) and f[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(f[1, 0]) or abs(np.hypot(f[1, 0], f[1, 1]) - 1) < 1e-9


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


def test_session_cells_masks_invalid_and_short_dwell() -> None:
    x = np.array([0.5] * 5 + [1.5] + [0.5] * 4)
    y = np.full(10, 0.5)
    valid = np.ones(10, dtype=bool)
    valid[0] = False
    cells = pm.session_cells(x, y, valid, MAZE, min_dwell_frames=2)
    c0 = MAZE.cell_to_idx[(0, 0)]
    assert cells[0] == -1 and cells[5] == -1
    assert np.all(cells[1:5] == c0) and np.all(cells[6:] == c0)


def test_running_visits() -> None:
    cells = np.array([0, 0, 0, 1, 1, 1, 1, -1, 2, 2, 2, 2, 3, 3])
    v = pm.running_visits(cells, [0, 8], [7, 14], min_visit_frames=3)
    assert v[["start", "end", "cell", "bout"]].values.tolist() == [
        [0, 3, 0, 0],
        [3, 7, 1, 0],
        [8, 12, 2, 1],
    ]
    assert pm.running_visits(cells, [5], [5]).empty


def test_cell_stays() -> None:
    c, s, e = pm.cell_stays(np.array([-1, 2, 2, -1, 2, 3, 3, -1]))
    assert c.tolist() == [2, 3] and s.tolist() == [1, 5] and e.tolist() == [5, 7]
    c, s, e = pm.cell_stays(np.full(4, -1))
    assert c.size == s.size == e.size == 0


@settings(max_examples=50, deadline=None)
@given(seed=st.integers(0, 10_000), n=st.integers(0, 60), n_bins=st.integers(1, 4))
def test_matched_subsample_balanced(seed: int, n: int, n_bins: int) -> None:
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, 2, n)
    dur = rng.gamma(2, 1, n) + lab
    spd = rng.normal(10, 2, n)
    if n:
        dur[0] = np.nan
    keep = pm.matched_subsample(lab, dur, spd, n_bins, rng)
    assert np.all(np.diff(keep) > 0)
    assert (lab[keep] == 0).sum() == (lab[keep] == 1).sum()
    assert np.all(np.isfinite(dur[keep]))


def test_matched_subsample_matches_covariates() -> None:
    rng = np.random.default_rng(0)
    lab = np.repeat([0, 1], 300)
    dur = np.concatenate([rng.uniform(0, 1, 300), rng.uniform(0.5, 1.5, 300)])
    keep = pm.matched_subsample(lab, dur, np.ones(600), 4, rng)
    d0, d1 = dur[keep][lab[keep] == 0], dur[keep][lab[keep] == 1]
    assert abs(np.median(d0) - np.median(d1)) < abs(np.median(dur[:300]) - np.median(dur[300:]))
    assert pm.matched_subsample(lab, np.full(600, np.nan), np.ones(600), 3, rng).size == 0


def _bouts(n_in: int, n_out: int) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = n_in + n_out + 3
    return pd.DataFrame(
        {
            "onset": np.arange(n) * 50,
            "offset": np.arange(n) * 50 + 20,
            "duration_s": rng.uniform(1, 3, n),
            "mean_speed": rng.uniform(5, 15, n),
            "run_type": ["into_dead_end"] * n_in
            + ["out_of_dead_end"] * n_out
            + ["junction_to_junction"] * 3,
        }
    )


def test_inout_samples() -> None:
    rng = np.random.default_rng(0)
    s = pm.inout_samples(_bouts(30, 30), rng, n_bins=2)
    assert s.reason == "" and s.n_classes == 2 and s.n_strata == 1
    assert (s.labels == 1).sum() == (s.labels == 0).sum() == s.info["inout_n_matched_per_class"]
    assert np.all(s.ends - s.starts == 20)
    assert "inout_median_speed_into" in s.info
    assert pm.inout_samples(_bouts(3, 3), rng).reason == "too few matched dead-end runs"
    none = pm.inout_samples(_bouts(0, 0), rng)
    assert none.reason == "no dead-end runs" and none.n == 0 and none.n_classes == 0
    empty = pm.inout_samples(_bouts(0, 0).iloc[:0], rng)
    assert empty.reason == "no dead-end runs"


def test_position_samples() -> None:
    v = pd.DataFrame(
        {"start": np.arange(12), "end": np.arange(12) + 1, "cell": [1] * 6 + [2] * 5 + [3]}
    )
    v["bout"] = 0
    s = pm.position_samples(v, min_class_samples=5)
    assert s.n == 11 and set(s.labels) == {1, 2} and s.n_classes == 2
    assert pm.position_samples(v, min_class_samples=6).reason != ""
    assert pm.position_samples(v.iloc[:0]).reason == "no running visits"


def _visits_for_path(path: list[int], bout: list[int]) -> pd.DataFrame:
    st_ = np.arange(len(path)) * 10
    return pd.DataFrame({"start": st_, "end": st_ + 5, "cell": path, "bout": bout})


def test_edge_direction_samples() -> None:
    a = MAZE.cell_to_idx[MAZE.cell_list[0]]
    b = MAZE.cell_to_idx[MAZE.adj[MAZE.cell_list[0]][0]]
    path = [a, b] * 12
    v = _visits_for_path(path, [0] * 24)
    s = pm.edge_direction_samples(v, MAZE, min_traversals=5)
    assert s.reason == "" and s.n_strata == 1 and s.n_classes == 2
    # every consecutive pair is a traversal: 12 a->b, 11 b->a
    assert s.info["n_traversals"] == 23
    assert np.all(s.ends - s.starts == 15)
    assert ((s.labels == 0).sum(), (s.labels == 1).sum()) in {(12, 11), (11, 12)}
    # different bouts break traversals
    v2 = _visits_for_path(path, list(range(24)))
    assert pm.edge_direction_samples(v2, MAZE).reason != ""
    assert pm.edge_direction_samples(v.iloc[:1], MAZE).reason == "too few running visits"


def test_junction_exit_samples() -> None:
    j = next(c for c in MAZE.junctions if MAZE.node_types[c] == "t_junction")
    arms = [MAZE.cell_to_idx[c] for c in MAZE.adj[j]]
    ji = MAZE.cell_to_idx[j]
    seq: list[int] = []
    for k in range(14):
        seq += [arms[0]] * 12 + [ji] * 4 + [arms[1 + k % 2]] * 12
    seq += [arms[0]] * 12 + [ji] * 4 + [arms[0]] * 12  # U-turn
    cells = np.asarray(seq)
    s = pm.junction_exit_samples(cells, MAZE, FPS, window_s=1.0, min_per_choice=5)
    assert s.reason == "" and s.n_strata == 1
    assert set(s.labels) == {arms[1], arms[2]}
    assert s.info["n_junction_uturns"] >= 1
    w = round(FPS)
    assert np.all(s.ends - s.starts == w)
    assert np.all(cells[s.ends - 1] == ji) and np.all(cells[s.ends] != ji)
    assert pm.junction_exit_samples(cells, MAZE, FPS, min_per_choice=20).reason != ""
    assert pm.junction_exit_samples(np.full(5, -1), MAZE, FPS).reason == "no T-junction passes"
    # window before the session start is dropped
    short = np.asarray([arms[0]] + [ji] + [arms[1]] * 3)
    assert pm.junction_exit_samples(short, MAZE, FPS, window_s=5.0).n == 0


# ---------------------------------------------------------------------------
# CV and scoring
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    n_blocks=st.integers(2, 6),
    gap=st.integers(0, 30),
)
def test_time_block_folds_disjoint_and_gapped(seed: int, n_blocks: int, gap: int) -> None:
    rng = np.random.default_rng(seed)
    n_frames = 1000
    starts = np.sort(rng.integers(0, n_frames - 20, 40))
    ends = starts + rng.integers(1, 20, 40)
    folds = pm.time_block_folds(starts, ends, n_frames, n_blocks, gap)
    tested = np.concatenate([te for _, te in folds])
    assert np.array_equal(np.sort(tested), np.arange(40))
    edges = np.linspace(0, n_frames, n_blocks + 1)
    block = np.clip(
        np.searchsorted(edges[1:-1], (starts + ends - 1) / 2, "right"), 0, n_blocks - 1
    )
    for tr, te in folds:
        assert np.intersect1d(tr, te).size == 0
        b = int(block[te[0]])
        assert np.all(block[te] == b)
        # training windows end before the test block minus the gap or start after it plus the gap
        assert np.all((ends[tr] <= edges[b] - gap) | (starts[tr] >= edges[b + 1] + gap))


def test_time_block_folds_gap_excludes_neighbours() -> None:
    starts = np.array([0, 45, 55, 95])
    ends = starts + 5
    folds = pm.time_block_folds(starts, ends, 100, n_blocks=2, gap_frames=10)
    tr0 = dict((int(te[0]), tr) for tr, te in folds)[0]
    assert tr0.tolist() == [3]  # sample at 55 is within the gap of block 0


def test_cv_predict_separable_and_edge_cases() -> None:
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 30)
    x = rng.normal(size=(60, 4))
    x[:, 0] += 4 * y
    folds = [(np.arange(30, 60), np.arange(0, 30)), (np.arange(0, 30), np.arange(30, 60))]
    # each training fold has only one class -> constant prediction
    pred = pm.cv_predict(x, y, folds)
    assert np.all(pred[:30] == 1) and np.all(pred[30:] == 0)
    perm = rng.permutation(60)
    folds = [(perm[:40], perm[40:]), (perm[20:], perm[:20])]
    pred = pm.cv_predict(x, y, folds)
    assert pm.balanced_accuracy(y[perm[:20]], pred[perm[:20]]) > 0.9
    pred = pm.cv_predict(x, y, [(np.zeros(0, int), np.arange(5))])
    assert np.all(pred == -1)


def test_cv_predict_multiclass() -> None:
    rng = np.random.default_rng(1)
    y = np.repeat([3, 7, 9], 30)
    x = rng.normal(size=(90, 3)) + np.eye(3)[np.repeat([0, 1, 2], 30)] * 5
    perm = rng.permutation(90)
    folds = [(perm[30:], perm[:30]), (perm[:60], perm[60:])]
    pred = pm.cv_predict(x, y, folds)
    assert set(np.unique(pred[np.r_[perm[:30], perm[60:]]])) <= {3, 7, 9}
    assert pm.balanced_accuracy(y[perm[:30]], pred[perm[:30]]) > 0.9


def test_cv_predict_hd_removal() -> None:
    """Features that carry the label only through HD lose it when HD is regressed out."""
    rng = np.random.default_rng(2)
    n = 400
    ang = rng.uniform(-np.pi, np.pi, n)
    hd = np.column_stack([np.sin(ang), np.cos(ang)])
    y = (np.sin(ang) > 0).astype(int)
    x = hd @ rng.normal(size=(2, 6)) + 0.05 * rng.normal(size=(n, 6))
    folds = pm.time_block_folds(np.arange(n), np.arange(n) + 1, n, 5, 0)
    acc_raw = pm.balanced_accuracy(y, pm.cv_predict(x, y, folds))
    acc_res = pm.balanced_accuracy(y, pm.cv_predict(x, y, folds, hd=hd))
    assert acc_raw > 0.9 and acc_res < 0.7


@settings(max_examples=50, deadline=None)
@given(seed=st.integers(0, 10_000), n=st.integers(1, 50), k=st.integers(1, 5))
def test_balanced_accuracy_bounds(seed: int, n: int, k: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    pred = rng.integers(-1, k, n)
    ba = pm.balanced_accuracy(y, pred)
    assert 0.0 <= ba <= 1.0
    assert pm.balanced_accuracy(y, y) == 1.0


def test_balanced_accuracy_values() -> None:
    assert pm.balanced_accuracy([0, 0, 0, 1], [0, 0, 0, 0]) == 0.5
    assert np.isnan(pm.balanced_accuracy([], []))


def test_class_mean_distance() -> None:
    d = MAZE.dist.astype(float)
    y = np.array([0, 0, 1])
    assert pm.class_mean_distance(y, y, d) == 0.0
    pred = np.array([0, 1, -1])
    expect = np.mean([d[0, 1] / 2, d.max()])
    assert pm.class_mean_distance(y, pred, d) == pytest.approx(expect)
    assert np.isnan(pm.class_mean_distance([], [], d))


def test_prepare_and_score_problem_stratified() -> None:
    rng = np.random.default_rng(3)
    n = 120
    strata = np.repeat([0, 1], n // 2)
    y = rng.integers(0, 2, n)
    x = rng.normal(size=(n, 3))
    x[strata == 0, 0] += 5 * y[strata == 0]  # informative only in stratum 0
    order = rng.permutation(n)
    starts = np.empty(n, dtype=np.int64)
    starts[order] = np.arange(n) * 10
    s = pm._make_samples("junction_exit", starts, starts + 5, y, strata, {})
    prob = pm.prepare_problem(s, n * 10, 5, 0)
    assert len(prob.strata) == 2
    res = pm.score_problem(prob, x, None, None)
    assert set(res) == {"balanced_accuracy"} and 0.6 < res["balanced_accuracy"] < 0.9


def test_null_summary() -> None:
    null = np.array([0.4, 0.5, 0.6, np.nan])
    r = pm.null_summary(0.55, null)
    assert r["n_null"] == 3 and r["p"] == pytest.approx(2 / 4)
    assert r["excess"] == pytest.approx(0.05)
    r = pm.null_summary(2.0, np.array([3.0, 4.0, 5.0]), lower_is_better=True)
    assert r["p"] == pytest.approx(1 / 4) and r["excess"] == pytest.approx(2.0)
    assert r["z"] < 0
    assert np.isnan(pm.null_summary(np.nan, null)["p"])
    assert pm.null_summary(0.5, np.zeros(0))["n_null"] == 0
    assert np.isnan(pm.null_summary(0.5, np.ones(3))["z"])


@settings(max_examples=30, deadline=None)
@given(n=st.integers(1, 2000), n_sh=st.integers(0, 50), min_shift=st.integers(0, 600))
def test_draw_shifts(n: int, n_sh: int, min_shift: int) -> None:
    sh = pm.draw_shifts(n, n_sh, min_shift, np.random.default_rng(0))
    assert sh[0] == 0
    if n_sh >= 1 and n > 2 * min_shift:
        assert sh.size == n_sh + 1
        assert np.all((sh[1:] >= min_shift) & (sh[1:] < n - min_shift))
    else:
        assert sh.size == 1


# ---------------------------------------------------------------------------
# Session-level evaluation
# ---------------------------------------------------------------------------


def test_decode_session_tables(session_result: pm.SessionResult) -> None:
    import pandera.pandas as pa

    dec = session_result.decoders
    schema = pa.DataFrameSchema(
        {
            "decoder": pa.Column(str, pa.Check.isin(pm.DECODERS)),
            "feature_set": pa.Column(str, pa.Check.isin(pm.FEATURE_SETS)),
            "metric": pa.Column(str),
            "observed": pa.Column(float, nullable=True),
            "p": pa.Column(float, pa.Check.in_range(0, 1), nullable=True),
            "n_null": pa.Column(int, pa.Check.ge(0)),
            "n_samples": pa.Column(int, pa.Check.ge(0)),
            "n_cells": pa.Column(int, pa.Check.eq(12)),
            "reason": pa.Column(str),
        }
    )
    schema.validate(dec)
    assert len(dec) == 3 * sum(len(m) for m in pm.METRICS.values())
    assert dec["headline"].sum() == 3 * len(pm.DECODERS)
    beh = dec[dec.feature_set == "behaviour"]
    assert (beh["n_features"] == 4).all()


def test_decode_session_detects_place_signal(session_result: pm.SessionResult) -> None:
    dec = session_result.decoders.set_index(["decoder", "feature_set", "metric"])
    pos = dec.loc[("position", "neural", "mean_distance")]
    assert pos["reason"] == "" and pos["excess"] > 1.0 and pos["p"] <= 1 / 13 + 1e-9
    assert pos["n_null"] == 12
    # heading decodes edge direction from behaviour alone
    edge = dec.loc[("edge_direction", "behaviour", "balanced_accuracy")]
    if edge["reason"] == "":
        assert edge["observed"] > 0.8


def test_decode_session_subsets_and_counts(session_result: pm.SessionResult) -> None:
    sub = session_result.subsets
    assert set(sub["n_cells_subset"]) == {5, 12}
    full = sub[sub["full_population"]]
    assert (full["n_draws"] == 1).all() and (full["n_null"] == 12).all()
    part = sub[~sub["full_population"]]
    assert (part["n_draws"] == 2).all() and (part["n_null"] <= 6).all()
    c = session_result.counts
    assert c["n_cells"] == 12 and c["n_bouts"] > 10
    for name in pm.DECODERS:
        assert f"{name}_n_samples" in c and f"{name}_reason" in c


def test_decode_session_parallel_matches_serial() -> None:
    s = _session(2500, 6, seed=4)
    a = _decode(s, n_shuffles=6).decoders
    b = _decode(s, n_shuffles=6, n_jobs=2).decoders
    pd.testing.assert_frame_equal(a, b)


def test_decode_session_short_and_empty() -> None:
    s = _session(700, 4, seed=5)
    res = _decode(s, n_shuffles=5, min_shift_s=60.0)  # session < 2 x 60 s minimum shift
    run = res.decoders[res.decoders["observed"].notna()]
    assert (run["reason"] == "session too short for shift null").all()
    assert (run["n_null"] == 0).all()
    s["speed"] = np.zeros_like(s["speed"])  # no running at all
    res = _decode(s, n_shuffles=5)
    running = res.decoders[res.decoders.decoder != "junction_exit"]
    assert running["observed"].isna().all() and (running["reason"] != "").all()


def test_evaluate_problem_requires_no_parallel_for_few_shifts() -> None:
    s = _session(2500, 5, seed=6)
    data = pm.SessionData(
        pm.WindowMeans(s["signal"]),
        pm.WindowMeans(pm.behaviour_matrix(s["hd"], s["speed"], s["ahv"])),
        MAZE.dist.astype(float),
    )
    on, off = pm.run_bouts(s["speed"], s["valid"], FPS)
    cells = pm.session_cells(s["x"], s["y"], s["valid"], MAZE)
    samp = pm.position_samples(pm.running_visits(cells, on, off))
    prob = pm.prepare_problem(samp, s["x"].size, 5, 0)
    out = pm.evaluate_problem(prob, data, np.array([0, 300]), ("behaviour",), n_jobs=4)
    assert set(out) == {("behaviour", "balanced_accuracy"), ("behaviour", "mean_distance")}
    assert out[("behaviour", "balanced_accuracy")].shape == (2,)


# ---------------------------------------------------------------------------
# Across-session statistics
# ---------------------------------------------------------------------------


def _sessions_table() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for i in range(8):
        animal = f"a{i // 2}"
        ct = "penk" if i < 4 else "nonpenk"
        for fs, shift in (("neural", 0.1), ("behaviour", 0.3), ("neural_hd_removed", 0.05)):
            ex = shift + rng.normal(0, 0.02)
            rows.append(
                {
                    "exp_id": f"e{i}",
                    "animal_id": animal,
                    "celltype": ct,
                    "decoder": "inout",
                    "metric": "balanced_accuracy",
                    "feature_set": fs,
                    "observed": 0.5 + ex,
                    "null_mean": 0.5,
                    "excess": ex,
                    "p": 0.01 if ex > 0.05 else 0.3,
                }
            )
    rows[0]["excess"] = np.nan
    return pd.DataFrame(rows)


def test_across_session_summary() -> None:
    out = pm.across_session_summary(_sessions_table())
    assert set(out["group"]) == {"all", "penk", "nonpenk"}
    row = out[(out.group == "all") & (out.feature_set == "behaviour")].iloc[0]
    assert row["n_sessions"] == 8 and row["n_animals"] == 4
    assert row["median_excess"] > 0.25 and row["session_wilcoxon_p"] < 0.05
    assert bool(row["headline"]) and not bool(row["descriptive"])
    neural = out[(out.group == "all") & (out.feature_set == "neural")].iloc[0]
    assert neural["n_sessions"] == 7 and neural["n_sessions_skipped"] == 1
    assert pm.across_session_summary(pd.DataFrame()).empty


def test_paired_feature_comparisons() -> None:
    out = pm.paired_feature_comparisons(_sessions_table())
    assert set(out["comparison"]) == {f"{a} - {b}" for a, b in pm.COMPARISONS}
    nb = out[out.comparison == "neural - behaviour"].iloc[0]
    assert nb["median_diff"] < 0 and nb["n_sessions"] == 7 and nb["n_animals"] == 4
    assert nb["session_wilcoxon_p"] < 0.05
    assert pm.paired_feature_comparisons(pd.DataFrame()).empty
    only = _sessions_table()
    only = only[only.feature_set == "neural"]
    assert pm.paired_feature_comparisons(only).empty


def test_wilcoxon_helper() -> None:
    assert np.isnan(pm._wilcoxon([1.0, 2.0]))
    assert np.isnan(pm._wilcoxon([0.0, 0.0, 0.0]))
    assert pm._wilcoxon([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) < 0.05


def test_subset_curve_summary(session_result: pm.SessionResult) -> None:
    sub = pd.concat(
        [session_result.subsets.assign(exp_id="a"), session_result.subsets.assign(exp_id="b")]
    )
    out = pm.subset_curve_summary(sub)
    assert set(out["n_cells_subset"]) == {5, -1}
    assert (out["n_sessions"] <= 2).all()
    assert pm.subset_curve_summary(pd.DataFrame()).empty
