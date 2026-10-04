"""Tests for ``hm2p.analysis.maze_memory`` (small synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from hm2p.analysis import maze_memory as mm
from hm2p.analysis import population_maze as pm
from hm2p.maze.topology import build_rose_maze

MAZE = build_rose_maze()
IDX = MAZE.cell_to_idx
FPS = 9.6
DE = mm.dead_end_indices(MAZE)


def c(col: int, row: int) -> int:
    """Cell index of (col, row)."""
    return IDX[(col, row)]


# ---------------------------------------------------------------------------
# Synthetic session: walk on the maze graph with dead-end dwells
# ---------------------------------------------------------------------------


def _walk(n_target: int, seed: int, p_back: float = 0.1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    cell, prev = MAZE.cell_list[0], None
    xs: list[float] = []
    ys: list[float] = []
    spd: list[float] = []
    hd: list[float] = []
    while len(xs) < n_target:
        fwd = [x for x in MAZE.adj[cell] if x != prev]
        nbs = MAZE.adj[cell] if (not fwd or rng.random() < p_back) else fwd
        nxt = nbs[int(rng.integers(len(nbs)))]
        k = int(rng.integers(3, 7))
        t = np.arange(k) / k
        xs += list(cell[0] + 0.5 + t * (nxt[0] - cell[0]))
        ys += list(cell[1] + 0.5 + t * (nxt[1] - cell[1]))
        spd += list(rng.uniform(6, 20, k))
        ang = np.degrees(np.arctan2(nxt[1] - cell[1], nxt[0] - cell[0]))
        hd += list(ang + rng.normal(0, 10, k))
        prev, cell = cell, nxt
        if len(MAZE.adj[cell]) == 1:
            k = int(rng.integers(6, 20))
            xs += [cell[0] + 0.5] * k
            ys += [cell[1] + 0.5] * k
            spd += list(rng.uniform(0, 1, k))
            hd += list(hd[-1] + rng.normal(0, 20, k))
    n = n_target
    x, y, h = np.asarray(xs[:n]), np.asarray(ys[:n]), np.asarray(hd[:n])
    return {
        "x": x,
        "y": y,
        "speed": np.asarray(spd[:n]),
        "hd": h,
        "ahv": np.degrees(np.gradient(np.unwrap(np.radians(h)))) * FPS,
        "light": (np.arange(n) // 576) % 2 == 0,
        "valid": np.ones(n, dtype=bool),
    }


def _origin_signal(w: dict[str, np.ndarray], n_cells: int, seed: int) -> np.ndarray:
    """Noise plus cells that fire during trips according to the trip's origin."""
    rng = np.random.default_rng(seed)
    cells = pm.session_cells(w["x"], w["y"], w["valid"], MAZE, 2)
    visits = mm.visit_table(cells, MAZE, 10, 2)
    trips = mm.trip_table(visits, MAZE)
    n = w["x"].size
    sig = rng.poisson(0.2, (n_cells, n)).astype(float)
    for o, a, b in zip(trips["origin"], trips["start"], trips["end"], strict=True):
        k = int(np.searchsorted(DE, o)) % n_cells
        sig[k, a:b] += 2.0
    return sig


def _params(**kw: object) -> mm.MazeMemParams:
    base = {"n_shuffles": 3, "n_subsets": 1, "n_subset_shuffles": 2, "subset_size": 3}
    base.update(kw)
    return mm.MazeMemParams(**base)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def session() -> tuple[dict[str, np.ndarray], mm.MazeMemResult]:
    w = _walk(6000, 0)
    sig = _origin_signal(w, 9, 1)
    sig[0, :5] = np.nan
    res = mm.maze_memory_session(
        sig,
        w["x"],
        w["y"],
        w["speed"],
        w["hd"],
        w["ahv"],
        w["light"],
        w["valid"],
        FPS,
        _params(n_shuffles=8),
        MAZE,
    )
    return w, res


# ---------------------------------------------------------------------------
# Visits and trips
# ---------------------------------------------------------------------------


def test_dead_end_indices() -> None:
    assert DE.size == 9
    assert all(MAZE.node_types[MAZE.cell_list[i]] == "dead_end" for i in DE)


def test_visit_table_basic_and_gap_merge() -> None:
    a, b, d = c(0, 0), c(1, 0), c(2, 0)
    cells = np.array([a, a, -1, a, b, b, b, d, d, d])
    v = mm.visit_table(cells, MAZE, max_gap_frames=2)
    assert v["cell"].tolist() == [a, b, d]
    assert v["start"].tolist() == [0, 4, 7] and v["end"].tolist() == [4, 7, 10]
    assert v["prev"].tolist() == [-1, a, b] and v["next"].tolist() == [b, d, -1]
    assert (v["seg"] == 0).all() and not v["interp"].any()


def test_visit_table_segment_breaks() -> None:
    a, b = c(0, 0), c(1, 0)
    long_gap = np.array([a, a] + [-1] * 5 + [b, b])
    v = mm.visit_table(long_gap, MAZE, max_gap_frames=2)
    assert v["seg"].tolist() == [0, 1]
    assert v["next"].tolist() == [-1, -1] and v["prev"].tolist() == [-1, -1]
    same = np.array([a, a] + [-1] * 5 + [a, a])
    assert mm.visit_table(same, MAZE, 2)["seg"].tolist() == [0, 1]
    far = np.array([c(0, 0)] * 2 + [c(6, 4)] * 2)
    assert mm.visit_table(far, MAZE, 2, max_jump_steps=2)["seg"].tolist() == [0, 1]


def test_visit_table_fills_jump() -> None:
    cells = np.array([c(0, 0)] * 3 + [c(1, 1)] * 3)  # (1,0) skipped
    v = mm.visit_table(cells, MAZE, 2, max_jump_steps=2)
    assert v["cell"].tolist() == [c(0, 0), c(1, 0), c(1, 1)]
    assert v["interp"].tolist() == [False, True, False]
    assert v.loc[1, "start"] == v.loc[1, "end"] == 3
    assert (v["seg"] == 0).all()


def test_visit_table_empty() -> None:
    v = mm.visit_table(np.full(5, -1), MAZE)
    assert v.empty and list(v.columns) == mm.VISIT_COLUMNS


def _visits_from_path(path: list[int], frames: int = 3) -> pd.DataFrame:
    return mm.visit_table(np.repeat(np.asarray(path), frames), MAZE, 2)


def test_trip_table_and_passthrough() -> None:
    a, j, b, k = c(0, 0), c(1, 0), c(2, 0), c(1, 1)
    # a -> j -> k -> j -> b : trip a->b with a U-turn at k
    v = _visits_from_path([a, j, k, j, b])
    trips = mm.trip_table(v, MAZE)
    assert len(trips) == 1
    t = trips.iloc[0]
    assert (t["origin"], t["dest"], t["n_steps"]) == (a, b, 4)
    assert t["start"] == 3 and t["end"] == 12
    pt = mm.passthrough_samples(v, trips, MAZE)
    # j (arrive a, leave k) and j (arrive k, leave b) pass through; k is a U-turn
    assert pt["cell"].tolist() == [j, j]
    assert pt["elapsed"].tolist() == [1, 3]
    assert pt["remaining"].tolist() == [1, 1]
    assert (pt["origin"] == a).all() and (pt["dest"] == b).all()


def test_trip_table_respects_segments() -> None:
    a, j, b = c(0, 0), c(1, 0), c(2, 0)
    cells = np.r_[[a] * 3, [j] * 3, [-1] * 20, [b] * 3]
    v = mm.visit_table(cells, MAZE, 2)
    assert mm.trip_table(v, MAZE).empty
    assert mm.trip_table(v.iloc[:0], MAZE).empty
    assert mm.passthrough_samples(v, mm.trip_table(v, MAZE), MAZE).empty


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.booleans(), min_size=1, max_size=60),
    st.integers(0, 59),
    st.integers(0, 60),
)
def test_light_condition(on: list[bool], a: int, length: int) -> None:
    arr = np.asarray(on)
    b = min(a + length, arr.size)
    out = mm.light_condition(arr, [a], [b], 0.9, 0.1)[0]
    if b <= a or a >= arr.size:
        assert out == "mixed"
        return
    frac = arr[a:b].mean()
    expect = "light" if frac >= 0.9 else "dark" if frac <= 0.1 else "mixed"
    assert out == expect


# ---------------------------------------------------------------------------
# Random walks and novelty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backtrack", [True, False])
def test_dead_end_absorption_rows(backtrack: bool) -> None:
    h = mm.dead_end_absorption(MAZE, backtrack)
    assert np.allclose(h[DE].sum(axis=1), 1.0)
    other = np.setdiff1d(np.arange(MAZE.n_cells), DE)
    assert np.allclose(h[other], 0.0) and np.allclose(h[:, other], 0.0)
    if not backtrack:
        assert np.allclose(np.diag(h)[DE], 0.0)  # no reversal -> never back to origin


def test_dead_end_absorption_matches_simulation() -> None:
    rng = np.random.default_rng(0)
    adj = [[IDX[x] for x in MAZE.adj[cl]] for cl in MAZE.cell_list]
    is_de = np.zeros(MAZE.n_cells, bool)
    is_de[DE] = True
    o = int(DE[0])
    for backtrack in (True, False):
        counts = np.zeros(MAZE.n_cells)
        for _ in range(4000):
            prev, cur = o, adj[o][0]
            while not is_de[cur]:
                opts = adj[cur] if backtrack else [x for x in adj[cur] if x != prev]
                prev, cur = cur, opts[int(rng.integers(len(opts)))]
            counts[cur] += 1
        h = mm.dead_end_absorption(MAZE, backtrack)[o]
        assert np.abs(counts / counts.sum() - h).max() < 0.03


def test_trip_novelty_labels() -> None:
    d0, d1, d2, d3 = (int(x) for x in DE[:4])
    seq = [d0, d1, d2, d3, d0, d3]
    v = pd.DataFrame(
        {
            "cell": seq,
            "start": np.arange(6) * 10,
            "end": np.arange(6) * 10 + 5,
            "seg": 0,
            "interp": False,
            "prev": -1,
            "next": -1,
        }
    )
    trips = pd.DataFrame(
        {"i_origin": [0, 1, 3, 4], "origin": [d0, d1, d3, d0], "dest": [d1, d2, d0, d3]}
    )
    nv = mm.trip_novelty(v, trips, MAZE, recent_k=3)
    # d3 -> d0: recent = {d3, d2, d1}, so d0 is novel-ish, but never-visited
    # dead ends are less recent than d0, so it is not the least recent
    assert nv["novel"].tolist() == [True, True, True, False]
    assert nv["lru"].tolist() == [True, True, False, False]
    assert nv["n_distinct_before"].tolist() == [1, 2, 4, 4]
    for m in mm.RW_MODELS:
        assert nv[f"p_novel_{m}"].between(0, 1).all()
        assert (nv[f"p_lru_{m}"] <= nv[f"p_novel_{m}"] + 1e-12).all()
    assert mm.trip_novelty(v, trips.iloc[:0], MAZE).empty
    one = mm.trip_novelty(v, trips, MAZE, absorption={"u": mm.dead_end_absorption(MAZE)})
    assert "p_novel_u" in one and "p_novel_uniform" not in one


def test_novelty_vs_random_walk() -> None:
    rng = np.random.default_rng(0)
    p = np.full(200, 0.3)
    hi = mm.novelty_vs_random_walk(np.ones(200, bool), p, 500, rng)
    assert hi["excess"] == pytest.approx(0.7) and hi["p"] < 0.01 and hi["z"] > 5
    mid = mm.novelty_vs_random_walk(np.arange(200) % 10 < 3, p, 500, rng)
    assert abs(mid["excess"]) < 1e-12 and mid["p"] > 0.5
    empty = mm.novelty_vs_random_walk(np.zeros(0, bool), np.zeros(0), 10, rng)
    assert empty["n_trips"] == 0 and np.isnan(empty["p"])
    const = mm.novelty_vs_random_walk(np.ones(5, bool), np.ones(5), 10, rng)
    assert np.isnan(const["z"])


# ---------------------------------------------------------------------------
# Behaviour channels
# ---------------------------------------------------------------------------


def test_time_since() -> None:
    out = mm._time_since(np.array([2, 5]), 8, 2.0)
    assert np.isnan(out[:2]).all()
    assert out[2:].tolist() == [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]


def test_behaviour_channels() -> None:
    a, j, b = c(0, 0), c(1, 0), c(2, 0)
    v = _visits_from_path([a, j, b, j, a], frames=2)
    n = 10
    valid = np.ones(n, bool)
    valid[9] = False
    ch = mm.behaviour_channels(np.full(n, 90.0), np.arange(n), -np.ones(n), v, MAZE, 2.0, valid)
    assert ch.shape == (len(mm.BEHAVIOUR_CHANNELS), n)
    assert np.allclose(ch[0, :9], 1.0) and np.allclose(ch[1, :9], 0.0, atol=1e-12)
    assert np.allclose(ch[3, :9], 1.0) and np.allclose(ch[4, :9], np.arange(9) / 2.0)
    assert np.isnan(ch[5, :2]).all() and ch[5, 2] == 0.0  # first exit at frame 2
    assert ch[6, 0] == 0.0 and ch[6, 4] == 0.0  # entries at 0 and 4
    assert ch[7].tolist()[:9] == [1, 1, 1, 1, 2, 2, 2, 2, 2]
    assert np.isnan(ch[:, 9]).all()
    nv = mm.behaviour_channels(np.zeros(3), np.zeros(3), np.zeros(3), v.iloc[:0], MAZE, 1.0)
    assert np.isnan(nv[5]).all() and (nv[7] == 0).all()


def test_behaviour_window_features() -> None:
    x = np.vstack([np.full(10, 0.5), np.full(10, 0.5), np.arange(10.0), np.ones(10)])
    x = np.vstack([x, np.zeros((4, 10))])
    f = mm.behaviour_window_features(pm.WindowMeans(x), [0, 2], [4, 6])
    assert np.allclose(f[:, 0], np.sqrt(0.5)) and np.allclose(f[:, 2], [1.5, 3.5])
    x[:2] = 0.0
    f = mm.behaviour_window_features(pm.WindowMeans(x), [0], [4])
    assert np.isnan(f[0, :2]).all()


# ---------------------------------------------------------------------------
# Condition-matched selection
# ---------------------------------------------------------------------------


@settings(max_examples=40, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.integers(0, 2), st.integers(0, 3), st.sampled_from(["light", "dark", "mixed"])
        ),
        min_size=0,
        max_size=80,
    ),
    st.integers(1, 3),
    st.integers(1, 3),
)
def test_matched_condition_indices(rows: list, min_n: int, min_labels: int) -> None:
    s = np.array([r[0] for r in rows], dtype=np.int64)
    lab = np.array([r[1] for r in rows], dtype=np.int64)
    cnd = np.array([r[2] for r in rows], dtype=object)
    out = mm.matched_condition_indices(
        s, lab, cnd, ("light", "dark"), min_n, min_labels, np.random.default_rng(0)
    )
    li, di = out["light"], out["dark"]
    assert (cnd[li] == "light").all() and (cnd[di] == "dark").all()
    pl = sorted(zip(s[li], lab[li], strict=True))
    pd_ = sorted(zip(s[di], lab[di], strict=True))
    assert pl == pd_  # identical (stratum, label) composition
    for st_ in np.unique(s[li]):
        labs, cnt = np.unique(lab[li][s[li] == st_], return_counts=True)
        assert labs.size >= min_labels and cnt.min() >= min_n


def test_condition_index_sets() -> None:
    cnd = np.array(["light"] * 6 + ["dark"] * 4 + ["mixed"] * 5, dtype=object)
    lab = np.array([0, 1] * 7 + [0])
    out = mm.condition_index_sets(
        np.zeros(15), lab, cnd, ("light", "dark", "all"), 2, 2, np.random.default_rng(0)
    )
    assert out["light"].size == out["dark"].size == 4
    assert out["all"].size == 10 and not np.isin(np.arange(10, 15), out["all"]).any()
    none = mm.condition_index_sets(
        np.zeros(3),
        np.zeros(3),
        np.array(["mixed"] * 3),
        ("light", "all"),
        1,
        1,
        np.random.default_rng(0),
    )
    assert none["light"].size == 0 and none["all"].size == 0
    few = mm.matched_condition_indices(
        np.zeros(4),
        np.arange(4),
        np.array(["light"] * 4),
        ("light",),
        2,
        1,
        np.random.default_rng(0),
    )
    assert few["light"].size == 0


# ---------------------------------------------------------------------------
# Batched fits
# ---------------------------------------------------------------------------


def test_group_demean_shapes() -> None:
    g = np.array([0, 0, 1, 1])
    y = np.array([1.0, 3.0, 10.0, 14.0])
    rtr, rte, known = mm._group_demean(y, g, np.array([2.0, 5.0]), np.array([0, 7]))
    assert rtr.tolist() == [-1.0, 1.0, -2.0, 2.0] and rte[0] == 0.0 and known.tolist() == [1, 0]
    x2 = np.column_stack([y, 2 * y])
    r2, _, _ = mm._group_demean(x2, g, x2[:1], g[:1])
    assert np.allclose(r2[:, 1], 2 * rtr)
    x3 = np.stack([x2, -x2])
    r3, t3, _ = mm._group_demean(x3, g, x3[:, :2], g[:2])
    assert r3.shape == (2, 4, 2) and np.allclose(r3[1], -r2) and t3.shape == (2, 2, 2)


@settings(max_examples=30, deadline=None)
@given(st.integers(1, 3), st.integers(1, 12), st.integers(1, 4), st.integers(0, 10_000))
def test_impute_batched_matches_impute_columns(s: int, n: int, d: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(s, n, d))
    x[rng.random(x.shape) < 0.3] = np.nan
    out = mm._impute_batched(x)
    for k in range(s):
        assert np.allclose(out[k], pm.impute_columns(x[k]))


def test_prepare_fold() -> None:
    rng = np.random.default_rng(0)
    xs = rng.normal(size=(2, 30, 3))
    cov = rng.normal(size=(30, 2))
    xs[:, :, 0] += 3 * cov[:, :1].T
    tr, te = np.arange(20), np.arange(20, 30)
    ztr, zte, known = mm.prepare_fold(xs, tr, te, cov)
    assert ztr.shape == (2, 20, 3) and zte.shape == (2, 10, 3) and known.all()
    assert np.allclose(ztr.mean(axis=1), 0, atol=1e-12)
    d = np.column_stack([np.ones(20), cov[tr]])
    assert np.allclose(d.T @ ztr[0], 0, atol=1e-9)  # residualised on the covariates
    groups = np.r_[np.zeros(10), np.ones(10), np.full(10, 2)].astype(np.int64)
    _, _, known = mm.prepare_fold(xs, tr, te, None, groups)
    assert not known.any()
    xs[:, :, 2] = 1.0  # constant column -> sd guard
    ztr, _, _ = mm.prepare_fold(xs, tr, te)
    assert np.allclose(ztr[:, :, 2], 0.0)


def test_logistic_newton_matches_sklearn() -> None:
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 4))
    y = (x[:, 0] - x[:, 1] + rng.normal(size=60) > 0).astype(float)
    w = rng.uniform(0.5, 2.0, 60)
    ref = LogisticRegression(C=0.5, max_iter=5000, tol=1e-10).fit(x, y, sample_weight=w)
    b = mm.logistic_newton(x[None], y[None], w[None], C=0.5, max_iter=50, tol=1e-10)
    assert np.allclose(b[0, 0, :-1], ref.coef_[0], atol=1e-6)
    assert np.isclose(b[0, 0, -1], ref.intercept_[0], atol=1e-6)


def test_batched_logistic_predict() -> None:
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 20)
    x = rng.normal(size=(60, 3)) + 4 * np.eye(3)[y]
    xs = np.stack([x, rng.permutation(x)])
    folds = pm.time_block_folds(np.arange(60), np.arange(1, 61), 60, 3, 0)
    folds = [(np.setdiff1d(np.arange(60), te), te) for _, te in folds]
    rng.shuffle(y)  # spread classes over folds
    xs[0] = rng.normal(size=(60, 3)) + 4 * np.eye(3)[y]
    pred = mm.batched_logistic_predict(xs, y, folds, C=1.0)
    assert pred.shape == (2, 60)
    acc = mm.balanced_accuracy_batched(y, pred)
    assert acc[0] > 0.9 and acc[1] < 0.7
    yb = (y > 0).astype(int)
    pb = mm.batched_logistic_predict(xs[:1], yb, folds, C=1.0, cov=rng.normal(size=(60, 2)))
    assert mm.balanced_accuracy_batched(yb, pb)[0] > 0.85
    one = mm.batched_logistic_predict(xs, np.zeros(60, int), folds)
    assert (one == 0).all()
    none = mm.batched_logistic_predict(xs, y, [(np.zeros(0, int), np.arange(5))])
    assert (none == -1).all()


@settings(max_examples=40, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 2), st.integers(-1, 2)), min_size=1, max_size=40))
def test_balanced_accuracy_batched(pairs: list) -> None:
    y = np.array([p[0] for p in pairs])
    pr = np.array([p[1] for p in pairs])
    out = mm.balanced_accuracy_batched(y, np.stack([pr, y]))
    assert out[0] == pytest.approx(pm.balanced_accuracy(y, pr))
    assert out[1] == pytest.approx(1.0)


def test_balanced_accuracy_batched_empty() -> None:
    assert np.isnan(mm.balanced_accuracy_batched(np.zeros(0), np.zeros((2, 0)))).all()


def test_batched_ridge_predict() -> None:
    rng = np.random.default_rng(0)
    n = 120
    y = rng.integers(0, 6, n).astype(float)
    x = rng.normal(size=(n, 5))
    x[:, 0] += y
    xs = np.stack([x, rng.permutation(x)])
    st_, en = np.arange(n), np.arange(1, n + 1)
    folds = pm.time_block_folds(st_, en, n, 4, 0)
    pred, yev = mm.batched_ridge_predict(xs, y, folds, alpha=1.0)
    assert np.allclose(yev, y)
    assert stats.spearmanr(pred[0], y)[0] > 0.8
    assert abs(stats.spearmanr(pred[1], y)[0]) < 0.4
    groups = np.repeat(np.arange(4), n // 4)
    pg, yg = mm.batched_ridge_predict(xs, y, folds, 1.0, rng.normal(size=(n, 1)), groups)
    assert np.isnan(pg).all() and np.isnan(yg).all()  # each block is its own group
    groups = np.arange(n) % 3
    pg, yg = mm.batched_ridge_predict(xs, y, folds, 1.0, None, groups)
    assert np.isfinite(pg).all()
    for gi in range(3):
        assert abs(np.mean(yg[groups == gi])) < 1.0
    tiny = [(np.arange(2), np.arange(2, 5))]
    p0, _ = mm.batched_ridge_predict(xs, y, tiny)
    assert np.isnan(p0).all()


# ---------------------------------------------------------------------------
# Partial Spearman
# ---------------------------------------------------------------------------


@settings(max_examples=40, deadline=None)
@given(st.integers(6, 60), st.integers(0, 10_000))
def test_partial_spearman_properties(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x, y, z = rng.normal(size=(3, n))
    r = mm.partial_spearman(x, y, None)
    assert r == pytest.approx(stats.spearmanr(x, y)[0], abs=1e-9)
    p = mm.partial_spearman(x, y, z)
    assert -1.0 <= p <= 1.0
    assert mm.partial_spearman(np.exp(x), y**3, z) == pytest.approx(p, abs=1e-9)


def test_partial_spearman_removes_confound() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    x = z + 0.3 * rng.normal(size=500)
    y = z + 0.3 * rng.normal(size=500)
    assert mm.partial_spearman(x, y, None) > 0.7
    assert abs(mm.partial_spearman(x, y, z[:, None])) < 0.15


def test_partial_spearman_degenerate() -> None:
    assert np.isnan(mm.partial_spearman([1.0, 2.0], [1.0, 2.0], None))
    assert np.isnan(mm.partial_spearman(np.ones(10), np.arange(10.0), None))
    x = np.arange(10.0)
    x[3] = np.nan
    assert np.isfinite(mm.partial_spearman(x, np.arange(10.0), np.arange(10.0) % 3))


# ---------------------------------------------------------------------------
# Tasks and scoring
# ---------------------------------------------------------------------------


def test_windows_and_task_metrics() -> None:
    w = mm._Windows()
    assert w.arrays()[0].size == 0
    assert w.add([0, 1], [2, 3]).tolist() == [0, 1]
    assert w.add([5], [6]).tolist() == [2]
    s, e = w.arrays()
    assert s.tolist() == [0, 1, 5] and e.tolist() == [2, 3, 6]
    t = mm._skip("a", "b", "c", "light", "reg", "why")
    assert t.metrics == mm.REG_METRICS and t.reason == "why"
    assert mm._skip("a", "b", "c", "light", "clf", "x").metrics == mm.CLF_METRICS


def test_task_builders_empty() -> None:
    p = _params()
    rng = np.random.default_rng(0)
    pt = mm.passthrough_samples(mm.visit_table(np.full(3, -1), MAZE), pd.DataFrame(), MAZE)
    w = mm._Windows()
    tasks = mm.splitter_tasks(pt, np.zeros(0, dtype=object), MAZE, w, 100, 5, p, rng)
    tasks += mm.distance_tasks(pt, np.zeros(0), np.zeros(0, dtype=object), w, 100, 5, p, rng)
    dep = mm.departure_table(pt, pd.DataFrame(), pd.DataFrame(), np.ones(100), FPS, 100, p)
    tasks += mm.planning_tasks(dep, w, 100, 5, p, rng)
    assert len(tasks) == (2 + 4 + 4) * len(p.conditions)
    assert all(t.reason for t in tasks)


def test_departure_table() -> None:
    a, j, b = c(0, 0), c(1, 0), c(2, 0)
    cells = np.r_[[a] * 12, [j] * 3, [b] * 3, [j] * 3, [a] * 3]
    v = mm.visit_table(cells, MAZE, 2)
    trips = mm.trip_table(v, MAZE)
    nov = mm.trip_novelty(v, trips, MAZE)
    p = _params(min_dead_end_dwell_s=0.5, plan_window_s=1.0)
    dep = mm.departure_table(v, trips, nov, np.ones(cells.size, bool), 10.0, cells.size, p)
    assert len(dep) == 1  # second departure dwell 0.3 s < 0.5 s
    r = dep.iloc[0]
    assert r["exit"] == 12 and r["dwell_s"] == pytest.approx(1.2)
    assert (r["pre_start"], r["pre_end"], r["post_start"], r["post_end"]) == (2, 12, 12, 22)
    assert r["pre_ok"] and r["post_ok"] and r["pre_cond"] == "light"


def test_score_task_and_parallel_equal() -> None:
    w = _walk(3000, 3)
    sig = _origin_signal(w, 9, 4)
    p = _params(n_jobs=1)
    rng = np.random.default_rng(0)
    cells = pm.session_cells(w["x"], w["y"], w["valid"], MAZE, 2)
    v = mm.visit_table(cells, MAZE, 10, 2)
    trips = mm.trip_table(v, MAZE)
    pt = mm.passthrough_samples(v, trips, MAZE)
    cond = np.full(len(pt), "light", dtype=object)
    win = mm._Windows()
    tasks = mm.splitter_tasks(pt, cond, MAZE, win, 3000, 48, p, rng)
    tasks += mm.distance_tasks(pt, np.ones(len(pt)), cond, win, 3000, 48, p, rng)
    s, e = win.arrays()
    beh = mm.behaviour_channels(w["hd"], w["speed"], w["ahv"], v, MAZE, FPS)
    bwm = pm.WindowMeans(beh)
    data = mm.SessionData(
        pm.WindowMeans(sig), bwm, s, e, mm.behaviour_window_features(bwm, s, e)[:, :2], p
    )
    shifts = np.array([0, 400, 900, 1500, 2000])
    one = mm.evaluate_tasks(tasks, data, shifts, mm.FEATURE_SETS, None, 1)
    two = mm.evaluate_tasks(tasks, data, shifts, mm.FEATURE_SETS, None, 2)
    assert one.keys() == two.keys() and len(one) > 0
    for k in one:
        assert np.allclose(one[k], two[k], equal_nan=True)
    live = [i for i, t in enumerate(tasks) if not t.reason]
    assert any(tasks[i].kind == "reg" for i in live)
    data.params = _params(shift_chunk=2)
    chunked = mm.evaluate_tasks(tasks, data, shifts)
    for k in one:
        assert np.allclose(one[k], chunked[k], equal_nan=True)


def test_diff_null_summary() -> None:
    res = mm.diff_null_summary(0.0, np.array([-1.0, 1.0, -1.0, 1.0]))
    assert res["p"] == 1.0
    res = mm.diff_null_summary(-5.0, np.linspace(-1, 1, 19))
    assert res["p"] == pytest.approx(1 / 20) and res["excess"] == pytest.approx(-5.0)
    assert np.isnan(mm.diff_null_summary(np.nan, np.ones(3))["p"])


# ---------------------------------------------------------------------------
# Session entry point
# ---------------------------------------------------------------------------


def test_session_outputs(session: tuple[dict[str, np.ndarray], mm.MazeMemResult]) -> None:
    _, res = session
    s = res.sessions
    assert set(s["analysis"]) == set(mm.ANALYSES)
    assert set(s["condition"]) == set(mm.CONDITIONS)
    assert set(s["feature_set"]) == set(mm.FEATURE_SETS)
    assert mm.PRO_MINUS_RETRO in set(s["target"])
    keys = mm.SESSION_KEYS
    assert not s.duplicated(keys).any()
    assert res.counts["n_trips"] > 20 and res.counts["n_shifts"] == 8
    assert set(res.behaviour["null_model"]) == set(mm.RW_MODELS)
    assert not res.subsets.empty and (res.subsets["n_cells_subset"] == 3).all()
    run = s[s["reason"] == ""]
    assert (run["n_null"] == 8).all() and run["observed"].notna().all()


def test_session_detects_origin_code(
    session: tuple[dict[str, np.ndarray], mm.MazeMemResult],
) -> None:
    _, res = session
    s = res.sessions.set_index(mm.SESSION_KEYS)
    key = ("splitter", "origin", "passthrough", "all", "neural", "balanced_accuracy")
    row = s.loc[key]
    assert row["observed"] > 0.8 and row["excess"] > 0.2 and row["p"] <= 1 / 9


def test_session_short_and_light_only() -> None:
    w = _walk(1500, 5)
    sig = np.random.default_rng(0).poisson(0.3, (4, 1500)).astype(float)
    p = _params(conditions=("light",), min_shift_s=100.0, subset_size=10)
    res = mm.maze_memory_session(
        sig,
        w["x"],
        w["y"],
        w["speed"],
        w["hd"],
        w["ahv"],
        np.ones(1500, bool),
        w["valid"],
        FPS,
        p,
    )
    assert set(res.sessions["condition"]) == {"light"}
    assert (res.sessions["reason"] != "").all()  # too short for the null, or too few samples
    assert res.subsets.empty


def test_behaviour_rows_and_schema(
    session: tuple[dict[str, np.ndarray], mm.MazeMemResult],
) -> None:
    _, res = session
    b = res.behaviour
    assert len(b) == len(mm.CONDITIONS) * 2 * len(mm.RW_MODELS)
    assert b["expected_frac"].between(0, 1).all()
    rows = mm.behaviour_as_session_rows(b.assign(exp_id="e1", animal_id="a1"))
    assert set(rows["feature_set"]) == {f"observed_vs_{m}_walk" for m in mm.RW_MODELS}
    assert (rows["exp_id"] == "e1").all()
    assert mm.behaviour_as_session_rows(pd.DataFrame()).empty
    empty = mm.behaviour_rows(
        pd.DataFrame(columns=mm.TRIP_COLUMNS),
        mm.trip_novelty(pd.DataFrame(columns=mm.VISIT_COLUMNS), pd.DataFrame(), MAZE),
        np.zeros(0, dtype=object),
        MAZE,
        FPS,
        _params(),
        np.random.default_rng(0),
    )
    assert (empty["reason"] == "no trips").all()


# ---------------------------------------------------------------------------
# Across-session statistics
# ---------------------------------------------------------------------------


def _multi_session(res: mm.MazeMemResult) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    parts = []
    for i, (animal, ct) in enumerate(
        [("a1", "penk"), ("a1", "penk"), ("a2", "nonpenk"), ("a3", "penk")]
    ):
        d = res.sessions.copy()
        d["excess"] = d["excess"] + rng.normal(0, 0.01, len(d))
        d["exp_id"], d["animal_id"], d["celltype"] = f"e{i}", animal, ct
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def test_across_session_summary(session: tuple[dict[str, np.ndarray], mm.MazeMemResult]) -> None:
    df = _multi_session(session[1])
    summ = mm.across_session_summary(df)
    assert set(summ["group"]) == {"all", "penk", "nonpenk"}
    a = summ[(summ["group"] == "all") & (summ["n_sessions"] == 4)]
    assert not a.empty and a["session_wilcoxon_p"].notna().any()
    assert (a["n_animals"] == 3).all()
    assert mm.across_session_summary(pd.DataFrame()).empty


def test_paired_comparisons(session: tuple[dict[str, np.ndarray], mm.MazeMemResult]) -> None:
    df = _multi_session(session[1])
    comp = mm.paired_comparisons(df)
    assert set(comp["comparison_type"]) == {
        "feature_set",
        "light_vs_dark",
        "prospective_vs_retrospective",
    }
    lvd = comp[comp["comparison_type"] == "light_vs_dark"]
    assert (lvd["condition"] == "light - dark").all()
    pr = comp[comp["comparison_type"] == "prospective_vs_retrospective"]
    assert (pr["target"] == "destination - origin").all()
    assert mm.paired_comparisons(pd.DataFrame()).empty


def test_subset_summary(session: tuple[dict[str, np.ndarray], mm.MazeMemResult]) -> None:
    sub = session[1].subsets
    df = pd.concat([sub.assign(exp_id="e1"), sub.assign(exp_id="e2")], ignore_index=True)
    out = mm.subset_summary(df)
    assert (out["n_sessions"] == 2).all() and "median_excess" in out
    assert mm.subset_summary(pd.DataFrame()).empty
