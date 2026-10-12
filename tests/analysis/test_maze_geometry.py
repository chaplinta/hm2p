"""Tests for ``hm2p.analysis.maze_geometry`` (small synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis import maze_geometry as mg
from hm2p.maze.topology import build_rose_maze

MAZE = build_rose_maze()
IDX = MAZE.cell_to_idx
FPS = 10.0


# ---------------------------------------------------------------------------
# Synthetic random walk on the maze graph and synthetic populations
# ---------------------------------------------------------------------------


def _walk(n_frames: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cell index, direction, head direction (deg) and speed per frame."""
    rng = np.random.default_rng(seed)
    cur, prev = IDX[(3, 2)], -1
    cells: list[int] = []
    while len(cells) < n_frames:
        cells += [cur] * int(rng.integers(4, 9))
        nb = [IDX[c] for c in MAZE.adj[MAZE.cell_list[cur]]]
        opts = [c for c in nb if c != prev] or nb
        prev, cur = cur, int(rng.choice(opts))
    c = np.array(cells[:n_frames])
    dirs = mg.frame_directions(c, MAZE)
    hd = np.where(dirs >= 0, 90.0 * dirs, rng.uniform(0, 360, n_frames)) + rng.normal(
        0, 10, n_frames
    )
    return c, dirs, hd, np.full(n_frames, 5.0)


def _population(kind: str, c: np.ndarray, dirs: np.ndarray, seed: int, n: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed + 100)
    cen = mg.cell_centres(MAZE)
    pref = rng.integers(0, 23, n)
    if kind == "graph":
        r = np.exp(-MAZE.dist[pref][:, c] / 3.0)
    elif kind == "metric":
        dd = np.linalg.norm(cen[pref][:, None, :] - cen[c][None], axis=2)
        r = np.exp(-(dd**2) / (2 * 2.0**2))
    else:  # role
        depth = mg.root_depth(MAZE)
        roles = [
            mg.state_role(MAZE.cell_list[a], int(b), MAZE, depth) if b >= 0 else "none"
            for a, b in zip(c, dirs, strict=True)
        ]
        names = sorted(set(roles))
        w = rng.normal(0, 1, (n, len(names)))
        r = w[:, [names.index(x) for x in roles]]
    return r + rng.normal(0, 1.0, r.shape)


def _analyse(kind: str, directed: bool, seed: int = 0) -> dict:
    c, dirs, hd, spd = _walk(16000, seed)
    x = mg.zscore_rows(_population(kind, c, dirs, seed), np.ones(c.size, bool))
    st_ = mg.frame_states(c, dirs, np.ones(c.size, bool), directed)
    p = mg.GeometryParams(n_shifts=15, n_perm=60)
    rng = np.random.default_rng(seed)
    sh = mg.draw_shifts(c.size, p.n_shifts, 300, rng)
    ns = 23 * 4 if directed else 23
    return mg.analyse_states(
        x, st_, ns, hd, spd, FPS, MAZE, directed, p, sh, rng, do_roles=directed
    )


# ---------------------------------------------------------------------------
# Maze geometry helpers
# ---------------------------------------------------------------------------


def test_cell_centres_shape_and_values():
    cen = mg.cell_centres(MAZE)
    assert cen.shape == (23, 2)
    assert np.allclose(cen[IDX[(0, 0)]], [0.5, 0.5])


def test_root_depth():
    dep = mg.root_depth(MAZE)
    assert dep[IDX[(3, 2)]] == 0
    assert dep[IDX[(3, 4)]] == 2
    assert dep[IDX[(0, 0)]] == 5


def test_direction_index():
    assert mg.direction_index(1, 0) == 0
    assert mg.direction_index(0, -1) == 3
    assert mg.direction_index(1, 1) == -1


@given(st.floats(-720, 720), st.floats(-720, 720))
@settings(max_examples=50, deadline=None)
def test_angle_diff_range_and_symmetry(a, b):
    d = float(mg.angle_diff_deg(a, b))
    assert 0 <= d <= 180 + 1e-9
    assert np.isclose(d, float(mg.angle_diff_deg(b, a)))


def test_angle_diff_examples():
    assert np.allclose(mg.angle_diff_deg([0, 350, 90], [270, 10, 270]), [90, 20, 180])


# ---------------------------------------------------------------------------
# Directions, states and roles
# ---------------------------------------------------------------------------


def test_frame_directions_straight_and_turn():
    a, b, c = IDX[(1, 1)], IDX[(1, 2)], IDX[(2, 2)]
    cells = np.array([a] * 4 + [b] * 4 + [c] * 4)
    d = mg.frame_directions(cells, MAZE)
    assert (d[:2] == -1).all()  # no previous stay
    assert (d[2:4] == 1).all()  # exit north to (1, 2)
    assert (d[4:6] == 1).all()  # entered (1, 2) moving north
    assert (d[6:8] == 0).all()  # exits east to (2, 2)
    assert (d[8:10] == 0).all()
    assert (d[10:] == -1).all()  # no next stay


def test_frame_directions_unassigned_and_nonadjacent():
    a, far = IDX[(0, 0)], IDX[(6, 4)]
    cells = np.array([a, a, -1, -1, far, far])
    d = mg.frame_directions(cells, MAZE)
    assert (d == -1).all()


def test_frame_states_modes():
    cells = np.array([0, 1, -1, 2])
    dirs = np.array([1, -1, 0, 3])
    m = np.array([True, True, True, False])
    assert list(mg.frame_states(cells, dirs, m, False)) == [0, 1, -1, -1]
    assert list(mg.frame_states(cells, dirs, m, True)) == [1, -1, -1, -1]


@pytest.mark.parametrize(
    ("cell", "direction", "role"),
    [
        ((0, 0), 2, "dead_end_in"),  # moving W into (0, 0)
        ((0, 0), 0, "dead_end_out"),
        ((3, 4), 1, "dead_end_in"),
        ((2, 2), 2, "corridor_out"),  # away from root
        ((2, 2), 0, "corridor_in"),
        ((2, 2), 1, "corridor_other"),
        ((1, 0), 3, "t_from_stem"),  # arrived from stem (1, 1) moving S
        ((1, 0), 1, "t_to_stem"),
        ((1, 0), 0, "t_bar"),
        ((1, 2), 2, "t_from_stem"),  # stem of (1, 2) is (2, 2)
        ((3, 2), 0, "root"),
    ],
)
def test_state_role(cell, direction, role):
    assert mg.state_role(cell, direction, MAZE, mg.root_depth(MAZE)) == role


def test_role_group():
    assert mg.role_group("dead_end_in") == "dead_end"
    assert mg.role_group("corridor_out") == "corridor"
    assert mg.role_group("t_bar") == "t"
    assert mg.role_group("root") == "root"


# ---------------------------------------------------------------------------
# Halves, means, crossnobis
# ---------------------------------------------------------------------------


def test_time_halves_alternate_and_gap():
    h = mg.time_halves(100, 10.0, block_s=2.0, gap_s=0.5)
    assert h[0] == -1  # within gap of the first boundary
    assert h[10] == 0
    assert h[30] == 1
    assert set(np.unique(h)) <= {-1, 0, 1}


def test_time_halves_offset_changes_assignment():
    a = mg.time_halves(100, 10.0, 2.0, 0.0, 0.0)
    b = mg.time_halves(100, 10.0, 2.0, 0.0, 0.5)
    assert not np.array_equal(a, b)


def test_state_means_values():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    s = np.array([0, 0, 1, -1])
    h = np.array([0, 1, 0, 1])
    m, c = mg.state_means(x, s, h, 2)
    assert m[0, 0, 0] == 1 and m[1, 0, 0] == 2 and m[0, 1, 0] == 3
    assert np.isnan(m[1, 1, 0])
    assert c.tolist() == [[1, 1], [1, 0]]


def test_crossnobis_noiseless_equals_squared_distance():
    a = np.array([[0.0, 0.0], [3.0, 4.0]])
    d = mg.crossnobis(a, a)
    assert np.isclose(d[0, 1], 25 / 2) and d[0, 0] == 0


@given(st.integers(0, 10_000))
@settings(max_examples=20, deadline=None)
def test_crossnobis_unbiased_under_null(seed):
    rng = np.random.default_rng(seed)
    vals = [
        mg.crossnobis(rng.normal(size=(5, 50)), rng.normal(size=(5, 50)))[0, 1] for _ in range(50)
    ]
    assert abs(np.mean(vals)) < 0.5  # plain distance would be ~2


def test_cv_dissimilarity_and_kept_states():
    c, dirs, hd, spd = _walk(3000, 1)
    x = mg.zscore_rows(_population("graph", c, dirs, 1), np.ones(c.size, bool))
    s = mg.frame_states(c, dirs, np.ones(c.size, bool), False)
    p = mg.GeometryParams()
    keep = mg.kept_states(s, 23, s.size, FPS, p)
    d, kept = mg.cv_dissimilarity(x, s, 23, FPS, p)
    assert np.array_equal(keep, kept)
    assert d.shape == (kept.size, kept.size)
    assert np.allclose(d, d.T)
    d2, _ = mg.cv_dissimilarity(x, s, 23, FPS, p, shift=500, keep=kept)
    assert d2.shape == d.shape and not np.allclose(d, d2)


def test_cv_dissimilarity_empty():
    p = mg.GeometryParams(min_frames=10_000)
    d, k = mg.cv_dissimilarity(np.zeros((2, 100)), np.zeros(100, int), 3, FPS, p)
    assert d.shape == (0, 0) and k.size == 0


# ---------------------------------------------------------------------------
# Descriptors and pair table
# ---------------------------------------------------------------------------


def _desc_pairs(directed: bool):
    c, dirs, hd, spd = _walk(3000, 2)
    s = mg.frame_states(c, dirs, np.ones(c.size, bool), directed)
    ns = 92 if directed else 23
    kept = mg.kept_states(s, ns, s.size, FPS, mg.GeometryParams())
    desc = mg.state_descriptors(s, kept, hd, spd, FPS, 60.0)
    return desc, mg.pair_table(desc, MAZE, directed)


def test_state_descriptors_columns():
    desc, _ = _desc_pairs(False)
    for col in ("state", "n_frames", "mean_time_s", "speed", "hd_c", "profile"):
        assert col in desc
    assert np.isclose(np.linalg.norm(desc["profile"].iloc[0]), 1.0)


def test_pair_table_cell_mode():
    desc, pairs = _desc_pairs(False)
    n = len(desc)
    assert len(pairs) == n * (n - 1) // 2
    row = pairs.iloc[0]
    assert row["graph"] == MAZE.dist[row["cell_i"], row["cell_j"]]
    assert "dir_diff" not in pairs


def test_pair_table_directed_mode():
    _, pairs = _desc_pairs(True)
    assert set(pairs["dir_diff"].unique()) <= {0.0, 90.0, 180.0}
    assert {"role_i", "role_j", "same_role"} <= set(pairs.columns)
    assert pairs["same_cell"].any()


def test_walled_pair_graph_far_euclid_near():
    i, j = IDX[(2, 4)], IDX[(3, 4)]
    assert MAZE.dist[i, j] == 7
    cen = mg.cell_centres(MAZE)
    assert np.isclose(np.linalg.norm(cen[i] - cen[j]), 1.0)


# ---------------------------------------------------------------------------
# Partial Spearman
# ---------------------------------------------------------------------------


def test_partial_spearman_removes_shared_driver():
    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    x = z + 0.1 * rng.normal(size=500)
    y = z + 0.1 * rng.normal(size=500)
    assert mg.partial_spearman(y, x) > 0.9
    assert abs(mg.partial_spearman(y, x, z[:, None])) < 0.15


def test_partial_spearman_nan_cases():
    assert np.isnan(mg.partial_spearman([1, 2], [1, 2]))
    assert np.isnan(mg.partial_spearman([1, 1, 1, 1, 1], [1, 2, 3, 4, 5]))


@given(st.integers(0, 10_000))
@settings(max_examples=20, deadline=None)
def test_partial_design_matches_partial_spearman(seed):
    rng = np.random.default_rng(seed)
    x, y = rng.normal(size=60), rng.normal(size=60)
    c = rng.normal(size=(60, 3))
    des = mg.PartialDesign(x, c)
    assert np.isclose(des(y), mg.partial_spearman(y, x, c))


def test_partial_design_rows_and_invalid():
    rng = np.random.default_rng(1)
    x, y, c = rng.normal(size=40), rng.normal(size=40), rng.normal(size=(40, 1))
    rows = np.arange(40) < 20
    assert np.isclose(mg.PartialDesign(x, c, rows)(y), mg.partial_spearman(y[:20], x[:20], c[:20]))
    assert np.isnan(mg.PartialDesign(x[:3], c[:3])(y[:3]))
    yy = y.copy()
    yy[0] = np.nan
    assert np.isnan(mg.PartialDesign(x, c)(yy))


def test_geometry_design_keys():
    _, pairs = _desc_pairs(True)
    des = mg.geometry_design(pairs, True)
    assert "prho_dir" in des and "prho_graph_local" in des


# ---------------------------------------------------------------------------
# Role contrast
# ---------------------------------------------------------------------------


def _toy_pairs():
    return pd.DataFrame(
        {
            "i": [0, 0, 1, 0],
            "j": [1, 2, 2, 3],
            "cell_i": [0, 0, 1, 0],
            "cell_j": [1, 1, 2, 4],
            "same_cell": [False, False, False, False],
            "role_i": ["a_x", "a_x", "b_y", "a_x"],
            "role_j": ["b_y", "a_x", "a_x", "root"],
            "graph": [1.0, 1.0, 1.0, 2.0],
            "euclid": [1.0, 1.0, 1.0, 2.0],
            "dir_diff": [0.0, 0.0, 90.0, 0.0],
        }
    )


def test_role_strata_modes():
    p = _toy_pairs()
    cp = mg.role_strata(p, "cellpair")
    assert cp[0] == cp[1] and cp[3] == -1
    ge = mg.role_strata(p, "geometry")
    assert ge[0] == ge[1] != ge[2]
    with pytest.raises(ValueError):
        mg.role_strata(p, "bogus")


def test_role_contrast_weighted():
    r = np.array([1.0, 0.0, 3.0, 5.0, 9.0])
    same = np.array([False, True, False, True, False])
    strata = np.array([0, 0, 1, 1, -1])
    val, n_same = mg.role_contrast(r, same, strata)
    assert np.isclose(val, ((1 - 0) + (3 - 5)) / 2)
    assert n_same == 2
    assert np.isnan(mg.role_contrast(r, np.zeros(5, bool), strata)[0])
    assert np.isnan(mg.role_contrast(r, same, -np.ones(5, int))[0])


def test_null_summary():
    out = mg.null_summary(3.0, [0.0, 1.0, 2.0, 1.0])
    assert out["p"] == pytest.approx(1 / 5)
    assert out["z"] > 0
    assert np.isnan(mg.null_summary(np.nan, [1, 2])["z"])


def test_zscore_rows_and_shifts():
    x = np.array([[1.0, 2.0, 3.0, np.nan], [5.0, 5.0, 5.0, 5.0]])
    z = mg.zscore_rows(x, np.array([True, True, True, False]))
    assert np.isclose(z[0, :3].mean(), 0) and (z[1] == 0).all()
    assert (mg.zscore_rows(x, np.zeros(4, bool)) == 0).all()
    rng = np.random.default_rng(0)
    s = mg.draw_shifts(1000, 20, 100, rng)
    assert s.size == 20 and s.min() >= 100 and s.max() < 900
    assert mg.draw_shifts(100, 20, 60, rng).size == 0


def test_mantel_null_shapes():
    _, pairs = _desc_pairs(False)
    n = int(max(pairs["i"].max(), pairs["j"].max()) + 1)
    rng = np.random.default_rng(0)
    d = rng.normal(size=(n, n))
    d = d + d.T
    out = mg.mantel_null(d, pairs, False, 5, rng, ("prho_graph",))
    assert out["prho_graph"].shape == (5,)


# ---------------------------------------------------------------------------
# Method validation on synthetic populations
# ---------------------------------------------------------------------------


def test_graph_population_graph_beats_euclid():
    r = _analyse("graph", directed=False)
    assert r["prho_graph_nt"] > r["prho_euclid_nt"]
    assert r["prho_graph_local"] > 0.3
    assert r["prho_graph_local_z"] > 2


def test_metric_population_euclid_beats_graph():
    r = _analyse("metric", directed=False)
    assert r["prho_euclid"] > 0.5
    assert r["prho_euclid"] > r["prho_graph"] + 0.3
    assert r["prho_euclid_z"] > 2


def test_role_population_generalises_and_place_does_not():
    rr = _analyse("role", directed=True)
    assert rr["role_contrast"] > 0.2 and rr["role_z"] > 4
    rg = _analyse("graph", directed=True)
    assert rg["role_z"] < 2.5
    assert rg["role_n_types_repeated"] >= 5


def test_analyse_states_too_few_states():
    c, dirs, hd, spd = _walk(500, 3)
    s = mg.frame_states(c, dirs, np.ones(c.size, bool), False)
    out = mg.analyse_states(
        np.zeros((3, c.size)),
        s,
        23,
        hd,
        spd,
        FPS,
        MAZE,
        False,
        mg.GeometryParams(),
        np.zeros(0, int),
        np.random.default_rng(0),
        keep=np.array([0, 1]),
    )
    assert out["reason"] == "fewer than 6 states"


# ---------------------------------------------------------------------------
# Session orchestration and summaries
# ---------------------------------------------------------------------------


def _session_dict(seed: int, animal: str) -> dict:
    c, dirs, _, _ = _walk(6000, seed)
    cen = mg.cell_centres(MAZE)
    rng = np.random.default_rng(seed)
    xy = cen[c] + rng.uniform(-0.3, 0.3, (c.size, 2))
    n = c.size
    return {
        "spikes": np.clip(_population("graph", c, dirs, seed), 0, None),
        "x_maze": xy[:, 0],
        "y_maze": xy[:, 1],
        "bad_behav": np.zeros(n, bool),
        "speed_cm_s": np.full(n, 6.0),
        "light_on": (np.arange(n) // 600) % 2 == 0,
        "hd_deg": rng.uniform(0, 360, n),
        "fps": FPS,
        "meta": {"exp_id": f"s{seed}", "animal_id": animal, "celltype": "penk"},
    }


def test_session_inputs_and_analyse_session():
    d = _session_dict(4, "a1")
    p = mg.GeometryParams(n_shifts=3, n_perm=5, min_frames=4)
    inp = mg.session_inputs(d, MAZE, p)
    assert inp["x"].shape == d["spikes"].shape and inp["run"].any()
    rows = mg.analyse_session(d, MAZE, p, modes=("cell",))
    assert [r["condition"] for r in rows] == ["all", "light", "dark"]
    assert rows[1]["n_states"] == rows[2]["n_states"]
    assert rows[0]["animal_id"] == "a1"


def test_wilcoxon_summary_and_animal_means():
    w = mg.wilcoxon_summary([1.0, 2.0, 3.0, 4.0, 5.0, -0.5])
    assert w["n"] == 6 and w["n_pos"] == 5 and 0 < w["rank_biserial"] <= 1
    assert np.isnan(mg.wilcoxon_summary([1.0, 0.0])["p"])
    df = pd.DataFrame({"animal_id": ["a", "a", "b"], "v": [1.0, 3.0, np.nan]})
    assert mg.animal_means(df, "v").to_dict() == {"a": 2.0}


def test_across_animals_and_light_dark():
    rows = []
    rng = np.random.default_rng(0)
    for k in range(8):
        for cond in ("all", "light", "dark"):
            rows.append(
                {
                    "exp_id": f"s{k}",
                    "animal_id": f"a{k // 2}",
                    "mode": "cell",
                    "condition": cond,
                    "prho_graph_z": rng.normal(1.0 if cond != "dark" else 0.0),
                    "prho_graph_p": 0.01,
                }
            )
    df = pd.DataFrame(rows)
    aa = mg.across_animals(df, ("prho_graph_z",))
    assert set(aa["condition"]) == {"all", "light", "dark"}
    assert (aa["n_animals"] == 4).all() and (aa["n_sessions_p05"] == 8).all()
    ld = mg.light_dark_paired(df, ("prho_graph_z", "missing"))
    assert len(ld) == 1 and ld["n_sessions"].iloc[0] == 8
