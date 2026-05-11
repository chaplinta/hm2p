"""Tests for hm2p.maze.neural -- maze-resolved neural analysis.

All tests use synthetic data only. The module is imported via pytest.importorskip
so all tests skip cleanly if the module is not yet available.

Reference: Koren Iton A et al. 2025. "NaviGraph." bioRxiv.
doi:10.1101/2025.05.18.654725
"""
from __future__ import annotations
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats as sp_stats
from hm2p.maze.topology import RoseMaze, build_rose_maze

neural = pytest.importorskip("hm2p.maze.neural",
                             reason="hm2p.maze.neural not yet implemented")

# ---------------------------------------------------------------------------
@pytest.fixture
def maze() -> RoseMaze:
    return build_rose_maze()

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)

def _walk(maze, rng, n=200):
    cl = maze.cell_list
    idx = rng.integers(0, len(cl))
    out = []
    for _ in range(n):
        if rng.random() < 0.05:
            out.append(-1)
        else:
            out.append(idx)
            nbs = maze.adj[cl[idx]]
            idx = maze.cell_to_idx[nbs[rng.integers(len(nbs))]]
    return np.array(out, dtype=np.int32)

def _sig(rng, nr, n):
    return rng.standard_normal((nr, n)).astype(np.float64) * 0.1 + 0.5

def _hd(rng, n):
    return rng.uniform(0, 360, size=n).astype(np.float64)

# =========================================================================
# node_activity_map -> ndarray (n_rois, n_cells)
# =========================================================================
class TestNodeActivityMap:
    def _call(self, sig, ci, mask, nc):
        """Wrapper to handle both tuple and ndarray return."""
        r = neural.node_activity_map(sig, ci, mask, nc)
        if isinstance(r, tuple):
            return r[0], r[1]
        return r, None

    def test_shape(self, maze, rng):
        act, _ = self._call(_sig(rng, 5, 100), _walk(maze, rng, 100),
                            np.ones(100, dtype=bool), maze.n_cells)
        assert act.shape == (5, maze.n_cells)

    def test_nan_unvisited(self, maze, rng):
        act, _ = self._call(_sig(rng, 3, 50), np.zeros(50, dtype=np.int32),
                            np.ones(50, dtype=bool), maze.n_cells)
        for c in range(1, maze.n_cells):
            assert np.all(np.isnan(act[:, c]))
        assert np.all(np.isfinite(act[:, 0]))

    def test_condition_mask(self, maze, rng):
        sig = _sig(rng, 2, 60)
        ci = np.zeros(60, dtype=np.int32)
        mask = np.zeros(60, dtype=bool); mask[:30] = True
        act, _ = self._call(sig, ci, mask, maze.n_cells)
        np.testing.assert_allclose(act[:, 0], np.mean(sig[:, :30], axis=1))

    def test_all_invalid(self, maze, rng):
        act, _ = self._call(_sig(rng, 2, 40), np.full(40, -1, dtype=np.int32),
                            np.ones(40, dtype=bool), maze.n_cells)
        assert np.all(np.isnan(act))

    def test_1d(self, maze, rng):
        act, _ = self._call(rng.standard_normal(50), _walk(maze, rng, 50),
                            np.ones(50, dtype=bool), maze.n_cells)
        assert act.shape == (1, maze.n_cells)

    def test_empty_mask(self, maze, rng):
        act, _ = self._call(_sig(rng, 2, 30), _walk(maze, rng, 30),
                            np.zeros(30, dtype=bool), maze.n_cells)
        assert np.all(np.isnan(act))

    def test_mean(self, maze):
        act, _ = self._call(np.array([[10., 20., 30.]]),
                            np.array([0, 1, 0], dtype=np.int32),
                            np.ones(3, dtype=bool), maze.n_cells)
        np.testing.assert_allclose(act[0, 0], 20.0)
        np.testing.assert_allclose(act[0, 1], 20.0)

    def test_empty(self, maze):
        act, _ = self._call(np.empty((2, 0)), np.array([], dtype=np.int32),
                            np.array([], dtype=bool), maze.n_cells)
        assert np.all(np.isnan(act))

    @given(nr=st.integers(1, 10), n=st.integers(5, 100), s=st.integers(0, 9999))
    @settings(max_examples=30, deadline=None)
    def test_prop_shape(self, nr, n, s):
        rng = np.random.default_rng(s); m = build_rose_maze()
        act, _ = self._call(rng.standard_normal((nr, n)),
                            rng.integers(-1, m.n_cells, size=n).astype(np.int32),
                            rng.random(n) > 0.3, m.n_cells)
        assert act.shape == (nr, m.n_cells)

    @given(n=st.integers(10, 100), s=st.integers(0, 9999))
    @settings(max_examples=30, deadline=None)
    def test_prop_nan(self, n, s):
        rng = np.random.default_rng(s); m = build_rose_maze()
        ci = rng.integers(-1, m.n_cells, size=n).astype(np.int32)
        mask = rng.random(n) > 0.2
        act, _ = self._call(rng.standard_normal((3, n)), ci, mask, m.n_cells)
        valid = mask & (ci >= 0) & (ci < m.n_cells)
        occ = np.zeros(m.n_cells, dtype=int)
        if valid.any(): np.add.at(occ, ci[valid], 1)
        for c in range(m.n_cells):
            if occ[c] == 0: assert np.all(np.isnan(act[:, c]))
            else: assert np.all(np.isfinite(act[:, c]))

# =========================================================================
# light_dark_node_contrast -> tuple or dict
# =========================================================================
class TestLightDarkNodeContrast:
    def _call(self, *args, **kw):
        r = neural.light_dark_node_contrast(*args, **kw)
        if isinstance(r, dict):
            return r["delta"], r.get("activity_light"), r.get("activity_dark"), r.get("valid_cells")
        # tuple: (delta, light_map, dark_map)
        return r[0], r[1], r[2], None

    def test_shapes(self, maze, rng):
        nr = 3
        d, lm, dm, _ = self._call(
            _sig(rng, nr, 200), _walk(maze, rng, 200),
            rng.random(200) > 0.5, np.ones(200, dtype=bool), maze.n_cells, min_frames=2)
        for a in (d, lm, dm):
            assert a.shape == (nr, maze.n_cells)

    def test_delta_consistency(self, maze, rng):
        d, lm, dm, vc = self._call(
            _sig(rng, 2, 400), _walk(maze, rng, 400),
            rng.random(400) > 0.5, np.ones(400, dtype=bool), maze.n_cells, min_frames=2)
        f = np.isfinite(d) & np.isfinite(lm) & np.isfinite(dm)
        if f.any():
            diff = dm[f] - lm[f]
            close = np.allclose(d[f], diff, atol=1e-12) or np.allclose(d[f], -diff, atol=1e-12)
            assert close

    def test_min_frames_nan(self, maze, rng):
        ci = np.zeros(40, dtype=np.int32)
        lo = np.zeros(40, dtype=bool); lo[:10] = True
        d, _, _, _ = self._call(_sig(rng, 2, 40), ci, lo,
                                np.ones(40, dtype=bool), maze.n_cells, min_frames=20)
        assert np.all(np.isnan(d[:, 0]))

    def test_min_frames_met(self, maze, rng):
        ci = np.zeros(100, dtype=np.int32)
        lo = np.zeros(100, dtype=bool); lo[:50] = True
        d, _, _, _ = self._call(_sig(rng, 2, 100), ci, lo,
                                np.ones(100, dtype=bool), maze.n_cells, min_frames=20)
        assert np.all(np.isfinite(d[:, 0]))

    def test_all_light(self, maze, rng):
        d, _, _, _ = self._call(_sig(rng, 2, 60), _walk(maze, rng, 60),
                                np.ones(60, dtype=bool), np.ones(60, dtype=bool),
                                maze.n_cells, min_frames=1)
        assert np.all(np.isnan(d))

    def test_all_dark(self, maze, rng):
        d, _, _, _ = self._call(_sig(rng, 2, 60), _walk(maze, rng, 60),
                                np.zeros(60, dtype=bool), np.ones(60, dtype=bool),
                                maze.n_cells, min_frames=1)
        assert np.all(np.isnan(d))

    def test_1d(self, maze, rng):
        d, _, _, _ = self._call(rng.standard_normal(100), _walk(maze, rng, 100),
                                rng.random(100) > 0.5, np.ones(100, dtype=bool),
                                maze.n_cells, min_frames=2)
        assert d.shape[0] == 1

# =========================================================================
# light_modulation_by_node_type
# =========================================================================
class TestLightModulationByNodeType:
    def test_keys(self, maze, rng):
        r = neural.light_modulation_by_node_type(
            rng.standard_normal((5, maze.n_cells)),
            np.ones(maze.n_cells, dtype=bool), maze,
            np.array(["penk"]*3 + ["nonpenk"]*2))
        assert set(r.keys()) == {"junction", "corridor", "dead_end"}

    def test_mean_delta_shape(self, maze, rng):
        nr = 8
        r = neural.light_modulation_by_node_type(
            rng.standard_normal((nr, maze.n_cells)),
            np.ones(maze.n_cells, dtype=bool), maze)
        for nt in r:
            assert r[nt]["mean_delta"].shape == (nr,)

    def test_wilcoxon_range(self, maze, rng):
        r = neural.light_modulation_by_node_type(
            rng.standard_normal((10, maze.n_cells)),
            np.ones(maze.n_cells, dtype=bool), maze)
        for nt in r:
            p = r[nt]["wilcoxon_p"]
            if np.isfinite(p): assert 0 <= p <= 1

    def test_mannwhitney(self, maze, rng):
        r = neural.light_modulation_by_node_type(
            rng.standard_normal((10, maze.n_cells)),
            np.ones(maze.n_cells, dtype=bool), maze,
            np.array(["penk"]*5 + ["nonpenk"]*5))
        for nt in r: assert "mannwhitney_p" in r[nt]

    def test_single_ct(self, maze, rng):
        r = neural.light_modulation_by_node_type(
            rng.standard_normal((6, maze.n_cells)),
            np.ones(maze.n_cells, dtype=bool), maze, np.array(["penk"]*6))
        for nt in r: assert np.isnan(r[nt]["mannwhitney_p"])

    def test_no_valid(self, maze, rng):
        r = neural.light_modulation_by_node_type(
            rng.standard_normal((4, maze.n_cells)),
            np.zeros(maze.n_cells, dtype=bool), maze)
        for nt in r: assert np.all(np.isnan(r[nt]["mean_delta"]))

    def test_no_labels(self, maze, rng):
        r = neural.light_modulation_by_node_type(
            rng.standard_normal((6, maze.n_cells)),
            np.ones(maze.n_cells, dtype=bool), maze)
        for nt in r: assert "mannwhitney_U" not in r[nt]

    def test_few_rois(self, maze, rng):
        r = neural.light_modulation_by_node_type(
            rng.standard_normal((4, maze.n_cells)),
            np.ones(maze.n_cells, dtype=bool), maze)
        for nt in r: assert np.isnan(r[nt]["wilcoxon_p"])

    def test_wilcoxon_scipy(self, maze):
        """Regression: verify Wilcoxon matches scipy."""
        rng2 = np.random.default_rng(99)
        delta = rng2.standard_normal((10, maze.n_cells)) * 2
        r = neural.light_modulation_by_node_type(
            delta, np.ones(maze.n_cells, dtype=bool), maze)
        for nt in r:
            md = r[nt]["mean_delta"]
            f = np.isfinite(md)
            if f.sum() >= 6:
                _, ep = sp_stats.wilcoxon(md[f], alternative="two-sided")
                np.testing.assert_allclose(r[nt]["wilcoxon_p"], ep, atol=1e-12)

# =========================================================================
# classify_frames_by_node_type
# =========================================================================
class TestClassifyFramesByNodeType:
    def test_keys(self, maze, rng):
        assert set(neural.classify_frames_by_node_type(
            _walk(maze, rng, 100), maze)) == {"junction","corridor","dead_end","invalid"}

    def test_exclusive(self, maze, rng):
        r = neural.classify_frames_by_node_type(_walk(maze, rng, 100), maze)
        np.testing.assert_array_equal(sum(r[k].astype(int) for k in r), 1)

    def test_exhaustive(self, maze, rng):
        r = neural.classify_frames_by_node_type(_walk(maze, rng, 100), maze)
        assert np.all(r["junction"]|r["corridor"]|r["dead_end"]|r["invalid"])

    def test_neg_invalid(self, maze):
        r = neural.classify_frames_by_node_type(np.array([-1,-1], dtype=np.int32), maze)
        assert np.all(r["invalid"])

    def test_junction(self, maze):
        r = neural.classify_frames_by_node_type(
            np.array([maze.cell_to_idx[maze.junctions[0]]], dtype=np.int32), maze)
        assert r["junction"][0]

    def test_dead_end(self, maze):
        r = neural.classify_frames_by_node_type(
            np.array([maze.cell_to_idx[maze.dead_ends[0]]], dtype=np.int32), maze)
        assert r["dead_end"][0]

    def test_corridor(self, maze):
        r = neural.classify_frames_by_node_type(
            np.array([maze.cell_to_idx[maze.corridors[0]]], dtype=np.int32), maze)
        assert r["corridor"][0]

    def test_oor(self, maze):
        r = neural.classify_frames_by_node_type(
            np.array([maze.n_cells], dtype=np.int32), maze)
        assert r["invalid"][0]

    @given(n=st.integers(1, 200), s=st.integers(0, 9999))
    @settings(max_examples=50, deadline=None)
    def test_prop(self, n, s):
        rng = np.random.default_rng(s); m = build_rose_maze()
        r = neural.classify_frames_by_node_type(
            rng.integers(-1, m.n_cells, size=n).astype(np.int32), m)
        np.testing.assert_array_equal(sum(r[k].astype(int) for k in r), 1)

# =========================================================================
# hd_tuning_by_location_type
# =========================================================================
class TestHdTuningByLocationType:
    def test_dict(self, maze, rng):
        n = 500
        r = neural.hd_tuning_by_location_type(
            rng.standard_normal(n), _hd(rng, n),
            neural.classify_frames_by_node_type(_walk(maze, rng, n), maze),
            np.ones(n, dtype=bool), min_frames=5)
        assert isinstance(r, dict)

    def test_tuple_values(self, maze, rng):
        n = 1000
        r = neural.hd_tuning_by_location_type(
            rng.standard_normal(n), _hd(rng, n),
            neural.classify_frames_by_node_type(_walk(maze, rng, n), maze),
            np.ones(n, dtype=bool), n_bins=36, min_frames=5)
        for _, (tc, bins, mvl) in r.items():
            assert tc.shape == (36,) and bins.shape == (36,)
            if np.isfinite(mvl): assert 0 <= mvl <= 1 + 1e-6

    def test_high_min(self, maze, rng):
        n = 100
        r = neural.hd_tuning_by_location_type(
            rng.standard_normal(n), _hd(rng, n),
            neural.classify_frames_by_node_type(_walk(maze, rng, n), maze),
            np.ones(n, dtype=bool), min_frames=n+1)
        assert len(r) == 0

# =========================================================================
# junction_vs_corridor_mvl
# =========================================================================
class TestJunctionVsCorridorMvl:
    def test_keys(self, maze, rng):
        r = neural.junction_vs_corridor_mvl(
            _sig(rng, 4, 300), _hd(rng, 300), _walk(maze, rng, 300),
            maze, np.ones(300, dtype=bool), min_frames=5)
        for k in ("junction_mvl","corridor_mvl","wilcoxon_stat","wilcoxon_p"):
            assert k in r

    def test_shapes(self, maze, rng):
        nr = 5
        r = neural.junction_vs_corridor_mvl(
            _sig(rng, nr, 400), _hd(rng, 400), _walk(maze, rng, 400),
            maze, np.ones(400, dtype=bool), min_frames=5)
        assert r["junction_mvl"].shape == (nr,)

    def test_range(self, maze, rng):
        r = neural.junction_vs_corridor_mvl(
            _sig(rng, 5, 500), _hd(rng, 500), _walk(maze, rng, 500),
            maze, np.ones(500, dtype=bool), min_frames=5)
        for k in ("junction_mvl","corridor_mvl"):
            f = r[k][np.isfinite(r[k])]
            if len(f): assert np.all(f >= 0) and np.all(f <= 1+1e-6)

    def test_few_rois(self, maze, rng):
        r = neural.junction_vs_corridor_mvl(
            _sig(rng, 3, 200), _hd(rng, 200), _walk(maze, rng, 200),
            maze, np.ones(200, dtype=bool), min_frames=5)
        assert np.isnan(r["wilcoxon_p"])

# =========================================================================
# count_corridor_traversals
# =========================================================================
class TestCountCorridorTraversals:
    def test_shape(self, maze, rng):
        assert neural.count_corridor_traversals(_walk(maze, rng, 100), maze).shape == (100,)
    def test_invalid(self, maze):
        np.testing.assert_array_equal(
            neural.count_corridor_traversals(np.array([-1,-1], dtype=np.int32), maze), 0)
    def test_non_corr(self, maze):
        ci = np.array([maze.cell_to_idx[maze.junctions[0]],
                        maze.cell_to_idx[maze.dead_ends[0]]], dtype=np.int32)
        np.testing.assert_array_equal(neural.count_corridor_traversals(ci, maze), 0)
    def test_single(self, maze):
        ci = np.full(10, maze.cell_to_idx[maze.corridors[0]], dtype=np.int32)
        assert np.all(neural.count_corridor_traversals(ci, maze) == 1)
    def test_reentry(self, maze):
        c, j = maze.cell_to_idx[maze.corridors[0]], maze.cell_to_idx[maze.junctions[0]]
        r = neural.count_corridor_traversals(np.array([j,c,c,j,c], dtype=np.int32), maze)
        assert r[1]==1 and r[2]==1 and r[4]==2 and r[0]==0 and r[3]==0
    def test_empty(self, maze):
        assert neural.count_corridor_traversals(np.array([], dtype=np.int32), maze).shape == (0,)
    @given(n=st.integers(10,200), s=st.integers(0,9999))
    @settings(max_examples=30, deadline=None)
    def test_prop_shape(self, n, s):
        rng = np.random.default_rng(s); m = build_rose_maze()
        assert neural.count_corridor_traversals(
            rng.integers(-1, m.n_cells, size=n).astype(np.int32), m).shape == (n,)
    @given(n=st.integers(10,150), s=st.integers(0,9999))
    @settings(max_examples=30, deadline=None)
    def test_prop_nondec(self, n, s):
        rng = np.random.default_rng(s); m = build_rose_maze()
        ci = rng.integers(-1, m.n_cells, size=n).astype(np.int32)
        r = neural.count_corridor_traversals(ci, m)
        for cidx in {m.cell_to_idx[c] for c in m.corridors}:
            v = r[ci == cidx]
            if len(v) > 1: assert np.all(np.diff(v) >= 0)
    @given(n=st.integers(10,200), s=st.integers(0,9999))
    @settings(max_examples=30, deadline=None)
    def test_prop_zero(self, n, s):
        rng = np.random.default_rng(s); m = build_rose_maze()
        ci = rng.integers(-1, m.n_cells, size=n).astype(np.int32)
        r = neural.count_corridor_traversals(ci, m)
        cs = {m.cell_to_idx[c] for c in m.corridors}
        for i in range(n):
            if ci[i] < 0 or ci[i] not in cs: assert r[i] == 0

# =========================================================================
# activity_by_traversal_number
# =========================================================================
class TestActivityByTraversalNumber:
    def test_keys(self, maze, rng):
        n = 300; ci = _walk(maze, rng, n)
        r = neural.activity_by_traversal_number(
            rng.standard_normal(n), ci, neural.count_corridor_traversals(ci, maze),
            np.ones(n, dtype=bool), 5)
        for k in ("traversal_nums","mean_activity","n_frames","spearman_r","spearman_p"):
            assert k in r
    def test_bounded(self, maze, rng):
        n = 300; ci = _walk(maze, rng, n)
        r = neural.activity_by_traversal_number(
            rng.standard_normal(n), ci, neural.count_corridor_traversals(ci, maze),
            np.ones(n, dtype=bool), 5)
        if len(r["traversal_nums"]):
            assert np.all(r["traversal_nums"]>=1) and np.all(r["traversal_nums"]<=5)
    def test_spearman_range(self, maze, rng):
        n = 500; ci = _walk(maze, rng, n)
        r = neural.activity_by_traversal_number(
            rng.standard_normal(n), ci, neural.count_corridor_traversals(ci, maze),
            np.ones(n, dtype=bool))
        if np.isfinite(r["spearman_r"]): assert -1 <= r["spearman_r"] <= 1
    def test_few_nan(self, maze):
        ci = np.full(10, maze.cell_to_idx[maze.corridors[0]], dtype=np.int32)
        r = neural.activity_by_traversal_number(
            np.ones(10), ci, neural.count_corridor_traversals(ci, maze),
            np.ones(10, dtype=bool), 5)
        assert np.isnan(r["spearman_r"])
    def test_spearman_match(self, maze, rng):
        """Regression: Spearman r matches scipy."""
        n = 600; ci = _walk(maze, rng, n)
        trav = neural.count_corridor_traversals(ci, maze)
        sig = -0.1*trav.astype(float) + rng.standard_normal(n)*0.01
        r = neural.activity_by_traversal_number(sig, ci, trav, np.ones(n, dtype=bool), 10)
        if len(r["traversal_nums"]) >= 3:
            exp, _ = sp_stats.spearmanr(r["traversal_nums"], r["mean_activity"])
            np.testing.assert_allclose(r["spearman_r"], exp, atol=1e-10)

# =========================================================================
# familiarity_effect_by_cell_type
# =========================================================================
class TestFamiliarityEffectByCellType:
    def test_keys(self, maze, rng):
        n, nr = 300, 6; ci = _walk(maze, rng, n)
        r = neural.familiarity_effect_by_cell_type(
            _sig(rng, nr, n), ci, neural.count_corridor_traversals(ci, maze),
            np.ones(n, dtype=bool), np.array(["penk"]*3+["nonpenk"]*3))
        for k in ("spearman_r","spearman_p","mannwhitney_U","mannwhitney_p"):
            assert k in r
    def test_per_roi(self, maze, rng):
        n, nr = 300, 4; ci = _walk(maze, rng, n)
        r = neural.familiarity_effect_by_cell_type(
            _sig(rng, nr, n), ci, neural.count_corridor_traversals(ci, maze),
            np.ones(n, dtype=bool), np.array(["penk"]*2+["nonpenk"]*2))
        assert r["spearman_r"].shape == (nr,)
    def test_single_ct(self, maze, rng):
        n, nr = 200, 4; ci = _walk(maze, rng, n)
        r = neural.familiarity_effect_by_cell_type(
            _sig(rng, nr, n), ci, neural.count_corridor_traversals(ci, maze),
            np.ones(n, dtype=bool), np.array(["penk"]*nr))
        assert np.isnan(r["mannwhitney_U"])
    def test_nan_sig(self, maze, rng):
        n = 100; ci = _walk(maze, rng, n)
        r = neural.familiarity_effect_by_cell_type(
            np.full((3, n), np.nan), ci, neural.count_corridor_traversals(ci, maze),
            np.ones(n, dtype=bool), np.array(["penk","penk","nonpenk"]))
        assert np.all(np.isnan(r["spearman_r"]))

# =========================================================================
# extract_junction_events
# =========================================================================
class TestExtractJunctionEvents:
    def test_list(self, maze, rng):
        assert isinstance(neural.extract_junction_events(_walk(maze, rng, 300), maze), list)

    def test_fields(self, maze, rng):
        for ev in neural.extract_junction_events(_walk(maze, rng, 300), maze, 2):
            # Works whether ev is dict or dataclass
            turn = ev["turn"] if isinstance(ev, dict) else ev.turn
            assert turn in ("left","right","back","forward")

    def test_junction_cells(self, maze, rng):
        js = {maze.cell_to_idx[j] for j in maze.junctions}
        for ev in neural.extract_junction_events(_walk(maze, rng, 300), maze, 2):
            jc = ev["junction"] if isinstance(ev, dict) else ev.junction_cell_idx
            assert jc in js

    def test_known(self, maze):
        path = [(0,0)]*3 + [(1,0)] + [(2,0)]*2
        ci = np.array([maze.cell_to_idx[c] for c in path], dtype=np.int32)
        evs = neural.extract_junction_events(ci, maze, 2)
        assert len(evs) >= 1
        ev = evs[0]
        jc = ev["junction"] if isinstance(ev, dict) else ev.junction_cell_idx
        assert jc == maze.cell_to_idx[(1,0)]

    def test_dead_end_only(self, maze):
        assert len(neural.extract_junction_events(
            np.full(30, maze.cell_to_idx[maze.dead_ends[0]], dtype=np.int32), maze)) == 0

    def test_min_pre(self, maze):
        ci = np.array([maze.cell_to_idx[c] for c in [(0,0),(1,0),(2,0)]], dtype=np.int32)
        assert len(neural.extract_junction_events(ci, maze, 2)) == 0

    def test_empty(self, maze):
        assert len(neural.extract_junction_events(np.array([], dtype=np.int32), maze)) == 0

    def test_invalid(self, maze):
        assert len(neural.extract_junction_events(np.full(30, -1, dtype=np.int32), maze)) == 0

    def test_multiple(self, maze):
        p = [(0,0)]*3 + [(1,0)] + [(2,0)]*3 + [(1,0)] + [(0,0)]*2
        ci = np.array([maze.cell_to_idx[c] for c in p], dtype=np.int32)
        assert len(neural.extract_junction_events(ci, maze, 2)) == 2

# =========================================================================
# pre_junction_population_vectors
# =========================================================================
class TestPreJunctionPopulationVectors:
    def test_X_y(self, maze, rng):
        n, nr = 800, 5
        evs = neural.extract_junction_events(_walk(maze, rng, n), maze, 2)
        X, y = neural.pre_junction_population_vectors(_sig(rng, nr, n), evs, 2)
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)

    def test_cols(self, maze, rng):
        n, nr = 800, 5
        evs = neural.extract_junction_events(_walk(maze, rng, n), maze, 2)
        X, y = neural.pre_junction_population_vectors(_sig(rng, nr, n), evs, 2)
        if X.shape[0] > 0:
            assert X.shape[1] == nr and X.shape[0] == y.shape[0]

    def test_empty(self, rng):
        X, y = neural.pre_junction_population_vectors(_sig(rng, 3, 100), [], 2)
        assert X.shape == (0, 3) and y.shape == (0,)

# =========================================================================
# decode_junction_choice
# =========================================================================
class TestDecodeJunctionChoice:
    def test_keys(self, rng):
        X = rng.standard_normal((40, 10))
        y = np.array([0]*20 + [1]*20)
        r = neural.decode_junction_choice(X, y, 5)
        assert "accuracy" in r and "n_events" in r

    def test_accuracy_range(self, rng):
        r = neural.decode_junction_choice(
            rng.standard_normal((40, 8)), np.array([0]*20+[1]*20), 5)
        assert 0 <= r["accuracy"] <= 1

    def test_separable(self, rng):
        X = np.zeros((40, 2)); X[:20, 0] = 10; X[20:, 0] = -10
        r = neural.decode_junction_choice(X, np.array([0]*20+[1]*20), 5)
        assert r["accuracy"] > 0.8

    def test_insufficient(self, rng):
        r = neural.decode_junction_choice(
            rng.standard_normal((10, 4)), np.array([0]*5+[1]*5), 5, min_events=15)
        assert np.isnan(r["accuracy"])

    def test_single_class(self, rng):
        r = neural.decode_junction_choice(
            rng.standard_normal((20, 5)), np.zeros(20, dtype=int), 5)
        assert np.isnan(r["accuracy"])

    def test_fold_count(self, rng):
        r = neural.decode_junction_choice(
            rng.standard_normal((40, 6)), np.array([0]*20+[1]*20), 5)
        assert len(r["fold_accuracies"]) == 5

    @given(n=st.integers(20, 60), f=st.integers(2, 15), s=st.integers(0, 9999))
    @settings(max_examples=20, deadline=None)
    def test_prop(self, n, f, s):
        rng = np.random.default_rng(s); X = rng.standard_normal((n, f))
        h = n // 2; y = np.array([0]*h + [1]*(n-h)); rng.shuffle(y)
        folds = min(5, min(h, n-h))
        if folds < 2: return
        r = neural.decode_junction_choice(X, y, folds)
        if np.isfinite(r["accuracy"]): assert 0 <= r["accuracy"] <= 1

# =========================================================================
# Integration
# =========================================================================
class TestIntegration:
    def test_analysis4(self, maze, rng):
        nr, n = 10, 600; sig = _sig(rng, nr, n)
        ci = _walk(maze, rng, n); lo = np.zeros(n, dtype=bool); lo[:n//2] = True
        act_r = neural.node_activity_map(sig, ci, np.ones(n, dtype=bool), maze.n_cells)
        if isinstance(act_r, tuple): act = act_r[0]
        else: act = act_r
        assert act.shape == (nr, maze.n_cells)
        cr = neural.light_dark_node_contrast(sig, ci, lo, np.ones(n, dtype=bool), maze.n_cells, 5)
        if isinstance(cr, dict):
            delta, vc = cr["delta"], cr["valid_cells"]
        else:
            delta = cr[0]; vc = np.any(np.isfinite(delta), axis=0)
        mod = neural.light_modulation_by_node_type(
            delta, vc, maze, np.array(["penk"]*5+["nonpenk"]*5))
        assert "junction" in mod

    def test_analysis1(self, maze, rng):
        n = 800
        loc = neural.classify_frames_by_node_type(_walk(maze, rng, n), maze)
        r = neural.hd_tuning_by_location_type(
            rng.standard_normal(n), _hd(rng, n), loc,
            np.ones(n, dtype=bool), min_frames=10)
        for _, (tc, _, _) in r.items(): assert tc.shape == (36,)

    def test_analysis2(self, maze, rng):
        nr, n = 6, 500; ci = _walk(maze, rng, n)
        r = neural.familiarity_effect_by_cell_type(
            _sig(rng, nr, n), ci, neural.count_corridor_traversals(ci, maze),
            np.ones(n, dtype=bool), np.array(["penk"]*3+["nonpenk"]*3))
        assert r["spearman_r"].shape == (nr,)

    def test_analysis3(self, maze, rng):
        nr, n = 6, 1000; sig = _sig(rng, nr, n); ci = _walk(maze, rng, n)
        evs = neural.extract_junction_events(ci, maze, 2)
        X, y = neural.pre_junction_population_vectors(sig, evs, 2)
        if X.shape[0] >= 15 and len(np.unique(y)) >= 2:
            r = neural.decode_junction_choice(X, y, 5)
            if np.isfinite(r["accuracy"]): assert 0 <= r["accuracy"] <= 1

# =========================================================================
# Edge cases
# =========================================================================
class TestEdgeCases:
    def _act(self, *a, **kw):
        r = neural.node_activity_map(*a, **kw)
        return r if not isinstance(r, tuple) else r[0]

    def test_single_frame(self, maze, rng):
        assert self._act(_sig(rng, 2, 1), np.array([0], dtype=np.int32),
                         np.array([True]), maze.n_cells).shape == (2, maze.n_cells)

    def test_nan_sig(self, maze, rng):
        n = 50; ci = _walk(maze, rng, n)
        act = self._act(np.full((3, n), np.nan), ci, np.ones(n, dtype=bool), maze.n_cells)
        r = neural.node_activity_map(np.full((3, n), np.nan), ci, np.ones(n, dtype=bool), maze.n_cells)
        occ = r[1] if isinstance(r, tuple) else None
        if occ is not None:
            for c in range(maze.n_cells):
                if occ[c] > 0: assert np.all(np.isnan(act[:, c]))

    def test_f32(self, maze, rng):
        assert self._act(rng.standard_normal((2, 100)).astype(np.float32),
                         _walk(maze, rng, 100), np.ones(100, dtype=bool),
                         maze.n_cells).shape == (2, maze.n_cells)

    def test_large(self, maze, rng):
        n = 50; ci = _walk(maze, rng, n)
        act = self._act(np.full((2, n), 1e15), ci, np.ones(n, dtype=bool), maze.n_cells)
        r = neural.node_activity_map(np.full((2, n), 1e15), ci, np.ones(n, dtype=bool), maze.n_cells)
        occ = r[1] if isinstance(r, tuple) else None
        if occ is not None:
            for c in range(maze.n_cells):
                if occ[c] > 0: assert np.all(np.isfinite(act[:, c]))

    def test_negative(self, maze, rng):
        n = 80; ci = _walk(maze, rng, n); sig = -np.abs(rng.standard_normal((3, n)))
        act = self._act(sig, ci, np.ones(n, dtype=bool), maze.n_cells)
        r = neural.node_activity_map(sig, ci, np.ones(n, dtype=bool), maze.n_cells)
        occ = r[1] if isinstance(r, tuple) else None
        if occ is not None:
            for c in range(maze.n_cells):
                if occ[c] > 0: assert np.all(act[:, c] < 0)

    def test_inf(self, maze, rng):
        n = 50; sig = rng.standard_normal((2, n)); sig[0, 10] = np.inf
        assert self._act(sig, _walk(maze, rng, n),
                         np.ones(n, dtype=bool), maze.n_cells).shape == (2, maze.n_cells)
