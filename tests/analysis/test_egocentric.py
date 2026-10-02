"""Tests for hm2p.analysis.egocentric (synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis import egocentric as ego
from hm2p.maze.topology import (
    RoseMaze,
    build_adjacency,
    build_rose_maze,
    classify_nodes,
    compute_distances,
)

FPS = 9.6


def _maze(cells: set[tuple[int, int]], adj: dict | None = None) -> RoseMaze:
    adj = build_adjacency(cells) if adj is None else adj
    dist, cell_list = compute_distances(cells, adj)
    return RoseMaze(
        cells=cells, adj=adj, node_types=classify_nodes(adj), dist=dist, cell_list=cell_list
    )


def _square_walls(size: float = 4.0) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, size, 0.0],
            [size, 0.0, size, size],
            [size, size, 0.0, size],
            [0.0, size, 0.0, 0.0],
        ]
    )


def _random_walk(n: int, seed: int, size: float = 4.0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, size - 0.1, n)
    y = rng.uniform(0.1, size - 0.1, n)
    heading = rng.uniform(0.0, 360.0, n)
    return x, y, heading


def _ebc_cell(n: int = 3000, seed: int = 0) -> tuple[np.ndarray, ...]:
    """Cell that fires when a wall lies at egocentric +90 deg within 0.5 units."""
    rng = np.random.default_rng(seed)
    x, y, heading = _random_walk(n, seed)
    walls = _square_walls()
    d90 = ego.ray_wall_distances(x, y, heading, walls, [90.0])[:, 0]
    signal = (d90 < 0.5).astype(float) + rng.normal(0, 0.1, n)
    return signal, x, y, heading, walls


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------


class TestWrapDeg:
    def test_values(self) -> None:
        out = ego.wrap_deg([0.0, 180.0, -180.0, 190.0, -190.0, 360.0, np.nan])
        np.testing.assert_allclose(out[:6], [0, 180, 180, -170, 170, 0])
        assert np.isnan(out[6])

    @given(st.floats(-1e4, 1e4, allow_nan=False))
    @settings(max_examples=100)
    def test_range_and_congruence(self, a: float) -> None:
        w = float(ego.wrap_deg(a))
        assert -180.0 < w <= 180.0
        assert np.isclose(np.cos(np.deg2rad(w)), np.cos(np.deg2rad(a)), atol=1e-6)
        assert np.isclose(np.sin(np.deg2rad(w)), np.sin(np.deg2rad(a)), atol=1e-6)


class TestHdToFrameHeading:
    def test_matches_pipeline_vector_angle_deg(self) -> None:
        from hm2p.kinematics.compute import _vector_angle_deg

        rng = np.random.default_rng(1)
        dx, dy = rng.normal(size=50), rng.normal(size=50)
        hd = _vector_angle_deg(np.zeros(50), np.zeros(50), dx, dy)
        phi = ego.hd_to_frame_heading(hd)
        expected = np.mod(np.degrees(np.arctan2(dy, dx)), 360.0)
        np.testing.assert_allclose(ego.wrap_deg(phi - expected), 0.0, atol=1e-9)

    def test_nan(self) -> None:
        assert np.isnan(ego.hd_to_frame_heading([np.nan])[0])


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class TestMazeWallSegments:
    def test_single_cell(self) -> None:
        walls = ego.maze_wall_segments(_maze({(0, 0)}))
        assert walls.shape == (4, 4)
        lengths = np.hypot(walls[:, 2] - walls[:, 0], walls[:, 3] - walls[:, 1])
        np.testing.assert_allclose(lengths, 1.0)

    def test_two_by_two_has_perimeter_only(self) -> None:
        walls = ego.maze_wall_segments(_maze({(0, 0), (1, 0), (0, 1), (1, 1)}))
        assert walls.shape == (8, 4)
        # no internal segment at x=1 between y in [0, 2] or y=1 between x in [0, 2]
        internal = (
            (walls[:, 0] == 1) & (walls[:, 2] == 1) & (walls[:, 1] < 2) & (walls[:, 1] >= 0)
        ) | ((walls[:, 1] == 1) & (walls[:, 3] == 1))
        assert not internal.any()

    def test_blocked_edge_listed_once(self) -> None:
        cells = {(0, 0), (1, 0)}
        adj: dict[tuple[int, int], list[tuple[int, int]]] = {(0, 0): [], (1, 0): []}
        walls = ego.maze_wall_segments(_maze(cells, adj))
        assert walls.shape == (7, 4)
        assert sum(1 for w in walls if tuple(w) == (1.0, 0.0, 1.0, 1.0)) == 1

    def test_rose_maze_default(self) -> None:
        walls = ego.maze_wall_segments()
        maze = build_rose_maze()
        assert walls.shape[1] == 4 and walls.shape[0] > len(maze.cells)
        assert (
            np.all(walls >= 0) and np.all(walls[:, [0, 2]] <= 7) and np.all(walls[:, [1, 3]] <= 5)
        )
        # the blocked top-row edge between (2,4) and (3,4) is a wall
        assert any(tuple(w) == (3.0, 4.0, 3.0, 5.0) for w in walls)

    def test_empty_maze(self) -> None:
        empty = RoseMaze(cells=set(), adj={}, node_types={}, dist=np.zeros((0, 0)), cell_list=[])
        assert ego.maze_wall_segments(empty).shape == (0, 4)


class TestRayWallDistances:
    def test_known_geometry(self) -> None:
        d = ego.ray_wall_distances([1.0], [2.0], [0.0], _square_walls(), [0, 90, 180, 270])
        np.testing.assert_allclose(d[0], [3.0, 2.0, 1.0, 2.0])

    def test_heading_rotates_rays(self) -> None:
        d = ego.ray_wall_distances([1.0], [2.0], [90.0], _square_walls(), [0, -90])
        np.testing.assert_allclose(d[0], [2.0, 3.0])

    def test_diagonal(self) -> None:
        d = ego.ray_wall_distances([2.0], [2.0], [45.0], _square_walls(), [0.0])
        np.testing.assert_allclose(d[0, 0], 2.0 * np.sqrt(2.0))

    def test_no_hit_and_nan_input(self) -> None:
        walls = np.array([[5.0, -1.0, 5.0, 1.0]])
        d = ego.ray_wall_distances([0.0, np.nan], [0.0, 0.0], [0.0, 0.0], walls, [0.0, 180.0])
        assert d[0, 0] == pytest.approx(5.0)
        assert np.isnan(d[0, 1])
        assert np.isnan(d[1]).all()

    def test_chunking_consistent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        x, y, h = _random_walk(50, 3)
        full = ego.ray_wall_distances(x, y, h, _square_walls(), [0, 45, 90])
        monkeypatch.setattr(ego, "_RAY_CHUNK", 7)
        np.testing.assert_allclose(
            ego.ray_wall_distances(x, y, h, _square_walls(), [0, 45, 90]), full
        )

    def test_empty_and_errors(self) -> None:
        assert ego.ray_wall_distances([], [], [], _square_walls(), [0.0]).shape == (0, 1)
        assert np.isnan(ego.ray_wall_distances([1.0], [1.0], [0.0], np.empty((0, 4)), [0.0])).all()
        with pytest.raises(ValueError, match="same length"):
            ego.ray_wall_distances([1.0, 2.0], [1.0], [0.0], _square_walls(), [0.0])
        with pytest.raises(ValueError, match="walls must have shape"):
            ego.ray_wall_distances([1.0], [1.0], [0.0], np.zeros((2, 3)), [0.0])


class TestWallDistance:
    def test_known(self) -> None:
        d = ego.wall_distance([1.0, 2.0, np.nan, 5.0], [2.0, 2.0, 1.0, 5.0], _square_walls())
        np.testing.assert_allclose(d[:2], [1.0, 2.0])
        assert np.isnan(d[2])
        assert d[3] == pytest.approx(np.sqrt(2.0))  # outside, nearest the (4, 4) corner

    def test_degenerate_segment_and_empty(self) -> None:
        d = ego.wall_distance([3.0], [4.0], np.array([[0.0, 0.0, 0.0, 0.0]]))
        assert d[0] == pytest.approx(5.0)
        assert np.isnan(ego.wall_distance([1.0], [1.0], np.empty((0, 4)))).all()
        with pytest.raises(ValueError, match="same length"):
            ego.wall_distance([1.0], [1.0, 2.0], _square_walls())


class TestHeadBodyAngle:
    def test_sign(self) -> None:
        out = ego.head_body_angle(
            [30.0, -30.0, 190.0, 0.0], [1, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]
        )
        np.testing.assert_allclose(out, [30.0, -30.0, -170.0, -90.0])

    def test_nan_and_coincident(self) -> None:
        out = ego.head_body_angle(
            [np.nan, 10.0, 10.0], [1, np.nan, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
        )
        assert np.isnan(out).all()

    @given(st.floats(-720, 720), st.floats(-5, 5), st.floats(-5, 5))
    @settings(max_examples=60)
    def test_range(self, h: float, hx: float, hy: float) -> None:
        out = ego.head_body_angle([h], [hx], [hy], [0.0], [0.0])[0]
        assert np.isnan(out) or -180.0 < out <= 180.0


# ---------------------------------------------------------------------------
# Shuffle helpers
# ---------------------------------------------------------------------------


class TestShuffleHelpers:
    def test_offsets_range_and_short_fallback(self) -> None:
        rng = np.random.default_rng(0)
        off = ego._shift_offsets(1000, 200, 96, rng)
        assert off.min() >= 96 and off.max() <= 904
        short = ego._shift_offsets(10, 50, 96, rng)
        assert short.min() >= 1 and short.max() <= 9

    def test_shifted_stack_matches_roll(self) -> None:
        s = np.arange(6, dtype=float)
        st_ = ego._shifted_stack(s, np.array([2, 5]))
        np.testing.assert_array_equal(st_[:, 0], s)
        np.testing.assert_array_equal(st_[:, 1], np.roll(s, 2))
        np.testing.assert_array_equal(st_[:, 2], np.roll(s, 5))

    def test_indicator_drops_negative(self) -> None:
        ind = ego._indicator(np.array([[0, 1], [-1, 1], [2, -1]]), 3, 3)
        np.testing.assert_array_equal(ind.toarray(), [[1, 0, 0], [1, 1, 0], [0, 0, 1]])

    def test_null_summary(self) -> None:
        null = np.linspace(0, 1, 99)
        r = ego._null_summary(2.0, null)
        assert r["p"] == pytest.approx(1 / 100) and r["z"] > 0
        r2 = ego._null_summary(-2.0, null - 0.5, two_sided=True)
        assert r2["p"] == pytest.approx(1 / 100)
        assert np.isnan(ego._null_summary(np.nan, null)["p"])
        assert np.isnan(ego._null_summary(1.0, np.array([np.nan]))["p"])
        assert np.isnan(ego._null_summary(1.0, np.ones(5))["z"])

    def test_min_shift_frames(self) -> None:
        assert ego._min_shift_frames(10.0, 9.6) == 96


# ---------------------------------------------------------------------------
# EBC ratemap, score and significance
# ---------------------------------------------------------------------------


class TestEbcRatemap:
    def test_shapes_and_defaults(self) -> None:
        signal, x, y, h, walls = _ebc_cell(n=500)
        res = ego.egocentric_boundary_ratemap(signal, x, y, h, walls, np.ones(500, bool))
        assert res["ratemap"].shape == (24, 12)
        assert res["occupancy"].shape == (24, 12)
        np.testing.assert_allclose(res["angle_centres"][:2], [7.5, 22.5])
        np.testing.assert_allclose(res["dist_centres"][0], 0.125)

    def test_occupancy_counts_and_mask(self) -> None:
        x, y, h = np.array([1.0, 1.0]), np.array([2.0, 2.0]), np.array([0.0, 0.0])
        res = ego.egocentric_boundary_ratemap(
            [1.0, 3.0],
            x,
            y,
            h,
            _square_walls(),
            [True, False],
            n_angle_bins=4,
            dist_edges=[0, 1.5, 2.5, 3.5],
            min_occupancy_frames=1,
        )
        # ray centres at 45, 135, 225, 315 deg; only frame 0 counted
        assert res["occupancy"].sum() == 4
        assert np.nanmax(res["ratemap"]) == 1.0

    def test_min_occupancy_nan(self) -> None:
        x, y, h = np.array([1.0]), np.array([2.0]), np.array([0.0])
        res = ego.egocentric_boundary_ratemap(
            [1.0], x, y, h, _square_walls(), [True], n_angle_bins=4, min_occupancy_frames=2
        )
        assert np.isnan(res["ratemap"]).all()

    def test_ray_dist_shape_checked(self) -> None:
        with pytest.raises(ValueError, match="ray_dist shape"):
            ego.egocentric_boundary_ratemap(
                [1.0], [1.0], [1.0], [0.0], _square_walls(), [True], ray_dist=np.zeros((1, 3))
            )

    def test_precomputed_ray_dist_identical(self) -> None:
        signal, x, y, h, walls = _ebc_cell(n=300)
        rd = ego.ray_wall_distances(x, y, h, walls, ego.ebc_angle_centres(24))
        a = ego.egocentric_boundary_ratemap(signal, x, y, h, walls, np.ones(300, bool))
        b = ego.egocentric_boundary_ratemap(
            signal, x, y, h, walls, np.ones(300, bool), ray_dist=rd
        )
        np.testing.assert_array_equal(np.isnan(a["ratemap"]), np.isnan(b["ratemap"]))
        np.testing.assert_allclose(np.nan_to_num(a["ratemap"]), np.nan_to_num(b["ratemap"]))


class TestEbcScore:
    def test_peak(self) -> None:
        ang = ego.ebc_angle_centres(8)
        rm = np.zeros((8, 3))
        rm[2, 1] = 5.0  # 112.5 deg, distance bin 1
        res = ego.ebc_score(rm, ang, np.array([0.5, 1.5, 2.5]))
        assert res["mrl"] == pytest.approx(1.0)
        assert res["preferred_angle"] == pytest.approx(112.5)
        assert res["preferred_distance"] == 1.5
        assert ego.ebc_score(rm, ang)["preferred_distance"] == 1.0

    def test_uniform_and_degenerate(self) -> None:
        ang = ego.ebc_angle_centres(8)
        assert ego.ebc_score(np.ones((8, 2)), ang)["mrl"] == pytest.approx(0.0, abs=1e-12)
        assert np.isnan(ego.ebc_score(np.full((8, 2), np.nan), ang)["mrl"])
        assert np.isnan(ego.ebc_score(-np.ones((8, 2)), ang)["mrl"])
        assert np.isnan(ego.ebc_score(np.ones((7, 2)), ang)["mrl"])

    def test_negative_bins_rectified(self) -> None:
        ang = ego.ebc_angle_centres(4)
        rm = np.array([[1.0], [-5.0], [-5.0], [-5.0]])
        assert ego.ebc_score(rm, ang)["mrl"] == pytest.approx(1.0)


class TestEbcSignificance:
    def test_synthetic_ebc_cell_recovered(self) -> None:
        signal, x, y, h, walls = _ebc_cell()
        res = ego.ebc_significance(
            signal,
            x,
            y,
            h,
            walls,
            np.ones(signal.size, bool),
            n_shuffles=40,
            rng=np.random.default_rng(0),
        )
        assert abs(float(ego.wrap_deg(res["preferred_angle"] - 90.0))) < 20.0
        assert res["preferred_distance"] < 0.5
        assert res["p"] < 0.05
        assert res["observed"] > res["null_95"]
        assert res["n_frames"] == signal.size

    def test_random_cell_not_significant(self) -> None:
        rng = np.random.default_rng(5)
        _, x, y, h, walls = _ebc_cell(n=2000, seed=2)
        res = ego.ebc_significance(
            rng.normal(size=2000), x, y, h, walls, np.ones(2000, bool), n_shuffles=40, rng=rng
        )
        assert res["p"] > 0.05

    def test_nan_signal_frames_excluded(self) -> None:
        signal, x, y, h, walls = _ebc_cell(n=400)
        signal[:10] = np.nan
        res = ego.ebc_significance(
            signal, x, y, h, walls, np.ones(400, bool), n_shuffles=5, min_shift_s=1.0
        )
        assert res["n_frames"] == 390 and np.isfinite(res["observed"])


# ---------------------------------------------------------------------------
# Angle and distance tuning
# ---------------------------------------------------------------------------


class TestAngleTuning:
    def test_monotonic_cell(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.uniform(-180, 180, 2000)
        sig = a / 180.0 + rng.normal(0, 0.2, 2000)
        res = ego.angle_tuning(sig, a, np.ones(2000, bool))
        assert res["slope_sign"] == 1 and res["slope_rho"] > 0.9
        assert res["modulation_index"] > 2.0
        assert res["curve"].shape == (18,) and res["centres"][0] == pytest.approx(-170.0)

    def test_degenerate(self) -> None:
        res = ego.angle_tuning(np.ones(30), np.linspace(-170, 170, 30), np.ones(30, bool))
        assert np.isnan(res["modulation_index"]) and res["slope_sign"] == 0
        empty = ego.angle_tuning(np.ones(5), np.zeros(5), np.zeros(5, bool))
        assert np.isnan(empty["modulation_index"])

    def test_boundary_bins(self) -> None:
        bins = ego._angle_bins(np.array([180.0, -180.0, -179.0, np.nan]), np.ones(4, bool), 18)
        np.testing.assert_array_equal(bins, [17, 17, 0, -1])

    def test_significance(self) -> None:
        rng = np.random.default_rng(1)
        a = np.cumsum(rng.normal(0, 20, 1500))  # autocorrelated angle trajectory
        sig = np.cos(np.deg2rad(a)) + rng.normal(0, 0.3, 1500)
        res = ego.angle_tuning_significance(
            sig, a, np.ones(1500, bool), n_shuffles=40, rng=np.random.default_rng(2)
        )
        assert res["p"] < 0.05 and res["z"] > 2
        noise = ego.angle_tuning_significance(
            np.random.default_rng(8).normal(size=1500),
            a,
            np.ones(1500, bool),
            n_shuffles=40,
            rng=np.random.default_rng(3),
        )
        assert noise["p"] > 0.05

    def test_significance_with_nan_signal(self) -> None:
        rng = np.random.default_rng(4)
        a = rng.uniform(-180, 180, 300)
        sig = rng.normal(size=300)
        sig[::17] = np.nan
        res = ego.angle_tuning_significance(sig, a, np.ones(300, bool), n_shuffles=5)
        assert np.isfinite(res["observed"]) and np.isfinite(res["p"])


class TestDistanceTuning:
    def test_positive_relation(self) -> None:
        rng = np.random.default_rng(0)
        d = np.abs(np.cumsum(rng.normal(0, 0.05, 1500))) % 2.0
        sig = d + rng.normal(0, 0.2, 1500)
        res = ego.distance_tuning(sig, d, np.ones(1500, bool), n_shuffles=40, rng=rng)
        assert res["rho"] > 0.5 and res["p"] < 0.05
        assert res["n_frames"] == 1500 and np.isfinite(res["curve"]).all()

    def test_noise_not_significant(self) -> None:
        rng = np.random.default_rng(1)
        d = rng.uniform(0, 2, 1000)
        res = ego.distance_tuning(
            rng.normal(size=1000), d, np.ones(1000, bool), n_shuffles=40, rng=rng
        )
        assert res["p"] > 0.05

    def test_nan_signal_slow_path_and_no_shuffles(self) -> None:
        rng = np.random.default_rng(2)
        d = rng.uniform(0, 2, 200)
        sig = d.copy()
        sig[5] = np.nan
        res = ego.distance_tuning(
            sig, d, np.ones(200, bool), n_shuffles=10, min_shift_s=1.0, rng=rng
        )
        assert res["rho"] == pytest.approx(1.0)
        res0 = ego.distance_tuning(d, d, np.ones(200, bool), n_shuffles=0)
        assert res0["rho"] == pytest.approx(1.0) and np.isnan(res0["p"])

    def test_too_few_frames_and_constant_d(self) -> None:
        res = ego.distance_tuning([1.0, 2.0], [0.0, 1.0], [True, True])
        assert np.isnan(res["rho"]) and np.isnan(res["curve"]).all()
        const = ego.distance_tuning(np.arange(20.0), np.ones(20), np.ones(20, bool), n_shuffles=3)
        assert np.isnan(const["rho"])

    def test_col_corr_zero_variance(self) -> None:
        out = ego._col_corr(np.ones((5, 2)), np.arange(5.0))
        assert np.isnan(out).all()
