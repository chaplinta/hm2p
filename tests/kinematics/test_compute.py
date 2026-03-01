"""Tests for kinematics/compute.py — HD, position, speed, light_on, bad_behav."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.kinematics.compute import (
    MAZE_POLYGON_COORDS,
    _clip_to_maze_polygon,
    _compute_hd_deg,
    _maze_linear_transform,
    _rotate_xy,
    compute_bad_behav_mask,
    compute_head_direction,
    compute_light_on,
    compute_maze_coords,
    compute_position_mm,
)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

KEYPOINTS = ["ear-left", "ear-right", "back-upper", "back-middle", "back-tail"]


def _make_pose_dataset(
    n_frames: int = 10,
    pos_data: np.ndarray | None = None,
    conf_data: np.ndarray | None = None,
) -> "xr.Dataset":
    """Build a minimal movement-style xarray Dataset for testing.

    Args:
        n_frames: Number of time steps.
        pos_data: (time, space, keypoints, individuals). Defaults to ones.
        conf_data: (time, keypoints, individuals). Defaults to ones.

    Returns:
        xarray.Dataset with 'position' and 'confidence' DataArrays.
    """
    import xarray as xr

    n_kp = len(KEYPOINTS)
    if pos_data is None:
        pos_data = np.ones((n_frames, 2, n_kp, 1), dtype=np.float64)
    if conf_data is None:
        conf_data = np.ones((n_frames, n_kp, 1), dtype=np.float64)

    position = xr.DataArray(
        pos_data,
        dims=["time", "space", "keypoints", "individuals"],
        coords={
            "time": np.arange(n_frames, dtype=float),
            "space": ["x", "y"],
            "keypoints": KEYPOINTS,
            "individuals": ["mouse"],
        },
    )
    confidence = xr.DataArray(
        conf_data,
        dims=["time", "keypoints", "individuals"],
        coords={
            "time": np.arange(n_frames, dtype=float),
            "keypoints": KEYPOINTS,
            "individuals": ["mouse"],
        },
    )
    return xr.Dataset({"position": position, "confidence": confidence})


# ---------------------------------------------------------------------------
# _compute_hd_deg
# ---------------------------------------------------------------------------


class TestComputeHdDeg:
    def test_pointing_south(self) -> None:
        """Ear vector pointing south: dx=0, dy=-1 → atan2(0,-1)=π → 180+180=360°."""
        # ear-left directly above ear-right in image coords (smaller y)
        hd = _compute_hd_deg(
            ear_left_x=np.array([5.0]),
            ear_left_y=np.array([0.0]),
            ear_right_x=np.array([5.0]),
            ear_right_y=np.array([1.0]),
        )
        np.testing.assert_allclose(hd[0], 360.0, atol=1e-4)

    def test_constant_angle_no_unwrap(self) -> None:
        """Constant angle → all output frames equal."""
        n = 20
        hd = _compute_hd_deg(
            ear_left_x=np.ones(n),
            ear_left_y=np.zeros(n),
            ear_right_x=np.zeros(n),
            ear_right_y=np.zeros(n),
        )
        assert np.allclose(hd, hd[0])

    def test_nan_preserved(self) -> None:
        """NaN ear positions produce NaN HD at those frames."""
        ear_left_x = np.array([1.0, np.nan, 1.0])
        ear_left_y = np.array([0.0, np.nan, 0.0])
        ear_right_x = np.array([0.0, np.nan, 0.0])
        ear_right_y = np.array([0.0, np.nan, 0.0])
        hd = _compute_hd_deg(ear_left_x, ear_left_y, ear_right_x, ear_right_y)
        assert np.isnan(hd[1])
        assert not np.isnan(hd[0])
        assert not np.isnan(hd[2])

    def test_all_nan_returns_nan(self) -> None:
        """All-NaN input returns all-NaN output."""
        n = 5
        nans = np.full(n, np.nan)
        hd = _compute_hd_deg(nans, nans, nans, nans)
        assert np.all(np.isnan(hd))

    def test_output_dtype_float32(self) -> None:
        n = 5
        hd = _compute_hd_deg(
            np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n)
        )
        assert hd.dtype == np.float32

    def test_output_shape(self) -> None:
        n = 50
        hd = _compute_hd_deg(
            np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n)
        )
        assert hd.shape == (n,)

    def test_unwrap_across_360_boundary(self) -> None:
        """Rotation passing through 360° should be unwrapped (no 360° jump)."""
        n = 100
        # Linearly increasing angle from 175° to 185° (crosses 180° = ~360 unwrapped)
        angles = np.linspace(175, 185, n)
        # Build synthetic ear positions for each angle
        # HD = 180 + atan2(lx-rx, ly-ry) = angle  →  atan2(lx-rx, ly-ry) = angle-180
        rad = np.deg2rad(angles - 180.0)
        lx = np.sin(rad)
        ly = np.cos(rad)
        rx = np.zeros(n)
        ry = np.zeros(n)
        hd = _compute_hd_deg(lx, ly, rx, ry)
        jumps = np.abs(np.diff(hd))
        assert np.all(jumps < 10.0), f"Large jump detected: {jumps.max():.1f}°"


# ---------------------------------------------------------------------------
# _rotate_xy
# ---------------------------------------------------------------------------


class TestRotateXY:
    def test_identity_zero_angle(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        xr, yr = _rotate_xy(x, y, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(xr, x)
        np.testing.assert_allclose(yr, y)

    def test_90_degree_rotation_around_origin(self) -> None:
        """90° CW rotation: (1, 0) → (0, -1) around origin."""
        x = np.array([1.0])
        y = np.array([0.0])
        xr, yr = _rotate_xy(x, y, 90.0, 0.0, 0.0)
        np.testing.assert_allclose(xr, [0.0], atol=1e-10)
        np.testing.assert_allclose(yr, [-1.0], atol=1e-10)

    def test_rotation_around_nonzero_centre(self) -> None:
        """360° rotation returns original point."""
        x = np.array([3.0])
        y = np.array([4.0])
        xr, yr = _rotate_xy(x, y, 360.0, 1.0, 1.0)
        np.testing.assert_allclose(xr, x, atol=1e-10)
        np.testing.assert_allclose(yr, y, atol=1e-10)

    def test_distance_preserved(self) -> None:
        """Rotation preserves distance from centre."""
        rng = np.random.default_rng(7)
        x = rng.uniform(-10, 10, 50)
        y = rng.uniform(-10, 10, 50)
        cx, cy = 2.0, -3.0
        xr, yr = _rotate_xy(x, y, 37.5, cx, cy)
        d_before = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        d_after = np.sqrt((xr - cx) ** 2 + (yr - cy) ** 2)
        np.testing.assert_allclose(d_after, d_before, atol=1e-10)

    @given(angle=st.floats(min_value=-360, max_value=360))
    @settings(max_examples=50)
    def test_rotation_preserves_shape(self, angle: float) -> None:
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        xr, yr = _rotate_xy(x, y, angle, 0.0, 0.0)
        assert xr.shape == x.shape
        assert yr.shape == y.shape


# ---------------------------------------------------------------------------
# _maze_linear_transform
# ---------------------------------------------------------------------------


class TestMazeLinearTransform:
    def test_origin_maps_to_zero(self) -> None:
        xm, ym = _maze_linear_transform(
            np.array([10.0]), np.array([20.0]),
            x1_mm=10.0, y1_mm=20.0, width_mm=100.0, height_mm=50.0,
        )
        np.testing.assert_allclose(xm, [0.0], atol=1e-6)
        np.testing.assert_allclose(ym, [0.0], atol=1e-6)

    def test_far_corner_maps_to_7_5(self) -> None:
        xm, ym = _maze_linear_transform(
            np.array([110.0]), np.array([70.0]),
            x1_mm=10.0, y1_mm=20.0, width_mm=100.0, height_mm=50.0,
        )
        np.testing.assert_allclose(xm, [7.0], atol=1e-6)
        np.testing.assert_allclose(ym, [5.0], atol=1e-6)

    def test_midpoint_maps_to_3_5_2_5(self) -> None:
        xm, ym = _maze_linear_transform(
            np.array([60.0]), np.array([45.0]),
            x1_mm=10.0, y1_mm=20.0, width_mm=100.0, height_mm=50.0,
        )
        np.testing.assert_allclose(xm, [3.5], atol=1e-6)
        np.testing.assert_allclose(ym, [2.5], atol=1e-6)

    def test_output_dtype_float32(self) -> None:
        xm, ym = _maze_linear_transform(
            np.array([0.0]), np.array([0.0]),
            x1_mm=0.0, y1_mm=0.0, width_mm=10.0, height_mm=10.0,
        )
        assert xm.dtype == np.float32
        assert ym.dtype == np.float32


# ---------------------------------------------------------------------------
# _clip_to_maze_polygon
# ---------------------------------------------------------------------------


def test_clip_inside_point_unchanged() -> None:
    """A point inside the valid maze polygon is not moved."""
    pytest.importorskip("shapely")
    # (3.5, 2.5) is confirmed inside the make_valid() decomposition
    x = np.array([3.5], dtype=np.float32)
    y = np.array([2.5], dtype=np.float32)
    xc, yc = _clip_to_maze_polygon(x, y)
    np.testing.assert_allclose(xc, x, atol=0.01)
    np.testing.assert_allclose(yc, y, atol=0.01)


def test_clip_outside_point_moves() -> None:
    """A point well outside the maze is moved to the boundary."""
    pytest.importorskip("shapely")
    x = np.array([10.0], dtype=np.float32)
    y = np.array([10.0], dtype=np.float32)
    xc, yc = _clip_to_maze_polygon(x, y)
    # Should be somewhere on the boundary, not at (10, 10)
    assert xc[0] <= 7.0 and yc[0] <= 5.0


def test_clip_nan_preserved() -> None:
    """NaN positions are passed through unchanged."""
    pytest.importorskip("shapely")
    x = np.array([np.nan, 3.5], dtype=np.float32)
    y = np.array([np.nan, 0.5], dtype=np.float32)
    xc, yc = _clip_to_maze_polygon(x, y)
    assert np.isnan(xc[0])
    assert np.isnan(yc[0])


def test_clip_output_dtype() -> None:
    pytest.importorskip("shapely")
    x = np.array([3.5], dtype=np.float32)
    y = np.array([0.5], dtype=np.float32)
    xc, yc = _clip_to_maze_polygon(x, y)
    assert xc.dtype == np.float32
    assert yc.dtype == np.float32


# ---------------------------------------------------------------------------
# compute_head_direction (xarray integration)
# ---------------------------------------------------------------------------


class TestComputeHeadDirection:
    def test_output_shape(self) -> None:
        pytest.importorskip("xarray")
        n = 15
        ds = _make_pose_dataset(n_frames=n)
        hd = compute_head_direction(ds)
        assert hd.shape == (n,)

    def test_output_dtype(self) -> None:
        pytest.importorskip("xarray")
        ds = _make_pose_dataset()
        hd = compute_head_direction(ds)
        assert hd.dtype == np.float32

    def test_known_angle(self) -> None:
        """Ear-left directly above ear-right → specific HD value."""
        pytest.importorskip("xarray")
        n = 5
        # ear-left at (5, 0), ear-right at (5, 1)
        # atan2(5-5, 0-1) = atan2(0, -1) = π  → 180+180 = 360
        pos_data = np.zeros((n, 2, len(KEYPOINTS), 1), dtype=np.float64)
        kp_idx = {k: i for i, k in enumerate(KEYPOINTS)}
        pos_data[:, 0, kp_idx["ear-left"], 0] = 5.0   # x
        pos_data[:, 1, kp_idx["ear-left"], 0] = 0.0   # y
        pos_data[:, 0, kp_idx["ear-right"], 0] = 5.0  # x
        pos_data[:, 1, kp_idx["ear-right"], 0] = 1.0  # y
        # Fill back keypoints with something reasonable
        for kp in ["back-upper", "back-middle", "back-tail"]:
            pos_data[:, 0, kp_idx[kp], 0] = 5.0
            pos_data[:, 1, kp_idx[kp], 0] = 3.0
        ds = _make_pose_dataset(n_frames=n, pos_data=pos_data)
        hd = compute_head_direction(ds)
        # arctan2(0, -1) = π, 180 + 180 = 360°
        np.testing.assert_allclose(hd[0], 360.0, atol=1.0)


# ---------------------------------------------------------------------------
# compute_position_mm (xarray integration)
# ---------------------------------------------------------------------------


class TestComputePositionMm:
    def test_output_shape(self) -> None:
        pytest.importorskip("xarray")
        n = 20
        ds = _make_pose_dataset(n_frames=n)
        x_mm, y_mm = compute_position_mm(ds, scale_mm_per_px=0.811)
        assert x_mm.shape == (n,)
        assert y_mm.shape == (n,)

    def test_output_dtype(self) -> None:
        pytest.importorskip("xarray")
        ds = _make_pose_dataset()
        x_mm, y_mm = compute_position_mm(ds, scale_mm_per_px=0.811)
        assert x_mm.dtype == np.float32
        assert y_mm.dtype == np.float32

    def test_scale_applied(self) -> None:
        """All-ones position × scale → constant output equal to scale."""
        pytest.importorskip("xarray")
        n = 5
        scale = 2.5
        ds = _make_pose_dataset(n_frames=n)
        x_mm, y_mm = compute_position_mm(ds, scale_mm_per_px=scale)
        np.testing.assert_allclose(x_mm, scale, rtol=1e-5)
        np.testing.assert_allclose(y_mm, scale, rtol=1e-5)

    def test_centroid_of_back_keypoints(self) -> None:
        """Position is mean of back-upper, back-middle, back-tail × scale."""
        pytest.importorskip("xarray")
        n = 3
        scale = 1.0
        pos_data = np.zeros((n, 2, len(KEYPOINTS), 1), dtype=np.float64)
        kp_idx = {k: i for i, k in enumerate(KEYPOINTS)}
        # Set back keypoints to known x values: 2, 4, 6 → mean = 4
        pos_data[:, 0, kp_idx["back-upper"], 0] = 2.0
        pos_data[:, 0, kp_idx["back-middle"], 0] = 4.0
        pos_data[:, 0, kp_idx["back-tail"], 0] = 6.0
        pos_data[:, 1, :, 0] = 1.0
        # Ear keypoints are irrelevant for position
        ds = _make_pose_dataset(n_frames=n, pos_data=pos_data)
        x_mm, _ = compute_position_mm(ds, scale_mm_per_px=scale)
        np.testing.assert_allclose(x_mm, 4.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# compute_maze_coords
# ---------------------------------------------------------------------------


class TestComputeMazeCoords:
    def _default_corners(self) -> np.ndarray:
        """Maze corners in pixels (from meta.txt typical values)."""
        return np.array(
            [[149.0, 72.0], [764.0, 82.0], [757.0, 509.0], [143.0, 500.0]],
            dtype=np.float64,
        )

    def test_output_shape(self) -> None:
        pytest.importorskip("shapely")
        n = 10
        x_mm = np.linspace(100.0, 500.0, n, dtype=np.float32)
        y_mm = np.linspace(60.0, 400.0, n, dtype=np.float32)
        corners = self._default_corners()
        xm, ym = compute_maze_coords(x_mm, y_mm, corners, scale_mm_per_px=0.811)
        assert xm.shape == (n,)
        assert ym.shape == (n,)

    def test_output_dtype(self) -> None:
        pytest.importorskip("shapely")
        x_mm = np.array([200.0], dtype=np.float32)
        y_mm = np.array([200.0], dtype=np.float32)
        corners = self._default_corners()
        xm, ym = compute_maze_coords(x_mm, y_mm, corners, scale_mm_per_px=0.811)
        assert xm.dtype == np.float32
        assert ym.dtype == np.float32

    def test_output_in_maze_bounds(self) -> None:
        """All output coords should be within [0, 7] × [0, 5] after clipping."""
        pytest.importorskip("shapely")
        rng = np.random.default_rng(99)
        # Wide range to ensure some are OOB
        x_mm = rng.uniform(-10, 800, 50).astype(np.float32)
        y_mm = rng.uniform(-10, 600, 50).astype(np.float32)
        corners = self._default_corners()
        xm, ym = compute_maze_coords(x_mm, y_mm, corners, scale_mm_per_px=0.811)
        assert np.all(xm >= -0.1) and np.all(xm <= 7.1)
        assert np.all(ym >= -0.1) and np.all(ym <= 5.1)


# ---------------------------------------------------------------------------
# compute_bad_behav_mask — pure numpy, testable immediately
# ---------------------------------------------------------------------------


def test_bad_behav_mask_empty_intervals() -> None:
    """Empty interval list produces all-False mask."""
    times = np.linspace(0, 600, 6000)
    mask = compute_bad_behav_mask(times, [])
    assert not mask.any()


def test_bad_behav_mask_single_interval() -> None:
    """Frames within a bad_behav interval are True."""
    times = np.array([0.0, 60.0, 120.0, 150.0, 165.0, 180.0, 240.0])
    intervals = [(120.0, 180.0)]
    mask = compute_bad_behav_mask(times, intervals)
    expected = np.array([False, False, True, True, True, True, False])
    np.testing.assert_array_equal(mask, expected)


def test_bad_behav_mask_multiple_intervals() -> None:
    """Frames in any bad_behav interval are True."""
    times = np.linspace(0, 600, 61)  # one sample per 10 s
    intervals = [(50.0, 100.0), (200.0, 250.0)]
    mask = compute_bad_behav_mask(times, intervals)
    for i, t in enumerate(times):
        expected = (50.0 <= t <= 100.0) or (200.0 <= t <= 250.0)
        assert mask[i] == expected, f"Frame at t={t:.1f} s: got {mask[i]}, expected {expected}"


def test_bad_behav_mask_shape_preserved() -> None:
    """Output shape matches input frame_times shape."""
    times = np.linspace(0, 600, 1234)
    mask = compute_bad_behav_mask(times, [(10.0, 20.0)])
    assert mask.shape == times.shape


# ---------------------------------------------------------------------------
# compute_light_on — pure numpy, testable immediately
# ---------------------------------------------------------------------------


def test_light_on_all_dark() -> None:
    """All frames before first light-on pulse are dark."""
    times = np.array([0.0, 10.0, 30.0, 59.9])
    light_on_times = np.array([60.0])
    light_off_times = np.array([120.0])
    result = compute_light_on(times, light_on_times, light_off_times)
    assert not result.any()


def test_light_on_alternating_cycle() -> None:
    """Light_on correctly alternates between on/off epochs."""
    # 1 min on / 1 min off, 4 cycles
    light_on = np.array([0.0, 120.0, 240.0, 360.0])
    light_off = np.array([60.0, 180.0, 300.0, 420.0])
    # Sample one frame per 30 s
    times = np.arange(0, 480, 30, dtype=float)
    result = compute_light_on(times, light_on, light_off)
    for i, t in enumerate(times):
        # Determine expected state: find which epoch we're in
        on_periods = [(on, off) for on, off in zip(light_on, light_off, strict=True)]
        expected = any(on <= t < off for on, off in on_periods)
        assert result[i] == expected, f"t={t}: got {result[i]}, expected {expected}"


# ---------------------------------------------------------------------------
# MAZE_POLYGON_COORDS — basic sanity
# ---------------------------------------------------------------------------


def test_maze_polygon_bounds() -> None:
    """MAZE_POLYGON_COORDS fit within the 7×5 rose-maze grid."""
    # The polygon has designed self-intersections (corridors) so is_valid=False
    # is expected. Use make_valid() at runtime for clipping.
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Polygon
    poly = Polygon(MAZE_POLYGON_COORDS)
    valid_poly = shapely.make_valid(poly)
    bounds = valid_poly.bounds
    assert bounds[0] >= 0, f"x_min={bounds[0]} < 0"
    assert bounds[1] >= 0, f"y_min={bounds[1]} < 0"
    assert bounds[2] <= 7, f"x_max={bounds[2]} > 7"
    assert bounds[3] <= 5, f"y_max={bounds[3]} > 5"


def test_maze_polygon_interior_point() -> None:
    """A known interior point (centre of maze) is inside the valid polygon."""
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Point, Polygon
    poly = Polygon(MAZE_POLYGON_COORDS)
    valid_poly = shapely.make_valid(poly)
    # (3.5, 2.5) is the approximate centre of the 7×5 maze — inside a corridor
    centre = Point(3.5, 2.5)
    assert valid_poly.contains(centre) or valid_poly.touches(centre)
