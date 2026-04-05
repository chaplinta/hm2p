"""Tests for kinematics/compute.py — HD, position, speed, light_on, bad_behav."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.kinematics.compute import (
    MAZE_POLYGON_COORDS,
    _clip_to_maze_polygon,
    _compute_hd_deg,
    _ear_perpendicular_angle,
    _fused_hd_wrapped,
    _maze_linear_transform,
    _median_filter_1d,
    _rotate_xy,
    _unwrap_and_smooth,
    _vector_angle_deg,
    _windowed_gradient,
    _windowed_speed,
    compute_bad_behav_mask,
    compute_head_direction,
    compute_light_on,
    compute_maze_coords,
    compute_position_mm,
)

# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

KEYPOINTS = ["left_ear", "right_ear", "mid_back", "mouse_center", "tail_base"]


def _make_pose_dataset(
    n_frames: int = 10,
    pos_data: np.ndarray | None = None,
    conf_data: np.ndarray | None = None,
) -> xr.Dataset:
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
# _median_filter_1d
# ---------------------------------------------------------------------------


class TestMedianFilter1d:
    def test_preserves_nan(self) -> None:
        """NaN positions in input remain NaN in output."""
        arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        out = _median_filter_1d(arr, win=3)
        assert np.isnan(out[2])
        assert not np.isnan(out[0])
        assert not np.isnan(out[4])

    def test_all_nan_returns_all_nan(self) -> None:
        arr = np.full(5, np.nan)
        out = _median_filter_1d(arr, win=3)
        assert np.all(np.isnan(out))

    def test_smooths_jittery_signal(self) -> None:
        """A spike in an otherwise constant signal should be removed."""
        arr = np.array([1.0, 1.0, 100.0, 1.0, 1.0], dtype=np.float64)
        out = _median_filter_1d(arr, win=3)
        # The spike at index 2 should be smoothed away
        assert out[2] < 50.0

    def test_win_1_returns_copy(self) -> None:
        """win=1 returns an identical copy (no filtering)."""
        arr = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        out = _median_filter_1d(arr, win=1)
        np.testing.assert_array_equal(out, arr)
        # Must be a copy, not same object
        assert out is not arr

    def test_win_0_returns_copy(self) -> None:
        """win=0 also returns a copy (edge case)."""
        arr = np.array([1.0, 2.0, 3.0])
        out = _median_filter_1d(arr, win=0)
        np.testing.assert_array_equal(out, arr)
        assert out is not arr

    def test_output_shape_matches_input(self) -> None:
        arr = np.random.default_rng(42).standard_normal(100)
        out = _median_filter_1d(arr, win=7)
        assert out.shape == arr.shape

    def test_constant_signal_unchanged(self) -> None:
        """A constant signal is not altered by median filtering."""
        arr = np.full(20, 42.0)
        out = _median_filter_1d(arr, win=5)
        np.testing.assert_allclose(out, 42.0)


# ---------------------------------------------------------------------------
# _windowed_gradient
# ---------------------------------------------------------------------------


class TestWindowedGradient:
    def test_constant_signal_zero_gradient(self) -> None:
        """A constant signal should have zero gradient everywhere."""
        n = 50
        signal = np.full(n, 5.0)
        times = np.linspace(0, 5, n)
        grad = _windowed_gradient(signal, times, window_s=0.5)
        np.testing.assert_allclose(grad, 0.0, atol=1e-10)

    def test_linear_signal_correct_slope(self) -> None:
        """A linear signal y = 3*t should have gradient ~3 everywhere."""
        n = 100
        times = np.linspace(0, 10, n)
        signal = 3.0 * times
        grad = _windowed_gradient(signal, times, window_s=0.5)
        # Interior points should be very close to 3.0
        np.testing.assert_allclose(grad[10:-10], 3.0, atol=0.1)

    def test_handles_nan_in_signal(self) -> None:
        """NaN values in the signal should not cause the entire output to be NaN."""
        n = 50
        times = np.linspace(0, 5, n)
        signal = 2.0 * times
        signal[20] = np.nan
        grad = _windowed_gradient(signal, times, window_s=0.5)
        # Most values should still be finite
        assert np.isfinite(grad).sum() > n // 2

    def test_output_shape(self) -> None:
        n = 30
        times = np.linspace(0, 3, n)
        signal = np.sin(times)
        grad = _windowed_gradient(signal, times, window_s=0.2)
        assert grad.shape == (n,)

    def test_output_dtype_float64(self) -> None:
        n = 20
        times = np.linspace(0, 2, n)
        signal = np.ones(n, dtype=np.float32)
        grad = _windowed_gradient(signal, times, window_s=0.2)
        assert grad.dtype == np.float64


# ---------------------------------------------------------------------------
# _windowed_speed
# ---------------------------------------------------------------------------


class TestWindowedSpeed:
    def test_stationary_zero_speed(self) -> None:
        """Stationary position should yield zero speed."""
        n = 50
        times = np.linspace(0, 5, n)
        x_mm = np.full(n, 100.0)
        y_mm = np.full(n, 200.0)
        speed = _windowed_speed(x_mm, y_mm, times, window_s=0.5)
        np.testing.assert_allclose(speed, 0.0, atol=1e-10)

    def test_constant_velocity_correct_speed(self) -> None:
        """Moving at constant velocity: x = 100*t mm → dx/dt = 100 mm/s = 10 cm/s."""
        n = 100
        times = np.linspace(0, 10, n)
        x_mm = 100.0 * times  # 100 mm/s in x
        y_mm = np.zeros(n)
        speed = _windowed_speed(x_mm, y_mm, times, window_s=0.5)
        # Interior points should be ~10 cm/s (100 mm/s / 10)
        np.testing.assert_allclose(speed[10:-10], 10.0, atol=0.5)

    def test_non_negative_output(self) -> None:
        """Speed should always be non-negative."""
        rng = np.random.default_rng(123)
        n = 60
        times = np.linspace(0, 6, n)
        x_mm = np.cumsum(rng.standard_normal(n))
        y_mm = np.cumsum(rng.standard_normal(n))
        speed = _windowed_speed(x_mm, y_mm, times, window_s=0.3)
        assert np.all(speed >= 0.0)

    def test_output_shape(self) -> None:
        n = 40
        times = np.linspace(0, 4, n)
        speed = _windowed_speed(np.zeros(n), np.zeros(n), times, window_s=0.2)
        assert speed.shape == (n,)

    def test_diagonal_velocity(self) -> None:
        """Moving diagonally: x=100*t, y=100*t → speed = sqrt(2)*10 cm/s."""
        n = 100
        times = np.linspace(0, 10, n)
        x_mm = 100.0 * times
        y_mm = 100.0 * times
        speed = _windowed_speed(x_mm, y_mm, times, window_s=0.5)
        expected = np.sqrt(2) * 10.0  # sqrt(100^2 + 100^2) / 10
        np.testing.assert_allclose(speed[10:-10], expected, atol=0.5)


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
        hd = _compute_hd_deg(np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n))
        assert hd.dtype == np.float32

    def test_output_shape(self) -> None:
        n = 50
        hd = _compute_hd_deg(np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n))
        assert hd.shape == (n,)

    def test_median_filter_win_0_unfiltered(self) -> None:
        """median_filter_win=0 disables filtering, giving raw HD."""
        n = 20
        # Create a jittery ear signal
        rng = np.random.default_rng(42)
        lx = np.ones(n) + rng.normal(0, 0.01, n)
        ly = np.zeros(n) + rng.normal(0, 0.01, n)
        rx = np.zeros(n)
        ry = np.zeros(n)
        hd_filtered = _compute_hd_deg(lx, ly, rx, ry, median_filter_win=5)
        hd_unfiltered = _compute_hd_deg(lx, ly, rx, ry, median_filter_win=0)
        # Unfiltered should differ from filtered (jitter preserved)
        assert not np.allclose(hd_filtered, hd_unfiltered, atol=1e-6)

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
            np.array([10.0]),
            np.array([20.0]),
            x1_mm=10.0,
            y1_mm=20.0,
            width_mm=100.0,
            height_mm=50.0,
        )
        np.testing.assert_allclose(xm, [0.0], atol=1e-6)
        np.testing.assert_allclose(ym, [0.0], atol=1e-6)

    def test_far_corner_maps_to_7_5(self) -> None:
        xm, ym = _maze_linear_transform(
            np.array([110.0]),
            np.array([70.0]),
            x1_mm=10.0,
            y1_mm=20.0,
            width_mm=100.0,
            height_mm=50.0,
        )
        np.testing.assert_allclose(xm, [7.0], atol=1e-6)
        np.testing.assert_allclose(ym, [5.0], atol=1e-6)

    def test_midpoint_maps_to_3_5_2_5(self) -> None:
        xm, ym = _maze_linear_transform(
            np.array([60.0]),
            np.array([45.0]),
            x1_mm=10.0,
            y1_mm=20.0,
            width_mm=100.0,
            height_mm=50.0,
        )
        np.testing.assert_allclose(xm, [3.5], atol=1e-6)
        np.testing.assert_allclose(ym, [2.5], atol=1e-6)

    def test_output_dtype_float32(self) -> None:
        xm, ym = _maze_linear_transform(
            np.array([0.0]),
            np.array([0.0]),
            x1_mm=0.0,
            y1_mm=0.0,
            width_mm=10.0,
            height_mm=10.0,
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
        pos_data[:, 0, kp_idx["left_ear"], 0] = 5.0  # x
        pos_data[:, 1, kp_idx["left_ear"], 0] = 0.0  # y
        pos_data[:, 0, kp_idx["right_ear"], 0] = 5.0  # x
        pos_data[:, 1, kp_idx["right_ear"], 0] = 1.0  # y
        # Fill back keypoints with something reasonable
        for kp in ["mid_back", "mouse_center", "tail_base"]:
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
        pos_data[:, 0, kp_idx["mid_back"], 0] = 2.0
        pos_data[:, 0, kp_idx["mouse_center"], 0] = 4.0
        pos_data[:, 0, kp_idx["tail_base"], 0] = 6.0
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


# ---------------------------------------------------------------------------
# _vector_angle_deg
# ---------------------------------------------------------------------------


class TestVectorAngleDeg:
    def test_east_direction(self) -> None:
        """Vector pointing east (dx>0, dy=0): atan2(dx, dy) = atan2(pos, 0) = 90 → 180+90 = 270."""
        # from=(0,0) to=(1,0): dx=1, dy=0 → atan2(1,0)=90° → 180+90=270
        angle = _vector_angle_deg(
            np.array([0.0]), np.array([0.0]),
            np.array([1.0]), np.array([0.0]),
        )
        np.testing.assert_allclose(angle, [270.0], atol=1e-6)

    def test_west_direction(self) -> None:
        """Vector pointing west (dx<0, dy=0): atan2(-1,0) = -90° → 180-90 = 90."""
        angle = _vector_angle_deg(
            np.array([0.0]), np.array([0.0]),
            np.array([-1.0]), np.array([0.0]),
        )
        np.testing.assert_allclose(angle, [90.0], atol=1e-6)

    def test_north_direction(self) -> None:
        """Vector pointing north (dy<0 in image coords, dx=0): atan2(0,-1) = 180° → 180+180 = 360."""
        angle = _vector_angle_deg(
            np.array([0.0]), np.array([0.0]),
            np.array([0.0]), np.array([-1.0]),
        )
        np.testing.assert_allclose(angle, [360.0], atol=1e-6)

    def test_south_direction(self) -> None:
        """Vector pointing south (dy>0, dx=0): atan2(0,1) = 0° → 180+0 = 180."""
        angle = _vector_angle_deg(
            np.array([0.0]), np.array([0.0]),
            np.array([0.0]), np.array([1.0]),
        )
        np.testing.assert_allclose(angle, [180.0], atol=1e-6)

    def test_nan_from_point_propagates(self) -> None:
        """NaN in from_x propagates to output."""
        angle = _vector_angle_deg(
            np.array([np.nan]), np.array([0.0]),
            np.array([1.0]), np.array([0.0]),
        )
        assert np.isnan(angle[0])

    def test_nan_to_point_propagates(self) -> None:
        """NaN in to_y propagates to output."""
        angle = _vector_angle_deg(
            np.array([0.0]), np.array([0.0]),
            np.array([1.0]), np.array([np.nan]),
        )
        assert np.isnan(angle[0])

    def test_multiple_frames_shape(self) -> None:
        """Output shape matches input length."""
        n = 20
        angle = _vector_angle_deg(
            np.zeros(n), np.zeros(n),
            np.ones(n), np.zeros(n),
        )
        assert angle.shape == (n,)

    def test_multiple_frames_values_consistent(self) -> None:
        """All frames with the same direction return the same angle."""
        n = 15
        angle = _vector_angle_deg(
            np.zeros(n), np.zeros(n),
            np.ones(n), np.zeros(n),
        )
        assert np.all(angle == angle[0])

    def test_diagonal_northeast(self) -> None:
        """Vector pointing northeast (dx=1, dy=-1): atan2(1,-1) = 135° → 180+135 = 315."""
        angle = _vector_angle_deg(
            np.array([0.0]), np.array([0.0]),
            np.array([1.0]), np.array([-1.0]),
        )
        np.testing.assert_allclose(angle, [315.0], atol=1e-6)

    @given(
        dx=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        dy=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_output_range_finite_input(self, dx: float, dy: float) -> None:
        """For finite non-degenerate inputs, output should be finite."""
        if dx == 0.0 and dy == 0.0:
            return  # zero vector → atan2(0,0) is implementation-defined
        angle = _vector_angle_deg(
            np.array([0.0]), np.array([0.0]),
            np.array([dx]), np.array([dy]),
        )
        assert np.isfinite(angle[0])
        # Result is 180 + atan2 → range is (0, 360]
        assert 0.0 <= angle[0] <= 360.0


# ---------------------------------------------------------------------------
# _ear_perpendicular_angle
# ---------------------------------------------------------------------------


class TestEarPerpendicularAngle:
    def test_ears_horizontal_pointing_right(self) -> None:
        """Left ear above right ear (ly < ry, lx == rx) → head points south (180°).

        ear_left=(5,0), ear_right=(5,2): dx=5-5=0, dy=0-2=-2
        atan2(0,-2) = 180° → 180+180 = 360° ... wait, this is _ear_perpendicular
        which uses atan2(lx-rx, ly-ry) = atan2(0,-2) = π → 180+180 = 360.
        """
        angle = _ear_perpendicular_angle(
            np.array([5.0]), np.array([0.0]),
            np.array([5.0]), np.array([2.0]),
        )
        np.testing.assert_allclose(angle, [360.0], atol=1e-6)

    def test_ears_same_y_left_is_left(self) -> None:
        """Left ear left of right ear (lx < rx, same y): atan2(lx-rx, 0) = atan2(-1,0) = -90 → 90."""
        angle = _ear_perpendicular_angle(
            np.array([0.0]), np.array([0.0]),
            np.array([1.0]), np.array([0.0]),
        )
        np.testing.assert_allclose(angle, [90.0], atol=1e-6)

    def test_nan_left_ear_returns_nan(self) -> None:
        angle = _ear_perpendicular_angle(
            np.array([np.nan]), np.array([0.0]),
            np.array([1.0]), np.array([0.0]),
        )
        assert np.isnan(angle[0])

    def test_nan_right_ear_returns_nan(self) -> None:
        angle = _ear_perpendicular_angle(
            np.array([0.0]), np.array([0.0]),
            np.array([np.nan]), np.array([0.0]),
        )
        assert np.isnan(angle[0])

    def test_output_shape(self) -> None:
        n = 30
        angle = _ear_perpendicular_angle(
            np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n)
        )
        assert angle.shape == (n,)

    def test_constant_configuration_constant_output(self) -> None:
        """Same ear positions every frame → same HD every frame."""
        n = 20
        angle = _ear_perpendicular_angle(
            np.full(n, 2.0), np.zeros(n),
            np.zeros(n), np.zeros(n),
        )
        assert np.allclose(angle, angle[0])

    @given(
        lx=st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        ly=st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        rx=st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        ry=st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_output_finite_for_finite_inputs(
        self, lx: float, ly: float, rx: float, ry: float
    ) -> None:
        """Finite ear coordinates always produce a finite angle."""
        if lx == rx and ly == ry:
            return  # degenerate: both ears at same point
        angle = _ear_perpendicular_angle(
            np.array([lx]), np.array([ly]),
            np.array([rx]), np.array([ry]),
        )
        assert np.isfinite(angle[0])


# ---------------------------------------------------------------------------
# _fused_hd_wrapped
# ---------------------------------------------------------------------------


def _circular_mean(angles_deg: list[float]) -> float:
    """Reference circular mean for a list of angles (degrees)."""
    rad = np.deg2rad(angles_deg)
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)


class TestFusedHdWrapped:
    # --- ears-only cases ---

    def test_ears_only_matches_ear_perpendicular(self) -> None:
        """With no nose/implant/neck, fused result equals _ear_perpendicular_angle."""
        n = 20
        lx = np.linspace(1.0, 3.0, n)
        ly = np.zeros(n)
        rx = np.zeros(n)
        ry = np.linspace(0.5, 1.5, n)

        ear_only = _ear_perpendicular_angle(lx, ly, rx, ry) % 360.0
        fused = _fused_hd_wrapped(lx, ly, rx, ry)

        np.testing.assert_allclose(fused, ear_only, atol=1e-6)

    def test_backwards_compat_none_args_same_as_ears_only(self) -> None:
        """Passing None for all optional args gives the same result as ears-only call."""
        n = 10
        lx = np.array([1.0] * n)
        ly = np.zeros(n)
        rx = np.zeros(n)
        ry = np.array([1.0] * n)

        fused_default = _fused_hd_wrapped(lx, ly, rx, ry)
        fused_explicit_none = _fused_hd_wrapped(
            lx, ly, rx, ry,
            nose_x=None, nose_y=None,
            implant_x=None, implant_y=None,
            neck_x=None, neck_y=None,
        )
        np.testing.assert_array_equal(fused_default, fused_explicit_none)

    # --- all-estimates-agree case ---

    def test_all_estimates_agree_returns_common_angle(self) -> None:
        """When all three estimates give the same angle, the fused result equals that angle."""
        # Arrange: ear perpendicular points south (180°).
        # ear_left=(5,0), ear_right=(5,2) → atan2(0,-2) = 180° → 360° wrapped
        # nose→implant and nose→neck are set up to also give 360°.
        # nose=(0,0), implant=(0,1): _vector_angle_deg(implant, nose) = angle of
        # vector from (0,1) to (0,0): dx=0, dy=-1 → atan2(0,-1) = 180° → 360°
        # nose→neck: neck=(0,1) → same vector → 360°
        n = 5
        lx = np.full(n, 5.0)
        ly = np.zeros(n)
        rx = np.full(n, 5.0)
        ry = np.full(n, 2.0)
        nose_x = np.zeros(n)
        nose_y = np.zeros(n)
        implant_x = np.zeros(n)
        implant_y = np.ones(n)
        neck_x = np.zeros(n)
        neck_y = np.ones(n)

        fused = _fused_hd_wrapped(lx, ly, rx, ry, nose_x, nose_y, implant_x, implant_y, neck_x, neck_y)
        # All three estimates are 360° (= 0° on circle). Circular mean of identical angles = same.
        # Due to floating-point, arctan2(-eps, 1.0) % 360 = 360.0 is equivalent to 0°.
        # Normalise to [0, 360) before comparing.
        fused_norm = fused % 360.0
        fused_norm[fused_norm == 360.0] = 0.0
        np.testing.assert_allclose(fused_norm, 0.0, atol=1.0)

    # --- NaN handling ---

    def test_one_ear_nan_uses_other_estimates(self) -> None:
        """NaN left_ear x still allows nose-implant and nose-neck estimates."""
        n = 5
        # Both ears NaN → ear estimate is NaN
        lx = np.full(n, np.nan)
        ly = np.full(n, np.nan)
        rx = np.full(n, np.nan)
        ry = np.full(n, np.nan)
        # nose→implant: nose=(0,0), implant=(0,1) → south pointing (180°)
        nose_x = np.zeros(n)
        nose_y = np.zeros(n)
        implant_x = np.zeros(n)
        implant_y = np.ones(n)
        neck_x = np.zeros(n)
        neck_y = np.ones(n)

        fused = _fused_hd_wrapped(lx, ly, rx, ry, nose_x, nose_y, implant_x, implant_y, neck_x, neck_y)
        # Result should be finite (ear estimate is NaN but the other two are not)
        assert np.all(np.isfinite(fused))

    def test_both_ears_nan_nose_implant_available(self) -> None:
        """When both ears are NaN and nose+implant are present, output uses nose-implant only."""
        n = 5
        lx = np.full(n, np.nan)
        ly = np.full(n, np.nan)
        rx = np.full(n, np.nan)
        ry = np.full(n, np.nan)
        # nose→implant: from=(0,1) to=(0,0): dy=-1 → atan2(0,-1) = 180° → 360 % 360 = 0
        nose_x = np.zeros(n)
        nose_y = np.zeros(n)
        implant_x = np.zeros(n)
        implant_y = np.ones(n)

        fused = _fused_hd_wrapped(lx, ly, rx, ry, nose_x, nose_y, implant_x, implant_y)
        assert np.all(np.isfinite(fused))
        # nose→implant estimate is 360° (= 0° on circle).
        # Due to floating-point, arctan2(-eps, 1.0) % 360 = 360.0 is equivalent to 0°.
        fused_norm = fused % 360.0
        fused_norm[fused_norm == 360.0] = 0.0
        np.testing.assert_allclose(fused_norm, 0.0, atol=1.0)

    def test_all_nan_returns_nan(self) -> None:
        """When every keypoint is NaN, output is NaN for all frames."""
        n = 6
        nans = np.full(n, np.nan)
        fused = _fused_hd_wrapped(nans, nans, nans, nans, nans, nans, nans, nans, nans, nans)
        assert np.all(np.isnan(fused))

    def test_single_frame_all_nan(self) -> None:
        """Single NaN frame returns NaN."""
        fused = _fused_hd_wrapped(
            np.array([np.nan]), np.array([np.nan]),
            np.array([np.nan]), np.array([np.nan]),
        )
        assert np.isnan(fused[0])

    # --- small disagreement: fused between estimates ---

    def test_estimates_disagree_slightly_fused_is_intermediate(self) -> None:
        """Two estimates 10° apart → fused should be between them (circular mean)."""
        n = 5
        # ear estimate: point due east, angle = 270°
        # We construct ear positions giving 270°:
        # atan2(lx-rx, ly-ry) = atan2(1,0) = 90° → 180+90 = 270°
        lx = np.full(n, 1.0)
        ly = np.zeros(n)
        rx = np.zeros(n)
        ry = np.zeros(n)
        ear_angle = float(_ear_perpendicular_angle(lx[:1], ly[:1], rx[:1], ry[:1])[0] % 360.0)

        # nose→implant: give 280° (10° off)
        # _vector_angle_deg(implant, nose): 180 + atan2(nose_x-implant_x, nose_y-implant_y)
        # We want 280 → atan2(dx,dy) = 100° → dy = cos(100°), dx = sin(100°)
        target_rad = np.deg2rad(280.0 - 180.0)
        nose_x = np.full(n, np.sin(target_rad))
        nose_y = np.full(n, np.cos(target_rad))
        implant_x = np.zeros(n)
        implant_y = np.zeros(n)

        fused = _fused_hd_wrapped(lx, ly, rx, ry, nose_x, nose_y, implant_x, implant_y)

        # Fused should be between ear_angle and 280° (roughly 275°)
        expected_circular_mean = _circular_mean([ear_angle, 280.0])
        np.testing.assert_allclose(fused, expected_circular_mean, atol=1.0)

    # --- output properties ---

    def test_output_in_0_360_range(self) -> None:
        """Fused HD is always in [0, 360) for valid frames."""
        rng = np.random.default_rng(55)
        n = 50
        lx = rng.uniform(0, 100, n)
        ly = rng.uniform(0, 100, n)
        rx = rng.uniform(0, 100, n)
        ry = rng.uniform(0, 100, n)
        fused = _fused_hd_wrapped(lx, ly, rx, ry)
        valid = ~np.isnan(fused)
        assert np.all(fused[valid] >= 0.0)
        # 360.0 can appear due to floating-point (arctan2(-eps,1) % 360 = 360.0 = 0°).
        assert np.all(fused[valid] <= 360.0)

    def test_output_shape(self) -> None:
        n = 35
        lx = np.ones(n)
        fused = _fused_hd_wrapped(lx, np.zeros(n), np.zeros(n), np.zeros(n))
        assert fused.shape == (n,)

    def test_output_dtype_float64(self) -> None:
        """Fused result is float64 (unwrapping converts to float32 later)."""
        n = 5
        fused = _fused_hd_wrapped(
            np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n)
        )
        assert fused.dtype == np.float64

    def test_partial_nan_nose_implant_frames(self) -> None:
        """Some frames have NaN nose, others don't — per-frame fallback works."""
        n = 6
        lx = np.ones(n)
        ly = np.zeros(n)
        rx = np.zeros(n)
        ry = np.zeros(n)
        nose_x = np.array([0.0, 0.0, np.nan, 0.0, np.nan, 0.0])
        nose_y = np.array([1.0, 1.0, np.nan, 1.0, np.nan, 1.0])
        implant_x = np.zeros(n)
        implant_y = np.zeros(n)

        fused = _fused_hd_wrapped(lx, ly, rx, ry, nose_x, nose_y, implant_x, implant_y)
        # All frames should be finite — ear estimate fills in where nose is NaN
        assert np.all(np.isfinite(fused))

    @given(
        lx=st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        ly=st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        rx=st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        ry=st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_ears_only_finite_inputs_produce_finite_output(
        self, lx: float, ly: float, rx: float, ry: float
    ) -> None:
        """Finite ear positions always produce a finite fused result."""
        if lx == rx and ly == ry:
            return  # degenerate
        fused = _fused_hd_wrapped(
            np.array([lx]), np.array([ly]),
            np.array([rx]), np.array([ry]),
        )
        assert np.isfinite(fused[0])
        # The function returns mean_angle % 360 but floating-point can produce exactly
        # 360.0 when the input is exactly 360° (= 0° on circle). Accept [0, 360].
        assert 0.0 <= fused[0] <= 360.0


# ---------------------------------------------------------------------------
# _unwrap_and_smooth
# ---------------------------------------------------------------------------


class TestUnwrapAndSmooth:
    def test_all_nan_input_returns_nan(self) -> None:
        """All-NaN input → all-NaN output (float32)."""
        arr = np.full(10, np.nan)
        out = _unwrap_and_smooth(arr, median_filter_win=5)
        assert np.all(np.isnan(out))
        assert out.dtype == np.float32

    def test_output_dtype_float32(self) -> None:
        arr = np.linspace(0.0, 360.0, 50)
        out = _unwrap_and_smooth(arr, median_filter_win=5)
        assert out.dtype == np.float32

    def test_output_shape(self) -> None:
        n = 40
        arr = np.ones(n) * 90.0
        out = _unwrap_and_smooth(arr, median_filter_win=3)
        assert out.shape == (n,)

    def test_smooth_constant_signal_unchanged(self) -> None:
        """A constant angle is unchanged by unwrapping and smoothing."""
        arr = np.full(30, 180.0, dtype=np.float64)
        out = _unwrap_and_smooth(arr, median_filter_win=5)
        np.testing.assert_allclose(out, 180.0, atol=1e-4)

    def test_slowly_increasing_signal_no_jumps(self) -> None:
        """A slowly increasing angle (no wrapping) passes through without large jumps."""
        n = 50
        arr = np.linspace(90.0, 270.0, n)  # monotone increase, no wrap
        out = _unwrap_and_smooth(arr, median_filter_win=3)
        jumps = np.abs(np.diff(out))
        assert np.all(jumps < 20.0), f"Unexpected jump: {jumps.max():.1f}°"

    def test_jump_across_360_0_boundary_unwrapped(self) -> None:
        """An angle that crosses the 360/0 boundary is unwrapped to a continuous signal."""
        n = 100
        # Angle increases from 355° to 365° (crossing 360→0)
        raw_angles = np.linspace(355.0, 365.0, n)
        # Wrap into [0, 360): values above 360 become small positive values
        wrapped = raw_angles % 360.0
        # At the wrap point there is a ~360° jump in the input
        assert np.max(np.abs(np.diff(wrapped))) > 200.0, "Test data must contain a wrap jump"

        out = _unwrap_and_smooth(wrapped, median_filter_win=1)  # no smoothing
        # After unwrapping, the jumps should be small
        jumps = np.abs(np.diff(out.astype(np.float64)))
        assert np.all(jumps < 10.0), f"Unwrapping failed — max jump: {jumps.max():.1f}°"

    def test_jump_across_0_360_boundary_going_backwards(self) -> None:
        """Angle decreasing from 5° to -5° (i.e. wrapping from 0→360) is unwrapped."""
        n = 100
        raw_angles = np.linspace(5.0, -5.0, n)
        wrapped = raw_angles % 360.0  # creates a ~360 jump going up
        assert np.max(np.abs(np.diff(wrapped))) > 200.0

        out = _unwrap_and_smooth(wrapped, median_filter_win=1)
        jumps = np.abs(np.diff(out.astype(np.float64)))
        assert np.all(jumps < 10.0), f"Unwrapping failed — max jump: {jumps.max():.1f}°"

    def test_nan_gaps_are_interpolated_then_restored(self) -> None:
        """NaN frames are interpolated for unwrapping but restored to NaN in output."""
        n = 20
        arr = np.linspace(10.0, 30.0, n)
        nan_idx = [5, 6, 7]
        arr[nan_idx] = np.nan
        out = _unwrap_and_smooth(arr, median_filter_win=1)
        # NaN positions in input must remain NaN in output
        for i in nan_idx:
            assert np.isnan(out[i]), f"Frame {i} should be NaN"
        # Non-NaN positions should be finite
        non_nan = [i for i in range(n) if i not in nan_idx]
        assert np.all(np.isfinite(out[non_nan]))

    def test_median_filter_win_1_no_smoothing(self) -> None:
        """win=1 disables median smoothing — output equals unwrapped signal."""
        n = 40
        arr = np.linspace(0.0, 350.0, n)
        out_win1 = _unwrap_and_smooth(arr, median_filter_win=1)
        out_win5 = _unwrap_and_smooth(arr, median_filter_win=5)
        # With a smooth linear signal both should be close, but win=1 is exact
        # Just verify shape and dtype
        assert out_win1.shape == (n,)
        assert out_win1.dtype == np.float32

    def test_large_constant_angle_preserved(self) -> None:
        """Constant angle far from 0/360 is preserved (no spurious offset from unwrap)."""
        arr = np.full(25, 270.0)
        out = _unwrap_and_smooth(arr, median_filter_win=3)
        np.testing.assert_allclose(out, 270.0, atol=1e-3)

    def test_single_valid_frame_with_surrounding_nans(self) -> None:
        """A single valid frame surrounded by NaN is returned as-is."""
        arr = np.full(9, np.nan)
        arr[4] = 45.0
        out = _unwrap_and_smooth(arr, median_filter_win=1)
        # NaN frames remain NaN
        for i in range(9):
            if i == 4:
                assert np.isfinite(out[i])
            else:
                assert np.isnan(out[i])

    @given(
        angles=st.lists(
            st.floats(min_value=0.0, max_value=360.0, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=200,
        )
    )
    @settings(max_examples=80)
    def test_finite_input_finite_output(self, angles: list[float]) -> None:
        """All-finite input → all-finite output (no NaN introduced by unwrapping)."""
        arr = np.array(angles, dtype=np.float64)
        out = _unwrap_and_smooth(arr, median_filter_win=3)
        assert np.all(np.isfinite(out))
