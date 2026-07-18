"""Tests for perspective correction comparison page helpers."""

from __future__ import annotations

import numpy as np
import pytest


class TestApplyPerspectiveInMazeCoords:
    def test_zero_height_no_change(self):
        from frontend.pages.perspective_compare_page import _apply_perspective_in_maze_coords

        x = np.array([1.0, 3.5, 6.0])
        y = np.array([0.5, 2.5, 4.5])
        xc, yc = _apply_perspective_in_maze_coords(x, y, (3.5, 2.5), 700, 0)
        np.testing.assert_array_equal(xc, x)
        np.testing.assert_array_equal(yc, y)

    def test_forward_moves_toward_center(self):
        from frontend.pages.perspective_compare_page import _apply_perspective_in_maze_coords

        cx, cy = 3.5, 2.5
        x = np.array([6.0])
        y = np.array([4.0])
        xc, yc = _apply_perspective_in_maze_coords(x, y, (cx, cy), 700, 30)
        # Should move toward center
        assert abs(xc[0] - cx) < abs(x[0] - cx)
        assert abs(yc[0] - cy) < abs(y[0] - cy)

    def test_inverse_moves_away_from_center(self):
        from frontend.pages.perspective_compare_page import _apply_perspective_in_maze_coords

        cx, cy = 3.5, 2.5
        x = np.array([6.0])
        y = np.array([4.0])
        xc, yc = _apply_perspective_in_maze_coords(x, y, (cx, cy), 700, 30, inverse=True)
        # Should move away from center
        assert abs(xc[0] - cx) > abs(x[0] - cx)
        assert abs(yc[0] - cy) > abs(y[0] - cy)

    def test_forward_then_inverse_roundtrip(self):
        from frontend.pages.perspective_compare_page import _apply_perspective_in_maze_coords

        x = np.array([1.0, 5.0, 3.5])
        y = np.array([0.5, 4.0, 2.5])
        cx, cy = 3.5, 2.5
        xf, yf = _apply_perspective_in_maze_coords(x, y, (cx, cy), 700, 40)
        xr, yr = _apply_perspective_in_maze_coords(xf, yf, (cx, cy), 700, 40, inverse=True)
        np.testing.assert_allclose(xr, x, atol=1e-10)
        np.testing.assert_allclose(yr, y, atol=1e-10)

    def test_center_unchanged(self):
        from frontend.pages.perspective_compare_page import _apply_perspective_in_maze_coords

        cx, cy = 4.0, 2.0
        x = np.array([cx])
        y = np.array([cy])
        xc, yc = _apply_perspective_in_maze_coords(x, y, (cx, cy), 700, 50)
        np.testing.assert_allclose(xc, [cx])
        np.testing.assert_allclose(yc, [cy])


class TestOutOfBoundsStats:
    def test_all_in_bounds(self):
        from frontend.pages.perspective_compare_page import _out_of_bounds_stats

        x = np.array([1.0, 3.5, 6.0])
        y = np.array([0.5, 2.5, 4.5])
        stats = _out_of_bounds_stats(x, y)
        assert stats["n_oob"] == 0
        assert stats["pct_oob"] == 0.0

    def test_some_out_of_bounds(self):
        from frontend.pages.perspective_compare_page import _out_of_bounds_stats

        x = np.array([1.0, 8.0, -1.0])  # 8.0 and -1.0 are OOB
        y = np.array([2.0, 2.0, 2.0])
        stats = _out_of_bounds_stats(x, y)
        assert stats["n_oob"] == 2
        assert stats["pct_oob"] == pytest.approx(200 / 3)

    def test_nan_excluded(self):
        from frontend.pages.perspective_compare_page import _out_of_bounds_stats

        x = np.array([1.0, np.nan, 3.0])
        y = np.array([2.0, 2.0, np.nan])
        stats = _out_of_bounds_stats(x, y)
        assert stats["n_valid"] == 1  # only first point is fully valid

    def test_empty(self):
        from frontend.pages.perspective_compare_page import _out_of_bounds_stats

        stats = _out_of_bounds_stats(np.array([]), np.array([]))
        assert stats["n_valid"] == 0


class TestEstimateCameraCenter:
    def test_returns_tuple(self):
        from frontend.pages.perspective_compare_page import _estimate_camera_center_maze

        result = _estimate_camera_center_maze(None, None)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_within_maze_bounds(self):
        from frontend.pages.perspective_compare_page import _estimate_camera_center_maze

        cx, cy = _estimate_camera_center_maze(None, None)
        assert 0 <= cx <= 7
        assert 0 <= cy <= 5
