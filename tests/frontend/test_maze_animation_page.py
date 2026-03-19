"""Tests for maze animation page helper functions."""

from __future__ import annotations

import numpy as np
import pytest


class TestSubsample:
    """Test the _subsample helper."""

    def test_step_1_identity(self):
        from frontend.pages.maze_animation_page import _subsample
        arr = np.arange(10)
        result = _subsample(arr, 1)
        np.testing.assert_array_equal(result, arr)

    def test_step_2(self):
        from frontend.pages.maze_animation_page import _subsample
        arr = np.arange(10)
        result = _subsample(arr, 2)
        np.testing.assert_array_equal(result, np.array([0, 2, 4, 6, 8]))

    def test_step_larger_than_array(self):
        from frontend.pages.maze_animation_page import _subsample
        arr = np.arange(5)
        result = _subsample(arr, 10)
        np.testing.assert_array_equal(result, np.array([0]))


class TestBuildAnimationFigure:
    """Test the animation figure builder."""

    def _make_test_data(self, n: int = 100):
        """Generate simple circular trajectory for testing."""
        t = np.linspace(0, 2 * np.pi, n)
        x = 3.5 + 2.0 * np.cos(t)
        y = 2.5 + 1.5 * np.sin(t)
        hd = np.degrees(t) % 360
        speed = np.ones(n) * 5.0
        light_on = np.ones(n, dtype=bool)
        light_on[n // 2:] = False
        frame_times = np.linspace(0, n / 9.6, n)
        return x, y, hd, speed, light_on, frame_times

    def test_returns_figure(self):
        from frontend.pages.maze_animation_page import _build_animation_figure
        x, y, hd, speed, light_on, ft = self._make_test_data()
        fig = _build_animation_figure(x, y, hd, speed, light_on, ft,
                                      trail_seconds=2.0, step=5, arrow_length=0.5)
        assert fig is not None
        assert hasattr(fig, "frames")
        assert len(fig.frames) > 0

    def test_frames_have_data(self):
        from frontend.pages.maze_animation_page import _build_animation_figure
        x, y, hd, speed, light_on, ft = self._make_test_data(50)
        fig = _build_animation_figure(x, y, hd, speed, light_on, ft,
                                      trail_seconds=1.0, step=1, arrow_length=0.5)
        for frame in fig.frames:
            assert len(frame.data) == 5  # walls, trail, head, arrow line, arrowhead

    def test_subsample_reduces_frames(self):
        from frontend.pages.maze_animation_page import _build_animation_figure
        x, y, hd, speed, light_on, ft = self._make_test_data(100)
        fig1 = _build_animation_figure(x, y, hd, speed, light_on, ft,
                                       trail_seconds=1.0, step=1, arrow_length=0.5)
        fig5 = _build_animation_figure(x, y, hd, speed, light_on, ft,
                                       trail_seconds=1.0, step=5, arrow_length=0.5)
        assert len(fig5.frames) < len(fig1.frames)

    def test_empty_input(self):
        from frontend.pages.maze_animation_page import _build_animation_figure
        fig = _build_animation_figure(
            np.array([]), np.array([]), np.array([]), np.array([]),
            np.array([], dtype=bool), np.array([]),
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )
        assert len(fig.frames) == 0

    def test_slider_present(self):
        from frontend.pages.maze_animation_page import _build_animation_figure
        x, y, hd, speed, light_on, ft = self._make_test_data(50)
        fig = _build_animation_figure(x, y, hd, speed, light_on, ft,
                                      trail_seconds=1.0, step=2, arrow_length=0.5)
        assert len(fig.layout.sliders) == 1
        assert len(fig.layout.updatemenus) == 1

    def test_play_pause_buttons(self):
        from frontend.pages.maze_animation_page import _build_animation_figure
        x, y, hd, speed, light_on, ft = self._make_test_data(30)
        fig = _build_animation_figure(x, y, hd, speed, light_on, ft,
                                      trail_seconds=1.0, step=1, arrow_length=0.5)
        buttons = fig.layout.updatemenus[0].buttons
        labels = [b.label for b in buttons]
        assert "Play" in labels
        assert "Pause" in labels


class TestMazeWalls:
    """Test maze wall polygon data."""

    def test_walls_closed(self):
        from frontend.pages.maze_animation_page import _MAZE_WALLS_X, _MAZE_WALLS_Y
        # Polygon should be closed (first == last)
        assert _MAZE_WALLS_X[0] == _MAZE_WALLS_X[-1]
        assert _MAZE_WALLS_Y[0] == _MAZE_WALLS_Y[-1]

    def test_walls_length(self):
        from frontend.pages.maze_animation_page import _MAZE_WALLS_X, _MAZE_WALLS_Y
        assert len(_MAZE_WALLS_X) == len(_MAZE_WALLS_Y)

    def test_walls_within_bounds(self):
        from frontend.pages.maze_animation_page import _MAZE_WALLS_X, _MAZE_WALLS_Y
        assert all(0 <= x <= 7 for x in _MAZE_WALLS_X)
        assert all(0 <= y <= 5 for y in _MAZE_WALLS_Y)


class TestDrawMaze:
    """Test the _draw_maze helper."""

    def test_adds_trace(self):
        import plotly.graph_objects as go
        from frontend.pages.maze_animation_page import _draw_maze
        fig = go.Figure()
        _draw_maze(fig)
        assert len(fig.data) == 1
        assert fig.data[0].mode == "lines"
