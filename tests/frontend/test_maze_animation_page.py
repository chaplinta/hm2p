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


class TestBuildCanvasPayload:
    """Test the canvas payload builder (replaced _build_animation_figure)."""

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

    def test_returns_dict(self):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        x, y, hd, speed, light_on, ft = self._make_test_data()
        payload = _build_canvas_payload(x, y, hd, speed, light_on, ft,
                                        trail_seconds=2.0, step=5, arrow_length=0.5)
        assert isinstance(payload, dict)
        assert payload["n_frames"] > 0

    def test_payload_has_required_keys(self):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        x, y, hd, speed, light_on, ft = self._make_test_data(50)
        payload = _build_canvas_payload(x, y, hd, speed, light_on, ft,
                                        trail_seconds=1.0, step=1, arrow_length=0.5)
        required_keys = {
            "n_frames", "bp_names", "skeleton", "bp_colors",
            "maze_walls_x", "maze_walls_y", "bp_x", "bp_y",
            "hd_deg", "speed", "light_on", "frame_times",
            "arrow_length", "trail_seconds", "show_position", "show_skeleton",
        }
        assert required_keys <= set(payload.keys())

    def test_subsample_reduces_frames(self):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        x, y, hd, speed, light_on, ft = self._make_test_data(100)
        p1 = _build_canvas_payload(x, y, hd, speed, light_on, ft,
                                   trail_seconds=1.0, step=1, arrow_length=0.5)
        p5 = _build_canvas_payload(x, y, hd, speed, light_on, ft,
                                   trail_seconds=1.0, step=5, arrow_length=0.5)
        assert p5["n_frames"] < p1["n_frames"]

    def test_empty_input(self):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            np.array([]), np.array([]), np.array([]), np.array([]),
            np.array([], dtype=bool), np.array([]),
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )
        assert payload["n_frames"] == 0

    def test_light_on_encoded_as_int(self):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        x, y, hd, speed, light_on, ft = self._make_test_data(50)
        payload = _build_canvas_payload(x, y, hd, speed, light_on, ft,
                                        trail_seconds=1.0, step=2, arrow_length=0.5)
        # light_on should be encoded as list of 0/1 ints
        assert all(v in (0, 1) for v in payload["light_on"])

    def test_show_position_and_skeleton_flags(self):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        x, y, hd, speed, light_on, ft = self._make_test_data(30)
        payload = _build_canvas_payload(x, y, hd, speed, light_on, ft,
                                        trail_seconds=1.0, step=1, arrow_length=0.5,
                                        show_position=False, show_skeleton=False)
        assert payload["show_position"] is False
        assert payload["show_skeleton"] is False

    def test_arrow_length_preserved(self):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        x, y, hd, speed, light_on, ft = self._make_test_data(20)
        payload = _build_canvas_payload(x, y, hd, speed, light_on, ft,
                                        trail_seconds=1.0, step=1, arrow_length=0.75)
        assert payload["arrow_length"] == 0.75

    def test_trail_seconds_preserved(self):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        x, y, hd, speed, light_on, ft = self._make_test_data(20)
        payload = _build_canvas_payload(x, y, hd, speed, light_on, ft,
                                        trail_seconds=5.0, step=1, arrow_length=0.5)
        assert payload["trail_seconds"] == 5.0


class TestCanvasPayloadLightState:
    """Test that light/dark state is encoded correctly in payload."""

    def _make_test_data(self, n: int = 20, all_dark: bool = False):
        t = np.linspace(0, 2 * np.pi, n)
        x = 3.5 + 2.0 * np.cos(t)
        y = 2.5 + 1.5 * np.sin(t)
        hd = np.degrees(t) % 360
        speed = np.ones(n) * 5.0
        light_on = np.zeros(n, dtype=bool) if all_dark else np.ones(n, dtype=bool)
        frame_times = np.linspace(0, n / 9.6, n)
        return x, y, hd, speed, light_on, frame_times

    def test_light_on_all_ones(self):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        x, y, hd, speed, light_on, ft = self._make_test_data(all_dark=False)
        payload = _build_canvas_payload(x, y, hd, speed, light_on, ft,
                                        trail_seconds=1.0, step=1, arrow_length=0.5)
        assert all(v == 1 for v in payload["light_on"])

    def test_dark_all_zeros(self):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        x, y, hd, speed, light_on, ft = self._make_test_data(all_dark=True)
        payload = _build_canvas_payload(x, y, hd, speed, light_on, ft,
                                        trail_seconds=1.0, step=1, arrow_length=0.5)
        assert all(v == 0 for v in payload["light_on"])


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
