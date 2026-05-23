"""Tests for the maze canvas component and its data payload builder.

Tests cover:
  - Data serialization (_nan_to_none, _build_canvas_payload)
  - Data integrity (array lengths, value domains, monotonicity)
  - Maze wall constants (_MAZE_WALLS_X, _MAZE_WALLS_Y)
  - Bodypart name aliasing (SA aliases, legacy DLC names)
  - Skeleton connection validation
  - Edge cases (all-NaN, single frame, very long sessions, missing bodyparts)
  - HTML builder output (build_maze_canvas_html)

All tests use small synthetic arrays — no real data files.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_arrays():
    """Minimal valid data for 100 frames at ~9.6 Hz."""
    n = 100
    t = np.linspace(0, 2 * np.pi, n)
    return {
        "x_maze": 3.5 + 2.0 * np.cos(t),
        "y_maze": 2.5 + 1.5 * np.sin(t),
        "hd_deg": np.degrees(t) % 360,
        "speed": np.ones(n) * 5.0,
        "light_on": np.concatenate([np.ones(50, dtype=bool), np.zeros(50, dtype=bool)]),
        "frame_times": np.linspace(0.0, n / 9.6, n),
    }


@pytest.fixture
def full_bp_maze():
    """Per-bodypart maze coordinates for a 100-frame session."""
    n = 100
    rng = np.random.default_rng(42)
    bp_names = [
        "nose_tip", "left_ear", "right_ear", "head_midpoint",
        "neck", "mid_back", "mouse_center", "tail_base",
    ]
    bp_maze = {}
    for bp in bp_names:
        bp_maze[bp] = {
            "x": rng.uniform(0, 7, n),
            "y": rng.uniform(0, 5, n),
        }
    return bp_maze


# ---------------------------------------------------------------------------
# _nan_to_none tests
# ---------------------------------------------------------------------------


class TestNanToNone:
    """Test _nan_to_none JSON-safe conversion."""

    def test_finite_values_preserved(self):
        from frontend.pages.maze_animation_page import _nan_to_none
        arr = np.array([1.0, 2.5, -3.7, 0.0])
        result = _nan_to_none(arr)
        assert result == [1.0, 2.5, -3.7, 0.0]
        assert all(isinstance(v, float) for v in result)

    def test_nan_becomes_none(self):
        from frontend.pages.maze_animation_page import _nan_to_none
        arr = np.array([1.0, np.nan, 3.0])
        result = _nan_to_none(arr)
        assert result[0] == 1.0
        assert result[1] is None
        assert result[2] == 3.0

    def test_inf_becomes_none(self):
        from frontend.pages.maze_animation_page import _nan_to_none
        arr = np.array([1.0, np.inf, -np.inf])
        result = _nan_to_none(arr)
        assert result[0] == 1.0
        assert result[1] is None
        assert result[2] is None

    def test_all_nan_returns_all_none(self):
        from frontend.pages.maze_animation_page import _nan_to_none
        arr = np.array([np.nan, np.nan, np.nan])
        result = _nan_to_none(arr)
        assert all(v is None for v in result)

    def test_empty_array(self):
        from frontend.pages.maze_animation_page import _nan_to_none
        arr = np.array([])
        result = _nan_to_none(arr)
        assert result == []

    def test_output_is_json_serializable(self):
        from frontend.pages.maze_animation_page import _nan_to_none
        arr = np.array([1.0, np.nan, np.inf, -np.inf, 0.0, -1.5])
        result = _nan_to_none(arr)
        # Must not raise — JSON cannot encode NaN or Infinity
        serialized = json.dumps(result)
        decoded = json.loads(serialized)
        assert decoded[0] == 1.0
        assert decoded[1] is None  # NaN -> null
        assert decoded[2] is None  # inf -> null
        assert decoded[3] is None  # -inf -> null

    def test_float32_input(self):
        from frontend.pages.maze_animation_page import _nan_to_none
        arr = np.array([1.0, np.nan], dtype=np.float32)
        result = _nan_to_none(arr)
        assert result[0] == pytest.approx(1.0)
        assert result[1] is None

    @given(arrays(np.float64, st.integers(0, 200),
                  elements=st.floats(allow_nan=True, allow_infinity=True)))
    @settings(max_examples=50, deadline=5000)
    def test_output_always_json_serializable(self, arr):
        from frontend.pages.maze_animation_page import _nan_to_none
        result = _nan_to_none(arr)
        # Must not raise
        serialized = json.dumps(result)
        decoded = json.loads(serialized)
        assert len(decoded) == len(arr)
        for orig, converted in zip(arr, decoded):
            if np.isfinite(orig):
                assert converted == pytest.approx(float(orig), abs=1e-10)
            else:
                assert converted is None


# ---------------------------------------------------------------------------
# _build_canvas_payload tests — serialization
# ---------------------------------------------------------------------------


class TestBuildCanvasPayloadSerialization:
    """Test that _build_canvas_payload produces JSON-serializable payloads."""

    def test_basic_payload_structure(self, minimal_arrays):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=2.0, step=1, arrow_length=0.5,
        )
        required_keys = {
            "n_frames", "bp_names", "skeleton", "bp_colors",
            "maze_walls_x", "maze_walls_y", "bp_x", "bp_y",
            "hd_deg", "speed", "light_on", "frame_times",
            "arrow_length", "trail_seconds", "show_position", "show_skeleton",
        }
        assert required_keys.issubset(set(payload.keys()))

    def test_payload_is_json_serializable(self, minimal_arrays):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=2.0, step=1, arrow_length=0.5,
        )
        # Must not raise
        serialized = json.dumps(payload)
        assert isinstance(serialized, str)
        # Round-trip
        decoded = json.loads(serialized)
        assert decoded["n_frames"] == payload["n_frames"]

    def test_numpy_arrays_converted_to_lists(self, minimal_arrays):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=2.0, step=1, arrow_length=0.5,
        )
        assert isinstance(payload["hd_deg"], list)
        assert isinstance(payload["speed"], list)
        assert isinstance(payload["light_on"], list)
        assert isinstance(payload["frame_times"], list)
        assert isinstance(payload["maze_walls_x"], list)
        assert isinstance(payload["maze_walls_y"], list)

    def test_nan_in_position_handled(self):
        """NaN values in position arrays must become null (None) in the payload."""
        from frontend.pages.maze_animation_page import _build_canvas_payload
        n = 50
        x = np.ones(n) * 3.5
        x[10:15] = np.nan
        y = np.ones(n) * 2.5
        hd = np.zeros(n)
        speed = np.ones(n) * 5.0
        light = np.ones(n, dtype=bool)
        ft = np.linspace(0, 5, n)

        payload = _build_canvas_payload(
            x, y, hd, speed, light, ft,
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )
        # mouse_center fallback — bp_x["mouse_center"] should have None at NaN positions
        mc_x = payload["bp_x"]["mouse_center"]
        assert mc_x[10] is None
        assert mc_x[14] is None
        assert mc_x[0] == pytest.approx(3.5)

    def test_nan_in_hd_handled(self):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        n = 30
        hd = np.ones(n) * 90.0
        hd[5] = np.nan
        payload = _build_canvas_payload(
            np.ones(n) * 3.5, np.ones(n) * 2.5,
            hd, np.ones(n), np.ones(n, dtype=bool),
            np.linspace(0, 3, n),
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )
        assert payload["hd_deg"][5] is None
        assert payload["hd_deg"][0] == pytest.approx(90.0)

    def test_nan_in_speed_handled(self):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        n = 20
        speed = np.ones(n) * 10.0
        speed[3] = np.nan
        payload = _build_canvas_payload(
            np.ones(n) * 3.5, np.ones(n) * 2.5,
            np.zeros(n), speed, np.ones(n, dtype=bool),
            np.linspace(0, 2, n),
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )
        assert payload["speed"][3] is None


# ---------------------------------------------------------------------------
# _build_canvas_payload tests — bodypart aliasing
# ---------------------------------------------------------------------------


class TestBuildCanvasPayloadAliasing:
    """Test bodypart name mapping including SA and legacy aliases."""

    def test_standard_bodypart_names(self, minimal_arrays, full_bp_maze):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=2.0, step=1, arrow_length=0.5,
            bp_maze=full_bp_maze,
        )
        # All standard names should be in bp_names
        for bp in ["nose_tip", "left_ear", "right_ear", "head_midpoint",
                    "neck", "mid_back", "mouse_center", "tail_base"]:
            assert bp in payload["bp_names"]

    def test_sa_alias_nose_preserved(self, minimal_arrays):
        """SuperAnimal alias 'nose' should be included if present in bp_maze."""
        from frontend.pages.maze_animation_page import _build_canvas_payload
        n = len(minimal_arrays["x_maze"])
        bp_maze = {
            "nose": {"x": np.ones(n) * 3.0, "y": np.ones(n) * 2.0},
            "left_ear": {"x": np.ones(n) * 3.5, "y": np.ones(n) * 2.5},
            "right_ear": {"x": np.ones(n) * 4.0, "y": np.ones(n) * 2.5},
        }
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=1.0, step=1, arrow_length=0.5,
            bp_maze=bp_maze,
        )
        assert "nose" in payload["bp_names"]
        assert "nose" in payload["bp_x"]
        assert "nose" in payload["bp_y"]

    def test_legacy_alias_implant_base_rear_preserved(self, minimal_arrays):
        """Legacy DLC alias 'implant_base_rear' should be included if present."""
        from frontend.pages.maze_animation_page import _build_canvas_payload
        n = len(minimal_arrays["x_maze"])
        bp_maze = {
            "implant_base_rear": {"x": np.ones(n) * 3.5, "y": np.ones(n) * 2.5},
        }
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=1.0, step=1, arrow_length=0.5,
            bp_maze=bp_maze,
        )
        assert "implant_base_rear" in payload["bp_names"]
        assert "implant_base_rear" in payload["bp_x"]


# ---------------------------------------------------------------------------
# _build_canvas_payload tests — maze wall coordinates
# ---------------------------------------------------------------------------


class TestMazeWallCoordinates:
    """Test that maze wall constants are correct for the q-rose (Rosenberg) maze."""

    def test_maze_walls_included_in_payload(self, minimal_arrays):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )
        assert "maze_walls_x" in payload
        assert "maze_walls_y" in payload
        assert len(payload["maze_walls_x"]) == 33
        assert len(payload["maze_walls_y"]) == 33

    def test_maze_walls_same_length(self):
        from frontend.pages.maze_animation_page import _MAZE_WALLS_X, _MAZE_WALLS_Y
        assert len(_MAZE_WALLS_X) == len(_MAZE_WALLS_Y)

    def test_maze_walls_33_points(self):
        from frontend.pages.maze_animation_page import _MAZE_WALLS_X, _MAZE_WALLS_Y
        assert len(_MAZE_WALLS_X) == 33

    def test_maze_polygon_closed(self):
        """First point must equal last point for a closed polygon."""
        from frontend.pages.maze_animation_page import _MAZE_WALLS_X, _MAZE_WALLS_Y
        assert _MAZE_WALLS_X[0] == _MAZE_WALLS_X[-1]
        assert _MAZE_WALLS_Y[0] == _MAZE_WALLS_Y[-1]

    def test_maze_walls_within_grid_bounds(self):
        """All points must lie within the 7x5 maze grid."""
        from frontend.pages.maze_animation_page import _MAZE_WALLS_X, _MAZE_WALLS_Y
        assert all(0 <= x <= 7 for x in _MAZE_WALLS_X)
        assert all(0 <= y <= 5 for y in _MAZE_WALLS_Y)

    def test_maze_walls_integer_coordinates(self):
        """Maze walls are on integer cell boundaries."""
        from frontend.pages.maze_animation_page import _MAZE_WALLS_X, _MAZE_WALLS_Y
        assert all(isinstance(x, int) for x in _MAZE_WALLS_X)
        assert all(isinstance(y, int) for y in _MAZE_WALLS_Y)

    def test_maze_walls_importable_from_animation_page(self):
        """perspective_compare_page imports these — they must be accessible."""
        from frontend.pages.maze_animation_page import _MAZE_WALLS_X, _MAZE_WALLS_Y
        assert isinstance(_MAZE_WALLS_X, list)
        assert isinstance(_MAZE_WALLS_Y, list)

    def test_perspective_compare_page_imports_walls(self):
        """Verify perspective_compare_page can import the maze wall constants."""
        # This is a cross-module dependency test — if the import path breaks,
        # perspective_compare_page is also broken.
        from frontend.pages.perspective_compare_page import (
            _MAZE_WALLS_X as imported_x,
        )
        from frontend.pages.maze_animation_page import (
            _MAZE_WALLS_X as source_x,
        )
        # They must be the exact same object (imported by reference)
        assert imported_x is source_x


# ---------------------------------------------------------------------------
# _build_canvas_payload tests — skeleton connections
# ---------------------------------------------------------------------------


class TestSkeletonConnections:
    """Test that skeleton connections reference valid bodypart names."""

    def test_skeleton_references_valid_bp_names(self, minimal_arrays, full_bp_maze):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=1.0, step=1, arrow_length=0.5,
            bp_maze=full_bp_maze,
        )
        bp_set = set(payload["bp_names"])
        for bp1, bp2 in payload["skeleton"]:
            assert bp1 in bp_set, f"skeleton references unknown bodypart '{bp1}'"
            assert bp2 in bp_set, f"skeleton references unknown bodypart '{bp2}'"

    def test_skeleton_filtered_to_present_bodyparts(self, minimal_arrays):
        """If only some bodyparts are present, skeleton should only include
        connections between those bodyparts."""
        from frontend.pages.maze_animation_page import _build_canvas_payload
        n = len(minimal_arrays["x_maze"])
        bp_maze = {
            "nose_tip": {"x": np.ones(n), "y": np.ones(n)},
            "head_midpoint": {"x": np.ones(n), "y": np.ones(n)},
            # neck is missing — so (head_midpoint, neck) should be excluded
        }
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=1.0, step=1, arrow_length=0.5,
            bp_maze=bp_maze,
        )
        for bp1, bp2 in payload["skeleton"]:
            assert bp1 in payload["bp_names"]
            assert bp2 in payload["bp_names"]

    def test_no_bodyparts_gives_no_skeleton(self, minimal_arrays):
        """With no bp_maze, skeleton should be empty (or only mouse_center)."""
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=1.0, step=1, arrow_length=0.5,
            bp_maze=None,
        )
        # Only mouse_center is synthesized as fallback — no connections
        assert len(payload["skeleton"]) == 0

    def test_full_skeleton_structure(self):
        """The raw _SKELETON constant should have 10 connections."""
        from frontend.pages.maze_animation_page import _SKELETON
        assert len(_SKELETON) == 10
        for pair in _SKELETON:
            assert len(pair) == 2
            assert isinstance(pair[0], str)
            assert isinstance(pair[1], str)


# ---------------------------------------------------------------------------
# _build_canvas_payload tests — empty/missing bodypart data
# ---------------------------------------------------------------------------


class TestBuildCanvasPayloadMissingBodyparts:
    """Test that missing or empty bodypart data produces valid payload."""

    def test_no_bp_maze_fallback_to_centroid(self, minimal_arrays):
        """Without bp_maze, payload should use mouse_center from centroid."""
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=1.0, step=1, arrow_length=0.5,
            bp_maze=None,
        )
        assert "mouse_center" in payload["bp_names"]
        assert "mouse_center" in payload["bp_x"]
        assert "mouse_center" in payload["bp_y"]
        assert len(payload["bp_x"]["mouse_center"]) == payload["n_frames"]

    def test_no_head_midpoint_still_valid(self, minimal_arrays):
        """Missing head_midpoint should produce a valid payload."""
        from frontend.pages.maze_animation_page import _build_canvas_payload
        n = len(minimal_arrays["x_maze"])
        bp_maze = {
            "nose_tip": {"x": np.ones(n), "y": np.ones(n)},
            "left_ear": {"x": np.ones(n), "y": np.ones(n)},
            "right_ear": {"x": np.ones(n), "y": np.ones(n)},
        }
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=1.0, step=1, arrow_length=0.5,
            bp_maze=bp_maze,
        )
        assert "head_midpoint" not in payload["bp_names"]
        # Payload must still be valid JSON
        json.dumps(payload)

    def test_empty_bp_maze_dict_treated_as_none(self, minimal_arrays):
        """An empty dict should behave like None — fall back to centroid."""
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=1.0, step=1, arrow_length=0.5,
            bp_maze={},
        )
        assert "mouse_center" in payload["bp_names"]


# ---------------------------------------------------------------------------
# _build_canvas_payload tests — data integrity
# ---------------------------------------------------------------------------


class TestBuildCanvasPayloadIntegrity:
    """Test that all arrays in the payload are consistent."""

    def test_all_arrays_same_length(self, minimal_arrays, full_bp_maze):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=2.0, step=1, arrow_length=0.5,
            bp_maze=full_bp_maze,
        )
        n = payload["n_frames"]
        assert len(payload["hd_deg"]) == n
        assert len(payload["speed"]) == n
        assert len(payload["light_on"]) == n
        assert len(payload["frame_times"]) == n
        for bp_name in payload["bp_names"]:
            assert len(payload["bp_x"][bp_name]) == n, f"bp_x[{bp_name}] length mismatch"
            assert len(payload["bp_y"][bp_name]) == n, f"bp_y[{bp_name}] length mismatch"

    def test_light_on_values_are_0_or_1(self, minimal_arrays):
        """light_on must be integers 0 or 1 — not Python bools (JSON compat)."""
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )
        for v in payload["light_on"]:
            assert v in (0, 1), f"light_on value {v} is not 0 or 1"
            assert isinstance(v, int), f"light_on value {v} is {type(v)}, expected int"

    def test_frame_times_monotonically_increasing(self, minimal_arrays):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )
        ft = payload["frame_times"]
        for i in range(1, len(ft)):
            assert ft[i] >= ft[i - 1], f"frame_times not monotonic at index {i}"

    def test_bp_colors_has_entry_for_every_bodypart(self, minimal_arrays, full_bp_maze):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=1.0, step=1, arrow_length=0.5,
            bp_maze=full_bp_maze,
        )
        for bp_name in payload["bp_names"]:
            assert bp_name in payload["bp_colors"], (
                f"bp_colors missing entry for bodypart '{bp_name}'"
            )
            # Colour should be a hex string
            color = payload["bp_colors"][bp_name]
            assert isinstance(color, str)
            assert color.startswith("#"), f"colour '{color}' for '{bp_name}' is not hex"

    def test_n_frames_matches_actual_data(self, minimal_arrays):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )
        n = payload["n_frames"]
        assert n == len(minimal_arrays["x_maze"])

    def test_subsample_reduces_n_frames(self, minimal_arrays):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload_s1 = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )
        payload_s5 = _build_canvas_payload(
            **minimal_arrays,
            trail_seconds=1.0, step=5, arrow_length=0.5,
        )
        assert payload_s5["n_frames"] < payload_s1["n_frames"]
        # step=5 on 100 frames: indices [0,5,10,...,95] = 20 frames
        assert payload_s5["n_frames"] == 20


# ---------------------------------------------------------------------------
# _build_canvas_payload tests — payload size
# ---------------------------------------------------------------------------


class TestPayloadSize:
    """Test that payload size is reasonable for typical sessions."""

    def test_typical_session_under_10mb(self):
        """A 17k-frame session (typical for this experiment) should produce
        a payload under 10 MB when serialized to JSON."""
        from frontend.pages.maze_animation_page import _build_canvas_payload
        n = 17000
        rng = np.random.default_rng(99)
        x = rng.uniform(0, 7, n)
        y = rng.uniform(0, 5, n)
        hd = rng.uniform(0, 360, n)
        speed = rng.uniform(0, 30, n)
        light = np.ones(n, dtype=bool)
        light[::2] = False
        ft = np.linspace(0, n / 9.6, n)

        bp_names = [
            "nose_tip", "left_ear", "right_ear", "head_midpoint",
            "neck", "mid_back", "mouse_center", "tail_base",
        ]
        bp_maze = {}
        for bp in bp_names:
            bp_maze[bp] = {"x": rng.uniform(0, 7, n), "y": rng.uniform(0, 5, n)}

        payload = _build_canvas_payload(
            x, y, hd, speed, light, ft,
            trail_seconds=5.0, step=1, arrow_length=0.5,
            bp_maze=bp_maze,
        )
        serialized = json.dumps(payload, separators=(",", ":"))
        size_mb = len(serialized) / (1024 * 1024)
        assert size_mb < 10, f"payload is {size_mb:.1f} MB, exceeds 10 MB limit"

    def test_subsampled_session_much_smaller(self):
        """Subsampling by 5 should reduce payload size roughly 5x."""
        from frontend.pages.maze_animation_page import _build_canvas_payload
        n = 5000
        rng = np.random.default_rng(7)
        x = rng.uniform(0, 7, n)
        y = rng.uniform(0, 5, n)
        hd = rng.uniform(0, 360, n)
        speed = rng.uniform(0, 30, n)
        light = np.ones(n, dtype=bool)
        ft = np.linspace(0, n / 9.6, n)

        payload_s1 = _build_canvas_payload(
            x, y, hd, speed, light, ft,
            trail_seconds=5.0, step=1, arrow_length=0.5,
        )
        payload_s5 = _build_canvas_payload(
            x, y, hd, speed, light, ft,
            trail_seconds=5.0, step=5, arrow_length=0.5,
        )
        size_s1 = len(json.dumps(payload_s1, separators=(",", ":")))
        size_s5 = len(json.dumps(payload_s5, separators=(",", ":")))
        ratio = size_s1 / size_s5
        assert ratio > 3.0, f"subsampling ratio {ratio:.1f} — expected > 3x reduction"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases: all-NaN, single frame, large sessions, missing data."""

    def test_all_nan_position_data(self):
        """All-NaN position should produce a valid payload with all-None bp data."""
        from frontend.pages.maze_animation_page import _build_canvas_payload
        n = 20
        payload = _build_canvas_payload(
            np.full(n, np.nan), np.full(n, np.nan),
            np.zeros(n), np.ones(n) * 5.0,
            np.ones(n, dtype=bool), np.linspace(0, 2, n),
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )
        assert payload["n_frames"] == n
        # All positions should be None
        mc_x = payload["bp_x"]["mouse_center"]
        assert all(v is None for v in mc_x)
        # Must still be JSON-serializable
        json.dumps(payload)

    def test_single_frame(self):
        """A single-frame session should produce a valid payload."""
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            np.array([3.5]), np.array([2.5]),
            np.array([180.0]), np.array([5.0]),
            np.array([True]), np.array([0.0]),
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )
        assert payload["n_frames"] == 1
        assert len(payload["hd_deg"]) == 1
        assert len(payload["frame_times"]) == 1
        json.dumps(payload)

    def test_very_long_session(self):
        """Sessions with >50k frames should still produce a valid payload."""
        from frontend.pages.maze_animation_page import _build_canvas_payload
        n = 55000
        rng = np.random.default_rng(123)
        payload = _build_canvas_payload(
            rng.uniform(0, 7, n), rng.uniform(0, 5, n),
            rng.uniform(0, 360, n), rng.uniform(0, 30, n),
            rng.choice([True, False], n), np.linspace(0, n / 9.6, n),
            trail_seconds=5.0, step=5, arrow_length=0.5,
        )
        # With step=5, n_frames should be 11000
        assert payload["n_frames"] == 11000
        assert len(payload["frame_times"]) == 11000
        json.dumps(payload)

    def test_zero_length_time_range(self):
        """An empty array (zero frames) should produce a valid empty payload."""
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            np.array([]), np.array([]),
            np.array([]), np.array([]),
            np.array([], dtype=bool), np.array([]),
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )
        assert payload["n_frames"] == 0
        assert len(payload["hd_deg"]) == 0
        json.dumps(payload)

    def test_all_light_on(self):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        n = 30
        payload = _build_canvas_payload(
            np.ones(n) * 3.5, np.ones(n) * 2.5,
            np.zeros(n), np.ones(n) * 5.0,
            np.ones(n, dtype=bool), np.linspace(0, 3, n),
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )
        assert all(v == 1 for v in payload["light_on"])

    def test_all_light_off(self):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        n = 30
        payload = _build_canvas_payload(
            np.ones(n) * 3.5, np.ones(n) * 2.5,
            np.zeros(n), np.ones(n) * 5.0,
            np.zeros(n, dtype=bool), np.linspace(0, 3, n),
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )
        assert all(v == 0 for v in payload["light_on"])


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------


class TestBuildCanvasPayloadHypothesis:
    """Property-based tests for _build_canvas_payload with random inputs."""

    @given(
        n=st.integers(min_value=2, max_value=500),
        step=st.integers(min_value=1, max_value=10),
        trail_s=st.floats(min_value=0.1, max_value=30.0),
        arrow_len=st.floats(min_value=0.1, max_value=2.0),
    )
    @settings(max_examples=30, deadline=10000)
    def test_payload_always_consistent(self, n, step, trail_s, arrow_len):
        """For any valid inputs, the payload must be consistent and serializable."""
        from frontend.pages.maze_animation_page import _build_canvas_payload
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 7, n)
        y = rng.uniform(0, 5, n)
        hd = rng.uniform(0, 360, n)
        speed = rng.uniform(0, 30, n)
        light = rng.choice([True, False], n)
        ft = np.sort(rng.uniform(0, n / 9.6, n))  # monotonic

        payload = _build_canvas_payload(
            x, y, hd, speed, light, ft,
            trail_seconds=trail_s, step=step, arrow_length=arrow_len,
        )

        nf = payload["n_frames"]
        assert nf > 0
        assert len(payload["hd_deg"]) == nf
        assert len(payload["speed"]) == nf
        assert len(payload["light_on"]) == nf
        assert len(payload["frame_times"]) == nf

        # All light_on values are 0 or 1
        for v in payload["light_on"]:
            assert v in (0, 1)

        # Must be JSON-serializable
        json.dumps(payload)

    @given(
        arrays(np.float64, st.integers(5, 100),
               elements=st.one_of(
                   st.floats(min_value=-1, max_value=8,
                             allow_nan=False, allow_infinity=False),
                   st.just(float("nan")),
               ))
    )
    @settings(max_examples=30, deadline=10000)
    def test_nan_positions_never_break_serialization(self, x_arr):
        """Any mix of NaN/finite positions should produce valid JSON."""
        from frontend.pages.maze_animation_page import _build_canvas_payload
        n = len(x_arr)
        y_arr = np.ones(n) * 2.5
        payload = _build_canvas_payload(
            x_arr, y_arr,
            np.zeros(n), np.ones(n),
            np.ones(n, dtype=bool), np.linspace(0, n / 9.6, n),
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )
        serialized = json.dumps(payload)
        decoded = json.loads(serialized)
        assert decoded["n_frames"] == n


# ---------------------------------------------------------------------------
# build_maze_canvas_html tests (HTML output)
# ---------------------------------------------------------------------------


class TestBuildMazeCanvasHtml:
    """Test the HTML builder from the maze_canvas component."""

    def _make_payload(self, n=50):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        rng = np.random.default_rng(0)
        return _build_canvas_payload(
            rng.uniform(0, 7, n), rng.uniform(0, 5, n),
            rng.uniform(0, 360, n), rng.uniform(0, 30, n),
            np.ones(n, dtype=bool), np.linspace(0, n / 9.6, n),
            trail_seconds=2.0, step=1, arrow_length=0.5,
        )

    def test_returns_string(self):
        from frontend.components.maze_canvas import build_maze_canvas_html
        payload = self._make_payload()
        html = build_maze_canvas_html(payload)
        assert isinstance(html, str)

    def test_contains_canvas_element(self):
        from frontend.components.maze_canvas import build_maze_canvas_html
        payload = self._make_payload()
        html = build_maze_canvas_html(payload)
        assert "<canvas" in html

    def test_contains_play_button(self):
        from frontend.components.maze_canvas import build_maze_canvas_html
        payload = self._make_payload()
        html = build_maze_canvas_html(payload)
        assert "Play" in html

    def test_contains_scrubber(self):
        from frontend.components.maze_canvas import build_maze_canvas_html
        payload = self._make_payload()
        html = build_maze_canvas_html(payload)
        assert "scrubber" in html

    def test_contains_speed_selector(self):
        from frontend.components.maze_canvas import build_maze_canvas_html
        payload = self._make_payload()
        html = build_maze_canvas_html(payload)
        assert "sel-speed" in html
        assert "0.25x" in html
        assert "1x" in html

    def test_contains_data_json(self):
        from frontend.components.maze_canvas import build_maze_canvas_html
        payload = self._make_payload()
        html = build_maze_canvas_html(payload)
        assert "__MAZE_DATA_" in html

    def test_contains_readout_labels(self):
        from frontend.components.maze_canvas import build_maze_canvas_html
        payload = self._make_payload()
        html = build_maze_canvas_html(payload)
        for label in ["Time:", "Frame:", "HD:", "Speed:", "Light:"]:
            assert label in html, f"Missing readout label '{label}'"

    def test_unique_ids_on_repeated_calls(self):
        """Each call should produce unique DOM element IDs."""
        from frontend.components.maze_canvas import build_maze_canvas_html
        payload = self._make_payload()
        html1 = build_maze_canvas_html(payload)
        html2 = build_maze_canvas_html(payload)
        # Extract canvas element IDs
        import re
        ids1 = re.findall(r'id="maze-canvas-([a-f0-9]+)"', html1)
        ids2 = re.findall(r'id="maze-canvas-([a-f0-9]+)"', html2)
        assert len(ids1) == 1
        assert len(ids2) == 1
        assert ids1[0] != ids2[0], "DOM IDs should be unique across calls"

    def test_empty_payload_produces_valid_html(self):
        """An empty (0-frame) payload should still produce valid HTML."""
        from frontend.components.maze_canvas import build_maze_canvas_html
        from frontend.pages.maze_animation_page import _build_canvas_payload
        payload = _build_canvas_payload(
            np.array([]), np.array([]),
            np.array([]), np.array([]),
            np.array([], dtype=bool), np.array([]),
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )
        html = build_maze_canvas_html(payload)
        assert isinstance(html, str)
        assert "<canvas" in html

    def test_html_contains_css(self):
        from frontend.components.maze_canvas import build_maze_canvas_html
        payload = self._make_payload()
        html = build_maze_canvas_html(payload)
        assert "<style>" in html
        assert "maze-container-" in html


# ---------------------------------------------------------------------------
# render_maze_canvas tests (Streamlit integration)
# ---------------------------------------------------------------------------


class TestRenderMazeCanvas:
    """Test that render_maze_canvas calls the correct Streamlit API."""

    def _make_payload(self, n=20):
        from frontend.pages.maze_animation_page import _build_canvas_payload
        return _build_canvas_payload(
            np.ones(n) * 3.5, np.ones(n) * 2.5,
            np.zeros(n), np.ones(n),
            np.ones(n, dtype=bool), np.linspace(0, 2, n),
            trail_seconds=1.0, step=1, arrow_length=0.5,
        )

    def test_render_uses_components_html(self):
        """render_maze_canvas should call st.components.v1.html."""
        from unittest.mock import MagicMock, patch
        from frontend.components.maze_canvas import render_maze_canvas

        payload = self._make_payload()
        mock_components_v1 = MagicMock()

        with patch.dict("sys.modules", {
            "streamlit.components": MagicMock(),
            "streamlit.components.v1": mock_components_v1,
        }):
            render_maze_canvas(payload, height=780)

        mock_components_v1.html.assert_called_once()
        html_arg = mock_components_v1.html.call_args[0][0]
        assert "<canvas" in html_arg


# ---------------------------------------------------------------------------
# _BP_COLORS tests
# ---------------------------------------------------------------------------


class TestBPColors:
    """Test the bodypart colour palette."""

    def test_all_standard_bodyparts_have_colors(self):
        from frontend.pages.maze_animation_page import _BP_COLORS
        standard_bps = [
            "nose_tip", "left_ear", "right_ear", "head_midpoint",
            "neck", "mid_back", "mouse_center", "tail_base",
        ]
        for bp in standard_bps:
            assert bp in _BP_COLORS, f"standard bodypart '{bp}' missing from _BP_COLORS"

    def test_aliases_have_matching_colors(self):
        """SA and legacy aliases should map to the same colour as the canonical name."""
        from frontend.pages.maze_animation_page import _BP_COLORS
        assert _BP_COLORS["nose"] == _BP_COLORS["nose_tip"]
        assert _BP_COLORS["implant_base_rear"] == _BP_COLORS["head_midpoint"]

    def test_colors_are_hex_strings(self):
        from frontend.pages.maze_animation_page import _BP_COLORS
        for bp, color in _BP_COLORS.items():
            assert isinstance(color, str)
            assert color.startswith("#"), f"colour for '{bp}' is '{color}', not hex"
            assert len(color) == 7, f"colour for '{bp}' is '{color}', expected #RRGGBB"


# ---------------------------------------------------------------------------
# _draw_maze tests (retained from old tests, verified with new code)
# ---------------------------------------------------------------------------


class TestDrawMaze:
    """Test the _draw_maze helper."""

    def test_adds_one_trace(self):
        import plotly.graph_objects as go
        from frontend.pages.maze_animation_page import _draw_maze
        fig = go.Figure()
        _draw_maze(fig)
        assert len(fig.data) == 1
        assert fig.data[0].mode == "lines"

    def test_trace_has_correct_wall_data(self):
        import plotly.graph_objects as go
        from frontend.pages.maze_animation_page import _draw_maze, _MAZE_WALLS_X, _MAZE_WALLS_Y
        fig = go.Figure()
        _draw_maze(fig)
        np.testing.assert_array_equal(fig.data[0].x, _MAZE_WALLS_X)
        np.testing.assert_array_equal(fig.data[0].y, _MAZE_WALLS_Y)

    def test_walls_hidden_from_legend(self):
        import plotly.graph_objects as go
        from frontend.pages.maze_animation_page import _draw_maze
        fig = go.Figure()
        _draw_maze(fig)
        assert fig.data[0].showlegend is False


# ---------------------------------------------------------------------------
# _subsample tests (retained from old tests)
# ---------------------------------------------------------------------------


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

    def test_preserves_dtype(self):
        from frontend.pages.maze_animation_page import _subsample
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _subsample(arr, 1)
        assert result.dtype == np.float32

    @given(
        arr=arrays(np.float64, st.integers(1, 200),
                   elements=st.floats(allow_nan=False, allow_infinity=False,
                                      min_value=-1e6, max_value=1e6)),
        step=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=50, deadline=5000)
    def test_subsample_length(self, arr, step):
        from frontend.pages.maze_animation_page import _subsample
        result = _subsample(arr, step)
        expected_len = len(arr[::step])
        assert len(result) == expected_len
