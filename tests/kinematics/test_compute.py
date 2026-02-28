"""Tests for kinematics/compute.py — HD, position, speed, light_on, bad_behav."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.kinematics.compute import (
    MAZE_POLYGON_COORDS,
    compute_bad_behav_mask,
    compute_light_on,
)

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
    shapely_geom = pytest.importorskip("shapely.geometry")
    shapely_val = pytest.importorskip("shapely.validation")
    poly = shapely_geom.Polygon(MAZE_POLYGON_COORDS)
    valid_poly = shapely_val.make_valid(poly)
    bounds = valid_poly.bounds
    assert bounds[0] >= 0, f"x_min={bounds[0]} < 0"
    assert bounds[1] >= 0, f"y_min={bounds[1]} < 0"
    assert bounds[2] <= 7, f"x_max={bounds[2]} > 7"
    assert bounds[3] <= 5, f"y_max={bounds[3]} > 5"


def test_maze_polygon_interior_point() -> None:
    """A known interior point (centre of maze) is inside the valid polygon."""
    shapely_geom = pytest.importorskip("shapely.geometry")
    shapely_val = pytest.importorskip("shapely.validation")
    poly = shapely_geom.Polygon(MAZE_POLYGON_COORDS)
    valid_poly = shapely_val.make_valid(poly)
    # (3.5, 2.5) is the approximate centre of the 7×5 maze — inside a corridor
    centre = shapely_geom.Point(3.5, 2.5)
    assert valid_poly.contains(centre) or valid_poly.touches(centre)
