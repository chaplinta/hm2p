"""Tests for maze position discretization."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.maze.discretize import (
    cell_sequence,
    discretize_position,
    discretize_position_fast,
    node_sequence,
)
from hm2p.maze.topology import build_rose_maze


@pytest.fixture
def maze():
    return build_rose_maze()


class TestDiscretizePosition:
    """Test continuous → discrete position mapping."""

    def test_cell_center_maps_to_correct_cell(self, maze):
        """Position at cell center should map to that cell."""
        x = np.array([0.5, 1.5, 6.5])
        y = np.array([0.5, 0.5, 4.5])
        result = discretize_position(x, y, maze)
        for i, (xi, yi) in enumerate(zip(x, y)):
            cell = (int(xi - 0.5), int(yi - 0.5))
            assert result[i] == maze.cell_to_idx[cell]

    def test_nan_maps_to_minus_one(self, maze):
        x = np.array([np.nan, 0.5])
        y = np.array([0.5, np.nan])
        result = discretize_position(x, y, maze)
        assert result[0] == -1
        assert result[1] == -1

    def test_inaccessible_position_maps_to_nearest(self, maze):
        """Position in a wall cell should map to nearest accessible cell."""
        # (3, 0) is a wall cell — center at (3.5, 0.5)
        x = np.array([3.5])
        y = np.array([0.5])
        result = discretize_position(x, y, maze)
        assert result[0] >= 0  # Should map to something valid
        # Nearest accessible should be (2,0) or (4,0)
        mapped_cell = maze.cell_list[result[0]]
        assert mapped_cell in [(2, 0), (4, 0)]

    def test_all_valid_positions_produce_valid_indices(self, maze):
        """Random valid positions should all produce valid indices."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 7, 100)
        y = rng.uniform(0, 5, 100)
        result = discretize_position(x, y, maze)
        assert np.all(result >= 0)
        assert np.all(result < maze.n_cells)

    def test_empty_input(self, maze):
        x = np.array([], dtype=float)
        y = np.array([], dtype=float)
        result = discretize_position(x, y, maze)
        assert len(result) == 0

    def test_all_nan(self, maze):
        x = np.full(5, np.nan)
        y = np.full(5, np.nan)
        result = discretize_position(x, y, maze)
        np.testing.assert_array_equal(result, -1)


class TestDiscretizePositionFast:
    """Test vectorized discretization."""

    def test_matches_scalar_version(self, maze):
        """Fast version should produce same results as scalar version."""
        rng = np.random.default_rng(123)
        x = rng.uniform(0, 7, 50)
        y = rng.uniform(0, 5, 50)
        # Add some NaNs
        x[5] = np.nan
        y[10] = np.nan
        result_scalar = discretize_position(x, y, maze)
        result_fast = discretize_position_fast(x, y, maze)
        np.testing.assert_array_equal(result_scalar, result_fast)

    def test_empty_input(self, maze):
        x = np.array([], dtype=float)
        y = np.array([], dtype=float)
        result = discretize_position_fast(x, y, maze)
        assert len(result) == 0


class TestCellSequence:
    """Test cell trajectory compression."""

    def test_removes_consecutive_duplicates(self):
        indices = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3])
        cells, times = cell_sequence(indices)
        np.testing.assert_array_equal(cells, [0, 1, 2, 3])
        np.testing.assert_array_equal(times, [0, 3, 5, 8])

    def test_skips_invalid(self):
        indices = np.array([-1, -1, 0, 0, 1, -1, 2])
        cells, times = cell_sequence(indices)
        np.testing.assert_array_equal(cells, [0, 1, 2])
        np.testing.assert_array_equal(times, [2, 4, 6])

    def test_all_invalid(self):
        indices = np.array([-1, -1, -1])
        cells, times = cell_sequence(indices)
        assert len(cells) == 0
        assert len(times) == 0

    def test_empty(self):
        indices = np.array([], dtype=np.int32)
        cells, times = cell_sequence(indices)
        assert len(cells) == 0

    def test_single_cell(self):
        indices = np.array([5, 5, 5])
        cells, times = cell_sequence(indices)
        np.testing.assert_array_equal(cells, [5])
        np.testing.assert_array_equal(times, [0])


class TestNodeSequence:
    """Test node (junction/dead-end) sequence extraction."""

    def test_filters_to_important_nodes(self, maze):
        """Should only include junctions and dead ends."""
        # Build a trajectory through a known path
        # (0,0) → (1,0) → (2,0) — dead end, T-junction, dead end
        path = maze.path((0, 0), (2, 0))
        assert path is not None
        indices = np.array([maze.cell_to_idx[c] for c in path])
        # Repeat each cell a few times (simulating staying)
        repeated = np.repeat(indices, 5)
        nodes, times = node_sequence(repeated, maze)
        # Should get 3 nodes: (0,0), (1,0), (2,0)
        assert len(nodes) == 3

    def test_removes_corridor_cells(self, maze):
        """Corridor cells should be filtered out."""
        # Path through corridor: (1,0) → (1,1) → (1,2)
        # (1,1) is a corridor cell, (1,0) is T-junction, (1,2) is T-junction
        path = [(1, 0), (1, 1), (1, 2)]
        indices = np.array([maze.cell_to_idx[c] for c in path])
        nodes, times = node_sequence(indices, maze)
        assert len(nodes) == 2
        assert nodes[0] == maze.cell_to_idx[(1, 0)]
        assert nodes[1] == maze.cell_to_idx[(1, 2)]

    def test_empty_input(self, maze):
        indices = np.array([], dtype=np.int32)
        nodes, times = node_sequence(indices, maze)
        assert len(nodes) == 0

    def test_all_invalid(self, maze):
        indices = np.full(10, -1, dtype=np.int32)
        nodes, times = node_sequence(indices, maze)
        assert len(nodes) == 0
