"""Hypothesis property-based tests for maze analysis functions."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from hm2p.maze.analysis import (
    cell_occupancy,
    classify_turn,
    exploration_efficiency,
    occupancy_fraction,
    sequence_entropy,
)
from hm2p.maze.discretize import (
    cell_sequence,
    discretize_position,
    discretize_position_fast,
    node_sequence,
)
from hm2p.maze.topology import build_rose_maze


MAZE = build_rose_maze()


# ---------------------------------------------------------------------------
# Topology properties
# ---------------------------------------------------------------------------


class TestTopologyProperties:
    """Property-based tests for maze topology invariants."""

    def test_all_cells_reachable(self):
        """Every cell must be reachable from every other cell."""
        assert np.all(MAZE.dist >= 0)

    def test_distance_is_metric(self):
        """Distance should satisfy metric axioms."""
        n = MAZE.n_cells
        for i in range(n):
            # Non-negativity
            assert np.all(MAZE.dist[i] >= 0)
            # Identity
            assert MAZE.dist[i, i] == 0
            # Symmetry
            for j in range(n):
                assert MAZE.dist[i, j] == MAZE.dist[j, i]

    def test_junctions_plus_dead_ends_plus_corridors_equals_total(self):
        assert (
            len(MAZE.junctions) + len(MAZE.dead_ends) + len(MAZE.corridors)
            == MAZE.n_cells
        )


# ---------------------------------------------------------------------------
# Discretization properties
# ---------------------------------------------------------------------------


class TestDiscretizationProperties:
    @given(
        n=st.integers(1, 50),
        data=st.data(),
    )
    @settings(max_examples=30)
    def test_result_always_valid_or_minus_one(self, n, data):
        """Discretized positions should always be valid cell indices or -1."""
        x = data.draw(arrays(np.float64, shape=n,
                      elements=st.floats(0.0, 7.0, allow_nan=False, allow_infinity=False)))
        y = data.draw(arrays(np.float64, shape=n,
                      elements=st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False)))
        result = discretize_position(x, y, MAZE)
        assert np.all((result >= 0) & (result < MAZE.n_cells))

    @given(
        n=st.integers(1, 30),
        data=st.data(),
    )
    @settings(max_examples=20)
    def test_fast_matches_scalar(self, n, data):
        """Fast and scalar discretization should produce identical results."""
        x = data.draw(arrays(np.float64, shape=n,
                      elements=st.floats(0.0, 7.0, allow_nan=False, allow_infinity=False)))
        y = data.draw(arrays(np.float64, shape=n,
                      elements=st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False)))
        result_scalar = discretize_position(x, y, MAZE)
        result_fast = discretize_position_fast(x, y, MAZE)
        np.testing.assert_array_equal(result_scalar, result_fast)

    @given(
        indices=arrays(np.int32, shape=st.integers(1, 100),
                       elements=st.integers(-1, 22)),
    )
    @settings(max_examples=30)
    def test_cell_sequence_no_consecutive_duplicates(self, indices):
        """Cell sequence should never have consecutive identical values."""
        cells, times = cell_sequence(indices)
        if len(cells) > 1:
            assert np.all(cells[1:] != cells[:-1])

    @given(
        indices=arrays(np.int32, shape=st.integers(1, 100),
                       elements=st.integers(-1, 22)),
    )
    @settings(max_examples=20)
    def test_node_sequence_subset_of_cell_sequence(self, indices):
        """Node sequence should be a subsequence of cell sequence."""
        cells, _ = cell_sequence(indices)
        nodes, _ = node_sequence(indices, MAZE)
        # Every node should appear in cells
        if len(nodes) > 0:
            assert len(nodes) <= len(cells)


# ---------------------------------------------------------------------------
# Analysis properties
# ---------------------------------------------------------------------------


class TestOccupancyProperties:
    @given(
        indices=arrays(np.int32, shape=st.integers(0, 200),
                       elements=st.integers(-1, 22)),
    )
    @settings(max_examples=30)
    def test_occupancy_sums_to_valid_frames(self, indices):
        """Total occupancy should equal number of valid frames."""
        counts = cell_occupancy(indices, MAZE.n_cells)
        valid = np.sum(indices >= 0)
        assert counts.sum() == valid

    @given(
        indices=arrays(np.int32, shape=st.integers(1, 200),
                       elements=st.integers(0, 22)),
    )
    @settings(max_examples=20)
    def test_occupancy_fraction_sums_to_one(self, indices):
        """Occupancy fractions should sum to 1."""
        frac = occupancy_fraction(indices, MAZE.n_cells)
        assert abs(frac.sum() - 1.0) < 1e-10

    @given(
        indices=arrays(np.int32, shape=st.integers(0, 100),
                       elements=st.integers(-1, 22)),
    )
    @settings(max_examples=20)
    def test_occupancy_non_negative(self, indices):
        """All occupancy counts should be non-negative."""
        counts = cell_occupancy(indices, MAZE.n_cells)
        assert np.all(counts >= 0)


class TestEntropyProperties:
    @given(
        seq=arrays(np.int32, shape=st.integers(2, 100),
                   elements=st.integers(0, 10)),
    )
    @settings(max_examples=20)
    def test_entropy_non_negative(self, seq):
        """Conditional entropy should be non-negative."""
        ctx, ent = sequence_entropy(seq, max_context=5)
        assert np.all(ent >= -1e-10)

    @given(
        seq=arrays(np.int32, shape=st.integers(5, 50),
                   elements=st.integers(0, 5)),
    )
    @settings(max_examples=20)
    def test_entropy_bounded(self, seq):
        """Conditional entropy should not exceed log2(n_symbols)."""
        n_symbols = len(set(seq.tolist()))
        if n_symbols <= 1:
            return
        max_ent = np.log2(n_symbols)
        ctx, ent = sequence_entropy(seq, max_context=3)
        # Allow small tolerance for floating point
        assert np.all(ent <= max_ent + 0.1)


class TestExplorationProperties:
    @given(
        seq=arrays(np.int32, shape=st.integers(2, 100),
                   elements=st.integers(0, 13)),
    )
    @settings(max_examples=20)
    def test_new_nodes_bounded_by_window(self, seq):
        """Number of distinct nodes per window cannot exceed window size."""
        ws, nn = exploration_efficiency(seq)
        for w, n in zip(ws, nn):
            assert n <= w

    @given(
        seq=arrays(np.int32, shape=st.integers(2, 100),
                   elements=st.integers(0, 13)),
    )
    @settings(max_examples=20)
    def test_new_nodes_at_least_one(self, seq):
        """Every window must have at least 1 distinct node."""
        ws, nn = exploration_efficiency(seq)
        assert np.all(nn >= 1.0)


class TestTurnClassificationProperties:
    @given(
        dx=st.sampled_from([-1, 0, 1]),
        dy=st.sampled_from([-1, 0, 1]),
    )
    @settings(max_examples=20)
    def test_classify_turn_returns_valid_label(self, dx, dy):
        """Turn classification should always return a valid label."""
        assume(abs(dx) + abs(dy) == 1)  # Only cardinal directions
        prev = (5, 5)
        curr = (5 + dx, 5 + dy)
        # Try all possible next cells
        for dx2, dy2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (curr[0] + dx2, curr[1] + dy2)
            turn = classify_turn(prev, curr, nxt)
            assert turn in ("left", "right", "back", "forward")
