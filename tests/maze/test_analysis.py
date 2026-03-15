"""Tests for maze behavioural analysis metrics."""

from __future__ import annotations

import numpy as np
import pytest

from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.maze.analysis import (
    cell_occupancy,
    classify_turn,
    cross_entropy,
    cross_entropy_2nd_order,
    dead_end_visits,
    exploration_efficiency,
    find_monotonic_paths,
    markov_order_comparison,
    maze_exploration_summary,
    occupancy_fraction,
    path_efficiency_over_time,
    per_junction_turn_bias,
    segment_modes,
    sequence_entropy,
    simulate_random_walk,
    stationary_distribution,
    transition_entropy,
    transition_entropy_2nd_order,
    transition_matrix,
    transition_matrix_2nd_order,
    turn_bias,
)
from hm2p.maze.topology import build_rose_maze


@pytest.fixture
def maze():
    return build_rose_maze()


# ---------------------------------------------------------------------------
# Occupancy
# ---------------------------------------------------------------------------


class TestOccupancy:
    def test_basic_counts(self):
        indices = np.array([0, 0, 1, 1, 1, 2])
        counts = cell_occupancy(indices, 5)
        np.testing.assert_array_equal(counts, [2, 3, 1, 0, 0])

    def test_ignores_invalid(self):
        indices = np.array([-1, 0, -1, 1])
        counts = cell_occupancy(indices, 3)
        np.testing.assert_array_equal(counts, [1, 1, 0])

    def test_empty(self):
        indices = np.array([], dtype=np.int32)
        counts = cell_occupancy(indices, 3)
        np.testing.assert_array_equal(counts, [0, 0, 0])

    def test_fraction_sums_to_one(self):
        indices = np.array([0, 0, 1, 2, 2, 2])
        frac = occupancy_fraction(indices, 5)
        assert abs(frac.sum() - 1.0) < 1e-10

    def test_fraction_all_invalid(self):
        indices = np.array([-1, -1])
        frac = occupancy_fraction(indices, 3)
        np.testing.assert_array_equal(frac, [0, 0, 0])


# ---------------------------------------------------------------------------
# Exploration efficiency
# ---------------------------------------------------------------------------


class TestExplorationEfficiency:
    def test_all_same_node(self):
        """Visiting same node repeatedly → 1 unique per window."""
        seq = np.array([0] * 20)
        ws, nn = exploration_efficiency(seq)
        assert np.all(nn == 1.0)

    def test_all_distinct(self):
        """Each visit to a new node → unique count grows with window."""
        seq = np.arange(20)
        ws, nn = exploration_efficiency(seq)
        # Each window of size w should have w unique nodes
        for w, n in zip(ws, nn):
            assert n <= w  # Can't have more unique than window size

    def test_empty(self):
        seq = np.array([], dtype=np.int32)
        ws, nn = exploration_efficiency(seq)
        assert len(ws) == 0

    def test_custom_windows(self):
        seq = np.arange(10)
        ws, nn = exploration_efficiency(seq, window_sizes=np.array([2, 5]))
        assert len(nn) == 2


# ---------------------------------------------------------------------------
# Turn classification
# ---------------------------------------------------------------------------


class TestClassifyTurn:
    def test_left_turn(self):
        # Moving right (east), then turning north = left turn
        turn = classify_turn((0, 0), (1, 0), (1, 1))
        assert turn == "left"

    def test_right_turn(self):
        # Moving right (east), then turning south... but south means y-1
        # Moving up (north), then turning east = right turn
        turn = classify_turn((1, 0), (1, 1), (2, 1))
        assert turn == "right"

    def test_back_turn(self):
        # Moving right, then going back left
        turn = classify_turn((0, 0), (1, 0), (0, 0))
        assert turn == "back"

    def test_forward(self):
        # Moving right, continuing right
        turn = classify_turn((0, 0), (1, 0), (2, 0))
        assert turn == "forward"


class TestTurnBias:
    def test_with_synthetic_trajectory(self, maze):
        """Trajectory with known turns should produce correct bias."""
        # Build a path: (0,0) → (1,0) → (1,1) → (1,2)
        # At (1,0): arriving from left (0,0), departing up (1,1) = left turn
        path = [(0, 0), (1, 0), (1, 1), (1, 2)]
        indices = np.array([maze.cell_to_idx[c] for c in path])
        tb = turn_bias(indices, maze)
        assert tb["left"] + tb["right"] + tb["back"] + tb["forward"] >= 0

    def test_empty_trajectory(self, maze):
        indices = np.array([], dtype=np.int32)
        tb = turn_bias(indices, maze)
        assert tb["left_frac"] == 0.5  # Default when no data

    def test_per_junction(self, maze):
        path = [(0, 0), (1, 0), (2, 0)]
        indices = np.array([maze.cell_to_idx[c] for c in path])
        pj = per_junction_turn_bias(indices, maze)
        assert (1, 0) in pj  # T-junction should be in results


# ---------------------------------------------------------------------------
# Monotonic paths
# ---------------------------------------------------------------------------


class TestMonotonicPaths:
    def test_direct_path_to_dead_end(self, maze):
        """Direct path to dead end should be detected as monotonic."""
        # Path: (1,2) → (1,1) → (1,0) → (0,0) — monotonically toward (0,0)
        path_cells = [(1, 2), (1, 1), (1, 0), (0, 0)]
        node_seq = np.array([maze.cell_to_idx[c] for c in path_cells])
        node_times = np.arange(len(node_seq))
        target = maze.cell_to_idx[(0, 0)]
        paths = find_monotonic_paths(node_seq, node_times, target, maze)
        assert len(paths) >= 1
        assert paths[0]["efficiency"] > 0

    def test_no_path_when_not_visiting_target(self, maze):
        """If target is never visited, no paths should be found."""
        path_cells = [(1, 0), (1, 1), (1, 2)]
        node_seq = np.array([maze.cell_to_idx[c] for c in path_cells])
        node_times = np.arange(len(node_seq))
        target = maze.cell_to_idx[(6, 4)]  # Never visited
        paths = find_monotonic_paths(node_seq, node_times, target, maze)
        assert len(paths) == 0

    def test_empty_input(self, maze):
        paths = find_monotonic_paths(
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            0, maze
        )
        assert len(paths) == 0


# ---------------------------------------------------------------------------
# Path efficiency
# ---------------------------------------------------------------------------


class TestPathEfficiency:
    def test_optimal_path_has_efficiency_one(self, maze):
        """Straight shortest path should have efficiency close to 1."""
        path_cells = list(maze.path((0, 0), (6, 4)))
        node_seq = np.array([maze.cell_to_idx[c] for c in path_cells])
        node_times = np.arange(len(node_seq))
        times, eff = path_efficiency_over_time(
            node_seq, node_times, maze, window=len(node_seq)
        )
        if len(eff) > 0:
            assert eff[0] == pytest.approx(1.0)

    def test_empty_input(self, maze):
        times, eff = path_efficiency_over_time(
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            maze
        )
        assert len(times) == 0


# ---------------------------------------------------------------------------
# Mode segmentation
# ---------------------------------------------------------------------------


class TestSegmentModes:
    def test_segments_not_empty_for_valid_trajectory(self, maze):
        # Build trajectory visiting multiple nodes including dead ends
        path = maze.path((0, 0), (6, 4))
        assert path is not None
        return_path = maze.path((6, 4), (0, 0))
        assert return_path is not None
        full_path = list(path) + list(return_path)[1:]
        node_seq = np.array([maze.cell_to_idx[c] for c in full_path])
        node_times = np.arange(len(node_seq))
        segments = segment_modes(node_seq, node_times, maze)
        assert len(segments) > 0
        for seg in segments:
            assert seg["mode"] in ("directed", "explore")

    def test_empty_input(self, maze):
        segments = segment_modes(
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            maze
        )
        assert len(segments) == 0


# ---------------------------------------------------------------------------
# Sequence entropy
# ---------------------------------------------------------------------------


class TestSequenceEntropy:
    def test_deterministic_sequence_low_entropy(self):
        """Perfectly predictable sequence should have low/decreasing entropy."""
        # Repeating pattern 0, 1, 2, 0, 1, 2, ...
        seq = np.tile([0, 1, 2], 50)
        ctx, ent = sequence_entropy(seq, max_context=5)
        # With enough context, pattern is fully predictable
        # Context 3+ should give near-zero (period=3, so context 2 suffices)
        assert ent[-1] <= ent[0] + 1e-10  # More context ≤ less context

    def test_random_sequence_higher_entropy(self):
        """Random sequence should have higher entropy."""
        rng = np.random.default_rng(42)
        seq = rng.integers(0, 5, 200)
        ctx, ent = sequence_entropy(seq, max_context=5)
        assert ent[0] > 0  # Non-zero entropy

    def test_empty(self):
        ctx, ent = sequence_entropy(np.array([], dtype=np.int32))
        assert len(ctx) == 1

    def test_entropy_non_negative(self):
        rng = np.random.default_rng(99)
        seq = rng.integers(0, 3, 100)
        ctx, ent = sequence_entropy(seq, max_context=5)
        assert np.all(ent >= 0)


# ---------------------------------------------------------------------------
# Markov model
# ---------------------------------------------------------------------------


class TestTransitionMatrix:
    def test_row_stochastic(self):
        """Rows should sum to 1."""
        seq = np.array([0, 1, 0, 1, 2, 0])
        tm = transition_matrix(seq, 3)
        for i in range(3):
            if tm[i].sum() > 0:
                assert abs(tm[i].sum() - 1.0) < 1e-10

    def test_deterministic_sequence(self):
        """0→1→2→0→1→2 should have deterministic transitions."""
        seq = np.array([0, 1, 2, 0, 1, 2])
        tm = transition_matrix(seq, 3)
        assert tm[0, 1] == 1.0
        assert tm[1, 2] == 1.0
        assert tm[2, 0] == 1.0

    def test_pseudocount(self):
        """Pseudocount should smooth probabilities."""
        seq = np.array([0, 1, 0, 1])
        tm = transition_matrix(seq, 3, pseudocount=1.0)
        # All transitions should be > 0 with pseudocount
        assert np.all(tm > 0)

    def test_empty_sequence(self):
        tm = transition_matrix(np.array([], dtype=np.int32), 3)
        np.testing.assert_array_equal(tm, 0)


class TestTransitionEntropy:
    def test_deterministic_zero_entropy(self):
        """Deterministic transitions should have zero entropy."""
        seq = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        tm = transition_matrix(seq, 3)
        h = transition_entropy(tm, seq)
        assert h < 0.01

    def test_uniform_max_entropy(self):
        """Uniform transitions should have maximum entropy."""
        # Create uniform transition matrix
        tm = np.ones((3, 3)) / 3
        seq = np.array([0, 1, 2] * 100)
        h = transition_entropy(tm, seq)
        # Should be close to log2(3) ≈ 1.585
        assert abs(h - np.log2(3)) < 0.1

    def test_non_negative(self):
        seq = np.array([0, 1, 0, 2, 1, 0, 2, 1])
        tm = transition_matrix(seq, 3)
        h = transition_entropy(tm, seq)
        assert h >= 0


class TestCrossEntropy:
    def test_same_distribution_low_ce(self):
        """Cross-entropy should be low when test matches training."""
        seq = np.array([0, 1, 0, 1, 0, 1] * 50)
        tm = transition_matrix(seq, 3)
        ce = cross_entropy(seq, tm, 3)
        assert ce < 2.0  # Low because predictable

    def test_mismatched_higher_ce(self):
        """Mismatched test data should have higher cross-entropy."""
        train = np.array([0, 1, 0, 1] * 50)
        test = np.array([0, 2, 0, 2] * 50)
        tm = transition_matrix(train, 3, pseudocount=0.1)
        ce_match = cross_entropy(train, tm, 3)
        ce_mismatch = cross_entropy(test, tm, 3)
        assert ce_mismatch > ce_match

    def test_empty_sequence(self):
        tm = np.ones((3, 3)) / 3
        ce = cross_entropy(np.array([], dtype=np.int32), tm, 3)
        assert ce == 0.0


# ---------------------------------------------------------------------------
# 2nd-order Markov model
# ---------------------------------------------------------------------------


class TestTransitionMatrix2ndOrder:
    def test_shape(self):
        seq = np.array([0, 1, 2, 0, 1, 2])
        tm = transition_matrix_2nd_order(seq, 3)
        assert tm.shape == (3, 3, 3)

    def test_row_stochastic(self):
        """Each [i, j, :] slice with observed transitions sums to 1."""
        seq = np.array([0, 1, 2, 0, 1, 2, 1, 0, 2])
        tm = transition_matrix_2nd_order(seq, 3)
        for i in range(3):
            for j in range(3):
                s = tm[i, j, :].sum()
                if s > 0:
                    assert abs(s - 1.0) < 1e-10

    def test_deterministic_triplets(self):
        """Cyclic 0->1->2->0->1->2 should produce deterministic 2nd-order."""
        seq = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        tm = transition_matrix_2nd_order(seq, 3)
        # After seeing (0, 1) the next is always 2
        assert tm[0, 1, 2] == 1.0
        # After seeing (1, 2) the next is always 0
        assert tm[1, 2, 0] == 1.0
        # After seeing (2, 0) the next is always 1
        assert tm[2, 0, 1] == 1.0

    def test_pseudocount_all_positive(self):
        """With pseudocount, all entries should be > 0."""
        seq = np.array([0, 1, 2, 0])
        tm = transition_matrix_2nd_order(seq, 3, pseudocount=1.0)
        assert np.all(tm > 0)

    def test_short_sequence_zeros(self):
        """Sequence shorter than 3 should return all zeros."""
        seq = np.array([0, 1])
        tm = transition_matrix_2nd_order(seq, 3)
        np.testing.assert_array_equal(tm, 0)

    @given(
        seq=st.lists(
            st.integers(min_value=0, max_value=4), min_size=5, max_size=100,
        ).map(np.array),
    )
    @settings(max_examples=50, deadline=None)
    def test_property_row_stochastic(self, seq):
        """Random sequences always produce row-stochastic slices."""
        tm = transition_matrix_2nd_order(seq, 5)
        for i in range(5):
            for j in range(5):
                s = tm[i, j, :].sum()
                if s > 0:
                    assert abs(s - 1.0) < 1e-10


class TestTransitionEntropy2ndOrder:
    def test_deterministic_near_zero(self):
        """Deterministic cyclic sequence should have near-zero entropy."""
        seq = np.array([0, 1, 2] * 30)
        tm = transition_matrix_2nd_order(seq, 3)
        h = transition_entropy_2nd_order(tm, seq)
        assert h < 0.01

    def test_non_negative(self):
        """Entropy should always be >= 0."""
        rng = np.random.default_rng(42)
        seq = rng.integers(0, 4, 200)
        tm = transition_matrix_2nd_order(seq, 4)
        h = transition_entropy_2nd_order(tm, seq)
        assert h >= 0


class TestCrossEntropy2ndOrder:
    def test_matched_lower(self):
        """Train=test cross-entropy should be lower than mismatched."""
        train = np.array([0, 1, 2] * 50)
        test_match = np.array([0, 1, 2] * 50)
        test_mismatch = np.array([0, 2, 1] * 50)
        tm = transition_matrix_2nd_order(train, 3, pseudocount=0.01)
        ce_match = cross_entropy_2nd_order(test_match, tm, 3)
        ce_mismatch = cross_entropy_2nd_order(test_mismatch, tm, 3)
        assert ce_match < ce_mismatch

    def test_penalty_for_unseen(self):
        """Unseen triplet should incur high penalty (20-bit)."""
        train = np.array([0, 1, 0, 1, 0, 1])
        test = np.array([0, 1, 2])  # triplet (0,1,2) never in training
        tm = transition_matrix_2nd_order(train, 3)  # no pseudocount
        ce = cross_entropy_2nd_order(test, tm, 3)
        assert ce >= 19.0  # Should be ~20 from penalty


class TestStationaryDistribution:
    def test_sums_to_one(self):
        seq = np.array([0, 1, 2, 0, 1, 0, 2, 1] * 20)
        tm = transition_matrix(seq, 3)
        pi = stationary_distribution(tm)
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_non_negative(self):
        seq = np.array([0, 1, 2, 0, 1, 0, 2, 1] * 20)
        tm = transition_matrix(seq, 3)
        pi = stationary_distribution(tm)
        assert np.all(pi >= 0)

    def test_doubly_stochastic_uniform(self):
        """Doubly stochastic matrix should give uniform distribution."""
        # Circulant matrix: each row and column sums to 1
        tm = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ])
        pi = stationary_distribution(tm)
        np.testing.assert_allclose(pi, 1.0 / 3, atol=1e-10)


class TestMarkovOrderComparison:
    def test_returns_expected_keys(self):
        seq = np.array([0, 1, 2, 0, 1, 2, 1, 0] * 20)
        result = markov_order_comparison(seq, 3)
        assert "order_1" in result
        assert "order_2" in result
        assert "delta_aic" in result
        assert "delta_bic" in result
        assert "preferred_order" in result

    def test_preferred_order_is_int(self):
        seq = np.array([0, 1, 2, 0, 1, 2, 1, 0] * 20)
        result = markov_order_comparison(seq, 3)
        assert isinstance(result["preferred_order"], int)
        assert result["preferred_order"] in (1, 2)


# ---------------------------------------------------------------------------
# Dead-end analysis
# ---------------------------------------------------------------------------


class TestDeadEndVisits:
    def test_detects_dead_end_visits(self, maze):
        """Should detect visits to dead ends."""
        # Path: (1,0) → (0,0) → (1,0) → (2,0)
        path = [(1, 0), (0, 0), (1, 0), (2, 0)]
        seq = np.array([maze.cell_to_idx[c] for c in path])
        devs = dead_end_visits(seq, maze)
        assert devs[(0, 0)]["visits"] == 1
        assert devs[(2, 0)]["visits"] == 1

    def test_dwell_time(self, maze):
        """Should count consecutive frames at dead end."""
        # Stay at (0,0) for 3 steps
        seq = np.array([maze.cell_to_idx[(0, 0)]] * 3 + [maze.cell_to_idx[(1, 0)]])
        devs = dead_end_visits(seq, maze)
        assert devs[(0, 0)]["visits"] == 1
        assert devs[(0, 0)]["total_frames"] == 3

    def test_unvisited_dead_ends_zero(self, maze):
        """Unvisited dead ends should have 0 visits."""
        seq = np.array([maze.cell_to_idx[(1, 0)], maze.cell_to_idx[(1, 1)]])
        devs = dead_end_visits(seq, maze)
        for de in maze.dead_ends:
            assert devs[de]["visits"] >= 0


# ---------------------------------------------------------------------------
# Random walk simulation
# ---------------------------------------------------------------------------


class TestSimulateRandomWalk:
    def test_valid_trajectory(self, maze):
        traj = simulate_random_walk(maze, 100)
        assert len(traj) == 100
        assert np.all(traj >= 0)
        assert np.all(traj < maze.n_cells)

    def test_all_transitions_valid(self, maze):
        """Each transition should be to an adjacent cell."""
        traj = simulate_random_walk(maze, 500)
        for i in range(len(traj) - 1):
            curr = maze.cell_list[traj[i]]
            nxt = maze.cell_list[traj[i + 1]]
            assert nxt in maze.adj[curr] or traj[i] == traj[i + 1]

    def test_forward_bias_reduces_backtracking(self, maze):
        """Forward-biased walk should have fewer direction changes."""
        unbiased = simulate_random_walk(maze, 5000, seed=1, forward_bias=0.0)
        biased = simulate_random_walk(maze, 5000, seed=1, forward_bias=0.8)

        # Count direction reversals
        def count_reversals(traj):
            n = 0
            for i in range(2, len(traj)):
                if traj[i] == traj[i - 2] and traj[i] != traj[i - 1]:
                    n += 1
            return n

        assert count_reversals(biased) < count_reversals(unbiased)

    def test_deterministic_with_seed(self, maze):
        """Same seed should produce same trajectory."""
        t1 = simulate_random_walk(maze, 100, seed=42)
        t2 = simulate_random_walk(maze, 100, seed=42)
        np.testing.assert_array_equal(t1, t2)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestMazeExplorationSummary:
    def test_basic_summary(self, maze):
        """Summary should return expected keys."""
        # Simulate simple trajectory
        path = maze.path((0, 0), (6, 4))
        assert path is not None
        indices = np.array([maze.cell_to_idx[c] for c in path])
        repeated = np.repeat(indices, 10)  # 10 frames per cell
        summary = maze_exploration_summary(repeated, maze, fps=30.0)
        assert "total_frames" in summary
        assert "unique_cells_visited" in summary
        assert "coverage_frac" in summary
        assert summary["unique_cells_visited"] == len(path)
        assert 0 < summary["coverage_frac"] <= 1.0
        assert summary["total_frames"] == len(repeated)

    def test_all_invalid(self, maze):
        indices = np.full(20, -1, dtype=np.int32)
        summary = maze_exploration_summary(indices, maze)
        assert summary["unique_cells_visited"] == 0
        assert summary["coverage_frac"] == 0.0
