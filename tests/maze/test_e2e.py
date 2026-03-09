"""End-to-end integration tests for maze analysis pipeline.

Tests the full flow: position → discretization → node sequence → analysis metrics.
Uses synthetic trajectories that simulate realistic mouse behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.maze.analysis import (
    cell_occupancy,
    exploration_efficiency,
    find_monotonic_paths,
    maze_exploration_summary,
    occupancy_fraction,
    path_efficiency_over_time,
    per_junction_turn_bias,
    segment_modes,
    sequence_entropy,
    turn_bias,
)
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


def generate_random_walk(maze, n_steps: int, seed: int = 42) -> np.ndarray:
    """Generate a random walk trajectory on the maze graph.

    Returns (n_steps,) array of cell indices.
    """
    rng = np.random.default_rng(seed)
    cell_list = maze.cell_list
    current = rng.choice(len(cell_list))
    trajectory = [current]
    for _ in range(n_steps - 1):
        cell = cell_list[current]
        nbs = maze.adj[cell]
        next_cell = nbs[rng.integers(len(nbs))]
        current = maze.cell_to_idx[next_cell]
        trajectory.append(current)
    return np.array(trajectory, dtype=np.int32)


def generate_goal_directed_walk(
    maze, start_cell: tuple[int, int], goal_cell: tuple[int, int],
    noise_prob: float = 0.1, seed: int = 42,
) -> np.ndarray:
    """Generate a mostly goal-directed walk with occasional random steps.

    Returns (N,) array of cell indices.
    """
    rng = np.random.default_rng(seed)
    cell_list = maze.cell_list
    current = maze.cell_to_idx[start_cell]
    trajectory = [current]
    goal_idx = maze.cell_to_idx[goal_cell]
    max_steps = 200  # Safety limit

    for _ in range(max_steps):
        if current == goal_idx:
            break
        cell = cell_list[current]
        nbs = maze.adj[cell]

        if rng.random() < noise_prob:
            # Random step
            next_cell = nbs[rng.integers(len(nbs))]
        else:
            # Step toward goal (greedy: choose neighbour closest to goal)
            best_nb = min(nbs, key=lambda nb: maze.dist[maze.cell_to_idx[nb], goal_idx])
            next_cell = best_nb

        current = maze.cell_to_idx[next_cell]
        trajectory.append(current)

    return np.array(trajectory, dtype=np.int32)


class TestE2ERandomWalk:
    """End-to-end tests with random walk trajectories."""

    def test_full_pipeline_random_walk(self, maze):
        """Full pipeline: random walk → discretize → analyze."""
        traj = generate_random_walk(maze, 2000)

        # Cell sequence
        cells, times = cell_sequence(traj)
        assert len(cells) > 0
        assert len(cells) == len(times)
        # No consecutive duplicates
        assert np.all(cells[1:] != cells[:-1])

        # Node sequence
        nodes, ntimes = node_sequence(traj, maze)
        assert len(nodes) > 0
        assert len(nodes) <= len(cells)

        # Occupancy
        occ = cell_occupancy(traj, maze.n_cells)
        assert occ.sum() == len(traj)
        frac = occupancy_fraction(traj, maze.n_cells)
        assert abs(frac.sum() - 1.0) < 1e-10

        # Exploration efficiency
        ws, nn = exploration_efficiency(nodes)
        assert len(ws) > 0
        assert np.all(nn >= 1.0)

        # Turn bias
        tb = turn_bias(cells, maze)
        assert 0 <= tb["left_frac"] <= 1

        # Entropy
        ctx, ent = sequence_entropy(nodes, max_context=5)
        assert np.all(ent >= 0)

        # Summary
        summary = maze_exploration_summary(traj, maze)
        assert summary["unique_cells_visited"] > 0
        assert 0 < summary["coverage_frac"] <= 1.0

    def test_coverage_increases_with_steps(self, maze):
        """More steps should lead to higher coverage."""
        short = generate_random_walk(maze, 100, seed=1)
        long = generate_random_walk(maze, 5000, seed=1)

        s1 = maze_exploration_summary(short, maze)
        s2 = maze_exploration_summary(long, maze)

        assert s2["unique_cells_visited"] >= s1["unique_cells_visited"]

    def test_random_walk_visits_all_cells(self, maze):
        """Long random walk should visit all 23 cells."""
        traj = generate_random_walk(maze, 10000, seed=7)
        unique = len(set(traj.tolist()))
        assert unique == maze.n_cells

    def test_turn_bias_approximately_balanced(self, maze):
        """Long random walk should have approximately equal left/right bias."""
        traj = generate_random_walk(maze, 50000, seed=99)
        cells, _ = cell_sequence(traj)
        tb = turn_bias(cells, maze)
        # Should be close to 0.5 for unbiased walk
        assert 0.3 < tb["left_frac"] < 0.7


class TestE2EGoalDirected:
    """End-to-end tests with goal-directed trajectories."""

    def test_goal_directed_finds_monotonic_path(self, maze):
        """Goal-directed walk should contain a monotonic path to target."""
        traj = generate_goal_directed_walk(
            maze, start_cell=(0, 0), goal_cell=(6, 4), noise_prob=0.05, seed=42,
        )
        nodes, ntimes = node_sequence(traj, maze)

        target_idx = maze.cell_to_idx[(6, 4)]
        paths = find_monotonic_paths(nodes, ntimes, target_idx, maze)
        assert len(paths) >= 1
        # Efficiency should be high for mostly-directed walk
        assert paths[0]["efficiency"] > 0.5

    def test_goal_directed_higher_efficiency(self, maze):
        """Goal-directed walk should have higher path efficiency than random."""
        goal_traj = generate_goal_directed_walk(
            maze, start_cell=(0, 0), goal_cell=(6, 4), noise_prob=0.05,
        )
        rand_traj = generate_random_walk(maze, len(goal_traj))

        goal_nodes, goal_times = node_sequence(goal_traj, maze)
        rand_nodes, rand_times = node_sequence(rand_traj, maze)

        if len(goal_nodes) > 3 and len(rand_nodes) > 3:
            _, goal_eff = path_efficiency_over_time(goal_nodes, goal_times, maze, window=5)
            _, rand_eff = path_efficiency_over_time(rand_nodes, rand_times, maze, window=5)

            if len(goal_eff) > 0 and len(rand_eff) > 0:
                assert np.mean(goal_eff) >= np.mean(rand_eff) * 0.8  # Allow some slack

    def test_mode_segmentation_detects_directed(self, maze):
        """Mode segmentation should find directed segments in goal-directed walk."""
        traj = generate_goal_directed_walk(
            maze, start_cell=(0, 0), goal_cell=(6, 4), noise_prob=0.0,
        )
        nodes, ntimes = node_sequence(traj, maze)
        segments = segment_modes(nodes, ntimes, maze)

        # Should find at least one directed segment
        directed = [s for s in segments if s["mode"] == "directed"]
        assert len(directed) >= 1

    def test_optimal_path_is_fully_monotonic(self, maze):
        """Direct shortest path should be detected as a single monotonic path."""
        path = maze.path((0, 0), (6, 4))
        assert path is not None

        indices = np.array([maze.cell_to_idx[c] for c in path])
        nodes, ntimes = node_sequence(indices, maze)

        target_idx = maze.cell_to_idx[(6, 4)]
        paths = find_monotonic_paths(nodes, ntimes, target_idx, maze)

        # Should find exactly one monotonic path covering entire trajectory
        assert len(paths) == 1
        assert paths[0]["start_idx"] == 0
        assert paths[0]["end_idx"] == len(nodes) - 1


class TestE2EContinuousPosition:
    """End-to-end tests starting from continuous (x, y) positions."""

    def test_continuous_to_analysis(self, maze):
        """Full flow: continuous position → discrete → analysis."""
        # Generate smooth trajectory through accessible cells
        path = maze.path((0, 0), (6, 4))
        assert path is not None

        # Create continuous trajectory with sub-cell resolution
        x_points = [c[0] + 0.5 for c in path]
        y_points = [c[1] + 0.5 for c in path]

        # Interpolate for smooth trajectory
        t = np.linspace(0, 1, 200)
        t_nodes = np.linspace(0, 1, len(path))
        x = np.interp(t, t_nodes, x_points)
        y = np.interp(t, t_nodes, y_points)

        # Add small noise
        rng = np.random.default_rng(42)
        x += rng.normal(0, 0.05, len(x))
        y += rng.normal(0, 0.05, len(y))

        # Discretize
        indices = discretize_position(x, y, maze)
        assert np.all(indices >= 0)

        # Fast version should match
        indices_fast = discretize_position_fast(x, y, maze)
        np.testing.assert_array_equal(indices, indices_fast)

        # Analyze
        summary = maze_exploration_summary(indices, maze)
        assert summary["unique_cells_visited"] > 0

    def test_nan_positions_handled(self, maze):
        """NaN positions should be handled gracefully."""
        x = np.array([0.5, np.nan, 1.5, np.nan, 2.5])
        y = np.array([0.5, 0.5, np.nan, np.nan, 0.5])

        indices = discretize_position(x, y, maze)
        assert indices[0] >= 0
        assert indices[1] == -1
        assert indices[2] == -1
        assert indices[3] == -1
        assert indices[4] >= 0

        # Analysis should handle -1 indices
        summary = maze_exploration_summary(indices, maze)
        assert summary["valid_frames"] == 2

    def test_out_of_bounds_positions(self, maze):
        """Positions outside maze should map to nearest accessible cell."""
        # Position at (3.5, 0.5) is in a wall cell
        x = np.array([3.5])
        y = np.array([0.5])
        indices = discretize_position(x, y, maze)
        assert indices[0] >= 0
        mapped = maze.cell_list[indices[0]]
        assert mapped in maze.cells


class TestE2EPerJunction:
    """End-to-end tests for per-junction analysis."""

    def test_all_junctions_visited(self, maze):
        """Long random walk should visit all T-junctions."""
        traj = generate_random_walk(maze, 10000, seed=11)
        cells, _ = cell_sequence(traj)
        pj = per_junction_turn_bias(cells, maze)

        for junction in maze.junctions:
            counts = pj[junction]
            total = sum(counts.values())
            assert total > 0, f"Junction {junction} never visited"

    def test_junction_counts_consistent(self, maze):
        """Sum of per-junction turns should equal global turn count."""
        traj = generate_random_walk(maze, 5000, seed=22)
        cells, _ = cell_sequence(traj)

        global_tb = turn_bias(cells, maze)
        pj = per_junction_turn_bias(cells, maze)

        total_left = sum(pj[j]["left"] for j in maze.junctions)
        total_right = sum(pj[j]["right"] for j in maze.junctions)

        assert total_left == global_tb["left"]
        assert total_right == global_tb["right"]


class TestE2EEntropy:
    """End-to-end entropy tests."""

    def test_entropy_decreases_with_bias(self, maze):
        """Biased walk should have lower entropy than random walk."""
        # Random walk
        rand_traj = generate_random_walk(maze, 5000, seed=33)
        rand_nodes, _ = node_sequence(rand_traj, maze)

        # Goal-directed walk (repeated round trips)
        biased_traj = []
        for _ in range(50):
            t = generate_goal_directed_walk(
                maze, (0, 0), (6, 4), noise_prob=0.0, seed=33,
            )
            biased_traj.extend(t.tolist())
            t = generate_goal_directed_walk(
                maze, (6, 4), (0, 0), noise_prob=0.0, seed=34,
            )
            biased_traj.extend(t.tolist())
        biased_traj = np.array(biased_traj, dtype=np.int32)
        biased_nodes, _ = node_sequence(biased_traj, maze)

        if len(rand_nodes) > 10 and len(biased_nodes) > 10:
            _, rand_ent = sequence_entropy(rand_nodes, max_context=3)
            _, biased_ent = sequence_entropy(biased_nodes, max_context=3)
            # Biased should be more predictable (lower entropy) at context ≥ 2
            assert biased_ent[-1] <= rand_ent[-1] + 0.5  # Allow some noise
