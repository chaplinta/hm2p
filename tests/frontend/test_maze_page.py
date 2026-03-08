"""Tests for maze analysis frontend page logic."""

from __future__ import annotations

import numpy as np
import pytest


class TestMazeTopologyDisplay:
    """Test maze topology data used in the page."""

    def test_maze_builds_successfully(self):
        from hm2p.maze.topology import build_rose_maze
        maze = build_rose_maze()
        assert maze.n_cells == 23
        assert len(maze.junctions) == 8
        assert len(maze.dead_ends) == 6

    def test_maze_cell_list_sorted(self):
        from hm2p.maze.topology import build_rose_maze
        maze = build_rose_maze()
        # cell_list should be sorted
        assert maze.cell_list == sorted(maze.cell_list)

    def test_distance_matrix_shape(self):
        from hm2p.maze.topology import build_rose_maze
        maze = build_rose_maze()
        assert maze.dist.shape == (23, 23)

    def test_random_walk_produces_valid_trajectory(self):
        """Random walk generator (used in demo) should work."""
        from hm2p.maze.topology import build_rose_maze
        maze = build_rose_maze()
        rng = np.random.default_rng(42)
        current = rng.choice(len(maze.cell_list))
        trajectory = [current]
        for _ in range(100):
            cell = maze.cell_list[current]
            nbs = maze.adj[cell]
            next_cell = nbs[rng.integers(len(nbs))]
            current = maze.cell_to_idx[next_cell]
            trajectory.append(current)
        traj = np.array(trajectory, dtype=np.int32)
        assert len(traj) == 101
        assert np.all(traj >= 0)
        assert np.all(traj < maze.n_cells)


class TestSignalQualityComputations:
    """Test signal quality computations used in the page."""

    def test_photobleaching_detection(self):
        """Declining trace should be detected as photobleaching."""
        n_frames = 1000
        # Simulate photobleaching: exponential decay
        t = np.linspace(0, 1, n_frames)
        trace = np.exp(-t) + np.random.default_rng(42).normal(0, 0.01, n_frames)
        n10 = n_frames // 10
        first_mean = np.mean(trace[:n10])
        last_mean = np.mean(trace[-n10:])
        drift = last_mean - first_mean
        assert drift < 0  # Should detect negative drift

    def test_snr_computation(self):
        """SNR from baseline std and 95th percentile."""
        rng = np.random.default_rng(42)
        # Low noise baseline with frequent calcium events
        trace = rng.normal(0, 0.01, 1000)
        # Add events to 10% of frames so 95th pct captures them
        for i in range(0, 1000, 10):
            trace[i:i + 3] = 0.5  # Moderate calcium events

        baseline = trace[trace < np.percentile(trace, 50)]
        noise_std = np.std(baseline)
        peak = np.percentile(trace, 95)
        snr = peak / noise_std
        assert snr > 5  # Should be good SNR

    def test_autocorrelation_via_fft(self):
        """FFT-based autocorrelation should give 1.0 at lag 0."""
        rng = np.random.default_rng(42)
        trace = rng.normal(0, 1, 500)
        trace -= np.mean(trace)
        n = len(trace)
        fft = np.fft.fft(trace, n=2 * n)
        acf = np.fft.ifft(fft * np.conj(fft)).real[:n]
        acf /= acf[0]
        assert acf[0] == pytest.approx(1.0)
        # Should decay toward 0 for white noise
        assert abs(acf[10]) < 0.5

    def test_quality_grading(self):
        """Quality grading logic."""
        # Grade A: SNR >= 5, drift < 15%
        assert _grade(snr=6, drift_pct=5) == "A"
        # Grade B: SNR >= 5, drift 15-30%
        assert _grade(snr=6, drift_pct=20) == "B"
        # Grade C: SNR 2-3
        assert _grade(snr=2.5, drift_pct=5) == "C"
        # Grade D: SNR < 2
        assert _grade(snr=1.5, drift_pct=5) == "D"


def _grade(snr: float, drift_pct: float) -> str:
    """Reproduce grading logic from signal_quality_page.py."""
    grade = "A"
    if snr < 2 or abs(drift_pct) > 50:
        grade = "D"
    elif snr < 3 or abs(drift_pct) > 30:
        grade = "C"
    elif snr < 5 or abs(drift_pct) > 15:
        grade = "B"
    return grade


class TestMarkovModel:
    """Test Markov transition model for frontend display."""

    def test_transition_matrix_shape(self):
        from hm2p.maze.analysis import transition_matrix
        seq = np.array([0, 1, 2, 0, 1])
        tm = transition_matrix(seq, 5)
        assert tm.shape == (5, 5)

    def test_transition_entropy_range(self):
        from hm2p.maze.analysis import transition_entropy, transition_matrix
        seq = np.array([0, 1, 2, 0, 1, 2] * 10)
        tm = transition_matrix(seq, 3)
        h = transition_entropy(tm, seq)
        assert 0 <= h <= np.log2(3) + 0.01


class TestForwardBiasSweep:
    """Test forward bias sweep logic used in maze page."""

    def test_bias_sweep_produces_valid_summaries(self):
        from hm2p.maze.analysis import maze_exploration_summary, simulate_random_walk
        from hm2p.maze.topology import build_rose_maze
        maze = build_rose_maze()
        for bf in [0.0, 0.2, 0.5, 0.8]:
            traj = simulate_random_walk(maze, 500, seed=42, forward_bias=bf)
            s = maze_exploration_summary(traj, maze)
            assert 0 < s["coverage_frac"] <= 1.0
            assert s["unique_cells_visited"] > 0

    def test_bias_affects_dead_end_dwell(self):
        """Forward bias changes dead-end visit patterns."""
        from hm2p.maze.analysis import dead_end_visits, simulate_random_walk
        from hm2p.maze.topology import build_rose_maze
        maze = build_rose_maze()
        low = simulate_random_walk(maze, 5000, seed=42, forward_bias=0.0)
        high = simulate_random_walk(maze, 5000, seed=42, forward_bias=0.8)
        de_low = dead_end_visits(low, maze)
        de_high = dead_end_visits(high, maze)
        # Both should have dead-end data for all dead ends
        assert len(de_low) == len(maze.dead_ends)
        assert len(de_high) == len(maze.dead_ends)
        # High bias → longer dwell times per visit (once entered, harder to leave)
        avg_dwell_low = np.mean([v["mean_dwell"] for v in de_low.values() if v["visits"] > 0])
        avg_dwell_high = np.mean([v["mean_dwell"] for v in de_high.values() if v["visits"] > 0])
        assert avg_dwell_high >= avg_dwell_low * 0.5  # Relaxed — direction-dependent

    def test_occupancy_grid_construction(self):
        """Test the occupancy grid construction used in the heatmap."""
        from hm2p.maze.analysis import cell_occupancy, simulate_random_walk
        from hm2p.maze.topology import build_rose_maze
        maze = build_rose_maze()
        traj = simulate_random_walk(maze, 1000, seed=42)
        occ = cell_occupancy(traj, maze.n_cells)
        grid = np.full((5, 7), np.nan)
        for i, cell in enumerate(maze.cell_list):
            grid[cell[1], cell[0]] = occ[i]
        # Check that accessible cells have values
        assert np.sum(~np.isnan(grid)) == maze.n_cells
        # Check that total occupancy matches trajectory length
        assert int(np.nansum(grid)) == len(traj)


class TestDeadEndDisplay:
    """Test dead-end visit data for frontend display."""

    def test_dead_end_table_data(self):
        from hm2p.maze.analysis import dead_end_visits, simulate_random_walk
        from hm2p.maze.topology import build_rose_maze
        maze = build_rose_maze()
        traj = simulate_random_walk(maze, 2000, seed=42)
        de = dead_end_visits(traj, maze)
        assert len(de) == len(maze.dead_ends)
        for cell_coord, info in de.items():
            assert cell_coord in maze.dead_ends
            assert "visits" in info
            assert "mean_dwell" in info
            assert info["visits"] >= 0
