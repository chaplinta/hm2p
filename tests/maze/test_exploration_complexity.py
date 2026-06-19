"""Tests for maze exploration-complexity measures. Synthetic inputs only."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.maze.exploration_complexity import (
    build_adjacency_indices,
    coverage_z_vs_null,
    lz76_complexity,
    normalized_lz76,
    occupancy_entropy,
    random_walk_coverage_null,
)
from hm2p.maze.topology import build_rose_maze


# ---------------------------------------------------------------------------
# occupancy_entropy
# ---------------------------------------------------------------------------


def test_entropy_uniform_is_log2_k():
    # 4 cells each occupied equally -> 2 bits
    ci = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    assert np.isclose(occupancy_entropy(ci), 2.0)


def test_entropy_single_cell_is_zero():
    assert occupancy_entropy(np.array([5, 5, 5, 5])) == 0.0


def test_entropy_ignores_invalid_and_empty():
    assert occupancy_entropy(np.array([-1, -1])) == 0.0
    # invalid frames dropped: two cells equally -> 1 bit
    assert np.isclose(occupancy_entropy(np.array([-1, 0, 1, -1])), 1.0)


def test_entropy_concentrated_lt_uniform():
    uniform = np.array([0, 1, 2, 3] * 5)
    skewed = np.array([0] * 17 + [1, 2, 3])
    assert occupancy_entropy(skewed) < occupancy_entropy(uniform)


# ---------------------------------------------------------------------------
# lz76_complexity
# ---------------------------------------------------------------------------


def test_lz76_empty_is_zero():
    assert lz76_complexity([]) == 0


def test_lz76_hand_traced_values():
    # Hand-traced against the Kaspar-Schuster parsing.
    assert lz76_complexity([0, 0, 0, 0]) == 2
    assert lz76_complexity([0, 0, 0, 1]) == 3


def test_lz76_repetitive_lt_varied():
    periodic = [0, 1] * 25
    rng = np.random.default_rng(0)
    varied = rng.integers(0, 6, size=50).tolist()
    assert lz76_complexity(periodic) < lz76_complexity(varied)


def test_normalized_lz_single_symbol_zero():
    assert normalized_lz76([3, 3, 3, 3, 3]) == 0.0


def test_normalized_lz_periodic_lt_random():
    periodic = [0, 1, 2] * 20
    rng = np.random.default_rng(1)
    rand = rng.integers(0, 3, size=60).tolist()
    assert normalized_lz76(periodic) < normalized_lz76(rand)


# ---------------------------------------------------------------------------
# random-walk null
# ---------------------------------------------------------------------------


def test_adjacency_indices_match_graph():
    maze = build_rose_maze()
    adj_idx = build_adjacency_indices(maze)
    assert len(adj_idx) == len(maze.cell_list)
    # neighbour count per cell matches the graph degree
    for i, cell in enumerate(maze.cell_list):
        assert adj_idx[i].size == len(maze.adj[cell])


def test_random_walk_counts_in_range():
    maze = build_rose_maze()
    adj_idx = build_adjacency_indices(maze)
    rng = np.random.default_rng(2)
    counts = random_walk_coverage_null(adj_idx, start_idx=0, n_steps=20, n_sims=50, rng=rng)
    assert counts.shape == (50,)
    assert counts.min() >= 1
    assert counts.max() <= maze.n_cells


def test_random_walk_more_steps_covers_more():
    maze = build_rose_maze()
    adj_idx = build_adjacency_indices(maze)
    rng = np.random.default_rng(3)
    few = random_walk_coverage_null(adj_idx, 0, n_steps=3, n_sims=200, rng=rng)
    many = random_walk_coverage_null(adj_idx, 0, n_steps=40, n_sims=200, rng=rng)
    assert np.mean(many) > np.mean(few)


def test_coverage_z_sign_and_nan():
    null = np.array([5, 6, 7, 6, 5, 7])
    assert coverage_z_vs_null(10, null) > 0
    assert coverage_z_vs_null(2, null) < 0
    assert np.isnan(coverage_z_vs_null(6, np.array([4, 4, 4])))


def test_coverage_z_empty_null_is_nan():
    assert np.isnan(coverage_z_vs_null(5, np.array([])))
