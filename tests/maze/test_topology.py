"""Tests for maze topology module."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.maze.topology import (
    RoseMaze,
    _HARDCODED_CELLS,
    build_adjacency,
    build_rose_maze,
    classify_nodes,
    compute_distances,
    get_accessible_cells,
    shortest_path,
)


class TestAccessibleCells:
    """Test maze cell enumeration."""

    def test_cell_count(self):
        cells = get_accessible_cells()
        assert len(cells) == 23

    def test_hardcoded_matches_shapely(self):
        """Hardcoded cells must match shapely computation."""
        cells = get_accessible_cells()
        assert cells == _HARDCODED_CELLS

    def test_bottom_row_has_gap_at_col3(self):
        cells = get_accessible_cells()
        assert (3, 0) not in cells
        assert (0, 0) in cells
        assert (4, 0) in cells

    def test_top_row_full(self):
        cells = get_accessible_cells()
        for col in range(7):
            assert (col, 4) in cells

    def test_no_cells_outside_grid(self):
        cells = get_accessible_cells()
        for col, row in cells:
            assert 0 <= col < 7
            assert 0 <= row < 5


class TestAdjacency:
    """Test graph adjacency construction."""

    def test_corner_cell_has_one_neighbour(self):
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        # (0,0) is a dead end at bottom-left
        assert len(adj[(0, 0)]) == 1
        assert (1, 0) in adj[(0, 0)]

    def test_top_row_interior_has_two_neighbours(self):
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        # (2,4) should connect to (1,4) and (3,4)
        assert len(adj[(2, 4)]) == 2

    def test_t_junction_has_three_neighbours(self):
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        # (1,0) connects to (0,0), (2,0), (1,1)
        assert len(adj[(1, 0)]) == 3

    def test_adjacency_is_symmetric(self):
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        for cell, nbs in adj.items():
            for nb in nbs:
                assert cell in adj[nb], f"{cell} in adj[{nb}] but not vice versa"

    def test_no_diagonal_connections(self):
        """Adjacency should be 4-connected only."""
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        for cell, nbs in adj.items():
            for nb in nbs:
                dc = abs(cell[0] - nb[0])
                dr = abs(cell[1] - nb[1])
                assert dc + dr == 1, f"Non-4-connected: {cell} -> {nb}"


class TestClassifyNodes:
    """Test junction classification."""

    def test_dead_end_count(self):
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        types = classify_nodes(adj)
        dead_ends = [c for c, t in types.items() if t == "dead_end"]
        assert len(dead_ends) == 6

    def test_t_junction_count(self):
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        types = classify_nodes(adj)
        junctions = [c for c, t in types.items() if t == "t_junction"]
        assert len(junctions) == 8

    def test_known_dead_ends(self):
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        types = classify_nodes(adj)
        for de in [(0, 0), (2, 0), (4, 0), (6, 0), (0, 4), (6, 4)]:
            assert types[de] == "dead_end", f"{de} should be dead_end"

    def test_known_t_junctions(self):
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        types = classify_nodes(adj)
        for tj in [(1, 0), (5, 0), (1, 2), (5, 2), (3, 2), (1, 4), (3, 4), (5, 4)]:
            assert types[tj] == "t_junction", f"{tj} should be t_junction"

    def test_no_crossroads(self):
        """Rose maze has no 4-way intersections."""
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        types = classify_nodes(adj)
        crossroads = [c for c, t in types.items() if t == "crossroads"]
        assert len(crossroads) == 0


class TestDistances:
    """Test shortest-path distance computation."""

    def test_self_distance_zero(self):
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        dist, cell_list = compute_distances(cells, adj)
        for i in range(len(cell_list)):
            assert dist[i, i] == 0

    def test_symmetric(self):
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        dist, _ = compute_distances(cells, adj)
        np.testing.assert_array_equal(dist, dist.T)

    def test_adjacent_cells_distance_one(self):
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        dist, cell_list = compute_distances(cells, adj)
        idx = {c: i for i, c in enumerate(cell_list)}
        # (0,0) and (1,0) are adjacent
        assert dist[idx[(0, 0)], idx[(1, 0)]] == 1

    def test_cross_maze_distance(self):
        """Distance from (0,0) to (6,4) should be 10."""
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        dist, cell_list = compute_distances(cells, adj)
        idx = {c: i for i, c in enumerate(cell_list)}
        d = dist[idx[(0, 0)], idx[(6, 4)]]
        assert d == 10

    def test_no_unreachable_cells(self):
        """All cells should be reachable from all others."""
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        dist, _ = compute_distances(cells, adj)
        assert np.all(dist >= 0)

    def test_triangle_inequality(self):
        """d(a,c) <= d(a,b) + d(b,c) for all triples."""
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        dist, _ = compute_distances(cells, adj)
        n = len(dist)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    assert dist[a, c] <= dist[a, b] + dist[b, c]


class TestShortestPath:
    """Test BFS shortest path."""

    def test_same_cell(self):
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        path = shortest_path((0, 0), (0, 0), adj)
        assert path == [(0, 0)]

    def test_adjacent_cells(self):
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        path = shortest_path((0, 0), (1, 0), adj)
        assert path == [(0, 0), (1, 0)]

    def test_longer_path(self):
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        path = shortest_path((0, 0), (2, 0), adj)
        assert path is not None
        assert len(path) == 3
        assert path[0] == (0, 0)
        assert path[-1] == (2, 0)

    def test_path_length_matches_distance(self):
        cells = get_accessible_cells()
        adj = build_adjacency(cells)
        dist, cell_list = compute_distances(cells, adj)
        idx = {c: i for i, c in enumerate(cell_list)}
        path = shortest_path((0, 0), (6, 4), adj)
        assert path is not None
        assert len(path) - 1 == dist[idx[(0, 0)], idx[(6, 4)]]

    def test_unreachable_returns_none(self):
        # Create disconnected graph
        adj = {(0, 0): [], (5, 5): []}
        path = shortest_path((0, 0), (5, 5), adj)
        assert path is None


class TestRoseMaze:
    """Test the RoseMaze dataclass."""

    @pytest.fixture
    def maze(self):
        return build_rose_maze()

    def test_n_cells(self, maze):
        assert maze.n_cells == 23

    def test_junctions_count(self, maze):
        assert len(maze.junctions) == 8

    def test_dead_ends_count(self, maze):
        assert len(maze.dead_ends) == 6

    def test_corridors_count(self, maze):
        # 23 total - 8 junctions - 6 dead ends = 9 corridors
        assert len(maze.corridors) == 9

    def test_distance_method(self, maze):
        assert maze.distance((0, 0), (1, 0)) == 1
        assert maze.distance((0, 0), (0, 0)) == 0

    def test_path_method(self, maze):
        path = maze.path((0, 0), (6, 4))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (6, 4)

    def test_left_right_symmetry(self, maze):
        """Maze should be approximately symmetric under x → 6-x reflection."""
        # Distance from (0,0) to (0,4) should equal (6,0) to (6,4)
        d1 = maze.distance((0, 0), (0, 4))
        d2 = maze.distance((6, 0), (6, 4))
        assert d1 == d2

    def test_max_distance(self, maze):
        """Maximum distance should be finite and reasonable."""
        max_d = maze.dist.max()
        assert max_d > 0
        assert max_d <= 23  # Can't be longer than total cells
