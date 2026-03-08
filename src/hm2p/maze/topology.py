"""Rose maze topology — graph representation of the 7×5 rose maze.

The maze is a 7×5 unit grid with internal walls creating corridors.
This module defines:
  - The corridor structure as a set of accessible 1×1 cells
  - A graph where nodes are cell centers and edges connect adjacent cells
  - Junction classification (dead-end, corridor, T-junction, crossroads)
  - Shortest-path distances between all cells

Inspired by Rosenberg et al. (2021) eLife, adapted for the hm2p rose maze.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Maze polygon (same as in kinematics/compute.py)
# ---------------------------------------------------------------------------
MAZE_POLYGON_COORDS: list[tuple[int, int]] = [
    (0, 0), (3, 0), (3, 1), (2, 1), (2, 2), (5, 2), (5, 1), (4, 1),
    (4, 0), (7, 0), (7, 1), (6, 1), (6, 4), (7, 4), (7, 5), (4, 5),
    (4, 4), (5, 4), (5, 3), (4, 3), (4, 5), (3, 5), (3, 3), (2, 3),
    (2, 4), (3, 4), (3, 5), (0, 5), (0, 4), (1, 4), (1, 1), (0, 1),
]

# ---------------------------------------------------------------------------
# Accessible cells — manually enumerated from the polygon
# ---------------------------------------------------------------------------
# Each cell is identified by its (col, row) integer coordinates where
# the cell spans [col, col+1) × [row, row+1).  The cell center is at
# (col+0.5, row+0.5).
#
# The rose maze corridor layout (y increases upward, shown as grid):
#
#   Row 4: [0,4] [_,_] [2,4] [3,4] [_,_] [5,4] [6,4]
#   Row 3: [0,3] [_,_] [2,3] [3,3] [_,_] [5,3] [6,3]
#   Row 2: [0,2] [_,_] [2,2] [3,2] [4,2] [5,2] [6,2]
#   Row 1: [0,1] [_,_] [2,1] [3,1] [4,1] [5,1] [6,1]
#   Row 0: [0,0] [1,0] [2,0] [_,_] [4,0] [5,0] [6,0]
#
# [_,_] = wall (inaccessible)

def _compute_accessible_cells() -> set[tuple[int, int]]:
    """Compute the set of accessible cells using the maze polygon.

    Uses shapely for point-in-polygon test on cell centers.
    Falls back to a hardcoded set if shapely is not available.
    """
    try:
        from shapely.geometry import Point, Polygon
        poly = Polygon(MAZE_POLYGON_COORDS)
        cells = set()
        for col in range(7):
            for row in range(5):
                center = Point(col + 0.5, row + 0.5)
                if poly.contains(center):
                    cells.add((col, row))
        return cells
    except ImportError:
        return _HARDCODED_CELLS.copy()


# Hardcoded fallback (computed once from shapely, verified)
# Grid layout (O = accessible, . = wall):
#   y=4   O O O O O O O   (top row: full 7-cell corridor)
#   y=3   . O . O . O .   (3 vertical pillars)
#   y=2   . O O O O O .   (central horizontal corridor)
#   y=1   . O . . . O .   (2 vertical corridors)
#   y=0   O O O . O O O   (bottom: two 3-cell arms, gap at col 3)
#         0 1 2 3 4 5 6
_HARDCODED_CELLS: set[tuple[int, int]] = {
    # Row 0 (bottom): left arm [0-2] + right arm [4-6], wall at col 3
    (0, 0), (1, 0), (2, 0),
    (4, 0), (5, 0), (6, 0),
    # Row 1: two vertical corridors at x=1 and x=5
    (1, 1), (5, 1),
    # Row 2: central horizontal corridor x=1..5
    (1, 2), (2, 2), (3, 2), (4, 2), (5, 2),
    # Row 3: three vertical pillars at x=1, x=3, x=5
    (1, 3), (3, 3), (5, 3),
    # Row 4 (top): full corridor x=0..6
    (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4),
}


def get_accessible_cells() -> set[tuple[int, int]]:
    """Return the set of accessible (col, row) cells in the maze."""
    return _compute_accessible_cells()


# ---------------------------------------------------------------------------
# Graph adjacency
# ---------------------------------------------------------------------------

# Cardinal directions: (dcol, drow)
_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def build_adjacency(cells: set[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Build adjacency dict: cell → list of accessible neighbours.

    Two cells are adjacent if they share an edge (4-connected).
    """
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {c: [] for c in cells}
    for c in cells:
        for dc, dr in _DIRECTIONS:
            nb = (c[0] + dc, c[1] + dr)
            if nb in cells:
                adj[c].append(nb)
    return adj


# ---------------------------------------------------------------------------
# Junction classification
# ---------------------------------------------------------------------------

def classify_nodes(
    adj: dict[tuple[int, int], list[tuple[int, int]]],
) -> dict[tuple[int, int], str]:
    """Classify each cell by its connectivity.

    Returns dict[cell → type] where type is one of:
        "dead_end"   — 1 neighbour
        "corridor"   — 2 neighbours
        "t_junction" — 3 neighbours
        "crossroads" — 4 neighbours
    """
    labels = {}
    for cell, neighbours in adj.items():
        n = len(neighbours)
        if n <= 1:
            labels[cell] = "dead_end"
        elif n == 2:
            labels[cell] = "corridor"
        elif n == 3:
            labels[cell] = "t_junction"
        else:
            labels[cell] = "crossroads"
    return labels


# ---------------------------------------------------------------------------
# Shortest-path distances (BFS on unweighted graph)
# ---------------------------------------------------------------------------

def compute_distances(
    cells: set[tuple[int, int]],
    adj: dict[tuple[int, int], list[tuple[int, int]]],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Compute all-pairs shortest-path distances.

    Returns:
        dist: (N, N) int array of shortest path distances.
        cell_list: ordered list of cells matching dist indices.
    """
    cell_list = sorted(cells)
    idx = {c: i for i, c in enumerate(cell_list)}
    n = len(cell_list)
    dist = np.full((n, n), -1, dtype=np.int32)

    for start in cell_list:
        si = idx[start]
        dist[si, si] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            curr = queue[head]
            head += 1
            ci = idx[curr]
            for nb in adj[curr]:
                ni = idx[nb]
                if dist[si, ni] == -1:
                    dist[si, ni] = dist[si, ci] + 1
                    queue.append(nb)

    return dist, cell_list


def shortest_path(
    start: tuple[int, int],
    end: tuple[int, int],
    adj: dict[tuple[int, int], list[tuple[int, int]]],
) -> list[tuple[int, int]] | None:
    """Find shortest path between two cells (BFS).

    Returns list of cells from start to end inclusive, or None if unreachable.
    """
    if start == end:
        return [start]
    visited = {start}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [start]
    head = 0
    while head < len(queue):
        curr = queue[head]
        head += 1
        for nb in adj[curr]:
            if nb not in visited:
                visited.add(nb)
                parent[nb] = curr
                if nb == end:
                    path = [end]
                    c = end
                    while c != start:
                        c = parent[c]
                        path.append(c)
                    return path[::-1]
                queue.append(nb)
    return None


# ---------------------------------------------------------------------------
# RoseMaze dataclass — complete topology
# ---------------------------------------------------------------------------

@dataclass
class RoseMaze:
    """Complete rose maze topology.

    Attributes:
        cells: set of accessible (col, row) cells.
        adj: adjacency dict.
        node_types: dict[cell → junction type].
        dist: (N, N) shortest-path distance matrix.
        cell_list: ordered list of cells (index matches dist).
        cell_to_idx: dict[cell → index in cell_list].
        junctions: list of cells that are T-junctions or crossroads.
        dead_ends: list of cells that are dead ends.
    """

    cells: set[tuple[int, int]]
    adj: dict[tuple[int, int], list[tuple[int, int]]]
    node_types: dict[tuple[int, int], str]
    dist: np.ndarray
    cell_list: list[tuple[int, int]]
    cell_to_idx: dict[tuple[int, int], int] = field(init=False)
    junctions: list[tuple[int, int]] = field(init=False)
    dead_ends: list[tuple[int, int]] = field(init=False)
    corridors: list[tuple[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.cell_to_idx = {c: i for i, c in enumerate(self.cell_list)}
        self.junctions = [c for c, t in self.node_types.items() if t in ("t_junction", "crossroads")]
        self.dead_ends = [c for c, t in self.node_types.items() if t == "dead_end"]
        self.corridors = [c for c, t in self.node_types.items() if t == "corridor"]

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def distance(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        """Shortest-path distance between two cells."""
        return int(self.dist[self.cell_to_idx[a], self.cell_to_idx[b]])

    def path(self, a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
        """Shortest path between two cells."""
        return shortest_path(a, b, self.adj)


def build_rose_maze() -> RoseMaze:
    """Construct the complete rose maze topology."""
    cells = get_accessible_cells()
    adj = build_adjacency(cells)
    node_types = classify_nodes(adj)
    dist, cell_list = compute_distances(cells, adj)
    return RoseMaze(
        cells=cells,
        adj=adj,
        node_types=node_types,
        dist=dist,
        cell_list=cell_list,
    )
