"""Position discretization — map continuous (x, y) to maze graph cells.

Converts continuous mouse position (in maze coordinates 0–7 × 0–5) to
discrete cell indices on the maze graph.
"""

from __future__ import annotations

import numpy as np

from hm2p.maze.topology import RoseMaze, build_rose_maze


def discretize_position(
    x: np.ndarray,
    y: np.ndarray,
    maze: RoseMaze | None = None,
) -> np.ndarray:
    """Map continuous (x, y) maze coordinates to nearest accessible cell index.

    Args:
        x: (N,) x positions in maze units [0, 7].
        y: (N,) y positions in maze units [0, 5].
        maze: RoseMaze instance (built if not provided).

    Returns:
        (N,) int array of cell indices into maze.cell_list.
        NaN positions map to -1.
    """
    if maze is None:
        maze = build_rose_maze()

    n = len(x)
    result = np.full(n, -1, dtype=np.int32)

    # Cell centers
    centers = np.array([(c[0] + 0.5, c[1] + 0.5) for c in maze.cell_list])

    # Process valid (non-NaN) positions
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return result

    xv = x[valid]
    yv = y[valid]
    valid_idx = np.flatnonzero(valid)

    # Find nearest accessible cell center for each position
    for i in range(len(xv)):
        dx = centers[:, 0] - xv[i]
        dy = centers[:, 1] - yv[i]
        dist = dx * dx + dy * dy
        result[valid_idx[i]] = int(np.argmin(dist))

    return result


def discretize_position_fast(
    x: np.ndarray,
    y: np.ndarray,
    maze: RoseMaze | None = None,
) -> np.ndarray:
    """Vectorized version of discretize_position for large arrays.

    Same interface as discretize_position but uses vectorized operations
    for better performance on long trajectories.
    """
    if maze is None:
        maze = build_rose_maze()

    n = len(x)
    result = np.full(n, -1, dtype=np.int32)

    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return result

    xv = x[valid]
    yv = y[valid]

    # Cell centers as array
    centers = np.array([(c[0] + 0.5, c[1] + 0.5) for c in maze.cell_list])

    # Compute distances from each position to each cell center
    # Shape: (n_valid, n_cells)
    dx = xv[:, None] - centers[None, :, 0]
    dy = yv[:, None] - centers[None, :, 1]
    dist_sq = dx * dx + dy * dy

    # Nearest cell for each position
    nearest = np.argmin(dist_sq, axis=1)
    result[valid] = nearest.astype(np.int32)

    return result


def cell_sequence(
    cell_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compress cell trajectory to sequence of distinct cells visited.

    Removes consecutive duplicates (staying in same cell) and invalid
    indices (-1).

    Args:
        cell_indices: (N,) int array from discretize_position.

    Returns:
        cells: (M,) int array of cell indices visited (no consecutive duplicates).
        times: (M,) int array of frame indices when each cell was first entered.
    """
    valid = cell_indices >= 0
    if not valid.any():
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # Get valid indices and their frame numbers
    valid_idx = np.flatnonzero(valid)
    valid_cells = cell_indices[valid_idx]

    # Find transitions (different from previous)
    changes = np.ones(len(valid_cells), dtype=bool)
    changes[1:] = valid_cells[1:] != valid_cells[:-1]

    return valid_cells[changes], valid_idx[changes]


def node_sequence(
    cell_indices: np.ndarray,
    maze: RoseMaze | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract sequence of junction/dead-end visits from trajectory.

    Like Rosenberg's ParseNodeTrajectory — filters cell sequence to only
    include visits to decision points (T-junctions) and dead ends,
    collapsing corridor transits.

    Args:
        cell_indices: (N,) int array from discretize_position.
        maze: RoseMaze instance.

    Returns:
        nodes: (M,) int array of junction/dead-end cell indices visited.
        times: (M,) int array of frame indices when each node was reached.
    """
    if maze is None:
        maze = build_rose_maze()

    cells, times = cell_sequence(cell_indices)
    if len(cells) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # Set of important nodes (junctions + dead ends)
    important = set()
    for c in maze.junctions:
        important.add(maze.cell_to_idx[c])
    for c in maze.dead_ends:
        important.add(maze.cell_to_idx[c])

    # Filter to important nodes
    is_important = np.array([c in important for c in cells])
    imp_cells = cells[is_important]
    imp_times = times[is_important]

    if len(imp_cells) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # Remove consecutive duplicates
    changes = np.ones(len(imp_cells), dtype=bool)
    changes[1:] = imp_cells[1:] != imp_cells[:-1]

    return imp_cells[changes], imp_times[changes]
