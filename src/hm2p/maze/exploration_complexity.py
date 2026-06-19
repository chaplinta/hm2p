"""Exploration-complexity measures for maze trajectories.

These supplement the simple per-epoch coverage measure (unique cells visited)
with measures that are sensitive to time distribution, route structure, and the
maze topology, so that a light/dark difference can be attributed to exploration
rather than to how much the animal moved:

- ``occupancy_entropy``: Shannon entropy of the time-per-cell distribution.
  Uniform exploration is high; dwelling on a few cells is low.
- ``lz76_complexity`` / ``normalized_lz76``: Lempel-Ziv complexity of the
  cell-visit sequence. Stereotyped, repeated routes compress more (lower
  complexity) than varied routes of the same length.
- ``random_walk_coverage_null`` / ``coverage_z_vs_null``: coverage benchmarked
  against uniform random walks of the same number of cell-transitions on the
  maze graph, so reduced coverage can be compared with what a memoryless walker
  would achieve given the same amount of movement and the same topology.

References
----------
Lempel & Ziv 1976. "On the complexity of finite sequences." IEEE Trans. Inf.
    Theory 22(1):75-81. doi:10.1109/TIT.1976.1055501
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def occupancy_entropy(
    cell_indices: npt.NDArray[np.integer],
    base: float = 2.0,
) -> float:
    """Shannon entropy of the cell-occupancy distribution.

    Parameters
    ----------
    cell_indices : (n,) int
        Per-frame maze-cell index; negative values (e.g. -1) are treated as
        invalid and ignored.
    base : float
        Log base (2 → bits).

    Returns
    -------
    float
        Entropy in the chosen base. 0 if no valid frames or all time in one cell.
    """
    ci = np.asarray(cell_indices)
    ci = ci[ci >= 0]
    if ci.size == 0:
        return 0.0
    counts = np.bincount(ci)
    p = counts[counts > 0] / ci.size
    return float(-np.sum(p * (np.log(p) / np.log(base))))


def lz76_complexity(sequence: npt.ArrayLike) -> int:
    """Lempel-Ziv (1976) complexity of a symbol sequence.

    Counts the number of distinct patterns produced when the sequence is parsed
    left to right (the LZ76 ``c`` measure; Kaspar & Schuster 1987 formulation).
    A more stereotyped / repetitive sequence yields a lower count.

    Parameters
    ----------
    sequence : array-like
        Sequence of hashable symbols (e.g. maze-cell indices).

    Returns
    -------
    int
        LZ76 complexity (>= 0; 0 for an empty sequence).
    """
    s = list(sequence)
    n = len(s)
    if n == 0:
        return 0
    c = 1
    ell = 1
    i = 0
    k = 1
    k_max = 1
    while True:
        if ell + k > n:
            c += 1
            break
        if s[i + k - 1] == s[ell + k - 1]:
            k += 1
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == ell:
                c += 1
                ell += k_max
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    return c


def normalized_lz76(sequence: npt.ArrayLike) -> float:
    """LZ76 complexity normalised by the random-sequence upper bound.

    For a sequence of length ``n`` over an alphabet of ``a`` symbols, the LZ76
    complexity of a random sequence grows as ``n / log_a(n)``; dividing by that
    bound gives a length- and alphabet-comparable value (≈1 for random,
    lower for stereotyped). Returns 0 for sequences too short to normalise.
    """
    s = list(sequence)
    n = len(s)
    if n < 2:
        return 0.0
    a = len(set(s))
    if a < 2:
        return 0.0  # single symbol → maximally compressible
    c = lz76_complexity(s)
    bound = n / (np.log(n) / np.log(a))
    return float(c / bound) if bound > 0 else 0.0


def build_adjacency_indices(maze) -> list[npt.NDArray[np.int_]]:
    """Neighbour indices per cell, indexed by position in ``maze.cell_list``.

    Returns a list where entry ``i`` is an int array of the indices of the
    accessible neighbours of ``maze.cell_list[i]``.
    """
    adj_idx: list[npt.NDArray[np.int_]] = []
    for cell in maze.cell_list:
        neigh = [maze.cell_to_idx[nb] for nb in maze.adj.get(cell, [])]
        adj_idx.append(np.asarray(neigh, dtype=int))
    return adj_idx


def random_walk_coverage_null(
    adj_idx: list[npt.NDArray[np.int_]],
    start_idx: int,
    n_steps: int,
    n_sims: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.int_]:
    """Unique-cell counts from uniform random walks on the maze graph.

    Each simulation starts at ``start_idx`` and takes ``n_steps`` cell-to-cell
    moves, each to a uniformly chosen accessible neighbour, counting the unique
    cells visited (including the start). This is the memoryless-walker
    null for "how many cells would you cover in this many cell-steps on this
    maze".

    Returns
    -------
    (n_sims,) int
        Unique-cell count per simulation.
    """
    counts = np.empty(n_sims, dtype=int)
    for s in range(n_sims):
        cur = int(start_idx)
        seen = {cur}
        for _ in range(n_steps):
            neigh = adj_idx[cur]
            if neigh.size == 0:
                break
            cur = int(neigh[rng.integers(neigh.size)])
            seen.add(cur)
        counts[s] = len(seen)
    return counts


def coverage_z_vs_null(observed_unique: int, null_counts: npt.NDArray[np.integer]) -> float:
    """Z-score of observed unique-cell count against a random-walk null.

    Positive → the animal covered more unique cells than a memoryless walker of
    the same step count; negative → fewer (more confined / repetitive). Returns
    ``nan`` if the null has zero spread.
    """
    null = np.asarray(null_counts, dtype=np.float64)
    if null.size == 0:
        return float("nan")
    sd = float(np.std(null))
    if sd == 0:
        return float("nan")
    return float((observed_unique - float(np.mean(null))) / sd)
