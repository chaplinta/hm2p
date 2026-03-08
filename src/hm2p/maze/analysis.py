"""Maze behavioural analysis — metrics inspired by Rosenberg et al. (2021).

Rosenberg et al. studied mice in a 6-level binary labyrinth (63 T-junctions,
64 dead ends). This module adapts their key analyses for the hm2p 7×5 rose
maze (8 T-junctions, 6 dead ends, 23 accessible cells).

Adapted metrics:
  - Exploration efficiency (new nodes per window)
  - Occupancy analysis (time per cell/node)
  - Turn bias at T-junctions
  - Monotonic path detection (goal-directed runs)
  - Path efficiency (actual vs optimal path length)
  - Behavioural mode segmentation (explore vs goal-directed)
  - Visit sequence entropy

Reference: Rosenberg, Zhang, Perona & Meister (2021). "Mice in a labyrinth
show rapid learning, sudden insight, and efficient exploration." eLife 10,
e66175. https://doi.org/10.7554/eLife.66175
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from hm2p.maze.topology import RoseMaze, build_rose_maze


# ---------------------------------------------------------------------------
# Occupancy analysis
# ---------------------------------------------------------------------------


def cell_occupancy(
    cell_indices: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    """Count frames spent in each cell.

    Args:
        cell_indices: (N,) int array of cell indices (-1 = invalid).
        n_cells: total number of cells in maze.

    Returns:
        (n_cells,) int array of frame counts per cell.
    """
    counts = np.zeros(n_cells, dtype=np.int64)
    valid = cell_indices[cell_indices >= 0]
    if len(valid) > 0:
        np.add.at(counts, valid, 1)
    return counts


def occupancy_fraction(
    cell_indices: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    """Fraction of time spent in each cell (normalized occupancy).

    Returns (n_cells,) float array summing to 1.0 (excluding invalid frames).
    """
    counts = cell_occupancy(cell_indices, n_cells)
    total = counts.sum()
    if total == 0:
        return np.zeros(n_cells, dtype=np.float64)
    return counts / total


# ---------------------------------------------------------------------------
# Exploration efficiency (Rosenberg NewNodes4)
# ---------------------------------------------------------------------------


def exploration_efficiency(
    node_seq: np.ndarray,
    window_sizes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute new distinct nodes per sliding window of visits.

    Measures how efficiently the animal explores the maze. Higher values
    at small windows indicate more diverse exploration.

    Inspired by Rosenberg et al. NewNodes4.

    Args:
        node_seq: (M,) int array of node indices visited (from node_sequence).
        window_sizes: array of window sizes to evaluate. If None, uses
            log-spaced sizes from 2 to len(node_seq).

    Returns:
        window_sizes: (K,) array of window sizes used.
        new_nodes: (K,) float array — mean distinct nodes per window.
    """
    n = len(node_seq)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    if window_sizes is None:
        # Log-spaced window sizes
        sizes = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
        window_sizes = np.array([s for s in sizes if s <= n] + [n])

    results = []
    for w in window_sizes:
        w = int(w)
        if w > n:
            break
        # Sliding windows
        counts = []
        step = max(1, (n - w) // max(1, n // w))
        for start in range(0, n - w + 1, step):
            window = node_seq[start : start + w]
            counts.append(len(set(window.tolist())))
        results.append(np.mean(counts))

    used = window_sizes[: len(results)]
    return used, np.array(results, dtype=np.float64)


# ---------------------------------------------------------------------------
# Turn bias at T-junctions
# ---------------------------------------------------------------------------


def classify_turn(
    prev_cell: tuple[int, int],
    junction: tuple[int, int],
    next_cell: tuple[int, int],
) -> str:
    """Classify a turn at a T-junction.

    Returns one of:
        "left", "right", "back" — relative to direction of approach.
    """
    # Direction of approach
    dx_in = junction[0] - prev_cell[0]
    dy_in = junction[1] - prev_cell[1]

    # Direction of departure
    dx_out = next_cell[0] - junction[0]
    dy_out = next_cell[1] - junction[1]

    # Cross product to determine turn direction
    # cross > 0 → left turn, cross < 0 → right turn, cross == 0 → straight/back
    cross = dx_in * dy_out - dy_in * dx_out

    # Dot product to check forward/back
    dot = dx_in * dx_out + dy_in * dy_out

    if dot < 0:
        return "back"
    elif cross > 0:
        return "left"
    elif cross < 0:
        return "right"
    else:
        # Straight through (forward) — treat as continuation
        return "forward"


def turn_bias(
    cell_seq: np.ndarray,
    maze: RoseMaze,
) -> dict[str, float]:
    """Compute global turn bias at all T-junctions.

    Returns dict with keys: "left", "right", "back", "forward", "left_frac".
    """
    counts: dict[str, int] = {"left": 0, "right": 0, "back": 0, "forward": 0}
    cell_list = maze.cell_list

    # Get set of junction indices
    junction_indices = {maze.cell_to_idx[j] for j in maze.junctions}

    for i in range(1, len(cell_seq) - 1):
        if cell_seq[i] in junction_indices:
            prev = cell_list[cell_seq[i - 1]]
            curr = cell_list[cell_seq[i]]
            nxt = cell_list[cell_seq[i + 1]]
            turn = classify_turn(prev, curr, nxt)
            counts[turn] += 1

    total_lr = counts["left"] + counts["right"]
    result = {
        "left": float(counts["left"]),
        "right": float(counts["right"]),
        "back": float(counts["back"]),
        "forward": float(counts["forward"]),
        "left_frac": counts["left"] / total_lr if total_lr > 0 else 0.5,
    }
    return result


def per_junction_turn_bias(
    cell_seq: np.ndarray,
    maze: RoseMaze,
) -> dict[tuple[int, int], dict[str, int]]:
    """Compute turn bias at each individual T-junction.

    Returns dict[junction_cell → {"left": n, "right": n, "back": n, "forward": n}].
    """
    cell_list = maze.cell_list
    junction_indices = {maze.cell_to_idx[j] for j in maze.junctions}

    result: dict[tuple[int, int], dict[str, int]] = {
        j: {"left": 0, "right": 0, "back": 0, "forward": 0} for j in maze.junctions
    }

    for i in range(1, len(cell_seq) - 1):
        if cell_seq[i] in junction_indices:
            prev = cell_list[cell_seq[i - 1]]
            curr = cell_list[cell_seq[i]]
            nxt = cell_list[cell_seq[i + 1]]
            turn = classify_turn(prev, curr, nxt)
            result[curr][turn] += 1

    return result


# ---------------------------------------------------------------------------
# Monotonic path detection (Rosenberg FindPathsToNode)
# ---------------------------------------------------------------------------


def find_monotonic_paths(
    node_seq: np.ndarray,
    node_times: np.ndarray,
    target_idx: int,
    maze: RoseMaze,
) -> list[dict]:
    """Find monotonic paths toward a target node.

    A monotonic path is a sequence where the graph distance to the target
    decreases at every step. This detects goal-directed runs.

    Inspired by Rosenberg et al. FindPathsToNode.

    Args:
        node_seq: (M,) int array of node indices visited.
        node_times: (M,) int array of frame times.
        target_idx: cell index of the target node.
        maze: RoseMaze instance.

    Returns:
        List of dicts with keys:
            "start_idx": index into node_seq where path begins.
            "end_idx": index where target is reached.
            "length": number of steps (nodes visited).
            "start_time": frame number at path start.
            "end_time": frame number at path end.
            "optimal_length": shortest possible path length.
            "efficiency": optimal_length / actual_length.
    """
    if len(node_seq) == 0:
        return []

    # Distance from each cell to target
    target_dist = maze.dist[target_idx]

    paths = []
    # Find all visits to the target
    target_visits = np.flatnonzero(node_seq == target_idx)

    for j in target_visits:
        # Trace backward as long as distance increases
        k = j - 1
        while k >= 0 and target_dist[node_seq[k]] > target_dist[node_seq[k + 1]]:
            k -= 1
        k += 1  # first node in this monotonic path

        if k < j:  # at least 2 nodes
            start_cell = maze.cell_list[node_seq[k]]
            opt_len = maze.distance(start_cell, maze.cell_list[target_idx])
            actual_len = j - k
            paths.append({
                "start_idx": k,
                "end_idx": j,
                "length": actual_len,
                "start_time": int(node_times[k]),
                "end_time": int(node_times[j]),
                "optimal_length": opt_len,
                "efficiency": opt_len / actual_len if actual_len > 0 else 1.0,
            })

    return paths


# ---------------------------------------------------------------------------
# Path efficiency
# ---------------------------------------------------------------------------


def path_efficiency_over_time(
    node_seq: np.ndarray,
    node_times: np.ndarray,
    maze: RoseMaze,
    window: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute path efficiency in sliding windows.

    Efficiency = optimal_distance / actual_steps for consecutive pairs.

    Args:
        node_seq: (M,) int array of node indices.
        node_times: (M,) int array of frame times.
        maze: RoseMaze instance.
        window: number of transitions per window.

    Returns:
        times: (K,) midpoint frame times.
        efficiency: (K,) mean efficiency per window.
    """
    if len(node_seq) < 2:
        return np.array([]), np.array([])

    # Pairwise optimal distances and actual steps
    n = len(node_seq)
    efficiencies = []
    mid_times = []

    for start in range(0, n - window, max(1, window // 2)):
        end = min(start + window, n)
        seg = node_seq[start:end]

        actual_steps = len(seg) - 1
        # Optimal = shortest path from first to last node
        opt = maze.dist[seg[0], seg[-1]]
        if actual_steps > 0 and opt >= 0:
            efficiencies.append(opt / actual_steps)
            mid_times.append((node_times[start] + node_times[min(end - 1, n - 1)]) / 2)

    return np.array(mid_times), np.array(efficiencies)


# ---------------------------------------------------------------------------
# Behavioural mode segmentation (Rosenberg SplitModeClips)
# ---------------------------------------------------------------------------


def segment_modes(
    node_seq: np.ndarray,
    node_times: np.ndarray,
    maze: RoseMaze,
    dead_end_targets: list[int] | None = None,
) -> list[dict]:
    """Segment trajectory into goal-directed and exploratory modes.

    Adapted from Rosenberg et al. SplitModeClips. Since the rose maze has
    no water reward, we detect goal-directed runs toward any dead end and
    classify everything else as exploration.

    Modes:
        "directed" — monotonically approaching a dead end
        "explore" — all other movement

    Args:
        node_seq: (M,) int array of node indices.
        node_times: (M,) int array of frame times.
        maze: RoseMaze instance.
        dead_end_targets: list of cell indices to use as targets.
            Defaults to all dead ends.

    Returns:
        List of dicts with keys: "mode", "start_idx", "end_idx",
        "start_time", "end_time", "target" (for directed).
    """
    if len(node_seq) < 2:
        return []

    if dead_end_targets is None:
        dead_end_targets = [maze.cell_to_idx[c] for c in maze.dead_ends]

    n = len(node_seq)
    labeled = np.full(n, -1, dtype=np.int32)  # -1 = unlabeled

    # Find monotonic paths to each dead end
    for target in dead_end_targets:
        paths = find_monotonic_paths(node_seq, node_times, target, maze)
        for p in paths:
            for idx in range(p["start_idx"], p["end_idx"] + 1):
                if labeled[idx] == -1:
                    labeled[idx] = target

    # Build segments
    segments = []
    i = 0
    while i < n:
        if labeled[i] >= 0:
            # Start of directed segment
            target = labeled[i]
            start = i
            while i < n and labeled[i] == target:
                i += 1
            segments.append({
                "mode": "directed",
                "start_idx": start,
                "end_idx": i - 1,
                "start_time": int(node_times[start]),
                "end_time": int(node_times[i - 1]),
                "target": int(target),
            })
        else:
            # Explore segment
            start = i
            while i < n and labeled[i] == -1:
                i += 1
            segments.append({
                "mode": "explore",
                "start_idx": start,
                "end_idx": i - 1,
                "start_time": int(node_times[start]),
                "end_time": int(node_times[i - 1]),
                "target": -1,
            })

    return segments


# ---------------------------------------------------------------------------
# Visit sequence entropy (Rosenberg StringEntropy)
# ---------------------------------------------------------------------------


def sequence_entropy(
    node_seq: np.ndarray,
    max_context: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute conditional entropy of node visit sequence.

    H(X_n | X_1, ..., X_{n-1}) for increasing context lengths.
    Lower entropy = more predictable navigation.

    Inspired by Rosenberg et al. StringEntropy.

    Args:
        node_seq: (M,) int array of node indices visited.
        max_context: maximum context length to evaluate.

    Returns:
        context_lengths: (K,) array of context lengths [1, ..., max_context].
        conditional_entropy: (K,) bits per node visit.
    """
    n = len(node_seq)
    if n < 2:
        return np.array([1]), np.array([0.0])

    max_context = min(max_context, n)
    context_lengths = np.arange(1, max_context + 1)
    cond_entropy = np.zeros(max_context)

    for k_idx, k in enumerate(context_lengths):
        if k >= n:
            cond_entropy[k_idx] = 0.0
            continue

        # Count k-grams and (k-1)-grams
        k_grams: dict[tuple, int] = Counter()
        k_minus_1: dict[tuple, int] = Counter()

        for i in range(n - k):
            kg = tuple(node_seq[i : i + k + 1].tolist())
            k_grams[kg] += 1
            km = tuple(node_seq[i : i + k].tolist())
            k_minus_1[km] += 1

        # H(X_{k+1} | X_1,...,X_k) = H(X_1,...,X_{k+1}) - H(X_1,...,X_k)
        total = sum(k_grams.values())
        if total == 0:
            cond_entropy[k_idx] = 0.0
            continue

        h_joint = 0.0
        for count in k_grams.values():
            p = count / total
            if p > 0:
                h_joint -= p * np.log2(p)

        h_history = 0.0
        total_hist = sum(k_minus_1.values())
        for count in k_minus_1.values():
            p = count / total_hist
            if p > 0:
                h_history -= p * np.log2(p)

        cond_entropy[k_idx] = max(0.0, h_joint - h_history)

    return context_lengths, cond_entropy


# ---------------------------------------------------------------------------
# First-order Markov transition model (Rosenberg FirstTransProb)
# ---------------------------------------------------------------------------


def transition_matrix(
    cell_seq: np.ndarray,
    n_cells: int,
    pseudocount: float = 0.0,
) -> np.ndarray:
    """Compute first-order transition probability matrix.

    P[i, j] = probability of moving to cell j given currently at cell i.

    Inspired by Rosenberg et al. FirstTransProb.

    Args:
        cell_seq: (M,) int array of cell indices (no consecutive duplicates).
        n_cells: total number of cells.
        pseudocount: additive smoothing (0 = MLE, >0 = Laplace).

    Returns:
        (n_cells, n_cells) float array — row-stochastic transition matrix.
    """
    counts = np.full((n_cells, n_cells), pseudocount)
    for i in range(len(cell_seq) - 1):
        a, b = cell_seq[i], cell_seq[i + 1]
        if 0 <= a < n_cells and 0 <= b < n_cells:
            counts[a, b] += 1

    # Normalize rows
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    return counts / row_sums


def transition_entropy(
    trans_mat: np.ndarray,
    cell_seq: np.ndarray,
) -> float:
    """Compute average transition entropy (bits per step).

    Weighted by empirical state occupancy.

    Args:
        trans_mat: (N, N) transition probability matrix.
        cell_seq: (M,) cell sequence (for occupancy weighting).

    Returns:
        Average conditional entropy H(X_{t+1} | X_t) in bits.
    """
    n = trans_mat.shape[0]
    # Empirical state distribution
    counts = np.zeros(n)
    for c in cell_seq:
        if 0 <= c < n:
            counts[c] += 1
    total = counts.sum()
    if total == 0:
        return 0.0
    pi = counts / total

    # Per-state entropy
    h = 0.0
    for i in range(n):
        if pi[i] > 0:
            row = trans_mat[i]
            for p in row:
                if p > 0:
                    h -= pi[i] * p * np.log2(p)
    return float(h)


def cross_entropy(
    cell_seq: np.ndarray,
    trans_mat: np.ndarray,
    n_cells: int,
) -> float:
    """Compute cross-entropy of a sequence under a transition model.

    Lower = better fit. Used for model evaluation (Rosenberg style).

    Args:
        cell_seq: (M,) test sequence.
        trans_mat: (N, N) transition matrix (from training data).
        n_cells: total number of cells.

    Returns:
        Cross-entropy in bits per step.
    """
    log_prob = 0.0
    n_transitions = 0
    for i in range(len(cell_seq) - 1):
        a, b = cell_seq[i], cell_seq[i + 1]
        if 0 <= a < n_cells and 0 <= b < n_cells:
            p = trans_mat[a, b]
            if p > 0:
                log_prob -= np.log2(p)
            else:
                log_prob += 20  # Penalty for impossible transition
            n_transitions += 1

    return log_prob / n_transitions if n_transitions > 0 else 0.0


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def maze_exploration_summary(
    cell_indices: np.ndarray,
    maze: RoseMaze | None = None,
    fps: float = 30.0,
) -> dict:
    """Compute comprehensive maze exploration summary statistics.

    Args:
        cell_indices: (N,) int array from discretize_position.
        maze: RoseMaze instance.
        fps: frame rate for time conversion.

    Returns:
        Dict with summary statistics.
    """
    if maze is None:
        maze = build_rose_maze()

    from hm2p.maze.discretize import cell_sequence, node_sequence

    cells_visited, cell_times = cell_sequence(cell_indices)
    nodes_visited, node_times = node_sequence(cell_indices, maze)

    # Basic counts
    valid_frames = np.sum(cell_indices >= 0)
    unique_cells = len(set(cells_visited.tolist())) if len(cells_visited) > 0 else 0
    unique_nodes = len(set(nodes_visited.tolist())) if len(nodes_visited) > 0 else 0

    # Occupancy
    occ = occupancy_fraction(cell_indices, maze.n_cells)

    # Coverage over time
    coverage = []
    seen = set()
    for i, c in enumerate(cells_visited):
        seen.add(int(c))
        if i % max(1, len(cells_visited) // 100) == 0:
            coverage.append(len(seen) / maze.n_cells)

    # Turn bias
    tb = turn_bias(cells_visited, maze) if len(cells_visited) > 2 else {
        "left": 0, "right": 0, "back": 0, "forward": 0, "left_frac": 0.5,
    }

    return {
        "total_frames": len(cell_indices),
        "valid_frames": int(valid_frames),
        "duration_s": len(cell_indices) / fps,
        "unique_cells_visited": unique_cells,
        "coverage_frac": unique_cells / maze.n_cells,
        "unique_nodes_visited": unique_nodes,
        "n_cell_transitions": len(cells_visited),
        "n_node_transitions": len(nodes_visited),
        "occupancy_entropy": float(-np.sum(occ[occ > 0] * np.log2(occ[occ > 0]))),
        "max_occupancy_cell": int(np.argmax(occ)) if valid_frames > 0 else -1,
        "turn_bias": tb,
    }
