"""Morphology processing: SWC loading, soma subtraction, rotation, and tree statistics.

Reimplements the TREES toolbox operations from the MATLAB pipeline using navis
(with a pure numpy/scipy fallback when navis is unavailable).

MATLAB sources (read-only reference):
    - morphology_readout.m — load SWC, soma subtract, rotate, extract stats
    - stats_tree_sw.m — tree statistics (length, branch points, Sholl, etc.)
    - getSurfDist.m — surface distance computation
    - rotate_tree.m — 2D rotation of tree coordinates
    - cat_tree_sw.m — concatenate multiple trees
    - dissect_tree_sw.m — dissect tree into branches

Citations
---------
navis: Schlegel et al. 2021. "navis: neuron analysis and visualization in
    Python." GitHub: https://github.com/navis-org/navis

TREES toolbox (original MATLAB implementation):
    Cuntz et al. 2010. "One rule to grow them all: a general theory of neuronal
    branching and its practical application." PLoS Comput Biol.
    doi:10.1371/journal.pcbi.1000877

Sholl analysis:
    Sholl 1953. "Dendritic organization in the neurons of the visual and motor
    cortices of the cat." J Anat 87(4):387-406.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import navis; fall back gracefully
# ---------------------------------------------------------------------------
try:
    import navis

    HAS_NAVIS = True
except ImportError:  # pragma: no cover
    HAS_NAVIS = False
    logger.info("navis not installed — using manual SWC fallback parser.")


# ============================================================================
# SWC fallback parser
# ============================================================================

#: Standard SWC column names
_SWC_COLUMNS = ["id", "type", "x", "y", "z", "radius", "parent_id"]


def _load_swc_manual(path: Path) -> dict[str, Any]:
    """Parse an SWC file into a dict with 'nodes' DataFrame and 'edges' array.

    SWC format columns: id, type, x, y, z, radius, parent_id.
    Lines beginning with ``#`` are treated as comments and skipped.

    Parameters
    ----------
    path : Path
        Path to the ``.swc`` file.

    Returns
    -------
    dict
        ``nodes`` — :class:`pandas.DataFrame` with columns
        ``id, type, x, y, z, radius, parent_id``.
        ``edges`` — ``(E, 2)`` int array of ``(parent_id, child_id)`` pairs
        (excluding the root whose parent_id is ``-1``).
    """
    rows: list[list[float]] = []
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 7:
                continue
            rows.append([float(v) for v in parts[:7]])

    if not rows:
        raise ValueError(f"No valid SWC data found in {path}")

    arr = np.array(rows)
    nodes = pd.DataFrame(arr, columns=_SWC_COLUMNS)
    nodes["id"] = nodes["id"].astype(int)
    nodes["type"] = nodes["type"].astype(int)
    nodes["parent_id"] = nodes["parent_id"].astype(int)

    # Build edges: (parent, child) for non-root nodes
    mask = nodes["parent_id"] != -1
    edges = nodes.loc[mask, ["parent_id", "id"]].values.astype(int)

    return {"nodes": nodes, "edges": edges}


# ============================================================================
# Public API
# ============================================================================


def load_morphology(tracing_path: Path) -> dict[str, dict[str, Any]]:
    """Load SWC files from a tracing directory.

    Looks for files named ``Soma.swc``, ``Apical*.swc``, ``Basal*.swc``,
    ``Surface.swc``, and ``Axon.swc`` (case-insensitive).

    Parameters
    ----------
    tracing_path : Path
        Directory containing SWC files.

    Returns
    -------
    dict
        Keyed by type (``'soma'``, ``'apical'``, ``'basal'``, ``'surface'``,
        ``'axon'``).  Each value is a dict with ``'nodes'``
        (:class:`~pandas.DataFrame`) and ``'edges'`` (ndarray).
        ``'basal'`` may contain concatenated data from multiple Basal*.swc
        files.  Keys for types that have no file are omitted.
    """
    tracing_path = Path(tracing_path)
    swc_files = list(tracing_path.glob("*.swc")) + list(tracing_path.glob("*.SWC"))

    result: dict[str, dict[str, Any]] = {}
    basal_parts: list[dict[str, Any]] = []

    for fp in swc_files:
        name_lower = fp.stem.lower()
        data = _load_swc_manual(fp)
        if "soma" in name_lower:
            result["soma"] = data
        elif "apical" in name_lower:
            result["apical"] = data
        elif "basal" in name_lower:
            basal_parts.append(data)
        elif "surface" in name_lower:
            result["surface"] = data
        elif "axon" in name_lower:
            result["axon"] = data

    # Concatenate basal trees --------------------------------------------------
    if len(basal_parts) == 1:
        result["basal"] = basal_parts[0]
    elif len(basal_parts) > 1:
        result["basal"] = _concatenate_trees(basal_parts)

    return result


def _concatenate_trees(trees: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate multiple SWC trees into one combined tree.

    Mimics ``cat_tree_sw.m``: the root of each subsequent tree is connected
    to the closest node in the accumulated tree (by Euclidean distance).
    """
    combined_nodes = trees[0]["nodes"].copy()
    combined_edges = trees[0]["edges"].copy() if len(trees[0]["edges"]) > 0 else np.empty((0, 2), dtype=int)

    for t in trees[1:]:
        offset = combined_nodes["id"].max() + 1
        new_nodes = t["nodes"].copy()

        # Find the root of the new tree (parent_id == -1)
        root_mask = new_nodes["parent_id"] == -1
        root_row = new_nodes.loc[root_mask].iloc[0]
        root_xyz = np.array([root_row["x"], root_row["y"], root_row["z"]])

        # Closest node in the combined tree
        combined_xyz = combined_nodes[["x", "y", "z"]].values
        dists = np.linalg.norm(combined_xyz - root_xyz, axis=1)
        closest_id = int(combined_nodes.iloc[np.argmin(dists)]["id"])

        # Re-id the new nodes
        old_to_new = {int(row["id"]): int(row["id"]) + offset for _, row in new_nodes.iterrows()}
        new_nodes["id"] = new_nodes["id"] + offset
        new_nodes["parent_id"] = new_nodes["parent_id"].apply(
            lambda pid: old_to_new[pid] if pid != -1 else closest_id
        )

        # New edges
        new_edges = new_nodes.loc[new_nodes["parent_id"] != -1, ["parent_id", "id"]].values.astype(int)
        # The former root now has closest_id as parent — add that edge
        former_root_new_id = old_to_new[int(root_row["id"])]
        root_edge = np.array([[closest_id, former_root_new_id]])
        new_edges = np.vstack([new_edges, root_edge]) if len(new_edges) > 0 else root_edge

        combined_nodes = pd.concat([combined_nodes, new_nodes], ignore_index=True)
        if len(combined_edges) > 0 and len(new_edges) > 0:
            combined_edges = np.vstack([combined_edges, new_edges])
        elif len(new_edges) > 0:
            combined_edges = new_edges

    return {"nodes": combined_nodes, "edges": combined_edges}


# ---------------------------------------------------------------------------
# Soma subtraction
# ---------------------------------------------------------------------------


def soma_subtract(
    neurons: dict[str, dict[str, Any]],
    soma_center: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Subtract soma centroid from all trees and flip Y axis.

    Implements ``morphology_readout.m`` lines 82-111:
    ``X = X - soma_mx; Y = (Y - soma_my) * -1; Z = Z - soma_mz``

    Parameters
    ----------
    neurons : dict
        As returned by :func:`load_morphology`.
    soma_center : ndarray, optional
        ``(3,)`` array ``[x, y, z]``.  If *None*, computed from the mean of
        the ``'soma'`` nodes.

    Returns
    -------
    dict
        New dict with centred coordinates (original is not mutated).
    """
    if soma_center is None:
        if "soma" not in neurons:
            raise ValueError("No soma data available and no soma_center provided.")
        soma_nodes = neurons["soma"]["nodes"]
        soma_center = np.array(
            [soma_nodes["x"].mean(), soma_nodes["y"].mean(), soma_nodes["z"].mean()]
        )

    result: dict[str, dict[str, Any]] = {}
    for key, tree in neurons.items():
        new_nodes = tree["nodes"].copy()
        new_nodes["x"] = new_nodes["x"] - soma_center[0]
        new_nodes["y"] = (new_nodes["y"] - soma_center[1]) * -1
        new_nodes["z"] = new_nodes["z"] - soma_center[2]
        result[key] = {"nodes": new_nodes, "edges": tree["edges"].copy()}

    return result


# ---------------------------------------------------------------------------
# Rotate to surface
# ---------------------------------------------------------------------------


def rotate_to_surface(
    neurons: dict[str, dict[str, Any]],
    surface_pts: np.ndarray,
    n_close_pts: int = 500,
) -> tuple[dict[str, dict[str, Any]], float]:
    """Rotate all trees so the pial surface is horizontal.

    Implements ``morphology_readout.m`` lines 117-153 and ``rotate_tree.m``.

    Parameters
    ----------
    neurons : dict
        Soma-subtracted neuron dict.
    surface_pts : ndarray
        ``(N, 2)`` array of surface X, Y coordinates (already soma-subtracted
        and Y-flipped).
    n_close_pts : int
        Number of surface points on each side of the closest-to-soma point to
        use for the linear fit (default 500, matching MATLAB).

    Returns
    -------
    tuple[dict, float]
        ``(rotated_neurons, angle_deg)``
    """
    soma_pt = np.array([[0.0, 0.0]])
    # Find closest surface point to soma
    dists = np.linalg.norm(surface_pts - soma_pt, axis=1)
    k = int(np.argmin(dists))

    # Select nearby surface points
    start = max(0, k - n_close_pts)
    end = min(len(surface_pts), k + n_close_pts + 1)
    close_pts = surface_pts[start:end]

    # Fit line: Y = slope * X + intercept
    slope = np.polyfit(close_pts[:, 0], close_pts[:, 1], 1)[0]

    # Compute rotation angle
    angle_rad = np.arctan(-slope)
    closest_pt = surface_pts[k]
    if closest_pt[1] < 0:
        # Surface is below soma — rotate the other way
        angle_rad = np.pi + angle_rad

    angle_deg = float(np.degrees(angle_rad))

    # Apply rotation to all trees
    y_offset = 0.0  # MATLAB uses surfFitOffset = 0
    rotated = _rotate_all(neurons, angle_rad, y_offset)

    return rotated, angle_deg


def _rotate_tree_2d(
    nodes: pd.DataFrame, angle_rad: float, y_offset: float
) -> pd.DataFrame:
    """Apply 2D rotation to X, Y columns of a node DataFrame.

    Implements ``rotate_tree.m``:
    ``pts2d = [X, Y - offset]'; rotated = R * pts2d; X = rotated[0]; Y = rotated[1] + offset``
    """
    new_nodes = nodes.copy()
    x = new_nodes["x"].values
    y = new_nodes["y"].values - y_offset
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    new_nodes["x"] = cos_a * x - sin_a * y
    new_nodes["y"] = sin_a * x + cos_a * y + y_offset
    return new_nodes


def _rotate_all(
    neurons: dict[str, dict[str, Any]], angle_rad: float, y_offset: float
) -> dict[str, dict[str, Any]]:
    """Rotate all trees by *angle_rad*."""
    result: dict[str, dict[str, Any]] = {}
    for key, tree in neurons.items():
        new_nodes = _rotate_tree_2d(tree["nodes"], angle_rad, y_offset)
        result[key] = {"nodes": new_nodes, "edges": tree["edges"].copy()}
    return result


# ---------------------------------------------------------------------------
# Tree statistics
# ---------------------------------------------------------------------------


def _build_adjacency(nodes: pd.DataFrame, edges: np.ndarray) -> dict[int, list[int]]:
    """Build parent -> [children] adjacency dict from edges."""
    children: dict[int, list[int]] = {int(nid): [] for nid in nodes["id"]}
    for parent_id, child_id in edges:
        children.setdefault(int(parent_id), []).append(int(child_id))
    return children


def _find_root(nodes: pd.DataFrame) -> int:
    """Return the id of the root node (parent_id == -1)."""
    roots = nodes.loc[nodes["parent_id"] == -1, "id"]
    if len(roots) == 0:
        # Fallback: find a node that is never a child
        child_ids = set(nodes["id"]) - set(nodes.loc[nodes["parent_id"] != -1, "parent_id"])
        if child_ids:
            return int(min(child_ids))
        return int(nodes["id"].iloc[0])
    return int(roots.iloc[0])


def _segment_lengths(nodes: pd.DataFrame, edges: np.ndarray) -> np.ndarray:
    """Euclidean length of each edge (segment) in the tree."""
    if len(edges) == 0:
        return np.array([])
    id_to_idx = {int(nid): i for i, nid in enumerate(nodes["id"])}
    xyz = nodes[["x", "y", "z"]].values
    lengths = np.empty(len(edges))
    for i, (pid, cid) in enumerate(edges):
        p_idx = id_to_idx[int(pid)]
        c_idx = id_to_idx[int(cid)]
        lengths[i] = np.linalg.norm(xyz[p_idx] - xyz[c_idx])
    return lengths


def _path_lengths_from_root(
    nodes: pd.DataFrame, edges: np.ndarray
) -> dict[int, float]:
    """Compute cumulative path length from root to every node via BFS."""
    children = _build_adjacency(nodes, edges)
    root_id = _find_root(nodes)
    id_to_idx = {int(nid): i for i, nid in enumerate(nodes["id"])}
    xyz = nodes[["x", "y", "z"]].values

    path_len: dict[int, float] = {root_id: 0.0}
    queue = [root_id]
    while queue:
        nid = queue.pop(0)
        for cid in children.get(nid, []):
            dist = float(np.linalg.norm(
                xyz[id_to_idx[cid]] - xyz[id_to_idx[nid]]
            ))
            path_len[cid] = path_len[nid] + dist
            queue.append(cid)

    return path_len


def _branch_order_per_node(
    nodes: pd.DataFrame, edges: np.ndarray
) -> dict[int, int]:
    """Compute branch order for each node (number of branch points on path to root)."""
    children = _build_adjacency(nodes, edges)
    root_id = _find_root(nodes)

    # Identify branch points (nodes with >1 child)
    bp_set = {nid for nid, ch in children.items() if len(ch) > 1}

    bo: dict[int, int] = {root_id: 0}
    queue = [root_id]
    while queue:
        nid = queue.pop(0)
        for cid in children.get(nid, []):
            # If cid itself is a branch point, increment
            bo[cid] = bo[nid] + (1 if cid in bp_set else 0)
            queue.append(cid)
    return bo


def _dissect_into_branches(
    nodes: pd.DataFrame, edges: np.ndarray
) -> list[list[int]]:
    """Dissect tree into branches (sequences of nodes between branch/terminal points).

    Returns a list of branches, each being a list of node ids from start to end.
    """
    children = _build_adjacency(nodes, edges)
    root_id = _find_root(nodes)

    # Branch points have >1 child, terminal nodes have 0 children
    bp_set = {nid for nid, ch in children.items() if len(ch) > 1}
    terminal_set = {nid for nid, ch in children.items() if len(ch) == 0}
    cut_set = bp_set | terminal_set

    branches: list[list[int]] = []

    # BFS to find branches
    queue: list[tuple[int, list[int]]] = [(root_id, [root_id])]
    while queue:
        nid, current_branch = queue.pop(0)
        for cid in children.get(nid, []):
            if cid in cut_set:
                # End this branch
                branches.append(current_branch + [cid])
                # If it's a branch point, start new branches from it
                if cid in bp_set:
                    queue.append((cid, [cid]))
            else:
                # Continue the branch
                queue.append((cid, current_branch + [cid]))

    return branches


def compute_tree_stats(
    nodes: pd.DataFrame, edges: np.ndarray
) -> dict[str, float]:
    """Compute morphological statistics for a single tree.

    Implements the global stats from ``stats_tree_sw.m``:

    - ``total_length`` — sum of all segment lengths (um)
    - ``max_path_length`` — longest path from root (um)
    - ``n_branch_points`` — number of nodes with > 1 child
    - ``max_branch_order`` — highest branch order in the tree
    - ``mean_branch_length`` — mean length of dissected branches (um)
    - ``mean_path_length`` — mean path-length-from-root across all nodes (um)
    - ``mean_branch_order`` — mean branch order at topological points
    - ``mean_path_eucl_ratio`` — mean ratio of path length to Euclidean distance
    - ``width`` — max X - min X
    - ``height`` — max Y - min Y
    - ``depth`` — max Z - min Z
    - ``width_height_ratio`` — width / height (0 if height == 0)
    - ``width_depth_ratio`` — width / depth (0 if depth == 0)

    Parameters
    ----------
    nodes : DataFrame
        Node table with ``id, x, y, z, radius, parent_id`` columns.
    edges : ndarray
        ``(E, 2)`` array of ``(parent_id, child_id)`` pairs.

    Returns
    -------
    dict
        Metric name -> value.
    """
    seg_lens = _segment_lengths(nodes, edges)
    total_length = float(seg_lens.sum()) if len(seg_lens) > 0 else 0.0

    path_lens = _path_lengths_from_root(nodes, edges)
    plen_values = np.array(list(path_lens.values()))
    max_path_length = float(plen_values.max()) if len(plen_values) > 0 else 0.0
    mean_path_length = float(plen_values.mean()) if len(plen_values) > 0 else 0.0

    children = _build_adjacency(nodes, edges)
    root_id = _find_root(nodes)
    bp_set = {nid for nid, ch in children.items() if len(ch) > 1}
    n_branch_points = len(bp_set)

    bo = _branch_order_per_node(nodes, edges)
    bo_values = np.array(list(bo.values()))
    max_branch_order = int(bo_values.max()) if len(bo_values) > 0 else 0

    # Topological points: branch + terminal
    terminal_set = {nid for nid, ch in children.items() if len(ch) == 0}
    topo_set = bp_set | terminal_set
    if topo_set:
        topo_bo = np.array([bo[nid] for nid in topo_set if nid in bo])
        mean_branch_order = float(topo_bo.mean()) if len(topo_bo) > 0 else 0.0
    else:
        mean_branch_order = 0.0

    # Mean path-length / Euclidean-distance ratio at topological points
    id_to_idx = {int(nid): i for i, nid in enumerate(nodes["id"])}
    xyz = nodes[["x", "y", "z"]].values
    root_xyz = xyz[id_to_idx[root_id]]
    ratios = []
    for nid in topo_set:
        if nid in path_lens and nid in id_to_idx:
            pl = path_lens[nid]
            eucl_dist = float(np.linalg.norm(xyz[id_to_idx[nid]] - root_xyz))
            if eucl_dist > 0 and pl > 0:
                ratios.append(pl / eucl_dist)
    mean_path_eucl_ratio = float(np.mean(ratios)) if ratios else float("nan")

    # Branch lengths via dissection
    branches = _dissect_into_branches(nodes, edges)
    branch_lengths = []
    for branch in branches:
        if len(branch) < 2:
            continue
        blen = 0.0
        for i in range(len(branch) - 1):
            p_idx = id_to_idx.get(branch[i])
            c_idx = id_to_idx.get(branch[i + 1])
            if p_idx is not None and c_idx is not None:
                blen += float(np.linalg.norm(xyz[p_idx] - xyz[c_idx]))
        if blen > 0.2:  # Match MATLAB threshold
            branch_lengths.append(blen)
    mean_branch_length = float(np.mean(branch_lengths)) if branch_lengths else 0.0

    # Spatial extents
    x_vals = nodes["x"].values
    y_vals = nodes["y"].values
    z_vals = nodes["z"].values
    width = float(x_vals.max() - x_vals.min()) if len(x_vals) > 0 else 0.0
    height = float(y_vals.max() - y_vals.min()) if len(y_vals) > 0 else 0.0
    depth = float(z_vals.max() - z_vals.min()) if len(z_vals) > 0 else 0.0

    width_height_ratio = width / height if height > 0 else 0.0
    width_depth_ratio = width / depth if depth > 0 else 0.0

    return {
        "total_length": total_length,
        "max_path_length": max_path_length,
        "n_branch_points": n_branch_points,
        "max_branch_order": max_branch_order,
        "mean_branch_length": mean_branch_length,
        "mean_path_length": mean_path_length,
        "mean_branch_order": mean_branch_order,
        "mean_path_eucl_ratio": mean_path_eucl_ratio,
        "width": width,
        "height": height,
        "depth": depth,
        "width_height_ratio": width_height_ratio,
        "width_depth_ratio": width_depth_ratio,
    }


# ---------------------------------------------------------------------------
# Sholl analysis
# ---------------------------------------------------------------------------


def compute_sholl(
    nodes: pd.DataFrame,
    soma_center: np.ndarray,
    radii: np.ndarray,
    edges: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Sholl intersection profile.

    For each radius in *radii*, count the number of tree segments that cross
    the sphere of that radius centred on *soma_center*.

    A segment between two nodes crosses a sphere of radius *r* if one node is
    inside and the other is outside (or on) the sphere.

    Parameters
    ----------
    nodes : DataFrame
        Node table with ``x, y, z`` columns.
    soma_center : ndarray
        ``(3,)`` centre point for Sholl shells.
    radii : ndarray
        Sorted array of shell radii.
    edges : ndarray, optional
        ``(E, 2)`` edge array.  If *None*, edges are inferred from
        ``parent_id`` column.

    Returns
    -------
    ndarray
        Integer array of length ``len(radii)`` with intersection counts.

    References
    ----------
    Sholl 1953. "Dendritic organization in the neurons of the visual and motor
    cortices of the cat." J Anat 87(4):387-406.
    """
    xyz = nodes[["x", "y", "z"]].values
    dists = np.linalg.norm(xyz - soma_center, axis=1)

    if edges is None:
        id_to_idx = {int(nid): i for i, nid in enumerate(nodes["id"])}
        edge_list = []
        for _, row in nodes.iterrows():
            pid = int(row["parent_id"])
            if pid != -1 and pid in id_to_idx:
                edge_list.append((id_to_idx[pid], id_to_idx[int(row["id"])]))
        edge_indices = np.array(edge_list, dtype=int) if edge_list else np.empty((0, 2), dtype=int)
    else:
        id_to_idx = {int(nid): i for i, nid in enumerate(nodes["id"])}
        edge_indices = np.array(
            [[id_to_idx[int(e[0])], id_to_idx[int(e[1])]] for e in edges
             if int(e[0]) in id_to_idx and int(e[1]) in id_to_idx],
            dtype=int,
        ) if len(edges) > 0 else np.empty((0, 2), dtype=int)

    counts = np.zeros(len(radii), dtype=int)
    if len(edge_indices) == 0:
        return counts

    d_parent = dists[edge_indices[:, 0]]
    d_child = dists[edge_indices[:, 1]]

    for i, r in enumerate(radii):
        # Crossing: one side <= r, other side > r
        crosses = (d_parent <= r) != (d_child <= r)
        counts[i] = int(crosses.sum())

    return counts


# ---------------------------------------------------------------------------
# Surface distance
# ---------------------------------------------------------------------------


def compute_surface_distance(
    surface_pts: np.ndarray, dendrite_pts: np.ndarray
) -> dict[str, float]:
    """Compute min and max distance from dendrite points to the surface.

    Implements ``getSurfDist.m``: for each dendrite point find the nearest
    surface point, then report the minimum (most superficial) and maximum
    (deepest) of those nearest-surface distances.

    Parameters
    ----------
    surface_pts : ndarray
        ``(N, D)`` array of surface coordinates (typically 2D: X, Y).
    dendrite_pts : ndarray
        ``(M, D)`` array of dendrite coordinates.

    Returns
    -------
    dict
        ``dist_superficial`` — minimum nearest-surface distance (closest
        dendrite to surface).
        ``dist_deep`` — maximum nearest-surface distance (furthest dendrite
        from its closest surface point).
    """
    tree = cKDTree(surface_pts)
    dists, _ = tree.query(dendrite_pts)
    return {
        "dist_superficial": float(np.min(dists)),
        "dist_deep": float(np.max(dists)),
    }
