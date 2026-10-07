"""Representational geometry of RSP population activity on the maze graph.

Question
--------
When population activity is averaged per maze state (maze cell, or maze cell x
travel direction), does the dissimilarity between states follow the maze's
corridor topology (shortest-path graph distance) or straight-line (Euclidean)
distance between cell centres? The 23-cell maze is a tree, so some cells are
close in space but far apart through the corridors (e.g. the three top-row dead
ends separated by walls), which lets the two distances be separated. A second
question asks whether states that share a local graph role (e.g. heading into a
dead end, leaving a T-junction towards its stem) at different locations have
more similar population vectors than states with different roles at matched
graph distance, Euclidean distance and heading difference.

States
------
Frames are assigned to maze cells (:func:`hm2p.analysis.population_maze.session_cells`)
and used only while running (speed >= ``run_cm_s``, no behavioural artefact).
Travel direction (E, N, W, S) is defined from successive cell stays: the first
half of each stay takes the entry direction (previous cell -> this cell), the
second half the exit direction (this cell -> next cell); halves whose
neighbouring stay is not an adjacent cell are unlabelled. A *directed state* is
(cell, travel direction).

Cross-validated dissimilarity
-----------------------------
Activity is z-scored per neuron over the session's valid frames. The session
is cut into blocks of ``block_s`` seconds assigned alternately to halves A and
B; frames within ``gap_s`` of a block boundary are dropped so a single visit
does not contribute to both halves. For each state the mean population vector
is computed in A and in B (states need at least ``min_frames`` frames in each
half). The dissimilarity between states i and j is the cross-validated squared
Euclidean distance (a crossnobis estimate without noise whitening)

    d_ij = (a_i - a_j) . (b_i - b_j) / n_neurons,

which is unbiased because the noise in A and B is independent: its expectation
is the squared distance between the true means and it is zero on average when
the states do not differ. The estimate is averaged over ``n_offsets`` block
phase offsets. Nili et al. 2014; Walther et al. 2016 (references below).

Geometry tests
--------------
Over all state pairs, the partial Spearman correlation of d with graph distance
controlling for Euclidean distance (and vice versa) is computed, with further
controls: absolute difference in mean visit time, distance between the states'
temporal occupancy profiles (visits per ``profile_bin_s`` bin; slow drift makes
states visited at similar times look similar), absolute difference in mean
speed, distance between mean head-direction unit vectors and between mean
axial (doubled-angle) vectors, and for directed states the travel-direction
difference. The same partials are also reported without the two temporal
controls (``_nt``; on a tree, graph-near states are visited close in time, so
the temporal controls can absorb genuine graph structure) and for local pairs
only (Euclidean distance <= ``LOCAL_EUCLID``; ``prho_graph_local``), where
nearby cells separated by walls carry the graph-vs-space contrast for codes
with narrow fields. Partial Spearman = Pearson correlation of the residuals of the
rank-transformed variables after linear regression on the ranked controls.

Null: circular shift of the activity relative to behaviour by at least
``min_shift_s`` (all state means and statistics recomputed per shift). A second
null permutes state identities of the dissimilarity matrix (Mantel). Session
z = (observed - null mean) / null sd. Animal-level inference: Wilcoxon
signed-rank test on per-animal mean session values.

Role generalisation
-------------------
The maze tree is rooted at its central T-junction (3, 2). Each directed state
gets a role from its cell type and whether its direction points away from or
towards the root: ``dead_end_in``, ``dead_end_out``, ``corridor_out``,
``corridor_in``, ``t_from_stem`` (moving away from the stem, i.e. arrived from
the root side), ``t_to_stem``, ``t_bar`` (along the junction's bar), ``root``
(the root junction; excluded). The dissimilarity is residualised (ranks, linear
regression) on time, occupancy-profile, speed and head-direction controls;
pairs of states at different cells are stratified by exact (graph distance,
Euclidean distance, direction difference). The statistic is the mean over
strata containing both kinds of pair of (mean residual for different-role
pairs - mean residual for same-role pairs), weighted by the smaller count.
Positive values mean same-role states are more similar. Primary null: role
labels permuted among states of the same node type (see :func:`role_test`);
secondary: permuted across all non-root states, and circular shift.

References
----------
Nili H, Wingfield C, Walther A, Su L, Marslen-Wilson W, Kriegeskorte N. 2014.
"A toolbox for representational similarity analysis." PLoS Comput Biol
10:e1003553. doi:10.1371/journal.pcbi.1003553
Walther A, Nili H, Ejaz N, Alink A, Kriegeskorte N, Diedrichsen J. 2016.
"Reliability of dissimilarity measures for multi-voxel pattern analysis."
NeuroImage 137:188-200. doi:10.1016/j.neuroimage.2015.12.012
https://github.com/rsagroup/rsatoolbox
Mantel N. 1967. "The detection of disease clustering and a generalized
regression approach." Cancer Research 27:209-220.
Legendre P, Legendre L. 2012. Numerical Ecology, 3rd ed. Elsevier (partial
Mantel / partial rank correlation on distance matrices).
Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth show
rapid learning, sudden insight, and efficient exploration." eLife 10:e66175.
doi:10.7554/eLife.66175
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse, stats

from hm2p.analysis.population_maze import cell_stays
from hm2p.maze.topology import RoseMaze

# Travel directions: index -> (dcol, drow); angle = 90 * index degrees (maze frame).
DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))
DIR_NAMES = ("E", "N", "W", "S")
ROOT_CELL = (3, 2)
ROLES = (
    "dead_end_in",
    "dead_end_out",
    "corridor_out",
    "corridor_in",
    "t_from_stem",
    "t_to_stem",
    "t_bar",
    "root",
)
LOCAL_EUCLID = 2.5
CONTROL_COLS = ("time_diff", "profile_dist", "speed_diff", "hd_dist", "hd_axial_dist")


@dataclass
class GeometryParams:
    """Parameters of the maze-geometry analysis (see module docstring)."""

    run_cm_s: float = 2.5
    block_s: float = 20.0
    gap_s: float = 2.0
    n_offsets: int = 3
    min_frames: int = 8
    profile_bin_s: float = 60.0
    n_shifts: int = 200
    min_shift_s: float = 30.0
    n_perm: int = 1000
    seed: int = 0


# ---------------------------------------------------------------------------
# Geometry of the maze
# ---------------------------------------------------------------------------


def cell_centres(maze: RoseMaze) -> np.ndarray:
    """Cell centres ``(n_cells, 2)`` in maze units, ordered as ``maze.cell_list``."""
    return np.array([(c + 0.5, r + 0.5) for c, r in maze.cell_list], dtype=np.float64)


def root_depth(maze: RoseMaze, root: tuple[int, int] = ROOT_CELL) -> np.ndarray:
    """Graph distance of every cell from *root* ``(n_cells,)``."""
    return maze.dist[maze.cell_to_idx[root]].astype(np.int64)


def direction_index(dc: int, dr: int) -> int:
    """Index of the unit step ``(dc, dr)`` in :data:`DIRECTIONS`, -1 otherwise."""
    try:
        return DIRECTIONS.index((int(dc), int(dr)))
    except ValueError:
        return -1


def angle_diff_deg(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    """Absolute wrapped angular difference in degrees, in ``[0, 180]``."""
    d = np.mod(np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0, 360.0) - 180.0
    return np.abs(d)


# ---------------------------------------------------------------------------
# Per-frame travel direction and states
# ---------------------------------------------------------------------------


def frame_directions(cells: npt.ArrayLike, maze: RoseMaze) -> np.ndarray:
    """Travel direction index per frame (-1 = undefined).

    Stays in one cell are taken from :func:`cell_stays` (gaps of unassigned
    frames are bridged). The first half of a stay (``start`` to
    ``start + len // 2``) gets the entry direction (previous -> current cell),
    the rest the exit direction (current -> next cell); a half whose
    neighbouring stay is missing or not adjacent stays -1. Unassigned frames
    (cell < 0) are always -1.
    """
    c = np.asarray(cells, dtype=np.int64)
    out = np.full(c.shape, -1, dtype=np.int64)
    sc, st, en = cell_stays(c)
    cl = maze.cell_list
    for k in range(sc.size):
        cur = cl[sc[k]]
        mid = st[k] + (en[k] - st[k]) // 2
        if k > 0:
            prev = cl[sc[k - 1]]
            if prev in maze.adj[cur]:
                out[st[k] : mid] = direction_index(cur[0] - prev[0], cur[1] - prev[1])
        if k + 1 < sc.size:
            nxt = cl[sc[k + 1]]
            if nxt in maze.adj[cur]:
                out[mid : en[k]] = direction_index(nxt[0] - cur[0], nxt[1] - cur[1])
    out[c < 0] = -1
    return out


def frame_states(
    cells: npt.ArrayLike, dirs: npt.ArrayLike, mask: npt.ArrayLike, directed: bool
) -> np.ndarray:
    """State id per frame: cell, or ``cell * 4 + direction`` if *directed* (-1 = unused)."""
    c = np.asarray(cells, dtype=np.int64)
    d = np.asarray(dirs, dtype=np.int64)
    m = np.asarray(mask, dtype=bool) & (c >= 0)
    if directed:
        m &= d >= 0
        s = c * 4 + d
    else:
        s = c.copy()
    return np.where(m, s, -1)


def state_role(cell: tuple[int, int], direction: int, maze: RoseMaze, depth: np.ndarray) -> str:
    """Local graph role of a directed state (see module docstring)."""
    if cell == ROOT_CELL:
        return "root"
    kind = maze.node_types[cell]
    dc, dr = DIRECTIONS[direction]
    d0 = depth[maze.cell_to_idx[cell]]
    nbrs = maze.adj[cell]
    if kind == "dead_end":
        par = nbrs[0]
        into = (cell[0] - par[0], cell[1] - par[1]) == (dc, dr)
        return "dead_end_in" if into else "dead_end_out"
    if kind == "corridor":
        fwd = (cell[0] + dc, cell[1] + dr)
        bwd = (cell[0] - dc, cell[1] - dr)
        if fwd in nbrs:
            outward = depth[maze.cell_to_idx[fwd]] > d0
        elif bwd in nbrs:
            outward = depth[maze.cell_to_idx[bwd]] < d0
        else:  # direction perpendicular to the corridor
            return "corridor_other"
        return "corridor_out" if outward else "corridor_in"
    # T-junction: the stem is the neighbour not collinear with any other neighbour.
    stem = None
    for n in nbrs:
        opp = (2 * cell[0] - n[0], 2 * cell[1] - n[1])
        if opp not in nbrs:
            stem = n
    if stem is None:
        return "junction_other"
    to_stem = (stem[0] - cell[0], stem[1] - cell[1])
    if (dc, dr) == to_stem:
        return "t_to_stem"
    if (-dc, -dr) == to_stem:
        return "t_from_stem"
    return "t_bar"


# ---------------------------------------------------------------------------
# Split halves and state means
# ---------------------------------------------------------------------------


def time_halves(
    n_frames: int, fps: float, block_s: float, gap_s: float, offset: float = 0.0
) -> np.ndarray:
    """Half (0 = A, 1 = B, -1 = dropped) per frame from alternating time blocks.

    Blocks of ``block_s`` seconds start at ``-offset * block_s``; frames within
    ``gap_s`` of a block boundary are dropped.
    """
    t = np.arange(n_frames) / float(fps) + float(offset) * block_s
    pos = t / block_s
    blk = np.floor(pos).astype(np.int64)
    within = (pos - blk) * block_s
    half = (blk % 2).astype(np.int64)
    drop = (within < gap_s) | (within > block_s - gap_s)
    half[drop] = -1
    return half


def state_means(
    x: npt.ArrayLike, states: npt.ArrayLike, halves: npt.ArrayLike, n_states: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-half state means and frame counts.

    Parameters
    ----------
    x : (n_neurons, n_frames) float, finite
    states : (n_frames,) int, -1 = unused
    halves : (n_frames,) int in {-1, 0, 1}

    Returns
    -------
    means : (2, n_states, n_neurons), NaN where a state has no frame
    counts : (2, n_states) int
    """
    a = np.asarray(x, dtype=np.float64)
    s = np.asarray(states, dtype=np.int64)
    h = np.asarray(halves, dtype=np.int64)
    means = np.full((2, n_states, a.shape[0]), np.nan)
    counts = np.zeros((2, n_states), dtype=np.int64)
    for k in (0, 1):
        sel = (s >= 0) & (h == k)
        idx = s[sel]
        cnt = np.bincount(idx, minlength=n_states)
        oh = sparse.csr_matrix(
            (np.ones(idx.size), (idx, np.arange(idx.size))), shape=(n_states, idx.size)
        )
        sums = np.asarray(oh @ a[:, sel].T)
        counts[k] = cnt
        with np.errstate(invalid="ignore", divide="ignore"):
            means[k] = np.where(cnt[:, None] > 0, sums / np.maximum(cnt, 1)[:, None], np.nan)
    return means, counts


def crossnobis(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    """Cross-validated squared Euclidean distance between rows, per neuron.

    ``d_ij = (a_i - a_j) . (b_i - b_j) / n_neurons`` for row-aligned state means
    ``a`` and ``b`` (``(n_states, n_neurons)``) from independent data halves.
    The diagonal is 0.
    """
    A = np.asarray(a, dtype=np.float64)
    B = np.asarray(b, dtype=np.float64)
    g = A @ B.T
    dg = np.diag(g)
    d = dg[:, None] + dg[None, :] - g - g.T
    np.fill_diagonal(d, 0.0)
    return d / A.shape[1]


def cv_dissimilarity(
    x: npt.ArrayLike,
    states: npt.ArrayLike,
    n_states: int,
    fps: float,
    params: GeometryParams,
    shift: int = 0,
    keep: npt.ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-validated dissimilarity matrix averaged over block offsets.

    Activity is rolled by *shift* frames relative to the states. A state is
    kept when it has at least ``params.min_frames`` frames in both halves for
    every offset (or as given by *keep*).

    Returns
    -------
    d : (n_kept, n_kept) float
    kept : (n_kept,) int, state ids
    """
    a = np.asarray(x, dtype=np.float64)
    if shift:
        a = np.roll(a, int(shift), axis=1)
    s = np.asarray(states, dtype=np.int64)
    n = s.size
    offsets = np.arange(params.n_offsets) / max(params.n_offsets, 1)
    parts = []
    ok = np.ones(n_states, dtype=bool)
    for off in offsets:
        h = time_halves(n, fps, params.block_s, params.gap_s, off)
        m, c = state_means(a, s, h, n_states)
        parts.append(m)
        ok &= (c >= params.min_frames).all(axis=0)
    kept = np.flatnonzero(ok) if keep is None else np.asarray(keep, dtype=np.int64)
    if kept.size == 0:
        return np.zeros((0, 0)), kept
    d = np.mean([crossnobis(m[0][kept], m[1][kept]) for m in parts], axis=0)
    return d, kept


def kept_states(
    states: npt.ArrayLike, n_states: int, n_frames: int, fps: float, params: GeometryParams
) -> np.ndarray:
    """States with at least ``min_frames`` frames in both halves for every block offset."""
    s = np.asarray(states, dtype=np.int64)
    ok = np.ones(n_states, dtype=bool)
    for off in np.arange(params.n_offsets) / max(params.n_offsets, 1):
        h = time_halves(n_frames, fps, params.block_s, params.gap_s, off)
        for k in (0, 1):
            ok &= np.bincount(s[(s >= 0) & (h == k)], minlength=n_states) >= params.min_frames
    return np.flatnonzero(ok)


# ---------------------------------------------------------------------------
# State descriptors and pair table
# ---------------------------------------------------------------------------


def state_descriptors(
    states: npt.ArrayLike,
    kept: npt.ArrayLike,
    hd_deg: npt.ArrayLike,
    speed: npt.ArrayLike,
    fps: float,
    profile_bin_s: float,
) -> pd.DataFrame:
    """Per kept state: mean time, occupancy profile, mean speed and head-direction vectors.

    Columns: ``state, n_frames, mean_time_s, speed, hd_c, hd_s, hd2_c, hd2_s,
    profile`` (``profile`` is the unit-normalised frame count per
    ``profile_bin_s`` time bin, an array).
    """
    s = np.asarray(states, dtype=np.int64)
    hd = np.deg2rad(np.asarray(hd_deg, dtype=float))
    sp = np.asarray(speed, dtype=float)
    t = np.arange(s.size) / float(fps)
    nb = max(1, int(np.ceil(s.size / fps / profile_bin_s)))
    tb = np.minimum((t // profile_bin_s).astype(np.int64), nb - 1)
    rows = []
    for k in np.asarray(kept, dtype=np.int64):
        sel = s == k
        h = hd[sel]
        h = h[np.isfinite(h)]
        prof = np.bincount(tb[sel], minlength=nb).astype(float)
        nrm = np.linalg.norm(prof)
        rows.append(
            {
                "state": int(k),
                "n_frames": int(sel.sum()),
                "mean_time_s": float(t[sel].mean()),
                "speed": float(np.nanmean(sp[sel])) if np.isfinite(sp[sel]).any() else np.nan,
                "hd_c": float(np.cos(h).mean()) if h.size else 0.0,
                "hd_s": float(np.sin(h).mean()) if h.size else 0.0,
                "hd2_c": float(np.cos(2 * h).mean()) if h.size else 0.0,
                "hd2_s": float(np.sin(2 * h).mean()) if h.size else 0.0,
                "profile": prof / nrm if nrm > 0 else prof,
            }
        )
    return pd.DataFrame(rows)


def pair_table(
    desc: pd.DataFrame, maze: RoseMaze, directed: bool, depth: np.ndarray | None = None
) -> pd.DataFrame:
    """All unordered state pairs (i < j over rows of *desc*) with geometry and controls.

    Columns: ``i, j`` (row positions), ``cell_i, cell_j, graph, euclid,
    same_cell, time_diff, profile_dist, speed_diff, hd_dist, hd_axial_dist``;
    for directed states also ``dir_diff`` (degrees) and ``role_i, role_j,
    same_role``.
    """
    st = desc["state"].to_numpy()
    cell = st // 4 if directed else st
    cen = cell_centres(maze)
    n = st.size
    ii, jj = np.triu_indices(n, k=1)
    prof = np.vstack(desc["profile"].to_numpy()) if n else np.zeros((0, 1))
    hd = desc[["hd_c", "hd_s"]].to_numpy()
    hd2 = desc[["hd2_c", "hd2_s"]].to_numpy()
    out = pd.DataFrame(
        {
            "i": ii,
            "j": jj,
            "cell_i": cell[ii],
            "cell_j": cell[jj],
            "graph": maze.dist[cell[ii], cell[jj]].astype(float),
            "euclid": np.linalg.norm(cen[cell[ii]] - cen[cell[jj]], axis=1),
            "same_cell": cell[ii] == cell[jj],
            "time_diff": np.abs(
                desc["mean_time_s"].to_numpy()[ii] - desc["mean_time_s"].to_numpy()[jj]
            ),
            "profile_dist": 1.0 - np.sum(prof[ii] * prof[jj], axis=1),
            "speed_diff": np.abs(desc["speed"].to_numpy()[ii] - desc["speed"].to_numpy()[jj]),
            "hd_dist": np.linalg.norm(hd[ii] - hd[jj], axis=1),
            "hd_axial_dist": np.linalg.norm(hd2[ii] - hd2[jj], axis=1),
        }
    )
    if directed:
        dr = st % 4
        out["dir_diff"] = angle_diff_deg(90.0 * dr[ii], 90.0 * dr[jj])
        if depth is None:
            depth = root_depth(maze)
        roles = np.array(
            [
                state_role(maze.cell_list[c], int(d), maze, depth)
                for c, d in zip(cell, dr, strict=True)
            ]
        )
        out["role_i"] = roles[ii]
        out["role_j"] = roles[jj]
        out["same_role"] = roles[ii] == roles[jj]
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _rank_cols(m: np.ndarray) -> np.ndarray:
    return np.column_stack([stats.rankdata(m[:, k]) for k in range(m.shape[1])]) if m.size else m


def _residualise(y: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Residuals of *y* (columns allowed) after OLS on *c* plus intercept."""
    X = np.column_stack([np.ones(y.shape[0]), c]) if c.size else np.ones((y.shape[0], 1))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_spearman(
    y: npt.ArrayLike, x: npt.ArrayLike, controls: npt.ArrayLike | None = None
) -> float:
    """Partial Spearman correlation of *y* and *x* given *controls* (columns).

    Ranks all variables, removes the linear effect of the ranked controls from
    the ranks of *y* and *x*, and returns the Pearson correlation of the
    residuals. NaN when fewer than ``n_controls + 3`` complete rows or a
    residual is constant.
    """
    yv = np.asarray(y, dtype=float).ravel()
    xv = np.asarray(x, dtype=float).ravel()
    c = (
        np.zeros((yv.size, 0))
        if controls is None
        else np.asarray(controls, dtype=float).reshape(yv.size, -1)
    )
    ok = np.isfinite(yv) & np.isfinite(xv) & np.isfinite(c).all(axis=1)
    if ok.sum() < c.shape[1] + 3:
        return float("nan")
    r = _rank_cols(np.column_stack([yv[ok], xv[ok]]))
    cr = _rank_cols(c[ok])
    res = _residualise(r, cr)
    sy, sx = res[:, 0].std(), res[:, 1].std()
    if sy < 1e-12 or sx < 1e-12:
        return float("nan")
    return float(np.corrcoef(res[:, 0], res[:, 1])[0, 1])


class PartialDesign:
    """Precomputed partial Spearman of a varying *y* with fixed *x* given fixed controls.

    The ranked controls (plus intercept) are orthogonalised once (QR); the
    residualised ranks of *x* are stored, so each call only ranks *y*.
    Numerically equal to :func:`partial_spearman` on the same rows.
    """

    def __init__(
        self, x: npt.ArrayLike, controls: npt.ArrayLike, rows: npt.ArrayLike | None = None
    ) -> None:
        xv = np.asarray(x, dtype=float).ravel()
        c = np.asarray(controls, dtype=float).reshape(xv.size, -1)
        ok = np.isfinite(xv) & np.isfinite(c).all(axis=1)
        if rows is not None:
            ok &= np.asarray(rows, dtype=bool)
        self.ok = ok
        self.valid = bool(ok.sum() >= c.shape[1] + 3)
        if not self.valid:
            return
        X = np.column_stack([np.ones(ok.sum()), _rank_cols(c[ok])])
        q, r = np.linalg.qr(X)
        keep = np.abs(np.diag(r)) > 1e-9 * max(1.0, np.abs(np.diag(r)).max())
        self.q = q[:, keep]
        rx = stats.rankdata(xv[ok])
        res = rx - self.q @ (self.q.T @ rx)
        nrm = np.linalg.norm(res)
        self.valid = bool(nrm > 1e-9)
        self.xres = res / nrm if self.valid else res

    def __call__(self, y: npt.ArrayLike) -> float:
        """Partial Spearman correlation for *y* (same rows as *x*)."""
        if not self.valid:
            return float("nan")
        yv = np.asarray(y, dtype=float).ravel()[self.ok]
        if not np.isfinite(yv).all():
            return float("nan")
        ry = stats.rankdata(yv)
        res = ry - self.q @ (self.q.T @ ry)
        nrm = np.linalg.norm(res)
        return float(res @ self.xres / nrm) if nrm > 1e-9 else float("nan")


def geometry_design(pairs: pd.DataFrame, directed: bool) -> dict[str, PartialDesign]:
    """Partial-correlation designs used by :func:`geometry_stats` (see its keys)."""
    g, e = pairs["graph"].to_numpy(float), pairs["euclid"].to_numpy(float)
    ctrl_cols = list(CONTROL_COLS) + (["dir_diff"] if directed else [])
    C = pairs[ctrl_cols].to_numpy(dtype=float)
    nt = np.column_stack(
        [C[:, k] for k, c in enumerate(ctrl_cols) if c not in ("time_diff", "profile_dist")]
    )
    loc = e <= LOCAL_EUCLID
    hd_k = ctrl_cols.index("hd_dist")
    des = {
        "prho_graph": PartialDesign(g, np.column_stack([e, C])),
        "prho_euclid": PartialDesign(e, np.column_stack([g, C])),
        "prho_graph_nt": PartialDesign(g, np.column_stack([e, nt])),
        "prho_euclid_nt": PartialDesign(e, np.column_stack([g, nt])),
        "prho_graph_local": PartialDesign(g, np.column_stack([e, C]), rows=loc),
        "prho_hd": PartialDesign(C[:, hd_k], np.column_stack([g, e, np.delete(C, hd_k, axis=1)])),
    }
    if directed:
        dk = ctrl_cols.index("dir_diff")
        des["prho_dir"] = PartialDesign(
            C[:, dk], np.column_stack([g, e, np.delete(C, dk, axis=1)])
        )
    return des


def geometry_stats(
    d: np.ndarray,
    pairs: pd.DataFrame,
    directed: bool,
    design: dict[str, PartialDesign] | None = None,
) -> dict[str, float]:
    """Partial Spearman statistics relating the dissimilarity matrix to maze geometry.

    Keys: ``rho_graph`` / ``rho_euclid`` (plain Spearman), ``prho_graph``
    (graph | Euclidean + controls), ``prho_euclid`` (Euclidean | graph +
    controls), ``graph_minus_euclid``; the same without the temporal controls
    (``*_nt``; on a tree, states near on the graph are also visited close in
    time, so these controls can absorb genuine graph structure);
    ``prho_graph_local`` (graph | Euclidean + controls over pairs with
    Euclidean distance <= :data:`LOCAL_EUCLID`, sensitive to narrow fields);
    ``prho_hd`` (HD-vector distance | graph, Euclidean, other controls); for
    directed states ``prho_dir`` (direction difference | all others);
    ``mean_d`` (mean cross-validated dissimilarity, a reliability check) and
    ``n_local_pairs``.
    """
    if design is None:
        design = geometry_design(pairs, directed)
    y = d[pairs["i"].to_numpy(), pairs["j"].to_numpy()]
    out = {k: f(y) for k, f in design.items()}
    out["graph_minus_euclid"] = out["prho_graph"] - out["prho_euclid"]
    out["graph_minus_euclid_nt"] = out["prho_graph_nt"] - out["prho_euclid_nt"]
    out["mean_d"] = float(np.mean(y)) if y.size else np.nan
    out["n_local_pairs"] = int((pairs["euclid"].to_numpy() <= LOCAL_EUCLID).sum())
    return out


def role_strata(pairs: pd.DataFrame, mode: str = "cellpair") -> np.ndarray:
    """Stratum id per pair; -1 = excluded.

    ``mode="cellpair"`` (primary): one stratum per unordered pair of maze
    cells, so same-role and different-role state pairs are compared at
    exactly the same two locations. ``mode="geometry"``: exact (graph
    distance, Euclidean distance, direction difference). Pairs at the same
    cell, or involving a ``root`` / undefined-role state, are excluded.
    """
    bad = {"root", "corridor_other", "junction_other"}
    ok = (
        ~pairs["same_cell"].to_numpy()
        & ~pairs["role_i"].isin(bad).to_numpy()
        & ~pairs["role_j"].isin(bad).to_numpy()
    )
    if mode == "cellpair":
        a = np.minimum(pairs["cell_i"].to_numpy(), pairs["cell_j"].to_numpy())
        b = np.maximum(pairs["cell_i"].to_numpy(), pairs["cell_j"].to_numpy())
        key = pd.Series(a * 1000 + b)
    elif mode == "geometry":
        key = (
            pairs["graph"].round(3).astype(str)
            + "|"
            + pairs["euclid"].round(3).astype(str)
            + "|"
            + pairs["dir_diff"].round(1).astype(str)
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")
    codes = pd.factorize(key)[0].astype(np.int64)
    return np.where(ok, codes, -1)


def role_contrast(
    resid: npt.ArrayLike, same: npt.ArrayLike, strata: npt.ArrayLike
) -> tuple[float, int]:
    """Weighted mean over strata of mean(resid | different role) - mean(resid | same role).

    Weights are ``min(n_same, n_diff)`` per stratum; strata lacking either kind
    of pair are skipped. Returns the contrast and the number of same-role pairs
    in usable strata (NaN, 0 if none).
    """
    r = np.asarray(resid, dtype=float)
    sm = np.asarray(same, dtype=bool)
    sk = np.asarray(strata, dtype=np.int64)
    use = (sk >= 0) & np.isfinite(r)
    if not use.any():
        return float("nan"), 0
    k, rr, ss = sk[use], r[use], sm[use]
    nk = int(k.max()) + 1
    n_s = np.bincount(k[ss], minlength=nk)
    n_d = np.bincount(k[~ss], minlength=nk)
    sum_s = np.bincount(k[ss], weights=rr[ss], minlength=nk)
    sum_d = np.bincount(k[~ss], weights=rr[~ss], minlength=nk)
    good = (n_s > 0) & (n_d > 0)
    if not good.any():
        return float("nan"), 0
    w = np.minimum(n_s, n_d)[good]
    diff = sum_d[good] / n_d[good] - sum_s[good] / n_s[good]
    return float(np.sum(w * diff) / np.sum(w)), int(n_s[good].sum())


def role_group(role: str) -> str:
    """Node-type group of a role: ``dead_end``, ``corridor``, ``t`` or the role itself."""
    for g in ("dead_end", "corridor", "t"):
        if role.startswith(g + "_"):
            return g
    return role


def role_test(
    d: np.ndarray,
    pairs: pd.DataFrame,
    n_perm: int,
    rng: np.random.Generator,
    mode: str = "cellpair",
) -> dict[str, Any]:
    """Role-generalisation contrast with two role-label permutation nulls.

    The dissimilarity is rank-transformed, residualised on the ranked controls
    (:data:`CONTROL_COLS` plus ``dir_diff``) and divided by the number of
    pairs before stratification (the contrast is a fraction of the rank
    range). Strata follow :func:`role_strata` (*mode*). With ``cellpair``
    strata, location is held fixed, so a code depending only on position
    gives no contrast; heading is handled by the ``dir_diff`` and HD
    regressors.

    Nulls: ``null`` permutes role labels across all non-root states;
    ``null_within`` permutes them only among states of the same node type
    (dead end, corridor, T-junction), so the node-type similarity shared by
    same-role pairs is kept in the null and only the direction-relative-to-
    root part of the role is tested. On synthetic populations whose activity
    depends only on graph or Euclidean position, the across-type null is
    anti-conservative (location-dependent geometry correlates with role) while
    the within-type null is calibrated, so ``null_within`` is the primary null.
    """
    y = d[pairs["i"].to_numpy(), pairs["j"].to_numpy()]
    C = pairs[[*CONTROL_COLS, "dir_diff"]].to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(C).all(axis=1)
    resid = np.full(y.shape, np.nan)
    if ok.sum() > C.shape[1] + 2:
        resid[ok] = (
            _residualise(stats.rankdata(y[ok])[:, None], _rank_cols(C[ok]))[:, 0] / ok.sum()
        )
    strata = np.where(ok, role_strata(pairs, mode), -1)
    obs, n_same = role_contrast(resid, pairs["same_role"].to_numpy(), strata)
    n_states = int(max(pairs["i"].max(), pairs["j"].max()) + 1) if len(pairs) else 0
    roles = np.empty(n_states, dtype=object)
    roles[pairs["i"].to_numpy()] = pairs["role_i"].to_numpy()
    roles[pairs["j"].to_numpy()] = pairs["role_j"].to_numpy()
    perm_idx = np.flatnonzero(~np.isin(roles, ["root", "corridor_other", "junction_other"]))
    groups = np.array([role_group(str(r)) for r in roles])
    ii, jj = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    null = np.full(n_perm, np.nan)
    null_w = np.full(n_perm, np.nan)
    if np.isfinite(obs):
        for p in range(n_perm):
            r2 = roles.copy()
            r2[perm_idx] = rng.permutation(roles[perm_idx])
            null[p] = role_contrast(resid, r2[ii] == r2[jj], strata)[0]
            r3 = roles.copy()
            for g in np.unique(groups[perm_idx]):
                gi = perm_idx[groups[perm_idx] == g]
                r3[gi] = rng.permutation(roles[gi])
            null_w[p] = role_contrast(resid, r3[ii] == r3[jj], strata)[0]
    role_counts = pd.Series(roles[perm_idx]).value_counts().to_dict() if n_states else {}
    return {
        "role_contrast": obs,
        "n_same_pairs": n_same,
        "null": null,
        "null_within": null_w,
        "role_counts": role_counts,
    }


def null_summary(observed: float, null: npt.ArrayLike) -> dict[str, float]:
    """z, null mean/sd and one-sided p (observed >= null) for a statistic."""
    nl = np.asarray(null, dtype=float)
    nl = nl[np.isfinite(nl)]
    if not np.isfinite(observed) or nl.size < 2:
        return {"z": np.nan, "null_mean": np.nan, "null_sd": np.nan, "p": np.nan}
    sd = float(nl.std())
    return {
        "z": (observed - float(nl.mean())) / sd if sd > 0 else np.nan,
        "null_mean": float(nl.mean()),
        "null_sd": sd,
        "p": float((1 + np.sum(nl >= observed)) / (1 + nl.size)),
    }


def mantel_null(
    d: np.ndarray,
    pairs: pd.DataFrame,
    directed: bool,
    n_perm: int,
    rng: np.random.Generator,
    keys: tuple[str, ...],
    design: dict[str, PartialDesign] | None = None,
) -> dict[str, np.ndarray]:
    """Statistics recomputed after permuting state identities of *d* (rows and columns)."""
    n = d.shape[0]
    out = {k: np.full(n_perm, np.nan) for k in keys}
    for p in range(n_perm):
        q = rng.permutation(n)
        s = geometry_stats(d[np.ix_(q, q)], pairs, directed, design)
        for k in keys:
            out[k][p] = s[k]
    return out


# ---------------------------------------------------------------------------
# Session-level pipeline
# ---------------------------------------------------------------------------


def zscore_rows(x: npt.ArrayLike, mask: npt.ArrayLike) -> np.ndarray:
    """Z-score rows with mean and SD over *mask* frames; non-finite and constant rows -> 0."""
    a = np.nan_to_num(np.asarray(x, dtype=np.float64))
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.zeros_like(a)
    mu = a[:, m].mean(axis=1, keepdims=True)
    sd = a[:, m].std(axis=1, keepdims=True)
    return np.where(sd > 0, (a - mu) / np.where(sd > 0, sd, 1.0), 0.0)


def draw_shifts(n_frames: int, n: int, min_shift: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` shifts uniform in ``[min_shift, n_frames - min_shift)`` (empty if too short)."""
    if n < 1 or n_frames <= 2 * min_shift:
        return np.zeros(0, dtype=np.int64)
    return rng.integers(min_shift, n_frames - min_shift, size=n).astype(np.int64)


GEOM_KEYS = (
    "prho_graph",
    "prho_euclid",
    "graph_minus_euclid",
    "prho_graph_nt",
    "prho_euclid_nt",
    "graph_minus_euclid_nt",
    "prho_graph_local",
    "prho_hd",
)


def analyse_states(
    x: np.ndarray,
    states: np.ndarray,
    n_states: int,
    hd_deg: np.ndarray,
    speed: np.ndarray,
    fps: float,
    maze: RoseMaze,
    directed: bool,
    params: GeometryParams,
    shifts: np.ndarray,
    rng: np.random.Generator,
    keep: np.ndarray | None = None,
    do_roles: bool = False,
) -> dict[str, Any]:
    """Observed statistics with shift and Mantel nulls for one state set.

    Returns a dict with ``n_states``, ``n_pairs``, ``rho_graph_euclid`` (Spearman
    between the two distances over pairs), observed statistics, and for each of
    :data:`GEOM_KEYS` (+ ``prho_dir``) the shift-null summary (``<key>_z``,
    ``<key>_p``) and Mantel p (``<key>_pmantel``); with *do_roles* the role
    contrast and its permutation null summary.
    """
    if keep is None:
        keep = kept_states(states, n_states, states.size, fps, params)
    out: dict[str, Any] = {"n_states": int(keep.size)}
    if keep.size < 6:
        out["reason"] = "fewer than 6 states"
        return out
    d, kept = cv_dissimilarity(x, states, n_states, fps, params, keep=keep)
    desc = state_descriptors(states, kept, hd_deg, speed, fps, params.profile_bin_s)
    pairs = pair_table(desc, maze, directed)
    design = geometry_design(pairs, directed)
    obs = geometry_stats(d, pairs, directed, design)
    y = d[pairs["i"].to_numpy(), pairs["j"].to_numpy()]
    obs["rho_graph"] = float(stats.spearmanr(y, pairs["graph"])[0])
    obs["rho_euclid"] = float(stats.spearmanr(y, pairs["euclid"])[0])
    out["n_pairs"] = len(pairs)
    out["n_cells_covered"] = int(np.unique(pairs[["cell_i", "cell_j"]].to_numpy()).size)
    out["rho_graph_euclid"] = float(stats.spearmanr(pairs["graph"], pairs["euclid"])[0])
    out.update(obs)
    keys = GEOM_KEYS + (("prho_dir",) if directed else ())
    null = {k: [] for k in keys + ("role_contrast",)}
    shift_roles: dict[str, list[float]] = {"cellpair": [], "geometry": []}
    for s in shifts:
        ds, _ = cv_dissimilarity(x, states, n_states, fps, params, shift=int(s), keep=kept)
        st = geometry_stats(ds, pairs, directed, design)
        for k in keys:
            null[k].append(st[k])
        if do_roles:
            for mode in shift_roles:
                shift_roles[mode].append(role_test(ds, pairs, 0, rng, mode)["role_contrast"])
    mant = mantel_null(d, pairs, directed, params.n_perm, rng, keys, design)
    for k in keys:
        ns = null_summary(obs[k], null[k])
        out[f"{k}_z"] = ns["z"]
        out[f"{k}_p"] = ns["p"]
        out[f"{k}_pmantel"] = null_summary(obs[k], mant[k])["p"]
    if do_roles:
        for mode, tag in (("cellpair", "role"), ("geometry", "rolegeom")):
            rt = role_test(d, pairs, params.n_perm, rng, mode)
            out[f"{tag}_contrast"] = rt["role_contrast"]
            out[f"{tag}_n_same_pairs"] = rt["n_same_pairs"]
            ns = null_summary(rt["role_contrast"], rt["null_within"])
            out[f"{tag}_z"], out[f"{tag}_p"] = ns["z"], ns["p"]
            ns = null_summary(rt["role_contrast"], rt["null"])
            out[f"{tag}_z_across"], out[f"{tag}_p_across"] = ns["z"], ns["p"]
            ns = null_summary(rt["role_contrast"], shift_roles[mode])
            out[f"{tag}_z_shift"], out[f"{tag}_p_shift"] = ns["z"], ns["p"]
        out["role_counts"] = rt["role_counts"]
        out["role_n_types_repeated"] = int(sum(v >= 2 for v in rt["role_counts"].values()))
    return out


# ---------------------------------------------------------------------------
# Across-session summaries
# ---------------------------------------------------------------------------


def wilcoxon_summary(v: npt.ArrayLike) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank vs 0 with matched-pairs rank-biserial effect size."""
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    out = {
        "n": int(a.size),
        "median": float(np.median(a)) if a.size else np.nan,
        "n_pos": int((a > 0).sum()),
    }
    nz = a[a != 0]
    if nz.size < 3:
        out.update({"W": np.nan, "p": np.nan, "rank_biserial": np.nan})
        return out
    res = stats.wilcoxon(nz)
    r = stats.rankdata(np.abs(nz))
    rb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    out.update({"W": float(res.statistic), "p": float(res.pvalue), "rank_biserial": float(rb)})
    return out


def animal_means(df: pd.DataFrame, col: str, animal_col: str = "animal_id") -> pd.Series:
    """Per-animal mean of *col* over sessions (NaN sessions dropped)."""
    return df.dropna(subset=[col]).groupby(animal_col)[col].mean()


# ---------------------------------------------------------------------------
# Session orchestration (dict of session arrays -> result rows)
# ---------------------------------------------------------------------------

CONDITIONS = ("all", "light", "dark")
MODES = ("cell", "directed")


def session_inputs(
    d: dict[str, Any], maze: RoseMaze, params: GeometryParams, signal: str = "spikes"
) -> dict[str, Any]:
    """Per-frame arrays for one session.

    Keys: ``x`` (z-scored activity over valid frames), ``cells``, ``dirs``,
    ``run`` (valid running frames), ``light`` (bool), ``hd``, ``speed``,
    ``fps``. Valid = no behavioural artefact and finite position.
    """
    from hm2p.analysis.population_maze import session_cells

    sig = np.nan_to_num(np.asarray(d[signal], dtype=float))
    n = sig.shape[1]

    def g(k: str) -> np.ndarray:
        return np.asarray(d[k], dtype=float)[:n]

    xm, ym = g("x_maze"), g("y_maze")
    valid = ~np.asarray(d["bad_behav"], dtype=bool)[:n] & np.isfinite(xm) & np.isfinite(ym)
    cells = session_cells(np.nan_to_num(xm), np.nan_to_num(ym), valid, maze)
    spd = g("speed_cm_s")
    run = valid & np.isfinite(spd) & (spd >= params.run_cm_s) & (cells >= 0)
    return {
        "x": zscore_rows(sig, valid),
        "cells": cells,
        "dirs": frame_directions(cells, maze),
        "run": run,
        "light": np.asarray(d["light_on"], dtype=bool)[:n],
        "hd": g("hd_deg"),
        "speed": spd,
        "fps": float(d["fps"]),
    }


def analyse_session(
    d: dict[str, Any],
    maze: RoseMaze,
    params: GeometryParams,
    signal: str = "spikes",
    modes: tuple[str, ...] = MODES,
) -> list[dict[str, Any]]:
    """Result rows (one per mode x condition) for one session.

    ``all`` uses every state passing the occupancy minimum over all running
    frames; ``light`` and ``dark`` use the states passing it in both
    conditions, so the paired light/dark comparison is over the same states.
    The same circular shifts are used for every mode and condition.
    """
    inp = session_inputs(d, maze, params, signal)
    n = inp["x"].shape[1]
    rng = np.random.default_rng(params.seed)
    shifts = draw_shifts(n, params.n_shifts, int(round(params.min_shift_s * inp["fps"])), rng)
    rows = []
    for mode in modes:
        directed = mode == "directed"
        ns = len(maze.cell_list) * (4 if directed else 1)
        st = {
            "all": frame_states(inp["cells"], inp["dirs"], inp["run"], directed),
            "light": frame_states(inp["cells"], inp["dirs"], inp["run"] & inp["light"], directed),
            "dark": frame_states(inp["cells"], inp["dirs"], inp["run"] & ~inp["light"], directed),
        }
        kp = {k: kept_states(v, ns, n, inp["fps"], params) for k, v in st.items()}
        common = np.intersect1d(kp["light"], kp["dark"])
        for cond in CONDITIONS:
            keep = kp["all"] if cond == "all" else common
            res = analyse_states(
                inp["x"],
                st[cond],
                ns,
                inp["hd"],
                inp["speed"],
                inp["fps"],
                maze,
                directed,
                params,
                shifts,
                rng,
                keep=keep,
                do_roles=directed,
            )
            res.update(
                {
                    "mode": mode,
                    "condition": cond,
                    "n_neurons": int(inp["x"].shape[0]),
                    "n_run_frames": int((st[cond] >= 0).sum()),
                    "n_shifts": int(shifts.size),
                }
            )
            rows.append(res)
    meta = d.get("meta", {})
    for r in rows:
        r["exp_id"] = meta.get("exp_id", "")
        r["animal_id"] = str(meta.get("animal_id", ""))
        r["celltype"] = meta.get("celltype", "")
        r["signal"] = signal
    return rows


def across_animals(sessions: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    """Animal-level Wilcoxon (vs 0) of per-animal means, per mode x condition x column.

    Also reports the session-level Wilcoxon and the number of sessions with
    shift-null p < 0.05 where a ``<col>_p`` column exists.
    """
    rows = []
    for (mode, cond), grp in sessions.groupby(["mode", "condition"]):
        for c in cols:
            if c not in grp:
                continue
            am = animal_means(grp, c)
            a = wilcoxon_summary(am.to_numpy())
            s = wilcoxon_summary(grp[c].to_numpy(dtype=float))
            pcol = c[:-2] + "_p" if c.endswith("_z") else c + "_p"
            row = {
                "mode": mode,
                "condition": cond,
                "stat": c,
                "n_animals": a["n"],
                "animal_median": a["median"],
                "animal_n_pos": a["n_pos"],
                "animal_p": a["p"],
                "animal_rank_biserial": a["rank_biserial"],
                "n_sessions": s["n"],
                "session_median": s["median"],
                "session_p": s["p"],
            }
            if pcol in grp and pcol != c:
                row["n_sessions_p05"] = int((grp[pcol] < 0.05).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def light_dark_paired(sessions: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    """Light minus dark per session (same states); animal-level Wilcoxon of animal means."""
    rows = []
    for mode, grp in sessions.groupby("mode"):
        lt = grp[grp["condition"] == "light"].set_index("exp_id")
        dk = grp[grp["condition"] == "dark"].set_index("exp_id")
        ids = lt.index.intersection(dk.index)
        for c in cols:
            if c not in lt:
                continue
            diff = pd.DataFrame(
                {
                    "diff": lt.loc[ids, c].astype(float) - dk.loc[ids, c].astype(float),
                    "animal_id": lt.loc[ids, "animal_id"],
                }
            ).dropna()
            a = wilcoxon_summary(animal_means(diff, "diff").to_numpy())
            s = wilcoxon_summary(diff["diff"].to_numpy())
            rows.append(
                {
                    "mode": mode,
                    "stat": c,
                    "n_sessions": s["n"],
                    "light_median": float(lt.loc[ids, c].median()),
                    "dark_median": float(dk.loc[ids, c].median()),
                    "session_median_diff": s["median"],
                    "session_p": s["p"],
                    "n_animals": a["n"],
                    "animal_median_diff": a["median"],
                    "animal_n_light_gt_dark": a["n_pos"],
                    "animal_p": a["p"],
                    "animal_rank_biserial": a["rank_biserial"],
                }
            )
    return pd.DataFrame(rows)
