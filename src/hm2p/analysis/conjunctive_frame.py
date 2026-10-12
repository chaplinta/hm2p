"""Controls and reference-frame tests for direction-dependent place coding in the maze.

Single-cell mutual information shows that position information rises when head
direction is held fixed (I(A; P | H) > I(A; P)), i.e. place responses depend on
direction. This module provides the pieces to ask (1) whether that survives
conditioning on acceleration, (2) whether travel direction or head direction
carries it, (3) in which reference frame a cell's direction preference is
consistent across maze locations, and (4) whether directed-state maps are
stable between light and dark.

Travel direction
----------------
Maze corridors are axis-aligned, so travel direction is one of four maze-axis
directions (0 = +x, 1 = +y, 2 = -x, 3 = -y), taken as the axis of largest
component of the Gaussian-smoothed maze-coordinate velocity. A direction is
kept only if the maze cell has a neighbour in that direction or the opposite
one (i.e. the move lies along a corridor through the cell).

Reference frames
----------------
Direction preferences are compared on a common set of locations: (maze cell,
travel axis) with both opposite directions sampled. Preference at location l
is p_l = (r+ - r-) / (|r+| + |r-|), r = occupancy-normalised activity while
travelling in the positive / negative axis direction (frames grouped into
passes, contiguous frames with the same location and direction; locations
need >= ``min_passes`` passes and >= ``min_frames`` frames each way). Each
frame maps a location to a sign sigma_l (which of the two directions is the
frame's positive label; 0 if the frame does not contrast them):

- ``allo``: lab/maze-axis frame; sigma = +1 everywhere, one group per axis
  (a cell consistent in this frame prefers one compass direction per axis).
- ``root``: away from vs toward the maze root, the graph centroid (cell
  minimising summed shortest-path distance; the central T-junction). The maze
  is a tree, so every edge has a label.
- ``deadend``: toward vs away from the nearest dead end (graph distance to the
  nearest dead end decreases); ties unlabelled. On this maze it agrees with
  ``root`` on 36 of 40 labelled directed edges.
- ``ego_side``: side arm of the next T-junction ahead on the left vs right.
  On this maze no location has both opposite directions labelled, so it cannot
  be tested as a within-location contrast (reported as such).
- ``ego_turn``: the turn actually taken at the next T-junction, left vs right,
  compared within the same directed state (location = cell x direction), i.e.
  prospective coding on the same approach.

Statistics per cell and frame: (1) a leave-one-location-out score, each
location's preference predicted from the frame-aligned sum of the other
locations in its group, score = sum(p_l * prediction_l) / sum|p_l| over all
qualifying locations (so frames that leave locations undefined are not
favoured); (2) consistency = |mean sigma_l p_l| (vector over groups). Null:
direction labels permuted among passes within each location (occupancy and
within-pass autocorrelation kept); z and one-sided p. Because sign patterns of
different frames are partly correlated across the sampled locations, a cell
coding one frame is also above the null in others; frames are compared by
paired score differences, not by significance against the null alone.

Left/right use the maze-coordinate handedness (+y counter-clockwise of +x).
If the maze coordinates are mirrored relative to the room, left and right swap
for the whole session; the statistics are sign-free and unaffected.

Splitter/prospective coding in the maze context: Wood ER, Dudchenko PA,
Robitsek RJ, Eichenbaum H. 2000. "Hippocampal neurons encode information about
different types of memory episodes occurring in the same location." Neuron
27:623-633. doi:10.1016/S0896-6273(00)00071-4
Route- and reference-frame-dependent RSC activity: Alexander AS, Nitz DA. 2015.
"Retrosplenial cortex maps the conjunction of internal and external spaces."
Nature Neuroscience 18:1143-1151. doi:10.1038/nn.4058
Mutual information and shuffle bias correction: Skaggs WE, McNaughton BL,
Gothard KM, Markus EJ. 1993. "An information-theoretic approach to deciphering
the hippocampal code." NIPS 5:1030-1037; Panzeri S, Senatore R, Montemurro MA,
Petersen RS. 2007. "Correcting for the sampling bias problem in spike train
information measures." J Neurophysiol 98:1064-1072. doi:10.1152/jn.00559.2007
Maze: Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth
show rapid learning, sudden insight, and efficient exploration." eLife
10:e66175. doi:10.7554/eLife.66175
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from hm2p.analysis.behaviour_mi import quantile_bins
from hm2p.analysis.population_maze import cell_stays
from hm2p.maze.topology import RoseMaze

DIRS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))
FRAMES = ("allo", "root", "deadend", "ego_side", "ego_turn")


# ---------------------------------------------------------------------------
# Mutual information, vectorised over circular shifts
# ---------------------------------------------------------------------------


def compact(x: npt.ArrayLike) -> np.ndarray:
    """Relabel non-negative ints to 0..k-1 (negatives stay -1)."""
    v = np.asarray(x, dtype=np.int64)
    out = np.full(v.shape, -1, dtype=np.int64)
    ok = v >= 0
    if ok.any():
        out[ok] = np.unique(v[ok], return_inverse=True)[1]
    return out


def joint(*arrs: npt.ArrayLike) -> np.ndarray:
    """Joint label of several int arrays (-1 if any is negative)."""
    a = [np.asarray(x, dtype=np.int64) for x in arrs]
    out = np.zeros(a[0].shape, dtype=np.int64)
    bad = np.zeros(a[0].shape, dtype=bool)
    for x in a:
        bad |= x < 0
        out = out * (int(x.max(initial=0)) + 1) + np.maximum(x, 0)
    out[bad] = -1
    return out


def cond_mi_rows(bins: npt.ArrayLike, y: npt.ArrayLike, c: npt.ArrayLike | None) -> np.ndarray:
    """I(bins_s; y | c) in bits for every row s of ``bins`` (S, N).

    All entries must be valid (>= 0); ``c=None`` gives plain MI. Equal to
    :func:`hm2p.analysis.behaviour_mi.conditional_mi` row by row.
    """
    b = np.atleast_2d(np.asarray(bins, dtype=np.int64))
    yy = compact(y)
    cc = np.zeros(yy.shape, dtype=np.int64) if c is None else compact(c)
    s_n, n = b.shape
    if n == 0:
        return np.zeros(s_n)
    k = int(b.max()) + 1
    ny, nc = int(yy.max()) + 1, int(cc.max()) + 1
    idx = (cc * ny + yy)[None, :] * k + b
    idx = idx + (np.arange(s_n) * nc * ny * k)[:, None]
    cnt = np.bincount(idx.ravel(), minlength=s_n * nc * ny * k).reshape(s_n, nc, ny, k)
    cnt = cnt.astype(float)
    n_c = cnt.sum(axis=(2, 3), keepdims=True)
    n_cy = cnt.sum(axis=3, keepdims=True)
    n_ck = cnt.sum(axis=2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = cnt * np.log2(cnt * n_c / (n_cy * n_ck))
    term[cnt == 0] = 0.0
    return term.sum(axis=(1, 2, 3)) / n


def shifted_bins(
    x: npt.ArrayLike, used: npt.ArrayLike, shifts: npt.ArrayLike, n_bins: int = 4
) -> np.ndarray:
    """Quantile bins of ``roll(x, s)[used]`` for each shift (row 0 = shift 0)."""
    sig = np.asarray(x, dtype=float)
    u = np.asarray(used, dtype=bool)
    sh = np.concatenate([[0], np.asarray(shifts, dtype=np.int64)])
    return np.stack([quantile_bins(np.roll(sig, int(s))[u], n_bins) for s in sh])


def mi_with_null(bins: np.ndarray, y: npt.ArrayLike, c: npt.ArrayLike | None) -> dict[str, float]:
    """Observed MI (row 0) with shift-null summary (rows 1..)."""
    v = cond_mi_rows(bins, y, c)
    obs, null = float(v[0]), v[1:]
    if null.size == 0:
        return {"mi": obs, "null_mean": np.nan, "mi_debiased": np.nan, "z": np.nan, "p": np.nan}
    m, sd = float(null.mean()), float(null.std())
    return {
        "mi": obs,
        "null_mean": m,
        "mi_debiased": obs - m,
        "z": (obs - m) / sd if sd > 0 else np.nan,
        "p": float((1 + np.sum(null >= obs)) / (1 + null.size)),
    }


# ---------------------------------------------------------------------------
# Kinematic labels
# ---------------------------------------------------------------------------


def _fill_nan(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if ok.all() or not ok.any():
        return np.nan_to_num(x)
    return np.interp(np.arange(x.size), np.flatnonzero(ok), x[ok])


def acceleration(speed: npt.ArrayLike, fps: float, sigma_s: float = 0.25) -> np.ndarray:
    """d(speed)/dt (units/s) of Gaussian-smoothed speed; NaN where speed is NaN."""
    s = np.asarray(speed, dtype=float)
    sm = gaussian_filter1d(_fill_nan(s), max(sigma_s * fps, 1e-6))
    a = np.gradient(sm) * fps
    a[~np.isfinite(s)] = np.nan
    return a


def tertiles(x: npt.ArrayLike, mask: npt.ArrayLike) -> np.ndarray:
    """Tertile bin 0..2 of ``x`` among ``mask`` frames (others -1)."""
    v = np.asarray(x, dtype=float)
    m = np.asarray(mask, dtype=bool) & np.isfinite(v)
    out = np.full(v.shape, -1, dtype=np.int64)
    if m.sum() >= 3:
        edges = np.quantile(v[m], [1 / 3, 2 / 3])
        out[m] = np.searchsorted(edges, v[m], side="right")
    return out


def allowed_dirs(maze: RoseMaze) -> np.ndarray:
    """(n_cells, 4) bool: move along d is along a corridor through the cell."""
    out = np.zeros((maze.n_cells, 4), dtype=bool)
    for c in maze.cell_list:
        i = maze.cell_to_idx[c]
        for d, (dx, dy) in enumerate(DIRS):
            fwd, back = (c[0] + dx, c[1] + dy), (c[0] - dx, c[1] - dy)
            out[i, d] = fwd in maze.adj[c] or back in maze.adj[c]
    return out


def travel_direction(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    cells: npt.ArrayLike,
    maze: RoseMaze,
    fps: float,
    sigma_s: float = 0.3,
) -> np.ndarray:
    """Maze-axis travel direction 0..3 per frame (-1 if position NaN, cell -1 or not allowed)."""
    xx, yy = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    c = np.asarray(cells, dtype=np.int64)
    sg = max(sigma_s * fps, 1e-6)
    vx = np.gradient(gaussian_filter1d(_fill_nan(xx), sg))
    vy = np.gradient(gaussian_filter1d(_fill_nan(yy), sg))
    proj = np.stack([vx, vy, -vx, -vy], axis=1)
    d = np.argmax(proj, axis=1).astype(np.int64)
    ok = np.isfinite(xx) & np.isfinite(yy) & (c >= 0) & (np.hypot(vx, vy) > 0)
    ok[ok] &= allowed_dirs(maze)[c[ok], d[ok]]
    d[~ok] = -1
    return d


# ---------------------------------------------------------------------------
# Reference-frame labels of directed states
# ---------------------------------------------------------------------------


def _dir_of(a: tuple[int, int], b: tuple[int, int]) -> int:
    return DIRS.index((b[0] - a[0], b[1] - a[1]))


def turn_code(arrive: int, leave: int) -> int:
    """Egocentric turn: 1 left, 0 right, 2 straight, -1 U-turn."""
    if leave == arrive:
        return 2
    if leave == (arrive + 1) % 4:
        return 1
    if leave == (arrive + 3) % 4:
        return 0
    return -1


def maze_root(maze: RoseMaze) -> int:
    """Graph centroid: cell index minimising the summed shortest-path distance."""
    return int(np.argmin(maze.dist.sum(axis=1)))


@dataclass
class StateLabels:
    """Per directed state (n_cells, 4) labels; -1 = undefined."""

    allowed: np.ndarray
    root_out: np.ndarray
    toward_deadend: np.ndarray
    ego_side: np.ndarray


def state_labels(maze: RoseMaze, root: int | None = None) -> StateLabels:
    """Root-relative, dead-end-relative and egocentric-geometry labels of directed states."""
    r = maze_root(maze) if root is None else root
    de = [maze.cell_to_idx[c] for c in maze.dead_ends]
    dde = maze.dist[:, de].min(axis=1)
    allowed = allowed_dirs(maze)
    shape = (maze.n_cells, 4)
    root_out = np.full(shape, -1, dtype=np.int64)
    toward = np.full(shape, -1, dtype=np.int64)
    side = np.full(shape, -1, dtype=np.int64)
    for c in maze.cell_list:
        i = maze.cell_to_idx[c]
        for d, (dx, dy) in enumerate(DIRS):
            if not allowed[i, d]:
                continue
            fwd = (c[0] + dx, c[1] + dy)
            if fwd in maze.adj[c]:
                a, b = c, fwd
            else:
                a, b = (c[0] - dx, c[1] - dy), c
            ia, ib = maze.cell_to_idx[a], maze.cell_to_idx[b]
            root_out[i, d] = int(maze.dist[r, ib] > maze.dist[r, ia])
            if dde[ib] != dde[ia]:
                toward[i, d] = int(dde[ib] < dde[ia])
            # walk to the next node that is not a 2-neighbour corridor
            prev, cur = a, b
            while len(maze.adj[cur]) == 2:
                nxt = [n for n in maze.adj[cur] if n != prev][0]
                prev, cur = cur, nxt
            if len(maze.adj[cur]) != 3:
                continue
            arr = _dir_of(prev, cur)
            turns = {turn_code(arr, _dir_of(cur, n)) for n in maze.adj[cur] if n != prev}
            if turns == {2, 1}:
                side[i, d] = 1
            elif turns == {2, 0}:
                side[i, d] = 0
    return StateLabels(allowed, root_out, toward, side)


def upcoming_turn(cells: npt.ArrayLike, maze: RoseMaze) -> np.ndarray:
    """Per frame, the turn taken at the next T-junction on the current approach.

    For every T-junction stay with adjacent, different arrival and exit cells,
    the junction frames and the preceding corridor stays leading to it
    (walking back while the cells are 2-neighbour corridors on a continuous
    path) get the turn code (1 left, 0 right, 2 straight). Other frames -1.
    """
    c = np.asarray(cells, dtype=np.int64)
    out = np.full(c.shape, -1, dtype=np.int64)
    cell, start, end = cell_stays(c)
    cl = maze.cell_list
    for k in range(1, cell.size - 1):
        j, p, q = int(cell[k]), int(cell[k - 1]), int(cell[k + 1])
        if len(maze.adj[cl[j]]) != 3 or maze.dist[p, j] != 1 or maze.dist[q, j] != 1 or p == q:
            continue
        t = turn_code(_dir_of(cl[p], cl[j]), _dir_of(cl[j], cl[q]))
        out[start[k] : end[k]] = t
        m, nxt = k - 1, j
        while (
            m >= 0 and len(maze.adj[cl[int(cell[m])]]) == 2 and maze.dist[int(cell[m]), nxt] == 1
        ):
            if m + 2 <= k and int(cell[m]) == int(cell[m + 2]):
                break  # reversal
            out[start[m] : end[m]] = t
            nxt = int(cell[m])
            m -= 1
    return out


def axis_locations(
    cells: npt.ArrayLike, tdir: npt.ArrayLike, mask: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Location = (maze cell, travel axis) as ``cell * 2 + axis``; label 1 = +x / +y travel.

    Frames outside ``mask`` or without cell / direction get -1.
    """
    c = np.asarray(cells, dtype=np.int64)
    d = np.asarray(tdir, dtype=np.int64)
    m = np.asarray(mask, dtype=bool) & (c >= 0) & (d >= 0)
    loc = np.where(m, c * 2 + d % 2, -1)
    lab = np.where(m, (d < 2).astype(np.int64), -1)
    return loc, lab


def frame_signs(maze: RoseMaze) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per frame, the sign and group of every (cell, axis) location.

    ``sigma[l]`` = +1 if travel in the positive axis direction is the frame's
    positive label at that location and the opposite direction its negative
    label, -1 for the reverse, 0 where the frame does not contrast the two
    directions. ``group[l]`` = locations whose preferences the frame says
    should agree (both axes for maze-relative frames; one group per axis for
    ``allo``). Frames: allo, root (away from the root), deadend (toward the
    nearest dead end), ego_side (side arm of the next junction on the left).
    """
    sl = state_labels(maze)
    n_loc = 2 * maze.n_cells
    axis = np.arange(n_loc) % 2
    cell = np.arange(n_loc) // 2
    ok = sl.allowed[cell, axis] & sl.allowed[cell, axis + 2]
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "allo": (ok.astype(float), axis.astype(np.int64))
    }
    for name, arr in (
        ("root", sl.root_out),
        ("deadend", sl.toward_deadend),
        ("ego_side", sl.ego_side),
    ):
        pos, neg = arr[cell, axis], arr[cell, axis + 2]
        sig = np.where((pos == 1) & (neg == 0), 1.0, np.where((pos == 0) & (neg == 1), -1.0, 0.0))
        out[name] = (np.where(ok, sig, 0.0), np.zeros(n_loc, dtype=np.int64))
    return out


def turn_locations(
    cells: npt.ArrayLike, tdir: npt.ArrayLike, turn: npt.ArrayLike, mask: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Location = directed state ``cell * 4 + dir``; label = upcoming turn left (1) / right (0)."""
    c = np.asarray(cells, dtype=np.int64)
    d = np.asarray(tdir, dtype=np.int64)
    t = np.asarray(turn, dtype=np.int64)
    m = np.asarray(mask, dtype=bool) & (c >= 0) & (d >= 0) & (t >= 0) & (t <= 1)
    return np.where(m, c * 4 + d, -1), np.where(m, t, -1)


# ---------------------------------------------------------------------------
# Frame consistency
# ---------------------------------------------------------------------------


def passes(loc: npt.ArrayLike, lab: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pass id per frame (-1 unused) plus per-pass location and label."""
    lo = np.asarray(loc, dtype=np.int64)
    la = np.asarray(lab, dtype=np.int64)
    used = (lo >= 0) & (la >= 0)
    key = np.where(used, lo * 2 + la, -1)
    new = used.copy()
    new[1:] &= (key[1:] != key[:-1]) | ~used[:-1]
    pid = np.cumsum(new) - 1
    pid[~used] = -1
    first = np.flatnonzero(new)
    return pid, lo[first], la[first]


def _prefs(r1: np.ndarray, r0: np.ndarray) -> np.ndarray:
    den = np.abs(r1) + np.abs(r0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (r1 - r0) / den
    return np.where(den > 0, p, 0.0)


def _rates(
    sums: np.ndarray, cnt: np.ndarray, ploc: np.ndarray, plab: np.ndarray, n_loc: int
) -> tuple:
    """(cells, n_loc) rates for label 1 and 0 from per-pass sums."""
    idx = ploc * 2 + plab
    tot = np.zeros((sums.shape[0], 2 * n_loc))
    np.add.at(tot.T, idx, sums.T)
    n = np.bincount(idx, weights=cnt, minlength=2 * n_loc)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = tot / n
    return r[:, 1::2], r[:, 0::2], n


def lolo_score(
    p: np.ndarray, valid: np.ndarray, sigma: np.ndarray, group: np.ndarray
) -> np.ndarray:
    """Leave-one-location-out frame score per cell.

    For each valid location with ``sigma != 0`` the preference is predicted
    as ``sigma_l * sign(sum over the other valid locations of the same group
    of sigma * p)``; score = sum of ``p_l * prediction`` / sum of ``|p_l|``
    over all valid locations (in [-1, 1]; 1 = every location's preference
    follows from the others under this frame).

    Parameters
    ----------
    p : (cells, L) preferences; valid : (L,) bool; sigma : (L,); group : (L,) int.
    """
    s = np.where(valid, sigma, 0.0)
    sp = p * s
    tot = np.zeros((p.shape[0], int(group.max()) + 1))
    np.add.at(tot.T, group, sp.T)
    pred = np.sign(tot[:, group] - sp) * s
    den = np.abs(np.where(valid, p, 0.0)).sum(axis=1)
    num = (np.where(valid, p, 0.0) * pred).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def location_prefs(
    signal: np.ndarray,
    pid: np.ndarray,
    ploc: np.ndarray,
    plab: np.ndarray,
    n_loc: int,
) -> np.ndarray:
    """(cells, n_loc) preference (r1 - r0) / (|r1| + |r0|) from frames grouped in passes."""
    used = pid >= 0
    cnt = np.bincount(pid[used], minlength=ploc.size).astype(float)
    sums = np.zeros((signal.shape[0], ploc.size))
    np.add.at(sums.T, pid[used], signal[:, used].T)
    r1, r0, _ = _rates(sums, cnt, ploc, plab, n_loc)
    return _prefs(r1, r0)


def frame_scores(
    signal: npt.ArrayLike,
    loc: npt.ArrayLike,
    lab: npt.ArrayLike,
    frames: dict[str, tuple[np.ndarray, np.ndarray]],
    n_shuffles: int,
    rng: np.random.Generator,
    min_passes: int = 3,
    min_frames: int = 10,
) -> pd.DataFrame:
    """Per cell and frame: cross-validated frame score and consistency vs a pass-label null.

    Locations need >= ``min_passes`` passes and >= ``min_frames`` frames for
    both labels (decided on the observed labels). Null: labels permuted among
    passes within each location. Returns roi, frame, n_loc (qualifying
    locations), n_def (of those, contrasted by the frame), score, score_null_mean,
    score_null_sd, score_z, score_p, consistency (|mean sigma*p| over defined
    locations, groups combined as a vector), cons_z, cons_p, mean_signed
    (mean sigma*p over defined locations, all groups).
    """
    sig = np.atleast_2d(np.nan_to_num(np.asarray(signal, dtype=float)))
    n_cells = sig.shape[0]
    pid, ploc, plab = passes(loc, lab)
    names = list(frames)
    n_loc_all = len(next(iter(frames.values()))[0])
    if ploc.size:
        cnt = np.bincount(pid[pid >= 0], minlength=ploc.size).astype(float)
        npass = np.zeros((n_loc_all, 2))
        nfr = np.zeros((n_loc_all, 2))
        np.add.at(npass, (ploc, plab), 1)
        np.add.at(nfr, (ploc, plab), cnt)
        valid = (npass.min(axis=1) >= min_passes) & (nfr.min(axis=1) >= min_frames)
    else:
        valid = np.zeros(n_loc_all, dtype=bool)

    def stats_of(p: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        res = {}
        for f in names:
            sgm, grp = frames[f]
            dv = valid & (sgm != 0)
            sp = np.where(dv, p * sgm, 0.0)
            vec = np.zeros((n_cells, int(grp.max()) + 1))
            np.add.at(vec.T, grp, sp.T)
            nd = max(int(dv.sum()), 1)
            res[f] = (lolo_score(p, valid, sgm, grp), np.linalg.norm(vec, axis=1) / nd)
        return res

    rows = []
    if valid.sum() < 2:
        for f in names:
            rows.append(
                pd.DataFrame(
                    {"roi": np.arange(n_cells), "frame": f, "n_loc": int(valid.sum()), "n_def": 0}
                )
            )
        return pd.concat(rows, ignore_index=True)
    p_obs = location_prefs(sig, pid, ploc, plab, n_loc_all)
    obs = stats_of(p_obs)
    # within-location permutation of pass labels (only qualifying locations matter)
    order_loc = np.argsort(ploc, kind="stable")
    nulls = {f: (np.zeros((n_shuffles, n_cells)), np.zeros((n_shuffles, n_cells))) for f in names}
    for s in range(n_shuffles):
        perm = np.lexsort((rng.random(ploc.size), ploc))
        new = np.empty_like(plab)
        new[perm] = plab[order_loc]
        st_ = stats_of(location_prefs(sig, pid, ploc, new, n_loc_all))
        for f in names:
            nulls[f][0][s], nulls[f][1][s] = st_[f]

    def zp(o: np.ndarray, nl: np.ndarray) -> tuple[np.ndarray, ...]:
        if n_shuffles == 0:
            nan = np.full(n_cells, np.nan)
            return nan, nan, nan, nan
        m, sd = nl.mean(axis=0), nl.std(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(sd > 0, (o - m) / sd, np.nan)
        p = (1 + (nl >= o[None, :]).sum(axis=0)) / (1 + n_shuffles)
        return m, sd, z, p

    for f in names:
        sgm, _ = frames[f]
        dv = valid & (sgm != 0)
        sc, co = obs[f]
        m, sd, z, p = zp(sc, nulls[f][0])
        _, _, cz, cp = zp(co, nulls[f][1])
        with np.errstate(invalid="ignore"):
            signed = (
                (np.where(dv, p_obs * sgm, 0.0).sum(axis=1) / dv.sum()) if dv.any() else np.nan
            )
        rows.append(
            pd.DataFrame(
                {
                    "roi": np.arange(n_cells),
                    "frame": f,
                    "n_loc": int(valid.sum()),
                    "n_def": int(dv.sum()),
                    "score": sc,
                    "score_null_mean": m,
                    "score_null_sd": sd,
                    "score_z": z,
                    "score_p": p,
                    "consistency": co if dv.any() else np.nan,
                    "cons_z": cz if dv.any() else np.nan,
                    "cons_p": cp if dv.any() else np.nan,
                    "mean_signed": signed,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def residualise(signal: npt.ArrayLike, strata: npt.ArrayLike) -> np.ndarray:
    """Subtract each cell's mean activity within each stratum (strata < 0 unchanged)."""
    sig = np.atleast_2d(np.asarray(signal, dtype=float)).copy()
    s = np.asarray(strata, dtype=np.int64)
    for k in np.unique(s[s >= 0]):
        m = s == k
        sig[:, m] -= sig[:, m].mean(axis=1, keepdims=True)
    return sig


# ---------------------------------------------------------------------------
# Directed-state maps and light/dark stability
# ---------------------------------------------------------------------------


def state_map(
    signal: npt.ArrayLike,
    state: npt.ArrayLike,
    mask: npt.ArrayLike,
    n_states: int,
    min_frames: int = 10,
) -> np.ndarray:
    """(cells, n_states) occupancy-normalised activity; NaN where occupancy < ``min_frames``."""
    sig = np.atleast_2d(np.asarray(signal, dtype=float))
    st = np.asarray(state, dtype=np.int64)
    m = np.asarray(mask, dtype=bool) & (st >= 0)
    occ = np.bincount(st[m], minlength=n_states).astype(float)
    tot = np.zeros((sig.shape[0], n_states))
    np.add.at(tot.T, st[m], sig[:, m].T)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = tot / occ
    r[:, occ < min_frames] = np.nan
    return r


def direction_residual(maps: np.ndarray, n_dirs: int = 4) -> np.ndarray:
    """Subtract each maze cell's mean over its sampled directions (cells with < 2 dirs -> NaN)."""
    m = np.atleast_2d(maps).reshape(maps.shape[0], -1, n_dirs)
    nok = np.isfinite(m).sum(axis=2, keepdims=True)
    mu = np.nansum(m, axis=2, keepdims=True) / np.maximum(nok, 1)
    out = m - mu
    out[np.broadcast_to(nok < 2, out.shape)] = np.nan
    return out.reshape(maps.shape)


def rowwise_spearman(a: np.ndarray, b: np.ndarray, min_n: int = 5) -> np.ndarray:
    """Spearman rho per row over columns finite in both (NaN if < ``min_n``)."""
    out = np.full(a.shape[0], np.nan)
    for i in range(a.shape[0]):
        ok = np.isfinite(a[i]) & np.isfinite(b[i])
        if ok.sum() >= min_n and np.ptp(a[i, ok]) > 0 and np.ptp(b[i, ok]) > 0:
            out[i] = stats.spearmanr(a[i, ok], b[i, ok]).statistic
    return out


def epoch_parity(light: npt.ArrayLike) -> np.ndarray:
    """0/1 parity of each frame's light epoch ordinal within its condition."""
    lt = np.asarray(light, dtype=bool)
    if lt.size == 0:
        return np.zeros(0, dtype=np.int64)
    start = np.r_[True, lt[1:] != lt[:-1]]
    epoch = np.cumsum(start) - 1  # conditions alternate, so epoch // 2 is the ordinal
    return (epoch // 2) % 2


# ---------------------------------------------------------------------------
# Animal-level statistics
# ---------------------------------------------------------------------------


def wilcoxon_summary(v: npt.ArrayLike) -> dict[str, Any]:
    """Wilcoxon signed-rank vs 0 (two-sided) with matched-pairs rank-biserial."""
    x = np.asarray(v, dtype=float)
    x = x[np.isfinite(x)]
    out: dict[str, Any] = {"n": int(x.size), "median": float(np.median(x)) if x.size else np.nan}
    nz = x[x != 0]
    if nz.size < 3:
        out.update(p=np.nan, rank_biserial=np.nan, n_pos=int((x > 0).sum()))
        return out
    ranks = stats.rankdata(np.abs(nz))
    wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    out.update(
        p=float(stats.wilcoxon(nz).pvalue),
        rank_biserial=float((wp - wm) / (wp + wm)),
        n_pos=int((x > 0).sum()),
    )
    return out


def animal_test(df: pd.DataFrame, col: str, animal_col: str = "animal_id") -> dict[str, Any]:
    """Median per animal of ``col`` (over its cells), then Wilcoxon vs 0 across animals."""
    a = df.groupby(animal_col)[col].median().dropna()
    r = wilcoxon_summary(a.to_numpy())
    r["n_cells"] = int(df[col].notna().sum())
    r["cell_median"] = float(df[col].median())
    r["frac_cells_pos"] = float((df[col].dropna() > 0).mean()) if df[col].notna().any() else np.nan
    return r
