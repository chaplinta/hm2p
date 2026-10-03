"""Running bouts in maze context: what a run-state cell's activity depends on.

A cell whose activity tracks run vs still could carry a generic locomotor
state signal, or it could carry information about where the run goes. This
module annotates each running bout with its maze context and asks, per cell,
whether bout-level activity depends on that context once run duration and
speed are accounted for.

Bout annotation (:func:`annotate_bouts`)
----------------------------------------
Positions are discretised onto the 23-cell maze graph
(:func:`hm2p.maze.discretize.discretize_position_fast`). Cell occupancies
shorter than ``min_dwell_frames`` are treated as boundary jitter and dropped
before the cell sequence is built. For each bout: start and end cell and their
node type (dead end / corridor / junction), the path length in cells (sum of
graph distances between consecutive cells), duration, mean speed, fraction of
frames with the lights on, whether the bout enters a cell not occupied earlier
in the session (and the fraction of entered cells that are novel), whether the
bout ends in an arm not occupied earlier in the current light epoch, the
change in graph distance to the nearest dead end, and the run type:

- ``into_dead_end``: ends in a dead end, starts elsewhere;
- ``out_of_dead_end``: starts in a dead end, ends elsewhere;
- ``junction_to_junction``: neither starts nor ends in a dead end (the name
  follows the analysis plan; start/end may be corridor cells);
- ``dead_end_to_dead_end``: starts and ends in (different) dead ends;
- ``no_transition``: no cell change during the bout;
- ``unknown``: no valid position during the bout.

An *arm* is a connected component of non-junction cells (corridors and dead
ends); each junction is its own unit. In this maze every arm is a single cell
except the corridor + dead end pair at (3, 3)-(3, 4).

Per-cell statistics (:func:`cell_statistics`)
---------------------------------------------
Bout activity is the mean signal over the bout's frames. Duration and mean
speed are **regressed out by rank**: bout activity is rank-transformed and the
ranks are residualised on an intercept and the ranks of duration and mean
speed (and the rank of bout onset time where stated). Contrasts are
differences of mean residual rank between two bout groups, in units of the
fraction of bouts (0.1 = one group sits ten percentile points higher).

1. ``approach_return``: into-dead-end minus out-of-dead-end bouts.
2. ``novel_repeat``: bouts entering at least one session-novel cell minus bouts
   entering only previously visited cells (onset time is an additional
   covariate, because novel bouts occur early in the session and a slow drift
   in the signal would otherwise appear as a novelty effect). ``edge_decay_rho``:
   partial Spearman correlation across bouts between activity and the median
   number of prior traversals of the bout's directed edges, controlling for
   duration, speed and onset time.
3. ``route_reliability``: split-half reliability of the per-route mean
   residual activity (route = directed start->end cell pair; traversals of each
   route alternate between halves in time order; Spearman across routes).
   Tested against permutation of route labels among bouts within duration
   quartiles. Near zero for a generic run-state signal; positive for
   place x direction or route coding.
4. ``light_dark``: light-bout minus dark-bout residual activity.
5. ``arm_first_entry``: bouts ending in an arm first entered in the current
   light epoch minus the rest (descriptive of re-exploration after a light
   switch; confounded with time since the switch).
6. Timing: Spearman cross-correlation (rank-transformed signal vs the
   standardised run indicator) at lags of +-``max_lag_s``. Positive lag means
   activity at t correlates with running at t + lag (activity leads).
   ``xc_peak_r`` is the correlation at the lag of largest magnitude and
   ``xc_peak_lag_s`` that lag. ``xc_lead_lag_asym`` is the mean correlation over
   positive lags minus that over negative lags, signed by the peak so that
   positive means activity modulation leads running for both increases and
   decreases. Lead/lag is tested through the asymmetry, because the location
   of the argmax under the shift null is uninformative.

Significance (items 1, 2, 4, 5, 6) uses a circular shift of the signal
relative to behaviour by at least ``min_shift_s``; the bout annotations are
fixed, so the null differs from the observed value only in signal-behaviour
alignment. p-values are two-sided ``(k + 1) / (n + 1)`` on absolute deviation
from the null mean, except ``xc_peak_r`` (one-sided on the peak magnitude)
and ``route_reliability`` (one-sided, label-permutation null).

Calcium indicator kinetics delay dF/F relative to spiking; with dF/F the
cross-correlation is biased toward activity lagging running. Inferred spikes
(CASCADE) reduce this bias.

References
----------
Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth show
rapid learning, sudden insight, and efficient exploration." eLife 10:e66175.
doi:10.7554/eLife.66175 (maze graph, dead ends, exploration of novel cells)

Saleem AB, Ayaz A, Jeffery KJ, Harris KD, Carandini M. 2013. "Integration of
visual motion and locomotion in mouse visual cortex." Nature Neuroscience
16:1864-1869. doi:10.1038/nn.3567 (light vs dark running: visual flow vs
self-motion)

Alexander AS, Nitz DA. 2015. "Retrosplenial cortex maps the conjunction of
internal and external spaces." Nature Neuroscience 18:1143-1151.
doi:10.1038/nn.4058 (route-position conjunctive coding in RSC)

Rupprecht P, Carta S, Hoffmann A, et al. 2021. "A database and deep learning
toolbox for noise-optimized, generalized spike inference from calcium
imaging." Nature Neuroscience 24:1324-1337. doi:10.1038/s41593-021-00895-5.
https://github.com/HelmchenLabSoftware/Cascade (inferred spikes)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from hm2p.analysis.event_triggered_behaviour import NODE_GROUPS
from hm2p.maze.discretize import cell_sequence, discretize_position_fast
from hm2p.maze.topology import RoseMaze, build_rose_maze

RUN_TYPES = (
    "into_dead_end",
    "out_of_dead_end",
    "junction_to_junction",
    "dead_end_to_dead_end",
    "no_transition",
    "unknown",
)
CONTRASTS = ("approach_return", "novel_repeat", "light_dark", "arm_first_entry")
SHIFT_TESTED = (*CONTRASTS, "edge_decay_rho", "xc_lead_lag_asym", "xc_peak_r")
TESTED = (*SHIFT_TESTED, "route_reliability")
LIGHT_FRAC_HI = 0.9
LIGHT_FRAC_LO = 0.1


# ---------------------------------------------------------------------------
# Maze helpers
# ---------------------------------------------------------------------------


def dwell_filtered_cells(cell_idx: npt.ArrayLike, min_dwell_frames: int = 2) -> np.ndarray:
    """Set cell occupancies shorter than *min_dwell_frames* to -1 (unassigned).

    Parameters
    ----------
    cell_idx : (n_frames,) int
        Cell index per frame, -1 where invalid.
    min_dwell_frames : int
        Minimum number of consecutive frames in a cell to count as a visit.

    Returns
    -------
    (n_frames,) int64
    """
    c = np.asarray(cell_idx, dtype=np.int64).copy()
    if c.size == 0 or min_dwell_frames <= 1:
        return c
    change = np.flatnonzero(np.diff(c) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [c.size]])
    for a, b in zip(starts, ends, strict=True):
        if b - a < min_dwell_frames:
            c[a:b] = -1
    return c


def arm_ids(maze: RoseMaze) -> np.ndarray:
    """Arm label per cell index: components of non-junction cells; junctions alone."""
    n = len(maze.cell_list)
    out = np.full(n, -1, dtype=np.int64)
    junction = {maze.cell_to_idx[c] for c in maze.junctions}
    label = 0
    for i in range(n):
        if out[i] >= 0:
            continue
        out[i] = label
        if i not in junction:
            stack = [i]
            while stack:
                cur = stack.pop()
                for nb in maze.adj[maze.cell_list[cur]]:
                    j = maze.cell_to_idx[nb]
                    if j not in junction and out[j] < 0:
                        out[j] = label
                        stack.append(j)
        label += 1
    return out


def node_groups(maze: RoseMaze) -> np.ndarray:
    """Node group per cell index: ``dead_end`` / ``corridor`` / ``junction``."""
    return np.array([NODE_GROUPS[maze.node_types[c]] for c in maze.cell_list], dtype=object)


def dead_end_distance(maze: RoseMaze) -> np.ndarray:
    """Graph distance from each cell to its nearest dead end."""
    de = [maze.cell_to_idx[c] for c in maze.dead_ends]
    return maze.dist[:, de].min(axis=1).astype(np.int64)


def classify_run(start_group: str | None, end_group: str | None, moved: bool) -> str:
    """Run type from start/end node groups (see module docstring)."""
    if start_group is None or end_group is None:
        return "unknown"
    if not moved:
        return "no_transition"
    s_de, e_de = start_group == "dead_end", end_group == "dead_end"
    if s_de and e_de:
        return "dead_end_to_dead_end"
    if e_de:
        return "into_dead_end"
    if s_de:
        return "out_of_dead_end"
    return "junction_to_junction"


def prior_edge_counts(seq: npt.ArrayLike, maze: RoseMaze) -> np.ndarray:
    """For each transition seq[k] -> seq[k+1]: earlier traversals of that directed edge.

    Non-adjacent transitions (tracking jumps, dropped short visits) are NaN.

    Returns
    -------
    (len(seq) - 1,) float
    """
    s = np.asarray(seq, dtype=np.int64)
    out = np.full(max(s.size - 1, 0), np.nan)
    seen: dict[tuple[int, int], int] = {}
    for k in range(s.size - 1):
        a, b = int(s[k]), int(s[k + 1])
        if maze.dist[a, b] != 1:
            continue
        out[k] = seen.get((a, b), 0)
        seen[(a, b)] = seen.get((a, b), 0) + 1
    return out


def light_epoch_starts(light_on: npt.ArrayLike) -> np.ndarray:
    """First frame of the light epoch containing each frame."""
    lo = np.asarray(light_on, dtype=bool).ravel()
    if lo.size == 0:
        return np.zeros(0, dtype=np.int64)
    trans = np.concatenate([[0], np.flatnonzero(np.diff(lo.astype(np.int8)) != 0) + 1])
    return trans[np.searchsorted(trans, np.arange(lo.size), side="right") - 1]


# ---------------------------------------------------------------------------
# Bout annotation
# ---------------------------------------------------------------------------


def light_condition(light_frac: float) -> str:
    """``light`` (>= 0.9 lit), ``dark`` (<= 0.1 lit) or ``mixed``."""
    if light_frac >= LIGHT_FRAC_HI:
        return "light"
    if light_frac <= LIGHT_FRAC_LO:
        return "dark"
    return "mixed"


def annotate_bouts(
    onsets: npt.ArrayLike,
    offsets: npt.ArrayLike,
    x_maze: npt.ArrayLike,
    y_maze: npt.ArrayLike,
    speed: npt.ArrayLike,
    light_on: npt.ArrayLike,
    fps: float,
    maze: RoseMaze | None = None,
    min_dwell_frames: int = 2,
) -> pd.DataFrame:
    """Maze-context annotation of running bouts (one row per bout).

    Parameters
    ----------
    onsets, offsets : (n_bouts,) int
        Bout bounds (offsets exclusive), e.g. from
        :func:`hm2p.analysis.running_coding.run_bouts`.
    x_maze, y_maze : (n_frames,) float
        Position in maze units.
    speed : (n_frames,) float
        Speed in cm/s.
    light_on : (n_frames,) bool
    fps : float
    maze : RoseMaze or None
    min_dwell_frames : int
        See :func:`dwell_filtered_cells`.

    Returns
    -------
    pandas.DataFrame
        Columns: ``onset, offset, onset_s, duration_s, mean_speed, light_frac,
        light_cond, start_cell, end_cell, start_node, end_node, path_cells,
        n_entered, n_novel, novel_frac, novel, repeat, first_arm_entry,
        d_dead_end_start, d_dead_end_end, delta_dead_end, run_type, route,
        edge_prior_traversals``. Cells are indices into ``maze.cell_list``
        (-1 when unknown).

    References
    ----------
    Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth show
    rapid learning, sudden insight, and efficient exploration." eLife
    10:e66175. doi:10.7554/eLife.66175
    """
    maze = maze or build_rose_maze()
    on = np.asarray(onsets, dtype=np.int64)
    off = np.asarray(offsets, dtype=np.int64)
    x = np.asarray(x_maze, dtype=np.float64).ravel()
    y = np.asarray(y_maze, dtype=np.float64).ravel()
    spd = np.asarray(speed, dtype=np.float64).ravel()
    lo = np.asarray(light_on, dtype=bool).ravel()
    cells = dwell_filtered_cells(discretize_position_fast(x, y, maze), min_dwell_frames)
    seq, times = cell_sequence(cells)
    seq = seq.astype(np.int64)
    times = times.astype(np.int64)
    first_time = np.full(len(maze.cell_list), np.iinfo(np.int64).max)
    for c, t in zip(seq[::-1], times[::-1], strict=True):
        first_time[c] = t
    edge_prior = prior_edge_counts(seq, maze)
    groups = node_groups(maze)
    arms = arm_ids(maze)
    d_de = dead_end_distance(maze)
    arm_frame = np.where(cells >= 0, arms[np.clip(cells, 0, None)], -1)
    epoch_start = light_epoch_starts(lo)

    rows: list[dict[str, Any]] = []
    for a, b in zip(on, off, strict=True):
        lf = float(np.mean(lo[a:b])) if b > a else float("nan")
        rec: dict[str, Any] = {
            "onset": int(a),
            "offset": int(b),
            "onset_s": a / float(fps),
            "duration_s": (b - a) / float(fps),
            "mean_speed": float(np.nanmean(spd[a:b])) if b > a else float("nan"),
            "light_frac": lf,
            "light_cond": light_condition(lf),
        }
        i0 = int(np.searchsorted(times, a, side="right")) - 1
        i1 = int(np.searchsorted(times, b, side="left")) - 1
        i0 = max(i0, 0)
        if seq.size == 0 or i1 < i0:
            rec.update(_unknown_bout())
            rows.append(rec)
            continue
        bseq = seq[i0 : i1 + 1]
        start, end = int(bseq[0]), int(bseq[-1])
        entered = np.unique(bseq[1:])
        novel_cells = entered[first_time[entered] >= a] if entered.size else entered
        path = int(sum(maze.dist[bseq[k], bseq[k + 1]] for k in range(bseq.size - 1)))
        priors = edge_prior[i0:i1] if i1 > i0 else np.zeros(0)
        priors = priors[np.isfinite(priors)]
        end_arm = arms[end]
        start_arm = arms[start]
        prev = arm_frame[epoch_start[a] : a] if a < lo.size else np.zeros(0, dtype=np.int64)
        rec.update(
            {
                "start_cell": start,
                "end_cell": end,
                "start_node": str(groups[start]),
                "end_node": str(groups[end]),
                "path_cells": path,
                "n_entered": int(entered.size),
                "n_novel": int(novel_cells.size),
                "novel_frac": (novel_cells.size / entered.size if entered.size else float("nan")),
                "novel": bool(novel_cells.size > 0),
                "repeat": bool(entered.size > 0 and novel_cells.size == 0),
                "first_arm_entry": bool(end_arm != start_arm and not np.any(prev == end_arm)),
                "d_dead_end_start": int(d_de[start]),
                "d_dead_end_end": int(d_de[end]),
                "delta_dead_end": int(d_de[end] - d_de[start]),
                "run_type": classify_run(str(groups[start]), str(groups[end]), path > 0),
                "route": f"{start}->{end}" if start != end else "",
                "edge_prior_traversals": (
                    float(np.median(priors)) if priors.size else float("nan")
                ),
            }
        )
        rows.append(rec)
    cols = [
        "onset",
        "offset",
        "onset_s",
        "duration_s",
        "mean_speed",
        "light_frac",
        "light_cond",
        *_unknown_bout().keys(),
    ]
    return pd.DataFrame(rows, columns=cols)


def _unknown_bout() -> dict[str, Any]:
    nan = float("nan")
    return {
        "start_cell": -1,
        "end_cell": -1,
        "start_node": "",
        "end_node": "",
        "path_cells": 0,
        "n_entered": 0,
        "n_novel": 0,
        "novel_frac": nan,
        "novel": False,
        "repeat": False,
        "first_arm_entry": False,
        "d_dead_end_start": -1,
        "d_dead_end_end": -1,
        "delta_dead_end": 0,
        "run_type": "unknown",
        "route": "",
        "edge_prior_traversals": nan,
    }


def summarise_runs(bouts: pd.DataFrame) -> pd.DataFrame:
    """Session-level run counts per light condition x run type.

    Returns one row per (``light_cond``, ``run_type``) present, with
    ``n_bouts, n_novel, n_repeat, n_first_arm_entry, median_duration_s,
    median_mean_speed, median_path_cells``.
    """
    cols = [
        "light_cond",
        "run_type",
        "n_bouts",
        "n_novel",
        "n_repeat",
        "n_first_arm_entry",
        "median_duration_s",
        "median_mean_speed",
        "median_path_cells",
    ]
    if bouts.empty:
        return pd.DataFrame(columns=cols)
    g = bouts.groupby(["light_cond", "run_type"])
    out = g.agg(
        n_bouts=("onset", "size"),
        n_novel=("novel", "sum"),
        n_repeat=("repeat", "sum"),
        n_first_arm_entry=("first_arm_entry", "sum"),
        median_duration_s=("duration_s", "median"),
        median_mean_speed=("mean_speed", "median"),
        median_path_cells=("path_cells", "median"),
    ).reset_index()
    return out[cols]


# ---------------------------------------------------------------------------
# Rank residualisation and per-bout activity
# ---------------------------------------------------------------------------


def _norm_ranks(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Average ranks scaled to (0, 1]."""
    return stats.rankdata(a, axis=axis) / a.shape[axis]


def rank_residualiser(covariates: npt.ArrayLike) -> np.ndarray:
    """Matrix ``I - H`` that residualises a vector on the ranks of *covariates*.

    Parameters
    ----------
    covariates : (n, k) float
        Each column is rank-transformed; an intercept is added.

    Returns
    -------
    (n, n) float
    """
    c = np.asarray(covariates, dtype=np.float64)
    if c.ndim == 1:
        c = c[:, None]
    n = c.shape[0]
    design = np.column_stack([np.ones(n), _norm_ranks(c, axis=0)])
    hat = design @ np.linalg.pinv(design)
    return np.eye(n) - hat


def shifted_bout_means(
    signal: npt.ArrayLike,
    onsets: npt.ArrayLike,
    offsets: npt.ArrayLike,
    shifts: npt.ArrayLike,
) -> np.ndarray:
    """Mean signal per bout after circularly shifting the signal by each shift.

    Shift ``s`` corresponds to ``np.roll(signal, s)``. Non-finite frames are
    excluded from each mean; a bout with no finite frames gets the mean of
    the other bouts in that row.

    Returns
    -------
    (n_shifts, n_bouts) float
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    n = x.size
    fin = np.isfinite(x)
    xs = np.where(fin, x, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(np.concatenate([xs, xs]))])
    cf = np.concatenate([[0.0], np.cumsum(np.concatenate([fin, fin]).astype(np.float64))])
    on = np.asarray(onsets, dtype=np.int64)
    length = np.asarray(offsets, dtype=np.int64) - on
    sh = np.asarray(shifts, dtype=np.int64)
    start = (on[None, :] - sh[:, None]) % n
    tot = cs[start + length[None, :]] - cs[start]
    cnt = cf[start + length[None, :]] - cf[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        means = tot / cnt
    bad = ~np.isfinite(means)
    if bad.any():
        with np.errstate(invalid="ignore"):
            fill = np.nanmean(np.where(bad, np.nan, means), axis=1)
        fill = np.where(np.isfinite(fill), fill, 0.0)
        means = np.where(bad, fill[:, None], means)
    return means


def _split_halves(labels: np.ndarray) -> np.ndarray:
    """Alternate traversals of each label (in array order) between halves 0/1; -1 if no label."""
    half = np.full(labels.size, -1, dtype=np.int64)
    counter: dict[int, int] = {}
    for i, lab in enumerate(labels):
        if lab < 0:
            continue
        k = counter.get(int(lab), 0)
        half[i] = k % 2
        counter[int(lab)] = k + 1
    return half


def split_half_reliability(
    values: npt.ArrayLike, labels: npt.ArrayLike, halves: npt.ArrayLike, min_labels: int = 4
) -> float:
    """Spearman correlation across labels of half-0 vs half-1 mean values.

    Labels lacking either half are ignored; NaN when fewer than *min_labels*
    remain or either half has no variance.
    """
    v = np.asarray(values, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64)
    h = np.asarray(halves, dtype=np.int64)
    ok = (lab >= 0) & (h >= 0) & np.isfinite(v)
    if not ok.any():
        return float("nan")
    nl = int(lab[ok].max()) + 1
    s = np.zeros((2, nl))
    c = np.zeros((2, nl))
    np.add.at(s, (h[ok], lab[ok]), v[ok])
    np.add.at(c, (h[ok], lab[ok]), 1.0)
    keep = (c[0] > 0) & (c[1] > 0)
    if keep.sum() < min_labels:
        return float("nan")
    m0 = s[0, keep] / c[0, keep]
    m1 = s[1, keep] / c[1, keep]
    if np.ptp(m0) == 0 or np.ptp(m1) == 0:
        return float("nan")
    return float(stats.spearmanr(m0, m1)[0])


def permute_within_strata(
    labels: npt.ArrayLike, strata: npt.ArrayLike, rng: np.random.Generator
) -> np.ndarray:
    """Permute labels among elements sharing a stratum (labels < 0 stay in place)."""
    lab = np.asarray(labels, dtype=np.int64).copy()
    st = np.asarray(strata, dtype=np.int64)
    for s in np.unique(st[lab >= 0]):
        idx = np.flatnonzero((st == s) & (lab >= 0))
        lab[idx] = lab[rng.permutation(idx)]
    return lab


def run_indicator(onsets: npt.ArrayLike, offsets: npt.ArrayLike, n_frames: int) -> np.ndarray:
    """Boolean per-frame indicator of running bouts."""
    out = np.zeros(int(n_frames), dtype=bool)
    for a, b in zip(np.asarray(onsets), np.asarray(offsets), strict=True):
        out[int(a) : int(b)] = True
    return out


def circular_xcorr(xz: npt.ArrayLike, rz: npt.ArrayLike, n_valid: int) -> np.ndarray:
    """``c[k] = sum_t xz[t] * rz[(t + k) mod n] / n_valid`` for every lag k (FFT).

    Shifting *xz* by ``s`` (``np.roll``) maps ``c[k]`` to ``c[k + s]``, so one
    transform gives the observed and every circular-shift null curve.
    """
    x = np.asarray(xz, dtype=np.float64)
    r = np.asarray(rz, dtype=np.float64)
    n = x.size
    c = np.fft.irfft(np.conj(np.fft.rfft(x)) * np.fft.rfft(r), n)
    return c / max(int(n_valid), 1)


def xcorr_summary(curves: np.ndarray, lags: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    """Peak r, peak lag (s) and signed lead-lag asymmetry for rows of *curves*."""
    c = np.atleast_2d(curves)
    pk = np.argmax(np.abs(c), axis=1)
    peak_r = c[np.arange(c.shape[0]), pk]
    sign = np.sign(peak_r)
    asym = sign * (c[:, lags > 0].mean(axis=1) - c[:, lags < 0].mean(axis=1))
    return {"xc_peak_r": peak_r, "xc_peak_lag_s": lags[pk] / float(fps), "xc_lead_lag_asym": asym}


# ---------------------------------------------------------------------------
# Session design and per-cell statistics
# ---------------------------------------------------------------------------


@dataclass
class MazeRunDesign:
    """Everything about a session's bouts that does not depend on the cell."""

    bouts: pd.DataFrame
    onsets: np.ndarray
    offsets: np.ndarray
    resid_base: np.ndarray
    resid_time: np.ndarray
    contrasts: dict[str, tuple[np.ndarray, np.ndarray, str]]
    prior_idx: np.ndarray
    prior_resid: np.ndarray | None
    prior_resid_mat: np.ndarray | None
    route_labels: np.ndarray
    route_halves: np.ndarray
    route_null_labels: np.ndarray
    route_null_halves: np.ndarray
    run_z: np.ndarray
    n_valid: int
    lags: np.ndarray
    fps: float


def build_design(
    bouts: pd.DataFrame,
    valid: npt.ArrayLike,
    fps: float,
    n_frames: int,
    n_route_perms: int = 200,
    min_group: int = 5,
    min_traversals: int = 4,
    n_duration_strata: int = 4,
    min_prior_bouts: int = 10,
    max_lag_s: float = 5.0,
    rng: np.random.Generator | None = None,
) -> MazeRunDesign:
    """Precompute residualisers, contrast groups, route permutations and run indicator.

    Contrast groups with fewer than *min_group* bouts are left empty (their
    statistics are NaN). Routes with fewer than *min_traversals* traversals
    are excluded from the reliability analysis.
    """
    rng = rng or np.random.default_rng(0)
    b = bouts.reset_index(drop=True)
    on = b["onset"].to_numpy(np.int64)
    off = b["offset"].to_numpy(np.int64)
    nb = len(b)
    empty = np.zeros(0, dtype=np.int64)
    if nb >= 3:
        cov = b[["duration_s", "mean_speed"]].to_numpy(np.float64)
        cov = np.where(np.isfinite(cov), cov, np.nanmedian(cov, axis=0))
        resid_base = rank_residualiser(cov)
        resid_time = rank_residualiser(np.column_stack([cov, b["onset_s"].to_numpy()]))
    else:
        resid_base = resid_time = np.eye(nb)

    def _grp(sel_a: np.ndarray, sel_b: np.ndarray, which: str) -> tuple[np.ndarray, ...]:
        ia, ib = np.flatnonzero(sel_a), np.flatnonzero(sel_b)
        if ia.size < min_group or ib.size < min_group:
            return empty, empty, which
        return ia, ib, which

    rt = b["run_type"].to_numpy() if nb else np.zeros(0, dtype=object)
    lc = b["light_cond"].to_numpy() if nb else np.zeros(0, dtype=object)
    known = rt != "unknown"
    moved = known & (rt != "no_transition")
    contrasts = {
        "approach_return": _grp(rt == "into_dead_end", rt == "out_of_dead_end", "base"),
        "novel_repeat": _grp(
            b["novel"].to_numpy(bool) if nb else empty.astype(bool),
            b["repeat"].to_numpy(bool) if nb else empty.astype(bool),
            "time",
        ),
        "light_dark": _grp(lc == "light", lc == "dark", "base"),
        "arm_first_entry": _grp(
            moved & (b["first_arm_entry"].to_numpy(bool) if nb else empty.astype(bool)),
            moved & ~(b["first_arm_entry"].to_numpy(bool) if nb else empty.astype(bool)),
            "base",
        ),
    }

    # edge-familiarity partial Spearman: subset with a finite prior count
    prior = b["edge_prior_traversals"].to_numpy(np.float64) if nb else np.zeros(0)
    prior_idx = np.flatnonzero(np.isfinite(prior))
    prior_resid = prior_mat = None
    if prior_idx.size >= min_prior_bouts and np.ptp(prior[prior_idx]) > 0:
        sub = b.iloc[prior_idx]
        cov = sub[["duration_s", "mean_speed", "onset_s"]].to_numpy(np.float64)
        cov = np.where(np.isfinite(cov), cov, np.nanmedian(cov, axis=0))
        prior_mat = rank_residualiser(cov)
        e = prior_mat @ _norm_ranks(prior[prior_idx])
        sd = float(np.std(e))
        prior_resid = (e - e.mean()) / sd if sd > 0 else None
        if prior_resid is None:
            prior_mat = None

    # routes
    route = b["route"].to_numpy(object) if nb else np.zeros(0, dtype=object)
    counts = pd.Series(route[route != ""]).value_counts() if nb else pd.Series(dtype=int)
    keep_routes = sorted(counts[counts >= min_traversals].index)
    code = {r: i for i, r in enumerate(keep_routes)}
    labels = np.array([code.get(r, -1) for r in route], dtype=np.int64)
    halves = _split_halves(labels)
    if nb and (labels >= 0).any() and n_route_perms > 0:
        dur = b["duration_s"].to_numpy(np.float64)
        q = np.quantile(dur[labels >= 0], np.linspace(0, 1, n_duration_strata + 1)[1:-1])
        strata = np.searchsorted(q, dur, side="right")
        null_labels = np.stack(
            [permute_within_strata(labels, strata, rng) for _ in range(n_route_perms)]
        )
        null_halves = np.stack([_split_halves(row) for row in null_labels])
    else:
        null_labels = null_halves = np.zeros((0, nb), dtype=np.int64)

    # run indicator for the cross-correlation
    v = np.asarray(valid, dtype=bool).ravel()[:n_frames]
    run = run_indicator(on, off, n_frames).astype(np.float64)
    n_valid = int(v.sum())
    run_z = np.zeros(n_frames)
    if n_valid > 1:
        mu, sd = run[v].mean(), run[v].std()
        if sd > 0:
            run_z[v] = (run[v] - mu) / sd
    max_lag = int(round(max_lag_s * fps))
    return MazeRunDesign(
        bouts=b,
        onsets=on,
        offsets=off,
        resid_base=resid_base,
        resid_time=resid_time,
        contrasts=contrasts,
        prior_idx=prior_idx,
        prior_resid=prior_resid,
        prior_resid_mat=prior_mat,
        route_labels=labels,
        route_halves=halves,
        route_null_labels=null_labels,
        route_null_halves=null_halves,
        run_z=run_z,
        n_valid=n_valid,
        lags=np.arange(-max_lag, max_lag + 1),
        fps=float(fps),
    )


def _bout_statistics(means: np.ndarray, d: MazeRunDesign) -> dict[str, np.ndarray]:
    """Contrast and edge-decay statistics for each row of bout means."""
    nrow, nb = means.shape
    out: dict[str, np.ndarray] = {}
    if nb == 0:
        return {k: np.full(nrow, np.nan) for k in (*CONTRASTS, "edge_decay_rho")}
    r = _norm_ranks(means, axis=1)
    e_base = r @ d.resid_base.T
    e_time = r @ d.resid_time.T
    for name, (ia, ib, which) in d.contrasts.items():
        if ia.size == 0:
            out[name] = np.full(nrow, np.nan)
            continue
        e = e_time if which == "time" else e_base
        out[name] = e[:, ia].mean(axis=1) - e[:, ib].mean(axis=1)
    if d.prior_resid is None or d.prior_resid_mat is None:
        out["edge_decay_rho"] = np.full(nrow, np.nan)
    else:
        sub = _norm_ranks(means[:, d.prior_idx], axis=1) @ d.prior_resid_mat.T
        sub = sub - sub.mean(axis=1, keepdims=True)
        sd = sub.std(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["edge_decay_rho"] = (sub @ d.prior_resid) / (sd * sub.shape[1])
    return out


def _shift_p(obs: float, null: np.ndarray) -> tuple[float, float]:
    """z and two-sided p of *obs* against a null sample."""
    nk = null[np.isfinite(null)]
    if not np.isfinite(obs) or nk.size < 10:
        return float("nan"), float("nan")
    mu, sd = float(np.mean(nk)), float(np.std(nk))
    z = float((obs - mu) / sd) if sd > 0 else float("nan")
    p = float((np.sum(np.abs(nk - mu) >= abs(obs - mu)) + 1) / (nk.size + 1))
    return z, p


def cell_statistics(
    signal: npt.ArrayLike,
    design: MazeRunDesign,
    n_shuffles: int = 200,
    min_shift_s: float = 10.0,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """All maze-running statistics for one cell, with nulls (see module docstring).

    Parameters
    ----------
    signal : (n_frames,) float
        dF/F, event mask or inferred spike rate.
    design : MazeRunDesign
        From :func:`build_design` for the same session.
    n_shuffles : int
        Circular shifts for the shift-null statistics.
    min_shift_s : float
        Minimum shift magnitude in seconds.
    rng : Generator or None

    Returns
    -------
    dict
        Observed statistics with ``_z`` and ``_p`` for each tested measure,
        ``xc_peak_lag_s``, raw group means (``rate_<group>``) and ``n_runs``.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(signal, dtype=np.float64).ravel()
    n = x.size
    d = design
    min_shift = int(round(min_shift_s * d.fps))
    can_shift = n > 2 * min_shift and n_shuffles >= 1
    shifts = np.concatenate(
        [[0], rng.integers(min_shift, n - min_shift, size=n_shuffles) if can_shift else []]
    ).astype(np.int64)

    means = shifted_bout_means(x, d.onsets, d.offsets, shifts)
    stat = _bout_statistics(means, d)

    # cross-correlation (Spearman: rank-transformed signal)
    fin = np.isfinite(x)
    xr = np.zeros(n)
    if fin.sum() > 1:
        rk = stats.rankdata(x[fin])
        sd = rk.std()
        if sd > 0:
            xr[fin] = (rk - rk.mean()) / sd
    c = circular_xcorr(xr, d.run_z, d.n_valid)
    curves = c[(d.lags[None, :] + shifts[:, None]) % n]
    stat.update(xcorr_summary(curves, d.lags, d.fps))

    out: dict[str, Any] = {"n_runs": int(d.onsets.size)}
    for k in (*CONTRASTS, "edge_decay_rho", "xc_lead_lag_asym"):
        out[k] = float(stat[k][0])
        out[f"{k}_z"], out[f"{k}_p"] = (
            _shift_p(out[k], stat[k][1:]) if can_shift else (float("nan"), float("nan"))
        )
    peak = float(stat["xc_peak_r"][0])
    out["xc_peak_r"] = peak
    out["xc_peak_lag_s"] = float(stat["xc_peak_lag_s"][0])
    null_peak = np.abs(stat["xc_peak_r"][1:])
    if can_shift and null_peak.size >= 10:
        mu, sdn = float(null_peak.mean()), float(null_peak.std())
        out["xc_peak_r_z"] = float((abs(peak) - mu) / sdn) if sdn > 0 else float("nan")
        out["xc_peak_r_p"] = float((np.sum(null_peak >= abs(peak)) + 1) / (null_peak.size + 1))
    else:
        out["xc_peak_r_z"] = out["xc_peak_r_p"] = float("nan")

    # route specificity: label-permutation null on residual activity
    obs_means = means[0]
    e = obs_means
    if obs_means.size:
        e = _norm_ranks(obs_means[None, :], axis=1)[0] @ d.resid_base.T
    rel = split_half_reliability(e, d.route_labels, d.route_halves)
    out["route_reliability"] = rel
    out["n_routes"] = int(np.unique(d.route_labels[d.route_labels >= 0]).size)
    null_rel = np.array(
        [
            split_half_reliability(e, lab, h)
            for lab, h in zip(d.route_null_labels, d.route_null_halves, strict=True)
        ]
    )
    nk = null_rel[np.isfinite(null_rel)]
    if np.isfinite(rel) and nk.size >= 10:
        sdn = float(nk.std())
        out["route_reliability_z"] = float((rel - nk.mean()) / sdn) if sdn > 0 else float("nan")
        out["route_reliability_p"] = float((np.sum(nk >= rel) + 1) / (nk.size + 1))
    else:
        out["route_reliability_z"] = out["route_reliability_p"] = float("nan")

    # raw (unresidualised) group means, descriptive
    group_names = {
        "approach_return": ("into_dead_end", "out_of_dead_end"),
        "novel_repeat": ("novel", "repeat"),
        "light_dark": ("light", "dark"),
        "arm_first_entry": ("first_arm_entry", "arm_reentry"),
    }
    for name, (ia, ib, _) in d.contrasts.items():
        ga, gb = group_names[name]
        out[f"rate_{ga}"] = float(np.mean(obs_means[ia])) if ia.size else float("nan")
        out[f"rate_{gb}"] = float(np.mean(obs_means[ib])) if ib.size else float("nan")
    return out
