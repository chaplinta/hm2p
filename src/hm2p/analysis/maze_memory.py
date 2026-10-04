"""Population coding of memory and planning during maze exploration.

All soma ROIs of a session are pooled (cell type is ignored for the main
result). Three analyses are run per session, each separately in light and in
dark epochs, with a circular-shift null and behaviour-only and head-direction
(HD)-removed controls. Decoding machinery (time-block cross-validation,
balanced logistic regression, window means under circular shift, null
summaries) is shared with :mod:`hm2p.analysis.population_maze`.

Shared preprocessing
--------------------
Positions are discretised onto the 23-cell maze graph; frames flagged as
behavioural artefacts are unassigned and cell occupancies shorter than
``min_dwell_frames`` are dropped (:func:`population_maze.session_cells`).
Neural activity and behaviour channels are set to NaN on invalid frames, so
window means use valid frames only. The cell sequence is converted into
*visits* (:func:`visit_table`): consecutive valid frames in one cell, merged
across gaps of at most ``max_gap_s``. Two consecutive visits more than
``max_gap_s`` apart, or more than ``max_jump_steps`` graph steps apart, start
a new *segment*; jumps of 2..``max_jump_steps`` steps are filled with the
unique intermediate cells of the tree as zero-length visits.

A *trip* (:func:`trip_table`) is the visit sequence between two consecutive
dead-end visits of one segment (leave dead end A ... arrive at dead end B; A
may equal B). The light condition of a trip is ``light`` when at least
``light_hi`` of its frames have the room lights on, ``dark`` when at most
``light_lo`` do, otherwise ``mixed`` (excluded).

Analysis 1 -- retrospective vs prospective ("splitter") coding
---------------------------------------------------------------
Samples: visits to non-dead-end cells inside a trip that pass through the
cell (arrival neighbour != departure neighbour); features = per-cell mean
activity over the visit. Labels: *origin* (dead end the trip started from;
retrospective) and *destination* (dead end the trip ends at; prospective).
Strata = (cell, arrival neighbour, departure neighbour), so place and travel
direction are identical within a stratum. Because the maze is a tree, the
arrival neighbour fixes the subtree the animal came from, so the origin is
decoded among the dead ends of that subtree only (and the destination among
those of the departure subtree). Within a stratum, labels with fewer than
``min_n`` samples are dropped and strata with fewer than two remaining labels
are dropped; balanced accuracy is computed per stratum and averaged weighted
by samples. ``prospective_minus_retrospective`` is the destination minus the
origin score at every shift (two-sided null p).

Analysis 2 -- distance to go
----------------------------
Same pass-through visits. Targets: *remaining* = graph distance from the
current cell to the destination dead end; *elapsed* = number of cell
transitions since leaving the origin dead end (path actually taken,
backtracks included). A ridge regression (Hoerl & Kennard 1970) from the
population is cross-validated over time blocks; the out-of-fold prediction is
scored by Spearman correlation with the target and by partial Spearman
correlation (Kim 2015) with the target controlling for the other step count
and visit speed. Variant ``place_controlled`` removes the per-cell mean of
the target and of every feature (training-fold means) before fitting, and
evaluates against the cell-demeaned target and covariates, so place coding
cannot contribute. Variant ``place_direction_controlled`` does the same per
(cell, arrival neighbour, departure neighbour) stratum: at a given cell the
travel direction also constrains the remaining distance, so a population that
encodes inbound vs outbound direction would otherwise score in the
place-controlled variant. It has its own sample set (light and dark equalised
per stratum x remaining steps; strata with fewer than two distinct remaining
values dropped).

Analysis 3 -- planning the next trip
------------------------------------
Samples: dead-end visits followed by a trip, dwell at least
``min_dead_end_dwell_s``. Windows: ``pre`` = the ``plan_window_s`` before
leaving the dead end; ``post`` = the ``plan_window_s`` after leaving. Labels:

- ``novel``: the destination is not among the ``recent_k`` most recently
  visited distinct dead ends (the current one included);
- ``lru``: the destination is a least recently visited dead end among all
  dead ends other than the origin (never-visited dead ends tie as least
  recent).

Condition = majority rule over the window and the following trip.

Variants are ``{window}{filter}``: filter ``""`` keeps all departures,
``_noreturn`` drops trips that return to the dead end they left (short
bounce-back trips, which always count as "recent"), and ``_noreturn_long``
additionally keeps only trips whose origin-to-destination graph distance is
at least ``long_trip_steps``. For planning only, the feature set
``behaviour_plus_syllable`` adds to the behaviour channels the fraction of
the window spent in each of the ``n_top_syllables`` most frequent
keypoint-MoSeq syllables of the session (Weinreb et al. 2024), as a control
for pre-departure posture and behavioural state.

Behaviour: for each trip, the probability that a random walk leaving the
same dead end first reaches a ``novel`` (or ``lru``) dead end is computed
exactly from an absorbing Markov chain (Kemeny & Snell 1960;
:func:`dead_end_absorption`) for two walkers: ``uniform`` (memoryless,
backtracking allowed, as in
:func:`hm2p.maze.exploration_complexity.random_walk_coverage_null`) and
``nonbacktracking`` (never reverses along the edge it arrived by). The second
model absorbs the forward bias of mice (Rosenberg et al. 2021), which on a
tree already raises the novel fraction well above the uniform walk without
any memory. Observed fractions are compared to the mean of these
probabilities; the per-session p value (two-sided) comes from Monte Carlo
draws of independent Bernoulli trials with those probabilities. This is the
trip-level counterpart of the coverage-vs-random-walk comparison (directed
exploration in light, near random in dark).

Feature sets and null
---------------------
- ``neural``: per-cell window means.
- ``behaviour``: sine and cosine of the circular mean HD, mean speed, mean
  |AHV|, time in session, time since leaving the last dead end (time since
  trip start), time since entering the last dead end (dwell) and number of
  distinct dead ends visited so far (:func:`behaviour_channels`); the same
  eight channels for all analyses.
- ``neural_hd_removed``: neural features residualised on sine/cosine of
  window HD (fitted on the training fold only).

Decoders: L2 logistic regression with balanced class weights (one-vs-rest
for more than two classes; same objective as scikit-learn's
``LogisticRegression``, Pedregosa et al. 2011) and ridge regression; features
z-scored and the HD residualisation fitted on the training fold only.
Cross-validation over ``n_blocks`` contiguous time blocks with a ``gap_s``
exclusion (Roberts et al. 2017; :func:`population_maze.time_block_folds`),
run within each stratum. Because samples, labels and folds are identical
at every null shift, the fits are solved for all shifts at once (Newton's
method / closed form, batched over shifts).

Null: circular shift of the predictors relative to the labels by at least
``min_shift_s`` (neural matrix for neural feature sets, with the HD
regressors and the regression covariates kept aligned; behaviour channels for
``behaviour``); features and the full CV are recomputed and the same shifts
are used for all analyses, conditions and feature sets.
``excess`` = observed minus null mean (positive = better than the null).

Light vs dark
-------------
Models are trained and tested within one condition. For every analysis the
light and dark sample sets are equalised by random subsampling: per
(stratum, label) in analysis 1, per (cell, remaining steps) in analysis 2 and
per class in analysis 3, both conditions keep the smaller of their two
counts. Condition ``all`` pools light and dark samples without equalisation
(more samples, used to test whether a code exists at all).

References
----------
Frank LM, Brown EN, Wilson M. 2000. "Trajectory encoding in the hippocampus
and entorhinal cortex." Neuron 27:169-178. doi:10.1016/S0896-6273(00)00018-0

Wood ER, Dudchenko PA, Robitsek RJ, Eichenbaum H. 2000. "Hippocampal neurons
encode information about different types of memory episodes occurring in the
same location." Neuron 27:623-633. doi:10.1016/S0896-6273(00)00071-4

Ferbinteanu J, Shapiro ML. 2003. "Prospective and retrospective memory
coding in the hippocampus." Neuron 40:1227-1239.
doi:10.1016/S0896-6273(03)00752-9

Alexander AS, Nitz DA. 2015. "Retrosplenial cortex maps the conjunction of
internal and external spaces." Nature Neuroscience 18:1143-1151.
doi:10.1038/nn.4058

Vedder LC, Miller AMP, Harrison MB, Smith DM. 2017. "Retrosplenial cortical
neurons encode navigational cues, trajectories and reward locations during
goal directed navigation." Cerebral Cortex 27:3713-3723.
doi:10.1093/cercor/bhw192

Howard LR, Javadi AH, Yu Y, et al. 2014. "The hippocampus and entorhinal
cortex encode the path and Euclidean distances to goals during navigation."
Current Biology 24:1331-1340. doi:10.1016/j.cub.2014.05.001

Pfeiffer BE, Foster DJ. 2013. "Hippocampal place-cell sequences depict future
paths to remembered goals." Nature 497:74-79. doi:10.1038/nature12112

Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth show
rapid learning, sudden insight, and efficient exploration." eLife 10:e66175.
doi:10.7554/eLife.66175

Weinreb C, Pearl JE, Lin S, et al. 2024. "Keypoint-MoSeq: parsing behavior by
linking point tracking to pose dynamics." Nature Methods 21:1329-1339.
doi:10.1038/s41592-024-02318-2

Kemeny JG, Snell JL. 1960. "Finite Markov Chains." Van Nostrand, Princeton.

Hoerl AE, Kennard RW. 1970. "Ridge regression: biased estimation for
nonorthogonal problems." Technometrics 12:55-67.
doi:10.1080/00401706.1970.10488634

Kim S. 2015. "ppcor: an R package for a fast calculation to semi-partial
correlation coefficients." Communications for Statistical Applications and
Methods 22:665-674. doi:10.5351/CSAM.2015.22.6.665

Roberts DR, Bahn V, Ciuti S, et al. 2017. "Cross-validation strategies for
data with temporal, spatial, hierarchical, or phylogenetic structure."
Ecography 40:913-929. doi:10.1111/ecog.02881

Brodersen KH, Ong CS, Stephan KE, Buhmann JM. 2010. "The balanced accuracy
and its posterior distribution." ICPR 2010, 3121-3124.
doi:10.1109/ICPR.2010.764

Pedregosa F, Varoquaux G, Gramfort A, et al. 2011. "Scikit-learn: Machine
Learning in Python." Journal of Machine Learning Research 12:2825-2830.
https://github.com/scikit-learn/scikit-learn
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from hm2p.analysis.population_maze import (
    Problem,
    Samples,
    WindowMeans,
    _make_samples,
    _wilcoxon,
    draw_shifts,
    impute_columns,
    null_summary,
    prepare_problem,
    session_cells,
    time_block_folds,
)
from hm2p.maze.topology import RoseMaze, build_rose_maze

log = logging.getLogger(__name__)

ANALYSES = ("splitter", "distance", "planning")
FEATURE_SETS = ("neural", "behaviour", "neural_hd_removed")
CONDITIONS = ("light", "dark", "all")
BEHAVIOUR_CHANNELS = (
    "hd_sin",
    "hd_cos",
    "speed",
    "abs_ahv",
    "t_session",
    "t_since_dead_end_exit",
    "t_since_dead_end_entry",
    "n_distinct_dead_ends",
)
CLF_METRICS = ("balanced_accuracy",)
REG_METRICS = ("spearman", "partial_spearman")
PRO_MINUS_RETRO = "prospective_minus_retrospective"
SYLLABLE_FEATURE_SET = "behaviour_plus_syllable"
FEATURE_COMPARISONS = (
    ("neural", "behaviour"),
    ("neural_hd_removed", "behaviour"),
    ("neural", "neural_hd_removed"),
    ("neural", SYLLABLE_FEATURE_SET),
    ("neural_hd_removed", SYLLABLE_FEATURE_SET),
)
DISTANCE_VARIANTS = ("pooled", "place_controlled", "place_direction_controlled")
PLAN_WINDOWS = ("pre", "post")
PLAN_FILTERS = ("", "_noreturn", "_noreturn_long")
SESSION_KEYS = ["analysis", "target", "variant", "condition", "feature_set", "metric"]


@dataclass
class MazeMemParams:
    """Parameters of the maze memory and planning analyses (see module docstring)."""

    n_shuffles: int = 200
    min_shift_s: float = 30.0
    n_blocks: int = 5
    gap_s: float = 5.0
    C: float = 1.0
    max_iter: int = 25
    ridge_alpha: float = 10.0
    min_dwell_frames: int = 2
    max_gap_s: float = 1.0
    max_jump_steps: int = 2
    light_hi: float = 0.9
    light_lo: float = 0.1
    min_n: int = 5
    min_reg_samples: int = 30
    min_class_samples: int = 5
    plan_window_s: float = 1.0
    long_trip_steps: int = 3
    n_top_syllables: int = 8
    min_dead_end_dwell_s: float = 0.5
    recent_k: int = 3
    n_rw_sims: int = 2000
    conditions: tuple[str, ...] = CONDITIONS
    subset_size: int = 10
    n_subsets: int = 5
    n_subset_shuffles: int = 10
    shift_chunk: int = 64
    n_jobs: int = 1
    seed: int = 0


# ---------------------------------------------------------------------------
# Visits, trips and conditions
# ---------------------------------------------------------------------------

VISIT_COLUMNS = ["cell", "start", "end", "seg", "interp", "prev", "next"]


def visit_table(
    cells: npt.ArrayLike,
    maze: RoseMaze,
    max_gap_frames: int = 10,
    max_jump_steps: int = 2,
) -> pd.DataFrame:
    """Sequence of cell visits with segment breaks and filled graph jumps.

    Parameters
    ----------
    cells : (n_frames,) int
        Maze cell index per frame, -1 where unassigned.
    maze : RoseMaze
    max_gap_frames : int
        Runs of the same cell separated by at most this many unassigned
        frames are merged; a longer gap between any two visits starts a new
        segment.
    max_jump_steps : int
        Consecutive visits whose cells are 2..``max_jump_steps`` graph steps
        apart are joined by zero-length visits to the intermediate cells
        (unique on the tree); larger jumps start a new segment.

    Returns
    -------
    pandas.DataFrame
        Columns ``cell, start, end`` (frames, end exclusive; ``start == end``
        for filled cells), ``seg`` (segment index), ``interp`` (filled cell),
        ``prev``/``next`` (neighbouring visit's cell in the same segment, -1
        at segment edges), in time order.
    """
    c = np.asarray(cells, dtype=np.int64).ravel()
    idx = np.flatnonzero(c >= 0)
    if idx.size == 0:
        return pd.DataFrame({k: np.zeros(0, dtype=np.int64) for k in VISIT_COLUMNS})
    v = c[idx]
    gap = np.diff(idx) - 1
    new = (np.diff(v) != 0) | (gap > max_gap_frames)
    first = np.concatenate([[0], np.flatnonzero(new) + 1])
    last = np.concatenate([first[1:] - 1, [v.size - 1]])
    rc, rs, re = v[first], idx[first], idx[last] + 1
    rows: list[tuple[int, int, int, int, int]] = []
    seg = 0
    for r in range(rc.size):
        if r > 0:
            a, b = int(rc[r - 1]), int(rc[r])
            d = int(maze.dist[a, b])
            if rs[r] - re[r - 1] > max_gap_frames or d > max_jump_steps or d <= 0:
                seg += 1
            elif d > 1:
                path = maze.path(maze.cell_list[a], maze.cell_list[b]) or []
                for mid in path[1:-1]:
                    e = int(re[r - 1])
                    rows.append((maze.cell_to_idx[mid], e, e, seg, 1))
        rows.append((int(rc[r]), int(rs[r]), int(re[r]), seg, 0))
    arr = np.asarray(rows, dtype=np.int64)
    df = pd.DataFrame(arr, columns=["cell", "start", "end", "seg", "interp"])
    df["interp"] = df["interp"].astype(bool)
    same_prev = np.concatenate([[False], arr[1:, 3] == arr[:-1, 3]])
    same_next = np.concatenate([arr[:-1, 3] == arr[1:, 3], [False]])
    df["prev"] = np.where(same_prev, np.roll(arr[:, 0], 1), -1)
    df["next"] = np.where(same_next, np.roll(arr[:, 0], -1), -1)
    return df


def dead_end_indices(maze: RoseMaze) -> np.ndarray:
    """Sorted cell indices of the maze dead ends."""
    return np.sort(np.asarray([maze.cell_to_idx[c] for c in maze.dead_ends], dtype=np.int64))


TRIP_COLUMNS = ["i_origin", "i_dest", "origin", "dest", "start", "end", "seg", "n_steps"]


def trip_table(visits: pd.DataFrame, maze: RoseMaze) -> pd.DataFrame:
    """Trips between consecutive dead-end visits of one segment.

    Returns
    -------
    pandas.DataFrame
        Columns ``i_origin, i_dest`` (row positions in *visits*), ``origin,
        dest`` (dead-end cells), ``start`` (frame the origin visit ends),
        ``end`` (frame the destination visit starts), ``seg`` and
        ``n_steps`` (cell transitions from origin to destination).
    """
    if visits.empty:
        return pd.DataFrame({k: np.zeros(0, dtype=np.int64) for k in TRIP_COLUMNS})
    cell = visits["cell"].to_numpy(np.int64)
    seg = visits["seg"].to_numpy(np.int64)
    de = np.flatnonzero(np.isin(cell, dead_end_indices(maze)))
    i, j = de[:-1], de[1:]
    ok = seg[i] == seg[j]
    i, j = i[ok], j[ok]
    return pd.DataFrame(
        {
            "i_origin": i,
            "i_dest": j,
            "origin": cell[i],
            "dest": cell[j],
            "start": visits["end"].to_numpy(np.int64)[i],
            "end": visits["start"].to_numpy(np.int64)[j],
            "seg": seg[i],
            "n_steps": j - i,
        }
    )


def light_condition(
    light_on: npt.ArrayLike,
    starts: npt.ArrayLike,
    ends: npt.ArrayLike,
    hi: float = 0.9,
    lo: float = 0.1,
) -> np.ndarray:
    """``light``/``dark``/``mixed`` per frame window ``[start, end)``.

    ``light`` when the fraction of light-on frames is at least *hi*, ``dark``
    when at most *lo*; ``mixed`` otherwise and for empty windows.
    """
    on = np.asarray(light_on, dtype=np.float64).ravel()
    cs = np.concatenate([[0.0], np.cumsum(on)])
    st = np.clip(np.asarray(starts, dtype=np.int64), 0, on.size)
    en = np.clip(np.asarray(ends, dtype=np.int64), 0, on.size)
    length = en - st
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(length > 0, (cs[en] - cs[st]) / np.maximum(length, 1), np.nan)
    out = np.full(st.size, "mixed", dtype=object)
    out[frac >= hi] = "light"
    out[frac <= lo] = "dark"
    return out


def passthrough_samples(visits: pd.DataFrame, trips: pd.DataFrame, maze: RoseMaze) -> pd.DataFrame:
    """Pass-through visits inside trips, with trip-level labels.

    A visit qualifies when it has frames (not filled) and its arrival
    neighbour differs from its departure neighbour.

    Returns
    -------
    pandas.DataFrame
        Columns ``start, end, cell, prev, next, origin, dest, trip`` (row in
        *trips*), ``elapsed`` (transitions since leaving the origin, >= 1) and
        ``remaining`` (graph distance to the destination).
    """
    cols = ["start", "end", "cell", "prev", "next", "origin", "dest", "trip"]
    cols += ["elapsed", "remaining"]
    if trips.empty:
        return pd.DataFrame({k: np.zeros(0, dtype=np.int64) for k in cols})
    st = visits["start"].to_numpy(np.int64)
    en = visits["end"].to_numpy(np.int64)
    cell = visits["cell"].to_numpy(np.int64)
    prv = visits["prev"].to_numpy(np.int64)
    nxt = visits["next"].to_numpy(np.int64)
    interp = visits["interp"].to_numpy(bool)
    io = trips["i_origin"].to_numpy(np.int64)
    idest = trips["i_dest"].to_numpy(np.int64)
    n_in = idest - io - 1
    trip = np.repeat(np.arange(len(trips)), n_in)
    k = np.concatenate([np.arange(a + 1, b) for a, b in zip(io, idest, strict=True)])
    k = k.astype(np.int64)
    keep = ~interp[k] & (prv[k] != nxt[k]) & (prv[k] >= 0) & (nxt[k] >= 0)
    k, trip = k[keep], trip[keep]
    dest = trips["dest"].to_numpy(np.int64)[trip]
    return pd.DataFrame(
        {
            "start": st[k],
            "end": en[k],
            "cell": cell[k],
            "prev": prv[k],
            "next": nxt[k],
            "origin": trips["origin"].to_numpy(np.int64)[trip],
            "dest": dest,
            "trip": trip,
            "elapsed": k - io[trip],
            "remaining": maze.dist[cell[k], dest].astype(np.int64),
        }
    )


# ---------------------------------------------------------------------------
# Dead-end history and random-walk expectation
# ---------------------------------------------------------------------------


RW_MODELS = ("uniform", "nonbacktracking")


def dead_end_absorption(maze: RoseMaze, backtrack: bool = True) -> np.ndarray:
    """First-hit dead-end distribution of a random walk leaving each dead end.

    The walk starts at dead end ``o`` and stops at the first dead end it
    reaches after leaving ``o`` (``o`` itself included). With
    ``backtrack=True`` it moves to a uniformly chosen neighbour at every
    step (the memoryless walker of
    :func:`hm2p.maze.exploration_complexity.random_walk_coverage_null`);
    with ``backtrack=False`` it never reverses along the edge it arrived by
    (uniform among the other neighbours), which matches a forward-biased
    walker without memory. Solved exactly as an absorbing Markov chain on
    cells (or on directed edges for the non-backtracking walk).

    Returns
    -------
    (n_cells, n_cells) float
        ``H[o, d]`` = probability of first reaching dead end ``d`` from dead
        end ``o``; rows of non-dead-end cells are zero; dead-end rows sum
        to 1.

    References
    ----------
    Kemeny JG, Snell JL. 1960. "Finite Markov Chains." Van Nostrand.
    """
    n = len(maze.cell_list)
    de = dead_end_indices(maze)
    is_de = np.zeros(n, dtype=bool)
    is_de[de] = True
    nbr = [[maze.cell_to_idx[x] for x in maze.adj[c]] for c in maze.cell_list]
    h = np.zeros((n, n))
    if backtrack:
        p = np.zeros((n, n))
        for i, nb in enumerate(nbr):
            if nb:
                p[i, nb] = 1.0 / len(nb)
        tr = np.flatnonzero(~is_de)
        b = np.linalg.solve(np.eye(tr.size) - p[np.ix_(tr, tr)], p[np.ix_(tr, de)])
        for o in de:
            h[o, de] = p[o, de] + p[o, tr] @ b
        return h
    edges = [(u, v) for u in range(n) for v in nbr[u]]
    eidx = {e: k for k, e in enumerate(edges)}
    trans = [k for k, (_, v) in enumerate(edges) if not is_de[v]]
    tpos = {k: j for j, k in enumerate(trans)}
    q = np.zeros((len(trans), len(trans)))
    r = np.zeros((len(trans), n))
    for k in trans:
        u, v = edges[k]
        out = [w for w in nbr[v] if w != u] or [u]
        for w in out:
            if is_de[w]:
                r[tpos[k], w] += 1.0 / len(out)
            else:
                q[tpos[k], tpos[eidx[(v, w)]]] += 1.0 / len(out)
    b = np.linalg.solve(np.eye(len(trans)) - q, r)
    for o in de:
        for v in nbr[o]:
            k = eidx[(o, v)]
            if is_de[v]:
                h[o, v] += 1.0 / len(nbr[o])
            else:
                h[o] += b[tpos[k]] / len(nbr[o])
    return h


def trip_novelty(
    visits: pd.DataFrame,
    trips: pd.DataFrame,
    maze: RoseMaze,
    recent_k: int = 3,
    absorption: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Novelty labels of each trip's destination and random-walk probabilities.

    History = all dead-end visits of the session up to and including the
    trip's origin visit (across segments).

    Parameters
    ----------
    absorption : dict or None
        Random-walk model name -> :func:`dead_end_absorption` matrix;
        default both models in :data:`RW_MODELS`.

    Returns
    -------
    pandas.DataFrame
        One row per trip: ``novel`` (destination not among the *recent_k*
        most recently visited distinct dead ends), ``lru`` (destination is a
        least recently visited dead end other than the origin; never-visited
        dead ends tie as least recent), ``p_novel_<model>`` /
        ``p_lru_<model>`` (random-walk probabilities of the same events) and
        ``n_distinct_before``.
    """
    if absorption is None:
        absorption = {
            "uniform": dead_end_absorption(maze, True),
            "nonbacktracking": dead_end_absorption(maze, False),
        }
    cols = ["novel", "lru"]
    cols += [f"p_{lab}_{m}" for m in absorption for lab in ("novel", "lru")]
    cols += ["n_distinct_before"]
    if trips.empty:
        return pd.DataFrame({k: np.zeros(0) for k in cols})
    de = dead_end_indices(maze)
    cell = visits["cell"].to_numpy(np.int64)
    de_rows = np.flatnonzero(np.isin(cell, de))
    pos = {int(r): k for k, r in enumerate(de_rows)}
    seq = cell[de_rows]
    rows = []
    for io, o, d in zip(
        trips["i_origin"].to_numpy(np.int64),
        trips["origin"].to_numpy(np.int64),
        trips["dest"].to_numpy(np.int64),
        strict=True,
    ):
        hist = seq[: pos[int(io)] + 1]
        recent: list[int] = []
        for x in hist[::-1]:
            if int(x) not in recent:
                recent.append(int(x))
                if len(recent) == recent_k:
                    break
        last = np.full(de.size, -1, dtype=np.int64)
        for t, x in enumerate(hist):
            last[np.searchsorted(de, x)] = t
        cand = de != o
        lru_set = de[cand & (last == last[cand].min())] if cand.any() else de[:0]
        not_recent = de[~np.isin(de, recent)]
        row: list[Any] = [bool(int(d) not in recent), bool(np.isin(d, lru_set))]
        for h in absorption.values():
            row += [float(h[o, not_recent].sum()), float(h[o, lru_set].sum())]
        rows.append((*row, int(np.unique(hist).size)))
    return pd.DataFrame(rows, columns=cols)


def novelty_vs_random_walk(
    observed: npt.ArrayLike,
    p_rw: npt.ArrayLike,
    n_sims: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Observed fraction of events vs the random-walk expectation.

    Parameters
    ----------
    observed : (n,) bool
        Event per trip (e.g. destination novel).
    p_rw : (n,) float
        Random-walk probability of the event for each trip.
    n_sims, rng
        Monte Carlo draws of independent Bernoulli(``p_rw``) trials.

    Returns
    -------
    dict
        ``n_trips, observed_frac, expected_frac, excess`` (observed minus
        expected), ``z`` (against the Monte Carlo distribution) and ``p``
        (two-sided, ``(1 + #|sim - E| >= |obs - E|) / (1 + n_sims)``).
    """
    o = np.asarray(observed, dtype=bool)
    p = np.clip(np.asarray(p_rw, dtype=np.float64), 0.0, 1.0)
    nan = float("nan")
    out = {"n_trips": int(o.size), "observed_frac": nan, "expected_frac": nan}
    out.update({"excess": nan, "z": nan, "p": nan})
    if o.size == 0:
        return out
    exp = float(p.mean())
    obs = float(o.mean())
    sims = (rng.random((n_sims, o.size)) < p[None, :]).mean(axis=1)
    sd = float(sims.std())
    out.update(
        {
            "observed_frac": obs,
            "expected_frac": exp,
            "excess": obs - exp,
            "z": (obs - exp) / sd if sd > 0 else nan,
            "p": float((1 + np.sum(np.abs(sims - exp) >= abs(obs - exp) - 1e-12)) / (1 + n_sims)),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Behaviour channels
# ---------------------------------------------------------------------------


def _time_since(events: np.ndarray, n: int, fps: float) -> np.ndarray:
    """Seconds since the most recent event frame (NaN before the first)."""
    f = np.arange(n)
    ev = np.sort(np.asarray(events, dtype=np.int64))
    k = np.searchsorted(ev, f, side="right") - 1
    out = np.full(n, np.nan)
    ok = k >= 0
    out[ok] = (f[ok] - ev[k[ok]]) / fps
    return out


def behaviour_channels(
    hd_deg: npt.ArrayLike,
    speed: npt.ArrayLike,
    ahv: npt.ArrayLike,
    visits: pd.DataFrame,
    maze: RoseMaze,
    fps: float,
    valid: npt.ArrayLike | None = None,
) -> np.ndarray:
    """Behaviour channels ``(8, n_frames)`` in the order of :data:`BEHAVIOUR_CHANNELS`.

    sin/cos HD, speed, |AHV|, time in session (s), time since the last
    dead-end exit (s), time since the last dead-end entry (s) and the number of
    distinct dead ends entered so far. Invalid frames are NaN.
    """
    hd = np.deg2rad(np.asarray(hd_deg, dtype=np.float64).ravel())
    n = hd.size
    cell = visits["cell"].to_numpy(np.int64)
    is_de = np.isin(cell, dead_end_indices(maze))
    entries = visits["start"].to_numpy(np.int64)[is_de]
    exits = visits["end"].to_numpy(np.int64)[is_de]
    seen: set[int] = set()
    n_seen = np.zeros(entries.size)
    for j, x in enumerate(cell[is_de]):
        seen.add(int(x))
        n_seen[j] = len(seen)
    order = np.argsort(entries, kind="stable")
    k = np.searchsorted(entries[order], np.arange(n), side="right") - 1
    n_distinct = np.zeros(n)
    if entries.size:
        n_distinct = np.where(k >= 0, n_seen[order][np.clip(k, 0, None)], 0.0)
    out = np.vstack(
        [
            np.sin(hd),
            np.cos(hd),
            np.asarray(speed, dtype=np.float64).ravel(),
            np.abs(np.asarray(ahv, dtype=np.float64).ravel()),
            np.arange(n) / fps,
            _time_since(exits, n, fps),
            _time_since(entries, n, fps),
            n_distinct,
        ]
    )
    if valid is not None:
        out[:, ~np.asarray(valid, dtype=bool).ravel()] = np.nan
    return out


def behaviour_window_features(
    bwm: WindowMeans, starts: npt.ArrayLike, ends: npt.ArrayLike, shift: int = 0
) -> np.ndarray:
    """Window features ``(n, 8)``: circular-mean HD (sin, cos) and the other channel means."""
    m = bwm.means(starts, ends, shift)
    r = np.hypot(m[:, 0], m[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        m[:, 0] = np.where(r > 0, m[:, 0] / r, np.nan)
        m[:, 1] = np.where(r > 0, m[:, 1] / r, np.nan)
    return m


# ---------------------------------------------------------------------------
# Condition-matched sample selection
# ---------------------------------------------------------------------------


def matched_condition_indices(
    strata: npt.ArrayLike,
    labels: npt.ArrayLike,
    cond: npt.ArrayLike,
    conditions: tuple[str, ...],
    min_n: int,
    min_labels: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Sample indices per condition with equal (stratum, label) counts across conditions.

    For every (stratum, label) pair the count in each condition is
    computed and ``m`` = the smallest; pairs with ``m < min_n`` are dropped,
    then strata with fewer than *min_labels* remaining labels are dropped.
    Each condition keeps a random subset of ``m`` samples per remaining
    pair, so all conditions have identical (stratum, label) counts. With a
    single condition this is a plain minimum-count filter.

    Returns
    -------
    dict
        Condition -> sorted int64 indices (empty arrays when nothing remains).
    """
    s = np.asarray(strata, dtype=np.int64)
    lab = np.asarray(labels, dtype=np.int64)
    cnd = np.asarray(cond, dtype=object)
    empty = {c: np.zeros(0, dtype=np.int64) for c in conditions}
    sel = np.isin(cnd, list(conditions))
    if not sel.any():
        return empty
    pairs = np.unique(np.column_stack([s[sel], lab[sel]]), axis=0)
    m = np.array(
        [
            min(int(np.sum((s == a) & (lab == b) & (cnd == c))) for c in conditions)
            for a, b in pairs
        ]
    )
    ok = m >= max(min_n, 1)
    pairs, m = pairs[ok], m[ok]
    if pairs.size == 0:
        return empty
    us, cnt = np.unique(pairs[:, 0], return_counts=True)
    good = np.isin(pairs[:, 0], us[cnt >= min_labels])
    pairs, m = pairs[good], m[good]
    out: dict[str, list[np.ndarray]] = {c: [] for c in conditions}
    for (a, b), k in zip(pairs, m, strict=True):
        for c in conditions:
            idx = np.flatnonzero((s == a) & (lab == b) & (cnd == c))
            out[c].append(idx if idx.size == k else rng.choice(idx, int(k), replace=False))
    return {
        c: np.sort(np.concatenate(v)).astype(np.int64) if v else empty[c] for c, v in out.items()
    }


def condition_index_sets(
    strata: npt.ArrayLike,
    labels: npt.ArrayLike,
    cond: npt.ArrayLike,
    conditions: tuple[str, ...],
    min_n: int,
    min_labels: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Equalised ``light``/``dark`` indices plus unequalised ``all`` (light + dark pooled)."""
    out: dict[str, np.ndarray] = {}
    pair = tuple(c for c in conditions if c in ("light", "dark"))
    if pair:
        out.update(matched_condition_indices(strata, labels, cond, pair, min_n, min_labels, rng))
    if "all" in conditions:
        c = np.asarray(cond, dtype=object)
        pooled = np.where(np.isin(c, ["light", "dark"]), "all", "mixed")
        out.update(
            matched_condition_indices(strata, labels, pooled, ("all",), min_n, min_labels, rng)
        )
    return out


# ---------------------------------------------------------------------------
# Cross-validated fits vectorised over circular shifts
# ---------------------------------------------------------------------------
#
# Every shift of the null has the same samples, labels and folds; only the
# feature matrix changes. The fits below therefore take a stack of feature
# matrices ``(n_shifts, n_samples, n_features)`` and solve all shifts at
# once with batched linear algebra, which is what makes 200 shifts per
# session affordable for problems with many small strata.


def _group_demean(
    xtr: np.ndarray, gtr: np.ndarray, xte: np.ndarray, gte: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract training-fold group means along the sample axis.

    *xtr*/*xte* are ``(n,)``, ``(n, d)`` or ``(S, n, d)``; the sample axis
    is the one matching ``gtr``/``gte``. Returns ``(xtr, xte, known)`` where
    ``known`` marks test samples whose group occurs in training.
    """
    ug, inv = np.unique(gtr, return_inverse=True)
    avg = np.zeros((ug.size, gtr.size))
    avg[inv, np.arange(gtr.size)] = 1.0
    avg /= avg.sum(axis=1, keepdims=True)
    pos = np.clip(np.searchsorted(ug, gte), 0, ug.size - 1)
    known = ug[pos] == gte
    if xtr.ndim == 1:
        means = avg @ xtr
        return xtr - means[inv], xte - means[pos], known
    means = avg @ xtr  # (G, d) or (S, G, d)
    if xtr.ndim == 2:
        return xtr - means[inv], xte - means[pos], known
    return xtr - means[:, inv], xte - means[:, pos], known


def _impute_batched(x: np.ndarray) -> np.ndarray:
    """:func:`population_maze.impute_columns` applied to each matrix of a ``(S, n, d)`` stack."""
    fin = np.isfinite(x)
    cnt = fin.sum(axis=1, keepdims=True)
    tot = np.where(fin, x, 0.0).sum(axis=1, keepdims=True)
    col = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
    return np.where(fin, x, col)


def prepare_fold(
    xs: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    cov: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training and test features of one fold for a ``(S, n, d)`` stack.

    Steps, all with training-fold statistics only: per-group demeaning
    (when *groups* is given), residualisation on ``[1, cov]`` by least
    squares (when *cov* is given; as in :func:`population_maze.cv_predict`),
    z-scoring.

    Returns
    -------
    ztr : (S, n_train, d)
    zte : (S, n_test, d)
    known : (n_test,) bool
        Test samples whose group occurs in training (all True without groups).
    """
    xtr, xte = xs[:, tr], xs[:, te]
    known = np.ones(te.size, dtype=bool)
    if groups is not None:
        xtr, xte, known = _group_demean(xtr, groups[tr], xte, groups[te])
    if cov is not None:
        dtr = np.column_stack([np.ones(tr.size), cov[tr]])
        dte = np.column_stack([np.ones(te.size), cov[te]])
        beta = np.linalg.pinv(dtr) @ xtr  # (S, k, d)
        xtr = xtr - dtr @ beta
        xte = xte - dte @ beta
    mu = xtr.mean(axis=1, keepdims=True)
    sd = xtr.std(axis=1, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (xtr - mu) / sd, (xte - mu) / sd, known


def logistic_newton(
    z: np.ndarray,
    y01: np.ndarray,
    w: np.ndarray,
    C: float = 1.0,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> np.ndarray:
    """L2-regularised weighted binary logistic regressions fitted by Newton's method.

    Minimises ``sum_i w_i * logloss_i + ||coef||^2 / (2 C)`` (intercept not
    penalised), the objective of scikit-learn's ``LogisticRegression`` with
    an lbfgs solver, for every shift and every binary problem at once.
    Steps are capped at 10 per coefficient to keep early iterations stable
    on separable data.

    Parameters
    ----------
    z : (S, n, d) float
        Features (finite; z-scored).
    y01 : (B, n) float
        0/1 targets of B binary problems sharing the features.
    w : (B, n) or (1, n) float
        Sample weights.
    C : float
        Inverse L2 strength.
    max_iter, tol
        Newton iterations and convergence threshold on the largest step.

    Returns
    -------
    (S, B, d + 1) float
        Coefficients, intercept last.

    References
    ----------
    Pedregosa F, et al. 2011. "Scikit-learn: Machine Learning in Python."
    Journal of Machine Learning Research 12:2825-2830.
    """
    from scipy.special import expit

    s, n, d = z.shape
    b = y01.shape[0]
    z1 = np.concatenate([z, np.ones((s, n, 1))], axis=2)
    lam = np.full(d + 1, 1.0 / C)
    lam[-1] = 1e-8
    beta = np.zeros((s, b, d + 1))
    zt = np.swapaxes(z1, 1, 2)  # (S, D, n)
    z4 = z1[:, None]  # (S, 1, n, D)
    diag = np.arange(d + 1)
    for _ in range(max_iter):
        pr = expit(beta @ zt)  # (S, B, n)
        g = (w * (pr - y01)) @ z1 + lam * beta
        v = w * pr * (1.0 - pr)
        h = np.swapaxes(z4 * v[..., None], -1, -2) @ z4  # (S, B, D, D)
        h[..., diag, diag] += lam
        step = np.linalg.solve(h, g[..., None])[..., 0]
        big = np.abs(step).max(axis=-1, keepdims=True)
        step *= np.minimum(1.0, 10.0 / np.maximum(big, 1e-300))
        beta -= step
        if float(np.abs(step).max()) < tol:
            break
    return beta


def batched_logistic_predict(
    xs: np.ndarray,
    y: npt.ArrayLike,
    folds: list[tuple[np.ndarray, np.ndarray]],
    C: float = 1.0,
    cov: np.ndarray | None = None,
    max_iter: int = 25,
) -> np.ndarray:
    """Out-of-fold predictions of balanced L2 logistic regression for every shift.

    Same model as :func:`population_maze.cv_predict` (balanced class
    weights ``n / (n_classes * n_c)``, optional training-fold HD
    residualisation, z-scoring); binary problems are one logistic
    regression, problems with more classes are one-vs-rest with the
    multiclass balanced weights and the arg-max decision value.

    Parameters
    ----------
    xs : (S, n, d) float
        Finite features for S shifts.
    y : (n,) int
    folds : list of (train_idx, test_idx)
    C : float
    cov : (n, k) float or None
        Residualisation covariates (not shifted).
    max_iter : int

    Returns
    -------
    (S, n) int64
        Predicted label; -1 where a sample is in no test fold or its fold
        had no training samples. A training fold with one class predicts it.
    """
    ya = np.asarray(y, dtype=np.int64)
    pred = np.full((xs.shape[0], ya.size), -1, dtype=np.int64)
    for tr, te in folds:
        if tr.size == 0 or te.size == 0:
            continue
        classes, inv, cnt = np.unique(ya[tr], return_inverse=True, return_counts=True)
        k = classes.size
        if k == 1:
            pred[:, te] = classes[0]
            continue
        ztr, zte, _ = prepare_fold(xs, tr, te, cov)
        w = (tr.size / (k * cnt))[inv][None, :]
        if k == 2:
            y01 = (inv == 1).astype(np.float64)[None, :]
        else:
            y01 = (inv[None, :] == np.arange(k)[:, None]).astype(np.float64)
        beta = logistic_newton(ztr, y01, w, C, max_iter)
        zte1 = np.concatenate([zte, np.ones(zte.shape[:2] + (1,))], axis=2)
        dec = beta @ np.swapaxes(zte1, 1, 2)  # (S, B, n_test)
        lab = (dec[:, 0] > 0).astype(np.int64) if k == 2 else np.argmax(dec, axis=1)
        pred[:, te] = classes[lab]
    return pred


def balanced_accuracy_batched(y: npt.ArrayLike, pred: np.ndarray) -> np.ndarray:
    """Balanced accuracy (mean per-class recall) of each row of ``pred`` ``(S, n)``.

    Brodersen KH, Ong CS, Stephan KE, Buhmann JM. 2010. "The balanced
    accuracy and its posterior distribution." ICPR 2010, 3121-3124.
    doi:10.1109/ICPR.2010.764
    """
    ya = np.asarray(y)
    if ya.size == 0:
        return np.full(pred.shape[0], np.nan)
    rec = [(pred[:, ya == c] == c).mean(axis=1) for c in np.unique(ya)]
    return np.mean(np.vstack(rec), axis=0)


def batched_ridge_predict(
    xs: np.ndarray,
    y: npt.ArrayLike,
    folds: list[tuple[np.ndarray, np.ndarray]],
    alpha: float = 10.0,
    cov: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold ridge regression predictions for every shift.

    Per fold (training statistics only): optional per-group demeaning of
    features and target (test samples of groups absent from training are
    left unpredicted), optional residualisation of the features on
    ``[1, cov]``, z-scoring, then ridge regression with penalty *alpha* on
    the centred target.

    Parameters
    ----------
    xs : (S, n, d) float
    y : (n,) float
    folds : list of (train_idx, test_idx)
    alpha : float
    cov : (n, k) float or None
    groups : (n,) int or None

    Returns
    -------
    pred : (S, n) float
        NaN where not predicted.
    y_eval : (n,) float
        The target the prediction is compared with (group-demeaned with
        training-fold means when *groups* is given); NaN where not predicted.

    References
    ----------
    Hoerl AE, Kennard RW. 1970. "Ridge regression: biased estimation for
    nonorthogonal problems." Technometrics 12:55-67.
    doi:10.1080/00401706.1970.10488634
    """
    ya = np.asarray(y, dtype=np.float64)
    pred = np.full((xs.shape[0], ya.size), np.nan)
    yev = np.full(ya.size, np.nan)
    eye = np.eye(xs.shape[2])
    for tr, te in folds:
        if tr.size < 3 or te.size == 0:
            continue
        ztr, zte, known = prepare_fold(xs, tr, te, cov, groups)
        ytr, yte = ya[tr], ya[te]
        if groups is not None:
            ytr, yte, _ = _group_demean(ytr, groups[tr], yte, groups[te])
        ym = ytr.mean()
        zt = np.swapaxes(ztr, 1, 2)
        coef = np.linalg.solve(zt @ ztr + alpha * eye, zt @ (ytr - ym)[:, None])
        p = (zte @ coef)[..., 0] + ym
        pred[:, te[known]] = p[:, known]
        yev[te[known]] = yte[known]
    return pred, yev


def _residual_ranks(v: np.ndarray, design: np.ndarray) -> np.ndarray:
    beta = np.linalg.lstsq(design, v, rcond=None)[0]
    return v - design @ beta


def partial_spearman(x: npt.ArrayLike, y: npt.ArrayLike, covs: npt.ArrayLike | None) -> float:
    """Spearman correlation of *x* and *y*, partialling *covs* (rank-based partial correlation).

    All variables are rank-transformed; the ranks of *x* and *y* are
    residualised on ``[1, ranks(covs)]`` by least squares and correlated.
    Rows with any non-finite value are dropped. NaN with fewer than
    ``n_covs + 3`` rows or a constant residual. With ``covs=None`` this is the
    ordinary Spearman correlation.

    References
    ----------
    Kim S. 2015. "ppcor: an R package for a fast calculation to semi-partial
    correlation coefficients." Communications for Statistical Applications
    and Methods 22:665-674. doi:10.5351/CSAM.2015.22.6.665
    """
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    ca = np.zeros((xa.size, 0)) if covs is None else np.asarray(covs, dtype=np.float64)
    ca = ca.reshape(xa.size, -1)
    ok = np.isfinite(xa) & np.isfinite(ya) & np.all(np.isfinite(ca), axis=1)
    if ok.sum() < ca.shape[1] + 3:
        return float("nan")
    rx = stats.rankdata(xa[ok])
    ry = stats.rankdata(ya[ok])
    rc = np.column_stack([stats.rankdata(c) for c in ca[ok].T]) if ca.shape[1] else ca[ok]
    design = np.column_stack([np.ones(rx.size), rc])
    ex = _residual_ranks(rx, design)
    ey = _residual_ranks(ry, design)
    sx, sy = np.sqrt(ex @ ex), np.sqrt(ey @ ey)
    if sx < 1e-9 * max(1.0, rx.size) or sy < 1e-9 * max(1.0, rx.size):
        return float("nan")
    return float(np.clip((ex @ ey) / (sx * sy), -1.0, 1.0))


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """One decoding problem: analysis x target x variant x condition.

    ``widx`` indexes the session's global window list. Classification tasks
    carry a :class:`Problem`; regression tasks the target, folds, optional
    groups (place control) and the evaluation covariates for the partial
    correlation.
    """

    analysis: str
    target: str
    variant: str
    condition: str
    kind: str
    widx: np.ndarray
    reason: str = ""
    n_samples: int = 0
    n_strata: int = 0
    n_classes: int = 0
    problem: Problem | None = None
    y: np.ndarray | None = None
    folds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    groups: np.ndarray | None = None
    covs: np.ndarray | None = None
    extra_feature_sets: tuple[str, ...] = ()

    @property
    def metrics(self) -> tuple[str, ...]:
        """Metrics reported for this task."""
        return CLF_METRICS if self.kind == "clf" else REG_METRICS


class _Windows:
    """Accumulates the frame windows of all tasks of a session."""

    def __init__(self) -> None:
        self.starts: list[np.ndarray] = []
        self.ends: list[np.ndarray] = []
        self.n = 0

    def add(self, starts: npt.ArrayLike, ends: npt.ArrayLike) -> np.ndarray:
        s = np.asarray(starts, dtype=np.int64)
        self.starts.append(s)
        self.ends.append(np.asarray(ends, dtype=np.int64))
        idx = np.arange(self.n, self.n + s.size)
        self.n += s.size
        return idx

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        z = np.zeros(0, dtype=np.int64)
        st = np.concatenate(self.starts) if self.starts else z
        en = np.concatenate(self.ends) if self.ends else z
        return st, en


def _skip(
    analysis: str, target: str, variant: str, condition: str, kind: str, reason: str
) -> Task:
    return Task(analysis, target, variant, condition, kind, np.zeros(0, dtype=np.int64), reason)


def _clf_task(
    analysis: str,
    target: str,
    variant: str,
    condition: str,
    starts: np.ndarray,
    ends: np.ndarray,
    labels: np.ndarray,
    strata: np.ndarray,
    windows: _Windows,
    n_frames: int,
    p: MazeMemParams,
    gap: int,
) -> Task:
    s: Samples = _make_samples(target, starts, ends, labels, strata, {})
    problem = prepare_problem(s, n_frames, p.n_blocks, gap)
    n_cls = int(max(np.unique(labels[strata == k]).size for k in np.unique(strata)))
    return Task(
        analysis,
        target,
        variant,
        condition,
        "clf",
        windows.add(starts, ends),
        n_samples=s.n,
        n_strata=s.n_strata,
        n_classes=n_cls,
        problem=problem,
    )


def splitter_tasks(
    pt: pd.DataFrame,
    cond: np.ndarray,
    maze: RoseMaze,
    windows: _Windows,
    n_frames: int,
    gap: int,
    p: MazeMemParams,
    rng: np.random.Generator,
) -> list[Task]:
    """Analysis 1 tasks: origin and destination decoding per condition.

    Strata are (cell, arrival neighbour, departure neighbour); labels with
    fewer than ``p.min_n`` samples and strata with fewer than two labels are
    dropped; light and dark are equalised per (stratum, label).
    """
    n = maze.dist.shape[0]
    tasks: list[Task] = []
    for target, col in (("origin", "origin"), ("destination", "dest")):
        if pt.empty:
            tasks += [
                _skip("splitter", target, "passthrough", c, "clf", "no pass-through visits")
                for c in p.conditions
            ]
            continue
        strata = (pt["cell"].to_numpy(np.int64) * n + pt["prev"].to_numpy(np.int64)) * n
        strata = strata + pt["next"].to_numpy(np.int64)
        labels = pt[col].to_numpy(np.int64)
        sets = condition_index_sets(strata, labels, cond, p.conditions, p.min_n, 2, rng)
        st = pt["start"].to_numpy(np.int64)
        en = pt["end"].to_numpy(np.int64)
        for c in p.conditions:
            idx = sets.get(c, np.zeros(0, dtype=np.int64))
            if idx.size == 0:
                reason = f"no stratum with >= 2 labels of >= {p.min_n} samples"
                tasks.append(_skip("splitter", target, "passthrough", c, "clf", reason))
                continue
            tasks.append(
                _clf_task(
                    "splitter",
                    target,
                    "passthrough",
                    c,
                    st[idx],
                    en[idx],
                    labels[idx],
                    strata[idx],
                    windows,
                    n_frames,
                    p,
                    gap,
                )
            )
    return tasks


def _cell_demean(v: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Subtract the per-cell mean (all samples) from each column of *v*."""
    out, _, _ = _group_demean(v, cell, v[:1], cell[:1])
    return out


def distance_tasks(
    pt: pd.DataFrame,
    visit_speed: np.ndarray,
    cond: np.ndarray,
    windows: _Windows,
    n_frames: int,
    gap: int,
    p: MazeMemParams,
    rng: np.random.Generator,
    n_cells_maze: int = 23,
) -> list[Task]:
    """Analysis 2 tasks: remaining and elapsed steps under three variants.

    - ``pooled`` and ``place_controlled`` (target, features and covariates
      demeaned per maze cell): light and dark equalised per (cell,
      remaining steps); at least ``p.min_reg_samples`` samples and three
      distinct values of each target.
    - ``place_direction_controlled``: demeaned per (cell, arrival neighbour,
      departure neighbour) stratum, so neither place nor travel direction
      through the cell (which also constrains the remaining distance) can
      contribute. Own sample set: light and dark equalised per (stratum,
      remaining steps); strata with fewer than two distinct remaining values
      dropped; at least ``p.min_reg_samples`` samples.
    """
    tasks: list[Task] = []
    targets = ("remaining", "elapsed")
    if pt.empty:
        return [
            _skip("distance", t, v, c, "reg", "no pass-through visits")
            for c in p.conditions
            for t in targets
            for v in DISTANCE_VARIANTS
        ]
    n = int(n_cells_maze)
    cell = pt["cell"].to_numpy(np.int64)
    stratum = (cell * n + pt["prev"].to_numpy(np.int64)) * n + pt["next"].to_numpy(np.int64)
    rem_i = pt["remaining"].to_numpy(np.int64)
    rem = rem_i.astype(np.float64)
    ela = pt["elapsed"].to_numpy(np.float64)
    st = pt["start"].to_numpy(np.int64)
    en = pt["end"].to_numpy(np.int64)
    spd = np.asarray(visit_speed, dtype=np.float64)
    zeros = np.zeros(cell.size)
    sets = condition_index_sets(zeros, cell * 1000 + rem_i, cond, p.conditions, 1, 1, rng)
    dsets = condition_index_sets(zeros, stratum * 1000 + rem_i, cond, p.conditions, 1, 1, rng)

    def _add(c: str, idx: np.ndarray, variants: tuple[str, ...], group: np.ndarray) -> None:
        widx = windows.add(st[idx], en[idx])
        folds = time_block_folds(st[idx], en[idx], n_frames, p.n_blocks, gap)
        g = group[idx]
        for target, y, other in (("remaining", rem, ela), ("elapsed", ela, rem)):
            covs = np.column_stack([other[idx], spd[idx]])
            for variant in variants:
                grouped = variant != "pooled"
                tasks.append(
                    Task(
                        "distance",
                        target,
                        variant,
                        c,
                        "reg",
                        widx,
                        n_samples=int(idx.size),
                        n_strata=int(np.unique(g).size) if grouped else 1,
                        n_classes=int(np.unique(y[idx]).size),
                        y=y[idx],
                        folds=folds,
                        groups=g if grouped else None,
                        covs=_cell_demean(covs, g) if grouped else covs,
                    )
                )

    for c in p.conditions:
        idx = sets.get(c, np.zeros(0, dtype=np.int64))
        reason = ""
        if idx.size < p.min_reg_samples:
            reason = f"fewer than {p.min_reg_samples} samples"
        elif np.unique(rem[idx]).size < 3 or np.unique(ela[idx]).size < 3:
            reason = "fewer than 3 distinct step counts"
        if reason:
            tasks += [
                _skip("distance", t, v, c, "reg", reason)
                for t in targets
                for v in DISTANCE_VARIANTS[:2]
            ]
        else:
            _add(c, idx, DISTANCE_VARIANTS[:2], cell)
        didx = dsets.get(c, np.zeros(0, dtype=np.int64))
        didx = didx[multi_value_groups(stratum[didx], rem_i[didx])]
        if didx.size < p.min_reg_samples:
            reason = f"fewer than {p.min_reg_samples} samples in strata with >= 2 remaining values"
            tasks += [
                _skip("distance", t, DISTANCE_VARIANTS[2], c, "reg", reason) for t in targets
            ]
        else:
            _add(c, didx, DISTANCE_VARIANTS[2:], stratum)
    return tasks


def multi_value_groups(groups: npt.ArrayLike, values: npt.ArrayLike) -> np.ndarray:
    """Boolean mask of samples whose group has at least two distinct values."""
    g = np.asarray(groups, dtype=np.int64)
    v = np.asarray(values, dtype=np.int64)
    if g.size == 0:
        return np.zeros(0, dtype=bool)
    pairs = np.unique(np.column_stack([g, v]), axis=0)
    ug, cnt = np.unique(pairs[:, 0], return_counts=True)
    return np.isin(g, ug[cnt >= 2])


def departure_table(
    visits: pd.DataFrame,
    trips: pd.DataFrame,
    novelty: pd.DataFrame,
    light_on: npt.ArrayLike,
    fps: float,
    n_frames: int,
    p: MazeMemParams,
    maze: RoseMaze | None = None,
) -> pd.DataFrame:
    """Analysis 3 samples: dead-end departures with dwell >= ``p.min_dead_end_dwell_s``.

    Returns
    -------
    pandas.DataFrame
        Columns ``exit`` (first frame after the dead-end visit), ``dwell_s``,
        ``trip_end``, ``novel``, ``lru``, ``origin``, ``dest``, ``trip_dist``
        (graph distance origin to destination), and per window ``w`` in
        (``pre``, ``post``): ``{w}_start``, ``{w}_end``, ``{w}_ok`` (window
        inside the session) and ``{w}_cond`` (light condition over the window
        and the following trip).
    """
    cols = ["exit", "dwell_s", "trip_end", "novel", "lru", "origin", "dest", "trip_dist"]
    if trips.empty:
        return pd.DataFrame({k: np.zeros(0) for k in cols})
    maze = maze or build_rose_maze()
    org = trips["origin"].to_numpy(np.int64)
    dst = trips["dest"].to_numpy(np.int64)
    w = max(1, int(round(p.plan_window_s * fps)))
    io = trips["i_origin"].to_numpy(np.int64)
    entry = visits["start"].to_numpy(np.int64)[io]
    ex = trips["start"].to_numpy(np.int64)
    tend = trips["end"].to_numpy(np.int64)
    dwell = (ex - entry) / fps
    keep = dwell >= p.min_dead_end_dwell_s
    df = pd.DataFrame(
        {
            "exit": ex,
            "dwell_s": dwell,
            "trip_end": tend,
            "novel": novelty["novel"].to_numpy(bool),
            "lru": novelty["lru"].to_numpy(bool),
            "origin": org,
            "dest": dst,
            "trip_dist": maze.dist[org, dst].astype(np.int64),
            "pre_start": ex - w,
            "pre_end": ex,
            "post_start": ex,
            "post_end": ex + w,
        }
    )[keep].reset_index(drop=True)
    for name in ("pre", "post"):
        a = df[f"{name}_start"].to_numpy(np.int64)
        b = np.maximum(df[f"{name}_end"].to_numpy(np.int64), df["trip_end"].to_numpy(np.int64))
        df[f"{name}_ok"] = (a >= 0) & (df[f"{name}_end"].to_numpy(np.int64) <= n_frames)
        df[f"{name}_cond"] = light_condition(light_on, a, b, p.light_hi, p.light_lo)
    return df


def plan_filter(dep: pd.DataFrame, name: str, long_trip_steps: int = 3) -> np.ndarray:
    """Departures kept by a planning variant filter.

    ``""``: all; ``"_noreturn"``: destination differs from origin (removes
    short bounce-back trips); ``"_noreturn_long"``: additionally a graph
    distance of at least *long_trip_steps* from origin to destination.
    """
    keep = np.ones(len(dep), dtype=bool)
    if name in ("_noreturn", "_noreturn_long"):
        keep &= dep["dest"].to_numpy(np.int64) != dep["origin"].to_numpy(np.int64)
    if name == "_noreturn_long":
        keep &= dep["trip_dist"].to_numpy(np.int64) >= long_trip_steps
    return keep


def planning_tasks(
    dep: pd.DataFrame,
    windows: _Windows,
    n_frames: int,
    gap: int,
    p: MazeMemParams,
    rng: np.random.Generator,
    extra_feature_sets: tuple[str, ...] = (),
) -> list[Task]:
    """Analysis 3 tasks: novel / least-recent destination per window x trip filter.

    Variants are ``{window}{filter}`` for windows :data:`PLAN_WINDOWS` and
    filters :data:`PLAN_FILTERS` (:func:`plan_filter`). *extra_feature_sets*
    (e.g. ``behaviour_plus_syllable``) are attached to every task run.
    """
    tasks: list[Task] = []
    for window in PLAN_WINDOWS:
        for filt in PLAN_FILTERS:
            variant = f"{window}{filt}"
            for target in ("novel", "lru"):
                if dep.empty:
                    tasks += [
                        _skip("planning", target, variant, c, "clf", "no dead-end departures")
                        for c in p.conditions
                    ]
                    continue
                ok = dep[f"{window}_ok"].to_numpy(bool) & plan_filter(dep, filt, p.long_trip_steps)
                cond = np.where(ok, dep[f"{window}_cond"].to_numpy(object), "mixed")
                lab = dep[target].to_numpy(bool).astype(np.int64)
                sets = condition_index_sets(
                    np.zeros(lab.size), lab, cond, p.conditions, p.min_class_samples, 2, rng
                )
                st = dep[f"{window}_start"].to_numpy(np.int64)
                en = dep[f"{window}_end"].to_numpy(np.int64)
                for c in p.conditions:
                    idx = sets.get(c, np.zeros(0, dtype=np.int64))
                    if idx.size == 0:
                        reason = f"fewer than {p.min_class_samples} departures per class"
                        tasks.append(_skip("planning", target, variant, c, "clf", reason))
                        continue
                    t = _clf_task(
                        "planning",
                        target,
                        variant,
                        c,
                        st[idx],
                        en[idx],
                        lab[idx],
                        np.zeros(idx.size, dtype=np.int64),
                        windows,
                        n_frames,
                        p,
                        gap,
                    )
                    t.extra_feature_sets = tuple(extra_feature_sets)
                    tasks.append(t)
    return tasks


def syllable_channels(
    syllable_id: npt.ArrayLike | None, valid: npt.ArrayLike, top_k: int = 8
) -> tuple[np.ndarray | None, np.ndarray]:
    """Indicator channels of the *top_k* most frequent syllables (valid frames).

    Window means of these channels are the fraction of the window spent in
    each syllable. Frames that are invalid or have no syllable (non-finite
    or negative id) are NaN.

    Returns
    -------
    channels : (k, n_frames) float or None
        None when no syllable labels are available.
    ids : (k,) int
        Syllable ids, most frequent first.
    """
    empty = np.zeros(0, dtype=np.int64)
    if syllable_id is None or top_k < 1:
        return None, empty
    s = np.asarray(syllable_id, dtype=np.float64).ravel()
    v = np.asarray(valid, dtype=bool).ravel()[: s.size]
    ok = np.isfinite(s) & (s >= 0) & v
    if not ok.any():
        return None, empty
    ids, cnt = np.unique(s[ok].astype(np.int64), return_counts=True)
    top = ids[np.argsort(-cnt, kind="stable")[:top_k]]
    ch = (s[None, :] == top[:, None]).astype(np.float64)
    ch[:, ~ok] = np.nan
    return ch, top


# ---------------------------------------------------------------------------
# Scoring over shifts
# ---------------------------------------------------------------------------


@dataclass
class SessionData:
    """Predictors and the global window list shared by all tasks of a session."""

    neural: WindowMeans
    behaviour: WindowMeans
    starts: np.ndarray
    ends: np.ndarray
    hd0: np.ndarray
    params: MazeMemParams
    syllable: WindowMeans | None = None


def score_task(
    task: Task, xs: np.ndarray, cov: np.ndarray | None, p: MazeMemParams
) -> dict[str, np.ndarray]:
    """Cross-validated metrics of one task for a stack of (finite) feature matrices.

    Returns metric -> ``(S,)`` array. Classification: balanced accuracy per
    stratum averaged weighted by stratum size. Regression: Spearman and
    partial Spearman (:func:`partial_spearman`) of the out-of-fold
    prediction with the target.
    """
    if task.kind == "clf":
        assert task.problem is not None
        y = task.problem.samples.labels
        accs, weights = [], []
        for idx, folds in task.problem.strata:
            c = None if cov is None else cov[idx]
            pred = batched_logistic_predict(xs[:, idx], y[idx], folds, p.C, c, p.max_iter)
            accs.append(balanced_accuracy_batched(y[idx], pred))
            weights.append(idx.size)
        return {"balanced_accuracy": np.average(np.vstack(accs), axis=0, weights=weights)}
    assert task.y is not None
    pred, yev = batched_ridge_predict(xs, task.y, task.folds, p.ridge_alpha, cov, task.groups)
    return {
        "spearman": np.array([partial_spearman(r, yev, None) for r in pred]),
        "partial_spearman": np.array([partial_spearman(r, yev, task.covs) for r in pred]),
    }


def _score_shifts(
    tasks: list[Task],
    data: SessionData,
    shifts: np.ndarray,
    feature_sets: tuple[str, ...],
    cols: np.ndarray | None,
) -> dict[tuple[int, str, str], np.ndarray]:
    """Metrics for every task x feature set at each shift (chunks of ``shift_chunk``)."""
    p = data.params
    out: dict[tuple[int, str, str], np.ndarray] = {}
    need_n = any(fs != "behaviour" for fs in feature_sets)
    chunks = [shifts[i : i + p.shift_chunk] for i in range(0, shifts.size, p.shift_chunk)]
    for i, t in enumerate(tasks):
        if t.reason:
            continue
        st, en = data.starts[t.widx], data.ends[t.widx]
        hd = impute_columns(data.hd0[t.widx])
        parts: dict[tuple[str, str], list[np.ndarray]] = {}
        extra = tuple(
            f
            for f in t.extra_feature_sets
            if "behaviour" in feature_sets and f not in feature_sets and data.syllable is not None
        )
        for ch in chunks:
            xn = xb = None
            if need_n:
                xn = np.stack([data.neural.means(st, en, int(s)) for s in ch])
                xn = _impute_batched(xn if cols is None else xn[:, :, cols])
            if "behaviour" in feature_sets:
                xb = np.stack(
                    [behaviour_window_features(data.behaviour, st, en, int(s)) for s in ch]
                )
                xb = _impute_batched(xb)
            for fs in feature_sets + extra:
                if fs == SYLLABLE_FEATURE_SET:
                    assert data.syllable is not None and xb is not None
                    xsy = np.stack([data.syllable.means(st, en, int(s)) for s in ch])
                    res = score_task(t, np.concatenate([xb, _impute_batched(xsy)], 2), None, p)
                elif fs == "behaviour":
                    res = score_task(t, xb, None, p)  # type: ignore[arg-type]
                elif fs == "neural":
                    res = score_task(t, xn, None, p)  # type: ignore[arg-type]
                else:
                    res = score_task(t, xn, hd, p)  # type: ignore[arg-type]
                for m in t.metrics:
                    parts.setdefault((fs, m), []).append(res[m])
        for (fs, m), v in parts.items():
            out[(i, fs, m)] = np.concatenate(v)
    return out


def evaluate_tasks(
    tasks: list[Task],
    data: SessionData,
    shifts: np.ndarray,
    feature_sets: tuple[str, ...] = FEATURE_SETS,
    cols: np.ndarray | None = None,
    n_jobs: int = 1,
) -> dict[tuple[int, str, str], np.ndarray]:
    """Metrics at shift 0 followed by every null shift (``shifts[0]`` must be 0).

    With ``n_jobs > 1`` the shifts are split into chunks scored in parallel
    (joblib, installed with scikit-learn).
    """
    sh = np.asarray(shifts, dtype=np.int64)
    if n_jobs <= 1 or sh.size < 2 * n_jobs:
        return _score_shifts(tasks, data, sh, feature_sets, cols)
    from joblib import Parallel, delayed

    parts = Parallel(n_jobs=n_jobs)(
        delayed(_score_shifts)(tasks, data, c, feature_sets, cols)
        for c in np.array_split(sh, n_jobs)
    )
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}


def diff_null_summary(observed: float, null: npt.ArrayLike) -> dict[str, Any]:
    """Like :func:`population_maze.null_summary` but two-sided, for a difference.

    ``p = (1 + #{|null - mean| >= |observed - mean|}) / (1 + n)``.
    """
    res = null_summary(observed, null)
    nk = np.asarray(null, dtype=np.float64)
    nk = nk[np.isfinite(nk)]
    if np.isfinite(observed) and nk.size:
        mu = nk.mean()
        res["p"] = float((1 + np.sum(np.abs(nk - mu) >= abs(observed - mu))) / (1 + nk.size))
    return res


# ---------------------------------------------------------------------------
# Session entry point
# ---------------------------------------------------------------------------


@dataclass
class MazeMemResult:
    """Outputs of :func:`maze_memory_session`."""

    sessions: pd.DataFrame
    behaviour: pd.DataFrame
    subsets: pd.DataFrame
    counts: dict[str, Any]


def _task_rows(
    tasks: list[Task],
    scores: dict[tuple[int, str, str], np.ndarray],
    feature_sets: tuple[str, ...],
    n_cells: int,
    short: bool,
    n_syllables: int = 0,
) -> list[dict[str, Any]]:
    n_feat = {"behaviour": len(BEHAVIOUR_CHANNELS)}
    n_feat[SYLLABLE_FEATURE_SET] = len(BEHAVIOUR_CHANNELS) + n_syllables
    rows = []
    for i, t in enumerate(tasks):
        extra = tuple(f for f in t.extra_feature_sets if n_syllables and f not in feature_sets)
        for fs in feature_sets + extra:
            for m in t.metrics:
                arr = scores.get((i, fs, m))
                obs = float("nan") if arr is None else float(arr[0])
                ns = null_summary(obs, np.zeros(0) if arr is None else arr[1:])
                reason = t.reason or ("session too short for shift null" if short else "")
                rows.append(
                    {
                        "analysis": t.analysis,
                        "target": t.target,
                        "variant": t.variant,
                        "condition": t.condition,
                        "feature_set": fs,
                        "metric": m,
                        "observed": obs,
                        **ns,
                        "n_samples": t.n_samples,
                        "n_strata": t.n_strata,
                        "n_classes": t.n_classes,
                        "n_cells": n_cells,
                        "n_features": n_feat.get(fs, n_cells),
                        "reason": reason,
                    }
                )
    return rows


def _pro_minus_retro_rows(
    tasks: list[Task],
    scores: dict[tuple[int, str, str], np.ndarray],
    feature_sets: tuple[str, ...],
    n_cells: int,
) -> list[dict[str, Any]]:
    """Destination minus origin balanced accuracy per condition x feature set."""
    find = {(t.target, t.condition): i for i, t in enumerate(tasks) if t.analysis == "splitter"}
    rows = []
    conds = sorted({t.condition for t in tasks if t.analysis == "splitter"})
    for c in conds:
        io, idd = find.get(("origin", c)), find.get(("destination", c))
        for fs in feature_sets:
            a = None if io is None else scores.get((io, fs, "balanced_accuracy"))
            b = None if idd is None else scores.get((idd, fs, "balanced_accuracy"))
            reason = ""
            if a is None or b is None:
                reason = "origin or destination not decoded"
                diff = np.full(1, np.nan)
            else:
                diff = b - a
            ns = diff_null_summary(float(diff[0]), diff[1:])
            rows.append(
                {
                    "analysis": "splitter",
                    "target": PRO_MINUS_RETRO,
                    "variant": "passthrough",
                    "condition": c,
                    "feature_set": fs,
                    "metric": "balanced_accuracy_diff",
                    "observed": float(diff[0]),
                    **ns,
                    "n_samples": 0,
                    "n_strata": 0,
                    "n_classes": 0,
                    "n_cells": n_cells,
                    "n_features": len(BEHAVIOUR_CHANNELS) if fs == "behaviour" else n_cells,
                    "reason": reason,
                }
            )
    return rows


def _subset_rows(
    tasks: list[Task],
    data: SessionData,
    shifts: np.ndarray,
    n_cells: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Neural scores with ``subset_size`` random cells (makes sessions comparable)."""
    p = data.params
    k = int(p.subset_size)
    if not 0 < k < n_cells or p.n_subsets < 1 or not any(not t.reason for t in tasks):
        return []
    sub_shifts = shifts[: p.n_subset_shuffles + 1]
    obs: dict[tuple[int, str], list[float]] = {}
    nul: dict[tuple[int, str], list[float]] = {}
    for _ in range(p.n_subsets):
        cols = np.sort(rng.choice(n_cells, k, replace=False))
        sc = evaluate_tasks(tasks, data, sub_shifts, ("neural",), cols, p.n_jobs)
        for (i, _fs, m), arr in sc.items():
            obs.setdefault((i, m), []).append(float(arr[0]))
            nul.setdefault((i, m), []).extend(arr[1:].tolist())
    rows = []
    for (i, m), o in obs.items():
        t = tasks[i]
        oa, na = np.asarray(o), np.asarray(nul[(i, m)])
        om = float(np.nanmean(oa)) if np.isfinite(oa).any() else float("nan")
        nm = float(np.nanmean(na)) if np.isfinite(na).any() else float("nan")
        rows.append(
            {
                "analysis": t.analysis,
                "target": t.target,
                "variant": t.variant,
                "condition": t.condition,
                "metric": m,
                "n_cells_subset": k,
                "n_draws": p.n_subsets,
                "observed_mean": om,
                "null_mean": nm,
                "excess": om - nm,
                "n_null": int(np.isfinite(na).sum()),
            }
        )
    return rows


def behaviour_rows(
    trips: pd.DataFrame,
    novelty: pd.DataFrame,
    trip_cond: np.ndarray,
    maze: RoseMaze,
    fps: float,
    p: MazeMemParams,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Per condition: destination novelty vs random walk and trip descriptives."""
    rows = []
    for c in p.conditions:
        sel = np.isin(trip_cond, ["light", "dark"]) if c == "all" else trip_cond == c
        tr = trips[sel]
        nv = novelty[sel]
        steps = tr["n_steps"].to_numpy(np.float64)
        org, dst = tr["origin"].to_numpy(np.int64), tr["dest"].to_numpy(np.int64)
        direct = (steps == maze.dist[org, dst]) & (org != dst)
        dur = (tr["end"].to_numpy(np.float64) - tr["start"].to_numpy(np.float64)) / fps
        for label, model in ((lab, m) for lab in ("novel", "lru") for m in RW_MODELS):
            res = novelty_vs_random_walk(
                nv[label].to_numpy(bool), nv[f"p_{label}_{model}"].to_numpy(), p.n_rw_sims, rng
            )
            rows.append(
                {
                    "condition": c,
                    "label": label,
                    "null_model": model,
                    **res,
                    "median_trip_steps": float(np.median(steps)) if steps.size else np.nan,
                    "frac_direct": float(direct.mean()) if steps.size else np.nan,
                    "frac_return_same": float((org == dst).mean()) if steps.size else np.nan,
                    "median_trip_s": float(np.median(dur)) if steps.size else np.nan,
                    "recent_k": p.recent_k,
                    "reason": "" if steps.size else "no trips",
                }
            )
    return pd.DataFrame(rows)


def maze_memory_session(
    signal: npt.ArrayLike,
    x_maze: npt.ArrayLike,
    y_maze: npt.ArrayLike,
    speed: npt.ArrayLike,
    hd_deg: npt.ArrayLike,
    ahv: npt.ArrayLike,
    light_on: npt.ArrayLike,
    valid: npt.ArrayLike,
    fps: float,
    params: MazeMemParams | None = None,
    maze: RoseMaze | None = None,
    syllable_id: npt.ArrayLike | None = None,
) -> MazeMemResult:
    """All three analyses for one session, every condition and feature set, with shift nulls.

    Parameters
    ----------
    signal : (n_cells, n_frames) float
        Pooled soma activity (dF/F or inferred spike rate).
    x_maze, y_maze : (n_frames,) float
        Position in maze units.
    speed : (n_frames,) float
        Speed, cm/s.
    hd_deg : (n_frames,) float
        Head direction, degrees.
    ahv : (n_frames,) float
        Angular head velocity, deg/s.
    light_on : (n_frames,) bool
    valid : (n_frames,) bool
        False on behavioural-artefact frames.
    fps : float
    params : MazeMemParams or None
    maze : RoseMaze or None
    syllable_id : (n_frames,) int or None
        Behavioural syllable per frame (keypoint-MoSeq). When given, planning
        tasks also get the ``behaviour_plus_syllable`` feature set: the
        behaviour channels plus the fraction of the window spent in each of
        the ``params.n_top_syllables`` most frequent syllables.

    Returns
    -------
    MazeMemResult
        ``sessions``: one row per analysis x target x variant x condition x
        feature set x metric (observed, null mean/sd, z, p, excess, counts,
        skip reason); ``behaviour``: destination novelty vs random walk per
        condition; ``subsets``: neural scores with ``subset_size`` cells;
        ``counts``: sample counts.

    References
    ----------
    See the module docstring.
    """
    p = params or MazeMemParams()
    maze = maze or build_rose_maze()
    rng = np.random.default_rng(p.seed)
    sig = np.array(np.atleast_2d(np.asarray(signal, dtype=np.float64)), copy=True)
    n_cells, n_frames = sig.shape
    v = np.asarray(valid, dtype=bool).ravel()[:n_frames]
    lit = np.asarray(light_on, dtype=bool).ravel()[:n_frames]
    sig[:, ~v] = np.nan

    cells = session_cells(x_maze, y_maze, v, maze, p.min_dwell_frames)
    visits = visit_table(cells, maze, int(round(p.max_gap_s * fps)), p.max_jump_steps)
    trips = trip_table(visits, maze)
    trip_cond = light_condition(lit, trips["start"], trips["end"], p.light_hi, p.light_lo)
    novelty = trip_novelty(visits, trips, maze, p.recent_k)
    pt = passthrough_samples(visits, trips, maze)
    pt_cond = trip_cond[pt["trip"].to_numpy(np.int64)] if len(pt) else np.zeros(0, dtype=object)

    beh = behaviour_channels(hd_deg, speed, ahv, visits, maze, fps, v)
    bwm = WindowMeans(beh)
    visit_speed = bwm.means(pt["start"], pt["end"])[:, 2] if len(pt) else np.zeros(0)
    dep = departure_table(visits, trips, novelty, lit, fps, n_frames, p, maze)
    syl_ch, syl_ids = syllable_channels(
        None if syllable_id is None else np.asarray(syllable_id).ravel()[:n_frames],
        v,
        p.n_top_syllables,
    )
    extra = (SYLLABLE_FEATURE_SET,) if syl_ch is not None else ()

    gap = int(round(p.gap_s * fps))
    windows = _Windows()
    tasks = splitter_tasks(pt, pt_cond, maze, windows, n_frames, gap, p, rng)
    tasks += distance_tasks(
        pt, visit_speed, pt_cond, windows, n_frames, gap, p, rng, maze.dist.shape[0]
    )
    tasks += planning_tasks(dep, windows, n_frames, gap, p, rng, extra)
    starts, ends = windows.arrays()
    data = SessionData(
        WindowMeans(sig),
        bwm,
        starts,
        ends,
        behaviour_window_features(bwm, starts, ends, 0)[:, :2],
        p,
        None if syl_ch is None else WindowMeans(syl_ch),
    )
    shifts = draw_shifts(n_frames, p.n_shuffles, int(round(p.min_shift_s * fps)), rng)
    scores = evaluate_tasks(tasks, data, shifts, FEATURE_SETS, None, p.n_jobs)
    short = shifts.size < 2
    rows = _task_rows(tasks, scores, FEATURE_SETS, n_cells, short, int(syl_ids.size))
    rows += _pro_minus_retro_rows(tasks, scores, FEATURE_SETS, n_cells)
    subsets = pd.DataFrame(_subset_rows(tasks, data, shifts, n_cells, rng))
    behaviour = behaviour_rows(trips, novelty, trip_cond, maze, fps, p, rng)

    counts: dict[str, Any] = {
        "n_frames": int(n_frames),
        "n_valid_frames": int(v.sum()),
        "n_cells": int(n_cells),
        "n_visits": int((~visits["interp"]).sum()) if len(visits) else 0,
        "n_filled_visits": int(visits["interp"].sum()) if len(visits) else 0,
        "n_segments": int(visits["seg"].nunique()) if len(visits) else 0,
        "n_trips": int(len(trips)),
        "n_passthrough": int(len(pt)),
        "n_departures": int(len(dep)),
        "n_shifts": int(shifts.size - 1),
    }
    for c in ("light", "dark", "mixed"):
        counts[f"n_trips_{c}"] = int(np.sum(trip_cond == c))
        counts[f"n_passthrough_{c}"] = int(np.sum(pt_cond == c))
    counts["n_top_syllables"] = int(syl_ids.size)
    counts["top_syllables"] = " ".join(str(int(x)) for x in syl_ids)
    log.debug("maze memory: %d cells, %d frames, %d tasks", n_cells, n_frames, len(tasks))
    return MazeMemResult(pd.DataFrame(rows), behaviour, subsets, counts)


# ---------------------------------------------------------------------------
# Across-session statistics
# ---------------------------------------------------------------------------


def behaviour_as_session_rows(behaviour: pd.DataFrame) -> pd.DataFrame:
    """Behaviour rows in the :data:`SESSION_KEYS` schema (expected = random walk)."""
    if behaviour.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "analysis": "novelty_behaviour",
            "target": behaviour["label"],
            "variant": "trips",
            "condition": behaviour["condition"],
            "feature_set": "observed_vs_" + behaviour["null_model"].astype(str) + "_walk",
            "metric": "destination_frac",
            "observed": behaviour["observed_frac"],
            "null_mean": behaviour["expected_frac"],
            "z": behaviour["z"],
            "p": behaviour["p"],
            "excess": behaviour["excess"],
            "n_samples": behaviour["n_trips"],
            "reason": behaviour["reason"],
        }
    )
    for c in ("exp_id", "animal_id", "celltype", "signal"):
        if c in behaviour:
            out[c] = behaviour[c].to_numpy()
    return out


def across_session_summary(
    sessions: pd.DataFrame, keys: list[str] | None = None, alpha: float = 0.05
) -> pd.DataFrame:
    """Excess over the null across sessions and across animal medians.

    One row per key combination x group, where group is ``all`` (pooled
    cells, the main result) or a cell type (descriptive only). Two-sided
    Wilcoxon signed-rank tests of ``excess`` against 0 on sessions and on
    per-animal medians (sessions of one animal are not independent).
    """
    keys = keys or SESSION_KEYS
    rows: list[dict[str, Any]] = []
    if sessions.empty:
        return pd.DataFrame(rows)
    for kv, g in sessions.groupby(keys, sort=False, dropna=False):
        groups = [("all", g)]
        if "celltype" in g:
            groups += [(str(ct), gg) for ct, gg in g.groupby("celltype")]
        for name, gg in groups:
            ok = gg[np.isfinite(gg["excess"].to_numpy(np.float64))]
            animal = ok.groupby("animal_id")["excess"].median()
            pv = ok["p"].to_numpy(np.float64)
            rows.append(
                {
                    **dict(zip(keys, kv, strict=True)),
                    "group": name,
                    "descriptive": name != "all",
                    "n_sessions": int(len(ok)),
                    "n_sessions_skipped": int(len(gg) - len(ok)),
                    "median_observed": float(ok["observed"].median()) if len(ok) else np.nan,
                    "median_null_mean": float(ok["null_mean"].median()) if len(ok) else np.nan,
                    "median_excess": float(ok["excess"].median()) if len(ok) else np.nan,
                    "n_sessions_sig": int((pv < alpha).sum()),
                    "frac_sessions_sig": float((pv < alpha).mean()) if pv.size else np.nan,
                    "session_wilcoxon_p": _wilcoxon(ok["excess"]),
                    "n_animals": int(animal.size),
                    "median_animal_excess": float(animal.median()) if animal.size else np.nan,
                    "animal_wilcoxon_p": _wilcoxon(animal),
                }
            )
    return pd.DataFrame(rows)


def _paired(
    df: pd.DataFrame, pivot: str, pairs: tuple[tuple[str, str], ...], kind: str
) -> list[dict[str, Any]]:
    keys = [k for k in SESSION_KEYS if k != pivot]
    rows: list[dict[str, Any]] = []
    for kv, g in df.groupby(keys, sort=False, dropna=False):
        wide = g.pivot_table(
            index=["exp_id", "animal_id"], columns=pivot, values="excess", aggfunc="first"
        )
        for a, b in pairs:
            if a not in wide or b not in wide:
                continue
            diff = (wide[a] - wide[b]).dropna()
            if diff.empty:
                continue
            animal = diff.groupby(level="animal_id").median()
            rows.append(
                {
                    "comparison_type": kind,
                    **dict(zip(keys, kv, strict=True)),
                    pivot: f"{a} - {b}",
                    "n_sessions": int(diff.size),
                    "median_diff": float(diff.median()),
                    "n_sessions_first_better": int((diff > 0).sum()),
                    "session_wilcoxon_p": _wilcoxon(diff),
                    "n_animals": int(animal.size),
                    "median_animal_diff": float(animal.median()),
                    "animal_wilcoxon_p": _wilcoxon(animal),
                }
            )
    return rows


def paired_comparisons(sessions: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon of per-session excess differences.

    - ``feature_set``: neural vs behaviour, neural with HD removed vs
      behaviour, neural vs neural with HD removed (same condition);
    - ``light_vs_dark``: light minus dark excess (same feature set);
    - ``prospective_vs_retrospective``: destination minus origin excess.

    Each difference is tested against 0 across sessions and across
    per-animal medians (two-sided Wilcoxon signed-rank).
    """
    if sessions.empty:
        return pd.DataFrame()
    df = sessions[sessions["target"] != PRO_MINUS_RETRO]
    rows = _paired(df, "feature_set", FEATURE_COMPARISONS, "feature_set")
    rows += _paired(df, "condition", (("light", "dark"),), "light_vs_dark")
    split = df[(df["analysis"] == "splitter")]
    rows += _paired(split, "target", (("destination", "origin"),), "prospective_vs_retrospective")
    return pd.DataFrame(rows)


def subset_summary(subsets: pd.DataFrame) -> pd.DataFrame:
    """Median observed, null and excess across sessions for the cell-subset scores."""
    if subsets.empty:
        return pd.DataFrame()
    keys = ["analysis", "target", "variant", "condition", "metric", "n_cells_subset"]
    return (
        subsets.groupby(keys, dropna=False)
        .agg(
            n_sessions=("excess", lambda v: int(np.isfinite(v).sum())),
            median_observed=("observed_mean", "median"),
            median_null=("null_mean", "median"),
            median_excess=("excess", "median"),
            session_wilcoxon_p=("excess", _wilcoxon),
        )
        .reset_index()
    )
