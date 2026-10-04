"""Pooled-population decoding of maze behaviour from RSP soma activity.

Question: with all soma ROIs of a session pooled (cell type ignored), does
population activity carry information about maze behaviour beyond running?
Four decoders are fitted per session, each with a circular-shift null and two
confound controls.

Samples and labels
------------------
Positions are discretised onto the 23-cell maze graph and cell occupancies
shorter than ``min_dwell_frames`` are dropped
(:func:`hm2p.analysis.maze_running.dwell_filtered_cells`); frames flagged as
behavioural artefacts are unassigned first. Running bouts follow the runshape
convention (:func:`hm2p.analysis.running_coding.run_bouts`).

1. ``inout`` -- runs into dead ends vs runs out of dead ends
   (:func:`hm2p.analysis.maze_running.annotate_bouts`). Classes are balanced and
   matched on run duration and mean speed by stratified subsampling within
   duration x speed quantile bins (:func:`matched_subsample`). Sample window =
   the run.
2. ``position`` -- contiguous stays in one maze cell during running bouts of
   at least ``min_visit_frames``; class = maze cell; cells with fewer than
   ``min_class_samples`` visits are dropped. Metrics: balanced accuracy and the
   class-balanced mean graph distance between decoded and true cell
   (``mean_distance``, the headline metric, lower is better), because
   confusing neighbouring cells is not equivalent to confusing distant ones.
3. ``edge_direction`` -- traversals of an undirected maze edge (two
   consecutive running visits to adjacent cells in one bout); label = travel
   direction; window = both visits. Edges with at least ``min_traversals``
   traversals in each direction are decoded separately and balanced accuracy
   is averaged over edges weighted by samples.
4. ``junction_exit`` -- T-junction visits (previous and next cell adjacent to
   the junction, U-turns excluded); label = exit arm; stratum = (junction,
   arrival arm) so the choice is binary within a stratum; window = the
   ``junction_window_s`` before leaving the junction (it extends into the
   arrival arm when the junction stay is shorter). Strata with at least
   ``min_per_choice`` of each choice are decoded separately and pooled as for
   edges.

Decoder
-------
Features are per-cell mean activity over each sample's window (non-finite
frames excluded), z-scored with training-fold statistics. The classifier is
L2-regularised logistic regression with balanced class weights (scikit-learn,
Pedregosa et al. 2011). Cross-validation is grouped by ``n_blocks`` contiguous
time blocks of the session; training samples within ``gap_s`` of the test
block are dropped so train and test are not adjacent in time (blocked CV,
Roberts et al. 2017). Out-of-fold predictions are scored with balanced
accuracy (mean per-class recall, Brodersen et al. 2010). Stratified decoders
(edge direction, junction exit) run the CV within each stratum.

Feature sets
------------
- ``neural``: per-cell mean activity.
- ``behaviour``: behaviour only -- sine and cosine of the circular mean head
  direction in the window, mean speed and mean |AHV|. This shows how much of
  each label is decodable from heading and locomotion alone. Head direction
  is expected to be strongly informative for ``inout``, ``edge_direction``
  and ``junction_exit`` (the maze arms have fixed orientations), so neural
  decoding of those labels is only interpretable relative to this baseline.
- ``neural_hd_removed``: per-cell activity with a linear regression on the
  sine and cosine of window head direction removed; the regression is fitted
  on the training fold only and applied to train and test.

Null
----
Circular shift of the predictors relative to the behaviour labels by a random
offset of at least ``min_shift_s`` from zero (neural matrix for ``neural`` and
``neural_hd_removed`` -- head direction regressors stay aligned with the
labels -- and the behaviour channels for ``behaviour``). Features and the full
CV pipeline are recomputed for each shift; the same shifts are used for all
decoders and feature sets. ``p = (1 + #{null at least as good}) / (1 + n)``,
one-sided in the direction of better decoding; ``z = (observed - null
mean) / null sd``; ``excess`` = observed minus null mean, sign-flipped for
``mean_distance`` so that positive always means better than the null.

The existing junction decoder in :mod:`hm2p.maze.neural` compares to a fixed
1 / n_classes chance level without a null; it is not used here.

References
----------
Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth show
rapid learning, sudden insight, and efficient exploration." eLife 10:e66175.
doi:10.7554/eLife.66175 (maze graph, dead ends)

Alexander AS, Nitz DA. 2015. "Retrosplenial cortex maps the conjunction of
internal and external spaces." Nature Neuroscience 18:1143-1151.
doi:10.1038/nn.4058 (route- and position-dependent RSC population activity)

Brodersen KH, Ong CS, Stephan KE, Buhmann JM. 2010. "The balanced accuracy
and its posterior distribution." Proceedings of the 20th International
Conference on Pattern Recognition, 3121-3124. doi:10.1109/ICPR.2010.764

Roberts DR, Bahn V, Ciuti S, et al. 2017. "Cross-validation strategies for
data with temporal, spatial, hierarchical, or phylogenetic structure."
Ecography 40:913-929. doi:10.1111/ecog.02881

Pedregosa F, Varoquaux G, Gramfort A, et al. 2011. "Scikit-learn: Machine
Learning in Python." Journal of Machine Learning Research 12:2825-2830.
https://github.com/scikit-learn/scikit-learn
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from hm2p.analysis.maze_running import annotate_bouts, dwell_filtered_cells
from hm2p.analysis.running_coding import run_bouts
from hm2p.maze.discretize import discretize_position_fast
from hm2p.maze.topology import RoseMaze, build_rose_maze

log = logging.getLogger(__name__)

DECODERS = ("inout", "position", "edge_direction", "junction_exit")
FEATURE_SETS = ("neural", "behaviour", "neural_hd_removed")
METRICS: dict[str, tuple[str, ...]] = {
    "inout": ("balanced_accuracy",),
    "position": ("balanced_accuracy", "mean_distance"),
    "edge_direction": ("balanced_accuracy",),
    "junction_exit": ("balanced_accuracy",),
}
LOWER_IS_BETTER = frozenset({"mean_distance"})
BEHAVIOUR_CHANNELS = ("hd_sin", "hd_cos", "speed", "abs_ahv")
HEADLINE_METRIC = {
    "inout": "balanced_accuracy",
    "position": "mean_distance",
    "edge_direction": "balanced_accuracy",
    "junction_exit": "balanced_accuracy",
}
COMPARISONS = (
    ("neural", "behaviour"),
    ("neural_hd_removed", "behaviour"),
    ("neural", "neural_hd_removed"),
)


@dataclass
class PopDecParams:
    """Parameters of the population maze decoding (see module docstring)."""

    n_shuffles: int = 200
    min_shift_s: float = 30.0
    n_blocks: int = 5
    gap_s: float = 5.0
    C: float = 1.0
    max_iter: int = 200
    min_dwell_frames: int = 2
    min_visit_frames: int = 3
    min_class_samples: int = 5
    n_match_bins: int = 3
    min_traversals: int = 5
    junction_window_s: float = 1.0
    min_per_choice: int = 5
    subset_sizes: tuple[int, ...] = (5, 10)
    n_subsets: int = 10
    n_subset_shuffles: int = 10
    n_jobs: int = 1
    seed: int = 0


# ---------------------------------------------------------------------------
# Window means with circular shift
# ---------------------------------------------------------------------------


class WindowMeans:
    """Mean of each channel over frame windows, for any circular shift.

    Shift ``s`` corresponds to ``np.roll(x, s, axis=1)``. Non-finite frames
    are excluded from each mean; a window with no finite frame is NaN.

    Parameters
    ----------
    x : (n_channels, n_frames) or (n_frames,) float
    """

    def __init__(self, x: npt.ArrayLike) -> None:
        a = np.atleast_2d(np.asarray(x, dtype=np.float64))
        self.n_channels, self.n_frames = a.shape
        fin = np.isfinite(a)
        zero = np.zeros((self.n_channels, 1))
        two = np.concatenate([np.where(fin, a, 0.0)] * 2, axis=1)
        self._cs = np.concatenate([zero, np.cumsum(two, axis=1)], axis=1)
        ftwo = np.concatenate([fin] * 2, axis=1).astype(np.float64)
        self._cf = np.concatenate([zero, np.cumsum(ftwo, axis=1)], axis=1)

    def means(self, starts: npt.ArrayLike, ends: npt.ArrayLike, shift: int = 0) -> np.ndarray:
        """Window means, shape ``(n_windows, n_channels)``; windows are ``[start, end)``."""
        st = np.asarray(starts, dtype=np.int64)
        length = np.asarray(ends, dtype=np.int64) - st
        if np.any(length < 0) or np.any(length > self.n_frames):
            raise ValueError("window lengths must lie in [0, n_frames]")
        s = (st - int(shift)) % self.n_frames
        tot = self._cs[:, s + length] - self._cs[:, s]
        cnt = self._cf[:, s + length] - self._cf[:, s]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(cnt > 0, tot / np.where(cnt > 0, cnt, 1.0), np.nan)
        return out.T


def impute_columns(x: npt.ArrayLike) -> np.ndarray:
    """Replace non-finite entries by the column mean of finite entries (0 if none)."""
    a = np.array(x, dtype=np.float64, copy=True)
    if a.size == 0:
        return a
    fin = np.isfinite(a)
    cnt = fin.sum(axis=0)
    col = np.where(cnt > 0, np.where(fin, a, 0.0).sum(axis=0) / np.maximum(cnt, 1), 0.0)
    return np.where(fin, a, col[None, :])


def behaviour_matrix(
    hd_deg: npt.ArrayLike, speed: npt.ArrayLike, ahv: npt.ArrayLike
) -> np.ndarray:
    """Behaviour channels ``(4, n_frames)``: sin HD, cos HD, speed, |AHV|."""
    hd = np.deg2rad(np.asarray(hd_deg, dtype=np.float64).ravel())
    return np.vstack(
        [
            np.sin(hd),
            np.cos(hd),
            np.asarray(speed, dtype=np.float64).ravel(),
            np.abs(np.asarray(ahv, dtype=np.float64).ravel()),
        ]
    )


def behaviour_features(
    bwm: WindowMeans, starts: npt.ArrayLike, ends: npt.ArrayLike, shift: int = 0
) -> np.ndarray:
    """Behaviour features per window ``(n, 4)``: sin/cos of circular mean HD, speed, |AHV|.

    The window means of sin and cos HD are normalised by their resultant
    length (NaN when it is zero or no frame is finite).
    """
    m = bwm.means(starts, ends, shift)
    r = np.hypot(m[:, 0], m[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(r > 0, m[:, 0] / r, np.nan)
        c = np.where(r > 0, m[:, 1] / r, np.nan)
    return np.column_stack([s, c, m[:, 2], m[:, 3]])


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


@dataclass
class Samples:
    """Decoding samples: frame windows ``[starts, ends)``, labels and strata.

    ``reason`` is non-empty when the decoder cannot be run for the session.
    """

    decoder: str
    starts: np.ndarray
    ends: np.ndarray
    labels: np.ndarray
    strata: np.ndarray
    reason: str = ""
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Number of samples."""
        return int(self.labels.size)

    @property
    def n_classes(self) -> int:
        """Number of distinct labels (2 for the stratified binary decoders)."""
        if self.n == 0:
            return 0
        if np.unique(self.strata).size > 1:
            return 2
        return int(np.unique(self.labels).size)

    @property
    def n_strata(self) -> int:
        """Number of strata."""
        return int(np.unique(self.strata).size)


def _empty_samples(decoder: str, reason: str, info: dict[str, Any] | None = None) -> Samples:
    z = np.zeros(0, dtype=np.int64)
    return Samples(decoder, z, z.copy(), z.copy(), z.copy(), reason, info or {})


def _make_samples(
    decoder: str,
    starts: npt.ArrayLike,
    ends: npt.ArrayLike,
    labels: npt.ArrayLike,
    strata: npt.ArrayLike,
    info: dict[str, Any],
) -> Samples:
    return Samples(
        decoder,
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(strata, dtype=np.int64),
        "",
        info,
    )


def session_cells(
    x_maze: npt.ArrayLike,
    y_maze: npt.ArrayLike,
    valid: npt.ArrayLike,
    maze: RoseMaze,
    min_dwell_frames: int = 2,
) -> np.ndarray:
    """Maze cell per frame (-1 for invalid frames and occupancies < *min_dwell_frames*)."""
    x = np.asarray(x_maze, dtype=np.float64).ravel()
    y = np.asarray(y_maze, dtype=np.float64).ravel()
    cells = discretize_position_fast(x, y, maze).astype(np.int64)
    cells[~np.asarray(valid, dtype=bool).ravel()] = -1
    return dwell_filtered_cells(cells, min_dwell_frames)


def running_visits(
    cells: npt.ArrayLike,
    onsets: npt.ArrayLike,
    offsets: npt.ArrayLike,
    min_visit_frames: int = 3,
) -> pd.DataFrame:
    """Contiguous stays in one maze cell within running bouts.

    Returns
    -------
    pandas.DataFrame
        Columns ``start, end`` (frames, end exclusive), ``cell``, ``bout``
        (index of the bout), sorted by ``start``.
    """
    c = np.asarray(cells, dtype=np.int64)
    rows: list[tuple[int, int, int, int]] = []
    for k, (a, b) in enumerate(zip(np.asarray(onsets), np.asarray(offsets), strict=True)):
        seg = c[int(a) : int(b)]
        if seg.size == 0:
            continue
        change = np.flatnonzero(np.diff(seg) != 0) + 1
        st = np.concatenate([[0], change])
        en = np.concatenate([change, [seg.size]])
        for s, e in zip(st, en, strict=True):
            if seg[s] >= 0 and e - s >= min_visit_frames:
                rows.append((int(a) + int(s), int(a) + int(e), int(seg[s]), k))
    return pd.DataFrame(rows, columns=["start", "end", "cell", "bout"])


def cell_stays(cells: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stays in one cell, merging runs of the same cell separated only by -1 frames.

    Returns
    -------
    cell, start, end : (n_stays,) int
        ``end`` is one past the last valid frame of the stay.
    """
    c = np.asarray(cells, dtype=np.int64)
    idx = np.flatnonzero(c >= 0)
    if idx.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z.copy(), z.copy()
    v = c[idx]
    first = np.concatenate([[0], np.flatnonzero(np.diff(v) != 0) + 1])
    last = np.concatenate([first[1:] - 1, [v.size - 1]])
    return v[first], idx[first], idx[last] + 1


def matched_subsample(
    labels: npt.ArrayLike,
    duration: npt.ArrayLike,
    speed: npt.ArrayLike,
    n_bins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Indices of a class-balanced subsample matched on duration and speed.

    Samples are binned by the pooled quantiles of duration and speed
    (``n_bins`` each); within every duration x speed bin the larger class is
    randomly subsampled to the size of the smaller one. Samples with a
    non-finite covariate are dropped.

    Parameters
    ----------
    labels : (n,) int
        Binary labels (0/1).
    duration, speed : (n,) float
    n_bins : int
    rng : Generator

    Returns
    -------
    (m,) int64
        Sorted indices; the two classes have equal counts.
    """
    lab = np.asarray(labels, dtype=np.int64)
    d = np.asarray(duration, dtype=np.float64)
    s = np.asarray(speed, dtype=np.float64)
    ok = np.isfinite(d) & np.isfinite(s)
    if not ok.any():
        return np.zeros(0, dtype=np.int64)
    inner = np.linspace(0, 1, n_bins + 1)[1:-1]
    bd = np.searchsorted(np.quantile(d[ok], inner), d, side="right")
    bs = np.searchsorted(np.quantile(s[ok], inner), s, side="right")
    cell = np.where(ok, bd * (n_bins + 1) + bs, -1)
    keep: list[np.ndarray] = []
    for b in np.unique(cell[cell >= 0]):
        i0 = np.flatnonzero((cell == b) & (lab == 0))
        i1 = np.flatnonzero((cell == b) & (lab == 1))
        m = min(i0.size, i1.size)
        if m:
            keep += [rng.choice(i0, m, replace=False), rng.choice(i1, m, replace=False)]
    return np.sort(np.concatenate(keep)) if keep else np.zeros(0, dtype=np.int64)


def inout_samples(
    bouts: pd.DataFrame,
    rng: np.random.Generator,
    n_bins: int = 3,
    min_class_samples: int = 5,
) -> Samples:
    """Runs into (label 1) vs out of (label 0) dead ends, matched on duration and speed."""
    rt = bouts["run_type"].to_numpy() if len(bouts) else np.zeros(0, dtype=object)
    sel = np.flatnonzero((rt == "into_dead_end") | (rt == "out_of_dead_end"))
    sub = bouts.iloc[sel]
    lab = (sub["run_type"].to_numpy() == "into_dead_end").astype(np.int64)
    info: dict[str, Any] = {
        "n_into_dead_end": int((lab == 1).sum()),
        "n_out_of_dead_end": int((lab == 0).sum()),
    }
    if sel.size == 0:
        return _empty_samples("inout", "no dead-end runs", info)
    keep = matched_subsample(
        lab, sub["duration_s"].to_numpy(), sub["mean_speed"].to_numpy(), n_bins, rng
    )
    info["inout_n_matched_per_class"] = int(keep.size // 2)
    if keep.size // 2 < min_class_samples:
        return _empty_samples("inout", "too few matched dead-end runs", info)
    m = sub.iloc[keep]
    lk = lab[keep]
    dur = m["duration_s"].to_numpy(np.float64)
    spd = m["mean_speed"].to_numpy(np.float64)
    for name, cls in (("into", 1), ("out", 0)):
        info[f"inout_median_duration_s_{name}"] = float(np.median(dur[lk == cls]))
        info[f"inout_median_speed_{name}"] = float(np.median(spd[lk == cls]))
    return _make_samples(
        "inout", m["onset"], m["offset"], lk, np.zeros(lk.size, dtype=np.int64), info
    )


def position_samples(visits: pd.DataFrame, min_class_samples: int = 5) -> Samples:
    """Running visits labelled by maze cell; cells with < *min_class_samples* visits dropped."""
    info: dict[str, Any] = {"n_running_visits": int(len(visits))}
    if visits.empty:
        return _empty_samples("position", "no running visits", info)
    counts = visits["cell"].value_counts()
    keep_cells = counts[counts >= min_class_samples].index
    v = visits[visits["cell"].isin(keep_cells)]
    info["position_n_classes"] = int(len(keep_cells))
    if len(keep_cells) < 2:
        return _empty_samples("position", "fewer than 2 cells with enough visits", info)
    return _make_samples(
        "position", v["start"], v["end"], v["cell"], np.zeros(len(v), dtype=np.int64), info
    )


def edge_direction_samples(
    visits: pd.DataFrame, maze: RoseMaze, min_traversals: int = 5
) -> Samples:
    """Edge traversals labelled by direction (0: lower to higher cell index, 1: reverse).

    A traversal is two consecutive visits in the same bout to adjacent cells;
    its window spans both visits. Stratum = undirected edge. Edges with fewer
    than *min_traversals* traversals in either direction are dropped.
    """
    info: dict[str, Any] = {"n_traversals": 0, "edge_n_edges_used": 0}
    if len(visits) < 2:
        return _empty_samples("edge_direction", "too few running visits", info)
    v = visits.sort_values("start")
    cell = v["cell"].to_numpy(np.int64)
    bout = v["bout"].to_numpy(np.int64)
    st = v["start"].to_numpy(np.int64)
    en = v["end"].to_numpy(np.int64)
    a, b = cell[:-1], cell[1:]
    ok = (bout[:-1] == bout[1:]) & (maze.dist[a, b] == 1)
    a, b = a[ok], b[ok]
    starts, ends = st[:-1][ok], en[1:][ok]
    lab = (a > b).astype(np.int64)
    edge = np.minimum(a, b) * maze.n_cells + np.maximum(a, b)
    info["n_traversals"] = int(lab.size)
    keep = np.zeros(lab.size, dtype=bool)
    for e in np.unique(edge):
        m = edge == e
        if (lab[m] == 0).sum() >= min_traversals and (lab[m] == 1).sum() >= min_traversals:
            keep |= m
    info["edge_n_edges_used"] = int(np.unique(edge[keep]).size)
    if not keep.any():
        return _empty_samples("edge_direction", "no edge with enough traversals each way", info)
    return _make_samples("edge_direction", starts[keep], ends[keep], lab[keep], edge[keep], info)


def junction_exit_samples(
    cells: npt.ArrayLike,
    maze: RoseMaze,
    fps: float,
    window_s: float = 1.0,
    min_per_choice: int = 5,
) -> Samples:
    """T-junction visits labelled by exit arm, stratified by (junction, arrival arm).

    Window = ``[exit - window_s, exit)`` where ``exit`` is one past the last
    frame in the junction. U-turns (exit arm = arrival arm) are excluded and
    counted.
    """
    cell, start, end = cell_stays(cells)
    jset = {maze.cell_to_idx[c] for c in maze.junctions if maze.node_types[c] == "t_junction"}
    w = max(1, int(round(window_s * fps)))
    rows: list[tuple[int, int, int, int]] = []
    n_uturn = 0
    for k in range(1, cell.size - 1):
        j, p, q = int(cell[k]), int(cell[k - 1]), int(cell[k + 1])
        if j not in jset or maze.dist[p, j] != 1 or maze.dist[q, j] != 1:
            continue
        if p == q:
            n_uturn += 1
            continue
        ex = int(end[k])
        if ex - w < 0:
            continue
        rows.append((ex - w, ex, q, j * maze.n_cells + p))
    info: dict[str, Any] = {
        "n_junction_visits": len(rows) + n_uturn,
        "n_junction_uturns": n_uturn,
        "junction_n_strata_used": 0,
    }
    if not rows:
        return _empty_samples("junction_exit", "no T-junction passes", info)
    arr = np.asarray(rows, dtype=np.int64)
    lab, strata = arr[:, 2], arr[:, 3]
    keep = np.zeros(lab.size, dtype=bool)
    for s in np.unique(strata):
        m = strata == s
        _, cnt = np.unique(lab[m], return_counts=True)
        if cnt.size == 2 and cnt.min() >= min_per_choice:
            keep |= m
    info["junction_n_strata_used"] = int(np.unique(strata[keep]).size)
    if not keep.any():
        return _empty_samples(
            "junction_exit", "no junction stratum with enough of each exit", info
        )
    return _make_samples(
        "junction_exit", arr[keep, 0], arr[keep, 1], lab[keep], strata[keep], info
    )


# ---------------------------------------------------------------------------
# Cross-validation and scoring
# ---------------------------------------------------------------------------


def time_block_folds(
    starts: npt.ArrayLike,
    ends: npt.ArrayLike,
    n_frames: int,
    n_blocks: int = 5,
    gap_frames: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Train/test index pairs for CV grouped by contiguous session time blocks.

    The session is split into *n_blocks* equal-length blocks; a sample
    belongs to the block containing its window midpoint. For each block with
    test samples, training samples are those of other blocks whose window
    ends at least *gap_frames* before the block starts or starts at least
    *gap_frames* after it ends.

    Roberts DR, Bahn V, Ciuti S, et al. 2017. "Cross-validation strategies
    for data with temporal, spatial, hierarchical, or phylogenetic
    structure." Ecography 40:913-929. doi:10.1111/ecog.02881
    """
    st = np.asarray(starts, dtype=np.int64)
    en = np.asarray(ends, dtype=np.int64)
    edges = np.linspace(0, n_frames, n_blocks + 1)
    mid = (st + en - 1) / 2.0
    block = np.clip(np.searchsorted(edges[1:-1], mid, side="right"), 0, n_blocks - 1)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for b in range(n_blocks):
        test = np.flatnonzero(block == b)
        if test.size == 0:
            continue
        lo, hi = edges[b] - gap_frames, edges[b + 1] + gap_frames
        train = np.flatnonzero((block != b) & ((en <= lo) | (st >= hi)))
        folds.append((train, test))
    return folds


def cv_predict(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    folds: list[tuple[np.ndarray, np.ndarray]],
    C: float = 1.0,
    hd: npt.ArrayLike | None = None,
    max_iter: int = 200,
) -> np.ndarray:
    """Out-of-fold predictions of an L2 logistic regression (balanced class weights).

    Parameters
    ----------
    x : (n, d) float
        Features (finite).
    y : (n,) int
    folds : list of (train_idx, test_idx)
    C : float
        Inverse L2 regularisation strength.
    hd : (n, 2) float or None
        When given, each feature is residualised on ``[1, hd]`` by least
        squares fitted on the training fold only.
    max_iter : int

    Notes
    -----
    Binary problems use the liblinear solver and multiclass problems lbfgs
    (multinomial loss); both are L2-penalised. liblinear also penalises the
    intercept, which has little effect with z-scored features and balanced
    class weights. Features must be finite (see :func:`impute_columns`);
    scikit-learn input validation is skipped for speed.

    Returns
    -------
    (n,) int64
        Predicted label; -1 for samples in no test fold or whose fold had no
        training samples. A training fold with one class predicts that class.

    References
    ----------
    Pedregosa F, et al. 2011. "Scikit-learn: Machine Learning in Python."
    Journal of Machine Learning Research 12:2825-2830.
    """
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.int64)
    ha = None if hd is None else np.asarray(hd, dtype=np.float64)
    pred = np.full(ya.size, -1, dtype=np.int64)
    for tr, te in folds:
        if tr.size == 0 or te.size == 0:
            continue
        classes = np.unique(ya[tr])
        if classes.size == 1:
            pred[te] = classes[0]
            continue
        xtr, xte = xa[tr], xa[te]
        if ha is not None:
            dtr = np.column_stack([np.ones(tr.size), ha[tr]])
            dte = np.column_stack([np.ones(te.size), ha[te]])
            beta = np.linalg.lstsq(dtr, xtr, rcond=None)[0]
            xtr = xtr - dtr @ beta
            xte = xte - dte @ beta
        mu = xtr.mean(axis=0)
        sd = xtr.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        solver = "liblinear" if classes.size == 2 else "lbfgs"
        _, inv, cnt = np.unique(ya[tr], return_inverse=True, return_counts=True)
        weight = (tr.size / (classes.size * cnt))[inv]  # balanced class weights
        clf = LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=0)
        with (
            warnings.catch_warnings(),
            sklearn.config_context(assume_finite=True, skip_parameter_validation=True),
        ):
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit((xtr - mu) / sd, ya[tr], sample_weight=weight)
            pred[te] = clf.predict((xte - mu) / sd)
    return pred


def balanced_accuracy(y: npt.ArrayLike, pred: npt.ArrayLike) -> float:
    """Mean per-class recall over the classes present in *y* (NaN if empty).

    Brodersen KH, Ong CS, Stephan KE, Buhmann JM. 2010. "The balanced
    accuracy and its posterior distribution." ICPR 2010, 3121-3124.
    doi:10.1109/ICPR.2010.764
    """
    ya = np.asarray(y)
    pa = np.asarray(pred)
    if ya.size == 0:
        return float("nan")
    return float(np.mean([np.mean(pa[ya == c] == c) for c in np.unique(ya)]))


def class_mean_distance(y: npt.ArrayLike, pred: npt.ArrayLike, dist: npt.ArrayLike) -> float:
    """Class-balanced mean graph distance between decoded and true cell.

    Per true class, the mean of ``dist[true, pred]`` over its samples, then
    the mean over classes. Unpredicted samples (``pred < 0``) count as the
    maximum distance in *dist*.
    """
    ya = np.asarray(y, dtype=np.int64)
    pa = np.asarray(pred, dtype=np.int64)
    d = np.asarray(dist, dtype=np.float64)
    if ya.size == 0:
        return float("nan")
    per = np.where(pa >= 0, d[ya, np.clip(pa, 0, None)], d.max())
    return float(np.mean([per[ya == c].mean() for c in np.unique(ya)]))


@dataclass
class Problem:
    """Samples plus per-stratum CV folds (stratum indices into the samples)."""

    samples: Samples
    strata: list[tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]]


def prepare_problem(samples: Samples, n_frames: int, n_blocks: int, gap_frames: int) -> Problem:
    """Build per-stratum time-block folds for *samples*."""
    out = []
    for s in np.unique(samples.strata):
        idx = np.flatnonzero(samples.strata == s)
        folds = time_block_folds(
            samples.starts[idx], samples.ends[idx], n_frames, n_blocks, gap_frames
        )
        out.append((idx, folds))
    return Problem(samples, out)


def score_problem(
    problem: Problem,
    x: np.ndarray,
    hd: np.ndarray | None,
    dist: np.ndarray | None,
    C: float = 1.0,
    max_iter: int = 200,
) -> dict[str, float]:
    """Cross-validated metrics for one feature matrix.

    Balanced accuracy is computed per stratum and averaged weighted by the
    number of samples; ``mean_distance`` is added for the position decoder
    (one stratum) when *dist* is given.
    """
    y = problem.samples.labels
    accs, weights = [], []
    out: dict[str, float] = {}
    for idx, folds in problem.strata:
        pred = cv_predict(
            x[idx], y[idx], folds, C=C, hd=None if hd is None else hd[idx], max_iter=max_iter
        )
        accs.append(balanced_accuracy(y[idx], pred))
        weights.append(idx.size)
        if dist is not None and problem.samples.decoder == "position":
            out["mean_distance"] = class_mean_distance(y[idx], pred, dist)
    out["balanced_accuracy"] = float(np.average(accs, weights=weights))
    return out


def null_summary(observed: float, null: npt.ArrayLike, lower_is_better: bool = False) -> dict:
    """Null mean, sd, z, one-sided p and excess of *observed* over a null sample.

    ``p = (1 + #{null at least as good as observed}) / (1 + n_null)``;
    ``excess`` is positive when the observed value is better than the null
    mean. NaN statistics when the observed value or the null is missing.
    """
    nk = np.asarray(null, dtype=np.float64)
    nk = nk[np.isfinite(nk)]
    nan = float("nan")
    res = {"null_mean": nan, "null_sd": nan, "z": nan, "p": nan, "excess": nan, "n_null": 0}
    res["n_null"] = int(nk.size)
    if not np.isfinite(observed) or nk.size == 0:
        return res
    mu, sd = float(nk.mean()), float(nk.std())
    better = nk <= observed if lower_is_better else nk >= observed
    res.update(
        {
            "null_mean": mu,
            "null_sd": sd,
            "z": float((observed - mu) / sd) if sd > 0 else nan,
            "p": float((1 + better.sum()) / (1 + nk.size)),
            "excess": float(mu - observed if lower_is_better else observed - mu),
        }
    )
    return res


# ---------------------------------------------------------------------------
# Session-level evaluation
# ---------------------------------------------------------------------------


@dataclass
class SessionData:
    """Predictor arrays shared by all decoders of one session."""

    neural: WindowMeans
    behaviour: WindowMeans
    dist: np.ndarray


def _score_shifts(
    problem: Problem,
    data: SessionData,
    shifts: np.ndarray,
    feature_sets: tuple[str, ...],
    cols: np.ndarray | None,
    C: float,
    max_iter: int,
) -> dict[tuple[str, str], np.ndarray]:
    """Metrics for each feature set at each shift (``(feature_set, metric)`` -> array)."""
    s = problem.samples
    metrics = METRICS[s.decoder]
    out = {(fs, m): np.full(shifts.size, np.nan) for fs in feature_sets for m in metrics}
    hd0 = impute_columns(behaviour_features(data.behaviour, s.starts, s.ends, 0)[:, :2])
    for k, sh in enumerate(shifts):
        xn = None
        if any(fs != "behaviour" for fs in feature_sets):
            xn = impute_columns(data.neural.means(s.starts, s.ends, int(sh)))
            if cols is not None:
                xn = xn[:, cols]
        for fs in feature_sets:
            if fs == "behaviour":
                x = impute_columns(behaviour_features(data.behaviour, s.starts, s.ends, int(sh)))
                res = score_problem(problem, x, None, data.dist, C, max_iter)
            elif fs == "neural":
                res = score_problem(problem, xn, None, data.dist, C, max_iter)  # type: ignore[arg-type]
            else:
                res = score_problem(problem, xn, hd0, data.dist, C, max_iter)  # type: ignore[arg-type]
            for m in metrics:
                out[(fs, m)][k] = res[m]
    return out


def evaluate_problem(
    problem: Problem,
    data: SessionData,
    shifts: np.ndarray,
    feature_sets: tuple[str, ...] = FEATURE_SETS,
    cols: np.ndarray | None = None,
    C: float = 1.0,
    max_iter: int = 200,
    n_jobs: int = 1,
) -> dict[tuple[str, str], np.ndarray]:
    """Metrics at shift 0 followed by every null shift.

    ``shifts`` must start with 0 (the observed alignment). With
    ``n_jobs > 1`` the shifts are split into chunks evaluated in parallel
    (joblib, installed with scikit-learn).
    """
    sh = np.asarray(shifts, dtype=np.int64)
    if n_jobs <= 1 or sh.size < 2 * n_jobs:
        return _score_shifts(problem, data, sh, feature_sets, cols, C, max_iter)
    from joblib import Parallel, delayed

    chunks = np.array_split(sh, n_jobs)
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_score_shifts)(problem, data, c, feature_sets, cols, C, max_iter) for c in chunks
    )
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def draw_shifts(
    n_frames: int, n_shuffles: int, min_shift: int, rng: np.random.Generator
) -> np.ndarray:
    """``[0, s_1, ..., s_n]`` with ``s_i`` uniform in ``[min_shift, n_frames - min_shift)``.

    Only ``[0]`` when the session is too short for the minimum shift.
    """
    if n_shuffles < 1 or n_frames <= 2 * min_shift:
        return np.zeros(1, dtype=np.int64)
    null = rng.integers(min_shift, n_frames - min_shift, size=n_shuffles)
    return np.concatenate([[0], null]).astype(np.int64)


@dataclass
class SessionResult:
    """Outputs of :func:`decode_session`."""

    decoders: pd.DataFrame
    subsets: pd.DataFrame
    counts: dict[str, Any]


def _result_rows(
    samples: Samples,
    scores: dict[tuple[str, str], np.ndarray] | None,
    n_cells: int,
    reason: str,
) -> list[dict[str, Any]]:
    rows = []
    for fs in FEATURE_SETS:
        for m in METRICS[samples.decoder]:
            low = m in LOWER_IS_BETTER
            arr = None if scores is None else scores[(fs, m)]
            obs = float("nan") if arr is None else float(arr[0])
            ns = null_summary(obs, np.zeros(0) if arr is None else arr[1:], low)
            rows.append(
                {
                    "decoder": samples.decoder,
                    "feature_set": fs,
                    "metric": m,
                    "headline": m == HEADLINE_METRIC[samples.decoder],
                    "lower_is_better": low,
                    "observed": obs,
                    **ns,
                    "n_samples": samples.n,
                    "n_classes": samples.n_classes,
                    "n_strata": samples.n_strata,
                    "n_cells": n_cells,
                    "n_features": len(BEHAVIOUR_CHANNELS) if fs == "behaviour" else n_cells,
                    "reason": reason,
                }
            )
    return rows


def build_samples(
    x_maze: npt.ArrayLike,
    y_maze: npt.ArrayLike,
    speed: npt.ArrayLike,
    light_on: npt.ArrayLike,
    valid: npt.ArrayLike,
    fps: float,
    params: PopDecParams,
    maze: RoseMaze,
    rng: np.random.Generator,
) -> tuple[dict[str, Samples], dict[str, Any]]:
    """Samples for every decoder plus session behaviour counts."""
    spd = np.asarray(speed, dtype=np.float64).ravel()
    v = np.asarray(valid, dtype=bool).ravel()
    on, off = run_bouts(spd, v, fps)
    bouts = annotate_bouts(
        on, off, x_maze, y_maze, spd, light_on, fps, maze, params.min_dwell_frames
    )
    cells = session_cells(x_maze, y_maze, v, maze, params.min_dwell_frames)
    visits = running_visits(cells, on, off, params.min_visit_frames)
    samples = {
        "inout": inout_samples(bouts, rng, params.n_match_bins, params.min_class_samples),
        "position": position_samples(visits, params.min_class_samples),
        "edge_direction": edge_direction_samples(visits, maze, params.min_traversals),
        "junction_exit": junction_exit_samples(
            cells, maze, fps, params.junction_window_s, params.min_per_choice
        ),
    }
    counts: dict[str, Any] = {
        "n_frames": int(spd.size),
        "n_valid_frames": int(v.sum()),
        "n_bouts": int(on.size),
    }
    for name, s in samples.items():
        counts.update(s.info)
        counts[f"{name}_n_samples"] = s.n
        counts[f"{name}_reason"] = s.reason
    return samples, counts


def decode_session(
    signal: npt.ArrayLike,
    x_maze: npt.ArrayLike,
    y_maze: npt.ArrayLike,
    speed: npt.ArrayLike,
    hd_deg: npt.ArrayLike,
    ahv: npt.ArrayLike,
    light_on: npt.ArrayLike,
    valid: npt.ArrayLike,
    fps: float,
    params: PopDecParams | None = None,
    maze: RoseMaze | None = None,
) -> SessionResult:
    """All four decoders, three feature sets, shift nulls and cell-subset curve.

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
    params : PopDecParams or None
    maze : RoseMaze or None

    Returns
    -------
    SessionResult
        ``decoders``: one row per decoder x feature set x metric with
        observed value, null statistics (:func:`null_summary`), sample,
        class, stratum and cell counts and the skip reason (empty if run).
        ``subsets``: neural decoding vs number of cells (random subsets of
        ``params.subset_sizes`` smaller than the population, averaged over
        ``params.n_subsets`` draws, null from the first
        ``params.n_subset_shuffles`` shifts; plus the full population from
        the main result). ``counts``: behaviour sample counts.

    References
    ----------
    See the module docstring.
    """
    p = params or PopDecParams()
    maze = maze or build_rose_maze()
    rng = np.random.default_rng(p.seed)
    sig = np.atleast_2d(np.asarray(signal, dtype=np.float64))
    n_cells, n_frames = sig.shape
    samples, counts = build_samples(x_maze, y_maze, speed, light_on, valid, fps, p, maze, rng)
    counts["n_cells"] = n_cells
    data = SessionData(
        WindowMeans(sig),
        WindowMeans(behaviour_matrix(hd_deg, speed, ahv)),
        maze.dist.astype(np.float64),
    )
    shifts = draw_shifts(n_frames, p.n_shuffles, int(round(p.min_shift_s * fps)), rng)
    gap = int(round(p.gap_s * fps))
    rows: list[dict[str, Any]] = []
    sub_rows: list[dict[str, Any]] = []
    for name in DECODERS:
        s = samples[name]
        if s.reason:
            rows += _result_rows(s, None, n_cells, s.reason)
            continue
        problem = prepare_problem(s, n_frames, p.n_blocks, gap)
        scores = evaluate_problem(
            problem, data, shifts, C=p.C, max_iter=p.max_iter, n_jobs=p.n_jobs
        )
        reason = "" if shifts.size > 1 else "session too short for shift null"
        rows += _result_rows(s, scores, n_cells, reason)
        sub_rows += _subset_rows(problem, data, shifts, scores, n_cells, p, rng)
    log.debug("decoded session: %d cells, %d frames", n_cells, n_frames)
    return SessionResult(pd.DataFrame(rows), pd.DataFrame(sub_rows), counts)


def _subset_rows(
    problem: Problem,
    data: SessionData,
    shifts: np.ndarray,
    full: dict[tuple[str, str], np.ndarray],
    n_cells: int,
    p: PopDecParams,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Neural decoding vs number of cells (see :func:`decode_session`)."""
    dec = problem.samples.decoder
    sub_shifts = shifts[: p.n_subset_shuffles + 1]
    rows: list[dict[str, Any]] = []
    for k in sorted({int(k) for k in p.subset_sizes if 0 < k < n_cells}):
        obs: dict[str, list[float]] = {m: [] for m in METRICS[dec]}
        nul: dict[str, list[float]] = {m: [] for m in METRICS[dec]}
        for _ in range(p.n_subsets):
            cols = np.sort(rng.choice(n_cells, k, replace=False))
            sc = evaluate_problem(
                problem, data, sub_shifts, ("neural",), cols, p.C, p.max_iter, p.n_jobs
            )
            for m in METRICS[dec]:
                obs[m].append(float(sc[("neural", m)][0]))
                nul[m].extend(sc[("neural", m)][1:].tolist())
        for m in METRICS[dec]:
            rows.append(_subset_row(dec, m, k, p.n_subsets, obs[m], nul[m], False))
    for m in METRICS[dec]:
        arr = full[("neural", m)]
        rows.append(_subset_row(dec, m, n_cells, 1, [float(arr[0])], arr[1:].tolist(), True))
    return rows


def _subset_row(
    dec: str,
    metric: str,
    k: int,
    n_draws: int,
    obs: list[float],
    null: list[float],
    full_population: bool,
) -> dict[str, Any]:
    o = np.asarray(obs, dtype=np.float64)
    nl = np.asarray(null, dtype=np.float64)
    om = float(np.nanmean(o)) if np.isfinite(o).any() else float("nan")
    nm = float(np.nanmean(nl)) if np.isfinite(nl).any() else float("nan")
    excess = nm - om if metric in LOWER_IS_BETTER else om - nm
    return {
        "decoder": dec,
        "metric": metric,
        "n_cells_subset": int(k),
        "full_population": bool(full_population),
        "n_draws": int(n_draws),
        "observed_mean": om,
        "null_mean": nm,
        "excess": excess,
        "n_null": int(np.isfinite(nl).sum()),
    }


# ---------------------------------------------------------------------------
# Across-session statistics
# ---------------------------------------------------------------------------


def _wilcoxon(v: npt.ArrayLike) -> float:
    """Two-sided Wilcoxon signed-rank p vs 0 (NaN with < 3 finite values or all zero)."""
    a = np.asarray(v, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 3 or not np.any(a != 0):
        return float("nan")
    return float(stats.wilcoxon(a).pvalue)


def across_session_summary(sessions: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Excess over the shift null across sessions and across animal medians.

    One row per decoder x metric x feature set x group, where group is
    ``all`` (pooled, the main result) or a cell type (descriptive). Tests are
    two-sided Wilcoxon signed-rank tests of ``excess`` against 0, on sessions
    (sessions of one animal are not independent) and on per-animal medians.

    Parameters
    ----------
    sessions : DataFrame
        Rows from :func:`decode_session` with ``exp_id``, ``animal_id`` and
        ``celltype`` attached.
    alpha : float
        Per-session significance level for ``frac_sessions_sig``.
    """
    rows: list[dict[str, Any]] = []
    if sessions.empty:
        return pd.DataFrame(rows)
    keys = ["decoder", "metric", "feature_set"]
    for (dec, met, fs), g in sessions.groupby(keys, sort=False):
        groups = [("all", g)]
        if "celltype" in g:
            groups += [(str(ct), gg) for ct, gg in g.groupby("celltype")]
        for name, gg in groups:
            ok = gg[np.isfinite(gg["excess"].to_numpy(np.float64))]
            animal = ok.groupby("animal_id")["excess"].median()
            pv = ok["p"].to_numpy(np.float64)
            rows.append(
                {
                    "decoder": dec,
                    "metric": met,
                    "feature_set": fs,
                    "headline": met == HEADLINE_METRIC.get(str(dec)),
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


def paired_feature_comparisons(sessions: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon of null-corrected excess between feature sets.

    For each decoder x metric and each pair in :data:`COMPARISONS`
    (``neural`` vs ``behaviour``, ``neural_hd_removed`` vs ``behaviour``,
    ``neural`` vs ``neural_hd_removed``): the per-session difference in
    ``excess`` (first minus second), tested against 0 across sessions and
    across per-animal medians of the difference.
    """
    rows: list[dict[str, Any]] = []
    if sessions.empty:
        return pd.DataFrame(rows)
    for (dec, met), g in sessions.groupby(["decoder", "metric"], sort=False):
        wide = g.pivot_table(
            index=["exp_id", "animal_id"], columns="feature_set", values="excess", aggfunc="first"
        )
        for a, b in COMPARISONS:
            if a not in wide or b not in wide:
                continue
            diff = (wide[a] - wide[b]).dropna()
            animal = diff.groupby(level="animal_id").median()
            rows.append(
                {
                    "decoder": dec,
                    "metric": met,
                    "headline": met == HEADLINE_METRIC.get(str(dec)),
                    "comparison": f"{a} - {b}",
                    "n_sessions": int(diff.size),
                    "median_diff": float(diff.median()) if diff.size else np.nan,
                    "n_sessions_first_better": int((diff > 0).sum()),
                    "session_wilcoxon_p": _wilcoxon(diff),
                    "n_animals": int(animal.size),
                    "median_animal_diff": float(animal.median()) if animal.size else np.nan,
                    "animal_wilcoxon_p": _wilcoxon(animal),
                }
            )
    return pd.DataFrame(rows)


def subset_curve_summary(subsets: pd.DataFrame) -> pd.DataFrame:
    """Median observed, null and excess across sessions per decoder x metric x subset size.

    The full-population rows have different sizes per session and are
    pooled under ``n_cells_subset = -1``.
    """
    if subsets.empty:
        return pd.DataFrame()
    df = subsets.copy()
    df.loc[df["full_population"].astype(bool), "n_cells_subset"] = -1
    g = df.groupby(["decoder", "metric", "n_cells_subset"])
    out = g.agg(
        n_sessions=("excess", lambda v: int(np.isfinite(v).sum())),
        median_observed=("observed_mean", "median"),
        median_null=("null_mean", "median"),
        median_excess=("excess", "median"),
        session_wilcoxon_p=("excess", _wilcoxon),
    ).reset_index()
    return out
