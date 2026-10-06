"""Within-epoch dynamics of the RSP population maze code.

Four questions, all on pooled soma populations of one session, with a
frame-wise cross-validated decoder trained on light and dark frames pooled:

1. **Drift and re-anchoring.** Decoding error of maze position (graph distance
   between decoded and true maze cell) and of travel direction as a function
   of time since the last light switch, separately in dark and light epochs.
   Path-integration maintenance predicts error growing with time in the dark
   and dropping after lights-on. Also a population-vector (PV) version: each
   frame's correlation with the light-condition template of its maze cell.
2. **Anticipatory reversals.** Mid-maze reversals of travel direction
   (U-turns away from dead ends). The decoded travel direction (decoder
   trained on frames away from reversals, causal smoothing) is aligned to the
   behavioural turn and compared with matched non-reversal decelerations and
   with a behaviour-only decoder fed the same smoothing.
3. **Decoding confidence before repeat choices.** At T-junction departures,
   choices are ``repeat`` (U-turn, or the next junction/dead end was among the
   last ``n_back`` junction/dead-end visits) or ``new``. Position-decoding
   confidence in the window before departure is compared between them.
4. **Representation vs exploration quality.** Per light/dark epoch, decoding
   error vs coverage of maze cells relative to a random walk of the same
   number of steps.

Labels
------
Maze cells come from :func:`hm2p.analysis.population_maze.session_cells`
(nearest cell, occupancies shorter than ``min_dwell_frames`` unassigned).
Travel direction is defined on the maze tree rooted at its graph centre: a
stay is ``outward`` (1) when the animal leaves it to a cell further from the
root and ``inward`` (0) otherwise; stays whose arrival and departure
directions differ (U-turns, crossing a junction between two child branches)
are unlabelled. Running frames follow
:func:`hm2p.analysis.running_coding.run_bouts`.

Decoder
-------
Per-frame features are the cells' activity smoothed with a boxcar (centred
for questions 1, 3, 4; causal for question 2 so that no future activity
enters a frame), z-scored with training statistics. L2 logistic regression
with balanced class weights (scikit-learn; Pedregosa et al. 2011) is trained
on labelled running frames; cross-validation is by ``n_blocks`` contiguous
time blocks with a ``gap_s`` exclusion around the test block (Roberts et al.
2017); every frame of the test block receives out-of-fold class
probabilities. The behaviour-only decoder uses sine/cosine head direction,
speed and |AHV| with the same smoothing.

Statistics
----------
Per-session effect sizes are Spearman or partial Spearman correlations
(rank residuals; Conover & Iman 1981) or median differences; inference is a
two-sided Wilcoxon signed-rank test across per-animal medians (sessions of
one animal are not independent), with the matched-pairs rank-biserial
correlation as the effect size (Kerby 2014).

References
----------
Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth show
rapid learning, sudden insight, and efficient exploration." eLife 10:e66175.
doi:10.7554/eLife.66175

Pedregosa F, et al. 2011. "Scikit-learn: Machine Learning in Python." Journal
of Machine Learning Research 12:2825-2830.
https://github.com/scikit-learn/scikit-learn

Roberts DR, Bahn V, Ciuti S, et al. 2017. "Cross-validation strategies for
data with temporal, spatial, hierarchical, or phylogenetic structure."
Ecography 40:913-929. doi:10.1111/ecog.02881

Conover WJ, Iman RL. 1981. "Rank transformations as a bridge between
parametric and nonparametric statistics." The American Statistician
35:124-129. doi:10.1080/00031305.1981.10479327

Kerby DS. 2014. "The simple difference formula: an approach to teaching
nonparametric correlation." Comprehensive Psychology 3:11.IT.3.1.
doi:10.2466/11.IT.3.1

Valerio S, Taube JS. 2012. "Path integration: how the head direction signal
maintains and corrects spatial orientation." Nature Neuroscience
15:1445-1453. doi:10.1038/nn.3215 (drift of HD in darkness, landmark
correction)

Alexander AS, Nitz DA. 2015. "Retrosplenial cortex maps the conjunction of
internal and external spaces." Nature Neuroscience 18:1143-1151.
doi:10.1038/nn.4058

Random-walk coverage null: :mod:`hm2p.maze.exploration_complexity`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from hm2p.analysis.maze_memory import partial_spearman
from hm2p.analysis.maze_running import light_epoch_starts
from hm2p.analysis.population_maze import cell_stays, session_cells
from hm2p.analysis.running_coding import run_bouts
from hm2p.maze.exploration_complexity import (
    build_adjacency_indices,
    coverage_z_vs_null,
    random_walk_coverage_null,
)
from hm2p.maze.topology import RoseMaze, build_rose_maze


@dataclass
class DynParams:
    """Parameters (see module docstring)."""

    smooth_s: float = 1.0
    rev_smooth_s: float = 0.3
    n_blocks: int = 10
    gap_s: float = 5.0
    C: float = 1.0
    max_iter: int = 300
    min_dwell_frames: int = 2
    min_class_frames: int = 20
    epoch_bin_s: float = 10.0
    epoch_len_s: float = 60.0
    edge_window_s: float = 15.0
    min_window_frames: int = 10
    rev_min_turn_deg: float = 120.0
    rev_max_stay_s: float = 6.0
    rev_hd_window_s: float = 1.0
    rev_exclude_s: float = 3.0
    rev_lag_s: float = 3.0
    pre_window_s: tuple[float, float] = (-1.0, -0.2)
    min_events: int = 5
    dep_n_back: int = 4
    dep_window_s: float = 2.0
    min_epoch_steps: int = 5
    min_epoch_run_frames: int = 30
    n_rw_sims: int = 500
    seed: int = 0


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def boxcar(x: npt.ArrayLike, w: int, causal: bool = False) -> np.ndarray:
    """NaN-aware boxcar mean along the last axis.

    Centred window ``[t - w//2, t - w//2 + w)`` or causal window
    ``[t - w + 1, t]`` (truncated at the edges). Frames with no finite value
    in the window are NaN.
    """
    a = np.atleast_2d(np.asarray(x, dtype=np.float64))
    n = a.shape[1]
    w = max(1, int(w))
    fin = np.isfinite(a)
    cs = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(np.where(fin, a, 0.0), 1)], 1)
    cf = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(fin, 1)], 1)
    t = np.arange(n)
    lo = t - w + 1 if causal else t - w // 2
    hi = lo + w
    lo, hi = np.clip(lo, 0, n), np.clip(hi, 0, n)
    tot, cnt = cs[:, hi] - cs[:, lo], cf[:, hi] - cf[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0, tot / np.where(cnt > 0, cnt, 1), np.nan)
    return out if np.ndim(x) > 1 else out[0]


def tree_depth(maze: RoseMaze) -> np.ndarray:
    """Graph distance of every cell from the maze graph centre (minimum eccentricity)."""
    root = int(np.argmin(maze.dist.max(axis=1)))
    return maze.dist[root].astype(np.int64)


def time_in_epoch(light_on: npt.ArrayLike, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Seconds since the start of each frame's light epoch, and the epoch index."""
    lo = np.asarray(light_on, dtype=bool).ravel()
    st = light_epoch_starts(lo)
    _, epoch = np.unique(st, return_inverse=True)
    return (np.arange(lo.size) - st) / fps, epoch.astype(np.int64)


def frame_directions(
    cells: npt.ArrayLike, maze: RoseMaze, max_gap: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame travel direction on the rooted maze tree.

    Returns
    -------
    direction : (n,) int
        1 outward, 0 inward, -1 unlabelled (see module docstring).
    arrive, leave : (n,) int
        Arrival and departure direction of the frame's stay (-1 unknown).
        Stays separated from their neighbour by more than ``max_gap``
        unassigned frames have unknown arrival / departure.
    """
    c = np.asarray(cells, dtype=np.int64)
    depth = tree_depth(maze)
    cell, st, en = cell_stays(c)
    out = np.full(c.size, -1, dtype=np.int64)
    arr_f, lea_f = out.copy(), out.copy()
    for k in range(cell.size):
        a = lv = -1
        if k > 0 and maze.dist[cell[k - 1], cell[k]] == 1 and st[k] - en[k - 1] <= max_gap:
            a = int(depth[cell[k]] > depth[cell[k - 1]])
        if (
            k < cell.size - 1
            and maze.dist[cell[k], cell[k + 1]] == 1
            and st[k + 1] - en[k] <= max_gap
        ):
            lv = int(depth[cell[k + 1]] > depth[cell[k]])
        sl = slice(st[k], en[k])
        arr_f[sl], lea_f[sl] = a, lv
        if lv >= 0 and (a < 0 or a == lv):
            out[sl] = lv
    out[c < 0] = -1
    return out, arr_f, lea_f


def running_mask(speed: npt.ArrayLike, valid: npt.ArrayLike, fps: float) -> np.ndarray:
    """Frames inside running bouts (:func:`run_bouts`)."""
    s = np.asarray(speed, dtype=np.float64)
    on, off = run_bouts(s, valid, fps)
    m = np.zeros(s.size, dtype=bool)
    for a, b in zip(on, off, strict=True):
        m[a:b] = True
    return m


def behaviour_frames(
    hd_deg: npt.ArrayLike, speed: npt.ArrayLike, ahv: npt.ArrayLike
) -> np.ndarray:
    """Behaviour channels ``(4, n)``: sin HD, cos HD, speed, |AHV|."""
    h = np.deg2rad(np.asarray(hd_deg, dtype=np.float64))
    return np.vstack(
        [np.sin(h), np.cos(h), np.asarray(speed, float), np.abs(np.asarray(ahv, float))]
    )


def behaviour_dir_frames(
    hd_deg: npt.ArrayLike,
    speed: npt.ArrayLike,
    ahv: npt.ArrayLike,
    cells: npt.ArrayLike,
    n_cells: int,
) -> np.ndarray:
    """Behaviour channels for travel direction ``(2 n_cells + 2, n)``.

    Sine and cosine of head direction multiplied by a one-hot code of the
    current maze cell (so a linear decoder can map heading to tree direction
    separately in every cell), plus speed and |AHV|. Frames without a cell
    have zero heading channels.
    """
    c = np.asarray(cells, dtype=np.int64)
    h = np.deg2rad(np.asarray(hd_deg, dtype=np.float64))
    out = np.zeros((2 * n_cells + 2, c.size))
    ok = c >= 0
    i = np.flatnonzero(ok)
    out[2 * c[ok], i] = np.sin(h[ok])
    out[2 * c[ok] + 1, i] = np.cos(h[ok])
    out[-2] = np.asarray(speed, dtype=np.float64)
    out[-1] = np.abs(np.asarray(ahv, dtype=np.float64))
    return out


def _impute(x: np.ndarray) -> np.ndarray:
    fin = np.isfinite(x)
    col = np.where(fin, x, 0).sum(0) / np.maximum(fin.sum(0), 1)
    return np.where(fin, x, col[None, :])


def frame_blocks(n_frames: int, n_blocks: int) -> np.ndarray:
    """Contiguous time-block index per frame."""
    edges = np.linspace(0, n_frames, n_blocks + 1)
    return np.clip(
        np.searchsorted(edges[1:-1], np.arange(n_frames), side="right"), 0, n_blocks - 1
    )


def cv_proba(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    train: npt.ArrayLike,
    n_blocks: int = 10,
    gap_frames: int = 0,
    C: float = 1.0,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold class probabilities for every frame (time-block CV).

    Parameters
    ----------
    x : (n_frames, d) float
    y : (n_frames,) int
        Labels; only frames with ``train`` True are used for fitting.
    train : (n_frames,) bool
    n_blocks, gap_frames : int
        Each block is a test block in turn; training frames within
        ``gap_frames`` of it are dropped.

    Returns
    -------
    proba : (n_frames, n_classes) float
        NaN rows for frames whose fold could not be fitted; classes absent
        from a fold's training set get probability 0.
    classes : (n_classes,) int
        Sorted labels of the training frames.

    References
    ----------
    Pedregosa F, et al. 2011. JMLR 12:2825-2830; Roberts DR, et al. 2017.
    Ecography 40:913-929. doi:10.1111/ecog.02881
    """
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    xa = _impute(np.asarray(x, dtype=np.float64))
    ya = np.asarray(y, dtype=np.int64)
    tr_all = np.asarray(train, dtype=bool)
    n = ya.size
    classes = np.unique(ya[tr_all])
    proba = np.full((n, classes.size), np.nan)
    if classes.size < 2:
        return proba, classes
    block = frame_blocks(n, n_blocks)
    idx = np.arange(n)
    for b in range(n_blocks):
        test = block == b
        if not test.any():
            continue
        lo, hi = idx[test][0], idx[test][-1] + 1
        tr = tr_all & ~test & ((idx < lo - gap_frames) | (idx >= hi + gap_frames))
        cl, inv, cnt = np.unique(ya[tr], return_inverse=True, return_counts=True)
        if cl.size < 2:
            continue
        mu, sd = xa[tr].mean(0), xa[tr].std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        clf = LogisticRegression(C=C, max_iter=max_iter, random_state=0)
        with warnings.catch_warnings(), sklearn.config_context(assume_finite=True):
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit((xa[tr] - mu) / sd, ya[tr], sample_weight=(tr.sum() / (cl.size * cnt))[inv])
            p = clf.predict_proba((xa[test] - mu) / sd)
        rows = np.zeros((test.sum(), classes.size))
        rows[:, np.searchsorted(classes, clf.classes_)] = p
        proba[test] = rows
    return proba, classes


def position_metrics(
    proba: np.ndarray, classes: np.ndarray, cells: npt.ArrayLike, dist: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-frame position-decoding metrics.

    ``err``: graph distance between decoded (argmax) and true cell;
    ``ptrue``: posterior of the true cell; ``experr``: posterior-expected
    graph distance. NaN where the true cell is not a decoder class or the
    frame was not predicted.
    """
    c = np.asarray(cells, dtype=np.int64)
    ok = np.isfinite(proba).all(1) & np.isin(c, classes)
    n = c.size
    err, ptrue, experr = (np.full(n, np.nan) for _ in range(3))
    if not ok.any():
        return {"err": err, "ptrue": ptrue, "experr": experr}
    pv = proba[ok]
    tc = c[ok]
    pred = classes[np.argmax(pv, 1)]
    err[ok] = dist[tc, pred]
    ptrue[ok] = pv[np.arange(tc.size), np.searchsorted(classes, tc)]
    experr[ok] = (pv * dist[np.ix_(tc, classes)]).sum(1)
    return {"err": err, "ptrue": ptrue, "experr": experr}


def cell_demean(values: npt.ArrayLike, cells: npt.ArrayLike, mask: npt.ArrayLike) -> np.ndarray:
    """*values* minus the mean over ``mask`` frames of the same true cell (excess error)."""
    v = np.asarray(values, dtype=np.float64)
    c = np.asarray(cells, dtype=np.int64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(v) & (c >= 0)
    out = np.full(v.size, np.nan)
    for k in np.unique(c[m]):
        sel = c == k
        out[sel] = v[sel] - v[m & sel].mean()
    return out


def pv_similarity(
    z: npt.ArrayLike,
    cells: npt.ArrayLike,
    template_mask: npt.ArrayLike,
    n_blocks: int,
    gap_frames: int,
    min_frames: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlation of each frame's population vector with its cell's template.

    Templates are mean vectors over ``template_mask`` frames (e.g. light
    running frames) of the other time blocks (gap excluded). Returns
    ``r_true`` and ``specificity`` = ``r_true`` minus the mean correlation
    with the other cells' templates. NaN where unavailable.
    """
    za = _impute(np.asarray(z, dtype=np.float64))
    c = np.asarray(cells, dtype=np.int64)
    tm = np.asarray(template_mask, dtype=bool) & (c >= 0)
    n = c.size
    r_true, spec = np.full(n, np.nan), np.full(n, np.nan)
    block = frame_blocks(n, n_blocks)
    idx = np.arange(n)
    for b in range(n_blocks):
        test = (block == b) & (c >= 0)
        if not test.any():
            continue
        lo, hi = idx[block == b][0], idx[block == b][-1] + 1
        tr = tm & (block != b) & ((idx < lo - gap_frames) | (idx >= hi + gap_frames))
        ks = [k for k in np.unique(c[tr]) if (c[tr] == k).sum() >= min_frames]
        if len(ks) < 2:
            continue
        tpl = np.vstack([za[tr & (c == k)].mean(0) for k in ks])
        tpl = tpl - tpl.mean(1, keepdims=True)
        tpl /= np.maximum(np.linalg.norm(tpl, axis=1, keepdims=True), 1e-12)
        ti = np.flatnonzero(test & np.isin(c, ks))
        xv = za[ti] - za[ti].mean(1, keepdims=True)
        xv /= np.maximum(np.linalg.norm(xv, axis=1, keepdims=True), 1e-12)
        r = xv @ tpl.T
        col = np.searchsorted(np.asarray(ks), c[ti])
        rt = r[np.arange(ti.size), col]
        r_true[ti] = rt
        spec[ti] = rt - (r.sum(1) - rt) / (len(ks) - 1)
    return r_true, spec


# ---------------------------------------------------------------------------
# Session frame table (question 1, used by 3 and 4)
# ---------------------------------------------------------------------------


def frame_table(
    signal: npt.ArrayLike,
    x_maze: npt.ArrayLike,
    y_maze: npt.ArrayLike,
    speed: npt.ArrayLike,
    hd_deg: npt.ArrayLike,
    ahv: npt.ArrayLike,
    light_on: npt.ArrayLike,
    valid: npt.ArrayLike,
    fps: float,
    params: DynParams | None = None,
    maze: RoseMaze | None = None,
) -> pd.DataFrame:
    """Per-frame labels, covariates and out-of-fold decoding metrics.

    Columns: ``t_s``, ``light``, ``t_epoch``, ``epoch``, ``running``,
    ``cell``, ``direction``, ``speed``, ``abs_ahv``; for each feature set
    ``fs`` in (``neural``, ``behaviour``): ``{fs}_pos_err``,
    ``{fs}_pos_ptrue``, ``{fs}_pos_experr``, ``{fs}_pos_excess`` (error minus
    the session mean error of the true cell over running frames),
    ``{fs}_dir_ptrue``, ``{fs}_dir_pout``; and ``pv_r``, ``pv_spec``
    (neural PV similarity to light templates).
    """
    p = params or DynParams()
    maze = maze or build_rose_maze()
    sig = np.atleast_2d(np.asarray(signal, dtype=np.float64))
    n = sig.shape[1]
    v = np.asarray(valid, dtype=bool)[:n]
    lo = np.asarray(light_on, dtype=bool)[:n]
    spd = np.asarray(speed, dtype=np.float64)[:n]
    cells = session_cells(x_maze, y_maze, v, maze, p.min_dwell_frames)[:n]
    direc, _, _ = frame_directions(cells, maze)
    run = running_mask(spd, v, fps) & (cells >= 0)
    t_ep, epoch = time_in_epoch(lo, fps)
    w = max(1, int(round(p.smooth_s * fps)))
    gap = int(round(p.gap_s * fps))
    df = pd.DataFrame(
        {
            "t_s": np.arange(n) / fps,
            "light": lo,
            "t_epoch": t_ep,
            "epoch": epoch,
            "running": run,
            "cell": cells,
            "direction": direc,
            "speed": spd,
            "abs_ahv": np.abs(np.asarray(ahv, dtype=np.float64)[:n]),
        }
    )
    counts = np.bincount(cells[run], minlength=maze.n_cells)
    pos_cls = np.flatnonzero(counts >= p.min_class_frames)
    pos_train = run & np.isin(cells, pos_cls)
    dir_train = run & (direc >= 0)
    nz = boxcar(sig, w).T
    bz = boxcar(behaviour_frames(hd_deg, spd, ahv)[:, :n], w).T
    hd_a = np.asarray(hd_deg, dtype=np.float64)[:n]
    ahv_a = np.asarray(ahv, dtype=np.float64)[:n]
    bdz = boxcar(behaviour_dir_frames(hd_a, spd, ahv_a, cells, maze.n_cells), w).T
    dist = maze.dist.astype(np.float64)
    for fs, x, xd in (("neural", nz, nz), ("behaviour", bz, bdz)):
        pr, cl = cv_proba(x, cells, pos_train, p.n_blocks, gap, p.C, p.max_iter)
        m = position_metrics(pr, cl, np.where(np.isin(cells, pos_cls), cells, -1), dist)
        df[f"{fs}_pos_err"] = m["err"]
        df[f"{fs}_pos_ptrue"] = m["ptrue"]
        df[f"{fs}_pos_experr"] = m["experr"]
        df[f"{fs}_pos_excess"] = cell_demean(m["err"], cells, run)
        pr, cl = cv_proba(xd, direc, dir_train, p.n_blocks, gap, p.C, p.max_iter)
        pout = pr[:, 1] if cl.size == 2 else np.full(n, np.nan)
        df[f"{fs}_dir_pout"] = pout
        df[f"{fs}_dir_ptrue"] = np.where(direc == 1, pout, np.where(direc == 0, 1 - pout, np.nan))
    zs = (nz - np.nanmean(nz, 0)) / np.where(np.nanstd(nz, 0) > 0, np.nanstd(nz, 0), 1.0)
    df["pv_r"], df["pv_spec"] = pv_similarity(
        zs, np.where(np.isin(cells, pos_cls), cells, -1), run & lo, p.n_blocks, gap
    )
    return df


def position_shift_null(
    signal: npt.ArrayLike,
    cells: npt.ArrayLike,
    train: npt.ArrayLike,
    shifts: npt.ArrayLike,
    smooth_frames: int,
    n_blocks: int,
    gap_frames: int,
    C: float = 1.0,
    max_iter: int = 300,
) -> np.ndarray:
    """Mean posterior of the true cell over *train* frames for each circular shift.

    The neural matrix is rolled by each shift relative to the labels and the
    whole cross-validated pipeline is refitted. Shift 0 gives the observed
    value; large shifts give a null with the autocorrelation of both
    preserved.
    """
    sig = np.atleast_2d(np.asarray(signal, dtype=np.float64))
    c = np.asarray(cells, dtype=np.int64)
    tr = np.asarray(train, dtype=bool)
    out = []
    for sh in np.asarray(shifts, dtype=np.int64):
        x = boxcar(np.roll(sig, int(sh), axis=1), smooth_frames).T
        pr, cl = cv_proba(x, c, tr, n_blocks, gap_frames, C, max_iter)
        ok = tr & np.isfinite(pr).all(1) & np.isin(c, cl)
        pt = pr[ok][np.arange(ok.sum()), np.searchsorted(cl, c[ok])]
        out.append(float(pt.mean()) if pt.size else np.nan)
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Question 1: drift and re-anchoring
# ---------------------------------------------------------------------------

DRIFT_METRICS = (
    "neural_pos_err",
    "neural_pos_excess",
    "neural_pos_ptrue",
    "neural_dir_ptrue",
    "pv_r",
    "pv_spec",
    "behaviour_pos_excess",
    "behaviour_dir_ptrue",
)


def drift_slopes(df: pd.DataFrame, metrics: tuple[str, ...] = DRIFT_METRICS) -> pd.DataFrame:
    """Per condition and metric: Spearman of metric vs time in epoch on running frames.

    ``rho`` is the raw Spearman correlation; ``prho`` partials speed, |AHV|
    and time in session. Positive ``rho`` for an error metric (or negative
    for ``ptrue`` / ``pv``) means decoding degrades with time in the epoch.
    """
    rows = []
    for cond, lval in (("dark", False), ("light", True)):
        d = df[df["running"] & (df["light"] == lval)]
        cov = d[["speed", "abs_ahv", "t_s"]].to_numpy(float)
        for m in metrics:
            y = d[m].to_numpy(float)
            t = d["t_epoch"].to_numpy(float)
            ok = np.isfinite(y)
            rows.append(
                {
                    "condition": cond,
                    "metric": m,
                    "n_frames": int(ok.sum()),
                    "rho": partial_spearman(t, y, None),
                    "prho": partial_spearman(t, y, cov),
                }
            )
    return pd.DataFrame(rows)


def epoch_profile(
    df: pd.DataFrame, metrics: tuple[str, ...], bin_s: float, epoch_s: float
) -> pd.DataFrame:
    """Mean of each metric on running frames per condition and time-in-epoch bin."""
    d = df[df["running"]].copy()
    d["bin"] = np.floor(d["t_epoch"] / bin_s) * bin_s
    d = d[d["bin"] < epoch_s]
    d["condition"] = np.where(d["light"], "light", "dark")
    g = d.groupby(["condition", "bin"])
    out = g[list(metrics)].mean().reset_index()
    out["n_frames"] = g.size().to_numpy()
    return out


def switch_table(
    df: pd.DataFrame,
    metrics: tuple[str, ...],
    fps: float,
    window_s: float = 15.0,
    min_frames: int = 10,
) -> pd.DataFrame:
    """Metric means around every light switch (running frames).

    For each switch from epoch ``e`` to ``e + 1``: ``pre_late`` = last
    ``window_s`` of epoch ``e``, ``post_early`` = first ``window_s`` of
    epoch ``e + 1``; ``pre_early`` = first ``window_s`` and ``pre_mid`` =
    ``[L - 2 window_s, L - window_s)`` of epoch ``e`` (length ``L``). For a
    dark-to-light switch, ``correction = pre_late - post_early`` and
    ``drift = pre_mid - pre_early`` share no frames, so their correlation
    across switches is not inflated by a common noise term.
    Windows with fewer than ``min_frames`` running frames are NaN.
    """
    rows = []
    run = df["running"].to_numpy(bool)
    ep = df["epoch"].to_numpy()
    lo = df["light"].to_numpy(bool)
    te = df["t_epoch"].to_numpy(float)
    ne = int(ep.max()) + 1 if ep.size else 0
    lengths = np.array([(ep == e).sum() / fps for e in range(ne)])
    for e in range(ne - 1):
        L = lengths[e]
        if 3 * window_s > L:
            continue
        cur, nxt = ep == e, ep == e + 1
        wins = {
            "pre_early": cur & (te < window_s),
            "pre_mid": cur & (te >= L - 2 * window_s) & (te < L - window_s),
            "pre_late": cur & (te >= L - window_s),
            "post_early": nxt & (te < window_s),
        }
        row: dict[str, Any] = {
            "epoch": e,
            "switch": "dark_to_light" if not lo[cur][0] else "light_to_dark",
        }
        for m in metrics:
            y = df[m].to_numpy(float)
            for wn, wm in wins.items():
                sel = wm & run & np.isfinite(y)
                row[f"{m}__{wn}"] = float(y[sel].mean()) if sel.sum() >= min_frames else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def switch_summary(sw: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    """Per switch type and metric: median correction and drift-correction Spearman.

    ``correction`` = ``pre_late - post_early``; ``drift`` = ``pre_mid -
    pre_early``; ``rho_drift_correction`` is their Spearman across switches.
    """
    rows = []
    for sw_type, g in sw.groupby("switch"):
        for m in metrics:
            corr = g[f"{m}__pre_late"] - g[f"{m}__post_early"]
            drift = g[f"{m}__pre_mid"] - g[f"{m}__pre_early"]
            ok = corr.notna() & drift.notna()
            rows.append(
                {
                    "switch": sw_type,
                    "metric": m,
                    "n_switches": int(corr.notna().sum()),
                    "median_correction": float(corr.median()) if corr.notna().any() else np.nan,
                    "rho_drift_correction": partial_spearman(drift[ok], corr[ok], None)
                    if ok.sum() >= 4
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Question 2: reversals
# ---------------------------------------------------------------------------


def _circ_mean_deg(h: np.ndarray) -> float:
    h = h[np.isfinite(h)]
    if h.size == 0:
        return np.nan
    r = np.deg2rad(h)
    return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def head_turn_frame(hd_deg: np.ndarray, lo: int, hi: int, h_in: float, h_out: float) -> int:
    """Final frame in ``[lo, hi)`` after which HD stays closer to *h_out* than *h_in*.

    Returns -1 if HD never settles closer to *h_out*.
    """
    h = np.deg2rad(np.asarray(hd_deg[lo:hi], dtype=np.float64))
    s = np.cos(h - np.deg2rad(h_out)) - np.cos(h - np.deg2rad(h_in))
    s = pd.Series(s).ffill().bfill().to_numpy()
    if s.size == 0 or not np.isfinite(s).all() or s[-1] <= 0:
        return -1
    neg = np.flatnonzero(s <= 0)
    return lo + (int(neg[-1]) + 1 if neg.size else 0)


def turn_events(
    cells: npt.ArrayLike,
    hd_deg: npt.ArrayLike,
    speed: npt.ArrayLike,
    ahv: npt.ArrayLike,
    light_on: npt.ArrayLike,
    fps: float,
    maze: RoseMaze,
    params: DynParams | None = None,
) -> pd.DataFrame:
    """Reversals and same-direction passes at non-dead-end maze cells.

    A stay ``k`` with neighbours ``k - 1`` and ``k + 1`` adjacent to it (gaps
    at most 0.5 s) and duration at most ``rev_max_stay_s`` is a
    ``reversal`` when ``cell[k-1] == cell[k+1]`` and the head turns by at
    least ``rev_min_turn_deg`` (circular mean HD over ``rev_hd_window_s``
    before entry vs after exit), and a ``pass`` when ``cell[k-1] !=
    cell[k+1]`` and arrival and departure tree directions agree.

    Columns: ``kind``, ``cell``, ``entry``, ``exit``, ``t_speedmin`` (frame of
    minimum smoothed speed in the stay), ``t_headturn`` (reversals; -1 if not
    found), ``dir_pre``, ``dir_post``, ``min_speed``, ``decel`` (mean speed in
    the 0.5 s before entry minus ``min_speed``), ``peak_abs_ahv``,
    ``dur_s``, ``turn_deg``, ``light_frac``.
    """
    p = params or DynParams()
    c = np.asarray(cells, dtype=np.int64)
    hd = np.asarray(hd_deg, dtype=np.float64)
    spd = boxcar(np.asarray(speed, dtype=np.float64), max(1, int(round(0.3 * fps))))
    aa = np.abs(np.asarray(ahv, dtype=np.float64))
    lo_ = np.asarray(light_on, dtype=float)
    depth = tree_depth(maze)
    dead = {maze.cell_to_idx[x] for x in maze.dead_ends}
    cell, st, en = cell_stays(c)
    g = max(1, int(round(0.5 * fps)))
    hw = max(1, int(round(p.rev_hd_window_s * fps)))
    n = c.size
    rows = []
    for k in range(1, cell.size - 1):
        a, b, q = int(cell[k - 1]), int(cell[k]), int(cell[k + 1])
        if b in dead or maze.dist[a, b] != 1 or maze.dist[b, q] != 1:
            continue
        if st[k] - en[k - 1] > g or st[k + 1] - en[k] > g:
            continue
        s0, s1 = int(st[k]), int(en[k])
        if (s1 - s0) / fps > p.rev_max_stay_s or s0 - hw < 0 or s1 + hw > n:
            continue
        d_pre = int(depth[b] > depth[a])
        d_post = int(depth[q] > depth[b])
        h_in = _circ_mean_deg(hd[s0 - hw : s0])
        h_out = _circ_mean_deg(hd[s1 : s1 + hw])
        turn = abs((h_out - h_in + 180) % 360 - 180) if np.isfinite(h_in + h_out) else np.nan
        if a == q:
            if not turn >= p.rev_min_turn_deg:
                continue
            kind = "reversal"
            th = head_turn_frame(hd, max(0, s0 - g), min(n, s1 + g), h_in, h_out)
        else:
            if d_pre != d_post:
                continue
            kind, th = "pass", -1
        seg = spd[s0:s1]
        if not np.isfinite(seg).any():
            continue
        tmin = s0 + int(np.nanargmin(seg))
        pre = spd[max(0, s0 - g) : s0]
        rows.append(
            {
                "kind": kind,
                "cell": b,
                "entry": s0,
                "exit": s1,
                "t_speedmin": tmin,
                "t_headturn": th,
                "dir_pre": d_pre,
                "dir_post": d_post,
                "min_speed": float(np.nanmin(seg)),
                "decel": float(np.nanmean(pre) - np.nanmin(seg))
                if np.isfinite(pre).any()
                else np.nan,
                "peak_abs_ahv": float(np.nanmax(aa[s0:s1]))
                if np.isfinite(aa[s0:s1]).any()
                else np.nan,
                "dur_s": (s1 - s0) / fps,
                "turn_deg": turn,
                "light_frac": float(np.nanmean(lo_[s0:s1])),
            }
        )
    cols = [
        "kind",
        "cell",
        "entry",
        "exit",
        "t_speedmin",
        "t_headturn",
        "dir_pre",
        "dir_post",
        "min_speed",
        "decel",
        "peak_abs_ahv",
        "dur_s",
        "turn_deg",
        "light_frac",
    ]
    return pd.DataFrame(rows, columns=cols)


def match_passes(
    ev: pd.DataFrame,
    covariates: tuple[str, ...] = ("min_speed", "decel", "peak_abs_ahv", "dur_s"),
) -> pd.DataFrame:
    """Greedy 1:1 nearest-neighbour match of each reversal to a pass.

    Distance is Euclidean on covariate ranks (pooled, scaled to [0, 1]) plus
    1 when the light condition (light fraction > 0.5) differs. Returns the
    reversals with a ``match`` column (row index of the matched pass, -1 if
    none left).
    """
    ev = ev.reset_index(drop=True)
    r = ev[ev["kind"] == "reversal"].copy()
    ps = ev[ev["kind"] == "pass"]
    z = np.column_stack(
        [stats.rankdata(ev[c].fillna(ev[c].median())) / max(len(ev), 1) for c in covariates]
    )
    lit = (ev["light_frac"] > 0.5).to_numpy()
    free = list(ps.index)
    match = []
    for i in r.index:
        if not free:
            match.append(-1)
            continue
        fa = np.asarray(free)
        d = np.linalg.norm(z[fa] - z[i], axis=1) + (lit[fa] != lit[i])
        j = int(np.argmin(d))
        match.append(int(fa[j]))
        free.pop(j)
    r["match"] = match
    return r


def aligned(x: npt.ArrayLike, frames: npt.ArrayLike, half: int) -> np.ndarray:
    """``(n_events, 2 half + 1)`` snippets of *x* around *frames* (NaN outside)."""
    a = np.asarray(x, dtype=np.float64)
    f = np.asarray(frames, dtype=np.int64)
    idx = f[:, None] + np.arange(-half, half + 1)[None, :]
    ok = (idx >= 0) & (idx < a.size)
    return np.where(ok, a[np.clip(idx, 0, a.size - 1)], np.nan)


def p_new(
    pout: npt.ArrayLike, frames: npt.ArrayLike, new_dir: npt.ArrayLike, half: int
) -> np.ndarray:
    """Aligned probability of the post-event (or opposite) direction."""
    s = aligned(pout, frames, half)
    nd = np.asarray(new_dir, dtype=np.int64)[:, None]
    return np.where(nd == 1, s, 1 - s)


def half_rise_time(
    curve: npt.ArrayLike,
    lags_s: npt.ArrayLike,
    base: tuple[float, float] = (-2.0, -1.2),
    plateau: tuple[float, float] = (1.0, 2.0),
    min_rise: float = 0.05,
) -> float:
    """Time at which *curve* last crosses upward half-way between baseline and plateau.

    Baseline and plateau are curve means over the given lag ranges; the
    crossing is linearly interpolated and searched before the plateau
    range. NaN when the rise is below *min_rise* or no crossing is found.
    """
    y = np.asarray(curve, dtype=np.float64)
    t = np.asarray(lags_s, dtype=np.float64)
    bm = np.nanmean(y[(t >= base[0]) & (t <= base[1])])
    pm = np.nanmean(y[(t >= plateau[0]) & (t <= plateau[1])])
    if not np.isfinite(bm + pm) or pm - bm < min_rise:
        return float("nan")
    half = 0.5 * (bm + pm)
    sel = np.flatnonzero(t < plateau[0])
    ys = y[sel]
    below = np.flatnonzero(ys <= half)
    if below.size == 0 or below[-1] + 1 >= ys.size:
        return float("nan")
    i = below[-1]
    y0, y1 = ys[i], ys[i + 1]
    t0, t1 = t[sel][i], t[sel][i + 1]
    return float(t0 + (half - y0) / (y1 - y0) * (t1 - t0)) if y1 != y0 else float(t0)


def lagged_xcorr_peak(
    a: npt.ArrayLike, b: npt.ArrayLike, mask: npt.ArrayLike, max_lag: int
) -> int:
    """Lag (frames) maximising Spearman corr of ``a[t]`` with ``b[t + lag]`` over *mask*.

    Positive lag means *a* leads *b*.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    n = x.size
    best, best_r = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        i = np.arange(max(0, -lag), min(n, n - lag))
        sel = i[m[i] & m[i + lag] & np.isfinite(x[i]) & np.isfinite(y[i + lag])]
        if sel.size < 10:
            continue
        r = stats.spearmanr(x[sel], y[sel + lag]).statistic
        if r > best_r:
            best, best_r = lag, r
    return best


def reversal_session(
    signal: npt.ArrayLike,
    x_maze: npt.ArrayLike,
    y_maze: npt.ArrayLike,
    speed: npt.ArrayLike,
    hd_deg: npt.ArrayLike,
    ahv: npt.ArrayLike,
    light_on: npt.ArrayLike,
    valid: npt.ArrayLike,
    fps: float,
    params: DynParams | None = None,
    maze: RoseMaze | None = None,
) -> dict[str, Any]:
    """Question 2 for one session.

    The direction decoders (neural and behaviour-only, causal boxcar of
    ``rev_smooth_s``) are trained on labelled running frames farther than
    ``rev_exclude_s`` from every reversal. Returns ``events``
    (reversals with matches and per-event pre-window indices), ``curves``
    (session-mean aligned curves per decoder, alignment and event kind) and
    ``summary`` (per-session scalars, see :func:`reversal_summary_row`).
    """
    p = params or DynParams()
    maze = maze or build_rose_maze()
    sig = np.atleast_2d(np.asarray(signal, dtype=np.float64))
    n = sig.shape[1]
    v = np.asarray(valid, dtype=bool)[:n]
    spd = np.asarray(speed, dtype=np.float64)[:n]
    cells = session_cells(x_maze, y_maze, v, maze, p.min_dwell_frames)[:n]
    direc, _, _ = frame_directions(cells, maze)
    run = running_mask(spd, v, fps) & (cells >= 0)
    ev = turn_events(cells, hd_deg, spd, ahv, light_on, fps, maze, p)
    ex = max(1, int(round(p.rev_exclude_s * fps)))
    near = np.zeros(n, dtype=bool)
    is_rev = (ev["kind"] == "reversal").to_numpy()
    for a, b in zip(ev["entry"][is_rev], ev["exit"][is_rev], strict=True):
        near[max(0, a - ex) : b + ex] = True
    train = run & (direc >= 0) & ~near
    w = max(1, int(round(p.rev_smooth_s * fps)))
    gap = int(round(p.gap_s * fps))
    feats = {
        "neural": boxcar(sig, w, causal=True).T,
        "behaviour": boxcar(
            behaviour_dir_frames(
                np.asarray(hd_deg, float)[:n], spd, np.asarray(ahv, float)[:n], cells, maze.n_cells
            ),
            w,
            causal=True,
        ).T,
    }
    pout = {}
    for fs, x in feats.items():
        pr, cl = cv_proba(x, direc, train, p.n_blocks, gap, p.C, p.max_iter)
        pout[fs] = pr[:, 1] if cl.size == 2 else np.full(n, np.nan)
    half = int(round(p.rev_lag_s * fps))
    lags = np.arange(-half, half + 1) / fps
    rev = match_passes(ev) if (ev["kind"] == "reversal").any() else ev.iloc[:0].assign(match=[])
    passes = ev.loc[rev["match"][rev["match"] >= 0]] if len(rev) else ev.iloc[:0]
    pre = (lags >= p.pre_window_s[0]) & (lags <= p.pre_window_s[1])
    curves = []
    rev = rev.copy()
    for fs in feats:
        for kind, d, frame_col in (
            ("reversal", rev, "t_speedmin"),
            ("pass_matched", passes, "t_speedmin"),
            ("reversal_headturn", rev[rev["t_headturn"] >= 0], "t_headturn"),
        ):
            if len(d) == 0:
                continue
            # Passes: the would-be new direction is the opposite of the current one.
            new_dir = d["dir_post"] if kind != "pass_matched" else 1 - d["dir_pre"]
            m = p_new(pout[fs], d[frame_col].to_numpy(), new_dir.to_numpy(), half)
            if kind == "reversal":
                rev[f"{fs}_pre"] = np.nanmean(m[:, pre], 1)
            if kind == "pass_matched":
                pm = np.full(len(rev), np.nan)
                pm[(rev["match"] >= 0).to_numpy()] = np.nanmean(m[:, pre], 1)
                rev[f"{fs}_pre_match"] = pm
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mc = np.nanmean(m, 0)
            for lag, val in zip(lags, mc, strict=True):
                curves.append(
                    {"feature_set": fs, "kind": kind, "lag_s": lag, "p_new": val, "n": len(d)}
                )
    lag_ref = lagged_xcorr_peak(
        pout["neural"], pout["behaviour"], run & ~near, int(round(2.0 * fps))
    )
    cdf = pd.DataFrame(curves)
    return {
        "events": rev,
        "curves": cdf,
        "summary": reversal_summary_row(rev, passes, cdf, lag_ref / fps, p),
    }


def reversal_summary_row(
    rev: pd.DataFrame, passes: pd.DataFrame, curves: pd.DataFrame, lag_ref_s: float, p: DynParams
) -> dict[str, Any]:
    """Session scalars for question 2.

    ``{fs}_anticip`` = median over reversals of (pre-window P(new) at the
    reversal minus that at its matched pass), aligned to speed minimum;
    ``{fs}_t_half`` / ``{fs}_t_half_headturn`` = half-rise time of the mean
    reversal curve aligned to speed minimum / head turn; ``lead_vs_behaviour``
    = behaviour minus neural half-rise time (positive: neural flips first);
    ``lag_ref_s`` = lag of the neural vs behaviour decoded direction
    cross-correlation on ordinary running (positive: neural leads).
    """
    out: dict[str, Any] = {
        "n_reversals": int(len(rev)),
        "n_matched": int((rev["match"] >= 0).sum()) if len(rev) else 0,
        "n_headturn": int((rev["t_headturn"] >= 0).sum()) if len(rev) else 0,
        "lag_ref_s": lag_ref_s,
    }
    for c in ("min_speed", "decel", "peak_abs_ahv", "dur_s"):
        out[f"rev_median_{c}"] = float(rev[c].median()) if len(rev) else np.nan
        out[f"pass_median_{c}"] = float(passes[c].median()) if len(passes) else np.nan
    for fs in ("neural", "behaviour"):
        if len(rev) >= p.min_events and f"{fs}_pre_match" in rev:
            out[f"{fs}_anticip"] = float(np.nanmedian(rev[f"{fs}_pre"] - rev[f"{fs}_pre_match"]))
        else:
            out[f"{fs}_anticip"] = np.nan
        for kind, key in (("reversal", "t_half"), ("reversal_headturn", "t_half_headturn")):
            c = (
                curves[(curves.get("feature_set") == fs) & (curves.get("kind") == kind)]
                if len(curves)
                else curves
            )
            ok = len(c) and int(c["n"].iloc[0]) >= p.min_events
            out[f"{fs}_{key}"] = half_rise_time(c["p_new"], c["lag_s"]) if ok else np.nan
    out["lead_vs_behaviour"] = out["behaviour_t_half"] - out["neural_t_half"]
    out["lead_vs_behaviour_headturn"] = (
        out["behaviour_t_half_headturn"] - out["neural_t_half_headturn"]
    )
    return out


# ---------------------------------------------------------------------------
# Question 3: junction departures
# ---------------------------------------------------------------------------


def junction_departures(
    cells: npt.ArrayLike, maze: RoseMaze, fps: float, n_back: int = 4, max_gap_s: float = 0.5
) -> pd.DataFrame:
    """T-junction departures labelled ``repeat`` (1) or ``new`` (0).

    Node stays are stays in T-junctions or dead ends. For a junction stay
    with adjacent arrival cell ``p`` and exit cell ``q``, the destination is
    the next node stay. ``repeat`` = U-turn (``q == p``), destination equal
    to this junction, or destination among the previous ``n_back`` node
    stays. Columns: ``junction``, ``depart`` (one past the last junction
    frame), ``arrive``, ``uturn``, ``dest``, ``repeat``.
    """
    c = np.asarray(cells, dtype=np.int64)
    cell, st, en = cell_stays(c)
    node = {maze.cell_to_idx[x] for x in maze.dead_ends} | {
        maze.cell_to_idx[x] for x in maze.junctions
    }
    tj = {maze.cell_to_idx[x] for x in maze.junctions if maze.node_types[x] == "t_junction"}
    g = max(1, int(round(max_gap_s * fps)))
    node_k = [k for k in range(cell.size) if int(cell[k]) in node]
    pos = {k: i for i, k in enumerate(node_k)}
    rows = []
    for k in range(1, cell.size - 1):
        j, a, q = int(cell[k]), int(cell[k - 1]), int(cell[k + 1])
        if j not in tj or maze.dist[a, j] != 1 or maze.dist[q, j] != 1:
            continue
        if st[k] - en[k - 1] > g or st[k + 1] - en[k] > g:
            continue
        i = pos[k]
        if i + 1 >= len(node_k):
            continue
        dest = int(cell[node_k[i + 1]])
        recent = {int(cell[node_k[m]]) for m in range(max(0, i - n_back), i)}
        rep = (q == a) or dest == j or dest in recent
        rows.append(
            {
                "junction": j,
                "depart": int(en[k]),
                "arrive": int(st[k]),
                "uturn": q == a,
                "dest": dest,
                "repeat": int(rep),
            }
        )
    return pd.DataFrame(rows, columns=["junction", "depart", "arrive", "uturn", "dest", "repeat"])


DEPARTURE_METRICS = (
    "neural_pos_ptrue",
    "neural_pos_err",
    "neural_pos_excess",
    "behaviour_pos_ptrue",
)


def departure_features(
    dep: pd.DataFrame, df: pd.DataFrame, fps: float, window_s: float = 2.0
) -> pd.DataFrame:
    """Window means (``[depart - window_s, depart)``) of decoding metrics and covariates."""
    w = max(1, int(round(window_s * fps)))
    out = dep.copy()
    n = len(df)
    cols = (*DEPARTURE_METRICS, "speed", "abs_ahv")
    arrs = {c: df[c].to_numpy(float) for c in cols}
    lo = df["light"].to_numpy(float)
    for c in cols:
        out[c] = [
            np.nanmean(arrs[c][max(0, d - w) : d])
            if np.isfinite(arrs[c][max(0, d - w) : d]).any()
            else np.nan
            for d in dep["depart"]
        ]
    out["light_frac"] = [lo[max(0, d - w) : min(n, d)].mean() for d in dep["depart"]]
    out["t_s"] = dep["depart"] / fps
    return out


def departure_effects(feat: pd.DataFrame, min_per_class: int = 5) -> pd.DataFrame:
    """Per condition and metric: repeat-vs-new effect within the session.

    ``diff`` = median(repeat) - median(new); ``prho`` = partial Spearman of
    the metric with the repeat label, partialling window speed, |AHV|, time
    in session and junction identity (dummies).
    """
    rows = []
    for cond, sel in (
        ("light", feat["light_frac"] >= 0.9),
        ("dark", feat["light_frac"] <= 0.1),
        ("all", np.ones(len(feat), dtype=bool)),
    ):
        d = feat[sel]
        n1, n0 = int((d["repeat"] == 1).sum()), int((d["repeat"] == 0).sum())
        dummies = pd.get_dummies(d["junction"], drop_first=True).to_numpy(float)
        cov = np.column_stack([d[["speed", "abs_ahv", "t_s"]].to_numpy(float), dummies])
        if cond == "all":
            cov = np.column_stack([cov, d["light_frac"].to_numpy(float)])
        for m in DEPARTURE_METRICS:
            ok = min(n1, n0) >= min_per_class
            y = d[m].to_numpy(float)
            r = d["repeat"].to_numpy(int)
            rows.append(
                {
                    "condition": cond,
                    "metric": m,
                    "n_repeat": n1,
                    "n_new": n0,
                    "diff": float(np.nanmedian(y[r == 1]) - np.nanmedian(y[r == 0]))
                    if ok
                    else np.nan,
                    "prho": partial_spearman(r, y, cov) if ok else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Question 4: epochs
# ---------------------------------------------------------------------------


EPOCH_METRICS = ("neural_pos_err", "neural_pos_excess", "neural_dir_ptrue", "pv_spec")


def epoch_exploration(
    df: pd.DataFrame,
    maze: RoseMaze,
    fps: float,
    rng: np.random.Generator,
    n_sims: int = 500,
    min_steps: int = 5,
    min_run_frames: int = 30,
    min_len_s: float = 50.0,
) -> pd.DataFrame:
    """Per light epoch: exploration efficiency and decoding quality.

    Exploration: the epoch's sequence of maze-cell stays, transitions between
    adjacent cells counted as steps; ``coverage_z`` = unique cells vs uniform
    random walks of the same number of steps from the same start cell
    (:func:`hm2p.maze.exploration_complexity.random_walk_coverage_null`);
    ``new_per_step`` = (unique cells - 1) / steps. Decoding: means of
    :data:`EPOCH_METRICS` over running frames. Covariates: ``speed`` (mean
    over running frames), ``frac_running``, ``t_start_s``.
    """
    adj = build_adjacency_indices(maze)
    rows = []
    for e, g in df.groupby("epoch"):
        if len(g) / fps < min_len_s:
            continue
        c = g["cell"].to_numpy(np.int64)
        seq, _, _ = cell_stays(c)
        steps = int(np.sum(maze.dist[seq[:-1], seq[1:]] == 1)) if seq.size > 1 else 0
        run = g["running"].to_numpy(bool)
        row: dict[str, Any] = {
            "epoch": int(e),
            "light": bool(g["light"].iloc[0]),
            "t_start_s": float(g["t_s"].iloc[0]),
            "n_steps": steps,
            "n_unique": int(np.unique(seq).size),
            "n_run_frames": int(run.sum()),
            "frac_running": float(run.mean()),
            "speed": float(g["speed"].to_numpy(float)[run].mean()) if run.any() else np.nan,
        }
        if steps >= min_steps:
            null = random_walk_coverage_null(adj, int(seq[0]), steps, n_sims, rng)
            row["coverage_z"] = coverage_z_vs_null(row["n_unique"], null)
            row["new_per_step"] = (row["n_unique"] - 1) / steps
        else:
            row["coverage_z"] = row["new_per_step"] = np.nan
        for m in EPOCH_METRICS:
            y = g[m].to_numpy(float)[run]
            row[m] = (
                float(np.nanmean(y))
                if run.sum() >= min_run_frames and np.isfinite(y).any()
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def epoch_correlations(ep: pd.DataFrame, min_epochs: int = 6) -> pd.DataFrame:
    """Within-session partial Spearman of each decoding metric vs ``coverage_z``.

    Partialling ``speed``, ``frac_running`` and ``t_start_s`` (plus ``light``
    for the pooled condition). NaN with fewer than *min_epochs* epochs.
    """
    rows = []
    for cond, sel in (
        ("light", ep["light"]),
        ("dark", ~ep["light"]),
        ("all", np.ones(len(ep), bool)),
    ):
        d = ep[np.asarray(sel, dtype=bool)]
        covs = ["speed", "frac_running", "t_start_s"] + (["light"] if cond == "all" else [])
        cov = d[covs].to_numpy(float)
        for m in EPOCH_METRICS:
            ok = d[m].notna() & d["coverage_z"].notna()
            rows.append(
                {
                    "condition": cond,
                    "metric": m,
                    "n_epochs": int(ok.sum()),
                    "rho": partial_spearman(d["coverage_z"], d[m], None)
                    if ok.sum() >= min_epochs
                    else np.nan,
                    "prho": partial_spearman(d["coverage_z"], d[m], cov)
                    if ok.sum() >= min_epochs
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Across-animal statistics
# ---------------------------------------------------------------------------


def wilcoxon_summary(values: npt.ArrayLike) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank vs 0 with matched-pairs rank-biserial r.

    Kerby DS. 2014. Comprehensive Psychology 3:11.IT.3.1. doi:10.2466/11.IT.3.1
    """
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    out = {
        "n": int(a.size),
        "median": float(np.median(a)) if a.size else np.nan,
        "n_pos": int((a > 0).sum()),
        "p": np.nan,
        "r_rb": np.nan,
    }
    nz = a[a != 0]
    if nz.size >= 3:
        out["p"] = float(stats.wilcoxon(nz).pvalue)
        rk = stats.rankdata(np.abs(nz))
        out["r_rb"] = float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum())
    return out


def animal_test(df: pd.DataFrame, value: str, animal: str = "animal_id") -> dict[str, Any]:
    """:func:`wilcoxon_summary` on per-animal medians of *value*, plus session-level p."""
    d = df[np.isfinite(df[value].to_numpy(float))]
    per = d.groupby(animal)[value].median()
    res = {f"animal_{k}": v for k, v in wilcoxon_summary(per).items()}
    res.update({f"session_{k}": v for k, v in wilcoxon_summary(d[value]).items()})
    return res


def bh_fdr(p: npt.ArrayLike) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values (NaN preserved).

    Benjamini Y, Hochberg Y. 1995. "Controlling the false discovery rate."
    J R Stat Soc B 57:289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
    """
    pa = np.asarray(p, dtype=np.float64)
    out = np.full(pa.size, np.nan)
    ok = np.flatnonzero(np.isfinite(pa))
    if ok.size == 0:
        return out
    order = ok[np.argsort(pa[ok])]
    m = ok.size
    adj = pa[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.minimum(adj, 1.0)
    return out
