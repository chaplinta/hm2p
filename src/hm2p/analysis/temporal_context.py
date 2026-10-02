"""Time and episode-context coding in calcium-imaging populations.

Tests whether population activity carries information about *when* in the
session (seconds to minutes) and *which* light/dark episode a moment belongs
to, independently of moment-to-moment behavioural variables. Activity is
first averaged in coarse time bins (default 10 s); all analyses then operate
on the binned population matrix ``(n_rois, n_bins)``.

Analyses
--------
- **Time decoding** — cross-validated regression of session time from
  population vectors with contiguous block folds; inference against a
  circular-shift null of the target (Mau et al. 2018; Tsao et al. 2018).
- **Epoch identity** — leave-one-bin-out nearest-centroid classification of
  which light/dark epoch a bin belongs to, among epochs of the same light
  state; inference against independent per-cell circular shifts of the
  binned activity (preserves each cell's autocorrelation and drift
  magnitude, breaks population alignment to the epochs).
- **Temporal distance** — Spearman correlation between the distance of epoch
  mean population vectors and their separation in time, with an epoch-order
  permutation null (Rubin et al. 2015; Ziv et al. 2013).
- **Per-cell drift** (circular-shift null) and **per-cell epoch selectivity**
  (Kruskal-Wallis H, boundary-misaligned label-shift null on fine bins).
- **First vs later dark epoch** mean activity (novelty of darkness), via
  :func:`hm2p.analysis.transitions.first_vs_later_epochs`.
- **Matched-N decoding** — repeated subsampling to a fixed cell count so that
  populations of different size can be compared.

All inference is non-parametric (permutation / rank-based). The ridge
regressor is used only as a decoder; its output is never used for parametric
inference.

References
----------
Mau W, Sullivan DW, Kinsky NR, Hasselmo ME, Howard MW, Eichenbaum H. 2018.
"The same hippocampal CA1 population simultaneously codes temporal information
over multiple timescales." Current Biology. doi:10.1016/j.cub.2018.03.051

Rubin A, Geva N, Sheintuch L, Ziv Y. 2015. "Hippocampal ensemble dynamics
timestamp events in long-term memory." eLife. doi:10.7554/eLife.12247

Ziv Y, Burns LD, Cocker ED, Hamel EO, Ghosh KK, Kitch LJ, El Gamal A,
Schnitzer MJ. 2013. "Long-term dynamics of CA1 hippocampal place codes."
Nature Neuroscience. doi:10.1038/nn.3329

Tsao A, Sugar J, Lu L, Wang C, Knierim JJ, Moser MB, Moser EI. 2018.
"Integrating time from experience in the lateral entorhinal cortex." Nature.
doi:10.1038/s41586-018-0459-6
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

log = logging.getLogger(__name__)

__all__ = [
    "MIN_VALID_FRACTION",
    "bin_activity",
    "epoch_index",
    "pure_epoch_bins",
    "time_decoding",
    "time_decoding_null",
    "epoch_identity_decoding",
    "temporal_distance",
    "per_cell_drift",
    "per_cell_epoch_selectivity",
    "first_vs_later_dark",
    "matched_n_decoding",
]

#: Minimum fraction of valid frames for a time bin to be used.
MIN_VALID_FRACTION = 0.5

_RIDGE_ALPHAS = np.logspace(-2, 4, 13)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Return a Generator from a Generator, seed or None."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _as_2d(x: npt.ArrayLike, name: str) -> np.ndarray:
    """Coerce to a float (n_rois, n) matrix; 1-D input becomes one row."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D; got shape {arr.shape}")
    return arr


def _perm_p_greater(observed: float, null: np.ndarray) -> float:
    """One-sided permutation p with the +1 correction (larger = more extreme)."""
    null = np.asarray(null, dtype=np.float64)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or null.size == 0:
        return float("nan")
    return float((1 + np.sum(null >= observed)) / (1 + null.size))


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    """Z-score each row; constant rows become zeros."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return (x - mu) / sd


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho, NaN when either input is constant or too short."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(a[ok], b[ok]).statistic)


def _shift_values(n: int, n_perms: int, rng: np.random.Generator) -> np.ndarray:
    """Random circular shifts excluding the smallest 10% of either direction."""
    lo = max(1, int(np.ceil(0.1 * n)))
    hi = n - lo
    if hi <= lo:
        lo, hi = 1, max(2, n)
    return rng.integers(lo, hi, size=n_perms)


def _circular_partition(labels: np.ndarray, shift: int) -> np.ndarray:
    """Circularly shifted block labels with the wrapped block split in two.

    After a circular shift, one block wraps from the end of the session to the
    start. Treating its two pieces as one group would join the first and last
    minutes of the session, which penalises the null under slow drift. The
    leading piece is given its own id ``-1`` so it forms a separate group.
    """
    rolled = np.roll(np.asarray(labels).astype(int), int(shift))
    if rolled.size > 1 and rolled[0] == rolled[-1]:
        differs = np.flatnonzero(rolled != rolled[0])
        if differs.size:
            rolled[: differs[0]] = -1
    return rolled


def _subtract_state_median(x: np.ndarray, state: np.ndarray | None) -> np.ndarray:
    """Remove each cell's median per light state (columns of *x* = bins)."""
    if state is None:
        return x
    out = x.copy()
    for st in np.unique(state):
        sel = state == st
        out[:, sel] -= np.median(x[:, sel], axis=1, keepdims=True)
    return out


def _valid_columns(binned: np.ndarray) -> np.ndarray:
    """Bins in which every cell is finite."""
    return np.all(np.isfinite(binned), axis=0)


# ---------------------------------------------------------------------------
# Binning and epoch labels
# ---------------------------------------------------------------------------


def bin_activity(
    signal: npt.ArrayLike,
    fps: float,
    bin_s: float = 10.0,
    mask: npt.ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average activity in fixed time bins.

    Parameters
    ----------
    signal : (n_rois, n_frames) or (n_frames,) float
        Activity per frame. NaN frames are ignored per ROI.
    fps : float
        Frame rate (Hz).
    bin_s : float
        Bin width in seconds.
    mask : (n_frames,) bool or None
        Frames to use (e.g. ``~bad_behav``). ``None`` uses all frames.

    Returns
    -------
    binned : (n_rois, n_bins) float
        Mean activity per bin; NaN in invalid bins.
    bin_centres_s : (n_bins,) float
        Bin centre times from session start (s).
    bin_valid : (n_bins,) bool
        True when at least ``MIN_VALID_FRACTION`` of the nominal frames in the
        bin are valid (masked-in and finite in every ROI).

    Raises
    ------
    ValueError
        If ``fps`` or ``bin_s`` are not positive or ``mask`` has the wrong length.
    """
    x = _as_2d(signal, "signal")
    if not fps > 0 or not bin_s > 0:
        raise ValueError(f"fps and bin_s must be positive; got {fps}, {bin_s}")
    n_rois, n_frames = x.shape
    m = np.ones(n_frames, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if m.shape != (n_frames,):
        raise ValueError(f"mask must have shape ({n_frames},); got {m.shape}")
    if n_frames == 0:
        return np.zeros((n_rois, 0)), np.zeros(0), np.zeros(0, dtype=bool)

    bin_of_frame = np.floor(np.arange(n_frames) / float(fps) / float(bin_s)).astype(int)
    n_bins = int(bin_of_frame[-1]) + 1
    frame_ok = m & np.all(np.isfinite(x), axis=0)

    counts = np.bincount(bin_of_frame[frame_ok], minlength=n_bins).astype(np.float64)
    sums = np.zeros((n_rois, n_bins))
    xv = np.where(frame_ok[None, :], x, 0.0)
    for r in range(n_rois):
        sums[r] = np.bincount(bin_of_frame, weights=xv[r], minlength=n_bins)

    expected = float(bin_s) * float(fps)
    bin_valid = counts >= MIN_VALID_FRACTION * expected
    with np.errstate(invalid="ignore", divide="ignore"):
        binned = sums / counts[None, :]
    binned[:, ~bin_valid] = np.nan
    centres = (np.arange(n_bins) + 0.5) * float(bin_s)
    return binned, centres, bin_valid


def epoch_index(
    light_on: npt.ArrayLike,
    fps: float,
    bin_centres_s: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Light/dark epoch membership of each time bin.

    Epochs are the runs of constant light state delimited by
    :func:`hm2p.analysis.anchoring.find_transitions`; they are numbered
    0, 1, 2, ... in time order across both light states. A bin is assigned to
    the epoch containing its centre frame.

    Parameters
    ----------
    light_on : (n_frames,) bool
    fps : float
    bin_centres_s : (n_bins,) float

    Returns
    -------
    epoch_idx : (n_bins,) int
        Epoch number of each bin.
    epoch_light : (n_bins,) bool
        Light state of that epoch.

    Raises
    ------
    ValueError
        If ``light_on`` is empty or ``fps`` is not positive.
    """
    from hm2p.analysis.anchoring import find_transitions

    light = np.asarray(light_on, dtype=bool)
    if light.ndim != 1 or light.size == 0:
        raise ValueError("light_on must be a non-empty 1-D array")
    if not fps > 0:
        raise ValueError(f"fps must be positive; got {fps}")
    tr = find_transitions(light)
    bounds = np.sort(np.concatenate([tr["dark_to_light"], tr["light_to_dark"]]))
    centres = np.asarray(bin_centres_s, dtype=np.float64)
    frames = np.clip(np.floor(centres * float(fps)).astype(int), 0, light.size - 1)
    idx = np.searchsorted(bounds, frames, side="right").astype(int)
    return idx, light[frames]


def pure_epoch_bins(
    light_on: npt.ArrayLike,
    fps: float,
    bin_centres_s: npt.ArrayLike,
    bin_s: float,
) -> np.ndarray:
    """Bins that lie entirely within one light/dark epoch.

    A bin that straddles a light switch mixes two epochs; these are excluded
    from epoch-identity and epoch-selectivity analyses.

    Parameters
    ----------
    light_on : (n_frames,) bool
    fps : float
    bin_centres_s : (n_bins,) float
    bin_s : float

    Returns
    -------
    (n_bins,) bool
    """
    centres = np.asarray(bin_centres_s, dtype=np.float64)
    start, _ = epoch_index(light_on, fps, centres - bin_s / 2.0)
    end, _ = epoch_index(light_on, fps, centres + bin_s / 2.0 - 1.0 / float(fps))
    return np.asarray(start == end)


# ---------------------------------------------------------------------------
# Time decoding
# ---------------------------------------------------------------------------


def _fold_ids(n: int, n_folds: int, block: bool, rng: np.random.Generator) -> np.ndarray:
    """Fold label per sample: contiguous blocks or a random partition."""
    folds = np.arange(n) * n_folds // n
    if not block:
        folds = rng.permutation(folds)
    return folds


def _ridge_cv_predict(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold ridge predictions; features standardised on the training fold."""
    from sklearn.linear_model import RidgeCV

    pred = np.full(y.shape, np.nan)
    for f in np.unique(folds):
        test = folds == f
        train = ~test
        if train.sum() < 2:
            continue
        mu = x[train].mean(axis=0)
        sd = x[train].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        model = RidgeCV(alphas=_RIDGE_ALPHAS)
        model.fit((x[train] - mu) / sd, y[train])
        pred[test] = model.predict((x[test] - mu) / sd)
    return pred


def time_decoding(
    binned: npt.ArrayLike,
    target: npt.ArrayLike,
    n_folds: int = 5,
    block: bool = True,
    rng: np.random.Generator | int | None = None,
) -> dict[str, Any]:
    """Cross-validated decoding of a continuous target (session time).

    A ridge regressor (``sklearn.linear_model.RidgeCV``) is fit on population
    vectors of the training folds and evaluated on the held-out fold. With
    ``block=True`` folds are contiguous in time, so the decoder must
    generalise to unseen periods rather than interpolate between neighbouring
    bins (Mau et al. 2018).

    Parameters
    ----------
    binned : (n_rois, n_bins) float
        Binned activity; bins with any NaN are dropped.
    target : (n_bins,) float
        Value to decode, typically bin centre time (s).
    n_folds : int
        Number of cross-validation folds (>= 2).
    block : bool
        Contiguous folds (True) or random folds (False).
    rng : Generator, int or None
        Only used when ``block=False``.

    Returns
    -------
    dict
        ``median_abs_error`` — median |pred - true| (target units).
        ``rho`` — Spearman rho between prediction and target.
        ``predicted`` — (n_used,) out-of-fold predictions.
        ``true`` — (n_used,) target values used.
        ``n_bins`` — number of bins used.

    Raises
    ------
    ValueError
        If shapes disagree or ``n_folds`` < 2.

    References
    ----------
    Mau W et al. 2018. "The same hippocampal CA1 population simultaneously
    codes temporal information over multiple timescales." Current Biology.
    doi:10.1016/j.cub.2018.03.051
    """
    x = _as_2d(binned, "binned")
    y = np.asarray(target, dtype=np.float64)
    if y.shape != (x.shape[1],):
        raise ValueError(f"target must have shape ({x.shape[1]},); got {y.shape}")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2; got {n_folds}")
    ok = _valid_columns(x) & np.isfinite(y)
    xs, ys = x[:, ok].T, y[ok]
    nan_out: dict[str, Any] = {
        "median_abs_error": float("nan"),
        "rho": float("nan"),
        "predicted": np.zeros(0),
        "true": ys,
        "n_bins": int(ys.size),
    }
    if ys.size < 2 * n_folds or xs.shape[1] == 0:
        return nan_out
    folds = _fold_ids(ys.size, n_folds, block, _as_rng(rng))
    pred = _ridge_cv_predict(xs, ys, folds)
    return {
        "median_abs_error": float(np.nanmedian(np.abs(pred - ys))),
        "rho": _spearman(pred, ys),
        "predicted": pred,
        "true": ys,
        "n_bins": int(ys.size),
    }


def time_decoding_null(
    binned: npt.ArrayLike,
    target: npt.ArrayLike,
    n_perms: int = 200,
    n_folds: int = 5,
    block: bool = True,
    rng: np.random.Generator | int | None = None,
) -> dict[str, Any]:
    """Permutation test for :func:`time_decoding` by circular target shifts.

    The target is circularly shifted relative to the population bins (shift
    drawn away from 0), preserving the autocorrelation of both the activity
    and the target while breaking their alignment.

    Parameters
    ----------
    binned : (n_rois, n_bins) float
    target : (n_bins,) float
    n_perms : int
        Number of shifts.
    n_folds, block : see :func:`time_decoding`.
    rng : Generator, int or None

    Returns
    -------
    dict
        Observed ``rho`` and ``median_abs_error``; ``null_rho`` and
        ``null_error`` (n_perms,); ``p_rho`` (one-sided, rho larger than
        null) and ``p_error`` (error smaller than null).

    References
    ----------
    Rubin A et al. 2015. "Hippocampal ensemble dynamics timestamp events in
    long-term memory." eLife. doi:10.7554/eLife.12247
    """
    gen = _as_rng(rng)
    x = _as_2d(binned, "binned")
    y = np.asarray(target, dtype=np.float64)
    obs = time_decoding(x, y, n_folds=n_folds, block=block, rng=gen)
    ok = _valid_columns(x) & np.isfinite(y)
    xv, yv = x[:, ok], y[ok]
    null_rho = np.full(n_perms, np.nan)
    null_err = np.full(n_perms, np.nan)
    if np.isfinite(obs["rho"]) and yv.size >= 3:
        for i, s in enumerate(_shift_values(yv.size, n_perms, gen)):
            res = time_decoding(xv, np.roll(yv, int(s)), n_folds=n_folds, block=block, rng=gen)
            null_rho[i] = res["rho"]
            null_err[i] = res["median_abs_error"]
    return {
        "rho": obs["rho"],
        "median_abs_error": obs["median_abs_error"],
        "n_bins": obs["n_bins"],
        "null_rho": null_rho,
        "null_error": null_err,
        "p_rho": _perm_p_greater(obs["rho"], null_rho),
        "p_error": _perm_p_greater(-obs["median_abs_error"], -null_err),
    }


def matched_n_decoding(
    binned: npt.ArrayLike,
    target: npt.ArrayLike,
    n_cells: int,
    n_draws: int = 20,
    rng: np.random.Generator | int | None = None,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Time decoding from repeated random subsets of a fixed number of cells.

    Decoding accuracy grows with population size; drawing the same number of
    cells for every session/population makes the comparison fair.

    Parameters
    ----------
    binned : (n_rois, n_bins) float
    target : (n_bins,) float
    n_cells : int
        Cells per draw; must be in [1, n_rois].
    n_draws : int
        Number of random subsets (>= 1).
    rng : Generator, int or None
    n_folds : int

    Returns
    -------
    dict
        ``errors`` (n_draws,), ``rhos`` (n_draws,), ``median_error``,
        ``median_rho``, ``n_cells``, ``n_draws``.

    Raises
    ------
    ValueError
        If ``n_cells`` is out of range or ``n_draws`` < 1.
    """
    x = _as_2d(binned, "binned")
    n_rois = x.shape[0]
    if not 1 <= n_cells <= n_rois:
        raise ValueError(f"n_cells must be in [1, {n_rois}]; got {n_cells}")
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1; got {n_draws}")
    gen = _as_rng(rng)
    errors = np.full(n_draws, np.nan)
    rhos = np.full(n_draws, np.nan)
    for d in range(n_draws):
        idx = np.sort(gen.choice(n_rois, size=n_cells, replace=False))
        res = time_decoding(x[idx], target, n_folds=n_folds, block=True)
        errors[d] = res["median_abs_error"]
        rhos[d] = res["rho"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med_err = float(np.nanmedian(errors))
        med_rho = float(np.nanmedian(rhos))
    return {
        "errors": errors,
        "rhos": rhos,
        "median_error": med_err,
        "median_rho": med_rho,
        "n_cells": int(n_cells),
        "n_draws": int(n_draws),
    }


# ---------------------------------------------------------------------------
# Epoch identity and temporal distance
# ---------------------------------------------------------------------------


def _loo_centroid_correct(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Leave-one-out nearest-centroid correctness per sample.

    Parameters
    ----------
    z : (n_samples, n_features) float
    labels : (n_samples,) int; every class must have >= 2 samples.

    Returns
    -------
    (n_samples,) bool
    """
    classes, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    sums = np.zeros((classes.size, z.shape[1]))
    np.add.at(sums, inv, z)
    centroids = sums / counts[:, None]
    d = ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    own = (sums[inv] - z) / (counts[inv] - 1)[:, None]
    d[np.arange(z.shape[0]), inv] = ((z - own) ** 2).sum(axis=1)
    return np.asarray(np.argmin(d, axis=1) == inv)


def _epoch_accuracy(
    z: np.ndarray, labels: np.ndarray, states: np.ndarray
) -> tuple[float, float, int, int]:
    """Pooled LOO nearest-centroid accuracy and chance within each state.

    Parameters
    ----------
    z : (n_bins, n_rois) float
    labels : (n_bins,) int
    states : (n_bins,) int or bool
        Classification is run separately within each state value.

    Returns
    -------
    accuracy, chance, n_bins_used, n_classes
        NaN accuracy / chance when no state has two classes of >= 2 bins.
    """
    correct = 0
    chance = 0.0
    n_used = 0
    n_classes = 0
    for st in np.unique(states):
        sel = states == st
        ids, counts = np.unique(labels[sel], return_counts=True)
        keep_ids = ids[counts >= 2]
        if keep_ids.size < 2:
            continue
        rows = sel & np.isin(labels, keep_ids)
        correct += int(_loo_centroid_correct(z[rows], labels[rows]).sum())
        chance += rows.sum() / keep_ids.size
        n_used += int(rows.sum())
        n_classes += int(keep_ids.size)
    if n_used == 0:
        return float("nan"), float("nan"), 0, 0
    return correct / n_used, chance / n_used, n_used, n_classes


def epoch_identity_decoding(
    binned: npt.ArrayLike,
    epoch_idx: npt.ArrayLike,
    epoch_light: npt.ArrayLike | None = None,
    n_perms: int = 200,
    rng: np.random.Generator | int | None = None,
) -> dict[str, Any]:
    """Classify which light/dark epoch each bin belongs to.

    Leave-one-bin-out nearest-centroid classification on z-scored population
    vectors. When ``epoch_light`` is given, each cell's median per light state
    is first removed and classification is among epochs of the same light
    state only (light and dark run separately, then pooled), so a response to
    the light state itself cannot be used. Chance is the bin-weighted mean of
    1 / n_epochs per state; the test statistic is accuracy minus chance.

    The null circularly shifts each cell's (state-median-removed) binned
    activity by an independent random amount while the epoch labels stay
    fixed. This preserves each cell's autocorrelation and the magnitude of
    its slow drift but breaks the alignment of the population to the epochs.
    A label-shift null is not used here because with ~6 bins per epoch it has
    only ~6 distinct partitions and cannot resolve p < 0.05. Population-
    coherent slow drift is only partly removed by this null, so a significant
    result means "the population distinguishes epochs", which includes drift;
    compare with :func:`temporal_distance` and :func:`time_decoding`.

    Parameters
    ----------
    binned : (n_rois, n_bins) float
        Full-session binned activity; bins with any NaN are excluded.
    epoch_idx : (n_bins,) int
        Contiguous epoch numbers (see :func:`epoch_index`).
    epoch_light : (n_bins,) bool or None
    n_perms : int
    rng : Generator, int or None

    Returns
    -------
    dict
        ``accuracy``, ``chance``, ``accuracy_minus_chance``, ``p_value``
        (one-sided), ``null_minus_chance`` (n_perms,), ``n_bins``,
        ``n_epochs``.

    References
    ----------
    Rubin A et al. 2015. "Hippocampal ensemble dynamics timestamp events in
    long-term memory." eLife. doi:10.7554/eLife.12247
    """
    x = _as_2d(binned, "binned")
    ep = np.asarray(epoch_idx).astype(int)
    if ep.shape != (x.shape[1],):
        raise ValueError(f"epoch_idx must have shape ({x.shape[1]},); got {ep.shape}")
    el = np.zeros(ep.size, dtype=bool) if epoch_light is None else np.asarray(epoch_light, bool)
    gen = _as_rng(rng)
    ok = _valid_columns(x)
    out: dict[str, Any] = {
        "accuracy": float("nan"),
        "chance": float("nan"),
        "accuracy_minus_chance": float("nan"),
        "p_value": float("nan"),
        "null_minus_chance": np.full(n_perms, np.nan),
        "n_bins": 0,
        "n_epochs": 0,
    }
    if ok.sum() < 4 or x.shape[0] == 0:
        return out
    xv = _subtract_state_median(x[:, ok], el[ok] if epoch_light is not None else None)
    z = _zscore_rows(xv).T

    acc, chance, n_used, n_classes = _epoch_accuracy(z, ep[ok], el[ok])
    if n_used == 0:
        return out
    null = np.full(n_perms, np.nan)
    lab_ok, st_ok = ep[ok], el[ok]
    n_valid, n_cells = z.shape
    rows = np.arange(n_valid)
    for i in range(n_perms):
        shifts = _shift_values(n_valid, n_cells, gen)
        zs = z[(rows[:, None] - shifts[None, :]) % n_valid, np.arange(n_cells)[None, :]]
        a_null, c_null, _, _ = _epoch_accuracy(zs, lab_ok, st_ok)
        null[i] = a_null - c_null
    out.update(
        {
            "accuracy": float(acc),
            "chance": float(chance),
            "accuracy_minus_chance": float(acc - chance),
            "p_value": _perm_p_greater(acc - chance, null),
            "null_minus_chance": null,
            "n_bins": int(n_used),
            "n_epochs": int(n_classes),
        }
    )
    return out


def temporal_distance(
    binned: npt.ArrayLike,
    epoch_idx: npt.ArrayLike,
    epoch_light: npt.ArrayLike | None = None,
    n_perms: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> dict[str, Any]:
    """Population-vector distance between epochs versus their time separation.

    Each epoch is summarised by its mean z-scored population vector. For every
    pair of epochs (restricted to same-light-state pairs when ``epoch_light``
    is given) the Euclidean distance is related to the separation of the epoch
    centres (in bins) by Spearman rho. A positive rho means representations
    become more different the further apart in time they are. The null
    permutes epoch vectors among epochs of the same light state (Mantel-style)
    and the p-value is one-sided (rho larger than null).

    Parameters
    ----------
    binned : (n_rois, n_bins) float
    epoch_idx : (n_bins,) int
    epoch_light : (n_bins,) bool or None
    n_perms : int
    rng : Generator, int or None

    Returns
    -------
    dict
        ``rho``, ``p_value``, ``n_pairs``, ``n_epochs``.

    References
    ----------
    Rubin A et al. 2015. "Hippocampal ensemble dynamics timestamp events in
    long-term memory." eLife. doi:10.7554/eLife.12247

    Ziv Y et al. 2013. "Long-term dynamics of CA1 hippocampal place codes."
    Nature Neuroscience. doi:10.1038/nn.3329
    """
    x = _as_2d(binned, "binned")
    ep = np.asarray(epoch_idx).astype(int)
    if ep.shape != (x.shape[1],):
        raise ValueError(f"epoch_idx must have shape ({x.shape[1]},); got {ep.shape}")
    el = np.zeros(ep.size, dtype=bool) if epoch_light is None else np.asarray(epoch_light, bool)
    gen = _as_rng(rng)
    ok = _valid_columns(x)
    nan_out = {"rho": float("nan"), "p_value": float("nan"), "n_pairs": 0, "n_epochs": 0}
    if ok.sum() < 3 or x.shape[0] == 0:
        return nan_out
    z = _zscore_rows(x[:, ok])
    epv, elv = ep[ok], el[ok]
    ids = np.unique(epv)
    vecs = np.vstack([z[:, epv == e].mean(axis=1) for e in ids])
    centre = np.array([np.flatnonzero(epv == e).mean() for e in ids])
    state = np.array([elv[epv == e][0] for e in ids])

    iu, ju = np.triu_indices(ids.size, k=1)
    same = state[iu] == state[ju]
    iu, ju = iu[same], ju[same]
    if iu.size < 3:
        return {**nan_out, "n_pairs": int(iu.size), "n_epochs": int(ids.size)}
    sep = np.abs(centre[iu] - centre[ju])

    def _rho(v: np.ndarray) -> float:
        dist = np.sqrt(((v[iu] - v[ju]) ** 2).sum(axis=1))
        return _spearman(dist, sep)

    obs = _rho(vecs)
    null = np.full(n_perms, np.nan)
    for i in range(n_perms):
        perm = np.arange(ids.size)
        for st in np.unique(state):
            members = np.flatnonzero(state == st)
            perm[members] = gen.permutation(members)
        null[i] = _rho(vecs[perm])
    return {
        "rho": obs,
        "p_value": _perm_p_greater(obs, null),
        "n_pairs": int(iu.size),
        "n_epochs": int(ids.size),
    }


# ---------------------------------------------------------------------------
# Per-cell measures
# ---------------------------------------------------------------------------


def _rank_rows(x: np.ndarray) -> np.ndarray:
    """Average ranks along each row."""
    return np.asarray(stats.rankdata(x, axis=1), dtype=np.float64)


def per_cell_drift(
    binned: npt.ArrayLike,
    n_perms: int = 500,
    rng: np.random.Generator | int | None = None,
) -> dict[str, np.ndarray]:
    """Per-cell monotonic trend of activity over the session.

    Spearman rho between each cell's binned activity and bin order. The null
    circularly shifts each cell's activity relative to time (the same shift
    for all cells per permutation); the p-value is two-sided on |rho|.

    Parameters
    ----------
    binned : (n_rois, n_bins) float
        Bins with any NaN are dropped.
    n_perms : int
    rng : Generator, int or None

    Returns
    -------
    dict
        ``rho`` (n_rois,), ``p_value`` (n_rois,); NaN for constant cells or
        fewer than 3 valid bins.

    References
    ----------
    Ziv Y et al. 2013. "Long-term dynamics of CA1 hippocampal place codes."
    Nature Neuroscience. doi:10.1038/nn.3329
    """
    x = _as_2d(binned, "binned")
    n_rois = x.shape[0]
    ok = _valid_columns(x)
    rho = np.full(n_rois, np.nan)
    pval = np.full(n_rois, np.nan)
    n = int(ok.sum())
    if n < 3 or n_rois == 0:
        return {"rho": rho, "p_value": pval}
    r = _rank_rows(x[:, ok])
    r = r - r.mean(axis=1, keepdims=True)
    norm = np.sqrt((r**2).sum(axis=1))
    const = norm == 0
    norm = np.where(const, 1.0, norm)
    t = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    t /= np.sqrt((t**2).sum())
    rho = (r @ t) / norm
    gen = _as_rng(rng)
    shifts = _shift_values(n, n_perms, gen)
    null = np.vstack([(np.roll(r, int(s), axis=1) @ t) / norm for s in shifts]).T
    pval = (1 + np.sum(np.abs(null) >= np.abs(rho)[:, None] - 1e-12, axis=1)) / (1 + n_perms)
    rho = np.where(const, np.nan, rho)
    pval = np.where(const, np.nan, pval)
    return {"rho": rho, "p_value": pval}


def _kruskal_h_rows(ranks: np.ndarray, labels: np.ndarray, tie: np.ndarray) -> np.ndarray:
    """Kruskal-Wallis H for every row of a rank matrix given group labels."""
    n = ranks.shape[1]
    _, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    onehot = np.zeros((n, counts.size))
    onehot[np.arange(n), inv] = 1.0
    rsum = ranks @ onehot
    h = 12.0 / (n * (n + 1)) * (rsum**2 / counts[None, :]).sum(axis=1) - 3.0 * (n + 1)
    return np.asarray(h / tie)


def _kw_epsilon_sq(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Kruskal-Wallis epsilon-squared, H / (N - 1), per row of *x*.

    H includes the tie correction. Rows with no variation give NaN.
    """
    n = x.shape[1]
    ranks = _rank_rows(x)
    tie = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        _, c = np.unique(ranks[i], return_counts=True)
        tie[i] = 1.0 - (c**3 - c).sum() / (n**3 - n)
    const = tie <= 0
    h = _kruskal_h_rows(ranks, labels, np.where(const, 1.0, tie))
    return np.where(const, np.nan, h / (n - 1))


def per_cell_epoch_selectivity(
    binned: npt.ArrayLike,
    epoch_idx: npt.ArrayLike,
    epoch_light: npt.ArrayLike | None = None,
    n_perms: int = 500,
    rng: np.random.Generator | int | None = None,
) -> dict[str, np.ndarray]:
    """Per-cell Kruskal-Wallis H of binned activity across epochs.

    H (with tie correction) measures how much a cell's activity differs
    between epochs. When ``epoch_light`` is given, each cell's median per
    light state is removed first, so a pure light/dark response does not
    count as epoch selectivity.

    The null circularly shifts the epoch-label sequence relative to the bins
    by a uniformly drawn non-zero shift (the full cyclic group, which makes
    the observed partition exchangeable with the null ones for stationary
    activity). The block that wraps from the session end to its start is
    split into two groups, so every null group is contiguous in time. Equal
    block lengths matter: for a monotonic drift any partition with unequal
    blocks has lower between-group variance than the true one, which would
    make drift look epoch-selective. The one extra group makes the null
    slightly conservative for stationary activity. Observed and null are
    compared on epsilon-squared, H / (N - 1). The p-value is one-sided.

    Resolution is limited by the number of bins per epoch (shifts s and
    s + L give nearly the same partition), so use bins much shorter than an
    epoch (e.g. 1 s bins for 60 s epochs).

    Parameters
    ----------
    binned : (n_rois, n_bins) float
    epoch_idx : (n_bins,) int
    epoch_light : (n_bins,) bool or None
    n_perms : int
    rng : Generator, int or None

    Returns
    -------
    dict
        ``H`` (n_rois,), ``epsilon_sq`` (n_rois,; H / (N - 1)) and
        ``p_value`` (n_rois,); NaN for constant cells or fewer than 2 epochs.

    References
    ----------
    Kruskal WH, Wallis WA. 1952. "Use of ranks in one-criterion variance
    analysis." Journal of the American Statistical Association.
    doi:10.1080/01621459.1952.10483441
    """
    x = _as_2d(binned, "binned")
    ep = np.asarray(epoch_idx).astype(int)
    if ep.shape != (x.shape[1],):
        raise ValueError(f"epoch_idx must have shape ({x.shape[1]},); got {ep.shape}")
    n_rois = x.shape[0]
    ok = _valid_columns(x)
    h_out = np.full(n_rois, np.nan)
    p_out = np.full(n_rois, np.nan)
    lab = ep[ok]
    if np.unique(lab).size < 2 or lab.size < 3 or n_rois == 0:
        return {"H": h_out, "epsilon_sq": h_out.copy(), "p_value": p_out}
    state = None if epoch_light is None else np.asarray(epoch_light, dtype=bool)[ok]
    xv = _subtract_state_median(x[:, ok], state)
    n = lab.size
    obs = _kw_epsilon_sq(xv, lab)
    gen = _as_rng(rng)
    if n_perms > 0:
        null = np.empty((n_rois, n_perms))
        for i, shift in enumerate(gen.integers(1, n, size=n_perms)):
            null[:, i] = _kw_epsilon_sq(xv, _circular_partition(lab, int(shift)))
        p = (1 + np.sum(null >= obs[:, None] - 1e-12, axis=1)) / (1 + n_perms)
        p_out = np.where(np.isfinite(obs), p, np.nan)
    return {"H": obs * (n - 1), "epsilon_sq": obs, "p_value": p_out}


def first_vs_later_dark(
    signal: npt.ArrayLike,
    light_on: npt.ArrayLike,
    bad_behav: npt.ArrayLike,
    fps: float,
    min_epoch_s: float = 5.0,
) -> dict[str, Any]:
    """Mean activity in the first dark epoch versus later dark epochs, per cell.

    Wraps :func:`hm2p.analysis.transitions.first_vs_later_epochs` with the
    NaN-aware mean as metric. A positive ``difference`` means a cell was more
    active in the first exposure to darkness than in later ones (novelty).

    Parameters
    ----------
    signal : (n_rois, n_frames) or (n_frames,) float
    light_on : (n_frames,) bool
    bad_behav : (n_frames,) bool
    fps : float
    min_epoch_s : float

    Returns
    -------
    dict
        ``first_value``, ``later_mean``, ``difference`` (each (n_rois,)), and
        ``n_epochs`` (int, number of dark epochs used).
    """
    from hm2p.analysis.transitions import first_vs_later_epochs

    x = _as_2d(signal, "signal")
    light = np.asarray(light_on, dtype=bool)
    bad = np.asarray(bad_behav, dtype=bool)

    def _mean(seg: np.ndarray) -> float:
        seg = seg[np.isfinite(seg)]
        return float(seg.mean()) if seg.size else float("nan")

    first = np.full(x.shape[0], np.nan)
    later = np.full(x.shape[0], np.nan)
    diff = np.full(x.shape[0], np.nan)
    n_epochs = 0
    for i in range(x.shape[0]):
        res = first_vs_later_epochs(x[i], light, bad, fps, _mean, min_epoch_s)
        first[i] = res["first_value"]
        later[i] = res["later_mean"]
        diff[i] = res["difference"]
        n_epochs = int(res["n_epochs"])
    return {"first_value": first, "later_mean": later, "difference": diff, "n_epochs": n_epochs}
