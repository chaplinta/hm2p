"""Tools for asking whether a labelled cell group is a discrete type or part of a continuum.

Four non-parametric tools:

- :func:`silverman_mode_test` -- Silverman's critical-bandwidth test for the
  number of modes of a 1-D distribution (is a measure unimodal within a group,
  or does it hide a sub-population?).
- :func:`shift_function` -- difference between two groups at each quantile,
  with a hierarchical (animal, then cell) bootstrap confidence band. A
  continuum shift moves every quantile; a discrete sub-population changes
  mainly one tail.
- :func:`rank_axis` -- a per-cell composite score: the mean percentile rank of
  several measures, each oriented by a sign.
- :func:`knn_label_mixing` -- how often a cell's nearest neighbours in feature
  space share its label, against a permutation null that reassigns labels
  between animals (cells of one animal always share a label in this design).

References
----------
Silverman BW. 1981. "Using kernel density estimates to investigate
multimodality." J R Stat Soc B 43:97-99. doi:10.1111/j.2517-6161.1981.tb01155.x

Hall P, York M. 2001. "On the calibration of Silverman's test for
multimodality." Statistica Sinica 11:515-536. (The uncalibrated test used here
is conservative: p values are too large, so a small p is credible.)

Rousselet GA, Pernet CR, Wilcox RR. 2017. "Beyond differences in means:
robust graphical methods to compare two groups in neuroscience." Eur J
Neurosci 46:1738-1748. doi:10.1111/ejn.13610

Doksum K. 1974. "Empirical probability plots and statistical inference for
nonlinear models in the two-sample case." Ann Stat 2:267-277.
doi:10.1214/aos/1176342662
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
import numpy.typing as npt

_GRID = 512


def _finite(x: npt.ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def count_modes(x: npt.ArrayLike, bandwidth: float, grid: np.ndarray | None = None) -> int:
    """Number of local maxima of a Gaussian kernel density estimate of *x*."""
    x = _finite(x)
    if x.size == 0:
        return 0
    if grid is None:
        span = x.max() - x.min()
        grid = np.linspace(
            x.min() - 3 * bandwidth, x.max() + 3 * bandwidth + 1e-12 * (span == 0), _GRID
        )
    dens = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bandwidth) ** 2).sum(axis=1)
    d = np.diff(dens)
    # a mode is a sign change of the derivative from + to - (ignoring flat runs)
    s = np.sign(d)
    s = s[s != 0]
    return int(np.sum((s[:-1] > 0) & (s[1:] < 0)))


def critical_bandwidth(x: npt.ArrayLike, k: int = 1, tol: float = 1e-4) -> float:
    """Smallest Gaussian bandwidth at which the KDE of *x* has at most *k* modes."""
    x = _finite(x)
    if x.size < 2 or np.ptp(x) == 0:
        return 0.0
    lo, hi = 1e-6 * np.ptp(x), np.ptp(x)
    while count_modes(x, hi) > k:
        hi *= 2
    while (hi - lo) > tol * np.ptp(x):
        mid = 0.5 * (lo + hi)
        if count_modes(x, mid) > k:
            lo = mid
        else:
            hi = mid
    return float(hi)


def silverman_mode_test(
    x: npt.ArrayLike, k: int = 1, n_boot: int = 500, seed: int = 0
) -> dict[str, Any]:
    """Silverman's test of H0: the distribution of *x* has at most *k* modes.

    Smoothed bootstrap from the KDE at the critical bandwidth, rescaled to the
    sample variance (Silverman 1981). Returns ``h_crit``, ``p`` (fraction of
    bootstrap samples with more than *k* modes at ``h_crit``) and ``n``.
    """
    x = _finite(x)
    n = x.size
    if n < 10 or np.ptp(x) == 0:
        return {"h_crit": float("nan"), "p": float("nan"), "n": int(n)}
    h = critical_bandwidth(x, k)
    rng = np.random.default_rng(seed)
    mu, var = x.mean(), x.var()
    scale = 1.0 / np.sqrt(1.0 + h**2 / var) if var > 0 else 1.0
    exceed = 0
    for _ in range(n_boot):
        y = rng.choice(x, size=n, replace=True) + h * rng.standard_normal(n)
        y = mu + (y - mu) * scale
        exceed += count_modes(y, h) > k
    return {"h_crit": h, "p": float(exceed / n_boot), "n": int(n)}


def _hier_resample(values: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample groups (animals) with replacement, then values within each chosen group."""
    uniq = np.unique(groups)
    picks = rng.choice(uniq, size=uniq.size, replace=True)
    parts = []
    for g in picks:
        v = values[groups == g]
        parts.append(rng.choice(v, size=v.size, replace=True))
    return np.concatenate(parts)


def shift_function(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    groups_a: npt.ArrayLike | None = None,
    groups_b: npt.ArrayLike | None = None,
    quantiles: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Quantile differences ``Q_a(q) - Q_b(q)`` with a bootstrap confidence band.

    With ``groups_a``/``groups_b`` (animal ids), the bootstrap resamples
    animals and then cells within animals, so the band reflects between-animal
    variability; without them it resamples cells.
    """
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    ga = np.zeros(a_arr.size) if groups_a is None else np.asarray(groups_a)
    gb = np.zeros(b_arr.size) if groups_b is None else np.asarray(groups_b)
    ka, kb = np.isfinite(a_arr), np.isfinite(b_arr)
    a_arr, ga, b_arr, gb = a_arr[ka], ga[ka], b_arr[kb], gb[kb]
    q = np.asarray(quantiles, float)
    if a_arr.size == 0 or b_arr.size == 0:
        nan = [float("nan")] * q.size
        return {"quantiles": q.tolist(), "diff": nan, "lo": nan, "hi": nan}
    diff = np.quantile(a_arr, q) - np.quantile(b_arr, q)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, q.size))
    for i in range(n_boot):
        boots[i] = np.quantile(_hier_resample(a_arr, ga, rng), q) - np.quantile(
            _hier_resample(b_arr, gb, rng), q
        )
    alpha = (1 - ci) / 2
    return {
        "quantiles": q.tolist(),
        "diff": diff.tolist(),
        "lo": np.quantile(boots, alpha, axis=0).tolist(),
        "hi": np.quantile(boots, 1 - alpha, axis=0).tolist(),
    }


def rank_axis(columns: Sequence[npt.ArrayLike], signs: Sequence[int]) -> np.ndarray:
    """Mean percentile rank (0-1) across *columns*, each multiplied by its sign first.

    NaN entries are ignored per cell; a cell with no finite entries gets NaN.
    """
    if len(columns) != len(signs) or not columns:
        raise ValueError("columns and signs must be non-empty and the same length")
    mats = []
    for col, sgn in zip(columns, signs, strict=True):
        v = np.asarray(col, float) * sgn
        r = np.full(v.shape, np.nan)
        ok = np.isfinite(v)
        if ok.sum() > 1:
            order = v[ok].argsort().argsort()
            r[ok] = order / (ok.sum() - 1)
        elif ok.sum() == 1:
            r[ok] = 0.5
        mats.append(r)
    m = np.vstack(mats)
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(m).sum(axis=0)
        out = np.where(cnt > 0, np.nansum(m, axis=0) / np.maximum(cnt, 1), np.nan)
    return out


def _same_label_fraction(nbrs: np.ndarray, labels: np.ndarray, target: Any) -> float:
    rows = np.flatnonzero(labels == target)
    if rows.size == 0:
        return float("nan")
    return float(np.mean(labels[nbrs[rows]] == target))


def knn_label_mixing(
    X: npt.ArrayLike,
    labels: npt.ArrayLike,
    animals: npt.ArrayLike,
    k: int = 10,
    max_perms: int = 2000,
    seed: int = 0,
    exclude_same_animal: bool = True,
) -> dict[str, Any]:
    """Same-label fraction among each cell's *k* nearest neighbours, per label.

    Features are z-scored per column. With ``exclude_same_animal`` the
    neighbours are drawn only from other animals, so the statistic is not
    driven by animal identity. The null reassigns labels between animals
    (exactly, when the number of label assignments is at most ``max_perms``),
    keeping the number of animals per label fixed. Returns per-label observed
    fractions, null means, and one-sided p values (observed >= null).
    """
    X = np.asarray(X, float)
    labels = np.asarray(labels)
    animals = np.asarray(animals)
    ok = np.all(np.isfinite(X), axis=1)
    X, labels, animals = X[ok], labels[ok], animals[ok]
    sd = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1)
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    if exclude_same_animal:
        d[animals[:, None] == animals[None, :]] = np.inf
    kk = min(k, max(1, int(np.isfinite(d).sum(axis=1).min())))
    nbrs = np.argsort(d, axis=1)[:, :kk]

    uniq_labels = sorted(set(labels.tolist()))
    animal_ids = np.unique(animals)
    animal_label = {a: labels[animals == a][0] for a in animal_ids}
    observed = {lab: _same_label_fraction(nbrs, labels, lab) for lab in uniq_labels}

    # label assignments over animals, keeping counts per label
    if len(uniq_labels) != 2:
        raise ValueError("knn_label_mixing supports exactly two labels")
    first = uniq_labels[0]
    n_first = sum(1 for a in animal_ids if animal_label[a] == first)
    combos = list(combinations(range(animal_ids.size), n_first))
    rng = np.random.default_rng(seed)
    if len(combos) > max_perms:
        combos = [
            tuple(sorted(rng.choice(animal_ids.size, n_first, replace=False)))
            for _ in range(max_perms)
        ]
    idx_of = {a: i for i, a in enumerate(animal_ids)}
    cell_animal_idx = np.array([idx_of[a] for a in animals])
    null = {lab: [] for lab in uniq_labels}
    for combo in combos:
        is_first = np.isin(cell_animal_idx, combo)
        perm = np.where(is_first, first, uniq_labels[1])
        for lab in uniq_labels:
            null[lab].append(_same_label_fraction(nbrs, perm, lab))
    out: dict[str, Any] = {"k": kk, "n_assignments": len(combos), "labels": {}}
    for lab in uniq_labels:
        nv = np.asarray(null[lab], float)
        nv = nv[np.isfinite(nv)]
        obs = observed[lab]
        out["labels"][str(lab)] = {
            "observed": obs,
            "null_mean": float(nv.mean()) if nv.size else float("nan"),
            "p_one_sided": float((np.sum(nv >= obs - 1e-12)) / nv.size)
            if nv.size
            else float("nan"),
            "n_cells": int(np.sum(labels == lab)),
        }
    return out
