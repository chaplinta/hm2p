"""Map-engagement: is a maze location re-instantiated as the same population
state each time it is visited?

Tests whether the RSP population's spatial representation is *used* during
exploration, without decoding (few HD cells). A "visit" to a maze cell is a
contiguous run of valid frames in that cell, summarised by its mean population
vector. If the spatial map is engaged, repeat visits to the same cell give
similar population vectors (high within-cell consistency) relative to visits to
different cells (across-cell baseline). The reported quantity is
``within - across`` consistency, which removes global drift / arousal that would
correlate everything.

See docs/plan-map-engagement-neural.md. The within-vs-across-location logic
follows population-vector reproducibility analyses (e.g. spatial-map stability;
Ziv et al. 2013, Nat. Neurosci. 16:264-266, doi:10.1038/nn.3329).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def extract_visit_vectors(
    signal: npt.NDArray[np.floating],
    cell_idx: npt.NDArray[np.integer],
    valid: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    """Collapse contiguous same-cell runs into one mean population vector each.

    Parameters
    ----------
    signal : (n_cells, n_frames) float
        Per-frame population activity (e.g. z-scored dF/F).
    cell_idx : (n_frames,) int
        Maze-cell index per frame; negative = no cell.
    valid : (n_frames,) bool
        Frames to include (e.g. moving and not bad_behav).

    Returns
    -------
    visit_cells : (n_visits,) int
        Maze cell of each visit.
    visit_vecs : (n_visits, n_cells) float
        Mean population vector over each visit's frames.
    """
    signal = np.asarray(signal, dtype=np.float64)
    cell_idx = np.asarray(cell_idx)
    valid = np.asarray(valid, dtype=bool)
    n = cell_idx.shape[0]
    if n == 0:
        return np.empty(0, dtype=int), np.empty((0, signal.shape[0]), dtype=np.float64)

    eff = np.where(valid, cell_idx, -1)
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = eff[1:] != eff[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], n)

    cells: list[int] = []
    vecs: list[np.ndarray] = []
    for s, e in zip(starts, ends, strict=False):
        c = int(eff[s])
        if c < 0:
            continue
        cells.append(c)
        vecs.append(signal[:, s:e].mean(axis=1))
    if not cells:
        return np.empty(0, dtype=int), np.empty((0, signal.shape[0]), dtype=np.float64)
    return np.asarray(cells, dtype=int), np.vstack(vecs)


def mean_pairwise_corr(vecs: npt.NDArray[np.floating]) -> float:
    """Mean off-diagonal Pearson correlation among rows of ``vecs``.

    Returns ``nan`` for fewer than two rows or no finite pairs.
    """
    vecs = np.asarray(vecs, dtype=np.float64)
    if vecs.shape[0] < 2:
        return float("nan")
    c = np.corrcoef(vecs)
    iu = np.triu_indices(c.shape[0], k=1)
    vals = c[iu]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")


def consistency_debiased(
    visit_cells: npt.NDArray[np.integer],
    visit_vecs: npt.NDArray[np.floating],
    k_visits: int = 3,
    n_cells_cap: int | None = None,
    n_boot: int = 50,
    rng: np.random.Generator | None = None,
) -> dict:
    """Within-cell minus across-cell population-vector consistency.

    Cells with at least ``k_visits`` visits are eligible. Optionally subsample to
    ``n_cells_cap`` eligible cells and exactly ``k_visits`` visits per cell
    (bootstrap ``n_boot`` times, averaged) so the estimate can be matched across
    conditions. ``within`` is the mean over cells of the mean pairwise visit-
    correlation; ``across`` is the mean pairwise correlation between cells'
    representative (mean) vectors.

    Returns
    -------
    dict with ``within``, ``across``, ``debiased`` (= within - across),
    ``n_eligible_cells``, ``n_used_cells``.
    """
    if rng is None:
        rng = np.random.default_rng()
    visit_cells = np.asarray(visit_cells, dtype=int)
    visit_vecs = np.asarray(visit_vecs, dtype=np.float64)

    uniq, counts = np.unique(visit_cells, return_counts=True)
    eligible = uniq[counts >= k_visits]
    n_eligible = int(eligible.size)
    cap = n_eligible if n_cells_cap is None else min(n_cells_cap, n_eligible)
    if cap < 2:
        return {
            "within": float("nan"),
            "across": float("nan"),
            "debiased": float("nan"),
            "n_eligible_cells": n_eligible,
            "n_used_cells": cap,
        }

    cell_to_visits = {c: np.flatnonzero(visit_cells == c) for c in eligible}

    within_boot, across_boot = [], []
    for _ in range(n_boot):
        chosen = rng.choice(eligible, size=cap, replace=False)
        within_per_cell = []
        reps = []
        for c in chosen:
            vi = cell_to_visits[c]
            pick = rng.choice(vi, size=k_visits, replace=False)
            vecs = visit_vecs[pick]
            within_per_cell.append(mean_pairwise_corr(vecs))
            reps.append(vecs.mean(axis=0))
        within_boot.append(np.nanmean(within_per_cell))
        across_boot.append(mean_pairwise_corr(np.vstack(reps)))

    within = float(np.nanmean(within_boot))
    across = float(np.nanmean(across_boot))
    return {
        "within": within,
        "across": across,
        "debiased": within - across,
        "n_eligible_cells": n_eligible,
        "n_used_cells": cap,
    }
