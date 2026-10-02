"""Separate location (corridor vs junction) from running in single-cell activity.

In a maze, corridor occupancy and running are confounded: animals run along
corridors and slow at junctions. Two contrasts disentangle them per cell:

- **Location at matched speed:** among running frames, corridor frames and
  junction frames are subsampled so their speed distributions match
  (:func:`hm2p.analysis.matched_tuning.match_indices_1d`), then the location
  index ``(r_corridor - r_junction) / (r_corridor + r_junction)`` is computed.
- **Running within a location:** within corridor frames (and separately
  within junction frames), ``(r_run - r_still) / (r_run + r_still)``.

Significance uses a circular shift of the signal relative to behaviour, which
keeps both autocorrelations (the same design as
:mod:`hm2p.analysis.significance`). The frame subsets are fixed before
shuffling so that the null differs from the observed value only by the
signal-behaviour alignment.

Maze topology: Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a
labyrinth show rapid learning, sudden insight, and efficient exploration."
eLife 10:e66175. doi:10.7554/eLife.66175
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from hm2p.analysis.matched_tuning import match_indices_1d

CORRIDOR, JUNCTION, DEAD_END = 0, 1, 2


def location_labels(node_channels: dict[str, np.ndarray]) -> np.ndarray:
    """Integer location per frame from one-hot node channels (-1 when invalid).

    Expects ``node_corridor``, ``node_junction`` and ``node_dead_end`` as
    produced by :func:`hm2p.analysis.event_triggered_behaviour.node_type_channels`.
    """
    corr = np.asarray(node_channels["node_corridor"], dtype=np.float64)
    junc = np.asarray(node_channels["node_junction"], dtype=np.float64)
    dead = np.asarray(node_channels["node_dead_end"], dtype=np.float64)
    out = np.full(corr.shape, -1, dtype=np.int64)
    out[corr == 1] = CORRIDOR
    out[junc == 1] = JUNCTION
    out[dead == 1] = DEAD_END
    return out


def _index(a: float, b: float) -> float:
    """``(a - b) / (a + b)``; NaN when the sum is not positive or inputs are NaN."""
    if not (np.isfinite(a) and np.isfinite(b)) or a + b <= 0:
        return float("nan")
    return float((a - b) / (a + b))


def contrast_frames(
    speed: npt.ArrayLike,
    location: npt.ArrayLike,
    mask: npt.ArrayLike,
    run_threshold: float = 2.5,
    n_speed_bins: int = 10,
    min_frames: int = 50,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Frame index sets for the two contrasts.

    Returns
    -------
    dict
        ``corr_run_matched`` / ``junc_run_matched``: running frames in
        corridors and junctions with speed distributions matched;
        ``corr_run`` / ``corr_still`` / ``junc_run`` / ``junc_still``: frames
        for the running-within-location contrasts. Sets with fewer than
        *min_frames* frames are returned empty.
    """
    spd = np.asarray(speed, dtype=np.float64)
    loc = np.asarray(location, dtype=np.int64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(spd)
    run = m & (spd >= run_threshold)
    still = m & (spd < run_threshold)
    sets = {
        "corr_run": np.flatnonzero(run & (loc == CORRIDOR)),
        "corr_still": np.flatnonzero(still & (loc == CORRIDOR)),
        "junc_run": np.flatnonzero(run & (loc == JUNCTION)),
        "junc_still": np.flatnonzero(still & (loc == JUNCTION)),
    }
    empty = np.zeros(0, dtype=np.int64)
    a, b = sets["corr_run"], sets["junc_run"]
    if a.size >= min_frames and b.size >= min_frames:
        ia, ib = match_indices_1d(
            spd[a],
            spd[b],
            n_bins=n_speed_bins,
            circular=False,
            rng=rng or np.random.default_rng(0),
        )
        sets["corr_run_matched"] = a[ia]
        sets["junc_run_matched"] = b[ib]
    else:
        sets["corr_run_matched"] = empty
        sets["junc_run_matched"] = empty
    return {k: (v if v.size >= min_frames else empty) for k, v in sets.items()}


def contrast_statistics(signal: npt.ArrayLike, frames: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean rates and contrast indices for one signal on fixed frame sets."""
    x = np.asarray(signal, dtype=np.float64)

    def _mean(idx: np.ndarray) -> float:
        return float(np.nanmean(x[idx])) if idx.size else float("nan")

    r = {k: _mean(v) for k, v in frames.items()}
    return {
        "rate_corr_run_matched": r["corr_run_matched"],
        "rate_junc_run_matched": r["junc_run_matched"],
        "loc_index": _index(r["corr_run_matched"], r["junc_run_matched"]),
        "run_index_corr": _index(r["corr_run"], r["corr_still"]),
        "run_index_junc": _index(r["junc_run"], r["junc_still"]),
    }


INDEX_KEYS = ("loc_index", "run_index_corr", "run_index_junc")


def contrast_significance(
    signal: npt.ArrayLike,
    frames: dict[str, np.ndarray],
    fps: float,
    n_shuffles: int = 200,
    min_shift_s: float = 10.0,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Observed contrasts with circular-shift z and two-sided p for each index."""
    x = np.asarray(signal, dtype=np.float64)
    obs = contrast_statistics(x, frames)
    rng = rng or np.random.default_rng(0)
    n = x.size
    min_shift = int(round(min_shift_s * fps))
    out: dict[str, Any] = dict(obs)
    if n <= 2 * min_shift or n_shuffles < 1:
        for k in INDEX_KEYS:
            out[f"{k}_z"] = float("nan")
            out[f"{k}_p"] = float("nan")
        return out
    shifts = rng.integers(min_shift, n - min_shift, size=n_shuffles)
    null = {k: np.full(n_shuffles, np.nan) for k in INDEX_KEYS}
    for i, s in enumerate(shifts):
        st = contrast_statistics(np.roll(x, int(s)), frames)
        for k in INDEX_KEYS:
            null[k][i] = st[k]
    for k in INDEX_KEYS:
        nk = null[k][np.isfinite(null[k])]
        o = obs[k]
        if not np.isfinite(o) or nk.size < 10:
            out[f"{k}_z"] = float("nan")
            out[f"{k}_p"] = float("nan")
            continue
        sd = float(np.std(nk))
        out[f"{k}_z"] = float((o - np.mean(nk)) / sd) if sd > 0 else float("nan")
        extreme = np.sum(np.abs(nk - np.mean(nk)) >= abs(o - np.mean(nk)))
        out[f"{k}_p"] = float((extreme + 1) / (nk.size + 1))
    return out
