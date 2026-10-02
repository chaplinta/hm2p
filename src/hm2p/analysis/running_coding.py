"""How a cell's activity relates to running: graded speed, onset vs sustained, bout length.

Three per-cell descriptions of a running-related signal:

- **Speed tuning shape.** Mean activity in fixed speed bins. ``step_index``
  contrasts all running bins with the still bin (a binary run-state code);
  ``graded_rho`` is the Spearman correlation between activity and speed
  across running bins only (a graded speed code). Both are tested against a
  circular shift of the signal relative to speed.
- **Within-bout time course.** For running bouts long enough, mean activity
  early in the bout versus late in the bout; ``bout_adaptation_index`` =
  (early - late) / (early + late). Positive values mean activity is front-
  loaded at run onset; zero means sustained.
- **Bout length.** Spearman correlation across bouts between bout duration and
  the summed activity during the bout (and the mean activity). If activity is
  sustained at a constant level, summed activity scales with duration.

Speed coding in cortex: Saleem et al. 2013. "Integration of visual motion
and locomotion in mouse visual cortex." Nature Neuroscience 16:1864-1869.
doi:10.1038/nn.3567. Speed cells: Kropff et al. 2015. "Speed cells in the
medial entorhinal cortex." Nature 523:419-424. doi:10.1038/nature14622.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from hm2p.analysis.state_dynamics import segment_bouts

DEFAULT_SPEED_EDGES = (0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0)


def speed_bin_means(
    signal: npt.ArrayLike,
    speed: npt.ArrayLike,
    mask: npt.ArrayLike,
    edges: tuple[float, ...] = DEFAULT_SPEED_EDGES,
    min_frames: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean signal per speed bin; bins with fewer than *min_frames* frames are NaN.

    The last bin is open-ended (speed >= last edge).
    """
    x = np.asarray(signal, dtype=np.float64)
    s = np.asarray(speed, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(s) & np.isfinite(x)
    e = np.asarray(edges, dtype=np.float64)
    nb = e.size
    means = np.full(nb, np.nan)
    counts = np.zeros(nb, dtype=np.int64)
    centres = np.empty(nb)
    for i in range(nb):
        lo = e[i]
        hi = e[i + 1] if i + 1 < nb else np.inf
        sel = m & (s >= lo) & (s < hi)
        counts[i] = int(sel.sum())
        centres[i] = (lo + hi) / 2 if np.isfinite(hi) else lo + (e[-1] - e[-2]) / 2
        if counts[i] >= min_frames:
            means[i] = float(np.mean(x[sel]))
    return means, centres, counts


def speed_shape(means: np.ndarray, centres: np.ndarray) -> dict[str, float]:
    """Step index (running bins vs bin 0) and graded rho across running bins."""
    still = means[0]
    run = means[1:]
    run_ok = np.isfinite(run)
    run_mean = float(np.mean(run[run_ok])) if run_ok.any() else float("nan")
    denom = still + run_mean
    step = float((run_mean - still) / denom) if np.isfinite(denom) and denom > 0 else float("nan")
    if run_ok.sum() >= 3 and np.ptp(run[run_ok]) > 0:
        graded = float(stats.spearmanr(centres[1:][run_ok], run[run_ok])[0])
    else:
        graded = float("nan")
    return {"step_index": step, "graded_rho": graded}


def speed_shape_significance(
    signal: npt.ArrayLike,
    speed: npt.ArrayLike,
    mask: npt.ArrayLike,
    fps: float,
    edges: tuple[float, ...] = DEFAULT_SPEED_EDGES,
    n_shuffles: int = 200,
    min_shift_s: float = 10.0,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Observed speed-shape statistics with circular-shift z and two-sided p."""
    x = np.asarray(signal, dtype=np.float64)
    means, centres, counts = speed_bin_means(x, speed, mask, edges)
    obs = speed_shape(means, centres)
    out: dict[str, Any] = {**obs, "speed_curve": means.tolist(), "speed_counts": counts.tolist()}
    n = x.size
    min_shift = int(round(min_shift_s * fps))
    rng = rng or np.random.default_rng(0)
    if n <= 2 * min_shift or n_shuffles < 1:
        for k in obs:
            out[f"{k}_z"] = out[f"{k}_p"] = float("nan")
        return out
    null: dict[str, list[float]] = {k: [] for k in obs}
    for s in rng.integers(min_shift, n - min_shift, size=n_shuffles):
        mm, cc, _ = speed_bin_means(np.roll(x, int(s)), speed, mask, edges)
        for k, v in speed_shape(mm, cc).items():
            null[k].append(v)
    for k, o in obs.items():
        nk = np.asarray(null[k], dtype=np.float64)
        nk = nk[np.isfinite(nk)]
        if not np.isfinite(o) or nk.size < 10:
            out[f"{k}_z"] = out[f"{k}_p"] = float("nan")
            continue
        mu, sd = float(np.mean(nk)), float(np.std(nk))
        out[f"{k}_z"] = float((o - mu) / sd) if sd > 0 else float("nan")
        out[f"{k}_p"] = float((np.sum(np.abs(nk - mu) >= abs(o - mu)) + 1) / (nk.size + 1))
    return out


def run_bouts(
    speed: npt.ArrayLike,
    mask: npt.ArrayLike,
    fps: float,
    threshold: float = 2.5,
    min_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Onsets/offsets (exclusive) of running bouts lasting at least *min_s*."""
    s = np.asarray(speed, dtype=np.float64)
    running = np.asarray(mask, dtype=bool) & np.isfinite(s) & (s >= threshold)
    return segment_bouts(running, min_frames=max(1, int(round(min_s * fps))))


def bout_time_course(
    signal: npt.ArrayLike,
    onsets: npt.ArrayLike,
    offsets: npt.ArrayLike,
    fps: float,
    early_s: tuple[float, float] = (0.0, 1.0),
    late_s: tuple[float, float] = (2.0, 3.0),
) -> dict[str, float]:
    """Early vs late activity within bouts at least ``late_s[1]`` long."""
    x = np.asarray(signal, dtype=np.float64)
    on = np.asarray(onsets, dtype=np.int64)
    off = np.asarray(offsets, dtype=np.int64)
    e0, e1 = (int(round(t * fps)) for t in early_s)
    l0, l1 = (int(round(t * fps)) for t in late_s)
    keep = (off - on) >= l1
    early = [np.nanmean(x[a + e0 : a + e1]) for a in on[keep]]
    late = [np.nanmean(x[a + l0 : a + l1]) for a in on[keep]]
    nan = float("nan")
    if len(early) < 3:
        return {
            "bout_early": nan,
            "bout_late": nan,
            "bout_adaptation_index": nan,
            "n_long_bouts": 0,
        }
    em, lm = float(np.mean(early)), float(np.mean(late))
    idx = (em - lm) / (em + lm) if em + lm > 0 else nan
    return {
        "bout_early": em,
        "bout_late": lm,
        "bout_adaptation_index": float(idx),
        "n_long_bouts": int(len(early)),
    }


def bout_duration_scaling(
    signal: npt.ArrayLike,
    onsets: npt.ArrayLike,
    offsets: npt.ArrayLike,
    min_bouts: int = 8,
) -> dict[str, float]:
    """Spearman rho across bouts of duration vs summed and vs mean activity."""
    x = np.asarray(signal, dtype=np.float64)
    on = np.asarray(onsets, dtype=np.int64)
    off = np.asarray(offsets, dtype=np.int64)
    nan = float("nan")
    if on.size < min_bouts:
        return {"bout_dur_rho_sum": nan, "bout_dur_rho_mean": nan, "n_bouts": int(on.size)}
    dur = (off - on).astype(np.float64)
    total = np.array([np.nansum(x[a:b]) for a, b in zip(on, off, strict=True)])
    mean = np.array([np.nanmean(x[a:b]) for a, b in zip(on, off, strict=True)])

    def _rho(a: np.ndarray, b: np.ndarray) -> float:
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < min_bouts or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
            return nan
        return float(stats.spearmanr(a[ok], b[ok])[0])

    return {
        "bout_dur_rho_sum": _rho(dur, total),
        "bout_dur_rho_mean": _rho(dur, mean),
        "n_bouts": int(on.size),
    }
