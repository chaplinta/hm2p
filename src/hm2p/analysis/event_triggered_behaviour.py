"""Event-triggered behaviour: what the animal is doing when a cell has an event.

The usual tuning analysis conditions neural activity on behaviour. This module
reverses the direction: it detects calcium events in each cell's dF/F trace
and summarises the behavioural variables around and during those events.

Per cell and per behavioural channel three questions are asked:

1. **Onset window.** Does the channel change after event onset? The statistic
   is the mean of the channel in a post-onset window minus its mean in a
   pre-onset baseline window, averaged over events. For occupancy-like
   channels (light state, maze node type, time in the light epoch) the
   statistic is the plain mean in the post-onset window, i.e. the fraction
   of event onsets that fall in that state.
2. **During events.** Is the channel different while the cell is in an event
   than elsewhere? The statistic is the mean of the channel over event frames
   minus its mean over non-event frames.
3. **Duration coupling.** Is event duration related to the behaviour during
   the event (Spearman rank correlation)?

Both shuffle tests use circular-shift nulls: the onset train (or the whole
event mask) is shifted circularly by a random offset of at least
``min_shift_s``, which keeps the number of events, their durations and their
inter-event intervals while breaking their alignment to behaviour. This is the
same design as :func:`hm2p.analysis.event_aligned.shuffle_test`, applied in the
opposite direction (behaviour is the signal, neural events are the train).
p-values are two-sided, ``(k + 1) / (n + 1)`` on absolute deviations from the
null mean.

Frames flagged in ``bad_behav`` (animal caught on the tether) are set to NaN
in every channel, and all statistics are NaN-aware.

All functions are pure numpy/scipy/pandas — no I/O.

References
----------
Voigts J, Harnett MT. 2020. "Somatic and dendritic encoding of spatial
variables in retrosplenial cortex differs during 2D navigation." Neuron
105(2):237-245. doi:10.1016/j.neuron.2019.10.016 (event detector)

Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth show
rapid learning, sudden insight, and efficient exploration." eLife
10:e66175. doi:10.7554/eLife.66175 (maze graph: junctions and dead ends)

Weinreb C, Pearl JE, Lin S, et al. 2024. "Keypoint-MoSeq: parsing behavior by
linking point tracking to pose dynamics." Nature Methods 21:1329-1339.
doi:10.1038/s41592-024-02318-2. https://github.com/dattalab/keypoint-moseq
(behavioural syllables)
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from hm2p.analysis.event_aligned import MIN_EVENTS, _draw_shifts, _seconds_to_frames

__all__ = [
    "OCCUPANCY_CHANNELS",
    "behaviour_channels",
    "body_head_bearing_deg",
    "cell_etb_table",
    "during_event_mean",
    "during_event_shuffle",
    "duration_behaviour_correlation",
    "event_bounds",
    "event_mask_from_bounds",
    "event_onsets",
    "event_triggered_average",
    "node_type_channels",
    "onset_vs_shuffle",
    "syllable_enrichment",
    "time_in_light_epoch_s",
    "wrap_deg",
]

log = logging.getLogger(__name__)

OCCUPANCY_CHANNELS: frozenset[str] = frozenset(
    {"light_on", "node_junction", "node_corridor", "node_dead_end", "time_in_light_epoch_s"}
)
"""Channels tested by the plain post-onset mean (no baseline subtraction)."""

NODE_GROUPS: dict[str, str] = {
    "t_junction": "junction",
    "crossroads": "junction",
    "corridor": "corridor",
    "dead_end": "dead_end",
}
"""Map from :func:`hm2p.maze.topology.classify_nodes` labels to channel groups."""

SYLLABLE_ENRICHED_Z = 3.0
"""z-score above which a syllable is called enriched at event onsets."""

DEFAULT_WINDOW_S: tuple[float, float] = (0.0, 2.0)
DEFAULT_BASELINE_S: tuple[float, float] = (-5.0, -2.0)

_CELL_COLUMNS = [
    "roi",
    "channel",
    "n_events",
    "median_duration_s",
    "onset_observed",
    "onset_null_mean",
    "onset_z",
    "onset_p",
    "during_observed",
    "during_mean",
    "outside_mean",
    "during_z",
    "during_p",
    "duration_rho",
    "duration_p",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _nanmean(arr: npt.NDArray[np.floating], axis: int | None = None) -> np.ndarray:
    """``np.nanmean`` returning NaN for all-NaN slices without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(np.nanmean(arr, axis=axis))


def _null_summary(observed: float, null: np.ndarray) -> tuple[float, float, float]:
    """Null mean, z and two-sided ``(k + 1) / (n + 1)`` p for one statistic.

    Deviations are measured from the null mean, as in
    :func:`hm2p.analysis.event_aligned.shuffle_test`.
    """
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return float("nan"), float("nan"), float("nan")
    mu = float(np.mean(null))
    sd = float(np.std(null, ddof=1)) if null.size > 1 else 0.0
    k = int(np.sum(np.abs(null - mu) >= abs(observed - mu) - 1e-12))
    p = (k + 1.0) / (null.size + 1.0)
    z = (observed - mu) / sd if sd > 0 else float("nan")
    return mu, float(z), float(p)


def wrap_deg(angle_deg: npt.ArrayLike) -> np.ndarray:
    """Wrap angles in degrees to ``[-180, 180)``."""
    a = np.asarray(angle_deg, dtype=np.float64)
    return np.asarray((a + 180.0) % 360.0 - 180.0)


def body_head_bearing_deg(
    x_body: npt.ArrayLike,
    y_body: npt.ArrayLike,
    x_head: npt.ArrayLike,
    y_head: npt.ArrayLike,
) -> np.ndarray:
    """Bearing of the body→head vector in the pipeline's HD convention.

    Uses the same convention as
    ``hm2p.kinematics.compute._vector_angle_deg`` (``atan2(dx, dy) + 90``),
    which is rotated to agree with the ear-perpendicular head direction, so a
    head pointing straight along the body axis gives ``hd_deg - bearing = 0``.

    Returns
    -------
    (n_frames,) float
        Bearing in degrees, wrapped to ``[0, 360)``; NaN where any input is NaN.
    """
    dx = np.asarray(x_head, dtype=np.float64) - np.asarray(x_body, dtype=np.float64)
    dy = np.asarray(y_head, dtype=np.float64) - np.asarray(y_body, dtype=np.float64)
    return np.asarray(np.mod(90.0 + np.degrees(np.arctan2(dx, dy)), 360.0))


def time_in_light_epoch_s(light_on: npt.ArrayLike, fps: float) -> np.ndarray:
    """Time since the most recent light transition, in seconds.

    Frames before the first transition are NaN because the start of that
    epoch precedes the recording.

    Parameters
    ----------
    light_on : (n_frames,) bool
    fps : float

    Returns
    -------
    (n_frames,) float
    """
    lo = np.asarray(light_on, dtype=bool).ravel()
    _seconds_to_frames(0.0, fps, "fps")  # validates fps
    out = np.full(lo.size, np.nan)
    if lo.size < 2:
        return out
    trans = np.flatnonzero(np.diff(lo.astype(np.int8)) != 0) + 1
    if trans.size == 0:
        return out
    frames = np.arange(lo.size)
    last = np.searchsorted(trans, frames, side="right") - 1
    ok = last >= 0
    out[ok] = (frames[ok] - trans[last[ok]]) / float(fps)
    return out


def node_type_channels(
    x_maze: npt.ArrayLike,
    y_maze: npt.ArrayLike,
    maze: Any = None,
) -> dict[str, np.ndarray]:
    """One-hot maze node type per frame (junction / corridor / dead end).

    Positions are discretised onto the maze graph with
    :func:`hm2p.maze.discretize.discretize_position_fast`. T-junctions and
    crossroads are grouped as ``junction``. Frames without a valid cell are
    NaN in all three channels.

    Parameters
    ----------
    x_maze, y_maze : (n_frames,) float
        Position in maze units.
    maze : RoseMaze or None
        Maze topology; built with :func:`hm2p.maze.topology.build_rose_maze`
        when None.

    Returns
    -------
    dict
        ``"node_junction"``, ``"node_corridor"``, ``"node_dead_end"`` —
        (n_frames,) float in {0, 1, NaN}.

    References
    ----------
    Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth show
    rapid learning, sudden insight, and efficient exploration." eLife
    10:e66175. doi:10.7554/eLife.66175
    """
    from hm2p.maze.discretize import discretize_position_fast
    from hm2p.maze.topology import build_rose_maze

    if maze is None:
        maze = build_rose_maze()
    x = np.asarray(x_maze, dtype=np.float64).ravel()
    y = np.asarray(y_maze, dtype=np.float64).ravel()
    idx = discretize_position_fast(x, y, maze).astype(np.int64)
    group_of = np.array([NODE_GROUPS[maze.node_types[c]] for c in maze.cell_list], dtype=object)
    valid = idx >= 0
    out: dict[str, np.ndarray] = {}
    for g in ("junction", "corridor", "dead_end"):
        ch = np.full(idx.size, np.nan)
        ch[valid] = (group_of[idx[valid]] == g).astype(np.float64)
        out[f"node_{g}"] = ch
    return out


# ---------------------------------------------------------------------------
# Behaviour channels
# ---------------------------------------------------------------------------


def behaviour_channels(arrays: Mapping[str, Any], maze: Any = None) -> dict[str, np.ndarray]:
    """Per-frame behavioural variables for event-triggered analysis.

    Parameters
    ----------
    arrays : Mapping
        Session arrays as produced by
        ``run_celltype_programme.read_session_arrays``. Required keys:
        ``speed_cm_s``, ``ahv_deg_s``, ``active``, ``light_on``, ``fps``.
        Optional: ``dff`` (sets the frame count), ``bad_behav``, ``hd_deg``
        with ``x_head_mm``/``y_head_mm``/``x_body_mm``/``y_body_mm``
        (head-body angle), ``x_maze``/``y_maze`` (node type).
    maze : RoseMaze or None
        Passed to :func:`node_type_channels`.

    Returns
    -------
    dict[str, (n_frames,) float]
        ``speed_cm_s``, ``abs_ahv``, ``active``, ``light_on``,
        ``time_in_light_epoch_s``; ``head_body_angle_deg`` (signed, wrapped
        to ``[-180, 180)``, ``hd_deg`` minus the body→head bearing; positive
        when HD is rotated in the positive HD direction relative to the body
        axis) and ``abs_head_body_angle_deg`` when head and body positions are
        present; ``node_junction``, ``node_corridor``, ``node_dead_end`` when
        maze coordinates are present. ``bad_behav`` frames are NaN in every
        channel. ``syllable_id`` is categorical and is not included; use
        :func:`syllable_enrichment` on it directly.

    Notes
    -----
    The head-body angle is invariant to a rotation applied to both HD and
    positions, but requires that ``hd_deg`` and the positions share a frame
    of reference (both camera frame, or both rotated by the session
    orientation).
    """
    fps = float(arrays["fps"])
    if arrays.get("dff") is not None:
        n = int(np.asarray(arrays["dff"]).shape[-1])
    else:
        n = int(np.asarray(arrays["speed_cm_s"]).size)

    def _get(key: str) -> np.ndarray:
        return np.array(arrays[key], dtype=np.float64).ravel()[:n]  # copy: masked below

    ch: dict[str, np.ndarray] = {
        "speed_cm_s": _get("speed_cm_s"),
        "abs_ahv": np.abs(_get("ahv_deg_s")),
        "active": _get("active"),
        "light_on": _get("light_on"),
        "time_in_light_epoch_s": time_in_light_epoch_s(
            np.asarray(arrays["light_on"], dtype=bool).ravel()[:n], fps
        ),
    }
    pos_keys = ("x_head_mm", "y_head_mm", "x_body_mm", "y_body_mm")
    if arrays.get("hd_deg") is not None and all(arrays.get(k) is not None for k in pos_keys):
        bearing = body_head_bearing_deg(
            _get("x_body_mm"), _get("y_body_mm"), _get("x_head_mm"), _get("y_head_mm")
        )
        angle = wrap_deg(_get("hd_deg") - bearing)
        ch["head_body_angle_deg"] = angle
        ch["abs_head_body_angle_deg"] = np.abs(angle)
    if arrays.get("x_maze") is not None and arrays.get("y_maze") is not None:
        ch.update(node_type_channels(_get("x_maze"), _get("y_maze"), maze))
    if arrays.get("bad_behav") is not None:
        bad = np.asarray(arrays["bad_behav"], dtype=bool).ravel()[:n]
        for v in ch.values():
            v[: bad.size][bad] = np.nan
    return ch


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def event_bounds(
    dff_trace: npt.ArrayLike,
    fps: float,
    min_duration_s: float = 0.0,
    method: str = "vh",
) -> tuple[np.ndarray, np.ndarray]:
    """Onset and offset frames of calcium events in one dF/F trace.

    Events are detected with :func:`hm2p.analysis.cell_features.detect_cell_events`
    (Voigts & Harnett 2020 by default). Offsets are exclusive.

    Parameters
    ----------
    dff_trace : (n_frames,) float
    fps : float
    min_duration_s : float
        Events shorter than this are dropped.
    method : {"vh", "sd"}

    Returns
    -------
    onsets, offsets : (n_events,) int
    """
    from hm2p.analysis.cell_features import detect_cell_events

    min_frames = _seconds_to_frames(min_duration_s, fps, "min_duration_s")
    res = detect_cell_events(np.asarray(dff_trace, dtype=np.float64), fps, method=method)
    on = np.asarray(res.onsets, dtype=np.int64)
    off = np.asarray(res.offsets, dtype=np.int64)
    keep = (off - on) >= max(1, min_frames)
    return on[keep], off[keep]


def event_onsets(
    dff_trace: npt.ArrayLike,
    fps: float,
    min_duration_s: float = 0.0,
    method: str = "vh",
) -> tuple[np.ndarray, np.ndarray]:
    """Event onset frames and durations (s) for one dF/F trace.

    See :func:`event_bounds`.

    Returns
    -------
    onsets : (n_events,) int
    durations_s : (n_events,) float
    """
    on, off = event_bounds(dff_trace, fps, min_duration_s, method)
    return on, (off - on).astype(np.float64) / float(fps)


def event_mask_from_bounds(
    onsets: npt.ArrayLike, offsets: npt.ArrayLike, n_frames: int
) -> np.ndarray:
    """Boolean per-frame mask that is True inside ``[onset, offset)``."""
    mask = np.zeros(int(n_frames), dtype=bool)
    for a, b in zip(np.asarray(onsets).ravel(), np.asarray(offsets).ravel(), strict=True):
        mask[max(0, int(a)) : max(0, min(int(b), mask.size))] = True
    return mask


# ---------------------------------------------------------------------------
# Event-triggered average
# ---------------------------------------------------------------------------


def event_triggered_average(
    channel: npt.ArrayLike,
    onsets: npt.ArrayLike,
    fps: float,
    pre_s: float = 5.0,
    post_s: float = 5.0,
) -> dict[str, Any]:
    """Mean behavioural channel aligned to event onsets.

    Events whose ``[-pre_s, +post_s)`` window leaves the recording are
    dropped. NaN samples are ignored per time point.

    Parameters
    ----------
    channel : (n_frames,) float
    onsets : (n_events,) int
    fps : float
    pre_s, post_s : float

    Returns
    -------
    dict
        ``"mean"`` — (pre+post,) NaN-aware mean; ``"n"`` — number of events
        used; ``"time_s"`` — (pre+post,) time axis in seconds.
    """
    x = np.asarray(channel, dtype=np.float64).ravel()
    pre = _seconds_to_frames(pre_s, fps, "pre_s")
    post = _seconds_to_frames(post_s, fps, "post_s")
    offs = np.arange(-pre, post)
    ev = np.asarray(onsets, dtype=np.int64).ravel()
    ev = ev[(ev - pre >= 0) & (ev + post <= x.size)]
    time_s = offs / float(fps)
    if ev.size == 0:
        return {"mean": np.full(offs.size, np.nan), "n": 0, "time_s": time_s}
    win = x[ev[:, None] + offs[None, :]]
    return {"mean": _nanmean(win, axis=0), "n": int(ev.size), "time_s": time_s}


# ---------------------------------------------------------------------------
# Onset-window shuffle test
# ---------------------------------------------------------------------------


def _window_to_frames(window: tuple[float, float], fps: float, name: str) -> tuple[int, int]:
    """Convert a ``(start_s, stop_s)`` window to half-open frame offsets."""
    a, b = float(window[0]), float(window[1])
    if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
        raise ValueError(f"{name} must satisfy start < stop; got {window}")
    fa = int(round(a * float(fps)))
    fb = int(round(b * float(fps)))
    if fb <= fa:
        raise ValueError(f"{name} must span at least one frame")
    return fa, fb


def _window_means(
    csum: np.ndarray, ccount: np.ndarray, starts: np.ndarray, stops: np.ndarray
) -> np.ndarray:
    """NaN-aware means of ``x[start:stop]`` from cumulative sums (any shape)."""
    s = csum[stops] - csum[starts]
    c = ccount[stops] - ccount[starts]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def _onset_statistic(
    x: np.ndarray,
    events: np.ndarray,
    win: tuple[int, int],
    base: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-event statistic for an (..., n_events) array of onset frames.

    Returns the mean over events of the statistic (shape ``events.shape[:-1]``)
    and the number of finite per-event values.
    """
    finite = np.isfinite(x)
    csum = np.concatenate([[0.0], np.cumsum(np.where(finite, x, 0.0))])
    ccount = np.concatenate([[0], np.cumsum(finite, dtype=np.int64)])
    lo = win[0] if base is None else min(win[0], base[0])
    hi = win[1] if base is None else max(win[1], base[1])
    inside = (events + lo >= 0) & (events + hi <= x.size)

    def _idx(off: int) -> np.ndarray:
        # out-of-range events are clipped here and masked to NaN below
        return np.clip(events + off, 0, x.size)

    vals = _window_means(csum, ccount, _idx(win[0]), _idx(win[1]))
    if base is not None:
        vals = vals - _window_means(csum, ccount, _idx(base[0]), _idx(base[1]))
    vals = np.where(inside, vals, np.nan)
    n_ok = np.sum(np.isfinite(vals), axis=-1)
    return _nanmean(vals, axis=-1), n_ok


def onset_vs_shuffle(
    channel: npt.ArrayLike,
    onsets: npt.ArrayLike,
    fps: float,
    window: tuple[float, float] = DEFAULT_WINDOW_S,
    n_shuffles: int = 500,
    min_shift_s: float = 10.0,
    rng: np.random.Generator | None = None,
    baseline: tuple[float, float] | None = DEFAULT_BASELINE_S,
) -> dict[str, float]:
    """Behaviour after event onset against a circular-shift null.

    Statistic: mean over events of (mean channel in *window* minus mean in
    *baseline*), both relative to onset. With ``baseline=None`` the statistic
    is the mean channel in *window* (use for occupancy-like channels). Events
    whose windows leave the recording are dropped. The null shifts the onset
    train circularly by a random offset in ``[min_shift_s, T - min_shift_s]``
    (as in :func:`hm2p.analysis.event_aligned.shuffle_test`).

    Parameters
    ----------
    channel : (n_frames,) float
    onsets : (n_events,) int
    fps : float
    window : (start_s, stop_s)
        Post-onset window, half-open.
    n_shuffles : int
    min_shift_s : float
    rng : numpy.random.Generator or None
    baseline : (start_s, stop_s) or None
        Pre-onset baseline window, half-open.

    Returns
    -------
    dict
        ``"observed"``, ``"null_mean"``, ``"z"``, ``"p"`` (two-sided,
        ``(k + 1) / (n + 1)``) and ``"n"`` (usable events). Statistics are NaN
        with fewer than :data:`~hm2p.analysis.event_aligned.MIN_EVENTS`
        usable events.

    Raises
    ------
    ValueError
        For an invalid window or a recording too short for *min_shift_s*.
    """
    x = np.asarray(channel, dtype=np.float64).ravel()
    win = _window_to_frames(window, fps, "window")
    base = None if baseline is None else _window_to_frames(baseline, fps, "baseline")
    ev = np.unique(np.asarray(onsets, dtype=np.int64).ravel())
    ev = ev[(ev >= 0) & (ev < x.size)]
    nan = float("nan")
    obs, n_ok = _onset_statistic(x, ev, win, base)
    n = int(n_ok)
    out = {"observed": float(obs), "null_mean": nan, "z": nan, "p": nan, "n": n}
    if n < MIN_EVENTS:
        out["observed"] = nan
        return out
    if n_shuffles < 1:
        return out
    rng = rng if rng is not None else np.random.default_rng()
    shifts = _draw_shifts(
        x.size, n_shuffles, _seconds_to_frames(min_shift_s, fps, "min_shift_s"), rng
    )
    shifted = (ev[None, :] + shifts[:, None]) % x.size
    null, _ = _onset_statistic(x, shifted, win, base)
    mu, z, p = _null_summary(float(obs), np.asarray(null, dtype=np.float64))
    out.update({"null_mean": mu, "z": z, "p": p})
    return out


# ---------------------------------------------------------------------------
# During-event test
# ---------------------------------------------------------------------------


def during_event_mean(
    channel: npt.ArrayLike, onsets: npt.ArrayLike, offsets: npt.ArrayLike
) -> float:
    """NaN-aware mean of the channel over all frames inside events."""
    x = np.asarray(channel, dtype=np.float64).ravel()
    m = event_mask_from_bounds(onsets, offsets, x.size)
    vals = x[m]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def _circular_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``out[s] = sum_t a[t] * b[(t - s) mod T]`` for every shift ``s`` (FFT)."""
    n = a.size
    return np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=n)


def during_event_shuffle(
    channel: npt.ArrayLike,
    onsets: npt.ArrayLike,
    offsets: npt.ArrayLike,
    fps: float,
    n_shuffles: int = 500,
    min_shift_s: float = 10.0,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Behaviour during events versus elsewhere, against a circular-shift null.

    Statistic: mean channel over event frames minus mean over non-event
    frames. The null circularly shifts the whole event mask (so event
    durations and spacing are kept), evaluated for every shift at once with
    an FFT circular correlation.

    Parameters
    ----------
    channel : (n_frames,) float
    onsets, offsets : (n_events,) int
        Half-open event bounds.
    fps : float
    n_shuffles : int
    min_shift_s : float
    rng : numpy.random.Generator or None

    Returns
    -------
    dict
        ``"observed"``, ``"during"``, ``"outside"``, ``"null_mean"``, ``"z"``,
        ``"p"`` (two-sided), ``"n"`` (number of events). NaN statistics with
        fewer than ``MIN_EVENTS`` events.
    """
    x = np.asarray(channel, dtype=np.float64).ravel()
    on = np.asarray(onsets, dtype=np.int64).ravel()
    nan = float("nan")
    out = {
        "observed": nan,
        "during": nan,
        "outside": nan,
        "null_mean": nan,
        "z": nan,
        "p": nan,
        "n": int(on.size),
    }
    if on.size < MIN_EVENTS or x.size == 0:
        return out
    mask = event_mask_from_bounds(on, offsets, x.size).astype(np.float64)
    finite = np.isfinite(x)
    x0 = np.where(finite, x, 0.0)
    fin = finite.astype(np.float64)
    tot_sum, tot_cnt = float(x0.sum()), float(fin.sum())

    def _diff(s_in: np.ndarray, c_in: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c_out = tot_cnt - c_in
        with np.errstate(invalid="ignore", divide="ignore"):
            d_in = np.where(c_in > 0, s_in / np.maximum(c_in, 1), np.nan)
            d_out = np.where(c_out > 0, (tot_sum - s_in) / np.maximum(c_out, 1), np.nan)
        return np.asarray(d_in - d_out), d_in, d_out

    s_obs = np.array([float(np.sum(x0 * mask))])
    c_obs = np.array([float(np.sum(fin * mask))])
    obs, d_in, d_out = _diff(s_obs, c_obs)
    out.update({"observed": float(obs[0]), "during": float(d_in[0]), "outside": float(d_out[0])})
    if n_shuffles < 1:
        return out
    rng = rng if rng is not None else np.random.default_rng()
    shifts = _draw_shifts(
        x.size, n_shuffles, _seconds_to_frames(min_shift_s, fps, "min_shift_s"), rng
    )
    s_all = _circular_correlation(x0, mask)[shifts]
    c_all = np.rint(_circular_correlation(fin, mask)[shifts])
    null, _, _ = _diff(s_all, c_all)
    mu, z, p = _null_summary(float(obs[0]), null)
    out.update({"null_mean": mu, "z": z, "p": p})
    return out


# ---------------------------------------------------------------------------
# Syllable enrichment
# ---------------------------------------------------------------------------


def syllable_enrichment(
    syllable_id: npt.ArrayLike,
    onsets: npt.ArrayLike,
    n_shuffles: int = 500,
    rng: np.random.Generator | None = None,
    fps: float = 9.6,
    min_shift_s: float = 10.0,
) -> pd.DataFrame:
    """Fraction of event onsets in each behavioural syllable versus shuffle.

    Syllables are taken at the onset frame. Negative syllable ids are
    treated as missing; onsets on missing frames are excluded from the
    fractions (in the observed and every shuffled train). The null shifts the
    onset train circularly, as in :func:`onset_vs_shuffle`.

    Parameters
    ----------
    syllable_id : (n_frames,) int
        keypoint-MoSeq syllable per frame (Weinreb et al. 2024).
    onsets : (n_events,) int
    n_shuffles : int
    rng : numpy.random.Generator or None
    fps : float
    min_shift_s : float

    Returns
    -------
    pandas.DataFrame
        One row per syllable present in *syllable_id*: ``syllable``,
        ``n_onsets``, ``observed_frac``, ``null_mean``, ``z``, ``p``. Empty
        with fewer than ``MIN_EVENTS`` onsets on valid frames.

    References
    ----------
    Weinreb C, Pearl JE, Lin S, et al. 2024. "Keypoint-MoSeq: parsing behavior
    by linking point tracking to pose dynamics." Nature Methods 21:1329-1339.
    doi:10.1038/s41592-024-02318-2
    """
    cols = ["syllable", "n_onsets", "observed_frac", "null_mean", "z", "p"]
    raw = np.asarray(syllable_id, dtype=np.float64).ravel()
    syl = np.where(np.isfinite(raw), raw, -1).astype(np.int64)
    labels = np.unique(syl[syl >= 0])
    ev = np.unique(np.asarray(onsets, dtype=np.int64).ravel())
    ev = ev[(ev >= 0) & (ev < syl.size)]
    if labels.size == 0 or np.sum(syl[ev] >= 0) < MIN_EVENTS:
        return pd.DataFrame(columns=cols)
    code = np.full(syl.size, -1, dtype=np.int64)
    code[syl >= 0] = np.searchsorted(labels, syl[syl >= 0])
    n_lab = labels.size

    def _fractions(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = code[frames]  # (..., n_events)
        onehot = c[..., None] == np.arange(n_lab)
        counts = onehot.sum(axis=-2).astype(np.float64)
        total = counts.sum(axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(total > 0, counts / np.maximum(total, 1), np.nan), counts

    obs, counts = _fractions(ev)
    rows = []
    null = np.full((0, n_lab), np.nan)
    if n_shuffles >= 1:
        rng = rng if rng is not None else np.random.default_rng()
        shifts = _draw_shifts(
            syl.size, n_shuffles, _seconds_to_frames(min_shift_s, fps, "min_shift_s"), rng
        )
        null, _ = _fractions((ev[None, :] + shifts[:, None]) % syl.size)
    for j, lab in enumerate(labels):
        mu, z, p = _null_summary(float(obs[j]), null[:, j])
        rows.append(
            {
                "syllable": int(lab),
                "n_onsets": int(counts[j]),
                "observed_frac": float(obs[j]),
                "null_mean": mu,
                "z": z,
                "p": p,
            }
        )
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Duration coupling
# ---------------------------------------------------------------------------


def duration_behaviour_correlation(
    durations_s: npt.ArrayLike,
    onsets: npt.ArrayLike,
    channel: npt.ArrayLike,
    fps: float,
    min_events: int = 5,
) -> dict[str, float]:
    """Spearman correlation between event duration and behaviour during it.

    The behaviour for each event is the NaN-aware mean of *channel* over
    ``[onset, onset + duration)``.

    Parameters
    ----------
    durations_s, onsets : (n_events,)
    channel : (n_frames,) float
    fps : float
    min_events : int
        Minimum number of events with finite values.

    Returns
    -------
    dict
        ``"rho"``, ``"p"``, ``"n"``. NaN when too few events or a constant
        variable.
    """
    from scipy.stats import spearmanr

    x = np.asarray(channel, dtype=np.float64).ravel()
    on = np.asarray(onsets, dtype=np.int64).ravel()
    dur = np.asarray(durations_s, dtype=np.float64).ravel()
    if on.size != dur.size:
        raise ValueError("durations_s and onsets must have the same length")
    off = on + np.maximum(1, np.rint(dur * float(fps)).astype(np.int64))
    beh = np.array(
        [during_event_mean(x, [a], [b]) for a, b in zip(on, off, strict=True)], dtype=np.float64
    )
    ok = np.isfinite(beh) & np.isfinite(dur)
    n = int(ok.sum())
    nan = float("nan")
    if n < min_events or np.ptp(beh[ok]) == 0 or np.ptp(dur[ok]) == 0:
        return {"rho": nan, "p": nan, "n": n}
    res = spearmanr(dur[ok], beh[ok])
    return {"rho": float(res.statistic), "p": float(res.pvalue), "n": n}


# ---------------------------------------------------------------------------
# Per-cell table
# ---------------------------------------------------------------------------


def cell_etb_table(
    dff: npt.ArrayLike,
    behaviour: Mapping[str, npt.ArrayLike],
    fps: float,
    n_shuffles: int = 300,
    rng: np.random.Generator | None = None,
    window: tuple[float, float] = DEFAULT_WINDOW_S,
    baseline: tuple[float, float] = DEFAULT_BASELINE_S,
    min_shift_s: float = 10.0,
    min_duration_s: float = 0.0,
    occupancy_channels: frozenset[str] = OCCUPANCY_CHANNELS,
    events: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]] | None = None,
) -> pd.DataFrame:
    """Event-triggered behaviour tests for every (cell, channel) pair.

    Events are detected on each row of *dff* (:func:`event_bounds`). For each
    channel the onset-window test (:func:`onset_vs_shuffle`; no baseline for
    channels in *occupancy_channels*), the during-event test
    (:func:`during_event_shuffle`) and the duration coupling
    (:func:`duration_behaviour_correlation`) are computed.

    Parameters
    ----------
    dff : (n_cells, n_frames) float
    behaviour : Mapping[str, (n_frames,) float]
        Output of :func:`behaviour_channels`.
    fps : float
    n_shuffles : int
    rng : numpy.random.Generator or None
    window, baseline : (start_s, stop_s)
    min_shift_s, min_duration_s : float
    occupancy_channels : frozenset of str
    events : sequence of (onsets, offsets) or None
        Precomputed event bounds per cell (e.g. from :func:`event_bounds`);
        detected here when None. *min_duration_s* is not applied to them.

    Returns
    -------
    pandas.DataFrame
        One row per (roi, channel) with ``n_events``, ``median_duration_s``,
        ``onset_observed``, ``onset_null_mean``, ``onset_z``, ``onset_p``,
        ``during_observed``, ``during_mean``, ``outside_mean``, ``during_z``,
        ``during_p``, ``duration_rho``, ``duration_p``.
    """
    sig = np.asarray(dff, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[None, :]
    if sig.ndim != 2:
        raise ValueError(f"dff must be 2-D; got shape {sig.shape}")
    if events is not None and len(events) != sig.shape[0]:
        raise ValueError("events must have one (onsets, offsets) pair per cell")
    rng = rng if rng is not None else np.random.default_rng()
    n_frames = sig.shape[1]
    chans = {k: np.asarray(v, dtype=np.float64).ravel()[:n_frames] for k, v in behaviour.items()}
    rows: list[dict[str, Any]] = []
    for roi in range(sig.shape[0]):
        if events is None:
            on, off = event_bounds(sig[roi], fps, min_duration_s)
        else:
            on = np.asarray(events[roi][0], dtype=np.int64).ravel()
            off = np.asarray(events[roi][1], dtype=np.int64).ravel()
        dur = (off - on) / float(fps)
        med_dur = float(np.median(dur)) if dur.size else float("nan")
        for name, x in chans.items():
            base = None if name in occupancy_channels else baseline
            ons = onset_vs_shuffle(x, on, fps, window, n_shuffles, min_shift_s, rng, baseline=base)
            dur_res = during_event_shuffle(x, on, off, fps, n_shuffles, min_shift_s, rng)
            corr = duration_behaviour_correlation(dur, on, x, fps)
            rows.append(
                {
                    "roi": roi,
                    "channel": name,
                    "n_events": int(on.size),
                    "median_duration_s": med_dur,
                    "onset_observed": ons["observed"],
                    "onset_null_mean": ons["null_mean"],
                    "onset_z": ons["z"],
                    "onset_p": ons["p"],
                    "during_observed": dur_res["observed"],
                    "during_mean": dur_res["during"],
                    "outside_mean": dur_res["outside"],
                    "during_z": dur_res["z"],
                    "during_p": dur_res["p"],
                    "duration_rho": corr["rho"],
                    "duration_p": corr["p"],
                }
            )
    return pd.DataFrame(rows, columns=_CELL_COLUMNS)
