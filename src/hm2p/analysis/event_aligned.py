"""Event-aligned responses of single cells to discrete behavioural moments.

Tests whether a cell's activity changes transiently around discrete
behavioural events: light switching on or off, movement onsets and offsets,
head-turn onsets, entries into maze junctions, and entries into dead ends.

Each cell is first tested against its own null distribution. The response
statistic is the post-minus-pre mean of the signal around each event,
averaged over events. The null is built by circularly shifting the whole
event train by a random offset of at least ``min_shift_s``, which keeps the
number of events and their inter-event intervals while breaking their
alignment to the neural signal. Circular time-shift nulls are standard
practice for testing tuning of spike and calcium signals (the same approach
is used for HD and place tuning in :mod:`hm2p.analysis.significance`); here
the event train is shifted instead of the signal, which is equivalent up to
edge handling.

Caveat for periodic event trains: when events recur at a fixed period (the
room lights alternate every 60 s), shifts close to a multiple of the period
re-align the shifted events with real events, so part of the null contains
the true response. The test then remains valid but is conservative (larger
p, smaller z) for strictly periodic events such as ``light_on``/``light_off``.

Sessions are then summarised by the fraction of responsive cells, tested
against the 5 % false-positive rate with a one-sided exact binomial test.

All functions are pure numpy/scipy/pandas — no I/O.

References
----------
Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth show
rapid learning, sudden insight, and efficient exploration." eLife
10:e66175. doi:10.7554/eLife.66175 (maze graph: junctions and dead ends)
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from hm2p.analysis.anchoring import find_transitions
from hm2p.analysis.state_dynamics import event_aligned_average, segment_bouts

__all__ = [
    "cell_event_table",
    "junction_events",
    "light_events",
    "movement_events",
    "peri_event_response",
    "population_peri_event",
    "session_summary",
    "shuffle_test",
    "turn_events",
]

log = logging.getLogger(__name__)

MIN_EVENTS = 3
"""Minimum number of usable events for a shuffle test."""

RESPONSIVE_ALPHA = 0.05
"""Per-cell significance level used to call a cell responsive."""

_SUMMARY_COLUMNS = [
    "event_type",
    "n_cells",
    "n_responsive",
    "frac_responsive",
    "frac_excited",
    "frac_inhibited",
    "median_z",
    "median_observed",
    "median_n_events",
    "binom_p",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nanmean(arr: npt.NDArray[np.floating], axis: int | None = None) -> np.ndarray:
    """``np.nanmean`` returning NaN for all-NaN slices without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(np.nanmean(arr, axis=axis))


def _seconds_to_frames(seconds: float, fps: float, name: str) -> int:
    """Convert a non-negative duration in seconds to a whole number of frames."""
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be a positive finite number; got {fps}")
    seconds = float(seconds)
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{name} must be a non-negative finite number; got {seconds}")
    return int(round(seconds * fps))


def _as_events(event_frames: npt.ArrayLike) -> np.ndarray:
    """Sorted unique int64 event frames."""
    return np.unique(np.asarray(event_frames, dtype=np.int64).ravel())


def _drop_in_mask(events: np.ndarray, exclude: npt.NDArray[np.bool_] | None) -> np.ndarray:
    """Remove events whose frame is True in *exclude*."""
    if exclude is None or events.size == 0:
        return events
    ex = np.asarray(exclude, dtype=bool)
    inside = (events >= 0) & (events < ex.size)
    keep = np.ones(events.size, dtype=bool)
    keep[inside] = ~ex[events[inside]]
    return events[keep]


def _window_frames(fps: float, pre_s: float, post_s: float, gap_s: float) -> tuple[int, int, int]:
    """Validate and convert the peri-event window to frames."""
    pre = _seconds_to_frames(pre_s, fps, "pre_s")
    post = _seconds_to_frames(post_s, fps, "post_s")
    gap = _seconds_to_frames(gap_s, fps, "gap_s")
    if post < 1:
        raise ValueError("post_s must span at least one frame")
    if pre - gap < 1:
        raise ValueError("pre_s - gap_s must span at least one frame")
    return pre, post, gap


def _delta_matrix(
    signals: np.ndarray, events: np.ndarray, pre: int, post: int, gap: int
) -> np.ndarray:
    """Per-cell, per-event post-minus-pre means (vectorised).

    Parameters
    ----------
    signals : (n_cells, n_frames) float
    events : (n_events,) int
        Event frames; events whose window leaves the recording are dropped.
    pre, post, gap : int
        Window sizes in frames. Pre window is ``[-pre, -gap)``, post window
        ``[0, post)``.

    Returns
    -------
    (n_cells, n_kept_events) float
    """
    n_frames = signals.shape[1]
    ev = events[(events - pre >= 0) & (events + post <= n_frames)]
    if ev.size == 0:
        return np.empty((signals.shape[0], 0), dtype=np.float64)
    offsets = np.arange(-pre, post)
    windows = signals[:, ev[:, None] + offsets[None, :]]  # (cells, events, win)
    pre_mean = _nanmean(windows[:, :, : pre - gap], axis=2)
    post_mean = _nanmean(windows[:, :, pre:], axis=2)
    return np.asarray(post_mean - pre_mean, dtype=np.float64)


def _draw_shifts(
    n_frames: int, n_shuffles: int, min_shift_frames: int, rng: np.random.Generator
) -> np.ndarray:
    """Random circular offsets in ``[min_shift, n_frames - min_shift]``.

    Raises
    ------
    ValueError
        If the recording is too short for the requested minimum shift.
    """
    lo = max(1, int(min_shift_frames))
    hi = int(n_frames) - lo
    if hi < lo:
        raise ValueError(
            f"recording of {n_frames} frames is too short for a minimum shift of {lo} frames"
        )
    return rng.integers(lo, hi + 1, size=int(n_shuffles))


def _shuffle_stats(
    signals: np.ndarray,
    events: np.ndarray,
    fps: float,
    n_shuffles: int,
    min_shift_s: float,
    rng: np.random.Generator,
    pre_s: float,
    post_s: float,
    gap_s: float,
) -> dict[str, np.ndarray]:
    """Observed mean delta and circular-shift null statistics for every cell.

    Returns arrays of length ``n_cells`` keyed ``observed``, ``null_mean``,
    ``null_sd``, ``z``, ``p``, ``n_events``. Cells with fewer than
    :data:`MIN_EVENTS` usable events get NaN statistics.
    """
    n_cells, n_frames = signals.shape
    pre, post, gap = _window_frames(fps, pre_s, post_s, gap_s)
    nan = np.full(n_cells, np.nan)
    deltas = _delta_matrix(signals, events, pre, post, gap)
    n_events = np.sum(np.isfinite(deltas), axis=1).astype(int)
    observed = _nanmean(deltas, axis=1) if deltas.shape[1] else nan.copy()
    out = {
        "observed": nan.copy(),
        "null_mean": nan.copy(),
        "null_sd": nan.copy(),
        "z": nan.copy(),
        "p": nan.copy(),
        "n_events": n_events,
    }
    ok = n_events >= MIN_EVENTS
    if not ok.any() or n_shuffles < 1:
        out["observed"][ok] = observed[ok]
        return out
    min_shift = _seconds_to_frames(min_shift_s, fps, "min_shift_s")
    shifts = _draw_shifts(n_frames, n_shuffles, min_shift, rng)
    null = np.full((n_cells, shifts.size), np.nan)
    for j, s in enumerate(shifts):
        d = _delta_matrix(signals, np.sort((events + s) % n_frames), pre, post, gap)
        if d.shape[1]:
            null[:, j] = _nanmean(d, axis=1)
    null_mean = _nanmean(null, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        null_sd = np.asarray(np.nanstd(null, axis=1, ddof=1))
    dev_null = np.abs(null - null_mean[:, None])
    dev_obs = np.abs(observed - null_mean)[:, None]
    n_valid = np.sum(np.isfinite(null), axis=1)
    k = np.sum(dev_null >= dev_obs - 1e-12, axis=1)
    p = (k + 1.0) / (n_valid + 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(null_sd > 0, (observed - null_mean) / null_sd, np.nan)
    good = ok & (n_valid > 0) & np.isfinite(observed)
    out["observed"][ok] = observed[ok]
    out["null_mean"][good] = null_mean[good]
    out["null_sd"][good] = null_sd[good]
    out["z"][good] = z[good]
    out["p"][good] = p[good]
    return out


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------


def light_events(
    light_on: npt.NDArray[np.bool_],
    bad_behav: npt.NDArray[np.bool_] | None = None,
) -> dict[str, np.ndarray]:
    """Frames at which the room lights switch on or off.

    Parameters
    ----------
    light_on : (n_frames,) bool
        Light state per frame.
    bad_behav : (n_frames,) bool or None
        Frames with artefactual behaviour (animal caught on the tether);
        transitions falling on such a frame are dropped.

    Returns
    -------
    dict
        ``"light_on"`` — first light frame after darkness (dark→light).
        ``"light_off"`` — first dark frame after light (light→dark).
    """
    lo = np.asarray(light_on, dtype=bool).ravel()
    tr = find_transitions(lo)
    return {
        "light_on": _drop_in_mask(_as_events(tr["dark_to_light"]), bad_behav),
        "light_off": _drop_in_mask(_as_events(tr["light_to_dark"]), bad_behav),
    }


def movement_events(
    active: npt.NDArray[np.bool_],
    fps: float,
    min_still_s: float = 1.0,
    min_move_s: float = 1.0,
    bad_behav: npt.NDArray[np.bool_] | None = None,
) -> dict[str, np.ndarray]:
    """Movement onsets and offsets flanked by sustained opposite states.

    A movement onset is the first frame of a moving bout lasting at least
    ``min_move_s`` that directly follows a still bout lasting at least
    ``min_still_s``. A movement offset is the first frame of a still bout
    lasting at least ``min_still_s`` that directly follows a moving bout
    lasting at least ``min_move_s``.

    Parameters
    ----------
    active : (n_frames,) bool
        Movement state per frame.
    fps : float
        Sampling rate.
    min_still_s, min_move_s : float
        Minimum durations of the still and moving bouts around the event.
    bad_behav : (n_frames,) bool or None
        Events whose qualifying still or moving bout overlaps a bad-behaviour
        frame are dropped (the tether artefact produces spurious immobility).

    Returns
    -------
    dict
        ``"move_onset"`` and ``"move_offset"`` — (n_events,) int frames.
    """
    act = np.asarray(active, dtype=bool).ravel()
    n_still = max(1, _seconds_to_frames(min_still_s, fps, "min_still_s"))
    n_move = max(1, _seconds_to_frames(min_move_s, fps, "min_move_s"))
    m_on, m_off = segment_bouts(act, min_frames=1)
    s_on, s_off = segment_bouts(~act, min_frames=1)
    still_len_by_end = dict(zip(s_off.tolist(), (s_off - s_on).tolist(), strict=True))
    move_len_by_end = dict(zip(m_off.tolist(), (m_off - m_on).tolist(), strict=True))
    bad = None if bad_behav is None else np.asarray(bad_behav, dtype=bool).ravel()

    def _clean(frame: int, before: int, after: int) -> bool:
        if bad is None:
            return True
        return not bool(bad[max(0, frame - before) : frame + after].any())

    onsets = [
        int(on)
        for on, off in zip(m_on, m_off, strict=True)
        if off - on >= n_move
        and still_len_by_end.get(int(on), 0) >= n_still
        and _clean(int(on), n_still, n_move)
    ]
    offsets = [
        int(on)
        for on, off in zip(s_on, s_off, strict=True)
        if off - on >= n_still
        and move_len_by_end.get(int(on), 0) >= n_move
        and _clean(int(on), n_move, n_still)
    ]
    return {
        "move_onset": np.asarray(onsets, dtype=np.int64),
        "move_offset": np.asarray(offsets, dtype=np.int64),
    }


def turn_events(
    ahv_deg_s: npt.NDArray[np.floating],
    fps: float,
    threshold_deg_s: float = 90.0,
    quiet_s: float = 1.0,
    mask: npt.NDArray[np.bool_] | None = None,
) -> dict[str, np.ndarray]:
    """Onsets of fast head turns preceded by a quiet period.

    A turn onset is a frame where ``|AHV| >= threshold_deg_s`` and every one
    of the preceding ``quiet_s`` seconds of frames had a finite
    ``|AHV| < threshold_deg_s`` (and was valid under *mask*, if given). The
    onset frame itself must be valid under *mask*.

    Direction follows the sign convention of :mod:`hm2p.analysis.ahv`:
    positive AHV is clockwise (viewed from above), labelled ``turn_right``;
    negative AHV is counter-clockwise, labelled ``turn_left``.

    Parameters
    ----------
    ahv_deg_s : (n_frames,) float
        Angular head velocity in deg/s. NaN frames are neither quiet nor
        turning.
    fps : float
        Sampling rate.
    threshold_deg_s : float
        Absolute AHV threshold defining a fast turn.
    quiet_s : float
        Minimum duration below threshold before the onset.
    mask : (n_frames,) bool or None
        Valid frames.

    Returns
    -------
    dict
        ``"turn_left"`` and ``"turn_right"`` — (n_events,) int frames.
    """
    ahv = np.asarray(ahv_deg_s, dtype=np.float64).ravel()
    if not np.isfinite(threshold_deg_s) or threshold_deg_s <= 0:
        raise ValueError(f"threshold_deg_s must be positive; got {threshold_deg_s}")
    q = max(1, _seconds_to_frames(quiet_s, fps, "quiet_s"))
    valid = np.isfinite(ahv)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).ravel()
    mag = np.abs(np.where(np.isfinite(ahv), ahv, 0.0))
    above = valid & (mag >= threshold_deg_s)
    quiet = valid & (mag < threshold_deg_s)
    csum = np.concatenate([[0], np.cumsum(quiet, dtype=np.int64)])
    frames = np.arange(ahv.size)
    cand = frames[above & (frames >= q)]
    onsets = cand[(csum[cand] - csum[cand - q]) == q]
    return {
        "turn_left": onsets[ahv[onsets] < 0].astype(np.int64),
        "turn_right": onsets[ahv[onsets] > 0].astype(np.int64),
    }


def junction_events(
    x_maze: npt.NDArray[np.floating],
    y_maze: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_] | None,
    maze: Any = None,
    min_pre_frames: int = 2,
) -> dict[str, np.ndarray]:
    """Entries into maze junctions and dead ends.

    Positions are discretised onto the maze graph with
    :func:`hm2p.maze.discretize.discretize_position_fast`; frames outside
    *mask* are set to the invalid cell -1.

    ``junction_entry`` is the first frame of each junction visit returned by
    :func:`hm2p.maze.neural.extract_junction_events` (valid preceding and
    following cells required). ``dead_end_entry`` is the first frame of each
    run in a degree-1 cell (``maze.dead_ends``) whose preceding run is a valid,
    different cell lasting at least ``min_pre_frames`` frames.

    Parameters
    ----------
    x_maze, y_maze : (n_frames,) float
        Position in maze units.
    mask : (n_frames,) bool or None
        Valid frames.
    maze : RoseMaze or None
        Maze topology; built with :func:`build_rose_maze` when None.
    min_pre_frames : int
        Minimum length of the run preceding the entry.

    Returns
    -------
    dict
        ``"junction_entry"`` and ``"dead_end_entry"`` — (n_events,) int frames.

    References
    ----------
    Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth show
    rapid learning, sudden insight, and efficient exploration." eLife
    10:e66175. doi:10.7554/eLife.66175
    """
    from hm2p.maze.discretize import discretize_position_fast
    from hm2p.maze.neural import extract_junction_events
    from hm2p.maze.topology import build_rose_maze

    if maze is None:
        maze = build_rose_maze()
    x = np.asarray(x_maze, dtype=np.float64).ravel()
    y = np.asarray(y_maze, dtype=np.float64).ravel()
    cell_idx = discretize_position_fast(x, y, maze).astype(np.int64)
    if mask is not None:
        cell_idx[~np.asarray(mask, dtype=bool).ravel()] = -1
    junc = [int(e["junction_frame"]) for e in extract_junction_events(cell_idx, maze)]

    dead = np.zeros(maze.n_cells, dtype=bool)
    for c in maze.dead_ends:
        dead[maze.cell_to_idx[c]] = True
    entries: list[int] = []
    if cell_idx.size:
        change = np.flatnonzero(np.diff(cell_idx) != 0) + 1
        starts = np.concatenate([[0], change])
        lengths = np.diff(np.concatenate([starts, [cell_idx.size]]))
        for r in range(1, starts.size):
            cur = int(cell_idx[starts[r]])
            prev = int(cell_idx[starts[r - 1]])
            if cur >= 0 and dead[cur] and prev >= 0 and lengths[r - 1] >= min_pre_frames:
                entries.append(int(starts[r]))
    return {
        "junction_entry": np.asarray(sorted(junc), dtype=np.int64),
        "dead_end_entry": np.asarray(entries, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Per-cell responses
# ---------------------------------------------------------------------------


def peri_event_response(
    signal: npt.NDArray[np.floating],
    event_frames: npt.ArrayLike,
    fps: float,
    pre_s: float = 2.0,
    post_s: float = 2.0,
    gap_s: float = 0.0,
) -> dict[str, Any]:
    """Post-minus-pre response of one signal around a set of events.

    The pre window is ``[-pre_s, -gap_s)`` and the post window ``[0, post_s)``
    relative to each event frame. Events whose full window does not fit in
    the recording are dropped.

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal (inferred spike rate or dF/F). NaN samples are ignored.
    event_frames : (n_events,) int
        Event frames.
    fps : float
        Sampling rate.
    pre_s, post_s : float
        Window lengths in seconds.
    gap_s : float
        Gap before the event excluded from the pre window.

    Returns
    -------
    dict
        ``"delta"`` — (n_kept,) post-minus-pre mean per event.
        ``"mean_delta"`` — mean of ``delta`` (NaN with no events).
        ``"n_events"`` — number of events with a finite ``delta``.
        ``"time_s"`` — (pre+post,) time axis in seconds.
        ``"mean_trace"`` — (pre+post,) event-aligned mean signal.
    """
    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim != 1:
        raise ValueError(f"signal must be 1-D; got shape {sig.shape}")
    pre, post, gap = _window_frames(fps, pre_s, post_s, gap_s)
    events = _as_events(event_frames)
    aligned = event_aligned_average(sig, events, pre, post)
    delta = _delta_matrix(sig[None, :], events, pre, post, gap)[0]
    finite = np.isfinite(delta)
    return {
        "delta": delta,
        "mean_delta": float(np.mean(delta[finite])) if finite.any() else float("nan"),
        "n_events": int(finite.sum()),
        "time_s": aligned["time_axis"] / float(fps),
        "mean_trace": aligned["mean"],
    }


def shuffle_test(
    signal: npt.NDArray[np.floating],
    event_frames: npt.ArrayLike,
    fps: float,
    n_shuffles: int = 500,
    min_shift_s: float = 10.0,
    rng: np.random.Generator | None = None,
    pre_s: float = 2.0,
    post_s: float = 2.0,
    gap_s: float = 0.0,
) -> dict[str, float]:
    """Test a peri-event response against a circular-shift null.

    The event train is shifted circularly by a random offset in
    ``[min_shift_s, T - min_shift_s]`` (``T`` the recording length) for each
    shuffle, keeping the number of events and their inter-event intervals.
    The statistic is the mean post-minus-pre delta
    (:func:`peri_event_response`). For strictly periodic event trains the
    test is conservative (see the module docstring).

    Parameters
    ----------
    signal : (n_frames,) float
    event_frames : (n_events,) int
    fps : float
    n_shuffles : int
    min_shift_s : float
        Minimum circular offset in seconds.
    rng : numpy.random.Generator or None
    pre_s, post_s, gap_s : float
        Window parameters passed to :func:`peri_event_response`.

    Returns
    -------
    dict
        ``"observed"``, ``"null_mean"``, ``"null_sd"``, ``"z"`` (observed
        minus null mean over null SD), ``"p"`` (two-sided, deviations from the
        null mean, ``(k + 1) / (n + 1)``), ``"n_events"``. All statistics are
        NaN when fewer than 3 events are usable.

    Raises
    ------
    ValueError
        If the recording is shorter than twice the minimum shift.
    """
    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim != 1:
        raise ValueError(f"signal must be 1-D; got shape {sig.shape}")
    rng = rng if rng is not None else np.random.default_rng()
    res = _shuffle_stats(
        sig[None, :],
        _as_events(event_frames),
        fps,
        n_shuffles,
        min_shift_s,
        rng,
        pre_s,
        post_s,
        gap_s,
    )
    out = {k: float(v[0]) for k, v in res.items() if k != "n_events"}
    out["n_events"] = int(res["n_events"][0])
    return out


def cell_event_table(
    signals: npt.NDArray[np.floating],
    events: Mapping[str, npt.ArrayLike],
    fps: float,
    n_shuffles: int = 500,
    rng: np.random.Generator | None = None,
    min_shift_s: float = 10.0,
    pre_s: float = 2.0,
    post_s: float = 2.0,
    gap_s: float = 0.0,
) -> pd.DataFrame:
    """Shuffle test of every cell against every event type.

    All cells share the same circular offsets within an event type, but each
    cell is compared with its own null distribution.

    Parameters
    ----------
    signals : (n_cells, n_frames) float
    events : Mapping[str, (n_events,) int]
        Event frames per event type.
    fps : float
    n_shuffles : int
    rng : numpy.random.Generator or None
    min_shift_s, pre_s, post_s, gap_s : float
        See :func:`shuffle_test`.

    Returns
    -------
    pandas.DataFrame
        One row per (roi, event_type) with ``roi`` (row index into
        *signals*), ``event_type``, ``observed``, ``null_mean``, ``null_sd``,
        ``z``, ``p``, ``n_events``, ``responsive`` (``p < 0.05``) and
        ``direction`` (+1 above the null mean, -1 below, 0 when undefined).
    """
    sig = np.asarray(signals, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[None, :]
    if sig.ndim != 2:
        raise ValueError(f"signals must be 2-D; got shape {sig.shape}")
    rng = rng if rng is not None else np.random.default_rng()
    tables = []
    for name, frames in events.items():
        res = _shuffle_stats(
            sig, _as_events(frames), fps, n_shuffles, min_shift_s, rng, pre_s, post_s, gap_s
        )
        df = pd.DataFrame(res)
        df.insert(0, "event_type", name)
        df.insert(0, "roi", np.arange(sig.shape[0]))
        tables.append(df)
    cols = ["roi", "event_type", "observed", "null_mean", "null_sd", "z", "p", "n_events"]
    if not tables:
        return pd.DataFrame(columns=cols + ["responsive", "direction"])
    out = pd.concat(tables, ignore_index=True)[cols]
    out["responsive"] = (out["p"] < RESPONSIVE_ALPHA).fillna(False).astype(bool)
    out["direction"] = np.sign(out["z"].fillna(0.0)).astype(int)
    return out


def population_peri_event(
    signals: npt.NDArray[np.floating],
    event_frames: npt.ArrayLike,
    fps: float,
    pre_s: float = 5.0,
    post_s: float = 5.0,
) -> dict[str, Any]:
    """Population-mean event-aligned trace.

    Each cell's event-aligned mean (:func:`event_aligned_average`) is
    averaged across cells.

    Parameters
    ----------
    signals : (n_cells, n_frames) float
    event_frames : (n_events,) int
    fps : float
    pre_s, post_s : float

    Returns
    -------
    dict
        ``"time_s"`` — (pre+post,) seconds; ``"mean"`` — population mean;
        ``"sem"`` — SEM across cells; ``"n_events"``; ``"n_cells"``.
    """
    sig = np.asarray(signals, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[None, :]
    pre = _seconds_to_frames(pre_s, fps, "pre_s")
    post = _seconds_to_frames(post_s, fps, "post_s")
    events = _as_events(event_frames)
    per_cell = []
    n_events = 0
    time_axis = np.arange(-pre, post, dtype=np.float64)
    for row in sig:
        res = event_aligned_average(row, events, pre, post)
        per_cell.append(res["mean"])
        n_events = res["n_events"]
    stack = np.vstack(per_cell) if per_cell else np.empty((0, pre + post))
    mean = _nanmean(stack, axis=0) if stack.shape[0] else np.full(pre + post, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n_ok = np.sum(np.isfinite(stack), axis=0)
        sd = np.nanstd(stack, axis=0, ddof=1) if stack.shape[0] > 1 else np.full(pre + post, 0.0)
    sem = np.where(n_ok > 1, sd / np.sqrt(np.maximum(n_ok, 1)), np.nan)
    return {
        "time_s": time_axis / float(fps),
        "mean": np.asarray(mean, dtype=np.float64),
        "sem": np.asarray(sem, dtype=np.float64),
        "n_events": int(n_events),
        "n_cells": int(sig.shape[0]),
    }


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------


def _binom_greater(k: int, n: int, p0: float = RESPONSIVE_ALPHA) -> float:
    """One-sided exact binomial p-value for ``k`` of ``n`` exceeding ``p0``."""
    from scipy.stats import binomtest

    if n < 1:
        return float("nan")
    return float(binomtest(int(k), int(n), p0, alternative="greater").pvalue)


def session_summary(cell_table: pd.DataFrame) -> pd.DataFrame:
    """Fraction of responsive cells and median effect per event type.

    Only cells with a defined p-value (enough events) are counted.

    Parameters
    ----------
    cell_table : pandas.DataFrame
        Output of :func:`cell_event_table`.

    Returns
    -------
    pandas.DataFrame
        One row per event type: ``n_cells``, ``n_responsive``,
        ``frac_responsive``, ``frac_excited``, ``frac_inhibited``,
        ``median_z``, ``median_observed``, ``median_n_events`` and
        ``binom_p`` — the one-sided exact binomial test of the responsive
        fraction against the 5 % false-positive rate.
    """
    rows = []
    if cell_table.empty:
        return pd.DataFrame(columns=_SUMMARY_COLUMNS)
    for et, g in cell_table.groupby("event_type", sort=False):
        tested = g[np.isfinite(g["p"].to_numpy(dtype=float))]
        n = len(tested)
        resp = tested["responsive"].to_numpy(dtype=bool)
        direc = tested["direction"].to_numpy(dtype=int)
        n_resp = int(resp.sum())
        rows.append(
            {
                "event_type": et,
                "n_cells": n,
                "n_responsive": n_resp,
                "frac_responsive": n_resp / n if n else np.nan,
                "frac_excited": float(np.sum(resp & (direc > 0))) / n if n else np.nan,
                "frac_inhibited": float(np.sum(resp & (direc < 0))) / n if n else np.nan,
                "median_z": float(np.nanmedian(tested["z"])) if n else np.nan,
                "median_observed": float(np.nanmedian(tested["observed"])) if n else np.nan,
                "median_n_events": float(np.median(tested["n_events"])) if n else np.nan,
                "binom_p": _binom_greater(n_resp, n),
            }
        )
    return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)
