"""Movement-state dynamics of neural activity — bouts, event-aligned averages, decay.

Quantifies how a neural signal evolves around movement onsets and offsets and
during immobility. The motivating comparison is between two RSP excitatory
populations: Penk+ neurons, reported to be low-rheobase and non-persistently
firing, versus Penk-negative CamKII+ neurons. A transient population responds
at movement onset and decays quickly once the animal stops; a sustained
population holds its activity through long movement bouts and through
immobility.

All functions are pure numpy/scipy — no I/O.

References
----------
Brennan EKW, Sudhakar SK, Jedrasiak-Cape I, John TT, Ahmed OJ. 2020.
"Hyperexcitable Neurons Enable Precise and Persistent Information Encoding in
the Superficial Retrosplenial Cortex." Cell Reports 30(5):1598-1612.
doi:10.1016/j.celrep.2019.12.093

Jedrasiak-Cape I, et al. 2025. "Cell type-specific cortical circuits of the
retrosplenial cortex." Progress in Neurobiology. doi:10.1101/2024.06.04.597341

Skaggs WE, McNaughton BL, Gothard KM, Markus EJ. 1993. "An
information-theoretic approach to deciphering the hippocampal code."
NeurIPS 5:1030-1037.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt

__all__ = [
    "activity_vs_bout_duration",
    "bout_durations",
    "event_aligned_average",
    "immobility_decay",
    "movement_state_summary",
    "onset_latency_frames",
    "segment_bouts",
    "syllable_conditioned_activity",
    "syllable_information",
    "transient_index",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nanmean(arr: npt.NDArray[np.floating], axis: int | None = None) -> np.ndarray:
    """``np.nanmean`` that returns NaN for all-NaN slices without warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(np.nanmean(arr, axis=axis))


def _nansem(arr: npt.NDArray[np.floating], axis: int = 0) -> np.ndarray:
    """Standard error of the mean ignoring NaN, NaN where fewer than 2 samples."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.asarray(np.nanstd(arr, axis=axis, ddof=1))
    counts = np.sum(np.isfinite(arr), axis=axis)
    sem = np.full(sd.shape, np.nan, dtype=np.float64)
    enough = counts > 1
    sem[enough] = sd[enough] / np.sqrt(counts[enough])
    return sem


def _as_float_1d(arr: npt.ArrayLike, name: str) -> np.ndarray:
    """Coerce to a 1-D float array, raising on the wrong dimensionality."""
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got shape {out.shape}")
    return out


def _as_bool_1d(arr: npt.ArrayLike, name: str) -> np.ndarray:
    """Coerce to a 1-D boolean array, raising on the wrong dimensionality."""
    out = np.asarray(arr, dtype=bool)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got shape {out.shape}")
    return out


def _check_fps(fps: float) -> float:
    """Validate a sampling rate."""
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be a positive finite number; got {fps}")
    return fps


# ---------------------------------------------------------------------------
# Bout segmentation
# ---------------------------------------------------------------------------


def segment_bouts(
    state: npt.NDArray[np.bool_],
    min_frames: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment a boolean state vector into contiguous bouts.

    Parameters
    ----------
    state : (n_frames,) bool
        State per frame (e.g. ``active``). Contiguous runs of True form bouts.
    min_frames : int
        Minimum bout length in frames. Bouts shorter than this are dropped.

    Returns
    -------
    onsets : (n_bouts,) int
        First frame of each bout (inclusive).
    offsets : (n_bouts,) int
        One past the last frame of each bout (exclusive), so bout length is
        ``offsets - onsets``.

    Raises
    ------
    ValueError
        If ``state`` is not 1-D or ``min_frames`` is less than 1.
    """
    state_arr = _as_bool_1d(state, "state")
    if int(min_frames) < 1:
        raise ValueError(f"min_frames must be >= 1; got {min_frames}")
    min_frames = int(min_frames)

    if state_arr.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    padded = np.concatenate([[0], state_arr.astype(np.int8), [0]])
    diff = np.diff(padded)
    onsets = np.flatnonzero(diff == 1)
    offsets = np.flatnonzero(diff == -1)

    keep = (offsets - onsets) >= min_frames
    return onsets[keep].astype(int), offsets[keep].astype(int)


def bout_durations(
    onsets: npt.NDArray[np.integer],
    offsets: npt.NDArray[np.integer],
    fps: float,
) -> np.ndarray:
    """Convert bout onset/offset frame indices to durations in seconds.

    Parameters
    ----------
    onsets : (n_bouts,) int
        Bout onset frames (inclusive), as returned by :func:`segment_bouts`.
    offsets : (n_bouts,) int
        Bout offset frames (exclusive).
    fps : float
        Sampling rate in frames per second.

    Returns
    -------
    (n_bouts,) float
        Bout duration in seconds.

    Raises
    ------
    ValueError
        If the two index arrays differ in shape or ``fps`` is not positive.
    """
    onsets_arr = np.asarray(onsets, dtype=np.int64)
    offsets_arr = np.asarray(offsets, dtype=np.int64)
    if onsets_arr.shape != offsets_arr.shape:
        raise ValueError(
            f"onsets and offsets must have the same shape; got {onsets_arr.shape} "
            f"and {offsets_arr.shape}"
        )
    fps = _check_fps(fps)
    return (offsets_arr - onsets_arr).astype(np.float64) / fps


# ---------------------------------------------------------------------------
# Event-aligned averaging
# ---------------------------------------------------------------------------


def event_aligned_average(
    signal: npt.NDArray[np.floating],
    event_frames: npt.NDArray[np.integer],
    pre_frames: int,
    post_frames: int,
    baseline_frames: int | None = None,
) -> dict:
    """Average a signal in windows aligned to a set of event frames.

    Events whose window would extend beyond either end of the signal are
    dropped. NaN samples (e.g. excluded frames) are ignored by the average.

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal. May contain NaN for excluded frames.
    event_frames : (n_events,) int
        Frame indices of the events to align to. The event frame itself is the
        first sample of the post-event window.
    pre_frames : int
        Number of frames before each event to include.
    post_frames : int
        Number of frames from the event onwards to include.
    baseline_frames : int or None
        If given, subtract the mean of the first ``baseline_frames`` samples of
        each window from that window before averaging.

    Returns
    -------
    dict
        ``"mean"`` — (pre+post,) mean across events.
        ``"sem"`` — (pre+post,) standard error across events.
        ``"n_events"`` — number of events that contributed.
        ``"time_axis"`` — (pre+post,) frame offset relative to the event,
        running from ``-pre_frames`` to ``post_frames - 1``.
        ``"windows"`` — (n_events, pre+post) per-event windows.

    Raises
    ------
    ValueError
        If the window lengths are negative, the total window is empty, or
        ``baseline_frames`` is outside the window.
    """
    sig = _as_float_1d(signal, "signal")
    events = np.asarray(event_frames, dtype=np.int64).ravel()

    pre_frames = int(pre_frames)
    post_frames = int(post_frames)
    if pre_frames < 0 or post_frames < 0:
        raise ValueError(
            f"pre_frames and post_frames must be >= 0; got {pre_frames}, {post_frames}"
        )
    win_len = pre_frames + post_frames
    if win_len < 1:
        raise ValueError("pre_frames + post_frames must be >= 1")

    if baseline_frames is not None:
        baseline_frames = int(baseline_frames)
        if baseline_frames < 1 or baseline_frames > win_len:
            raise ValueError(f"baseline_frames must be in [1, {win_len}]; got {baseline_frames}")

    time_axis = np.arange(-pre_frames, post_frames, dtype=np.float64)

    n = sig.size
    valid = (events - pre_frames >= 0) & (events + post_frames <= n)
    kept = events[valid]

    windows = np.empty((kept.size, win_len), dtype=np.float64)
    for i, ev in enumerate(kept):
        windows[i] = sig[ev - pre_frames : ev + post_frames]

    if baseline_frames is not None and kept.size > 0:
        base = _nanmean(windows[:, :baseline_frames], axis=1)
        windows = windows - base[:, np.newaxis]

    if kept.size == 0:
        mean = np.full(win_len, np.nan, dtype=np.float64)
        sem = np.full(win_len, np.nan, dtype=np.float64)
    else:
        mean = _nanmean(windows, axis=0)
        sem = _nansem(windows, axis=0)

    return {
        "mean": mean,
        "sem": sem,
        "n_events": int(kept.size),
        "time_axis": time_axis,
        "windows": windows,
    }


def transient_index(
    aligned_mean: npt.NDArray[np.floating],
    pre_frames: int,
    peak_window_frames: int,
    sustained_window_frames: int,
    eps: float = 1e-9,
) -> float:
    """Ratio of the early peak response to the late sustained response.

    The index is ``(peak - baseline) / (sustained - baseline)`` where the peak
    is the maximum of the early post-event window, the sustained level is the
    mean of the final ``sustained_window_frames`` samples of the window, and
    the baseline is the mean of the pre-event samples. Values near 1 indicate a
    sustained response; large values indicate a transient response that decays.

    Parameters
    ----------
    aligned_mean : (pre+post,) float
        Event-aligned mean trace, as returned by :func:`event_aligned_average`.
    pre_frames : int
        Number of pre-event samples in ``aligned_mean``. If 0, the baseline is
        taken as 0.
    peak_window_frames : int
        Length of the early post-event window searched for the peak.
    sustained_window_frames : int
        Length of the late window (taken from the end of the trace) used as the
        sustained level.
    eps : float
        Tolerance below which the sustained response is treated as zero.

    Returns
    -------
    float
        Transient index; NaN when the sustained response is within ``eps`` of
        the baseline, or when either window contains no finite samples.

    Raises
    ------
    ValueError
        If the window lengths are not positive or exceed the trace length.
    """
    trace = _as_float_1d(aligned_mean, "aligned_mean")
    pre_frames = int(pre_frames)
    peak_window_frames = int(peak_window_frames)
    sustained_window_frames = int(sustained_window_frames)

    if pre_frames < 0 or pre_frames >= trace.size:
        raise ValueError(f"pre_frames must be in [0, {trace.size - 1}]; got {pre_frames}")
    if peak_window_frames < 1 or sustained_window_frames < 1:
        raise ValueError("peak_window_frames and sustained_window_frames must be >= 1")
    if peak_window_frames > trace.size - pre_frames:
        raise ValueError("peak_window_frames exceeds the post-event window length")
    if sustained_window_frames > trace.size - pre_frames:
        raise ValueError("sustained_window_frames exceeds the post-event window length")

    baseline = float(_nanmean(trace[:pre_frames])) if pre_frames > 0 else 0.0
    peak_window = trace[pre_frames : pre_frames + peak_window_frames]
    sustained_window = trace[trace.size - sustained_window_frames :]

    if not np.isfinite(baseline):
        return float("nan")
    if not np.any(np.isfinite(peak_window)) or not np.any(np.isfinite(sustained_window)):
        return float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peak = float(np.nanmax(peak_window))
    sustained = float(_nanmean(sustained_window))

    denom = sustained - baseline
    if abs(denom) <= eps:
        return float("nan")
    return (peak - baseline) / denom


def onset_latency_frames(
    aligned_mean: npt.NDArray[np.floating],
    pre_frames: int,
    threshold_frac: float = 0.5,
) -> float:
    """Frames from the event to the first crossing of a fraction of the peak.

    Parameters
    ----------
    aligned_mean : (pre+post,) float
        Event-aligned mean trace.
    pre_frames : int
        Number of pre-event samples; the baseline is their mean (0 if
        ``pre_frames`` is 0).
    threshold_frac : float
        Fraction of the baseline-to-peak amplitude defining the threshold,
        in (0, 1].

    Returns
    -------
    float
        Latency in frames from the event to the first post-event sample at or
        above the threshold. NaN if the baseline or the whole post-event window
        is NaN, or the post-event peak does not exceed the baseline.

    Raises
    ------
    ValueError
        If ``pre_frames`` is out of range or ``threshold_frac`` is not in (0, 1].
    """
    trace = _as_float_1d(aligned_mean, "aligned_mean")
    pre_frames = int(pre_frames)
    if pre_frames < 0 or pre_frames >= trace.size:
        raise ValueError(f"pre_frames must be in [0, {trace.size - 1}]; got {pre_frames}")
    if not 0.0 < float(threshold_frac) <= 1.0:
        raise ValueError(f"threshold_frac must be in (0, 1]; got {threshold_frac}")

    baseline = float(_nanmean(trace[:pre_frames])) if pre_frames > 0 else 0.0
    post = trace[pre_frames:]
    if not np.isfinite(baseline) or not np.any(np.isfinite(post)):
        return float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peak = float(np.nanmax(post))
    if not np.isfinite(peak) or peak <= baseline:
        return float("nan")

    # The peak sample itself always satisfies the threshold, so at least one
    # crossing exists once the peak exceeds the baseline.
    threshold = baseline + float(threshold_frac) * (peak - baseline)
    crossings = np.flatnonzero(np.isfinite(post) & (post >= threshold))
    return float(crossings[0])


# ---------------------------------------------------------------------------
# Immobility and bout-duration dependence
# ---------------------------------------------------------------------------


def immobility_decay(
    signal: npt.NDArray[np.floating],
    active: npt.NDArray[np.bool_],
    fps: float,
    min_bout_s: float = 3.0,
    max_bout_s: float = 20.0,
) -> dict:
    """Pooled activity time course during immobility and its decay time.

    Immobility bouts (``active`` False) of duration between ``min_bout_s`` and
    ``max_bout_s`` are aligned to their onset (the movement offset) and pooled.
    The decay time constant is estimated non-parametrically as the first time
    the pooled curve falls to 1/e of its initial value; no exponential model is
    fitted.

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal; NaN frames are ignored.
    active : (n_frames,) bool
        Movement state per frame.
    fps : float
        Sampling rate in frames per second.
    min_bout_s, max_bout_s : float
        Bout duration limits in seconds. Bouts longer than ``max_bout_s`` are
        still used but truncated to ``max_bout_s``.

    Returns
    -------
    dict
        ``"time_s"`` — (n_samples,) time since movement offset.
        ``"mean_timecourse"`` — (n_samples,) pooled mean signal.
        ``"sem_timecourse"`` — (n_samples,) standard error across bouts.
        ``"n_bouts"`` — number of bouts pooled.
        ``"initial_value"`` — pooled value at time 0.
        ``"decay_time_s"`` — time to fall to 1/e of the initial value, NaN if
        the curve never does or the initial value is not positive.

    Raises
    ------
    ValueError
        If the inputs differ in length, ``fps`` is not positive, or the bout
        limits are invalid.
    """
    sig = _as_float_1d(signal, "signal")
    act = _as_bool_1d(active, "active")
    if sig.shape != act.shape:
        raise ValueError(
            f"signal and active must have the same shape; got {sig.shape}, {act.shape}"
        )
    fps = _check_fps(fps)
    if not 0 < float(min_bout_s) <= float(max_bout_s):
        raise ValueError(f"require 0 < min_bout_s <= max_bout_s; got {min_bout_s}, {max_bout_s}")

    min_frames = max(1, int(round(float(min_bout_s) * fps)))
    max_frames = max(min_frames, int(round(float(max_bout_s) * fps)))

    onsets, offsets = segment_bouts(~act, min_frames=min_frames)
    time_s = np.arange(max_frames, dtype=np.float64) / fps

    if onsets.size == 0:
        return {
            "time_s": time_s,
            "mean_timecourse": np.full(max_frames, np.nan),
            "sem_timecourse": np.full(max_frames, np.nan),
            "n_bouts": 0,
            "initial_value": float("nan"),
            "decay_time_s": float("nan"),
        }

    pooled = np.full((onsets.size, max_frames), np.nan, dtype=np.float64)
    for i, (start, end) in enumerate(zip(onsets, offsets, strict=True)):
        stop = min(end, start + max_frames)
        pooled[i, : stop - start] = sig[start:stop]

    mean_tc = _nanmean(pooled, axis=0)
    sem_tc = _nansem(pooled, axis=0)

    initial = float(mean_tc[0])
    decay_time = float("nan")
    if np.isfinite(initial) and initial > 0:
        target = initial / np.e
        below = np.flatnonzero(np.isfinite(mean_tc) & (mean_tc <= target))
        if below.size > 0:
            decay_time = float(time_s[below[0]])

    return {
        "time_s": time_s,
        "mean_timecourse": mean_tc,
        "sem_timecourse": sem_tc,
        "n_bouts": int(onsets.size),
        "initial_value": initial,
        "decay_time_s": decay_time,
    }


def activity_vs_bout_duration(
    signal: npt.NDArray[np.floating],
    active: npt.NDArray[np.bool_],
    fps: float,
    n_duration_bins: int = 5,
) -> dict:
    """Mean within-bout activity as a function of movement bout duration.

    Movement bouts are binned by duration quantiles. A sustained population
    keeps a similar mean activity across duration bins; a transient population
    shows lower mean activity in long bouts, because the onset response is
    diluted by the rest of the bout.

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal; NaN frames are ignored.
    active : (n_frames,) bool
        Movement state per frame.
    fps : float
        Sampling rate in frames per second.
    n_duration_bins : int
        Number of duration quantile bins.

    Returns
    -------
    dict
        ``"bin_edges_s"`` — (n_bins+1,) duration quantile edges.
        ``"bin_centers_s"`` — (n_bins,) median bout duration per bin.
        ``"mean_signal"`` — (n_bins,) mean of the per-bout mean signal.
        ``"sem_signal"`` — (n_bins,) standard error across bouts within a bin.
        ``"n_bouts_per_bin"`` — (n_bins,) int counts.
        ``"bout_durations_s"`` — (n_bouts,) duration of every bout.
        ``"bout_means"`` — (n_bouts,) mean signal within every bout.

    Raises
    ------
    ValueError
        If the inputs differ in length, ``fps`` is not positive, or
        ``n_duration_bins`` is less than 1.
    """
    sig = _as_float_1d(signal, "signal")
    act = _as_bool_1d(active, "active")
    if sig.shape != act.shape:
        raise ValueError(
            f"signal and active must have the same shape; got {sig.shape}, {act.shape}"
        )
    fps = _check_fps(fps)
    n_bins = int(n_duration_bins)
    if n_bins < 1:
        raise ValueError(f"n_duration_bins must be >= 1; got {n_bins}")

    onsets, offsets = segment_bouts(act, min_frames=1)
    durations = bout_durations(onsets, offsets, fps)
    bout_means = np.array(
        [float(_nanmean(sig[s:e])) for s, e in zip(onsets, offsets, strict=True)],
        dtype=np.float64,
    )

    empty_bins = np.full(n_bins, np.nan, dtype=np.float64)
    if onsets.size == 0:
        return {
            "bin_edges_s": np.full(n_bins + 1, np.nan),
            "bin_centers_s": empty_bins,
            "mean_signal": empty_bins.copy(),
            "sem_signal": empty_bins.copy(),
            "n_bouts_per_bin": np.zeros(n_bins, dtype=int),
            "bout_durations_s": durations,
            "bout_means": bout_means,
        }

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(durations, quantiles)
    bin_idx = np.clip(np.digitize(durations, bin_edges[1:-1], right=False), 0, n_bins - 1)

    mean_signal = np.full(n_bins, np.nan, dtype=np.float64)
    sem_signal = np.full(n_bins, np.nan, dtype=np.float64)
    centers = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        sel = bin_idx == b
        counts[b] = int(sel.sum())
        if counts[b] == 0:
            continue
        centers[b] = float(np.median(durations[sel]))
        mean_signal[b] = float(_nanmean(bout_means[sel]))
        sem_signal[b] = float(_nansem(bout_means[sel][:, np.newaxis], axis=0)[0])

    return {
        "bin_edges_s": bin_edges,
        "bin_centers_s": centers,
        "mean_signal": mean_signal,
        "sem_signal": sem_signal,
        "n_bouts_per_bin": counts,
        "bout_durations_s": durations,
        "bout_means": bout_means,
    }


def movement_state_summary(
    signal: npt.NDArray[np.floating],
    active: npt.NDArray[np.bool_],
    bad_behav: npt.NDArray[np.bool_],
    fps: float,
    pre_s: float = 2.0,
    post_s: float = 5.0,
    min_bout_s: float = 1.0,
    long_bout_s: float = 4.0,
) -> dict:
    """Summarise movement-onset, movement-offset and immobility dynamics.

    Frames flagged in ``bad_behav`` (artefactual immobility, e.g. the animal
    caught on the imaging fibre) are set to NaN and excluded from every
    average.

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal.
    active : (n_frames,) bool
        Movement state per frame.
    bad_behav : (n_frames,) bool
        Frames to exclude.
    fps : float
        Sampling rate in frames per second.
    pre_s, post_s : float
        Window around movement onsets/offsets, in seconds.
    min_bout_s : float
        Minimum movement bout duration used to define onsets and offsets.
    long_bout_s : float
        Minimum duration of a movement bout used for ``sustained_ratio``.

    Returns
    -------
    dict
        ``"onset_transient_index"``, ``"offset_transient_index"`` — transient
        index at movement onset and offset (see :func:`transient_index`).
        ``"onset_latency_s"`` — latency to half the onset peak, in seconds.
        ``"n_onsets"``, ``"n_offsets"`` — number of events averaged.
        ``"immobility_decay_time_s"`` — see :func:`immobility_decay`.
        ``"sustained_ratio"`` — mean signal in the second half of long movement
        bouts divided by the mean in the first half; ~1 for a sustained
        response, below 1 for a transient one.
        ``"onset_mean"``, ``"offset_mean"`` — event-aligned mean traces.
        ``"time_s"`` — time axis of those traces, in seconds.

    Raises
    ------
    ValueError
        If the inputs differ in length or the window lengths are not positive.
    """
    sig = _as_float_1d(signal, "signal")
    act = _as_bool_1d(active, "active")
    bad = _as_bool_1d(bad_behav, "bad_behav")
    if not (sig.shape == act.shape == bad.shape):
        raise ValueError(
            "signal, active and bad_behav must have the same shape; "
            f"got {sig.shape}, {act.shape}, {bad.shape}"
        )
    fps = _check_fps(fps)

    pre_frames = max(1, int(round(float(pre_s) * fps)))
    post_frames = max(2, int(round(float(post_s) * fps)))

    masked = np.where(bad, np.nan, sig)
    min_frames = max(1, int(round(float(min_bout_s) * fps)))
    onsets, offsets = segment_bouts(act, min_frames=min_frames)

    onset_aligned = event_aligned_average(
        masked, onsets, pre_frames, post_frames, baseline_frames=pre_frames
    )
    offset_aligned = event_aligned_average(
        masked, offsets, pre_frames, post_frames, baseline_frames=pre_frames
    )

    peak_win = max(1, min(post_frames, int(round(1.0 * fps))))
    sustained_win = max(1, min(post_frames, int(round(2.0 * fps))))

    onset_ti = transient_index(onset_aligned["mean"], pre_frames, peak_win, sustained_win)
    offset_ti = transient_index(offset_aligned["mean"], pre_frames, peak_win, sustained_win)
    latency_frames = onset_latency_frames(onset_aligned["mean"], pre_frames)
    latency_s = latency_frames / fps if np.isfinite(latency_frames) else float("nan")

    decay = immobility_decay(masked, act, fps)

    sustained_ratio = _sustained_ratio(masked, onsets, offsets, fps, long_bout_s)

    return {
        "onset_transient_index": onset_ti,
        "offset_transient_index": offset_ti,
        "onset_latency_s": latency_s,
        "n_onsets": onset_aligned["n_events"],
        "n_offsets": offset_aligned["n_events"],
        "immobility_decay_time_s": decay["decay_time_s"],
        "sustained_ratio": sustained_ratio,
        "onset_mean": onset_aligned["mean"],
        "offset_mean": offset_aligned["mean"],
        "time_s": onset_aligned["time_axis"] / fps,
    }


def _sustained_ratio(
    signal: npt.NDArray[np.floating],
    onsets: npt.NDArray[np.integer],
    offsets: npt.NDArray[np.integer],
    fps: float,
    long_bout_s: float,
    eps: float = 1e-12,
) -> float:
    """Second-half over first-half mean signal across long movement bouts.

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal, already masked for excluded frames.
    onsets, offsets : (n_bouts,) int
        Movement bout boundaries.
    fps : float
        Sampling rate in frames per second.
    long_bout_s : float
        Minimum bout duration, in seconds, for a bout to contribute.
    eps : float
        Tolerance below which the first-half mean is treated as zero.

    Returns
    -------
    float
        Ratio of second-half to first-half mean; NaN if no bout qualifies or
        the first-half mean is within ``eps`` of zero.
    """
    min_len = max(2, int(round(float(long_bout_s) * fps)))
    first_halves: list[float] = []
    second_halves: list[float] = []
    for start, end in zip(onsets, offsets, strict=True):
        if end - start < min_len:
            continue
        mid = start + (end - start) // 2
        first_halves.append(float(_nanmean(signal[start:mid])))
        second_halves.append(float(_nanmean(signal[mid:end])))

    if not first_halves:
        return float("nan")

    first = float(_nanmean(np.asarray(first_halves)))
    second = float(_nanmean(np.asarray(second_halves)))
    if not np.isfinite(first) or not np.isfinite(second) or abs(first) <= eps:
        return float("nan")
    return second / first


# ---------------------------------------------------------------------------
# Syllable-conditioned activity
# ---------------------------------------------------------------------------


def syllable_conditioned_activity(
    signal: npt.NDArray[np.floating],
    syllable_id: npt.NDArray[np.integer],
    mask: npt.NDArray[np.bool_],
    min_frames: int = 20,
) -> dict[int, float]:
    """Mean signal per behavioural syllable.

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal; NaN frames are ignored.
    syllable_id : (n_frames,) int
        Syllable label per frame. Negative labels are treated as unassigned and
        skipped.
    mask : (n_frames,) bool
        Valid frames (e.g. not ``bad_behav``).
    min_frames : int
        Minimum number of valid frames for a syllable to be reported.

    Returns
    -------
    dict[int, float]
        Mean signal keyed by syllable id, for syllables with enough frames.

    Raises
    ------
    ValueError
        If the inputs differ in length or ``min_frames`` is less than 1.
    """
    sig = _as_float_1d(signal, "signal")
    syl = np.asarray(syllable_id).ravel().astype(np.int64)
    msk = _as_bool_1d(mask, "mask")
    if not (sig.shape == syl.shape == msk.shape):
        raise ValueError(
            "signal, syllable_id and mask must have the same shape; "
            f"got {sig.shape}, {syl.shape}, {msk.shape}"
        )
    if int(min_frames) < 1:
        raise ValueError(f"min_frames must be >= 1; got {min_frames}")

    valid = msk & (syl >= 0) & np.isfinite(sig)
    out: dict[int, float] = {}
    for label in np.unique(syl[valid]):
        sel = valid & (syl == label)
        if int(sel.sum()) < int(min_frames):
            continue
        out[int(label)] = float(np.mean(sig[sel]))
    return out


def syllable_information(
    signal: npt.NDArray[np.floating],
    syllable_id: npt.NDArray[np.integer],
    mask: npt.NDArray[np.bool_],
    n_signal_bins: int = 10,
) -> float:
    """Mutual information in bits between a binned signal and syllable identity.

    The signal is discretised into equal-occupancy (quantile) bins and the
    mutual information of the joint distribution with syllable identity is
    computed as ``sum p(x, y) log2[p(x, y) / (p(x) p(y))]``. This follows the
    information-theoretic framework of Skaggs et al. (1993) applied to
    behavioural syllables rather than spatial bins.

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal; NaN frames are ignored.
    syllable_id : (n_frames,) int
        Syllable label per frame; negative labels are skipped.
    mask : (n_frames,) bool
        Valid frames.
    n_signal_bins : int
        Number of quantile bins for the signal.

    Returns
    -------
    float
        Mutual information in bits, 0.0 when fewer than two signal bins or
        fewer than two syllables are present.

    Raises
    ------
    ValueError
        If the inputs differ in length or ``n_signal_bins`` is less than 2.

    References
    ----------
    Skaggs WE, McNaughton BL, Gothard KM, Markus EJ. 1993. "An
    information-theoretic approach to deciphering the hippocampal code."
    NeurIPS 5:1030-1037.
    """
    sig = _as_float_1d(signal, "signal")
    syl = np.asarray(syllable_id).ravel().astype(np.int64)
    msk = _as_bool_1d(mask, "mask")
    if not (sig.shape == syl.shape == msk.shape):
        raise ValueError(
            "signal, syllable_id and mask must have the same shape; "
            f"got {sig.shape}, {syl.shape}, {msk.shape}"
        )
    if int(n_signal_bins) < 2:
        raise ValueError(f"n_signal_bins must be >= 2; got {n_signal_bins}")

    valid = msk & (syl >= 0) & np.isfinite(sig)
    if int(valid.sum()) < 2:
        return 0.0

    x = sig[valid]
    y = syl[valid]
    labels = np.unique(y)
    if labels.size < 2:
        return 0.0

    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, int(n_signal_bins) + 1)))
    if edges.size < 3:
        return 0.0
    x_idx = np.clip(np.digitize(x, edges[1:-1], right=False), 0, edges.size - 2)
    y_idx = np.searchsorted(labels, y)

    joint = np.zeros((edges.size - 1, labels.size), dtype=np.float64)
    np.add.at(joint, (x_idx, y_idx), 1.0)
    joint /= joint.sum()

    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denom = px * py
    nz = (joint > 0) & (denom > 0)
    mi = float(np.sum(joint[nz] * np.log2(joint[nz] / denom[nz])))
    return max(mi, 0.0)
