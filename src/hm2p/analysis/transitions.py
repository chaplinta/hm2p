"""Responses to light/dark transitions and recovery of HD tuning.

Room lights alternate 1 min on / 1 min off, giving 10-15 transitions per
session. Darkness is total, so a light-to-dark transition removes all visual
cues at a known instant. This module aligns neural signals to those
transitions, quantifies the early versus late response, and tracks how head
direction tuning recovers after the lights return.

Transition indices come from :func:`hm2p.analysis.anchoring.find_transitions`
and event-aligned averaging from
:func:`hm2p.analysis.state_dynamics.event_aligned_average`; neither is
reimplemented here.

All functions are pure numpy/scipy — no I/O. Statistics are non-parametric
(Wilcoxon signed-rank across transitions).

References
----------
Brennan EKW, Sudhakar SK, Jedrasiak-Cape I, John TT, Ahmed OJ. 2020.
"Hyperexcitable Neurons Enable Precise and Persistent Information Encoding in
the Superficial Retrosplenial Cortex." Cell Reports 30(5):1598-1612.
doi:10.1016/j.celrep.2019.12.093

Taube JS, Muller RU, Ranck JB. 1990. "Head-direction cells recorded from the
postsubiculum in freely moving rats. II. Effects of environmental
manipulations." J Neurosci 10(2):436-447.
doi:10.1523/JNEUROSCI.10-02-00436.1990
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy.stats import wilcoxon

from hm2p.analysis.anchoring import find_transitions

# Package-internal validation helpers are shared with state_dynamics rather than
# duplicated, alongside the public event-aligned averaging used here.
from hm2p.analysis.state_dynamics import (
    _as_bool_1d,
    _as_float_1d,
    _check_fps,
    _nanmean,
    _nansem,
    event_aligned_average,
    segment_bouts,
)
from hm2p.analysis.tuning import (
    compute_hd_tuning_curve,
    mean_vector_length,
    preferred_direction,
)

__all__ = [
    "first_vs_later_epochs",
    "population_transition_response",
    "recovery_time_s",
    "transition_aligned_responses",
    "transition_transient",
    "tuning_recovery_timecourse",
]

_DIRECTIONS = ("light_to_dark", "dark_to_light")


def _window_mask(
    time_s: npt.NDArray[np.floating],
    bounds: tuple[float, float],
) -> np.ndarray:
    """Boolean mask selecting ``bounds[0] <= time_s < bounds[1]``.

    Parameters
    ----------
    time_s : (n_samples,) float
        Time axis in seconds.
    bounds : (float, float)
        Half-open interval in seconds.

    Returns
    -------
    (n_samples,) bool

    Raises
    ------
    ValueError
        If the interval is empty or inverted.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    if not hi > lo:
        raise ValueError(f"window bounds must satisfy lo < hi; got {bounds}")
    return (time_s >= lo) & (time_s < hi)


def transition_aligned_responses(
    signal: npt.NDArray[np.floating],
    light_on: npt.NDArray[np.bool_],
    bad_behav: npt.NDArray[np.bool_],
    fps: float,
    pre_s: float = 10.0,
    post_s: float = 30.0,
    direction: str = "light_to_dark",
) -> dict:
    """Average a signal around light/dark transitions.

    The pre-transition window is used as the baseline and subtracted from each
    transition window. Frames flagged in ``bad_behav`` are set to NaN and
    ignored by the averages.

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal for one ROI.
    light_on : (n_frames,) bool
        Room-light state per frame.
    bad_behav : (n_frames,) bool
        Frames to exclude.
    fps : float
        Sampling rate in frames per second.
    pre_s, post_s : float
        Window before and after the transition, in seconds.
    direction : {"light_to_dark", "dark_to_light"}
        Which transitions to align to.

    Returns
    -------
    dict
        ``"mean"``, ``"sem"`` — (n_samples,) baseline-subtracted average.
        ``"windows"`` — (n_transitions, n_samples) per-transition windows.
        ``"time_s"`` — (n_samples,) time relative to the transition, seconds.
        ``"n_transitions"`` — number of transitions averaged.
        ``"pre_frames"`` — number of pre-transition samples.
        ``"direction"`` — the direction used.

    Raises
    ------
    ValueError
        If the inputs differ in length, ``fps`` is not positive, the window
        lengths are not positive, or ``direction`` is unknown.
    """
    sig = _as_float_1d(signal, "signal")
    light = _as_bool_1d(light_on, "light_on")
    bad = _as_bool_1d(bad_behav, "bad_behav")
    if not (sig.shape == light.shape == bad.shape):
        raise ValueError(
            "signal, light_on and bad_behav must have the same shape; "
            f"got {sig.shape}, {light.shape}, {bad.shape}"
        )
    fps = _check_fps(fps)
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}; got {direction!r}")

    pre_frames = max(1, int(round(float(pre_s) * fps)))
    post_frames = max(1, int(round(float(post_s) * fps)))

    events = find_transitions(light)[direction]
    masked = np.where(bad, np.nan, sig)

    aligned = event_aligned_average(
        masked, events, pre_frames, post_frames, baseline_frames=pre_frames
    )

    return {
        "mean": aligned["mean"],
        "sem": aligned["sem"],
        "windows": aligned["windows"],
        "time_s": aligned["time_axis"] / fps,
        "n_transitions": aligned["n_events"],
        "pre_frames": pre_frames,
        "direction": direction,
    }


def transition_transient(
    aligned: dict,
    fps: float,
    early_s: tuple[float, float] = (0.0, 5.0),
    late_s: tuple[float, float] = (15.0, 30.0),
) -> dict:
    """Early versus late amplitude of a transition-aligned response.

    Amplitudes are relative to the pre-transition baseline, which
    :func:`transition_aligned_responses` has already subtracted. The sign of
    the early response is tested across transitions with a Wilcoxon
    signed-rank test (non-parametric, paired within ROI).

    Parameters
    ----------
    aligned : dict
        Output of :func:`transition_aligned_responses`.
    fps : float
        Sampling rate in frames per second; retained for interface symmetry,
        the time axis in ``aligned`` is used directly.
    early_s, late_s : (float, float)
        Half-open post-transition windows, in seconds.

    Returns
    -------
    dict
        ``"early_amplitude"``, ``"late_amplitude"`` — mean across transitions.
        ``"early_per_transition"``, ``"late_per_transition"`` — per-transition
        window means.
        ``"sign"`` — +1, -1 or 0: the sign of the median early amplitude when
        the Wilcoxon test is significant at p < 0.05, else 0.
        ``"p_value"`` — Wilcoxon p-value; NaN when fewer than 5 transitions.
        ``"n_transitions"`` — number of transitions used.

    Raises
    ------
    ValueError
        If ``aligned`` lacks the required keys, ``fps`` is not positive, or a
        window is empty.
    """
    for key in ("windows", "time_s"):
        if key not in aligned:
            raise ValueError(f"aligned must contain {key!r}")
    _check_fps(fps)

    time_s = np.asarray(aligned["time_s"], dtype=np.float64)
    windows = np.atleast_2d(np.asarray(aligned["windows"], dtype=np.float64))

    early_mask = _window_mask(time_s, early_s)
    late_mask = _window_mask(time_s, late_s)

    n_trans = int(windows.shape[0]) if windows.size else 0
    if n_trans == 0 or not early_mask.any() or not late_mask.any():
        return {
            "early_amplitude": float("nan"),
            "late_amplitude": float("nan"),
            "early_per_transition": np.zeros(0, dtype=np.float64),
            "late_per_transition": np.zeros(0, dtype=np.float64),
            "sign": 0,
            "p_value": float("nan"),
            "n_transitions": n_trans,
        }

    early_per = np.asarray(_nanmean(windows[:, early_mask], axis=1), dtype=np.float64)
    late_per = np.asarray(_nanmean(windows[:, late_mask], axis=1), dtype=np.float64)

    early_amp = float(_nanmean(early_per))
    late_amp = float(_nanmean(late_per))

    p_value = float("nan")
    finite = early_per[np.isfinite(early_per)]
    if finite.size >= 5 and np.any(finite != 0.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_value = float(wilcoxon(finite).pvalue)

    sign = 0
    if np.isfinite(p_value) and p_value < 0.05 and finite.size > 0:
        median = float(np.median(finite))
        sign = int(np.sign(median))

    return {
        "early_amplitude": early_amp,
        "late_amplitude": late_amp,
        "early_per_transition": early_per,
        "late_per_transition": late_per,
        "sign": sign,
        "p_value": p_value,
        "n_transitions": n_trans,
    }


def population_transition_response(
    signals: npt.NDArray[np.floating],
    light_on: npt.NDArray[np.bool_],
    bad_behav: npt.NDArray[np.bool_],
    fps: float,
    pre_s: float = 10.0,
    post_s: float = 30.0,
    direction: str = "light_to_dark",
    early_s: tuple[float, float] = (0.0, 5.0),
    late_s: tuple[float, float] = (15.0, 30.0),
) -> dict:
    """Transition responses for every ROI in a population.

    Parameters
    ----------
    signals : (n_rois, n_frames) float
        Neural signal per ROI.
    light_on : (n_frames,) bool
        Room-light state per frame.
    bad_behav : (n_frames,) bool
        Frames to exclude.
    fps : float
        Sampling rate in frames per second.
    pre_s, post_s : float
        Window before and after the transition, in seconds.
    direction : {"light_to_dark", "dark_to_light"}
        Which transitions to align to.
    early_s, late_s : (float, float)
        Post-transition windows passed to :func:`transition_transient`.

    Returns
    -------
    dict
        ``"early_amplitude"``, ``"late_amplitude"``, ``"p_value"`` — (n_rois,).
        ``"sign"`` — (n_rois,) int.
        ``"mean_timecourse"`` — (n_samples,) mean across ROIs of the per-ROI
        mean response.
        ``"sem_timecourse"`` — (n_samples,) standard error across ROIs.
        ``"time_s"`` — (n_samples,) time relative to the transition.
        ``"n_transitions"`` — transitions available.
        ``"n_rois"`` — number of ROIs.

    Raises
    ------
    ValueError
        If ``signals`` is not 2-D or its frame axis does not match the masks.
    """
    sig = np.asarray(signals, dtype=np.float64)
    if sig.ndim != 2:
        raise ValueError(f"signals must be 2-D (n_rois, n_frames); got shape {sig.shape}")
    n_rois, n_frames = sig.shape
    light = _as_bool_1d(light_on, "light_on")
    bad = _as_bool_1d(bad_behav, "bad_behav")
    if light.size != n_frames or bad.size != n_frames:
        raise ValueError(
            "light_on and bad_behav must match the frame axis of signals; "
            f"got {light.size}, {bad.size} vs {n_frames}"
        )

    early = np.full(n_rois, np.nan, dtype=np.float64)
    late = np.full(n_rois, np.nan, dtype=np.float64)
    pvals = np.full(n_rois, np.nan, dtype=np.float64)
    signs = np.zeros(n_rois, dtype=int)

    per_roi_mean: list[np.ndarray] = []
    time_s = np.zeros(0, dtype=np.float64)
    n_transitions = 0

    for r in range(n_rois):
        aligned = transition_aligned_responses(
            sig[r], light, bad, fps, pre_s=pre_s, post_s=post_s, direction=direction
        )
        stats = transition_transient(aligned, fps, early_s=early_s, late_s=late_s)
        early[r] = stats["early_amplitude"]
        late[r] = stats["late_amplitude"]
        pvals[r] = stats["p_value"]
        signs[r] = stats["sign"]
        per_roi_mean.append(np.asarray(aligned["mean"], dtype=np.float64))
        time_s = np.asarray(aligned["time_s"], dtype=np.float64)
        n_transitions = int(aligned["n_transitions"])

    if per_roi_mean:
        stack = np.vstack(per_roi_mean)
        mean_tc = np.asarray(_nanmean(stack, axis=0), dtype=np.float64)
        sem_tc = np.asarray(_nansem(stack, axis=0), dtype=np.float64)
    else:
        mean_tc = np.zeros(0, dtype=np.float64)
        sem_tc = np.zeros(0, dtype=np.float64)

    return {
        "early_amplitude": early,
        "late_amplitude": late,
        "p_value": pvals,
        "sign": signs,
        "mean_timecourse": mean_tc,
        "sem_timecourse": sem_tc,
        "time_s": time_s,
        "n_transitions": n_transitions,
        "n_rois": int(n_rois),
    }


def tuning_recovery_timecourse(
    signal: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    light_on: npt.NDArray[np.bool_],
    fps: float,
    reference_pd: float | None = None,
    window_s: float = 30.0,
    step_s: float = 10.0,
    post_s: float = 60.0,
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
) -> dict:
    """HD tuning strength and drift after each dark-to-light transition.

    Sliding windows starting at fixed offsets after each dark-to-light
    transition give a mean vector length and a preferred direction. The
    absolute deviation of that preferred direction from a reference (by default
    the preferred direction over all valid light frames) measures how far the
    tuning has drifted; its decline over time measures re-anchoring to the
    restored visual cues. Windows are on a common time axis so they can be
    averaged across transitions.

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal for one ROI.
    hd_deg : (n_frames,) float
        Head direction in degrees.
    mask : (n_frames,) bool
        Valid frames (e.g. moving and not ``bad_behav``).
    light_on : (n_frames,) bool
        Room-light state per frame.
    fps : float
        Sampling rate in frames per second.
    reference_pd : float or None
        Reference preferred direction in degrees. If None it is computed from
        all valid light frames.
    window_s : float
        Length of each sliding window, in seconds.
    step_s : float
        Spacing between window start times, in seconds.
    post_s : float
        Last window start time after the transition, in seconds.
    n_bins : int
        Number of HD bins.
    smoothing_sigma_deg : float
        Gaussian smoothing of the tuning curve, in degrees.

    Returns
    -------
    dict
        ``"time_s"`` — (n_points,) window start times after the transition.
        ``"mvl"`` — (n_points,) mean vector length averaged over transitions.
        ``"pd_deviation_deg"`` — (n_points,) mean absolute deviation of the
        preferred direction from the reference, in degrees.
        ``"n_transitions"`` — number of transitions contributing at least one
        window.
        ``"reference_pd"`` — reference preferred direction used.

    Raises
    ------
    ValueError
        If the inputs differ in length or the window parameters are not
        positive.
    """
    sig = _as_float_1d(signal, "signal")
    hd = _as_float_1d(hd_deg, "hd_deg")
    msk = _as_bool_1d(mask, "mask")
    light = _as_bool_1d(light_on, "light_on")
    if not (sig.shape == hd.shape == msk.shape == light.shape):
        raise ValueError(
            "signal, hd_deg, mask and light_on must have the same shape; "
            f"got {sig.shape}, {hd.shape}, {msk.shape}, {light.shape}"
        )
    fps = _check_fps(fps)
    if float(window_s) <= 0 or float(step_s) <= 0 or float(post_s) < 0:
        raise ValueError(
            f"require window_s > 0, step_s > 0, post_s >= 0; got {window_s}, {step_s}, {post_s}"
        )

    window_frames = max(1, int(round(float(window_s) * fps)))
    step_frames = max(1, int(round(float(step_s) * fps)))
    post_frames = int(round(float(post_s) * fps))

    start_offsets = np.arange(0, post_frames + 1, step_frames, dtype=np.int64)
    time_s = start_offsets.astype(np.float64) / fps

    if reference_pd is None:
        light_mask = msk & light
        if int(light_mask.sum()) >= n_bins:
            tc, bc = compute_hd_tuning_curve(
                sig, hd, light_mask, n_bins=n_bins, smoothing_sigma_deg=smoothing_sigma_deg
            )
            reference_pd = preferred_direction(tc, bc)
        else:
            reference_pd = 0.0
    reference_pd = float(reference_pd)

    events = find_transitions(light)["dark_to_light"]
    n = sig.size

    if events.size == 0:
        return {
            "time_s": time_s,
            "mvl": np.full(time_s.size, np.nan),
            "pd_deviation_deg": np.full(time_s.size, np.nan),
            "n_transitions": 0,
            "reference_pd": reference_pd,
        }

    mvls = np.full((events.size, time_s.size), np.nan, dtype=np.float64)
    devs = np.full((events.size, time_s.size), np.nan, dtype=np.float64)

    for i, event in enumerate(events):
        for j, offset in enumerate(start_offsets):
            start = int(event) + int(offset)
            end = start + window_frames
            if start < 0 or end > n:
                continue
            win_mask = np.zeros_like(msk)
            win_mask[start:end] = msk[start:end]
            if int(win_mask.sum()) < n_bins:
                continue
            tc, bc = compute_hd_tuning_curve(
                sig, hd, win_mask, n_bins=n_bins, smoothing_sigma_deg=smoothing_sigma_deg
            )
            mvls[i, j] = mean_vector_length(tc, bc)
            pd = preferred_direction(tc, bc)
            devs[i, j] = abs(((pd - reference_pd + 180.0) % 360.0) - 180.0)

    contributing = int(np.sum(np.any(np.isfinite(mvls), axis=1)))

    return {
        "time_s": time_s,
        "mvl": np.asarray(_nanmean(mvls, axis=0), dtype=np.float64),
        "pd_deviation_deg": np.asarray(_nanmean(devs, axis=0), dtype=np.float64),
        "n_transitions": contributing,
        "reference_pd": reference_pd,
    }


def recovery_time_s(
    time_s: npt.NDArray[np.floating],
    curve: npt.NDArray[np.floating],
    target_frac: float = 0.5,
    direction: str = "down",
) -> float:
    """Time at which a curve first crosses a fraction of the way to its asymptote.

    The target is ``initial + target_frac * (final - initial)``, where the
    initial and final values are the first and last finite samples of the
    curve. This makes the measure usable both for a quantity that falls towards
    its asymptote (e.g. preferred-direction deviation, ``direction="down"``)
    and one that rises towards it (e.g. mean vector length,
    ``direction="up"``).

    Parameters
    ----------
    time_s : (n_points,) float
        Time axis in seconds.
    curve : (n_points,) float
        Curve values; NaN samples are skipped.
    target_frac : float
        Fraction of the initial-to-final change, in (0, 1].
    direction : {"down", "up"}
        Whether the crossing is downward or upward.

    Returns
    -------
    float
        First time at or beyond the target, NaN if the curve has fewer than
        two finite samples.

    Raises
    ------
    ValueError
        If the arrays differ in length, ``target_frac`` is outside (0, 1], or
        ``direction`` is unknown.
    """
    t = _as_float_1d(time_s, "time_s")
    y = _as_float_1d(curve, "curve")
    if t.shape != y.shape:
        raise ValueError(f"time_s and curve must have the same shape; got {t.shape}, {y.shape}")
    if not 0.0 < float(target_frac) <= 1.0:
        raise ValueError(f"target_frac must be in (0, 1]; got {target_frac}")
    if direction not in ("down", "up"):
        raise ValueError(f"direction must be 'down' or 'up'; got {direction!r}")

    finite = np.flatnonzero(np.isfinite(y) & np.isfinite(t))
    if finite.size < 2:
        return float("nan")

    initial = float(y[finite[0]])
    final = float(y[finite[-1]])
    target = initial + float(target_frac) * (final - initial)

    values = y[finite]
    times = t[finite]
    # The target always lies between the initial and final values, so at least
    # one sample satisfies the crossing once two finite samples exist.
    crossed = values <= target if direction == "down" else values >= target
    idx = np.flatnonzero(crossed)
    return float(times[idx[0]])


def first_vs_later_epochs(
    signal: npt.NDArray[np.floating],
    light_on: npt.NDArray[np.bool_],
    bad_behav: npt.NDArray[np.bool_],
    fps: float,
    metric_fn: Callable[[np.ndarray], float],
    min_epoch_s: float = 5.0,
) -> dict:
    """Compare a metric in the first dark epoch with later dark epochs.

    Tests for adaptation across repeated light removals: a response present
    only in the first dark epoch indicates novelty or startle rather than a
    stable effect of darkness.

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal.
    light_on : (n_frames,) bool
        Room-light state per frame.
    bad_behav : (n_frames,) bool
        Frames to exclude; set to NaN before ``metric_fn`` is applied.
    fps : float
        Sampling rate in frames per second.
    metric_fn : callable
        Function mapping a 1-D signal segment to a float.
    min_epoch_s : float
        Minimum dark-epoch duration, in seconds, for an epoch to be used.

    Returns
    -------
    dict
        ``"first_value"`` — metric for the first dark epoch, NaN if none.
        ``"later_values"`` — (n_epochs-1,) metric for each later dark epoch.
        ``"later_mean"`` — mean of ``later_values``, NaN if there are none.
        ``"difference"`` — ``first_value - later_mean``.
        ``"n_epochs"`` — number of dark epochs used.

    Raises
    ------
    ValueError
        If the inputs differ in length, ``fps`` is not positive, or
        ``metric_fn`` is not callable.
    """
    sig = _as_float_1d(signal, "signal")
    light = _as_bool_1d(light_on, "light_on")
    bad = _as_bool_1d(bad_behav, "bad_behav")
    if not (sig.shape == light.shape == bad.shape):
        raise ValueError(
            "signal, light_on and bad_behav must have the same shape; "
            f"got {sig.shape}, {light.shape}, {bad.shape}"
        )
    fps = _check_fps(fps)
    if not callable(metric_fn):
        raise ValueError("metric_fn must be callable")

    min_frames = max(1, int(round(float(min_epoch_s) * fps)))
    onsets, offsets = segment_bouts(~light, min_frames=min_frames)

    masked = np.where(bad, np.nan, sig)
    values = np.array(
        [float(metric_fn(masked[s:e])) for s, e in zip(onsets, offsets, strict=True)],
        dtype=np.float64,
    )

    first = float(values[0]) if values.size > 0 else float("nan")
    later = values[1:] if values.size > 1 else np.zeros(0, dtype=np.float64)
    later_mean = float(_nanmean(later)) if later.size > 0 else float("nan")
    both_finite = np.isfinite(first) and np.isfinite(later_mean)
    difference = first - later_mean if both_finite else float("nan")

    return {
        "first_value": first,
        "later_values": later,
        "later_mean": later_mean,
        "difference": difference,
        "n_epochs": int(values.size),
    }
