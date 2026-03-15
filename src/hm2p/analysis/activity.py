"""Condition-split activity analysis.

Computes per-cell activity metrics across four conditions defined by
movement state (moving vs stationary) and illumination (light vs dark).

All functions are pure numpy — no I/O, no classes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hm2p.constants import SPEED_ACTIVE_THRESHOLD


def split_conditions(
    speed: npt.NDArray[np.floating],
    light_on: npt.NDArray[np.bool_],
    active_mask: npt.NDArray[np.bool_],
    speed_threshold: float = SPEED_ACTIVE_THRESHOLD,
) -> dict[str, npt.NDArray[np.bool_]]:
    """Split frames into four conditions based on movement and illumination.

    Parameters
    ----------
    speed : (n_frames,) float array
        Instantaneous speed (e.g. cm/s).
    light_on : (n_frames,) bool array
        True when room lights are on.
    active_mask : (n_frames,) bool array
        True for frames to include (e.g. not bad_behav).
    speed_threshold : float
        Speed >= this value counts as moving. Default from
        ``hm2p.constants.SPEED_ACTIVE_THRESHOLD`` (0.5 cm/s).

    Returns
    -------
    dict
        Keys ``"moving_light"``, ``"moving_dark"``, ``"stationary_light"``,
        ``"stationary_dark"`` each mapping to a boolean mask of shape
        ``(n_frames,)``.  All masks are AND-ed with *active_mask*.
    """
    moving = speed >= speed_threshold
    stationary = ~moving
    light = light_on.astype(bool)
    dark = ~light

    return {
        "moving_light": active_mask & moving & light,
        "moving_dark": active_mask & moving & dark,
        "stationary_light": active_mask & stationary & light,
        "stationary_dark": active_mask & stationary & dark,
    }


def condition_event_rate(
    event_mask: npt.NDArray[np.bool_],
    condition_mask: npt.NDArray[np.bool_],
    fps: float,
) -> float:
    """Count event onsets within a condition and return rate (events/s).

    An onset is a transition from ``False`` to ``True`` in *event_mask*
    at frames where *condition_mask* is ``True``.

    Parameters
    ----------
    event_mask : (n_frames,) bool array
        True during detected calcium events.
    condition_mask : (n_frames,) bool array
        True for frames belonging to the condition of interest.
    fps : float
        Sampling rate in Hz.

    Returns
    -------
    float
        Event onset rate in events per second.  Returns 0.0 if no
        condition frames exist.
    """
    event_mask = np.asarray(event_mask, dtype=bool)
    condition_mask = np.asarray(condition_mask, dtype=bool)

    n_condition = condition_mask.sum()
    if n_condition == 0:
        return 0.0

    # Onset = frame where event_mask goes from False to True.
    onset = np.zeros_like(event_mask)
    onset[0] = event_mask[0]
    onset[1:] = event_mask[1:] & ~event_mask[:-1]

    n_onsets = (onset & condition_mask).sum()
    duration_s = n_condition / fps
    return float(n_onsets / duration_s)


def condition_mean_signal(
    signal: npt.NDArray[np.floating],
    condition_mask: npt.NDArray[np.bool_],
) -> float:
    """Mean of *signal* during condition frames.

    Parameters
    ----------
    signal : (n_frames,) float array
        E.g. dF/F or deconvolved spike rate.
    condition_mask : (n_frames,) bool array
        True for frames belonging to the condition.

    Returns
    -------
    float
        Mean value, or ``nan`` if no valid frames.
    """
    if condition_mask.sum() == 0:
        return float("nan")
    return float(np.mean(signal[condition_mask]))


def condition_mean_amplitude(
    signal: npt.NDArray[np.floating],
    event_mask: npt.NDArray[np.bool_],
    condition_mask: npt.NDArray[np.bool_],
) -> float:
    """Mean of *signal* during frames that are both event and condition.

    Parameters
    ----------
    signal : (n_frames,) float array
        E.g. dF/F or deconvolved spike rate.
    event_mask : (n_frames,) bool array
        True during detected events.
    condition_mask : (n_frames,) bool array
        True for frames belonging to the condition.

    Returns
    -------
    float
        Mean amplitude, or ``nan`` if no qualifying frames.
    """
    mask = event_mask & condition_mask
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(signal[mask]))


def modulation_index(rate_a: float, rate_b: float) -> float:
    """Compute modulation index ``(a - b) / (a + b)``.

    Parameters
    ----------
    rate_a, rate_b : float
        Two rates to compare (e.g. moving vs stationary event rates).

    Returns
    -------
    float
        Modulation index in ``[-1, 1]``.  Returns ``0.0`` if both rates
        are zero.
    """
    total = rate_a + rate_b
    if total == 0.0:
        return 0.0
    return float((rate_a - rate_b) / total)


def compute_cell_activity(
    signal: npt.NDArray[np.floating],
    event_mask: npt.NDArray[np.bool_],
    speed: npt.NDArray[np.floating],
    light_on: npt.NDArray[np.bool_],
    active_mask: npt.NDArray[np.bool_],
    fps: float,
    speed_threshold: float = SPEED_ACTIVE_THRESHOLD,
) -> dict[str, float]:
    """Compute all condition-split activity metrics for one cell.

    Parameters
    ----------
    signal : (n_frames,) float array
        dF/F or deconvolved trace for this cell.
    event_mask : (n_frames,) bool array
        True during detected events for this cell.
    speed : (n_frames,) float array
        Animal speed (cm/s).
    light_on : (n_frames,) bool array
        True when lights are on.
    active_mask : (n_frames,) bool array
        True for valid (non-artefact) frames.
    fps : float
        Sampling rate in Hz.
    speed_threshold : float
        Threshold for moving vs stationary.

    Returns
    -------
    dict
        Flat dictionary with keys:

        - ``{cond}_event_rate`` — event onset rate (events/s)
        - ``{cond}_mean_signal`` — mean signal value
        - ``{cond}_mean_amplitude`` — mean signal during events
        - ``movement_modulation`` — modulation index (moving vs stationary)
        - ``light_modulation`` — modulation index (light vs dark)

        where *cond* is one of ``moving_light``, ``moving_dark``,
        ``stationary_light``, ``stationary_dark``.
    """
    conditions = split_conditions(speed, light_on, active_mask, speed_threshold)

    result: dict[str, float] = {}
    for cond_name, cond_mask in conditions.items():
        result[f"{cond_name}_event_rate"] = condition_event_rate(
            event_mask, cond_mask, fps
        )
        result[f"{cond_name}_mean_signal"] = condition_mean_signal(signal, cond_mask)
        result[f"{cond_name}_mean_amplitude"] = condition_mean_amplitude(
            signal, event_mask, cond_mask
        )

    # Aggregate rates across light conditions for modulation indices.
    moving_rate = (
        result["moving_light_event_rate"] + result["moving_dark_event_rate"]
    ) / 2.0
    stationary_rate = (
        result["stationary_light_event_rate"] + result["stationary_dark_event_rate"]
    ) / 2.0
    light_rate = (
        result["moving_light_event_rate"] + result["stationary_light_event_rate"]
    ) / 2.0
    dark_rate = (
        result["moving_dark_event_rate"] + result["stationary_dark_event_rate"]
    ) / 2.0

    result["movement_modulation"] = modulation_index(moving_rate, stationary_rate)
    result["light_modulation"] = modulation_index(light_rate, dark_rate)

    return result


def compute_batch_activity(
    signals: npt.NDArray[np.floating],
    event_masks: npt.NDArray[np.bool_],
    speed: npt.NDArray[np.floating],
    light_on: npt.NDArray[np.bool_],
    active_mask: npt.NDArray[np.bool_],
    fps: float,
    speed_threshold: float = SPEED_ACTIVE_THRESHOLD,
) -> list[dict[str, float]]:
    """Compute condition-split activity for every ROI.

    Parameters
    ----------
    signals : (n_rois, n_frames) float array
        dF/F or deconvolved traces, one row per cell.
    event_masks : (n_rois, n_frames) bool array
        Event masks, one row per cell.
    speed : (n_frames,) float array
        Animal speed (cm/s).
    light_on : (n_frames,) bool array
        True when lights are on.
    active_mask : (n_frames,) bool array
        True for valid frames.
    fps : float
        Sampling rate in Hz.
    speed_threshold : float
        Threshold for moving vs stationary.

    Returns
    -------
    list[dict]
        One dict per ROI, as returned by :func:`compute_cell_activity`.
    """
    n_rois = signals.shape[0]
    return [
        compute_cell_activity(
            signals[i],
            event_masks[i],
            speed,
            light_on,
            active_mask,
            fps,
            speed_threshold,
        )
        for i in range(n_rois)
    ]
