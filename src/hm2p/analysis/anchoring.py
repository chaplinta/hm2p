"""Cue anchoring analysis — HD re-anchoring after visual cue restoration.

Key science question: when lights turn back on after a dark period,
how quickly does the HD network snap back to the visually anchored PD?
Fast re-anchoring suggests visual dominance; slow/incomplete suggests
path integration independence.

All functions pure numpy — no I/O.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hm2p.analysis.tuning import (
    compute_hd_tuning_curve,
    mean_vector_length,
    preferred_direction,
)


def find_transitions(
    light_on: npt.NDArray[np.bool_],
) -> dict:
    """Find light→dark and dark→light transitions.

    Parameters
    ----------
    light_on : (n_frames,) bool

    Returns
    -------
    dict
        ``"dark_to_light"`` — frame indices where dark→light.
        ``"light_to_dark"`` — frame indices where light→dark.
    """
    diff = np.diff(light_on.astype(int))
    dark_to_light = np.where(diff == 1)[0] + 1
    light_to_dark = np.where(diff == -1)[0] + 1

    return {
        "dark_to_light": dark_to_light,
        "light_to_dark": light_to_dark,
    }


def anchoring_time_course(
    signal: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    light_on: npt.NDArray[np.bool_],
    reference_pd: float | None = None,
    window_frames: int = 300,
    step_frames: int = 60,
    pre_transition_s: float = 10.0,
    post_transition_s: float = 30.0,
    fps: float = 30.0,
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
) -> dict:
    """Compute PD relative to reference around dark→light transitions.

    Parameters
    ----------
    signal : (n_frames,) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
    light_on : (n_frames,) bool
    reference_pd : float or None
        Reference PD (in degrees) to compute deviation from. If None,
        uses the PD from the first light epoch.
    window_frames : int
        Sliding window for tuning curve computation.
    step_frames : int
        Step between windows.
    pre_transition_s : float
        Time before transition to include.
    post_transition_s : float
        Time after transition to include.
    fps : float
    n_bins : int
    smoothing_sigma_deg : float

    Returns
    -------
    dict
        ``"time_offsets_s"`` — time relative to transition.
        ``"pd_deviations"`` — PD deviation from reference per time point.
        ``"mvls"`` — MVL per time point.
        ``"reference_pd"`` — reference PD used.
        ``"n_transitions"`` — number of transitions averaged.
    """
    n = len(signal)
    transitions = find_transitions(light_on)
    d2l = transitions["dark_to_light"]

    if len(d2l) == 0:
        return {
            "time_offsets_s": np.array([]),
            "pd_deviations": np.array([]),
            "mvls": np.array([]),
            "reference_pd": reference_pd or 0.0,
            "n_transitions": 0,
        }

    # Determine reference PD from first light epoch
    if reference_pd is None:
        mask_light = mask & light_on
        if mask_light.sum() >= n_bins:
            tc, bc = compute_hd_tuning_curve(
                signal, hd_deg, mask_light, n_bins=n_bins,
                smoothing_sigma_deg=smoothing_sigma_deg,
            )
            reference_pd = preferred_direction(tc, bc)
        else:
            reference_pd = 0.0

    pre_frames = int(pre_transition_s * fps)
    post_frames = int(post_transition_s * fps)

    # Build time grid
    time_offsets_frames = np.arange(-pre_frames, post_frames, step_frames)
    time_offsets_s = time_offsets_frames / fps

    # Accumulate PD deviations across all transitions
    all_pds = []
    all_mvls = []
    valid_transitions = 0

    for trans in d2l:
        pds = []
        mvls_t = []
        valid_window = True

        for t_offset in time_offsets_frames:
            center = trans + t_offset
            start = center - window_frames // 2
            end = start + window_frames

            if start < 0 or end > n:
                pds.append(np.nan)
                mvls_t.append(np.nan)
                continue

            win_mask = np.zeros_like(mask)
            win_mask[start:end] = mask[start:end]

            if win_mask.sum() < n_bins:
                pds.append(np.nan)
                mvls_t.append(np.nan)
                continue

            tc, bc = compute_hd_tuning_curve(
                signal, hd_deg, win_mask, n_bins=n_bins,
                smoothing_sigma_deg=smoothing_sigma_deg,
            )
            pd = preferred_direction(tc, bc)
            mvl = mean_vector_length(tc, bc)

            # Deviation from reference
            dev = ((pd - reference_pd + 180) % 360) - 180
            pds.append(dev)
            mvls_t.append(mvl)

        all_pds.append(pds)
        all_mvls.append(mvls_t)
        valid_transitions += 1

    if valid_transitions == 0:
        return {
            "time_offsets_s": time_offsets_s,
            "pd_deviations": np.full_like(time_offsets_s, np.nan),
            "mvls": np.full_like(time_offsets_s, np.nan),
            "reference_pd": reference_pd,
            "n_transitions": 0,
        }

    # Average across transitions
    pds_arr = np.array(all_pds, dtype=np.float64)
    mvls_arr = np.array(all_mvls, dtype=np.float64)

    mean_pds = np.nanmean(pds_arr, axis=0)
    mean_mvls = np.nanmean(mvls_arr, axis=0)

    return {
        "time_offsets_s": time_offsets_s,
        "pd_deviations": mean_pds,
        "mvls": mean_mvls,
        "reference_pd": reference_pd,
        "n_transitions": valid_transitions,
    }


def anchoring_speed(
    pd_deviations: npt.NDArray[np.floating],
    time_offsets_s: npt.NDArray[np.floating],
) -> dict:
    """Estimate speed of re-anchoring from PD deviation time course.

    Parameters
    ----------
    pd_deviations : (n_points,) float
        PD deviation from reference.
    time_offsets_s : (n_points,) float
        Time relative to transition.

    Returns
    -------
    dict
        ``"pre_deviation"`` — mean abs deviation before transition.
        ``"post_deviation"`` — mean abs deviation after transition (late period).
        ``"half_time_s"`` — estimated time to reach halfway between pre and post deviation.
        ``"anchoring_strength"`` — (pre - post) / pre; 1 = full re-anchoring, 0 = none.
    """
    valid = np.isfinite(pd_deviations) & np.isfinite(time_offsets_s)
    if valid.sum() < 3:
        return {
            "pre_deviation": np.nan,
            "post_deviation": np.nan,
            "half_time_s": np.nan,
            "anchoring_strength": np.nan,
        }

    # Pre-transition: t < 0
    pre_mask = valid & (time_offsets_s < 0)
    post_mask = valid & (time_offsets_s > time_offsets_s[valid].max() * 0.5)

    pre_dev = float(np.mean(np.abs(pd_deviations[pre_mask]))) if pre_mask.any() else np.nan
    post_dev = float(np.mean(np.abs(pd_deviations[post_mask]))) if post_mask.any() else np.nan

    # Half-time estimation
    half_time = np.nan
    if np.isfinite(pre_dev) and np.isfinite(post_dev) and pre_dev > post_dev:
        target = (pre_dev + post_dev) / 2.0
        post_points = valid & (time_offsets_s > 0)
        if post_points.any():
            abs_devs = np.abs(pd_deviations[post_points])
            times_post = time_offsets_s[post_points]
            crossings = np.where(abs_devs < target)[0]
            if len(crossings) > 0:
                half_time = float(times_post[crossings[0]])

    # Anchoring strength
    if np.isfinite(pre_dev) and pre_dev > 0:
        strength = (pre_dev - post_dev) / pre_dev
    else:
        strength = np.nan

    return {
        "pre_deviation": pre_dev,
        "post_deviation": post_dev,
        "half_time_s": half_time,
        "anchoring_strength": float(strength) if np.isfinite(strength) else np.nan,
    }
