"""Per-cell feature table construction for cell-type comparison.

Builds a wide per-ROI feature table from a session's dF/F traces and
behavioural covariates. Four feature families are computed:

``tr_``
    Amplitude-invariant trace-shape statistics (skewness, kurtosis,
    autocorrelation time, burstiness, dynamic-range ratio). These do not
    depend on the absolute dF/F scale, so they are comparable across
    animals, sessions and expression levels.
``ev_``
    Calcium event kinetics (rate, amplitude, duration, rise/decay time,
    inter-event intervals, plateau fraction) from either the Voigts &
    Harnett detector or the SD-threshold detector.
``tun_``
    Tuning features (HD mean vector length, preferred direction, tuning
    width, HD Skaggs information, AHV modulation, speed modulation),
    including light/dark splits of the HD mean vector length.
``act_``
    Condition-split activity metrics (moving/stationary x light/dark).

The module is pure numpy/pandas — no I/O. Session-level metadata columns
(animal_id, session_id, celltype, equipment, virus) are attached by the
caller, not here.

Motivation
----------
Hypothesis H2 predicts that Penk+ and Penk-CamKII+ retrosplenial neurons
differ in activity kinetics (event rate, dF/F skewness, event duration,
inter-event interval), consistent with the low-rheobase phenotype
described for Penk+ cortical neurons. Trace skewness has been used as a
cell-type signature of calcium dynamics.

References
----------
Voigts & Harnett 2020. "Somatic and dendritic encoding of spatial variables
    in retrosplenial cortex differs during 2D navigation." Neuron
    105(2):237-245. doi:10.1016/j.neuron.2019.10.016
    https://github.com/jvoigts/cell_labeling_bhv
Zong et al. 2022. "Large-scale two-photon calcium imaging in freely moving
    mice." Cell 185(7):1240-1256. doi:10.1016/j.cell.2022.02.017
Brennan et al. 2020. "Thalamus and claustrum control parallel layer 1
    circuits in retrosplenial cortex." Cell Reports 30(4):1226-1238.
    doi:10.1016/j.celrep.2019.12.093
Niethard et al. 2021. "Cortical circuit activity underlying sleep slow
    oscillations and spindles." J Neurosci 41:4212.
    doi:10.1523/JNEUROSCI.1957-20.2021
Skaggs et al. 1993. "An information-theoretic approach to deciphering the
    hippocampal code." NIPS 5:1030-1037. doi:10.1162/neco.1996.8.6.1345
Lubba et al. 2019. "catch22: CAnonical Time-series CHaracteristics."
    Data Mining and Knowledge Discovery 33:1821-1852.
    doi:10.1007/s10618-019-00647-x
    https://github.com/DynamicsAndNeuralSystems/pycatch22
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from hm2p.analysis.activity import compute_cell_activity
from hm2p.analysis.ahv import ahv_modulation_index, ahv_tuning_curve
from hm2p.analysis.information import skaggs_info_rate
from hm2p.analysis.speed import speed_modulation_index
from hm2p.analysis.tuning import (
    compute_hd_tuning_curve,
    mean_vector_length,
    preferred_direction,
    tuning_width_fwhm,
)
from hm2p.calcium.events import (
    EventResult,
    characterize_events,
    compute_event_rate,
    detect_events_sd_single,
    detect_events_single,
    summarize_cell_dynamics,
)

__all__ = [
    "trace_statistics",
    "event_statistics",
    "tuning_features",
    "catch22_features",
    "cell_feature_row",
    "session_feature_table",
    "feature_families",
]

# Fraction of the event peak within which a frame counts as "plateau".
PLATEAU_TOLERANCE: float = 0.2

# Minimum number of valid frames required before tuning features are computed.
MIN_TUNING_FRAMES: int = 10

# Event-detection methods accepted by :func:`event_statistics`.
EVENT_METHODS: tuple[str, ...] = ("vh", "sd")


def _iei_statistics(onsets: npt.NDArray[np.integer], fps: float) -> tuple[float, float]:
    """Coefficient of variation and median of inter-event intervals.

    Parameters
    ----------
    onsets : (n_events,) int
        Event onset frame indices.
    fps : float
        Sampling rate in Hz.

    Returns
    -------
    iei_cv : float
        Standard deviation of inter-event intervals divided by their mean.
        NaN when fewer than three onsets are available (fewer than two
        intervals) or when the mean interval is zero.
    median_iei_s : float
        Median inter-event interval in seconds; NaN when fewer than two
        onsets are available.
    """
    onsets = np.asarray(onsets)
    if onsets.size < 2 or fps <= 0:
        return float("nan"), float("nan")

    ieis = np.diff(np.sort(onsets.astype(np.float64))) / fps
    median_iei = float(np.median(ieis))

    if ieis.size < 2:
        return float("nan"), median_iei

    mean_iei = float(np.mean(ieis))
    if mean_iei <= 0:
        return float("nan"), median_iei
    return float(np.std(ieis, ddof=1) / mean_iei), median_iei


def _plateau_fraction(
    dff_trace: npt.NDArray[np.floating],
    event_result: EventResult,
    tolerance: float = PLATEAU_TOLERANCE,
) -> float:
    """Fraction of event frames sitting within *tolerance* of the event peak.

    A sustained (plateau-like) transient spends many frames near its peak;
    a sharp transient spends few. The measure is scale-free because the
    tolerance is expressed relative to each event's own peak amplitude.

    Parameters
    ----------
    dff_trace : (n_frames,) float
        dF/F trace for one ROI.
    event_result : EventResult
        Detected events for this ROI.
    tolerance : float
        Relative tolerance around the event peak (0.2 = within 20 %).

    Returns
    -------
    float
        Fraction in [0, 1], or NaN when no events with a positive peak exist.
    """
    trace = np.asarray(dff_trace, dtype=np.float64)
    n_plateau = 0
    n_event_frames = 0

    for onset, offset in zip(event_result.onsets, event_result.offsets, strict=False):
        segment = trace[int(onset) : int(offset)]
        if segment.size == 0:
            continue
        peak = float(np.max(segment))
        if not np.isfinite(peak) or peak <= 0:
            continue
        n_event_frames += segment.size
        n_plateau += int(np.count_nonzero(segment >= (1.0 - tolerance) * peak))

    if n_event_frames == 0:
        return float("nan")
    return float(n_plateau / n_event_frames)


def _duration_cv(events: list[dict]) -> float:
    """Coefficient of variation of event durations.

    Parameters
    ----------
    events : list of dict
        Output of :func:`hm2p.calcium.events.characterize_events`.

    Returns
    -------
    float
        Standard deviation of ``duration_s`` divided by its mean, or NaN
        when fewer than two events exist or the mean duration is zero.
    """
    if len(events) < 2:
        return float("nan")
    durations = np.asarray([e["duration_s"] for e in events], dtype=np.float64)
    mean_duration = float(np.mean(durations))
    if mean_duration <= 0:
        return float("nan")
    return float(np.std(durations, ddof=1) / mean_duration)


def _event_summary(
    dff_trace: npt.NDArray[np.floating],
    event_result: EventResult,
    fps: float,
    bad_frames: npt.NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    """Aggregate per-event metrics for one ROI into a flat dictionary.

    Parameters
    ----------
    dff_trace : (n_frames,) float
        dF/F trace for one ROI.
    event_result : EventResult
        Detected events for this ROI.
    fps : float
        Imaging frame rate in Hz.
    bad_frames : (n_frames,) bool, optional
        Frames to exclude from rate and duty-cycle calculations.

    Returns
    -------
    dict
        All keys of :func:`hm2p.calcium.events.summarize_cell_dynamics`
        plus ``iei_cv``, ``median_iei_s``, ``plateau_fraction`` and
        ``duration_cv``.
    """
    summary: dict[str, float] = dict(
        summarize_cell_dynamics(dff_trace, event_result, fps, bad_frames)
    )
    iei_cv, median_iei = _iei_statistics(event_result.onsets, fps)
    summary["iei_cv"] = iei_cv
    summary["median_iei_s"] = median_iei
    summary["plateau_fraction"] = _plateau_fraction(dff_trace, event_result)
    summary["duration_cv"] = _duration_cv(characterize_events(dff_trace, event_result, fps))
    return summary


def _moments(trace: npt.NDArray[np.floating]) -> tuple[float, float]:
    """Bias-corrected sample skewness and excess kurtosis.

    ``scipy.stats`` emits a precision-loss RuntimeWarning when the values
    are nearly identical; in that regime the moments are not meaningful,
    so NaN is returned instead of an unreliable number.

    Parameters
    ----------
    trace : (n_frames,) float
        Finite, non-constant signal.

    Returns
    -------
    skewness, kurtosis : float
        NaN when the moment calculation is numerically unreliable.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            return (
                float(stats.skew(trace, bias=False)),
                float(stats.kurtosis(trace, bias=False)),
            )
        except RuntimeWarning:
            return float("nan"), float("nan")


def _autocorr_time_s(
    trace: npt.NDArray[np.floating],
    fps: float,
    max_lag_frames: int | None = None,
) -> float:
    """Lag at which the normalised autocorrelation first drops below 1/e.

    Parameters
    ----------
    trace : (n_frames,) float
        Signal; assumed to have non-zero variance (checked by the caller).
    fps : float
        Sampling rate in Hz.
    max_lag_frames : int, optional
        Largest lag to examine. ``None`` examines every available lag.

    Returns
    -------
    float
        Decay time in seconds, or NaN if the autocorrelation never drops
        below 1/e within the examined lags.
    """
    x = np.asarray(trace, dtype=np.float64)
    n = x.size
    if n < 2 or fps <= 0:
        return float("nan")

    x = x - np.mean(x)
    denom = float(np.dot(x, x))
    if denom <= 0:
        return float("nan")

    full = np.correlate(x, x, mode="full")
    acf = full[n - 1 :] / denom
    if max_lag_frames is not None:
        acf = acf[: max_lag_frames + 1]

    below = np.flatnonzero(acf < np.exp(-1.0))
    if below.size == 0:
        return float("nan")
    return float(below[0] / fps)


def trace_statistics(
    dff_trace: npt.NDArray[np.floating],
    fps: float,
    onsets: npt.NDArray[np.integer] | None = None,
) -> dict[str, float]:
    """Amplitude-invariant statistics of a dF/F trace.

    All returned quantities are invariant (or close to invariant) to a
    multiplicative rescaling of the trace, so they can be compared across
    animals with different expression levels.

    Skewness of the calcium trace is used as a cell-type signature of
    activity dynamics (Niethard et al. 2021,
    doi:10.1523/JNEUROSCI.1957-20.2021).

    Parameters
    ----------
    dff_trace : (n_frames,) float
        dF/F trace for one ROI.
    fps : float
        Imaging frame rate in Hz.
    onsets : (n_events,) int, optional
        Event onset frame indices. When given, ``burstiness`` is the
        coefficient of variation of the inter-event intervals; otherwise
        it is NaN.

    Returns
    -------
    dict
        ``skewness`` — bias-corrected sample skewness.
        ``kurtosis`` — bias-corrected excess (Fisher) kurtosis.
        ``autocorr_time_s`` — lag at which the normalised autocorrelation
        first drops below 1/e, in seconds (NaN if it never does).
        ``burstiness`` — coefficient of variation of inter-event intervals.
        ``fraction_above_2sd`` — fraction of frames more than two standard
        deviations above the mean.
        ``signal_range_ratio`` — (95th - 5th percentile) divided by the
        median absolute deviation.

        Every value is NaN for a constant trace or a trace shorter than
        four frames.

    Raises
    ------
    ValueError
        If ``dff_trace`` is not one-dimensional.
    """
    trace = np.asarray(dff_trace, dtype=np.float64)
    if trace.ndim != 1:
        raise ValueError(f"dff_trace must be 1-D; got shape {trace.shape}")

    burstiness, _ = _iei_statistics(
        np.asarray([], dtype=np.int64) if onsets is None else np.asarray(onsets),
        fps,
    )

    nan_result: dict[str, float] = {
        "skewness": float("nan"),
        "kurtosis": float("nan"),
        "autocorr_time_s": float("nan"),
        "burstiness": burstiness,
        "fraction_above_2sd": float("nan"),
        "signal_range_ratio": float("nan"),
    }

    finite = np.isfinite(trace)
    if trace.size < 4 or not finite.all():
        return nan_result

    sd = float(np.std(trace))
    if sd <= 0:
        # Constant trace: higher moments and the range ratio are undefined.
        return nan_result

    mad = float(np.median(np.abs(trace - np.median(trace))))
    if mad > 0:
        rng_ratio = float((np.percentile(trace, 95.0) - np.percentile(trace, 5.0)) / mad)
    else:
        rng_ratio = float("nan")

    skewness, kurtosis = _moments(trace)

    return {
        "skewness": skewness,
        "kurtosis": kurtosis,
        "autocorr_time_s": _autocorr_time_s(trace, fps),
        "burstiness": burstiness,
        "fraction_above_2sd": float(np.mean(trace > np.mean(trace) + 2.0 * sd)),
        "signal_range_ratio": rng_ratio,
    }


def detect_cell_events(
    dff_trace: npt.NDArray[np.floating],
    fps: float,
    method: str = "vh",
) -> EventResult:
    """Run the requested event detector on a single trace.

    Parameters
    ----------
    dff_trace : (n_frames,) float
        dF/F trace for one ROI.
    fps : float
        Imaging frame rate in Hz.
    method : {"vh", "sd"}
        ``"vh"`` uses the Voigts & Harnett noise-probability detector
        (doi:10.1016/j.neuron.2019.10.016); ``"sd"`` uses the
        SD-threshold detector (Zong et al. 2022,
        doi:10.1016/j.cell.2022.02.017).

    Returns
    -------
    EventResult

    Raises
    ------
    ValueError
        If *method* is not one of ``EVENT_METHODS``.
    """
    if method not in EVENT_METHODS:
        raise ValueError(f"method must be one of {EVENT_METHODS}; got {method!r}")
    trace = np.asarray(dff_trace, dtype=np.float64)
    if method == "vh":
        return detect_events_single(trace)
    return detect_events_sd_single(trace, fps)


def event_statistics(
    dff_trace: npt.NDArray[np.floating],
    fps: float,
    bad_frames: npt.NDArray[np.bool_] | None = None,
    method: str = "vh",
) -> dict[str, float]:
    """Calcium event kinetics for one ROI.

    Runs the requested detector and summarises the resulting events.

    Parameters
    ----------
    dff_trace : (n_frames,) float
        dF/F trace for one ROI.
    fps : float
        Imaging frame rate in Hz.
    bad_frames : (n_frames,) bool, optional
        Frames to exclude from rate and duty-cycle calculations.
    method : {"vh", "sd"}
        Event detector, see :func:`detect_cell_events`.

    Returns
    -------
    dict
        All keys returned by
        :func:`hm2p.calcium.events.summarize_cell_dynamics`, plus
        ``iei_cv`` (coefficient of variation of inter-event intervals),
        ``median_iei_s``, ``plateau_fraction`` (fraction of event frames
        within 20 % of the event peak) and ``duration_cv``.
    """
    trace = np.asarray(dff_trace, dtype=np.float64)
    result = detect_cell_events(trace, fps, method=method)
    return _event_summary(trace, result, fps, bad_frames)


def _hd_occupancy(
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_bins: int,
) -> np.ndarray:
    """Frame counts per head-direction bin over the masked frames.

    Parameters
    ----------
    hd_deg : (n_frames,) float
        Head direction in degrees.
    mask : (n_frames,) bool
        Frames to include.
    n_bins : int
        Number of bins spanning [0, 360).

    Returns
    -------
    (n_bins,) float
        Occupancy in frames.
    """
    hd_mod = np.mod(np.asarray(hd_deg, dtype=np.float64)[np.asarray(mask, dtype=bool)], 360.0)
    edges = np.linspace(0.0, 360.0, n_bins + 1)
    idx = np.clip(np.digitize(hd_mod, edges) - 1, 0, n_bins - 1)
    occupancy = np.zeros(n_bins, dtype=np.float64)
    np.add.at(occupancy, idx, 1.0)
    return occupancy


def _nan_tuning_features() -> dict[str, float]:
    """Tuning feature dict with all values NaN (used when frames are scarce)."""
    return {
        "mvl": float("nan"),
        "preferred_direction": float("nan"),
        "fwhm": float("nan"),
        "hd_skaggs_info": float("nan"),
        "ahv_modulation_index": float("nan"),
        "ahv_modulation_depth": float("nan"),
        "ahv_asymmetry_index": float("nan"),
        "speed_modulation_index": float("nan"),
        "speed_correlation": float("nan"),
    }


def tuning_features(
    signal: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    ahv_deg_s: npt.NDArray[np.floating],
    speed_cm_s: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    fps: float,
    n_bins: int = 36,
    smoothing_sigma_deg: float = 6.0,
) -> dict[str, float]:
    """Head-direction, angular-velocity and speed tuning features for one cell.

    Parameters
    ----------
    signal : (n_frames,) float
        Neural signal (dF/F, event mask or deconvolved rate).
    hd_deg : (n_frames,) float
        Head direction in degrees.
    ahv_deg_s : (n_frames,) float
        Angular head velocity in deg/s.
    speed_cm_s : (n_frames,) float
        Running speed in cm/s.
    mask : (n_frames,) bool
        Valid frames (e.g. active and not bad_behav).
    fps : float
        Sampling rate in Hz. Retained for interface symmetry; the tuning
        statistics themselves are rate-independent.
    n_bins : int
        Number of head-direction bins.
    smoothing_sigma_deg : float
        Gaussian smoothing sigma for the HD tuning curve, in degrees.

    Returns
    -------
    dict
        ``mvl``, ``preferred_direction``, ``fwhm``, ``hd_skaggs_info``,
        ``ahv_modulation_index`` (modulation depth divided by the mean of
        the AHV tuning curve), ``ahv_modulation_depth``,
        ``ahv_asymmetry_index``, ``speed_modulation_index`` and
        ``speed_correlation``. All NaN when fewer than
        ``MIN_TUNING_FRAMES`` frames are valid.
    """
    del fps  # rate-independent statistics; kept for a uniform call signature
    sig = np.asarray(signal, dtype=np.float64)
    hd_arr = np.asarray(hd_deg, dtype=np.float64)
    ahv_arr = np.asarray(ahv_deg_s, dtype=np.float64)
    speed_arr = np.asarray(speed_cm_s, dtype=np.float64)
    # Pose gaps leave NaN in the behavioural channels; a NaN speed makes the
    # median threshold NaN and every comparison False, so restrict to finite frames.
    mask_arr = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(sig)
        & np.isfinite(hd_arr)
        & np.isfinite(ahv_arr)
        & np.isfinite(speed_arr)
    )

    if int(mask_arr.sum()) < MIN_TUNING_FRAMES:
        return _nan_tuning_features()

    tc, bin_centers = compute_hd_tuning_curve(
        sig,
        np.asarray(hd_deg, dtype=np.float64),
        mask_arr,
        n_bins=n_bins,
        smoothing_sigma_deg=smoothing_sigma_deg,
    )
    occupancy = _hd_occupancy(hd_deg, mask_arr, n_bins)

    ahv_tc, ahv_centers = ahv_tuning_curve(sig, np.asarray(ahv_deg_s, dtype=np.float64), mask_arr)
    ahv_stats = ahv_modulation_index(ahv_tc, ahv_centers)
    ahv_mean = float(np.nanmean(ahv_tc)) if np.any(~np.isnan(ahv_tc)) else 0.0
    if ahv_mean > 0:
        ahv_mi = float(ahv_stats["modulation_depth"] / ahv_mean)
    else:
        ahv_mi = float("nan")

    speed_stats = speed_modulation_index(sig, np.asarray(speed_cm_s, dtype=np.float64), mask_arr)

    return {
        "mvl": float(mean_vector_length(tc, bin_centers)),
        "preferred_direction": float(preferred_direction(tc, bin_centers)),
        "fwhm": float(tuning_width_fwhm(tc, bin_centers)),
        "hd_skaggs_info": float(skaggs_info_rate(tc, occupancy)),
        "ahv_modulation_index": ahv_mi,
        "ahv_modulation_depth": float(ahv_stats["modulation_depth"]),
        "ahv_asymmetry_index": float(ahv_stats["asymmetry_index"]),
        "speed_modulation_index": float(speed_stats["speed_modulation_index"]),
        "speed_correlation": float(speed_stats["speed_correlation"]),
    }


def catch22_features(trace: npt.NDArray[np.floating]) -> dict[str, float]:
    """Optional catch22 time-series features for one trace.

    catch22 is a set of 22 canonical time-series characteristics selected
    for low redundancy and high classification performance.

    Lubba et al. 2019. "catch22: CAnonical Time-series CHaracteristics."
    Data Mining and Knowledge Discovery 33:1821-1852.
    doi:10.1007/s10618-019-00647-x
    https://github.com/DynamicsAndNeuralSystems/pycatch22

    Parameters
    ----------
    trace : (n_frames,) float
        Signal to characterise.

    Returns
    -------
    dict
        Feature name (prefixed ``c22_``) to value. Empty dict when
        ``pycatch22`` is not installed.
    """
    try:
        import pycatch22
    except ImportError:
        return {}

    values = np.asarray(trace, dtype=np.float64)
    out = pycatch22.catch22_all(values.tolist())
    return {
        f"c22_{name}": float(value)
        for name, value in zip(out["names"], out["values"], strict=False)
    }


def cell_feature_row(
    roi_idx: int,
    dff_trace: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    ahv_deg_s: npt.NDArray[np.floating],
    speed_cm_s: npt.NDArray[np.floating],
    light_on: npt.NDArray[np.bool_],
    active: npt.NDArray[np.bool_],
    bad_behav: npt.NDArray[np.bool_],
    fps: float,
    *,
    event_method: str = "vh",
    include_catch22: bool = False,
) -> dict[str, Any]:
    """Compute the full feature dictionary for one ROI.

    Parameters
    ----------
    roi_idx : int
        ROI index within the session.
    dff_trace : (n_frames,) float
        dF/F trace for this ROI.
    hd_deg : (n_frames,) float
        Head direction in degrees.
    ahv_deg_s : (n_frames,) float
        Angular head velocity in deg/s.
    speed_cm_s : (n_frames,) float
        Running speed in cm/s.
    light_on : (n_frames,) bool
        True when room lights are on.
    active : (n_frames,) bool
        True for frames where the animal is moving.
    bad_behav : (n_frames,) bool
        True for frames flagged as behavioural artefact (e.g. the animal
        caught on the imaging fibre); these are excluded everywhere.
    fps : float
        Imaging frame rate in Hz.
    event_method : {"vh", "sd"}
        Event detector to use.
    include_catch22 : bool
        Whether to append catch22 features (no-op if pycatch22 is absent).

    Returns
    -------
    dict
        ``roi_idx`` plus features prefixed ``tr_``, ``ev_``, ``tun_`` and
        ``act_``, plus light/dark splits ``tun_mvl_light``,
        ``tun_mvl_dark``, ``ev_event_rate_light`` and
        ``ev_event_rate_dark``.

    Raises
    ------
    ValueError
        If the input arrays do not all have the same length.
    """
    trace = np.asarray(dff_trace, dtype=np.float64)
    hd = np.asarray(hd_deg, dtype=np.float64)
    ahv = np.asarray(ahv_deg_s, dtype=np.float64)
    speed = np.asarray(speed_cm_s, dtype=np.float64)
    light = np.asarray(light_on, dtype=bool)
    active_arr = np.asarray(active, dtype=bool)
    bad = np.asarray(bad_behav, dtype=bool)

    lengths = {arr.shape[0] for arr in (trace, hd, ahv, speed, light, active_arr, bad)}
    if len(lengths) != 1:
        raise ValueError(f"all inputs must have the same length; got lengths {sorted(lengths)}")

    n_frames = trace.shape[0]
    good = ~bad
    valid = active_arr & good

    event_result = detect_cell_events(trace, fps, method=event_method)
    event_mask = event_result.event_mask.astype(bool)

    row: dict[str, Any] = {"roi_idx": int(roi_idx)}

    for key, value in _event_summary(trace, event_result, fps, bad).items():
        row[f"ev_{key}"] = value

    # Light/dark event rates: frames of the opposite condition (and bad
    # frames) are treated as excluded for the rate denominator.
    row["ev_event_rate_light"] = compute_event_rate(
        event_result.onsets, n_frames, fps, bad_frames=~(light & good)
    )
    row["ev_event_rate_dark"] = compute_event_rate(
        event_result.onsets, n_frames, fps, bad_frames=~(~light & good)
    )

    for key, value in trace_statistics(trace, fps, onsets=event_result.onsets).items():
        row[f"tr_{key}"] = value

    for key, value in tuning_features(trace, hd, ahv, speed, valid, fps).items():
        row[f"tun_{key}"] = value

    row["tun_mvl_light"] = tuning_features(trace, hd, ahv, speed, valid & light, fps)["mvl"]
    row["tun_mvl_dark"] = tuning_features(trace, hd, ahv, speed, valid & ~light, fps)["mvl"]

    for key, value in compute_cell_activity(trace, event_mask, speed, light, good, fps).items():
        row[f"act_{key}"] = value

    if include_catch22:
        row.update(catch22_features(trace))

    return row


def session_feature_table(
    dff: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    ahv_deg_s: npt.NDArray[np.floating],
    speed_cm_s: npt.NDArray[np.floating],
    light_on: npt.NDArray[np.bool_],
    active: npt.NDArray[np.bool_],
    bad_behav: npt.NDArray[np.bool_],
    fps: float,
    roi_types: npt.NDArray[np.str_] | list[str] | None = None,
    *,
    event_method: str = "vh",
    include_catch22: bool = False,
) -> pd.DataFrame:
    """Build the per-ROI feature table for one session.

    Parameters
    ----------
    dff : (n_rois, n_frames) float
        dF/F traces.
    hd_deg, ahv_deg_s, speed_cm_s : (n_frames,) float
        Behavioural covariates.
    light_on, active, bad_behav : (n_frames,) bool
        Frame masks, see :func:`cell_feature_row`.
    fps : float
        Imaging frame rate in Hz.
    roi_types : sequence of str, optional
        Per-ROI type label (e.g. ``"soma"`` / ``"dendrite"``). Added as a
        ``roi_type`` column when given.
    event_method : {"vh", "sd"}
        Event detector to use.
    include_catch22 : bool
        Whether to append catch22 features.

    Returns
    -------
    pandas.DataFrame
        One row per ROI, with ``roi_idx`` first. Session-level metadata
        (animal_id, session_id, celltype, ...) is attached by the caller.

    Raises
    ------
    ValueError
        If ``dff`` is not two-dimensional or ``roi_types`` has the wrong
        length.
    """
    traces = np.asarray(dff, dtype=np.float64)
    if traces.ndim != 2:
        raise ValueError(f"dff must be 2-D (n_rois, n_frames); got shape {traces.shape}")

    n_rois = traces.shape[0]
    if roi_types is not None and len(roi_types) != n_rois:
        raise ValueError(f"roi_types has length {len(roi_types)} but dff has {n_rois} ROIs")

    rows: list[dict[str, Any]] = []
    for i in range(n_rois):
        row = cell_feature_row(
            i,
            traces[i],
            hd_deg,
            ahv_deg_s,
            speed_cm_s,
            light_on,
            active,
            bad_behav,
            fps,
            event_method=event_method,
            include_catch22=include_catch22,
        )
        if roi_types is not None:
            row["roi_type"] = str(roi_types[i])
        rows.append(row)

    table = pd.DataFrame(rows)
    if roi_types is not None and not table.empty:
        cols = ["roi_idx", "roi_type"] + [
            c for c in table.columns if c not in ("roi_idx", "roi_type")
        ]
        table = table[cols]
    return table


def feature_families() -> dict[str, list[str]]:
    """Map false-discovery-rate family names to feature column names.

    Grouping features into families keeps the multiple-comparison
    correction within scientifically coherent sets rather than applying
    one correction across all features at once.

    Returns
    -------
    dict
        ``"kinetics"`` — event rate, amplitude, duration and interval
        features (hypothesis H2).
        ``"trace_shape"`` — amplitude-invariant trace statistics.
        ``"tuning"`` — head-direction, angular-velocity and speed tuning.
        ``"activity"`` — condition-split activity metrics.
    """
    return {
        "kinetics": [
            "ev_event_rate",
            "ev_event_rate_light",
            "ev_event_rate_dark",
            "ev_mean_amplitude",
            "ev_median_amplitude",
            "ev_mean_duration_s",
            "ev_median_duration_s",
            "ev_mean_rise_time_s",
            "ev_mean_decay_time_s",
            "ev_mean_auc",
            "ev_fraction_active",
            "ev_mean_iei_s",
            "ev_median_iei_s",
            "ev_iei_cv",
            "ev_duration_cv",
            "ev_plateau_fraction",
            "ev_snr",
        ],
        "trace_shape": [
            "tr_skewness",
            "tr_kurtosis",
            "tr_autocorr_time_s",
            "tr_burstiness",
            "tr_fraction_above_2sd",
            "tr_signal_range_ratio",
        ],
        "tuning": [
            "tun_mvl",
            "tun_mvl_light",
            "tun_mvl_dark",
            "tun_fwhm",
            "tun_hd_skaggs_info",
            "tun_ahv_modulation_index",
            "tun_ahv_modulation_depth",
            "tun_ahv_asymmetry_index",
            "tun_speed_modulation_index",
            "tun_speed_correlation",
        ],
        "activity": [
            "act_moving_light_event_rate",
            "act_moving_dark_event_rate",
            "act_stationary_light_event_rate",
            "act_stationary_dark_event_rate",
            "act_moving_light_mean_signal",
            "act_moving_dark_mean_signal",
            "act_stationary_light_mean_signal",
            "act_stationary_dark_mean_signal",
            "act_movement_modulation",
            "act_light_modulation",
        ],
    }
