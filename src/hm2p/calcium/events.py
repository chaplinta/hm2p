"""Stage 4 — Voigts & Harnett calcium transient detection.

Detects individual calcium transient events in dF/F0 traces using a
percentile-based Gaussian noise model and CDF probability thresholds.

Algorithm (per ROI):
  1. Rectify trace (clip negatives to 0), optionally Gaussian-smooth.
  2. Normalize to [0, 1].
  3. Estimate noise distribution: mean from the prc_mean percentile,
     std from the prc_high - prc_low percentile range.
  4. Compute per-frame noise probability via the normal CDF:
     noise_prob = 2 * (1 - CDF(trace, mean, std)).
     Values near 0 = likely signal; values near 1 = likely noise.
  5. Detect event onsets where (1 - noise_prob) crosses (1 - prob_onset)
     from below (i.e. noise_prob drops below prob_onset).
  6. Find offsets where noise_prob rises above prob_offset and is increasing.
  7. Optionally require at least one frame with noise_prob < alpha for
     the event to be kept (significance filter).
  8. Record event onset/offset indices and peak dF/F amplitude.

Reference:
    Voigts & Harnett 2020. "Somatic and dendritic encoding of spatial
    variables in retrosplenial cortex differs during 2D navigation."
    Neuron 105(2):237-245. doi:10.1016/j.neuron.2019.10.016
    https://github.com/jvoigts/cell_labeling_bhv
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, stats


# Default parameters matching the legacy pipeline
SMOOTH_SIGMA: int = 3
PRC_MEAN: int = 40
PRC_LOW: int = 10
PRC_HIGH: int = 90
PROB_ONSET: float = 0.2
PROB_OFFSET: float = 0.7
ALPHA: float = 1.0  # legacy default; set < 1 (e.g. 0.05) for significance filtering


@dataclass
class EventResult:
    """Per-ROI calcium event detection results."""

    onsets: np.ndarray  # (n_events,) int — frame indices of event onsets
    offsets: np.ndarray  # (n_events,) int — frame indices of event offsets
    amplitudes: np.ndarray  # (n_events,) float — peak dF/F during each event
    event_mask: np.ndarray  # (n_frames,) int — 1 during event, 0 outside
    noise_prob: np.ndarray  # (n_frames,) float — per-frame noise probability


@dataclass
class BatchEventResult:
    """Event detection results for all ROIs."""

    events: list[EventResult] = field(default_factory=list)
    event_masks: np.ndarray = field(
        default_factory=lambda: np.empty(0)
    )  # (n_rois, n_frames)
    noise_probs: np.ndarray = field(
        default_factory=lambda: np.empty(0)
    )  # (n_rois, n_frames)


def _get_crossings(data: np.ndarray, threshold: float) -> np.ndarray:
    """Find indices where data crosses threshold from below (rising edge)."""
    return np.flatnonzero((data[:-1] <= threshold) & (data[1:] >= threshold)) + 1


def estimate_noise_probability(
    trace: np.ndarray,
    smooth_sigma: int | None = SMOOTH_SIGMA,
    prc_mean: int = PRC_MEAN,
    prc_low: int = PRC_LOW,
    prc_high: int = PRC_HIGH,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-frame noise probability using percentile-based Gaussian model.

    Args:
        trace: (n_frames,) float — dF/F0 trace for a single ROI.
        smooth_sigma: Gaussian smoothing sigma in frames. None to skip smoothing.
        prc_mean: Percentile for estimating distribution mean.
        prc_low: Low percentile for std estimation.
        prc_high: High percentile for std estimation.

    Returns:
        Tuple of:
            noise_prob: (n_frames,) float in [0, 1] — probability each frame is noise.
                        0 = definitely signal, 1 = definitely noise.
            trace_norm: (n_frames,) float in [0, 1] — normalized smoothed trace.
    """
    # Rectify: clip negatives
    trace_rect = np.clip(trace.copy(), 0, None)

    if smooth_sigma is not None and smooth_sigma > 0:
        trace_smooth = ndimage.gaussian_filter1d(trace_rect, sigma=smooth_sigma)
        trace_smooth = np.clip(trace_smooth, 0, None)
    else:
        trace_smooth = trace_rect

    # Normalize to [0, 1]
    vmin = trace_smooth.min()
    vmax = trace_smooth.max()
    if vmax - vmin > 0:
        trace_norm = (trace_smooth - vmin) / (vmax - vmin)
    else:
        trace_norm = np.zeros_like(trace_smooth)

    # Estimate noise distribution from percentiles
    dist_mean = np.percentile(trace_norm, prc_mean)
    dist_std = np.percentile(trace_norm, prc_high) - np.percentile(trace_norm, prc_low)
    if dist_std <= 0:
        dist_std = np.finfo(np.float64).eps

    # Fold: everything below mean is noise
    trace_folded = trace_norm.copy()
    trace_folded[trace_folded < dist_mean] = dist_mean

    # Compute noise probability via normal CDF
    noise_prob = 1 - stats.norm.cdf(trace_folded, loc=dist_mean, scale=dist_std)
    noise_prob = noise_prob * 2  # scale to [0, 1] range (one-tailed → two-tailed)
    noise_prob = np.clip(noise_prob, 0, 1)

    return noise_prob.astype(np.float64), trace_norm.astype(np.float64)


def detect_events_single(
    dff_trace: np.ndarray,
    smooth_sigma: int | None = SMOOTH_SIGMA,
    prc_mean: int = PRC_MEAN,
    prc_low: int = PRC_LOW,
    prc_high: int = PRC_HIGH,
    prob_onset: float = PROB_ONSET,
    prob_offset: float = PROB_OFFSET,
    alpha: float = ALPHA,
) -> EventResult:
    """Detect calcium transient events in a single dF/F0 trace.

    Args:
        dff_trace: (n_frames,) float — dF/F0 trace for one ROI.
        smooth_sigma: Gaussian smoothing sigma in frames for noise estimation.
        prc_mean: Percentile for noise distribution mean.
        prc_low: Low percentile for noise std.
        prc_high: High percentile for noise std.
        prob_onset: Noise probability threshold for event onset (< this = onset).
        prob_offset: Noise probability threshold for event offset (> this = offset).
        alpha: Significance threshold — event must have >= 1 frame with
               noise_prob < alpha to be kept. Set to 1.0 to disable.

    Returns:
        EventResult with onset/offset indices, amplitudes, mask, and noise probs.
    """
    n_frames = len(dff_trace)
    trace = dff_trace.astype(np.float64)

    noise_prob, _ = estimate_noise_probability(
        trace,
        smooth_sigma=smooth_sigma,
        prc_mean=prc_mean,
        prc_low=prc_low,
        prc_high=prc_high,
    )

    # Find candidate onsets: where (1 - noise_prob) crosses (1 - prob_onset)
    onset_candidates = _get_crossings(1 - noise_prob, 1 - prob_onset)

    onsets = []
    offsets = []
    amplitudes = []
    event_mask = np.zeros(n_frames, dtype=np.int32)

    for i_onset in onset_candidates:
        # Skip if this onset falls within a previously detected event
        if onsets and i_onset < offsets[-1]:
            continue

        # Search forward for offset
        sub_probs = noise_prob[i_onset:]
        found_offset = False

        for i_sub in range(1, len(sub_probs)):
            is_rising = sub_probs[i_sub] > sub_probs[i_sub - 1]
            above_offset = sub_probs[i_sub] > prob_offset

            if is_rising and above_offset:
                i_offset = i_onset + i_sub

                # Significance check: at least one frame with noise_prob < alpha
                if np.any(noise_prob[i_onset:i_offset] < alpha):
                    onsets.append(i_onset)
                    offsets.append(i_offset)
                    amplitudes.append(np.max(trace[i_onset:i_offset]))
                    event_mask[i_onset:i_offset] = 1

                found_offset = True
                break

        # If no offset found, event runs to end of trace (don't include it)
        if not found_offset:
            pass

    return EventResult(
        onsets=np.array(onsets, dtype=np.int64),
        offsets=np.array(offsets, dtype=np.int64),
        amplitudes=np.array(amplitudes, dtype=np.float64),
        event_mask=event_mask,
        noise_prob=noise_prob,
    )


def detect_events(
    dff: np.ndarray,
    fps: float,
    smooth_sigma: int | None = SMOOTH_SIGMA,
    prc_mean: int = PRC_MEAN,
    prc_low: int = PRC_LOW,
    prc_high: int = PRC_HIGH,
    prob_onset: float = PROB_ONSET,
    prob_offset: float = PROB_OFFSET,
    alpha: float = ALPHA,
) -> np.ndarray:
    """Detect calcium transient events for all ROIs.

    This is the batch interface matching the original function signature.
    Returns a binary event mask (n_rois, n_frames).

    Args:
        dff: (n_rois, n_frames) float32 — dF/F0 traces.
        fps: Imaging frame rate (Hz). Reserved for future use (e.g. duration filter).
        smooth_sigma: Gaussian smoothing sigma in frames.
        prc_mean: Percentile for noise distribution mean.
        prc_low: Low percentile for noise std.
        prc_high: High percentile for noise std.
        prob_onset: Noise probability threshold for event onset.
        prob_offset: Noise probability threshold for event offset.
        alpha: Significance threshold (1.0 to disable).

    Returns:
        (n_rois, n_frames) float32 — binary event mask.
    """
    n_rois, n_frames = dff.shape
    masks = np.zeros((n_rois, n_frames), dtype=np.float32)

    for i in range(n_rois):
        result = detect_events_single(
            dff[i],
            smooth_sigma=smooth_sigma,
            prc_mean=prc_mean,
            prc_low=prc_low,
            prc_high=prc_high,
            prob_onset=prob_onset,
            prob_offset=prob_offset,
            alpha=alpha,
        )
        masks[i] = result.event_mask.astype(np.float32)

    return masks


def detect_events_batch(
    dff: np.ndarray,
    fps: float,
    smooth_sigma: int | None = SMOOTH_SIGMA,
    prc_mean: int = PRC_MEAN,
    prc_low: int = PRC_LOW,
    prc_high: int = PRC_HIGH,
    prob_onset: float = PROB_ONSET,
    prob_offset: float = PROB_OFFSET,
    alpha: float = ALPHA,
) -> BatchEventResult:
    """Detect calcium transient events for all ROIs, returning full results.

    Unlike `detect_events` which returns only the mask, this returns
    per-ROI EventResult objects with onsets, offsets, amplitudes, and
    noise probabilities.

    Args:
        dff: (n_rois, n_frames) float32 — dF/F0 traces.
        fps: Imaging frame rate (Hz).
        smooth_sigma: Gaussian smoothing sigma in frames.
        prc_mean: Percentile for noise distribution mean.
        prc_low: Low percentile for noise std.
        prc_high: High percentile for noise std.
        prob_onset: Noise probability threshold for event onset.
        prob_offset: Noise probability threshold for event offset.
        alpha: Significance threshold (1.0 to disable).

    Returns:
        BatchEventResult with per-ROI events, stacked masks, and noise probs.
    """
    n_rois, n_frames = dff.shape
    events = []
    masks = np.zeros((n_rois, n_frames), dtype=np.float32)
    noise_probs = np.zeros((n_rois, n_frames), dtype=np.float64)

    for i in range(n_rois):
        result = detect_events_single(
            dff[i],
            smooth_sigma=smooth_sigma,
            prc_mean=prc_mean,
            prc_low=prc_low,
            prc_high=prc_high,
            prob_onset=prob_onset,
            prob_offset=prob_offset,
            alpha=alpha,
        )
        events.append(result)
        masks[i] = result.event_mask.astype(np.float32)
        noise_probs[i] = result.noise_prob

    return BatchEventResult(
        events=events,
        event_masks=masks,
        noise_probs=noise_probs,
    )


def compute_event_snr(
    dff: np.ndarray,
    event_mask: np.ndarray,
    amplitudes: np.ndarray,
    bad_frames: np.ndarray | None = None,
) -> float:
    """Compute event-based SNR for a single ROI.

    SNR = mean(event_amplitudes) / std(dF/F during non-event periods).

    Args:
        dff: (n_frames,) float — dF/F0 trace.
        event_mask: (n_frames,) int — 1 during event, 0 outside.
        amplitudes: (n_events,) float — peak dF/F per event.
        bad_frames: Optional (n_frames,) bool — frames to exclude.

    Returns:
        SNR as float. Returns NaN if no events or no non-event frames.
    """
    if len(amplitudes) == 0:
        return np.nan

    good = np.ones(len(dff), dtype=bool)
    if bad_frames is not None:
        good &= ~bad_frames

    non_event = good & (event_mask == 0)
    if non_event.sum() == 0:
        return np.nan

    signal = np.mean(amplitudes)
    noise = np.std(dff[non_event])
    if noise <= 0:
        return np.nan

    return float(signal / noise)


def compute_event_rate(
    onsets: np.ndarray,
    n_frames: int,
    fps: float,
    bad_frames: np.ndarray | None = None,
) -> float:
    """Compute event rate (events/min) for a single ROI.

    Args:
        onsets: (n_events,) int — event onset frame indices.
        n_frames: Total number of frames.
        fps: Imaging frame rate (Hz).
        bad_frames: Optional (n_frames,) bool — frames to exclude from duration.

    Returns:
        Event rate in events/min.
    """
    if bad_frames is not None:
        n_good = n_frames - bad_frames.sum()
        n_good_events = sum(1 for o in onsets if not bad_frames[o])
    else:
        n_good = n_frames
        n_good_events = len(onsets)

    if n_good <= 0:
        return 0.0

    duration_min = n_good / fps / 60.0
    return float(n_good_events / duration_min) if duration_min > 0 else 0.0


# ---------------------------------------------------------------------------
# Event dynamics characterization
# ---------------------------------------------------------------------------

def characterize_events(
    dff_trace: np.ndarray,
    event_result: EventResult,
    fps: float,
) -> list[dict]:
    """Characterize each detected event's dynamics.

    Per-event metrics following Voigts & Harnett 2020
    (doi:10.1016/j.neuron.2019.10.016):
      - amplitude: peak dF/F during event
      - duration_s: onset-to-offset in seconds
      - rise_frames: onset to peak frame count
      - rise_time_s: onset to peak in seconds
      - decay_frames: peak to offset frame count
      - decay_time_s: peak to offset in seconds
      - auc: area under curve (integral of dF/F during event)
      - mean_dff: mean dF/F during event

    Args:
        dff_trace: (n_frames,) float — dF/F0 trace.
        event_result: EventResult from detect_events_single.
        fps: Imaging frame rate (Hz).

    Returns:
        List of dicts, one per event.
    """
    events = []
    for onset, offset, amp in zip(
        event_result.onsets, event_result.offsets, event_result.amplitudes
    ):
        segment = dff_trace[onset:offset]
        duration_frames = offset - onset
        peak_idx_local = int(np.argmax(segment))

        events.append({
            "onset": int(onset),
            "offset": int(offset),
            "amplitude": float(amp),
            "duration_frames": duration_frames,
            "duration_s": duration_frames / fps,
            "rise_frames": peak_idx_local,
            "rise_time_s": peak_idx_local / fps,
            "decay_frames": duration_frames - peak_idx_local,
            "decay_time_s": (duration_frames - peak_idx_local) / fps,
            "auc": float(np.sum(segment)) / fps,
            "mean_dff": float(np.mean(segment)),
        })
    return events


def summarize_cell_dynamics(
    dff_trace: np.ndarray,
    event_result: EventResult,
    fps: float,
    bad_frames: np.ndarray | None = None,
) -> dict:
    """Compute per-cell summary statistics of calcium event dynamics.

    Aggregates characterize_events() output into population-level metrics.

    Args:
        dff_trace: (n_frames,) float — dF/F0 trace.
        event_result: EventResult from detect_events_single.
        fps: Imaging frame rate (Hz).
        bad_frames: Optional (n_frames,) bool — frames to exclude.

    Returns:
        Dict with summary statistics.
    """
    events = characterize_events(dff_trace, event_result, fps)
    n_events = len(events)

    rate = compute_event_rate(event_result.onsets, len(dff_trace), fps, bad_frames)
    snr = compute_event_snr(dff_trace, event_result.event_mask,
                            event_result.amplitudes, bad_frames)

    if n_events == 0:
        return {
            "n_events": 0,
            "event_rate": rate,
            "snr": snr,
            "mean_amplitude": np.nan,
            "median_amplitude": np.nan,
            "mean_duration_s": np.nan,
            "median_duration_s": np.nan,
            "mean_rise_time_s": np.nan,
            "mean_decay_time_s": np.nan,
            "mean_auc": np.nan,
            "mean_dff_during_events": np.nan,
            "fraction_active": 0.0,
            "mean_iei_s": np.nan,
        }

    amps = np.array([e["amplitude"] for e in events])
    durations = np.array([e["duration_s"] for e in events])
    rises = np.array([e["rise_time_s"] for e in events])
    decays = np.array([e["decay_time_s"] for e in events])
    aucs = np.array([e["auc"] for e in events])
    mean_dffs = np.array([e["mean_dff"] for e in events])

    # Fraction of recording that is "active" (event frames / total frames)
    active_frames = int(event_result.event_mask.sum())
    total_frames = len(dff_trace)
    if bad_frames is not None:
        total_frames = int((~bad_frames).sum())
        active_frames = int((event_result.event_mask.astype(bool) & ~bad_frames).sum())
    fraction_active = active_frames / max(total_frames, 1)

    # Inter-event intervals
    if n_events > 1:
        ieis = np.diff(event_result.onsets) / fps
        mean_iei = float(np.mean(ieis))
    else:
        mean_iei = np.nan

    return {
        "n_events": n_events,
        "event_rate": rate,
        "snr": snr,
        "mean_amplitude": float(np.mean(amps)),
        "median_amplitude": float(np.median(amps)),
        "mean_duration_s": float(np.mean(durations)),
        "median_duration_s": float(np.median(durations)),
        "mean_rise_time_s": float(np.mean(rises)),
        "mean_decay_time_s": float(np.mean(decays)),
        "mean_auc": float(np.mean(aucs)),
        "mean_dff_during_events": float(np.mean(mean_dffs)),
        "fraction_active": fraction_active,
        "mean_iei_s": mean_iei,
    }
