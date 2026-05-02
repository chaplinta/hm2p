"""Stage 4 — per-ROI quality control metrics.

Computes a structured QC table with one row per ROI. Metrics are written into
ca.h5 as a group of equal-length 1D arrays (``roi_qc/``). They are used by
``roi_viewer_page.py`` to flag potentially unreliable ROIs; they do NOT remove
any ROIs from the dataset.

Metrics
-------
roi_index       int32    ROI index (same order as dff / F_raw arrays)
snr_event       float32  Event-based SNR = mean(event amplitudes) / std(dF/F outside events)
decay_tau_s     float32  Median exponential decay time constant (s) across detected events.
                         NaN if fewer than 3 events, or if all exponential fits fail.
fneu_dff_corr   float32  Spearman rank correlation between this ROI's dff and the
                         mean Fneu trace across all ROIs (Spearman, non-parametric).
bleach_slope    float32  Fractional bleaching slope: (F_end_mean - F_start_mean) / F_start_mean.
                         First/last 10 % of frames used as windows. Negative = bleaching loss.
active_fraction float32  Fraction of frames where the event mask is 1. Falls back to
                         fraction where dff > 3*MAD if no event mask is available.

Recommended thresholds (for flagging — not for automatic exclusion):
  snr_event       < 3.0
  decay_tau_s     outside [0.2, 4.0] s
  fneu_dff_corr   > 0.6
  bleach_slope    < -0.4  (> 40 % loss)
  active_fraction < 0.05

References:
    Pnevmatikakis et al. 2016. "Simultaneous Denoising, Deconvolution, and
    Demixing of Calcium Imaging Data." Neuron 89(2):285-299.
    doi:10.1016/j.neuron.2015.11.037
    (SNR and decay-time QC metrics for calcium imaging.)

    Voigts & Harnett 2020. "Somatic and dendritic encoding of spatial
    variables in retrosplenial cortex differs during 2D navigation."
    Neuron 105(2):237-245. doi:10.1016/j.neuron.2019.10.016
    (V&H event detection used for SNR and active-fraction metrics.)
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import optimize, stats

from hm2p.calcium.events import EventResult, compute_event_snr

log = logging.getLogger(__name__)

# --- Recommended QC thresholds (for display / flagging only) ---
SNR_MIN: float = 3.0
TAU_MIN_S: float = 0.2
TAU_MAX_S: float = 4.0
FNEU_CORR_MAX: float = 0.6
BLEACH_MAX_LOSS: float = -0.4  # slope < -0.4 flags > 40% loss
ACTIVE_FRAC_MIN: float = 0.05

# Fraction of frames used for bleaching slope estimation (first/last window)
BLEACH_WINDOW_FRAC: float = 0.10

# Minimum events required to estimate decay tau
MIN_EVENTS_FOR_TAU: int = 3


def _fit_exponential_decay(
    segment: np.ndarray,
    fps: float,
) -> float:
    """Fit a single-exponential decay to a post-peak segment.

    Fits ``A * exp(-t / tau) + C`` to the segment starting from the peak.
    Returns ``tau`` in seconds, or ``np.nan`` on failure.

    Parameters
    ----------
    segment : np.ndarray
        (n,) float — dF/F segment from event onset to offset.
    fps : float
        Imaging frame rate (Hz). Used to convert frame counts to seconds.

    Returns
    -------
    float
        Decay time constant tau (s), or np.nan if fit fails.
    """
    if len(segment) < 4:
        return np.nan

    peak_idx = int(np.argmax(segment))
    decay_seg = segment[peak_idx:]

    if len(decay_seg) < 3:
        return np.nan

    t = np.arange(len(decay_seg), dtype=float) / fps
    y = decay_seg.astype(float)

    # Guard against non-positive values that break log-space fitting
    if y[0] <= 0:
        return np.nan

    def _exp_decay(t_: np.ndarray, tau: float, c: float) -> np.ndarray:
        return (y[0] - c) * np.exp(-t_ / tau) + c

    try:
        # Initial guess: tau = half the segment length in seconds, c = final value
        tau0 = max(t[-1] / 2.0, 0.01)
        c0 = float(y[-1])
        popt, _ = optimize.curve_fit(
            _exp_decay,
            t,
            y,
            p0=[tau0, c0],
            bounds=([1e-4, -np.inf], [100.0, np.inf]),
            maxfev=400,
        )
        tau_fit = float(popt[0])
        if tau_fit <= 0:
            return np.nan
        return tau_fit
    except (RuntimeError, ValueError):
        return np.nan


def compute_decay_tau(
    dff_trace: np.ndarray,
    event_result: EventResult,
    fps: float,
) -> float:
    """Compute median exponential decay time constant across detected events.

    For each event, fits an exponential decay to the post-peak dF/F segment
    and returns the median tau across all events with successful fits.

    Parameters
    ----------
    dff_trace : np.ndarray
        (n_frames,) float — dF/F0 trace for one ROI.
    event_result : EventResult
        Detected events from ``detect_events_single`` or ``detect_events_batch``.
    fps : float
        Imaging frame rate (Hz).

    Returns
    -------
    float
        Median decay tau (s) across events. NaN if fewer than ``MIN_EVENTS_FOR_TAU``
        events or all fits fail.
    """
    n_events = len(event_result.onsets)
    if n_events < MIN_EVENTS_FOR_TAU:
        return np.nan

    taus = []
    for onset, offset in zip(event_result.onsets, event_result.offsets, strict=False):
        segment = dff_trace[int(onset) : int(offset)]
        tau = _fit_exponential_decay(segment, fps)
        if np.isfinite(tau):
            taus.append(tau)

    if len(taus) == 0:
        return np.nan

    return float(np.median(taus))


def compute_fneu_dff_corr(
    dff_trace: np.ndarray,
    mean_fneu: np.ndarray,
) -> float:
    """Compute Spearman rank correlation between ROI dff and mean neuropil trace.

    A high correlation indicates the ROI signal may be dominated by residual
    neuropil contamination.

    Uses Spearman rank correlation (non-parametric, per CLAUDE.md policy).

    Parameters
    ----------
    dff_trace : np.ndarray
        (n_frames,) float — dF/F0 for one ROI.
    mean_fneu : np.ndarray
        (n_frames,) float — mean Fneu across all ROIs.

    Returns
    -------
    float
        Spearman r in [-1, 1], or NaN if fewer than 10 frames or all-constant.

    References
    ----------
    scipy.stats.spearmanr is used directly (Spearman rank correlation is
    a non-parametric measure; CLAUDE.md prohibits Pearson correlation).
    """
    n = len(dff_trace)
    if n < 10 or len(mean_fneu) != n:
        return np.nan

    # Guard against constant traces (spearmanr returns NaN cleanly, but log it)
    if np.std(dff_trace) == 0 or np.std(mean_fneu) == 0:
        return np.nan

    result = stats.spearmanr(dff_trace, mean_fneu)
    corr = float(result.statistic)
    if not np.isfinite(corr):
        return np.nan
    return corr


def compute_bleach_slope(
    F_raw: np.ndarray,
    window_frac: float = BLEACH_WINDOW_FRAC,
) -> float:
    """Estimate fractional photobleaching slope from raw fluorescence.

    Computes ``(F_end - F_start) / F_start`` where ``F_start`` and ``F_end``
    are the means of the first and last ``window_frac`` of frames respectively.
    A negative value indicates bleaching loss.

    Parameters
    ----------
    F_raw : np.ndarray
        (n_frames,) float — raw fluorescence for one ROI (pre-subtraction).
    window_frac : float
        Fraction of total frames to use as start/end windows. Default 0.10
        (first/last 10 %).

    Returns
    -------
    float
        Fractional bleaching slope (dimensionless). Negative = bleaching loss.
        NaN if F_start mean is zero or trace is too short.
    """
    n = len(F_raw)
    window = max(1, int(round(n * window_frac)))

    f_start = float(np.mean(F_raw[:window]))
    f_end = float(np.mean(F_raw[-window:]))

    if f_start <= 0 or not np.isfinite(f_start):
        return np.nan

    return (f_end - f_start) / f_start


def compute_active_fraction(
    dff_trace: np.ndarray,
    event_mask: np.ndarray | None,
) -> float:
    """Compute fraction of frames where the ROI is active.

    If an event mask is provided, active fraction = sum(event_mask) / n_frames.
    Otherwise, falls back to fraction of frames where dff > 3 * MAD(|dff|).
    The MAD-based proxy is documented explicitly here because the fallback
    avoids a circular dependency on event detection.

    Parameters
    ----------
    dff_trace : np.ndarray
        (n_frames,) float — dF/F0 for one ROI.
    event_mask : np.ndarray or None
        (n_frames,) int/bool — 1/True during events, 0/False outside.
        If None, uses the 3*MAD proxy.

    Returns
    -------
    float
        Active fraction in [0, 1].
    """
    n = len(dff_trace)
    if n == 0:
        return np.nan

    if event_mask is not None and len(event_mask) == n:
        return float(np.sum(event_mask > 0)) / n

    # Fallback: 3*MAD threshold (robust, no event detection required)
    # MAD-based SD estimate: sigma_MAD = median(|x - median(x)|) * 1.4826
    mad = float(np.median(np.abs(dff_trace - np.median(dff_trace)))) * 1.4826
    if mad <= 0:
        return 0.0
    threshold = 3.0 * mad
    return float(np.sum(dff_trace > threshold)) / n


def compute_roi_qc(
    dff: np.ndarray,
    F_raw: np.ndarray,
    Fneu_raw: np.ndarray,
    event_results: list[EventResult] | None,
    event_masks: np.ndarray | None,
    fps: float,
    bad_frames: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-ROI QC metrics for all ROIs.

    Parameters
    ----------
    dff : np.ndarray
        (n_rois, n_frames) float32 — dF/F0 traces.
    F_raw : np.ndarray
        (n_rois, n_frames) float32 — raw fluorescence (pre-subtraction).
    Fneu_raw : np.ndarray
        (n_rois, n_frames) float32 — neuropil traces for all ROIs.
    event_results : list of EventResult or None
        Per-ROI event detection results (from ``detect_events_batch``).
        If None, event-dependent metrics (snr_event, decay_tau_s) will be NaN
        and active_fraction uses the MAD proxy.
    event_masks : np.ndarray or None
        (n_rois, n_frames) float32 — binary event masks. Used for
        active_fraction when event_results is None but masks are available.
    fps : float
        Imaging frame rate (Hz).
    bad_frames : np.ndarray or None
        (n_frames,) bool — frames to exclude from SNR calculation.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: roi_index, snr_event, decay_tau_s, fneu_dff_corr,
        bleach_slope, active_fraction.
        All arrays have length n_rois.

    References
    ----------
    Pnevmatikakis et al. 2016. "Simultaneous Denoising, Deconvolution, and
    Demixing of Calcium Imaging Data." Neuron 89(2):285-299.
    doi:10.1016/j.neuron.2015.11.037

    Voigts & Harnett 2020. "Somatic and dendritic encoding of spatial
    variables in retrosplenial cortex differs during 2D navigation."
    Neuron 105(2):237-245. doi:10.1016/j.neuron.2019.10.016
    """
    n_rois, n_frames = dff.shape

    # Mean Fneu across all ROIs — used for contamination correlation
    mean_fneu = np.mean(Fneu_raw, axis=0).astype(np.float64)

    roi_index = np.arange(n_rois, dtype=np.int32)
    snr_event = np.full(n_rois, np.nan, dtype=np.float32)
    decay_tau_s = np.full(n_rois, np.nan, dtype=np.float32)
    fneu_dff_corr = np.full(n_rois, np.nan, dtype=np.float32)
    bleach_slope = np.full(n_rois, np.nan, dtype=np.float32)
    active_fraction = np.full(n_rois, np.nan, dtype=np.float32)

    for i in range(n_rois):
        trace = dff[i].astype(np.float64)

        # SNR and decay tau require event results
        ev_mask_i: np.ndarray | None = None
        if event_results is not None and i < len(event_results):
            er = event_results[i]
            snr_event[i] = compute_event_snr(trace, er.event_mask, er.amplitudes, bad_frames)
            decay_tau_s[i] = compute_decay_tau(trace, er, fps)
            ev_mask_i = er.event_mask
        elif event_masks is not None and i < event_masks.shape[0]:
            ev_mask_i = event_masks[i]

        # Spearman correlation with mean neuropil (non-parametric)
        fneu_dff_corr[i] = compute_fneu_dff_corr(trace, mean_fneu)

        # Bleaching slope from raw F
        bleach_slope[i] = compute_bleach_slope(F_raw[i].astype(np.float64))

        # Active fraction
        active_fraction[i] = compute_active_fraction(trace, ev_mask_i)

    log.info(
        "ROI QC computed for %d ROIs: median SNR=%.1f, median tau=%.2f s, median active_frac=%.3f",
        n_rois,
        float(np.nanmedian(snr_event)),
        float(np.nanmedian(decay_tau_s)),
        float(np.nanmedian(active_fraction)),
    )

    return {
        "roi_qc/roi_index": roi_index,
        "roi_qc/snr_event": snr_event,
        "roi_qc/decay_tau_s": decay_tau_s,
        "roi_qc/fneu_dff_corr": fneu_dff_corr,
        "roi_qc/bleach_slope": bleach_slope,
        "roi_qc/active_fraction": active_fraction,
    }


def flag_roi_qc(
    qc: dict[str, np.ndarray],
    snr_min: float = SNR_MIN,
    tau_min_s: float = TAU_MIN_S,
    tau_max_s: float = TAU_MAX_S,
    fneu_corr_max: float = FNEU_CORR_MAX,
    bleach_max_loss: float = BLEACH_MAX_LOSS,
    active_frac_min: float = ACTIVE_FRAC_MIN,
) -> np.ndarray:
    """Return a boolean array marking ROIs that fail one or more QC thresholds.

    Parameters
    ----------
    qc : dict[str, np.ndarray]
        Output of ``compute_roi_qc`` (keys use ``roi_qc/`` prefix).
    snr_min : float
        Minimum acceptable event SNR. ROIs below this are flagged.
    tau_min_s : float
        Minimum acceptable decay tau (s).
    tau_max_s : float
        Maximum acceptable decay tau (s).
    fneu_corr_max : float
        Maximum acceptable Spearman correlation with neuropil.
    bleach_max_loss : float
        Minimum acceptable bleach slope (< this = too much bleaching).
    active_frac_min : float
        Minimum acceptable active fraction.

    Returns
    -------
    np.ndarray
        (n_rois,) bool — True = flagged (fails at least one criterion).
    """
    snr = qc["roi_qc/snr_event"]
    tau = qc["roi_qc/decay_tau_s"]
    corr = qc["roi_qc/fneu_dff_corr"]
    bleach = qc["roi_qc/bleach_slope"]
    active = qc["roi_qc/active_fraction"]

    n = len(snr)
    flagged = np.zeros(n, dtype=bool)

    # A NaN metric does not flag the ROI (it is unknown, not bad)
    flagged |= np.where(np.isfinite(snr), snr < snr_min, False)
    flagged |= np.where(np.isfinite(tau), (tau < tau_min_s) | (tau > tau_max_s), False)
    flagged |= np.where(np.isfinite(corr), corr > fneu_corr_max, False)
    flagged |= np.where(np.isfinite(bleach), bleach < bleach_max_loss, False)
    flagged |= np.where(np.isfinite(active), active < active_frac_min, False)

    return flagged
