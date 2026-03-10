"""Electrophysiology trace processing: filtering, deconcat, spike detection.

Reimplements the MATLAB functions ``pc_lowpassfilt.m``, inline
deconcatenation from ``ephys_intr.m``, and ``spike_times.m`` (Berg 2006).

References
----------
Berg, R. W. 2006. spike_times — action potential detection by threshold
crossing. www.berg-lab.net

Chen, C. & Bhatt, D. H. & Bhatt, D. & Bhatt, D. & Bhatt, D. 2000.
Filter parameters: 4-pole Butterworth at 1 kHz for current traces,
500 Hz for minis.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from .config import FILTER_CUTOFF, FILTER_ORDER, SAMPLE_RATE


def lowpass_filter(
    signal: np.ndarray,
    order: int = FILTER_ORDER,
    cutoff: float = FILTER_CUTOFF,
    fs: float = SAMPLE_RATE,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.
    order : int
        Filter order (number of poles).
    cutoff : float
        Cutoff frequency in Hz.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        Filtered signal, same shape as *signal*.

    Raises
    ------
    ValueError
        If *signal* is empty or *cutoff* >= Nyquist frequency.
    """
    if signal.size == 0:
        return signal.copy()

    nyq = fs / 2.0
    if cutoff >= nyq:
        raise ValueError(
            f"Cutoff ({cutoff} Hz) must be below the Nyquist frequency ({nyq} Hz)."
        )

    b, a = butter(order, cutoff / nyq, btype="low")
    # padlen must be < signal length; filtfilt default is 3 * max(len(a), len(b))
    padlen = min(3 * max(len(a), len(b)), signal.size - 1)
    return filtfilt(b, a, signal, padlen=padlen)


def deconcat_traces(
    filtered: np.ndarray,
    delay: int,
    delay_bp: int,
    pulse_dur: int,
    n_pulses: int,
    sr: float,
) -> np.ndarray:
    """Slice a concatenated trace into per-sweep columns.

    Mirrors the MATLAB deconcatenation logic in ``ephys_intr.m``::

        startpoint = delay/2 : pulse_dur + delay_bp : trace_length
        endpoint   = startpoint(2) : pulse_dur + delay_bp : trace_length
        traces_deconc(:, t) = filtered(startpoint(t)+1 : endpoint(t))

    Parameters
    ----------
    filtered : np.ndarray
        1-D filtered concatenated trace.
    delay : int
        Pre-stimulus delay in *samples*.
    delay_bp : int
        Delay between pulses in *samples*.
    pulse_dur : int
        Pulse duration in *samples*.
    n_pulses : int
        Number of stimulus pulses (sweeps).
    sr : float
        Sampling rate in Hz (unused here but kept for API consistency).

    Returns
    -------
    np.ndarray
        2-D array of shape ``(samples_per_pulse, n_pulses)`` where each
        column is one deconcat sweep.
    """
    step = pulse_dur + delay_bp
    half_delay = delay // 2

    # Build start indices (0-based, matching MATLAB startpoint(t)+1 → Python [start:end])
    startpoints = np.arange(half_delay, len(filtered), step)
    # Endpoints start from the second startpoint
    endpoints = np.arange(half_delay + step, len(filtered) + 1, step)

    n_available = min(len(startpoints), len(endpoints), n_pulses)

    # Determine samples per sweep from first pair
    samples_per_sweep = endpoints[0] - startpoints[0]

    traces = np.empty((samples_per_sweep, n_available), dtype=filtered.dtype)
    for t in range(n_available):
        traces[:, t] = filtered[startpoints[t] : startpoints[t] + samples_per_sweep]

    return traces


def build_stim_vector(
    first_amp: float, amp_change: float, n_pulses: int
) -> np.ndarray:
    """Build the stimulus amplitude vector for a step protocol.

    Matches the MATLAB expression::

        stimvec = [firstamp : ampch : (pulsenr * ampch + (firstamp - ampch))]

    Parameters
    ----------
    first_amp : float
        Amplitude of the first pulse (pA or nA).
    amp_change : float
        Amplitude increment per pulse.
    n_pulses : int
        Number of pulses.

    Returns
    -------
    np.ndarray
        1-D array of length *n_pulses* with stimulus amplitudes.
    """
    return np.arange(n_pulses) * amp_change + first_amp


def detect_spikes(
    trace: np.ndarray, threshold_factor: float = 0.5
) -> np.ndarray:
    """Detect spike times by threshold crossing of the membrane potential.

    Reimplements ``spike_times.m`` (Berg 2006). The threshold is set at
    ``threshold_factor * max(trace)``. Consecutive above-threshold samples
    are grouped; the index of the peak within each group is returned.

    Parameters
    ----------
    trace : np.ndarray
        1-D membrane voltage trace.
    threshold_factor : float
        Fraction of the trace maximum used as threshold (0–1).

    Returns
    -------
    np.ndarray
        1-D int array of spike peak indices. Empty if no spikes detected.
    """
    if trace.size == 0:
        return np.array([], dtype=np.intp)

    peak_val = np.max(trace)
    threshold = threshold_factor * peak_val

    # If max is non-positive, no spikes
    if peak_val <= 0:
        return np.array([], dtype=np.intp)

    above = trace > threshold
    if not np.any(above):
        return np.array([], dtype=np.intp)

    above_idx = np.where(above)[0]

    # Group consecutive indices
    diffs = np.diff(above_idx)
    group_breaks = np.where(diffs > 1)[0]

    # Build groups: list of (start_idx, end_idx) into above_idx
    group_starts = np.concatenate([[0], group_breaks + 1])
    group_ends = np.concatenate([group_breaks + 1, [len(above_idx)]])

    spike_peaks = np.empty(len(group_starts), dtype=np.intp)
    for i, (gs, ge) in enumerate(zip(group_starts, group_ends)):
        group_indices = above_idx[gs:ge]
        # Find peak within this group
        local_peak = group_indices[np.argmax(trace[group_indices])]
        spike_peaks[i] = local_peak

    return spike_peaks


def count_spikes(
    traces: np.ndarray, threshold_factor: float = 0.5
) -> np.ndarray:
    """Count spikes in each column of a 2-D traces array.

    Parameters
    ----------
    traces : np.ndarray
        2-D array ``(n_samples, n_sweeps)``.
    threshold_factor : float
        Threshold factor passed to :func:`detect_spikes`.

    Returns
    -------
    np.ndarray
        1-D int array of length ``n_sweeps`` with spike counts.
    """
    if traces.ndim == 1:
        traces = traces[:, np.newaxis]

    n_sweeps = traces.shape[1]
    counts = np.empty(n_sweeps, dtype=np.intp)
    for t in range(n_sweeps):
        counts[t] = len(detect_spikes(traces[:, t], threshold_factor))
    return counts


def compute_rmp(traces: np.ndarray, baseline_samples: int) -> float:
    """Compute resting membrane potential from pre-stimulus baseline.

    Matches the MATLAB logic: ``mean(traces(1:delay/2, :))``.

    Parameters
    ----------
    traces : np.ndarray
        2-D array ``(n_samples, n_sweeps)`` of deconcatenated traces.
    baseline_samples : int
        Number of samples from the start of each sweep to average over
        (typically ``delay // 2``).

    Returns
    -------
    float
        Mean baseline voltage (mV), rounded to 1 decimal place.
    """
    if traces.ndim == 1:
        traces = traces[:, np.newaxis]
    baseline = traces[:baseline_samples, :]
    return round(float(np.mean(baseline)), 1)


def compute_steady_state(
    trace: np.ndarray,
    baseline_start: int,
    baseline_end: int,
    ss_start: int,
    ss_end: int,
) -> float:
    """Compute steady-state voltage deflection for a single sweep.

    Reimplements ``getSS.m``: ``Vss - Vrest`` where *Vrest* is the mean of
    the baseline window and *Vss* is the mean of the steady-state window.

    Parameters
    ----------
    trace : np.ndarray
        1-D or 2-D array. If 2-D, computes per-column means then the
        overall difference.
    baseline_start : int
        Start sample of baseline window (inclusive).
    baseline_end : int
        End sample of baseline window (exclusive).
    ss_start : int
        Start sample of steady-state window (inclusive).
    ss_end : int
        End sample of steady-state window (exclusive).

    Returns
    -------
    float
        Steady-state voltage minus baseline voltage.
    """
    if trace.ndim == 1:
        v_rest = float(np.mean(trace[baseline_start:baseline_end]))
        v_ss = float(np.mean(trace[ss_start:ss_end]))
    else:
        v_rest = float(np.mean(trace[baseline_start:baseline_end, :]))
        v_ss = float(np.mean(trace[ss_start:ss_end, :]))
    return v_ss - v_rest
