"""Indicator and expression controls for between-population calcium comparisons.

Two populations imaged with different viral constructs can differ in event
shape because of indicator expression, calcium buffering or clearance rather
than firing. These functions measure event shape on events least affected by
summation (isolated, small events) and the baseline fluorescence of each ROI,
so that kinetic differences can be checked against expression differences.

The isolated small-event decay approximates the single-event response of the
indicator in each cell: for GCaMP7f the in vivo half-decay after one action
potential is about 0.27 s and after 10 action potentials about 0.5 s
(Dana et al. 2019. "High-performance calcium sensors for imaging activity in
neuronal populations and microcompartments." Nature Methods 16:649-657.
doi:10.1038/s41592-019-0435-6). Decay times well above that in isolated small
events indicate either sustained firing or slower calcium clearance.

All functions are pure numpy; no I/O.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _event_peak(trace: np.ndarray, onset: int, offset: int) -> int:
    """Frame index of the maximum of *trace* within [onset, offset]."""
    stop = max(onset + 1, min(offset + 1, trace.size))
    return int(onset + np.argmax(trace[onset:stop]))


def isolated_event_mask(
    onsets: npt.ArrayLike,
    offsets: npt.ArrayLike,
    n_frames: int,
    fps: float,
    isolation_s: float = 10.0,
) -> np.ndarray:
    """Boolean flag per event: no other event within *isolation_s* on either side.

    The gap is measured from the previous event's offset to this onset and
    from this offset to the next onset. Events touching the recording edges
    within the isolation window are not isolated.
    """
    on = np.asarray(onsets, dtype=np.int64)
    off = np.asarray(offsets, dtype=np.int64)
    n = on.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    gap = int(round(isolation_s * fps))
    out = np.ones(n, dtype=bool)
    for i in range(n):
        prev_end = off[i - 1] if i > 0 else -(10**12)
        next_start = on[i + 1] if i < n - 1 else 10**12
        if on[i] - prev_end < gap or next_start - off[i] < gap:
            out[i] = False
        if on[i] < gap or off[i] > n_frames - 1 - gap:
            out[i] = False
    return out


def event_decay_time(
    trace: npt.ArrayLike,
    onset: int,
    offset: int,
    fps: float,
    max_window_s: float = 10.0,
) -> float:
    """Time from the event peak until the trace first falls below peak / e.

    Baseline is taken as 0 dF/F (dF/F is already baseline-normalised). Returns
    NaN when the peak is not positive or the trace does not fall below
    peak / e within *max_window_s*.
    """
    x = np.asarray(trace, dtype=np.float64)
    peak_i = _event_peak(x, int(onset), int(offset))
    peak = x[peak_i]
    if not np.isfinite(peak) or peak <= 0:
        return float("nan")
    stop = min(x.size, peak_i + int(round(max_window_s * fps)) + 1)
    after = x[peak_i + 1 : stop]
    below = np.flatnonzero(after < peak / np.e)
    if below.size == 0:
        return float("nan")
    return float((below[0] + 1) / fps)


def event_fwhm(trace: npt.ArrayLike, onset: int, offset: int, fps: float) -> float:
    """Full width at half maximum of an event, in seconds.

    Searches outward from the peak within a window of the event plus its own
    length on either side. NaN when the peak is not positive or either
    half-maximum crossing is not found.
    """
    x = np.asarray(trace, dtype=np.float64)
    onset, offset = int(onset), int(offset)
    peak_i = _event_peak(x, onset, offset)
    peak = x[peak_i]
    if not np.isfinite(peak) or peak <= 0:
        return float("nan")
    span = max(1, offset - onset)
    lo = max(0, onset - span)
    hi = min(x.size, offset + span + 1)
    half = peak / 2.0
    left = np.flatnonzero(x[lo:peak_i] < half)
    right = np.flatnonzero(x[peak_i + 1 : hi] < half)
    if left.size == 0 or right.size == 0:
        return float("nan")
    left_i = lo + left[-1]
    right_i = peak_i + 1 + right[0]
    return float((right_i - left_i) / fps)


def isolated_small_event_kinetics(
    trace: npt.ArrayLike,
    onsets: npt.ArrayLike,
    offsets: npt.ArrayLike,
    amplitudes: npt.ArrayLike,
    fps: float,
    isolation_s: float = 10.0,
    small_quantile: float = 0.5,
    max_window_s: float = 10.0,
    min_events: int = 3,
) -> dict[str, float]:
    """Per-cell kinetics of isolated, small calcium events.

    Small means amplitude at or below the cell's *small_quantile* amplitude.
    Returns the median decay time (peak to peak/e), median FWHM, median
    amplitude, and the number of qualifying events. Medians are NaN when
    fewer than *min_events* events qualify.
    """
    x = np.asarray(trace, dtype=np.float64)
    on = np.asarray(onsets, dtype=np.int64)
    off = np.asarray(offsets, dtype=np.int64)
    amp = np.asarray(amplitudes, dtype=np.float64)
    nan = float("nan")
    result = {
        "iso_n_events": 0.0,
        "iso_decay_s": nan,
        "iso_fwhm_s": nan,
        "iso_amplitude": nan,
    }
    if on.size == 0:
        return result
    iso = isolated_event_mask(on, off, x.size, fps, isolation_s)
    small = amp <= np.nanquantile(amp, small_quantile)
    keep = np.flatnonzero(iso & small)
    result["iso_n_events"] = float(keep.size)
    if keep.size < min_events:
        return result
    decays = np.array([event_decay_time(x, on[i], off[i], fps, max_window_s) for i in keep])
    widths = np.array([event_fwhm(x, on[i], off[i], fps) for i in keep])
    result["iso_decay_s"] = float(np.nanmedian(decays)) if np.isfinite(decays).any() else nan
    result["iso_fwhm_s"] = float(np.nanmedian(widths)) if np.isfinite(widths).any() else nan
    result["iso_amplitude"] = float(np.median(amp[keep]))
    return result


def baseline_fluorescence(f_raw: npt.ArrayLike, percentile: float = 10.0) -> float:
    """Low-percentile raw fluorescence of a trace (an expression-level proxy).

    Uses a low percentile of F rather than the mean so that activity does not
    raise the estimate. NaN for an empty or all-NaN trace.
    """
    f = np.asarray(f_raw, dtype=np.float64)
    f = f[np.isfinite(f)]
    if f.size == 0:
        return float("nan")
    return float(np.percentile(f, percentile))
