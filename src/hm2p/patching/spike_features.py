"""Spike waveform feature extraction via eFEL.

Replaces the PANDORA toolbox (MATLAB) for extracting action potential parameters
from patch-clamp recordings.

Citations
---------
eFEL:
    Van Geit et al. 2016. "BluePyOpt: Leveraging open source software and cloud
    computing for neuroscience optimization problems." Frontiers in Neuroinformatics.
    doi:10.3389/fninf.2016.00017
    GitHub: https://github.com/BlueBrain/eFEL

PANDORA (original MATLAB toolbox replaced by this module):
    Gunay et al. 2009. "Channel density distributions explain spiking variability
    in the globus pallidus: a data-driven approach." J Neurosci.
    doi:10.1523/JNEUROSCI.2929-09.2009
    GitHub: https://github.com/cengique/pandora-matlab

Spike detection (Berg 2006) used in the original MATLAB pipeline:
    Berg, R.W. 2006. spike_times.m — threshold-crossing spike detection.
"""

from __future__ import annotations

import numpy as np


def extract_spike_features(
    trace: np.ndarray,
    time: np.ndarray,
    stim_start: float,
    stim_end: float,
    spike_index: int = 1,
    threshold: float = -20.0,
) -> dict | None:
    """Extract spike features from a voltage trace using eFEL.

    Parameters
    ----------
    trace : np.ndarray
        Membrane voltage trace in mV, shape ``(N,)``.
    time : np.ndarray
        Time vector in ms, shape ``(N,)``, matching ``trace``.
    stim_start : float
        Stimulus onset time in ms.
    stim_end : float
        Stimulus offset time in ms.
    spike_index : int, optional
        0-based index of the spike to analyse. Default is 1 (the 2nd spike),
        matching the MATLAB convention of analysing the 2nd spike at rheobase.
        Falls back to the first spike (index 0) if ``spike_index`` exceeds the
        number of detected spikes.
    threshold : float, optional
        Spike detection threshold in mV. Default -20.0 mV.

    Returns
    -------
    dict or None
        Dictionary with keys: ``min_vm``, ``peak_vm``, ``max_vm_slope``,
        ``half_vm``, ``amplitude``, ``max_ahp``, ``half_width``,
        ``spike_time``.  Returns ``None`` if no spikes are found.

    Raises
    ------
    ImportError
        If eFEL is not installed.
    """
    try:
        import efel
    except ImportError:
        raise ImportError(
            "eFEL is required for spike feature extraction. "
            "Install it with: pip install efel"
        )

    efel.reset()
    efel.setDoubleSetting("Threshold", threshold)

    trace_dict = {
        "T": time.astype(float),
        "V": trace.astype(float),
        "stim_start": [float(stim_start)],
        "stim_end": [float(stim_end)],
    }

    feature_list = [
        "minimum_voltage",
        "peak_voltage",
        "AP_rise_rate",
        "AP_begin_voltage",
        "AP_amplitude",
        "min_AHP_values",
        "AP_duration_half_width",
        "peak_time",
    ]

    results = efel.getFeatureValues([trace_dict], feature_list)[0]

    # Check if any spikes were detected
    if results["peak_voltage"] is None or len(results["peak_voltage"]) == 0:
        return None

    n_spikes = len(results["peak_voltage"])

    # Select spike index, falling back to first spike if out of range
    idx = spike_index if spike_index < n_spikes else 0

    # Derive half_vm: AP_begin_voltage + AP_amplitude / 2
    ap_begin = (
        results["AP_begin_voltage"][idx]
        if results["AP_begin_voltage"] is not None and len(results["AP_begin_voltage"]) > idx
        else np.nan
    )
    ap_amp = (
        results["AP_amplitude"][idx]
        if results["AP_amplitude"] is not None and len(results["AP_amplitude"]) > idx
        else np.nan
    )
    half_vm = ap_begin + ap_amp / 2.0

    def _safe_get(arr: np.ndarray | None, i: int) -> float:
        """Safely index into an eFEL result array."""
        if arr is None or len(arr) <= i:
            return np.nan
        return float(arr[i])

    return {
        "min_vm": _safe_get(results["minimum_voltage"], 0),  # single value per trace
        "peak_vm": _safe_get(results["peak_voltage"], idx),
        "max_vm_slope": _safe_get(results["AP_rise_rate"], idx),
        "half_vm": float(half_vm),
        "amplitude": _safe_get(results["AP_amplitude"], idx),
        "max_ahp": _safe_get(results["min_AHP_values"], idx),
        "half_width": _safe_get(results["AP_duration_half_width"], idx),
        "spike_time": _safe_get(results["peak_time"], idx),
    }


def extract_waveform(
    trace: np.ndarray,
    spike_time_idx: int,
    sr: int = 20_000,
    pre_ms: float = 7.0,
    post_ms: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a voltage waveform window around a spike.

    Mirrors the window extraction in ``sumSpikeWaveforms.m`` (lines 40-50):
    pre_ms = 7 ms before the spike peak, post_ms = 20 ms after.

    Parameters
    ----------
    trace : np.ndarray
        Full voltage trace in mV, shape ``(N,)``.
    spike_time_idx : int
        Sample index of the spike peak within ``trace``.
    sr : int, optional
        Sampling rate in Hz. Default 20000.
    pre_ms : float, optional
        Time before spike peak to include, in ms. Default 7.0.
    post_ms : float, optional
        Time after spike peak to include, in ms. Default 20.0.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(time_ms, voltage)`` where ``time_ms`` is relative to the spike peak
        (negative before, positive after) and ``voltage`` is the corresponding
        membrane potential in mV.

    Raises
    ------
    ValueError
        If the requested window exceeds the trace boundaries.
    """
    sr_khz = sr / 1000.0
    pre_samples = int(round(pre_ms * sr_khz))
    post_samples = int(round(post_ms * sr_khz))

    start_idx = spike_time_idx - pre_samples
    end_idx = spike_time_idx + post_samples

    if start_idx < 0:
        raise ValueError(
            f"Window starts before trace: spike at index {spike_time_idx}, "
            f"pre_samples={pre_samples} would start at index {start_idx}."
        )
    if end_idx >= len(trace):
        raise ValueError(
            f"Window extends beyond trace: spike at index {spike_time_idx}, "
            f"post_samples={post_samples} would end at index {end_idx}, "
            f"but trace length is {len(trace)}."
        )

    voltage = trace[start_idx : end_idx + 1]
    time_ms = np.linspace(-pre_ms, post_ms, len(voltage))

    return time_ms, voltage
