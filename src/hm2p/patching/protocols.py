"""Protocol-specific extraction from WaveSurfer patch-clamp recordings.

Implements extraction for five standard current-clamp protocols:
- **IV** (current--voltage relationship): stepped current injections
- **Rheobase**: fine-stepped currents to find minimum spiking threshold
- **Passive**: small hyperpolarising steps for input resistance and time constant
- **Sag**: large hyperpolarising steps for Ih sag ratio
- **Ramp**: ramping current injection

Each extractor reads a ``ws_data`` dict (as returned by ``hm2p.patching.io.load_wavesurfer``)
containing sweep traces and an HDF5 header with stimulus parameters.

MATLAB source reference: ``old-penk-patching/ephys/ephys_intr.m``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from hm2p.patching.ephys import (
    build_stim_vector,
    compute_rmp,
    count_spikes,
    deconcat_traces,
    lowpass_filter,
)

logger = logging.getLogger(__name__)

# Default filter parameters (from pc_lowpassfilt.m and ephys_intr.m)
_FILTER_CUTOFF = 1000  # Hz
_FILTER_ORDER = 4


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IVResult:
    """Results from IV protocol extraction."""

    traces: np.ndarray  # (samples_per_pulse, n_pulses)
    stim_vec: np.ndarray  # (n_pulses,) in nA or pA
    spike_counts: np.ndarray  # (n_pulses,)
    rmp: float  # resting membrane potential in mV
    filtered: np.ndarray  # full filtered trace before deconcatenation


@dataclass
class RheobaseResult:
    """Results from rheobase protocol extraction."""

    traces: np.ndarray  # (samples_per_pulse, n_pulses)
    stim_vec: np.ndarray  # (n_pulses,)
    spike_counts: np.ndarray  # (n_pulses,)
    rheo_current: float  # rheobase current amplitude
    rmp: float
    filtered: np.ndarray


@dataclass
class PassiveResult:
    """Results from passive protocol extraction."""

    traces: np.ndarray  # (samples_per_pulse, n_pulses)
    stim_vec: np.ndarray  # (n_pulses,)
    rin: float  # input resistance in MOhm
    tau: np.ndarray  # membrane time constants per sweep (ms)
    capacitance: float  # tau_mean / Rin in pF
    rmp: float


@dataclass
class SagResult:
    """Results from sag protocol extraction."""

    traces: np.ndarray
    stim_vec: np.ndarray
    sag_ratio: np.ndarray  # sag ratio per sweep (%)
    rmp: float


@dataclass
class RampResult:
    """Results from ramp protocol extraction."""

    traces: np.ndarray
    filtered: np.ndarray


# ---------------------------------------------------------------------------
# Stimulus parameter parsing
# ---------------------------------------------------------------------------


def _parse_stim_params(header: dict, element_key: str, sr: float) -> dict:
    """Extract stimulus parameters from the WaveSurfer HDF5 header.

    The MATLAB code reads from paths like::

        header.StimulusLibrary.Stimuli.<element_key>.Delegate.<param>

    Parameters
    ----------
    header : dict
        The ``ws_data["header"]`` dictionary from a loaded WaveSurfer file.
    element_key : str
        Stimulus element name, e.g. ``"element1"`` for IV.
    sr : float
        Sampling rate in Hz.

    Returns
    -------
    dict
        Keys: ``delay``, ``delay_bp``, ``pulse_dur``, ``n_pulses``,
        ``first_amp``, ``amp_change`` (all in samples where applicable,
        amplitudes in native units).
    """
    stim = header["StimulusLibrary"]["Stimuli"][element_key]["Delegate"]

    def _num(val: Any) -> float:
        """Convert a value that may be bytes, str, or numeric to float."""
        if isinstance(val, (bytes, np.bytes_)):
            val = val.decode("utf-8")
        if isinstance(val, str):
            return float(val)
        return float(val)

    delay = _num(stim["Delay"]) * sr
    delay_bp = _num(stim["DelayBetweenPulses"]) * sr
    n_pulses = int(_num(stim["PulseCount"]))
    pulse_dur = _num(stim["PulseDuration"]) * sr
    first_amp = _num(stim["FirstPulseAmplitude"])
    amp_change = _num(stim["AmplitudeChangePerPulse"])

    return {
        "delay": int(round(delay)),
        "delay_bp": int(round(delay_bp)),
        "pulse_dur": int(round(pulse_dur)),
        "n_pulses": n_pulses,
        "first_amp": first_amp,
        "amp_change": amp_change,
    }


# ---------------------------------------------------------------------------
# Helper: filter + deconcat common pattern
# ---------------------------------------------------------------------------


def _filter_and_deconcat(
    raw_trace: np.ndarray,
    sr: float,
    stim_params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter a raw concatenated trace and split into per-pulse sweeps.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(traces, stim_vec, filtered)`` where ``traces`` has shape
        ``(samples_per_pulse, n_pulses)``.
    """
    filtered = lowpass_filter(
        raw_trace,
        order=_FILTER_ORDER,
        cutoff=_FILTER_CUTOFF,
        fs=sr,
    )

    stim_vec = build_stim_vector(
        stim_params["first_amp"],
        stim_params["amp_change"],
        stim_params["n_pulses"],
    )

    traces = deconcat_traces(
        filtered,
        delay=stim_params["delay"],
        delay_bp=stim_params["delay_bp"],
        pulse_dur=stim_params["pulse_dur"],
        n_pulses=stim_params["n_pulses"],
        sr=sr,
    )

    return traces, stim_vec, filtered


# ---------------------------------------------------------------------------
# Protocol extractors
# ---------------------------------------------------------------------------


def extract_iv(ws_data: dict) -> IVResult:
    """Extract IV (current-voltage) protocol data.

    Reads stimulus parameters from header ``element1``. Deconcatenates the
    filtered trace into per-pulse sweeps, counts spikes per sweep, and
    computes the resting membrane potential from the pre-stimulus baseline.

    Parameters
    ----------
    ws_data : dict
        Loaded WaveSurfer data with keys ``"traces"`` (raw voltage, 1-D) and
        ``"header"`` (nested dict of HDF5 header).

    Returns
    -------
    IVResult
    """
    sr = float(ws_data["header"]["StimulationSampleRate"])
    stim_params = _parse_stim_params(ws_data["header"], "element1", sr)
    traces, stim_vec, filtered = _filter_and_deconcat(
        ws_data["traces"], sr, stim_params
    )

    spike_counts = count_spikes(traces, threshold_factor=0.5)
    rmp = compute_rmp(traces, baseline_samples=stim_params["delay"] // 2)

    return IVResult(
        traces=traces,
        stim_vec=stim_vec,
        spike_counts=spike_counts,
        rmp=rmp,
        filtered=filtered,
    )


def extract_rheobase(ws_data: dict) -> RheobaseResult:
    """Extract rheobase protocol data.

    Reads stimulus parameters from header ``element9``. Finds the first sweep
    containing at least one spike — the corresponding current is the rheobase.

    Parameters
    ----------
    ws_data : dict
        Loaded WaveSurfer data.

    Returns
    -------
    RheobaseResult

    Raises
    ------
    ValueError
        If no spikes are found in any sweep.
    """
    sr = float(ws_data["header"]["StimulationSampleRate"])
    stim_params = _parse_stim_params(ws_data["header"], "element9", sr)
    traces, stim_vec, filtered = _filter_and_deconcat(
        ws_data["traces"], sr, stim_params
    )

    spike_counts = count_spikes(traces, threshold_factor=0.5)

    # Find first sweep with at least one spike
    spike_idx = np.where(spike_counts > 0)[0]
    if len(spike_idx) == 0:
        raise ValueError("No spikes found in any rheobase sweep.")
    rheo_current = float(stim_vec[spike_idx[0]])

    rmp = compute_rmp(traces, baseline_samples=stim_params["delay"] // 2)

    return RheobaseResult(
        traces=traces,
        stim_vec=stim_vec,
        spike_counts=spike_counts,
        rheo_current=rheo_current,
        rmp=rmp,
        filtered=filtered,
    )


def extract_passive(ws_data: dict) -> PassiveResult:
    """Extract passive membrane properties from small hyperpolarising steps.

    Reads stimulus parameters from header ``element6``. Computes:
    - Input resistance (Rin) from slope of steady-state dV vs injected current.
    - Membrane time constant (tau) from exponential fit to the voltage onset.
    - Membrane capacitance as tau_mean / Rin.

    Parameters
    ----------
    ws_data : dict
        Loaded WaveSurfer data.

    Returns
    -------
    PassiveResult
    """
    sr = float(ws_data["header"]["StimulationSampleRate"])
    sr_khz = sr / 1000.0
    stim_params = _parse_stim_params(ws_data["header"], "element6", sr)
    traces, stim_vec, filtered = _filter_and_deconcat(
        ws_data["traces"], sr, stim_params
    )

    delay = stim_params["delay"]

    # Resting voltage: baseline before stimulus
    v_rest = np.mean(traces[:delay, :], axis=0)

    # Steady-state voltage: within the pulse (MATLAB: 2*delay:3*delay)
    ss_start = 2 * delay
    ss_end = min(3 * delay, traces.shape[0])
    v_ss = np.mean(traces[ss_start:ss_end, :], axis=0)

    dv = v_ss - v_rest

    # Linear fit: dV = Rin * I (Rin in MOhm when I in nA, V in mV)
    # polyfit returns [slope, intercept]
    coeffs = np.polyfit(stim_vec, dv, 1)
    rin = coeffs[0] * 1000.0  # convert to MOhm (MATLAB: P(1)*1000)

    rmp = float(np.round(np.mean(v_rest), 1))

    # Time constant: exponential fit to onset transient
    # MATLAB fits traces from 0.5*delay to delay for the first 5 sweeps
    tau_start = delay // 2
    tau_end = delay
    n_tau_sweeps = min(5, traces.shape[1])

    def _exp_decay(x: np.ndarray, a: float, b: float, tau: float) -> np.ndarray:
        return a - b * np.exp(-x / tau)

    tau_values = np.full(n_tau_sweeps, np.nan)
    for j in range(n_tau_sweeps):
        segment = traces[tau_start:tau_end, j]
        x = np.arange(len(segment), dtype=float)
        try:
            popt, _ = curve_fit(
                _exp_decay,
                x,
                segment,
                p0=[segment[0], segment[-1], 0.3 * sr_khz],
                maxfev=10000,
            )
            tau_values[j] = popt[2] / sr_khz  # convert samples to ms
        except (RuntimeError, ValueError):
            tau_values[j] = np.nan

    tau_mean = float(np.nanmean(tau_values))
    # Capacitance: C = tau / Rin (in pF when tau in ms and Rin in MOhm)
    capacitance = tau_mean / rin * 1000.0 if rin != 0 else np.nan

    return PassiveResult(
        traces=traces,
        stim_vec=stim_vec,
        rin=float(np.round(rin, 0)),
        tau=tau_values,
        capacitance=capacitance,
        rmp=rmp,
    )


def extract_sag(ws_data: dict) -> SagResult:
    """Extract sag ratio from large hyperpolarising current steps.

    Reads stimulus parameters from header ``element5``.

    Sag ratio: ``100 * (Vss - Vmin) / (Vrest - Vmin)`` per sweep,
    where Vrest is the pre-stimulus baseline, Vmin is the minimum voltage
    during the step, and Vss is the steady-state voltage during the step.

    Parameters
    ----------
    ws_data : dict
        Loaded WaveSurfer data.

    Returns
    -------
    SagResult
    """
    sr = float(ws_data["header"]["StimulationSampleRate"])
    stim_params = _parse_stim_params(ws_data["header"], "element5", sr)

    # For sag, stim_vec is constant (all pulses same amplitude)
    n_pulses = stim_params["n_pulses"]
    stim_vec = np.ones(n_pulses) * stim_params["first_amp"]

    # Override amp_change to 0 for deconcat (sag uses constant amplitude)
    stim_params_for_deconcat = stim_params.copy()
    stim_params_for_deconcat["amp_change"] = 0.0

    filtered = lowpass_filter(
        ws_data["traces"],
        order=_FILTER_ORDER,
        cutoff=_FILTER_CUTOFF,
        fs=sr,
    )

    traces = deconcat_traces(
        filtered,
        delay=stim_params["delay"],
        delay_bp=stim_params["delay_bp"],
        pulse_dur=stim_params["pulse_dur"],
        n_pulses=n_pulses,
        sr=sr,
    )

    delay = stim_params["delay"]

    # Resting voltage: pre-stimulus baseline
    v_rest = np.mean(traces[: delay // 2, :], axis=0)

    # Steady-state: MATLAB uses 2*delay:4*delay
    ss_start = 2 * delay
    ss_end = min(4 * delay, traces.shape[0])
    v_ss = np.mean(traces[ss_start:ss_end, :], axis=0)

    # Minimum voltage during the step (MATLAB: 0.5*delay:delay)
    min_start = delay // 2
    min_end = delay
    v_min = np.min(traces[min_start:min_end, :], axis=0)

    # Sag ratio: 100 * (Vss - Vmin) / (Vrest - Vmin)
    denom = v_rest - v_min
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        sag_ratio = np.where(denom != 0, 100.0 * (v_ss - v_min) / denom, np.nan)

    rmp = float(np.round(np.mean(v_rest), 1))

    return SagResult(
        traces=traces,
        stim_vec=stim_vec,
        sag_ratio=sag_ratio,
        rmp=rmp,
    )


def extract_ramp(ws_data: dict) -> RampResult:
    """Extract ramp protocol data.

    Reads from header ``element3``. Simply filters the trace; no
    deconcatenation is needed for ramp protocols.

    Parameters
    ----------
    ws_data : dict
        Loaded WaveSurfer data.

    Returns
    -------
    RampResult
    """
    sr = float(ws_data["header"]["StimulationSampleRate"])

    filtered = lowpass_filter(
        ws_data["traces"],
        order=_FILTER_ORDER,
        cutoff=_FILTER_CUTOFF,
        fs=sr,
    )

    return RampResult(
        traces=ws_data["traces"],
        filtered=filtered,
    )


# ---------------------------------------------------------------------------
# Protocol identification and batch processing
# ---------------------------------------------------------------------------


def identify_protocol(filename: str) -> str:
    """Identify the protocol type from a WaveSurfer filename.

    The MATLAB code uses ``contains(stimuli_type, 'IV')``, etc.

    Parameters
    ----------
    filename : str
        Filename (not full path) of the WaveSurfer H5 file.

    Returns
    -------
    str
        One of ``"iv"``, ``"rheobase"``, ``"passive"``, ``"sag"``, ``"ramp"``,
        or ``"unknown"``.
    """
    name_lower = filename.lower()
    # Order matters: check more specific names first
    if "rheobase" in name_lower:
        return "rheobase"
    if "passive" in name_lower:
        return "passive"
    if "sag" in name_lower:
        return "sag"
    if "ramp" in name_lower:
        return "ramp"
    if "iv" in name_lower:
        return "iv"
    return "unknown"


def process_all_protocols(h5_dir: Path) -> dict:
    """Find all H5 files in a directory, identify protocols, and extract each.

    Parameters
    ----------
    h5_dir : Path
        Directory containing WaveSurfer ``.h5`` files.

    Returns
    -------
    dict
        Keys are protocol names (``"iv"``, ``"rheobase"``, etc.), values are
        the corresponding result dataclasses. Missing protocols are not included.
    """
    from hm2p.patching.io import load_wavesurfer

    h5_dir = Path(h5_dir)
    results: dict = {}

    extractors = {
        "iv": extract_iv,
        "rheobase": extract_rheobase,
        "passive": extract_passive,
        "sag": extract_sag,
        "ramp": extract_ramp,
    }

    for h5_file in sorted(h5_dir.glob("*.h5")):
        protocol = identify_protocol(h5_file.name)
        if protocol == "unknown":
            logger.warning("Unknown protocol for file: %s", h5_file.name)
            continue

        try:
            ws_data = load_wavesurfer(h5_file)
            results[protocol] = extractors[protocol](ws_data)
            logger.info("Extracted %s from %s", protocol, h5_file.name)
        except Exception:
            logger.exception("Failed to extract %s from %s", protocol, h5_file.name)

    return results
