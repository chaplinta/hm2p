"""Tests for hm2p.patching.protocols.

Uses synthetic WaveSurfer-like data structures — never real data files.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from hm2p.patching.protocols import (
    IVResult,
    PassiveResult,
    RampResult,
    RheobaseResult,
    SagResult,
    _parse_stim_params,
    extract_iv,
    extract_passive,
    extract_ramp,
    extract_rheobase,
    extract_sag,
    identify_protocol,
)


# ---------------------------------------------------------------------------
# Helpers: build synthetic WaveSurfer-like data
# ---------------------------------------------------------------------------

_SR = 20_000  # 20 kHz


def _make_stim_element(
    delay: float = 0.5,
    delay_bp: float = 0.0,
    pulse_dur: float = 0.5,
    n_pulses: int = 5,
    first_amp: float = -0.2,
    amp_change: float = 0.05,
) -> dict:
    """Create a stimulus element dict mirroring HDF5 header structure.

    All time values in seconds (as stored in WaveSurfer H5).
    """
    return {
        "Delegate": {
            "Delay": str(delay),
            "DelayBetweenPulses": str(delay_bp),
            "PulseDuration": str(pulse_dur),
            "PulseCount": str(int(n_pulses)),
            "FirstPulseAmplitude": str(first_amp),
            "AmplitudeChangePerPulse": str(amp_change),
        }
    }


def _make_ws_data(
    element_key: str = "element1",
    delay: float = 0.5,
    delay_bp: float = 0.0,
    pulse_dur: float = 0.5,
    n_pulses: int = 5,
    first_amp: float = -0.2,
    amp_change: float = 0.05,
    rmp: float = -70.0,
    add_spikes: bool = False,
    spike_in_sweep: int | None = None,
) -> dict:
    """Build a synthetic ws_data dict matching WaveSurfer loaded structure.

    Creates a concatenated trace of ``n_pulses`` sweeps, each with a flat
    baseline at ``rmp`` (with small voltage steps proportional to current).

    If ``add_spikes`` is True, adds a spike-like transient (peak at +20 mV)
    in sweeps at or above ``spike_in_sweep``.
    """
    delay_samples = int(delay * _SR)
    delay_bp_samples = int(delay_bp * _SR)
    pulse_dur_samples = int(pulse_dur * _SR)

    samples_per_pulse = pulse_dur_samples + delay_bp_samples
    total_samples = delay_samples + n_pulses * samples_per_pulse

    # Build concatenated trace
    trace = np.full(total_samples, rmp)

    if amp_change != 0:
        stim_vec = np.arange(first_amp, first_amp + n_pulses * amp_change, amp_change)
        if len(stim_vec) > n_pulses:
            stim_vec = stim_vec[:n_pulses]
    else:
        stim_vec = np.ones(n_pulses) * first_amp

    # Add voltage deflections proportional to current (Rin ~ 200 MOhm, realistic)
    # Only apply deflection during the stimulus portion (after baseline),
    # matching real electrophysiology where baseline is pre-stimulus.
    rin_mohm = 200.0
    baseline_end_in_sweep = delay_samples  # baseline occupies first delay samples of each sweep
    for i in range(n_pulses):
        sweep_start = delay_samples // 2 + i * samples_per_pulse
        # Stimulus starts after baseline within the sweep
        stim_start_abs = sweep_start + baseline_end_in_sweep // 2
        stim_end_abs = sweep_start + pulse_dur_samples
        if stim_end_abs <= total_samples:
            dv = stim_vec[i] * rin_mohm  # dV in mV
            trace[stim_start_abs:stim_end_abs] += dv

    if add_spikes and spike_in_sweep is not None:
        for i in range(spike_in_sweep, n_pulses):
            start = delay_samples // 2 + i * samples_per_pulse
            spike_pos = start + pulse_dur_samples // 3
            if spike_pos + 60 < total_samples:
                # Create a realistic-width spike (~2ms total) that survives
                # 1kHz low-pass filtering. At 20kHz, 2ms = 40 samples.
                rise_n = 10  # 0.5ms rise
                fall_n = 15  # 0.75ms fall
                ahp_n = 20   # 1ms AHP recovery
                # Rising phase
                for k in range(rise_n):
                    frac = k / rise_n
                    trace[spike_pos - rise_n + k] = rmp + frac * (40.0 - rmp)
                # Peak (hold for a few samples)
                for k in range(3):
                    trace[spike_pos + k] = 40.0
                # Falling phase
                for k in range(fall_n):
                    frac = k / fall_n
                    trace[spike_pos + 3 + k] = 40.0 + frac * (-80.0 - 40.0)
                # AHP recovery
                for k in range(ahp_n):
                    frac = k / ahp_n
                    trace[spike_pos + 3 + fall_n + k] = -80.0 + frac * (rmp + 80.0)

    header = {
        "StimulationSampleRate": float(_SR),
        "StimulusLibrary": {
            "Stimuli": {
                element_key: _make_stim_element(
                    delay=delay,
                    delay_bp=delay_bp,
                    pulse_dur=pulse_dur,
                    n_pulses=n_pulses,
                    first_amp=first_amp,
                    amp_change=amp_change,
                ),
            }
        },
    }

    return {"traces": trace, "header": header}


# ---------------------------------------------------------------------------
# Tests for _parse_stim_params
# ---------------------------------------------------------------------------


class TestParseStimParams:
    """Tests for _parse_stim_params."""

    def test_basic_parsing(self):
        header = {
            "StimulusLibrary": {
                "Stimuli": {
                    "element1": _make_stim_element(
                        delay=0.5, pulse_dur=0.5, n_pulses=19,
                        first_amp=-0.2, amp_change=0.025,
                    )
                }
            }
        }
        params = _parse_stim_params(header, "element1", sr=_SR)

        assert params["delay"] == int(0.5 * _SR)
        assert params["pulse_dur"] == int(0.5 * _SR)
        assert params["n_pulses"] == 19
        assert params["first_amp"] == pytest.approx(-0.2)
        assert params["amp_change"] == pytest.approx(0.025)

    def test_bytes_values(self):
        """WaveSurfer H5 may store values as byte strings."""
        header = {
            "StimulusLibrary": {
                "Stimuli": {
                    "element6": {
                        "Delegate": {
                            "Delay": b"0.25",
                            "DelayBetweenPulses": b"0.0",
                            "PulseDuration": b"0.5",
                            "PulseCount": b"7",
                            "FirstPulseAmplitude": b"-0.05",
                            "AmplitudeChangePerPulse": b"0.01",
                        }
                    }
                }
            }
        }
        params = _parse_stim_params(header, "element6", sr=_SR)
        assert params["n_pulses"] == 7
        assert params["first_amp"] == pytest.approx(-0.05)

    def test_numeric_values(self):
        """Values may already be numeric (not strings)."""
        header = {
            "StimulusLibrary": {
                "Stimuli": {
                    "element5": {
                        "Delegate": {
                            "Delay": 0.5,
                            "DelayBetweenPulses": 0.0,
                            "PulseDuration": 1.0,
                            "PulseCount": 3,
                            "FirstPulseAmplitude": -0.3,
                            "AmplitudeChangePerPulse": 0.0,
                        }
                    }
                }
            }
        }
        params = _parse_stim_params(header, "element5", sr=_SR)
        assert params["n_pulses"] == 3
        assert params["delay"] == int(0.5 * _SR)


# ---------------------------------------------------------------------------
# Tests for identify_protocol
# ---------------------------------------------------------------------------


class TestIdentifyProtocol:
    """Tests for identify_protocol."""

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("SW0001_IV_001.h5", "iv"),
            ("cell3_Rheobase_002.h5", "rheobase"),
            ("SW0005_Passive.h5", "passive"),
            ("experiment_Sag_test.h5", "sag"),
            ("SW0002_Ramp_001.h5", "ramp"),
            ("something_else.h5", "unknown"),
            # Case insensitive
            ("CELL1_iv_recording.h5", "iv"),
            ("myfile_RHEOBASE.h5", "rheobase"),
            ("data_PASSIVE_01.h5", "passive"),
        ],
    )
    def test_identification(self, filename: str, expected: str):
        assert identify_protocol(filename) == expected

    def test_rheobase_before_iv(self):
        """'Rheobase' contains 'base' not 'iv', but test that order is correct."""
        # A filename with both should match the more specific one first
        assert identify_protocol("IV_Rheobase.h5") == "rheobase"


# ---------------------------------------------------------------------------
# Tests for extract_iv
# ---------------------------------------------------------------------------


class TestExtractIV:
    """Tests for extract_iv."""

    def test_basic_iv(self):
        ws = _make_ws_data(
            element_key="element1",
            delay=0.5,
            pulse_dur=0.5,
            n_pulses=5,
            first_amp=-0.2,
            amp_change=0.1,
            rmp=-70.0,
        )
        result = extract_iv(ws)

        assert isinstance(result, IVResult)
        assert result.traces.shape[1] == 5
        assert len(result.stim_vec) == 5
        assert len(result.spike_counts) == 5
        assert result.stim_vec[0] == pytest.approx(-0.2)
        assert result.stim_vec[1] == pytest.approx(-0.1)
        # RMP should be near -70 mV
        assert result.rmp == pytest.approx(-70.0, abs=5.0)

    def test_iv_with_spikes(self):
        ws = _make_ws_data(
            element_key="element1",
            delay=0.5,
            pulse_dur=0.5,
            n_pulses=5,
            first_amp=-0.1,
            amp_change=0.05,
            rmp=-70.0,
            add_spikes=True,
            spike_in_sweep=3,
        )
        result = extract_iv(ws)
        # Sweeps before spike_in_sweep should have 0 spikes
        assert result.spike_counts[0] == 0
        assert result.spike_counts[1] == 0


# ---------------------------------------------------------------------------
# Tests for extract_rheobase
# ---------------------------------------------------------------------------


class TestExtractRheobase:
    """Tests for extract_rheobase."""

    def test_finds_rheobase(self):
        ws = _make_ws_data(
            element_key="element9",
            delay=0.5,
            pulse_dur=0.5,
            n_pulses=10,
            first_amp=0.0,
            amp_change=0.01,
            rmp=-70.0,
            add_spikes=True,
            spike_in_sweep=5,
        )
        result = extract_rheobase(ws)

        assert isinstance(result, RheobaseResult)
        # Rheobase should be at sweep 5 (0-indexed), current = 0.0 + 5*0.01 = 0.05
        assert result.rheo_current == pytest.approx(0.05, abs=0.02)
        assert result.rmp == pytest.approx(-70.0, abs=5.0)

    def test_no_spikes_raises(self):
        ws = _make_ws_data(
            element_key="element9",
            delay=0.5,
            pulse_dur=0.5,
            n_pulses=5,
            first_amp=0.0,
            amp_change=0.01,
            rmp=-70.0,
            add_spikes=False,
        )
        with pytest.raises(ValueError, match="No spikes found"):
            extract_rheobase(ws)


# ---------------------------------------------------------------------------
# Tests for extract_passive
# ---------------------------------------------------------------------------


class TestExtractPassive:
    """Tests for extract_passive."""

    def test_basic_passive(self):
        ws = _make_ws_data(
            element_key="element6",
            delay=0.25,
            delay_bp=0.25,
            pulse_dur=0.75,
            n_pulses=7,
            first_amp=-0.05,
            amp_change=0.01,
            rmp=-70.0,
        )
        result = extract_passive(ws)

        assert isinstance(result, PassiveResult)
        assert result.traces.shape[1] == 7
        assert len(result.stim_vec) == 7
        # Rin should be a finite number
        assert np.isfinite(result.rin)
        assert result.rmp == pytest.approx(-70.0, abs=5.0)

    def test_capacitance_is_computed(self):
        ws = _make_ws_data(
            element_key="element6",
            delay=0.25,
            delay_bp=0.25,
            pulse_dur=0.75,
            n_pulses=7,
            first_amp=-0.05,
            amp_change=0.01,
            rmp=-65.0,
        )
        result = extract_passive(ws)
        # Capacitance = tau / Rin * 1000; may be NaN if fit fails
        assert isinstance(result.capacitance, float)


# ---------------------------------------------------------------------------
# Tests for extract_sag
# ---------------------------------------------------------------------------


class TestExtractSag:
    """Tests for extract_sag."""

    def test_basic_sag(self):
        ws = _make_ws_data(
            element_key="element5",
            delay=0.25,
            delay_bp=0.25,
            pulse_dur=0.75,
            n_pulses=3,
            first_amp=-0.3,
            amp_change=0.0,
            rmp=-70.0,
        )
        result = extract_sag(ws)

        assert isinstance(result, SagResult)
        assert result.traces.shape[1] == 3
        assert len(result.stim_vec) == 3
        # All stim values should be the same (constant current)
        assert np.all(result.stim_vec == pytest.approx(-0.3))
        assert result.rmp == pytest.approx(-70.0, abs=5.0)

    def test_sag_ratio_shape(self):
        ws = _make_ws_data(
            element_key="element5",
            delay=0.25,
            delay_bp=0.25,
            pulse_dur=0.75,
            n_pulses=4,
            first_amp=-0.3,
            amp_change=0.0,
        )
        result = extract_sag(ws)
        assert len(result.sag_ratio) == 4


# ---------------------------------------------------------------------------
# Tests for extract_ramp
# ---------------------------------------------------------------------------


class TestExtractRamp:
    """Tests for extract_ramp."""

    def test_basic_ramp(self):
        # Ramp needs element3 in header but doesn't parse pulse params
        header = {
            "StimulationSampleRate": float(_SR),
            "StimulusLibrary": {
                "Stimuli": {
                    "element3": {
                        "Delegate": {
                            "Delay": "0.5",
                            "Duration": "2.0",
                            "EndTime": str(3.0 * _SR),
                            "Amplitude": "0.5",
                        }
                    }
                }
            },
        }
        trace = np.linspace(-70, -70, int(3.0 * _SR))  # 3s flat
        ws = {"traces": trace, "header": header}

        result = extract_ramp(ws)

        assert isinstance(result, RampResult)
        assert len(result.filtered) == len(trace)
        assert len(result.traces) == len(trace)


# ---------------------------------------------------------------------------
# Tests for dataclass structure
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Verify dataclass fields exist and are correctly typed."""

    def test_iv_result_fields(self):
        r = IVResult(
            traces=np.zeros((100, 5)),
            stim_vec=np.arange(5, dtype=float),
            spike_counts=np.zeros(5, dtype=int),
            rmp=-70.0,
            filtered=np.zeros(600),
        )
        assert r.rmp == -70.0
        assert r.traces.shape == (100, 5)

    def test_rheobase_result_fields(self):
        r = RheobaseResult(
            traces=np.zeros((100, 10)),
            stim_vec=np.arange(10, dtype=float),
            spike_counts=np.zeros(10, dtype=int),
            rheo_current=0.05,
            rmp=-70.0,
            filtered=np.zeros(1100),
        )
        assert r.rheo_current == 0.05

    def test_passive_result_fields(self):
        r = PassiveResult(
            traces=np.zeros((100, 7)),
            stim_vec=np.arange(7, dtype=float),
            rin=150.0,
            tau=np.array([10.0, 11.0]),
            capacitance=66.7,
            rmp=-70.0,
        )
        assert r.rin == 150.0

    def test_sag_result_fields(self):
        r = SagResult(
            traces=np.zeros((100, 3)),
            stim_vec=np.ones(3) * -0.3,
            sag_ratio=np.array([15.0, 14.0, 16.0]),
            rmp=-70.0,
        )
        assert len(r.sag_ratio) == 3

    def test_ramp_result_fields(self):
        r = RampResult(
            traces=np.zeros(5000),
            filtered=np.zeros(5000),
        )
        assert len(r.traces) == 5000
