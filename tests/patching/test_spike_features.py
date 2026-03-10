"""Tests for hm2p.patching.spike_features.

Uses synthetic action potential waveforms — never real data files.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: synthetic spike generation
# ---------------------------------------------------------------------------


def _make_synthetic_spike_trace(
    sr: int = 20_000,
    duration_ms: float = 200.0,
    rmp: float = -70.0,
    peak: float = 40.0,
    ahp: float = -80.0,
    spike_time_ms: float = 50.0,
    spike_width_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a minimal synthetic trace containing one action potential.

    The AP is modelled as a triangular rise/fall with an AHP undershoot,
    sitting on a flat baseline at ``rmp``.

    Returns (time_ms, voltage_mV).
    """
    n_samples = int(sr * duration_ms / 1000.0)
    time_ms = np.linspace(0, duration_ms, n_samples)
    voltage = np.full(n_samples, rmp)

    # Spike peak
    spike_idx = int(spike_time_ms / duration_ms * n_samples)
    half_width_samples = int(spike_width_ms / 1000.0 * sr / 2)
    ahp_samples = int(2.0 / 1000.0 * sr)  # 2ms AHP

    # Rising phase
    rise_start = max(0, spike_idx - half_width_samples)
    for i in range(rise_start, spike_idx):
        frac = (i - rise_start) / max(1, spike_idx - rise_start)
        voltage[i] = rmp + frac * (peak - rmp)

    # Peak
    voltage[spike_idx] = peak

    # Falling phase to AHP
    fall_end = min(n_samples - 1, spike_idx + half_width_samples)
    for i in range(spike_idx + 1, fall_end + 1):
        frac = (i - spike_idx) / max(1, fall_end - spike_idx)
        voltage[i] = peak + frac * (ahp - peak)

    # AHP recovery
    ahp_end = min(n_samples - 1, fall_end + ahp_samples)
    for i in range(fall_end + 1, ahp_end + 1):
        frac = (i - fall_end) / max(1, ahp_end - fall_end)
        voltage[i] = ahp + frac * (rmp - ahp)

    return time_ms, voltage


def _make_two_spike_trace(
    sr: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a trace with two spikes for testing spike_index selection."""
    duration_ms = 300.0
    n_samples = int(sr * duration_ms / 1000.0)
    time_ms = np.linspace(0, duration_ms, n_samples)
    voltage = np.full(n_samples, -70.0)

    for spike_time_ms in [80.0, 150.0]:
        spike_idx = int(spike_time_ms / duration_ms * n_samples)
        hw = int(0.5 / 1000.0 * sr)
        ahp_s = int(2.0 / 1000.0 * sr)
        rise_start = max(0, spike_idx - hw)
        for i in range(rise_start, spike_idx):
            frac = (i - rise_start) / max(1, spike_idx - rise_start)
            voltage[i] = -70.0 + frac * 110.0
        voltage[spike_idx] = 40.0
        fall_end = min(n_samples - 1, spike_idx + hw)
        for i in range(spike_idx + 1, fall_end + 1):
            frac = (i - spike_idx) / max(1, fall_end - spike_idx)
            voltage[i] = 40.0 + frac * (-80.0 - 40.0)
        ahp_end = min(n_samples - 1, fall_end + ahp_s)
        for i in range(fall_end + 1, ahp_end + 1):
            frac = (i - fall_end) / max(1, ahp_end - fall_end)
            voltage[i] = -80.0 + frac * 10.0

    return time_ms, voltage


# ---------------------------------------------------------------------------
# Tests for extract_waveform
# ---------------------------------------------------------------------------


class TestExtractWaveform:
    """Tests for extract_waveform."""

    def test_basic_window(self):
        from hm2p.patching.spike_features import extract_waveform

        sr = 20_000
        trace = np.zeros(10_000)
        spike_idx = 5000
        trace[spike_idx] = 40.0

        time_ms, voltage = extract_waveform(trace, spike_idx, sr=sr)

        # pre=7ms, post=20ms at 20kHz: 140 + 400 + 1 = 541 samples
        expected_len = int(7.0 * 20) + int(20.0 * 20) + 1
        assert len(time_ms) == expected_len
        assert len(voltage) == expected_len
        assert time_ms[0] == pytest.approx(-7.0, abs=0.1)
        assert time_ms[-1] == pytest.approx(20.0, abs=0.1)

    def test_spike_in_window(self):
        from hm2p.patching.spike_features import extract_waveform

        trace = np.zeros(10_000)
        spike_idx = 5000
        trace[spike_idx] = 40.0
        time_ms, voltage = extract_waveform(trace, spike_idx, sr=20_000)

        # The spike peak should be at time ~0
        peak_idx = np.argmax(voltage)
        assert time_ms[peak_idx] == pytest.approx(0.0, abs=0.2)
        assert voltage[peak_idx] == pytest.approx(40.0)

    def test_custom_window(self):
        from hm2p.patching.spike_features import extract_waveform

        trace = np.ones(20_000) * -65.0
        spike_idx = 10_000
        time_ms, voltage = extract_waveform(
            trace, spike_idx, sr=20_000, pre_ms=5.0, post_ms=10.0
        )
        expected_len = int(5.0 * 20) + int(10.0 * 20) + 1
        assert len(time_ms) == expected_len

    def test_raises_on_early_spike(self):
        from hm2p.patching.spike_features import extract_waveform

        trace = np.zeros(10_000)
        with pytest.raises(ValueError, match="starts before trace"):
            extract_waveform(trace, spike_time_idx=10, sr=20_000, pre_ms=7.0)

    def test_raises_on_late_spike(self):
        from hm2p.patching.spike_features import extract_waveform

        trace = np.zeros(10_000)
        with pytest.raises(ValueError, match="extends beyond trace"):
            extract_waveform(trace, spike_time_idx=9999, sr=20_000, post_ms=20.0)


# ---------------------------------------------------------------------------
# Tests for extract_spike_features
# ---------------------------------------------------------------------------


class TestExtractSpikeFeatures:
    """Tests for extract_spike_features (requires eFEL)."""

    @pytest.fixture(autouse=True)
    def _check_efel(self):
        """Skip if eFEL is not installed."""
        pytest.importorskip("efel")

    def test_single_spike_features(self):
        from hm2p.patching.spike_features import extract_spike_features

        time_ms, voltage = _make_synthetic_spike_trace(
            sr=20_000, peak=40.0, rmp=-70.0, ahp=-80.0, spike_time_ms=50.0
        )

        result = extract_spike_features(
            voltage, time_ms, stim_start=10.0, stim_end=190.0, spike_index=0
        )

        assert result is not None
        assert "peak_vm" in result
        assert "amplitude" in result
        assert "half_width" in result
        assert "spike_time" in result
        # Peak should be near +40 mV
        assert result["peak_vm"] > 20.0
        # Amplitude should be positive
        assert result["amplitude"] > 0

    def test_no_spike_returns_none(self):
        from hm2p.patching.spike_features import extract_spike_features

        time_ms = np.linspace(0, 200, 4000)
        voltage = np.full_like(time_ms, -70.0)

        result = extract_spike_features(
            voltage, time_ms, stim_start=10.0, stim_end=190.0
        )
        assert result is None

    def test_spike_index_fallback(self):
        """If spike_index > n_spikes, should fall back to first spike."""
        from hm2p.patching.spike_features import extract_spike_features

        time_ms, voltage = _make_synthetic_spike_trace(spike_time_ms=50.0)

        result = extract_spike_features(
            voltage, time_ms, stim_start=10.0, stim_end=190.0, spike_index=5
        )

        # Should still return features (fell back to first spike)
        assert result is not None
        assert result["peak_vm"] > 0

    def test_two_spikes_selects_second(self):
        """With spike_index=1, should select the 2nd spike."""
        from hm2p.patching.spike_features import extract_spike_features

        time_ms, voltage = _make_two_spike_trace()

        result = extract_spike_features(
            voltage, time_ms, stim_start=10.0, stim_end=290.0, spike_index=1
        )

        assert result is not None
        # Second spike is at ~150ms
        assert result["spike_time"] > 100.0

    def test_result_keys(self):
        from hm2p.patching.spike_features import extract_spike_features

        time_ms, voltage = _make_synthetic_spike_trace()
        result = extract_spike_features(
            voltage, time_ms, stim_start=10.0, stim_end=190.0, spike_index=0
        )

        expected_keys = {
            "min_vm", "peak_vm", "max_vm_slope", "half_vm",
            "amplitude", "max_ahp", "half_width", "spike_time",
        }
        assert result is not None
        assert set(result.keys()) == expected_keys


class TestExtractSpikeFeaturesImportError:
    """Test ImportError handling when eFEL is not installed."""

    def test_import_error_message(self, monkeypatch):
        """Verify helpful error message when eFEL is missing."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "efel":
                raise ImportError("No module named 'efel'")
            return real_import(name, *args, **kwargs)

        # Need to reload the module to trigger the import
        import importlib
        import hm2p.patching.spike_features as sf_mod

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pip install efel"):
            sf_mod.extract_spike_features(
                np.zeros(100), np.linspace(0, 10, 100), 1.0, 9.0
            )
