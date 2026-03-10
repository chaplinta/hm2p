"""Tests for patching.ephys — filtering, deconcat, spike detection."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from hm2p.patching.ephys import (
    build_stim_vector,
    compute_rmp,
    compute_steady_state,
    count_spikes,
    deconcat_traces,
    detect_spikes,
    lowpass_filter,
)


# ---------------------------------------------------------------------------
# lowpass_filter
# ---------------------------------------------------------------------------


class TestLowpassFilter:
    """Test Butterworth low-pass filter."""

    def test_dc_signal_unchanged(self) -> None:
        """A constant (DC) signal should pass through a low-pass filter."""
        dc = np.full(500, 42.0)
        filtered = lowpass_filter(dc, fs=20000, cutoff=1000)
        np.testing.assert_allclose(filtered, 42.0, atol=1e-10)

    def test_high_frequency_attenuated(self) -> None:
        """A signal well above cutoff should be strongly attenuated."""
        fs = 20000
        t = np.arange(5000) / fs
        # 5 kHz sine — well above 1 kHz cutoff
        signal = np.sin(2 * np.pi * 5000 * t)
        filtered = lowpass_filter(signal, fs=fs, cutoff=1000, order=4)
        # RMS of filtered should be much less than RMS of original
        rms_orig = np.sqrt(np.mean(signal**2))
        rms_filt = np.sqrt(np.mean(filtered**2))
        assert rms_filt < 0.05 * rms_orig

    def test_low_frequency_passes(self) -> None:
        """A signal well below cutoff should pass with minimal attenuation."""
        fs = 20000
        t = np.arange(5000) / fs
        signal = np.sin(2 * np.pi * 50 * t)  # 50 Hz
        filtered = lowpass_filter(signal, fs=fs, cutoff=1000, order=4)
        rms_orig = np.sqrt(np.mean(signal**2))
        rms_filt = np.sqrt(np.mean(filtered**2))
        assert rms_filt > 0.95 * rms_orig

    def test_empty_signal(self) -> None:
        result = lowpass_filter(np.array([]))
        assert len(result) == 0

    def test_cutoff_above_nyquist_raises(self) -> None:
        with pytest.raises(ValueError, match="Nyquist"):
            lowpass_filter(np.ones(100), cutoff=10001, fs=20000)

    @given(
        dc_val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    )
    @settings(max_examples=20)
    def test_dc_hypothesis(self, dc_val: float) -> None:
        """Any DC value should pass through unchanged."""
        dc = np.full(200, dc_val)
        filtered = lowpass_filter(dc, fs=20000, cutoff=1000)
        np.testing.assert_allclose(filtered, dc_val, atol=1e-6)

    def test_output_shape_matches_input(self) -> None:
        signal = np.random.randn(1000)
        filtered = lowpass_filter(signal)
        assert filtered.shape == signal.shape


# ---------------------------------------------------------------------------
# deconcat_traces
# ---------------------------------------------------------------------------


class TestDeconcatTraces:
    """Test concatenated trace slicing."""

    def test_basic_slicing(self) -> None:
        """Known geometry: 3 pulses, check shape and values."""
        delay = 200  # samples
        pulse_dur = 1000
        delay_bp = 200
        n_pulses = 3
        # Total length: at least half_delay + n_pulses * (pulse_dur + delay_bp)
        total = delay // 2 + n_pulses * (pulse_dur + delay_bp) + delay
        trace = np.arange(total, dtype=np.float64)
        result = deconcat_traces(
            trace, delay=delay, delay_bp=delay_bp,
            pulse_dur=pulse_dur, n_pulses=n_pulses, sr=20000,
        )
        assert result.shape[1] == n_pulses
        assert result.shape[0] == pulse_dur + delay_bp

    def test_single_pulse(self) -> None:
        delay = 100
        pulse_dur = 500
        delay_bp = 100
        total = 1000
        trace = np.ones(total)
        result = deconcat_traces(
            trace, delay=delay, delay_bp=delay_bp,
            pulse_dur=pulse_dur, n_pulses=1, sr=20000,
        )
        assert result.shape[1] == 1

    def test_values_are_correct_slice(self) -> None:
        """Each column should contain the corresponding slice of the input."""
        delay = 100
        pulse_dur = 200
        delay_bp = 50
        step = pulse_dur + delay_bp
        half_delay = delay // 2
        n_pulses = 2
        total = half_delay + (n_pulses + 1) * step
        trace = np.arange(total, dtype=np.float64)
        result = deconcat_traces(
            trace, delay=delay, delay_bp=delay_bp,
            pulse_dur=pulse_dur, n_pulses=n_pulses, sr=20000,
        )
        # First column starts at half_delay
        expected_start_0 = half_delay
        np.testing.assert_array_equal(
            result[:, 0], trace[expected_start_0 : expected_start_0 + step]
        )


# ---------------------------------------------------------------------------
# build_stim_vector
# ---------------------------------------------------------------------------


class TestBuildStimVector:
    """Test stimulus amplitude vector construction."""

    def test_basic(self) -> None:
        vec = build_stim_vector(first_amp=-200, amp_change=50, n_pulses=5)
        np.testing.assert_array_equal(vec, [-200, -150, -100, -50, 0])

    def test_single_pulse(self) -> None:
        vec = build_stim_vector(first_amp=10, amp_change=5, n_pulses=1)
        np.testing.assert_array_equal(vec, [10])

    def test_length(self) -> None:
        vec = build_stim_vector(first_amp=0, amp_change=10, n_pulses=19)
        assert len(vec) == 19

    @given(
        first_amp=st.floats(min_value=-1000, max_value=1000, allow_nan=False),
        amp_change=st.floats(min_value=-100, max_value=100, allow_nan=False),
        n_pulses=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=30)
    def test_length_hypothesis(
        self, first_amp: float, amp_change: float, n_pulses: int
    ) -> None:
        vec = build_stim_vector(first_amp, amp_change, n_pulses)
        assert len(vec) == n_pulses

    @given(
        first_amp=st.floats(min_value=-1000, max_value=1000, allow_nan=False),
        amp_change=st.floats(
            min_value=0.1, max_value=100, allow_nan=False
        ),
        n_pulses=st.integers(min_value=2, max_value=50),
    )
    @settings(max_examples=30)
    def test_monotonic_increasing(
        self, first_amp: float, amp_change: float, n_pulses: int
    ) -> None:
        vec = build_stim_vector(first_amp, amp_change, n_pulses)
        assert np.all(np.diff(vec) > 0)


# ---------------------------------------------------------------------------
# detect_spikes
# ---------------------------------------------------------------------------


class TestDetectSpikes:
    """Test spike detection by threshold crossing."""

    def test_single_spike(self) -> None:
        """A single Gaussian peak should yield one spike."""
        t = np.arange(1000)
        trace = -60 + 80 * np.exp(-((t - 500) ** 2) / (2 * 5**2))
        spikes = detect_spikes(trace, threshold_factor=0.5)
        assert len(spikes) == 1
        assert spikes[0] == 500  # peak at center

    def test_two_spikes(self) -> None:
        t = np.arange(1000)
        trace = (
            -60
            + 80 * np.exp(-((t - 200) ** 2) / (2 * 5**2))
            + 80 * np.exp(-((t - 700) ** 2) / (2 * 5**2))
        )
        spikes = detect_spikes(trace, threshold_factor=0.5)
        assert len(spikes) == 2

    def test_no_spikes_flat(self) -> None:
        trace = np.full(100, -70.0)
        spikes = detect_spikes(trace, threshold_factor=0.5)
        assert len(spikes) == 0

    def test_no_spikes_negative(self) -> None:
        """All-negative trace should yield no spikes."""
        trace = np.full(100, -10.0)
        spikes = detect_spikes(trace, threshold_factor=0.5)
        assert len(spikes) == 0

    def test_empty_trace(self) -> None:
        spikes = detect_spikes(np.array([]))
        assert len(spikes) == 0

    def test_threshold_factor_controls_sensitivity(self) -> None:
        """Lower threshold should detect more spikes."""
        t = np.arange(1000)
        # Two peaks: big one and small one
        trace = (
            -60
            + 80 * np.exp(-((t - 300) ** 2) / (2 * 5**2))
            + 20 * np.exp(-((t - 700) ** 2) / (2 * 5**2))
        )
        high_thresh = detect_spikes(trace, threshold_factor=0.8)
        low_thresh = detect_spikes(trace, threshold_factor=0.1)
        assert len(low_thresh) >= len(high_thresh)

    def test_spike_at_peak(self) -> None:
        """Returned index should correspond to the actual peak."""
        t = np.arange(200)
        trace = np.zeros(200)
        trace[100] = 50.0
        trace[99] = 30.0
        trace[101] = 30.0
        spikes = detect_spikes(trace, threshold_factor=0.5)
        assert len(spikes) == 1
        assert spikes[0] == 100


# ---------------------------------------------------------------------------
# count_spikes
# ---------------------------------------------------------------------------


class TestCountSpikes:
    """Test per-column spike counting."""

    def test_basic(self) -> None:
        t = np.arange(500)
        col1 = -60 + 80 * np.exp(-((t - 250) ** 2) / (2 * 5**2))
        col2 = np.full(500, -70.0)
        traces = np.column_stack([col1, col2])
        counts = count_spikes(traces, threshold_factor=0.5)
        assert counts[0] == 1
        assert counts[1] == 0

    def test_1d_input(self) -> None:
        t = np.arange(500)
        trace = -60 + 80 * np.exp(-((t - 250) ** 2) / (2 * 5**2))
        counts = count_spikes(trace)
        assert len(counts) == 1
        assert counts[0] == 1


# ---------------------------------------------------------------------------
# compute_rmp
# ---------------------------------------------------------------------------


class TestComputeRmp:
    """Test resting membrane potential computation."""

    def test_constant_baseline(self) -> None:
        traces = np.full((200, 3), -65.0)
        rmp = compute_rmp(traces, baseline_samples=100)
        assert rmp == -65.0

    def test_uses_only_baseline(self) -> None:
        """Values after baseline_samples should not affect result."""
        traces = np.zeros((200, 2))
        traces[:50, :] = -70.0
        traces[50:, :] = 100.0  # should be ignored
        rmp = compute_rmp(traces, baseline_samples=50)
        assert rmp == -70.0

    def test_1d_input(self) -> None:
        trace = np.full(100, -60.0)
        rmp = compute_rmp(trace, baseline_samples=50)
        assert rmp == -60.0

    @given(
        val=st.floats(min_value=-200, max_value=100, allow_nan=False),
    )
    @settings(max_examples=20)
    def test_rmp_rounded(self, val: float) -> None:
        traces = np.full((100, 1), val)
        rmp = compute_rmp(traces, baseline_samples=50)
        assert rmp == round(val, 1)


# ---------------------------------------------------------------------------
# compute_steady_state
# ---------------------------------------------------------------------------


class TestComputeSteadyState:
    """Test steady-state voltage deflection."""

    def test_known_deflection(self) -> None:
        """If baseline = -70 and ss = -50, deflection = 20."""
        trace = np.zeros(1000)
        trace[:200] = -70.0  # baseline window
        trace[400:600] = -50.0  # steady-state window
        ss = compute_steady_state(trace, 0, 200, 400, 600)
        assert ss == pytest.approx(20.0)

    def test_zero_deflection(self) -> None:
        trace = np.full(500, -65.0)
        ss = compute_steady_state(trace, 0, 100, 200, 300)
        assert ss == pytest.approx(0.0)

    def test_negative_deflection(self) -> None:
        trace = np.zeros(500)
        trace[:100] = -60.0
        trace[200:300] = -80.0
        ss = compute_steady_state(trace, 0, 100, 200, 300)
        assert ss == pytest.approx(-20.0)

    def test_2d_input(self) -> None:
        traces = np.zeros((500, 3))
        traces[:100, :] = -70.0
        traces[200:300, :] = -50.0
        ss = compute_steady_state(traces, 0, 100, 200, 300)
        assert ss == pytest.approx(20.0)

    @given(
        baseline=st.floats(min_value=-200, max_value=0, allow_nan=False),
        ss_val=st.floats(min_value=-200, max_value=0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_deflection_is_difference(self, baseline: float, ss_val: float) -> None:
        trace = np.zeros(500)
        trace[:100] = baseline
        trace[200:300] = ss_val
        result = compute_steady_state(trace, 0, 100, 200, 300)
        assert result == pytest.approx(ss_val - baseline, abs=1e-10)
