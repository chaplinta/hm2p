"""Tests for calcium/events.py — Voigts & Harnett calcium transient detection."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.calcium.events import (
    BatchEventResult,
    EventResult,
    _get_crossings,
    compute_event_rate,
    compute_event_snr,
    detect_events,
    detect_events_batch,
    detect_events_single,
    estimate_noise_probability,
)


# ---------------------------------------------------------------------------
# _get_crossings
# ---------------------------------------------------------------------------


class TestGetCrossings:
    def test_simple_crossing(self):
        data = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        result = _get_crossings(data, 0.5)
        assert list(result) == [2]

    def test_no_crossing(self):
        data = np.array([0.0, 0.0, 0.0])
        result = _get_crossings(data, 0.5)
        assert len(result) == 0

    def test_multiple_crossings(self):
        data = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        result = _get_crossings(data, 0.5)
        assert list(result) == [1, 3]

    def test_exact_threshold(self):
        # [0.0, 0.5, 1.0]: both 0→0.5 and 0.5→1.0 satisfy <= and >=
        data = np.array([0.0, 0.5, 1.0])
        result = _get_crossings(data, 0.5)
        assert len(result) == 2
        assert list(result) == [1, 2]


# ---------------------------------------------------------------------------
# estimate_noise_probability
# ---------------------------------------------------------------------------


class TestEstimateNoiseProbability:
    def test_output_shape(self, rng):
        trace = rng.standard_normal(500)
        noise_prob, trace_norm = estimate_noise_probability(trace)
        assert noise_prob.shape == (500,)
        assert trace_norm.shape == (500,)

    def test_noise_prob_range(self, rng):
        trace = rng.standard_normal(1000).astype(np.float64)
        trace = np.abs(trace)  # ensure positive
        noise_prob, _ = estimate_noise_probability(trace)
        assert noise_prob.min() >= 0.0
        assert noise_prob.max() <= 1.0

    def test_trace_norm_range(self, rng):
        trace = np.abs(rng.standard_normal(500))
        _, trace_norm = estimate_noise_probability(trace)
        assert trace_norm.min() >= -1e-10  # floating point tolerance
        assert trace_norm.max() <= 1.0 + 1e-10

    def test_flat_signal(self):
        trace = np.ones(200) * 5.0
        noise_prob, trace_norm = estimate_noise_probability(trace)
        # Flat signal → all zeros after normalization
        assert np.allclose(trace_norm, 0.0)

    def test_no_smoothing(self, rng):
        trace = np.abs(rng.standard_normal(300))
        noise_prob, _ = estimate_noise_probability(trace, smooth_sigma=None)
        assert noise_prob.shape == (300,)

    def test_large_transient_low_noise_prob(self):
        """A large transient should have low noise probability."""
        trace = np.zeros(500)
        trace[200:230] = 10.0
        noise_prob, _ = estimate_noise_probability(trace, smooth_sigma=1)
        # During the transient, noise prob should be low
        assert noise_prob[210:220].mean() < 0.3


# ---------------------------------------------------------------------------
# detect_events_single
# ---------------------------------------------------------------------------


class TestDetectEventsSingle:
    def test_returns_event_result(self):
        trace = np.zeros(300)
        trace[100:130] = 5.0
        result = detect_events_single(trace, smooth_sigma=1)
        assert isinstance(result, EventResult)

    def test_output_shapes(self):
        trace = np.zeros(300)
        trace[100:130] = 5.0
        result = detect_events_single(trace, smooth_sigma=1)
        assert result.event_mask.shape == (300,)
        assert result.noise_prob.shape == (300,)
        assert len(result.onsets) == len(result.offsets)
        assert len(result.onsets) == len(result.amplitudes)

    def test_flat_signal_no_events(self):
        trace = np.zeros(300)
        result = detect_events_single(trace)
        assert len(result.onsets) == 0
        assert result.event_mask.sum() == 0

    def test_large_transient_detected(self):
        """A clear, large transient should be detected."""
        trace = np.random.default_rng(0).normal(0, 0.01, 1000)
        trace = np.abs(trace)
        # Add a clear transient
        trace[400:430] = 2.0
        result = detect_events_single(trace, smooth_sigma=2)
        assert len(result.onsets) >= 1
        # At least some frames in the transient region should be masked
        assert result.event_mask[400:430].sum() > 0

    def test_onset_before_offset(self):
        """Every onset should be before its corresponding offset."""
        rng = np.random.default_rng(1)
        trace = np.abs(rng.standard_normal(1000))
        trace[200:250] = 5.0
        trace[600:650] = 5.0
        result = detect_events_single(trace, smooth_sigma=2)
        for onset, offset in zip(result.onsets, result.offsets):
            assert onset < offset

    def test_amplitudes_positive(self):
        """Event amplitudes should be positive."""
        trace = np.abs(np.random.default_rng(2).standard_normal(500))
        trace[200:230] = 3.0
        result = detect_events_single(trace, smooth_sigma=2)
        if len(result.amplitudes) > 0:
            assert np.all(result.amplitudes > 0)

    def test_alpha_significance_filter(self):
        """With strict alpha, marginal events should be filtered out."""
        rng = np.random.default_rng(3)
        # Noisy trace with small bumps
        trace = np.abs(rng.normal(0, 0.1, 1000))
        trace[300:310] = 0.5  # small bump
        result_lenient = detect_events_single(trace, smooth_sigma=2, alpha=1.0)
        result_strict = detect_events_single(trace, smooth_sigma=2, alpha=0.01)
        assert len(result_strict.onsets) <= len(result_lenient.onsets)

    def test_non_overlapping_events(self):
        """Events should not overlap."""
        trace = np.abs(np.random.default_rng(4).standard_normal(1000))
        trace[200:230] = 3.0
        trace[400:430] = 3.0
        trace[700:730] = 3.0
        result = detect_events_single(trace, smooth_sigma=2)
        for i in range(len(result.onsets) - 1):
            assert result.offsets[i] <= result.onsets[i + 1]


# ---------------------------------------------------------------------------
# detect_events (batch interface)
# ---------------------------------------------------------------------------


class TestDetectEvents:
    def test_output_shape(self, rng):
        dff = rng.standard_normal((10, 500)).astype(np.float32)
        result = detect_events(dff, fps=30.0)
        assert result.shape == dff.shape

    def test_output_dtype_float32(self, rng):
        dff = rng.standard_normal((5, 200)).astype(np.float32)
        result = detect_events(dff, fps=30.0)
        assert result.dtype == np.float32

    def test_output_binary(self, rng):
        dff = rng.standard_normal((8, 300)).astype(np.float32)
        result = detect_events(dff, fps=30.0)
        unique = np.unique(result)
        assert set(unique).issubset({0.0, 1.0})

    def test_flat_signal_no_events(self):
        dff = np.zeros((3, 300), dtype=np.float32)
        result = detect_events(dff, fps=30.0)
        assert result.sum() == 0.0

    def test_large_transient_detected(self):
        dff = np.zeros((1, 1000), dtype=np.float32)
        dff[0, :] = np.abs(np.random.default_rng(0).normal(0, 0.01, 1000))
        dff[0, 400:430] = 2.0
        result = detect_events(dff, fps=30.0, smooth_sigma=2)
        assert result[0, 400:430].sum() > 0

    @given(
        n_rois=st.integers(min_value=1, max_value=20),
        n_frames=st.integers(min_value=10, max_value=200),
    )
    @settings(max_examples=30)
    def test_shape_property(self, n_rois: int, n_frames: int):
        rng = np.random.default_rng(1)
        dff = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
        result = detect_events(dff, fps=10.0)
        assert result.shape == (n_rois, n_frames)


# ---------------------------------------------------------------------------
# detect_events_batch
# ---------------------------------------------------------------------------


class TestDetectEventsBatch:
    def test_returns_batch_result(self, rng):
        dff = rng.standard_normal((5, 300)).astype(np.float32)
        result = detect_events_batch(dff, fps=30.0)
        assert isinstance(result, BatchEventResult)
        assert len(result.events) == 5
        assert result.event_masks.shape == (5, 300)
        assert result.noise_probs.shape == (5, 300)

    def test_events_are_event_results(self, rng):
        dff = rng.standard_normal((3, 200)).astype(np.float32)
        result = detect_events_batch(dff, fps=30.0)
        for ev in result.events:
            assert isinstance(ev, EventResult)


# ---------------------------------------------------------------------------
# compute_event_snr
# ---------------------------------------------------------------------------


class TestComputeEventSNR:
    def test_no_events_nan(self):
        dff = np.random.default_rng(0).standard_normal(100)
        mask = np.zeros(100, dtype=np.int32)
        snr = compute_event_snr(dff, mask, np.array([]))
        assert np.isnan(snr)

    def test_positive_snr(self):
        dff = np.random.default_rng(0).normal(0, 0.1, 500)
        mask = np.zeros(500, dtype=np.int32)
        mask[200:230] = 1
        amps = np.array([2.0])
        snr = compute_event_snr(dff, mask, amps)
        assert snr > 0

    def test_bad_frames_excluded(self):
        dff = np.random.default_rng(0).normal(0, 0.1, 500)
        dff[0:50] = 100.0  # bad frames with huge values
        mask = np.zeros(500, dtype=np.int32)
        mask[200:230] = 1
        amps = np.array([2.0])
        bad = np.zeros(500, dtype=bool)
        bad[0:50] = True
        snr_with_bad = compute_event_snr(dff, mask, amps, bad_frames=bad)
        snr_without = compute_event_snr(dff, mask, amps)
        # Excluding the bad high-value frames should increase SNR
        assert snr_with_bad > snr_without


# ---------------------------------------------------------------------------
# compute_event_rate
# ---------------------------------------------------------------------------


class TestComputeEventRate:
    def test_basic_rate(self):
        # 10 events in 600 frames at 30 fps = 600/30 = 20s = 1/3 min
        # rate = 10 / (1/3) = 30 events/min
        onsets = np.arange(10)
        rate = compute_event_rate(onsets, n_frames=600, fps=30.0)
        assert abs(rate - 30.0) < 0.1

    def test_no_events(self):
        rate = compute_event_rate(np.array([], dtype=np.int64), n_frames=600, fps=30.0)
        assert rate == 0.0

    def test_bad_frames_excluded(self):
        onsets = np.array([10, 20, 30, 100, 200])
        bad = np.zeros(300, dtype=bool)
        bad[10] = True  # one onset is in a bad frame
        rate_clean = compute_event_rate(onsets, 300, 30.0, bad_frames=bad)
        rate_all = compute_event_rate(onsets, 300, 30.0)
        # Fewer good events and slightly shorter duration
        assert rate_clean < rate_all

    def test_all_frames_bad(self):
        """All frames bad gives zero rate."""
        onsets = np.array([10, 20])
        bad = np.ones(100, dtype=bool)
        rate = compute_event_rate(onsets, 100, 30.0, bad_frames=bad)
        assert rate == 0.0


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


class TestEstimateNoiseProbabilityEdgeCases:
    def test_all_zeros(self):
        """All-zero trace should produce all-zero normalised trace."""
        trace = np.zeros(200)
        noise_prob, trace_norm = estimate_noise_probability(trace)
        assert np.allclose(trace_norm, 0.0)

    def test_negative_values_clipped(self):
        """Negative values are rectified (clipped to 0) before processing."""
        trace = np.full(200, -5.0)
        noise_prob, trace_norm = estimate_noise_probability(trace)
        assert noise_prob.shape == (200,)


class TestDetectEventsSingleEdgeCases:
    def test_very_short_trace(self):
        """Detection on a very short trace (5 frames) should not crash."""
        trace = np.array([0.0, 0.0, 5.0, 0.0, 0.0])
        result = detect_events_single(trace, smooth_sigma=None)
        assert result.event_mask.shape == (5,)

    def test_constant_high_signal(self):
        """Constant high trace should have no events (no rising edge)."""
        trace = np.ones(300) * 10.0
        result = detect_events_single(trace)
        # Flat signal normalises to zero, no crossings
        assert len(result.onsets) == 0


class TestComputeEventSNREdgeCases:
    def test_all_event_frames(self):
        """When all frames are in events, non-event noise is zero -> NaN."""
        dff = np.ones(100)
        mask = np.ones(100, dtype=np.int32)
        amps = np.array([2.0])
        snr = compute_event_snr(dff, mask, amps)
        assert np.isnan(snr)

    def test_zero_noise_std(self):
        """When non-event frames have zero std, SNR is NaN."""
        dff = np.ones(100)  # constant everywhere
        mask = np.zeros(100, dtype=np.int32)
        mask[50:60] = 1
        amps = np.array([1.0])
        snr = compute_event_snr(dff, mask, amps)
        assert np.isnan(snr)
