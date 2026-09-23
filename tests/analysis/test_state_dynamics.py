"""Tests for hm2p.analysis.state_dynamics — movement-state dynamics."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis.state_dynamics import (
    _as_bool_1d,
    _as_float_1d,
    _check_fps,
    _nanmean,
    _nansem,
    _sustained_ratio,
    activity_vs_bout_duration,
    bout_durations,
    event_aligned_average,
    immobility_decay,
    movement_state_summary,
    onset_latency_frames,
    segment_bouts,
    syllable_conditioned_activity,
    syllable_information,
    transient_index,
)

FPS = 10.0


def _alternating_state(block: int, n_blocks: int, start_true: bool = True) -> np.ndarray:
    """Build a boolean array of alternating equal-length blocks."""
    blocks = [np.full(block, (i % 2 == 0) == start_true, dtype=bool) for i in range(n_blocks)]
    return np.concatenate(blocks)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    def test_nanmean_all_nan_returns_nan(self):
        out = _nanmean(np.full(5, np.nan))
        assert np.isnan(out)

    def test_nanmean_axis(self):
        arr = np.array([[1.0, np.nan], [3.0, 5.0]])
        assert np.allclose(_nanmean(arr, axis=0), [2.0, 5.0])

    def test_nansem_requires_two_samples(self):
        arr = np.array([[1.0, np.nan], [3.0, np.nan]])
        sem = _nansem(arr, axis=0)
        assert np.isclose(sem[0], np.std([1.0, 3.0], ddof=1) / np.sqrt(2))
        assert np.isnan(sem[1])

    def test_as_float_1d_rejects_2d(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            _as_float_1d(np.zeros((2, 2)), "signal")

    def test_as_bool_1d_rejects_2d(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            _as_bool_1d(np.zeros((2, 2)), "mask")

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
    def test_check_fps_rejects_invalid(self, bad):
        with pytest.raises(ValueError, match="fps must be"):
            _check_fps(bad)

    def test_check_fps_returns_float(self):
        assert _check_fps(9.6) == pytest.approx(9.6)


# ---------------------------------------------------------------------------
# segment_bouts / bout_durations
# ---------------------------------------------------------------------------


class TestSegmentBouts:
    def test_simple_pattern(self):
        state = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)
        onsets, offsets = segment_bouts(state)
        assert onsets.tolist() == [1, 5]
        assert offsets.tolist() == [3, 6]

    def test_all_true_single_bout(self):
        onsets, offsets = segment_bouts(np.ones(10, dtype=bool))
        assert onsets.tolist() == [0]
        assert offsets.tolist() == [10]

    def test_all_false_no_bouts(self):
        onsets, offsets = segment_bouts(np.zeros(10, dtype=bool))
        assert onsets.size == 0
        assert offsets.size == 0

    def test_single_frame_bout(self):
        state = np.array([0, 1, 0], dtype=bool)
        onsets, offsets = segment_bouts(state)
        assert onsets.tolist() == [1]
        assert offsets.tolist() == [2]

    def test_min_frames_filter(self):
        state = np.array([1, 0, 1, 1, 1, 0, 1], dtype=bool)
        onsets, offsets = segment_bouts(state, min_frames=3)
        assert onsets.tolist() == [2]
        assert offsets.tolist() == [5]

    def test_empty_input(self):
        onsets, offsets = segment_bouts(np.zeros(0, dtype=bool))
        assert onsets.size == 0
        assert offsets.size == 0

    def test_bout_touching_both_edges(self):
        state = np.array([1, 1, 0, 1], dtype=bool)
        onsets, offsets = segment_bouts(state)
        assert onsets.tolist() == [0, 3]
        assert offsets.tolist() == [2, 4]

    def test_min_frames_below_one_raises(self):
        with pytest.raises(ValueError, match="min_frames must be"):
            segment_bouts(np.ones(3, dtype=bool), min_frames=0)

    def test_two_dimensional_raises(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            segment_bouts(np.ones((2, 2), dtype=bool))

    @settings(max_examples=50, deadline=None)
    @given(st.lists(st.booleans(), max_size=80))
    def test_onsets_precede_offsets_and_cover_true_frames(self, values):
        state = np.array(values, dtype=bool)
        onsets, offsets = segment_bouts(state)
        assert np.all(onsets < offsets)
        assert int(np.sum(offsets - onsets)) == int(state.sum())

    @settings(max_examples=50, deadline=None)
    @given(st.lists(st.booleans(), max_size=80), st.integers(min_value=1, max_value=5))
    def test_min_frames_respected(self, values, min_frames):
        state = np.array(values, dtype=bool)
        onsets, offsets = segment_bouts(state, min_frames=min_frames)
        assert np.all((offsets - onsets) >= min_frames)
        assert int(np.sum(offsets - onsets)) <= int(state.sum())


class TestBoutDurations:
    def test_known_durations(self):
        durations = bout_durations(np.array([0, 10]), np.array([5, 30]), fps=10.0)
        assert durations.tolist() == [0.5, 2.0]

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            bout_durations(np.array([0, 1]), np.array([2]), fps=10.0)

    def test_bad_fps_raises(self):
        with pytest.raises(ValueError, match="fps must be"):
            bout_durations(np.array([0]), np.array([1]), fps=0.0)


# ---------------------------------------------------------------------------
# event_aligned_average
# ---------------------------------------------------------------------------


class TestEventAlignedAverage:
    def test_constant_signal(self):
        signal = np.full(100, 2.0)
        out = event_aligned_average(signal, np.array([20, 50]), 5, 10)
        assert out["n_events"] == 2
        assert out["windows"].shape == (2, 15)
        assert np.allclose(out["mean"], 2.0)
        assert np.allclose(out["sem"], 0.0)
        assert out["time_axis"][0] == -5
        assert out["time_axis"][-1] == 9

    def test_step_response_recovered(self):
        signal = np.zeros(100)
        signal[50:] = 3.0
        out = event_aligned_average(signal, np.array([50]), 10, 10)
        assert np.allclose(out["mean"][:10], 0.0)
        assert np.allclose(out["mean"][10:], 3.0)

    def test_edge_events_dropped(self):
        signal = np.arange(50, dtype=float)
        out = event_aligned_average(signal, np.array([1, 25, 49]), 5, 5)
        assert out["n_events"] == 1

    def test_no_valid_events_returns_nan(self):
        signal = np.arange(10, dtype=float)
        out = event_aligned_average(signal, np.array([0]), 5, 5)
        assert out["n_events"] == 0
        assert np.all(np.isnan(out["mean"]))
        assert np.all(np.isnan(out["sem"]))
        assert out["windows"].shape == (0, 10)

    def test_baseline_subtraction(self):
        signal = np.zeros(100)
        signal[:50] = 1.0
        signal[50:] = 4.0
        out = event_aligned_average(signal, np.array([50]), 10, 10, baseline_frames=10)
        assert np.allclose(out["mean"][:10], 0.0)
        assert np.allclose(out["mean"][10:], 3.0)

    def test_nan_frames_ignored(self):
        signal = np.ones(60)
        signal[30] = np.nan
        out = event_aligned_average(signal, np.array([20, 40]), 5, 5)
        assert np.allclose(out["mean"], 1.0)

    def test_single_event_sem_is_nan(self):
        out = event_aligned_average(np.ones(50), np.array([25]), 5, 5)
        assert np.all(np.isnan(out["sem"]))

    def test_negative_window_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            event_aligned_average(np.ones(10), np.array([5]), -1, 3)

    def test_empty_window_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            event_aligned_average(np.ones(10), np.array([5]), 0, 0)

    @pytest.mark.parametrize("baseline", [0, 30])
    def test_bad_baseline_raises(self, baseline):
        with pytest.raises(ValueError, match="baseline_frames"):
            event_aligned_average(np.ones(50), np.array([25]), 5, 5, baseline_frames=baseline)


# ---------------------------------------------------------------------------
# transient_index / onset_latency_frames
# ---------------------------------------------------------------------------


class TestTransientIndex:
    def test_pure_sustained_response_is_one(self):
        trace = np.concatenate([np.zeros(10), np.ones(20)])
        assert transient_index(trace, 10, 5, 5) == pytest.approx(1.0)

    def test_pure_transient_response_is_large(self):
        trace = np.concatenate([np.zeros(10), np.full(3, 1.0), np.full(17, 0.1)])
        assert transient_index(trace, 10, 3, 5) == pytest.approx(10.0)

    def test_zero_sustained_returns_nan(self):
        trace = np.concatenate([np.zeros(10), np.full(3, 1.0), np.zeros(17)])
        assert np.isnan(transient_index(trace, 10, 3, 5))

    def test_no_pre_window_uses_zero_baseline(self):
        trace = np.concatenate([np.full(3, 2.0), np.full(7, 1.0)])
        assert transient_index(trace, 0, 3, 5) == pytest.approx(2.0)

    def test_all_nan_windows_return_nan(self):
        trace = np.concatenate([np.zeros(10), np.full(20, np.nan)])
        assert np.isnan(transient_index(trace, 10, 5, 5))

    def test_nan_baseline_returns_nan(self):
        trace = np.concatenate([np.full(10, np.nan), np.ones(20)])
        assert np.isnan(transient_index(trace, 10, 5, 5))

    def test_bad_pre_frames_raises(self):
        with pytest.raises(ValueError, match="pre_frames must be"):
            transient_index(np.ones(10), 10, 2, 2)

    def test_non_positive_windows_raise(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            transient_index(np.ones(10), 2, 0, 2)

    def test_peak_window_too_long_raises(self):
        with pytest.raises(ValueError, match="peak_window_frames exceeds"):
            transient_index(np.ones(10), 5, 6, 2)

    def test_sustained_window_too_long_raises(self):
        with pytest.raises(ValueError, match="sustained_window_frames exceeds"):
            transient_index(np.ones(10), 5, 2, 6)


class TestOnsetLatencyFrames:
    def test_step_latency(self):
        trace = np.concatenate([np.zeros(10), np.zeros(5), np.ones(15)])
        assert onset_latency_frames(trace, 10) == pytest.approx(5.0)

    def test_immediate_response(self):
        trace = np.concatenate([np.zeros(10), np.ones(10)])
        assert onset_latency_frames(trace, 10) == pytest.approx(0.0)

    def test_flat_trace_returns_nan(self):
        assert np.isnan(onset_latency_frames(np.zeros(20), 10))

    def test_no_pre_window_uses_zero_baseline(self):
        trace = np.concatenate([np.zeros(4), np.ones(6)])
        assert onset_latency_frames(trace, 0) == pytest.approx(4.0)

    def test_all_nan_post_returns_nan(self):
        trace = np.concatenate([np.zeros(10), np.full(10, np.nan)])
        assert np.isnan(onset_latency_frames(trace, 10))

    def test_bad_pre_frames_raises(self):
        with pytest.raises(ValueError, match="pre_frames must be"):
            onset_latency_frames(np.ones(5), 5)

    @pytest.mark.parametrize("frac", [0.0, 1.5])
    def test_bad_threshold_raises(self, frac):
        with pytest.raises(ValueError, match="threshold_frac"):
            onset_latency_frames(np.ones(10), 2, threshold_frac=frac)


# ---------------------------------------------------------------------------
# immobility_decay
# ---------------------------------------------------------------------------


class TestImmobilityDecay:
    def test_exponential_decay_recovered(self):
        tau = 2.0
        n = 400
        active = np.ones(n, dtype=bool)
        active[50:200] = False
        signal = np.zeros(n)
        t = np.arange(150) / FPS
        signal[50:200] = np.exp(-t / tau)
        out = immobility_decay(signal, active, FPS)
        assert out["n_bouts"] == 1
        assert out["initial_value"] == pytest.approx(1.0)
        assert out["decay_time_s"] == pytest.approx(tau, abs=0.15)

    def test_multiple_bouts_pooled(self):
        active = _alternating_state(50, 8, start_true=True)
        signal = np.where(active, 1.0, 0.2)
        out = immobility_decay(signal, active, FPS)
        assert out["n_bouts"] == 4
        assert out["mean_timecourse"][0] == pytest.approx(0.2)
        assert np.isfinite(out["sem_timecourse"][0])

    def test_constant_signal_never_decays(self):
        active = np.ones(300, dtype=bool)
        active[50:250] = False
        signal = np.ones(300)
        out = immobility_decay(signal, active, FPS)
        assert np.isnan(out["decay_time_s"])

    def test_no_qualifying_bouts(self):
        active = np.ones(100, dtype=bool)
        out = immobility_decay(np.ones(100), active, FPS)
        assert out["n_bouts"] == 0
        assert np.isnan(out["decay_time_s"])
        assert np.all(np.isnan(out["mean_timecourse"]))

    def test_non_positive_initial_returns_nan(self):
        active = np.ones(300, dtype=bool)
        active[50:250] = False
        signal = np.full(300, -1.0)
        out = immobility_decay(signal, active, FPS)
        assert np.isnan(out["decay_time_s"])

    def test_bout_truncated_to_max(self):
        active = np.zeros(400, dtype=bool)
        signal = np.ones(400)
        out = immobility_decay(signal, active, FPS, min_bout_s=1.0, max_bout_s=5.0)
        assert out["mean_timecourse"].size == 50

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            immobility_decay(np.ones(10), np.ones(5, dtype=bool), FPS)

    def test_bad_bout_limits_raise(self):
        with pytest.raises(ValueError, match="min_bout_s"):
            immobility_decay(
                np.ones(10), np.ones(10, dtype=bool), FPS, min_bout_s=5.0, max_bout_s=1.0
            )


# ---------------------------------------------------------------------------
# activity_vs_bout_duration
# ---------------------------------------------------------------------------


class TestActivityVsBoutDuration:
    def _build(self):
        """Six movement bouts of increasing duration separated by immobility."""
        active = np.zeros(400, dtype=bool)
        signal = np.zeros(400)
        start = 5
        for i, length in enumerate([5, 8, 12, 20, 30, 45]):
            active[start : start + length] = True
            signal[start : start + length] = float(i + 1)
            start += length + 10
        return signal, active

    def test_bin_counts_and_values(self):
        signal, active = self._build()
        out = activity_vs_bout_duration(signal, active, FPS, n_duration_bins=3)
        assert out["bout_durations_s"].size == 6
        assert out["n_bouts_per_bin"].sum() == 6
        assert np.allclose(out["bout_means"], [1, 2, 3, 4, 5, 6])
        # Longer-duration bins hold the later (higher-valued) bouts.
        assert out["mean_signal"][-1] > out["mean_signal"][0]

    def test_bin_centers_are_median_durations(self):
        signal, active = self._build()
        out = activity_vs_bout_duration(signal, active, FPS, n_duration_bins=1)
        assert out["bin_centers_s"][0] == pytest.approx(np.median(out["bout_durations_s"]))

    def test_empty_bins_are_nan(self):
        active = np.zeros(50, dtype=bool)
        active[10:15] = True
        out = activity_vs_bout_duration(np.ones(50), active, FPS, n_duration_bins=4)
        assert out["n_bouts_per_bin"].sum() == 1
        assert np.isnan(out["mean_signal"]).sum() >= 3

    def test_no_bouts(self):
        out = activity_vs_bout_duration(np.ones(50), np.zeros(50, dtype=bool), FPS)
        assert out["bout_durations_s"].size == 0
        assert np.all(np.isnan(out["mean_signal"]))
        assert out["n_bouts_per_bin"].sum() == 0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            activity_vs_bout_duration(np.ones(10), np.ones(5, dtype=bool), FPS)

    def test_bad_bin_count_raises(self):
        with pytest.raises(ValueError, match="n_duration_bins"):
            activity_vs_bout_duration(np.ones(10), np.ones(10, dtype=bool), FPS, n_duration_bins=0)


# ---------------------------------------------------------------------------
# _sustained_ratio / movement_state_summary
# ---------------------------------------------------------------------------


class TestSustainedRatio:
    def test_sustained_signal_ratio_near_one(self):
        onsets = np.array([0, 100])
        offsets = np.array([80, 180])
        signal = np.ones(200)
        assert _sustained_ratio(signal, onsets, offsets, FPS, 4.0) == pytest.approx(1.0)

    def test_decaying_signal_ratio_below_one(self):
        onsets = np.array([0])
        offsets = np.array([80])
        signal = np.zeros(100)
        signal[:40] = 1.0
        signal[40:80] = 0.1
        assert _sustained_ratio(signal, onsets, offsets, FPS, 4.0) == pytest.approx(0.1)

    def test_no_long_bouts_returns_nan(self):
        onsets = np.array([0])
        offsets = np.array([5])
        assert np.isnan(_sustained_ratio(np.ones(50), onsets, offsets, FPS, 4.0))

    def test_zero_first_half_returns_nan(self):
        onsets = np.array([0])
        offsets = np.array([80])
        signal = np.zeros(100)
        signal[40:80] = 1.0
        assert np.isnan(_sustained_ratio(signal, onsets, offsets, FPS, 4.0))


class TestMovementStateSummary:
    def _sustained_session(self):
        active = _alternating_state(60, 10, start_true=False)
        signal = active.astype(float)
        bad = np.zeros(active.size, dtype=bool)
        return signal, active, bad

    def _transient_session(self):
        active = _alternating_state(60, 10, start_true=False)
        onsets, _ = segment_bouts(active)
        signal = np.zeros(active.size)
        t = np.arange(active.size) / FPS
        for onset in onsets:
            decay = np.exp(-(t - t[onset]) / 0.5)
            decay[: int(onset)] = 0.0
            signal += decay
        bad = np.zeros(active.size, dtype=bool)
        return signal, active, bad

    def test_sustained_session_has_flat_ratio(self):
        signal, active, bad = self._sustained_session()
        out = movement_state_summary(signal, active, bad, FPS)
        assert out["n_onsets"] > 0
        assert out["n_offsets"] > 0
        assert out["onset_transient_index"] == pytest.approx(1.0, abs=0.1)
        assert out["sustained_ratio"] == pytest.approx(1.0, abs=0.05)
        assert out["onset_latency_s"] == pytest.approx(0.0, abs=0.2)

    def test_transient_session_decays_within_bouts(self):
        signal, active, bad = self._transient_session()
        out = movement_state_summary(signal, active, bad, FPS)
        assert out["sustained_ratio"] < 0.5
        assert out["onset_transient_index"] > 1.5

    def test_time_axis_and_traces_match(self):
        signal, active, bad = self._sustained_session()
        out = movement_state_summary(signal, active, bad, FPS, pre_s=1.0, post_s=2.0)
        assert out["time_s"].size == out["onset_mean"].size == out["offset_mean"].size
        assert out["time_s"][0] == pytest.approx(-1.0)

    def test_bad_behav_frames_excluded(self):
        signal, active, _ = self._sustained_session()
        bad = np.ones(active.size, dtype=bool)
        out = movement_state_summary(signal, active, bad, FPS)
        assert np.isnan(out["onset_transient_index"])
        assert np.isnan(out["sustained_ratio"])
        assert np.isnan(out["immobility_decay_time_s"])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            movement_state_summary(
                np.ones(10), np.ones(10, dtype=bool), np.ones(5, dtype=bool), FPS
            )


# ---------------------------------------------------------------------------
# Syllable-conditioned activity
# ---------------------------------------------------------------------------


class TestSyllableConditionedActivity:
    def test_means_per_syllable(self):
        syl = np.repeat([0, 1], 50)
        signal = np.where(syl == 0, 1.0, 3.0)
        mask = np.ones(100, dtype=bool)
        out = syllable_conditioned_activity(signal, syl, mask, min_frames=10)
        assert out == {0: pytest.approx(1.0), 1: pytest.approx(3.0)}

    def test_min_frames_filter(self):
        syl = np.concatenate([np.zeros(50, dtype=int), np.ones(5, dtype=int)])
        signal = np.ones(55)
        mask = np.ones(55, dtype=bool)
        out = syllable_conditioned_activity(signal, syl, mask, min_frames=10)
        assert set(out) == {0}

    def test_negative_labels_skipped(self):
        syl = np.concatenate([np.full(30, -1), np.zeros(30, dtype=int)])
        signal = np.ones(60)
        mask = np.ones(60, dtype=bool)
        out = syllable_conditioned_activity(signal, syl, mask, min_frames=5)
        assert set(out) == {0}

    def test_mask_applied(self):
        syl = np.repeat([0, 1], 50)
        signal = np.ones(100)
        mask = syl == 0
        out = syllable_conditioned_activity(signal, syl, mask, min_frames=5)
        assert set(out) == {0}

    def test_nan_frames_ignored(self):
        syl = np.zeros(40, dtype=int)
        signal = np.ones(40)
        signal[:10] = np.nan
        mask = np.ones(40, dtype=bool)
        out = syllable_conditioned_activity(signal, syl, mask, min_frames=5)
        assert out[0] == pytest.approx(1.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            syllable_conditioned_activity(np.ones(10), np.zeros(5, dtype=int), np.ones(10, bool))

    def test_bad_min_frames_raises(self):
        with pytest.raises(ValueError, match="min_frames"):
            syllable_conditioned_activity(
                np.ones(10), np.zeros(10, dtype=int), np.ones(10, bool), min_frames=0
            )


class TestSyllableInformation:
    def test_deterministic_mapping_is_informative(self):
        syl = np.repeat([0, 1, 2], 100)
        signal = syl.astype(float) * 10.0
        mask = np.ones(300, dtype=bool)
        mi = syllable_information(signal, syl, mask, n_signal_bins=3)
        assert mi == pytest.approx(np.log2(3), abs=0.05)

    def test_independent_signal_has_low_information(self):
        rng = np.random.default_rng(0)
        syl = rng.integers(0, 3, 3000)
        signal = rng.normal(size=3000)
        mask = np.ones(3000, dtype=bool)
        mi = syllable_information(signal, syl, mask, n_signal_bins=10)
        assert 0.0 <= mi < 0.1

    def test_single_syllable_returns_zero(self):
        syl = np.zeros(100, dtype=int)
        mi = syllable_information(np.arange(100.0), syl, np.ones(100, dtype=bool))
        assert mi == 0.0

    def test_constant_signal_returns_zero(self):
        syl = np.repeat([0, 1], 50)
        mi = syllable_information(np.ones(100), syl, np.ones(100, dtype=bool))
        assert mi == 0.0

    def test_too_few_valid_frames_returns_zero(self):
        syl = np.array([0, 1, 2], dtype=int)
        mask = np.array([True, False, False])
        mi = syllable_information(np.arange(3.0), syl, mask)
        assert mi == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            syllable_information(np.ones(10), np.zeros(5, dtype=int), np.ones(10, bool))

    def test_bad_bin_count_raises(self):
        with pytest.raises(ValueError, match="n_signal_bins"):
            syllable_information(
                np.ones(10), np.zeros(10, dtype=int), np.ones(10, bool), n_signal_bins=1
            )

    @settings(max_examples=25, deadline=None)
    @given(
        st.lists(
            st.floats(min_value=-10, max_value=10, allow_nan=False), min_size=40, max_size=120
        )
    )
    def test_information_is_non_negative(self, values):
        signal = np.array(values, dtype=float)
        syl = np.arange(signal.size) % 3
        mask = np.ones(signal.size, dtype=bool)
        assert syllable_information(signal, syl, mask, n_signal_bins=4) >= 0.0
