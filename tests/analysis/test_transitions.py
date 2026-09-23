"""Tests for hm2p.analysis.transitions — light/dark transition responses."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis.transitions import (
    _window_mask,
    first_vs_later_epochs,
    population_transition_response,
    recovery_time_s,
    transition_aligned_responses,
    transition_transient,
    tuning_recovery_timecourse,
)

FPS = 10.0
BLOCK = 40  # frames per light or dark epoch (4 s at 10 fps)
N_BLOCKS = 20


def _light_blocks(block: int = BLOCK, n_blocks: int = N_BLOCKS) -> np.ndarray:
    """Alternating light/dark epochs starting with light."""
    return np.concatenate([np.full(block, i % 2 == 0, dtype=bool) for i in range(n_blocks)])


def _step_signal(light_on: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    """Signal that steps to ``amplitude`` whenever the lights go out."""
    return np.where(light_on, 0.0, amplitude)


class TestWindowMask:
    def test_half_open_interval(self):
        time_s = np.array([-1.0, 0.0, 1.0, 2.0])
        mask = _window_mask(time_s, (0.0, 2.0))
        assert mask.tolist() == [False, True, True, False]

    @pytest.mark.parametrize("bounds", [(1.0, 1.0), (2.0, 1.0)])
    def test_invalid_bounds_raise(self, bounds):
        with pytest.raises(ValueError, match="lo < hi"):
            _window_mask(np.arange(3.0), bounds)


class TestTransitionAlignedResponses:
    def test_step_amplitude_recovered(self):
        light = _light_blocks()
        signal = _step_signal(light, amplitude=2.5)
        bad = np.zeros(light.size, dtype=bool)
        out = transition_aligned_responses(
            signal, light, bad, FPS, pre_s=2.0, post_s=3.0, direction="light_to_dark"
        )
        assert out["n_transitions"] >= 8
        pre = out["time_s"] < 0
        post = out["time_s"] >= 0
        assert np.allclose(out["mean"][pre], 0.0, atol=1e-9)
        assert np.allclose(out["mean"][post], 2.5, atol=1e-9)

    def test_dark_to_light_direction(self):
        light = _light_blocks()
        signal = _step_signal(light, amplitude=2.0)
        bad = np.zeros(light.size, dtype=bool)
        out = transition_aligned_responses(
            signal, light, bad, FPS, pre_s=2.0, post_s=3.0, direction="dark_to_light"
        )
        post = out["time_s"] >= 0
        assert np.allclose(out["mean"][post], -2.0, atol=1e-9)
        assert out["direction"] == "dark_to_light"

    def test_window_shapes_and_time_axis(self):
        light = _light_blocks()
        signal = _step_signal(light)
        bad = np.zeros(light.size, dtype=bool)
        out = transition_aligned_responses(signal, light, bad, FPS, pre_s=1.0, post_s=2.0)
        assert out["pre_frames"] == 10
        assert out["windows"].shape == (out["n_transitions"], 30)
        assert out["time_s"][0] == pytest.approx(-1.0)
        assert out["sem"].size == 30

    def test_bad_behav_frames_excluded(self):
        light = _light_blocks()
        signal = _step_signal(light)
        bad = ~light  # every dark frame excluded
        out = transition_aligned_responses(signal, light, bad, FPS, pre_s=2.0, post_s=3.0)
        post = out["time_s"] >= 0
        assert np.all(np.isnan(out["mean"][post]))

    def test_no_transitions(self):
        light = np.ones(200, dtype=bool)
        out = transition_aligned_responses(
            np.ones(200), light, np.zeros(200, dtype=bool), FPS, pre_s=1.0, post_s=1.0
        )
        assert out["n_transitions"] == 0
        assert np.all(np.isnan(out["mean"]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            transition_aligned_responses(
                np.ones(10), np.ones(10, dtype=bool), np.ones(5, dtype=bool), FPS
            )

    def test_bad_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be"):
            transition_aligned_responses(
                np.ones(10),
                np.ones(10, dtype=bool),
                np.zeros(10, dtype=bool),
                FPS,
                direction="sideways",
            )

    def test_bad_fps_raises(self):
        with pytest.raises(ValueError, match="fps must be"):
            transition_aligned_responses(
                np.ones(10), np.ones(10, dtype=bool), np.zeros(10, dtype=bool), -1.0
            )


class TestTransitionTransient:
    def _aligned(self, amplitude: float = 1.0, direction: str = "light_to_dark") -> dict:
        light = _light_blocks()
        signal = _step_signal(light, amplitude=amplitude)
        bad = np.zeros(light.size, dtype=bool)
        return transition_aligned_responses(
            signal, light, bad, FPS, pre_s=1.0, post_s=3.0, direction=direction
        )

    def test_positive_step_amplitudes_and_sign(self):
        out = transition_transient(self._aligned(1.0), FPS, early_s=(0.0, 1.0), late_s=(2.0, 3.0))
        assert out["early_amplitude"] == pytest.approx(1.0)
        assert out["late_amplitude"] == pytest.approx(1.0)
        assert out["sign"] == 1
        assert out["p_value"] < 0.05
        assert out["early_per_transition"].size == out["n_transitions"]

    def test_negative_step_sign(self):
        out = transition_transient(
            self._aligned(1.0, direction="dark_to_light"),
            FPS,
            early_s=(0.0, 1.0),
            late_s=(2.0, 3.0),
        )
        assert out["early_amplitude"] == pytest.approx(-1.0)
        assert out["sign"] == -1

    def test_few_transitions_give_nan_p(self):
        light = np.concatenate([np.ones(60, dtype=bool), np.zeros(60, dtype=bool)])
        signal = _step_signal(light)
        aligned = transition_aligned_responses(
            signal, light, np.zeros(light.size, dtype=bool), FPS, pre_s=1.0, post_s=3.0
        )
        out = transition_transient(aligned, FPS, early_s=(0.0, 1.0), late_s=(2.0, 3.0))
        assert out["n_transitions"] == 1
        assert np.isnan(out["p_value"])
        assert out["sign"] == 0

    def test_zero_response_gives_nan_p(self):
        light = _light_blocks()
        aligned = transition_aligned_responses(
            np.ones(light.size),
            light,
            np.zeros(light.size, dtype=bool),
            FPS,
            pre_s=1.0,
            post_s=3.0,
        )
        out = transition_transient(aligned, FPS, early_s=(0.0, 1.0), late_s=(2.0, 3.0))
        assert out["early_amplitude"] == pytest.approx(0.0)
        assert np.isnan(out["p_value"])
        assert out["sign"] == 0

    def test_no_transitions_returns_nan(self):
        light = np.ones(200, dtype=bool)
        aligned = transition_aligned_responses(
            np.ones(200), light, np.zeros(200, dtype=bool), FPS, pre_s=1.0, post_s=3.0
        )
        out = transition_transient(aligned, FPS, early_s=(0.0, 1.0), late_s=(2.0, 3.0))
        assert out["n_transitions"] == 0
        assert np.isnan(out["early_amplitude"])
        assert out["early_per_transition"].size == 0

    def test_window_outside_time_axis_returns_nan(self):
        aligned = self._aligned()
        out = transition_transient(aligned, FPS, early_s=(50.0, 60.0), late_s=(2.0, 3.0))
        assert np.isnan(out["early_amplitude"])

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="aligned must contain"):
            transition_transient({"time_s": np.arange(3.0)}, FPS)

    def test_bad_fps_raises(self):
        with pytest.raises(ValueError, match="fps must be"):
            transition_transient(self._aligned(), 0.0)


class TestPopulationTransitionResponse:
    def test_per_roi_amplitudes(self):
        light = _light_blocks()
        bad = np.zeros(light.size, dtype=bool)
        signals = np.vstack(
            [
                _step_signal(light, amplitude=1.0),
                _step_signal(light, amplitude=-2.0),
                np.zeros(light.size),
            ]
        )
        out = population_transition_response(
            signals, light, bad, FPS, pre_s=1.0, post_s=3.0, early_s=(0.0, 1.0), late_s=(2.0, 3.0)
        )
        assert out["n_rois"] == 3
        assert out["early_amplitude"] == pytest.approx([1.0, -2.0, 0.0])
        assert out["sign"].tolist() == [1, -1, 0]
        assert out["p_value"][0] < 0.05
        assert np.isnan(out["p_value"][2])

    def test_population_timecourse_shapes(self):
        light = _light_blocks()
        bad = np.zeros(light.size, dtype=bool)
        signals = np.vstack([_step_signal(light, 1.0), _step_signal(light, 3.0)])
        out = population_transition_response(signals, light, bad, FPS, pre_s=1.0, post_s=2.0)
        assert out["mean_timecourse"].size == out["time_s"].size == 30
        post = out["time_s"] >= 0
        assert np.allclose(out["mean_timecourse"][post], 2.0)
        assert np.all(np.isfinite(out["sem_timecourse"][post]))
        assert out["n_transitions"] >= 8

    def test_empty_population(self):
        light = _light_blocks()
        out = population_transition_response(
            np.zeros((0, light.size)), light, np.zeros(light.size, dtype=bool), FPS
        )
        assert out["n_rois"] == 0
        assert out["mean_timecourse"].size == 0
        assert out["early_amplitude"].size == 0

    def test_one_dimensional_signals_raise(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            population_transition_response(
                np.ones(10), np.ones(10, dtype=bool), np.zeros(10, dtype=bool), FPS
            )

    def test_frame_mismatch_raises(self):
        with pytest.raises(ValueError, match="must match the frame axis"):
            population_transition_response(
                np.ones((2, 10)), np.ones(5, dtype=bool), np.zeros(5, dtype=bool), FPS
            )


class TestTuningRecoveryTimecourse:
    def _session(self, n: int = 2400, pd_deg: float = 90.0):
        hd = (np.arange(n) * 37.0) % 360.0
        light = (np.arange(n) // 300) % 2 == 1
        signal = 1.0 + np.cos(np.deg2rad(hd - pd_deg))
        mask = np.ones(n, dtype=bool)
        return signal, hd, mask, light

    def test_stable_tuning_has_small_deviation(self):
        signal, hd, mask, light = self._session()
        out = tuning_recovery_timecourse(
            signal, hd, mask, light, FPS, window_s=10.0, step_s=5.0, post_s=20.0
        )
        assert out["time_s"].tolist() == [0.0, 5.0, 10.0, 15.0, 20.0]
        assert out["n_transitions"] >= 3
        assert out["reference_pd"] == pytest.approx(90.0, abs=10.0)
        assert np.nanmax(out["pd_deviation_deg"]) < 30.0
        assert np.nanmin(out["mvl"]) > 0.1

    def test_explicit_reference_pd_used(self):
        signal, hd, mask, light = self._session()
        out = tuning_recovery_timecourse(
            signal,
            hd,
            mask,
            light,
            FPS,
            reference_pd=270.0,
            window_s=10.0,
            step_s=10.0,
            post_s=10.0,
        )
        assert out["reference_pd"] == pytest.approx(270.0)
        # Tuning is at 90 deg, so the deviation from a 270 deg reference is large.
        assert np.nanmin(out["pd_deviation_deg"]) > 120.0

    def test_sparse_mask_falls_back_to_zero_reference(self):
        signal, hd, mask, light = self._session(n=600)
        sparse = np.zeros(mask.size, dtype=bool)
        sparse[:5] = True
        out = tuning_recovery_timecourse(
            signal, hd, sparse, light, FPS, window_s=5.0, step_s=5.0, post_s=5.0
        )
        assert out["reference_pd"] == 0.0
        assert np.all(np.isnan(out["mvl"]))
        assert out["n_transitions"] == 0

    def test_no_transitions(self):
        signal, hd, mask, _ = self._session(n=600)
        light = np.ones(600, dtype=bool)
        out = tuning_recovery_timecourse(
            signal, hd, mask, light, FPS, window_s=5.0, step_s=5.0, post_s=5.0
        )
        assert out["n_transitions"] == 0
        assert np.all(np.isnan(out["mvl"]))
        assert np.all(np.isnan(out["pd_deviation_deg"]))

    def test_windows_beyond_recording_are_skipped(self):
        signal, hd, mask, light = self._session(n=900)
        out = tuning_recovery_timecourse(
            signal, hd, mask, light, FPS, window_s=10.0, step_s=10.0, post_s=200.0
        )
        assert np.isnan(out["mvl"][-1])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            tuning_recovery_timecourse(
                np.ones(10), np.ones(5), np.ones(10, dtype=bool), np.ones(10, dtype=bool), FPS
            )

    @pytest.mark.parametrize("kwargs", [{"window_s": 0.0}, {"step_s": 0.0}, {"post_s": -1.0}])
    def test_bad_window_parameters_raise(self, kwargs):
        signal, hd, mask, light = self._session(n=300)
        with pytest.raises(ValueError, match="require window_s"):
            tuning_recovery_timecourse(signal, hd, mask, light, FPS, **kwargs)


class TestRecoveryTimeS:
    def test_downward_crossing(self):
        time_s = np.arange(5.0)
        curve = np.array([10.0, 8.0, 4.0, 1.0, 0.0])
        # target = 10 + 0.5 * (0 - 10) = 5
        assert recovery_time_s(time_s, curve) == pytest.approx(2.0)

    def test_upward_crossing(self):
        time_s = np.arange(5.0)
        curve = np.array([0.0, 0.1, 0.3, 0.5, 0.6])
        # target = 0 + 0.5 * 0.6 = 0.3
        assert recovery_time_s(time_s, curve, direction="up") == pytest.approx(2.0)

    def test_nan_samples_skipped(self):
        time_s = np.arange(5.0)
        curve = np.array([10.0, np.nan, 4.0, np.nan, 0.0])
        assert recovery_time_s(time_s, curve) == pytest.approx(2.0)

    def test_all_nan_returns_nan(self):
        assert np.isnan(recovery_time_s(np.arange(4.0), np.full(4, np.nan)))

    def test_single_finite_point_returns_nan(self):
        curve = np.array([1.0, np.nan, np.nan])
        assert np.isnan(recovery_time_s(np.arange(3.0), curve))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            recovery_time_s(np.arange(3.0), np.arange(4.0))

    @pytest.mark.parametrize("frac", [0.0, 1.5])
    def test_bad_target_frac_raises(self, frac):
        with pytest.raises(ValueError, match="target_frac"):
            recovery_time_s(np.arange(3.0), np.arange(3.0), target_frac=frac)

    def test_bad_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be"):
            recovery_time_s(np.arange(3.0), np.arange(3.0), direction="sideways")

    @settings(max_examples=30, deadline=None)
    @given(
        st.lists(st.floats(min_value=-5, max_value=5, allow_nan=False), min_size=2, max_size=30)
    )
    def test_result_is_within_time_range(self, values):
        curve = np.array(values, dtype=float)
        time_s = np.arange(curve.size, dtype=float)
        out = recovery_time_s(time_s, curve)
        assert np.isnan(out) or (time_s[0] <= out <= time_s[-1])


class TestFirstVsLaterEpochs:
    def _session(self, first_value: float = 5.0, later_value: float = 1.0):
        light = _light_blocks(block=50, n_blocks=8)
        signal = np.where(light, 0.0, later_value)
        onset = int(np.flatnonzero(~light)[0])
        signal[onset : onset + 50] = first_value
        bad = np.zeros(light.size, dtype=bool)
        return signal, light, bad

    def test_adaptation_detected(self):
        signal, light, bad = self._session()
        out = first_vs_later_epochs(signal, light, bad, FPS, np.nanmean, min_epoch_s=1.0)
        assert out["n_epochs"] == 4
        assert out["first_value"] == pytest.approx(5.0)
        assert out["later_mean"] == pytest.approx(1.0)
        assert out["difference"] == pytest.approx(4.0)
        assert out["later_values"].size == 3

    def test_bad_behav_frames_excluded(self):
        signal, light, _ = self._session()
        bad = np.zeros(light.size, dtype=bool)
        onset = int(np.flatnonzero(~light)[0])
        bad[onset : onset + 25] = True
        signal[onset : onset + 25] = 100.0
        out = first_vs_later_epochs(signal, light, bad, FPS, np.nanmean, min_epoch_s=1.0)
        assert out["first_value"] == pytest.approx(5.0)

    def test_single_dark_epoch(self):
        light = np.concatenate([np.ones(50, dtype=bool), np.zeros(50, dtype=bool)])
        signal = np.ones(100)
        out = first_vs_later_epochs(
            signal, light, np.zeros(100, dtype=bool), FPS, np.nanmean, min_epoch_s=1.0
        )
        assert out["n_epochs"] == 1
        assert np.isnan(out["later_mean"])
        assert np.isnan(out["difference"])

    def test_no_dark_epochs(self):
        light = np.ones(100, dtype=bool)
        out = first_vs_later_epochs(
            np.ones(100), light, np.zeros(100, dtype=bool), FPS, np.nanmean
        )
        assert out["n_epochs"] == 0
        assert np.isnan(out["first_value"])
        assert out["later_values"].size == 0

    def test_short_epochs_dropped(self):
        light = np.ones(100, dtype=bool)
        light[10:12] = False
        light[40:80] = False
        out = first_vs_later_epochs(
            np.ones(100), light, np.zeros(100, dtype=bool), FPS, np.nanmean, min_epoch_s=1.0
        )
        assert out["n_epochs"] == 1

    def test_custom_metric_function(self):
        signal, light, bad = self._session()
        out = first_vs_later_epochs(
            signal, light, bad, FPS, lambda seg: float(np.nanmax(seg)), min_epoch_s=1.0
        )
        assert out["first_value"] == pytest.approx(5.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            first_vs_later_epochs(
                np.ones(10), np.ones(5, dtype=bool), np.zeros(5, dtype=bool), FPS, np.nanmean
            )

    def test_non_callable_metric_raises(self):
        with pytest.raises(ValueError, match="metric_fn must be callable"):
            first_vs_later_epochs(
                np.ones(10), np.ones(10, dtype=bool), np.zeros(10, dtype=bool), FPS, "mean"
            )
