"""Tests for hm2p.sync.diagnostics — pure-function diagnostics module.

Covers all functions per the test plan in
``tests/sync/TEST_PLAN.md`` §1.1, §3, §9. Hypothesis property tests
target boundary behaviour around each threshold.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hm2p.sync.diagnostics import (
    _CODE_LUT,
    _DEFAULT_CONFIG,
    FLOAT_SENTINEL,
    INT_SENTINEL,
    ChannelScalars,
    CrossChannelScalars,
    LightScalars,
    SyncScalars,
    build_scalars,
    channel_scalars,
    classify,
    code_message,
    cross_channel_scalars,
    decode_codes_json,
    drift_slope,
    encode_codes_json,
    infer_light_polarity_ok,
    light_scalars,
    load_config,
    scalars_to_diag_attrs,
)
from tests.sync.conftest import (
    synthetic_clean_pulse_train,
    synthetic_corrupted_pulse_train,
    synthetic_drifted_pulse_train,
)

# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_no_path_returns_defaults(self) -> None:
        cfg = load_config(None)
        assert cfg["hard"]["frame_count_diff_max"] == 5

    def test_missing_path_returns_defaults(self, tmp_path) -> None:
        cfg = load_config(tmp_path / "does-not-exist.yaml")
        assert cfg["hard"]["temporal_overlap_min_frac"] == pytest.approx(0.95)

    def test_loads_yaml_file(self, tmp_path) -> None:
        path = tmp_path / "sync.yaml"
        path.write_text("hard:\n  frame_count_diff_max: 9\n")
        cfg = load_config(path)
        assert cfg["hard"]["frame_count_diff_max"] == 9
        # Other defaults are preserved.
        assert cfg["warn"]["cv_cam_max"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# drift_slope
# ---------------------------------------------------------------------------


class TestDriftSlope:
    def test_constant_isi_zero_drift(self) -> None:
        times = np.linspace(0.0, 10.0, 1001, dtype=np.float64)
        slope_ppm, r2 = drift_slope(times)
        assert abs(slope_ppm) < 1e-3
        assert r2 == pytest.approx(1.0, abs=1e-9)

    def test_recovers_known_drift(self) -> None:
        rng = np.random.default_rng(0)
        for ppm in (50, 100, 500, 1000, -200):
            times = synthetic_drifted_pulse_train(rng, fps=100.0, duration_s=10.0, drift_ppm=ppm)
            # Comparison must be against nominal fps, not median ISI: a
            # uniform multiplicative drift moves slope AND median together,
            # so ppm-vs-median is always ~0. ppm-vs-nominal recovers the
            # input drift.
            slope_ppm, r2 = drift_slope(times, fps_nominal=100.0)
            assert abs(slope_ppm - ppm) <= max(2.0, abs(ppm) * 0.05), (ppm, slope_ppm)
            assert r2 > 0.999

    def test_empty_returns_nan(self) -> None:
        slope_ppm, r2 = drift_slope(np.empty(0))
        assert np.isnan(slope_ppm)
        assert np.isnan(r2)

    def test_single_element_returns_nan(self) -> None:
        slope_ppm, r2 = drift_slope(np.array([1.0]))
        assert np.isnan(slope_ppm)
        assert np.isnan(r2)

    def test_constant_time_returns_nan(self) -> None:
        times = np.zeros(10, dtype=np.float64)
        slope_ppm, r2 = drift_slope(times)
        assert np.isnan(slope_ppm)
        assert np.isnan(r2)

    @given(
        n=st.integers(min_value=10, max_value=2000),
        seed=st.integers(min_value=0, max_value=9999),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_r2_in_unit_interval(self, n, seed) -> None:
        rng = np.random.default_rng(seed)
        times = np.cumsum(rng.uniform(1e-3, 1e-1, size=n))
        slope_ppm, r2 = drift_slope(times)
        assert np.isfinite(slope_ppm)
        assert 0.0 <= r2 <= 1.0


# ---------------------------------------------------------------------------
# channel_scalars
# ---------------------------------------------------------------------------


class TestChannelScalars:
    def test_clean_train(self) -> None:
        times = synthetic_clean_pulse_train(
            np.random.default_rng(0), fps=100.0, duration_s=10.0, jitter_ms=0.0
        )
        s = channel_scalars(times, fps_nominal=100.0)
        assert s.n_pulses == times.size
        assert s.isi_median_ms == pytest.approx(10.0, abs=1e-6)
        assert s.isi_cv == pytest.approx(0.0, abs=1e-9)
        assert s.n_isi_outliers == 0
        assert abs(s.drift_slope_ppm) < 0.01

    def test_perfectly_uniform_train_no_false_outliers(self) -> None:
        """QA 2.5 — perfectly uniform synthetic train must report 0 outliers.

        ``np.diff(np.linspace(...))`` on float64 introduces floating-point
        noise on the order of 10^-11 s at 100 Hz. The previous tolerance
        of 1e-9 × median_isi (~10^-11 for 100 Hz) was BELOW that noise
        floor, so a perfectly uniform synthetic train falsely reported
        many outliers when MAD == 0. The fix uses 1e-6 × median_isi.
        """
        # Construct a perfectly uniform 100 Hz train via cumsum of a
        # constant ISI — the cleanest possible pulse train.
        n = 1000
        isi = 1.0 / 100.0
        times = np.arange(n, dtype=np.float64) * isi
        s = channel_scalars(times, fps_nominal=100.0)
        # MAD is exactly 0 → triggers the QA 2.5 code path.
        assert s.isi_mad_ms == pytest.approx(0.0, abs=1e-12)
        assert s.n_isi_outliers == 0, (
            f"Expected 0 outliers for a perfectly uniform train; got "
            f"{s.n_isi_outliers}. Tolerance is too tight for float64 noise."
        )

    def test_perfectly_uniform_train_one_real_outlier_detected(self) -> None:
        """A genuine deviation (10 % of ISI) is still detected when MAD == 0.

        Pin the boundary: 1 ppm tolerance of the median is small enough
        that any meaningful corruption is still flagged as an outlier.
        """
        n = 1000
        isi = 1.0 / 100.0
        times = np.arange(n, dtype=np.float64) * isi
        # Shift one pulse by 10 % of ISI — this is far above 1 ppm.
        times[500] += 0.1 * isi
        s = channel_scalars(times, fps_nominal=100.0)
        # Single corrupted ISI affects two diffs (at index 499 and 500).
        assert s.n_isi_outliers >= 1

    def test_jitter_increases_cv(self) -> None:
        rng = np.random.default_rng(0)
        clean = channel_scalars(
            synthetic_clean_pulse_train(rng, fps=100.0, duration_s=10.0, jitter_ms=0.0),
            fps_nominal=100.0,
        )
        jittery = channel_scalars(
            synthetic_clean_pulse_train(rng, fps=100.0, duration_s=10.0, jitter_ms=3.0),
            fps_nominal=100.0,
        )
        assert jittery.isi_cv > clean.isi_cv

    def test_duplicate_pulse_low_min_isi(self) -> None:
        rng = np.random.default_rng(0)
        times = synthetic_corrupted_pulse_train(
            rng, fps=100.0, duration_s=10.0, duplicate_idxs=(100,)
        )
        s = channel_scalars(times, fps_nominal=100.0)
        assert s.min_isi_ms < 1e-2  # nearly zero
        assert s.n_isi_outliers >= 1

    def test_missing_pulse_outlier_count(self) -> None:
        rng = np.random.default_rng(0)
        times = synthetic_corrupted_pulse_train(
            rng, fps=100.0, duration_s=10.0, missing_idxs=(100,)
        )
        s = channel_scalars(times, fps_nominal=100.0)
        # Median ISI is still ~10 ms; one ISI is 2× → flagged outlier.
        assert s.isi_median_ms == pytest.approx(10.0, abs=0.5)
        assert s.n_isi_outliers >= 1

    def test_empty_returns_zero_n(self) -> None:
        s = channel_scalars(np.empty(0), fps_nominal=100.0)
        assert s.n_pulses == 0
        assert np.isnan(s.isi_median_ms)

    def test_single_element(self) -> None:
        s = channel_scalars(np.array([1.0]), fps_nominal=100.0)
        assert s.n_pulses == 1
        assert s.duration_s == 0.0
        assert np.isnan(s.isi_median_ms)

    def test_two_elements(self) -> None:
        s = channel_scalars(np.array([0.0, 0.01]), fps_nominal=100.0)
        assert s.n_pulses == 2
        assert s.isi_median_ms == pytest.approx(10.0)
        assert s.isi_cv == pytest.approx(0.0)

    def test_constant_time_zero_cv_or_nan(self) -> None:
        s = channel_scalars(np.zeros(10, dtype=np.float64), fps_nominal=100.0)
        # Median ISI = 0 → CV undefined → sentinel
        assert np.isnan(s.isi_cv)

    @given(
        fps=st.floats(min_value=10.0, max_value=200.0),
        duration_s=st.floats(min_value=2.0, max_value=60.0),
        jitter_ms=st.floats(min_value=0.0, max_value=2.0),
        seed=st.integers(min_value=0, max_value=9999),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_median_close_to_inverse_fps(self, fps, duration_s, jitter_ms, seed) -> None:
        rng = np.random.default_rng(seed)
        times = synthetic_clean_pulse_train(
            rng, fps=fps, duration_s=duration_s, jitter_ms=jitter_ms
        )
        s = channel_scalars(times, fps_nominal=fps)
        expected_ms = 1000.0 / fps
        # Median should be close to nominal — non-parametric is robust.
        assert abs(s.isi_median_ms - expected_ms) < max(0.5, 5 * jitter_ms)
        assert s.isi_cv >= 0.0
        assert s.n_isi_outliers >= 0


# ---------------------------------------------------------------------------
# cross_channel_scalars
# ---------------------------------------------------------------------------


class TestCrossChannelScalars:
    def test_identical_spans(self) -> None:
        a = np.linspace(0, 6, 600, dtype=np.float64)
        c = cross_channel_scalars(a, a)
        assert c.overlap_s == pytest.approx(6.0)
        assert c.start_offset_ms == pytest.approx(0.0)
        assert c.end_offset_ms == pytest.approx(0.0)

    def test_disjoint(self) -> None:
        cam = np.linspace(0, 5, 500, dtype=np.float64)
        img = np.linspace(10, 15, 50, dtype=np.float64)
        c = cross_channel_scalars(cam, img)
        assert c.overlap_s == 0.0
        assert c.start_offset_ms == pytest.approx(10000.0)

    def test_partial_overlap(self) -> None:
        cam = np.linspace(0, 10, 1000, dtype=np.float64)
        img = np.linspace(5, 15, 100, dtype=np.float64)
        c = cross_channel_scalars(cam, img)
        assert c.overlap_s == pytest.approx(5.0)
        assert c.start_offset_ms == pytest.approx(5000.0)
        assert c.end_offset_ms == pytest.approx(5000.0)

    def test_empty_inputs(self) -> None:
        c = cross_channel_scalars(np.empty(0), np.linspace(0, 1, 10))
        assert np.isnan(c.overlap_s)


# ---------------------------------------------------------------------------
# light_scalars
# ---------------------------------------------------------------------------


class TestLightScalars:
    def test_regular_60_60(self) -> None:
        on = np.array([120.0, 240.0, 360.0], dtype=np.float64)  # 3 cycles
        off = np.array([60.0, 180.0, 300.0], dtype=np.float64)
        s = light_scalars(on, off, duration_s=480.0)
        assert s.n_on == 3
        assert s.n_off == 3
        assert s.period_median_s == pytest.approx(120.0)
        assert s.first_state_at_t0 == 1  # off comes first → was on
        # Duty cycle: integrate state. Walking edges starting state=1:
        # [0,60] on (60s), [60,120] off (60s), [120,180] on, [180,240] off,
        # [240,300] on, [300,360] off, [360,480] on (120s) → on_total=300/480
        assert s.duty_cycle == pytest.approx(300 / 480, abs=1e-6)

    def test_empty(self) -> None:
        s = light_scalars(np.empty(0), np.empty(0), duration_s=60.0)
        assert s.n_on == 0
        assert s.first_state_at_t0 == -1
        assert np.isnan(s.duty_cycle)

    def test_first_edge_on_means_was_off(self) -> None:
        on = np.array([5.0], dtype=np.float64)
        off = np.array([20.0], dtype=np.float64)
        s = light_scalars(on, off, duration_s=30.0)
        assert s.first_state_at_t0 == 0

    def test_mismatched_counts_handled(self) -> None:
        on = np.array([10.0, 130.0], dtype=np.float64)
        off = np.array([70.0], dtype=np.float64)
        s = light_scalars(on, off, duration_s=200.0)
        assert s.n_on == 2
        assert s.n_off == 1

    def test_negative_time_edges_dropped_not_double_counted(self) -> None:
        """QA 2.4 — pre-window edges must not freeze the prior state.

        Construct a clean cycle plus one spurious edge at t = -10.
        The previous ``if t < 0: continue`` inside the integration loop
        kept the prior state through to the next non-negative edge,
        which double-counted the time before that edge in the wrong
        state. Dropping negative edges before the walk fixes this.
        """
        # Standard cycle without the negative edge: on=[120, 240], off=[60, 180]
        on_clean = np.array([120.0, 240.0], dtype=np.float64)
        off_clean = np.array([60.0, 180.0], dtype=np.float64)
        s_clean = light_scalars(on_clean, off_clean, duration_s=300.0)

        # Same cycle plus one rogue light_off at t=-10 (e.g. DAQ
        # pre-trigger artefact).
        on_rogue = on_clean.copy()
        off_rogue = np.array([-10.0, 60.0, 180.0], dtype=np.float64)
        s_rogue = light_scalars(on_rogue, off_rogue, duration_s=300.0)

        # Both should produce the same duty cycle once the negative edge
        # is correctly excluded from the integration walk.
        assert s_rogue.duty_cycle == pytest.approx(s_clean.duty_cycle, abs=1e-6)

    def test_polarity_helper_in_range(self) -> None:
        s = LightScalars(duty_cycle=0.5)
        assert infer_light_polarity_ok(s, _DEFAULT_CONFIG)

    def test_polarity_helper_out_of_range(self) -> None:
        s = LightScalars(duty_cycle=1.0)
        assert not infer_light_polarity_ok(s, _DEFAULT_CONFIG)
        s2 = LightScalars(duty_cycle=FLOAT_SENTINEL)
        assert not infer_light_polarity_ok(s2, _DEFAULT_CONFIG)

    @given(
        on=st.lists(st.floats(min_value=0, max_value=1000), min_size=0, max_size=20),
        off=st.lists(st.floats(min_value=0, max_value=1000), min_size=0, max_size=20),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_first_state_in_valid_set(self, on, off) -> None:
        on_arr = np.sort(np.asarray(on, dtype=np.float64))
        off_arr = np.sort(np.asarray(off, dtype=np.float64))
        s = light_scalars(on_arr, off_arr, duration_s=1000.0)
        assert s.first_state_at_t0 in (-1, 0, 1)


# ---------------------------------------------------------------------------
# classify — tier predicates (test plan §3.1)
# ---------------------------------------------------------------------------


def _ok_scalars() -> SyncScalars:
    """Return a SyncScalars that should classify OK."""
    return SyncScalars(
        timestamps_present=True,
        cam=ChannelScalars(
            n_pulses=600,
            duration_s=6.0,
            isi_median_ms=10.0,
            isi_mad_ms=0.1,
            isi_cv=0.005,
            drift_slope_ppm=10.0,
            drift_r2=0.999,
            n_isi_outliers=0,
            min_isi_ms=9.5,
        ),
        img=ChannelScalars(
            n_pulses=180,
            duration_s=6.0,
            isi_median_ms=33.3,
            isi_mad_ms=0.05,
            isi_cv=0.002,
            drift_slope_ppm=5.0,
            drift_r2=0.999,
            n_isi_outliers=0,
            min_isi_ms=33.0,
        ),
        line=ChannelScalars(n_pulses=29160, isi_median_ms=0.2, duration_s=6.0),
        cross=CrossChannelScalars(overlap_s=6.0, start_offset_ms=0.0, end_offset_ms=0.0),
        light=LightScalars(
            n_on=3, n_off=3, period_median_s=120.0, duty_cycle=0.5, first_state_at_t0=1
        ),
        n_tiff_frames=180,
        pulse_count_diff=0,
        pulse_count_diff_after_off_by_one=0,
        cam_min=0.0,
        cam_max=1.0,
        sci_min=0.0,
        sci_max=1.0,
        light_min=0.0,
        light_max=1.0,
    )


class TestClassifyTiers:
    def test_ok(self) -> None:
        status, warnings, failures = classify(_ok_scalars())
        assert status == "OK"
        assert warnings == []
        assert failures == []

    def test_failed_no_timestamps(self) -> None:
        s = _ok_scalars()
        s.timestamps_present = False
        status, w, f = classify(s)
        assert status == "FAILED_NO_TIMESTAMPS"
        assert any(x.startswith("no_timestamps") for x in f)

    def test_failed_no_pulses(self) -> None:
        s = _ok_scalars()
        s.cam = replace(s.cam, n_pulses=0)
        status, _w, f = classify(s)
        assert status == "FAILED_NO_PULSES"
        assert any(x.startswith("no_pulses") for x in f)

    def test_failed_frame_count_mismatch(self) -> None:
        s = _ok_scalars()
        s.pulse_count_diff_after_off_by_one = 50
        s.pulse_count_diff = 50
        status, _w, f = classify(s)
        assert status == "FAILED_FRAME_COUNT_MISMATCH"
        assert any(x.startswith("frame_count_mismatch") for x in f)

    def test_failed_temporal_overlap(self) -> None:
        s = _ok_scalars()
        s.cross = CrossChannelScalars(overlap_s=4.0, start_offset_ms=0.0, end_offset_ms=0.0)
        # cam_dur = img_dur = 6.0 → overlap_frac = 4/6 ~ 0.67
        status, _w, f = classify(s)
        assert status == "FAILED_TEMPORAL_OVERLAP"
        assert any(x.startswith("temporal_overlap_hard") for x in f)

    def test_camera_overshoot_not_failed(self) -> None:
        """Camera recording 384s beyond imaging should NOT fail overlap check.

        The overlap fraction is relative to imaging duration, not the
        longer camera duration. Session 20220531 had this exact issue:
        camera ran ~384s past imaging end but data within the imaging
        window was perfect.
        """
        s = _ok_scalars()
        # Imaging: 600s, Camera: 984s (384s overshoot)
        s.img = replace(s.img, duration_s=600.0)
        s.cam = replace(s.cam, duration_s=984.0)
        # Full imaging window is covered by camera → overlap = 600s
        s.cross = CrossChannelScalars(
            overlap_s=600.0, start_offset_ms=0.0, end_offset_ms=-384000.0
        )
        status, warnings, _f = classify(s)
        assert status != "FAILED_TEMPORAL_OVERLAP", (
            f"Camera overshoot should not fail overlap: status={status}"
        )
        assert status in ("OK", "OK_WITH_WARNINGS")

    def test_failed_truncated_camera(self) -> None:
        s = _ok_scalars()
        s.cam = replace(s.cam, duration_s=2.0)
        # cam/img = 0.33 < 0.5
        # Force overlap to look fine to bypass earlier tier
        s.cross = CrossChannelScalars(overlap_s=2.0, start_offset_ms=0.0, end_offset_ms=0.0)
        # overlap/max(2,6)=0.33 < 0.95 — temporal overlap fires first
        # so increase overlap above 0.95 of max_dur via making img also short
        s.img = replace(s.img, duration_s=2.05)  # cam/img ~0.976 → still > 0.5
        # Use a shorter img with overlap of 1.95 s → 1.95/2.05 ≈ 0.95
        s.cross = CrossChannelScalars(overlap_s=2.0, start_offset_ms=0.0, end_offset_ms=0.0)
        # Recompute: cam_dur=2.0, img_dur=2.05 → 2/2.05 ≈ 0.976, NOT truncated
        # Reset cam to test truncation-only: use img_dur=10
        s.cam = replace(s.cam, duration_s=4.0)
        s.img = replace(s.img, duration_s=10.0)
        s.cross = CrossChannelScalars(overlap_s=10.0, start_offset_ms=0.0, end_offset_ms=0.0)
        # overlap/max=10/10=1.0 OK; cam/img=0.4 < 0.5 → truncated_camera
        status, _w, f = classify(s)
        assert status == "FAILED_TRUNCATED_CAMERA"
        assert any(x.startswith("truncated_camera") for x in f)

    def test_ok_with_warnings_high_cam_jitter(self) -> None:
        s = _ok_scalars()
        s.cam = replace(s.cam, isi_cv=0.025)  # > 0.02
        status, warnings, _f = classify(s)
        assert status == "OK_WITH_WARNINGS"
        assert "high_camera_jitter" in warnings

    def test_ok_with_warnings_high_img_jitter(self) -> None:
        s = _ok_scalars()
        s.img = replace(s.img, isi_cv=0.01)  # > 0.005
        status, warnings, _f = classify(s)
        assert status == "OK_WITH_WARNINGS"
        assert "high_imaging_jitter" in warnings

    def test_ok_with_warnings_drift_camera(self) -> None:
        s = _ok_scalars()
        s.cam = replace(s.cam, drift_slope_ppm=200.0)
        status, warnings, _f = classify(s)
        assert status == "OK_WITH_WARNINGS"
        assert "linear_drift_camera" in warnings

    def test_ok_with_warnings_duplicate_pulses(self) -> None:
        s = _ok_scalars()
        s.cam = replace(s.cam, min_isi_ms=0.1)  # vs median 10ms → << 0.25*median
        status, warnings, _f = classify(s)
        assert status == "OK_WITH_WARNINGS"
        assert "duplicate_pulses_camera" in warnings

    def test_ok_with_warnings_non_saturated_digital(self) -> None:
        s = _ok_scalars()
        s.cam_max = 0.7
        status, warnings, _f = classify(s)
        assert status == "OK_WITH_WARNINGS"
        assert "non_saturated_digital" in warnings

    def test_ok_with_warnings_truncated_camera_warning_at_below_50(self) -> None:
        # Truncation between cam/img < 0.5 is the FAILED tier; an overlap
        # in [0.95, 0.99] is the warning-band.
        s = _ok_scalars()
        s.cam = replace(s.cam, duration_s=6.0)
        s.img = replace(s.img, duration_s=6.05)
        s.cross = CrossChannelScalars(overlap_s=5.95, start_offset_ms=0.0, end_offset_ms=0.0)
        status, warnings, _f = classify(s)
        assert status == "OK_WITH_WARNINGS"
        assert "temporal_overlap_low" in warnings

    def test_ok_with_warnings_light_period_drift(self) -> None:
        s = _ok_scalars()
        s.light = LightScalars(
            n_on=3, n_off=3, period_median_s=140.0, duty_cycle=0.5, first_state_at_t0=1
        )
        status, warnings, _f = classify(s)
        assert status == "OK_WITH_WARNINGS"
        assert "light_period_drift" in warnings

    def test_ok_with_warnings_light_count_mismatch(self) -> None:
        s = _ok_scalars()
        s.light = LightScalars(
            n_on=5, n_off=7, period_median_s=120.0, duty_cycle=0.5, first_state_at_t0=1
        )
        status, warnings, _f = classify(s)
        assert status == "OK_WITH_WARNINGS"
        assert "light_count_mismatch" in warnings

    def test_ok_with_warnings_non_uniform_pose_decimation(self) -> None:
        s = _ok_scalars()
        s.kin_pose_decimation_uniform = 0
        status, warnings, _f = classify(s)
        assert status == "OK_WITH_WARNINGS"
        assert "non_uniform_pose_decimation" in warnings

    def test_ok_with_warnings_missing_tiff_frame_count(self) -> None:
        s = _ok_scalars()
        s.n_tiff_frames = INT_SENTINEL
        s.pulse_count_diff = INT_SENTINEL
        s.pulse_count_diff_after_off_by_one = INT_SENTINEL
        status, warnings, _f = classify(s)
        assert status == "OK_WITH_WARNINGS"
        assert "missing_tiff_frame_count" in warnings

    def test_ok_with_warnings_cross_start_offset_high(self) -> None:
        s = _ok_scalars()
        s.cross = CrossChannelScalars(overlap_s=6.0, start_offset_ms=100.0, end_offset_ms=0.0)
        status, warnings, _f = classify(s)
        assert status == "OK_WITH_WARNINGS"
        assert "cross_start_offset_high" in warnings

    def test_ok_with_warnings_s2p_off_by_one_fix(self) -> None:
        s = _ok_scalars()
        s.s2p_off_by_one_fix_applied = 1
        status, warnings, _f = classify(s)
        assert status == "OK_WITH_WARNINGS"
        assert "s2p_off_by_one_fix_applied" in warnings

    def test_ok_with_warnings_frame_count_off_by_one(self) -> None:
        s = _ok_scalars()
        # raw diff = 1, after fix = 0 → frame_count_off_by_one warning
        s.pulse_count_diff = 1
        s.pulse_count_diff_after_off_by_one = 0
        status, warnings, _f = classify(s)
        assert status == "OK_WITH_WARNINGS"
        assert "frame_count_off_by_one" in warnings

    def test_ok_with_warnings_frame_count_minor_mismatch(self) -> None:
        s = _ok_scalars()
        s.pulse_count_diff = 3
        s.pulse_count_diff_after_off_by_one = 3
        status, warnings, _f = classify(s)
        assert status == "OK_WITH_WARNINGS"
        assert "frame_count_minor_mismatch" in warnings


class TestClassifyTierOrdering:
    def test_no_timestamps_precedes_no_pulses(self) -> None:
        s = _ok_scalars()
        s.timestamps_present = False
        s.cam = replace(s.cam, n_pulses=0)
        status, _w, _f = classify(s)
        assert status == "FAILED_NO_TIMESTAMPS"

    def test_no_pulses_precedes_frame_count_mismatch(self) -> None:
        s = _ok_scalars()
        s.cam = replace(s.cam, n_pulses=0)
        s.pulse_count_diff_after_off_by_one = 999
        status, _w, _f = classify(s)
        assert status == "FAILED_NO_PULSES"

    def test_frame_count_precedes_overlap(self) -> None:
        s = _ok_scalars()
        s.pulse_count_diff_after_off_by_one = 50
        s.cross = CrossChannelScalars(overlap_s=2.0, start_offset_ms=0.0, end_offset_ms=0.0)
        status, _w, _f = classify(s)
        assert status == "FAILED_FRAME_COUNT_MISMATCH"

    def test_failed_precedes_warnings(self) -> None:
        s = _ok_scalars()
        s.pulse_count_diff_after_off_by_one = 50
        s.cam = replace(s.cam, isi_cv=0.05)
        status, _w, _f = classify(s)
        assert status.startswith("FAILED_")

    def test_warning_demotes_from_ok(self) -> None:
        s = _ok_scalars()
        s.cam = replace(s.cam, isi_cv=0.025)
        status, warnings, _f = classify(s)
        assert status == "OK_WITH_WARNINGS"
        assert len(warnings) >= 1


class TestClassifyBoundaries:
    @given(diff=st.integers(min_value=-20, max_value=20))
    @settings(max_examples=40)
    def test_frame_count_threshold_property(self, diff) -> None:
        s = _ok_scalars()
        s.pulse_count_diff = diff
        s.pulse_count_diff_after_off_by_one = diff
        status, _w, _f = classify(s)
        if abs(diff) <= 5:
            assert status in ("OK", "OK_WITH_WARNINGS"), (diff, status)
        else:
            assert status == "FAILED_FRAME_COUNT_MISMATCH", (diff, status)

    @given(overlap_frac=st.floats(min_value=0.5, max_value=1.0))
    @settings(max_examples=40)
    def test_overlap_threshold_property(self, overlap_frac) -> None:
        s = _ok_scalars()
        s.cam = replace(s.cam, duration_s=10.0)
        s.img = replace(s.img, duration_s=10.0)
        s.cross = CrossChannelScalars(
            overlap_s=10.0 * overlap_frac, start_offset_ms=0.0, end_offset_ms=0.0
        )
        status, warnings, _f = classify(s)
        if overlap_frac < 0.95:
            assert status == "FAILED_TEMPORAL_OVERLAP"
        elif overlap_frac < 0.99:
            assert status == "OK_WITH_WARNINGS"
            assert "temporal_overlap_low" in warnings
        else:
            assert status in ("OK", "OK_WITH_WARNINGS")

    @given(cv=st.floats(min_value=0.0, max_value=0.05))
    @settings(max_examples=30)
    def test_cv_cam_threshold_property(self, cv) -> None:
        s = _ok_scalars()
        s.cam = replace(s.cam, isi_cv=cv)
        status, warnings, _f = classify(s)
        if cv > 0.02:
            assert status == "OK_WITH_WARNINGS"
            assert "high_camera_jitter" in warnings
        else:
            # Below threshold, no jitter warning fires (other warnings may not).
            assert "high_camera_jitter" not in warnings


class TestClassifyInvariants:
    @given(
        cv_cam=st.floats(min_value=0.0, max_value=0.1),
        cv_img=st.floats(min_value=0.0, max_value=0.05),
        diff=st.integers(min_value=-50, max_value=50),
        drift_cam=st.floats(min_value=-1500, max_value=1500),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_invariants(self, cv_cam, cv_img, diff, drift_cam) -> None:
        s = _ok_scalars()
        s.cam = replace(s.cam, isi_cv=cv_cam, drift_slope_ppm=drift_cam)
        s.img = replace(s.img, isi_cv=cv_img)
        s.pulse_count_diff = diff
        s.pulse_count_diff_after_off_by_one = diff
        status, warnings, failures = classify(s)
        # Status is always a recognised code.
        from hm2p.io.hdf5 import SYNC_STATUS_CODES

        assert status in SYNC_STATUS_CODES
        # If status starts with FAILED_, failures is non-empty.
        if status.startswith("FAILED_"):
            assert len(failures) >= 1
        # If warnings exist, status is OK_WITH_WARNINGS or FAILED_ (failures
        # may co-exist with warnings, but warnings can't co-exist with bare OK).
        if status == "OK":
            assert warnings == []
            assert failures == []


# ---------------------------------------------------------------------------
# JSON encoding helpers
# ---------------------------------------------------------------------------


class TestJsonEncoding:
    def test_round_trip(self) -> None:
        codes = ["high_camera_jitter", "linear_drift_camera"]
        encoded = encode_codes_json(codes)
        decoded = decode_codes_json(encoded)
        assert decoded == codes

    def test_round_trip_bytes(self) -> None:
        codes = ["a", "b"]
        encoded = encode_codes_json(codes).encode("utf-8")
        decoded = decode_codes_json(encoded)
        assert decoded == codes

    def test_decode_non_list_raises(self) -> None:
        with pytest.raises(ValueError):
            decode_codes_json('{"a": 1}')


# ---------------------------------------------------------------------------
# code_message LUT smoke
# ---------------------------------------------------------------------------


def test_code_lut_covers_failure_codes() -> None:
    for code in (
        "no_timestamps",
        "no_pulses",
        "frame_count_mismatch",
        "temporal_overlap_hard",
        "truncated_camera",
    ):
        assert code in _CODE_LUT


def test_code_lut_covers_warning_codes() -> None:
    expected = (
        "frame_count_off_by_one",
        "frame_count_minor_mismatch",
        "high_camera_jitter",
        "high_imaging_jitter",
        "linear_drift_camera",
        "linear_drift_imaging",
        "duplicate_pulses_camera",
        "non_saturated_digital",
        "light_period_drift",
        "light_count_mismatch",
        "non_uniform_pose_decimation",
        "missing_tiff_frame_count",
        "cross_start_offset_high",
        "s2p_off_by_one_fix_applied",
        "temporal_overlap_low",
    )
    for code in expected:
        assert code in _CODE_LUT


def test_code_message_unknown_returns_input() -> None:
    assert code_message("not-a-real-code") == "not-a-real-code"


# ---------------------------------------------------------------------------
# build_scalars / scalars_to_diag_attrs
# ---------------------------------------------------------------------------


class TestBuildScalars:
    def test_clean_session(self) -> None:
        rng = np.random.default_rng(0)
        cam = synthetic_clean_pulse_train(rng, fps=100.0, duration_s=10.0, jitter_ms=0.0)
        img = synthetic_clean_pulse_train(rng, fps=30.0, duration_s=10.0, jitter_ms=0.0)
        line = synthetic_clean_pulse_train(rng, fps=30.0 * 162, duration_s=10.0, jitter_ms=0.0)
        s = build_scalars(
            timestamps_present=True,
            cam_times=cam,
            img_times=img,
            line_times=line,
            light_on=np.array([3.0]),
            light_off=np.array([7.0]),
            n_tiff_frames=img.size,
        )
        assert s.cam.n_pulses == cam.size
        assert s.img.n_pulses == img.size
        assert s.line.n_pulses == line.size
        assert s.cross.overlap_s == pytest.approx(10.0, abs=0.1)
        assert s.pulse_count_diff == 0
        assert s.pulse_count_diff_after_off_by_one == 0

    def test_off_by_one_correction(self) -> None:
        rng = np.random.default_rng(0)
        img = synthetic_clean_pulse_train(rng, fps=30.0, duration_s=10.0, jitter_ms=0.0)
        cam = synthetic_clean_pulse_train(rng, fps=100.0, duration_s=10.0, jitter_ms=0.0)
        # n_tiff_frames = img - 1 → diff=1 → after off-by-one correction = 0
        s = build_scalars(
            timestamps_present=True,
            cam_times=cam,
            img_times=img,
            line_times=np.empty(0),
            light_on=np.empty(0),
            light_off=np.empty(0),
            n_tiff_frames=img.size - 1,
        )
        assert s.pulse_count_diff == 1
        assert s.pulse_count_diff_after_off_by_one == 0

    def test_missing_timestamps(self) -> None:
        s = build_scalars(
            timestamps_present=False,
            cam_times=None,
            img_times=None,
            line_times=None,
            light_on=None,
            light_off=None,
        )
        assert s.timestamps_present is False
        status, _w, _f = classify(s)
        assert status == "FAILED_NO_TIMESTAMPS"


class TestScalarsToDiagAttrs:
    def test_keys_complete(self) -> None:
        from hm2p.sync.diagnostics import SYNC_DIAG_FLOAT_KEYS, SYNC_DIAG_INT_KEYS

        attrs = scalars_to_diag_attrs(_ok_scalars())
        for k in SYNC_DIAG_INT_KEYS:
            assert k in attrs, k
            assert isinstance(attrs[k], int)
        for k in SYNC_DIAG_FLOAT_KEYS:
            assert k in attrs, k
            assert isinstance(attrs[k], float)

    def test_round_trip_via_validator(self) -> None:
        # The flattened attrs must satisfy the parquet validator after
        # being put into a single-row DataFrame.
        import pandas as pd

        from hm2p.io.hdf5 import validate_sync_report_parquet

        attrs = scalars_to_diag_attrs(_ok_scalars())
        attrs.update(
            {
                "exp_id": "test_session",
                "sub": "sub-test",
                "ses": "ses-test",
                "sync_status": "OK",
                "sync_warnings": "[]",
                "sync_failures": "[]",
                "dlc_champion_id": "champ-1",
                "read_error": "",
            }
        )
        df = pd.DataFrame([attrs])
        validate_sync_report_parquet(df)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_sentinels() -> None:
    assert INT_SENTINEL == -9999
    assert np.isnan(FLOAT_SENTINEL)
