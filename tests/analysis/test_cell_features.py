"""Tests for hm2p.analysis.cell_features."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from hm2p.analysis import cell_features as cf
from hm2p.calcium.events import EventResult

FPS = 9.6


def _make_event_result(
    onsets: list[int],
    offsets: list[int],
    n_frames: int,
) -> EventResult:
    """Build a minimal EventResult for direct unit testing."""
    mask = np.zeros(n_frames, dtype=np.int32)
    for on, off in zip(onsets, offsets, strict=True):
        mask[on:off] = 1
    return EventResult(
        onsets=np.asarray(onsets, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
        amplitudes=np.ones(len(onsets), dtype=np.float64),
        event_mask=mask,
        noise_prob=np.zeros(n_frames, dtype=np.float64),
    )


def _synthetic_session(
    n_rois: int = 3,
    n_frames: int = 400,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Small synthetic session with HD-tuned transients."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames)

    hd = np.mod(t * 3.0, 360.0)
    ahv = np.full(n_frames, 3.0 * FPS)
    speed = 2.0 + rng.random(n_frames)
    light = np.zeros(n_frames, dtype=bool)
    light[: n_frames // 2] = True
    active = speed > 1.0
    bad = np.zeros(n_frames, dtype=bool)
    bad[-5:] = True

    dff = np.zeros((n_rois, n_frames), dtype=np.float64)
    for i in range(n_rois):
        base = 0.02 * rng.standard_normal(n_frames)
        preferred = 90.0 * i
        tuning = np.exp(np.cos(np.deg2rad(hd - preferred)) * 2.0)
        transient = np.zeros(n_frames)
        for onset in range(10 + i, n_frames - 20, 40):
            decay = np.exp(-np.arange(12) / 4.0)
            transient[onset : onset + 12] += decay * tuning[onset]
        dff[i] = base + 0.05 * transient

    return {
        "dff": dff,
        "hd_deg": hd,
        "ahv_deg_s": ahv,
        "speed_cm_s": speed,
        "light_on": light,
        "active": active,
        "bad_behav": bad,
    }


# ---------------------------------------------------------------------------
# _iei_statistics
# ---------------------------------------------------------------------------


def test_iei_statistics_empty_returns_nan() -> None:
    cv, median = cf._iei_statistics(np.asarray([], dtype=np.int64), FPS)
    assert np.isnan(cv)
    assert np.isnan(median)


def test_iei_statistics_two_onsets_gives_median_only() -> None:
    cv, median = cf._iei_statistics(np.asarray([0, 10], dtype=np.int64), 10.0)
    assert np.isnan(cv)
    assert median == pytest.approx(1.0)


def test_iei_statistics_regular_onsets_have_zero_cv() -> None:
    cv, median = cf._iei_statistics(np.asarray([0, 10, 20, 30], dtype=np.int64), 10.0)
    assert cv == pytest.approx(0.0, abs=1e-12)
    assert median == pytest.approx(1.0)


def test_iei_statistics_nonpositive_fps_returns_nan() -> None:
    cv, median = cf._iei_statistics(np.asarray([0, 10, 20], dtype=np.int64), 0.0)
    assert np.isnan(cv)
    assert np.isnan(median)


def test_iei_statistics_duplicate_onsets_give_nan_cv() -> None:
    cv, median = cf._iei_statistics(np.asarray([5, 5, 5], dtype=np.int64), 10.0)
    assert np.isnan(cv)
    assert median == pytest.approx(0.0)


def test_iei_statistics_irregular_onsets_have_positive_cv() -> None:
    cv, _ = cf._iei_statistics(np.asarray([0, 1, 50, 51, 52], dtype=np.int64), 10.0)
    assert cv > 0.5


# ---------------------------------------------------------------------------
# _plateau_fraction
# ---------------------------------------------------------------------------


def test_plateau_fraction_square_event_is_one() -> None:
    trace = np.zeros(50)
    trace[10:20] = 1.0
    result = _make_event_result([10], [20], 50)
    assert cf._plateau_fraction(trace, result) == pytest.approx(1.0)


def test_plateau_fraction_sharp_event_is_small() -> None:
    trace = np.zeros(50)
    trace[10:20] = np.linspace(0.1, 1.0, 10)
    result = _make_event_result([10], [20], 50)
    frac = cf._plateau_fraction(trace, result)
    assert 0.0 < frac < 0.5


def test_plateau_fraction_no_events_is_nan() -> None:
    assert np.isnan(cf._plateau_fraction(np.zeros(20), _make_event_result([], [], 20)))


def test_plateau_fraction_skips_nonpositive_peak_events() -> None:
    trace = -np.ones(30)
    result = _make_event_result([5], [10], 30)
    assert np.isnan(cf._plateau_fraction(trace, result))


def test_plateau_fraction_skips_empty_segment() -> None:
    trace = np.ones(30)
    result = _make_event_result([5], [5], 30)
    assert np.isnan(cf._plateau_fraction(trace, result))


# ---------------------------------------------------------------------------
# _duration_cv / _event_summary
# ---------------------------------------------------------------------------


def test_duration_cv_fewer_than_two_events_is_nan() -> None:
    assert np.isnan(cf._duration_cv([]))
    assert np.isnan(cf._duration_cv([{"duration_s": 1.0}]))


def test_duration_cv_equal_durations_is_zero() -> None:
    events = [{"duration_s": 2.0}, {"duration_s": 2.0}, {"duration_s": 2.0}]
    assert cf._duration_cv(events) == pytest.approx(0.0)


def test_duration_cv_zero_mean_duration_is_nan() -> None:
    assert np.isnan(cf._duration_cv([{"duration_s": 0.0}, {"duration_s": 0.0}]))


def test_duration_cv_varied_durations_is_positive() -> None:
    events = [{"duration_s": 1.0}, {"duration_s": 5.0}, {"duration_s": 9.0}]
    assert cf._duration_cv(events) > 0.5


def test_event_summary_contains_derived_keys() -> None:
    trace = np.zeros(60)
    trace[10:20] = 1.0
    trace[30:40] = 1.0
    result = _make_event_result([10, 30], [20, 40], 60)
    out = cf._event_summary(trace, result, 10.0)
    assert out["n_events"] == 2
    assert out["duration_cv"] == pytest.approx(0.0)
    assert out["plateau_fraction"] == pytest.approx(1.0)
    assert out["median_iei_s"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _moments
# ---------------------------------------------------------------------------


def test_moments_match_scipy_for_well_conditioned_input() -> None:
    from scipy import stats as sp_stats

    rng = np.random.default_rng(11)
    trace = rng.exponential(1.0, 300)
    skewness, kurtosis = cf._moments(trace)
    assert skewness == pytest.approx(float(sp_stats.skew(trace, bias=False)))
    assert kurtosis == pytest.approx(float(sp_stats.kurtosis(trace, bias=False)))


def test_moments_nearly_identical_values_return_nan() -> None:
    trace = np.full(50, 1e8)
    trace[0] += 1e-8
    skewness, kurtosis = cf._moments(trace)
    assert np.isnan(skewness)
    assert np.isnan(kurtosis)


# ---------------------------------------------------------------------------
# _autocorr_time_s
# ---------------------------------------------------------------------------


def test_autocorr_time_white_noise_is_fast() -> None:
    rng = np.random.default_rng(1)
    tau = cf._autocorr_time_s(rng.standard_normal(500), 10.0)
    assert 0.0 < tau <= 0.5


def test_autocorr_time_smooth_signal_is_slower_than_noise() -> None:
    rng = np.random.default_rng(2)
    noise = rng.standard_normal(500)
    smooth = np.convolve(noise, np.ones(20) / 20.0, mode="same")
    assert cf._autocorr_time_s(smooth, 10.0) > cf._autocorr_time_s(noise, 10.0)


def test_autocorr_time_constant_trace_is_nan() -> None:
    assert np.isnan(cf._autocorr_time_s(np.ones(100), 10.0))


def test_autocorr_time_too_short_is_nan() -> None:
    assert np.isnan(cf._autocorr_time_s(np.asarray([1.0]), 10.0))


def test_autocorr_time_nonpositive_fps_is_nan() -> None:
    assert np.isnan(cf._autocorr_time_s(np.asarray([1.0, 2.0, 3.0]), 0.0))


def test_autocorr_time_never_decaying_within_max_lag_is_nan() -> None:
    # A ramp stays above 1/e for the first ten lags of a 100-frame signal.
    ramp = np.arange(100, dtype=np.float64)
    assert np.isnan(cf._autocorr_time_s(ramp, 10.0, max_lag_frames=10))
    assert np.isfinite(cf._autocorr_time_s(ramp, 10.0))


# ---------------------------------------------------------------------------
# trace_statistics
# ---------------------------------------------------------------------------


def test_trace_statistics_constant_trace_is_all_nan() -> None:
    out = cf.trace_statistics(np.ones(100), FPS)
    for key, value in out.items():
        assert np.isnan(value), key


def test_trace_statistics_short_trace_is_all_nan() -> None:
    out = cf.trace_statistics(np.asarray([0.0, 1.0, 2.0]), FPS)
    assert np.isnan(out["skewness"])


def test_trace_statistics_nonfinite_trace_is_all_nan() -> None:
    trace = np.arange(20, dtype=np.float64)
    trace[5] = np.nan
    assert np.isnan(cf.trace_statistics(trace, FPS)["kurtosis"])


def test_trace_statistics_positively_skewed_trace() -> None:
    rng = np.random.default_rng(3)
    trace = rng.exponential(1.0, 2000)
    out = cf.trace_statistics(trace, FPS)
    assert out["skewness"] > 0.5
    assert out["kurtosis"] > 0.0
    assert 0.0 <= out["fraction_above_2sd"] <= 1.0
    assert out["signal_range_ratio"] > 0.0


def test_trace_statistics_burstiness_from_onsets() -> None:
    rng = np.random.default_rng(4)
    trace = rng.standard_normal(200)
    out = cf.trace_statistics(trace, 10.0, onsets=np.asarray([0, 10, 20, 30]))
    assert out["burstiness"] == pytest.approx(0.0, abs=1e-12)


def test_trace_statistics_zero_mad_gives_nan_range_ratio() -> None:
    trace = np.zeros(100)
    trace[:2] = 5.0
    out = cf.trace_statistics(trace, FPS)
    assert np.isnan(out["signal_range_ratio"])
    assert np.isfinite(out["skewness"])


def test_trace_statistics_rejects_2d_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        cf.trace_statistics(np.zeros((2, 10)), FPS)


def test_trace_statistics_is_scale_invariant_for_skewness() -> None:
    rng = np.random.default_rng(5)
    trace = rng.exponential(1.0, 500)
    a = cf.trace_statistics(trace, FPS)["skewness"]
    b = cf.trace_statistics(trace * 7.0, FPS)["skewness"]
    assert a == pytest.approx(b, rel=1e-9)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    hnp.arrays(
        np.float64,
        hnp.array_shapes(min_dims=1, max_dims=1, min_side=4, max_side=200),
        elements=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    )
)
def test_trace_statistics_finite_or_nan_for_finite_input(trace: np.ndarray) -> None:
    out = cf.trace_statistics(trace, FPS)
    assert set(out) == {
        "skewness",
        "kurtosis",
        "autocorr_time_s",
        "burstiness",
        "fraction_above_2sd",
        "signal_range_ratio",
    }
    for key, value in out.items():
        assert isinstance(value, float), key
        assert np.isnan(value) or np.isfinite(value), key


# ---------------------------------------------------------------------------
# detect_cell_events / event_statistics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["vh", "sd"])
def test_detect_cell_events_returns_event_result(method: str) -> None:
    trace = _synthetic_session(n_rois=1)["dff"][0]
    result = cf.detect_cell_events(trace, FPS, method=method)
    assert isinstance(result, EventResult)
    assert result.event_mask.shape == trace.shape


def test_detect_cell_events_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        cf.detect_cell_events(np.zeros(50), FPS, method="oasis")


@pytest.mark.parametrize("method", ["vh", "sd"])
def test_event_statistics_has_expected_keys(method: str) -> None:
    trace = _synthetic_session(n_rois=1)["dff"][0]
    out = cf.event_statistics(trace, FPS, method=method)
    for key in (
        "n_events",
        "event_rate",
        "iei_cv",
        "median_iei_s",
        "plateau_fraction",
        "duration_cv",
    ):
        assert key in out


def test_event_statistics_respects_bad_frames() -> None:
    trace = _synthetic_session(n_rois=1)["dff"][0]
    bad = np.zeros(trace.size, dtype=bool)
    # Exclude a stretch that contains no event onsets: the event count is
    # unchanged but the recording duration shrinks, so the rate must rise.
    bad[:8] = True
    full = cf.event_statistics(trace, FPS)
    masked = cf.event_statistics(trace, FPS, bad_frames=bad)
    assert masked["n_events"] == full["n_events"]
    assert masked["event_rate"] > full["event_rate"]


def test_event_statistics_flat_trace_has_no_events() -> None:
    out = cf.event_statistics(np.zeros(200), FPS, method="sd")
    assert out["n_events"] == 0
    assert np.isnan(out["plateau_fraction"])


# ---------------------------------------------------------------------------
# _hd_occupancy / _nan_tuning_features / tuning_features
# ---------------------------------------------------------------------------


def test_hd_occupancy_sums_to_masked_frame_count() -> None:
    hd = np.linspace(0.0, 359.0, 100)
    mask = np.ones(100, dtype=bool)
    mask[:10] = False
    occ = cf._hd_occupancy(hd, mask, 36)
    assert occ.sum() == pytest.approx(90.0)
    assert occ.shape == (36,)


def test_hd_occupancy_wraps_360() -> None:
    occ = cf._hd_occupancy(np.asarray([360.0, 0.0]), np.ones(2, dtype=bool), 4)
    assert occ[0] == 2.0


def test_nan_tuning_features_all_nan() -> None:
    out = cf._nan_tuning_features()
    assert all(np.isnan(v) for v in out.values())


def test_tuning_features_too_few_frames_returns_nan() -> None:
    n = 20
    mask = np.zeros(n, dtype=bool)
    mask[:3] = True
    out = cf.tuning_features(np.zeros(n), np.zeros(n), np.zeros(n), np.ones(n), mask, FPS)
    assert np.isnan(out["mvl"])


def test_tuning_features_tuned_cell_has_higher_mvl_than_untuned() -> None:
    n = 2000
    rng = np.random.default_rng(6)
    hd = np.mod(np.arange(n) * 7.0, 360.0)
    tuned = np.exp(3.0 * np.cos(np.deg2rad(hd - 45.0)))
    untuned = rng.random(n)
    mask = np.ones(n, dtype=bool)
    ahv = np.full(n, 7.0 * FPS)
    speed = np.full(n, 3.0)

    tuned_out = cf.tuning_features(tuned, hd, ahv, speed, mask, FPS)
    untuned_out = cf.tuning_features(untuned, hd, ahv, speed, mask, FPS)

    assert tuned_out["mvl"] > untuned_out["mvl"]
    assert tuned_out["preferred_direction"] == pytest.approx(45.0, abs=15.0)
    assert np.isfinite(tuned_out["fwhm"])
    assert tuned_out["hd_skaggs_info"] > untuned_out["hd_skaggs_info"]


def test_tuning_features_keys_are_stable() -> None:
    n = 500
    hd = np.mod(np.arange(n) * 5.0, 360.0)
    out = cf.tuning_features(
        np.ones(n),
        hd,
        np.full(n, 10.0),
        np.full(n, 2.0),
        np.ones(n, dtype=bool),
        FPS,
    )
    assert set(out) == set(cf._nan_tuning_features())


def test_tuning_features_nonpositive_ahv_mean_gives_nan_index() -> None:
    n = 500
    hd = np.mod(np.arange(n) * 5.0, 360.0)
    out = cf.tuning_features(
        -np.ones(n),
        hd,
        np.linspace(-100.0, 100.0, n),
        np.full(n, 2.0),
        np.ones(n, dtype=bool),
        FPS,
    )
    assert np.isnan(out["ahv_modulation_index"])


# ---------------------------------------------------------------------------
# catch22_features
# ---------------------------------------------------------------------------


def test_catch22_features_absent_package_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pycatch22", None)
    with monkeypatch.context() as ctx:
        ctx.delitem(sys.modules, "pycatch22", raising=False)
        real_import = __import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "pycatch22":
                raise ImportError("no pycatch22")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        ctx.setattr("builtins.__import__", fake_import)
        assert cf.catch22_features(np.arange(50, dtype=np.float64)) == {}


def test_catch22_features_present_package_returns_prefixed_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = types.ModuleType("pycatch22")

    def catch22_all(values: list[float]) -> dict[str, list]:
        return {
            "names": [f"feat{i}" for i in range(22)],
            "values": [float(i) * len(values) for i in range(22)],
        }

    fake.catch22_all = catch22_all  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pycatch22", fake)

    out = cf.catch22_features(np.arange(10, dtype=np.float64))
    assert len(out) == 22
    assert all(key.startswith("c22_") for key in out)
    assert out["c22_feat1"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# cell_feature_row
# ---------------------------------------------------------------------------


def test_cell_feature_row_has_prefixed_keys() -> None:
    session = _synthetic_session(n_rois=1)
    row = cf.cell_feature_row(
        0,
        session["dff"][0],
        session["hd_deg"],
        session["ahv_deg_s"],
        session["speed_cm_s"],
        session["light_on"],
        session["active"],
        session["bad_behav"],
        FPS,
    )
    assert row["roi_idx"] == 0
    for prefix in ("ev_", "tr_", "tun_", "act_"):
        assert any(key.startswith(prefix) for key in row)
    assert "tun_mvl_light" in row
    assert "tun_mvl_dark" in row
    assert "ev_event_rate_light" in row
    assert "ev_event_rate_dark" in row


def test_cell_feature_row_rejects_mismatched_lengths() -> None:
    session = _synthetic_session(n_rois=1, n_frames=100)
    with pytest.raises(ValueError, match="same length"):
        cf.cell_feature_row(
            0,
            session["dff"][0][:50],
            session["hd_deg"],
            session["ahv_deg_s"],
            session["speed_cm_s"],
            session["light_on"],
            session["active"],
            session["bad_behav"],
            FPS,
        )


def test_cell_feature_row_with_catch22(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("pycatch22")
    fake.catch22_all = lambda values: {"names": ["a"], "values": [1.0]}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pycatch22", fake)

    session = _synthetic_session(n_rois=1, n_frames=200)
    row = cf.cell_feature_row(
        0,
        session["dff"][0],
        session["hd_deg"],
        session["ahv_deg_s"],
        session["speed_cm_s"],
        session["light_on"],
        session["active"],
        session["bad_behav"],
        FPS,
        include_catch22=True,
    )
    assert row["c22_a"] == pytest.approx(1.0)


def test_cell_feature_row_sd_method_runs() -> None:
    session = _synthetic_session(n_rois=1, n_frames=200)
    row = cf.cell_feature_row(
        0,
        session["dff"][0],
        session["hd_deg"],
        session["ahv_deg_s"],
        session["speed_cm_s"],
        session["light_on"],
        session["active"],
        session["bad_behav"],
        FPS,
        event_method="sd",
    )
    assert row["ev_n_events"] >= 0


# ---------------------------------------------------------------------------
# session_feature_table
# ---------------------------------------------------------------------------


def test_session_feature_table_shape_and_columns() -> None:
    session = _synthetic_session(n_rois=3, n_frames=300)
    table = cf.session_feature_table(
        session["dff"],
        session["hd_deg"],
        session["ahv_deg_s"],
        session["speed_cm_s"],
        session["light_on"],
        session["active"],
        session["bad_behav"],
        FPS,
        roi_types=["soma", "soma", "dendrite"],
    )
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 3
    assert list(table.columns[:2]) == ["roi_idx", "roi_type"]
    assert table["roi_idx"].tolist() == [0, 1, 2]
    assert table["roi_type"].tolist() == ["soma", "soma", "dendrite"]


def test_session_feature_table_without_roi_types() -> None:
    session = _synthetic_session(n_rois=2, n_frames=200)
    table = cf.session_feature_table(
        session["dff"],
        session["hd_deg"],
        session["ahv_deg_s"],
        session["speed_cm_s"],
        session["light_on"],
        session["active"],
        session["bad_behav"],
        FPS,
    )
    assert "roi_type" not in table.columns
    assert len(table) == 2


def test_session_feature_table_rejects_1d_dff() -> None:
    session = _synthetic_session(n_rois=1, n_frames=100)
    with pytest.raises(ValueError, match="2-D"):
        cf.session_feature_table(
            session["dff"][0],
            session["hd_deg"],
            session["ahv_deg_s"],
            session["speed_cm_s"],
            session["light_on"],
            session["active"],
            session["bad_behav"],
            FPS,
        )


def test_session_feature_table_rejects_bad_roi_types_length() -> None:
    session = _synthetic_session(n_rois=2, n_frames=100)
    with pytest.raises(ValueError, match="roi_types has length"):
        cf.session_feature_table(
            session["dff"],
            session["hd_deg"],
            session["ahv_deg_s"],
            session["speed_cm_s"],
            session["light_on"],
            session["active"],
            session["bad_behav"],
            FPS,
            roi_types=["soma"],
        )


# ---------------------------------------------------------------------------
# feature_families
# ---------------------------------------------------------------------------


def test_feature_families_keys() -> None:
    families = cf.feature_families()
    assert set(families) == {"kinetics", "trace_shape", "tuning", "activity"}
    for cols in families.values():
        assert len(cols) == len(set(cols))


def test_feature_families_columns_exist_in_table() -> None:
    session = _synthetic_session(n_rois=2, n_frames=300)
    table = cf.session_feature_table(
        session["dff"],
        session["hd_deg"],
        session["ahv_deg_s"],
        session["speed_cm_s"],
        session["light_on"],
        session["active"],
        session["bad_behav"],
        FPS,
    )
    for family, cols in cf.feature_families().items():
        missing = sorted(set(cols) - set(table.columns))
        assert not missing, f"{family} references missing columns: {missing}"


def test_tuning_features_with_nan_behaviour_frames() -> None:
    """Pose gaps (NaN hd/speed/ahv) are excluded rather than degrading the indices."""
    from hm2p.analysis.cell_features import tuning_features

    rng = np.random.default_rng(3)
    n = 1500
    hd = np.mod(np.cumsum(rng.normal(0, 5, n)), 360.0)
    speed = rng.uniform(0, 20, n)
    ahv = rng.normal(0, 50, n)
    signal = 0.05 * speed + np.exp(np.cos(np.deg2rad(hd - 45))) / 5 + rng.normal(0, 0.05, n)
    speed[::11] = np.nan
    hd[::13] = np.nan
    ahv[::17] = np.nan
    out = tuning_features(signal, hd, ahv, speed, np.ones(n, dtype=bool), fps=9.6)
    assert np.isfinite(out["mvl"]) and out["mvl"] > 0.05
    assert out["speed_modulation_index"] > 0.1
    assert out["speed_correlation"] > 0.3
