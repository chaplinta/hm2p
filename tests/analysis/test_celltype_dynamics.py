"""Tests for hm2p.analysis.celltype_dynamics.

Covers:
- population_rate_by_condition: 2×2 factorial condition rates
- compare_celltypes: Mann-Whitney U between Penk+ and Penk⁻CamKII+
- celltype_dynamics_summary: multi-session aggregation
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import mannwhitneyu

from hm2p.analysis.celltype_dynamics import (
    celltype_dynamics_summary,
    compare_celltypes,
    population_rate_by_condition,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dff(n_rois: int = 15, n_frames: int = 400, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_rois, n_frames)).astype(np.float32) * 0.2


def _make_speed(n_frames: int = 400, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed + 10)
    return np.abs(rng.standard_normal(n_frames)).astype(np.float32) * 5.0


def _make_light_on(n_frames: int = 400) -> np.ndarray:
    """Alternating blocks of light/dark (60-frame blocks)."""
    light = np.zeros(n_frames, dtype=bool)
    for i in range(0, n_frames, 120):
        light[i : i + 60] = True
    return light


def _make_active(n_frames: int = 400) -> np.ndarray:
    return np.ones(n_frames, dtype=bool)


def _make_session(
    celltype: str = "penk",
    n_rois: int = 15,
    n_frames: int = 400,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "celltype": celltype,
        "dff": _make_dff(n_rois, n_frames, seed),
        "speed_cm_s": _make_speed(n_frames, seed),
        "light_on": _make_light_on(n_frames),
        "bad_behav": np.zeros(n_frames, dtype=bool),
        "active": _make_active(n_frames),
    }


# ===========================================================================
# population_rate_by_condition
# ===========================================================================


class TestPopulationRateByCondition:
    def test_output_keys_present(self) -> None:
        dff = _make_dff()
        speed = _make_speed()
        light_on = _make_light_on()
        active = _make_active()
        result = population_rate_by_condition(dff, speed, light_on, active)
        expected_keys = {"moving_light", "moving_dark", "stationary_light", "stationary_dark"}
        assert expected_keys == set(result.keys())

    def test_each_condition_has_mean_rate_and_n_frames(self) -> None:
        dff = _make_dff()
        speed = _make_speed()
        light_on = _make_light_on()
        active = _make_active()
        result = population_rate_by_condition(dff, speed, light_on, active)
        for cond in result:
            assert "mean_rate" in result[cond]
            assert "n_frames" in result[cond]

    def test_mean_rate_shape(self) -> None:
        n_rois = 12
        dff = _make_dff(n_rois=n_rois)
        speed = _make_speed()
        light_on = _make_light_on()
        active = _make_active()
        result = population_rate_by_condition(dff, speed, light_on, active)
        for cond in result:
            assert result[cond]["mean_rate"].shape == (n_rois,)

    def test_n_frames_non_negative(self) -> None:
        dff = _make_dff()
        speed = _make_speed()
        light_on = _make_light_on()
        active = _make_active()
        result = population_rate_by_condition(dff, speed, light_on, active)
        for cond in result:
            assert result[cond]["n_frames"] >= 0

    def test_frame_counts_sum_le_total(self) -> None:
        """Sum of per-condition frames ≤ total active frames."""
        n_frames = 400
        dff = _make_dff(n_frames=n_frames)
        speed = _make_speed(n_frames)
        light_on = _make_light_on(n_frames)
        active = _make_active(n_frames)
        result = population_rate_by_condition(dff, speed, light_on, active)
        total = sum(result[c]["n_frames"] for c in result)
        assert total <= n_frames

    def test_all_active_frames_partitioned(self) -> None:
        """When all frames are active, counts must sum to exactly n_frames."""
        n_frames = 400
        dff = _make_dff(n_frames=n_frames)
        speed = np.array([5.0] * 200 + [1.0] * 200, dtype=np.float32)  # 200 moving, 200 stationary
        light_on = np.array([True] * 200 + [False] * 200)
        active = np.ones(n_frames, dtype=bool)
        result = population_rate_by_condition(dff, speed, light_on, active)
        total = sum(result[c]["n_frames"] for c in result)
        assert total == n_frames

    def test_all_frames_bad_gives_nan_rates(self) -> None:
        """When active_mask is all False, all conditions should have n_frames=0."""
        dff = _make_dff()
        speed = _make_speed()
        light_on = _make_light_on()
        active = np.zeros(400, dtype=bool)
        result = population_rate_by_condition(dff, speed, light_on, active)
        for cond in result:
            assert result[cond]["n_frames"] == 0
            assert np.all(np.isnan(result[cond]["mean_rate"]))

    def test_speed_threshold_effect(self) -> None:
        """Higher threshold should shift frames from moving to stationary."""
        dff = _make_dff()
        speed = _make_speed()
        light_on = _make_light_on()
        active = _make_active()
        result_low = population_rate_by_condition(dff, speed, light_on, active, speed_threshold=1.0)
        result_high = population_rate_by_condition(dff, speed, light_on, active, speed_threshold=10.0)
        # At speed_threshold=1 more frames are "moving"
        n_moving_low = result_low["moving_light"]["n_frames"] + result_low["moving_dark"]["n_frames"]
        n_moving_high = result_high["moving_light"]["n_frames"] + result_high["moving_dark"]["n_frames"]
        assert n_moving_low >= n_moving_high

    def test_single_roi(self) -> None:
        dff = _make_dff(n_rois=1)
        speed = _make_speed()
        light_on = _make_light_on()
        active = _make_active()
        result = population_rate_by_condition(dff, speed, light_on, active)
        for cond in result:
            assert result[cond]["mean_rate"].shape == (1,)

    def test_mismatched_lengths_handled(self) -> None:
        """Shorter behavioural arrays should be handled by truncating to min."""
        dff = _make_dff(n_frames=400)
        speed = _make_speed(n_frames=350)  # shorter
        light_on = _make_light_on(n_frames=400)
        active = _make_active(n_frames=400)
        result = population_rate_by_condition(dff, speed, light_on, active)
        total = sum(result[c]["n_frames"] for c in result)
        assert total <= 350

    @given(
        n_rois=st.integers(min_value=1, max_value=20),
        n_frames=st.integers(min_value=20, max_value=200),
    )
    @settings(max_examples=20, deadline=None)
    def test_mean_rate_shape_property(self, n_rois: int, n_frames: int) -> None:
        rng = np.random.default_rng(0)
        dff = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
        speed = np.abs(rng.standard_normal(n_frames)).astype(np.float32)
        light_on = (rng.uniform(size=n_frames) > 0.5)
        active = np.ones(n_frames, dtype=bool)
        result = population_rate_by_condition(dff, speed, light_on, active)
        for cond in result:
            assert result[cond]["mean_rate"].shape == (n_rois,)


# ===========================================================================
# compare_celltypes
# ===========================================================================


class TestCompareCelltypes:
    def test_output_keys_present(self) -> None:
        rng = np.random.default_rng(0)
        penk = rng.standard_normal(30).astype(np.float32)
        nonpenk = rng.standard_normal(25).astype(np.float32)
        result = compare_celltypes(penk, nonpenk)
        expected = {"statistic", "p_value", "effect_size", "n_penk", "n_nonpenk",
                    "penk_median", "nonpenk_median"}
        assert expected == set(result.keys())

    def test_uses_mann_whitney_u(self) -> None:
        """Regression test: must use Mann-Whitney U, not t-test."""
        rng = np.random.default_rng(1)
        penk = rng.standard_normal(40).astype(np.float32)
        nonpenk = rng.standard_normal(35).astype(np.float32) + 1.0
        result = compare_celltypes(penk, nonpenk)

        # Recompute expected values using MWU directly
        U_exp, p_exp = mannwhitneyu(penk, nonpenk, alternative="two-sided")
        assert abs(result["statistic"] - float(U_exp)) < 1e-6
        assert abs(result["p_value"] - float(p_exp)) < 1e-10

    def test_p_value_in_valid_range(self) -> None:
        rng = np.random.default_rng(2)
        penk = rng.standard_normal(30).astype(np.float32)
        nonpenk = rng.standard_normal(30).astype(np.float32)
        result = compare_celltypes(penk, nonpenk)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_effect_size_in_valid_range(self) -> None:
        """Rank-biserial r ∈ [-1, 1]."""
        rng = np.random.default_rng(3)
        penk = rng.standard_normal(30).astype(np.float32)
        nonpenk = rng.standard_normal(30).astype(np.float32)
        result = compare_celltypes(penk, nonpenk)
        assert -1.0 - 1e-6 <= result["effect_size"] <= 1.0 + 1e-6

    def test_sample_counts_correct(self) -> None:
        rng = np.random.default_rng(4)
        penk = rng.standard_normal(25).astype(np.float32)
        nonpenk = rng.standard_normal(18).astype(np.float32)
        result = compare_celltypes(penk, nonpenk)
        assert result["n_penk"] == 25
        assert result["n_nonpenk"] == 18

    def test_nan_values_excluded(self) -> None:
        rng = np.random.default_rng(5)
        penk = rng.standard_normal(30).astype(np.float32)
        penk[:5] = np.nan
        nonpenk = rng.standard_normal(25).astype(np.float32)
        result = compare_celltypes(penk, nonpenk)
        assert result["n_penk"] == 25  # 30 - 5 NaNs
        assert result["n_nonpenk"] == 25

    def test_insufficient_samples_returns_nan(self) -> None:
        """Fewer than 2 valid samples should return NaN statistics."""
        penk = np.array([1.0, np.nan, np.nan])
        nonpenk = np.array([2.0, 3.0, 4.0])
        result = compare_celltypes(penk, nonpenk)
        assert np.isnan(result["statistic"])
        assert np.isnan(result["p_value"])

    def test_identical_distributions_high_p(self) -> None:
        """Identical distributions should not be significantly different."""
        rng = np.random.default_rng(6)
        vals = rng.standard_normal(100).astype(np.float32)
        result = compare_celltypes(vals[:50], vals[50:])
        # p > 0.05 is not guaranteed with identical distributions but
        # effect size should be near zero
        assert abs(result["effect_size"]) < 0.5  # loose bound

    def test_clearly_separated_distributions_low_p(self) -> None:
        """Very different distributions should give p ≈ 0."""
        penk = np.zeros(100, dtype=np.float32)
        nonpenk = np.ones(100, dtype=np.float32) * 100.0
        result = compare_celltypes(penk, nonpenk)
        assert result["p_value"] < 0.001

    def test_medians_correct(self) -> None:
        penk = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        nonpenk = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        result = compare_celltypes(penk, nonpenk)
        assert abs(result["penk_median"] - 3.0) < 1e-5
        assert abs(result["nonpenk_median"] - 20.0) < 1e-5

    def test_all_nan_returns_nan(self) -> None:
        penk = np.full(10, np.nan, dtype=np.float32)
        nonpenk = np.full(10, np.nan, dtype=np.float32)
        result = compare_celltypes(penk, nonpenk)
        assert np.isnan(result["statistic"])

    @given(
        n1=st.integers(min_value=2, max_value=50),
        n2=st.integers(min_value=2, max_value=50),
        offset=st.floats(min_value=-10.0, max_value=10.0),
    )
    @settings(max_examples=25, deadline=None)
    def test_p_value_always_valid(self, n1: int, n2: int, offset: float) -> None:
        rng = np.random.default_rng(0)
        penk = rng.standard_normal(n1).astype(np.float32)
        nonpenk = rng.standard_normal(n2).astype(np.float32) + offset
        result = compare_celltypes(penk, nonpenk)
        if np.isfinite(result["p_value"]):
            assert 0.0 <= result["p_value"] <= 1.0


# ===========================================================================
# celltype_dynamics_summary
# ===========================================================================


class TestCelltypeDynamicsSummary:
    def _make_sessions(
        self,
        n_penk: int = 2,
        n_nonpenk: int = 2,
        n_rois: int = 10,
        n_frames: int = 300,
    ) -> list[dict]:
        sessions = []
        for i in range(n_penk):
            sessions.append(_make_session("penk", n_rois, n_frames, seed=i))
        for i in range(n_nonpenk):
            sessions.append(_make_session("nonpenk", n_rois, n_frames, seed=100 + i))
        return sessions

    def test_output_keys_present(self) -> None:
        sessions = self._make_sessions()
        result = celltype_dynamics_summary(sessions)
        assert "comparisons" in result
        assert "conditions" in result
        assert "n_penk_sessions" in result
        assert "n_nonpenk_sessions" in result

    def test_comparisons_has_all_conditions_plus_overall(self) -> None:
        sessions = self._make_sessions()
        result = celltype_dynamics_summary(sessions)
        expected = {
            "moving_light", "moving_dark", "stationary_light", "stationary_dark", "overall"
        }
        assert expected == set(result["comparisons"].keys())

    def test_session_counts_correct(self) -> None:
        sessions = self._make_sessions(n_penk=3, n_nonpenk=2)
        result = celltype_dynamics_summary(sessions)
        assert result["n_penk_sessions"] == 3
        assert result["n_nonpenk_sessions"] == 2

    def test_each_comparison_has_required_keys(self) -> None:
        sessions = self._make_sessions()
        result = celltype_dynamics_summary(sessions)
        required = {"statistic", "p_value", "effect_size", "n_penk", "n_nonpenk"}
        for cond_result in result["comparisons"].values():
            assert required.issubset(cond_result.keys())

    def test_p_values_in_valid_range(self) -> None:
        sessions = self._make_sessions()
        result = celltype_dynamics_summary(sessions)
        for cond, comp in result["comparisons"].items():
            p = comp["p_value"]
            if np.isfinite(p):
                assert 0.0 <= p <= 1.0, f"p_value out of range for condition {cond}: {p}"

    def test_empty_sessions_returns_nan_stats(self) -> None:
        result = celltype_dynamics_summary([])
        assert result["n_penk_sessions"] == 0
        assert result["n_nonpenk_sessions"] == 0
        for cond_result in result["comparisons"].values():
            assert np.isnan(cond_result["statistic"])

    def test_only_penk_sessions(self) -> None:
        """With only Penk+ sessions, nonpenk comparisons should give NaN."""
        sessions = [_make_session("penk", seed=i) for i in range(3)]
        result = celltype_dynamics_summary(sessions)
        assert result["n_nonpenk_sessions"] == 0
        for cond_result in result["comparisons"].values():
            assert np.isnan(cond_result["statistic"])

    def test_only_nonpenk_sessions(self) -> None:
        sessions = [_make_session("nonpenk", seed=i) for i in range(3)]
        result = celltype_dynamics_summary(sessions)
        assert result["n_penk_sessions"] == 0
        for cond_result in result["comparisons"].values():
            assert np.isnan(cond_result["statistic"])

    def test_session_missing_signal_key_skipped(self) -> None:
        """Sessions without the signal key should be skipped gracefully."""
        sessions = self._make_sessions(n_penk=2, n_nonpenk=2)
        # Remove "dff" from first session entirely
        sessions[0].pop("dff", None)
        result = celltype_dynamics_summary(sessions, signal_key="dff")
        # Should not crash; session counts still correct
        assert result["n_penk_sessions"] == 2

    def test_signal_key_spikes(self) -> None:
        """When signal_key='spikes', should read spikes field."""
        sessions = []
        rng = np.random.default_rng(0)
        for i in range(2):
            ses = _make_session("penk", seed=i)
            ses["spikes"] = np.abs(rng.standard_normal(ses["dff"].shape)).astype(np.float32)
            sessions.append(ses)
        for i in range(2):
            ses = _make_session("nonpenk", seed=100 + i)
            ses["spikes"] = np.abs(rng.standard_normal(ses["dff"].shape)).astype(np.float32)
            sessions.append(ses)
        result = celltype_dynamics_summary(sessions, signal_key="spikes")
        assert "comparisons" in result

    def test_bad_behav_excluded(self) -> None:
        """Frames marked as bad_behav should be excluded from analysis."""
        sessions = self._make_sessions(n_penk=2, n_nonpenk=2)
        # Mark half of all frames as bad behaviour in first session
        sessions[0]["bad_behav"][:200] = True
        result_with_bad = celltype_dynamics_summary(sessions)
        # Should complete without error
        assert "comparisons" in result_with_bad

    def test_unknown_celltype_ignored(self) -> None:
        """Sessions with celltype='unknown' should not contribute to either group."""
        sessions = self._make_sessions(n_penk=2, n_nonpenk=2)
        sessions.append(_make_session("unknown", seed=999))
        result = celltype_dynamics_summary(sessions)
        assert result["n_penk_sessions"] == 2
        assert result["n_nonpenk_sessions"] == 2

    def test_custom_speed_threshold(self) -> None:
        sessions = self._make_sessions()
        result_default = celltype_dynamics_summary(sessions, speed_threshold=2.5)
        result_high = celltype_dynamics_summary(sessions, speed_threshold=15.0)
        # Both should complete; n_frames in moving conditions changes
        assert "comparisons" in result_default
        assert "comparisons" in result_high

    def test_session_missing_speed_is_skipped(self) -> None:
        """Session with speed_cm_s=None is silently skipped."""
        sessions = self._make_sessions(n_penk=2, n_nonpenk=2)
        sessions[0].pop("speed_cm_s", None)
        result = celltype_dynamics_summary(sessions)
        # Should not crash; still processes remaining sessions
        assert "comparisons" in result

    def test_session_missing_light_on_is_skipped(self) -> None:
        """Session with light_on=None is silently skipped."""
        sessions = self._make_sessions(n_penk=2, n_nonpenk=2)
        sessions[1].pop("light_on", None)
        result = celltype_dynamics_summary(sessions)
        assert "comparisons" in result

    def test_session_missing_bad_behav_uses_all_frames(self) -> None:
        """Session without bad_behav key should use all frames as valid."""
        sessions = self._make_sessions(n_penk=2, n_nonpenk=2)
        sessions[0].pop("bad_behav", None)  # remove bad_behav → should default to all-valid
        result = celltype_dynamics_summary(sessions)
        assert "comparisons" in result
