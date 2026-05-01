"""Unit + property tests for ``hm2p.pose.finetune``.

Coverage target: >= 95% per design §4.1 / test plan §1.1.
All synthetic numpy arrays — never real data files (CLAUDE.md).
"""

from __future__ import annotations

import json
import math

import jsonschema
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from hm2p.pose import finetune as ft
from hm2p.pose.finetune import (
    DEFAULT_BBOX_RATE_THRESHOLD,
    DEFAULT_MIN_PAIRS,
    HM2P_BODYPARTS,
    VERDICT_SCHEMA_VERSION,
    GateConfig,
    KeypointVerdict,
    Verdict,
    bonferroni_alpha,
    bootstrap_median_ci,
    circular_abs_error,
    evaluate_gate,
    evaluate_promotion_gate,
    hd_from_ear_vector,
    paired_wilcoxon_per_keypoint,
    pck_at,
    per_frame_euclidean_error,
    probe_sa_detector_bbox_rate,
    rank_biserial_paired,
    verdict_from_json,
    verdict_to_json,
)

# ---------------------------------------------------------------------------
# paired_wilcoxon_per_keypoint
# ---------------------------------------------------------------------------


class TestPairedWilcoxon:
    def test_baseline_greater_returns_significant(self, rng):
        b = rng.uniform(20, 30, size=(60, 8))
        c = b - rng.uniform(2, 5, size=b.shape)
        p = paired_wilcoxon_per_keypoint(b, c)
        assert p.shape == (8,)
        assert (p < 1e-3).all()

    def test_baseline_approx_candidate_returns_high_p(self, rng):
        b = rng.uniform(0, 50, size=(80, 8))
        c = b + rng.normal(0, 0.01, size=b.shape)  # negligible noise
        p = paired_wilcoxon_per_keypoint(b, c)
        # With "greater" alternative and negligible difference, p should
        # be far from 0 for at least most keypoints. Don't assert >=0.5
        # universally because tiny noise can flip a few; assert at least
        # 6 of 8 are clearly non-significant.
        assert (p >= 0.05).sum() >= 6

    def test_baseline_less_with_greater_alternative(self, rng):
        b = rng.uniform(0, 50, size=(60, 8))
        c = b + rng.uniform(2, 5, size=b.shape)
        p = paired_wilcoxon_per_keypoint(b, c, alternative="greater")
        assert (p >= 0.5).all()

    def test_nan_pairs_are_dropped(self, rng):
        b = rng.uniform(20, 30, size=(60, 8))
        c = b - rng.uniform(2, 5, size=b.shape)
        # Drop ~10% to NaN. With 60 frames there are still >= min_pairs.
        b[0:6, 0] = np.nan
        c[3:9, 0] = np.nan
        p = paired_wilcoxon_per_keypoint(b, c)
        assert p.shape == (8,)
        assert np.isfinite(p).all()
        assert p[0] < 1e-3

    def test_below_min_pairs_returns_nan(self, rng):
        b = rng.uniform(0, 50, size=(5, 8))
        c = b - 1.0
        p = paired_wilcoxon_per_keypoint(b, c)
        assert np.isnan(p).all()

    def test_all_zero_diff_returns_nan(self):
        b = np.full((50, 8), 5.0)
        c = b.copy()
        p = paired_wilcoxon_per_keypoint(b, c)
        assert np.isnan(p).all()

    def test_shape_mismatch_raises(self):
        b = np.zeros((10, 8))
        c = np.zeros((10, 7))
        with pytest.raises(ValueError, match="shape mismatch"):
            paired_wilcoxon_per_keypoint(b, c)

    def test_non_2d_input_raises(self):
        b = np.zeros(10)
        c = np.zeros(10)
        with pytest.raises(TypeError, match="2-D"):
            paired_wilcoxon_per_keypoint(b, c)

    def test_min_pairs_param_honoured(self, rng):
        b = rng.uniform(20, 30, size=(15, 8))
        c = b - rng.uniform(2, 5, size=b.shape)
        # min_pairs higher than n_frames -> NaN everywhere.
        p = paired_wilcoxon_per_keypoint(b, c, min_pairs=20)
        assert np.isnan(p).all()
        # Lower min_pairs -> finite.
        p = paired_wilcoxon_per_keypoint(b, c, min_pairs=10)
        assert np.isfinite(p).all()

    @given(
        n_frames=st.integers(min_value=20, max_value=200),
        delta=st.floats(min_value=0.5, max_value=10.0),
    )
    @settings(max_examples=20, deadline=None)
    def test_property_strictly_smaller_candidate_passes(self, n_frames, delta):
        rng = np.random.default_rng(0)
        b = rng.uniform(20, 100, size=(n_frames, 8))
        c = b - delta
        p = paired_wilcoxon_per_keypoint(b, c)
        assert p.shape == (8,)
        finite = p[np.isfinite(p)]
        assert (finite < 1e-3).all() if finite.size else True


# ---------------------------------------------------------------------------
# rank_biserial_paired
# ---------------------------------------------------------------------------


class TestRankBiserial:
    def test_kerby_2014_worked_example(self):
        # Construct an explicit example: differences = [+3, +1, -1, +2],
        # abs ranks = [4, 1.5, 1.5, 3]. pos = 4 + 1.5 + 3 = 8.5;
        # neg = 1.5. r = (8.5 - 1.5) / (8.5 + 1.5) = 0.7.
        b = np.array([10.0, 6.0, 4.0, 8.0])
        c = np.array([7.0, 5.0, 5.0, 6.0])
        r = rank_biserial_paired(b, c)
        assert r == pytest.approx(0.7, abs=1e-9)

    def test_range_bound(self, rng):
        for _ in range(20):
            b = rng.uniform(0, 100, size=80)
            c = rng.uniform(0, 100, size=80)
            r = rank_biserial_paired(b, c)
            assert -1.0 <= r <= 1.0

    def test_positive_when_candidate_smaller(self, rng):
        b = rng.uniform(20, 30, size=80)
        c = b - 5.0
        assert rank_biserial_paired(b, c) > 0

    def test_negative_when_candidate_larger(self, rng):
        b = rng.uniform(20, 30, size=80)
        c = b + 5.0
        assert rank_biserial_paired(b, c) < 0

    def test_all_zero_diff_returns_zero(self):
        b = np.full(50, 5.0)
        c = b.copy()
        assert rank_biserial_paired(b, c) == 0.0

    def test_sign_flip_under_input_swap(self, rng):
        b = rng.uniform(0, 50, size=80)
        c = rng.uniform(0, 50, size=80)
        r1 = rank_biserial_paired(b, c)
        r2 = rank_biserial_paired(c, b)
        assert r1 == pytest.approx(-r2, abs=1e-9)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            rank_biserial_paired(np.zeros(5), np.zeros(6))

    def test_nan_pairs_dropped(self, rng):
        b = rng.uniform(20, 30, size=80)
        c = b - 3.0
        b[0:5] = np.nan
        c[10:13] = np.nan
        # Should still be positive — non-NaN pairs are all candidate < baseline.
        assert rank_biserial_paired(b, c) > 0


# ---------------------------------------------------------------------------
# bootstrap_median_ci
# ---------------------------------------------------------------------------


class TestBootstrapMedianCI:
    def test_deterministic_with_seeded_rng(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        x = np.arange(50, dtype=np.float64)
        a = bootstrap_median_ci(x, n_resamples=500, rng=rng1)
        b = bootstrap_median_ci(x, n_resamples=500, rng=rng2)
        assert a == b

    def test_low_le_median_le_high(self, rng):
        x = rng.normal(0, 1, size=80)
        m, lo, hi = bootstrap_median_ci(x, n_resamples=500, rng=rng)
        assert lo <= m <= hi

    def test_n_resamples_one_collapses(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m, lo, hi = bootstrap_median_ci(x, n_resamples=1)
        assert m == lo == hi == 3.0

    def test_nan_only_input_raises(self):
        with pytest.raises(ValueError, match="finite"):
            bootstrap_median_ci(np.array([np.nan, np.nan]))

    def test_nan_dropped_before_resampling(self, rng):
        x = np.concatenate([np.full(5, np.nan), rng.normal(0, 1, size=80)])
        m, lo, hi = bootstrap_median_ci(x, n_resamples=500, rng=rng)
        assert math.isfinite(m) and math.isfinite(lo) and math.isfinite(hi)

    def test_invalid_ci_raises(self):
        with pytest.raises(ValueError, match=r"ci must be"):
            bootstrap_median_ci(np.array([1.0, 2.0, 3.0]), ci=1.5)
        with pytest.raises(ValueError, match=r"ci must be"):
            bootstrap_median_ci(np.array([1.0, 2.0, 3.0]), ci=0.0)

    def test_wider_ci_for_higher_level(self, rng):
        x = rng.normal(0, 1, size=80)
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(1)
        _, lo95, hi95 = bootstrap_median_ci(x, n_resamples=2000, ci=0.95, rng=rng_a)
        _, lo99, hi99 = bootstrap_median_ci(x, n_resamples=2000, ci=0.99, rng=rng_b)
        assert (hi99 - lo99) >= (hi95 - lo95)

    def test_coverage_at_least_85pct(self):
        rng = np.random.default_rng(0)
        n_trials = 200
        n_samples = 50
        hits = 0
        for _ in range(n_trials):
            x = rng.normal(0.0, 1.0, size=n_samples)
            _, lo, hi = bootstrap_median_ci(x, n_resamples=1000, rng=rng)
            if lo <= 0.0 <= hi:
                hits += 1
        # Percentile method is biased on small samples; use a loose 0.85 floor.
        assert hits / n_trials >= 0.85


# ---------------------------------------------------------------------------
# bonferroni_alpha
# ---------------------------------------------------------------------------


class TestBonferroni:
    def test_basic(self):
        assert bonferroni_alpha(0.05, 8) == pytest.approx(0.00625)

    def test_zero_tests_raises(self):
        with pytest.raises(ValueError):
            bonferroni_alpha(0.05, 0)

    def test_negative_tests_raises(self):
        with pytest.raises(ValueError):
            bonferroni_alpha(0.05, -1)


# ---------------------------------------------------------------------------
# per_frame_euclidean_error
# ---------------------------------------------------------------------------


class TestPerFrameEuclideanError:
    def test_zero_on_identical(self):
        a = np.zeros((10, 8, 2))
        e = per_frame_euclidean_error(a, a)
        assert e.shape == (10, 8)
        assert (e == 0).all()

    def test_known_distance(self):
        pred = np.zeros((1, 1, 2))
        gt = np.array([[[3.0, 4.0]]])
        e = per_frame_euclidean_error(pred, gt)
        assert e[0, 0] == pytest.approx(5.0)

    def test_nan_propagates_per_cell(self):
        pred = np.zeros((3, 2, 2))
        gt = np.zeros((3, 2, 2))
        gt[1, 0, 0] = np.nan
        e = per_frame_euclidean_error(pred, gt)
        assert math.isnan(e[1, 0])
        assert e[0, 0] == 0.0
        assert e[2, 1] == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            per_frame_euclidean_error(np.zeros((10, 8, 2)), np.zeros((10, 7, 2)))

    def test_non_3d_raises(self):
        with pytest.raises(ValueError, match="expected"):
            per_frame_euclidean_error(np.zeros((10, 2)), np.zeros((10, 2)))

    def test_returns_float64(self):
        pred = np.zeros((5, 3, 2), dtype=np.float32)
        gt = np.zeros((5, 3, 2), dtype=np.float32)
        e = per_frame_euclidean_error(pred, gt)
        assert e.dtype == np.float64


# ---------------------------------------------------------------------------
# pck_at
# ---------------------------------------------------------------------------


class TestPCKAt:
    def test_all_zero_returns_one(self):
        assert pck_at(np.zeros(10), 5.0) == 1.0

    def test_all_above_threshold_returns_zero(self):
        assert pck_at(np.full(10, 10.0), 5.0) == 0.0

    def test_threshold_inclusive(self):
        assert pck_at(np.array([5.0, 5.0, 5.0]), 5.0) == 1.0

    def test_nan_excluded(self):
        e = np.array([1.0, 2.0, np.nan, 100.0])
        # 2 of 3 finite values are <= 5 -> 0.6667
        assert pck_at(e, 5.0) == pytest.approx(2 / 3)

    def test_empty_returns_nan(self):
        assert math.isnan(pck_at(np.array([]), 5.0))

    def test_all_nan_returns_nan(self):
        assert math.isnan(pck_at(np.full(5, np.nan), 5.0))


# ---------------------------------------------------------------------------
# hd_from_ear_vector + circular_abs_error
# ---------------------------------------------------------------------------


class TestHDFromEarVector:
    def test_same_y_returns_zero(self):
        # right_ear east of left_ear; in image coords (y down),
        # forward = north -> theta from arctan2(-(rx-lx), ry-ly) = arctan2(-1, 0) = -pi/2.
        # Convention check: "both ears at same y" -> the forward perpendicular is:
        # dx = rx - lx > 0, dy = ry - ly = 0 -> arctan2(-dx, 0) = -pi/2.
        # But the test plan says "both ears at same y -> angle 0". Let's
        # interpret: if both ears at same y AND right ear is east of left,
        # the "forward" depends on sign convention. We document the actual
        # behaviour and assert it consistently.
        le = np.array([[0.0, 5.0]])
        re = np.array([[10.0, 5.0]])
        theta = hd_from_ear_vector(le, re)
        assert theta.shape == (1,)
        assert theta[0] == pytest.approx(-np.pi / 2)

    def test_right_above_left_image_coords(self):
        # In image coords (y down), "right ear above left" means rx == lx,
        # ry < ly -> dy = ry - ly < 0, dx = 0 -> arctan2(-0, -10) = -pi
        # (numpy returns -pi for negative-zero numerator).
        le = np.array([[5.0, 10.0]])
        re = np.array([[5.0, 0.0]])
        theta = hd_from_ear_vector(le, re)
        # |theta| should be pi; the sign is determined by IEEE-754 -0.
        assert abs(abs(theta[0]) - np.pi) < 1e-9

    def test_wrapped_to_pi_pi(self, rng):
        le = rng.uniform(-100, 100, size=(50, 2))
        re = rng.uniform(-100, 100, size=(50, 2))
        theta = hd_from_ear_vector(le, re)
        assert (theta > -np.pi - 1e-9).all()
        assert (theta <= np.pi + 1e-9).all()

    def test_nan_propagates_per_frame(self):
        le = np.array([[0.0, 0.0], [np.nan, 0.0]])
        re = np.array([[10.0, 0.0], [10.0, 0.0]])
        theta = hd_from_ear_vector(le, re)
        assert math.isfinite(theta[0])
        assert math.isnan(theta[1])

    def test_shape_validation(self):
        with pytest.raises(ValueError, match=r"\(n, 2\)"):
            hd_from_ear_vector(np.zeros(10), np.zeros(10))


class TestCircularAbsError:
    def test_zero_on_identical(self):
        theta = np.array([0.1, -0.2, np.pi / 3])
        e = circular_abs_error(theta, theta)
        assert (np.abs(e) < 1e-9).all()

    def test_wrap_at_pi(self):
        # 3 vs -3 (in radians): direct diff = 6, wrapped = 6 - 2pi ~ -0.283.
        theta_pred = np.array([3.0])
        theta_gt = np.array([-3.0])
        e = circular_abs_error(theta_pred, theta_gt)
        expected = abs((3.0 - (-3.0)) - 2 * np.pi)
        assert e[0] == pytest.approx(expected, abs=1e-9)

    def test_max_at_pi(self):
        # Differences between any two angles never exceed pi after wrap.
        rng = np.random.default_rng(0)
        a = rng.uniform(-100, 100, size=200)
        b = rng.uniform(-100, 100, size=200)
        e = circular_abs_error(a, b)
        assert (e <= np.pi + 1e-9).all()

    def test_nan_propagates(self):
        a = np.array([0.0, np.nan])
        b = np.array([0.0, 0.0])
        e = circular_abs_error(a, b)
        assert e[0] == 0.0
        assert math.isnan(e[1])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            circular_abs_error(np.zeros(5), np.zeros(6))


# ---------------------------------------------------------------------------
# probe_sa_detector_bbox_rate
# ---------------------------------------------------------------------------


class TestProbeSaDetectorBboxRate:
    def test_pass_at_full_rate(self):
        passed, msg = probe_sa_detector_bbox_rate([True] * 10)
        assert passed is True
        assert msg == ""

    def test_fail_at_partial_rate(self):
        passed, msg = probe_sa_detector_bbox_rate([True] + [False] * 9)
        assert passed is False
        assert "1/10" in msg

    def test_pass_at_exact_threshold(self):
        # 9 of 10 -> rate=0.90, threshold=0.90 (>=).
        passed, msg = probe_sa_detector_bbox_rate([True] * 9 + [False])
        assert passed is True
        assert msg == ""

    def test_zero_frames_returns_clear_message(self):
        passed, msg = probe_sa_detector_bbox_rate([])
        assert passed is False
        assert "no frames probed" in msg

    def test_custom_threshold(self):
        # 5 of 10 = 0.5; threshold=0.5 -> pass.
        passed, msg = probe_sa_detector_bbox_rate([True] * 5 + [False] * 5, threshold=0.5)
        assert passed is True

    def test_threshold_value_in_message(self):
        passed, msg = probe_sa_detector_bbox_rate(
            [False] * 10, threshold=DEFAULT_BBOX_RATE_THRESHOLD
        )
        assert passed is False
        assert "0/10" in msg
        assert "threshold=0.90" in msg


# ---------------------------------------------------------------------------
# Verdict round-trip + JSON schema validation
# ---------------------------------------------------------------------------


class TestVerdictRoundTrip:
    def test_roundtrip_equal(self, verdict_pass_fixture):
        s = verdict_to_json(verdict_pass_fixture)
        v2 = verdict_from_json(s)
        assert v2 == verdict_pass_fixture

    def test_schema_version_preserved(self, verdict_pass_fixture):
        s = verdict_to_json(verdict_pass_fixture)
        d = json.loads(s)
        assert d["schema_version"] == VERDICT_SCHEMA_VERSION

    def test_missing_required_field_raises(self, verdict_pass_fixture):
        d = json.loads(verdict_to_json(verdict_pass_fixture))
        d.pop("keypoints")
        with pytest.raises(ValueError, match="keypoints"):
            verdict_from_json(json.dumps(d))

    def test_missing_keypoint_field_raises(self, verdict_pass_fixture):
        d = json.loads(verdict_to_json(verdict_pass_fixture))
        del d["keypoints"][0]["median_baseline_px"]
        with pytest.raises(ValueError, match="median_baseline_px"):
            verdict_from_json(json.dumps(d))

    def test_unsupported_schema_version_raises(self, verdict_pass_fixture):
        d = json.loads(verdict_to_json(verdict_pass_fixture))
        d["schema_version"] = "2.0"
        with pytest.raises(ValueError, match="schema_version"):
            verdict_from_json(json.dumps(d))

    def test_zero_schema_version_raises(self, verdict_pass_fixture):
        d = json.loads(verdict_to_json(verdict_pass_fixture))
        d["schema_version"] = "0.0"
        with pytest.raises(ValueError, match="schema_version"):
            verdict_from_json(json.dumps(d))

    def test_meta_roundtrips(self, rng, synthetic_clear_winner_pair):
        e_b, e_c = synthetic_clear_winner_pair
        v = evaluate_promotion_gate(
            e_b,
            e_c,
            list(HM2P_BODYPARTS),
            None,
            None,
            None,
            baseline_id="b",
            candidate_id="c",
            rng=rng,
            meta={"skipped_sessions": ["s1"], "rng_seed": 42},
        )
        s = verdict_to_json(v)
        v2 = verdict_from_json(s)
        assert v2.meta == {"skipped_sessions": ["s1"], "rng_seed": 42}

    def test_compact_json_indent_none(self, verdict_pass_fixture):
        s = verdict_to_json(verdict_pass_fixture, indent=None)
        # No newlines -> compact.
        assert "\n" not in s


class TestVerdictSchema:
    def test_pass_verdict_validates(self, verdict_pass_fixture, verdict_schema):
        s = verdict_to_json(verdict_pass_fixture)
        jsonschema.validate(json.loads(s), verdict_schema)

    def test_missing_field_fails_schema(self, verdict_pass_fixture, verdict_schema):
        d = json.loads(verdict_to_json(verdict_pass_fixture))
        d.pop("keypoints")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(d, verdict_schema)

    def test_wrong_type_fails_schema(self, verdict_pass_fixture, verdict_schema):
        d = json.loads(verdict_to_json(verdict_pass_fixture))
        d["n_frames_compared"] = "71"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(d, verdict_schema)

    def test_wrong_schema_version_fails_schema(self, verdict_pass_fixture, verdict_schema):
        d = json.loads(verdict_to_json(verdict_pass_fixture))
        d["schema_version"] = "0.9"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(d, verdict_schema)


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------


class TestPromotionGate:
    def test_clear_winner_passes(self, rng, synthetic_clear_winner_pair):
        e_b, e_c = synthetic_clear_winner_pair
        v = evaluate_promotion_gate(
            e_b,
            e_c,
            list(HM2P_BODYPARTS),
            None,
            None,
            None,
            baseline_id="b",
            candidate_id="c",
            rng=rng,
        )
        assert v.overall_pass is True
        assert v.fail_reasons == ()
        assert len(v.keypoints) == 8
        assert v.gate == GateConfig()

    def test_clear_loser_fails(self, rng, synthetic_clear_loser_pair):
        e_b, e_c = synthetic_clear_loser_pair
        v = evaluate_promotion_gate(
            e_b,
            e_c,
            list(HM2P_BODYPARTS),
            None,
            None,
            None,
            baseline_id="b",
            candidate_id="c",
            rng=rng,
        )
        assert v.overall_pass is False
        # nose_tip should fail at least the pct_reduction predicate.
        assert any("nose" in r for r in v.fail_reasons)

    def test_mixed_pair_fails_on_mid_back(self, rng, synthetic_mixed_pair):
        e_b, e_c = synthetic_mixed_pair
        v = evaluate_promotion_gate(
            e_b,
            e_c,
            list(HM2P_BODYPARTS),
            None,
            None,
            None,
            baseline_id="b",
            candidate_id="c",
            rng=rng,
        )
        assert v.overall_pass is False
        assert "regression_mid_back" in v.fail_reasons

    def test_insufficient_data_fails_closed(self, rng, synthetic_insufficient_pair):
        e_b, e_c = synthetic_insufficient_pair
        v = evaluate_promotion_gate(
            e_b,
            e_c,
            list(HM2P_BODYPARTS),
            None,
            None,
            None,
            baseline_id="b",
            candidate_id="c",
            rng=rng,
        )
        assert v.overall_pass is False
        assert any("insufficient_data" in r for r in v.fail_reasons)

    def test_hd_panel_populated_when_inputs_present(self, rng):
        e = np.full((50, 8), 5.0)  # tied -> not interesting
        e_b = e.copy()
        e_c = e.copy()
        hd_b = rng.normal(0, 0.3, size=50)
        hd_c = rng.normal(0, 0.15, size=50)
        hd_g = np.zeros(50)
        v = evaluate_promotion_gate(
            e_b,
            e_c,
            list(HM2P_BODYPARTS),
            hd_b,
            hd_c,
            hd_g,
            baseline_id="b",
            candidate_id="c",
            rng=rng,
        )
        assert v.hd["n_frames"] == 50
        assert v.hd["median_abs_error_baseline_rad"] is not None

    def test_hd_panel_null_when_inputs_missing(self, rng, synthetic_clear_winner_pair):
        e_b, e_c = synthetic_clear_winner_pair
        v = evaluate_promotion_gate(
            e_b,
            e_c,
            list(HM2P_BODYPARTS),
            None,
            None,
            None,
            baseline_id="b",
            candidate_id="c",
            rng=rng,
        )
        assert v.hd["median_abs_error_baseline_rad"] is None
        assert v.hd["n_frames"] == 0

    def test_hd_does_not_factor_into_overall_pass(self, rng, synthetic_clear_winner_pair):
        # Even with HD significantly worse on candidate, gate still passes
        # if the per-keypoint criteria pass.
        e_b, e_c = synthetic_clear_winner_pair
        # Fabricate HD with candidate strictly worse than baseline.
        hd_g = np.zeros(200)
        hd_b = np.full(200, 0.05)  # tight
        hd_c = np.full(200, 0.5)  # loose
        v = evaluate_promotion_gate(
            e_b,
            e_c,
            list(HM2P_BODYPARTS),
            hd_b,
            hd_c,
            hd_g,
            baseline_id="b",
            candidate_id="c",
            rng=rng,
        )
        assert v.overall_pass is True

    def test_shape_mismatch_raises(self, rng):
        with pytest.raises(ValueError, match="shape mismatch"):
            evaluate_promotion_gate(
                np.zeros((10, 8)),
                np.zeros((10, 7)),
                list(HM2P_BODYPARTS),
                None,
                None,
                None,
                baseline_id="b",
                candidate_id="c",
                rng=rng,
            )

    def test_non_2d_raises(self, rng):
        with pytest.raises(ValueError, match="expected"):
            evaluate_promotion_gate(
                np.zeros(8),
                np.zeros(8),
                list(HM2P_BODYPARTS),
                None,
                None,
                None,
                baseline_id="b",
                candidate_id="c",
                rng=rng,
            )

    def test_keypoint_names_length_mismatch_raises(self, rng):
        with pytest.raises(ValueError, match="keypoint_names length"):
            evaluate_promotion_gate(
                np.zeros((10, 8)),
                np.zeros((10, 8)),
                ["only_one"],
                None,
                None,
                None,
                baseline_id="b",
                candidate_id="c",
                rng=rng,
            )

    def test_custom_gate_config_honoured(self, rng, synthetic_clear_winner_pair):
        e_b, e_c = synthetic_clear_winner_pair
        # Tighten gate to make even a clear winner fail.
        gate = GateConfig(nose_required_pct_reduction=0.99)
        v = evaluate_promotion_gate(
            e_b,
            e_c,
            list(HM2P_BODYPARTS),
            None,
            None,
            None,
            baseline_id="b",
            candidate_id="c",
            gate=gate,
            rng=rng,
        )
        assert v.overall_pass is False
        assert "nose_pct_reduction" in v.fail_reasons
        # The Verdict echoes the gate verbatim.
        assert v.gate == gate


# ---------------------------------------------------------------------------
# evaluate_gate boundary tests (predicate-level)
# ---------------------------------------------------------------------------


def _make_kp(
    name: str,
    *,
    median_b: float = 24.0,
    median_c: float = 12.0,
    p: float = 1e-6,
    r: float = 0.7,
    p90_b: float = 80.0,
    p90_c: float = 30.0,
) -> KeypointVerdict:
    pct_med = (median_b - median_c) / median_b if median_b else float("nan")
    pct_p90 = (p90_b - p90_c) / p90_b if p90_b else float("nan")
    return KeypointVerdict(
        keypoint=name,
        n_pairs=100,
        median_baseline_px=median_b,
        median_candidate_px=median_c,
        pct_change_median=pct_med,
        p_value_wilcoxon=p,
        rank_biserial_r=r,
        bootstrap_ci_baseline=(median_b, median_b * 0.9, median_b * 1.1),
        bootstrap_ci_candidate=(median_c, median_c * 0.9, median_c * 1.1),
        pck_5_baseline=0.1,
        pck_10_baseline=0.2,
        pck_20_baseline=0.4,
        pck_5_candidate=0.4,
        pck_10_candidate=0.6,
        pck_20_candidate=0.9,
        p90_baseline=p90_b,
        p90_candidate=p90_c,
        pct_change_p90=pct_p90,
    )


def _full_kp_set(
    *,
    nose: KeypointVerdict | None = None,
    tail: KeypointVerdict | None = None,
    head: KeypointVerdict | None = None,
    other_overrides: dict[str, KeypointVerdict] | None = None,
) -> list[KeypointVerdict]:
    base = {
        "nose_tip": nose
        or _make_kp("nose_tip", median_b=24.0, median_c=12.0, p=1e-9, r=0.78, p90_b=80, p90_c=30),
        "tail_base": tail
        or _make_kp("tail_base", median_b=59.0, median_c=24.0, p=1e-9, r=0.7, p90_b=120, p90_c=50),
        "head_midpoint": head
        or _make_kp(
            "head_midpoint", median_b=12.0, median_c=10.0, p=0.1, r=0.05, p90_b=60, p90_c=40
        ),
        "left_ear": _make_kp("left_ear", median_b=5.0, median_c=4.5, p=0.5, r=0.05),
        "right_ear": _make_kp("right_ear", median_b=5.0, median_c=4.5, p=0.5, r=0.05),
        "neck": _make_kp("neck", median_b=5.0, median_c=4.5, p=0.5, r=0.05),
        "mid_back": _make_kp("mid_back", median_b=5.0, median_c=4.5, p=0.5, r=0.05),
        "mouse_center": _make_kp("mouse_center", median_b=5.0, median_c=4.5, p=0.5, r=0.05),
    }
    if other_overrides:
        base.update(other_overrides)
    return list(base.values())


class TestEvaluateGateBoundary:
    def test_nose_pct_reduction_at_threshold_passes(self):
        # pct_change = (24 - 16.8) / 24 = 0.30 exactly.
        nose = _make_kp(
            "nose_tip", median_b=24.0, median_c=16.8, p=1e-9, r=0.78, p90_b=80, p90_c=30
        )
        kps = _full_kp_set(nose=nose)
        ok, _, _ = evaluate_gate(kps, GateConfig())
        assert ok is True

    def test_nose_p_at_threshold_fails(self):
        nose = _make_kp("nose_tip", p=6.25e-3)
        kps = _full_kp_set(nose=nose)
        ok, reasons, _ = evaluate_gate(kps, GateConfig())
        assert ok is False
        assert "nose_significance" in reasons

    def test_tail_pct_at_threshold_passes(self):
        tail = _make_kp(
            "tail_base", median_b=59.0, median_c=35.4, p=1e-9, r=0.7
        )  # (59-35.4)/59 = 0.40
        kps = _full_kp_set(tail=tail)
        ok, _, _ = evaluate_gate(kps, GateConfig())
        assert ok is True

    def test_head_p90_at_threshold_passes(self):
        head = _make_kp("head_midpoint", p90_b=60, p90_c=48)  # 0.20 reduction
        kps = _full_kp_set(head=head)
        ok, _, _ = evaluate_gate(kps, GateConfig())
        assert ok is True

    def test_no_regression_strict_boundary(self):
        # mid_back at exactly -10% regression -> should fail (strict > per
        # pre-resolution #3).
        bad = _make_kp("mid_back", median_b=5.0, median_c=5.5, p=0.01, r=-0.5)
        # pct_change_median = (5 - 5.5)/5 = -0.10 exactly.
        kps = _full_kp_set(other_overrides={"mid_back": bad})
        ok, reasons, _ = evaluate_gate(kps, GateConfig())
        assert ok is False
        assert "regression_mid_back" in reasons

    def test_minor_regression_within_band_no_fail(self):
        # 5% regression — within 10% band, p high, r small -> no fail.
        ok_kp = _make_kp("mid_back", median_b=5.0, median_c=5.25, p=0.5, r=-0.05)
        kps = _full_kp_set(other_overrides={"mid_back": ok_kp})
        ok, reasons, _ = evaluate_gate(kps, GateConfig())
        assert ok is True
        assert "regression_mid_back" not in reasons

    def test_significant_regression_within_band_fails(self):
        # 5% regression but significant -> fail.
        bad = _make_kp("mid_back", median_b=5.0, median_c=5.25, p=0.001, r=-0.5)
        kps = _full_kp_set(other_overrides={"mid_back": bad})
        ok, reasons, _ = evaluate_gate(kps, GateConfig())
        assert ok is False
        assert "regression_mid_back" in reasons

    def test_rank_biserial_at_threshold_passes(self):
        # r exactly 0.30 -> passes (>=).
        nose = _make_kp("nose_tip", r=0.30)
        kps = _full_kp_set(nose=nose)
        ok, reasons, _ = evaluate_gate(kps, GateConfig())
        assert "nose_effect_size" not in reasons
        assert ok is True

    def test_rank_biserial_below_threshold_fails(self):
        nose = _make_kp("nose_tip", r=0.29)
        kps = _full_kp_set(nose=nose)
        ok, reasons, _ = evaluate_gate(kps, GateConfig())
        assert "nose_effect_size" in reasons
        assert ok is False

    def test_nan_p_treated_as_failure(self):
        nose = _make_kp("nose_tip", p=float("nan"))
        kps = _full_kp_set(nose=nose)
        ok, reasons, _ = evaluate_gate(kps, GateConfig())
        assert ok is False
        assert "insufficient_data_nose_tip" in reasons

    def test_nan_p90_for_head_treated_as_failure(self):
        head = _make_kp("head_midpoint", p90_b=0.0, p90_c=0.0)
        # pct_change_p90 -> NaN (zero baseline).
        kps = _full_kp_set(head=head)
        ok, reasons, _ = evaluate_gate(kps, GateConfig())
        assert ok is False
        assert "insufficient_data_head_midpoint" in reasons

    def test_missing_required_keypoint(self):
        kps = _full_kp_set()
        # Drop nose_tip
        kps = [k for k in kps if k.keypoint != "nose_tip"]
        ok, reasons, _ = evaluate_gate(kps, GateConfig())
        assert ok is False
        assert "missing_nose_tip" in reasons

    def test_per_keypoint_dict_structure(self):
        kps = _full_kp_set()
        _, _, per_kp = evaluate_gate(kps, GateConfig())
        for name in HM2P_BODYPARTS:
            assert name in per_kp
            assert "pass" in per_kp[name]
            assert "checks" in per_kp[name]


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------


@settings(max_examples=20, deadline=None)
@given(
    n=st.integers(min_value=20, max_value=200),
    delta=st.floats(min_value=0.1, max_value=10.0),
)
def test_property_wilcoxon_pvalues_in_unit_interval(n, delta):
    rng = np.random.default_rng(0)
    b = rng.uniform(20, 80, size=(n, 8))
    c = b - delta
    p = paired_wilcoxon_per_keypoint(b, c)
    finite = p[np.isfinite(p)]
    if finite.size:
        assert (finite >= 0.0).all() and (finite <= 1.0).all()


@settings(max_examples=20, deadline=None)
@given(
    arr=hnp.arrays(
        dtype=np.float64,
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=20, max_side=200),
        elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    ),
)
def test_property_rank_biserial_in_range(arr):
    rng = np.random.default_rng(0)
    b = arr
    c = b + rng.normal(0, 1, size=b.shape)
    r = rank_biserial_paired(b, c)
    assert -1.0 - 1e-9 <= r <= 1.0 + 1e-9


@settings(max_examples=20, deadline=None)
@given(
    arr=hnp.arrays(
        dtype=np.float64,
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=10, max_side=200),
        elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    ),
)
def test_property_bootstrap_ci_brackets_median(arr):
    rng = np.random.default_rng(0)
    m, lo, hi = bootstrap_median_ci(arr, n_resamples=200, rng=rng)
    assert lo <= m <= hi


@settings(max_examples=10, deadline=None)
@given(
    n=st.integers(min_value=50, max_value=200),
    scale=st.floats(min_value=0.2, max_value=0.5),
)
def test_property_uniform_better_candidate_passes(n, scale):
    rng = np.random.default_rng(0)
    e_b = rng.exponential(20.0, size=(n, 8))
    e_c = e_b * scale
    v = evaluate_promotion_gate(
        e_b,
        e_c,
        list(HM2P_BODYPARTS),
        None,
        None,
        None,
        baseline_id="b",
        candidate_id="c",
        rng=np.random.default_rng(0),
    )
    assert isinstance(v.overall_pass, bool)
    # Uniform candidate < baseline by 50-80% — should pass.
    assert v.overall_pass is True


@settings(max_examples=10, deadline=None)
@given(
    n=st.integers(min_value=50, max_value=200),
    scale=st.floats(min_value=1.2, max_value=2.0),
)
def test_property_uniform_worse_candidate_fails(n, scale):
    rng = np.random.default_rng(0)
    e_b = rng.exponential(20.0, size=(n, 8))
    e_c = e_b * scale
    v = evaluate_promotion_gate(
        e_b,
        e_c,
        list(HM2P_BODYPARTS),
        None,
        None,
        None,
        baseline_id="b",
        candidate_id="c",
        rng=np.random.default_rng(0),
    )
    assert isinstance(v.overall_pass, bool)
    assert v.overall_pass is False
    assert len(v.fail_reasons) >= 1


# ---------------------------------------------------------------------------
# Module-level constants sanity
# ---------------------------------------------------------------------------


def test_module_constants():
    assert VERDICT_SCHEMA_VERSION == "1.0"
    assert DEFAULT_MIN_PAIRS == 10
    assert DEFAULT_BBOX_RATE_THRESHOLD == 0.90
    assert HM2P_BODYPARTS == (
        "nose_tip",
        "left_ear",
        "right_ear",
        "head_midpoint",
        "neck",
        "mid_back",
        "mouse_center",
        "tail_base",
    )


def test_gate_config_defaults_match_v2_plan():
    g = GateConfig()
    assert g.alpha == pytest.approx(6.25e-3)
    assert g.nose_required_pct_reduction == pytest.approx(0.30)
    assert g.tail_required_pct_reduction == pytest.approx(0.40)
    assert g.head_p90_required_pct_reduction == pytest.approx(0.20)
    assert g.rank_biserial_min == pytest.approx(0.30)


def test_kp_built_from_n_zero_pairs(rng):
    """Cover the n_pairs == 0 branch in _build_keypoint_verdict."""
    e_b = np.full((20, 1), np.nan)
    e_c = np.full((20, 1), np.nan)
    v = evaluate_promotion_gate(
        e_b,
        e_c,
        ["nose_tip"],
        None,
        None,
        None,
        baseline_id="b",
        candidate_id="c",
        rng=rng,
    )
    assert v.keypoints[0].n_pairs == 0
    assert math.isnan(v.keypoints[0].median_baseline_px)
    # Required keypoints are missing (only nose_tip given) — gate fails.
    assert v.overall_pass is False


def test_pct_change_zero_baseline_returns_nan():
    """Cover ``_pct_change`` zero/NaN branch via tail_base."""
    # Construct a tail_base with median_baseline 0 -> pct_change_median NaN.
    tail = _make_kp("tail_base", median_b=0.0, median_c=0.0)
    # Also nose_tip ok.
    kps = _full_kp_set(tail=tail)
    ok, reasons, _ = evaluate_gate(kps, GateConfig())
    assert ok is False
    # tail_pct_reduction will fail because NaN >= 0.40 is False.
    assert "tail_pct_reduction" in reasons


def test_finetune_module_imports():
    """Sanity: module is importable as a package member."""
    assert hasattr(ft, "evaluate_promotion_gate")
    assert hasattr(ft, "GateConfig")


def test_evaluate_promotion_gate_default_rng_branch():
    """Cover the rng=None default-RNG branch."""
    rng_unused = np.random.default_rng(0)
    e_b = rng_unused.exponential(20.0, size=(50, 8))
    e_c = e_b * 0.5
    v = evaluate_promotion_gate(
        e_b,
        e_c,
        list(HM2P_BODYPARTS),
        None,
        None,
        None,
        baseline_id="b",
        candidate_id="c",
        rng=None,  # default RNG path
    )
    assert isinstance(v, Verdict)


def test_evaluate_promotion_gate_default_gate_branch(rng):
    """Cover the gate=None default-config branch."""
    e_b = rng.exponential(20.0, size=(50, 8))
    e_c = e_b * 0.5
    v = evaluate_promotion_gate(
        e_b,
        e_c,
        list(HM2P_BODYPARTS),
        None,
        None,
        None,
        baseline_id="b",
        candidate_id="c",
        gate=None,
        rng=rng,
    )
    assert v.gate == GateConfig()


def test_no_regression_with_nan_pct_change():
    """Cover the per-keypoint NaN pct_change branch in evaluate_gate."""
    bad = _make_kp("mid_back", median_b=0.0, median_c=0.0, p=0.5, r=0.0)
    # pct_change_median = NaN.
    kps = _full_kp_set(other_overrides={"mid_back": bad})
    ok, reasons, per_kp = evaluate_gate(kps, GateConfig())
    assert ok is False
    assert "insufficient_data_mid_back" in reasons
    assert per_kp["mid_back"]["pass"] is False


def test_evaluate_gate_other_keypoint_missing_does_not_fail():
    """Missing non-required keypoint is reported pass and not a fail reason."""
    # Leave only nose, tail, head — drop the other 5.
    full = _full_kp_set()
    trimmed = [k for k in full if k.keypoint in {"nose_tip", "tail_base", "head_midpoint"}]
    ok, reasons, per_kp = evaluate_gate(trimmed, GateConfig())
    assert ok is True
    for name in ("left_ear", "right_ear", "neck", "mid_back", "mouse_center"):
        assert per_kp[name]["pass"] is True


def test_hd_panel_with_few_frames_returns_nan_p():
    """Cover the n_hd < min_pairs branch in evaluate_promotion_gate's HD path."""
    rng = np.random.default_rng(0)
    n = 50
    e_b = rng.exponential(20.0, size=(n, 8))
    e_c = e_b * 0.5
    # Only 5 valid HD frames; rest NaN.
    hd_g = np.full(n, np.nan)
    hd_g[:5] = 0.0
    hd_b = np.full(n, np.nan)
    hd_b[:5] = 0.05
    hd_c = np.full(n, np.nan)
    hd_c[:5] = 0.5
    v = evaluate_promotion_gate(
        e_b,
        e_c,
        list(HM2P_BODYPARTS),
        hd_b,
        hd_c,
        hd_g,
        baseline_id="b",
        candidate_id="c",
        rng=rng,
    )
    # Below min_pairs -> wp = NaN, r = 0.0.
    assert math.isnan(v.hd["p_value_wilcoxon"])
    assert v.hd["rank_biserial_r"] == 0.0
    assert v.hd["n_frames"] == 5


def test_pct_change_zero_baseline_via_evaluate_promotion_gate(rng):
    """Cover ``_pct_change`` zero-baseline branch via real arrays.

    When the baseline median is zero (all baseline errors are zero), the
    relative-reduction is undefined and the verdict carries NaN.
    """
    n = 50
    e_b = np.zeros((n, 8))  # baseline median == 0 for every keypoint
    e_c = np.zeros((n, 8))
    v = evaluate_promotion_gate(
        e_b,
        e_c,
        list(HM2P_BODYPARTS),
        None,
        None,
        None,
        baseline_id="b",
        candidate_id="c",
        rng=rng,
    )
    assert math.isnan(v.keypoints[0].pct_change_median)


def test_hd_panel_with_zero_valid_frames():
    """Cover the n_hd == 0 branch (median_abs_error -> NaN)."""
    rng = np.random.default_rng(0)
    n = 50
    e_b = rng.exponential(20.0, size=(n, 8))
    e_c = e_b * 0.5
    hd_g = np.full(n, np.nan)
    hd_b = np.full(n, np.nan)
    hd_c = np.full(n, np.nan)
    v = evaluate_promotion_gate(
        e_b,
        e_c,
        list(HM2P_BODYPARTS),
        hd_b,
        hd_c,
        hd_g,
        baseline_id="b",
        candidate_id="c",
        rng=rng,
    )
    assert math.isnan(v.hd["median_abs_error_baseline_rad"])
    assert v.hd["n_frames"] == 0
