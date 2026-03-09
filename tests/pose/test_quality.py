"""Tests for pose tracking quality diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.pose.quality import (
    body_length_consistency,
    detect_ear_distance_outliers,
    detect_frozen_keypoint,
    detect_jumps,
    ear_distance,
    likelihood_summary,
    session_quality_report,
    stratified_frame_selection,
    worst_frames,
)


class TestLikelihoodSummary:
    def test_perfect_likelihood(self):
        lik = np.ones(100)
        result = likelihood_summary(lik)
        assert result["mean"] == 1.0
        assert result["pct_above_90"] == 1.0
        assert result["n_frames"] == 100

    def test_mixed_likelihood(self):
        lik = np.array([0.1, 0.5, 0.8, 0.95, 1.0])
        result = likelihood_summary(lik)
        assert result["pct_above_90"] == pytest.approx(0.4)  # 2 out of 5
        assert result["pct_above_50"] == pytest.approx(0.8)  # 4 out of 5

    def test_empty_array(self):
        result = likelihood_summary(np.array([]))
        assert result["n_frames"] == 0
        assert np.isnan(result["mean"])


class TestDetectJumps:
    def test_no_jumps_smooth_trajectory(self):
        x = np.linspace(0, 100, 500)
        y = np.sin(x / 10) * 20 + 300
        jumps = detect_jumps(x, y, threshold_px=50.0)
        assert not jumps.any()

    def test_detects_teleport(self):
        x = np.ones(100) * 400.0
        y = np.ones(100) * 300.0
        # Insert teleport at frame 50
        x[50] = 100.0
        y[50] = 100.0
        jumps = detect_jumps(x, y, threshold_px=50.0)
        assert jumps[50]  # Jump TO the anomalous position
        assert jumps[51]  # Jump BACK from anomalous position

    def test_single_frame(self):
        jumps = detect_jumps(np.array([1.0]), np.array([1.0]))
        assert len(jumps) == 1
        assert not jumps[0]

    def test_threshold_sensitivity(self):
        x = np.array([0.0, 10.0, 20.0, 100.0, 110.0])
        y = np.zeros(5)
        # Default threshold 50px — only 20→100 is a jump (80px)
        jumps = detect_jumps(x, y, threshold_px=50.0)
        assert not jumps[1]  # 10px
        assert not jumps[2]  # 10px
        assert jumps[3]      # 80px


class TestDetectFrozenKeypoint:
    def test_moving_keypoint_not_frozen(self):
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.normal(0, 5, 200))
        y = np.cumsum(rng.normal(0, 5, 200))
        frozen = detect_frozen_keypoint(x, y, window=30)
        assert frozen.sum() < len(frozen) * 0.1  # Mostly not frozen

    def test_stuck_keypoint_detected(self):
        x = np.ones(100) * 400.0
        y = np.ones(100) * 300.0
        frozen = detect_frozen_keypoint(x, y, window=30)
        assert frozen.all()  # Completely stuck

    def test_short_sequence(self):
        x = np.ones(5) * 100.0
        y = np.ones(5) * 100.0
        frozen = detect_frozen_keypoint(x, y, window=30)
        assert not frozen.any()  # Too short to detect


class TestEarDistance:
    def test_known_distance(self):
        # Ears 10px apart horizontally
        dist = ear_distance(
            np.array([10.0]), np.array([0.0]),
            np.array([0.0]), np.array([0.0]),
        )
        assert dist[0] == pytest.approx(10.0)

    def test_diagonal_distance(self):
        dist = ear_distance(
            np.array([3.0]), np.array([4.0]),
            np.array([0.0]), np.array([0.0]),
        )
        assert dist[0] == pytest.approx(5.0)


class TestEarDistanceOutliers:
    def test_consistent_ears_no_outliers(self):
        rng = np.random.default_rng(42)
        n = 500
        # Ears consistently ~20px apart
        left_x = 300 + rng.normal(0, 2, n)
        left_y = 300 + rng.normal(0, 2, n)
        right_x = left_x - 20 + rng.normal(0, 1, n)
        right_y = left_y + rng.normal(0, 1, n)
        result = detect_ear_distance_outliers(left_x, left_y, right_x, right_y)
        assert result["n_outliers"] < n * 0.10  # Few outliers

    def test_swapped_ears_detected(self):
        n = 200
        left_x = np.ones(n) * 300
        left_y = np.ones(n) * 300
        right_x = np.ones(n) * 280
        right_y = np.ones(n) * 300
        # Swap ears at frames 50-60
        left_x[50:60] = 100
        left_y[50:60] = 100
        result = detect_ear_distance_outliers(left_x, left_y, right_x, right_y)
        assert result["n_outliers"] >= 10  # Swapped frames detected

    def test_too_few_frames(self):
        result = detect_ear_distance_outliers(
            np.array([1.0]), np.array([1.0]),
            np.array([2.0]), np.array([2.0]),
        )
        assert result["n_outliers"] == 0
        assert np.isnan(result["median"])


class TestBodyLengthConsistency:
    def test_consistent_body_length(self):
        rng = np.random.default_rng(42)
        n = 500
        head_x = 400 + rng.normal(0, 2, n)
        head_y = 300 + rng.normal(0, 2, n)
        tail_x = head_x - 80 + rng.normal(0, 2, n)
        tail_y = head_y + rng.normal(0, 2, n)
        result = body_length_consistency(head_x, head_y, tail_x, tail_y)
        assert result["median"] == pytest.approx(80, abs=5)
        assert result["n_outliers"] < n * 0.10

    def test_stretched_body_outlier(self):
        n = 200
        head_x = np.ones(n) * 400
        head_y = np.ones(n) * 300
        tail_x = np.ones(n) * 320
        tail_y = np.ones(n) * 300
        # Extreme stretch at frame 100
        tail_x[100] = 0.0
        result = body_length_consistency(head_x, head_y, tail_x, tail_y)
        assert result["is_outlier"][100]


class TestSessionQualityReport:
    def test_good_session(self):
        rng = np.random.default_rng(42)
        n = 1000
        kp_data = {
            "left_ear": {
                "x": 300 + np.cumsum(rng.normal(0, 1, n)),
                "y": 300 + np.cumsum(rng.normal(0, 1, n)),
                "likelihood": np.ones(n) * 0.99,
            },
            "right_ear": {
                "x": 280 + np.cumsum(rng.normal(0, 1, n)),
                "y": 300 + np.cumsum(rng.normal(0, 1, n)),
                "likelihood": np.ones(n) * 0.98,
            },
        }
        report = session_quality_report(kp_data)
        assert report["overall_score"] > 80
        assert report["pct_good"] > 0.9
        assert report["n_frames"] == n
        assert len(report["issues"]) == 0

    def test_bad_session(self):
        n = 1000
        kp_data = {
            "nose": {
                "x": np.ones(n) * 100,
                "y": np.ones(n) * 100,
                "likelihood": np.ones(n) * 0.3,
            },
        }
        report = session_quality_report(kp_data)
        assert report["overall_score"] < 50
        assert len(report["issues"]) > 0

    def test_empty_data(self):
        report = session_quality_report({})
        assert report["overall_score"] == 0.0


class TestWorstFrames:
    def test_selects_lowest_likelihood(self):
        lik = np.ones(100) * 0.99
        lik[10] = 0.1
        lik[50] = 0.2
        lik[80] = 0.3
        indices = worst_frames(lik, n_frames=3, min_spacing=5)
        assert 10 in indices
        assert 50 in indices
        assert 80 in indices

    def test_spacing_enforced(self):
        lik = np.ones(100) * 0.99
        lik[10:15] = 0.1  # 5 consecutive bad frames
        indices = worst_frames(lik, n_frames=3, min_spacing=10)
        # Can only pick one from the cluster of 5
        in_cluster = sum(1 for i in indices if 10 <= i <= 14)
        assert in_cluster == 1

    def test_2d_input(self):
        lik = np.ones((100, 3)) * 0.99
        lik[5, :] = 0.1
        indices = worst_frames(lik, n_frames=1)
        assert 5 in indices


class TestStratifiedFrameSelection:
    def test_covers_quality_range(self):
        rng = np.random.default_rng(42)
        # Uniformly distributed likelihoods
        lik = rng.uniform(0, 1, 1000)
        result = stratified_frame_selection(lik, n_per_bin=3, min_spacing=10)
        assert result["total_selected"] > 0
        assert len(result["bins"]) == 4
        assert len(result["indices"]) == result["total_selected"]

    def test_all_good_still_selects(self):
        lik = np.ones(500) * 0.99
        result = stratified_frame_selection(lik, n_per_bin=3, min_spacing=10)
        # All frames are in the "good" bin
        assert result["total_selected"] >= 1
