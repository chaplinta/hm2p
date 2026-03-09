"""Tests for Tracking Quality page logic."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.pose.quality import (
    body_length_consistency,
    detect_ear_distance_outliers,
    detect_frozen_keypoint,
    detect_jumps,
    session_quality_report,
    stratified_frame_selection,
    worst_frames,
)
from hm2p.pose.retrain import select_retraining_frames


class TestTrackingQualityWorkflow:
    """End-to-end tests mimicking the page workflow."""

    def _make_session_data(self, n=3000, seed=42):
        """Create synthetic keypoint data for one session."""
        rng = np.random.default_rng(seed)
        base_x = 400 + np.cumsum(rng.normal(0, 1, n))
        base_y = 300 + np.cumsum(rng.normal(0, 1, n))
        return {
            "left_ear": {
                "x": base_x + 10,
                "y": base_y,
                "likelihood": np.clip(rng.normal(0.9, 0.1, n), 0, 1),
            },
            "right_ear": {
                "x": base_x - 10,
                "y": base_y,
                "likelihood": np.clip(rng.normal(0.9, 0.1, n), 0, 1),
            },
            "tail_base": {
                "x": base_x - 60,
                "y": base_y,
                "likelihood": np.clip(rng.normal(0.85, 0.15, n), 0, 1),
            },
        }

    def test_full_quality_pipeline(self):
        """Report → worst frames → retraining selection."""
        kp_data = self._make_session_data()
        report = session_quality_report(kp_data)

        assert report["n_frames"] == 3000
        assert 0 <= report["overall_score"] <= 100
        assert 0 <= report["pct_good"] <= 1
        assert isinstance(report["issues"], list)
        assert report["problem_frames"].shape == (3000,)

    def test_anatomical_checks(self):
        """Ear distance and body length checks work on page data."""
        kp_data = self._make_session_data()

        ear_result = detect_ear_distance_outliers(
            kp_data["left_ear"]["x"], kp_data["left_ear"]["y"],
            kp_data["right_ear"]["x"], kp_data["right_ear"]["y"],
        )
        assert ear_result["median"] == pytest.approx(20.0, abs=5.0)
        assert "is_outlier" in ear_result

        body_result = body_length_consistency(
            kp_data["left_ear"]["x"], kp_data["left_ear"]["y"],
            kp_data["tail_base"]["x"], kp_data["tail_base"]["y"],
        )
        assert body_result["median"] > 0
        assert "is_outlier" in body_result

    def test_retraining_selection(self):
        """Frame selection for retraining works on page data."""
        kp_data = self._make_session_data()
        lik_matrix = np.column_stack([
            kp_data[bp]["likelihood"] for bp in kp_data
        ])

        result = select_retraining_frames(
            lik_matrix, method="stratified", n_frames=20,
        )
        assert len(result["indices"]) > 0
        assert result["method"] == "stratified"

    def test_degraded_session_flagged(self):
        """Session with injected tracking failures gets low score."""
        kp_data = self._make_session_data()
        # Inject bad tracking: low likelihood + jumps
        kp_data["left_ear"]["likelihood"][500:1000] = 0.1
        kp_data["left_ear"]["x"][700] = -9999  # teleport

        report = session_quality_report(kp_data)
        assert report["overall_score"] < 80
        assert len(report["issues"]) > 0
