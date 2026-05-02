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
            kp_data["left_ear"]["x"],
            kp_data["left_ear"]["y"],
            kp_data["right_ear"]["x"],
            kp_data["right_ear"]["y"],
        )
        assert ear_result["median"] == pytest.approx(20.0, abs=5.0)
        assert "is_outlier" in ear_result

        body_result = body_length_consistency(
            kp_data["left_ear"]["x"],
            kp_data["left_ear"]["y"],
            kp_data["tail_base"]["x"],
            kp_data["tail_base"]["y"],
        )
        assert body_result["median"] > 0
        assert "is_outlier" in body_result

    def test_retraining_selection(self):
        """Frame selection for retraining works on page data."""
        kp_data = self._make_session_data()
        lik_matrix = np.column_stack([kp_data[bp]["likelihood"] for bp in kp_data])

        result = select_retraining_frames(
            lik_matrix,
            method="stratified",
            n_frames=20,
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


class TestMethodsExpanderCitation:
    """The Methods & References expander must cite Ye et al. 2024.

    Per CLAUDE.md citation policy, any analysis method taken from a paper
    must be cited in three places: code docstring, docs, and frontend.
    The frontend slot for the SuperAnimal fine-tune work is the Methods
    & References expander on the tracking-quality page.
    """

    def _read_page_source(self) -> str:
        from pathlib import Path

        page = (
            Path(__file__).resolve().parent.parent.parent
            / "frontend"
            / "pages"
            / "tracking_quality_page.py"
        )
        return page.read_text()

    def test_page_imports_clean(self):
        """Importing the page module raises nothing.

        Streamlit pages are top-level scripts, so we read the source and
        compile it. AppTest-based smoke tests live in
        tests/frontend/test_app_rendering.py.
        """
        src = self._read_page_source()
        compile(src, "tracking_quality_page.py", "exec")

    def test_methods_expander_present(self):
        src = self._read_page_source()
        assert 'st.expander("Methods & References")' in src

    def test_methods_expander_cites_ye_2024(self):
        src = self._read_page_source()
        assert "Ye S" in src or "Ye, S" in src
        assert "10.1038/s41467-024-48792-2" in src

    def test_methods_expander_cites_kerby_2014(self):
        """Effect-size method has its own citation (Kerby 2014)."""
        src = self._read_page_source()
        assert "Kerby" in src
        assert "10.2466/11.IT.3.1" in src

    def test_methods_expander_describes_gate(self):
        src = self._read_page_source()
        # Sanity check that the expander documents the gate, not
        # just the citation.
        assert "Wilcoxon" in src
        assert "rank-biserial" in src
        assert "Bonferroni" in src


class TestVerdictDisplay:
    """Verdict-display section consumes the schema-1.0 contract."""

    def _read_page_source(self) -> str:
        from pathlib import Path

        return (
            Path(__file__).resolve().parent.parent.parent
            / "frontend" / "pages" / "tracking_quality_page.py"
        ).read_text()

    def test_uses_load_verdict_helper(self):
        src = self._read_page_source()
        assert "load_verdict" in src

    def test_display_logic_handles_missing_verdict(self):
        """The page must show a friendly banner when verdict is None."""
        src = self._read_page_source()
        assert "Verdict not yet computed" in src

    def test_display_logic_handles_pass_and_fail(self):
        src = self._read_page_source()
        assert "Promotion gate: PASS" in src
        assert "Promotion gate: FAIL" in src

    def test_display_logic_rejects_unsupported_schema(self):
        src = self._read_page_source()
        assert "Unsupported verdict schema version" in src


class TestLoadVerdict:
    """Unit tests on the cached loader in frontend/data.py."""

    def test_returns_none_when_object_absent(self, monkeypatch):
        from frontend import data as fdata

        monkeypatch.setattr(fdata, "download_s3_bytes", lambda b, k: None)
        assert fdata.load_verdict() is None

    def test_returns_dict_for_valid_json(self, monkeypatch):
        import json

        from frontend import data as fdata

        payload = {"schema_version": "1.0", "baseline_id": "b", "candidate_id": "c"}
        monkeypatch.setattr(
            fdata, "download_s3_bytes",
            lambda b, k: json.dumps(payload).encode(),
        )
        loaded = fdata.load_verdict()
        assert loaded == payload

    def test_returns_none_for_corrupt_json(self, monkeypatch):
        from frontend import data as fdata

        monkeypatch.setattr(fdata, "download_s3_bytes", lambda b, k: b"not json")
        assert fdata.load_verdict() is None

    def test_verdict_s3_key_is_canonical(self):
        from frontend import data as fdata

        assert fdata.VERDICT_S3_KEY == "dlc-retrain/models/_compare_verdict.json"
