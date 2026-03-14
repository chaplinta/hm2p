"""Streamlit AppTest — automated UI rendering tests for frontend pages.

Uses Streamlit's built-in AppTest (https://docs.streamlit.io/develop/api-reference/app-testing)
to verify that pages render without errors. External services (S3, EC2) are
mocked to avoid hitting real AWS resources.

All tests use synthetic data only — no real experimental data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure repo root is on path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from streamlit.testing.v1 import AppTest

# ---------------------------------------------------------------------------
# Shared mock data
# ---------------------------------------------------------------------------

MOCK_EXPERIMENTS = [
    {"exp_id": "20220804_13_52_02_1117646", "lens": "16x", "orientation": "15",
     "fibre": "", "primary_exp": "1", "bad_2p_frames": "", "bad_behav_times": "",
     "exclude": "0", "zstack_id": "", "Notes": ""},
    {"exp_id": "20221015_10_00_00_1116663", "lens": "16x", "orientation": "0",
     "fibre": "", "primary_exp": "1", "bad_2p_frames": "", "bad_behav_times": "",
     "exclude": "0", "zstack_id": "", "Notes": ""},
]

MOCK_ANIMALS = [
    {"animal_id": "1117646", "celltype": "penk", "strain": "Penk-Cre",
     "gcamp": "GCaMP7f", "virus_id": "ADD3", "hemisphere": "right", "sex": "M"},
    {"animal_id": "1116663", "celltype": "nonpenk", "strain": "Penk-Cre",
     "gcamp": "GCaMP8f", "virus_id": "344", "hemisphere": "left", "sex": "F"},
]

MOCK_PIPELINE_STATUS = {
    "20220804_13_52_02_1117646": {
        "ca_extraction": True, "pose": True, "kinematics": True,
        "calcium": True, "sync": True, "analysis": True,
    },
    "20221015_10_00_00_1116663": {
        "ca_extraction": True, "pose": True, "kinematics": False,
        "calcium": True, "sync": False, "analysis": False,
    },
}

MOCK_STAGE_SUMMARY = {
    "ingest": {"label": "Stage 0 — Ingest", "short": "Ingest", "expected": 26, "done": 26, "status": "Complete", "color": "green"},
    "ca_extraction": {"label": "Stage 1 — Suite2p", "short": "Suite2p", "expected": 26, "done": 26, "status": "Complete", "color": "green"},
    "pose": {"label": "Stage 2 — DLC", "short": "DLC", "expected": 26, "done": 26, "status": "Complete", "color": "green"},
    "kinematics": {"label": "Stage 3 — Kinematics", "short": "Kinematics", "expected": 21, "done": 21, "status": "Complete", "color": "green"},
    "calcium": {"label": "Stage 4 — Calcium", "short": "Calcium", "expected": 26, "done": 26, "status": "Complete", "color": "green"},
    "sync": {"label": "Stage 5 — Sync", "short": "Sync", "expected": 21, "done": 21, "status": "Complete", "color": "green"},
    "analysis": {"label": "Stage 6 — Analysis", "short": "Analysis", "expected": 21, "done": 21, "status": "Complete", "color": "green"},
    "kpms": {"label": "Stage 3b — MoSeq", "short": "MoSeq", "expected": 26, "done": 0, "status": "Not started", "color": "red"},
}


def _mock_s3_client():
    """Return a mock S3 client that returns empty results."""
    mock = MagicMock()
    mock.list_objects_v2.return_value = {"Contents": [], "KeyCount": 0}
    mock.get_object.side_effect = Exception("NoSuchKey")
    mock.exceptions = MagicMock()
    mock.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
    return mock


# Common patches applied to all page tests
_COMMON_PATCHES = {
    "frontend.data.load_experiments": lambda: MOCK_EXPERIMENTS,
    "frontend.data.load_animals": lambda: MOCK_ANIMALS,
    "frontend.data.get_pipeline_status": lambda: MOCK_PIPELINE_STATUS,
    "frontend.data.get_stage_summary": lambda: MOCK_STAGE_SUMMARY,
    "frontend.data.get_s3_client": _mock_s3_client,
    "frontend.data.get_ec2_instances": lambda: [],
    "frontend.data.get_progress": lambda prefix: None,
    "frontend.data.download_s3_bytes": lambda bucket, key: None,
    "frontend.data.download_s3_numpy": lambda bucket, key: None,
}


def _pages_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "frontend" / "pages"


def _run_page(page_filename: str, timeout: int = 10) -> AppTest:
    """Run a single page script with mocked data and return the AppTest."""
    page_path = _pages_dir() / page_filename
    assert page_path.exists(), f"Page not found: {page_path}"

    at = AppTest.from_file(str(page_path), default_timeout=timeout)

    # Apply all common patches
    for target, replacement in _COMMON_PATCHES.items():
        at = at.run()  # Can't patch before first run, so we'll use context manager approach
        break

    return at


# ---------------------------------------------------------------------------
# Tests — each verifies a page renders without uncaught exceptions
# ---------------------------------------------------------------------------


def _has_real_exception(at: AppTest) -> str | None:
    """Return first non-page_link exception message, or None.

    st.page_link raises KeyError('url_pathname') in AppTest because pages
    aren't registered through st.navigation. This is expected — skip it.
    """
    for exc in at.exception:
        if "url_pathname" in str(exc):
            continue
        return str(exc)
    return None


class TestHomePageRendering:
    """Test that the home page renders correctly."""

    def test_home_renders(self):
        page_path = str(_pages_dir() / "home_page.py")
        at = AppTest.from_file(page_path, default_timeout=10)

        with patch("frontend.data.load_experiments", return_value=MOCK_EXPERIMENTS), \
             patch("frontend.data.load_animals", return_value=MOCK_ANIMALS), \
             patch("frontend.data.get_stage_summary", return_value=MOCK_STAGE_SUMMARY):
            at.run()

        err = _has_real_exception(at)
        assert err is None, f"Home page raised: {err}"

    def test_home_shows_title(self):
        page_path = str(_pages_dir() / "home_page.py")
        at = AppTest.from_file(page_path, default_timeout=10)

        with patch("frontend.data.load_experiments", return_value=MOCK_EXPERIMENTS), \
             patch("frontend.data.load_animals", return_value=MOCK_ANIMALS), \
             patch("frontend.data.get_stage_summary", return_value=MOCK_STAGE_SUMMARY):
            at.run()

        titles = [t.value for t in at.title]
        assert any("hm2p" in t for t in titles), f"Expected 'hm2p' in titles: {titles}"

    def test_home_shows_metrics(self):
        page_path = str(_pages_dir() / "home_page.py")
        at = AppTest.from_file(page_path, default_timeout=10)

        with patch("frontend.data.load_experiments", return_value=MOCK_EXPERIMENTS), \
             patch("frontend.data.load_animals", return_value=MOCK_ANIMALS), \
             patch("frontend.data.get_stage_summary", return_value=MOCK_STAGE_SUMMARY):
            at.run()

        # Should have metrics for sessions, animals, etc.
        assert len(at.metric) > 0, "Expected metrics on home page"


class TestPipelinePageRendering:
    """Test that the pipeline page renders correctly."""

    def test_pipeline_renders(self):
        page_path = str(_pages_dir() / "pipeline_page.py")
        at = AppTest.from_file(page_path, default_timeout=10)

        with patch("frontend.data.get_stage_summary", return_value=MOCK_STAGE_SUMMARY), \
             patch("frontend.data.get_ec2_instances", return_value=[]), \
             patch("frontend.data.get_progress", return_value=None), \
             patch("frontend.data.get_pipeline_status", return_value=MOCK_PIPELINE_STATUS), \
             patch("frontend.data.load_experiments", return_value=MOCK_EXPERIMENTS):
            at.run()

        assert not at.exception, f"Pipeline page raised: {at.exception}"


class TestSessionsPageRendering:
    """Test that the sessions page renders correctly."""

    def test_sessions_renders(self):
        page_path = str(_pages_dir() / "sessions_page.py")
        at = AppTest.from_file(page_path, default_timeout=10)

        with patch("frontend.data.load_experiments", return_value=MOCK_EXPERIMENTS), \
             patch("frontend.data.load_animals", return_value=MOCK_ANIMALS), \
             patch("frontend.data.get_pipeline_status", return_value=MOCK_PIPELINE_STATUS):
            at.run()

        assert not at.exception, f"Sessions page raised: {at.exception}"


class TestAnimalsPageRendering:
    """Test that the animals page renders correctly."""

    def test_animals_renders(self):
        page_path = str(_pages_dir() / "animals_page.py")
        at = AppTest.from_file(page_path, default_timeout=10)

        with patch("frontend.data.load_experiments", return_value=MOCK_EXPERIMENTS), \
             patch("frontend.data.load_animals", return_value=MOCK_ANIMALS), \
             patch("frontend.data.get_pipeline_status", return_value=MOCK_PIPELINE_STATUS):
            at.run()

        assert not at.exception, f"Animals page raised: {at.exception}"


class TestChangelogPageRendering:
    """Test that the changelog page renders correctly."""

    def test_changelog_renders(self):
        page_path = str(_pages_dir() / "changelog_page.py")
        at = AppTest.from_file(page_path, default_timeout=10)
        at.run()
        assert not at.exception, f"Changelog page raised: {at.exception}"
