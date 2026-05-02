"""Smoke tests for frontend pages — verifies all pages are importable."""

from __future__ import annotations

from pathlib import Path

import pytest

_PAGES_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "pages"

# All page modules that should be importable
_UNTESTED_PAGES = [
    "analysis_page",
    "anatomy_page",
    "animals_page",
    "aws_page",
    "batch_page",
    "behaviour_page",
    "calcium_page",
    "changelog_page",
    "compare_page",
    "correlations_page",
    "cost_page",
    "dlc_page",
    "dlc_viewer_page",
    "event_dynamics_page",
    "events_page",
    "explorer_page",
    "gain_page",
    "gallery_page",
    "home_page",
    "hypotheses_page",
    "info_theory_page",
    "light_compare_page",
    "light_page",
    "moseq_explore_page",
    "moseq_page",
    "patching_morph_page",
    "patching_page",
    "pipeline_page",
    "place_tuning_page",
    "population_page",
    "qc_report_page",
    "roi_curation_page",
    "roi_viewer_page",
    "sessions_page",
    "speed_page",
    "stability_page",
    "stats_page",
    "suite2p_page",
    "sync_page",
    "timeline_page",
    "trace_compare_page",
    "zdrift_page",
]


class TestPageFilesExist:
    """Verify all expected page files exist on disk."""

    @pytest.mark.parametrize("page", _UNTESTED_PAGES)
    def test_page_file_exists(self, page):
        path = _PAGES_DIR / f"{page}.py"
        assert path.exists(), f"Page file not found: {path}"


class TestPagesImportable:
    """Verify page modules can be imported without crashing.

    These are smoke tests — they don't test page rendering (which requires
    Streamlit runtime), just that the Python source is valid.
    Note: Some pages call st.title() etc. at module level, so we cannot
    actually import them outside the Streamlit context. Instead we compile
    the source to verify it is syntactically valid Python.
    """

    @pytest.mark.parametrize("page", _UNTESTED_PAGES)
    def test_page_source_is_valid_python(self, page):
        """Verify the page source parses as valid Python (compile check)."""
        path = _PAGES_DIR / f"{page}.py"
        source = path.read_text()
        compile(source, str(path), "exec")
