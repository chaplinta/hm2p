"""Tests for scripts/build_qc_reports_html.py (synthetic report data only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from build_qc_reports_html import (  # noqa: E402
    QCCSS_MARK,
    QCJS_MARK,
    REPORTS,
    build_page,
    json_safe,
    page_template,
)

from tests.qc.report_fixtures import DOCS  # noqa: E402


@pytest.mark.parametrize("name", REPORTS)
def test_template_inlines_shared_parts(name):
    tpl = page_template(name)
    assert QCCSS_MARK not in tpl and QCJS_MARK not in tpl
    assert "const QC = (() =>" in tpl
    assert "/*__DATA__*/null" in tpl


@pytest.mark.parametrize("name", REPORTS)
def test_build_page_fills_every_placeholder(name):
    html = build_page(name, DOCS[name](), "test-build", "/* plotly */")
    for mark in (
        "/*__DATA__*/null",
        "__BUILD__",
        "<!--__PLOTLY__-->",
        "<!--__CSS__-->",
        QCCSS_MARK,
        QCJS_MARK,
    ):
        assert mark not in html
    assert html.startswith("<!doctype html>")
    assert f'QC.nav("qc-{name}.html")' in html  # navigation is built at load time
    assert (
        "https://" not in html.split("<script>")[0].split("<style>")[0]
    )  # no external resources in <head>


def test_json_safe_replaces_non_finite_but_keeps_strings():
    out = json_safe({"a": float("nan"), "b": [1.0, float("inf")], "c": "NaNInfinity", "d": (2.0,)})
    assert out == {"a": None, "b": [1.0, None], "c": "NaNInfinity", "d": [2.0]}


def test_strings_containing_nan_letters_survive_build():
    # base64 traces can contain "NaN"; the shared results-page serialiser would corrupt them
    html = build_page("tracking", {"sessions": [], "probe": "xxNaNInfinityxx"}, "b", "")
    assert "xxNaNInfinityxx" in html


def test_missing_placeholder_raises(tmp_path):
    (tmp_path / "qc_x.html").write_text("<html></html>")
    (tmp_path / "qc_common.css").write_text("")
    (tmp_path / "qc_common.js").write_text("")
    with pytest.raises(ValueError):
        page_template("x", tmp_path)
