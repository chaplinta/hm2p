"""Tests for ``scripts/build_rsp_all_cells_html.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import build_rsp_all_cells_html as b  # noqa: E402


def test_shared_style_extracts_first_block() -> None:
    src = "<p>x</p><style>a{}</style><style>b{}</style>"
    assert b.shared_style(src) == "<style>a{}</style>"
    with pytest.raises(ValueError):
        b.shared_style("<p>none</p>")


def test_assemble_fills_all_markers() -> None:
    tpl = (
        f"{b.CSS_MARK}{b.PLOTLY_MARK}<p>{b.BUILD_MARK}</p>"
        f"<script>const D = {b.DATA_MARK};</script>"
    )
    out = b.assemble(tpl, "<style>s{}</style>", {"k": 1}, "B1", "var P;")
    assert out.startswith("<style>s{}</style><script>var P;</script>")
    assert "B1" in out and '{"k":1}' in out
    with pytest.raises(ValueError):
        b.assemble("no css marker", "<style></style>", {}, "", "")


def test_template_has_markers() -> None:
    tpl = b.TEMPLATE.read_text()
    for m in (b.CSS_MARK, b.PLOTLY_MARK, b.BUILD_MARK, b.DATA_MARK):
        assert m in tpl
