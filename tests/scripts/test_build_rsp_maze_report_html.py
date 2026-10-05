"""Tests for ``scripts/build_rsp_maze_report_html.py``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import build_rsp_all_cells_html as ac  # noqa: E402
import build_rsp_maze_report_html as b  # noqa: E402


def test_template_has_all_markers() -> None:
    tpl = b.TEMPLATE.read_text()
    for m in (ac.CSS_MARK, ac.PLOTLY_MARK, ac.BUILD_MARK, ac.DATA_MARK):
        assert m in tpl


def test_assemble_on_template() -> None:
    out = ac.assemble(b.TEMPLATE.read_text(), "<style>x{}</style>", {"k": 1}, "B", "var P;")
    assert "<style>x{}</style>" in out and '{"k":1}' in out and "<script>var P;</script>" in out
