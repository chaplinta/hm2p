"""Tests for ``scripts/build_penk_report_html.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import build_penk_report_html as b  # noqa: E402


def test_load_data(tmp_path: Path) -> None:
    for n in b.REQUIRED:
        (tmp_path / f"{n}.json").write_text(json.dumps({"name": n}))
    data = b.load_data(tmp_path)
    assert set(data) == set(b.REQUIRED) and data["composites"]["name"] == "composites"
    (tmp_path / "composites.json").unlink()
    with pytest.raises(FileNotFoundError):
        b.load_data(tmp_path)


def test_json_for_script_handles_nan_and_script_close() -> None:
    text = b._json_for_script({"a": float("nan"), "b": float("inf"), "c": "</script>"})
    assert "NaN" not in text and "Infinity" not in text
    assert "</script>" not in text
    assert json.loads(text.replace("<\\/", "</"))["a"] is None


def test_render_fills_markers() -> None:
    tpl = f"{b.PLOTLY_MARK}<p>{b.BUILD_MARK}</p><script>const D = {b.DATA_MARK};</script>"
    out = b.render(tpl, {"x": 1}, "2026-10-02", "var P=1;'</script>'")
    assert "2026-10-02" in out and '{"x":1}' in out
    assert out.startswith("<script>var P=1;") and out.count("</script>") == 2
    with pytest.raises(ValueError):
        b.render("<p>no markers</p>", {}, "x", "")


def test_plotly_js_is_bundled() -> None:
    assert "plotly.js" in b.plotly_js()[:500]


def test_build_label_has_date() -> None:
    assert b.build_label()[:4].isdigit()


def test_template_has_markers() -> None:
    tpl = b.TEMPLATE.read_text()
    assert b.DATA_MARK in tpl and b.BUILD_MARK in tpl and b.PLOTLY_MARK in tpl
    assert "cdnjs" not in tpl


def test_maze_layout_is_a_tree() -> None:
    m = b.maze_layout()
    assert len(m["cells"]) == 23 and len(m["edges"]) == 22
    assert {c[2] for c in m["cells"]} == {"dead_end", "t_junction", "corridor"}
