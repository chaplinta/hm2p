#!/usr/bin/env python3
"""Build the standalone all-cells results page (``docs/results-rsp-all-cells.html``).

Embeds ``docs/figures/rsp_all/data/all_cells.json`` (from
``scripts/make_rsp_all_cells.py``) and Plotly.js into
``scripts/templates/rsp_all_cells.html``. The stylesheet is shared with the
Penk results page: it is copied from ``scripts/templates/penk_report.html``.

Usage
-----
    python scripts/make_rsp_all_cells.py
    python scripts/build_rsp_all_cells_html.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_penk_report_html import (  # noqa: E402
    BUILD_MARK,
    DATA_MARK,
    PLOTLY_MARK,
    build_label,
    plotly_js,
    render,
)

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "scripts" / "templates" / "rsp_all_cells.html"
STYLE_SOURCE = REPO / "scripts" / "templates" / "penk_report.html"
DATA = REPO / "docs" / "figures" / "rsp_all" / "data" / "all_cells.json"
OUT = REPO / "docs" / "results-rsp-all-cells.html"
CSS_MARK = "<!--__CSS__-->"


def shared_style(source: str) -> str:
    """The first ``<style>...</style>`` block of *source*."""
    m = re.search(r"<style>.*?</style>", source, flags=re.S)
    if m is None:
        raise ValueError("no <style> block in style source")
    return m.group(0)


def assemble(template: str, style_source: str, data: dict, build: str, plotly: str) -> str:
    """Insert the shared stylesheet, then fill data, build label and Plotly."""
    if CSS_MARK not in template:
        raise ValueError("template is missing the CSS placeholder")
    return render(template.replace(CSS_MARK, shared_style(style_source)), data, build, plotly)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - file I/O entry point
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    html = assemble(
        TEMPLATE.read_text(),
        STYLE_SOURCE.read_text(),
        json.loads(DATA.read_text()),
        build_label(),
        plotly_js(),
    )
    args.out.write_text(html)
    print(f"wrote {args.out} ({len(html) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["BUILD_MARK", "DATA_MARK", "PLOTLY_MARK", "assemble", "shared_style"]
