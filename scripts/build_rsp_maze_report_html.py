#!/usr/bin/env python3
"""Build the standalone maze-exploration findings report page.

Output: ``docs/results-rsp-maze-exploration.html``.

Embeds ``docs/figures/rsp_maze/data/report.json`` (from
``scripts/make_rsp_maze_report.py``) and Plotly.js into
``scripts/templates/rsp_maze_report.html``, with the stylesheet shared with the
other results pages.

Usage
-----
    python scripts/make_rsp_maze_report.py
    python scripts/build_rsp_maze_report_html.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_penk_report_html import build_label, plotly_js  # noqa: E402
from build_rsp_all_cells_html import STYLE_SOURCE, assemble  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "scripts" / "templates" / "rsp_maze_report.html"
DATA = REPO / "docs" / "figures" / "rsp_maze" / "data" / "report.json"
OUT = REPO / "docs" / "results-rsp-maze-exploration.html"


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
