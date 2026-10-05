#!/usr/bin/env python3
"""Build the five standalone data-quality report pages.

Reads ``results/qc/data/{tracking,movement,syllables,rois,spikes}.json``
(written by ``scripts/make_qc_reports.py``) and writes
``docs/qc/qc-{name}.html``. Each page embeds its data, Plotly.js (copied from
the installed ``plotly`` package), the stylesheet shared with the results
pages and the QC helpers in ``scripts/templates/qc_common.{css,js}``, so it
opens offline from ``file://``. The pages link to each other by relative path
and should stay in the same folder.

Usage
-----
    python scripts/make_qc_reports.py
    python scripts/build_qc_reports_html.py               # all pages with data
    python scripts/build_qc_reports_html.py --reports tracking
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_penk_report_html import (  # noqa: E402
    BUILD_MARK,
    DATA_MARK,
    PLOTLY_MARK,
    build_label,
    plotly_js,
)
from build_rsp_all_cells_html import CSS_MARK, STYLE_SOURCE, shared_style  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "scripts" / "templates"
DATA_DIR = REPO / "results" / "qc" / "data"
OUT_DIR = REPO / "docs" / "qc"
REPORTS = ("tracking", "movement", "syllables", "rois", "spikes")
QCCSS_MARK = "<!--__QCCSS__-->"
QCJS_MARK = "<!--__QCJS__-->"


def page_template(name: str, templates: Path = TEMPLATES) -> str:
    """Report template with the shared QC stylesheet and helper script inlined."""
    tpl = (templates / f"qc_{name}.html").read_text()
    if QCCSS_MARK not in tpl or QCJS_MARK not in tpl:
        raise ValueError(f"qc_{name}.html is missing a QC placeholder")
    css = (templates / "qc_common.css").read_text()
    js = (templates / "qc_common.js").read_text()
    return tpl.replace(QCCSS_MARK, css).replace(QCJS_MARK, js)


def json_safe(o: Any) -> Any:
    """Replace non-finite floats with None, recursively.

    Done on the Python side rather than by text substitution on the JSON
    string (as ``build_penk_report_html._json_for_script`` does), because the
    reports embed base64 traces that can contain the letters "NaN".
    """
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [json_safe(v) for v in o]
    return o


def build_page(name: str, data: dict, build: str, plotly: str, templates: Path = TEMPLATES) -> str:
    """Complete standalone HTML for one report."""
    text = json.dumps(json_safe(data), separators=(",", ":"), allow_nan=False).replace(
        "</", "<\\/"
    )
    tpl = page_template(name, templates).replace(CSS_MARK, shared_style(STYLE_SOURCE.read_text()))
    if DATA_MARK not in tpl or BUILD_MARK not in tpl or PLOTLY_MARK not in tpl:
        raise ValueError(f"qc_{name}.html is missing a placeholder")
    lib = "<script>" + plotly.replace("</script", "<\\/script") + "</script>"
    return tpl.replace(DATA_MARK, text).replace(BUILD_MARK, build).replace(PLOTLY_MARK, lib)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - file I/O entry point
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reports", nargs="+", choices=REPORTS, default=list(REPORTS))
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plotly = plotly_js()
    build = build_label()
    for name in args.reports:
        src = args.data_dir / f"{name}.json"
        if not src.exists():
            print(f"skip {name}: {src} not found (run scripts/make_qc_reports.py)")
            continue
        html = build_page(name, json.loads(src.read_text()), build, plotly)
        out = args.out_dir / f"qc-{name}.html"
        out.write_text(html)
        print(f"wrote {out} ({len(html) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
