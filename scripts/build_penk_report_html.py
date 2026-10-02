#!/usr/bin/env python3
"""Build the interactive Penk+ vs Penk-CamKII+ results page.

Injects the figure data written by ``scripts/make_penk_figures.py``
(``docs/figures/penk/data/*.json``) into ``scripts/templates/penk_report.html``
and writes ``docs/results-penk-vs-nonpenk.html``. The page loads Plotly from
cdnjs and renders all charts client-side from the embedded data.

Usage
-----
    python scripts/make_penk_figures.py        # refresh the data first
    python scripts/build_penk_report_html.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "scripts" / "templates" / "penk_report.html"
DATA_DIR = REPO / "docs" / "figures" / "penk" / "data"
OUT = REPO / "docs" / "results-penk-vs-nonpenk.html"
REQUIRED = (
    "composites",
    "event_kinetics",
    "running_around_events",
    "running_state",
    "light_transitions",
    "patching_adaptation",
)
DATA_MARK = "/*__DATA__*/null"
BUILD_MARK = "__BUILD__"


def load_data(data_dir: Path = DATA_DIR) -> dict:
    """Read every required figure JSON into one dict keyed by figure name."""
    missing = [n for n in REQUIRED if not (data_dir / f"{n}.json").exists()]
    if missing:
        raise FileNotFoundError(f"missing figure data: {missing} in {data_dir}")
    return {n: json.loads((data_dir / f"{n}.json").read_text()) for n in REQUIRED}


def _json_for_script(obj: dict) -> str:
    """Compact JSON safe to embed inside a <script> element (NaN -> null)."""
    text = json.dumps(obj, separators=(",", ":"), allow_nan=True)
    text = text.replace("NaN", "null").replace("-Infinity", "null").replace("Infinity", "null")
    return text.replace("</", "<\\/")


def render(template: str, data: dict, build: str) -> str:
    """Fill the data and build markers in *template*."""
    if DATA_MARK not in template or BUILD_MARK not in template:
        raise ValueError("template is missing a placeholder")
    return template.replace(DATA_MARK, _json_for_script(data)).replace(BUILD_MARK, build)


def build_label() -> str:
    """Date plus short git commit, or the date alone outside a repository."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = ""
    return f"{date.today().isoformat()}" + (f" · {sha}" if sha else "")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - file I/O entry point
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    html = render(TEMPLATE.read_text(), load_data(), build_label())
    args.out.write_text(html)
    print(f"wrote {args.out} ({len(html) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
