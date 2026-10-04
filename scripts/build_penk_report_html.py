#!/usr/bin/env python3
"""Build the interactive Penk+ vs Penk-CamKII+ results page.

Injects the figure data written by ``scripts/make_penk_figures.py``
(``docs/figures/penk/data/*.json``) into ``scripts/templates/penk_report.html``
and writes ``docs/results-penk-vs-nonpenk.html``. Plotly.js is copied from the
installed ``plotly`` Python package into the page, so the file is fully
standalone and works offline.

Usage
-----
    python scripts/make_penk_figures.py        # refresh the animal-level data
    python scripts/make_penk_cell_level.py     # refresh the cell-level data
    python scripts/make_penk_maze.py           # refresh the maze-context data
    python scripts/make_penk_continuum.py      # refresh the continuum analyses
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
ALL_CELLS = REPO / "docs" / "figures" / "rsp_all" / "data" / "all_cells.json"
REQUIRED = (
    "composites",
    "event_kinetics",
    "running_around_events",
    "running_state",
    "light_transitions",
    "patching_adaptation",
    "cell_level",
    "maze_running",
    "continuum",
)
DATA_MARK = "/*__DATA__*/null"
BUILD_MARK = "__BUILD__"
PLOTLY_MARK = "<!--__PLOTLY__-->"


def plotly_js() -> str:
    """Return the plotly.min.js bundled with the installed ``plotly`` package."""
    import plotly

    return (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text()


def load_data(data_dir: Path = DATA_DIR) -> dict:
    """Read every required figure JSON into one dict keyed by figure name."""
    missing = [n for n in REQUIRED if not (data_dir / f"{n}.json").exists()]
    if missing:
        raise FileNotFoundError(f"missing figure data: {missing} in {data_dir}")
    return {n: json.loads((data_dir / f"{n}.json").read_text()) for n in REQUIRED}


def maze_layout() -> dict:
    """Maze cells (col, row, node type) and corridor edges from the topology module."""
    from hm2p.maze.topology import build_rose_maze

    maze = build_rose_maze()
    cells = [[c, r, maze.node_types[(c, r)]] for c, r in sorted(maze.cells)]
    edges = sorted({tuple(sorted((a, b))) for a, nbrs in maze.adj.items() for b in nbrs})
    return {"cells": cells, "edges": [[list(a), list(b)] for a, b in edges]}


def _json_for_script(obj: dict) -> str:
    """Compact JSON safe to embed inside a <script> element (NaN -> null)."""
    text = json.dumps(obj, separators=(",", ":"), allow_nan=True)
    text = text.replace("NaN", "null").replace("-Infinity", "null").replace("Infinity", "null")
    return text.replace("</", "<\\/")


def render(template: str, data: dict, build: str, plotly: str) -> str:
    """Fill the data, build label and inline Plotly markers in *template*."""
    if any(m not in template for m in (DATA_MARK, BUILD_MARK, PLOTLY_MARK)):
        raise ValueError("template is missing a placeholder")
    lib = "<script>" + plotly.replace("</script", "<\\/script") + "</script>"
    out = template.replace(DATA_MARK, _json_for_script(data)).replace(BUILD_MARK, build)
    return out.replace(PLOTLY_MARK, lib)


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
    data = load_data()
    data["maze"] = maze_layout()
    data["all_cells"] = json.loads(ALL_CELLS.read_text()) if ALL_CELLS.exists() else {}
    html = render(TEMPLATE.read_text(), data, build_label(), plotly_js())
    args.out.write_text(html)
    print(f"wrote {args.out} ({len(html) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
