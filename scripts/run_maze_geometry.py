#!/usr/bin/env python3
"""Representational geometry on the maze graph, all cells (``hm2p.analysis.maze_geometry``).

Per session (local cache of pickled session arrays), for maze-cell states and
directed (cell x travel direction) states, in all running frames and in light
and dark (same states in both): cross-validated population dissimilarity
related to graph distance vs Euclidean distance (partial Spearman with
temporal, speed and head-direction controls; circular-shift and Mantel
nulls), and for directed states the role-generalisation contrast.

Outputs (``--out``): ``sessions_<signal>.csv`` (one row per session x mode x
condition), ``across_animals_<signal>.csv`` (animal-level Wilcoxon vs 0),
``light_dark_<signal>.csv`` (paired light - dark), ``role_counts_<signal>.csv``.

Usage
-----
    python scripts/run_maze_geometry.py --cache results/session_cache --signal spikes
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

from hm2p.analysis import maze_geometry as mg
from hm2p.maze.topology import build_rose_maze

REPO = Path(__file__).resolve().parent.parent

STATS = (
    "prho_graph",
    "prho_euclid",
    "graph_minus_euclid",
    "prho_graph_nt",
    "prho_euclid_nt",
    "graph_minus_euclid_nt",
    "prho_graph_local",
    "prho_hd",
    "prho_dir",
    "role_contrast",
    "rolegeom_contrast",
)
Z_STATS = tuple(f"{s[:-9] if s.endswith('_contrast') else s}_z" for s in STATS)


def run_one(path: Path, params: mg.GeometryParams, signal: str) -> list[dict[str, Any]]:
    """Result rows for one cached session."""
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    t0 = time.time()
    rows = mg.analyse_session(d, build_rose_maze(), params, signal)
    print(f"done {path.stem} {time.time() - t0:.0f}s", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - file I/O entry point
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--signal", default="spikes", choices=["spikes", "dff"])
    ap.add_argument("--n-shifts", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "maze_geometry")
    args = ap.parse_args(argv)
    params = mg.GeometryParams(n_shifts=args.n_shifts, n_perm=args.n_perm)
    files = sorted(args.cache.glob("*.pkl"))
    parts = Parallel(n_jobs=args.n_jobs)(delayed(run_one)(f, params, args.signal) for f in files)
    rows = [r for p in parts for r in p]
    counts = [
        {"exp_id": r["exp_id"], "condition": r["condition"], "role": k, "n_states": v}
        for r in rows
        if r["mode"] == "directed"
        for k, v in (r.get("role_counts") or {}).items()
    ]
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "role_counts"} for r in rows])
    args.out.mkdir(parents=True, exist_ok=True)
    s = args.signal
    df.to_csv(args.out / f"sessions_{s}.csv", index=False)
    pd.DataFrame(counts).to_csv(args.out / f"role_counts_{s}.csv", index=False)
    cols = STATS + Z_STATS + ("mean_d",)
    mg.across_animals(df, cols).to_csv(args.out / f"across_animals_{s}.csv", index=False)
    mg.light_dark_paired(df, cols).to_csv(args.out / f"light_dark_{s}.csv", index=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
