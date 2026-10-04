#!/usr/bin/env python3
"""Run the dead-end recency analysis (``hm2p.analysis.dead_end_recency``) on all sessions.

All soma ROIs are pooled (no cell-type split). Sessions come from S3 via
``run_celltype_programme.iter_sessions`` or from a local cache of pickled
session arrays (``--cache``, as written by a previous run with ``--write-cache``).

Outputs in ``results/dead_end_recency/<signal>_w<window>/``: cells.csv
(per cell x condition), entries.csv (per entry: recency and behaviour),
summary.csv (cell, session and animal-level tests, paired dark - light).

Usage
-----
    python scripts/run_dead_end_recency.py --signal spikes
    python scripts/run_dead_end_recency.py --cache <dir> --signal dff --window 1.0
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hm2p.analysis.dead_end_recency import (  # noqa: E402
    dead_end_entries,
    entry_covariates,
    maze_visits,
    recency_cell_table,
    summarise,
)
from hm2p.maze.discretize import discretize_position_fast  # noqa: E402
from hm2p.maze.topology import build_rose_maze  # noqa: E402

log = logging.getLogger("dead_end_recency")
REPO = Path(__file__).resolve().parent.parent
KEEP = (
    "spikes",
    "dff",
    "speed_cm_s",
    "ahv_deg_s",
    "light_on",
    "bad_behav",
    "x_maze",
    "y_maze",
    "frame_times",
)


def session_result(
    arrays: dict[str, Any],
    meta: dict[str, Any],
    signal: str,
    window_s: float,
    n_shift: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Per-cell table and per-entry table for one session, or None when unusable."""
    if signal == "dff_rise" and "dff" in arrays:
        dff = np.asarray(arrays["dff"], dtype=float)
        rise = np.clip(np.diff(dff, axis=1, prepend=dff[:, :1]), 0.0, None)
        arrays = {**arrays, "dff_rise": rise}  # onset-weighted dF/F, no spike inference
    if signal not in arrays or "x_maze" not in arrays or "frame_times" not in arrays:
        return None
    maze = build_rose_maze()
    dead = {maze.cell_to_idx[c] for c in maze.dead_ends}
    ft = np.asarray(arrays["frame_times"], dtype=float)
    fps = 1.0 / float(np.median(np.diff(ft)))
    w = int(round(window_s * fps))
    x = np.asarray(arrays["x_maze"], dtype=float).copy()
    y = np.asarray(arrays["y_maze"], dtype=float).copy()
    x[np.asarray(arrays["bad_behav"], dtype=bool)] = np.nan
    cells_idx = discretize_position_fast(x, y, maze)
    entries = dead_end_entries(maze_visits(cells_idx), dead, arrays["light_on"], fps)
    n_frames = np.asarray(arrays[signal]).shape[1]
    fr = entries["frame"].to_numpy()
    entries = entries[(fr - w >= 0) & (fr + w < n_frames)].reset_index(drop=True)
    if len(entries) < 15:
        return None
    cov = entry_covariates(
        entries, arrays["speed_cm_s"], arrays["ahv_deg_s"], arrays["light_on"], fps, w
    )
    table = recency_cell_table(
        arrays[signal], entries, cov, w, n_shift=n_shift, min_shift=int(30 * fps), seed=seed
    )
    ent = pd.concat([entries, cov], axis=1)
    for t in (table, ent):
        t["exp_id"] = meta["exp_id"]
        t["animal_id"] = str(meta["animal_id"])
        t["celltype"] = meta.get("celltype")
    return table, ent


def iter_cache(cache: Path) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:  # pragma: no cover
    for f in sorted(cache.glob("*.pkl")):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        yield d.get("meta", {"exp_id": f.stem, "animal_id": "?"}), d


def iter_s3(
    signal: str, write_cache: Path | None
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:  # pragma: no cover
    import run_celltype_programme as rcp

    args = argparse.Namespace(
        sessions=None, include_excluded=False, primary_only=False, all_rois=False, signal=signal
    )
    for row, arrays in rcp.iter_sessions(args):
        meta = row.to_dict()
        if write_cache is not None:
            write_cache.mkdir(parents=True, exist_ok=True)
            keep = {k: v for k, v in arrays.items() if k in KEEP}
            keep["meta"] = meta
            with open(write_cache / f"{meta['exp_id']}.pkl", "wb") as fh:
                pickle.dump(keep, fh)
        yield meta, arrays


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - entry point
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signal", default="spikes", choices=["spikes", "dff", "dff_rise"])
    ap.add_argument("--window", type=float, default=1.5, help="response window (s)")
    ap.add_argument("--n-shift", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--write-cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "dead_end_recency")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    src = iter_cache(args.cache) if args.cache else iter_s3(args.signal, args.write_cache)
    cells, ents = [], []
    for meta, arrays in src:
        res = session_result(arrays, meta, args.signal, args.window, args.n_shift, args.seed)
        if res is None:
            log.warning("skipped %s", meta.get("exp_id"))
            continue
        cells.append(res[0])
        ents.append(res[1])
        log.info("done %s (%d entries)", meta.get("exp_id"), len(res[1]))
    out = args.out / f"{args.signal}_w{args.window:g}"
    out.mkdir(parents=True, exist_ok=True)
    c = pd.concat(cells, ignore_index=True)
    c.to_csv(out / "cells.csv", index=False)
    pd.concat(ents, ignore_index=True).to_csv(out / "entries.csv", index=False)
    s = summarise(c)
    s.to_csv(out / "summary.csv", index=False)
    print(s.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
