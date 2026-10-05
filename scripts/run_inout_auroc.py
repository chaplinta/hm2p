#!/usr/bin/env python3
"""Single-cell AUROC for runs into vs out of dead ends, all cells (``hm2p.analysis.inout_auroc``).

Uses the same duration- and speed-matched runs as the population decoder
(``hm2p.analysis.population_maze.build_samples``), for all frames and for
light-only and dark-only frames. Sessions come from a local cache of pickled
session arrays.

Outputs in ``results/inout_auroc/``: cells.csv (per cell x condition) and
null_<condition>.npz (pooled shift-null AUROCs for plotting).

Usage
-----
    python scripts/run_inout_auroc.py --cache <dir>
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hm2p.analysis.inout_auroc import cell_inout_auroc
from hm2p.analysis.population_maze import PopDecParams, build_samples, draw_shifts
from hm2p.maze.topology import build_rose_maze

REPO = Path(__file__).resolve().parent.parent
CONDITIONS = ("all", "light", "dark")


def session_auroc(
    d: dict[str, Any], n_shuffles: int = 500, signal: str = "spikes", seed: int = 0
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Per-cell AUROC rows for one session (each condition) and the kept null samples."""
    sig = np.nan_to_num(np.asarray(d[signal], dtype=float))
    n = sig.shape[1]

    def g(k: str) -> np.ndarray:
        return np.asarray(d[k], dtype=float)[:n]

    valid = ~np.asarray(d["bad_behav"], dtype=bool)[:n]
    light = np.asarray(d["light_on"], dtype=bool)[:n]
    fps = float(d["fps"])
    params = PopDecParams()
    maze = build_rose_maze()
    rows, nulls = [], {}
    for cond in CONDITIONS:
        mask = valid if cond == "all" else valid & (light if cond == "light" else ~light)
        rng = np.random.default_rng(seed)
        samples, _ = build_samples(
            g("x_maze"), g("y_maze"), g("speed_cm_s"), light, mask, fps, params, maze, rng
        )
        s = samples["inout"]
        if s.reason:
            continue
        shifts = draw_shifts(n, n_shuffles, int(round(params.min_shift_s * fps)), rng)[1:]
        t, keep = cell_inout_auroc(sig, s.starts, s.ends, s.labels, g("hd_deg"), shifts)
        t["condition"] = cond
        t["n_into"] = int((s.labels == 1).sum())
        t["n_out"] = int((s.labels == 0).sum())
        t["exp_id"] = d["meta"]["exp_id"]
        t["animal_id"] = str(d["meta"]["animal_id"])
        rows.append(t)
        nulls[cond] = keep
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), nulls


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - file I/O entry point
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--signal", default="spikes", choices=["spikes", "dff"])
    ap.add_argument("--n-shuffles", type=int, default=500)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "inout_auroc")
    args = ap.parse_args(argv)
    rows, pooled = [], {c: {"raw": [], "hd": []} for c in CONDITIONS}
    for f in sorted(args.cache.glob("*.pkl")):
        with open(f, "rb") as fh:
            t, nulls = session_auroc(pickle.load(fh), args.n_shuffles, args.signal)
        if len(t):
            rows.append(t)
        for c, k in nulls.items():
            for name in ("raw", "hd"):
                pooled[c][name].append(k[name])
        print("done", f.stem, len(t), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(args.out / f"cells_{args.signal}.csv", index=False)
    for c, k in pooled.items():
        np.savez_compressed(
            args.out / f"null_{args.signal}_{c}.npz",
            **{name: np.concatenate(v) if v else np.zeros(0) for name, v in k.items()},
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
