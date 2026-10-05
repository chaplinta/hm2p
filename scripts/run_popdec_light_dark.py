#!/usr/bin/env python3
"""Pooled population maze decoders run on light-only and dark-only frames.

Uses ``hm2p.analysis.population_maze.decode_session``.

The decoders' valid-frame mask is restricted to frames with the room lights on
or off, so every run, visit and junction pass used by a decoder lies within one
condition. Sessions come from a local cache of pickled session arrays (as
written by ``scripts/run_dead_end_recency.py --write-cache``).

Usage
-----
    python scripts/run_popdec_light_dark.py --cache <dir>
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hm2p.analysis.population_maze import PopDecParams, decode_session

REPO = Path(__file__).resolve().parent.parent


def session_rows(d: dict[str, Any], params: PopDecParams, signal: str = "spikes") -> pd.DataFrame:
    """Decoder rows for one session, light and dark separately."""
    sig = np.nan_to_num(np.asarray(d[signal], dtype=float))
    n = sig.shape[1]

    def g(k: str) -> np.ndarray:
        return np.asarray(d[k], dtype=float)[:n]

    valid = ~np.asarray(d["bad_behav"], dtype=bool)[:n]
    light = np.asarray(d["light_on"], dtype=bool)[:n]
    out = []
    for cond, m in (("light", light), ("dark", ~light)):
        r = decode_session(
            sig,
            g("x_maze"),
            g("y_maze"),
            g("speed_cm_s"),
            g("hd_deg"),
            g("ahv_deg_s"),
            light,
            valid & m,
            float(d["fps"]),
            params=params,
        )
        t = r.decoders.copy()
        t["condition"] = cond
        t["exp_id"] = d["meta"]["exp_id"]
        t["animal_id"] = str(d["meta"]["animal_id"])
        out.append(t)
    return pd.concat(out, ignore_index=True)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - file I/O entry point
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--signal", default="spikes", choices=["spikes", "dff"])
    ap.add_argument("--n-shuffles", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "popdec_light_dark")
    args = ap.parse_args(argv)
    params = PopDecParams(n_shuffles=args.n_shuffles, subset_sizes=(), n_jobs=args.n_jobs)
    rows = []
    for f in sorted(args.cache.glob("*.pkl")):
        with open(f, "rb") as fh:
            rows.append(session_rows(pickle.load(fh), params, args.signal))
        print("done", f.stem, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(
        args.out / f"sessions_{args.signal}.csv", index=False
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
