#!/usr/bin/env python3
"""Runner 'mazemem': population coding of memory and planning in the maze.

All soma ROIs of a session are pooled (cell type is kept as a column for
reference and descriptive per-type summaries only). Three analyses are run
per session with ``hm2p.analysis.maze_memory.maze_memory_session``, each
separately in light and in dark (light and dark sample counts equalised) and
in light + dark pooled:

1. retrospective vs prospective ("splitter") coding: decode the trip's origin
   and destination dead end from pass-through visits, stratified by (cell,
   arrival neighbour, departure neighbour);
2. distance to go: cross-validated ridge prediction of remaining graph steps
   to the destination and of elapsed steps since the origin (partial
   Spearman), pooled and place-controlled;
3. planning the next trip: decode whether the next destination is novel-ish
   (or the least recently visited dead end) from the 1 s before / after
   leaving a dead end; plus behaviour: destination novelty vs uniform and
   non-backtracking random walks.

Each decoder is run on neural activity, behaviour only and neural activity
with head direction removed, with a circular-shift null shared across feature
sets.

Outputs (``results/celltype_programme/mazemem/``):

- ``sessions.csv``: session x analysis x target x variant x condition x
  feature set x metric (observed, null mean/sd, z, p, excess, n, reason).
- ``behaviour.csv``: per session x condition x label x random-walk model,
  observed vs expected fraction of trips to novel-ish destinations.
- ``sample_counts.csv``: visit, trip and sample counts per session.
- ``subset_curve.csv`` / ``subset_curve_summary.csv``: neural scores with a
  fixed number of random cells (comparability across sessions).
- ``summary.csv``: excess across sessions and animal medians (Wilcoxon
  signed-rank vs 0), pooled (main result) and per cell type (descriptive);
  includes the behaviour rows.
- ``comparisons.csv``: paired Wilcoxon of excess differences: neural vs
  behaviour / HD-removed, light vs dark, prospective vs retrospective.
- ``run_info.json``: parameters, signal and counts.

Registered through ``run_celltype_programme.load_extra_runners``.

References: see ``hm2p.analysis.maze_memory``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_celltype_hypotheses import SessionIter, _empty_result, _signal_matrix  # noqa: E402
from run_celltype_programme import attach_session_meta, write_outputs  # noqa: E402

log = logging.getLogger("celltype_programme")

RUNNER_KEY = "mazemem"
DEFAULT_SHUFFLES = 200


def _get(args: argparse.Namespace, name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    return default if value is None else value


def params_from_args(args: argparse.Namespace) -> Any:
    """``MazeMemParams`` from the programme command line (defaults where absent)."""
    from hm2p.analysis.maze_memory import MazeMemParams

    d = MazeMemParams()
    return MazeMemParams(
        n_shuffles=int(_get(args, "mazemem_shuffles", DEFAULT_SHUFFLES)),
        recent_k=int(_get(args, "mazemem_recent_k", d.recent_k)),
        min_n=int(_get(args, "mazemem_min_n", d.min_n)),
        n_jobs=int(_get(args, "n_jobs", 1)),
        seed=int(_get(args, "seed", 0)),
    )


def signal_used(arrays: dict[str, Any], signal: str) -> str:
    """Name of the signal ``_signal_matrix`` returns (it falls back to dF/F)."""
    if signal == "dff" or arrays.get(signal) is None:
        return "dff"
    return signal


def valid_frames(arrays: dict[str, Any], n: int) -> np.ndarray:
    """Frames without behavioural artefacts (``~bad_behav``; ``mask`` if absent)."""
    if "bad_behav" in arrays:
        return ~np.asarray(arrays["bad_behav"], dtype=bool)[:n]
    return np.asarray(arrays["mask"], dtype=bool)[:n]


def session_tables(arrays: dict[str, Any], args: argparse.Namespace) -> Any:
    """``MazeMemResult`` for one session (signal name attached); None without maze position."""
    from hm2p.analysis.maze_memory import maze_memory_session

    if arrays.get("x_maze") is None or arrays.get("y_maze") is None:
        return None
    sig = _signal_matrix(arrays, args.signal)
    n = sig.shape[1]

    def _f(key: str) -> np.ndarray:
        return np.asarray(arrays[key], dtype=np.float64)[:n]

    res = maze_memory_session(
        sig,
        _f("x_maze"),
        _f("y_maze"),
        _f("speed_cm_s"),
        _f("hd_deg"),
        _f("ahv_deg_s"),
        np.asarray(arrays["light_on"], dtype=bool)[:n],
        valid_frames(arrays, n),
        float(arrays["fps"]),
        params=params_from_args(args),
    )
    used = signal_used(arrays, args.signal)
    for df in (res.sessions, res.behaviour, res.subsets):
        df["signal"] = used
    res.counts["signal"] = used
    return res


def run_mazemem(args: argparse.Namespace, sessions: SessionIter, out_dir: Path) -> dict[str, Any]:
    """Run all sessions and write the outputs listed in the module docstring."""
    from hm2p.analysis.maze_memory import (
        across_session_summary,
        behaviour_as_session_rows,
        paired_comparisons,
        subset_summary,
    )

    ses_rows, beh_rows, sub_rows, count_rows = [], [], [], []
    for row, arrays in sessions:
        t0 = time.time()
        res = session_tables(arrays, args)
        if res is None:
            log.warning("mazemem: %s has no maze position; skipped", row.get("exp_id"))
            count_rows.append(attach_session_meta(pd.DataFrame([{"reason": "no maze"}]), row))
            continue
        ses_rows.append(attach_session_meta(res.sessions, row))
        beh_rows.append(attach_session_meta(res.behaviour, row))
        if not res.subsets.empty:
            sub_rows.append(attach_session_meta(res.subsets, row))
        counts = pd.DataFrame([{**res.counts, "reason": ""}])
        count_rows.append(attach_session_meta(counts, row))
        log.info("mazemem: %s done in %.0f s", row.get("exp_id"), time.time() - t0)
    if not ses_rows:
        return _empty_result("mazemem", out_dir)
    sessions_df = pd.concat(ses_rows, ignore_index=True)
    behaviour = pd.concat(beh_rows, ignore_index=True)
    subsets = pd.concat(sub_rows, ignore_index=True) if sub_rows else pd.DataFrame()
    write_outputs(out_dir, "sessions", sessions_df)
    write_outputs(out_dir, "behaviour", behaviour)
    write_outputs(out_dir, "sample_counts", pd.concat(count_rows, ignore_index=True))
    write_outputs(out_dir, "subset_curve", subsets)
    write_outputs(out_dir, "subset_curve_summary", subset_summary(subsets))
    combined = pd.concat([sessions_df, behaviour_as_session_rows(behaviour)], ignore_index=True)
    summary = across_session_summary(combined)
    comparisons = paired_comparisons(combined)
    write_outputs(out_dir, "summary", summary)
    write_outputs(out_dir, "comparisons", comparisons)
    info = {
        "hypothesis": "mazemem",
        "signal": str(_get(args, "signal", "dff")),
        "n_sessions": int(sessions_df["exp_id"].nunique()),
        "n_animals": int(sessions_df["animal_id"].nunique()),
        "params": asdict(params_from_args(args)),
    }
    write_outputs(out_dir, "run_info", info)
    return {**info, "summary": summary, "comparisons": comparisons}


RUNNER = run_mazemem
