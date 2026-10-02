#!/usr/bin/env python3
"""Runner 'locrun': separate corridor location from running in single cells.

For each soma ROI: the corridor-vs-junction location index among running
frames with speed distributions matched, and the running-vs-still index
within corridors and within junctions, each against a circular-shift null of
the cell's own signal (``hm2p.analysis.location_running``). Then per cell
type: the fraction of significant cells (pooled binomial vs 0.05,
descriptive) and an animal-level Wilcoxon of the animal median index against
0; and the four-level between-group report on each index.

Registered through ``run_celltype_programme.load_extra_runners``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_celltype_hypotheses import (  # noqa: E402
    SessionIter,
    _empty_result,
    _signal_matrix,
    between_group_report,
)
from run_celltype_programme import attach_session_meta, write_outputs  # noqa: E402

log = logging.getLogger("celltype_programme")

RUNNER_KEY = "locrun"
EXTRA_SYNC_KEYS: list[str] = []
INDICES = ("loc_index", "run_index_corr", "run_index_junc")
ALPHA = 0.05


def _get(args: argparse.Namespace, name: str, default: float) -> float:
    value = getattr(args, name, None)
    return default if value is None else value


def session_cells(arrays: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame | None:
    """Per-cell contrast table for one session; None if maze position is missing."""
    from hm2p.analysis.event_triggered_behaviour import node_type_channels
    from hm2p.analysis.location_running import (
        contrast_frames,
        contrast_significance,
        location_labels,
    )

    if arrays.get("x_maze") is None or arrays.get("y_maze") is None:
        return None
    sig = _signal_matrix(arrays, args.signal)
    n = sig.shape[1]
    loc = location_labels(node_type_channels(arrays["x_maze"][:n], arrays["y_maze"][:n]))
    rng = np.random.default_rng(int(_get(args, "seed", 0)))
    frames = contrast_frames(
        np.asarray(arrays["speed_cm_s"], dtype=np.float64)[:n],
        loc,
        np.asarray(arrays["mask"], dtype=bool)[:n],
        run_threshold=float(_get(args, "run_threshold_locrun", 2.5)),
        rng=rng,
    )
    rows = []
    for i in range(sig.shape[0]):
        res = contrast_significance(
            sig[i],
            frames,
            float(arrays["fps"]),
            n_shuffles=int(_get(args, "n_shuffles_evt", 200)),
            rng=rng,
        )
        rows.append({"roi_idx": int(arrays["roi_idx"][i]), **res})
    df = pd.DataFrame(rows)
    df["n_frames_corr_run_matched"] = frames["corr_run_matched"].size
    df["n_frames_junc_run_matched"] = frames["junc_run_matched"].size
    return df


def within_group(cells: pd.DataFrame) -> pd.DataFrame:
    """Per cell type and index: fraction significant and animal-level Wilcoxon vs 0."""
    from scipy import stats

    out = []
    for ct, g in cells.groupby("celltype"):
        for k in INDICES:
            p = g[f"{k}_p"].dropna()
            n_sig = int((p < ALPHA).sum())
            animal_med = g.groupby("animal_id")[k].median().dropna()
            wp = (
                float(stats.wilcoxon(animal_med).pvalue)
                if animal_med.size >= 3 and np.any(animal_med != 0)
                else np.nan
            )
            out.append(
                {
                    "celltype": ct,
                    "index": k,
                    "n_cells": int(p.size),
                    "n_sig": n_sig,
                    "frac_sig": n_sig / p.size if p.size else np.nan,
                    "binom_p": (
                        float(stats.binomtest(n_sig, int(p.size), ALPHA, "greater").pvalue)
                        if p.size
                        else np.nan
                    ),
                    "n_animals": int(animal_med.size),
                    "median_animal_index": (
                        float(animal_med.median()) if animal_med.size else np.nan
                    ),
                    "wilcoxon_p": wp,
                }
            )
    return pd.DataFrame(out)


def run_locrun(args: argparse.Namespace, sessions: SessionIter, out_dir: Path) -> dict[str, Any]:
    rows = []
    for row, arrays in sessions:
        df = session_cells(arrays, args)
        if df is None:
            log.warning("locrun: %s has no maze position; skipped", row.get("exp_id"))
            continue
        rows.append(attach_session_meta(df, row))
    if not rows:
        return _empty_result("locrun", out_dir)
    cells = pd.concat(rows, ignore_index=True)
    write_outputs(out_dir, "cells", cells)
    wg = within_group(cells)
    write_outputs(out_dir, "within_group", wg)
    report = between_group_report(
        cells,
        list(INDICES),
        family="locrun",
        n_perms=int(_get(args, "n_perms", 10000)),
        seed=int(_get(args, "seed", 0)),
    )
    write_outputs(out_dir, "between_group_report", report)
    return {"n_sessions": len(rows), "n_cells": len(cells), "report": report, "within": wg}


RUNNER = run_locrun
