#!/usr/bin/env python3
"""Runner 'mazerun': running bouts in maze context, per cell.

Running bouts (speed >= 2.5 cm/s for >= 1 s on non-artefact frames, the
runshape convention) are annotated with their maze context
(``hm2p.analysis.maze_running.annotate_bouts``). Per soma ROI, bout-level
activity with duration and mean speed regressed out by rank is compared
between approach and return runs, session-novel and repeated runs, light and
dark runs, and first and repeated arm entries within a light epoch; it is
correlated with prior traversals of the same directed edge; its route
specificity is measured by split-half reliability across start->end routes
(label-permutation null among duration-matched runs); and the lagged
cross-correlation with the run indicator gives lead/lag timing. Shift-tested
measures use a circular shift of the signal relative to behaviour.

Outputs (``results/celltype_programme/mazerun/``): ``cells.csv``,
``runs.csv`` (one row per bout), ``run_summary.csv`` (counts per light
condition x run type per session), ``within_group.csv`` and
``between_group_report.csv``. Registered through
``run_celltype_programme.load_extra_runners``.

Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth show
rapid learning, sudden insight, and efficient exploration." eLife 10:e66175.
doi:10.7554/eLife.66175
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

RUNNER_KEY = "mazerun"
EXTRA_SYNC_KEYS: list[str] = []
TESTED = (
    "approach_return",
    "novel_repeat",
    "edge_decay_rho",
    "route_reliability",
    "light_dark",
    "arm_first_entry",
    "xc_lead_lag_asym",
    "xc_peak_r",
)
DESCRIPTIVE = ("xc_peak_lag_s",)
ALPHA = 0.05


def _get(args: argparse.Namespace, name: str, default: float) -> float:
    value = getattr(args, name, None)
    return default if value is None else value


def session_tables(
    arrays: dict[str, Any], args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Per-cell table, per-bout table and run summary for one session.

    Returns None when maze position is missing.
    """
    from hm2p.analysis.maze_running import (
        annotate_bouts,
        build_design,
        cell_statistics,
        summarise_runs,
    )
    from hm2p.analysis.running_coding import run_bouts

    if arrays.get("x_maze") is None or arrays.get("y_maze") is None:
        return None
    sig = _signal_matrix(arrays, args.signal)
    n = sig.shape[1]
    fps = float(arrays["fps"])
    speed = np.asarray(arrays["speed_cm_s"], dtype=np.float64)[:n]
    if "bad_behav" in arrays:
        valid = ~np.asarray(arrays["bad_behav"], dtype=bool)[:n]
    else:
        valid = np.asarray(arrays["mask"], dtype=bool)[:n]
    rng = np.random.default_rng(int(_get(args, "seed", 0)))
    n_shuffles = int(_get(args, "n_shuffles_evt", 200))
    on, off = run_bouts(speed, valid, fps)
    bouts = annotate_bouts(
        on,
        off,
        np.asarray(arrays["x_maze"], dtype=np.float64)[:n],
        np.asarray(arrays["y_maze"], dtype=np.float64)[:n],
        speed,
        np.asarray(arrays["light_on"], dtype=bool)[:n],
        fps,
    )
    design = build_design(bouts, valid, fps, n, n_route_perms=n_shuffles, rng=rng)
    rows = [
        {
            "roi_idx": int(arrays["roi_idx"][i]),
            **cell_statistics(sig[i], design, n_shuffles=n_shuffles, rng=rng),
        }
        for i in range(sig.shape[0])
    ]
    return pd.DataFrame(rows), bouts, summarise_runs(bouts)


def within_group(cells: pd.DataFrame) -> pd.DataFrame:
    """Per cell type and measure: cell- and animal-level Wilcoxon vs 0, fraction significant.

    The cell-level Wilcoxon treats cells as independent and is descriptive;
    the animal-level Wilcoxon on animal medians is the inferential test.
    """
    from scipy import stats

    def _wilcoxon(v: pd.Series) -> float:
        v = v.dropna()
        if v.size < 3 or not np.any(v != 0):
            return np.nan
        return float(stats.wilcoxon(v).pvalue)

    out = []
    for ct, g in cells.groupby("celltype"):
        for k in TESTED + DESCRIPTIVE:
            if k not in g:
                continue
            animal_med = g.groupby("animal_id")[k].median().dropna()
            vals = g[k].dropna()
            rec: dict[str, Any] = {
                "celltype": ct,
                "measure": k,
                "n_cells_finite": int(vals.size),
                "median_cell_value": float(vals.median()) if vals.size else np.nan,
                "cell_wilcoxon_p": _wilcoxon(vals),
                "n_animals": int(animal_med.size),
                "median_animal_value": float(animal_med.median()) if animal_med.size else np.nan,
                "animal_wilcoxon_p": _wilcoxon(animal_med),
            }
            if k in TESTED:
                p = g[f"{k}_p"].dropna()
                n_sig = int((p < ALPHA).sum())
                rec.update(
                    {
                        "n_cells_tested": int(p.size),
                        "n_sig": n_sig,
                        "frac_sig": n_sig / p.size if p.size else np.nan,
                        "binom_p": (
                            float(stats.binomtest(n_sig, int(p.size), ALPHA, "greater").pvalue)
                            if p.size
                            else np.nan
                        ),
                    }
                )
            out.append(rec)
    return pd.DataFrame(out)


def run_mazerun(args: argparse.Namespace, sessions: SessionIter, out_dir: Path) -> dict[str, Any]:
    cell_rows, run_rows, summary_rows = [], [], []
    for row, arrays in sessions:
        res = session_tables(arrays, args)
        if res is None:
            log.warning("mazerun: %s has no maze position; skipped", row.get("exp_id"))
            continue
        cells, runs, summary = res
        cell_rows.append(attach_session_meta(cells, row))
        run_rows.append(attach_session_meta(runs, row))
        summary_rows.append(attach_session_meta(summary, row))
    if not cell_rows:
        return _empty_result("mazerun", out_dir)
    cells = pd.concat(cell_rows, ignore_index=True)
    write_outputs(out_dir, "cells", cells)
    write_outputs(out_dir, "runs", pd.concat(run_rows, ignore_index=True))
    write_outputs(out_dir, "run_summary", pd.concat(summary_rows, ignore_index=True))
    wg = within_group(cells)
    write_outputs(out_dir, "within_group", wg)
    report = between_group_report(
        cells,
        list(TESTED + DESCRIPTIVE),
        family="mazerun",
        n_perms=int(_get(args, "n_perms", 10000)),
        seed=int(_get(args, "seed", 0)),
    )
    write_outputs(out_dir, "between_group_report", report)
    return {"n_sessions": len(cell_rows), "n_cells": len(cells), "report": report, "within": wg}


RUNNER = run_mazerun
