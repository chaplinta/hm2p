#!/usr/bin/env python3
"""Runner 'runshape': how single cells relate to running.

Per soma ROI (``hm2p.analysis.running_coding``): speed-tuning shape (step
index for run vs still, graded rho across running speeds, each against a
circular-shift null), within-bout adaptation (early vs late activity in runs
of at least 3 s), and bout-length scaling (Spearman across runs of duration vs
summed and mean activity). Within-group and between-group reports as in the
other extra runners. Registered through
``run_celltype_programme.load_extra_runners``.
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

RUNNER_KEY = "runshape"
EXTRA_SYNC_KEYS: list[str] = []
TESTED = ("step_index", "graded_rho")
DESCRIPTIVE = ("bout_adaptation_index", "bout_dur_rho_sum", "bout_dur_rho_mean")
ALPHA = 0.05


def _get(args: argparse.Namespace, name: str, default: float) -> float:
    value = getattr(args, name, None)
    return default if value is None else value


def session_cells(arrays: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    """Per-cell running-coding table for one session."""
    from hm2p.analysis.running_coding import (
        bout_duration_scaling,
        bout_time_course,
        run_bouts,
        speed_shape_significance,
    )

    sig = _signal_matrix(arrays, args.signal)
    n = sig.shape[1]
    fps = float(arrays["fps"])
    speed = np.asarray(arrays["speed_cm_s"], dtype=np.float64)[:n]
    mask = np.asarray(arrays["mask"], dtype=bool)[:n]
    # stillness frames are excluded by "mask" (active) in the programme; use
    # the non-artefact mask so the still bin is populated
    valid = ~np.asarray(arrays["bad_behav"], dtype=bool)[:n] if "bad_behav" in arrays else mask
    rng = np.random.default_rng(int(_get(args, "seed", 0)))
    on, off = run_bouts(speed, valid, fps)
    rows = []
    for i in range(sig.shape[0]):
        res = speed_shape_significance(
            sig[i], speed, valid, fps, n_shuffles=int(_get(args, "n_shuffles_evt", 200)), rng=rng
        )
        res.pop("speed_curve")
        res.pop("speed_counts")
        rows.append(
            {
                "roi_idx": int(arrays["roi_idx"][i]),
                **res,
                **bout_time_course(sig[i], on, off, fps),
                **bout_duration_scaling(sig[i], on, off),
            }
        )
    return pd.DataFrame(rows)


def within_group(cells: pd.DataFrame) -> pd.DataFrame:
    """Fraction significant (tested measures) and animal Wilcoxon vs 0 (all measures)."""
    from scipy import stats

    out = []
    for ct, g in cells.groupby("celltype"):
        for k in TESTED + DESCRIPTIVE:
            animal_med = g.groupby("animal_id")[k].median().dropna()
            rec: dict[str, Any] = {
                "celltype": ct,
                "measure": k,
                "n_animals": int(animal_med.size),
                "median_animal_value": float(animal_med.median()) if animal_med.size else np.nan,
                "wilcoxon_p": (
                    float(stats.wilcoxon(animal_med).pvalue)
                    if animal_med.size >= 3 and np.any(animal_med != 0)
                    else np.nan
                ),
            }
            if k in TESTED:
                p = g[f"{k}_p"].dropna()
                n_sig = int((p < ALPHA).sum())
                rec.update(
                    {
                        "n_cells": int(p.size),
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


def run_runshape(args: argparse.Namespace, sessions: SessionIter, out_dir: Path) -> dict[str, Any]:
    rows = [attach_session_meta(session_cells(arrays, args), row) for row, arrays in sessions]
    if not rows:
        return _empty_result("runshape", out_dir)
    cells = pd.concat(rows, ignore_index=True)
    write_outputs(out_dir, "cells", cells)
    wg = within_group(cells)
    write_outputs(out_dir, "within_group", wg)
    report = between_group_report(
        cells,
        list(TESTED + DESCRIPTIVE),
        family="runshape",
        n_perms=int(_get(args, "n_perms", 10000)),
        seed=int(_get(args, "seed", 0)),
    )
    write_outputs(out_dir, "between_group_report", report)
    return {"n_sessions": len(rows), "n_cells": len(cells), "report": report, "within": wg}


RUNNER = run_runshape
