#!/usr/bin/env python3
"""Runner 'movestate': translation vs rotation coding, and syllable information.

Per session, speed and AHV (``speed_cm_s``, ``ahv_deg_s`` in sync.h5) are
smoothed (0.5 s centred moving average; signed AHV smoothed before
rectification) on non-artefact frames (``~bad_behav``), and frames are
assigned to still / straight running / turning in place / running+turning
(``hm2p.analysis.movement_state``; defaults 1 and 3 cm/s, 30 and 90 deg/s,
state bouts >= 0.3 s). Per soma ROI: mean activity per state, run, turn,
run-vs-turn and interaction indices, graded |AHV| tuning within running at
matched speed, and, when ``syllable_id`` is present, syllable activity
profile vs syllable speed/|AHV| and mutual information with syllable
identity, with three syllable speed classes and within the fast class. All
statistics are tested against a circular shift of the signal relative to
behaviour.

Outputs (``results/celltype_programme/movestate/``): ``cells.csv``,
``state_summary.csv`` (one row per session), ``syllable_summary.csv`` (one
row per session x syllable), ``within_group.csv`` and
``between_group_report.csv``. Registered through
``run_celltype_programme.load_extra_runners``.

Weinreb C, Pearl JE, Lin S, et al. 2024. "Keypoint-MoSeq: parsing behavior
by linking point tracking to pose dynamics." Nature Methods 21:1329-1339.
doi:10.1038/s41592-024-02318-2. https://github.com/dattalab/keypoint-moseq
Panzeri S, Senatore R, Montemurro MA, Petersen RS. 2007. "Correcting for
the sampling bias problem in spike train information measures." Journal of
Neurophysiology 98:1064-1072. doi:10.1152/jn.00559.2007
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

RUNNER_KEY = "movestate"
EXTRA_SYNC_KEYS: list[str] = []
TESTED = (
    "run_index",
    "turn_index",
    "run_vs_turn_bias",
    "interaction_index",
    "ahv_rho_matched",
    "syl_speed_rho",
    "syl_ahv_rho",
    "syl_mi",
    "class_mi",
    "within_fast_mi",
)
# measures whose value column differs from the test key (MI is reported debiased)
VALUE_COLUMN = {k: f"{k}_debiased" for k in ("syl_mi", "class_mi", "within_fast_mi")}
DESCRIPTIVE = (
    "run_d",
    "turn_d",
    "bias_d",
    "interaction_d",
    "interaction_contrast_d",
    "class_frac_of_syl_mi",
    "cond_syl_mi_given_class",
)
ALPHA = 0.05


def _get(args: argparse.Namespace, name: str, default: float) -> float:
    value = getattr(args, name, None)
    return default if value is None else value


def design_kwargs(args: argparse.Namespace) -> dict[str, float]:
    """State thresholds from the command line (``--ms-*``), else module defaults."""
    from hm2p.analysis import movement_state as ms

    return {
        "speed_lo": float(_get(args, "ms_speed_lo", ms.DEFAULT_SPEED_LO)),
        "speed_hi": float(_get(args, "ms_speed_hi", ms.DEFAULT_SPEED_HI)),
        "ahv_lo": float(_get(args, "ms_ahv_lo", ms.DEFAULT_AHV_LO)),
        "ahv_hi": float(_get(args, "ms_ahv_hi", ms.DEFAULT_AHV_HI)),
        "smooth_s": float(_get(args, "ms_smooth_s", ms.DEFAULT_SMOOTH_S)),
        "min_bout_s": float(_get(args, "ms_min_bout_s", ms.DEFAULT_MIN_BOUT_S)),
    }


def session_tables(
    arrays: dict[str, Any], args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-cell table, one-row state summary and syllable summary for one session."""
    from hm2p.analysis.movement_state import build_design, cell_statistics

    sig = _signal_matrix(arrays, args.signal)
    n = sig.shape[1]
    fps = float(arrays["fps"])
    speed = np.asarray(arrays["speed_cm_s"], dtype=np.float64)[:n]
    ahv = np.asarray(arrays["ahv_deg_s"], dtype=np.float64)[:n]
    # the programme "mask" requires active frames; still frames are needed here
    if "bad_behav" in arrays:
        valid = ~np.asarray(arrays["bad_behav"], dtype=bool)[:n]
    else:
        valid = np.ones(n, dtype=bool)
    syl = arrays.get("syllable_id")
    syl = None if syl is None else np.asarray(syl)[:n]
    light = arrays.get("light_on")
    light = None if light is None else np.asarray(light, dtype=bool)[:n]
    design, summary, syl_df = build_design(
        speed, ahv, valid, fps, syl, light, **design_kwargs(args)
    )
    summary["has_syllables"] = syl is not None
    if summary["rare_states"]:
        log.warning("movestate: rare states (< min frames): %s", summary["rare_states"])
    rng = np.random.default_rng(int(_get(args, "seed", 0)))
    n_shuffles = int(_get(args, "n_shuffles_evt", 200))
    rows = [
        {
            "roi_idx": int(arrays["roi_idx"][i]),
            **cell_statistics(sig[i], design, n_shuffles=n_shuffles, rng=rng),
        }
        for i in range(sig.shape[0])
    ]
    return pd.DataFrame(rows), pd.DataFrame([summary]), syl_df


def within_group(cells: pd.DataFrame) -> pd.DataFrame:
    """Per cell type and measure: cell- and animal-level Wilcoxon vs 0, fraction significant.

    The cell-level Wilcoxon treats cells as independent and is descriptive;
    the animal-level Wilcoxon on animal medians is the inferential test.
    For shift-tested measures the significant cells are also split by the
    sign of the shift z (``n_sig_pos`` / ``n_sig_neg``).
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
            col = VALUE_COLUMN.get(k, k)
            if col not in g:
                continue
            animal_med = g.groupby("animal_id")[col].median().dropna()
            vals = g[col].dropna()
            rec: dict[str, Any] = {
                "celltype": ct,
                "measure": col,
                "n_cells_finite": int(vals.size),
                "median_cell_value": float(vals.median()) if vals.size else np.nan,
                "cell_wilcoxon_p": _wilcoxon(vals),
                "n_animals": int(animal_med.size),
                "median_animal_value": float(animal_med.median()) if animal_med.size else np.nan,
                "animal_wilcoxon_p": _wilcoxon(animal_med),
            }
            if k in TESTED and f"{k}_p" in g:
                sub = g[[f"{k}_p", f"{k}_z"]].dropna(subset=[f"{k}_p"])
                sig = sub[f"{k}_p"] < ALPHA
                n_sig = int(sig.sum())
                rec.update(
                    {
                        "n_cells_tested": int(len(sub)),
                        "n_sig": n_sig,
                        "n_sig_pos": int((sig & (sub[f"{k}_z"] > 0)).sum()),
                        "n_sig_neg": int((sig & (sub[f"{k}_z"] < 0)).sum()),
                        "frac_sig": n_sig / len(sub) if len(sub) else np.nan,
                        "binom_p": (
                            float(stats.binomtest(n_sig, int(len(sub)), ALPHA, "greater").pvalue)
                            if len(sub)
                            else np.nan
                        ),
                    }
                )
            out.append(rec)
    return pd.DataFrame(out)


def report_metrics(cells: pd.DataFrame) -> list[str]:
    """Value columns entering the between-group report (present in *cells*)."""
    cols = [VALUE_COLUMN.get(k, k) for k in TESTED + DESCRIPTIVE]
    return [c for c in cols if c in cells.columns and cells[c].notna().any()]


def run_movestate(
    args: argparse.Namespace, sessions: SessionIter, out_dir: Path
) -> dict[str, Any]:
    cell_rows, state_rows, syl_rows = [], [], []
    for row, arrays in sessions:
        cells, summary, syl_df = session_tables(arrays, args)
        cell_rows.append(attach_session_meta(cells, row))
        state_rows.append(attach_session_meta(summary, row))
        if len(syl_df):
            syl_rows.append(attach_session_meta(syl_df, row))
    if not cell_rows:
        return _empty_result("movestate", out_dir)
    cells = pd.concat(cell_rows, ignore_index=True)
    write_outputs(out_dir, "cells", cells)
    write_outputs(out_dir, "state_summary", pd.concat(state_rows, ignore_index=True))
    syl_all = pd.concat(syl_rows, ignore_index=True) if syl_rows else pd.DataFrame()
    write_outputs(out_dir, "syllable_summary", syl_all)
    wg = within_group(cells)
    write_outputs(out_dir, "within_group", wg)
    report = between_group_report(
        cells,
        report_metrics(cells),
        family="movestate",
        n_perms=int(_get(args, "n_perms", 10000)),
        seed=int(_get(args, "seed", 0)),
    )
    write_outputs(out_dir, "between_group_report", report)
    return {"n_sessions": len(cell_rows), "n_cells": len(cells), "report": report, "within": wg}


RUNNER = run_movestate
