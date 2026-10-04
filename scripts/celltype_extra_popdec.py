#!/usr/bin/env python3
"""Runner 'popdec': pooled-population decoding of maze behaviour.

All soma ROIs of a session are pooled (cell type is ignored for the main
result). Four decoders are run per session with
``hm2p.analysis.population_maze.decode_session``: runs into vs out of dead
ends (matched on run duration and speed), maze cell during running (balanced
accuracy and mean graph distance, the latter the headline), travel direction
within maze edges, and exit arm at T-junctions (stratified by junction and
arrival arm). Each is fitted on neural activity, on behaviour only (head
direction, speed, |AHV|) and on neural activity with head direction
regressed out, with a circular-shift null for every combination.

Outputs (``results/celltype_programme/popdec/``):

- ``sessions.csv``: one row per session x decoder x feature set x metric
  (observed, null mean/sd, z, p, excess over null, sample/class/cell counts,
  skip reason, cell type).
- ``subset_curve.csv``: neural decoding vs number of cells per session;
  ``subset_curve_summary.csv``: medians across sessions.
- ``behaviour_counts.csv``: per-session sample counts and skip reasons.
- ``summary.csv``: excess over the null across sessions and across animal
  medians (Wilcoxon signed-rank), pooled (main result) and per cell type
  (descriptive).
- ``comparisons.csv``: paired Wilcoxon of excess between feature sets
  (neural vs behaviour, neural with HD removed vs behaviour, neural vs
  neural with HD removed).
- ``run_info.json``: parameters, signal and counts.

Registered through ``run_celltype_programme.load_extra_runners``.

References: see ``hm2p.analysis.population_maze`` (Rosenberg et al. 2021
eLife doi:10.7554/eLife.66175; Alexander & Nitz 2015 Nat Neurosci
doi:10.1038/nn.4058; Brodersen et al. 2010 ICPR doi:10.1109/ICPR.2010.764;
Roberts et al. 2017 Ecography doi:10.1111/ecog.02881).
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

RUNNER_KEY = "popdec"
DEFAULT_SHUFFLES = 200


def _get(args: argparse.Namespace, name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    return default if value is None else value


def params_from_args(args: argparse.Namespace) -> Any:
    """``PopDecParams`` from the programme command line (defaults where absent)."""
    from hm2p.analysis.population_maze import PopDecParams

    return PopDecParams(
        n_shuffles=int(_get(args, "popdec_shuffles", DEFAULT_SHUFFLES)),
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


def session_tables(
    arrays: dict[str, Any], args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | None:
    """Decoder table, cell-subset table and behaviour counts for one session.

    Returns None when maze position is missing.
    """
    from hm2p.analysis.population_maze import decode_session

    if arrays.get("x_maze") is None or arrays.get("y_maze") is None:
        return None
    sig = _signal_matrix(arrays, args.signal)
    n = sig.shape[1]

    def _f(key: str) -> np.ndarray:
        return np.asarray(arrays[key], dtype=np.float64)[:n]

    res = decode_session(
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
    res.decoders["signal"] = used
    res.subsets["signal"] = used
    return res.decoders, res.subsets, {**res.counts, "signal": used}


def run_popdec(args: argparse.Namespace, sessions: SessionIter, out_dir: Path) -> dict[str, Any]:
    """Run all sessions and write the outputs listed in the module docstring."""
    from hm2p.analysis.population_maze import (
        across_session_summary,
        paired_feature_comparisons,
        subset_curve_summary,
    )

    dec_rows, sub_rows, count_rows = [], [], []
    for row, arrays in sessions:
        t0 = time.time()
        res = session_tables(arrays, args)
        if res is None:
            log.warning("popdec: %s has no maze position; skipped", row.get("exp_id"))
            count_rows.append(attach_session_meta(pd.DataFrame([{"reason": "no maze"}]), row))
            continue
        dec, sub, counts = res
        dec_rows.append(attach_session_meta(dec, row))
        sub_rows.append(attach_session_meta(sub, row))
        count_rows.append(attach_session_meta(pd.DataFrame([{**counts, "reason": ""}]), row))
        log.info("popdec: %s done in %.0f s", row.get("exp_id"), time.time() - t0)
    if not dec_rows:
        return _empty_result("popdec", out_dir)
    sessions_df = pd.concat(dec_rows, ignore_index=True)
    subsets = pd.concat(sub_rows, ignore_index=True)
    write_outputs(out_dir, "sessions", sessions_df)
    write_outputs(out_dir, "subset_curve", subsets)
    write_outputs(out_dir, "subset_curve_summary", subset_curve_summary(subsets))
    write_outputs(out_dir, "behaviour_counts", pd.concat(count_rows, ignore_index=True))
    summary = across_session_summary(sessions_df)
    comparisons = paired_feature_comparisons(sessions_df)
    write_outputs(out_dir, "summary", summary)
    write_outputs(out_dir, "comparisons", comparisons)
    info = {
        "hypothesis": "popdec",
        "signal": str(_get(args, "signal", "dff")),
        "n_sessions": int(sessions_df["exp_id"].nunique()),
        "n_animals": int(sessions_df["animal_id"].nunique()),
        "params": asdict(params_from_args(args)),
    }
    write_outputs(out_dir, "run_info", info)
    return {**info, "summary": summary, "comparisons": comparisons}


RUNNER = run_popdec
