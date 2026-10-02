#!/usr/bin/env python3
"""TCTX runner — time and episode-context coding, Penk+ vs Penk-CamKII+.

Plugs into ``run_celltype_programme.py`` with the same ``(args, sessions,
out_dir)`` signature as the runners in ``run_celltype_hypotheses.py``.
Each session's signal is averaged in coarse time bins (``--bin-s-ctx``,
default 10 s) and analysed with :mod:`hm2p.analysis.temporal_context`:

- session time decoding (full N and matched N) with a circular-shift null;
- epoch-identity decoding among same-light-state epochs;
- temporal-distance rho between epoch population vectors (Rubin 2015);
- per-cell drift, per-cell epoch selectivity (Kruskal-Wallis H, computed on
  ``--bin-s-sel`` bins, default 1 s, so its label-shift null has enough
  distinct partitions) and the first-vs-later dark epoch difference.

Outputs: ``cells.csv``, ``sessions.csv``, ``within_group.csv``,
``between_group_report.csv`` (families ``tctx_session`` and ``tctx_cell``).
The runner contains no S3 code and is tested with synthetic sessions.

References
----------
Mau W et al. 2018. "The same hippocampal CA1 population simultaneously codes
temporal information over multiple timescales." Current Biology.
doi:10.1016/j.cub.2018.03.051
Rubin A et al. 2015. "Hippocampal ensemble dynamics timestamp events in
long-term memory." eLife. doi:10.7554/eLife.12247
Ziv Y et al. 2013. "Long-term dynamics of CA1 hippocampal place codes."
Nature Neuroscience. doi:10.1038/nn.3329
Tsao A et al. 2018. "Integrating time from experience in the lateral
entorhinal cortex." Nature. doi:10.1038/s41586-018-0459-6
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_celltype_hypotheses import (  # noqa: E402
    SessionIter,
    _empty_result,
    _report_families,
    _signal_matrix,
)
from run_celltype_programme import attach_session_meta, write_outputs  # noqa: E402

log = logging.getLogger("celltype_programme")

RUNNER_KEY = "tctx"
EXTRA_SYNC_KEYS: list[str] = []

ALPHA = 0.05
MIN_ANIMALS_WILCOXON = 2

TCTX_FAMILIES = {
    "tctx_session": ["time_mae_matched_s", "time_rho_matched", "epoch_acc_matched_minus_chance"],
    # epsilon-squared = H / (N - 1): H normalised for the number of bins
    "tctx_cell": ["drift_abs_rho", "epoch_sel_eps2"],
}

#: Session-level columns tested against 0 at animal level (Wilcoxon).
SESSION_WILCOXON_COLS = [
    "time_rho_full",
    "time_rho_matched",
    "epoch_acc_minus_chance",
    "epoch_acc_matched_minus_chance",
    "tdist_rho",
]
#: Per-cell p-value columns summarised as fraction significant.
CELL_P_COLS = ["drift_p", "epoch_sel_p"]


def _params(args: argparse.Namespace) -> dict[str, Any]:
    """Runner parameters with defaults for attributes absent from *args*."""
    return {
        "bin_s": float(getattr(args, "bin_s_ctx", 10.0)),
        "bin_s_sel": float(getattr(args, "bin_s_sel", 1.0)),
        "n_matched": int(getattr(args, "n_cells_matched", 8)),
        "n_draws": int(getattr(args, "n_draws_tctx", 20)),
        "n_perms_decode": int(getattr(args, "n_perms_tctx_decode", 200)),
        "n_shuffles": int(getattr(args, "n_shuffles_tctx", 500)),
        "n_folds": int(getattr(args, "n_folds_tctx", 5)),
    }


def matched_epoch_accuracy(
    binned: np.ndarray,
    epoch_idx: np.ndarray,
    epoch_light: np.ndarray,
    n_cells: int,
    n_draws: int,
    rng: np.random.Generator,
) -> float:
    """Median epoch-identity accuracy minus chance over random cell subsets."""
    from hm2p.analysis.temporal_context import epoch_identity_decoding

    n_rois = binned.shape[0]
    if n_cells > n_rois or n_cells < 1:
        return float("nan")
    vals = []
    for _ in range(n_draws):
        idx = np.sort(rng.choice(n_rois, size=n_cells, replace=False))
        res = epoch_identity_decoding(binned[idx], epoch_idx, epoch_light, n_perms=0)
        vals.append(res["accuracy_minus_chance"])
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.nanmedian(arr)) if np.isfinite(arr).any() else float("nan")


def session_tctx(
    sig: np.ndarray, arrays: dict[str, Any], params: dict[str, Any], rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """All temporal-context measures for one session.

    Returns the per-cell table (columns ``roi`` = row index into *sig*) and
    the session-level record.
    """
    from hm2p.analysis.temporal_context import (
        bin_activity,
        epoch_identity_decoding,
        epoch_index,
        first_vs_later_dark,
        matched_n_decoding,
        per_cell_drift,
        per_cell_epoch_selectivity,
        pure_epoch_bins,
        temporal_distance,
        time_decoding_null,
    )

    fps = float(arrays["fps"])
    bin_s = params["bin_s"]
    light = np.asarray(arrays["light_on"], dtype=bool)
    bad = np.asarray(arrays["bad_behav"], dtype=bool)
    binned, centres, valid = bin_activity(sig, fps, bin_s=bin_s, mask=~bad)
    n_rois = sig.shape[0]
    rec: dict[str, Any] = {"n_cells": n_rois, "n_bins": int(valid.sum())}

    dec = time_decoding_null(
        binned, centres, n_perms=params["n_perms_decode"], n_folds=params["n_folds"], rng=rng
    )
    rec.update(
        {
            "time_rho_full": dec["rho"],
            "time_mae_full_s": dec["median_abs_error"],
            "time_p_full": dec["p_rho"],
        }
    )
    n_m = params["n_matched"]
    if n_rois >= n_m:
        mdec = matched_n_decoding(
            binned, centres, n_m, n_draws=params["n_draws"], rng=rng, n_folds=params["n_folds"]
        )
        rec["time_mae_matched_s"] = mdec["median_error"]
        rec["time_rho_matched"] = mdec["median_rho"]
    else:
        rec["time_mae_matched_s"] = np.nan
        rec["time_rho_matched"] = np.nan

    ep_idx, ep_light = epoch_index(light, fps, centres)
    pure = pure_epoch_bins(light, fps, centres, bin_s)
    ep_binned = binned.copy()
    ep_binned[:, ~pure] = np.nan
    ep = epoch_identity_decoding(
        ep_binned, ep_idx, ep_light, n_perms=params["n_perms_decode"], rng=rng
    )
    rec.update(
        {
            "epoch_acc": ep["accuracy"],
            "epoch_chance": ep["chance"],
            "epoch_acc_minus_chance": ep["accuracy_minus_chance"],
            "epoch_p": ep["p_value"],
            "n_epochs": ep["n_epochs"],
            "epoch_acc_matched_minus_chance": matched_epoch_accuracy(
                ep_binned, ep_idx, ep_light, n_m, params["n_draws"], rng
            ),
        }
    )
    td = temporal_distance(ep_binned, ep_idx, ep_light, n_perms=params["n_shuffles"], rng=rng)
    rec.update({"tdist_rho": td["rho"], "tdist_p": td["p_value"], "tdist_n_pairs": td["n_pairs"]})

    drift = per_cell_drift(binned, n_perms=params["n_shuffles"], rng=rng)
    # Selectivity uses fine bins: its label-shift null has about
    # (bins per epoch) distinct partitions, too few at 10 s bins.
    fine, fine_c, _ = bin_activity(sig, fps, bin_s=params["bin_s_sel"], mask=~bad)
    fine_idx, fine_light = epoch_index(light, fps, fine_c)
    fine[:, ~pure_epoch_bins(light, fps, fine_c, params["bin_s_sel"])] = np.nan
    sel = per_cell_epoch_selectivity(
        fine, fine_idx, fine_light, n_perms=params["n_shuffles"], rng=rng
    )
    fl = first_vs_later_dark(sig, light, bad, fps)
    cells = pd.DataFrame(
        {
            "roi": np.arange(n_rois),
            "drift_rho": drift["rho"],
            "drift_abs_rho": np.abs(drift["rho"]),
            "drift_p": drift["p_value"],
            "epoch_sel_H": sel["H"],
            "epoch_sel_eps2": sel["epsilon_sq"],
            "epoch_sel_p": sel["p_value"],
            "dark_first_value": fl["first_value"],
            "dark_later_mean": fl["later_mean"],
            "dark_first_minus_later": fl["difference"],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rec["frac_drift_sig"] = _frac_sig(cells["drift_p"])
        rec["frac_epoch_sel_sig"] = _frac_sig(cells["epoch_sel_p"])
        rec["median_dark_first_minus_later"] = float(np.nanmedian(fl["difference"]))
    rec["n_dark_epochs"] = fl["n_epochs"]
    return cells, rec


def _frac_sig(p: pd.Series) -> float:
    """Fraction of finite p-values below ``ALPHA``."""
    v = p.dropna().to_numpy(dtype=np.float64)
    return float((v < ALPHA).mean()) if v.size else float("nan")


def tctx_within_group(cells: pd.DataFrame, sess: pd.DataFrame) -> pd.DataFrame:
    """Within-group tests that each population carries temporal context.

    Three kinds of rows per cell type:

    - ``cell_frac`` — pooled fraction of cells with p < 0.05 for each per-cell
      measure, one-sided exact binomial against 0.05 (descriptive; cells are
      not independent).
    - ``session_frac`` — fraction of sessions whose own permutation test is
      significant (decoding, epoch identity, temporal distance), binomial
      against 0.05.
    - ``animal_wilcoxon`` — Wilcoxon signed-rank of animal medians against 0
      (the inferential test).
    """
    from scipy import stats

    from hm2p.analysis.event_aligned import _binom_greater

    rows: list[dict[str, Any]] = []
    for ct, g in cells.groupby("celltype", sort=True):
        for col in CELL_P_COLS:
            v = g[col].dropna().to_numpy(dtype=np.float64)
            k = int((v < ALPHA).sum())
            rows.append(
                {
                    "celltype": ct,
                    "test": "cell_frac",
                    "metric": col,
                    "n": int(v.size),
                    "k_significant": k,
                    "fraction": k / v.size if v.size else np.nan,
                    "p_value": _binom_greater(k, int(v.size)),
                }
            )
    for ct, g in sess.groupby("celltype", sort=True):
        for col in ["time_p_full", "epoch_p", "tdist_p"]:
            v = g[col].dropna().to_numpy(dtype=np.float64)
            k = int((v < ALPHA).sum())
            rows.append(
                {
                    "celltype": ct,
                    "test": "session_frac",
                    "metric": col,
                    "n": int(v.size),
                    "k_significant": k,
                    "fraction": k / v.size if v.size else np.nan,
                    "p_value": _binom_greater(k, int(v.size)),
                }
            )
        for col in SESSION_WILCOXON_COLS:
            animal = g.groupby("animal_id")[col].median().dropna().to_numpy(dtype=np.float64)
            rec: dict[str, Any] = {
                "celltype": ct,
                "test": "animal_wilcoxon",
                "metric": col,
                "n": int(animal.size),
                "median": float(np.median(animal)) if animal.size else np.nan,
                "p_value": np.nan,
            }
            if animal.size >= MIN_ANIMALS_WILCOXON and np.any(animal != 0):
                rec["p_value"] = float(stats.wilcoxon(animal).pvalue)
            rows.append(rec)
    return pd.DataFrame(rows)


def run_tctx(args: argparse.Namespace, sessions: SessionIter, out_dir: Path) -> dict[str, Any]:
    """Time / episode-context coding per session, then group statistics."""
    params = _params(args)
    rng = np.random.default_rng(args.seed)
    cell_rows: list[pd.DataFrame] = []
    sess_rows: list[pd.DataFrame] = []
    for row, arrays in sessions:
        sig = _signal_matrix(arrays, args.signal)
        if sig.shape[0] == 0 or sig.shape[1] == 0:
            log.warning("%s: empty signal, skipped", row["exp_id"])
            continue
        cells, rec = session_tctx(sig, arrays, params, rng)
        cells["roi_idx"] = np.asarray(arrays["roi_idx"])[cells["roi"].to_numpy(dtype=int)]
        cell_rows.append(attach_session_meta(cells.drop(columns=["roi"]), row))
        sess_rows.append(attach_session_meta(pd.DataFrame([rec]), row))
    if not cell_rows:
        return _empty_result("tctx", out_dir)
    cells_df = pd.concat(cell_rows, ignore_index=True)
    sess_df = pd.concat(sess_rows, ignore_index=True)
    write_outputs(out_dir, "cells", cells_df)
    write_outputs(out_dir, "sessions", sess_df)

    within = tctx_within_group(cells_df, sess_df)
    write_outputs(out_dir, "within_group", within)

    # Session- and cell-level families have different units of observation,
    # so each is reported from its own table and the two are concatenated.
    rep_sess = _report_families(
        sess_df, {"tctx_session": TCTX_FAMILIES["tctx_session"]}, out_dir, args
    )
    rep_cell = _report_families(cells_df, {"tctx_cell": TCTX_FAMILIES["tctx_cell"]}, out_dir, args)
    report = pd.concat([rep_sess, rep_cell], ignore_index=True)
    write_outputs(out_dir, "between_group_report", report)
    return {
        "n_sessions": len(sess_rows),
        "n_cells": len(cells_df),
        "within": within,
        "report": report,
    }


RUNNER = run_tctx
