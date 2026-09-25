#!/usr/bin/env python3
"""Hypothesis runners for ``run_celltype_programme.py`` (H2–H10).

Each runner takes ``(args, sessions, out_dir)`` where ``sessions`` yields
``(meta_row, arrays)`` pairs produced by ``run_celltype_programme.iter_sessions``
(see that module for the array keys), computes per-cell or per-session
measures with the pure functions in ``hm2p.analysis``, writes the raw tables,
and writes the four-level between-group report for each declared metric
family. The runners contain no S3 code so they are unit-tested with synthetic
sessions. See docs/plan-penk-vs-nonpenk.md.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from run_celltype_programme import attach_session_meta, between_group_report, write_outputs

log = logging.getLogger("celltype_programme")

SessionIter = Iterable[tuple[pd.Series, dict[str, Any]]]
Runner = Callable[[argparse.Namespace, SessionIter, Path], dict[str, Any]]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _signal_matrix(arrays: dict[str, Any], signal: str) -> np.ndarray:
    """Select the (n_rois, n_frames) signal requested on the command line."""
    if signal == "dff" or signal not in arrays or arrays[signal] is None:
        return np.asarray(arrays["dff"], dtype=np.float64)
    return np.asarray(arrays[signal], dtype=np.float64)


def _per_animal(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Collapse a per-cell/per-session table to one row per animal (medians)."""
    keep = ["animal_id", "celltype", "fibre", "lens", "virus_id", "gcamp"]
    keep = [c for c in keep if c in df.columns]
    return df.groupby(keep, as_index=False)[metrics].median()


def _report_families(
    df: pd.DataFrame,
    families: dict[str, list[str]],
    out_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Between-group report for every family, concatenated and written."""
    reports = []
    for family, metrics in families.items():
        present = [m for m in metrics if m in df.columns]
        if not present:
            continue
        reports.append(
            between_group_report(df, present, family=family, n_perms=args.n_perms, seed=args.seed)
        )
    report = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    write_outputs(out_dir, "between_group_report", report)
    return report


def _empty_result(name: str, out_dir: Path) -> dict[str, Any]:
    log.warning("%s: no usable sessions", name)
    write_outputs(out_dir, "summary", {"hypothesis": name, "n_sessions": 0})
    return {"n_sessions": 0}


# ---------------------------------------------------------------------------
# H4 — state dependence: transient vs sustained
# ---------------------------------------------------------------------------

H4_FAMILIES = {
    "state_dynamics": [
        "onset_transient_index",
        "offset_transient_index",
        "onset_latency_s",
        "immobility_decay_time_s",
        "sustained_ratio",
    ],
    "syllables": ["syllable_information_bits"],
}


def run_h4(args: argparse.Namespace, sessions: SessionIter, out_dir: Path) -> dict[str, Any]:
    from hm2p.analysis.state_dynamics import movement_state_summary, syllable_information

    rows: list[pd.DataFrame] = []
    for row, arrays in sessions:
        sig = _signal_matrix(arrays, args.signal)
        recs = []
        for i in range(sig.shape[0]):
            rec = movement_state_summary(
                sig[i], arrays["active"], arrays["bad_behav"], arrays["fps"]
            )
            rec = {k: v for k, v in rec.items() if np.isscalar(v)}
            rec["roi_idx"] = int(arrays["roi_idx"][i])
            if "syllable_id" in arrays:
                rec["syllable_information_bits"] = syllable_information(
                    sig[i], arrays["syllable_id"], arrays["mask"]
                )
            recs.append(rec)
        rows.append(attach_session_meta(pd.DataFrame(recs), row))
    if not rows:
        return _empty_result("h4", out_dir)
    cells = pd.concat(rows, ignore_index=True)
    write_outputs(out_dir, "cells", cells)
    report = _report_families(cells, H4_FAMILIES, out_dir, args)
    return {"n_sessions": len(rows), "n_cells": len(cells), "report": report}


# ---------------------------------------------------------------------------
# H5 — light/dark transition dynamics
# ---------------------------------------------------------------------------

TRANSITION_GRID_S = np.round(np.arange(-10.0, 30.0, 0.1), 3)


def _resample_timecourse(time_s: Any, values: Any) -> np.ndarray:
    """Interpolate a transition-aligned mean onto the common 10 Hz grid.

    Sessions differ slightly in imaging rate, so their aligned windows have
    different lengths; averaging across sessions needs a shared time axis.
    """
    t = np.asarray(time_s, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(v)
    if ok.sum() < 2:
        return np.full(TRANSITION_GRID_S.shape, np.nan)
    out = np.interp(TRANSITION_GRID_S, t[ok], v[ok], left=np.nan, right=np.nan)
    return out


H5_FAMILIES = {
    "transition_light_to_dark": ["ltd_early_amplitude", "ltd_late_amplitude", "ltd_frac_sig"],
    "transition_dark_to_light": ["dtl_early_amplitude", "dtl_late_amplitude", "dtl_frac_sig"],
    "recovery": ["mvl_recovery_time_s", "pd_recovery_time_s"],
}


def run_h5(args: argparse.Namespace, sessions: SessionIter, out_dir: Path) -> dict[str, Any]:
    from hm2p.analysis.transitions import (
        population_transition_response,
        recovery_time_s,
        tuning_recovery_timecourse,
    )

    cell_rows: list[pd.DataFrame] = []
    session_rows: list[dict[str, Any]] = []
    timecourses: dict[str, list[np.ndarray]] = {"ltd": [], "dtl": []}
    for row, arrays in sessions:
        sig = _signal_matrix(arrays, args.signal)
        fps = arrays["fps"]
        recs = pd.DataFrame({"roi_idx": arrays["roi_idx"]})
        ses: dict[str, Any] = {"n_rois": sig.shape[0]}
        for tag, direction in (("ltd", "light_to_dark"), ("dtl", "dark_to_light")):
            res = population_transition_response(
                sig, arrays["light_on"], arrays["bad_behav"], fps, direction=direction
            )
            recs[f"{tag}_early_amplitude"] = res["early_amplitude"]
            recs[f"{tag}_late_amplitude"] = res["late_amplitude"]
            recs[f"{tag}_p_value"] = res["p_value"]
            ses[f"{tag}_n_transitions"] = res["n_transitions"]
            ses[f"{tag}_frac_sig"] = float(np.nanmean(np.asarray(res["p_value"]) < 0.05))
            ses[f"{tag}_early_amplitude"] = float(np.nanmean(res["early_amplitude"]))
            ses[f"{tag}_late_amplitude"] = float(np.nanmean(res["late_amplitude"]))
            if res["n_transitions"] > 0:
                timecourses[tag].append(
                    _resample_timecourse(res["time_s"], res["mean_timecourse"])
                )
        # tuning recovery after dark->light, per cell, then session medians
        mvl_rt, pd_rt = [], []
        for i in range(sig.shape[0]):
            tc = tuning_recovery_timecourse(
                sig[i], arrays["hd_deg"], arrays["mask"], arrays["light_on"], fps
            )
            if tc["n_transitions"] == 0:
                continue
            mvl_rt.append(recovery_time_s(tc["time_s"], tc["mvl"], direction="up"))
            pd_rt.append(recovery_time_s(tc["time_s"], tc["pd_deviation_deg"], direction="down"))
        recs["mvl_recovery_time_s"] = (mvl_rt + [np.nan] * sig.shape[0])[: sig.shape[0]]
        recs["pd_recovery_time_s"] = (pd_rt + [np.nan] * sig.shape[0])[: sig.shape[0]]
        ses["mvl_recovery_time_s"] = float(np.nanmedian(mvl_rt)) if mvl_rt else np.nan
        ses["pd_recovery_time_s"] = float(np.nanmedian(pd_rt)) if pd_rt else np.nan
        cell_rows.append(attach_session_meta(recs, row))
        session_rows.append(attach_session_meta(pd.DataFrame([ses]), row).iloc[0].to_dict())
    if not cell_rows:
        return _empty_result("h5", out_dir)
    cells = pd.concat(cell_rows, ignore_index=True)
    sess = pd.DataFrame(session_rows)
    write_outputs(out_dir, "cells", cells)
    write_outputs(
        out_dir,
        "sessions",
        sess.drop(
            columns=[
                c
                for c in sess
                if c.endswith("_time_s") and c not in ("mvl_recovery_time_s", "pd_recovery_time_s")
            ],
            errors="ignore",
        ),
    )
    report = _report_families(sess, H5_FAMILIES, out_dir, args)
    write_outputs(
        out_dir,
        "population_timecourses",
        {
            "time_s": TRANSITION_GRID_S,
            **{k: np.nanmean(np.vstack(v), axis=0) if v else [] for k, v in timecourses.items()},
        },
    )
    return {"n_sessions": len(cell_rows), "n_cells": len(cells), "report": report}


# ---------------------------------------------------------------------------
# H2 — calcium activity signature (per-cell feature table)
# ---------------------------------------------------------------------------


def build_feature_table(
    args: argparse.Namespace, sessions: SessionIter, include_catch22: bool = False
) -> pd.DataFrame:
    """Per-cell feature table across sessions with metadata columns attached.

    With ``--signal spikes`` the table is built from the CASCADE inferred
    spike rates (``sp_`` rate family plus the ``tun_`` and ``act_``
    families) instead of dF/F; the dF/F event-kinetics and trace-shape
    families are then absent, and sessions without a ``spikes`` array are
    skipped. ``include_catch22`` applies to the dF/F table only.
    """
    from hm2p.analysis.cell_features import session_feature_table, session_spike_feature_table

    use_spikes = getattr(args, "signal", "dff") == "spikes"
    rows: list[pd.DataFrame] = []
    for row, arrays in sessions:
        behaviour = (
            arrays["hd_deg"],
            arrays["ahv_deg_s"],
            arrays["speed_cm_s"],
            arrays["light_on"],
            arrays["active"],
            arrays["bad_behav"],
            arrays["fps"],
        )
        if use_spikes:
            if arrays.get("spikes") is None:
                log.warning("%s: no inferred spikes, skipping session", row["exp_id"])
                continue
            table = session_spike_feature_table(
                np.asarray(arrays["spikes"], dtype=np.float64),
                *behaviour,
                roi_types=arrays.get("roi_types"),
            )
        else:
            table = session_feature_table(
                _signal_matrix(arrays, "dff"),
                *behaviour,
                roi_types=arrays.get("roi_types"),
                include_catch22=include_catch22,
            )
        table["roi_idx"] = arrays["roi_idx"]
        if arrays.get("spikes_model") is not None:
            table["spikes_model"] = str(arrays["spikes_model"])
        rows.append(attach_session_meta(table, row))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def virus_variant_check(cells: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Kruskal-Wallis across Penk+ virus constructs on animal medians."""
    from scipy import stats

    penk = cells[cells["celltype"] == "penk"]
    out = []
    for m in metrics:
        if m not in penk.columns:
            continue
        per_animal = penk.groupby(["animal_id", "virus_id"])[m].median().reset_index()
        groups = [g[m].dropna().to_numpy() for _, g in per_animal.groupby("virus_id")]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            out.append({"metric": m, "n_variants": len(groups), "p_value": np.nan})
            continue
        try:
            p = float(stats.kruskal(*groups).pvalue)
        except ValueError:
            p = np.nan
        out.append({"metric": m, "n_variants": len(groups), "p_value": p})
    return pd.DataFrame(out)


def snr_confounds(cells: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Spearman correlation of each metric with event SNR (expression proxy)."""
    from hm2p.analysis.mixed_stats import confound_check

    rows = []
    for m in metrics:
        if m in cells.columns and "ev_snr" in cells.columns and m != "ev_snr":
            sub = cells.dropna(subset=[m, "ev_snr"])
            if len(sub) >= 5:
                for r in confound_check(sub, m, ["ev_snr"]):
                    rows.append({"metric": m, **r})
    return pd.DataFrame(rows)


def snr_matched_cells(
    cells: pd.DataFrame, snr_col: str = "ev_snr", n_bins: int = 10, seed: int = 42
) -> pd.DataFrame:
    """Subsample cells so the two groups have matched SNR distributions.

    Expression level and optics differ between mice; matching on event SNR
    (``match_indices_1d`` on the pooled cells) removes the part of any
    kinetics difference that tracks signal quality.
    """
    from hm2p.analysis.matched_tuning import match_indices_1d

    if snr_col not in cells.columns:
        # the inferred-spike table carries no event SNR; matching is not defined
        log.info("snr_matched_cells: %s absent, skipping SNR matching", snr_col)
        return cells.iloc[0:0]
    sub = cells.dropna(subset=[snr_col, "celltype"])
    a = sub[sub["celltype"] == "penk"]
    b = sub[sub["celltype"] == "nonpenk"]
    if len(a) < 5 or len(b) < 5:
        return sub.iloc[0:0]
    ia, ib = match_indices_1d(
        a[snr_col].to_numpy(dtype=float),
        b[snr_col].to_numpy(dtype=float),
        n_bins=n_bins,
        circular=False,
        rng=np.random.default_rng(seed),
    )
    return pd.concat([a.iloc[ia], b.iloc[ib]], ignore_index=True)


def run_h2(args: argparse.Namespace, sessions: SessionIter, out_dir: Path) -> dict[str, Any]:
    from hm2p.analysis.cell_features import feature_families

    if getattr(args, "features", None) is not None:
        cells = pd.read_csv(args.features)
    else:
        cells = build_feature_table(args, sessions, include_catch22=True)
    if cells.empty:
        return _empty_result("h2", out_dir)
    write_outputs(out_dir, "cells", cells)
    families = feature_families()
    report = _report_families(cells, families, out_dir, args)
    all_metrics = [m for fam in families.values() for m in fam]
    write_outputs(out_dir, "virus_variant_kruskal", virus_variant_check(cells, all_metrics))
    write_outputs(out_dir, "snr_confounds", snr_confounds(cells, all_metrics))
    matched = snr_matched_cells(cells, seed=args.seed)
    if not matched.empty:
        write_outputs(out_dir, "cells_snr_matched", matched)
        reports = []
        for family, metrics in families.items():
            present = [m for m in metrics if m in matched.columns]
            if present:
                reports.append(
                    between_group_report(
                        matched, present, family=family, n_perms=args.n_perms, seed=args.seed
                    )
                )
        write_outputs(
            out_dir,
            "between_group_report_snr_matched",
            pd.concat(reports, ignore_index=True) if reports else pd.DataFrame(),
        )
    return {"n_sessions": int(cells["exp_id"].nunique()), "n_cells": len(cells), "report": report}


# ---------------------------------------------------------------------------
# H3 — angular head velocity coding (speed-matched light vs dark)
# ---------------------------------------------------------------------------

H3_FAMILIES = {
    "ahv": [
        "ahv_modulation_depth",
        "ahv_asymmetry_index",
        "ahv_modulation_depth_light",
        "ahv_modulation_depth_dark",
        "ahv_modulation_depth_matched_light",
        "ahv_modulation_depth_matched_dark",
        "ahv_dark_minus_light_matched",
        "atd_best_lag_ms",
    ],
}


def _ahv_depth(signal: np.ndarray, ahv: np.ndarray, mask: np.ndarray) -> float:
    from hm2p.analysis.ahv import ahv_modulation_index, ahv_tuning_curve

    if mask.sum() < 50:
        return np.nan
    curve, centers = ahv_tuning_curve(signal, ahv, mask)
    return float(ahv_modulation_index(curve, centers)["modulation_depth"])


def run_h3(args: argparse.Namespace, sessions: SessionIter, out_dir: Path) -> dict[str, Any]:
    from hm2p.analysis.ahv import ahv_modulation_index, ahv_tuning_curve, anticipatory_time_delay
    from hm2p.analysis.matched_tuning import match_indices_1d

    rng = np.random.default_rng(args.seed)
    rows: list[pd.DataFrame] = []
    for row, arrays in sessions:
        sig = _signal_matrix(arrays, args.signal)
        ahv, mask, light = arrays["ahv_deg_s"], arrays["mask"], arrays["light_on"]
        speed = arrays["speed_cm_s"]
        li = np.where(mask & light)[0]
        di = np.where(mask & ~light)[0]
        ml = np.zeros_like(mask)
        md = np.zeros_like(mask)
        if len(li) >= 50 and len(di) >= 50:
            a, b = match_indices_1d(speed[li], speed[di], n_bins=20, circular=False, rng=rng)
            ml[li[a]] = True
            md[di[b]] = True
        recs = []
        for i in range(sig.shape[0]):
            rec: dict[str, Any] = {"roi_idx": int(arrays["roi_idx"][i])}
            if mask.sum() >= 50:
                curve, centers = ahv_tuning_curve(sig[i], ahv, mask)
                mi = ahv_modulation_index(curve, centers)
                rec["ahv_modulation_depth"] = mi["modulation_depth"]
                rec["ahv_asymmetry_index"] = mi["asymmetry_index"]
                atd = anticipatory_time_delay(sig[i], arrays["hd_deg"], mask, fps=arrays["fps"])
                rec["atd_best_lag_ms"] = atd["best_lag_ms"]
            rec["ahv_modulation_depth_light"] = _ahv_depth(sig[i], ahv, mask & light)
            rec["ahv_modulation_depth_dark"] = _ahv_depth(sig[i], ahv, mask & ~light)
            rec["ahv_modulation_depth_matched_light"] = _ahv_depth(sig[i], ahv, ml)
            rec["ahv_modulation_depth_matched_dark"] = _ahv_depth(sig[i], ahv, md)
            rec["ahv_dark_minus_light_matched"] = (
                rec["ahv_modulation_depth_matched_dark"]
                - rec["ahv_modulation_depth_matched_light"]
            )
            recs.append(rec)
        rows.append(attach_session_meta(pd.DataFrame(recs), row))
    if not rows:
        return _empty_result("h3", out_dir)
    cells = pd.concat(rows, ignore_index=True)
    write_outputs(out_dir, "cells", cells)
    report = _report_families(cells, H3_FAMILIES, out_dir, args)
    return {"n_sessions": len(rows), "n_cells": len(cells), "report": report}


# ---------------------------------------------------------------------------
# H6 — heterogeneity and omnibus separability
# ---------------------------------------------------------------------------


def center_features_by_animal(cells: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Subtract each animal's mean from every feature.

    Removes between-animal offsets (expression, optics, virus construct) so
    that dispersion and separability reflect within-animal cell-to-cell
    structure only. Between-group mean differences are removed as well, so
    this is a control for the heterogeneity tests, not a replacement for the
    location tests.
    """
    out = cells.copy()
    means = out.groupby("animal_id")[feature_cols].transform("mean")
    out[feature_cols] = out[feature_cols] - means
    return out


def run_h6(args: argparse.Namespace, sessions: SessionIter, out_dir: Path) -> dict[str, Any]:
    from hm2p.analysis.cell_features import feature_families
    from hm2p.analysis.heterogeneity import (
        classifier_permutation_test,
        dispersion_ratio_test,
        energy_distance_test,
        gmm_components_by_group,
        loao_classifier,
        penk_like_fraction,
        prepare_feature_matrix,
    )

    if args.features is not None:
        cells = pd.read_csv(args.features)
    else:
        cells = build_feature_table(args, sessions, include_catch22=True)
    if cells.empty:
        return _empty_result("h6", out_dir)
    fams = feature_families()
    feature_cols = [c for c in cells.columns if c.startswith("c22_")]
    feature_cols += fams["kinetics"] + fams["trace_shape"] + fams["tuning"]
    feature_cols = [c for c in feature_cols if c in cells.columns]
    if getattr(args, "center_by_animal", False):
        cells = center_features_by_animal(cells, feature_cols)
    x, meta, used = prepare_feature_matrix(cells, feature_cols)
    if x.shape[0] < 10 or meta["celltype"].nunique() != 2:
        return _empty_result("h6", out_dir)
    labels = meta["celltype"].to_numpy()
    animals = meta["animal_id"].to_numpy()
    rng = np.random.default_rng(args.seed)
    n_perms = min(args.n_perms, 1000)
    summary: dict[str, Any] = {
        "n_cells": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "features": used,
        "dispersion": dispersion_ratio_test(x, labels, animals, n_perms=n_perms, rng=rng),
        "energy_distance": energy_distance_test(x, labels, animals, n_perms=n_perms, rng=rng),
        "gmm": gmm_components_by_group(x, labels, rng=rng),
        "penk_like": penk_like_fraction(x, labels),
    }
    models = ("logistic", "gbm") if getattr(args, "gbm", True) else ("logistic",)
    for model in models:
        clf = loao_classifier(x, labels, animals, model=model, rng=rng, feature_names=used)
        summary[f"classifier_{model}"] = {
            k: v for k, v in clf.items() if k not in ("importance", "importance_by_feature")
        }
        if "importance_by_feature" in clf:
            write_outputs(
                out_dir,
                f"importance_{model}",
                pd.DataFrame(
                    {
                        "feature": list(clf["importance_by_feature"]),
                        "importance": list(clf["importance_by_feature"].values()),
                    }
                ),
            )
    summary["classifier_permutation"] = classifier_permutation_test(
        x, labels, animals, n_perms=min(n_perms, 200), rng=rng
    )
    write_outputs(out_dir, "summary", summary)
    return {"n_cells": int(x.shape[0]), "summary": summary}


# ---------------------------------------------------------------------------
# H8 — encoding models
# ---------------------------------------------------------------------------

H8_VARIABLES = ["hd", "ahv", "speed", "position", "light", "syllable"]
H8_FAMILIES = {
    "encoding": ["full_dev_expl"] + [f"part_{v}" for v in H8_VARIABLES],
}


def _glm_response(arrays: dict[str, Any], signal: str) -> np.ndarray:
    """Non-negative per-frame response for the Poisson GLM."""
    if signal == "spikes" and "spikes" in arrays:
        return np.clip(np.asarray(arrays["spikes"], dtype=np.float64), 0, None)
    if "event_masks" in arrays:
        return np.asarray(arrays["event_masks"], dtype=np.float64)
    return np.clip(np.asarray(arrays["dff"], dtype=np.float64), 0, None)


def run_h8(args: argparse.Namespace, sessions: SessionIter, out_dir: Path) -> dict[str, Any]:
    from hm2p.analysis.encoding import build_design_matrix, population_encoding_profiles

    rows: list[pd.DataFrame] = []
    for row, arrays in sessions:
        y = _glm_response(arrays, args.signal)
        design = build_design_matrix(
            hd_deg=arrays["hd_deg"],
            ahv_deg_s=arrays["ahv_deg_s"],
            speed_cm_s=arrays["speed_cm_s"],
            x_mm=arrays.get("x_mm"),
            y_mm=arrays.get("y_mm"),
            light_on=arrays["light_on"],
            syllable_id=arrays.get("syllable_id"),
        )
        design.valid &= arrays["mask"]
        prof = population_encoding_profiles(
            y,
            design,
            rng=np.random.default_rng(args.seed),
            run_selection=getattr(args, "glm_selection", True),
        )
        prof = prof.rename(columns={"roi": "roi_idx"})
        prof["roi_idx"] = arrays["roi_idx"][prof["roi_idx"].to_numpy()]
        rows.append(attach_session_meta(prof, row))
    if not rows:
        return _empty_result("h8", out_dir)
    cells = pd.concat(rows, ignore_index=True)
    cells["selected"] = cells["selected"].astype(str)
    write_outputs(out_dir, "cells", cells)
    report = _report_families(cells, H8_FAMILIES, out_dir, args)
    dom = cells.groupby(["celltype", "dominant"]).size().unstack(fill_value=0)
    write_outputs(out_dir, "dominant_by_celltype", dom.reset_index())
    return {"n_sessions": len(rows), "n_cells": len(cells), "report": report}


# ---------------------------------------------------------------------------
# H10 — behaviour-coupled navigational coding
# ---------------------------------------------------------------------------

H10_FAMILIES = {
    "navigation_cells": [
        "place_skaggs_info",
        "syllable_information_bits",
        "familiarity_spearman_r",
    ],
    "navigation_sessions": ["junction_decode_accuracy_minus_chance"],
}


def run_h10(args: argparse.Namespace, sessions: SessionIter, out_dir: Path) -> dict[str, Any]:
    from hm2p.analysis.state_dynamics import syllable_information
    from hm2p.analysis.tuning import compute_place_rate_map, spatial_information
    from hm2p.maze.discretize import discretize_position_fast
    from hm2p.maze.neural import (
        activity_by_traversal_number,
        count_corridor_traversals,
        decode_junction_choice,
        extract_junction_events,
        pre_junction_population_vectors,
    )
    from hm2p.maze.topology import build_rose_maze

    maze = build_rose_maze()
    cell_rows: list[pd.DataFrame] = []
    session_rows: list[dict[str, Any]] = []
    for row, arrays in sessions:
        sig = _signal_matrix(arrays, args.signal)
        mask, fps = arrays["mask"], arrays["fps"]
        recs = []
        have_maze = "x_maze" in arrays and "y_maze" in arrays
        cell_idx = None
        traversal = None
        if have_maze:
            cell_idx = discretize_position_fast(
                np.asarray(arrays["x_maze"], dtype=np.float64),
                np.asarray(arrays["y_maze"], dtype=np.float64),
                maze,
            ).astype(np.int64)
            cell_idx[~mask] = -1
            traversal = count_corridor_traversals(cell_idx, maze)
        for i in range(sig.shape[0]):
            rec: dict[str, Any] = {"roi_idx": int(arrays["roi_idx"][i])}
            if "x_mm" in arrays and "y_mm" in arrays:
                rate_map, occ, _, _ = compute_place_rate_map(
                    sig[i], arrays["x_mm"] / 10.0, arrays["y_mm"] / 10.0, mask, fps=fps
                )
                rec["place_skaggs_info"] = spatial_information(rate_map, occ)
            if "syllable_id" in arrays:
                rec["syllable_information_bits"] = syllable_information(
                    sig[i], arrays["syllable_id"], mask
                )
            if cell_idx is not None and traversal is not None:
                fam = activity_by_traversal_number(sig[i], cell_idx, traversal, mask)
                rec["familiarity_spearman_r"] = fam.get("spearman_r", np.nan)
            recs.append(rec)
        cell_rows.append(attach_session_meta(pd.DataFrame(recs), row))
        ses: dict[str, Any] = {"n_rois": sig.shape[0]}
        if cell_idx is not None:
            events = extract_junction_events(cell_idx, maze)
            if events:
                x, yv = pre_junction_population_vectors(sig, events)
                dec = decode_junction_choice(x, yv)
                ses["junction_decode_accuracy"] = dec["accuracy"]
                ses["junction_chance"] = dec["chance_level"]
                ses["junction_n_events"] = dec["n_events"]
                ses["junction_decode_accuracy_minus_chance"] = (
                    dec["accuracy"] - dec["chance_level"]
                )
        session_rows.append(attach_session_meta(pd.DataFrame([ses]), row).iloc[0].to_dict())
    if not cell_rows:
        return _empty_result("h10", out_dir)
    cells = pd.concat(cell_rows, ignore_index=True)
    sess = pd.DataFrame(session_rows)
    write_outputs(out_dir, "cells", cells)
    write_outputs(out_dir, "sessions", sess)
    rep_cells = _report_families(
        cells, {"navigation_cells": H10_FAMILIES["navigation_cells"]}, out_dir, args
    )
    rep_sess = pd.DataFrame()
    if "junction_decode_accuracy_minus_chance" in sess.columns:
        rep_sess = between_group_report(
            sess,
            H10_FAMILIES["navigation_sessions"],
            family="navigation_sessions",
            n_perms=args.n_perms,
            seed=args.seed,
        )
    report = pd.concat([rep_cells, rep_sess], ignore_index=True)
    write_outputs(out_dir, "between_group_report", report)
    return {"n_sessions": len(cell_rows), "n_cells": len(cells), "report": report}


# ---------------------------------------------------------------------------
# H7 — population HD code: decoding and topology (session level)
# ---------------------------------------------------------------------------

H7_FAMILIES = {
    "topology": [
        "ring_angle_hd_corr_all",
        "ring_angle_hd_corr_light",
        "ring_angle_hd_corr_dark",
        "ring_radius_cv_all",
        "persistence_ring_score",
        "matched_decode_error_median",
    ],
}


def run_h7(args: argparse.Namespace, sessions: SessionIter, out_dir: Path) -> dict[str, Any]:
    from hm2p.analysis.topology import topology_summary

    n_matched = getattr(args, "n_cells_matched", None)
    session_rows: list[dict[str, Any]] = []
    for row, arrays in sessions:
        sig = _signal_matrix(arrays, args.signal)
        if sig.shape[0] < 3:
            continue
        rng = np.random.default_rng(args.seed)
        matched = n_matched if n_matched and sig.shape[0] >= n_matched else None
        summ = topology_summary(
            sig,
            arrays["hd_deg"],
            arrays["mask"],
            arrays["light_on"],
            n_cells_matched=matched,
            rng=rng,
        )
        ses: dict[str, Any] = {
            "n_rois": sig.shape[0],
            "ripser_available": summ["ripser_available"],
        }
        for cond in ("all", "light", "dark"):
            ring = summ[f"pca_{cond}"]
            ses[f"ring_angle_hd_corr_{cond}"] = abs(float(ring["angle_hd_correlation"]))
            ses[f"ring_radius_cv_{cond}"] = float(ring["radius_cv"])
            ses[f"ring_uniformity_p_{cond}"] = float(ring["angle_uniformity_p"])
        ses["persistence_ring_score"] = float(summ["persistence"].get("ring_score", np.nan))
        md = summ.get("matched_decoding")
        ses["matched_decode_error_median"] = float(md["median"]) if md else np.nan
        session_rows.append(attach_session_meta(pd.DataFrame([ses]), row).iloc[0].to_dict())
    if not session_rows:
        return _empty_result("h7", out_dir)
    sess = pd.DataFrame(session_rows)
    write_outputs(out_dir, "sessions", sess)
    report = _report_families(sess, H7_FAMILIES, out_dir, args)
    return {"n_sessions": len(sess), "report": report}


# ---------------------------------------------------------------------------
# H9 — network coupling
# ---------------------------------------------------------------------------

H9_FAMILIES = {
    "coupling": [
        "pop_coupling_all",
        "pop_coupling_light",
        "pop_coupling_dark",
        "mean_noise_corr",
        "xcorr_asymmetry",
        "xcorr_peak_lag",
    ],
}


def run_h9(args: argparse.Namespace, sessions: SessionIter, out_dir: Path) -> dict[str, Any]:
    from hm2p.analysis.coupling import coupling_summary

    rows: list[pd.DataFrame] = []
    for row, arrays in sessions:
        sig = _signal_matrix(arrays, args.signal)
        if sig.shape[0] < 3:
            continue
        table = coupling_summary(sig, arrays["hd_deg"], arrays["mask"], arrays["light_on"])
        table = table.reset_index()
        table["roi_idx"] = arrays["roi_idx"][table["roi"].to_numpy()]
        rows.append(attach_session_meta(table.drop(columns=["roi"]), row))
    if not rows:
        return _empty_result("h9", out_dir)
    cells = pd.concat(rows, ignore_index=True)
    write_outputs(out_dir, "cells", cells)
    report = _report_families(cells, H9_FAMILIES, out_dir, args)
    return {"n_sessions": len(rows), "n_cells": len(cells), "report": report}


# ---------------------------------------------------------------------------
# Registry (filled in as hypotheses are implemented)
# ---------------------------------------------------------------------------

RUNNERS: dict[str, Runner] = {
    "h2": run_h2,
    "h3": run_h3,
    "h4": run_h4,
    "h5": run_h5,
    "h6": run_h6,
    "h7": run_h7,
    "h8": run_h8,
    "h9": run_h9,
    "h10": run_h10,
}
