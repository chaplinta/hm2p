#!/usr/bin/env python3
"""Test all hypotheses from docs/hypotheses.md and generate a report.

Loads analysis.h5 + sync.h5 + metadata, pools cells with animal/session
metadata, runs non-parametric tests (animal-level Mann-Whitney, cluster
permutation), checks signal quality confounds, applies FDR correction,
and outputs a structured markdown + CSV report.

Usage:
    python scripts/test_hypotheses.py                  # full report
    python scripts/test_hypotheses.py --signal dff     # dF/F only (default)
    python scripts/test_hypotheses.py --signal events   # event-based
    python scripts/test_hypotheses.py --n-perms 1000   # faster (fewer perms)
    python scripts/test_hypotheses.py --output results/hypothesis_report
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hm2p.analysis.mixed_stats import (
    animal_summary_test,
    cluster_permutation_test,
    confound_check,
    fdr_correct,
    interaction_contrast,
    run_between_group_test,
    within_cell_test,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("hypotheses")

BUCKET = "hm2p-derivatives"

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


_S3_CLIENT = None


def _s3():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        import boto3
        _S3_CLIENT = boto3.Session(profile_name="hm2p-agent").client("s3")
    return _S3_CLIENT


def _download_h5(key: str) -> h5py.File | None:
    """Download an HDF5 file from S3 into memory."""
    try:
        obj = _s3().get_object(Bucket=BUCKET, Key=key)
        data = obj["Body"].read()
        return h5py.File(io.BytesIO(data), "r")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load animals.csv and experiments.csv."""
    base = Path(__file__).resolve().parent.parent / "metadata"
    animals = pd.read_csv(base / "animals.csv")
    animals["animal_id"] = animals["animal_id"].astype(str)
    exps = pd.read_csv(base / "experiments.csv")
    exps["animal_id"] = exps["exp_id"].str.split("_").str[-1]
    return animals, exps


def load_all_analysis(
    animals: pd.DataFrame,
    exps: pd.DataFrame,
    signal: str = "dff",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all analysis.h5 + sync.h5 files and build per-cell + per-session DataFrames.

    Returns
    -------
    cell_df : DataFrame
        One row per (session, roi) with analysis metrics + metadata.
    session_df : DataFrame
        One row per session with behavioural metrics (speed, active, light/dark).
    """
    s3 = _s3()
    cell_rows: list[dict] = []
    session_rows: list[dict] = []

    valid_exps = exps[exps["exclude"].astype(str).str.strip() != "1"]
    log.info("Loading %d sessions...", len(valid_exps))

    for _, exp in valid_exps.iterrows():
        exp_id = exp["exp_id"]
        animal_id = exp["animal_id"]
        parts = exp_id.split("_")
        sub = f"sub-{animal_id}"
        ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"

        animal_row = animals[animals["animal_id"] == animal_id]
        if animal_row.empty:
            continue
        animal_info = animal_row.iloc[0]
        celltype = str(animal_info.get("celltype", ""))

        # Load sync.h5
        sync_key = f"sync/{sub}/{ses}/sync.h5"
        sync_f = _download_h5(sync_key)
        dff_data = None
        if sync_f is not None and "dff" in sync_f:
            dff_data = sync_f["dff"][:]

        # --- Extract session-level behavioural metrics from sync.h5 ---
        if sync_f is not None:
            _extract_session_behav(
                sync_f, session_rows, exp_id, animal_id, celltype, sub, ses, exp,
            )

        # Load analysis.h5
        key = f"analysis/{sub}/{ses}/analysis.h5"
        f = _download_h5(key)
        if f is None:
            log.warning("Missing: %s", key)
            if sync_f is not None:
                sync_f.close()
            continue

        try:
            if signal not in f:
                log.warning("Signal '%s' not in %s", signal, key)
                continue

            grp = f[signal]
            n_rois = grp["hd/all/mvl"].shape[0] if "hd/all/mvl" in grp else 0
            if n_rois == 0:
                continue

            for roi in range(n_rois):
                row: dict = {
                    "exp_id": exp_id,
                    "animal_id": animal_id,
                    "sub": sub,
                    "ses": ses,
                    "roi_idx": roi,
                    "celltype": celltype,
                    "sex": str(animal_info.get("sex", "")),
                    "hemisphere": str(animal_info.get("hemisphere", "")),
                    "inj_ap": float(animal_info.get("inj_ap", np.nan)),
                    "inj_ml": float(animal_info.get("inj_ml", np.nan)),
                    "inj_dv": float(animal_info.get("inj_dv", np.nan)),
                    "maze_session_num": int(exp["maze_session_num"]) if pd.notna(exp.get("maze_session_num")) else 0,
                    "signal": signal,
                }

                # ROI type (from sync.h5 if available)
                if sync_f is not None and "roi_types" in sync_f:
                    row["roi_type"] = int(sync_f["roi_types"][roi])
                else:
                    row["roi_type"] = 0

                # Activity metrics
                act = grp.get("activity")
                if act is not None:
                    for k in act:
                        row[k] = float(act[k][roi])

                # HD tuning
                for cond in ("all", "light", "dark"):
                    hd = grp.get(f"hd/{cond}")
                    if hd is not None:
                        for k in ("mvl", "preferred_direction", "tuning_width",
                                  "p_value", "significant"):
                            if k in hd:
                                val = hd[k][roi]
                                row[f"hd_{cond}_{k}"] = float(val) if k != "significant" else bool(val)

                # HD comparison
                hd_comp = grp.get("hd/comparison")
                if hd_comp is not None:
                    for k in ("correlation", "pd_shift", "mvl_ratio"):
                        if k in hd_comp:
                            row[f"hd_comp_{k}"] = float(hd_comp[k][roi])

                # Place coding
                for cond in ("all", "light", "dark"):
                    pl = grp.get(f"place/{cond}")
                    if pl is not None:
                        for k in ("spatial_info", "spatial_coherence", "sparsity",
                                  "p_value", "significant"):
                            if k in pl:
                                val = pl[k][roi]
                                row[f"place_{cond}_{k}"] = float(val) if k != "significant" else bool(val)

                # Place comparison
                pl_comp = grp.get("place/comparison")
                if pl_comp is not None:
                    if "correlation" in pl_comp:
                        row["place_comp_correlation"] = float(pl_comp["correlation"][roi])

                # Signal quality confounds
                if dff_data is not None and roi < dff_data.shape[0]:
                    trace = dff_data[roi]
                    baseline_std = float(np.nanstd(trace[trace < np.nanpercentile(trace, 25)]))
                    peak_dff = float(np.nanmax(trace))
                    row["snr"] = peak_dff / baseline_std if baseline_std > 0 else np.nan
                    row["peak_dff"] = peak_dff
                    row["baseline_std"] = baseline_std
                    # Bleaching: linear slope of baseline over time
                    n = len(trace)
                    if n > 100:
                        q10 = np.nanpercentile(trace, 10)
                        baseline_mask = trace < q10
                        if baseline_mask.sum() > 10:
                            x = np.where(baseline_mask)[0].astype(float)
                            y = trace[baseline_mask]
                            slope, _, _, _, _ = stats.linregress(x, y)
                            row["bleaching_slope"] = float(slope)

                cell_rows.append(row)

            if sync_f is not None:
                sync_f.close()
        finally:
            f.close()

    cell_df = pd.DataFrame(cell_rows)
    session_df = pd.DataFrame(session_rows)
    log.info(
        "Loaded %d cells from %d sessions, %d sessions with behaviour",
        len(cell_df),
        cell_df["exp_id"].nunique() if not cell_df.empty else 0,
        len(session_df),
    )
    return cell_df, session_df


def _extract_session_behav(
    sync_f: h5py.File,
    session_rows: list[dict],
    exp_id: str,
    animal_id: str,
    celltype: str,
    sub: str,
    ses: str,
    exp: pd.Series,
) -> None:
    """Extract session-level behavioural metrics from an open sync.h5 file."""
    has_speed = "speed_cm_s" in sync_f
    has_active = "active" in sync_f
    has_light = "light_on" in sync_f
    has_bad = "bad_behav" in sync_f

    if not has_speed and not has_active:
        return

    speed = sync_f["speed_cm_s"][:] if has_speed else None
    active = sync_f["active"][:].astype(bool) if has_active else None
    light_on = sync_f["light_on"][:].astype(bool) if has_light else None
    bad_behav = sync_f["bad_behav"][:].astype(bool) if has_bad else None

    # Build valid mask: not bad_behav
    valid = ~bad_behav if bad_behav is not None else np.ones(len(speed if speed is not None else active), dtype=bool)

    srow: dict = {
        "exp_id": exp_id,
        "animal_id": animal_id,
        "celltype": celltype,
        "sub": sub,
        "ses": ses,
    }

    # Overall metrics
    if active is not None:
        srow["frac_active"] = float(np.mean(active[valid])) if valid.sum() > 0 else np.nan

    if speed is not None and active is not None:
        active_valid = active & valid
        if active_valid.sum() > 0:
            srow["mean_speed"] = float(np.mean(speed[active_valid]))
        else:
            srow["mean_speed"] = np.nan
    elif speed is not None:
        srow["mean_speed"] = float(np.mean(speed[valid])) if valid.sum() > 0 else np.nan

    # Light vs dark behavioural metrics
    if light_on is not None:
        light_valid = valid & light_on
        dark_valid = valid & ~light_on

        if active is not None:
            srow["frac_active_light"] = float(np.mean(active[light_valid])) if light_valid.sum() > 0 else np.nan
            srow["frac_active_dark"] = float(np.mean(active[dark_valid])) if dark_valid.sum() > 0 else np.nan

        if speed is not None:
            # Mean speed per condition over valid frames with finite speed.
            # Do NOT gate on the `active` mask: if that mask is empty/misaligned
            # the intersection can be empty, yielding all-NaN columns and a
            # spurious "0 valid pairs" error in the within-session speed test.
            # Speed itself carries the movement information.
            sp = np.asarray(speed, dtype=float)
            light_sp = light_valid & np.isfinite(sp)
            dark_sp = dark_valid & np.isfinite(sp)
            srow["mean_speed_light"] = float(np.mean(sp[light_sp])) if light_sp.sum() > 0 else np.nan
            srow["mean_speed_dark"] = float(np.mean(sp[dark_sp])) if dark_sp.sum() > 0 else np.nan

    session_rows.append(srow)


# ---------------------------------------------------------------------------
# Hypothesis definitions
# ---------------------------------------------------------------------------


def define_hypotheses() -> list[dict]:
    """Return list of hypothesis dicts with test specifications."""
    h = []

    # --- H1: Activity (2x2 movement x light) ---
    h.append({"id": "H1.1", "name": "Movement increases activity",
              "type": "within_cell",
              "col_a": "moving_light_event_rate", "col_b": "stationary_light_event_rate"})
    h.append({"id": "H1.2", "name": "Light increases activity",
              "type": "within_cell",
              "col_a": "moving_light_event_rate", "col_b": "moving_dark_event_rate"})
    h.append({"id": "H1.3", "name": "Movement x light interaction",
              "type": "within_cell_interaction",
              "cols": ["moving_light_event_rate", "stationary_light_event_rate",
                       "moving_dark_event_rate", "stationary_dark_event_rate"]})
    h.append({"id": "H1.4", "name": "Baseline activity differs",
              "type": "between_group", "metric": "moving_light_event_rate"})
    h.append({"id": "H1.5", "name": "Movement modulation differs",
              "type": "between_group", "metric": "movement_modulation"})
    h.append({"id": "H1.6", "name": "Movement x light interaction differs",
              "type": "between_group_interaction",
              "cols": ["moving_light_event_rate", "stationary_light_event_rate",
                       "moving_dark_event_rate", "stationary_dark_event_rate"]})

    # --- H2: HD tuning ---
    h.append({"id": "H2.1", "name": "RSP has HD cells",
              "type": "descriptive", "metric": "hd_all_significant"})
    h.append({"id": "H2.2", "name": "HD strength differs",
              "type": "between_group", "metric": "hd_all_mvl"})
    h.append({"id": "H2.3", "name": "Tuning width differs",
              "type": "between_group", "metric": "hd_all_tuning_width"})

    # --- H3: Visual cue dependence ---
    # Neutral name: two-sided light-vs-dark test; direction is reported from the
    # sign of (light - dark). On clean FISSA data MVL is HIGHER in dark.
    h.append({"id": "H3.1", "name": "HD tuning (MVL): light vs dark",
              "type": "within_cell",
              "col_a": "hd_light_mvl", "col_b": "hd_dark_mvl"})
    h.append({"id": "H3.2", "name": "PD drifts in darkness",
              "type": "within_cell_onesample", "metric": "hd_comp_pd_shift"})
    h.append({"id": "H3.4", "name": "Visual cue dependence differs (KEY)",
              "type": "between_group", "metric": "hd_comp_mvl_ratio"})
    h.append({"id": "H3.4b", "name": "PD shift differs between types",
              "type": "between_group", "metric": "hd_comp_pd_shift"})
    h.append({"id": "H3.5", "name": "Light modulation differs",
              "type": "between_group", "metric": "light_modulation"})

    # --- H4: Behavioural differences (session-level, from sync.h5) ---
    h.append({"id": "H4.1a", "name": "Movement speed differs between types",
              "type": "between_group_session", "metric": "mean_speed"})
    h.append({"id": "H4.1b", "name": "Fraction active differs between types",
              "type": "between_group_session", "metric": "frac_active"})
    h.append({"id": "H4.3", "name": "Speed changes in darkness (within session)",
              "type": "within_session",
              "col_a": "mean_speed_light", "col_b": "mean_speed_dark"})
    h.append({"id": "H4.4", "name": "Activity fraction changes in darkness (within session)",
              "type": "within_session",
              "col_a": "frac_active_light", "col_b": "frac_active_dark"})
    h.append({"id": "H4.5", "name": "Light-dark speed change differs between types",
              "type": "between_group_session", "metric": "speed_light_dark_diff"})

    # --- H5: Spatial coding ---
    h.append({"id": "H5.1", "name": "RSP has spatial info",
              "type": "descriptive", "metric": "place_all_significant"})
    h.append({"id": "H5.2", "name": "Spatial info differs",
              "type": "between_group", "metric": "place_all_spatial_info"})
    # Neutral name: direction reported from sign of (light - dark). On clean
    # data spatial information is HIGHER in dark (per-cell; see caveats in docs).
    h.append({"id": "H5.3", "name": "Spatial info: light vs dark",
              "type": "within_cell",
              "col_a": "place_light_spatial_info", "col_b": "place_dark_spatial_info"})

    return h


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_hypotheses(
    cell_df: pd.DataFrame,
    session_df: pd.DataFrame,
    n_perms: int = 10000,
) -> tuple[list[dict], list[dict]]:
    """Run all hypothesis tests, return (results, confound_checks)."""
    hypotheses = define_hypotheses()
    results: list[dict] = []
    confounds: list[dict] = []
    confound_cols = ["snr", "peak_dff", "baseline_std", "bleaching_slope"]

    # Filter to soma only for cell-level tests
    soma = cell_df[cell_df["roi_type"] == 0].copy()
    log.info("Testing on %d soma ROIs (%d total ROIs)", len(soma), len(cell_df))

    # Prepare session_df with derived metric for H4.5
    ses_df = session_df.copy()
    if "mean_speed_light" in ses_df.columns and "mean_speed_dark" in ses_df.columns:
        ses_df["speed_light_dark_diff"] = ses_df["mean_speed_light"] - ses_df["mean_speed_dark"]

    for hyp in hypotheses:
        hid = hyp["id"]
        hname = hyp["name"]
        htype = hyp["type"]
        log.info("  %s: %s (%s)", hid, hname, htype)

        try:
            if htype == "between_group":
                metric = hyp["metric"]
                r = run_between_group_test(soma, metric, n_perms=n_perms)
                r["hypothesis"] = hid
                r["hypothesis_name"] = hname
                results.append(r)

                # Confound check if either test is trending
                p_summary = r.get("summary_p_value", 1)
                p_perm = r.get("perm_p_value", 1)
                if p_summary < 0.1 or p_perm < 0.1:
                    available = [c for c in confound_cols if c in soma.columns]
                    if available:
                        checks = confound_check(soma, metric, available)
                        for c in checks:
                            c["hypothesis"] = hid
                        confounds.extend(checks)

            elif htype == "between_group_session":
                metric = hyp["metric"]
                if metric not in ses_df.columns:
                    log.warning("    Skipping %s: column '%s' not in session data", hid, metric)
                    continue
                # Collapse sessions to animal means, then between-group test
                r = run_between_group_test(ses_df, metric, n_perms=n_perms)
                r["hypothesis"] = hid
                r["hypothesis_name"] = hname
                r["level"] = "session"
                results.append(r)

            elif htype == "within_cell":
                col_a, col_b = hyp["col_a"], hyp["col_b"]
                if col_a not in soma.columns or col_b not in soma.columns:
                    log.warning("    Skipping %s: missing columns", hid)
                    continue
                r = within_cell_test(soma, col_a, col_b)
                r["test"] = "wilcoxon"
                r["hypothesis"] = hid
                r["hypothesis_name"] = hname
                r["col_a"], r["col_b"] = col_a, col_b
                results.append(r)

            elif htype == "within_session":
                col_a, col_b = hyp["col_a"], hyp["col_b"]
                if col_a not in ses_df.columns or col_b not in ses_df.columns:
                    log.warning("    Skipping %s: missing columns in session data", hid)
                    continue
                r = within_cell_test(ses_df, col_a, col_b)
                r["test"] = "wilcoxon"
                r["hypothesis"] = hid
                r["hypothesis_name"] = hname
                r["col_a"], r["col_b"] = col_a, col_b
                r["level"] = "session"
                results.append(r)

            elif htype == "within_cell_interaction":
                cols = hyp["cols"]
                missing = [c for c in cols if c not in soma.columns]
                if missing:
                    log.warning("    Skipping %s: missing columns %s", hid, missing)
                    continue
                contrast = interaction_contrast(soma[cols].dropna(), cols)
                nonzero = contrast[contrast != 0]
                if len(nonzero) > 5:
                    stat, p = stats.wilcoxon(nonzero, alternative="two-sided")
                    results.append({
                        "test": "wilcoxon_interaction", "hypothesis": hid,
                        "hypothesis_name": hname,
                        "statistic": float(stat), "p_value": float(p),
                        "n_cells": len(contrast),
                        "mean_contrast": float(contrast.mean()),
                    })

            elif htype == "between_group_interaction":
                cols = hyp["cols"]
                needed = cols + ["celltype", "animal_id"]
                missing = [c for c in needed if c not in soma.columns]
                if missing:
                    log.warning("    Skipping %s: missing columns %s", hid, missing)
                    continue
                sub = soma[needed].dropna().copy()
                if len(sub) > 5:
                    sub["interaction"] = interaction_contrast(sub, cols)
                    r = run_between_group_test(sub, "interaction", n_perms=n_perms)
                    r["hypothesis"] = hid
                    r["hypothesis_name"] = hname
                    results.append(r)

            elif htype == "within_cell_onesample":
                metric = hyp["metric"]
                if metric not in soma.columns:
                    log.warning("    Skipping %s: missing column '%s'", hid, metric)
                    continue
                vals = soma[metric].dropna().values
                nonzero = vals[vals != 0]
                if len(nonzero) > 5:
                    stat, p = stats.wilcoxon(nonzero, alternative="two-sided")
                    results.append({
                        "test": "wilcoxon_onesample", "hypothesis": hid,
                        "hypothesis_name": hname,
                        "statistic": float(stat), "p_value": float(p),
                        "n_cells": len(vals),
                        "mean": float(np.mean(vals)),
                        "median": float(np.median(vals)),
                    })

            elif htype == "descriptive":
                metric = hyp["metric"]
                if metric not in soma.columns:
                    log.warning("    Skipping %s: missing column '%s'", hid, metric)
                    continue
                vals = soma[metric].dropna()
                if metric.endswith("_significant"):
                    frac = vals.mean()
                    n_sig = int(vals.sum())
                    results.append({
                        "test": "descriptive", "hypothesis": hid,
                        "hypothesis_name": hname,
                        "metric": metric,
                        "fraction_significant": float(frac),
                        "n_significant": n_sig,
                        "n_total": len(vals),
                    })
                else:
                    results.append({
                        "test": "descriptive", "hypothesis": hid,
                        "hypothesis_name": hname,
                        "metric": metric,
                        "mean": float(vals.mean()),
                        "median": float(vals.median()),
                        "std": float(vals.std()),
                        "n": len(vals),
                    })

        except (ValueError, KeyError) as exc:
            log.warning("    %s failed: %s", hid, exc)
            results.append({
                "test": "error", "hypothesis": hid,
                "hypothesis_name": hname, "error": str(exc),
            })

    # --- FDR correction across all testable results ---
    testable = [r for r in results if r.get("test") not in ("descriptive", "error")]
    if testable:
        # Collect p-values: use perm_p_value for between-group, p_value for others
        p_vals = []
        for r in testable:
            p = r.get("perm_p_value", r.get("p_value", np.nan))
            p_vals.append(p)

        # Apply FDR via the mixed_stats helper
        # Build temporary dicts with p_value key for fdr_correct
        temp = [{"p_value": p} for p in p_vals]
        valid_temp = [t for t in temp if not np.isnan(t["p_value"])]
        if valid_temp:
            corrected = fdr_correct(valid_temp)
            # Map back to original results
            vi = 0
            for i, r in enumerate(testable):
                p = p_vals[i]
                if not np.isnan(p):
                    r["p_fdr"] = corrected[vi]["p_fdr"]
                    r["significant_fdr"] = corrected[vi]["significant_fdr"]
                    vi += 1

    return results, confounds


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    results: list[dict],
    confounds: list[dict],
    cell_df: pd.DataFrame,
    session_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write markdown + CSV report."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Save raw results
    pd.DataFrame(results).to_csv(output_path / "hypothesis_results.csv", index=False)
    if confounds:
        pd.DataFrame(confounds).to_csv(output_path / "confound_checks.csv", index=False)

    # Generate markdown
    lines: list[str] = []
    lines.append("# Hypothesis Test Report")
    lines.append("")
    lines.append(f"**Signal:** {cell_df['signal'].iloc[0] if 'signal' in cell_df.columns and len(cell_df) > 0 else 'dff'}")
    n_soma = len(cell_df[cell_df['roi_type'] == 0]) if len(cell_df) > 0 else 0
    lines.append(f"**Cells:** {n_soma} soma ROIs from "
                 f"{cell_df['animal_id'].nunique() if len(cell_df) > 0 else 0} animals "
                 f"({cell_df['exp_id'].nunique() if len(cell_df) > 0 else 0} sessions)")
    n_penk = cell_df[cell_df["celltype"] == "penk"]["animal_id"].nunique() if len(cell_df) > 0 else 0
    n_nonpenk = cell_df[cell_df["celltype"] == "nonpenk"]["animal_id"].nunique() if len(cell_df) > 0 else 0
    lines.append(f"**Groups:** {n_penk} Penk+ animals, {n_nonpenk} CamKII+ animals")
    lines.append(f"**Sessions with behaviour:** {len(session_df)}")
    lines.append("")

    # Count significant results (uncorrected and FDR-corrected)
    testable = [r for r in results if r.get("test") not in ("descriptive", "error")]
    sig_uncorrected = sum(
        1 for r in testable
        if r.get("perm_p_value", r.get("p_value", 1)) < 0.05
    )
    sig_fdr = sum(1 for r in testable if r.get("significant_fdr", False))
    lines.append(f"**Significant (uncorrected p < 0.05):** {sig_uncorrected}/{len(testable)}")
    lines.append(f"**Significant (FDR-corrected):** {sig_fdr}/{len(testable)}")
    lines.append("")

    # Organise by hypothesis
    hyp_ids = []
    seen = set()
    for r in results:
        hid = r.get("hypothesis", "?")
        if hid not in seen:
            hyp_ids.append(hid)
            seen.add(hid)

    for hid in hyp_ids:
        hyp_results = [r for r in results if r.get("hypothesis") == hid]
        if not hyp_results:
            continue

        hname = hyp_results[0].get("hypothesis_name", hid)
        lines.append(f"## {hid}: {hname}")
        lines.append("")

        for r in hyp_results:
            test = r.get("test", "?")
            level_tag = f" [{r['level']}]" if "level" in r else ""
            fdr_tag = ""
            if "p_fdr" in r and not np.isnan(r.get("p_fdr", np.nan)):
                fdr_tag = f" (FDR p = {r['p_fdr']:.4f}{'**' if r.get('significant_fdr') else ''})"

            if test == "error":
                lines.append(f"- **Error:** {r.get('error', 'unknown')}")

            elif "verdict" in r:
                # Combined between-group result from run_between_group_test
                p_summary = r.get("summary_p_value", np.nan)
                p_perm = r.get("perm_p_value", np.nan)
                verdict = r.get("verdict", "?")
                p_s = f"p = {p_summary:.4f}" if not np.isnan(p_summary) else "p = n/a"
                p_p = f"p = {p_perm:.4f}" if not np.isnan(p_perm) else "p = n/a"
                sig_s = " *" if p_summary < 0.05 else ""
                sig_p = " *" if p_perm < 0.05 else ""
                lines.append(
                    f"- **Animal-level Mann-Whitney:** {p_s}{sig_s} "
                    f"(Penk+ mean={r.get('summary_penk_mean', 0):.4f}, "
                    f"CamKII+ mean={r.get('summary_nonpenk_mean', 0):.4f}, "
                    f"N={r.get('summary_n_penk', 0)} vs {r.get('summary_n_nonpenk', 0)} animals, "
                    f"CLES={r.get('summary_effect_size', 0):.2f})"
                )
                lines.append(
                    f"- **Cluster permutation:** {p_p}{sig_p} "
                    f"(observed diff={r.get('perm_observed', 0):.4f}, "
                    f"null std={r.get('perm_null_std', 0):.4f})"
                )
                lines.append(f"- **Verdict:** {verdict}{level_tag}{fdr_tag}")

            elif test in ("wilcoxon", "wilcoxon_interaction"):
                p = r.get("p_value", np.nan)
                p_str = f"p = {p:.4f}" if not np.isnan(p) else "p = n/a"
                sig = " *" if p < 0.05 else ""
                mdiff = r.get("mean_diff", r.get("mean_contrast", 0))
                # Explicit direction: mean_diff = col_a - col_b. State which side
                # is higher so the label can never imply the wrong direction.
                col_a, col_b = r.get("col_a"), r.get("col_b")
                dir_str = ""
                if col_a and col_b and not np.isnan(mdiff) and mdiff != 0:
                    hi, lo = (col_a, col_b) if mdiff > 0 else (col_b, col_a)
                    dir_str = f" [{hi} > {lo}]"
                lines.append(
                    f"- **Wilcoxon signed-rank:** {p_str}{sig} "
                    f"(mean diff={mdiff:.4f}, "
                    f"N={r.get('n_cells', 0)}){dir_str}{level_tag}{fdr_tag}"
                )
            elif test == "wilcoxon_onesample":
                p = r.get("p_value", np.nan)
                p_str = f"p = {p:.4f}" if not np.isnan(p) else "p = n/a"
                sig = " *" if p < 0.05 else ""
                lines.append(
                    f"- **Wilcoxon (vs 0):** {p_str}{sig} "
                    f"(mean={r.get('mean', 0):.4f}, median={r.get('median', 0):.4f}, "
                    f"N={r.get('n_cells', 0)}){fdr_tag}"
                )
            elif test == "descriptive":
                if "fraction_significant" in r:
                    lines.append(
                        f"- **Descriptive:** {r['n_significant']}/{r['n_total']} "
                        f"({r['fraction_significant']:.1%}) cells significant"
                    )
                else:
                    lines.append(
                        f"- **Descriptive:** mean={r.get('mean', 0):.4f}, "
                        f"median={r.get('median', 0):.4f}, N={r.get('n', 0)}"
                    )

        # Confound checks for this hypothesis
        hyp_confounds = [c for c in confounds if c.get("hypothesis") == hid]
        flagged = [c for c in hyp_confounds if c.get("flagged")]
        if flagged:
            lines.append("")
            lines.append("**Confound warnings:**")
            for c in flagged:
                lines.append(
                    f"- {c['confound']}: Spearman rho = {c['rho']:.3f} "
                    f"(p = {c['p_value']:.4f}, N={c.get('n', '?')})"
                )

        lines.append("")

    # Write report
    report_text = "\n".join(lines)
    report_path = output_path / "hypothesis_report.md"
    report_path.write_text(report_text)
    log.info("Report written to %s", report_path)
    print(report_text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Test hypotheses from docs/hypotheses.md")
    parser.add_argument("--signal", default="dff", choices=["dff", "events", "deconv"],
                        help="Signal type to analyse (default: dff)")
    parser.add_argument("--n-perms", type=int, default=10000,
                        help="Number of permutations for cluster test (default: 10000)")
    parser.add_argument("--output", type=Path,
                        default=Path("results/hypotheses"),
                        help="Output directory for report")
    parser.add_argument("--soma-only", action="store_true", default=True,
                        help="Only include soma ROIs (default: True)")
    args = parser.parse_args()

    animals, exps = load_metadata()
    cell_df, session_df = load_all_analysis(animals, exps, signal=args.signal)

    if cell_df.empty:
        log.error("No data loaded. Check S3 access and metadata.")
        sys.exit(1)

    results, confounds = run_hypotheses(cell_df, session_df, n_perms=args.n_perms)
    generate_report(results, confounds, cell_df, session_df, args.output)


if __name__ == "__main__":
    main()
