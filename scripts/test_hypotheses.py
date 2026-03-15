#!/usr/bin/env python3
"""Test all hypotheses from docs/hypotheses.md and generate a report.

Loads analysis.h5 + sync.h5 + metadata, pools cells with animal/session
metadata, runs non-parametric tests (animal-level Mann-Whitney, cluster
permutation), checks signal quality confounds, and outputs a structured
markdown + CSV report.

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
) -> pd.DataFrame:
    """Load all analysis.h5 files and build a per-cell DataFrame.

    Returns a DataFrame with one row per (session, roi) containing all
    analysis metrics plus metadata (animal_id, celltype, sex, etc.).
    """
    s3 = _s3()
    rows: list[dict] = []

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

        # Load analysis.h5
        key = f"analysis/{sub}/{ses}/analysis.h5"
        f = _download_h5(key)
        if f is None:
            log.warning("Missing: %s", key)
            continue

        try:
            if signal not in f:
                log.warning("Signal '%s' not in %s", signal, key)
                continue

            grp = f[signal]
            n_rois = grp["hd/all/mvl"].shape[0] if "hd/all/mvl" in grp else 0
            if n_rois == 0:
                continue

            # Load sync.h5 for SNR/confound metrics
            sync_key = f"sync/{sub}/{ses}/sync.h5"
            sync_f = _download_h5(sync_key)
            dff_data = None
            if sync_f is not None and "dff" in sync_f:
                dff_data = sync_f["dff"][:]

            for roi in range(n_rois):
                row: dict = {
                    "exp_id": exp_id,
                    "animal_id": animal_id,
                    "sub": sub,
                    "ses": ses,
                    "roi_idx": roi,
                    "celltype": str(animal_info.get("celltype", "")),
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

                rows.append(row)

            if sync_f is not None:
                sync_f.close()
        finally:
            f.close()

    df = pd.DataFrame(rows)
    log.info("Loaded %d cells from %d sessions", len(df), df["exp_id"].nunique())
    return df


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def animal_summary_test(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str = "celltype",
    animal_col: str = "animal_id",
) -> dict:
    """Collapse to animal means, then Mann-Whitney U."""
    sub = df[[animal_col, group_col, metric_col]].dropna()
    if sub.empty:
        return {"test": "animal_summary", "metric": metric_col,
                "statistic": np.nan, "p_value": np.nan,
                "n_penk": 0, "n_nonpenk": 0, "effect_size": np.nan}

    animal_means = sub.groupby([animal_col, group_col])[metric_col].mean().reset_index()
    penk = animal_means.loc[animal_means[group_col] == "penk", metric_col].values
    nonpenk = animal_means.loc[animal_means[group_col] == "nonpenk", metric_col].values

    if len(penk) < 2 or len(nonpenk) < 2:
        return {"test": "animal_summary", "metric": metric_col,
                "statistic": np.nan, "p_value": np.nan,
                "n_penk": len(penk), "n_nonpenk": len(nonpenk),
                "effect_size": np.nan}

    stat, p = stats.mannwhitneyu(penk, nonpenk, alternative="two-sided")
    # Common language effect size
    cles = stat / (len(penk) * len(nonpenk))
    return {
        "test": "animal_summary",
        "metric": metric_col,
        "statistic": float(stat),
        "p_value": float(p),
        "n_penk": len(penk),
        "n_nonpenk": len(nonpenk),
        "penk_mean": float(np.mean(penk)),
        "nonpenk_mean": float(np.mean(nonpenk)),
        "effect_size": float(cles),
    }


def cluster_permutation_test(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str = "celltype",
    cluster_col: str = "animal_id",
    n_perms: int = 10000,
) -> dict:
    """Permutation test shuffling group labels at the cluster (animal) level."""
    sub = df[[cluster_col, group_col, metric_col]].dropna()
    if sub.empty:
        return {"test": "cluster_perm", "metric": metric_col,
                "p_value": np.nan, "observed": np.nan}

    penk_vals = sub.loc[sub[group_col] == "penk", metric_col].values
    nonpenk_vals = sub.loc[sub[group_col] == "nonpenk", metric_col].values

    if len(penk_vals) < 1 or len(nonpenk_vals) < 1:
        return {"test": "cluster_perm", "metric": metric_col,
                "p_value": np.nan, "observed": np.nan}

    observed = float(np.mean(penk_vals) - np.mean(nonpenk_vals))

    # Get cluster-level group assignments
    cluster_groups = sub.groupby(cluster_col)[group_col].first()
    cluster_ids = cluster_groups.index.values
    n_nonpenk_clusters = (cluster_groups == "nonpenk").sum()

    rng = np.random.default_rng(42)
    null_stats = np.empty(n_perms)

    for i in range(n_perms):
        perm_nonpenk = rng.choice(cluster_ids, size=n_nonpenk_clusters, replace=False)
        perm_penk = np.setdiff1d(cluster_ids, perm_nonpenk)

        a = sub.loc[sub[cluster_col].isin(perm_penk), metric_col].values
        b = sub.loc[sub[cluster_col].isin(perm_nonpenk), metric_col].values

        null_stats[i] = np.mean(a) - np.mean(b) if len(a) > 0 and len(b) > 0 else 0.0

    p_value = float((np.sum(np.abs(null_stats) >= np.abs(observed)) + 1) / (n_perms + 1))

    return {
        "test": "cluster_perm",
        "metric": metric_col,
        "observed": observed,
        "p_value": p_value,
        "null_mean": float(np.mean(null_stats)),
        "null_std": float(np.std(null_stats)),
    }


def within_cell_test(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    name: str,
) -> dict:
    """Paired Wilcoxon signed-rank test across all cells."""
    sub = df[[col_a, col_b]].dropna()
    if len(sub) < 5:
        return {"test": "wilcoxon", "hypothesis": name,
                "p_value": np.nan, "n_cells": len(sub)}

    diff = sub[col_a].values - sub[col_b].values
    diff = diff[diff != 0]  # Wilcoxon excludes ties at 0
    if len(diff) < 5:
        return {"test": "wilcoxon", "hypothesis": name,
                "p_value": np.nan, "n_cells": len(sub)}

    stat, p = stats.wilcoxon(diff, alternative="two-sided")
    return {
        "test": "wilcoxon",
        "hypothesis": name,
        "statistic": float(stat),
        "p_value": float(p),
        "n_cells": len(sub),
        "mean_diff": float(np.mean(sub[col_a].values - sub[col_b].values)),
        "median_diff": float(np.median(sub[col_a].values - sub[col_b].values)),
    }


def confound_check(
    df: pd.DataFrame,
    metric_col: str,
    confound_cols: list[str],
) -> list[dict]:
    """Spearman correlation between metric and each confound."""
    results = []
    for conf in confound_cols:
        sub = df[[metric_col, conf]].dropna()
        if len(sub) < 5:
            results.append({"metric": metric_col, "confound": conf,
                            "rho": np.nan, "p_value": np.nan, "n": len(sub)})
            continue
        rho, p = stats.spearmanr(sub[metric_col], sub[conf])
        results.append({
            "metric": metric_col,
            "confound": conf,
            "rho": float(rho),
            "p_value": float(p),
            "n": len(sub),
            "flagged": abs(rho) > 0.3,
        })
    return results


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
    h.append({"id": "H3.1", "name": "Darkness degrades tuning",
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

    # --- H5: Spatial coding ---
    h.append({"id": "H5.1", "name": "RSP has spatial info",
              "type": "descriptive", "metric": "place_all_significant"})
    h.append({"id": "H5.2", "name": "Spatial info differs",
              "type": "between_group", "metric": "place_all_spatial_info"})
    h.append({"id": "H5.3", "name": "Spatial info drops in dark",
              "type": "within_cell",
              "col_a": "place_light_spatial_info", "col_b": "place_dark_spatial_info"})

    return h


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_hypotheses(
    df: pd.DataFrame,
    n_perms: int = 10000,
) -> tuple[list[dict], list[dict]]:
    """Run all hypothesis tests, return (results, confound_checks)."""
    hypotheses = define_hypotheses()
    results: list[dict] = []
    confounds: list[dict] = []
    confound_cols = ["snr", "peak_dff", "baseline_std", "bleaching_slope"]

    # Filter to soma only
    soma = df[df["roi_type"] == 0].copy()
    log.info("Testing on %d soma ROIs (%d total ROIs)", len(soma), len(df))

    for hyp in hypotheses:
        hid = hyp["id"]
        hname = hyp["name"]
        htype = hyp["type"]
        log.info("  %s: %s (%s)", hid, hname, htype)

        if htype == "between_group":
            metric = hyp["metric"]
            r1 = animal_summary_test(soma, metric)
            r1["hypothesis"] = hid
            r1["hypothesis_name"] = hname
            results.append(r1)

            r2 = cluster_permutation_test(soma, metric, n_perms=n_perms)
            r2["hypothesis"] = hid
            r2["hypothesis_name"] = hname
            results.append(r2)

            # Confound check if significant
            if r1.get("p_value", 1) < 0.1 or r2.get("p_value", 1) < 0.1:
                available = [c for c in confound_cols if c in soma.columns]
                checks = confound_check(soma, metric, available)
                for c in checks:
                    c["hypothesis"] = hid
                confounds.extend(checks)

        elif htype == "within_cell":
            r = within_cell_test(soma, hyp["col_a"], hyp["col_b"], hid)
            r["hypothesis"] = hid
            r["hypothesis_name"] = hname
            results.append(r)

        elif htype == "within_cell_interaction":
            # Compute interaction contrast per cell
            cols = hyp["cols"]
            sub = soma[cols].dropna()
            if len(sub) > 5:
                contrast = (sub[cols[0]] - sub[cols[1]]) - (sub[cols[2]] - sub[cols[3]])
                nonzero = contrast[contrast != 0]
                if len(nonzero) > 5:
                    stat, p = stats.wilcoxon(nonzero, alternative="two-sided")
                    results.append({
                        "test": "wilcoxon_interaction", "hypothesis": hid,
                        "hypothesis_name": hname,
                        "statistic": float(stat), "p_value": float(p),
                        "n_cells": len(sub),
                        "mean_contrast": float(contrast.mean()),
                    })

        elif htype == "between_group_interaction":
            # Compute interaction contrast, then compare between groups
            cols = hyp["cols"]
            sub = soma[cols + ["celltype", "animal_id"]].dropna()
            if len(sub) > 5:
                sub = sub.copy()
                sub["interaction"] = (sub[cols[0]] - sub[cols[1]]) - (sub[cols[2]] - sub[cols[3]])
                r1 = animal_summary_test(sub, "interaction")
                r1["hypothesis"] = hid
                r1["hypothesis_name"] = hname
                results.append(r1)
                r2 = cluster_permutation_test(sub, "interaction", n_perms=n_perms)
                r2["hypothesis"] = hid
                r2["hypothesis_name"] = hname
                results.append(r2)

        elif htype == "within_cell_onesample":
            metric = hyp["metric"]
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

    return results, confounds


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    results: list[dict],
    confounds: list[dict],
    df: pd.DataFrame,
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
    lines.append(f"**Signal:** {df['signal'].iloc[0] if 'signal' in df.columns else 'dff'}")
    lines.append(f"**Cells:** {len(df[df['roi_type'] == 0])} soma ROIs from "
                 f"{df['animal_id'].nunique()} animals ({df['exp_id'].nunique()} sessions)")
    n_penk = df[df["celltype"] == "penk"]["animal_id"].nunique()
    n_nonpenk = df[df["celltype"] == "nonpenk"]["animal_id"].nunique()
    lines.append(f"**Groups:** {n_penk} Penk+ animals, {n_nonpenk} Penk⁻CamKII+ animals")
    lines.append("")

    # Count significant results
    sig_results = [r for r in results if r.get("p_value", 1) < 0.05
                   and r.get("test") not in ("descriptive",)]
    lines.append(f"**Significant results (p < 0.05):** {len(sig_results)}")
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
            p = r.get("p_value", np.nan)
            p_str = f"p = {p:.4f}" if not np.isnan(p) else "p = n/a"
            sig = " **\\***" if p < 0.05 else ""

            if test == "animal_summary":
                lines.append(
                    f"- **Animal-level Mann-Whitney U:** {p_str}{sig} "
                    f"(Penk+ mean={r.get('penk_mean', 0):.4f}, "
                    f"Penk⁻CamKII+ mean={r.get('nonpenk_mean', 0):.4f}, "
                    f"N={r.get('n_penk', 0)} vs {r.get('n_nonpenk', 0)} animals, "
                    f"CLES={r.get('effect_size', 0):.2f})"
                )
            elif test == "cluster_perm":
                lines.append(
                    f"- **Cluster permutation:** {p_str}{sig} "
                    f"(observed diff={r.get('observed', 0):.4f}, "
                    f"null std={r.get('null_std', 0):.4f})"
                )
            elif test in ("wilcoxon", "wilcoxon_interaction"):
                lines.append(
                    f"- **Wilcoxon signed-rank:** {p_str}{sig} "
                    f"(mean diff={r.get('mean_diff', r.get('mean_contrast', 0)):.4f}, "
                    f"N={r.get('n_cells', 0)} cells)"
                )
            elif test == "wilcoxon_onesample":
                lines.append(
                    f"- **Wilcoxon (vs 0):** {p_str}{sig} "
                    f"(mean={r.get('mean', 0):.4f}, median={r.get('median', 0):.4f}, "
                    f"N={r.get('n_cells', 0)} cells)"
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
                    f"- {c['confound']}: Spearman ρ = {c['rho']:.3f} "
                    f"(p = {c['p_value']:.4f}, N={c['n']})"
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
    df = load_all_analysis(animals, exps, signal=args.signal)

    if df.empty:
        log.error("No data loaded. Check S3 access and metadata.")
        sys.exit(1)

    results, confounds = run_hypotheses(df, n_perms=args.n_perms)
    generate_report(results, confounds, df, args.output)


if __name__ == "__main__":
    main()
