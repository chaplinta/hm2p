"""Changelog page — tracks features, bug fixes, and updates with timestamps."""

from __future__ import annotations

import streamlit as st

st.title("Changelog")
st.caption("Features, bug fixes, and updates to the hm2p dashboard and pipeline.")

# Changelog entries — newest first
# Each entry: (date, time, category, description)
CHANGELOG = [
    ("2026-03-08", "13:00", "feature", "Add Signal Quality page — photobleaching trend analysis, noise floor estimation, temporal autocorrelation with half-decay time, per-ROI quality grading (A-D), SNR vs skewness scatter, CSV export"),
    ("2026-03-08", "12:00", "feature", "Add Maze Analysis module and page — rose maze topology graph (23 cells, 8 T-junctions, 6 dead ends), exploration efficiency, turn bias, path efficiency, monotonic path detection, behavioural mode segmentation, sequence entropy. Inspired by Rosenberg et al. (2021) eLife"),
    ("2026-03-08", "11:00", "feature", "Add Publication Statistics page — ready-to-report methods text, Penk vs non-Penk comparisons (SNR, event rate, active fraction, max dF/F), per-session table, CSV export"),
    ("2026-03-08", "10:30", "feature", "Add Cross-Session Light Analysis — aggregate light modulation index across all sessions, Penk vs non-Penk comparison with Mann-Whitney, KS test, Wilcoxon"),
    ("2026-03-08", "10:00", "feature", "Add Light/Dark Analysis page — compare activity between light-on and light-off epochs using timestamps.h5, modulation index, per-cycle analysis, population response"),
    ("2026-03-08", "09:30", "feature", "Add Home page — project dashboard with pipeline status overview, quick navigation links, recent changelog"),
    ("2026-03-08", "09:00", "feature", "Add downstream pipeline orchestrator (run_downstream_pipeline.py) — auto-runs Stages 3/5/6 when DLC completes, with --watch mode"),
    ("2026-03-08", "08:30", "feature", "Add Animal Summary page — per-animal ROI counts, quality metrics, Penk vs non-Penk stats, session list per animal"),
    ("2026-03-08", "08:15", "fix", "Remove duplicate st.set_page_config from analysis page (fixes StreamlitAPIException in multipage mode)"),
    ("2026-03-08", "08:00", "feature", "Add Trace Comparison page — overlay/stack/normalize multiple ROI traces, auto-find most correlated pair, cross-correlation lag analysis"),
    ("2026-03-08", "07:30", "feature", "Add QC Report page — automated quality grading (A-D), per-ROI quality checks, metadata display"),
    ("2026-03-08", "07:15", "improvement", "Reorganize navigation into sections: Overview, Pipeline, Explore, Analysis, System"),
    ("2026-03-08", "07:10", "fix", "Fix flaky hypothesis test (SI non-negativity tolerance 1e-10 -> 1e-9 for denormalized floats)"),
    ("2026-03-08", "07:00", "feature", "Add Correlations & Ensembles page — pairwise correlation matrix with hierarchical clustering, PCA dimensionality, population co-activation analysis"),
    ("2026-03-08", "06:30", "feature", "Add Event Browser page — browse individual calcium transients, waveform gallery with mean, event statistics (IEI, duration, peak), population raster"),
    ("2026-03-08", "06:00", "feature", "Add Batch Overview page — at-a-glance quality metrics for all sessions, ROI counts, SNR bars, event rate comparisons, CSV export"),
    ("2026-03-08", "05:30", "feature", "Add ROI Gallery page — grid view of all ROIs with mini traces, event overlays, sortable by SNR/event rate, quality filtering"),
    ("2026-03-08", "05:15", "feature", "Add Session Timeline page — temporal overview with light cycles, speed, HD, population dF/F heatmap, event rate, per-ROI browser"),
    ("2026-03-08", "05:00", "improvement", "Update analysis-plan.md with multi-signal pipeline documentation and analysis.h5 schema"),
    ("2026-03-08", "04:30", "feature", "Add Population Overview page — aggregate SNR, event rates, skewness distributions across all 391 ROIs, Penk vs non-Penk stats, quality filtering, CSV export"),
    ("2026-03-08", "04:00", "feature", "Add Data Explorer page — unified drill-down with calcium traces, event overlays, light cycle overlay, timestamps, pose trajectories, S3 file browser"),
    ("2026-03-08", "03:30", "feature", "Add Changelog page to track all features and fixes with timestamps"),
    ("2026-03-08", "03:25", "feature", "Add Cross-Session Comparison page with Penk vs non-Penk analysis, Mann-Whitney U tests"),
    ("2026-03-08", "03:20", "feature", "Add DLC Pose monitoring page with live progress, EC2 status, trajectory viewer, likelihood QC"),
    ("2026-03-08", "03:15", "feature", "Add Calcium data viewer with interactive traces, event overlays, heatmaps, correlation matrix, cell drill-down"),
    ("2026-03-08", "03:10", "feature", "Expand Analysis page with multi-signal comparison (dF/F, deconv, events), MVL cross-signal scatter, significance agreement (Jaccard)"),
    ("2026-03-08", "03:05", "feature", "Add Population Summary tab with per-ROI metrics across all signal types, CSV export"),
    ("2026-03-08", "03:00", "feature", "Add Stage 6 analysis script (run_stage6_analysis.py) — runs analysis with all signal types and saves to analysis.h5 on S3"),
    ("2026-03-08", "02:55", "feature", "Add analysis/save.py — HDF5 persistence for multi-signal analysis results"),
    ("2026-03-08", "02:50", "fix", "Update pipeline page with accurate stage statuses (Stage 4 complete at 26/26, Stage 6 ready)"),
    ("2026-03-08", "02:45", "feature", "Add analysis stage to S3 tracking (STAGE_PREFIXES now has 6 stages)"),
    ("2026-03-07", "22:33", "fix", "Fix Suite2p off-by-one: trim frame_times to match dF/F columns in sync"),
    ("2026-03-07", "20:00", "feature", "Expand test coverage to 626 tests (92% coverage)"),
    ("2026-03-07", "18:00", "fix", "Fix kinematics tests for movement 0.14 API (load_poses.from_file)"),
    ("2026-03-07", "16:00", "feature", "Add pipeline navigation and sync status to sessions page"),
    ("2026-03-07", "15:00", "fix", "Fix movement API (0.14) and kinematics stats keys"),
]

CATEGORY_COLORS = {
    "feature": "green",
    "fix": "orange",
    "improvement": "blue",
    "breaking": "red",
}

CATEGORY_ICONS = {
    "feature": ":sparkles:",
    "fix": ":wrench:",
    "improvement": ":chart_with_upwards_trend:",
    "breaking": ":warning:",
}

# Filters
categories = sorted(set(c[2] for c in CHANGELOG))
selected_cats = st.multiselect("Filter by category", categories, default=categories)

# Group by date
from collections import defaultdict

by_date: dict[str, list] = defaultdict(list)
for date, time, cat, desc in CHANGELOG:
    if cat in selected_cats:
        by_date[date].append((time, cat, desc))

for date in sorted(by_date.keys(), reverse=True):
    st.subheader(date)
    for time, cat, desc in sorted(by_date[date], reverse=True):
        color = CATEGORY_COLORS.get(cat, "gray")
        icon = CATEGORY_ICONS.get(cat, "")
        st.markdown(f"**{time}** — :{color}[{cat}] — {desc}")

st.markdown("---")
st.caption(
    "Entries are added automatically when features are built or bugs are fixed. "
    "Newest entries appear at the top."
)
