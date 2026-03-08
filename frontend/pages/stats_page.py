"""Publication Statistics — ready-to-report summary statistics for the dataset."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    download_s3_bytes,
    load_animals,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.stats")

st.title("Publication Statistics")
st.caption("Ready-to-report summary statistics for methods and results sections.")


@st.cache_data(ttl=600)
def compute_dataset_stats() -> dict:
    """Compute comprehensive dataset statistics."""
    import h5py

    experiments = load_experiments()
    animals = load_animals()
    animal_map = {a["animal_id"]: a for a in animals}

    # Session-level
    session_stats = []
    roi_stats = []

    unique_animals = set()
    penk_animals = set()
    nonpenk_animals = set()

    for exp in experiments:
        exp_id = exp["exp_id"]
        animal_id = exp_id.split("_")[-1]
        animal = animal_map.get(animal_id, {})
        celltype = animal.get("celltype", "?")
        sub, ses = parse_session_id(exp_id)

        unique_animals.add(animal_id)
        if celltype == "penk":
            penk_animals.add(animal_id)
        elif celltype == "nonpenk":
            nonpenk_animals.add(animal_id)

        data = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
        if data is None:
            continue

        try:
            f = h5py.File(io.BytesIO(data), "r")
            dff = f["dff"][:]
            n_rois, n_frames = dff.shape
            fps = float(f.attrs.get("fps_imaging", 9.8))
            duration_s = n_frames / fps

            session_stats.append({
                "exp_id": exp_id,
                "animal_id": animal_id,
                "celltype": celltype,
                "n_rois": n_rois,
                "n_frames": n_frames,
                "fps": fps,
                "duration_s": duration_s,
                "duration_min": duration_s / 60,
            })

            for roi in range(n_rois):
                trace = dff[roi]
                baseline_std = np.std(trace[trace < np.percentile(trace, 50)])
                peak = np.percentile(trace, 95)
                snr = peak / baseline_std if baseline_std > 0 else 0

                n_events = 0
                active_frac = 0.0
                if "event_masks" in f:
                    em = f["event_masks"][roi].astype(bool)
                    onsets = np.flatnonzero(em[1:] & ~em[:-1])
                    n_events = len(onsets) + (1 if em[0] else 0)
                    active_frac = float(em.mean())

                roi_stats.append({
                    "celltype": celltype,
                    "snr": snr,
                    "n_events": n_events,
                    "event_rate": n_events / (duration_s / 60),
                    "active_frac": active_frac,
                    "mean_dff": float(np.nanmean(trace)),
                    "max_dff": float(np.nanmax(trace)),
                })
            f.close()
        except Exception:
            continue

    ses_df = pd.DataFrame(session_stats)
    roi_df = pd.DataFrame(roi_stats)

    return {
        "sessions": ses_df,
        "rois": roi_df,
        "n_animals": len(unique_animals),
        "n_penk_animals": len(penk_animals),
        "n_nonpenk_animals": len(nonpenk_animals),
    }


with st.spinner("Computing dataset statistics..."):
    stats = compute_dataset_stats()

ses_df = stats["sessions"]
roi_df = stats["rois"]

if ses_df.empty:
    st.warning("No data available.")
    st.stop()

# --- Methods section text ---
st.subheader("Methods — Dataset Description")

n_sessions = len(ses_df)
n_animals = stats["n_animals"]
n_penk = stats["n_penk_animals"]
n_nonpenk = stats["n_nonpenk_animals"]
total_rois = len(roi_df)
penk_rois = len(roi_df[roi_df["celltype"] == "penk"])
nonpenk_rois = len(roi_df[roi_df["celltype"] == "nonpenk"])
mean_dur = ses_df["duration_min"].mean()
std_dur = ses_df["duration_min"].std()
mean_fps = ses_df["fps"].mean()
mean_rois = ses_df["n_rois"].mean()
std_rois = ses_df["n_rois"].std()

methods_text = f"""Two-photon calcium imaging was performed in {n_sessions} sessions from
{n_animals} mice ({n_penk} Penk-Cre, {n_nonpenk} CamKII non-Penk). Sessions lasted
{mean_dur:.1f} +/- {std_dur:.1f} min (mean +/- SD) at {mean_fps:.1f} Hz imaging rate.
A total of {total_rois} ROIs were identified ({penk_rois} Penk+, {nonpenk_rois} non-Penk),
with {mean_rois:.1f} +/- {std_rois:.1f} ROIs per session."""

st.markdown(f">{methods_text}")
st.caption("Copy-paste ready for methods section.")

# --- Results section stats ---
st.subheader("Results — Summary Statistics")

# SNR
penk_snr = roi_df[roi_df["celltype"] == "penk"]["snr"]
nonpenk_snr = roi_df[roi_df["celltype"] == "nonpenk"]["snr"]

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Penk+ ROIs:**")
    st.markdown(f"- N = {penk_rois}")
    st.markdown(f"- SNR: {penk_snr.mean():.2f} +/- {penk_snr.std():.2f} (mean +/- SD)")
    st.markdown(f"- SNR median: {penk_snr.median():.2f} [IQR: {penk_snr.quantile(0.25):.2f}-{penk_snr.quantile(0.75):.2f}]")
    if len(roi_df[roi_df["celltype"] == "penk"]["event_rate"]) > 0:
        er = roi_df[roi_df["celltype"] == "penk"]["event_rate"]
        st.markdown(f"- Event rate: {er.median():.1f} [IQR: {er.quantile(0.25):.1f}-{er.quantile(0.75):.1f}] events/min")

with col2:
    st.markdown("**Non-Penk ROIs:**")
    st.markdown(f"- N = {nonpenk_rois}")
    st.markdown(f"- SNR: {nonpenk_snr.mean():.2f} +/- {nonpenk_snr.std():.2f} (mean +/- SD)")
    st.markdown(f"- SNR median: {nonpenk_snr.median():.2f} [IQR: {nonpenk_snr.quantile(0.25):.2f}-{nonpenk_snr.quantile(0.75):.2f}]")
    if len(roi_df[roi_df["celltype"] == "nonpenk"]["event_rate"]) > 0:
        er = roi_df[roi_df["celltype"] == "nonpenk"]["event_rate"]
        st.markdown(f"- Event rate: {er.median():.1f} [IQR: {er.quantile(0.25):.1f}-{er.quantile(0.75):.1f}] events/min")

# Statistical tests
st.subheader("Statistical Comparisons")

from scipy.stats import mannwhitneyu

tests_results = []

if len(penk_snr) > 0 and len(nonpenk_snr) > 0:
    stat, pval = mannwhitneyu(penk_snr, nonpenk_snr, alternative="two-sided")
    tests_results.append({
        "Test": "SNR: Penk vs Non-Penk",
        "Statistic": f"U = {stat:.0f}",
        "p-value": f"{pval:.4f}",
        "Significant": "Yes" if pval < 0.05 else "No",
        "Penk (median)": f"{penk_snr.median():.2f}",
        "Non-Penk (median)": f"{nonpenk_snr.median():.2f}",
    })

penk_er = roi_df[roi_df["celltype"] == "penk"]["event_rate"]
nonpenk_er = roi_df[roi_df["celltype"] == "nonpenk"]["event_rate"]
if len(penk_er) > 0 and len(nonpenk_er) > 0:
    stat, pval = mannwhitneyu(penk_er, nonpenk_er, alternative="two-sided")
    tests_results.append({
        "Test": "Event rate: Penk vs Non-Penk",
        "Statistic": f"U = {stat:.0f}",
        "p-value": f"{pval:.4f}",
        "Significant": "Yes" if pval < 0.05 else "No",
        "Penk (median)": f"{penk_er.median():.1f}",
        "Non-Penk (median)": f"{nonpenk_er.median():.1f}",
    })

penk_active = roi_df[roi_df["celltype"] == "penk"]["active_frac"]
nonpenk_active = roi_df[roi_df["celltype"] == "nonpenk"]["active_frac"]
if len(penk_active) > 0 and len(nonpenk_active) > 0:
    stat, pval = mannwhitneyu(penk_active, nonpenk_active, alternative="two-sided")
    tests_results.append({
        "Test": "Active fraction: Penk vs Non-Penk",
        "Statistic": f"U = {stat:.0f}",
        "p-value": f"{pval:.4f}",
        "Significant": "Yes" if pval < 0.05 else "No",
        "Penk (median)": f"{penk_active.median():.3f}",
        "Non-Penk (median)": f"{nonpenk_active.median():.3f}",
    })

penk_maxdff = roi_df[roi_df["celltype"] == "penk"]["max_dff"]
nonpenk_maxdff = roi_df[roi_df["celltype"] == "nonpenk"]["max_dff"]
if len(penk_maxdff) > 0 and len(nonpenk_maxdff) > 0:
    stat, pval = mannwhitneyu(penk_maxdff, nonpenk_maxdff, alternative="two-sided")
    tests_results.append({
        "Test": "Max dF/F: Penk vs Non-Penk",
        "Statistic": f"U = {stat:.0f}",
        "p-value": f"{pval:.4f}",
        "Significant": "Yes" if pval < 0.05 else "No",
        "Penk (median)": f"{penk_maxdff.median():.3f}",
        "Non-Penk (median)": f"{nonpenk_maxdff.median():.3f}",
    })

if tests_results:
    st.dataframe(pd.DataFrame(tests_results), use_container_width=True, hide_index=True)

# --- Per-session table ---
st.subheader("Per-Session Statistics")
ses_display = ses_df[[
    "exp_id", "celltype", "n_rois", "duration_min", "fps",
]].copy()
ses_display["duration_min"] = ses_display["duration_min"].round(1)
ses_display["fps"] = ses_display["fps"].round(1)
st.dataframe(ses_display, use_container_width=True)

# Note about what's missing
st.markdown("---")
st.info(
    "**Note:** These statistics are calcium-only (from ca.h5). "
    "HD tuning, place coding, and movement modulation statistics "
    "will be available after DLC pose estimation, kinematics, and sync are complete."
)

# Download
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    csv_sessions = ses_df.to_csv(index=False)
    st.download_button("Download session stats (CSV)", csv_sessions, "session_stats.csv", "text/csv")
with col2:
    csv_rois = roi_df.to_csv(index=False)
    st.download_button("Download ROI stats (CSV)", csv_rois, "roi_stats.csv", "text/csv")
