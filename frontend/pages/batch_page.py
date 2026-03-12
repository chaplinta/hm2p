"""Batch Overview — at-a-glance quality metrics for all sessions in a single table."""

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
from hm2p.constants import CELLTYPE_HEX

log = logging.getLogger("hm2p.frontend.batch")

st.title("Batch Overview")
st.caption("At-a-glance summary of all sessions — ROI counts, quality, events, durations.")


@st.cache_data(ttl=600)
def load_batch_summary() -> pd.DataFrame:
    """Load per-session summary metrics from ca.h5 files."""
    import h5py

    experiments = load_experiments()
    animals = load_animals()
    animal_map = {a["animal_id"]: a for a in animals}
    rows = []

    for exp in experiments:
        exp_id = exp["exp_id"]
        animal_id = exp_id.split("_")[-1]
        animal = animal_map.get(animal_id, {})
        sub, ses = parse_session_id(exp_id)

        row = {
            "exp_id": exp_id,
            "animal": animal_id,
            "celltype": animal.get("celltype", "?"),
            "sub": sub,
            "ses": ses,
            "exclude": exp.get("exclude", "0"),
        }

        data = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
        if data is None:
            row.update({
                "n_rois": 0, "n_frames": 0, "duration_s": 0, "fps": 0,
                "median_snr": 0, "mean_snr": 0, "n_good_rois": 0,
                "median_event_rate": 0, "mean_max_dff": 0,
                "has_events": False, "has_deconv": False,
            })
            rows.append(row)
            continue

        try:
            f = h5py.File(io.BytesIO(data), "r")
            dff = f["dff"][:]
            n_rois, n_frames = dff.shape
            fps = float(f.attrs.get("fps_imaging", 9.8))

            # Per-ROI SNR
            snrs = []
            for i in range(n_rois):
                trace = dff[i]
                baseline_std = np.std(trace[trace < np.percentile(trace, 50)])
                peak = np.percentile(trace, 95)
                snrs.append(peak / baseline_std if baseline_std > 0 else 0)
            snrs = np.array(snrs)

            # Event stats
            has_events = "event_masks" in f
            event_rates = []
            if has_events:
                em = f["event_masks"][:]
                for i in range(n_rois):
                    mask = em[i].astype(bool)
                    onsets = np.flatnonzero(mask[1:] & ~mask[:-1])
                    n_events = len(onsets) + (1 if mask[0] else 0)
                    event_rates.append(n_events / (n_frames / fps / 60))
                event_rates = np.array(event_rates)

            row.update({
                "n_rois": n_rois,
                "n_frames": n_frames,
                "duration_s": round(n_frames / fps, 1),
                "fps": round(fps, 1),
                "median_snr": round(float(np.median(snrs)), 2),
                "mean_snr": round(float(np.mean(snrs)), 2),
                "n_good_rois": int(np.sum(snrs >= 3)),
                "median_event_rate": round(float(np.median(event_rates)), 1) if len(event_rates) > 0 else 0,
                "mean_max_dff": round(float(np.mean([np.nanmax(dff[i]) for i in range(n_rois)])), 3),
                "has_events": has_events,
                "has_deconv": "spks" in f,
            })
            f.close()
        except Exception:
            row.update({
                "n_rois": 0, "n_frames": 0, "duration_s": 0, "fps": 0,
                "median_snr": 0, "mean_snr": 0, "n_good_rois": 0,
                "median_event_rate": 0, "mean_max_dff": 0,
                "has_events": False, "has_deconv": False,
            })

        rows.append(row)

    return pd.DataFrame(rows)


with st.spinner("Loading batch summary (this may take a moment)..."):
    df = load_batch_summary()

# --- Summary cards ---
n_sessions = len(df)
total_rois = df["n_rois"].sum()
total_good = df["n_good_rois"].sum()
total_duration = df["duration_s"].sum()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Sessions", n_sessions)
col2.metric("Total ROIs", total_rois)
col3.metric("Good ROIs (SNR>=3)", total_good)
col4.metric("Total recording", f"{total_duration/3600:.1f} hrs")
col5.metric("Mean session", f"{df['duration_s'].mean():.0f}s")

# --- Celltype breakdown ---
col1, col2 = st.columns(2)
with col1:
    penk = df[df["celltype"] == "penk"]
    st.markdown(f"**Penk+**: {len(penk)} sessions, {penk['n_rois'].sum()} ROIs ({penk['n_good_rois'].sum()} good)")
with col2:
    nonpenk = df[df["celltype"] == "nonpenk"]
    st.markdown(f"**Non-Penk**: {len(nonpenk)} sessions, {nonpenk['n_rois'].sum()} ROIs ({nonpenk['n_good_rois'].sum()} good)")

# --- Main table ---
st.subheader("Session Summary Table")

# Color-code by quality
def style_snr(val):
    if isinstance(val, (int, float)):
        if val >= 5:
            return "background-color: #d4edda"
        elif val >= 3:
            return "background-color: #fff3cd"
        elif val > 0:
            return "background-color: #f8d7da"
    return ""

def style_rois(val):
    if isinstance(val, (int, float)) and val == 0:
        return "background-color: #f8d7da"
    return ""

display_cols = [
    "exp_id", "celltype", "n_rois", "n_good_rois", "duration_s",
    "median_snr", "mean_snr", "median_event_rate", "mean_max_dff",
    "has_events", "has_deconv", "exclude",
]

styled = df[display_cols].style.map(
    style_snr, subset=["median_snr", "mean_snr"]
).map(
    style_rois, subset=["n_rois"]
)

st.dataframe(styled, use_container_width=True, height=500)

# --- Visualizations ---
st.subheader("Cross-Session Comparisons")

import plotly.express as px
import plotly.graph_objects as go

tab_rois, tab_quality, tab_duration = st.tabs(["ROI Counts", "Quality", "Duration"])

with tab_rois:
    fig = px.bar(
        df.sort_values("celltype"),
        x="exp_id",
        y="n_rois",
        color="celltype",
        color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
        title="ROIs per Session",
        hover_data=["n_good_rois", "median_snr"],
    )
    fig.update_layout(height=350, xaxis=dict(tickangle=45))
    st.plotly_chart(fig, use_container_width=True)

    # Good vs total
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df["exp_id"],
        y=df["n_rois"],
        name="Total ROIs",
        marker_color="lightgray",
    ))
    fig2.add_trace(go.Bar(
        x=df["exp_id"],
        y=df["n_good_rois"],
        name="Good ROIs (SNR>=3)",
        marker_color="green",
    ))
    fig2.update_layout(
        barmode="overlay",
        title="Total vs Good ROIs",
        height=350,
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab_quality:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            df.sort_values("median_snr", ascending=False),
            x="exp_id",
            y="median_snr",
            color="celltype",
            color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
            title="Median SNR by Session",
        )
        fig.add_hline(y=3, line_dash="dash", line_color="red", opacity=0.5)
        fig.update_layout(height=350, xaxis=dict(tickangle=45))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            df[df["median_event_rate"] > 0].sort_values("median_event_rate", ascending=False),
            x="exp_id",
            y="median_event_rate",
            color="celltype",
            color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
            title="Median Event Rate (events/min)",
        )
        fig.update_layout(height=350, xaxis=dict(tickangle=45))
        st.plotly_chart(fig, use_container_width=True)

    # SNR scatter: per session
    fig = px.scatter(
        df,
        x="median_snr",
        y="median_event_rate",
        color="celltype",
        size="n_rois",
        hover_data=["exp_id"],
        color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
        title="SNR vs Event Rate (per session)",
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

with tab_duration:
    fig = px.bar(
        df,
        x="exp_id",
        y="duration_s",
        color="celltype",
        color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
        title="Session Duration (seconds)",
    )
    fig.update_layout(height=350, xaxis=dict(tickangle=45))
    st.plotly_chart(fig, use_container_width=True)

# --- Download ---
st.markdown("---")
csv = df.to_csv(index=False)
st.download_button("Download batch summary CSV", csv, "batch_summary.csv", "text/csv")
