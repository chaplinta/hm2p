"""Population overview — aggregate calcium metrics across all sessions.

Shows total ROI counts, calcium quality metrics, and cross-session
distributions without requiring kinematics data.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    download_s3_bytes,
    load_animals,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.population")

st.title("Population Overview")
st.caption("Aggregate calcium metrics across all sessions. No kinematics required.")


@st.cache_data(ttl=600)
def _load_all_calcium_summaries() -> list[dict]:
    """Load calcium summary stats for all sessions."""
    import h5py

    experiments = load_experiments()
    animals = load_animals()
    animal_map = {a["animal_id"]: a for a in animals}
    summaries = []

    for exp in experiments:
        exp_id = exp["exp_id"]
        animal_id = exp_id.split("_")[-1]
        animal = animal_map.get(animal_id, {})
        sub, ses = parse_session_id(exp_id)

        data = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
        if data is None:
            continue

        try:
            f = h5py.File(io.BytesIO(data), "r")
            dff = f["dff"][:]
            n_rois, n_frames = dff.shape
            fps = float(f.attrs.get("fps_imaging", 9.8))

            summary = {
                "exp_id": exp_id,
                "animal_id": animal_id,
                "celltype": animal.get("celltype", "?"),
                "sub": sub,
                "ses": ses,
                "n_rois": n_rois,
                "n_frames": n_frames,
                "fps": fps,
                "duration_s": n_frames / fps,
            }

            # Per-ROI stats
            for roi in range(n_rois):
                trace = dff[roi]
                # Compute SNR
                f0 = np.percentile(trace, 10)
                if f0 > 0:
                    dff_trace = (trace - f0) / f0
                else:
                    dff_trace = trace
                baseline_std = np.std(dff_trace[dff_trace < np.percentile(dff_trace, 50)])
                peak = np.percentile(dff_trace, 95)
                snr = peak / baseline_std if baseline_std > 0 else 0

                # Event stats
                n_events = 0
                active_frac = 0.0
                if "event_masks" in f:
                    em = f["event_masks"][roi].astype(bool)
                    onsets = np.flatnonzero(em[1:] & ~em[:-1])
                    n_events = len(onsets) + (1 if em[0] else 0)
                    active_frac = float(em.mean())

                summaries.append({
                    **summary,
                    "roi_idx": roi,
                    "mean_dff": float(np.nanmean(trace)),
                    "max_dff": float(np.nanmax(trace)),
                    "std_dff": float(np.nanstd(trace)),
                    "snr": snr,
                    "n_events": n_events,
                    "event_rate": n_events / (n_frames / fps / 60) if n_frames > 0 else 0,
                    "active_frac": active_frac,
                    "skewness": float(np.nan_to_num(
                        ((trace - trace.mean()) ** 3).mean() / (trace.std() ** 3)
                        if trace.std() > 0 else 0
                    )),
                })
            f.close()
        except Exception:
            continue

    return summaries


with st.spinner("Loading calcium data for all sessions..."):
    all_data = _load_all_calcium_summaries()

if not all_data:
    st.warning("No calcium data found on S3.")
    st.stop()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.DataFrame(all_data)

# --- Summary metrics ---
n_sessions = df["exp_id"].nunique()
n_rois_total = len(df)
n_penk = len(df[df["celltype"] == "penk"])
n_nonpenk = len(df[df["celltype"] == "nonpenk"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sessions", n_sessions)
col2.metric("Total ROIs", n_rois_total)
col3.metric("Penk+ ROIs", n_penk)
col4.metric("Non-Penk ROIs", n_nonpenk)

# --- Tabs ---
tab_dist, tab_quality, tab_events, tab_table = st.tabs([
    "Distributions", "Quality Metrics", "Event Analysis", "Full Table",
])


with tab_dist:
    st.subheader("Distribution of Key Metrics")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x="snr", color="celltype", nbins=40,
            barmode="overlay", opacity=0.7,
            color_discrete_map={"penk": "green", "nonpenk": "blue"},
            title="SNR Distribution by Cell Type",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df, x="event_rate", color="celltype", nbins=40,
            barmode="overlay", opacity=0.7,
            color_discrete_map={"penk": "green", "nonpenk": "blue"},
            title="Event Rate (events/min)",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x="max_dff", color="celltype", nbins=40,
            barmode="overlay", opacity=0.7,
            color_discrete_map={"penk": "green", "nonpenk": "blue"},
            title="Max dF/F Distribution",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df, x="skewness", color="celltype", nbins=40,
            barmode="overlay", opacity=0.7,
            color_discrete_map={"penk": "green", "nonpenk": "blue"},
            title="Trace Skewness (higher = burstier)",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


with tab_quality:
    st.subheader("Quality Metrics")

    # SNR vs skewness scatter
    fig = px.scatter(
        df, x="snr", y="skewness", color="celltype",
        hover_data=["exp_id", "roi_idx"],
        color_discrete_map={"penk": "green", "nonpenk": "blue"},
        title="SNR vs Skewness (good cells = high SNR + high skew)",
        opacity=0.5,
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Per-session box plots
    st.subheader("SNR by Session")
    fig = px.box(
        df, x="exp_id", y="snr", color="celltype",
        color_discrete_map={"penk": "green", "nonpenk": "blue"},
    )
    fig.update_layout(
        height=400,
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Quality thresholds
    st.subheader("Quality Thresholds")
    snr_thresh = st.slider("Min SNR", 0.0, 10.0, 2.0, 0.5)
    skew_thresh = st.slider("Min Skewness", -1.0, 5.0, 0.5, 0.1)

    good = df[(df["snr"] >= snr_thresh) & (df["skewness"] >= skew_thresh)]
    st.metric("ROIs passing quality threshold", f"{len(good)}/{len(df)} ({len(good)/len(df)*100:.0f}%)")

    if len(good) > 0:
        col1, col2 = st.columns(2)
        col1.metric("Penk passing", len(good[good["celltype"] == "penk"]))
        col2.metric("Non-Penk passing", len(good[good["celltype"] == "nonpenk"]))


with tab_events:
    st.subheader("Event Analysis")

    # Event rate by cell type
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        for ct, color in [("penk", "green"), ("nonpenk", "blue")]:
            ct_data = df[df["celltype"] == ct]
            fig.add_trace(go.Box(
                y=ct_data["event_rate"],
                name=f"{ct} (n={len(ct_data)})",
                marker_color=color,
                boxmean=True,
            ))
        fig.update_layout(
            title="Event Rate by Cell Type",
            yaxis_title="Events/min",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        for ct, color in [("penk", "green"), ("nonpenk", "blue")]:
            ct_data = df[df["celltype"] == ct]
            fig.add_trace(go.Box(
                y=ct_data["active_frac"],
                name=f"{ct} (n={len(ct_data)})",
                marker_color=color,
                boxmean=True,
            ))
        fig.update_layout(
            title="Active Fraction by Cell Type",
            yaxis_title="Fraction of frames in events",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Event rate vs SNR
    fig = px.scatter(
        df, x="snr", y="event_rate", color="celltype",
        hover_data=["exp_id", "roi_idx"],
        color_discrete_map={"penk": "green", "nonpenk": "blue"},
        title="Event Rate vs SNR",
        opacity=0.5,
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Statistical comparison
    st.subheader("Statistical Comparison")
    from scipy.stats import mannwhitneyu

    penk_rates = df[df["celltype"] == "penk"]["event_rate"].values
    nonpenk_rates = df[df["celltype"] == "nonpenk"]["event_rate"].values

    if len(penk_rates) > 0 and len(nonpenk_rates) > 0:
        stat, pval = mannwhitneyu(penk_rates, nonpenk_rates, alternative="two-sided")
        st.markdown(
            f"**Event rate:** Penk median = {np.median(penk_rates):.2f}, "
            f"Non-Penk median = {np.median(nonpenk_rates):.2f}, "
            f"Mann-Whitney U p = {pval:.4f} "
            f"{'(significant)' if pval < 0.05 else '(not significant)'}"
        )


with tab_table:
    st.subheader("Full Population Table")

    # Filtering
    col1, col2 = st.columns(2)
    with col1:
        ct_filter = st.multiselect("Cell type", ["penk", "nonpenk"], default=["penk", "nonpenk"])
    with col2:
        min_snr = st.number_input("Min SNR filter", 0.0, 20.0, 0.0, 0.5)

    filtered = df[df["celltype"].isin(ct_filter) & (df["snr"] >= min_snr)]
    st.markdown(f"**{len(filtered)} ROIs** shown")

    display_cols = [
        "exp_id", "celltype", "roi_idx", "snr", "mean_dff", "max_dff",
        "std_dff", "skewness", "n_events", "event_rate", "active_frac",
    ]
    st.dataframe(
        filtered[display_cols].round(4),
        use_container_width=True,
        height=500,
    )

    # Download
    csv = filtered[display_cols].to_csv(index=False)
    st.download_button("Download as CSV", csv, "population_data.csv", "text/csv")
