"""Cross-session comparison page — compare calcium metrics across sessions and cell types.

Enables comparison of calcium metrics, activity distributions, and cell-type
differences across sessions, animals, and cell types (Penk vs non-Penk).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import load_all_ca_data, session_filter_sidebar
from hm2p.constants import CELLTYPE_HEX, HEX_PENK, HEX_NONPENK

log = logging.getLogger("hm2p.frontend.compare")

st.title("Cross-Session Comparison")

# --- Load pooled ca data ---
with st.spinner("Loading calcium data for all sessions..."):
    all_sessions = load_all_ca_data()

sessions = session_filter_sidebar(
    all_sessions, show_roi_filter=True, key_prefix="compare"
)

if not sessions:
    st.warning("No calcium data found for the selected filters.")
    st.stop()

# --- Build session summaries from cached data ---
session_data = []
for s in sessions:
    dff = s["dff"]
    n_rois, n_frames = s["n_rois"], s["n_frames"]
    fps = s["fps"]

    summary = {
        "exp_id": s["exp_id"],
        "animal_id": s["animal_id"],
        "celltype": s["celltype"],
        "sub": s["sub"],
        "ses": s["ses"],
        "n_rois": n_rois,
        "n_frames": n_frames,
        "fps": fps,
        "duration_s": n_frames / fps,
        "mean_dff": float(np.nanmean(dff)),
        "max_dff": float(np.nanmax(dff)),
        "per_roi_mean": np.nanmean(dff, axis=1).tolist(),
        "per_roi_max": np.nanmax(dff, axis=1).tolist(),
        "per_roi_std": np.nanstd(dff, axis=1).tolist(),
    }

    em = s.get("event_masks")
    if em is not None:
        summary["per_roi_active_frac"] = em.mean(axis=1).tolist()
        counts = []
        for i in range(n_rois):
            m = em[i].astype(bool)
            onsets = np.flatnonzero(m[1:] & ~m[:-1])
            counts.append(len(onsets) + (1 if m[0] else 0))
        summary["per_roi_event_count"] = counts
        summary["per_roi_event_rate"] = [c / (n_frames / fps / 60) for c in counts]

    session_data.append(summary)

n_sessions = len(session_data)
n_total_rois = sum(s["n_rois"] for s in session_data)
col1, col2 = st.columns(2)
col1.metric("Sessions", n_sessions)
col2.metric("Total ROIs", n_total_rois)

# --- Overview table ---
st.subheader("Session Overview")

overview_rows = []
for s in session_data:
    overview_rows.append({
        "Session": s["exp_id"][:15],
        "Animal": s["animal_id"],
        "Type": s["celltype"],
        "ROIs": s["n_rois"],
        "Duration (s)": f"{s['duration_s']:.0f}",
        "FPS": f"{s['fps']:.1f}",
        "Mean dF/F0": f"{s['mean_dff']:.4f}",
        "Max dF/F0": f"{s['max_dff']:.2f}",
    })

df_overview = pd.DataFrame(overview_rows)
st.dataframe(df_overview, use_container_width=True, hide_index=True)

# --- Tabs ---
tab_rois, tab_activity, tab_celltypes = st.tabs([
    "ROI Counts & Quality",
    "Activity Comparison",
    "Cell Type Comparison",
])


with tab_rois:
    st.subheader("ROI Count and Quality Across Sessions")

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[s["exp_id"][:15] for s in session_data],
            y=[s["n_rois"] for s in session_data],
            marker_color=[HEX_PENK if s["celltype"] == "penk" else HEX_NONPENK for s in session_data],
            text=[s["celltype"] for s in session_data],
        ))
        fig.update_layout(
            title="ROIs per Session",
            xaxis_title="Session", yaxis_title="N ROIs",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Total ROIs by cell type
        penk_rois = sum(s["n_rois"] for s in session_data if s["celltype"] == "penk")
        nonpenk_rois = sum(s["n_rois"] for s in session_data if s["celltype"] == "nonpenk")

        fig = go.Figure(data=[go.Pie(
            labels=["Penk+", "Penk⁻CamKII+"],
            values=[penk_rois, nonpenk_rois],
            marker_colors=[HEX_PENK, HEX_NONPENK],
        )])
        fig.update_layout(title="Total ROIs by Cell Type", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Per-ROI mean dF/F0 distribution across sessions
    st.subheader("Per-ROI Mean dF/F0 Distribution")
    fig = go.Figure()
    for s in session_data:
        fig.add_trace(go.Box(
            y=s["per_roi_mean"],
            name=f"{s['exp_id'][:10]}",
            boxmean=True,
            marker_color=HEX_PENK if s["celltype"] == "penk" else HEX_NONPENK,
        ))
    fig.update_layout(
        title="Mean dF/F0 per ROI (each box = one session)",
        yaxis_title="Mean dF/F0",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


with tab_activity:
    st.subheader("Event Rate Comparison")

    sessions_with_events = [s for s in session_data if "per_roi_event_rate" in s]

    if not sessions_with_events:
        st.info("No event data available.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            for s in sessions_with_events:
                fig.add_trace(go.Box(
                    y=s["per_roi_event_rate"],
                    name=f"{s['exp_id'][:10]}",
                    boxmean=True,
                    marker_color=HEX_PENK if s["celltype"] == "penk" else HEX_NONPENK,
                ))
            fig.update_layout(
                title="Event Rate per ROI (events/min)",
                yaxis_title="Events/min",
                height=400,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            for s in sessions_with_events:
                fig.add_trace(go.Box(
                    y=s["per_roi_active_frac"],
                    name=f"{s['exp_id'][:10]}",
                    boxmean=True,
                    marker_color=HEX_PENK if s["celltype"] == "penk" else HEX_NONPENK,
                ))
            fig.update_layout(
                title="Active Fraction per ROI",
                yaxis_title="Fraction of frames in events",
                height=400,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Session-level summary
        st.subheader("Session-Level Event Summary")
        event_rows = []
        for s in sessions_with_events:
            rates = s["per_roi_event_rate"]
            event_rows.append({
                "Session": s["exp_id"][:15],
                "Type": s["celltype"],
                "ROIs": s["n_rois"],
                "Mean rate": f"{np.mean(rates):.1f}",
                "Median rate": f"{np.median(rates):.1f}",
                "Total events": sum(s["per_roi_event_count"]),
                "Active ROIs": sum(1 for r in rates if r > 0),
            })
        st.dataframe(pd.DataFrame(event_rows), use_container_width=True, hide_index=True)


with tab_celltypes:
    st.subheader("Penk+ vs Non-Penk Comparison")

    penk_sessions = [s for s in session_data if s["celltype"] == "penk"]
    nonpenk_sessions = [s for s in session_data if s["celltype"] == "nonpenk"]

    if not penk_sessions or not nonpenk_sessions:
        st.info("Need both Penk and non-Penk sessions for comparison.")
    else:
        penk_rois = sum(s["n_rois"] for s in penk_sessions)
        nonpenk_rois = sum(s["n_rois"] for s in nonpenk_sessions)

        col1, col2, col3 = st.columns(3)
        col1.metric("Penk sessions", len(penk_sessions))
        col2.metric("Non-Penk sessions", len(nonpenk_sessions))
        col3.metric("Total ROIs", f"{penk_rois} + {nonpenk_rois}")

        # Pool all ROI metrics by cell type
        penk_means = []
        nonpenk_means = []
        for s in penk_sessions:
            penk_means.extend(s["per_roi_mean"])
        for s in nonpenk_sessions:
            nonpenk_means.extend(s["per_roi_mean"])

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=penk_means, name="Penk+", opacity=0.7,
                marker_color=HEX_PENK, nbinsx=30,
            ))
            fig.add_trace(go.Histogram(
                x=nonpenk_means, name="Non-Penk", opacity=0.7,
                marker_color=HEX_NONPENK, nbinsx=30,
            ))
            fig.update_layout(
                barmode="overlay",
                title="Mean dF/F0 Distribution",
                xaxis_title="Mean dF/F0",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Event rates by cell type
            penk_rates = []
            nonpenk_rates = []
            for s in penk_sessions:
                if "per_roi_event_rate" in s:
                    penk_rates.extend(s["per_roi_event_rate"])
            for s in nonpenk_sessions:
                if "per_roi_event_rate" in s:
                    nonpenk_rates.extend(s["per_roi_event_rate"])

            if penk_rates and nonpenk_rates:
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=penk_rates, name="Penk+",
                    marker_color=HEX_PENK, boxmean=True,
                ))
                fig.add_trace(go.Box(
                    y=nonpenk_rates, name="Non-Penk",
                    marker_color=HEX_NONPENK, boxmean=True,
                ))
                fig.update_layout(
                    title="Event Rate by Cell Type",
                    yaxis_title="Events/min",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        st.subheader("Summary Statistics")
        from scipy.stats import mannwhitneyu

        summary = []
        for metric_name, penk_vals, nonpenk_vals in [
            ("Mean dF/F0", penk_means, nonpenk_means),
            ("Event rate", penk_rates, nonpenk_rates),
        ]:
            if penk_vals and nonpenk_vals:
                try:
                    stat, pval = mannwhitneyu(penk_vals, nonpenk_vals, alternative="two-sided")
                    summary.append({
                        "Metric": metric_name,
                        "Penk mean": f"{np.mean(penk_vals):.4f}",
                        "Penk median": f"{np.median(penk_vals):.4f}",
                        "Penk n": len(penk_vals),
                        "NonPenk mean": f"{np.mean(nonpenk_vals):.4f}",
                        "NonPenk median": f"{np.median(nonpenk_vals):.4f}",
                        "NonPenk n": len(nonpenk_vals),
                        "MWU p": f"{pval:.4f}",
                        "Significant": "Y" if pval < 0.05 else "-",
                    })
                except Exception as e:
                    st.info(f"Statistical test failed for {metric_name}: {e}")

        if summary:
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
