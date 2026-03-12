"""Cross-session comparison page — compare HD tuning across sessions and cell types.

Enables comparison of calcium metrics, HD tuning distributions, and place
coding across sessions, animals, and cell types (Penk vs non-Penk).
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
    REGION,
    download_s3_bytes,
    load_animals,
    load_experiments,
    parse_session_id,
)
from hm2p.constants import CELLTYPE_HEX, HEX_PENK, HEX_NONPENK

log = logging.getLogger("hm2p.frontend.compare")

st.title("Cross-Session Comparison")

# --- Load metadata ---
experiments = load_experiments()
animals = load_animals()
animal_map = {a["animal_id"]: a for a in animals}


@st.cache_data(ttl=600)
def _load_ca_summary(sub: str, ses: str) -> dict | None:
    """Load ca.h5 from S3 and return summary stats."""
    import h5py

    data = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
    if data is None:
        return None
    try:
        f = h5py.File(io.BytesIO(data), "r")
        dff = f["dff"][:]
        n_rois, n_frames = dff.shape
        fps = float(f.attrs.get("fps_imaging", 9.8))

        result = {
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

        if "event_masks" in f:
            em = f["event_masks"][:]
            result["per_roi_active_frac"] = em.mean(axis=1).tolist()
            # Event counts
            counts = []
            for i in range(n_rois):
                m = em[i].astype(bool)
                onsets = np.flatnonzero(m[1:] & ~m[:-1])
                counts.append(len(onsets) + (1 if m[0] else 0))
            result["per_roi_event_count"] = counts
            result["per_roi_event_rate"] = [c / (n_frames / fps / 60) for c in counts]

        f.close()
        return result
    except Exception:
        return None


# --- Load summaries for all sessions ---
st.sidebar.header("Filters")

# Cell type filter
celltypes = sorted(set(animal_map.get(e["exp_id"].split("_")[-1], {}).get("celltype", "?") for e in experiments))
celltype_filter = st.sidebar.multiselect("Cell type", celltypes, default=celltypes)

# Build session list
session_data = []
with st.spinner("Loading calcium summaries from S3..."):
    for exp in experiments:
        exp_id = exp["exp_id"]
        animal_id = exp_id.split("_")[-1]
        animal = animal_map.get(animal_id, {})
        celltype = animal.get("celltype", "?")

        if celltype not in celltype_filter:
            continue

        sub, ses = parse_session_id(exp_id)
        summary = _load_ca_summary(sub, ses)
        if summary is not None:
            summary["exp_id"] = exp_id
            summary["animal_id"] = animal_id
            summary["celltype"] = celltype
            summary["sub"] = sub
            summary["ses"] = ses
            session_data.append(summary)

if not session_data:
    st.warning("No calcium data found for the selected filters.")
    st.stop()

st.sidebar.markdown(f"**{len(session_data)} sessions** loaded")

# --- Overview table ---
st.subheader("Session Overview")

import pandas as pd
import plotly.graph_objects as go

overview_rows = []
for s in session_data:
    overview_rows.append({
        "Session": s["exp_id"][:15],
        "Animal": s["animal_id"],
        "Type": s["celltype"],
        "ROIs": s["n_rois"],
        "Duration (s)": f"{s['duration_s']:.0f}",
        "FPS": f"{s['fps']:.1f}",
        "Mean dF/F": f"{s['mean_dff']:.4f}",
        "Max dF/F": f"{s['max_dff']:.2f}",
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
            labels=["Penk+", "Non-Penk CamKII+"],
            values=[penk_rois, nonpenk_rois],
            marker_colors=[HEX_PENK, HEX_NONPENK],
        )])
        fig.update_layout(title="Total ROIs by Cell Type", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Per-ROI mean dF/F distribution across sessions
    st.subheader("Per-ROI Mean dF/F Distribution")
    fig = go.Figure()
    for s in session_data:
        fig.add_trace(go.Box(
            y=s["per_roi_mean"],
            name=f"{s['exp_id'][:10]}",
            boxmean=True,
            marker_color=HEX_PENK if s["celltype"] == "penk" else HEX_NONPENK,
        ))
    fig.update_layout(
        title="Mean dF/F per ROI (each box = one session)",
        yaxis_title="Mean dF/F",
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
                title="Mean dF/F Distribution",
                xaxis_title="Mean dF/F",
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
            ("Mean dF/F", penk_means, nonpenk_means),
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
                except Exception:
                    pass

        if summary:
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
