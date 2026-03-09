"""Cue Anchoring — HD re-anchoring after visual cue restoration.

Core science: when lights turn back on, how fast does the HD network
snap back to the visually anchored PD? Compares Penk vs non-Penk.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from frontend.data import load_all_sync_data, session_filter_sidebar
from hm2p.analysis.anchoring import (
    anchoring_speed,
    anchoring_time_course,
    find_transitions,
)
from hm2p.analysis.tuning import compute_hd_tuning_curve

st.title("Cue Anchoring")
st.caption(
    "How quickly does the HD network re-anchor to visual landmarks "
    "when lights turn back on after darkness?"
)

# Load real data
all_data = load_all_sync_data()
if all_data["n_sessions"] == 0:
    st.warning(
        "No data available yet. This page will populate when the relevant "
        "pipeline stage completes."
    )
    st.stop()

sessions = session_filter_sidebar(all_data["sessions"])
if not sessions:
    st.warning("No sessions match the current filters.")
    st.stop()

# Session selector
session_labels = [f"{s['exp_id']} ({s['celltype']}, {s['n_rois']} ROIs)" for s in sessions]
sel_idx = st.sidebar.selectbox("Session", range(len(sessions)),
                                format_func=lambda i: session_labels[i], key="anch_ses")
sess = sessions[sel_idx]

signals = sess["dff"]  # (n_rois, n_frames)
hd = sess["hd_deg"]
mask = sess["active"] & ~sess["bad_behav"]
light_on = sess["light_on"]
n_cells = signals.shape[0]
fps = 30.0

if n_cells == 0:
    st.warning("No ROIs in this session after filtering.")
    st.stop()

# Cell selector in sidebar
cell_main = st.sidebar.selectbox("Primary cell", range(n_cells),
                                  format_func=lambda x: f"Cell {x+1}", key="anch_main_cell")

signal = signals[cell_main]

# Estimate cycle length from light transitions
transitions = find_transitions(light_on)
if len(transitions) >= 2:
    cycle_frames = int(np.median(np.diff([t["frame"] for t in transitions])))
else:
    cycle_frames = int(60 * fps)  # default 60s

cycle_s = cycle_frames / fps

tab_tc, tab_speed, tab_compare = st.tabs(["Time Course", "Anchoring Speed", "Multi-Cell"])

with tab_tc:
    st.subheader("PD Deviation Around Dark->Light Transitions")

    result = anchoring_time_course(
        signal, hd, mask, light_on, fps=fps,
        window_frames=int(cycle_frames * 0.3),
        step_frames=int(cycle_frames * 0.05),
        pre_transition_s=cycle_s * 0.8,
        post_transition_s=cycle_s * 0.8,
    )

    col1, col2 = st.columns(2)
    col1.metric("Transitions averaged", result["n_transitions"])
    col2.metric("Reference PD", f"{result['reference_pd']:.1f}")

    if result["n_transitions"] > 0:
        t = result["time_offsets_s"]
        devs = result["pd_deviations"]
        mvls = result["mvls"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t, y=np.abs(devs),
            mode="lines+markers", name="|PD deviation|",
            marker=dict(size=4),
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="red",
                      annotation_text="Lights ON")
        fig.add_vrect(x0=min(t), x1=0, fillcolor="gray", opacity=0.1, line_width=0)
        fig.update_layout(
            height=300, title="PD Deviation from Reference (transition-aligned)",
            xaxis_title="Time from transition (s)",
            yaxis_title="|PD deviation| (deg)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # MVL time course
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t, y=mvls, mode="lines+markers",
            marker=dict(size=4, color="green"),
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.add_vrect(x0=min(t), x1=0, fillcolor="gray", opacity=0.1, line_width=0)
        fig.update_layout(
            height=250, title="MVL Around Transition",
            xaxis_title="Time from transition (s)",
            yaxis_title="MVL",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_speed:
    st.subheader("Re-Anchoring Speed")

    if result["n_transitions"] > 0:
        speed_result = anchoring_speed(result["pd_deviations"], result["time_offsets_s"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pre-deviation", f"{speed_result['pre_deviation']:.1f}")
        col2.metric("Post-deviation", f"{speed_result['post_deviation']:.1f}")
        col3.metric("Half-time", f"{speed_result['half_time_s']:.1f}s" if np.isfinite(speed_result['half_time_s']) else "N/A")
        col4.metric("Anchoring strength", f"{speed_result['anchoring_strength']:.2f}" if np.isfinite(speed_result['anchoring_strength']) else "N/A")

        st.markdown(
            "**Interpretation:** Anchoring strength near 1 means the cell fully "
            "re-anchors to the visual reference. Near 0 means the drift persists. "
            "Negative means the cell drifts *more* after lights come on."
        )

        # Visual summary
        if np.isfinite(speed_result["anchoring_strength"]):
            if speed_result["anchoring_strength"] > 0.7:
                st.success("Strong visual anchoring -- PD snaps back to reference.")
            elif speed_result["anchoring_strength"] > 0.3:
                st.info("Moderate anchoring -- partial re-alignment to visual cues.")
            else:
                st.warning("Weak anchoring -- HD representation largely path-integration driven.")
    else:
        st.warning("No transitions found.")

with tab_compare:
    st.subheader("Multi-Cell Comparison")
    st.markdown(
        "Compare anchoring across all cells in this session."
    )

    strengths = []
    half_times = []
    cell_labels = []

    for i in range(n_cells):
        tc_result = anchoring_time_course(
            signals[i], hd, mask, light_on, fps=fps,
            window_frames=int(cycle_frames * 0.3),
            step_frames=int(cycle_frames * 0.05),
            pre_transition_s=cycle_s * 0.8,
            post_transition_s=cycle_s * 0.8,
        )
        if tc_result["n_transitions"] > 0:
            sp = anchoring_speed(tc_result["pd_deviations"], tc_result["time_offsets_s"])
            strengths.append(sp["anchoring_strength"])
            half_times.append(sp["half_time_s"])
        else:
            strengths.append(np.nan)
            half_times.append(np.nan)
        cell_labels.append(f"Cell {i+1}")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(data=[go.Bar(
            x=cell_labels,
            y=strengths,
            marker_color=["green" if (np.isfinite(s) and s > 0.5) else "orange" if (np.isfinite(s) and s > 0.2) else "red"
                          for s in strengths],
        )])
        fig.update_layout(
            height=300, title="Anchoring Strength by Cell",
            yaxis_title="Anchoring strength",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        finite_mask = [np.isfinite(s) for s in strengths]
        fig = go.Figure(data=[go.Scatter(
            x=list(range(n_cells)),
            y=strengths,
            mode="markers+lines",
            marker=dict(size=10, color="royalblue"),
            text=cell_labels,
        )])
        fig.update_layout(
            height=300, title="Anchoring Strength Across Cells",
            xaxis_title="Cell index",
            yaxis_title="Anchoring strength",
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption(
    "Transition-aligned analysis averages PD deviation across all dark->light "
    "transitions. Gray shading = dark period before transition. "
    "Red dashed line = lights on."
)
