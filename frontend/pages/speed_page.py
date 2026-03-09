"""Speed Modulation — running speed effects on HD tuning.

Examines whether HD cell activity scales with movement speed,
an important control for separating speed from visual anchoring effects.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from frontend.data import load_all_sync_data, session_filter_sidebar
from hm2p.analysis.speed import (
    hd_tuning_by_speed,
    speed_modulation_index,
    speed_tuning_curve,
)
from hm2p.analysis.tuning import compute_hd_tuning_curve

st.title("Speed Modulation")
st.caption(
    "How does running speed affect HD cell activity and tuning? "
    "Important control for isolating visual vs speed-related gain changes."
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
                                format_func=lambda i: session_labels[i], key="spd_ses")
sess = sessions[sel_idx]

signals = sess["dff"]  # (n_rois, n_frames)
hd = sess["hd_deg"]
speed = sess["speed_cm_s"]
mask = sess["active"] & ~sess["bad_behav"]
n_cells = signals.shape[0]

if n_cells == 0:
    st.warning("No ROIs in this session after filtering.")
    st.stop()

tab_pop, tab_single, tab_hd = st.tabs(["Population", "Single Cell", "HD by Speed"])

with tab_pop:
    st.subheader("Population Speed Modulation")

    smis = []
    corrs = []
    for i in range(n_cells):
        result = speed_modulation_index(signals[i], speed, mask)
        smis.append(result["speed_modulation_index"])
        corrs.append(result["speed_correlation"])

    labels = [f"Cell {i+1}" for i in range(n_cells)]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(data=[go.Bar(
            x=labels, y=smis,
            marker_color=["green" if s > 0 else "red" for s in smis],
        )])
        fig.update_layout(
            height=300, title="Speed Modulation Index",
            xaxis_title="Cell", yaxis_title="SMI",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(data=[go.Bar(
            x=labels, y=corrs,
            marker_color=["royalblue" if c > 0 else "salmon" for c in corrs],
        )])
        fig.update_layout(
            height=300, title="Speed-Signal Correlation",
            xaxis_title="Cell", yaxis_title="Pearson r",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    mean_smi = np.mean(smis)
    st.markdown(f"**Mean SMI:** {mean_smi:.3f} — **Mean r:** {np.mean(corrs):.3f}")

with tab_single:
    st.subheader("Single Cell Speed Tuning")
    cell_idx = st.selectbox("Cell", range(n_cells),
                            format_func=lambda x: f"Cell {x+1}", key="spd_cell")

    tc, bc = speed_tuning_curve(signals[cell_idx], speed, mask, n_bins=15)
    result = speed_modulation_index(signals[cell_idx], speed, mask)

    col1, col2, col3 = st.columns(3)
    col1.metric("SMI", f"{result['speed_modulation_index']:.3f}")
    col2.metric("Speed-signal r", f"{result['speed_correlation']:.3f}")
    col3.metric("Fast/Slow ratio",
                f"{result['mean_signal_fast'] / max(result['mean_signal_slow'], 1e-10):.2f}")

    fig = go.Figure(data=[go.Scatter(
        x=bc, y=tc, mode="lines+markers",
        marker=dict(size=6, color="royalblue"),
    )])
    fig.update_layout(
        height=300, title=f"Cell {cell_idx+1} — Signal vs Speed",
        xaxis_title="Speed (cm/s)", yaxis_title="Mean signal",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Signal vs speed scatter (subsample for performance)
    n_show = min(2000, len(speed))
    idx = np.random.default_rng(0).choice(len(speed), n_show, replace=False)
    fig = go.Figure(data=[go.Scatter(
        x=speed[idx], y=signals[cell_idx][idx],
        mode="markers", marker=dict(size=2, opacity=0.3, color="gray"),
    )])
    fig.update_layout(
        height=250, title="Raw Signal vs Speed",
        xaxis_title="Speed (cm/s)", yaxis_title="Signal",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_hd:
    st.subheader("HD Tuning by Speed Group")
    cell_idx2 = st.selectbox("Cell", range(n_cells),
                             format_func=lambda x: f"Cell {x+1}", key="spd_hd_cell")

    hd_result = hd_tuning_by_speed(signals[cell_idx2], hd, speed, mask)

    # Polar overlay of tuning curves per speed group
    bc_hd = hd_result["bin_centers"]
    theta_plot = np.concatenate([np.deg2rad(bc_hd), [np.deg2rad(bc_hd[0])]])

    colors = ["blue", "green", "red"]
    fig = go.Figure()
    for i, (tc_i, label) in enumerate(zip(hd_result["tuning_curves"],
                                           hd_result["speed_labels"])):
        if not np.all(np.isnan(tc_i)):
            fig.add_trace(go.Scatterpolar(
                r=np.concatenate([tc_i, [tc_i[0]]]),
                theta=np.rad2deg(theta_plot),
                mode="lines", line=dict(color=colors[i % len(colors)], width=2),
                name=label,
            ))
    fig.update_layout(
        height=350, title=f"Cell {cell_idx2+1} — HD Tuning by Speed",
        polar=dict(radialaxis=dict(showticklabels=True)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # MVL comparison
    col1, col2, col3 = st.columns(3)
    for i, (label, mvl) in enumerate(zip(hd_result["speed_labels"], hd_result["mvls"])):
        [col1, col2, col3][i % 3].metric(label, f"MVL={mvl:.3f}" if np.isfinite(mvl) else "N/A")

# Footer
st.markdown("---")
st.caption(
    "SMI = (fast - slow) / (fast + slow). Positive = higher activity at speed. "
    "HD tuning by speed shows whether tuning sharpness changes with movement. "
    "Speed modulation should be controlled for when interpreting light/dark effects."
)
