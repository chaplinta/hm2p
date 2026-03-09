"""Stability Analysis — HD tuning stability over time and across conditions.

Visualizes temporal stability of head direction tuning: first/second half
comparison, sliding window MVL/PD tracking, and light/dark epoch comparison.

Requires real sync.h5 data from S3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

st.title("Tuning Stability")
st.caption(
    "Temporal stability of head direction tuning — tracks how tuning "
    "changes over time and between light/dark conditions."
)

import plotly.graph_objects as go

from hm2p.analysis.stability import (
    light_dark_stability,
    sliding_window_stability,
    split_temporal_halves,
)


# --- Data loading ---

def _try_load_real():
    """Attempt to load real sync.h5 data."""
    try:
        from frontend.data import load_all_sync_data, session_filter_sidebar
        all_data = load_all_sync_data()
        if all_data["n_sessions"] > 0:
            sessions = session_filter_sidebar(all_data["sessions"])
            return sessions, True
    except Exception:
        pass
    return None, False


real_sessions, has_real = _try_load_real()

if not has_real or not real_sessions:
    st.warning(
        "No data available yet. This page will populate when the relevant "
        "pipeline stage completes."
    )
    st.stop()

st.success(
    f"Loaded {len(real_sessions)} sessions, "
    f"{sum(s['n_rois'] for s in real_sessions)} total cells"
)

# Session and cell selection
session_labels = [s["exp_id"] for s in real_sessions]
sel_session_idx = st.selectbox("Session", range(len(session_labels)),
                                format_func=lambda i: session_labels[i],
                                key="stab_session")
ses_data = real_sessions[sel_session_idx]
n_rois = ses_data["n_rois"]
sel_cell = st.slider("Cell index", 0, max(0, n_rois - 1), 0, key="stab_cell")

signal = ses_data["dff"][sel_cell]
hd = ses_data["hd_deg"]
mask = ses_data["active"] & ~ses_data["bad_behav"]

tab_halves, tab_sliding, tab_lightdark = st.tabs([
    "First/Second Half", "Sliding Window", "Light/Dark",
])

# --- Split halves ---
with tab_halves:
    st.subheader("First Half vs Second Half")
    result = split_temporal_halves(signal, hd, mask)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Correlation", f"{result['correlation']:.3f}")
    col2.metric("PD shift", f"{result['pd_shift_deg']:.1f} deg")
    col3.metric("MVL (1st half)", f"{result['mvl_half1']:.3f}")
    col4.metric("MVL (2nd half)", f"{result['mvl_half2']:.3f}")

    # Overlay polar plots
    tc1 = result["tuning_curve_1"]
    tc2 = result["tuning_curve_2"]
    bc = result["bin_centers"]

    theta_p = np.concatenate([np.deg2rad(bc), [np.deg2rad(bc[0])]])
    r1 = np.concatenate([np.where(np.isnan(tc1), 0, tc1), [tc1[0] if not np.isnan(tc1[0]) else 0]])
    r2 = np.concatenate([np.where(np.isnan(tc2), 0, tc2), [tc2[0] if not np.isnan(tc2[0]) else 0]])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r1, theta=np.rad2deg(theta_p),
        mode="lines", fill="toself",
        fillcolor="rgba(65, 105, 225, 0.2)",
        line=dict(color="royalblue", width=2),
        name="First half",
    ))
    fig.add_trace(go.Scatterpolar(
        r=r2, theta=np.rad2deg(theta_p),
        mode="lines", fill="toself",
        fillcolor="rgba(225, 65, 105, 0.2)",
        line=dict(color="#E14169", width=2),
        name="Second half",
    ))
    fig.update_layout(
        height=400,
        polar=dict(angularaxis=dict(direction="clockwise", rotation=90)),
        title=f"Tuning Curve Comparison (r={result['correlation']:.3f})",
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Sliding window ---
with tab_sliding:
    st.subheader("Sliding Window Analysis")

    col_w, col_s = st.columns(2)
    with col_w:
        win_frames = st.slider("Window (frames)", 500, 3000, 1500, 100, key="stab_win")
    with col_s:
        step = st.slider("Step (frames)", 100, 1000, 300, 50, key="stab_step")

    sw = sliding_window_stability(signal, hd, mask, window_frames=win_frames, step_frames=step)

    if sw["n_windows"] > 1:
        col_sw1, col_sw2 = st.columns(2)
        with col_sw1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sw["window_centers"], y=sw["mvls"],
                mode="lines+markers", marker_color="royalblue",
            ))
            fig.update_layout(
                height=300, title="MVL Over Time",
                xaxis_title="Frame", yaxis_title="MVL",
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_sw2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sw["window_centers"], y=sw["preferred_dirs"],
                mode="lines+markers", marker_color="orange",
            ))
            fig.update_layout(
                height=300, title="Preferred Direction Over Time",
                xaxis_title="Frame", yaxis_title="PD (deg)",
                yaxis=dict(range=[0, 360]),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        st.markdown(
            f"**Windows:** {sw['n_windows']} --- "
            f"**MVL range:** {sw['mvls'].min():.3f}--{sw['mvls'].max():.3f} --- "
            f"**MVL std:** {np.std(sw['mvls']):.3f} --- "
            f"**PD range:** {sw['preferred_dirs'].min():.0f} deg--{sw['preferred_dirs'].max():.0f} deg"
        )
    else:
        st.warning("Not enough frames for sliding window analysis.")

# --- Light/Dark ---
with tab_lightdark:
    st.subheader("Light vs Dark Tuning")

    # Use light_on from the session data if available
    if "light_on" in ses_data:
        light_on = ses_data["light_on"]
    else:
        st.warning(
            "No light_on data available for this session. "
            "Light/dark analysis requires TDMS-derived light timestamps in sync.h5."
        )
        st.stop()

    n_light = light_on.sum()
    n_dark = (~light_on).sum()
    st.markdown(f"**Light frames:** {n_light} ({n_light/len(signal)*100:.0f}%) --- "
                f"**Dark frames:** {n_dark} ({n_dark/len(signal)*100:.0f}%)")

    ld = light_dark_stability(signal, hd, mask, light_on)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Correlation", f"{ld['correlation']:.3f}")
    col2.metric("PD shift", f"{ld['pd_shift_deg']:.1f} deg")
    col3.metric("MVL (light)", f"{ld['mvl_light']:.3f}")
    col4.metric("MVL (dark)", f"{ld['mvl_dark']:.3f}")

    tc_l = ld["tuning_curve_light"]
    tc_d = ld["tuning_curve_dark"]
    bc_ld = ld["bin_centers"]

    theta_ld = np.concatenate([np.deg2rad(bc_ld), [np.deg2rad(bc_ld[0])]])
    r_l = np.concatenate([np.where(np.isnan(tc_l), 0, tc_l), [tc_l[0] if not np.isnan(tc_l[0]) else 0]])
    r_d = np.concatenate([np.where(np.isnan(tc_d), 0, tc_d), [tc_d[0] if not np.isnan(tc_d[0]) else 0]])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_l, theta=np.rad2deg(theta_ld),
        mode="lines", fill="toself",
        fillcolor="rgba(255, 200, 50, 0.2)",
        line=dict(color="gold", width=2),
        name="Light ON",
    ))
    fig.add_trace(go.Scatterpolar(
        r=r_d, theta=np.rad2deg(theta_ld),
        mode="lines", fill="toself",
        fillcolor="rgba(100, 100, 200, 0.2)",
        line=dict(color="slateblue", width=2),
        name="Light OFF",
    ))
    fig.update_layout(
        height=400,
        polar=dict(angularaxis=dict(direction="clockwise", rotation=90)),
        title=f"Light vs Dark Tuning (r={ld['correlation']:.3f})",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "In real data, visual cue removal (lights off) may cause HD tuning to drift "
        "or become less sharp. This is the key manipulation in the hm2p experiment: "
        "testing whether Penk+ and non-Penk CamKII+ RSP neurons differ in their "
        "reliance on visual vs path-integration cues."
    )

# Footer
st.markdown("---")
st.caption(
    "Stability analysis compares HD tuning across time periods using Pearson "
    "correlation of occupancy-normalised tuning curves and circular preferred "
    "direction shift."
)
