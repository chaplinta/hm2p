"""AHV Analysis — angular head velocity tuning and anticipatory time delay.

Visualizes how neural activity relates to angular head velocity: AHV tuning
curves, CW/CCW asymmetry, and anticipatory time delay estimation.

Requires real sync.h5 data from S3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

st.title("AHV Analysis")
st.caption(
    "Angular head velocity tuning — how cell activity relates to rotational "
    "speed and direction. Anticipatory time delay estimation."
)

import plotly.graph_objects as go

from hm2p.analysis.ahv import (
    ahv_modulation_index,
    ahv_tuning_curve,
    anticipatory_time_delay,
    compute_ahv,
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
    except Exception as e:
        st.warning(f"Could not load sync data: {e}")
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
                                key="ahv_session")
ses_data = real_sessions[sel_session_idx]
n_rois = ses_data["n_rois"]
sel_cell = st.slider("Cell index", 0, max(0, n_rois - 1), 0, key="ahv_cell")

signal = ses_data["dff"][sel_cell]
hd = ses_data["hd_deg"]
mask = ses_data["active"] & ~ses_data["bad_behav"]
ahv = compute_ahv(hd, fps=30.0, smoothing_frames=3)

tab_tuning, tab_atd, tab_summary = st.tabs(["AHV Tuning", "Time Delay", "Summary"])

with tab_tuning:
    st.subheader("AHV Tuning Curve")

    tc, bc = ahv_tuning_curve(signal, ahv, mask, n_bins=30)
    mod = ahv_modulation_index(tc, bc)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("CW mean", f"{mod['cw_mean']:.3f}")
    col_m2.metric("CCW mean", f"{mod['ccw_mean']:.3f}")
    col_m3.metric("Asymmetry", f"{mod['asymmetry_index']:.3f}")
    col_m4.metric("Preferred AHV", f"{mod['preferred_ahv']:.0f} deg/s")

    col_tc, col_hist = st.columns(2)
    with col_tc:
        tc_plot = np.where(np.isnan(tc), 0, tc)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bc, y=tc_plot, mode="lines+markers",
            line=dict(color="royalblue", width=2),
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=mod["preferred_ahv"], line_dash="dot", line_color="red",
                      annotation_text=f"Pref={mod['preferred_ahv']:.0f}")
        fig.update_layout(
            height=350, title="AHV Tuning Curve",
            xaxis_title="Angular Head Velocity (deg/s)",
            yaxis_title="Mean Signal",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_hist:
        fig = go.Figure(data=[go.Histogram(
            x=ahv[mask], nbinsx=50, marker_color="lightblue",
        )])
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            height=350, title="AHV Distribution",
            xaxis_title="AHV (deg/s)", yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Signal vs AHV scatter
    with st.expander("Raw Data"):
        subsample = np.linspace(0, len(signal) - 1, min(2000, len(signal)), dtype=int)
        fig = go.Figure(data=[go.Scattergl(
            x=ahv[subsample], y=signal[subsample],
            mode="markers", marker=dict(size=2, opacity=0.3, color="gray"),
        )])
        fig.add_trace(go.Scatter(
            x=bc, y=tc_plot, mode="lines",
            line=dict(color="royalblue", width=3), name="Tuning curve",
        ))
        fig.update_layout(
            height=300, title="Signal vs AHV",
            xaxis_title="AHV (deg/s)", yaxis_title="Signal",
        )
        st.plotly_chart(fig, use_container_width=True)


with tab_atd:
    st.subheader("Anticipatory Time Delay")
    st.markdown(
        "Tests whether the cell's HD tuning leads or lags the actual head direction. "
        "Positive delay = anticipatory (cell fires before the head reaches preferred direction)."
    )

    max_lag = st.slider("Max lag (frames)", 3, 20, 8, 1, key="atd_lag")

    atd = anticipatory_time_delay(signal, hd, mask, max_lag_frames=max_lag, fps=30.0)

    col_a1, col_a2 = st.columns(2)
    col_a1.metric("Best lag", f"{atd['best_lag_ms']:.1f} ms")
    col_a2.metric("Best MVL", f"{atd['best_mvl']:.4f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=atd["lags_ms"], y=atd["mvls"],
        mode="lines+markers", marker_color="royalblue",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=atd["best_lag_ms"], line_dash="dot", line_color="red",
                  annotation_text=f"Best={atd['best_lag_ms']:.0f}ms")
    fig.update_layout(
        height=350,
        title="MVL vs Time Lag (positive = anticipatory)",
        xaxis_title="Lag (ms)", yaxis_title="MVL",
    )
    st.plotly_chart(fig, use_container_width=True)


with tab_summary:
    st.subheader("AHV Analysis Summary")

    st.markdown("""
    **Angular Head Velocity (AHV) analysis characterises:**

    1. **AHV tuning** — How does activity vary with rotation speed/direction?
       - CW/CCW asymmetry index reveals directional preference
       - Modulation depth indicates strength of AHV modulation

    2. **Anticipatory Time Delay (ATD)** — Does the cell predict head direction?
       - Positive ATD: cell fires before head reaches preferred direction
       - Typical HD cells: 20-40ms anticipatory (Taube & Muller, 1998)
       - RSP cells may show different ATD than classic HD cells in PoS

    3. **Clinical relevance:**
       - Penk+ vs non-Penk cells may differ in AHV sensitivity
       - Light/dark manipulation may alter AHV modulation
       - AHV modulation may reflect path integration contributions
    """)

# Footer
st.markdown("---")
st.caption(
    "AHV analysis methods: Taube & Muller (1998), Blair & Sharp (1995). "
    "ATD estimation: MVL maximisation over time-shifted tuning curves."
)
