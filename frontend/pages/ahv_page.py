"""AHV Analysis — angular head velocity tuning and anticipatory time delay.

Visualizes how neural activity relates to angular head velocity: AHV tuning
curves, CW/CCW asymmetry, and anticipatory time delay estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

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


def _make_ahv_cell(n=5000, pref_hd=90.0, kappa=3.0, ahv_gain=0.002,
                    preferred_ahv=100.0, seed=42):
    """Generate synthetic cell with HD + AHV modulation."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 8, n)) % 360.0
    ahv = compute_ahv(hd, fps=30.0, smoothing_frames=3)
    theta = np.deg2rad(hd)
    # HD tuning
    signal = 0.1 + np.exp(kappa * np.cos(theta - np.deg2rad(pref_hd)))
    signal /= signal.max()
    # AHV modulation (Gaussian centered on preferred AHV)
    ahv_mod = 1 + ahv_gain * np.exp(-(ahv - preferred_ahv)**2 / (2 * 200**2))
    signal *= ahv_mod
    signal += rng.normal(0, 0.1, n)
    signal = np.clip(signal, 0, None)
    mask = np.ones(n, dtype=bool)
    return signal, hd, ahv, mask


# Controls
col1, col2, col3, col4 = st.columns(4)
with col1:
    pref_hd = st.slider("Preferred HD (°)", 0, 359, 90, 10, key="ahv_pref")
with col2:
    kappa = st.slider("HD tuning κ", 0.5, 8.0, 3.0, 0.5, key="ahv_kappa")
with col3:
    ahv_gain = st.slider("AHV modulation", 0.0, 0.01, 0.003, 0.001, key="ahv_gain")
with col4:
    pref_ahv = st.slider("Preferred AHV (°/s)", -300, 300, 100, 50, key="ahv_pref_ahv")

signal, hd, ahv, mask = _make_ahv_cell(
    pref_hd=pref_hd, kappa=kappa, ahv_gain=ahv_gain, preferred_ahv=pref_ahv,
)

tab_tuning, tab_atd, tab_summary = st.tabs(["AHV Tuning", "Time Delay", "Summary"])

with tab_tuning:
    st.subheader("AHV Tuning Curve")

    tc, bc = ahv_tuning_curve(signal, ahv, mask, n_bins=30)
    mod = ahv_modulation_index(tc, bc)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("CW mean", f"{mod['cw_mean']:.3f}")
    col_m2.metric("CCW mean", f"{mod['ccw_mean']:.3f}")
    col_m3.metric("Asymmetry", f"{mod['asymmetry_index']:.3f}")
    col_m4.metric("Preferred AHV", f"{mod['preferred_ahv']:.0f}°/s")

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
            xaxis_title="Angular Head Velocity (°/s)",
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
            xaxis_title="AHV (°/s)", yaxis_title="Count",
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
            xaxis_title="AHV (°/s)", yaxis_title="Signal",
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
