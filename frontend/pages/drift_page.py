"""Drift Analysis — PD drift during darkness and visual cue removal.

Core science question: do HD cells maintain their preferred direction
when visual landmarks are removed (lights off), or does the internal
compass drift? Compare Penk vs non-Penk drift rates.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from hm2p.analysis.stability import (
    dark_drift_rate,
    drift_per_epoch,
    light_dark_stability,
    sliding_window_stability,
)
from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length

st.title("Drift Analysis")
st.caption(
    "Track preferred direction drift during light/dark cycles. "
    "Key question: does HD tuning drift when visual cues are removed?"
)


def _make_cell_with_dark_drift(
    n_frames=9000, pref=90.0, kappa=3.0, dark_drift_deg=0.0,
    noise=0.15, cycle_frames=1800, seed=42,
):
    """Generate HD cell with optional drift during dark epochs."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n_frames)) % 360.0
    theta = np.deg2rad(hd)
    signal = np.zeros(n_frames)

    # Build light_on mask
    light_on = np.zeros(n_frames, dtype=bool)
    for start in range(0, n_frames, 2 * cycle_frames):
        light_on[start:min(start + cycle_frames, n_frames)] = True

    # Track cumulative drift within dark epochs
    current_pref = pref
    drift_per_frame = dark_drift_deg / cycle_frames
    for i in range(n_frames):
        if not light_on[i]:
            current_pref += drift_per_frame
        else:
            current_pref = pref  # Snap back to anchored PD in light
        signal[i] = 0.1 + np.exp(kappa * np.cos(theta[i] - np.deg2rad(current_pref)))

    signal /= signal.max()
    signal += rng.normal(0, noise, n_frames)
    signal = np.clip(signal, 0, None)
    mask = np.ones(n_frames, dtype=bool)
    return signal, hd, mask, light_on


# Sidebar controls
st.sidebar.header("Cell Parameters")
kappa = st.sidebar.slider("κ (concentration)", 0.5, 8.0, 3.0, 0.5, key="drift_kappa")
noise = st.sidebar.slider("Noise σ", 0.05, 0.5, 0.15, 0.05, key="drift_noise")
dark_drift = st.sidebar.slider("Dark drift (°/epoch)", 0.0, 90.0, 20.0, 5.0, key="drift_deg")
n_frames = st.sidebar.slider("Frames", 6000, 18000, 9000, 1800, key="drift_n")
cycle_s = st.sidebar.slider("Cycle duration (s)", 30, 120, 60, 10, key="drift_cycle")
fps = 30.0
cycle_frames = int(cycle_s * fps)

signal, hd, mask, light_on = _make_cell_with_dark_drift(
    n_frames=n_frames, kappa=kappa, noise=noise,
    dark_drift_deg=dark_drift, cycle_frames=cycle_frames,
)

tab_epoch, tab_drift_rate, tab_compare, tab_ld = st.tabs([
    "Epoch PD Tracking", "Drift Rate", "Light vs Dark Tuning", "Sliding Window",
])

with tab_epoch:
    st.subheader("PD Drift Across Light/Dark Epochs")
    result = drift_per_epoch(signal, hd, mask, light_on)

    if result["n_epochs"] > 0:
        col1, col2 = st.columns(2)
        with col1:
            # PD per epoch
            colors = ["gold" if il else "midnightblue" for il in result["epoch_is_light"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=result["epoch_centers"] / fps,
                y=result["epoch_pds"],
                mode="markers+lines",
                marker=dict(color=colors, size=10),
                line=dict(color="gray", width=1),
                name="PD",
            ))
            fig.update_layout(
                height=300, title="Preferred Direction per Epoch",
                xaxis_title="Time (s)", yaxis_title="PD (°)",
                yaxis=dict(range=[0, 360]),
            )
            # Add light/dark background shading
            for start in range(0, n_frames, 2 * cycle_frames):
                fig.add_vrect(
                    x0=start / fps, x1=min(start + cycle_frames, n_frames) / fps,
                    fillcolor="yellow", opacity=0.1, line_width=0,
                )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Cumulative drift
            fig = go.Figure(data=[go.Scatter(
                x=result["epoch_centers"] / fps,
                y=result["cumulative_drift"],
                mode="markers+lines",
                marker=dict(color=colors, size=10),
                line=dict(color="gray", width=1),
            )])
            fig.update_layout(
                height=300, title="Cumulative PD Drift",
                xaxis_title="Time (s)", yaxis_title="Cumulative drift (°)",
            )
            for start in range(0, n_frames, 2 * cycle_frames):
                fig.add_vrect(
                    x0=start / fps, x1=min(start + cycle_frames, n_frames) / fps,
                    fillcolor="yellow", opacity=0.1, line_width=0,
                )
            st.plotly_chart(fig, use_container_width=True)

        # MVL per epoch
        fig = go.Figure(data=[go.Bar(
            x=[f"Epoch {i+1}" for i in range(result["n_epochs"])],
            y=result["epoch_mvls"],
            marker_color=colors,
        )])
        fig.update_layout(
            height=250, title="MVL per Epoch",
            xaxis_title="Epoch", yaxis_title="MVL",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data for epoch analysis.")

with tab_drift_rate:
    st.subheader("Drift Rate: Light vs Dark")

    dr = dark_drift_rate(
        signal, hd, mask, light_on, fps=fps,
        window_frames=int(cycle_frames * 0.6), step_frames=int(cycle_frames * 0.2),
    )

    col1, col2 = st.columns(2)
    col1.metric("Light drift rate", f"{dr['light_drift_deg_per_s']:.2f} °/s")
    col2.metric("Dark drift rate", f"{dr['dark_drift_deg_per_s']:.2f} °/s")

    if dark_drift > 0:
        ratio = dr["dark_drift_deg_per_s"] / max(dr["light_drift_deg_per_s"], 1e-10)
        st.markdown(f"**Dark/Light ratio:** {ratio:.1f}x")

    # PD trajectories
    fig = go.Figure()
    if len(dr["light_times_s"]) > 0:
        fig.add_trace(go.Scatter(
            x=dr["light_times_s"], y=dr["light_pds"],
            mode="markers+lines", name="Light",
            marker=dict(color="gold", size=6),
            line=dict(color="gold", width=1),
        ))
    if len(dr["dark_times_s"]) > 0:
        fig.add_trace(go.Scatter(
            x=dr["dark_times_s"], y=dr["dark_pds"],
            mode="markers+lines", name="Dark",
            marker=dict(color="midnightblue", size=6),
            line=dict(color="midnightblue", width=1),
        ))
    fig.update_layout(
        height=300, title="PD Trajectory (Sliding Windows)",
        xaxis_title="Time (s)", yaxis_title="PD (°)",
        yaxis=dict(range=[0, 360]),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_compare:
    st.subheader("Light vs Dark Tuning Curves")

    ld = light_dark_stability(signal, hd, mask, light_on)

    col1, col2, col3 = st.columns(3)
    col1.metric("Correlation", f"{ld['correlation']:.3f}")
    col2.metric("PD shift", f"{ld['pd_shift_deg']:.1f}°")
    col3.metric("MVL ratio", f"{ld['mvl_dark'] / max(ld['mvl_light'], 1e-10):.2f}")

    # Polar overlay
    bc = ld["bin_centers"]
    theta_plot = np.concatenate([np.deg2rad(bc), [np.deg2rad(bc[0])]])

    tc_light = ld["tuning_curve_light"]
    tc_dark = ld["tuning_curve_dark"]

    fig = go.Figure()
    if not np.all(np.isnan(tc_light)):
        fig.add_trace(go.Scatterpolar(
            r=np.concatenate([tc_light, [tc_light[0]]]),
            theta=np.rad2deg(theta_plot),
            mode="lines", line=dict(color="gold", width=3),
            name="Light",
        ))
    if not np.all(np.isnan(tc_dark)):
        fig.add_trace(go.Scatterpolar(
            r=np.concatenate([tc_dark, [tc_dark[0]]]),
            theta=np.rad2deg(theta_plot),
            mode="lines", line=dict(color="midnightblue", width=3),
            name="Dark",
        ))
    fig.update_layout(
        height=350,
        title="Tuning Curve Overlay",
        polar=dict(radialaxis=dict(showticklabels=False)),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_ld:
    st.subheader("Sliding Window MVL & PD")

    sw = sliding_window_stability(
        signal, hd, mask,
        window_frames=int(cycle_frames * 0.8),
        step_frames=int(cycle_frames * 0.2),
    )

    if sw["n_windows"] > 0:
        times = sw["window_centers"] / fps

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=sw["mvls"],
            mode="lines+markers", name="MVL",
            marker=dict(size=5),
        ))
        fig.update_layout(
            height=250, title="MVL Over Time",
            xaxis_title="Time (s)", yaxis_title="MVL",
        )
        for start in range(0, n_frames, 2 * cycle_frames):
            fig.add_vrect(
                x0=start / fps, x1=min(start + cycle_frames, n_frames) / fps,
                fillcolor="yellow", opacity=0.1, line_width=0,
            )
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=sw["preferred_dirs"],
            mode="lines+markers", name="PD",
            marker=dict(size=5),
        ))
        fig.update_layout(
            height=250, title="Preferred Direction Over Time",
            xaxis_title="Time (s)", yaxis_title="PD (°)",
            yaxis=dict(range=[0, 360]),
        )
        for start in range(0, n_frames, 2 * cycle_frames):
            fig.add_vrect(
                x0=start / fps, x1=min(start + cycle_frames, n_frames) / fps,
                fillcolor="yellow", opacity=0.1, line_width=0,
            )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption(
    "Yellow shading = light on. Dark blue = lights off (darkness). "
    "If HD cells rely on visual landmarks, PD should drift during darkness. "
    "Cells anchored by path integration should maintain stable PD."
)
