"""Cue Anchoring — HD re-anchoring after visual cue restoration.

Core science: when lights turn back on, how fast does the HD network
snap back to the visually anchored PD? Compares Penk vs non-Penk.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

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


def _make_cell(n_frames=12000, pref=90.0, kappa=3.0, drift=0.0,
               noise=0.15, cycle_frames=1800, seed=42):
    """Generate cell with drift in dark, snap-back in light."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n_frames)) % 360.0

    light_on = np.zeros(n_frames, dtype=bool)
    for start in range(0, n_frames, 2 * cycle_frames):
        light_on[start:min(start + cycle_frames, n_frames)] = True

    current_pref = pref
    drift_per_frame = drift / cycle_frames
    signal = np.zeros(n_frames)
    for i in range(n_frames):
        if not light_on[i]:
            current_pref += drift_per_frame
        else:
            current_pref = pref
        signal[i] = 0.1 + np.exp(kappa * np.cos(np.deg2rad(hd[i]) - np.deg2rad(current_pref)))

    signal /= signal.max()
    signal += rng.normal(0, noise, n_frames)
    signal = np.clip(signal, 0, None)
    mask = np.ones(n_frames, dtype=bool)
    return signal, hd, mask, light_on


# Parameters
st.sidebar.header("Cell")
kappa = st.sidebar.slider("κ", 0.5, 8.0, 3.5, 0.5, key="anch_kappa")
noise = st.sidebar.slider("Noise", 0.05, 0.5, 0.15, 0.05, key="anch_noise")
drift_deg = st.sidebar.slider("Dark drift (°/epoch)", 0.0, 90.0, 30.0, 5.0, key="anch_drift")
cycle_s = st.sidebar.slider("Cycle (s)", 30, 120, 60, 10, key="anch_cycle")

fps = 30.0
cycle_frames = int(cycle_s * fps)
n_frames = int(cycle_frames * 7)  # ~3.5 full cycles

signal, hd, mask, light_on = _make_cell(
    n_frames=n_frames, kappa=kappa, noise=noise,
    drift=drift_deg, cycle_frames=cycle_frames,
)

tab_tc, tab_speed, tab_compare = st.tabs(["Time Course", "Anchoring Speed", "Multi-Cell"])

with tab_tc:
    st.subheader("PD Deviation Around Dark→Light Transitions")

    result = anchoring_time_course(
        signal, hd, mask, light_on, fps=fps,
        window_frames=int(cycle_frames * 0.3),
        step_frames=int(cycle_frames * 0.05),
        pre_transition_s=cycle_s * 0.8,
        post_transition_s=cycle_s * 0.8,
    )

    col1, col2 = st.columns(2)
    col1.metric("Transitions averaged", result["n_transitions"])
    col2.metric("Reference PD", f"{result['reference_pd']:.1f}°")

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
            yaxis_title="|PD deviation| (°)",
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
        speed = anchoring_speed(result["pd_deviations"], result["time_offsets_s"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pre-deviation", f"{speed['pre_deviation']:.1f}°")
        col2.metric("Post-deviation", f"{speed['post_deviation']:.1f}°")
        col3.metric("Half-time", f"{speed['half_time_s']:.1f}s" if np.isfinite(speed['half_time_s']) else "N/A")
        col4.metric("Anchoring strength", f"{speed['anchoring_strength']:.2f}" if np.isfinite(speed['anchoring_strength']) else "N/A")

        st.markdown(
            "**Interpretation:** Anchoring strength near 1 means the cell fully "
            "re-anchors to the visual reference. Near 0 means the drift persists. "
            "Negative means the cell drifts *more* after lights come on."
        )

        # Visual summary
        if np.isfinite(speed["anchoring_strength"]):
            if speed["anchoring_strength"] > 0.7:
                st.success("Strong visual anchoring — PD snaps back to reference.")
            elif speed["anchoring_strength"] > 0.3:
                st.info("Moderate anchoring — partial re-alignment to visual cues.")
            else:
                st.warning("Weak anchoring — HD representation largely path-integration driven.")
    else:
        st.warning("No transitions found.")

with tab_compare:
    st.subheader("Multi-Cell Comparison")
    st.markdown(
        "Compare anchoring across cells with different drift amounts. "
        "Simulates what Penk vs non-Penk comparison would look like."
    )

    n_compare = st.slider("Cells to compare", 2, 8, 4, 1, key="anch_compare_n")

    strengths = []
    half_times = []
    drifts = np.linspace(5, 80, n_compare)

    for i, d in enumerate(drifts):
        sig, hd_c, mask_c, lo = _make_cell(
            n_frames=n_frames, kappa=kappa, noise=noise,
            drift=d, cycle_frames=cycle_frames, seed=i + 42,
        )
        tc_result = anchoring_time_course(
            sig, hd_c, mask_c, lo, fps=fps,
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

    labels = [f"Cell {i+1}\n(drift={d:.0f}°)" for i, d in enumerate(drifts)]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=strengths,
            marker_color=["green" if s > 0.5 else "orange" if s > 0.2 else "red"
                          for s in (s if np.isfinite(s) else 0 for s in strengths)],
        )])
        fig.update_layout(
            height=300, title="Anchoring Strength by Cell",
            yaxis_title="Anchoring strength",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(data=[go.Scatter(
            x=list(drifts), y=strengths,
            mode="markers+lines",
            marker=dict(size=10, color="royalblue"),
        )])
        fig.update_layout(
            height=300, title="Drift vs Anchoring Strength",
            xaxis_title="Dark drift (°/epoch)",
            yaxis_title="Anchoring strength",
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption(
    "Transition-aligned analysis averages PD deviation across all dark→light "
    "transitions. Gray shading = dark period before transition. "
    "Red dashed line = lights on."
)
