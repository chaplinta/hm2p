"""Gain Modulation — response amplitude changes between light and dark.

Visual cue removal may modulate not just preferred direction (drift) but
also the amplitude of HD tuning. Gain changes indicate direct visual input
to HD circuit, distinct from path integration drift.
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hm2p.analysis.gain import (
    epoch_gain_tracking,
    gain_modulation_index,
    population_gain_modulation,
)
from hm2p.analysis.tuning import compute_hd_tuning_curve

st.title("Gain Modulation")
st.caption(
    "Compare response amplitude between light and dark. "
    "Gain changes reveal direct visual modulation of HD tuning."
)


def _make_population(n_cells=10, n_frames=9000, kappa=3.0, noise=0.15,
                     dark_gain=1.0, cycle_frames=1800, seed=42):
    """Generate population with optional gain reduction in dark."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n_frames)) % 360.0
    theta = np.deg2rad(hd)
    prefs = np.linspace(0, 360, n_cells, endpoint=False)

    light_on = np.zeros(n_frames, dtype=bool)
    for start in range(0, n_frames, 2 * cycle_frames):
        light_on[start:min(start + cycle_frames, n_frames)] = True

    signals = np.zeros((n_cells, n_frames))
    for i in range(n_cells):
        k = np.clip(rng.normal(kappa, 0.5), 0.5, 10.0)
        signals[i] = 0.1 + np.exp(k * np.cos(theta - np.deg2rad(prefs[i])))
        signals[i] /= signals[i].max()
        # Apply gain reduction in dark
        cell_dark_gain = np.clip(rng.normal(dark_gain, 0.1), 0.1, 2.0)
        signals[i][~light_on] *= cell_dark_gain
        signals[i] += rng.normal(0, noise, n_frames)
        signals[i] = np.clip(signals[i], 0, None)

    mask = np.ones(n_frames, dtype=bool)
    return signals, hd, mask, light_on


# Parameters
st.sidebar.header("Population")
n_cells = st.sidebar.slider("Cells", 3, 15, 8, 1, key="gain_n")
kappa = st.sidebar.slider("κ", 0.5, 8.0, 3.0, 0.5, key="gain_kappa")
noise = st.sidebar.slider("Noise σ", 0.05, 0.5, 0.15, 0.05, key="gain_noise")
dark_gain = st.sidebar.slider("Dark gain", 0.2, 1.5, 0.7, 0.05, key="gain_dark")

signals, hd, mask, light_on = _make_population(
    n_cells=n_cells, kappa=kappa, noise=noise, dark_gain=dark_gain,
)

tab_pop, tab_single, tab_epoch = st.tabs(["Population", "Single Cell", "Epoch Tracking"])

with tab_pop:
    st.subheader("Population Gain Modulation")
    results = population_gain_modulation(signals, hd, mask, light_on)

    gains = [r["gain_index"] for r in results]
    peaks_l = [r["peak_light"] for r in results]
    peaks_d = [r["peak_dark"] for r in results]
    labels = [f"Cell {i+1}" for i in range(n_cells)]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(data=[go.Bar(
            x=labels, y=gains,
            marker_color=["green" if g > 0 else "red" for g in gains],
        )])
        fig.update_layout(
            height=300, title="Gain Modulation Index",
            xaxis_title="Cell", yaxis_title="GMI",
            yaxis=dict(range=[-1, 1]),
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Light", x=labels, y=peaks_l, marker_color="gold",
        ))
        fig.add_trace(go.Bar(
            name="Dark", x=labels, y=peaks_d, marker_color="midnightblue",
        ))
        fig.update_layout(
            height=300, title="Peak Response",
            xaxis_title="Cell", yaxis_title="Peak",
            barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Dynamic range comparison
    dr_l = [r["dynamic_range_light"] for r in results]
    dr_d = [r["dynamic_range_dark"] for r in results]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dr_l, y=dr_d, mode="markers+text",
        marker=dict(size=10, color="royalblue"),
        text=labels, textposition="top center",
    ))
    max_dr = max(max(dr_l), max(dr_d)) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_dr], y=[0, max_dr],
        mode="lines", line=dict(dash="dash", color="gray"),
        showlegend=False,
    ))
    fig.update_layout(
        height=300, title="Dynamic Range: Light vs Dark",
        xaxis_title="Light dynamic range",
        yaxis_title="Dark dynamic range",
    )
    st.plotly_chart(fig, use_container_width=True)

    mean_gmi = np.mean(gains)
    st.markdown(
        f"**Mean GMI:** {mean_gmi:.3f} — "
        f"{'Light > Dark' if mean_gmi > 0.05 else 'Dark > Light' if mean_gmi < -0.05 else 'Similar gain'}"
    )

with tab_single:
    st.subheader("Single Cell Detail")
    cell_idx = st.selectbox("Cell", range(n_cells),
                            format_func=lambda x: f"Cell {x+1}", key="gain_cell")

    result = results[cell_idx]
    col1, col2, col3 = st.columns(3)
    col1.metric("Gain Index", f"{result['gain_index']:.3f}")
    col2.metric("Peak (Light)", f"{result['peak_light']:.3f}")
    col3.metric("Peak (Dark)", f"{result['peak_dark']:.3f}")

    # Tuning curves overlay
    mask_light = mask & light_on
    mask_dark = mask & ~light_on
    tc_l, bc = compute_hd_tuning_curve(signals[cell_idx], hd, mask_light)
    tc_d, _ = compute_hd_tuning_curve(signals[cell_idx], hd, mask_dark)

    theta_plot = np.concatenate([np.deg2rad(bc), [np.deg2rad(bc[0])]])
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=np.concatenate([tc_l, [tc_l[0]]]),
        theta=np.rad2deg(theta_plot),
        mode="lines", line=dict(color="gold", width=3), name="Light",
    ))
    if not np.all(np.isnan(tc_d)):
        fig.add_trace(go.Scatterpolar(
            r=np.concatenate([tc_d, [tc_d[0]]]),
            theta=np.rad2deg(theta_plot),
            mode="lines", line=dict(color="midnightblue", width=3), name="Dark",
        ))
    fig.update_layout(
        height=350, title=f"Cell {cell_idx+1} — Light vs Dark Tuning",
        polar=dict(radialaxis=dict(showticklabels=True)),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_epoch:
    st.subheader("Gain Tracking Across Epochs")
    cell_idx2 = st.selectbox("Cell", range(n_cells),
                             format_func=lambda x: f"Cell {x+1}", key="gain_epoch_cell")

    et = epoch_gain_tracking(signals[cell_idx2], hd, mask, light_on)
    fps = 30.0

    if et["n_epochs"] > 0:
        times = et["epoch_centers"] / fps
        colors = ["gold" if il else "midnightblue" for il in et["epoch_is_light"]]

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(data=[go.Scatter(
                x=times, y=et["epoch_peaks"],
                mode="markers+lines", marker=dict(color=colors, size=10),
                line=dict(color="gray", width=1),
            )])
            fig.update_layout(
                height=250, title="Peak Response per Epoch",
                xaxis_title="Time (s)", yaxis_title="Peak",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(data=[go.Scatter(
                x=times, y=et["epoch_dynamic_ranges"],
                mode="markers+lines", marker=dict(color=colors, size=10),
                line=dict(color="gray", width=1),
            )])
            fig.update_layout(
                height=250, title="Dynamic Range per Epoch",
                xaxis_title="Time (s)", yaxis_title="Dynamic Range",
            )
            st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure(data=[go.Scatter(
            x=times, y=et["epoch_mvls"],
            mode="markers+lines", marker=dict(color=colors, size=10),
            line=dict(color="gray", width=1),
        )])
        fig.update_layout(
            height=250, title="MVL per Epoch",
            xaxis_title="Time (s)", yaxis_title="MVL",
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption(
    "GMI = (peak_light - peak_dark) / (peak_light + peak_dark). "
    "Positive = higher gain in light. A gain reduction in dark suggests "
    "visual input directly modulates HD tuning amplitude."
)
