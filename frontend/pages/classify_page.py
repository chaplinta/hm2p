"""Cell Classification — automated HD cell identification.

Combines MVL, shuffle significance, split-half reliability, and mutual
information to classify cells as HD-tuned or non-HD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hm2p.analysis.classify import (
    classification_summary_table,
    classify_population,
)
from hm2p.analysis.tuning import compute_hd_tuning_curve

st.title("Cell Classification")
st.caption(
    "Automated HD cell identification using MVL, shuffle significance, "
    "split-half reliability, and mutual information."
)


def _make_population(n_hd=6, n_noise=6, n_frames=3000, kappa=3.0,
                     noise=0.2, seed=42):
    """Generate a mixed population of HD and non-HD cells."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n_frames)) % 360.0
    theta = np.deg2rad(hd)
    n_total = n_hd + n_noise
    signals = np.zeros((n_total, n_frames))

    # HD-tuned cells with varying preferences
    prefs = np.linspace(0, 360, n_hd, endpoint=False)
    for i in range(n_hd):
        kappas_i = np.clip(rng.normal(kappa, 0.8), 0.5, 10.0)
        signals[i] = 0.1 + np.exp(kappas_i * np.cos(theta - np.deg2rad(prefs[i])))
        signals[i] /= signals[i].max()
        signals[i] += rng.normal(0, noise, n_frames)
        signals[i] = np.clip(signals[i], 0, None)

    # Non-HD noise cells
    for i in range(n_hd, n_total):
        signals[i] = np.abs(rng.normal(1, 0.5, n_frames))

    mask = np.ones(n_frames, dtype=bool)
    return signals, hd, mask


# Parameters
st.sidebar.header("Population")
n_hd = st.sidebar.slider("HD cells", 2, 15, 6, 1, key="cls_nhd")
n_noise = st.sidebar.slider("Noise cells", 2, 15, 6, 1, key="cls_nnoise")
kappa = st.sidebar.slider("Mean κ", 0.5, 8.0, 3.0, 0.5, key="cls_kappa")
noise = st.sidebar.slider("Noise σ", 0.05, 0.8, 0.2, 0.05, key="cls_noise")

st.sidebar.header("Thresholds")
mvl_thresh = st.sidebar.slider("MVL threshold", 0.05, 0.5, 0.15, 0.01, key="cls_mvl")
p_thresh = st.sidebar.slider("p-value threshold", 0.001, 0.1, 0.05, 0.005,
                              key="cls_p", format="%.3f")
rel_thresh = st.sidebar.slider("Reliability threshold", 0.1, 0.9, 0.5, 0.05,
                                key="cls_rel")
n_shuffles = st.sidebar.slider("Shuffles", 100, 1000, 300, 50, key="cls_shuf")

signals, hd, mask = _make_population(n_hd=n_hd, n_noise=n_noise,
                                      kappa=kappa, noise=noise)

with st.spinner("Classifying cells..."):
    pop = classify_population(
        signals, hd, mask,
        mvl_threshold=mvl_thresh,
        p_threshold=p_thresh,
        reliability_threshold=rel_thresh,
        n_shuffles=n_shuffles,
        rng=np.random.default_rng(42),
    )

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cells", n_hd + n_noise)
col2.metric("HD Cells", pop["n_hd"])
col3.metric("Non-HD Cells", pop["n_non_hd"])
col4.metric("HD Fraction", f"{pop['fraction_hd']:.1%}")

tab_table, tab_scatter, tab_tuning = st.tabs(["Summary Table", "Metric Scatter", "Tuning Curves"])

with tab_table:
    table = classification_summary_table(pop)
    df = pd.DataFrame(table)
    df["cell"] = df["cell"].apply(lambda x: f"Cell {x+1}")
    df["mvl"] = df["mvl"].apply(lambda x: f"{x:.3f}")
    df["p_value"] = df["p_value"].apply(lambda x: f"{x:.4f}")
    df["reliability"] = df["reliability"].apply(lambda x: f"{x:.3f}")
    df["mi"] = df["mi"].apply(lambda x: f"{x:.4f}")
    df["preferred_direction"] = df["preferred_direction"].apply(lambda x: f"{x:.1f}°")

    # Color HD vs non-HD
    def _highlight_hd(row):
        if row["is_hd"]:
            return ["background-color: rgba(0, 180, 0, 0.15)"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_highlight_hd, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown(
        "**Grades:** A = strong HD (MVL≥0.4, reliability≥0.8) · "
        "B = moderate HD (MVL≥0.25) · C = weak HD · D = non-HD"
    )

with tab_scatter:
    # MVL vs p-value scatter
    cells = pop["cells"]
    mvls = [c["mvl"] for c in cells]
    pvals = [c["p_value"] for c in cells]
    reliabilities = [c["reliability"] for c in cells]
    mis = [c["mi"] for c in cells]
    labels = [f"Cell {i+1}" for i in range(len(cells))]
    is_hd = [c["is_hd"] for c in cells]
    colors = ["HD" if h else "Non-HD" for h in is_hd]

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.scatter(
            x=mvls, y=pvals, color=colors, text=labels,
            labels={"x": "MVL", "y": "p-value", "color": "Class"},
            title="MVL vs Significance",
            color_discrete_map={"HD": "green", "Non-HD": "red"},
        )
        fig.add_hline(y=p_thresh, line_dash="dash", line_color="gray",
                      annotation_text=f"p={p_thresh}")
        fig.add_vline(x=mvl_thresh, line_dash="dash", line_color="gray",
                      annotation_text=f"MVL={mvl_thresh}")
        fig.update_yaxes(type="log")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.scatter(
            x=mvls, y=reliabilities, color=colors, text=labels,
            labels={"x": "MVL", "y": "Split-half r", "color": "Class"},
            title="MVL vs Reliability",
            color_discrete_map={"HD": "green", "Non-HD": "red"},
        )
        fig.add_hline(y=rel_thresh, line_dash="dash", line_color="gray",
                      annotation_text=f"r={rel_thresh}")
        fig.add_vline(x=mvl_thresh, line_dash="dash", line_color="gray")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # MI vs MVL
    fig = px.scatter(
        x=mvls, y=mis, color=colors, text=labels,
        labels={"x": "MVL", "y": "MI (bits)", "color": "Class"},
        title="MVL vs Mutual Information",
        color_discrete_map={"HD": "green", "Non-HD": "red"},
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

with tab_tuning:
    st.subheader("Tuning Curves by Classification")

    n_total = len(cells)
    hd_idx = pop["hd_indices"]
    non_hd_idx = [i for i in range(n_total) if i not in hd_idx]

    col_hd, col_non = st.columns(2)
    with col_hd:
        st.markdown("**HD Cells**")
        if hd_idx:
            for idx in hd_idx:
                tc, bc = compute_hd_tuning_curve(signals[idx], hd, mask, n_bins=36)
                theta_plot = np.concatenate([np.deg2rad(bc), [np.deg2rad(bc[0])]])
                r_plot = np.concatenate([tc, [tc[0]]])
                fig = go.Figure(data=[go.Scatterpolar(
                    r=r_plot, theta=np.rad2deg(theta_plot),
                    mode="lines", line=dict(color="green", width=2),
                    name=f"Cell {idx+1}",
                )])
                cell_info = cells[idx]
                fig.update_layout(
                    height=220, margin=dict(l=30, r=30, t=40, b=20),
                    title=f"Cell {idx+1} (MVL={cell_info['mvl']:.3f}, grade={classification_summary_table(pop)[idx]['grade']})",
                    polar=dict(radialaxis=dict(showticklabels=False)),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No HD cells detected.")

    with col_non:
        st.markdown("**Non-HD Cells**")
        for idx in non_hd_idx[:6]:  # Limit display
            tc, bc = compute_hd_tuning_curve(signals[idx], hd, mask, n_bins=36)
            theta_plot = np.concatenate([np.deg2rad(bc), [np.deg2rad(bc[0])]])
            r_plot = np.concatenate([tc, [tc[0]]])
            fig = go.Figure(data=[go.Scatterpolar(
                r=r_plot, theta=np.rad2deg(theta_plot),
                mode="lines", line=dict(color="red", width=2),
                name=f"Cell {idx+1}",
            )])
            fig.update_layout(
                height=220, margin=dict(l=30, r=30, t=40, b=20),
                title=f"Cell {idx+1} (MVL={cells[idx]['mvl']:.3f})",
                polar=dict(radialaxis=dict(showticklabels=False)),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption(
    "Classification uses three criteria: (1) MVL exceeds threshold, "
    "(2) shuffle test p-value below threshold, (3) split-half reliability "
    "above threshold. All three must pass for HD classification."
)
