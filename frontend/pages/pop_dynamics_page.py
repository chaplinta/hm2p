"""Population Dynamics — PCA, ensemble structure, and HD ring topology.

Visualizes population-level properties of HD cell ensembles: dimensionality,
pairwise correlations, population vector structure, and temporal coherence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

st.title("Population Dynamics")
st.caption(
    "Population-level analysis: PCA dimensionality, pairwise correlations, "
    "population vector ring structure, and ensemble coherence."
)

import plotly.express as px
import plotly.graph_objects as go

from hm2p.analysis.population import (
    ensemble_coherence,
    pairwise_correlations,
    population_pca,
    population_vector_correlation,
)


def _make_hd_population(n_cells=15, n_frames=3000, kappa=3.0, noise=0.15, seed=42):
    """Generate HD-tuned population."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n_frames)) % 360.0
    theta = np.deg2rad(hd)
    prefs = np.linspace(0, 360, n_cells, endpoint=False)
    signals = np.zeros((n_cells, n_frames))
    for i in range(n_cells):
        signals[i] = 0.1 + np.exp(kappa * np.cos(theta - np.deg2rad(prefs[i])))
        signals[i] /= signals[i].max()
        signals[i] += rng.normal(0, noise, n_frames)
        signals[i] = np.clip(signals[i], 0, None)
    mask = np.ones(n_frames, dtype=bool)
    return signals, hd, mask, prefs


# Controls
col_c1, col_c2, col_c3 = st.columns(3)
with col_c1:
    n_cells = st.slider("Number of cells", 5, 40, 15, 5, key="popdyn_n")
with col_c2:
    kappa = st.slider("Tuning κ", 0.5, 8.0, 3.0, 0.5, key="popdyn_kappa")
with col_c3:
    noise = st.slider("Noise level", 0.05, 0.8, 0.15, 0.05, key="popdyn_noise")

signals, hd, mask, prefs = _make_hd_population(
    n_cells=n_cells, kappa=kappa, noise=noise,
)

tab_pca, tab_corr, tab_pv, tab_coherence = st.tabs([
    "PCA", "Correlations", "Pop. Vector", "Coherence",
])

# --- PCA ---
with tab_pca:
    st.subheader("PCA Dimensionality")

    pca = population_pca(signals)

    col1, col2 = st.columns(2)
    col1.metric("Components for 95% var", pca["n_components_95"])
    col2.metric("PC1 variance", f"{pca['explained_variance_ratio'][0]:.1%}")

    col_sc, col_ev = st.columns(2)
    with col_sc:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(min(10, len(pca['explained_variance_ratio'])))],
            y=pca["explained_variance_ratio"][:10],
            marker_color="royalblue",
        ))
        fig.update_layout(
            height=300, title="Scree Plot",
            xaxis_title="Component", yaxis_title="Explained Variance",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_ev:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(pca["cumulative_variance"]) + 1)),
            y=pca["cumulative_variance"],
            mode="lines+markers", marker_color="orange",
        ))
        fig.add_hline(y=0.95, line_dash="dash", line_color="red",
                      annotation_text="95%")
        fig.update_layout(
            height=300, title="Cumulative Variance",
            xaxis_title="Components", yaxis_title="Cumulative Variance",
        )
        st.plotly_chart(fig, use_container_width=True)

    # PC1 vs PC2 colored by HD
    if pca["components"].shape[0] >= 2:
        pc1 = pca["components"][0]
        pc2 = pca["components"][1]
        subsample = np.linspace(0, len(pc1) - 1, min(1000, len(pc1)), dtype=int)
        fig = go.Figure(data=[go.Scattergl(
            x=pc1[subsample], y=pc2[subsample],
            mode="markers",
            marker=dict(
                size=3, opacity=0.5,
                color=hd[subsample], colorscale="HSV",
                colorbar=dict(title="HD (°)"),
            ),
        )])
        fig.update_layout(
            height=400, title="PC1 vs PC2 (colored by HD)",
            xaxis_title=f"PC1 ({pca['explained_variance_ratio'][0]:.1%})",
            yaxis_title=f"PC2 ({pca['explained_variance_ratio'][1]:.1%})",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "If HD cells form a ring attractor, PC1 vs PC2 should show a **circular** "
            "trajectory colored by head direction."
        )

# --- Correlations ---
with tab_corr:
    st.subheader("Pairwise Correlations")

    corr = pairwise_correlations(signals)
    labels = [f"Cell {i+1}" for i in range(n_cells)]

    fig = px.imshow(
        corr, x=labels, y=labels,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Pairwise Correlation Matrix",
        aspect="equal",
    )
    fig.update_layout(height=max(300, n_cells * 20 + 100))
    st.plotly_chart(fig, use_container_width=True)

    # Correlation distribution
    upper = corr[np.triu_indices_from(corr, k=1)]
    fig = go.Figure(data=[go.Histogram(x=upper, nbinsx=30, marker_color="royalblue")])
    fig.update_layout(
        height=300, title=f"Off-Diagonal Correlation Distribution (mean={np.mean(upper):.3f})",
        xaxis_title="Pearson r", yaxis_title="Count",
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Population Vector ---
with tab_pv:
    st.subheader("Population Vector Correlation")
    st.markdown(
        "The population vector correlation matrix shows the similarity between "
        "population activity patterns at different head directions. HD cell "
        "populations should show a **banded diagonal** reflecting the circular "
        "topology of HD encoding."
    )

    pvc = population_vector_correlation(signals, hd, mask, n_bins=36)
    bin_labels = [f"{i*10}°" for i in range(36)]

    fig = px.imshow(
        pvc,
        x=bin_labels, y=bin_labels,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Population Vector Correlation (HD × HD)",
        aspect="equal",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# --- Coherence ---
with tab_coherence:
    st.subheader("Ensemble Coherence Over Time")

    win = st.slider("Window (frames)", 50, 500, 150, 25, key="coh_win")
    centers, coh = ensemble_coherence(signals, window_frames=win)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=centers, y=coh, mode="lines",
        line=dict(color="royalblue"),
    ))
    fig.update_layout(
        height=300,
        title=f"Mean Pairwise Correlation Over Time (window={win})",
        xaxis_title="Frame", yaxis_title="Mean r",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Mean coherence", f"{np.mean(coh):.3f}")
    st.metric("Coherence std", f"{np.std(coh):.3f}")

# Footer
st.markdown("---")
st.caption(
    "PCA uses SVD on mean-subtracted signals. Population vector correlation "
    "computes mean activity per HD bin then correlates between bins. "
    "Ensemble coherence tracks mean pairwise correlation over sliding windows."
)
