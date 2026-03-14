"""Population Dynamics — PCA, ensemble structure, and HD ring topology.

Visualizes population-level properties of HD cell ensembles: dimensionality,
pairwise correlations, population vector structure, and temporal coherence.
All data loaded from real sync.h5 files on S3.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hm2p.analysis.population import (
    ensemble_coherence,
    pairwise_correlations,
    population_pca,
    population_vector_correlation,
)

log = logging.getLogger("hm2p.frontend.pop_dynamics")

st.title("Population Dynamics")
st.caption(
    "Population-level analysis: PCA dimensionality, pairwise correlations, "
    "population vector ring structure, and ensemble coherence."
)

# ── Load real data ────────────────────────────────────────────────────────

try:
    from frontend.data import load_all_sync_data, session_filter_sidebar
except ImportError:
    st.error("Frontend data module not available.")
    st.stop()

try:
    all_data = load_all_sync_data()
except Exception as e:
    log.warning("Could not load sync data: %s", e)
    st.warning("Could not load sync data from S3. Check server logs.")
    st.stop()

if all_data["n_sessions"] == 0:
    st.warning(
        "No sync data available yet. This page will populate automatically "
        "when Stage 5 (sync) completes."
    )
    st.stop()

sessions = session_filter_sidebar(all_data["sessions"])

if not sessions:
    st.warning("No sessions match the current filter.")
    st.stop()

# Session selector
exp_ids = [s["exp_id"] for s in sessions]
selected_exp = st.selectbox("Session", exp_ids, key="popdyn_session")
ses_data = next(s for s in sessions if s["exp_id"] == selected_exp)

signals = ses_data["dff"]
hd = ses_data["hd_deg"]
mask = ses_data["active"] & ~ses_data["bad_behav"]
n_cells = ses_data["n_rois"]
n_frames = ses_data["n_frames"]

st.success(f"Loaded {n_cells} cells, {n_frames} frames from {selected_exp}")

if n_cells < 3:
    st.warning("Need at least 3 cells for population dynamics analysis.")
    st.stop()

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
                colorbar=dict(title="HD (deg)"),
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
    labels = [f"Cell {i}" for i in range(n_cells)]

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
        height=300, title=f"Off-Diagonal Correlation Distribution (mean={np.nanmean(upper):.3f})",
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
    bin_labels = [f"{i*10}" for i in range(36)]

    fig = px.imshow(
        pvc,
        x=bin_labels, y=bin_labels,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Population Vector Correlation (HD x HD)",
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

    if len(coh) > 0:
        st.metric("Mean coherence", f"{np.nanmean(coh):.3f}")
        st.metric("Coherence std", f"{np.nanstd(coh):.3f}")

# Footer
st.markdown("---")
st.caption(
    "PCA uses SVD on mean-subtracted signals. Population vector correlation "
    "computes mean activity per HD bin then correlates between bins. "
    "Ensemble coherence tracks mean pairwise correlation over sliding windows."
)
