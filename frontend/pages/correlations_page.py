"""Correlations & Ensembles — pairwise neural correlations and ensemble detection."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    download_s3_bytes,
    load_animals,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.correlations")

st.title("Correlations & Ensembles")
st.caption("Pairwise neural correlations, ensemble detection, and dimensionality analysis.")

# --- Session selector ---
experiments = load_experiments()
animals = load_animals()
animal_map = {a["animal_id"]: a for a in animals}

exp_ids = [e["exp_id"] for e in experiments]
selected = st.selectbox(
    "Session",
    exp_ids,
    format_func=lambda x: f"{x} ({animal_map.get(x.split('_')[-1], {}).get('celltype', '?')})",
    key="corr_session",
)

if not selected:
    st.stop()

sub, ses = parse_session_id(selected)
animal_id = selected.split("_")[-1]
celltype = animal_map.get(animal_id, {}).get("celltype", "?")


@st.cache_data(ttl=300)
def load_corr_data(sub: str, ses: str) -> dict | None:
    """Load ca.h5 for correlation analysis."""
    import h5py

    data = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
    if data is None:
        return None

    f = h5py.File(io.BytesIO(data), "r")
    result = {
        "dff": f["dff"][:],
        "fps": float(f.attrs.get("fps_imaging", 9.8)),
    }
    if "event_masks" in f:
        result["event_masks"] = f["event_masks"][:]
    if "spks" in f:
        result["spks"] = f["spks"][:]
    f.close()
    return result


with st.spinner("Loading calcium data..."):
    data = load_corr_data(sub, ses)

if data is None:
    st.warning("No calcium data found.")
    st.stop()

dff = data["dff"]
n_rois, n_frames = dff.shape
fps = data["fps"]

st.markdown(f"**{sub} / {ses}** — {celltype} — {n_rois} ROIs, {n_frames/fps:.0f}s")

# Signal selection
signal_options = ["dff"]
if "spks" in data:
    signal_options.append("deconv")
if "event_masks" in data:
    signal_options.append("events")

_signal_labels = {"dff": "dF/F\u2080", "deconv": "Deconv", "events": "Events"}

col1, col2 = st.columns(2)
with col1:
    signal_type = st.selectbox("Signal", signal_options, format_func=lambda x: _signal_labels.get(x, x), key="corr_signal")
with col2:
    smooth_window = st.slider("Smoothing (frames)", 0, 30, 5, key="corr_smooth")

# Get signal
if signal_type == "dff":
    signal = dff.copy()
elif signal_type == "deconv":
    signal = data["spks"].copy()
else:
    signal = data["event_masks"].astype(np.float32)

# Optional smoothing
if smooth_window > 1:
    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.zeros_like(signal)
    for i in range(n_rois):
        smoothed[i] = np.convolve(signal[i], kernel, mode="same")
    signal = smoothed

# --- Tabs ---
tab_matrix, tab_pca, tab_ensembles = st.tabs([
    "Correlation Matrix", "PCA / Dimensionality", "Co-activation",
])

import plotly.express as px
import plotly.graph_objects as go

# --- Tab 1: Correlation matrix ---
with tab_matrix:
    st.subheader("Pairwise Correlation Matrix")

    # Compute correlation matrix
    corr_matrix = np.corrcoef(signal)
    np.fill_diagonal(corr_matrix, np.nan)

    # Hierarchical clustering for ordering
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    # Replace NaN diagonal for clustering
    corr_for_cluster = corr_matrix.copy()
    np.fill_diagonal(corr_for_cluster, 1.0)

    # Use 1-correlation as distance
    dist = 1 - corr_for_cluster
    dist = np.clip(dist, 0, 2)  # Ensure valid distance
    np.fill_diagonal(dist, 0)

    try:
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="ward")
        order = leaves_list(Z)
        reordered = corr_matrix[np.ix_(order, order)]
        use_clustering = True
    except Exception:
        reordered = corr_matrix
        order = np.arange(n_rois)
        use_clustering = False

    col1, col2 = st.columns([1, 1])
    with col1:
        cluster_order = st.checkbox("Hierarchical clustering order", value=True, key="corr_cluster")

    with st.expander("About ensembles & clustering"):
        st.markdown(
            "**Ensembles** are groups of neurons that tend to be co-active (fire together), "
            "suggesting they may be part of a functional circuit or encode similar information "
            "(e.g. a shared head-direction preference). "
            "**Hierarchical clustering** groups ROIs by similarity of their activity patterns "
            "(pairwise correlation), building a tree (dendrogram) where nearby branches are "
            "more correlated. Reordering the matrix by this tree reveals ensemble structure as "
            "bright blocks along the diagonal."
        )

    display_matrix = reordered if (cluster_order and use_clustering) else corr_matrix

    fig = go.Figure(data=go.Heatmap(
        z=display_matrix,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-0.5,
        zmax=0.5,
        colorbar=dict(title="r"),
        hovertemplate="ROI %{x} vs ROI %{y}<br>r = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=600,
        width=600,
        title="Pairwise Correlation Matrix" + (" (clustered)" if cluster_order else ""),
        xaxis_title="ROI",
        yaxis_title="ROI",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distribution of correlations
    upper_tri = corr_matrix[np.triu_indices(n_rois, k=1)]
    upper_tri = upper_tri[~np.isnan(upper_tri)]

    fig2 = px.histogram(
        x=upper_tri, nbins=50,
        title=f"Distribution of Pairwise Correlations (n={len(upper_tri)})",
        labels={"x": "Pearson r", "y": "Count"},
    )
    fig2.add_vline(x=0, line_dash="dash", line_color="gray")
    fig2.add_vline(x=np.median(upper_tri), line_dash="dash", line_color="red",
                   annotation_text=f"median={np.median(upper_tri):.3f}")
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean r", f"{np.nanmean(upper_tri):.3f}")
    col2.metric("Median r", f"{np.nanmedian(upper_tri):.3f}")
    col3.metric("% positive", f"{(upper_tri > 0).mean() * 100:.0f}%")
    col4.metric("% |r| > 0.3", f"{(np.abs(upper_tri) > 0.3).mean() * 100:.1f}%")


# --- Tab 2: PCA ---
with tab_pca:
    st.subheader("PCA — Population Dimensionality")

    from sklearn.decomposition import PCA

    # Z-score each ROI
    z_signal = (signal - signal.mean(axis=1, keepdims=True)) / (signal.std(axis=1, keepdims=True) + 1e-10)

    max_components = min(n_rois, 20)
    pca = PCA(n_components=max_components)
    scores = pca.fit_transform(z_signal.T)  # (n_frames, n_components)

    # Scree plot
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=np.arange(1, max_components + 1),
            y=pca.explained_variance_ratio_ * 100,
            name="Individual",
            marker_color="steelblue",
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(1, max_components + 1),
            y=np.cumsum(pca.explained_variance_ratio_) * 100,
            name="Cumulative",
            mode="lines+markers",
            line=dict(color="red"),
        ))
        fig.update_layout(
            height=350,
            title="Explained Variance",
            xaxis_title="PC",
            yaxis_title="% Variance",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # How many PCs to reach 80%?
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_80 = np.searchsorted(cumvar, 0.8) + 1
        n_90 = np.searchsorted(cumvar, 0.9) + 1
        n_95 = np.searchsorted(cumvar, 0.95) + 1

        st.markdown("### Dimensionality")
        st.metric("PCs for 80% var", n_80)
        st.metric("PCs for 90% var", n_90)
        st.metric("PCs for 95% var", n_95)
        st.metric("PC1 explains", f"{pca.explained_variance_ratio_[0]*100:.1f}%")

    # PC1 vs PC2 scatter
    st.subheader("PC1 vs PC2 (colored by time)")
    time_s = np.arange(n_frames) / fps
    ds = max(1, n_frames // 3000)

    fig = px.scatter(
        x=scores[::ds, 0],
        y=scores[::ds, 1],
        color=time_s[::ds],
        color_continuous_scale="Viridis",
        labels={"x": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                "y": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                "color": "Time (s)"},
        title="Population state space",
        opacity=0.4,
    )
    fig.update_layout(height=500)
    fig.update_traces(marker_size=3)
    st.plotly_chart(fig, use_container_width=True)

    # PC loadings
    st.subheader("PC Loadings")
    pc_to_show = st.selectbox("PC", list(range(1, min(6, max_components + 1))),
                               format_func=lambda x: f"PC{x}", key="corr_pc")
    loadings = pca.components_[pc_to_show - 1]
    fig = px.bar(
        x=np.arange(n_rois),
        y=loadings,
        title=f"PC{pc_to_show} Loadings",
        labels={"x": "ROI", "y": "Loading"},
    )
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)


# --- Tab 3: Co-activation ---
with tab_ensembles:
    st.subheader("Population Co-activation")

    if "event_masks" not in data:
        st.info("Event masks not available — using dF/F\u2080 threshold instead.")
        # Use dF/F0 > 2*std as proxy
        threshold = 2.0
        active = (dff > dff.mean(axis=1, keepdims=True) + threshold * dff.std(axis=1, keepdims=True))
    else:
        active = data["event_masks"].astype(bool)

    # Number of co-active cells per frame
    n_active = active.sum(axis=0)
    time_s = np.arange(n_frames) / fps

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        # Downsample for plotting
        ds = max(1, n_frames // 2000)
        fig.add_trace(go.Scatter(
            x=time_s[::ds],
            y=n_active[::ds],
            mode="lines",
            line=dict(color="black", width=0.8),
            name="Co-active cells",
        ))
        fig.update_layout(
            height=250,
            title="Co-active Cells Over Time",
            xaxis_title="Time (s)",
            yaxis_title="N cells active",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            x=n_active, nbins=n_rois,
            title="Distribution of Co-active Cell Count",
            labels={"x": "N cells active", "y": "N frames"},
        )
        # Expected from independent Poisson
        mean_active = n_active.mean()
        fig.add_vline(x=mean_active, line_dash="dash", line_color="red",
                      annotation_text=f"mean={mean_active:.1f}")
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    # Population event detection
    st.subheader("Population Events")

    with st.expander("About population events"):
        st.markdown(
            "A **population event** is a time bin where many neurons are simultaneously "
            "active — more than expected by chance if cells fired independently. "
            "High co-activation rates indicate ensemble-level coordination: neurons "
            "that consistently participate together likely share tuning properties or "
            "circuit connectivity. The participation bar chart below shows which ROIs "
            "are most reliably recruited during these events."
        )

    pop_threshold = st.slider(
        "Min co-active cells for population event",
        1, max(2, n_rois // 2), max(2, n_rois // 4),
        key="corr_pop_thresh",
    )

    pop_events = n_active >= pop_threshold
    pop_rate = pop_events.sum() / (n_frames / fps)

    col1, col2, col3 = st.columns(3)
    col1.metric("Population event frames", f"{pop_events.sum()}")
    col2.metric("Fraction of time", f"{pop_events.mean():.3f}")
    col3.metric("Events/s", f"{pop_rate:.2f}")

    # Which cells participate most in population events?
    participation = active[:, pop_events].mean(axis=1) if pop_events.any() else np.zeros(n_rois)
    fig = px.bar(
        x=np.arange(n_rois),
        y=participation,
        title="ROI Participation in Population Events",
        labels={"x": "ROI", "y": "Participation rate"},
    )
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)
