"""Population Decoder — Bayesian HD decoding from population activity.

Decodes head direction from the activity of a population of HD cells
using a Bayesian maximum-likelihood approach.

Requires real sync.h5 data from S3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

st.title("Population Decoder")
st.caption(
    "Bayesian maximum-likelihood head direction decoding from population activity. "
    "Cross-validated decoding accuracy from real HD cell populations."
)

import plotly.express as px
import plotly.graph_objects as go

from hm2p.analysis.decoder import (
    build_decoder,
    cross_validated_decode,
    decode_error,
    decode_hd,
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

# Session selection
session_labels = [s["exp_id"] for s in real_sessions]
sel_session_idx = st.selectbox("Session", range(len(session_labels)),
                                format_func=lambda i: session_labels[i],
                                key="dec_session")
ses_data = real_sessions[sel_session_idx]

signals = ses_data["dff"]
hd = ses_data["hd_deg"]
mask = ses_data["active"] & ~ses_data["bad_behav"]
n_cells = ses_data["n_rois"]
n_frames = signals.shape[1]

tab_decode, tab_cv = st.tabs(["Decode", "Cross-Validation"])

# --- Single decode ---
with tab_decode:
    st.subheader("Frame-by-Frame Decoding")

    dec = build_decoder(signals, hd, mask)
    decoded, posterior = decode_hd(signals, dec)
    errs = decode_error(decoded, hd % 360.0)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean abs error", f"{errs['mean_abs_error']:.1f} deg")
    col2.metric("Median abs error", f"{errs['median_abs_error']:.1f} deg")
    col3.metric("Circular std", f"{errs['circular_std_error']:.1f} deg")
    col4.metric("Cells used", n_cells)

    # Decoded vs actual (time series)
    n_show = min(500, n_frames)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=hd[:n_show] % 360, mode="lines",
        line=dict(color="gray", width=1), name="Actual HD",
    ))
    fig.add_trace(go.Scatter(
        y=decoded[:n_show], mode="markers",
        marker=dict(size=2, color="royalblue", opacity=0.5), name="Decoded HD",
    ))
    fig.update_layout(
        height=300, title=f"Decoded vs Actual HD (first {n_show} frames)",
        xaxis_title="Frame", yaxis_title="HD (deg)",
        yaxis=dict(range=[0, 360]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Error distribution
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig = go.Figure(data=[go.Histogram(
            x=errs["errors_deg"], nbinsx=36,
            marker_color="royalblue",
        )])
        fig.add_vline(x=0, line_color="red", line_dash="dash")
        fig.update_layout(
            height=300, title="Error Distribution",
            xaxis_title="Error (deg)", yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_e2:
        # Decoded vs actual scatter
        subsample = np.linspace(0, n_frames - 1, min(1000, n_frames), dtype=int)
        fig = go.Figure(data=[go.Scattergl(
            x=hd[subsample] % 360, y=decoded[subsample],
            mode="markers", marker=dict(size=2, opacity=0.3, color="royalblue"),
        )])
        fig.add_trace(go.Scatter(
            x=[0, 360], y=[0, 360], mode="lines",
            line=dict(color="red", dash="dash"), name="Perfect",
        ))
        fig.update_layout(
            height=300, title="Decoded vs Actual",
            xaxis_title="Actual HD (deg)", yaxis_title="Decoded HD (deg)",
            xaxis=dict(range=[0, 360]), yaxis=dict(range=[0, 360]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Posterior heatmap (subset)
    with st.expander("Posterior Probability"):
        n_post = min(200, n_frames)
        fig = px.imshow(
            posterior[:n_post].T,
            x=list(range(n_post)),
            y=[f"{b:.0f}" for b in dec["bin_centers"]],
            labels=dict(x="Frame", y="HD (deg)", color="P"),
            color_continuous_scale="Hot",
            title=f"Posterior P(HD | activity) -- first {n_post} frames",
            aspect="auto",
        )
        # Overlay actual HD
        fig.add_trace(go.Scatter(
            x=list(range(n_post)),
            y=[f"{hd[i] % 360:.0f}" for i in range(n_post)],
            mode="markers", marker=dict(size=2, color="cyan"),
            name="Actual HD",
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# --- Cross-validation ---
with tab_cv:
    st.subheader("Cross-Validated Decoding")

    n_folds = st.select_slider("Number of folds", [3, 5, 10], value=5, key="cv_folds")

    with st.spinner("Running cross-validation..."):
        cv_result = cross_validated_decode(
            signals, hd, mask, n_folds=n_folds,
            rng=np.random.default_rng(42),
        )

    cv_errs = cv_result["errors"]
    col1, col2, col3 = st.columns(3)
    col1.metric("CV mean abs error", f"{cv_errs['mean_abs_error']:.1f} deg")
    col2.metric("CV median abs error", f"{cv_errs['median_abs_error']:.1f} deg")
    col3.metric("Folds", n_folds)

    # Comparison: train vs CV
    st.markdown(
        f"**Train error:** {errs['mean_abs_error']:.1f} deg --- "
        f"**CV error:** {cv_errs['mean_abs_error']:.1f} deg --- "
        f"**Overfit gap:** {cv_errs['mean_abs_error'] - errs['mean_abs_error']:.1f} deg"
    )

    # CV error histogram
    fig = go.Figure(data=[go.Histogram(
        x=cv_errs["errors_deg"], nbinsx=36, marker_color="orange",
    )])
    fig.add_vline(x=0, line_color="red", line_dash="dash")
    fig.update_layout(
        height=300, title="CV Error Distribution",
        xaxis_title="Error (deg)", yaxis_title="Count",
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Footer ---
st.markdown("---")
st.caption(
    "Bayesian maximum-likelihood decoder assumes Poisson-like firing with "
    "tuning curves as rate model and flat prior. Cross-validation uses "
    "k-fold with shuffled frame assignment. Zhang, Sejnowski & Bhatt (1998)."
)
