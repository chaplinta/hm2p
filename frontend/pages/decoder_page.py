"""Population Decoder — Bayesian HD decoding from population activity.

Demonstrates how head direction can be decoded from the activity of a
population of HD cells using a Bayesian maximum-likelihood approach.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

st.title("Population Decoder")
st.caption(
    "Bayesian maximum-likelihood head direction decoding from population activity. "
    "Demonstrates cross-validated decoding accuracy with synthetic HD cell populations."
)

import plotly.express as px
import plotly.graph_objects as go

from hm2p.analysis.decoder import (
    build_decoder,
    cross_validated_decode,
    decode_error,
    decode_hd,
)


def _make_population(n_cells=10, n_frames=3000, kappa=3.0, noise=0.1, seed=42):
    """Generate synthetic population of HD cells."""
    rng = np.random.default_rng(seed)
    hd_deg = np.cumsum(rng.normal(0, 5, n_frames)) % 360.0
    theta = np.deg2rad(hd_deg)
    mask = np.ones(n_frames, dtype=bool)
    prefs = np.linspace(0, 360, n_cells, endpoint=False)
    signals = np.zeros((n_cells, n_frames), dtype=np.float64)
    for i in range(n_cells):
        pref_rad = np.deg2rad(prefs[i])
        rate = 0.1 + np.exp(kappa * np.cos(theta - pref_rad))
        rate /= rate.max()
        rate += rng.normal(0, noise, n_frames)
        signals[i] = np.clip(rate, 0, None)
    return signals, hd_deg, mask, prefs


# --- Controls ---
col_c1, col_c2, col_c3, col_c4 = st.columns(4)
with col_c1:
    n_cells = st.slider("Number of cells", 3, 50, 15, 1, key="dec_n")
with col_c2:
    kappa = st.slider("Tuning sharpness (κ)", 0.5, 8.0, 3.0, 0.5, key="dec_kappa")
with col_c3:
    noise_level = st.slider("Noise level", 0.05, 1.0, 0.2, 0.05, key="dec_noise")
with col_c4:
    n_frames = st.select_slider("Frames", [500, 1000, 2000, 3000, 5000], value=2000,
                                 key="dec_frames")

tab_decode, tab_cv, tab_sweep = st.tabs(["Decode", "Cross-Validation", "Parameter Sweep"])

# --- Single decode ---
with tab_decode:
    st.subheader("Frame-by-Frame Decoding")

    signals, hd, mask, prefs = _make_population(
        n_cells=n_cells, n_frames=n_frames, kappa=kappa, noise=noise_level,
    )
    dec = build_decoder(signals, hd, mask)
    decoded, posterior = decode_hd(signals, dec)
    errs = decode_error(decoded, hd % 360.0)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean abs error", f"{errs['mean_abs_error']:.1f}°")
    col2.metric("Median abs error", f"{errs['median_abs_error']:.1f}°")
    col3.metric("Circular std", f"{errs['circular_std_error']:.1f}°")
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
        xaxis_title="Frame", yaxis_title="HD (°)",
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
            xaxis_title="Error (°)", yaxis_title="Count",
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
            xaxis_title="Actual HD (°)", yaxis_title="Decoded HD (°)",
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
            labels=dict(x="Frame", y="HD (°)", color="P"),
            color_continuous_scale="Hot",
            title=f"Posterior P(HD | activity) — first {n_post} frames",
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
    col1.metric("CV mean abs error", f"{cv_errs['mean_abs_error']:.1f}°")
    col2.metric("CV median abs error", f"{cv_errs['median_abs_error']:.1f}°")
    col3.metric("Folds", n_folds)

    # Comparison: train vs CV
    st.markdown(
        f"**Train error:** {errs['mean_abs_error']:.1f}° — "
        f"**CV error:** {cv_errs['mean_abs_error']:.1f}° — "
        f"**Overfit gap:** {cv_errs['mean_abs_error'] - errs['mean_abs_error']:.1f}°"
    )

    # CV error histogram
    fig = go.Figure(data=[go.Histogram(
        x=cv_errs["errors_deg"], nbinsx=36, marker_color="orange",
    )])
    fig.add_vline(x=0, line_color="red", line_dash="dash")
    fig.update_layout(
        height=300, title="CV Error Distribution",
        xaxis_title="Error (°)", yaxis_title="Count",
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Parameter sweep ---
with tab_sweep:
    st.subheader("How Population Size & Tuning Affect Decoding")

    sweep_type = st.radio(
        "Sweep", ["Number of cells", "Tuning sharpness (κ)"],
        horizontal=True, key="dec_sweep",
    )

    sweep_data = []
    if sweep_type == "Number of cells":
        for n in [3, 5, 8, 10, 15, 20, 30]:
            sig, h, m, _ = _make_population(n_cells=n, n_frames=1500, kappa=kappa,
                                              noise=noise_level, seed=42)
            d = build_decoder(sig, h, m)
            dec_hd, _ = decode_hd(sig, d)
            e = decode_error(dec_hd, h % 360)
            sweep_data.append({"Value": n, "MAE (°)": e["mean_abs_error"]})
        x_label = "Number of cells"
    else:
        for k in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
            sig, h, m, _ = _make_population(n_cells=n_cells, n_frames=1500,
                                              kappa=k, noise=noise_level, seed=42)
            d = build_decoder(sig, h, m)
            dec_hd, _ = decode_hd(sig, d)
            e = decode_error(dec_hd, h % 360)
            sweep_data.append({"Value": k, "MAE (°)": e["mean_abs_error"]})
        x_label = "κ"

    sweep_df = pd.DataFrame(sweep_data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep_df["Value"], y=sweep_df["MAE (°)"],
        mode="lines+markers", marker_color="royalblue",
    ))
    fig.add_hline(y=90, line_color="red", line_dash="dash",
                  annotation_text="Chance level (90°)")
    fig.update_layout(
        height=350, title=f"Decoding Error vs {x_label}",
        xaxis_title=x_label, yaxis_title="Mean Abs Error (°)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sweep_df, hide_index=True)

# --- Footer ---
st.markdown("---")
st.caption(
    "Bayesian maximum-likelihood decoder assumes Poisson-like firing with "
    "tuning curves as rate model and flat prior. Cross-validation uses "
    "k-fold with shuffled frame assignment. Zhang, Sejnowski & Bhatt (1998)."
)
