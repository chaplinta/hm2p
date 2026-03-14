"""HD Tuning Explorer — interactive head direction tuning curve analysis.

Dedicated page for visualizing and analysing head direction tuning:
polar tuning curves, MVL distributions, preferred direction maps,
tuning width, and significance testing with circular shuffle.

Requires real sync.h5 data from S3.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

log = logging.getLogger("hm2p.frontend.hd_tuning")

st.title("HD Tuning Explorer")
st.caption(
    "Interactive head direction tuning curve analysis — polar plots, MVL, "
    "preferred direction, tuning width, and significance testing."
)

import plotly.express as px
import plotly.graph_objects as go

from hm2p.analysis.tuning import (
    compute_hd_tuning_curve,
    mean_vector_length,
    peak_to_trough_ratio,
    preferred_direction,
    tuning_width_fwhm,
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

# --- Tabs ---
tab_single, tab_population, tab_significance = st.tabs([
    "Single Cell", "Population", "Significance",
])

# --- Single Cell ---
with tab_single:
    st.subheader("Single Cell Tuning Curve")

    # Session and cell selection
    session_labels = [s["exp_id"] for s in real_sessions]
    sel_session_idx = st.selectbox("Session", range(len(session_labels)),
                                   format_func=lambda i: session_labels[i],
                                   key="hd_session")
    ses_data = real_sessions[sel_session_idx]
    n_rois = ses_data["n_rois"]
    sel_cell = st.slider("Cell index", 0, max(0, n_rois - 1), 0, key="hd_cell")

    n_bins = st.select_slider("Number of bins", options=[12, 18, 24, 36, 72], value=36,
                               key="hd_nbins")
    sigma = st.slider("Smoothing sigma (deg)", 0.0, 30.0, 6.0, 1.0, key="hd_sigma")

    signal = ses_data["dff"][sel_cell]
    hd_deg = ses_data["hd_deg"]
    mask = ses_data["active"] & ~ses_data["bad_behav"]

    tc, bin_centers = compute_hd_tuning_curve(
        signal, hd_deg, mask, n_bins=n_bins, smoothing_sigma_deg=sigma,
    )

    mvl = mean_vector_length(tc, bin_centers)
    pd_deg = preferred_direction(tc, bin_centers)
    fwhm = tuning_width_fwhm(tc, bin_centers)
    ptr = peak_to_trough_ratio(tc)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MVL", f"{mvl:.3f}")
    col2.metric("Preferred dir", f"{pd_deg:.1f}deg")
    col3.metric("FWHM", f"{fwhm:.1f}deg")
    col4.metric("Peak/Trough", f"{ptr:.2f}" if not np.isnan(ptr) else "---")

    # Polar plot
    col_polar, col_linear = st.columns(2)
    with col_polar:
        theta_plot = np.concatenate([np.deg2rad(bin_centers), [np.deg2rad(bin_centers[0])]])
        r_plot = np.concatenate([tc, [tc[0]]])
        r_plot = np.where(np.isnan(r_plot), 0, r_plot)

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r_plot, theta=np.rad2deg(theta_plot),
            mode="lines", fill="toself",
            fillcolor="rgba(65, 105, 225, 0.3)",
            line=dict(color="royalblue", width=2),
            name="Tuning curve",
        ))
        fig.add_trace(go.Scatterpolar(
            r=[0, np.nanmax(tc) * mvl], theta=[pd_deg, pd_deg],
            mode="lines",
            line=dict(color="red", width=3),
            name=f"MVL vector ({mvl:.3f})",
        ))
        fig.update_layout(
            height=400,
            polar=dict(
                radialaxis=dict(visible=False),
                angularaxis=dict(direction="clockwise", rotation=90, showticklabels=False),
            ),
            title="Polar Tuning Curve",
            showlegend=True,
            legend=dict(x=0, y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_linear:
        fig = go.Figure()
        tc_plot = np.where(np.isnan(tc), 0, tc)
        fig.add_trace(go.Bar(
            x=bin_centers, y=tc_plot,
            marker_color="royalblue",
            name="Tuning curve",
        ))
        fig.add_vline(x=pd_deg, line_color="red", line_dash="dash",
                      annotation_text=f"PD={pd_deg:.0f}deg")
        fig.update_layout(
            height=400,
            title="Linear Tuning Curve",
            xaxis_title="Head Direction (deg)",
            yaxis_title="Mean signal",
            xaxis=dict(range=[0, 360]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Raw data scatter
    with st.expander("Raw Data"):
        n_show = min(2000, len(signal))
        idx = np.linspace(0, len(signal) - 1, n_show, dtype=int)
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=hd_deg[idx], y=signal[idx],
            mode="markers", marker=dict(size=2, opacity=0.3, color="gray"),
            name="Raw data",
        ))
        fig.add_trace(go.Scatter(
            x=bin_centers, y=tc_plot,
            mode="lines", line=dict(color="royalblue", width=3),
            name="Tuning curve",
        ))
        fig.update_layout(
            height=300, title="Signal vs Head Direction",
            xaxis_title="HD (deg)", yaxis_title="Signal",
            xaxis=dict(range=[0, 360]),
        )
        st.plotly_chart(fig, use_container_width=True)


# --- Population ---
with tab_population:
    st.subheader("Population HD Tuning")

    all_tcs = []
    pop_data = []
    bc_shared = None

    for ses_data in real_sessions:
        signals = ses_data["dff"]
        hd = ses_data["hd_deg"]
        msk = ses_data["active"] & ~ses_data["bad_behav"]
        exp_id = ses_data["exp_id"]
        celltype = ses_data["celltype"]

        for ci in range(ses_data["n_rois"]):
            tc_i, bc_i = compute_hd_tuning_curve(signals[ci], hd, msk, n_bins=36)
            if bc_shared is None:
                bc_shared = bc_i
            mvl_i = mean_vector_length(tc_i, bc_i)
            pd_i = preferred_direction(tc_i, bc_i)
            fwhm_i = tuning_width_fwhm(tc_i, bc_i)
            pop_data.append({
                "Session": exp_id,
                "Cell": ci,
                "Celltype": celltype,
                "Pref Dir (deg)": f"{pd_i:.0f}",
                "MVL": mvl_i,
                "FWHM (deg)": f"{fwhm_i:.0f}",
            })
            all_tcs.append(tc_i)

    n_cells = len(all_tcs)

    if n_cells == 0:
        st.warning("No cells found across loaded sessions.")
    else:
        # MVL histogram
        mvls = [d["MVL"] for d in pop_data]
        col_hist, col_rose = st.columns(2)
        with col_hist:
            fig = go.Figure(data=[go.Histogram(x=mvls, nbinsx=15, marker_color="royalblue")])
            fig.update_layout(
                height=300, title="MVL Distribution",
                xaxis_title="Mean Vector Length", yaxis_title="Count",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_rose:
            pds = [preferred_direction(tc_i, bc_shared) for tc_i in all_tcs]
            fig = go.Figure()
            fig.add_trace(go.Barpolar(
                r=[1] * len(pds), theta=pds,
                marker_color=mvls,
                marker_colorscale="Viridis",
                marker_showscale=True,
                marker_colorbar=dict(title="MVL"),
                width=360 / max(n_cells, 1) * 0.8,
            ))
            fig.update_layout(
                height=300,
                polar=dict(radialaxis=dict(visible=False), angularaxis=dict(direction="clockwise", rotation=90, showticklabels=False)),
                title="Preferred Directions",
            )
            st.plotly_chart(fig, use_container_width=True)

        # All tuning curves heatmap
        tc_matrix = np.array([np.where(np.isnan(t), 0, t) for t in all_tcs])
        row_max = tc_matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        tc_norm = tc_matrix / row_max

        sort_idx = np.argsort(pds)
        tc_sorted = tc_norm[sort_idx]

        fig = px.imshow(
            tc_sorted,
            x=[f"{b:.0f}" for b in bc_shared],
            y=[f"Cell {sort_idx[i]}" for i in range(n_cells)],
            labels=dict(x="HD (deg)", y="Cell", color="Norm. rate"),
            color_continuous_scale="Hot",
            title="Population Tuning Curves (sorted by preferred direction)",
            aspect="auto",
        )
        fig.update_layout(height=max(300, n_cells * 15 + 100))
        st.plotly_chart(fig, use_container_width=True)

        # Data table
        with st.expander("Cell Details"):
            df_pop = pd.DataFrame(pop_data)
            df_pop["MVL"] = df_pop["MVL"].apply(lambda x: f"{x:.3f}")
            st.dataframe(df_pop, hide_index=True)


# --- Significance Testing ---
with tab_significance:
    st.subheader("Circular Shuffle Significance Test")
    st.markdown(
        "Tests whether observed HD tuning is significantly greater than "
        "expected by chance, using circular time-shift shuffles "
        "(Skaggs et al., 1993)."
    )

    # Session and cell selection for significance
    sig_session_idx = st.selectbox("Session", range(len(session_labels)),
                                    format_func=lambda i: session_labels[i],
                                    key="sig_session")
    sig_ses = real_sessions[sig_session_idx]
    sig_cell = st.slider("Cell index", 0, max(0, sig_ses["n_rois"] - 1), 0, key="sig_cell")
    n_shuffles = st.select_slider("Shuffles", [100, 500, 1000], value=500, key="sig_n")

    signal_s = sig_ses["dff"][sig_cell]
    hd_s = sig_ses["hd_deg"]
    mask_s = sig_ses["active"] & ~sig_ses["bad_behav"]

    with st.spinner("Running shuffle test..."):
        from hm2p.analysis.significance import hd_tuning_significance

        result = hd_tuning_significance(
            signal_s, hd_s, mask_s, n_shuffles=n_shuffles,
            metric="mvl", rng=np.random.default_rng(42),
        )

    p_val = result["p_value"]
    obs_mvl = result["observed"]
    shuf_dist = result["shuffle_distribution"]

    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Observed MVL", f"{obs_mvl:.4f}")
    col_p2.metric("p-value", f"{p_val:.4f}")
    col_p3.metric("Significant?", "Yes" if p_val < 0.05 else "No")

    # Shuffle distribution histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=shuf_dist, nbinsx=30,
        marker_color="lightgray", name="Shuffle dist",
    ))
    fig.add_vline(x=obs_mvl, line_color="red", line_width=3,
                  annotation_text=f"Observed={obs_mvl:.4f}")
    percentile_95 = np.percentile(shuf_dist, 95)
    fig.add_vline(x=percentile_95, line_color="orange", line_dash="dash",
                  annotation_text="95th pctile")
    fig.update_layout(
        height=350,
        title=f"Shuffle Distribution (n={n_shuffles}, p={p_val:.4f})",
        xaxis_title="MVL", yaxis_title="Count",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show the tuning curve
    tc_sig = result["tuning_curve"]
    bc_sig = result["bin_centers"]
    theta_s = np.concatenate([np.deg2rad(bc_sig), [np.deg2rad(bc_sig[0])]])
    r_s = np.concatenate([tc_sig, [tc_sig[0]]])
    r_s = np.where(np.isnan(r_s), 0, r_s)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_s, theta=np.rad2deg(theta_s),
        mode="lines", fill="toself",
        fillcolor="rgba(65, 105, 225, 0.3)" if p_val < 0.05 else "rgba(200, 200, 200, 0.3)",
        line=dict(color="royalblue" if p_val < 0.05 else "gray", width=2),
    ))
    fig.update_layout(
        height=350,
        polar=dict(radialaxis=dict(visible=False), angularaxis=dict(direction="clockwise", rotation=90, showticklabels=False)),
        title=f"Tuning Curve (p={p_val:.4f})",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rayleigh test and split-half reliability
    st.markdown("---")
    col_r, col_sh = st.columns(2)

    with col_r:
        st.markdown("**Rayleigh Test**")
        from hm2p.analysis.comparison import rayleigh_test
        ray = rayleigh_test(hd_s[mask_s], weights=signal_s[mask_s])
        st.metric("Rayleigh Z", f"{ray['z']:.2f}")
        st.metric("Rayleigh p", f"{ray['p_value']:.4f}")
        st.metric("Mean resultant R", f"{ray['mean_resultant_length']:.4f}")

    with col_sh:
        st.markdown("**Split-Half Reliability**")
        from hm2p.analysis.comparison import split_half_reliability
        sh = split_half_reliability(signal_s, hd_s, mask_s)
        st.metric("Half correlation", f"{sh['correlation']:.3f}")
        st.metric("MVL half 1", f"{sh['mvl_half1']:.4f}")
        st.metric("MVL half 2", f"{sh['mvl_half2']:.4f}")
        st.metric("PD shift", f"{sh['pd_shift']:.1f}deg")


# --- Footer ---
st.markdown("---")
st.caption(
    "HD tuning analysis uses occupancy-normalised circular histograms with "
    "Nadaraya-Watson Gaussian smoothing. Significance via circular time-shift "
    "shuffle (Skaggs et al., 1993). MVL = mean vector length (Rayleigh, 1919)."
)
