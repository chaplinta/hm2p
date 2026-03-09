"""Information Theory — mutual information, coding efficiency, redundancy.

Visualizes information-theoretic properties of HD cell populations:
per-cell MI, Skaggs information rate, and synergy/redundancy analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

st.title("Information Theory")
st.caption(
    "Information-theoretic analysis of HD coding — mutual information, "
    "Skaggs information rate, and population redundancy."
)

import plotly.graph_objects as go

from frontend.data import load_all_sync_data, session_filter_sidebar
from hm2p.analysis.information import (
    information_per_cell,
    mutual_information_binned,
    skaggs_info_rate,
    synergy_redundancy,
)
from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length

# Load real data
all_data = load_all_sync_data()
if all_data["n_sessions"] == 0:
    st.warning(
        "No data available yet. This page will populate when the relevant "
        "pipeline stage completes."
    )
    st.stop()

sessions = session_filter_sidebar(all_data["sessions"])
if not sessions:
    st.warning("No sessions match the current filters.")
    st.stop()

# Session selector
session_labels = [f"{s['exp_id']} ({s['celltype']}, {s['n_rois']} ROIs)" for s in sessions]
sel_idx = st.sidebar.selectbox("Session", range(len(sessions)),
                                format_func=lambda i: session_labels[i], key="info_ses")
sess = sessions[sel_idx]

signals = sess["dff"]  # (n_rois, n_frames)
hd = sess["hd_deg"]
mask = sess["active"] & ~sess["bad_behav"]
n_cells = signals.shape[0]

if n_cells == 0:
    st.warning("No ROIs in this session after filtering.")
    st.stop()

tab_mi, tab_skaggs, tab_redundancy = st.tabs(["Mutual Information", "Skaggs Info", "Redundancy"])

with tab_mi:
    st.subheader("Per-Cell Mutual Information")

    info = information_per_cell(signals, hd, mask)
    mvls = []
    for i in range(n_cells):
        tc, bc = compute_hd_tuning_curve(signals[i], hd, mask)
        mvls.append(mean_vector_length(tc, bc))

    col_bar, col_scatter = st.columns(2)
    with col_bar:
        sort_idx = np.argsort(info)[::-1]
        fig = go.Figure(data=[go.Bar(
            x=[f"Cell {i+1}" for i in sort_idx],
            y=info[sort_idx],
            marker_color="royalblue",
        )])
        fig.update_layout(
            height=300, title="MI per Cell (sorted)",
            xaxis_title="Cell", yaxis_title="MI (bits)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_scatter:
        fig = go.Figure(data=[go.Scatter(
            x=mvls, y=info,
            mode="markers", marker=dict(size=10, color="royalblue"),
            text=[f"Cell {i+1}" for i in range(n_cells)],
        )])
        fig.update_layout(
            height=300, title="MI vs MVL",
            xaxis_title="Mean Vector Length", yaxis_title="MI (bits)",
        )
        st.plotly_chart(fig, use_container_width=True)

    total_mi = float(np.sum(info))
    mean_mi = float(np.mean(info))
    st.markdown(f"**Total MI (sum):** {total_mi:.3f} bits — **Mean MI:** {mean_mi:.3f} bits")

with tab_skaggs:
    st.subheader("Skaggs Information Rate")

    skaggs_rates = []
    for i in range(n_cells):
        tc, bc = compute_hd_tuning_curve(signals[i], hd, mask, n_bins=36)
        hd_mod = np.mod(hd[mask], 360)
        bin_edges = np.linspace(0, 360, 37)
        occ = np.histogram(hd_mod, bins=bin_edges)[0].astype(float)
        si = skaggs_info_rate(tc, occ)
        skaggs_rates.append(si)

    fig = go.Figure(data=[go.Bar(
        x=[f"Cell {i+1}" for i in range(n_cells)],
        y=skaggs_rates,
        marker_color="orange",
    )])
    fig.update_layout(
        height=300, title="Skaggs Information Rate (bits/spike)",
        xaxis_title="Cell", yaxis_title="SI (bits/spike)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Skaggs vs MI comparison
    fig = go.Figure(data=[go.Scatter(
        x=info, y=skaggs_rates,
        mode="markers", marker=dict(size=10, color="green"),
        text=[f"Cell {i+1}" for i in range(n_cells)],
    )])
    fig.update_layout(
        height=300, title="Skaggs SI vs Binned MI",
        xaxis_title="Binned MI (bits)", yaxis_title="Skaggs SI (bits/spike)",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_redundancy:
    st.subheader("Pairwise Synergy / Redundancy")
    st.markdown(
        "Compares the joint information of cell pairs with the sum of individual "
        "information. Positive = redundant (overlap), negative = synergistic "
        "(complementary coding)."
    )

    if n_cells >= 2:
        col_a, col_b = st.columns(2)
        with col_a:
            cell_a = st.selectbox("Cell A", range(n_cells),
                                   format_func=lambda x: f"Cell {x+1}", key="red_a")
        with col_b:
            cell_b = st.selectbox("Cell B", range(n_cells), index=min(1, n_cells-1),
                                   format_func=lambda x: f"Cell {x+1}", key="red_b")

        if cell_a != cell_b:
            sr = synergy_redundancy(signals, hd, mask, cell_a, cell_b)
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric(f"MI Cell {cell_a+1}", f"{sr['info_a']:.4f}")
            col_r2.metric(f"MI Cell {cell_b+1}", f"{sr['info_b']:.4f}")
            col_r3.metric("Joint MI", f"{sr['info_joint']:.4f}")
            col_r4.metric("Redundancy", f"{sr['redundancy']:.4f}")

            category = "Redundant" if sr["redundancy"] > 0.01 else (
                "Synergistic" if sr["redundancy"] < -0.01 else "Independent"
            )
            st.markdown(f"**Coding type:** {category}")
        else:
            st.warning("Select two different cells.")

        # Redundancy matrix
        if st.checkbox("Show full redundancy matrix", key="red_matrix"):
            with st.spinner("Computing pairwise redundancy..."):
                red_mat = np.zeros((n_cells, n_cells))
                for i in range(n_cells):
                    for j in range(i+1, n_cells):
                        sr = synergy_redundancy(signals, hd, mask, i, j,
                                                n_hd_bins=12, n_signal_bins=4)
                        red_mat[i, j] = sr["redundancy"]
                        red_mat[j, i] = sr["redundancy"]

            import plotly.express as px
            labels = [f"Cell {i+1}" for i in range(n_cells)]
            fig = px.imshow(
                red_mat, x=labels, y=labels,
                color_continuous_scale="RdBu_r",
                title="Pairwise Redundancy (positive=redundant, negative=synergistic)",
                aspect="equal",
            )
            fig.update_layout(height=max(300, n_cells * 25))
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption(
    "MI estimated using binned method with quantile-based signal binning. "
    "Skaggs SI follows Skaggs et al. (1993). Redundancy uses sum-MI "
    "decomposition approximation."
)
