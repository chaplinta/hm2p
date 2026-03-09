"""Cell Classification — automated HD cell identification.

Shows ALL cells across ALL sessions by default. Optional filtering
by cell type or animal in the sidebar. Requires sync.h5 data from S3.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from hm2p.analysis.classify import (
    classification_summary_table,
    classify_population,
)
from hm2p.analysis.tuning import compute_hd_tuning_curve

log = logging.getLogger("hm2p.frontend.classify")

st.title("Cell Classification")
st.caption(
    "Automated HD cell identification using MVL, shuffle significance, "
    "split-half reliability, and mutual information. "
    "Shows all cells across all sessions by default."
)


# ── Data loading ────────────────────────────────────────────────────────────

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


# ── Parameters (sidebar) ───────────────────────────────────────────────────

real_sessions, has_real = _try_load_real()

if not has_real or not real_sessions:
    st.warning(
        "No data available yet. This page will populate when the relevant "
        "pipeline stage completes."
    )
    st.stop()

st.sidebar.header("Thresholds")
mvl_thresh = st.sidebar.slider("MVL threshold", 0.05, 0.5, 0.15, 0.01, key="cls_mvl")
p_thresh = st.sidebar.slider("p-value threshold", 0.001, 0.1, 0.05, 0.005,
                              key="cls_p", format="%.3f")
rel_thresh = st.sidebar.slider("Reliability threshold", 0.1, 0.9, 0.5, 0.05,
                                key="cls_rel")
n_shuffles = st.sidebar.slider("Shuffles", 100, 1000, 300, 50, key="cls_shuf")


# ── Run classification ──────────────────────────────────────────────────────

st.success(
    f"Loaded {len(real_sessions)} sessions, "
    f"{sum(s['n_rois'] for s in real_sessions)} total cells"
)

all_cells = []
all_signals_for_tuning = {}  # (exp_id, cell_idx) -> (signal, hd, mask)

with st.spinner("Classifying all cells..."):
    for ses_data in real_sessions:
        signals = ses_data["dff"]
        hd = ses_data["hd_deg"]
        mask = ses_data["active"] & ~ses_data["bad_behav"]
        exp_id = ses_data["exp_id"]
        celltype = ses_data["celltype"]

        pop = classify_population(
            signals, hd, mask,
            mvl_threshold=mvl_thresh,
            p_threshold=p_thresh,
            reliability_threshold=rel_thresh,
            n_shuffles=n_shuffles,
            rng=np.random.default_rng(42),
        )
        table = classification_summary_table(pop)
        for row in table:
            row["exp_id"] = exp_id
            row["celltype"] = celltype
            row["animal_id"] = ses_data["animal_id"]
            all_cells.append(row)
            # Store for tuning curve plots
            idx = row["cell"]
            all_signals_for_tuning[(exp_id, idx)] = (signals[idx], hd, mask)

df = pd.DataFrame(all_cells)
n_total = len(df)


# ── Summary metrics ────────────────────────────────────────────────────────

n_hd_found = int(df["is_hd"].sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cells", n_total)
col2.metric("HD Cells", n_hd_found)
col3.metric("Non-HD Cells", n_total - n_hd_found)
col4.metric("HD Fraction", f"{n_hd_found / n_total:.1%}" if n_total > 0 else "N/A")

# Per-celltype breakdown
for ct in df["celltype"].unique():
    sub = df[df["celltype"] == ct]
    n_ct_hd = sub["is_hd"].sum()
    st.caption(f"**{ct}**: {len(sub)} cells, {n_ct_hd} HD ({n_ct_hd/len(sub):.0%})")


# ── Tabs ────────────────────────────────────────────────────────────────────

tab_table, tab_scatter, tab_tuning = st.tabs(["Summary Table", "Metric Scatter", "Tuning Curves"])

with tab_table:
    df_show = df[["cell", "is_hd", "grade", "mvl", "p_value", "reliability",
                   "mi", "preferred_direction"]].copy()
    df_show.insert(0, "session", df["exp_id"])
    df_show.insert(1, "celltype", df["celltype"])

    df_show["mvl"] = df_show["mvl"].apply(lambda x: f"{x:.3f}")
    df_show["p_value"] = df_show["p_value"].apply(lambda x: f"{x:.4f}")
    df_show["reliability"] = df_show["reliability"].apply(lambda x: f"{x:.3f}")
    df_show["mi"] = df_show["mi"].apply(lambda x: f"{x:.4f}")
    df_show["preferred_direction"] = df_show["preferred_direction"].apply(lambda x: f"{x:.1f}°")

    def _highlight_hd(row):
        if row["is_hd"]:
            return ["background-color: rgba(0, 180, 0, 0.15)"] * len(row)
        return [""] * len(row)

    styled = df_show.style.apply(_highlight_hd, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=400)

    st.markdown(
        "**Grades:** A = strong HD (MVL>=0.4, reliability>=0.8) · "
        "B = moderate HD (MVL>=0.25) · C = weak HD · D = non-HD"
    )

with tab_scatter:
    mvls = df["mvl"].values
    pvals = df["p_value"].values
    reliabilities = df["reliability"].values
    mis = df["mi"].values
    labels = [f"Cell {r['cell']}" for _, r in df.iterrows()]

    colors = df["celltype"].values
    color_map = None  # Let plotly auto-assign

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.scatter(
            x=mvls, y=pvals, color=colors, text=labels,
            labels={"x": "MVL", "y": "p-value", "color": "Type"},
            title="MVL vs Significance",
            color_discrete_map=color_map,
        )
        fig.add_hline(y=p_thresh, line_dash="dash", line_color="gray",
                      annotation_text=f"p={p_thresh}")
        fig.add_vline(x=mvl_thresh, line_dash="dash", line_color="gray",
                      annotation_text=f"MVL={mvl_thresh}")
        fig.update_yaxes(type="log")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True, key="cls_mvl_p")

    with col_b:
        fig = px.scatter(
            x=mvls, y=reliabilities, color=colors, text=labels,
            labels={"x": "MVL", "y": "Split-half r", "color": "Type"},
            title="MVL vs Reliability",
            color_discrete_map=color_map,
        )
        fig.add_hline(y=rel_thresh, line_dash="dash", line_color="gray",
                      annotation_text=f"r={rel_thresh}")
        fig.add_vline(x=mvl_thresh, line_dash="dash", line_color="gray")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True, key="cls_mvl_r")

    # MI vs MVL
    fig = px.scatter(
        x=mvls, y=mis, color=colors, text=labels,
        labels={"x": "MVL", "y": "MI (bits)", "color": "Type"},
        title="MVL vs Mutual Information",
        color_discrete_map=color_map,
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True, key="cls_mvl_mi")

with tab_tuning:
    st.subheader("Tuning Curves by Classification")

    hd_cells = df[df["is_hd"]]
    non_hd_cells = df[~df["is_hd"]]

    col_hd, col_non = st.columns(2)
    with col_hd:
        st.markdown(f"**HD Cells ({len(hd_cells)})**")
        for _, row in hd_cells.head(8).iterrows():
            key = (row["exp_id"], row["cell"])
            if key in all_signals_for_tuning:
                sig, hd_arr, msk = all_signals_for_tuning[key]
                tc, bc = compute_hd_tuning_curve(sig, hd_arr, msk, n_bins=36)
                theta_plot = np.concatenate([np.deg2rad(bc), [np.deg2rad(bc[0])]])
                r_plot = np.concatenate([tc, [tc[0]]])
                fig = go.Figure(data=[go.Scatterpolar(
                    r=r_plot, theta=np.rad2deg(theta_plot),
                    mode="lines", line=dict(color="green", width=2),
                )])
                title = f"{row['exp_id'][-7:]} C{row['cell']} ({row['celltype']})"
                fig.update_layout(
                    height=220, margin=dict(l=30, r=30, t=40, b=20),
                    title=f"{title} MVL={row['mvl']:.3f} [{row['grade']}]",
                    polar=dict(radialaxis=dict(showticklabels=False)),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True,
                               key=f"tc_hd_{row['exp_id']}_{row['cell']}")
        if len(hd_cells) > 8:
            st.caption(f"Showing 8 of {len(hd_cells)} HD cells")

    with col_non:
        st.markdown(f"**Non-HD Cells ({len(non_hd_cells)})**")
        for _, row in non_hd_cells.head(6).iterrows():
            key = (row["exp_id"], row["cell"])
            if key in all_signals_for_tuning:
                sig, hd_arr, msk = all_signals_for_tuning[key]
                tc, bc = compute_hd_tuning_curve(sig, hd_arr, msk, n_bins=36)
                theta_plot = np.concatenate([np.deg2rad(bc), [np.deg2rad(bc[0])]])
                r_plot = np.concatenate([tc, [tc[0]]])
                fig = go.Figure(data=[go.Scatterpolar(
                    r=r_plot, theta=np.rad2deg(theta_plot),
                    mode="lines", line=dict(color="red", width=2),
                )])
                fig.update_layout(
                    height=220, margin=dict(l=30, r=30, t=40, b=20),
                    title=f"Cell {row['cell']} MVL={row['mvl']:.3f}",
                    polar=dict(radialaxis=dict(showticklabels=False)),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True,
                               key=f"tc_nhd_{row['exp_id']}_{row['cell']}")


# Footer
st.markdown("---")
st.caption(
    "Classification uses three criteria: (1) MVL exceeds threshold, "
    "(2) shuffle test p-value below threshold, (3) split-half reliability "
    "above threshold. All three must pass for HD classification."
)
