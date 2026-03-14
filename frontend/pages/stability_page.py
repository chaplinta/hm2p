"""Stability Analysis — HD tuning stability over time and across conditions.

Visualises temporal stability of head direction tuning: first/second half
comparison, sliding window MVL/PD tracking, light/dark epoch comparison,
and per-epoch drift tracking.

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

log = logging.getLogger("hm2p.frontend.stability")

st.title("Tuning Stability")
st.caption(
    "Temporal stability of head direction tuning — tracks how tuning "
    "changes over time and between light/dark conditions."
)

import plotly.express as px
import plotly.graph_objects as go

from hm2p.analysis.stability import (
    drift_per_epoch,
    light_dark_stability,
    sliding_window_stability,
    split_temporal_halves,
)


# ---------------------------------------------------------------------------
# Data loading — pooled across all sessions
# ---------------------------------------------------------------------------

def _try_load_real():
    """Attempt to load real sync.h5 data."""
    try:
        from frontend.data import load_all_sync_data, session_filter_sidebar
        all_data = load_all_sync_data()
        if all_data["n_sessions"] > 0:
            sessions = session_filter_sidebar(
                all_data["sessions"], key_prefix="stability"
            )
            return sessions, True
    except Exception:
        log.exception("Error loading sync data")
    return None, False


real_sessions, has_real = _try_load_real()

if not has_real or not real_sessions:
    st.warning(
        "No data available yet. This page will populate when the relevant "
        "pipeline stage completes."
    )
    st.stop()

total_rois = sum(s["n_rois"] for s in real_sessions)
st.success(f"Loaded {len(real_sessions)} sessions, {total_rois} total cells")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v: float, fmt: str = ".3f") -> str:
    return f"{v:{fmt}}" if np.isfinite(v) else "N/A"


def _build_cell_index(sessions: list[dict]) -> list[dict]:
    """Build a flat list of (session_idx, roi_idx, label) for all cells."""
    cells = []
    for si, s in enumerate(sessions):
        for ri in range(s["n_rois"]):
            cells.append({
                "session_idx": si,
                "roi_idx": ri,
                "label": f"{s['exp_id']} — ROI {ri}",
                "exp_id": s["exp_id"],
                "animal_id": s["animal_id"],
                "celltype": s["celltype"],
            })
    return cells


all_cells = _build_cell_index(real_sessions)


def _compute_split_half_for_cell(s: dict, roi_idx: int) -> dict | None:
    """Compute split-half stability for a single cell. Returns None on error."""
    try:
        signal = s["dff"][roi_idx]
        hd = s["hd_deg"]
        mask = s["active"] & ~s["bad_behav"]
        if mask.sum() < 72:
            return None
        return split_temporal_halves(signal, hd, mask)
    except Exception:
        return None


def _compute_light_dark_for_cell(s: dict, roi_idx: int) -> dict | None:
    """Compute light/dark stability for a single cell. Returns None on error."""
    try:
        signal = s["dff"][roi_idx]
        hd = s["hd_deg"]
        mask = s["active"] & ~s["bad_behav"]
        light_on = s["light_on"]
        if mask.sum() < 72:
            return None
        return light_dark_stability(signal, hd, mask, light_on)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_population, tab_single, tab_sliding, tab_lightdark, tab_drift = st.tabs([
    "Population Overview",
    "Single Cell",
    "Sliding Window",
    "Light/Dark",
    "Epoch Drift",
])


# ---------------------------------------------------------------------------
# Population Overview — pooled across all sessions
# ---------------------------------------------------------------------------

with tab_population:
    st.subheader("Population Stability Metrics")
    st.markdown(
        "Split-half correlation and light/dark PD shift pooled across **all** "
        "cells from all sessions. Higher split-half correlation indicates more "
        "temporally stable HD tuning."
    )

    with st.spinner("Computing stability metrics for all cells..."):
        pop_rows = []
        for si, s in enumerate(real_sessions):
            for ri in range(s["n_rois"]):
                sh = _compute_split_half_for_cell(s, ri)
                ld = _compute_light_dark_for_cell(s, ri)
                row = {
                    "exp_id": s["exp_id"],
                    "animal_id": s["animal_id"],
                    "celltype": s["celltype"],
                    "roi_idx": ri,
                    "split_half_r": sh["correlation"] if sh else np.nan,
                    "pd_shift_deg": sh["pd_shift_deg"] if sh else np.nan,
                    "mvl_half1": sh["mvl_half1"] if sh else np.nan,
                    "mvl_half2": sh["mvl_half2"] if sh else np.nan,
                    "ld_correlation": ld["correlation"] if ld else np.nan,
                    "ld_pd_shift_deg": ld["pd_shift_deg"] if ld else np.nan,
                    "mvl_light": ld["mvl_light"] if ld else np.nan,
                    "mvl_dark": ld["mvl_dark"] if ld else np.nan,
                }
                pop_rows.append(row)

    if not pop_rows:
        st.warning("No cells with sufficient data for stability analysis.")
        st.stop()

    df = pd.DataFrame(pop_rows)
    valid = df.dropna(subset=["split_half_r"])

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total cells", len(df))
    col2.metric("Valid cells", len(valid))
    col3.metric(
        "Median split-half r",
        _fmt(valid["split_half_r"].median()) if len(valid) else "N/A",
    )
    ld_valid = df.dropna(subset=["ld_correlation"])
    col4.metric(
        "Median L/D r",
        _fmt(ld_valid["ld_correlation"].median()) if len(ld_valid) else "N/A",
    )

    # Split-half correlation histogram
    st.markdown("#### Split-half correlation distribution")
    if len(valid) > 0:
        fig = px.histogram(
            valid,
            x="split_half_r",
            color="celltype",
            nbins=30,
            barmode="overlay",
            opacity=0.7,
            labels={"split_half_r": "Split-half r", "celltype": "Cell type"},
        )
        fig.update_layout(
            height=350,
            xaxis_title="Split-half correlation (r)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid split-half results to plot.")

    # PD shift histogram
    st.markdown("#### PD shift distribution (first vs second half)")
    if len(valid) > 0:
        fig = px.histogram(
            valid,
            x="pd_shift_deg",
            color="celltype",
            nbins=36,
            barmode="overlay",
            opacity=0.7,
            labels={"pd_shift_deg": "PD shift (deg)", "celltype": "Cell type"},
        )
        fig.update_layout(
            height=350,
            xaxis_title="PD shift (deg)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Light/dark correlation histogram
    st.markdown("#### Light vs dark correlation distribution")
    if len(ld_valid) > 0:
        fig = px.histogram(
            ld_valid,
            x="ld_correlation",
            color="celltype",
            nbins=30,
            barmode="overlay",
            opacity=0.7,
            labels={"ld_correlation": "Light/Dark r", "celltype": "Cell type"},
        )
        fig.update_layout(
            height=350,
            xaxis_title="Light/Dark correlation (r)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    # MVL scatter: half1 vs half2
    st.markdown("#### MVL first half vs second half")
    if len(valid) > 0:
        fig = px.scatter(
            valid,
            x="mvl_half1",
            y="mvl_half2",
            color="celltype",
            hover_data=["exp_id", "roi_idx"],
            opacity=0.7,
            labels={
                "mvl_half1": "MVL (first half)",
                "mvl_half2": "MVL (second half)",
                "celltype": "Cell type",
            },
        )
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", line=dict(dash="dash", color="grey", width=1),
            showlegend=False,
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Summary table
    with st.expander("Full stability table"):
        st.dataframe(
            df.round(3),
            use_container_width=True,
            height=400,
        )


# ---------------------------------------------------------------------------
# Single Cell — session + ROI selector
# ---------------------------------------------------------------------------

with tab_single:
    st.subheader("Single Cell Stability")

    if not all_cells:
        st.warning("No cells available.")
    else:
        # Session selector
        session_labels = [s["exp_id"] for s in real_sessions]
        sel_session_idx = st.selectbox(
            "Session",
            range(len(session_labels)),
            format_func=lambda i: session_labels[i],
            key="stab_single_session",
        )
        ses_data = real_sessions[sel_session_idx]
        n_rois = ses_data["n_rois"]
        sel_cell = st.slider(
            "Cell index", 0, max(0, n_rois - 1), 0, key="stab_single_cell"
        )

        signal = ses_data["dff"][sel_cell]
        hd = ses_data["hd_deg"]
        mask = ses_data["active"] & ~ses_data["bad_behav"]

        st.markdown(
            f"**Session:** {ses_data['exp_id']} --- "
            f"**Animal:** {ses_data['animal_id']} --- "
            f"**Cell type:** {ses_data['celltype']} --- "
            f"**ROI:** {sel_cell}/{n_rois - 1}"
        )

        result = split_temporal_halves(signal, hd, mask)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Correlation", _fmt(result["correlation"]))
        col2.metric("PD shift", f"{_fmt(result['pd_shift_deg'], '.1f')} deg")
        col3.metric("MVL (1st half)", _fmt(result["mvl_half1"]))
        col4.metric("MVL (2nd half)", _fmt(result["mvl_half2"]))

        if not np.isfinite(result["correlation"]):
            st.warning(
                "Correlation is NaN — cell may have insufficient activity or HD "
                "sampling in one half of the session, producing a flat tuning curve."
            )

        # Overlay polar plots
        tc1 = result["tuning_curve_1"]
        tc2 = result["tuning_curve_2"]
        bc = result["bin_centers"]

        theta_p = np.concatenate([np.deg2rad(bc), [np.deg2rad(bc[0])]])
        r1 = np.concatenate([
            np.where(np.isnan(tc1), 0, tc1),
            [tc1[0] if not np.isnan(tc1[0]) else 0],
        ])
        r2 = np.concatenate([
            np.where(np.isnan(tc2), 0, tc2),
            [tc2[0] if not np.isnan(tc2[0]) else 0],
        ])

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r1, theta=np.rad2deg(theta_p),
            mode="lines", fill="toself",
            fillcolor="rgba(65, 105, 225, 0.2)",
            line=dict(color="royalblue", width=2),
            name="First half",
        ))
        fig.add_trace(go.Scatterpolar(
            r=r2, theta=np.rad2deg(theta_p),
            mode="lines", fill="toself",
            fillcolor="rgba(225, 65, 105, 0.2)",
            line=dict(color="#E14169", width=2),
            name="Second half",
        ))
        fig.update_layout(
            height=400,
            polar=dict(
                radialaxis=dict(visible=False),
                angularaxis=dict(
                    direction="clockwise", rotation=90, showticklabels=False
                ),
            ),
            title=f"dF/F0 Tuning Curve — 1st vs 2nd Half (r={result['correlation']:.3f})",
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Sliding Window — single cell
# ---------------------------------------------------------------------------

with tab_sliding:
    st.subheader("Sliding Window Analysis")

    # Session + cell selector
    session_labels_sw = [s["exp_id"] for s in real_sessions]
    sel_sw_session = st.selectbox(
        "Session",
        range(len(session_labels_sw)),
        format_func=lambda i: session_labels_sw[i],
        key="stab_sw_session",
    )
    sw_ses = real_sessions[sel_sw_session]
    sel_sw_cell = st.slider(
        "Cell index", 0, max(0, sw_ses["n_rois"] - 1), 0, key="stab_sw_cell"
    )

    col_w, col_s = st.columns(2)
    with col_w:
        win_frames = st.slider(
            "Window (frames)", 500, 3000, 1500, 100, key="stab_win"
        )
    with col_s:
        step = st.slider(
            "Step (frames)", 100, 1000, 300, 50, key="stab_step"
        )

    sw_signal = sw_ses["dff"][sel_sw_cell]
    sw_hd = sw_ses["hd_deg"]
    sw_mask = sw_ses["active"] & ~sw_ses["bad_behav"]

    sw = sliding_window_stability(
        sw_signal, sw_hd, sw_mask, window_frames=win_frames, step_frames=step
    )

    if sw["n_windows"] > 1:
        col_sw1, col_sw2 = st.columns(2)
        with col_sw1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sw["window_centers"], y=sw["mvls"],
                mode="lines+markers", marker_color="royalblue",
            ))
            fig.update_layout(
                height=300, title="MVL Over Time",
                xaxis_title="Frame", yaxis_title="MVL",
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_sw2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sw["window_centers"], y=sw["preferred_dirs"],
                mode="lines+markers", marker_color="orange",
            ))
            fig.update_layout(
                height=300, title="Preferred Direction Over Time",
                xaxis_title="Frame", yaxis_title="PD (deg)",
                yaxis=dict(range=[0, 360]),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"**Windows:** {sw['n_windows']} --- "
            f"**MVL range:** {sw['mvls'].min():.3f}--{sw['mvls'].max():.3f} --- "
            f"**MVL std:** {np.std(sw['mvls']):.3f} --- "
            f"**PD range:** {sw['preferred_dirs'].min():.0f} deg--"
            f"{sw['preferred_dirs'].max():.0f} deg"
        )
    else:
        st.warning("Not enough frames for sliding window analysis.")


# ---------------------------------------------------------------------------
# Light/Dark — single cell + population scatter
# ---------------------------------------------------------------------------

with tab_lightdark:
    st.subheader("Light vs Dark Tuning")

    # Session + cell selector
    session_labels_ld = [s["exp_id"] for s in real_sessions]
    sel_ld_session = st.selectbox(
        "Session",
        range(len(session_labels_ld)),
        format_func=lambda i: session_labels_ld[i],
        key="stab_ld_session",
    )
    ld_ses = real_sessions[sel_ld_session]

    if "light_on" not in ld_ses:
        st.warning(
            "No light_on data available for this session. "
            "Light/dark analysis requires TDMS-derived light timestamps in sync.h5."
        )
    else:
        sel_ld_cell = st.slider(
            "Cell index", 0, max(0, ld_ses["n_rois"] - 1), 0,
            key="stab_ld_cell",
        )

        ld_signal = ld_ses["dff"][sel_ld_cell]
        ld_hd = ld_ses["hd_deg"]
        ld_mask = ld_ses["active"] & ~ld_ses["bad_behav"]
        light_on = ld_ses["light_on"]

        n_light = light_on.sum()
        n_dark = (~light_on).sum()
        st.markdown(
            f"**Light frames:** {n_light} ({n_light / len(ld_signal) * 100:.0f}%) --- "
            f"**Dark frames:** {n_dark} ({n_dark / len(ld_signal) * 100:.0f}%)"
        )

        ld = light_dark_stability(ld_signal, ld_hd, ld_mask, light_on)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Correlation", _fmt(ld["correlation"]))
        col2.metric("PD shift", f"{_fmt(ld['pd_shift_deg'], '.1f')} deg")
        col3.metric("MVL (light)", _fmt(ld["mvl_light"]))
        col4.metric("MVL (dark)", _fmt(ld["mvl_dark"]))

        if not np.isfinite(ld["correlation"]):
            st.warning(
                "Correlation is NaN — cell may have insufficient activity during "
                "light or dark epochs, producing a flat tuning curve."
            )

        tc_l = ld["tuning_curve_light"]
        tc_d = ld["tuning_curve_dark"]
        bc_ld = ld["bin_centers"]

        theta_ld = np.concatenate([np.deg2rad(bc_ld), [np.deg2rad(bc_ld[0])]])
        r_l = np.concatenate([
            np.where(np.isnan(tc_l), 0, tc_l),
            [tc_l[0] if not np.isnan(tc_l[0]) else 0],
        ])
        r_d = np.concatenate([
            np.where(np.isnan(tc_d), 0, tc_d),
            [tc_d[0] if not np.isnan(tc_d[0]) else 0],
        ])

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r_l, theta=np.rad2deg(theta_ld),
            mode="lines", fill="toself",
            fillcolor="rgba(255, 200, 50, 0.2)",
            line=dict(color="gold", width=2),
            name="Light ON",
        ))
        fig.add_trace(go.Scatterpolar(
            r=r_d, theta=np.rad2deg(theta_ld),
            mode="lines", fill="toself",
            fillcolor="rgba(100, 100, 200, 0.2)",
            line=dict(color="slateblue", width=2),
            name="Light OFF",
        ))
        fig.update_layout(
            height=400,
            polar=dict(
                radialaxis=dict(visible=False),
                angularaxis=dict(
                    direction="clockwise", rotation=90, showticklabels=False
                ),
            ),
            title=f"dF/F0 Light vs Dark Tuning (r={ld['correlation']:.3f})",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "In real data, visual cue removal (lights off) may cause HD tuning to drift "
        "or become less sharp. This is the key manipulation in the hm2p experiment: "
        "testing whether Penk+ and non-Penk CamKII+ RSP neurons differ in their "
        "reliance on visual vs path-integration cues."
    )


# ---------------------------------------------------------------------------
# Epoch Drift — per-epoch PD tracking
# ---------------------------------------------------------------------------

with tab_drift:
    st.subheader("Per-Epoch Drift Tracking")
    st.markdown(
        "Tracks preferred direction across sequential light/dark epochs within "
        "a session. Cumulative drift reveals whether HD tuning walks during "
        "darkness."
    )

    session_labels_dr = [s["exp_id"] for s in real_sessions]
    sel_dr_session = st.selectbox(
        "Session",
        range(len(session_labels_dr)),
        format_func=lambda i: session_labels_dr[i],
        key="stab_drift_session",
    )
    dr_ses = real_sessions[sel_dr_session]

    if "light_on" not in dr_ses:
        st.warning("No light_on data for this session.")
    else:
        sel_dr_cell = st.slider(
            "Cell index", 0, max(0, dr_ses["n_rois"] - 1), 0,
            key="stab_drift_cell",
        )

        dr_signal = dr_ses["dff"][sel_dr_cell]
        dr_hd = dr_ses["hd_deg"]
        dr_mask = dr_ses["active"] & ~dr_ses["bad_behav"]
        dr_light = dr_ses["light_on"]

        drift = drift_per_epoch(dr_signal, dr_hd, dr_mask, dr_light)

        if drift["n_epochs"] > 1:
            colors = [
                "gold" if is_l else "slateblue"
                for is_l in drift["epoch_is_light"]
            ]
            labels = [
                "Light" if is_l else "Dark"
                for is_l in drift["epoch_is_light"]
            ]

            col_dr1, col_dr2 = st.columns(2)
            with col_dr1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=drift["epoch_centers"],
                    y=drift["epoch_pds"],
                    mode="lines+markers",
                    marker=dict(color=colors, size=8),
                    text=labels,
                    hovertemplate="Frame %{x}<br>PD=%{y:.1f} deg<br>%{text}",
                ))
                fig.update_layout(
                    height=300,
                    title="Preferred Direction per Epoch",
                    xaxis_title="Frame",
                    yaxis_title="PD (deg)",
                    yaxis=dict(range=[0, 360]),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_dr2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=drift["epoch_centers"],
                    y=drift["cumulative_drift"],
                    mode="lines+markers",
                    marker=dict(color=colors, size=8),
                    text=labels,
                    hovertemplate="Frame %{x}<br>Cum. drift=%{y:.1f} deg<br>%{text}",
                ))
                fig.update_layout(
                    height=300,
                    title="Cumulative PD Drift from First Epoch",
                    xaxis_title="Frame",
                    yaxis_title="Cumulative drift (deg)",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                f"**Epochs:** {drift['n_epochs']} --- "
                f"**Light epochs:** {sum(drift['epoch_is_light'])} --- "
                f"**Dark epochs:** {sum(~drift['epoch_is_light'])} --- "
                f"**Total drift:** {drift['cumulative_drift'][-1]:.1f} deg"
            )
        else:
            st.warning(
                "Not enough epochs for drift analysis (need at least 2 "
                "light/dark transitions)."
            )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Stability analysis compares HD tuning across time periods using Pearson "
    "correlation of occupancy-normalised dF/F0 tuning curves and circular "
    "preferred direction shift."
)
