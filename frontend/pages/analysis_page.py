"""Analysis page -- multi-signal HD & place tuning, pooled across sessions.

Loads sync.h5 data for all sessions via load_all_sync_data(), with optional
filtering by celltype, animal, ROI type, and individual session.  All six
analysis tabs work on the pooled (or filtered) data.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

log = logging.getLogger("hm2p.frontend.analysis")

_signal_labels = {"dff": "dF/F\u2080", "deconv": "Deconv", "events": "Events"}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

st.title("Analysis -- Multi-Signal HD & Place Tuning")

from frontend.data import load_all_sync_data, session_filter_sidebar

all_data = load_all_sync_data()
if all_data["n_sessions"] == 0:
    st.warning("No data available yet. Run pipeline stages 0-5 first.")
    st.stop()

sessions = session_filter_sidebar(all_data["sessions"], key_prefix="analysis")

if not sessions:
    st.warning("No sessions match the current filters.")
    st.stop()

# Optional session selector (default = all sessions pooled)
session_labels = ["All sessions (pooled)"] + [s["exp_id"] for s in sessions]
sel_session_label = st.sidebar.selectbox(
    "Session", session_labels, index=0, key="analysis_session_select",
)

if sel_session_label == "All sessions (pooled)":
    active_sessions = sessions
else:
    active_sessions = [s for s in sessions if s["exp_id"] == sel_session_label]

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

st.sidebar.header("Parameters")
speed_threshold = st.sidebar.slider("Speed threshold (cm/s)", 0.0, 10.0, 2.5, 0.5)

st.sidebar.subheader("HD Tuning")
hd_n_bins = st.sidebar.select_slider("HD bins", [12, 18, 24, 36, 72], value=36)
hd_sigma = st.sidebar.slider("HD smoothing (deg)", 0.0, 20.0, 6.0, 1.0)

st.sidebar.subheader("Place Tuning")
place_bin = st.sidebar.slider("Place bin size (cm)", 1.0, 10.0, 2.5, 0.5)
place_sigma = st.sidebar.slider("Place smoothing (cm)", 0.0, 10.0, 3.0, 0.5)

n_shuffles = st.sidebar.number_input("Bootstrap shuffles", 100, 2000, 500, 100)

# Summary
total_rois = sum(s["n_rois"] for s in active_sessions)
total_frames = sum(s["n_frames"] for s in active_sessions)
st.sidebar.markdown(
    f"**Sessions:** {len(active_sessions)} | "
    f"**ROIs:** {total_rois} | "
    f"**Total frames:** {total_frames}"
)

st.success(
    f"Loaded {len(active_sessions)} session(s), "
    f"{total_rois} total ROIs"
)

# Determine which signals are available across loaded sessions
available_signals = ["dff"]
if any(s.get("deconv") is not None for s in sessions):
    available_signals.append("deconv")
if any(s.get("event_masks") is not None for s in sessions):
    available_signals.append("events")

# ---------------------------------------------------------------------------
# Helpers -- per-session data access
# ---------------------------------------------------------------------------


def _get_signal(ses: dict, roi: int, signal_type: str) -> np.ndarray:
    """Get signal array for a given ROI and signal type.

    Falls back to dff if the requested signal is not available.
    """
    if signal_type == "deconv":
        arr = ses.get("deconv")
        if arr is not None:
            return arr[roi]
        return ses["dff"][roi]
    elif signal_type == "events":
        arr = ses.get("event_masks")
        if arr is not None:
            return arr[roi].astype(np.float32)
        return np.zeros(ses["n_frames"], dtype=np.float32)
    return ses["dff"][roi]


def _get_session_masks(ses: dict) -> dict:
    """Compute common masks for a session."""
    active_mask = ses["active"] & ~ses["bad_behav"]
    speed = ses["speed_cm_s"]
    moving_mask = (speed >= speed_threshold) & active_mask
    light_on = ses["light_on"].astype(bool)
    return {
        "active": active_mask,
        "moving": moving_mask,
        "light_on": light_on,
        "speed": speed,
        "hd_deg": ses["hd_deg"],
    }


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_compare, tab_activity, tab_hd, tab_place, tab_robust, tab_population = st.tabs([
    "Signal Comparison",
    "Activity by Condition",
    "HD Tuning",
    "Place Tuning",
    "Robustness",
    "Population Summary",
])

# ---- Tab 1: Multi-signal comparison ----
with tab_compare:
    st.subheader("Cross-Signal Comparison")
    if len(available_signals) > 1:
        st.markdown(
            "Compare HD tuning metrics across signal types (dF/F, deconvolved, "
            "events). If the same cells are significant across signals, "
            "conclusions are robust."
        )
    else:
        st.markdown(
            "Compare HD tuning metrics computed with dF/F across all sessions. "
            "Deconvolved spikes and event masks will appear here when sync.h5 "
            "files are regenerated with Stage 5."
        )

    from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length

    # Compute MVL for each ROI across all sessions, for each signal
    mvl_by_signal: dict[str, np.ndarray] = {}
    roi_labels = []
    for sig_type in available_signals:
        mvl_data = []
        for ses in active_sessions:
            m = _get_session_masks(ses)
            for roi in range(ses["n_rois"]):
                if m["moving"].sum() > 50:
                    sig = _get_signal(ses, roi, sig_type)
                    tc, centers = compute_hd_tuning_curve(
                        sig, m["hd_deg"], m["moving"],
                        n_bins=hd_n_bins, smoothing_sigma_deg=hd_sigma,
                    )
                    mvl_data.append(mean_vector_length(tc, centers))
                else:
                    mvl_data.append(np.nan)
                if sig_type == available_signals[0]:
                    roi_labels.append(f"{ses['exp_id']}:ROI{roi}")
        mvl_by_signal[sig_type] = np.array(mvl_data)

    # For backwards compat (used in significance tab)
    mvl_arr = mvl_by_signal["dff"]

    # MVL distribution — overlaid per signal
    st.subheader("MVL Distribution")
    fig = go.Figure()
    for sig_type in available_signals:
        vals = mvl_by_signal[sig_type]
        vals = vals[np.isfinite(vals)]
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=20, name=_signal_labels.get(sig_type, sig_type),
            opacity=0.6,
        ))
    fig.update_layout(
        xaxis_title="Mean Vector Length",
        yaxis_title="Count",
        barmode="overlay",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cross-signal MVL scatter (if multiple signals available)
    if len(available_signals) >= 2:
        st.subheader("Cross-Signal MVL Agreement")
        ref_sig = available_signals[0]
        for other_sig in available_signals[1:]:
            ref_vals = mvl_by_signal[ref_sig]
            other_vals = mvl_by_signal[other_sig]
            valid = np.isfinite(ref_vals) & np.isfinite(other_vals)
            if valid.sum() > 2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ref_vals[valid], y=other_vals[valid],
                    mode="markers", marker=dict(size=5),
                    text=[roi_labels[i] for i in np.flatnonzero(valid)],
                ))
                maxv = max(ref_vals[valid].max(), other_vals[valid].max())
                fig.add_trace(go.Scatter(
                    x=[0, maxv], y=[0, maxv], mode="lines",
                    line=dict(dash="dash", color="gray"), showlegend=False,
                ))
                corr = np.corrcoef(ref_vals[valid], other_vals[valid])[0, 1]
                fig.update_layout(
                    title=f"{_signal_labels[ref_sig]} vs {_signal_labels[other_sig]} (r={corr:.3f})",
                    xaxis_title=f"MVL ({_signal_labels[ref_sig]})",
                    yaxis_title=f"MVL ({_signal_labels[other_sig]})",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

    # Per-ROI bar chart (mean dff)
    st.subheader("Mean dF/F per ROI")
    mean_vals = []
    for ses in active_sessions:
        for roi in range(ses["n_rois"]):
            mean_vals.append(np.nanmean(ses["dff"][roi]))

    fig = go.Figure()
    fig.add_trace(go.Bar(y=mean_vals))
    fig.update_layout(
        xaxis_title="ROI index (pooled)",
        yaxis_title="Mean dF/F",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Significance agreement
    st.subheader("Significance Agreement")
    st.caption(
        "Run significance testing and see which ROIs are HD-tuned."
    )

    if st.button("Run significance test", key="sig_compare"):
        from hm2p.analysis.significance import hd_tuning_significance

        sig_results = []
        progress = st.progress(0)
        roi_idx = 0

        for ses in active_sessions:
            m = _get_session_masks(ses)
            for roi in range(ses["n_rois"]):
                p_val = np.nan
                is_sig = False
                if m["moving"].sum() > 50:
                    res = hd_tuning_significance(
                        ses["dff"][roi], m["hd_deg"], m["moving"],
                        n_shuffles=n_shuffles,
                        n_bins=hd_n_bins,
                        smoothing_sigma_deg=hd_sigma,
                        rng=np.random.default_rng(roi_idx),
                    )
                    p_val = res["p_value"]
                    is_sig = p_val < 0.05
                sig_results.append({
                    "Session": ses["exp_id"],
                    "Celltype": ses["celltype"],
                    "ROI": roi,
                    "MVL": f"{mvl_arr[roi_idx]:.4f}" if np.isfinite(mvl_arr[roi_idx]) else "---",
                    "p-value": f"{p_val:.4f}" if np.isfinite(p_val) else "---",
                    "Significant": "Y" if is_sig else "-",
                })
                roi_idx += 1
                progress.progress(roi_idx / total_rois)

        progress.empty()

        df_sig = pd.DataFrame(sig_results)
        n_sig = sum(1 for r in sig_results if r["Significant"] == "Y")
        st.markdown(
            f"**{n_sig} / {total_rois} ROIs** significantly HD-tuned "
            f"({n_sig / total_rois:.1%})"
        )
        st.dataframe(df_sig, use_container_width=True)


# ---- Tab 2: Activity by condition ----
with tab_activity:
    st.subheader("Activity by Condition (2x2: Movement x Light)")

    from hm2p.analysis.activity import compute_batch_activity
    from hm2p.plotting import format_pvalue, paired_condition_scatter

    # Collect activity results across all sessions
    all_results = []
    all_session_labels = []
    fps_values = []

    for ses in active_sessions:
        m = _get_session_masks(ses)
        n_rois_ses = ses["n_rois"]
        n_frames_ses = ses["n_frames"]
        # Estimate fps from frame_times
        ft = ses["frame_times"]
        if len(ft) > 1:
            fps = 1.0 / np.median(np.diff(ft))
        else:
            fps = 9.8

        # Use real event_masks from sync.h5 if available, else zeros
        evts = ses.get("event_masks")
        if evts is None:
            evts = np.zeros_like(ses["dff"], dtype=bool)

        results = compute_batch_activity(
            ses["dff"], evts, m["speed"], m["light_on"],
            m["active"], fps, speed_threshold=speed_threshold,
        )
        all_results.extend(results)
        all_session_labels.extend([ses["exp_id"]] * n_rois_ses)

    if not all_results:
        st.info("No activity data computed.")
    else:
        n_rois_total = len(all_results)

        # Paired scatter: Light vs Dark within each movement state
        st.markdown("**Event rate: Light vs Dark (paired per ROI)**")
        col1, col2 = st.columns(2)
        with col1:
            mov_light = np.array([r["moving_light_event_rate"] for r in all_results])
            mov_dark = np.array([r["moving_dark_event_rate"] for r in all_results])
            fig_ml, stat_ml = paired_condition_scatter(
                mov_light, mov_dark, "Light", "Dark",
                "Event Rate (moving)", height=400, width=400,
            )
            st.plotly_chart(fig_ml, use_container_width=True)
            st.markdown(f"Wilcoxon: {format_pvalue(stat_ml['p'])}, n={stat_ml['n']}")

        with col2:
            stat_light = np.array([r["stationary_light_event_rate"] for r in all_results])
            stat_dark = np.array([r["stationary_dark_event_rate"] for r in all_results])
            fig_sl, stat_sl = paired_condition_scatter(
                stat_light, stat_dark, "Light", "Dark",
                "Event Rate (stationary)", height=400, width=400,
            )
            st.plotly_chart(fig_sl, use_container_width=True)
            st.markdown(f"Wilcoxon: {format_pvalue(stat_sl['p'])}, n={stat_sl['n']}")

        # Modulation indices scatter
        st.markdown("**Modulation Indices**")
        mov_mod = [r["movement_modulation"] for r in all_results]
        light_mod = [r["light_modulation"] for r in all_results]
        fig_mod = go.Figure()
        fig_mod.add_trace(go.Scatter(
            x=mov_mod, y=light_mod, mode="markers",
            text=[f"{all_session_labels[i]}:ROI{i}" for i in range(n_rois_total)],
            marker=dict(size=8),
        ))
        fig_mod.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_mod.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_mod.update_layout(
            xaxis_title="Movement Modulation",
            yaxis_title="Light Modulation",
            height=400,
        )
        st.plotly_chart(fig_mod, use_container_width=True)

        # Mean signal: Light vs Dark
        st.subheader("Mean Signal by Condition")
        col1, col2 = st.columns(2)
        with col1:
            ml_sig = np.array([r["moving_light_mean_signal"] for r in all_results])
            md_sig = np.array([r["moving_dark_mean_signal"] for r in all_results])
            fig_ms, stat_ms = paired_condition_scatter(
                ml_sig, md_sig, "Light", "Dark",
                "Mean Signal (moving)", height=400, width=400,
            )
            st.plotly_chart(fig_ms, use_container_width=True)
            st.markdown(f"Wilcoxon: {format_pvalue(stat_ms['p'])}, n={stat_ms['n']}")

        with col2:
            sl_sig = np.array([r["stationary_light_mean_signal"] for r in all_results])
            sd_sig = np.array([r["stationary_dark_mean_signal"] for r in all_results])
            fig_ss, stat_ss = paired_condition_scatter(
                sl_sig, sd_sig, "Light", "Dark",
                "Mean Signal (stationary)", height=400, width=400,
            )
            st.plotly_chart(fig_ss, use_container_width=True)
            st.markdown(f"Wilcoxon: {format_pvalue(stat_ss['p'])}, n={stat_ss['n']}")

        # Summary table
        st.markdown("### Per-ROI Summary")
        rows = []
        roi_global = 0
        for ses in active_sessions:
            for roi in range(ses["n_rois"]):
                r = all_results[roi_global]
                rows.append({
                    "Session": ses["exp_id"],
                    "ROI": roi,
                    "Move+Light": f"{r['moving_light_event_rate']:.3f}",
                    "Move+Dark": f"{r['moving_dark_event_rate']:.3f}",
                    "Still+Light": f"{r['stationary_light_event_rate']:.3f}",
                    "Still+Dark": f"{r['stationary_dark_event_rate']:.3f}",
                    "Move MI": f"{r['movement_modulation']:.3f}",
                    "Light MI": f"{r['light_modulation']:.3f}",
                })
                roi_global += 1
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---- Tab 3: HD Tuning ----
with tab_hd:
    st.subheader("Head Direction Tuning")

    from hm2p.analysis.tuning import (
        compute_hd_tuning_curve,
        mean_vector_length,
        preferred_direction,
    )

    # Signal selector
    if len(available_signals) > 1:
        hd_signal_type = st.radio(
            "Signal", available_signals,
            format_func=lambda s: _signal_labels.get(s, s),
            horizontal=True, key="hd_signal",
        )
    else:
        hd_signal_type = "dff"

    # Session + ROI picker for single-cell view
    hd_session_labels = [s["exp_id"] for s in active_sessions]
    hd_ses_idx = st.selectbox(
        "Session (single cell view)", range(len(hd_session_labels)),
        format_func=lambda i: hd_session_labels[i], key="hd_ses",
    )
    ses_hd = active_sessions[hd_ses_idx]
    n_rois_hd = ses_hd["n_rois"]
    roi_select = st.selectbox("ROI", list(range(n_rois_hd)), key="hd_roi")
    m_hd = _get_session_masks(ses_hd)

    sig = _get_signal(ses_hd, roi_select, hd_signal_type)
    moving_mask = m_hd["moving"]
    hd_deg = m_hd["hd_deg"]
    light_on = m_hd["light_on"]
    moving_light_mask = moving_mask & light_on
    moving_dark_mask = moving_mask & ~light_on

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### All moving frames")
        if moving_mask.sum() > 50:
            tc, centers = compute_hd_tuning_curve(
                sig, hd_deg, moving_mask,
                n_bins=hd_n_bins, smoothing_sigma_deg=hd_sigma,
            )
            mvl = mean_vector_length(tc, centers)
            pd_val = preferred_direction(tc, centers)

            tc_closed = np.append(tc, tc[0])
            centers_closed = np.append(centers, centers[0] + 360)
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=tc_closed, theta=centers_closed,
                mode="lines", fill="toself", name="All",
            ))
            fig.update_layout(
                title=f"ROI {roi_select} -- MVL={mvl:.3f}, PD={pd_val:.0f}",
                polar=dict(radialaxis=dict(visible=False), angularaxis=dict(showticklabels=False)),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Light vs Dark")
        if moving_light_mask.sum() > 50 and moving_dark_mask.sum() > 50:
            tc_l, centers = compute_hd_tuning_curve(
                sig, hd_deg, moving_light_mask,
                n_bins=hd_n_bins, smoothing_sigma_deg=hd_sigma,
            )
            tc_d, _ = compute_hd_tuning_curve(
                sig, hd_deg, moving_dark_mask,
                n_bins=hd_n_bins, smoothing_sigma_deg=hd_sigma,
            )
            mvl_l = mean_vector_length(tc_l, centers)
            mvl_d = mean_vector_length(tc_d, centers)

            tc_l_c = np.append(tc_l, tc_l[0])
            tc_d_c = np.append(tc_d, tc_d[0])
            centers_c = np.append(centers, centers[0] + 360)

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=tc_l_c, theta=centers_c, mode="lines",
                name=f"Light (MVL={mvl_l:.3f})", line=dict(color="gold"),
            ))
            fig.add_trace(go.Scatterpolar(
                r=tc_d_c, theta=centers_c, mode="lines",
                name=f"Dark (MVL={mvl_d:.3f})", line=dict(color="navy"),
            ))
            fig.update_layout(
                title=f"ROI {roi_select} -- Light vs Dark",
                polar=dict(radialaxis=dict(visible=False), angularaxis=dict(showticklabels=False)),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            from hm2p.analysis.comparison import (
                preferred_direction_shift,
                tuning_curve_correlation,
            )

            corr = tuning_curve_correlation(tc_l, tc_d)
            pd_shift = preferred_direction_shift(tc_l, tc_d, centers)
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("TC correlation", f"{corr:.3f}")
            col_m2.metric("PD shift (deg)", f"{pd_shift:.1f}")

    # Population HD summary (all sessions pooled)
    st.markdown("### Population HD Summary (all sessions)")
    mvls_pop = []
    pds_pop = []
    pop_labels = []
    for ses in active_sessions:
        m = _get_session_masks(ses)
        for roi in range(ses["n_rois"]):
            if m["moving"].sum() > 50:
                pop_sig = _get_signal(ses, roi, hd_signal_type)
                tc_i, c_i = compute_hd_tuning_curve(
                    pop_sig, m["hd_deg"], m["moving"],
                    n_bins=hd_n_bins, smoothing_sigma_deg=hd_sigma,
                )
                mvls_pop.append(mean_vector_length(tc_i, c_i))
                pds_pop.append(preferred_direction(tc_i, c_i))
                pop_labels.append(f"{ses['exp_id']}:ROI{roi}")

    if mvls_pop:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=mvls_pop, nbinsx=20))
            fig.update_layout(
                title="MVL Distribution",
                xaxis_title="MVL", yaxis_title="Count", height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=mvls_pop, theta=pds_pop,
                mode="markers",
                marker=dict(size=6),
                text=pop_labels,
            ))
            fig.update_layout(
                title="Preferred Directions (radius = MVL)",
                polar=dict(radialaxis=dict(visible=False), angularaxis=dict(showticklabels=False)),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)


# ---- Tab 4: Place Tuning ----
with tab_place:
    st.subheader("Place Tuning")

    from hm2p.analysis.tuning import (
        compute_place_rate_map,
        spatial_coherence,
        spatial_information,
        spatial_sparsity,
    )

    # Session + ROI picker
    pl_session_labels = [s["exp_id"] for s in active_sessions]
    pl_ses_idx = st.selectbox(
        "Session", range(len(pl_session_labels)),
        format_func=lambda i: pl_session_labels[i], key="pl_ses",
    )
    ses_pl = active_sessions[pl_ses_idx]
    n_rois_pl = ses_pl["n_rois"]
    roi_select_pl = st.selectbox("ROI", list(range(n_rois_pl)), key="place_roi")

    m_pl = _get_session_masks(ses_pl)
    sig_pl = ses_pl["dff"][roi_select_pl]
    moving_mask_pl = m_pl["moving"]
    light_on_pl = m_pl["light_on"]

    # Estimate fps
    ft_pl = ses_pl["frame_times"]
    if len(ft_pl) > 1:
        fps_pl = 1.0 / np.median(np.diff(ft_pl))
    else:
        fps_pl = 9.8

    # Check if position data is available
    if ses_pl.get("x_mm") is not None:
        # Convert mm to cm
        x_cm = ses_pl["x_mm"] / 10.0
        y_cm = ses_pl["y_mm"] / 10.0

        # --- All moving frames rate map ---
        st.subheader("Place Rate Map (all moving frames)")
        rate_map, occ_map, bex, bey = compute_place_rate_map(
            sig_pl, x_cm, y_cm, moving_mask_pl,
            bin_size=place_bin, smoothing_sigma=place_sigma,
            min_occupancy_s=0.5, fps=fps_pl,
        )

        fig = go.Figure(data=go.Heatmap(
            z=rate_map,
            x=0.5 * (bex[:-1] + bex[1:]),
            y=0.5 * (bey[:-1] + bey[1:]),
            colorscale="Hot",
            colorbar=dict(title="dF/F\u2080"),
        ))
        fig.update_layout(
            xaxis_title="X (cm)", yaxis_title="Y (cm)",
            height=450, yaxis=dict(scaleanchor="x"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Spatial metrics
        si = spatial_information(rate_map, occ_map)
        coh = spatial_coherence(rate_map)
        spar = spatial_sparsity(rate_map, occ_map)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Spatial Info (bits/event)", f"{si:.4f}")
        mc2.metric("Coherence", f"{coh:.3f}" if np.isfinite(coh) else "N/A")
        mc3.metric("Sparsity", f"{spar:.3f}" if np.isfinite(spar) else "N/A")

        # --- Light vs Dark rate maps ---
        st.subheader("Light vs Dark Rate Maps")
        moving_light_pl = moving_mask_pl & light_on_pl
        moving_dark_pl = moving_mask_pl & ~light_on_pl

        col_l, col_d = st.columns(2)
        with col_l:
            st.markdown("**Light ON**")
            if moving_light_pl.sum() > 50:
                rm_l, om_l, bex_l, bey_l = compute_place_rate_map(
                    sig_pl, x_cm, y_cm, moving_light_pl,
                    bin_size=place_bin, smoothing_sigma=place_sigma,
                    min_occupancy_s=0.5, fps=fps_pl,
                )
                fig_l = go.Figure(data=go.Heatmap(
                    z=rm_l,
                    x=0.5 * (bex_l[:-1] + bex_l[1:]),
                    y=0.5 * (bey_l[:-1] + bey_l[1:]),
                    colorscale="Hot",
                    colorbar=dict(title="dF/F\u2080"),
                ))
                fig_l.update_layout(
                    xaxis_title="X (cm)", yaxis_title="Y (cm)",
                    height=400, yaxis=dict(scaleanchor="x"),
                )
                st.plotly_chart(fig_l, use_container_width=True)
                si_l = spatial_information(rm_l, om_l)
                st.caption(f"SI = {si_l:.4f} bits/event")
            else:
                st.info("Too few light-on moving frames for rate map.")

        with col_d:
            st.markdown("**Light OFF (dark)**")
            if moving_dark_pl.sum() > 50:
                rm_d, om_d, bex_d, bey_d = compute_place_rate_map(
                    sig_pl, x_cm, y_cm, moving_dark_pl,
                    bin_size=place_bin, smoothing_sigma=place_sigma,
                    min_occupancy_s=0.5, fps=fps_pl,
                )
                fig_d = go.Figure(data=go.Heatmap(
                    z=rm_d,
                    x=0.5 * (bex_d[:-1] + bex_d[1:]),
                    y=0.5 * (bey_d[:-1] + bey_d[1:]),
                    colorscale="Hot",
                    colorbar=dict(title="dF/F\u2080"),
                ))
                fig_d.update_layout(
                    xaxis_title="X (cm)", yaxis_title="Y (cm)",
                    height=400, yaxis=dict(scaleanchor="x"),
                )
                st.plotly_chart(fig_d, use_container_width=True)
                si_d = spatial_information(rm_d, om_d)
                st.caption(f"SI = {si_d:.4f} bits/event")
            else:
                st.info("Too few dark moving frames for rate map.")

    else:
        st.info(
            "Place tuning requires x/y position data, which is not available "
            "in this session's sync.h5. Position-based rate maps will appear "
            "when x_mm/y_mm are added to the sync output."
        )

        # Occupancy in HD space as a proxy
        st.subheader("HD Occupancy (proxy)")
        hd_moving = m_pl["hd_deg"][moving_mask_pl]
        if len(hd_moving) > 50:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=hd_moving, nbinsx=36, name="HD occupancy"))
            fig.update_layout(
                xaxis_title="Head Direction (deg)",
                yaxis_title="Frame count",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)


# ---- Tab 5: Robustness ----
with tab_robust:
    st.subheader("Robustness -- Parameter Sensitivity")
    st.markdown(
        "### Parameter Grid (HD Tuning)\n"
        "Test how the fraction of significant cells changes across "
        "bin sizes and smoothing levels, pooled across all sessions."
    )

    bin_options = st.multiselect(
        "HD bins", [12, 18, 24, 36, 72], default=[18, 36], key="rob_bins",
    )
    sigma_options = st.multiselect(
        "Smoothing (deg)", [0, 3, 6, 9, 12], default=[3, 6], key="rob_sigma",
    )
    grid_shuffles = st.number_input(
        "Shuffles per cell", 50, 1000, 100, 50, key="grid_shuf",
    )

    grid_signals = st.multiselect(
        "Signals to test", available_signals, default=available_signals,
        format_func=lambda s: _signal_labels.get(s, s), key="rob_signals",
    )

    if st.button("Run Parameter Grid", key="run_grid"):
        from hm2p.analysis.significance import hd_tuning_significance
        from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length

        progress = st.progress(0)
        total = len(grid_signals) * len(bin_options) * len(sigma_options)
        grid_results = []
        step = 0

        for sig_type in grid_signals:
            for nb in bin_options:
                for sg in sigma_options:
                    n_sig = 0
                    n_tested = 0
                    roi_global = 0
                    for ses in active_sessions:
                        m = _get_session_masks(ses)
                        for roi in range(ses["n_rois"]):
                            if m["moving"].sum() > 50:
                                sig_arr = _get_signal(ses, roi, sig_type)
                                res = hd_tuning_significance(
                                    sig_arr, m["hd_deg"], m["moving"],
                                    n_shuffles=grid_shuffles,
                                    n_bins=nb, smoothing_sigma_deg=float(sg),
                                    rng=np.random.default_rng(roi_global),
                                )
                                if res["p_value"] < 0.05:
                                    n_sig += 1
                                n_tested += 1
                            roi_global += 1

                    frac = n_sig / n_tested if n_tested > 0 else 0
                    grid_results.append({
                        "signal": _signal_labels.get(sig_type, sig_type),
                        "bins": nb,
                        "sigma": sg,
                        "n_significant": n_sig,
                        "n_tested": n_tested,
                        "frac_significant": frac,
                    })
                    step += 1
                    progress.progress(step / total)

        progress.empty()

        df = pd.DataFrame(grid_results)
        st.dataframe(df, use_container_width=True)

        if len(df) > 1:
            import plotly.express as px

            fig = px.scatter(
                df, x="bins", y="sigma", size="frac_significant",
                color="signal", hover_data=["n_significant", "n_tested"],
                title="Fraction of HD-tuned cells across parameters",
            )
            st.plotly_chart(fig, use_container_width=True)


# ---- Tab 6: Population Summary ----
with tab_population:
    st.subheader("Population Summary")
    st.markdown(
        "Overview of all ROIs across all loaded sessions, with metrics "
        "for each available signal type."
    )

    from hm2p.analysis.tuning import (
        compute_hd_tuning_curve,
        mean_vector_length,
        preferred_direction,
    )

    pop_rows = []
    for ses in active_sessions:
        m = _get_session_masks(ses)
        for roi in range(ses["n_rois"]):
            row = {
                "Session": ses["exp_id"],
                "Animal": ses["animal_id"],
                "Celltype": ses["celltype"],
                "ROI": roi,
                "dff_mean": f"{np.nanmean(ses['dff'][roi]):.4f}",
                "dff_max": f"{np.nanmax(ses['dff'][roi]):.3f}",
            }

            for sig_type in available_signals:
                sig_arr = _get_signal(ses, roi, sig_type)
                label = _signal_labels.get(sig_type, sig_type)
                if m["moving"].sum() > 50:
                    tc, centers = compute_hd_tuning_curve(
                        sig_arr, m["hd_deg"], m["moving"],
                        n_bins=hd_n_bins, smoothing_sigma_deg=hd_sigma,
                    )
                    row[f"{label}_mvl"] = f"{mean_vector_length(tc, centers):.4f}"
                    row[f"{label}_pd"] = f"{preferred_direction(tc, centers):.0f}"
                else:
                    row[f"{label}_mvl"] = "---"
                    row[f"{label}_pd"] = "---"

            pop_rows.append(row)

    df_pop = pd.DataFrame(pop_rows)
    st.dataframe(df_pop, use_container_width=True, height=400)

    # Download as CSV
    csv_data = df_pop.to_csv(index=False)
    st.download_button(
        "Download as CSV",
        csv_data,
        "population_summary_pooled.csv",
        "text/csv",
    )
