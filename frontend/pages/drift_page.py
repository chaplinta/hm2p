"""Drift Analysis — PD drift during darkness and visual cue removal.

Core science question: do HD cells maintain their preferred direction
when visual landmarks are removed (lights off), or does the internal
compass drift? Compare Penk vs non-Penk drift rates.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.data import load_all_sync_data, session_filter_sidebar
from hm2p.analysis.stability import (
    dark_drift_rate,
    drift_per_epoch,
    light_dark_stability,
    sliding_window_stability,
)
from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length

st.title("Drift Analysis")
st.caption(
    "Track preferred direction drift during light/dark cycles. "
    "Key question: does HD tuning drift when visual cues are removed?"
)

# Load real data
all_data = load_all_sync_data()
if all_data["n_sessions"] == 0:
    st.warning(
        "No data available yet. This page will populate when the relevant "
        "pipeline stage completes."
    )
    st.stop()

sessions = session_filter_sidebar(all_data["sessions"], key_prefix="drift")
if not sessions:
    st.warning("No sessions match the current filters.")
    st.stop()


# ── Helper: estimate fps from frame_times ─────────────────────────────────
def _estimate_fps(frame_times: np.ndarray, n_frames: int) -> float:
    """Estimate sampling rate from frame_times array."""
    if frame_times is not None and len(frame_times) >= 2:
        duration = frame_times[-1] - frame_times[0]
        if duration > 0:
            return (len(frame_times) - 1) / duration
    # Fallback: sync.h5 is at imaging rate (~9.6 Hz)
    return 9.6


# ── Tabs ───────────────────────────────────────────────────────────────────
tab_overview, tab_epoch, tab_drift_rate, tab_compare, tab_ld = st.tabs([
    "Population Overview",
    "Epoch PD Tracking",
    "Drift Rate",
    "Light vs Dark Tuning",
    "Sliding Window",
])

# ═══════════════════════════════════════════════════════════════════════════
# Tab 1: Population Overview (all cells, all sessions)
# ═══════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Population Drift Overview")
    st.markdown(
        "Light/dark stability and drift rate for **every cell** across all "
        "sessions (after sidebar filters). Based on dF/F0 tuning curves."
    )

    # Compute per-cell metrics across all sessions
    rows = []
    progress = st.progress(0, text="Computing drift metrics...")
    total_cells = sum(s["n_rois"] for s in sessions)
    cell_count = 0

    for sess in sessions:
        signals = sess["dff"]
        hd = sess["hd_deg"]
        mask = sess["active"] & ~sess["bad_behav"]
        light_on = sess["light_on"]
        fps = _estimate_fps(sess["frame_times"], sess["n_frames"])
        n_frames = sess["n_frames"]

        # Estimate cycle length from light_on transitions
        diffs_lo = np.diff(light_on.astype(int))
        transition_frames = np.where(np.abs(diffs_lo) > 0)[0]
        if len(transition_frames) >= 2:
            cycle_frames = int(np.median(np.diff(transition_frames)))
        else:
            cycle_frames = int(60 * fps)

        for ci in range(sess["n_rois"]):
            cell_count += 1
            if cell_count % 10 == 0 or cell_count == total_cells:
                progress.progress(
                    min(cell_count / max(total_cells, 1), 1.0),
                    text=f"Computing drift metrics... ({cell_count}/{total_cells})",
                )

            sig = signals[ci]

            # Light/dark stability
            try:
                ld = light_dark_stability(sig, hd, mask, light_on)
            except Exception:
                continue

            # Drift rate
            try:
                dr = dark_drift_rate(
                    sig, hd, mask, light_on, fps=fps,
                    window_frames=max(int(cycle_frames * 0.6), 10),
                    step_frames=max(int(cycle_frames * 0.2), 5),
                )
            except Exception:
                dr = {"dark_drift_deg_per_s": np.nan, "light_drift_deg_per_s": np.nan}

            mvl_ratio = ld["mvl_dark"] / max(ld["mvl_light"], 1e-10)

            rows.append({
                "session": sess["exp_id"],
                "animal": sess["animal_id"],
                "celltype": sess["celltype"],
                "cell": ci,
                "correlation": round(ld["correlation"], 3),
                "pd_shift_deg": round(ld["pd_shift_deg"], 1),
                "mvl_light": round(ld["mvl_light"], 3),
                "mvl_dark": round(ld["mvl_dark"], 3),
                "mvl_ratio": round(mvl_ratio, 3),
                "light_drift_deg_s": round(dr["light_drift_deg_per_s"], 2),
                "dark_drift_deg_s": round(dr["dark_drift_deg_per_s"], 2),
            })

    progress.empty()

    if not rows:
        st.warning("No cells could be analysed with current filters.")
        st.stop()

    import pandas as pd
    df = pd.DataFrame(rows)

    # Summary metrics
    n_cells = len(df)
    n_sess = df["session"].nunique()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cells", n_cells)
    col2.metric("Sessions", n_sess)
    col3.metric("Median PD shift", f"{df['pd_shift_deg'].median():.1f} deg")
    col4.metric("Median dark drift", f"{df['dark_drift_deg_s'].median():.1f} deg/s")

    # ── Histogram: PD shift by celltype ────────────────────────────────────
    st.markdown("#### PD Shift (Light vs Dark) by Cell Type")
    fig_pd = px.histogram(
        df, x="pd_shift_deg", color="celltype",
        nbins=40, barmode="overlay", opacity=0.7,
        labels={"pd_shift_deg": "PD shift (deg)", "celltype": "Cell type"},
        color_discrete_map={"penk": "#e41a1c", "nonpenk": "#377eb8", "unknown": "#999999"},
    )
    fig_pd.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_pd.update_layout(height=350)
    st.plotly_chart(fig_pd, use_container_width=True)

    # ── Histogram: dark drift rate by celltype ─────────────────────────────
    st.markdown("#### Dark Drift Rate by Cell Type")
    df_valid_drift = df.dropna(subset=["dark_drift_deg_s"])
    fig_dr = px.histogram(
        df_valid_drift, x="dark_drift_deg_s", color="celltype",
        nbins=40, barmode="overlay", opacity=0.7,
        labels={"dark_drift_deg_s": "Dark drift rate (deg/s)", "celltype": "Cell type"},
        color_discrete_map={"penk": "#e41a1c", "nonpenk": "#377eb8", "unknown": "#999999"},
    )
    fig_dr.update_layout(height=350)
    st.plotly_chart(fig_dr, use_container_width=True)

    # ── Scatter: light vs dark drift rate ──────────────────────────────────
    st.markdown("#### Light vs Dark Drift Rate")
    df_scatter = df.dropna(subset=["light_drift_deg_s", "dark_drift_deg_s"])
    fig_sc = px.scatter(
        df_scatter, x="light_drift_deg_s", y="dark_drift_deg_s",
        color="celltype", opacity=0.7,
        labels={
            "light_drift_deg_s": "Light drift rate (deg/s)",
            "dark_drift_deg_s": "Dark drift rate (deg/s)",
            "celltype": "Cell type",
        },
        hover_data=["session", "cell"],
        color_discrete_map={"penk": "#e41a1c", "nonpenk": "#377eb8", "unknown": "#999999"},
    )
    # Unity line
    max_val = max(
        df_scatter["light_drift_deg_s"].max() if len(df_scatter) else 1,
        df_scatter["dark_drift_deg_s"].max() if len(df_scatter) else 1,
        1,
    )
    fig_sc.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", line=dict(dash="dash", color="gray", width=1),
        showlegend=False,
    ))
    fig_sc.update_layout(height=400)
    st.plotly_chart(fig_sc, use_container_width=True)
    st.caption(
        "Dashed line = unity. Points above the line drift faster in darkness "
        "than in light, consistent with visual cue dependence."
    )

    # ── Summary table ──────────────────────────────────────────────────────
    st.markdown("#### Per-Cell Summary")
    st.dataframe(
        df.rename(columns={
            "correlation": "L/D corr",
            "pd_shift_deg": "PD shift (deg)",
            "mvl_light": "MVL light",
            "mvl_dark": "MVL dark",
            "mvl_ratio": "MVL ratio",
            "light_drift_deg_s": "Light drift (deg/s)",
            "dark_drift_deg_s": "Dark drift (deg/s)",
        }),
        use_container_width=True,
        height=400,
    )

# ═══════════════════════════════════════════════════════════════════════════
# Single-cell tabs: shared session + cell selector
# ═══════════════════════════════════════════════════════════════════════════

# Session selector (shared across single-cell tabs)
session_labels = [
    f"{s['exp_id']} ({s['celltype']}, {s['n_rois']} ROIs)" for s in sessions
]
sel_idx = st.sidebar.selectbox(
    "Session", range(len(sessions)),
    format_func=lambda i: session_labels[i], key="drift_ses",
)
sess = sessions[sel_idx]

signals = sess["dff"]  # (n_rois, n_frames)
hd = sess["hd_deg"]
mask = sess["active"] & ~sess["bad_behav"]
light_on = sess["light_on"]
n_cells = signals.shape[0]
n_frames = signals.shape[1]
fps = _estimate_fps(sess["frame_times"], n_frames)

if n_cells == 0:
    st.warning("No ROIs in this session after filtering.")
    st.stop()

# Cell selector in sidebar
cell_main = st.sidebar.selectbox(
    "Primary cell", range(n_cells),
    format_func=lambda x: f"Cell {x+1}", key="drift_main_cell",
)
signal = signals[cell_main]

# Estimate cycle length from light_on transitions
diffs = np.diff(light_on.astype(int))
transition_frames = np.where(np.abs(diffs) > 0)[0]
if len(transition_frames) >= 2:
    cycle_frames = int(np.median(np.diff(transition_frames)))
else:
    cycle_frames = int(60 * fps)  # default 60s
cycle_s = cycle_frames / fps


def _add_light_shading(fig, n_frames, cycle_frames, fps):
    """Add yellow background shading for light-on epochs."""
    for start in range(0, n_frames, 2 * cycle_frames):
        fig.add_vrect(
            x0=start / fps, x1=min(start + cycle_frames, n_frames) / fps,
            fillcolor="yellow", opacity=0.1, line_width=0,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Tab 2: Epoch PD Tracking
# ═══════════════════════════════════════════════════════════════════════════
with tab_epoch:
    st.subheader("PD Drift Across Light/Dark Epochs")
    result = drift_per_epoch(signal, hd, mask, light_on)

    if result["n_epochs"] > 0:
        col1, col2 = st.columns(2)
        with col1:
            colors = ["gold" if il else "midnightblue" for il in result["epoch_is_light"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=result["epoch_centers"] / fps,
                y=result["epoch_pds"],
                mode="markers+lines",
                marker=dict(color=colors, size=10),
                line=dict(color="gray", width=1),
                name="PD",
            ))
            fig.update_layout(
                height=300, title="Preferred Direction per Epoch",
                xaxis_title="Time (s)", yaxis_title="PD (deg)",
                yaxis=dict(range=[0, 360]),
            )
            _add_light_shading(fig, n_frames, cycle_frames, fps)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(data=[go.Scatter(
                x=result["epoch_centers"] / fps,
                y=result["cumulative_drift"],
                mode="markers+lines",
                marker=dict(color=colors, size=10),
                line=dict(color="gray", width=1),
            )])
            fig.update_layout(
                height=300, title="Cumulative PD Drift",
                xaxis_title="Time (s)", yaxis_title="Cumulative drift (deg)",
            )
            _add_light_shading(fig, n_frames, cycle_frames, fps)
            st.plotly_chart(fig, use_container_width=True)

        # MVL per epoch
        fig = go.Figure(data=[go.Bar(
            x=[f"Epoch {i+1}" for i in range(result["n_epochs"])],
            y=result["epoch_mvls"],
            marker_color=colors,
        )])
        fig.update_layout(
            height=250, title="MVL per Epoch",
            xaxis_title="Epoch", yaxis_title="MVL",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data for epoch analysis.")

# ═══════════════════════════════════════════════════════════════════════════
# Tab 3: Drift Rate
# ═══════════════════════════════════════════════════════════════════════════
with tab_drift_rate:
    st.subheader("Drift Rate: Light vs Dark")

    dr = dark_drift_rate(
        signal, hd, mask, light_on, fps=fps,
        window_frames=max(int(cycle_frames * 0.6), 10),
        step_frames=max(int(cycle_frames * 0.2), 5),
    )

    col1, col2 = st.columns(2)
    col1.metric("Light drift rate", f"{dr['light_drift_deg_per_s']:.2f} deg/s")
    col2.metric("Dark drift rate", f"{dr['dark_drift_deg_per_s']:.2f} deg/s")

    if dr["dark_drift_deg_per_s"] > 0:
        ratio = dr["dark_drift_deg_per_s"] / max(dr["light_drift_deg_per_s"], 1e-10)
        st.markdown(f"**Dark/Light ratio:** {ratio:.1f}x")

    # PD trajectories
    fig = go.Figure()
    if len(dr["light_times_s"]) > 0:
        fig.add_trace(go.Scatter(
            x=dr["light_times_s"], y=dr["light_pds"],
            mode="markers+lines", name="Light",
            marker=dict(color="gold", size=6),
            line=dict(color="gold", width=1),
        ))
    if len(dr["dark_times_s"]) > 0:
        fig.add_trace(go.Scatter(
            x=dr["dark_times_s"], y=dr["dark_pds"],
            mode="markers+lines", name="Dark",
            marker=dict(color="midnightblue", size=6),
            line=dict(color="midnightblue", width=1),
        ))
    fig.update_layout(
        height=300, title="PD Trajectory (Sliding Windows)",
        xaxis_title="Time (s)", yaxis_title="PD (deg)",
        yaxis=dict(range=[0, 360]),
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Tab 4: Light vs Dark Tuning
# ═══════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Light vs Dark Tuning Curves")

    ld = light_dark_stability(signal, hd, mask, light_on)

    col1, col2, col3 = st.columns(3)
    col1.metric("Correlation", f"{ld['correlation']:.3f}")
    col2.metric("PD shift", f"{ld['pd_shift_deg']:.1f}")
    col3.metric("MVL ratio", f"{ld['mvl_dark'] / max(ld['mvl_light'], 1e-10):.2f}")

    # Polar overlay
    bc = ld["bin_centers"]
    theta_plot = np.concatenate([np.deg2rad(bc), [np.deg2rad(bc[0])]])

    tc_light = ld["tuning_curve_light"]
    tc_dark = ld["tuning_curve_dark"]

    fig = go.Figure()
    if not np.all(np.isnan(tc_light)):
        fig.add_trace(go.Scatterpolar(
            r=np.concatenate([tc_light, [tc_light[0]]]),
            theta=np.rad2deg(theta_plot),
            mode="lines", line=dict(color="gold", width=3),
            name="Light",
        ))
    if not np.all(np.isnan(tc_dark)):
        fig.add_trace(go.Scatterpolar(
            r=np.concatenate([tc_dark, [tc_dark[0]]]),
            theta=np.rad2deg(theta_plot),
            mode="lines", line=dict(color="midnightblue", width=3),
            name="Dark",
        ))
    fig.update_layout(
        height=350,
        title="Tuning Curve Overlay (dF/F0)",
        polar=dict(radialaxis=dict(visible=False), angularaxis=dict(showticklabels=False)),
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Tab 5: Sliding Window
# ═══════════════════════════════════════════════════════════════════════════
with tab_ld:
    st.subheader("Sliding Window MVL & PD")

    sw = sliding_window_stability(
        signal, hd, mask,
        window_frames=max(int(cycle_frames * 0.8), 10),
        step_frames=max(int(cycle_frames * 0.2), 5),
    )

    if sw["n_windows"] > 0:
        times = sw["window_centers"] / fps

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=sw["mvls"],
            mode="lines+markers", name="MVL",
            marker=dict(size=5),
        ))
        fig.update_layout(
            height=250, title="MVL Over Time",
            xaxis_title="Time (s)", yaxis_title="MVL",
        )
        _add_light_shading(fig, n_frames, cycle_frames, fps)
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=sw["preferred_dirs"],
            mode="lines+markers", name="PD",
            marker=dict(size=5),
        ))
        fig.update_layout(
            height=250, title="Preferred Direction Over Time",
            xaxis_title="Time (s)", yaxis_title="PD (deg)",
            yaxis=dict(range=[0, 360]),
        )
        _add_light_shading(fig, n_frames, cycle_frames, fps)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption(
    "Yellow shading = light on. Dark blue = lights off (darkness). "
    "If HD cells rely on visual landmarks, PD should drift during darkness. "
    "Cells anchored by path integration should maintain stable PD. "
    "All tuning computed from dF/F0."
)
