"""Light Analysis — compare neural activity between light-on and light-off epochs.

Uses the standard pooled-data pattern: loads ALL sessions by default via
load_all_sync_data(), with optional sidebar filtering by celltype/animal/ROI.
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

from frontend.data import load_all_sync_data, session_filter_sidebar

log = logging.getLogger("hm2p.frontend.light")

st.title("Light/Dark Analysis")
st.caption(
    "Compare neural activity between light-on and light-off epochs across all sessions."
)

# ── Load pooled data ────────────────────────────────────────────────────
with st.spinner("Loading sync data for all sessions..."):
    all_data = load_all_sync_data()

if all_data["n_sessions"] == 0:
    st.warning("No sync data available yet.")
    st.stop()

# ── Sidebar filters ─────────────────────────────────────────────────────
sessions = session_filter_sidebar(
    all_data["sessions"], show_roi_filter=True, key_prefix="light"
)

if not sessions:
    st.warning("No sessions match the current filters.")
    st.stop()

# ── Optional session selector ───────────────────────────────────────────
exp_ids = ["All sessions (pooled)"] + [s["exp_id"] for s in sessions]
selected_session = st.selectbox(
    "Session", exp_ids, index=0, key="light_session_select"
)

if selected_session != "All sessions (pooled)":
    sessions = [s for s in sessions if s["exp_id"] == selected_session]

# ── Summary metrics ─────────────────────────────────────────────────────
n_sessions = len(sessions)
n_total_rois = sum(s["n_rois"] for s in sessions)
st.markdown(f"**{n_sessions} session(s)** — **{n_total_rois} ROIs**")

# ── Compute per-ROI light/dark metrics across all filtered sessions ─────


def _detect_dark_epochs(light_on: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end) index pairs for dark epochs."""
    epochs = []
    in_dark = False
    start = 0
    for i in range(len(light_on)):
        if not light_on[i] and not in_dark:
            start = i
            in_dark = True
        elif light_on[i] and in_dark:
            epochs.append((start, i))
            in_dark = False
    if in_dark:
        epochs.append((start, len(light_on)))
    return epochs


def _count_transitions(light_on: np.ndarray) -> int:
    """Count light-on/off transitions."""
    if len(light_on) < 2:
        return 0
    return int(np.sum(light_on[1:] != light_on[:-1]))


# Aggregate metrics
all_light_means = []
all_dark_means = []
all_snrs = []
all_celltypes = []
all_session_labels = []

total_light_frames = 0
total_dark_frames = 0
total_transitions = 0

for sess in sessions:
    dff = sess["dff"]
    light_on = sess["light_on"]
    n_rois, n_frames = dff.shape

    # Apply valid mask (active & not bad_behav)
    valid = sess["active"] & ~sess["bad_behav"]
    light_valid = light_on & valid
    dark_valid = ~light_on & valid

    total_light_frames += int(light_valid.sum())
    total_dark_frames += int(dark_valid.sum())
    total_transitions += _count_transitions(light_on)

    for i in range(n_rois):
        trace = dff[i]
        lm = float(np.nanmean(trace[light_valid])) if light_valid.any() else 0.0
        dm = float(np.nanmean(trace[dark_valid])) if dark_valid.any() else 0.0
        all_light_means.append(lm)
        all_dark_means.append(dm)

        baseline_std = np.std(trace[trace < np.percentile(trace, 50)])
        peak = np.percentile(trace, 95)
        all_snrs.append(peak / baseline_std if baseline_std > 0 else 0.0)

        all_celltypes.append(sess["celltype"])
        all_session_labels.append(sess["exp_id"])

light_means = np.array(all_light_means)
dark_means = np.array(all_dark_means)
snrs = np.array(all_snrs)

# ── Top-level metrics ───────────────────────────────────────────────────
total_frames = total_light_frames + total_dark_frames
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Light ON frames",
    f"{total_light_frames:,} ({total_light_frames / max(1, total_frames) * 100:.1f}%)",
)
col2.metric(
    "Light OFF frames",
    f"{total_dark_frames:,} ({total_dark_frames / max(1, total_frames) * 100:.1f}%)",
)
col3.metric("Total transitions", f"{total_transitions:,}")
col4.metric("Sessions / ROIs", f"{n_sessions} / {n_total_rois}")

# ── Tabs ────────────────────────────────────────────────────────────────
tab_overview, tab_per_roi, tab_modulation, tab_population = st.tabs(
    ["Overview", "Per-ROI", "Modulation Index", "Population"]
)


# --- Tab 1: Overview ---
with tab_overview:
    st.subheader("Light vs Dark Activity")

    from hm2p.plotting import format_pvalue, paired_condition_scatter

    col1, col2 = st.columns(2)
    with col1:
        fig_dff, stat_dff = paired_condition_scatter(
            light_means,
            dark_means,
            "Light",
            "Dark",
            "Mean dF/F0",
            height=400,
            width=400,
        )
        st.plotly_chart(fig_dff, use_container_width=True)
        st.markdown(
            f"Wilcoxon signed-rank: {format_pvalue(stat_dff['p'])}, n={stat_dff['n']}"
        )

    # Light cycle timeline (show per-session or pooled mean)
    st.subheader("Light Cycle Timeline")

    if len(sessions) == 1:
        sess = sessions[0]
        dff_plot = sess["dff"]
        light_on_plot = sess["light_on"]
        n_frames_plot = sess["n_frames"]
        frame_times_plot = sess["frame_times"]
        fps_est = (
            1.0 / np.median(np.diff(frame_times_plot))
            if len(frame_times_plot) > 1
            else 9.8
        )
        time_s = frame_times_plot - frame_times_plot[0]

        ds = max(1, n_frames_plot // 2000)
        mean_trace = dff_plot.mean(axis=0)
        kernel = np.ones(max(1, int(fps_est))) / max(1, int(fps_est))
        mean_smooth = np.convolve(mean_trace, kernel, mode="same")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time_s[::ds],
                y=mean_smooth[::ds],
                mode="lines",
                name="Mean dF/F0",
                line=dict(color="black", width=1),
            )
        )

        # Shade dark periods
        dark_epochs = _detect_dark_epochs(light_on_plot)
        for ds_idx, de_idx in dark_epochs:
            fig.add_vrect(
                x0=time_s[ds_idx],
                x1=time_s[min(de_idx, n_frames_plot) - 1],
                fillcolor="rgba(50,50,50,0.15)",
                line_width=0,
                layer="below",
            )

        fig.update_layout(
            height=300,
            title="Population Mean dF/F0 with Light Cycles",
            xaxis_title="Time (s)",
            yaxis_title="Mean dF/F0",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Select a single session to see the light cycle timeline trace. "
            "Pooled view shows aggregate statistics in other tabs."
        )


# --- Tab 2: Per-ROI ---
with tab_per_roi:
    st.subheader("Per-ROI Light vs Dark")

    # Build DataFrame for coloring by celltype
    roi_df = pd.DataFrame(
        {
            "light_mean": light_means,
            "dark_mean": dark_means,
            "celltype": all_celltypes,
            "session": all_session_labels,
        }
    )

    fig = px.scatter(
        roi_df,
        x="light_mean",
        y="dark_mean",
        color="celltype",
        hover_data=["session"],
        labels={
            "light_mean": "Mean dF/F0 (Light ON)",
            "dark_mean": "Mean dF/F0 (Light OFF)",
        },
        title=f"Per-ROI: Light vs Dark Mean dF/F0 (n={len(roi_df)})",
        opacity=0.6,
    )
    # Unity line
    max_val = max(
        np.max(light_means) if len(light_means) else 1,
        np.max(dark_means) if len(dark_means) else 1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            showlegend=False,
        )
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


# --- Tab 3: Modulation Index ---
with tab_modulation:
    st.subheader("Light Modulation Index")
    st.markdown(
        "**LMI = (light - dark) / (light + dark)**. "
        "Positive = more active in light, Negative = more active in dark."
    )

    lmi_dff = (light_means - dark_means) / (light_means + dark_means + 1e-10)

    # Histogram colored by celltype
    lmi_df = pd.DataFrame(
        {"LMI": lmi_dff, "celltype": all_celltypes, "session": all_session_labels}
    )

    fig = px.histogram(
        lmi_df,
        x="LMI",
        color="celltype",
        nbins=30,
        title="Light Modulation Index (dF/F0)",
        barmode="overlay",
        opacity=0.7,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.add_vline(
        x=float(np.median(lmi_dff)),
        line_dash="dash",
        line_color="red",
        annotation_text=f"median={np.median(lmi_dff):.3f}",
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean LMI", f"{np.mean(lmi_dff):.3f}")
    col2.metric("Median LMI", f"{np.median(lmi_dff):.3f}")
    col3.metric("% prefer dark", f"{(lmi_dff < 0).mean() * 100:.0f}%")

    # Per-celltype breakdown
    if len(set(all_celltypes)) > 1:
        st.subheader("LMI by Cell Type")
        for ct in sorted(set(all_celltypes)):
            ct_mask = np.array(all_celltypes) == ct
            ct_lmi = lmi_dff[ct_mask]
            st.markdown(
                f"**{ct}** (n={ct_mask.sum()}): "
                f"mean={np.mean(ct_lmi):.3f}, "
                f"median={np.median(ct_lmi):.3f}, "
                f"% prefer dark={((ct_lmi < 0).mean() * 100):.0f}%"
            )

    # LMI vs SNR
    fig = px.scatter(
        x=snrs,
        y=lmi_dff,
        color=all_celltypes,
        labels={"x": "SNR", "y": "Light Modulation Index", "color": "celltype"},
        title="LMI vs SNR (high-SNR cells more reliable)",
        opacity=0.6,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


# --- Tab 4: Population ---
with tab_population:
    st.subheader("Population Light/Dark Response")

    # Aggregate population rate across sessions
    pop_light_rates = []
    pop_dark_rates = []
    pop_session_labels = []

    for sess in sessions:
        dff = sess["dff"]
        light_on = sess["light_on"]
        valid = sess["active"] & ~sess["bad_behav"]
        light_valid = light_on & valid
        dark_valid = ~light_on & valid

        if light_valid.any():
            pop_light_rates.append(float(np.nanmean(dff[:, light_valid])))
        else:
            pop_light_rates.append(0.0)
        if dark_valid.any():
            pop_dark_rates.append(float(np.nanmean(dff[:, dark_valid])))
        else:
            pop_dark_rates.append(0.0)
        pop_session_labels.append(sess["exp_id"])

    if pop_light_rates:
        col1, col2, col3 = st.columns(3)
        mean_light = np.mean(pop_light_rates)
        mean_dark = np.mean(pop_dark_rates)
        col1.metric("Mean pop. dF/F0 (Light)", f"{mean_light:.4f}")
        col2.metric("Mean pop. dF/F0 (Dark)", f"{mean_dark:.4f}")
        col3.metric(
            "Dark/Light ratio",
            f"{mean_dark / mean_light:.2f}" if mean_light > 0 else "N/A",
        )

    # Per-cycle analysis
    st.subheader("Per-Cycle Analysis")

    cycle_rows = []
    for sess in sessions:
        light_on = sess["light_on"]
        dff = sess["dff"]
        valid = sess["active"] & ~sess["bad_behav"]
        frame_times = sess["frame_times"]

        # Detect transitions: find on->off and off->on edges
        if len(light_on) < 2:
            continue

        changes = np.diff(light_on.astype(np.int8))
        # on_starts: where light turns on (0->1)
        on_starts = np.flatnonzero(changes == 1) + 1
        # off_starts: where light turns off (1->0)
        off_starts = np.flatnonzero(changes == -1) + 1

        # If first frame is light-on, add index 0 as an on_start
        if light_on[0]:
            on_starts = np.concatenate([[0], on_starts])
        # If first frame is dark, add index 0 as off_start
        else:
            off_starts = np.concatenate([[0], off_starts])

        # Pair each light-on epoch with its following dark epoch
        cycle_num = 0
        for on_idx in on_starts:
            # Find next off_start after on_idx
            later_offs = off_starts[off_starts > on_idx]
            if len(later_offs) == 0:
                continue
            off_idx = later_offs[0]

            # Find next on_start after off_idx (end of dark epoch)
            later_ons = on_starts[on_starts > off_idx]
            dark_end = later_ons[0] if len(later_ons) > 0 else len(light_on)

            light_mask = np.zeros(len(light_on), dtype=bool)
            light_mask[on_idx:off_idx] = True
            light_mask &= valid

            dark_mask = np.zeros(len(light_on), dtype=bool)
            dark_mask[off_idx:dark_end] = True
            dark_mask &= valid

            if not light_mask.any() or not dark_mask.any():
                continue

            cycle_num += 1
            light_dur = (
                frame_times[min(off_idx, len(frame_times) - 1)] - frame_times[on_idx]
            )
            dark_dur = (
                frame_times[min(dark_end, len(frame_times)) - 1]
                - frame_times[off_idx]
            )

            cycle_rows.append(
                {
                    "session": sess["exp_id"],
                    "celltype": sess["celltype"],
                    "cycle": cycle_num,
                    "light_dur_s": float(light_dur),
                    "dark_dur_s": float(dark_dur),
                    "light_mean_dff": float(np.nanmean(dff[:, light_mask])),
                    "dark_mean_dff": float(np.nanmean(dff[:, dark_mask])),
                }
            )

    if cycle_rows:
        cycle_df = pd.DataFrame(cycle_rows)

        # Aggregate: mean across sessions per cycle number
        agg_df = (
            cycle_df.groupby("cycle")
            .agg(
                light_mean=("light_mean_dff", "mean"),
                dark_mean=("dark_mean_dff", "mean"),
                light_sem=("light_mean_dff", "sem"),
                dark_sem=("dark_mean_dff", "sem"),
                n_sessions=("session", "nunique"),
            )
            .reset_index()
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=agg_df["cycle"],
                y=agg_df["light_mean"],
                error_y=dict(type="data", array=agg_df["light_sem"].tolist()),
                mode="lines+markers",
                name="Light ON",
                marker=dict(color="gold"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=agg_df["cycle"],
                y=agg_df["dark_mean"],
                error_y=dict(type="data", array=agg_df["dark_sem"].tolist()),
                mode="lines+markers",
                name="Light OFF",
                marker=dict(color="gray"),
            )
        )
        fig.update_layout(
            height=350,
            title=f"Mean Population dF/F0 per Light Cycle (n={len(sessions)} sessions)",
            xaxis_title="Cycle",
            yaxis_title="Mean dF/F0",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show aggregate table
        with st.expander("Cycle details"):
            st.dataframe(
                agg_df.rename(
                    columns={
                        "light_mean": "Light dF/F0",
                        "dark_mean": "Dark dF/F0",
                        "light_sem": "Light SEM",
                        "dark_sem": "Dark SEM",
                        "n_sessions": "# sessions",
                    }
                ).round(4),
                use_container_width=True,
            )
    else:
        st.info("No light/dark cycles detected in the selected sessions.")
