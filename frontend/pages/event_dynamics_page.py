"""Event Dynamics — compare calcium event properties across cell types and animals.

Loads ca.h5 data (Stage 4 output) for each session, runs Voigts & Harnett 2020
event detection, characterizes each event (amplitude, duration, rise/decay time,
AUC), and compares distributions between Penk+ vs non-Penk CamKII+ populations
and between individual animals.

Reference:
    Voigts & Harnett 2020. "Somatic and dendritic encoding of spatial
    variables in retrosplenial cortex differs during 2D navigation."
    Neuron 105(2):237-245. doi:10.1016/j.neuron.2019.10.016
    https://github.com/jvoigts/cell_labeling_bhv
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

from hm2p.calcium.events import (
    characterize_events,
    detect_events_single,
    summarize_cell_dynamics,
)

log = logging.getLogger("hm2p.frontend.event_dynamics")

st.title("Event Dynamics")
st.caption(
    "Compare calcium transient properties between Penk+ and non-Penk CamKII+ "
    "populations, and between animals within each cell type. "
    "Method: Voigts & Harnett 2020 (doi:10.1016/j.neuron.2019.10.016)."
)


# ── Data loading (from ca.h5, Stage 4) ─────────────────────────────────────

with st.spinner("Loading calcium data..."):
    from frontend.data import load_all_ca_data
    all_sessions = load_all_ca_data()

if not all_sessions:
    st.warning("No calcium data (ca.h5) available yet. This page will populate "
               "automatically when Stage 4 (calcium processing) completes.")
    st.stop()


# ── Sidebar filters ────────────────────────────────────────────────────────

celltypes = sorted(set(s["celltype"] for s in all_sessions))
animals = sorted(set(s["animal_id"] for s in all_sessions))

with st.sidebar:
    st.header("Filters")
    sel_celltypes = st.multiselect(
        "Cell type", celltypes, default=celltypes, key="ed_filter_ct",
    )
    sel_animals = st.multiselect(
        "Animal", animals, default=animals, key="ed_filter_animal",
    )
    roi_filter = st.radio(
        "ROI type",
        ["Soma only", "Dendrite only", "All ROIs"],
        index=0,
        key="ed_filter_roi",
    )

# Apply filters
sessions = [
    s for s in all_sessions
    if s["celltype"] in sel_celltypes and s["animal_id"] in sel_animals
]

# Apply ROI type filter
if roi_filter != "All ROIs":
    target_code = 0 if roi_filter == "Soma only" else 1
    filtered = []
    for s in sessions:
        mask = s["roi_types"] == target_code
        if mask.any():
            s_copy = dict(s)
            s_copy["dff"] = s["dff"][mask]
            s_copy["roi_types"] = s["roi_types"][mask]
            s_copy["n_rois"] = int(mask.sum())
            filtered.append(s_copy)
    sessions = filtered

if not sessions:
    st.warning("No sessions match the current filters.")
    st.stop()

st.success(f"Loaded {len(sessions)} sessions, "
           f"{sum(s['n_rois'] for s in sessions)} total cells")


# ── Run event detection + characterization ──────────────────────────────────

@st.cache_data(ttl=600)
def _compute_dynamics(_sessions_hash: str, sessions_data: list[dict]) -> pd.DataFrame:
    """Run event detection and dynamics characterization for all cells."""
    rows = []
    for ses in sessions_data:
        dff = ses["dff"]
        ses_fps = ses["fps"]

        for roi in range(ses["n_rois"]):
            trace = dff[roi].astype(np.float64)
            er = detect_events_single(trace)
            summary = summarize_cell_dynamics(trace, er, ses_fps)
            summary["celltype"] = ses["celltype"]
            summary["animal_id"] = ses["animal_id"]
            summary["exp_id"] = ses["exp_id"]
            summary["roi"] = roi
            rows.append(summary)
    return pd.DataFrame(rows)


# Hash for caching
sessions_hash = "|".join(f"{s['exp_id']}:{s['n_rois']}" for s in sessions)

with st.spinner("Running event detection and dynamics characterization..."):
    df = _compute_dynamics(sessions_hash, sessions)

if df.empty:
    st.warning("No data available after filtering.")
    st.stop()

# Filter cells with at least 1 event for meaningful comparisons
df_active = df[df["n_events"] > 0].copy()
n_total = len(df)
n_active = len(df_active)
n_silent = n_total - n_active


# ── Top metrics ─────────────────────────────────────────────────────────────

st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Cells", n_total)
col2.metric("Active Cells", n_active)
col3.metric("Silent Cells", n_silent)
col4.metric("Median Event Rate", f"{df_active['event_rate'].median():.1f}/min" if n_active else "N/A")
col5.metric("Median SNR", f"{df_active['snr'].median():.1f}" if n_active else "N/A")

ct_counts = df["celltype"].value_counts()
ct_str = " | ".join(f"{ct}: {n}" for ct, n in ct_counts.items())
st.caption(f"{len(sessions)} sessions | {ct_str}")

st.markdown("---")


# ── Tabs ────────────────────────────────────────────────────────────────────

tab_overview, tab_celltype, tab_animal, tab_table = st.tabs([
    "Overview", "Penk vs Non-Penk", "By Animal", "Data Table",
])


# Helper: box + strip comparison plot
def _comparison_plot(data: pd.DataFrame, metric: str, group_col: str,
                     title: str, xlab: str = "", ylab: str = "",
                     height: int = 350) -> go.Figure:
    """Create a box + strip plot comparing groups."""
    fig = px.box(
        data, x=group_col, y=metric, color=group_col,
        points="all", title=title,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(marker=dict(size=3, opacity=0.4))
    fig.update_layout(
        height=height,
        xaxis_title=xlab or group_col,
        yaxis_title=ylab or metric,
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=30),
    )
    return fig


# ── Tab 1: Overview ─────────────────────────────────────────────────────────

with tab_overview:
    st.subheader("Event Dynamics Overview")

    metrics = [
        ("event_rate", "Event Rate (events/min)"),
        ("mean_amplitude", "Mean Amplitude (dF/F)"),
        ("mean_duration_s", "Mean Duration (s)"),
        ("mean_rise_time_s", "Mean Rise Time (s)"),
        ("mean_decay_time_s", "Mean Decay Time (s)"),
        ("snr", "SNR"),
    ]

    for row_start in range(0, len(metrics), 3):
        cols = st.columns(3)
        for col_idx, (metric, label) in enumerate(metrics[row_start:row_start + 3]):
            with cols[col_idx]:
                vals = df_active[metric].dropna()
                if len(vals) > 0:
                    fig = px.histogram(
                        df_active, x=metric, nbins=30, title=label,
                        color_discrete_sequence=["steelblue"],
                    )
                    fig.add_vline(x=vals.median(), line_dash="dash", line_color="red",
                                  annotation_text=f"median={vals.median():.3f}")
                    fig.update_layout(height=250, margin=dict(l=40, r=20, t=50, b=30),
                                      xaxis_title=label, yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True, key=f"overview_{metric}")

    # Rise vs decay scatter
    st.subheader("Rise vs Decay Time")
    fig = px.scatter(
        df_active, x="mean_rise_time_s", y="mean_decay_time_s",
        color="celltype",
        opacity=0.5,
        title="Rise Time vs Decay Time (per cell)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.add_shape(type="line", x0=0, y0=0,
                  x1=df_active["mean_rise_time_s"].quantile(0.95),
                  y1=df_active["mean_rise_time_s"].quantile(0.95),
                  line=dict(dash="dash", color="gray"))
    fig.update_layout(
        height=400,
        xaxis_title="Mean Rise Time (s)",
        yaxis_title="Mean Decay Time (s)",
        margin=dict(l=50, r=20, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True, key="rise_decay_scatter")

    # Amplitude vs event rate
    st.subheader("Amplitude vs Event Rate")
    fig = px.scatter(
        df_active, x="event_rate", y="mean_amplitude",
        color="celltype",
        opacity=0.5,
        title="Event Rate vs Mean Amplitude",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        height=350,
        xaxis_title="Event Rate (events/min)",
        yaxis_title="Mean Amplitude (dF/F)",
        margin=dict(l=50, r=20, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True, key="amp_rate_scatter")


# ── Tab 2: Penk vs Non-Penk ────────────────────────────────────────────────

with tab_celltype:
    st.subheader("Penk+ vs Non-Penk CamKII+")

    celltypes_present = df_active["celltype"].unique()
    if len(celltypes_present) < 2:
        st.warning(f"Only one cell type present ({list(celltypes_present)}). "
                   "Need both penk and nonpenk for comparison.")
    else:
        comparison_metrics = [
            ("event_rate", "Event Rate (events/min)"),
            ("mean_amplitude", "Mean Amplitude (dF/F)"),
            ("median_amplitude", "Median Amplitude (dF/F)"),
            ("mean_duration_s", "Mean Duration (s)"),
            ("mean_rise_time_s", "Mean Rise Time (s)"),
            ("mean_decay_time_s", "Mean Decay Time (s)"),
            ("snr", "SNR"),
            ("fraction_active", "Fraction Active"),
            ("mean_auc", "Mean AUC"),
            ("mean_iei_s", "Mean Inter-Event Interval (s)"),
        ]

        for row_start in range(0, len(comparison_metrics), 3):
            cols = st.columns(3)
            for col_idx, (metric, label) in enumerate(comparison_metrics[row_start:row_start + 3]):
                with cols[col_idx]:
                    fig = _comparison_plot(
                        df_active, metric, "celltype", label,
                        xlab="Cell Type", ylab=label,
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                   key=f"ct_{metric}")

        # Summary statistics table
        st.subheader("Summary Statistics by Cell Type")
        summary_cols = ["event_rate", "mean_amplitude", "mean_duration_s",
                        "mean_rise_time_s", "mean_decay_time_s", "snr",
                        "fraction_active", "mean_auc"]
        summary = df_active.groupby("celltype")[summary_cols].agg(["median", "mean", "std", "count"])
        summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
        st.dataframe(summary.T, use_container_width=True)

        # Mann-Whitney U tests
        st.subheader("Statistical Comparisons (Mann-Whitney U)")
        from scipy.stats import mannwhitneyu

        ct_names = sorted(celltypes_present)
        if len(ct_names) >= 2:
            g1 = df_active[df_active["celltype"] == ct_names[0]]
            g2 = df_active[df_active["celltype"] == ct_names[1]]

            stat_rows = []
            for metric, label in comparison_metrics:
                v1 = g1[metric].dropna()
                v2 = g2[metric].dropna()
                if len(v1) >= 3 and len(v2) >= 3:
                    U, p = mannwhitneyu(v1, v2, alternative="two-sided")
                    stat_rows.append({
                        "Metric": label,
                        f"{ct_names[0]} (n={len(v1)})": f"{v1.median():.4f}",
                        f"{ct_names[1]} (n={len(v2)})": f"{v2.median():.4f}",
                        "U": f"{U:.0f}",
                        "p-value": f"{p:.4f}",
                        "Sig": "**" if p < 0.01 else ("*" if p < 0.05 else ""),
                    })
            if stat_rows:
                st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)


# ── Tab 3: By Animal ───────────────────────────────────────────────────────

with tab_animal:
    st.subheader("Event Dynamics by Animal")

    animal_ids = sorted(df_active["animal_id"].unique())
    if len(animal_ids) < 2:
        st.info("Only one animal in filtered data.")
    else:
        # Group by celltype then animal
        for ct in sorted(df_active["celltype"].unique()):
            ct_data = df_active[df_active["celltype"] == ct]
            ct_animals = sorted(ct_data["animal_id"].unique())

            if len(ct_animals) < 2:
                st.markdown(f"**{ct}** — only 1 animal, skipping within-type comparison.")
                continue

            st.markdown(f"### {ct} ({len(ct_animals)} animals)")

            animal_metrics = [
                ("event_rate", "Event Rate"),
                ("mean_amplitude", "Mean Amplitude"),
                ("mean_duration_s", "Mean Duration (s)"),
                ("snr", "SNR"),
            ]

            cols = st.columns(len(animal_metrics))
            for col_idx, (metric, label) in enumerate(animal_metrics):
                with cols[col_idx]:
                    fig = _comparison_plot(
                        ct_data, metric, "animal_id",
                        f"{label} — {ct}",
                        xlab="Animal", ylab=label, height=300,
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                   key=f"animal_{ct}_{metric}")

            # Per-animal summary
            animal_summary = ct_data.groupby("animal_id")[
                ["n_events", "event_rate", "mean_amplitude", "mean_duration_s", "snr"]
            ].agg(["count", "median", "mean"])
            animal_summary.columns = [f"{c}_{s}" for c, s in animal_summary.columns]
            st.dataframe(animal_summary, use_container_width=True)
            st.markdown("---")


# ── Tab 4: Data Table ──────────────────────────────────────────────────────

with tab_table:
    st.subheader("Per-Cell Dynamics Data")

    display_cols = [
        "celltype", "animal_id", "exp_id", "roi", "n_events", "event_rate",
        "snr", "mean_amplitude", "median_amplitude", "mean_duration_s",
        "mean_rise_time_s", "mean_decay_time_s", "mean_auc",
        "fraction_active", "mean_iei_s",
    ]
    available_cols = [c for c in display_cols if c in df.columns]
    df_display = df[available_cols].copy()

    # Format numeric columns
    for col in df_display.select_dtypes(include=[np.floating]).columns:
        df_display[col] = df_display[col].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        )

    st.dataframe(df_display, use_container_width=True, hide_index=True, height=500)

    st.download_button(
        "Download CSV",
        df[available_cols].to_csv(index=False),
        "event_dynamics.csv",
        "text/csv",
    )


# ── Footer ──────────────────────────────────────────────────────────────────

st.markdown("---")
with st.expander("Methods"):
    st.markdown("""
**Event detection:** Percentile-based noise model with CDF thresholding
(Voigts & Harnett 2020, doi:10.1016/j.neuron.2019.10.016).
[GitHub](https://github.com/jvoigts/cell_labeling_bhv)

**Per-event metrics:** amplitude (peak dF/F), duration (onset to offset),
rise time (onset to peak), decay time (peak to offset), AUC (integral of dF/F).

**Per-cell summaries:** event rate, SNR (mean amplitude / baseline std),
fraction active, mean inter-event interval, and aggregated event metrics.

**Statistical tests:** Mann-Whitney U (non-parametric, two-sided) for
cell-type comparisons. No correction for multiple comparisons shown
— interpret exploratorily.
""")

st.caption(
    "Event dynamics characterization compares calcium transient properties "
    "between Penk+ and non-Penk CamKII+ RSP populations. "
    "Voigts & Harnett 2020, doi:10.1016/j.neuron.2019.10.016."
)
