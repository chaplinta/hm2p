"""Cross-Session Light Analysis — compare light modulation across sessions and cell types."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import load_all_sync_data, session_filter_sidebar
from hm2p.constants import CELLTYPE_HEX

log = logging.getLogger("hm2p.frontend.light_compare")

st.title("Cross-Session Light Analysis")
st.caption("Compare light modulation index across all sessions — Penk vs non-Penk.")

# --- Load pooled sync data ---
with st.spinner("Loading sync data for all sessions..."):
    sync_data = load_all_sync_data()

sessions = session_filter_sidebar(
    sync_data["sessions"], show_roi_filter=True, key_prefix="lc"
)

if not sessions:
    st.warning("No sync data with light cycle information found.")
    st.stop()

n_sessions = len(sessions)
n_total_rois = sum(s["n_rois"] for s in sessions)
col1, col2 = st.columns(2)
col1.metric("Sessions", n_sessions)
col2.metric("Total ROIs", n_total_rois)

# --- Compute light modulation from cached sync data ---
rows = []
for s in sessions:
    dff = s["dff"]
    light_on = s["light_on"].astype(bool)
    n_rois = s["n_rois"]
    n_frames = s["n_frames"]
    fps = n_frames / (s["frame_times"][-1] - s["frame_times"][0]) if n_frames > 1 else 9.8
    event_masks = s.get("event_masks")

    # Skip sessions with no light variation
    if light_on.all() or (~light_on).all():
        continue

    for roi in range(n_rois):
        trace = dff[roi]
        light_mean = float(np.nanmean(trace[light_on]))
        dark_mean = float(np.nanmean(trace[~light_on]))
        lmi = (light_mean - dark_mean) / (light_mean + dark_mean + 1e-10)

        # SNR
        baseline_std = np.std(trace[trace < np.percentile(trace, 50)])
        peak = np.percentile(trace, 95)
        snr = peak / baseline_std if baseline_std > 0 else 0

        row = {
            "exp_id": s["exp_id"],
            "animal_id": s["animal_id"],
            "celltype": s["celltype"],
            "roi": roi,
            "snr": snr,
            "light_mean": light_mean,
            "dark_mean": dark_mean,
            "lmi_dff": lmi,
        }

        # Event-based modulation
        if event_masks is not None:
            em = event_masks[roi].astype(bool)
            light_events = em & light_on
            dark_events = em & ~light_on

            light_onsets = np.flatnonzero(light_events[1:] & ~light_events[:-1])
            dark_onsets = np.flatnonzero(dark_events[1:] & ~dark_events[:-1])

            light_dur = light_on.sum() / fps / 60
            dark_dur = (~light_on).sum() / fps / 60

            lr = len(light_onsets) / light_dur if light_dur > 0 else 0
            dr = len(dark_onsets) / dark_dur if dark_dur > 0 else 0

            row["light_event_rate"] = lr
            row["dark_event_rate"] = dr
            row["lmi_events"] = (lr - dr) / (lr + dr + 1e-10)

        rows.append(row)

df = pd.DataFrame(rows)

if df.empty:
    st.warning("No data with light cycle information found.")
    st.stop()

# --- Summary ---
n_rois = len(df)
n_sessions_with_light = df["exp_id"].nunique()
n_penk = len(df[df["celltype"] == "penk"])
n_nonpenk = len(df[df["celltype"] == "nonpenk"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sessions with light data", n_sessions_with_light)
col2.metric("Total ROIs", n_rois)
col3.metric("Penk ROIs", n_penk)
col4.metric("Non-Penk ROIs", n_nonpenk)

# --- Quality filter ---
snr_filter = st.slider("Min SNR", 0.0, 10.0, 2.0, 0.5, key="lc_snr")
df_filtered = df[df["snr"] >= snr_filter]
st.markdown(f"**{len(df_filtered)} ROIs** after SNR filter (>= {snr_filter})")

# --- Tabs ---
import plotly.express as px
import plotly.graph_objects as go

tab_dist, tab_celltype, tab_sessions, tab_stats = st.tabs([
    "LMI Distribution", "Penk vs Non-Penk", "Per-Session", "Statistics",
])

with tab_dist:
    st.subheader("Light Modulation Index Distribution")

    fig = px.histogram(
        df_filtered, x="lmi_dff", color="celltype", nbins=50,
        barmode="overlay", opacity=0.7,
        color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
        title="Light Modulation Index (dF/F0) — All ROIs",
        labels={"lmi_dff": "LMI (dF/F0)"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    if "lmi_events" in df_filtered.columns:
        fig = px.histogram(
            df_filtered.dropna(subset=["lmi_events"]),
            x="lmi_events", color="celltype", nbins=50,
            barmode="overlay", opacity=0.7,
            color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
            title="Light Modulation Index (Event Rate) — All ROIs",
            labels={"lmi_events": "LMI (events)"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


with tab_celltype:
    st.subheader("Penk vs Non-Penk Light Modulation")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for ct, color in CELLTYPE_HEX.items():
            ct_data = df_filtered[df_filtered["celltype"] == ct]
            if len(ct_data) > 0:
                fig.add_trace(go.Box(
                    y=ct_data["lmi_dff"],
                    name=f"{ct} (n={len(ct_data)})",
                    marker_color=color,
                    boxmean=True,
                ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=400, title="LMI (dF/F0) by Cell Type", yaxis_title="LMI")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "lmi_events" in df_filtered.columns:
            fig = go.Figure()
            for ct, color in CELLTYPE_HEX.items():
                ct_data = df_filtered[df_filtered["celltype"] == ct].dropna(subset=["lmi_events"])
                if len(ct_data) > 0:
                    fig.add_trace(go.Box(
                        y=ct_data["lmi_events"],
                        name=f"{ct} (n={len(ct_data)})",
                        marker_color=color,
                        boxmean=True,
                    ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=400, title="LMI (Events) by Cell Type", yaxis_title="LMI")
            st.plotly_chart(fig, use_container_width=True)

    # Scatter: light vs dark
    fig = px.scatter(
        df_filtered, x="light_mean", y="dark_mean", color="celltype",
        color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
        opacity=0.4,
        title="Mean dF/F0: Light vs Dark (per ROI)",
        labels={"light_mean": "Light ON mean dF/F0", "dark_mean": "Light OFF mean dF/F0"},
    )
    max_val = max(df_filtered["light_mean"].max(), df_filtered["dark_mean"].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", line=dict(dash="dash", color="gray"),
        showlegend=False,
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


with tab_sessions:
    st.subheader("Per-Session Light Modulation")

    session_stats = df_filtered.groupby(["exp_id", "celltype"]).agg(
        median_lmi=("lmi_dff", "median"),
        mean_lmi=("lmi_dff", "mean"),
        n_rois=("roi", "count"),
        pct_dark_prefer=("lmi_dff", lambda x: (x < 0).mean() * 100),
    ).reset_index()

    fig = px.bar(
        session_stats.sort_values("celltype"),
        x="exp_id", y="median_lmi", color="celltype",
        color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
        title="Median LMI per Session",
        hover_data=["n_rois", "pct_dark_prefer"],
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=400, xaxis=dict(tickangle=45))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(session_stats.round(3), use_container_width=True)


with tab_stats:
    st.subheader("Statistical Tests")

    penk_lmi = df_filtered[df_filtered["celltype"] == "penk"]["lmi_dff"].values
    nonpenk_lmi = df_filtered[df_filtered["celltype"] == "nonpenk"]["lmi_dff"].values

    if len(penk_lmi) > 0 and len(nonpenk_lmi) > 0:
        from scipy.stats import mannwhitneyu, ks_2samp

        # Mann-Whitney U
        stat_mw, pval_mw = mannwhitneyu(penk_lmi, nonpenk_lmi, alternative="two-sided")
        st.markdown(
            f"**Mann-Whitney U (LMI dF/F0):** "
            f"Penk median = {np.median(penk_lmi):.4f}, "
            f"Non-Penk median = {np.median(nonpenk_lmi):.4f}, "
            f"U = {stat_mw:.0f}, p = {pval_mw:.4f} "
            f"{'**(significant)**' if pval_mw < 0.05 else '(not significant)'}"
        )

        # KS test
        stat_ks, pval_ks = ks_2samp(penk_lmi, nonpenk_lmi)
        st.markdown(
            f"**Kolmogorov-Smirnov:** D = {stat_ks:.4f}, p = {pval_ks:.4f} "
            f"{'**(significant)**' if pval_ks < 0.05 else '(not significant)'}"
        )

        # Within-cell type: are cells modulated?
        from scipy.stats import wilcoxon as scipy_wilcoxon

        for ct, lmi_vals in [("Penk", penk_lmi), ("Non-Penk", nonpenk_lmi)]:
            if len(lmi_vals) >= 3:
                try:
                    stat_w, pval_w = scipy_wilcoxon(lmi_vals)
                    st.markdown(
                        f"**{ct} Wilcoxon (LMI != 0):** "
                        f"median = {np.median(lmi_vals):.4f}, "
                        f"p = {pval_w:.4f} "
                        f"{'**(significant)**' if pval_w < 0.05 else '(not significant)'}"
                    )
                except Exception:
                    pass

    # Event rate comparison
    if "lmi_events" in df_filtered.columns:
        st.markdown("---")
        penk_ev = df_filtered[df_filtered["celltype"] == "penk"]["lmi_events"].dropna().values
        nonpenk_ev = df_filtered[df_filtered["celltype"] == "nonpenk"]["lmi_events"].dropna().values

        if len(penk_ev) > 0 and len(nonpenk_ev) > 0:
            stat, pval = mannwhitneyu(penk_ev, nonpenk_ev, alternative="two-sided")
            st.markdown(
                f"**Mann-Whitney U (LMI Events):** "
                f"Penk median = {np.median(penk_ev):.4f}, "
                f"Non-Penk median = {np.median(nonpenk_ev):.4f}, "
                f"p = {pval:.4f} "
                f"{'**(significant)**' if pval < 0.05 else '(not significant)'}"
            )

    # Download
    st.markdown("---")
    csv = df_filtered.to_csv(index=False)
    st.download_button("Download light modulation data (CSV)", csv, "light_modulation.csv", "text/csv")
