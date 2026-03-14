"""Cell Summary — combined cell classification + batch quality overview.

Merges the former Summary and Batch Overview pages into one.
Tab 1: Cell classification, MVL, gain, speed modulation across all sessions.
Tab 2: Per-session quality metrics (ROI counts, SNR, event rates, duration).
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    download_s3_bytes,
    load_animals,
    load_experiments,
    parse_session_id,
)
from hm2p.constants import CELLTYPE_HEX

log = logging.getLogger("hm2p.frontend.summary")

st.title("Cell Summary")
st.caption(
    "At-a-glance overview of ALL cells across ALL sessions. "
    "Filter by cell type or animal in the sidebar."
)

tab_cells, tab_batch = st.tabs(["Cell Classification", "Batch Quality"])

# ══════════════════════════════════════════════════════════════════════════
# Tab 1: Cell Classification
# ══════════════════════════════════════════════════════════════════════════

with tab_cells:

    def _try_load_real_data():
        """Attempt to load real sync.h5 data from S3."""
        try:
            from frontend.data import load_all_sync_data, session_filter_sidebar
            all_data = load_all_sync_data()
            if all_data["n_sessions"] > 0:
                sessions = session_filter_sidebar(all_data["sessions"])
                return sessions, True
        except Exception as e:
            log.debug("Could not load real data: %s", e)
        return None, False

    real_sessions, has_real = _try_load_real_data()

    if not (has_real and real_sessions):
        st.warning("No sync data available yet. This tab will populate automatically "
                   "when Stage 5 (sync) completes.")
    else:
        from hm2p.analysis.classify import classify_population, classification_summary_table
        from hm2p.analysis.gain import population_gain_modulation
        from hm2p.analysis.speed import speed_modulation_index

        st.success(
            f"Loaded {len(real_sessions)} sessions, "
            f"{sum(s['n_rois'] for s in real_sessions)} total cells"
        )

        # Pooled data for analysis — run per session, aggregate results
        all_cells_info = []

        for ses_data in real_sessions:
            signals = ses_data["dff"]
            hd = ses_data["hd_deg"]
            mask = ses_data["active"] & ~ses_data["bad_behav"]
            light_on = ses_data["light_on"]
            speed = ses_data["speed_cm_s"]
            n_rois = ses_data["n_rois"]
            exp_id = ses_data["exp_id"]
            celltype = ses_data["celltype"]

            # Classify per session
            pop = classify_population(
                signals, hd, mask, n_shuffles=200,
                rng=np.random.default_rng(42),
            )
            table = classification_summary_table(pop)
            for row in table:
                row["exp_id"] = exp_id
                row["celltype"] = celltype
                row["animal_id"] = ses_data["animal_id"]
                all_cells_info.append(row)

            # Gain + speed per cell
            gains = population_gain_modulation(signals, hd, mask, light_on)
            for i in range(n_rois):
                smi = speed_modulation_index(signals[i], speed, mask)
                all_cells_info[-n_rois + i]["gain_index"] = gains[i]["gain_index"]
                all_cells_info[-n_rois + i]["smi"] = smi["speed_modulation_index"]

        # Build DataFrame
        df = pd.DataFrame(all_cells_info)
        n_total = len(df)
        n_hd = df["is_hd"].sum()

        # ── Top metrics ────────────────────────────────────────────────────
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Cells", n_total)
        col2.metric("HD Cells", int(n_hd))
        col3.metric("Non-HD", int(n_total - n_hd))
        col4.metric("HD Fraction", f"{n_hd / n_total:.0%}" if n_total > 0 else "N/A")

        hd_mvls = df[df["is_hd"]]["mvl"].values
        col5.metric("Mean HD MVL", f"{np.mean(hd_mvls):.3f}" if len(hd_mvls) > 0 else "N/A")

        n_sessions = len(real_sessions)
        celltypes = df["celltype"].value_counts()
        ct_str = " | ".join(f"{ct}: {n}" for ct, n in celltypes.items())
        st.caption(f"{n_sessions} sessions | {ct_str}")

        st.markdown("---")

        # ── Classification table ───────────────────────────────────────────
        st.subheader("Cell Classification")
        df_display = df[["cell", "is_hd", "grade", "mvl", "p_value", "reliability", "mi",
                          "preferred_direction"]].copy()
        df_display.insert(0, "session", df["exp_id"])
        df_display.insert(1, "celltype", df["celltype"])

        df_display["mvl"] = df_display["mvl"].apply(lambda x: f"{x:.3f}")
        df_display["p_value"] = df_display["p_value"].apply(lambda x: f"{x:.4f}")
        df_display["reliability"] = df_display["reliability"].apply(lambda x: f"{x:.3f}")
        df_display["mi"] = df_display["mi"].apply(lambda x: f"{x:.4f}")
        df_display["preferred_direction"] = df_display["preferred_direction"].apply(lambda x: f"{x:.1f}°")

        st.dataframe(df_display, use_container_width=True, hide_index=True, height=300)

        st.markdown(
            "**Grades:** A = strong HD (MVL>=0.4, reliability>=0.8) · "
            "B = moderate HD (MVL>=0.25) · C = weak HD · D = non-HD"
        )

        # ── Key metrics panels ─────────────────────────────────────────────
        st.subheader("Population Metrics")
        col_mvl, col_gain, col_speed = st.columns(3)

        with col_mvl:
            st.markdown("**MVL Distribution**")
            fig = go.Figure()
            for ct in df["celltype"].unique():
                subset = df[df["celltype"] == ct]
                fig.add_trace(go.Histogram(x=subset["mvl"], name=ct, opacity=0.7))
            fig.update_layout(height=250, xaxis_title="MVL", yaxis_title="Count",
                               barmode="overlay", margin=dict(t=10, b=30))
            st.plotly_chart(fig, use_container_width=True, key="mvl_hist")

        with col_gain:
            st.markdown("**Gain Modulation**")
            if "gain_index" in df.columns:
                mean_gmi = df["gain_index"].mean()
                st.metric("Mean GMI", f"{mean_gmi:.3f}")
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df["gain_index"], marker_color="orange"))
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                fig.update_layout(height=200, xaxis_title="GMI", margin=dict(t=10, b=30))
                st.plotly_chart(fig, use_container_width=True, key="gmi_hist")

        with col_speed:
            st.markdown("**Speed Modulation**")
            if "smi" in df.columns:
                mean_smi = df["smi"].mean()
                st.metric("Mean SMI", f"{mean_smi:.3f}")
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df["smi"], marker_color="green"))
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                fig.update_layout(height=200, xaxis_title="SMI", margin=dict(t=10, b=30))
                st.plotly_chart(fig, use_container_width=True, key="smi_hist")

        # ── Grade breakdown by celltype ────────────────────────────────────
        if "celltype" in df.columns:
            st.markdown("---")
            st.subheader("Classification by Cell Type")
            for ct in df["celltype"].unique():
                subset = df[df["celltype"] == ct]
                n_ct = len(subset)
                n_ct_hd = subset["is_hd"].sum()
                grade_counts = subset["grade"].value_counts().to_dict()
                grades_str = " | ".join(f"{g}: {n}" for g, n in sorted(grade_counts.items()))
                st.markdown(
                    f"**{ct}** — {n_ct} cells, {n_ct_hd} HD ({n_ct_hd/n_ct:.0%}) — {grades_str}"
                )


# ══════════════════════════════════════════════════════════════════════════
# Tab 2: Batch Quality
# ══════════════════════════════════════════════════════════════════════════

with tab_batch:

    @st.cache_data(ttl=600)
    def load_batch_summary() -> pd.DataFrame:
        """Load per-session summary metrics from ca.h5 files."""
        import h5py

        experiments = load_experiments()
        animals = load_animals()
        animal_map = {a["animal_id"]: a for a in animals}
        rows = []

        for exp in experiments:
            exp_id = exp["exp_id"]
            animal_id = exp_id.split("_")[-1]
            animal = animal_map.get(animal_id, {})
            sub, ses = parse_session_id(exp_id)

            row = {
                "exp_id": exp_id,
                "animal": animal_id,
                "celltype": animal.get("celltype", "?"),
                "sub": sub,
                "ses": ses,
                "exclude": exp.get("exclude", "0"),
            }

            data = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
            if data is None:
                row.update({
                    "n_rois": 0, "n_frames": 0, "duration_s": 0, "fps": 0,
                    "median_snr": 0, "mean_snr": 0, "n_good_rois": 0,
                    "median_event_rate": 0, "mean_max_dff": 0,
                    "has_events": False, "has_deconv": False,
                })
                rows.append(row)
                continue

            try:
                f = h5py.File(io.BytesIO(data), "r")
                dff = f["dff"][:]
                n_rois, n_frames = dff.shape
                fps = float(f.attrs.get("fps_imaging", 9.8))

                # Per-ROI SNR
                snrs = []
                for i in range(n_rois):
                    trace = dff[i]
                    baseline_std = np.std(trace[trace < np.percentile(trace, 50)])
                    peak = np.percentile(trace, 95)
                    snrs.append(peak / baseline_std if baseline_std > 0 else 0)
                snrs = np.array(snrs)

                # Event stats
                has_events = "event_masks" in f
                event_rates = []
                if has_events:
                    em = f["event_masks"][:]
                    for i in range(n_rois):
                        emask = em[i].astype(bool)
                        onsets = np.flatnonzero(emask[1:] & ~emask[:-1])
                        n_events = len(onsets) + (1 if emask[0] else 0)
                        event_rates.append(n_events / (n_frames / fps / 60))
                    event_rates = np.array(event_rates)

                row.update({
                    "n_rois": n_rois,
                    "n_frames": n_frames,
                    "duration_s": round(n_frames / fps, 1),
                    "fps": round(fps, 1),
                    "median_snr": round(float(np.median(snrs)), 2),
                    "mean_snr": round(float(np.mean(snrs)), 2),
                    "n_good_rois": int(np.sum(snrs >= 3)),
                    "median_event_rate": round(float(np.median(event_rates)), 1) if len(event_rates) > 0 else 0,
                    "mean_max_dff": round(float(np.mean([np.nanmax(dff[i]) for i in range(n_rois)])), 3),
                    "has_events": has_events,
                    "has_deconv": "spks" in f,
                })
                f.close()
            except Exception:
                row.update({
                    "n_rois": 0, "n_frames": 0, "duration_s": 0, "fps": 0,
                    "median_snr": 0, "mean_snr": 0, "n_good_rois": 0,
                    "median_event_rate": 0, "mean_max_dff": 0,
                    "has_events": False, "has_deconv": False,
                })

            rows.append(row)

        return pd.DataFrame(rows)

    with st.spinner("Loading batch summary (this may take a moment)..."):
        batch_df = load_batch_summary()

    # --- Summary cards ---
    n_sessions = len(batch_df)
    total_rois = batch_df["n_rois"].sum()
    total_good = batch_df["n_good_rois"].sum()
    total_duration = batch_df["duration_s"].sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Sessions", n_sessions)
    col2.metric("Total ROIs", total_rois)
    col3.metric("Good ROIs (SNR>=3)", total_good)
    col4.metric("Total recording", f"{total_duration/3600:.1f} hrs")
    col5.metric("Mean session", f"{batch_df['duration_s'].mean():.0f}s")

    # --- Celltype breakdown ---
    col1, col2 = st.columns(2)
    with col1:
        penk = batch_df[batch_df["celltype"] == "penk"]
        st.markdown(f"**Penk+**: {len(penk)} sessions, {penk['n_rois'].sum()} ROIs ({penk['n_good_rois'].sum()} good)")
    with col2:
        nonpenk = batch_df[batch_df["celltype"] == "nonpenk"]
        st.markdown(f"**Non-Penk**: {len(nonpenk)} sessions, {nonpenk['n_rois'].sum()} ROIs ({nonpenk['n_good_rois'].sum()} good)")

    # --- Main table ---
    st.subheader("Session Summary Table")

    def style_snr(val):
        if isinstance(val, (int, float)):
            if val >= 5:
                return "background-color: #d4edda"
            elif val >= 3:
                return "background-color: #fff3cd"
            elif val > 0:
                return "background-color: #f8d7da"
        return ""

    def style_rois(val):
        if isinstance(val, (int, float)) and val == 0:
            return "background-color: #f8d7da"
        return ""

    display_cols = [
        "exp_id", "celltype", "n_rois", "n_good_rois", "duration_s",
        "median_snr", "mean_snr", "median_event_rate", "mean_max_dff",
        "has_events", "has_deconv", "exclude",
    ]

    styled = batch_df[display_cols].style.map(
        style_snr, subset=["median_snr", "mean_snr"]
    ).map(
        style_rois, subset=["n_rois"]
    )

    st.dataframe(styled, use_container_width=True, height=500)

    # --- Visualizations ---
    st.subheader("Cross-Session Comparisons")

    tab_rois, tab_quality, tab_duration = st.tabs(["ROI Counts", "Quality", "Duration"])

    with tab_rois:
        fig = px.bar(
            batch_df.sort_values("celltype"),
            x="exp_id",
            y="n_rois",
            color="celltype",
            color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
            title="ROIs per Session",
            hover_data=["n_good_rois", "median_snr"],
        )
        fig.update_layout(height=350, xaxis=dict(tickangle=45))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=batch_df["exp_id"],
            y=batch_df["n_rois"],
            name="Total ROIs",
            marker_color="lightgray",
        ))
        fig2.add_trace(go.Bar(
            x=batch_df["exp_id"],
            y=batch_df["n_good_rois"],
            name="Good ROIs (SNR>=3)",
            marker_color="green",
        ))
        fig2.update_layout(
            barmode="overlay",
            title="Total vs Good ROIs",
            height=350,
            xaxis=dict(tickangle=45),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab_quality:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                batch_df.sort_values("median_snr", ascending=False),
                x="exp_id",
                y="median_snr",
                color="celltype",
                color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
                title="Median SNR by Session",
            )
            fig.add_hline(y=3, line_dash="dash", line_color="red", opacity=0.5)
            fig.update_layout(height=350, xaxis=dict(tickangle=45))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                batch_df[batch_df["median_event_rate"] > 0].sort_values("median_event_rate", ascending=False),
                x="exp_id",
                y="median_event_rate",
                color="celltype",
                color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
                title="Median Event Rate (events/min)",
            )
            fig.update_layout(height=350, xaxis=dict(tickangle=45))
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            batch_df,
            x="median_snr",
            y="median_event_rate",
            color="celltype",
            size="n_rois",
            hover_data=["exp_id"],
            color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
            title="SNR vs Event Rate (per session)",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with tab_duration:
        fig = px.bar(
            batch_df,
            x="exp_id",
            y="duration_s",
            color="celltype",
            color_discrete_map={**CELLTYPE_HEX, "?": "gray"},
            title="Session Duration (seconds)",
        )
        fig.update_layout(height=350, xaxis=dict(tickangle=45))
        st.plotly_chart(fig, use_container_width=True)

    # --- Download ---
    st.markdown("---")
    csv = batch_df.to_csv(index=False)
    st.download_button("Download batch summary CSV", csv, "batch_summary.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════
# Methods & References
# ══════════════════════════════════════════════════════════════════════════

st.markdown("---")

with st.expander("Methods & References"):
    st.markdown("""
**dF/F0 baseline:** Rolling Gaussian smooth + min + max filter
(Pachitariu et al. 2017, [doi:10.1101/061507](https://doi.org/10.1101/061507)).
[Suite2p GitHub](https://github.com/MouseLand/suite2p)

**Event detection:** Percentile-based noise model with CDF thresholding
(Voigts & Harnett 2020, [doi:10.1016/j.neuron.2019.10.016](https://doi.org/10.1016/j.neuron.2019.10.016)).
[GitHub](https://github.com/jvoigts/cell_labeling_bhv)

**HD tuning curves:** Occupancy-normalized spike/calcium rate per angular bin
(Taube et al. 1990, [doi:10.1523/JNEUROSCI.10-02-00420.1990](https://doi.org/10.1523/JNEUROSCI.10-02-00420.1990)).

**Mean vector length (MVL):** Resultant vector length of tuning curve
(Skaggs et al. 1996, [doi:10.1002/(SICI)1098-1063(1996)6:2<149::AID-HIPO6>3.0.CO;2-K](https://doi.org/10.1002/(SICI)1098-1063(1996)6:2<149::AID-HIPO6>3.0.CO;2-K)).

**Spatial information:** Skaggs information rate in bits/spike
(Skaggs et al. 1993, [doi:10.1162/neco.1996.8.6.1345](https://doi.org/10.1162/neco.1996.8.6.1345)).

**Significance testing:** Circular time-shift shuffle
(Muller et al. 1987, [doi:10.1523/JNEUROSCI.07-07-01951.1987](https://doi.org/10.1523/JNEUROSCI.07-07-01951.1987)).

**Soma/dendrite classification:** Aspect ratio heuristic from Suite2p stat.npy
(aspect_ratio > 2.5 = dendrite). Analysis defaults to **soma only**.
""")
