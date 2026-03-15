"""Place Tuning Explorer — 2-D spatial rate map analysis.

Dedicated page for visualising and analysing place tuning:
rate maps, spatial information, coherence, sparsity, and
significance testing with circular shuffle.

Requires real sync.h5 data from S3.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

log = logging.getLogger("hm2p.frontend.place_tuning")

st.title("Place Tuning Explorer")
st.caption(
    "Interactive 2-D spatial rate map analysis — occupancy-normalised rate maps, "
    "spatial information, coherence, sparsity, and significance testing."
)

import plotly.express as px
import plotly.graph_objects as go

from hm2p.analysis.tuning import (
    compute_place_rate_map,
    spatial_coherence,
    spatial_information,
    spatial_sparsity,
)
from hm2p.constants import HEX_NONPENK, HEX_PENK


# --- Data loading ---


def _try_load_real():
    """Attempt to load real sync.h5 data."""
    try:
        from frontend.data import load_all_sync_data, session_filter_sidebar

        all_data = load_all_sync_data()
        if all_data["n_sessions"] > 0:
            sessions = session_filter_sidebar(
                all_data["sessions"], key_prefix="place_filter"
            )
            return sessions, True
    except Exception as e:
        st.warning(f"Could not load sync data: {e}")
    return None, False


real_sessions, has_real = _try_load_real()

if not has_real or not real_sessions:
    st.warning(
        "No data available yet. This page will populate when the relevant "
        "pipeline stage completes."
    )
    st.stop()

# Filter to sessions that have position data
sessions_with_pos = [s for s in real_sessions if s.get("x_mm") is not None]

st.success(
    f"Loaded {len(real_sessions)} sessions, "
    f"{sum(s['n_rois'] for s in real_sessions)} total cells "
    f"({len(sessions_with_pos)} sessions with position data)"
)


# --- Load precomputed analysis.h5 results ---
@st.cache_data(ttl=600)
def _load_analysis_results():
    """Load per-cell place tuning results from analysis.h5 files on S3."""
    import h5py as _h5py

    from frontend.data import DERIVATIVES_BUCKET, download_s3_bytes

    base = Path(__file__).resolve().parent.parent.parent / "metadata"
    animals_df = pd.read_csv(base / "animals.csv")
    animals_df["animal_id"] = animals_df["animal_id"].astype(str)
    exps_df = pd.read_csv(base / "experiments.csv")
    exps_df["animal_id"] = exps_df["exp_id"].str.split("_").str[-1]

    rows = []
    for _, exp in exps_df.iterrows():
        if str(exp.get("exclude", "0")).strip() == "1":
            continue
        eid = exp["exp_id"]
        aid = exp["animal_id"]
        parts = eid.split("_")
        sub = f"sub-{aid}"
        ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"

        animal = animals_df[animals_df["animal_id"] == aid]
        if animal.empty:
            continue
        celltype = str(animal.iloc[0].get("celltype", ""))

        data = download_s3_bytes(
            DERIVATIVES_BUCKET, f"analysis/{sub}/{ses}/analysis.h5"
        )
        if data is None:
            continue

        try:
            with _h5py.File(io.BytesIO(data), "r") as f:
                if "dff/place/all/spatial_info" not in f:
                    continue
                n = f["dff/place/all/spatial_info"].shape[0]
                for ri in range(n):
                    row = {
                        "exp_id": eid,
                        "animal_id": aid,
                        "celltype": celltype,
                        "roi": ri,
                    }
                    for cond in ("all", "light", "dark"):
                        grp = f.get(f"dff/place/{cond}")
                        if grp:
                            for k in (
                                "spatial_info",
                                "spatial_coherence",
                                "sparsity",
                                "p_value",
                                "significant",
                            ):
                                if k in grp:
                                    v = grp[k][ri]
                                    row[f"place_{cond}_{k}"] = (
                                        bool(v) if k == "significant" else float(v)
                                    )
                    comp = f.get("dff/place/comparison")
                    if comp:
                        if "correlation" in comp:
                            row["place_comp_correlation"] = float(comp["correlation"][ri])
                    # ROI type
                    sync_data = download_s3_bytes(
                        DERIVATIVES_BUCKET, f"sync/{sub}/{ses}/sync.h5"
                    )
                    if sync_data:
                        with _h5py.File(io.BytesIO(sync_data), "r") as sf:
                            if "roi_types" in sf:
                                row["roi_type"] = int(sf["roi_types"][ri])
                    rows.append(row)
        except Exception:
            continue

    return pd.DataFrame(rows) if rows else None


analysis_df = _load_analysis_results()

# --- Tabs ---
tab_single, tab_population, tab_lightdark, tab_celltype, tab_gallery, tab_methods = (
    st.tabs([
        "Single Cell",
        "Population",
        "Light vs Dark",
        "Penk+ vs Penk\u207bCamKII+",
        "Rate Map Gallery",
        "Methods & References",
    ])
)


# --- Helper: compute rate map for a session/ROI ---
def _compute_rate_map_for_cell(ses_data, cell_idx, bin_size=2.5, sigma=3.0):
    """Compute rate map for a single cell, returning None if no position data."""
    if ses_data.get("x_mm") is None:
        return None
    x_cm = ses_data["x_mm"] / 10.0
    y_cm = ses_data["y_mm"] / 10.0
    signal = ses_data["dff"][cell_idx]
    mask = ses_data["active"] & ~ses_data["bad_behav"]

    rate_map, occ_map, bx, by = compute_place_rate_map(
        signal, x_cm, y_cm, mask, bin_size=bin_size, smoothing_sigma=sigma, fps=9.8
    )
    si = spatial_information(rate_map, occ_map)
    coh = spatial_coherence(rate_map)
    spar = spatial_sparsity(rate_map, occ_map)
    return {
        "rate_map": rate_map,
        "occ_map": occ_map,
        "bin_edges_x": bx,
        "bin_edges_y": by,
        "spatial_info": si,
        "coherence": coh,
        "sparsity": spar,
    }


# --- Tab 1: Single Cell ---
with tab_single:
    st.subheader("Single Cell Rate Map")

    if not sessions_with_pos:
        st.info("No sessions with position data available.")
    else:
        session_labels = [s["exp_id"] for s in sessions_with_pos]
        sel_session_idx = st.selectbox(
            "Session",
            range(len(session_labels)),
            format_func=lambda i: session_labels[i],
            key="place_session",
        )
        ses_data = sessions_with_pos[sel_session_idx]
        n_rois = ses_data["n_rois"]
        sel_cell = st.slider(
            "Cell index", 0, max(0, n_rois - 1), 0, key="place_cell"
        )

        col_params1, col_params2 = st.columns(2)
        with col_params1:
            bin_size = st.select_slider(
                "Bin size (cm)",
                options=[1.0, 1.5, 2.0, 2.5, 3.0, 5.0],
                value=2.5,
                key="place_binsize",
            )
        with col_params2:
            sigma = st.slider(
                "Smoothing sigma (bins)", 0.0, 10.0, 3.0, 0.5, key="place_sigma"
            )

        result = _compute_rate_map_for_cell(ses_data, sel_cell, bin_size, sigma)

        if result is None:
            st.warning("Position data not available for this session.")
        else:
            # Metrics row
            c1, c2, c3 = st.columns(3)
            c1.metric("Spatial Info (bits/event)", f"{result['spatial_info']:.4f}")
            c2.metric("Coherence", f"{result['coherence']:.3f}")
            c3.metric(
                "Sparsity",
                f"{result['sparsity']:.3f}"
                if not np.isnan(result["sparsity"])
                else "---",
            )

            # Rate map and occupancy map side by side
            col_rate, col_occ = st.columns(2)

            rm = result["rate_map"]
            om = result["occ_map"]
            bx = result["bin_edges_x"]
            by = result["bin_edges_y"]
            # Bin centres for axis labels
            cx = (bx[:-1] + bx[1:]) / 2
            cy = (by[:-1] + by[1:]) / 2

            with col_rate:
                rm_display = np.where(np.isnan(rm), 0, rm)
                fig = go.Figure(
                    data=go.Heatmap(
                        z=rm_display,
                        x=cx,
                        y=cy,
                        colorscale="Hot",
                        colorbar=dict(title="dF/F"),
                        hoverongaps=False,
                    )
                )
                fig.update_layout(
                    height=400,
                    title="Rate Map",
                    xaxis_title="X (cm)",
                    yaxis_title="Y (cm)",
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_occ:
                fig = go.Figure(
                    data=go.Heatmap(
                        z=om,
                        x=cx,
                        y=cy,
                        colorscale="Blues",
                        colorbar=dict(title="Seconds"),
                        hoverongaps=False,
                    )
                )
                fig.update_layout(
                    height=400,
                    title="Occupancy Map",
                    xaxis_title="X (cm)",
                    yaxis_title="Y (cm)",
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                )
                st.plotly_chart(fig, use_container_width=True)


# --- Tab 2: Population ---
with tab_population:
    st.subheader("Population Place Tuning")

    if analysis_df is None or analysis_df.empty:
        st.info("No precomputed analysis results. Run Stage 6 first.")
    else:
        # Filter to soma
        soma_df = (
            analysis_df[analysis_df.get("roi_type", 0) == 0]
            if "roi_type" in analysis_df.columns
            else analysis_df
        )

        n_total = len(soma_df)
        n_sig_all = (
            soma_df["place_all_significant"].sum()
            if "place_all_significant" in soma_df.columns
            else 0
        )
        n_sig_light = (
            soma_df["place_light_significant"].sum()
            if "place_light_significant" in soma_df.columns
            else 0
        )
        n_sig_dark = (
            soma_df["place_dark_significant"].sum()
            if "place_dark_significant" in soma_df.columns
            else 0
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total soma ROIs", n_total)
        c2.metric(
            "Place tuned (all)",
            f"{n_sig_all} ({100 * n_sig_all / n_total:.1f}%)" if n_total else "0",
        )
        c3.metric(
            "Place tuned (light)",
            f"{n_sig_light} ({100 * n_sig_light / n_total:.1f}%)" if n_total else "0",
        )
        c4.metric(
            "Place tuned (dark)",
            f"{n_sig_dark} ({100 * n_sig_dark / n_total:.1f}%)" if n_total else "0",
        )

        # By celltype
        st.subheader("By cell type")
        for ct in ["penk", "nonpenk"]:
            ct_df = soma_df[soma_df["celltype"] == ct]
            n_ct = len(ct_df)
            if n_ct == 0:
                continue
            label = "Penk+" if ct == "penk" else "Penk\u207bCamKII+"
            n_sig = (
                ct_df["place_all_significant"].sum()
                if "place_all_significant" in ct_df.columns
                else 0
            )
            mean_si = (
                ct_df["place_all_spatial_info"].mean()
                if "place_all_spatial_info" in ct_df.columns
                else 0
            )
            st.markdown(
                f"**{label}:** {n_ct} cells, {n_sig} place tuned "
                f"({100 * n_sig / n_ct:.1f}%), mean SI = {mean_si:.4f} bits/event"
            )

        # SI distribution histogram
        if "place_all_spatial_info" in soma_df.columns:
            st.subheader("Spatial Information Distribution")
            si_vals = soma_df["place_all_spatial_info"].dropna()
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=si_vals,
                    nbinsx=25,
                    marker_color="royalblue",
                    name="All cells",
                )
            )
            fig.update_layout(
                height=350,
                xaxis_title="Spatial Information (bits/event)",
                yaxis_title="Count",
                title="SI Distribution (all conditions)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Coherence vs Sparsity scatter
        if (
            "place_all_spatial_coherence" in soma_df.columns
            and "place_all_sparsity" in soma_df.columns
        ):
            st.subheader("Coherence vs Sparsity")
            sub = soma_df[
                ["place_all_spatial_coherence", "place_all_sparsity", "celltype"]
            ].dropna()
            if not sub.empty:
                fig = go.Figure()
                for ct, color, name in [
                    ("penk", HEX_PENK, "Penk+"),
                    ("nonpenk", HEX_NONPENK, "Penk\u207bCamKII+"),
                ]:
                    ct_sub = sub[sub["celltype"] == ct]
                    if ct_sub.empty:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=ct_sub["place_all_spatial_coherence"],
                            y=ct_sub["place_all_sparsity"],
                            mode="markers",
                            marker=dict(size=5, opacity=0.6, color=color),
                            name=name,
                        )
                    )
                fig.update_layout(
                    height=400,
                    xaxis_title="Spatial Coherence (r)",
                    yaxis_title="Sparsity",
                    title="Coherence vs Sparsity",
                )
                st.plotly_chart(fig, use_container_width=True)


# --- Tab 3: Light vs Dark ---
with tab_lightdark:
    st.subheader("Light vs Dark Place Tuning")

    if analysis_df is None or analysis_df.empty:
        st.info("No precomputed analysis results.")
    else:
        soma_df = (
            analysis_df[analysis_df.get("roi_type", 0) == 0]
            if "roi_type" in analysis_df.columns
            else analysis_df
        )

        if (
            "place_light_spatial_info" in soma_df.columns
            and "place_dark_spatial_info" in soma_df.columns
        ):
            sub = soma_df[
                ["place_light_spatial_info", "place_dark_spatial_info"]
            ].dropna()

            # Paired Wilcoxon
            from scipy.stats import wilcoxon as _wilcoxon

            diff = (
                sub["place_light_spatial_info"].values
                - sub["place_dark_spatial_info"].values
            )
            nonzero = diff[diff != 0]
            if len(nonzero) > 5:
                _, p_si = _wilcoxon(nonzero, alternative="two-sided")
            else:
                p_si = np.nan

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Light SI (mean)", f"{sub['place_light_spatial_info'].mean():.4f}"
            )
            c2.metric(
                "Dark SI (mean)", f"{sub['place_dark_spatial_info'].mean():.4f}"
            )
            c3.metric(
                "Wilcoxon p", f"{p_si:.4f}" if not np.isnan(p_si) else "n/a"
            )

            # Scatter: light vs dark SI
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=sub["place_light_spatial_info"],
                    y=sub["place_dark_spatial_info"],
                    mode="markers",
                    marker=dict(size=4, opacity=0.5, color="royalblue"),
                    name="Cells",
                )
            )
            max_val = max(
                sub["place_light_spatial_info"].max(),
                sub["place_dark_spatial_info"].max(),
            )
            max_val = max_val * 1.1 if max_val > 0 else 1.0
            fig.add_trace(
                go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode="lines",
                    line=dict(color="gray", dash="dash"),
                    name="Unity",
                )
            )
            fig.update_layout(
                height=400,
                width=400,
                xaxis_title="Light SI (bits/event)",
                yaxis_title="Dark SI (bits/event)",
                title=(
                    f"Light vs Dark SI (Wilcoxon p = {p_si:.4f})"
                    if not np.isnan(p_si)
                    else "Light vs Dark SI"
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "Light/dark spatial info not available in precomputed results."
            )

        # Rate map correlation distribution
        if "place_comp_correlation" in soma_df.columns:
            corrs = soma_df["place_comp_correlation"].dropna()
            if not corrs.empty:
                st.subheader("Rate Map Correlation (light vs dark)")
                fig = go.Figure(
                    data=[
                        go.Histogram(
                            x=corrs, nbinsx=20, marker_color="teal"
                        )
                    ]
                )
                fig.update_layout(
                    height=300,
                    xaxis_title="Pearson r",
                    yaxis_title="Count",
                    title=f"Light-dark rate map correlation (mean = {corrs.mean():.3f}, N = {len(corrs)})",
                )
                st.plotly_chart(fig, use_container_width=True)


# --- Tab 4: Penk+ vs Penk-CamKII+ ---
with tab_celltype:
    st.subheader("Penk+ vs Penk\u207bCamKII+ Place Tuning")

    if analysis_df is None or analysis_df.empty:
        st.info("No precomputed analysis results.")
    else:
        from scipy.stats import mannwhitneyu as _mwu

        soma_df = (
            analysis_df[analysis_df.get("roi_type", 0) == 0]
            if "roi_type" in analysis_df.columns
            else analysis_df
        )

        metrics_to_compare = [
            ("place_all_spatial_info", "SI (all)"),
            ("place_light_spatial_info", "SI (light)"),
            ("place_dark_spatial_info", "SI (dark)"),
            ("place_all_spatial_coherence", "Coherence (all)"),
            ("place_all_sparsity", "Sparsity (all)"),
            ("place_comp_correlation", "Rate map corr (light-dark)"),
        ]

        # Animal-level Mann-Whitney for each metric
        st.markdown("**Between-group tests (animal-level Mann-Whitney U):**")
        summary_rows = []
        for col, label in metrics_to_compare:
            if col not in soma_df.columns:
                continue
            sub = soma_df[["animal_id", "celltype", col]].dropna()
            animal_means = (
                sub.groupby(["animal_id", "celltype"])[col].mean().reset_index()
            )
            penk_vals = animal_means.loc[
                animal_means["celltype"] == "penk", col
            ].values
            nonpenk_vals = animal_means.loc[
                animal_means["celltype"] == "nonpenk", col
            ].values

            if len(penk_vals) >= 2 and len(nonpenk_vals) >= 2:
                stat, p = _mwu(penk_vals, nonpenk_vals, alternative="two-sided")
                sig = " *" if p < 0.05 else ""
            else:
                p = np.nan
                sig = ""

            summary_rows.append({
                "Metric": label,
                "Penk+ (mean)": (
                    f"{penk_vals.mean():.4f}" if len(penk_vals) > 0 else "n/a"
                ),
                "Penk\u207bCamKII+ (mean)": (
                    f"{nonpenk_vals.mean():.4f}" if len(nonpenk_vals) > 0 else "n/a"
                ),
                "N animals": f"{len(penk_vals)} vs {len(nonpenk_vals)}",
                "p": f"{p:.4f}{sig}" if not np.isnan(p) else "n/a",
            })

        if summary_rows:
            st.dataframe(
                pd.DataFrame(summary_rows), use_container_width=True, hide_index=True
            )

        # Box plots
        st.subheader("Distributions by cell type")
        n_cols = 3
        metrics_avail = [
            (c, l) for c, l in metrics_to_compare if c in soma_df.columns
        ]
        for i in range(0, len(metrics_avail), n_cols):
            batch = metrics_avail[i : i + n_cols]
            cols = st.columns(n_cols)
            for j, (col, label) in enumerate(batch):
                with cols[j]:
                    fig = go.Figure()
                    for ct, color, name in [
                        ("penk", HEX_PENK, "Penk+"),
                        ("nonpenk", HEX_NONPENK, "Penk\u207bCamKII+"),
                    ]:
                        vals = soma_df.loc[
                            soma_df["celltype"] == ct, col
                        ].dropna()
                        fig.add_trace(
                            go.Box(
                                y=vals,
                                name=name,
                                marker_color=color,
                                boxpoints="all",
                                jitter=0.3,
                                pointpos=-1.5,
                                marker=dict(size=3, opacity=0.5),
                            )
                        )
                    fig.update_layout(
                        height=300,
                        showlegend=False,
                        title=dict(text=label, font_size=12),
                        yaxis_title=label,
                        margin=dict(t=40, b=30),
                    )
                    st.plotly_chart(
                        fig, use_container_width=True, key=f"place_ct_box_{col}"
                    )

        # Fraction place-tuned by celltype
        st.subheader("Fraction place tuned")
        if "place_all_significant" in soma_df.columns:
            frac_rows = []
            for ct in ["penk", "nonpenk"]:
                ct_df = soma_df[soma_df["celltype"] == ct]
                label = "Penk+" if ct == "penk" else "Penk\u207bCamKII+"
                n_ct = len(ct_df)
                if n_ct == 0:
                    continue
                row = {"Cell type": label, "N cells": n_ct}
                for cond in ["all", "light", "dark"]:
                    sig_col = f"place_{cond}_significant"
                    if sig_col in ct_df.columns:
                        n_sig = ct_df[sig_col].sum()
                        pct = 100 * n_sig / n_ct
                        row[f"Place tuned ({cond})"] = f"{n_sig}/{n_ct} ({pct:.1f}%)"
                frac_rows.append(row)
            if frac_rows:
                st.dataframe(
                    pd.DataFrame(frac_rows), use_container_width=True, hide_index=True
                )


# --- Tab 5: Rate Map Gallery ---
with tab_gallery:
    st.subheader("Top Cells by Spatial Information")

    if analysis_df is None or analysis_df.empty or not sessions_with_pos:
        st.info(
            "Requires both precomputed analysis results and sessions with "
            "position data."
        )
    else:
        soma_df = (
            analysis_df[analysis_df.get("roi_type", 0) == 0]
            if "roi_type" in analysis_df.columns
            else analysis_df
        )

        if "place_all_spatial_info" not in soma_df.columns:
            st.info("No spatial info data in precomputed results.")
        else:
            n_top = st.slider("Number of top cells", 5, 40, 20, key="place_n_top")
            top_df = (
                soma_df.dropna(subset=["place_all_spatial_info"])
                .sort_values("place_all_spatial_info", ascending=False)
                .head(n_top)
            )

            if top_df.empty:
                st.info("No cells with spatial information values.")
            else:
                # Build a lookup for sessions with position data
                ses_lookup = {s["exp_id"]: s for s in sessions_with_pos}

                n_gallery_cols = 4
                gallery_items = []
                for _, row in top_df.iterrows():
                    eid = row["exp_id"]
                    roi = int(row["roi"])
                    ct = row.get("celltype", "unknown")
                    si = row["place_all_spatial_info"]
                    gallery_items.append((eid, roi, ct, si))

                for i in range(0, len(gallery_items), n_gallery_cols):
                    batch = gallery_items[i : i + n_gallery_cols]
                    cols = st.columns(n_gallery_cols)
                    for j, (eid, roi, ct, si) in enumerate(batch):
                        with cols[j]:
                            ses = ses_lookup.get(eid)
                            if ses is None:
                                st.caption(f"{eid} ROI {roi}")
                                st.info("No position data")
                                continue

                            # Check that ROI index is valid
                            if roi >= ses["n_rois"]:
                                st.caption(f"{eid} ROI {roi}")
                                st.info("ROI index out of range")
                                continue

                            result = _compute_rate_map_for_cell(
                                ses, roi, bin_size=2.5, sigma=3.0
                            )
                            if result is None:
                                st.caption(f"{eid} ROI {roi}")
                                st.info("No position data")
                                continue

                            rm = result["rate_map"]
                            rm_display = np.where(np.isnan(rm), 0, rm)
                            bx = result["bin_edges_x"]
                            by = result["bin_edges_y"]
                            cx = (bx[:-1] + bx[1:]) / 2
                            cy = (by[:-1] + by[1:]) / 2

                            border_color = (
                                HEX_PENK if ct == "penk" else HEX_NONPENK
                            )
                            ct_label = (
                                "Penk+" if ct == "penk" else "Penk\u207bCamKII+"
                            )

                            fig = go.Figure(
                                data=go.Heatmap(
                                    z=rm_display,
                                    x=cx,
                                    y=cy,
                                    colorscale="Hot",
                                    showscale=False,
                                    hoverongaps=False,
                                )
                            )
                            fig.update_layout(
                                height=220,
                                margin=dict(l=5, r=5, t=30, b=5),
                                title=dict(
                                    text=f"SI={si:.3f}",
                                    font_size=10,
                                ),
                                xaxis=dict(
                                    visible=False, scaleanchor="y", scaleratio=1
                                ),
                                yaxis=dict(visible=False),
                                # Border effect via paper_bgcolor
                                paper_bgcolor=border_color,
                                plot_bgcolor="black",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(
                                f"{eid}\nROI {roi} ({ct_label})"
                            )


# --- Tab 6: Methods & References ---
with tab_methods:
    st.subheader("Methods & References")

    st.markdown("""
**Rate map computation:**
2-D occupancy-normalised rate maps are computed by binning animal position into
spatial bins (default 2.5 cm), accumulating signal (dF/F) and occupancy per bin,
dividing signal by occupancy, and applying Gaussian smoothing with
Nadaraya-Watson normalisation to handle NaN (unvisited) bins.

**Spatial information:**
Skaggs spatial information quantifies how much information (in bits per event)
a cell's firing rate carries about the animal's location:

SI = sum_i p_i * (r_i / r_mean) * log2(r_i / r_mean)

where p_i is the occupancy probability in bin i, r_i is the mean rate in bin i,
and r_mean is the overall mean rate.

**Spatial coherence:**
Pearson correlation between each bin's rate and the mean rate of its 8 nearest
neighbours. High coherence indicates spatially smooth firing fields.

**Spatial sparsity:**
Measures how concentrated the spatial firing is. Low sparsity indicates a cell
fires in few spatial bins (place-like). Computed as:

sparsity = (sum(p_i * r_i))^2 / sum(p_i * r_i^2)

**Significance testing:**
Circular time-shift shuffle: the neural signal is circularly shifted by a random
offset (minimum 30 s) relative to position, and spatial information is recomputed.
This is repeated N times (default 500) to build a null distribution. The cell is
deemed significantly place-tuned if its observed SI exceeds the 95th percentile
of the shuffle distribution (p < 0.05).

---

**References:**

- Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. 1993.
  "An information-theoretic approach to deciphering the hippocampal code."
  *Advances in Neural Information Processing Systems* 5.
  doi:10.1162/neco.1996.8.6.1345

- Muller, R. U., Kubie, J. L., & Ranck, J. B. 1987.
  "Spatial firing patterns of hippocampal complex-spike cells in a fixed
  environment." *Journal of Neuroscience* 7(7), 1951-1968.
  doi:10.1523/JNEUROSCI.07-07-01951.1987
  (Circular shuffle significance testing method)
""")


# --- Footer ---
st.markdown("---")
st.caption(
    "Place tuning analysis uses occupancy-normalised 2-D rate maps with "
    "Gaussian smoothing. Spatial information: Skaggs et al. 1993. "
    "Significance via circular time-shift shuffle (Muller et al. 1987)."
)
