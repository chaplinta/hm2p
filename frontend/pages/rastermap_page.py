"""Rastermap — sorted neural population visualization.

Rastermap (Stringer et al. 2025) sorts neurons so nearby ones have similar
activity, revealing sequences, sustained states, and tuning without prior
knowledge of what drives the activity.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    check_stale_data_warning,
    download_s3_bytes,
    load_animals,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.rastermap")


def _page() -> None:
    st.title("Rastermap")
    check_stale_data_warning(stages=["sync"])
    st.caption(
        "Neurons sorted by activity similarity (Stringer et al. 2025, Nat Neurosci). "
        "Reveals sequential activation, sustained states, and tuning structure."
    )

    experiments = load_experiments()
    animals = load_animals()
    animal_map = {a["animal_id"]: a for a in animals}

    exp_ids = [e["exp_id"] for e in experiments if e.get("exclude", "0") != "1"]
    selected = st.selectbox(
        "Session", exp_ids,
        format_func=lambda x: f"{x} ({animal_map.get(x.split('_')[-1], {}).get('celltype', '?')})",
        key="rmap_session",
    )
    if not selected:
        st.stop()

    sub, ses = parse_session_id(selected)
    celltype = animal_map.get(selected.split("_")[-1], {}).get("celltype", "?")

    @st.cache_data(ttl=600)
    def _load(sub: str, ses: str):
        import h5py

        # Load ca.h5
        ca_bytes = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
        if ca_bytes is None:
            return None
        result = {}
        with h5py.File(io.BytesIO(ca_bytes), "r") as f:
            result["dff"] = f["dff"][:]
            result["fps"] = float(f.attrs.get("fps_imaging", 9.8))
            if "roi_types" in f:
                result["roi_types"] = f["roi_types"][:]

        # Load sync for behaviour
        sync_bytes = download_s3_bytes(DERIVATIVES_BUCKET, f"sync/{sub}/{ses}/sync.h5")
        if sync_bytes:
            with h5py.File(io.BytesIO(sync_bytes), "r") as f:
                for key in ("speed_cm_s", "ahv_deg_s", "hd_deg", "light_on", "active", "bad_behav"):
                    if key in f:
                        result[key] = f[key][:]
        return result

    with st.spinner("Loading data..."):
        data = _load(sub, ses)

    if data is None:
        st.warning("No calcium data found.")
        st.stop()

    dff = data["dff"]
    n_rois, n_frames = dff.shape
    fps = data["fps"]
    time_s = np.arange(n_frames) / fps
    roi_types = data.get("roi_types", np.zeros(n_rois, dtype=np.uint8))

    # ROI filter
    roi_filter = st.radio("ROIs", ["Soma only", "All ROIs"], horizontal=True, key="rmap_rois")
    if roi_filter == "Soma only":
        mask = roi_types == 0
        dff_filt = dff[mask]
        rt_filt = roi_types[mask]
    else:
        dff_filt = dff
        rt_filt = roi_types

    if dff_filt.shape[0] < 3:
        st.warning("Not enough ROIs for Rastermap.")
        st.stop()

    st.markdown(f"**{sub}/{ses}** — {celltype} — {dff_filt.shape[0]} ROIs, {n_frames/fps:.0f}s")

    # Compute Rastermap
    from hm2p.analysis.rastermap_analysis import (
        compute_rastermap,
        compute_superneurons,
        superneuron_behaviour_correlations,
    )

    n_clusters = st.slider("Clusters", 10, 200, min(100, dff_filt.shape[0]), 10, key="rmap_nc")

    with st.spinner("Computing Rastermap sorting..."):
        rmap = compute_rastermap(dff_filt, n_clusters=n_clusters)

    isort = rmap["isort"]

    tabs = st.tabs(["Sorted Raster", "Superneurons", "Conditions"])

    # ── Tab 1: Sorted raster ──
    with tabs[0]:
        st.subheader("Sorted Raster Plot")

        sorted_dff = dff_filt[isort]
        ds = max(1, n_frames // 3000)

        # Clip for contrast
        vmax = np.nanpercentile(sorted_dff, 98)
        clipped = np.clip(sorted_dff[:, ::ds], 0, vmax)

        n_rows = 2 if "speed_cm_s" in data else 1
        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                            row_heights=[0.2, 0.8] if n_rows == 2 else [1.0],
                            vertical_spacing=0.03)

        if "speed_cm_s" in data:
            speed = data["speed_cm_s"][:n_frames]
            fig.add_trace(go.Scatter(
                x=time_s[::ds], y=speed[::ds],
                mode="lines", line=dict(color="orange", width=0.8), name="Speed",
            ), row=1, col=1)
            if "light_on" in data:
                light = data["light_on"][:n_frames].astype(bool)
                dark_starts, dark_ends = [], []
                in_dark = not light[0]
                if in_dark:
                    dark_starts.append(time_s[0])
                for i in range(1, n_frames):
                    if not light[i] and light[i-1]:
                        dark_starts.append(time_s[i])
                    elif light[i] and not light[i-1]:
                        dark_ends.append(time_s[i])
                if len(dark_starts) > len(dark_ends):
                    dark_ends.append(time_s[-1])
                for ds_t, de_t in zip(dark_starts, dark_ends):
                    for r in range(1, n_rows + 1):
                        fig.add_vrect(x0=ds_t, x1=de_t, fillcolor="rgba(50,50,50,0.12)",
                                      layer="below", line_width=0, row=r, col=1)

        fig.add_trace(go.Heatmap(
            z=clipped, x=time_s[::ds],
            colorscale="Hot", showscale=True,
            colorbar=dict(title="dF/F0", len=0.4),
        ), row=n_rows, col=1)

        fig.update_layout(
            height=500, margin=dict(l=50, r=20, t=20, b=40),
        )
        fig.update_xaxes(title_text="Time (s)", row=n_rows, col=1)
        fig.update_yaxes(title_text="Neuron (sorted)", row=n_rows, col=1)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Superneurons ──
    with tabs[1]:
        st.subheader("Superneuron Behaviour Correlations")

        bin_size = st.slider("Neurons per superneuron", 3, 30, 10, 1, key="rmap_bin")
        superneurons = compute_superneurons(dff_filt, isort, bin_size=bin_size)

        hd = data.get("hd_deg")
        speed = data.get("speed_cm_s")
        light = data.get("light_on")

        corrs = superneuron_behaviour_correlations(superneurons, hd, speed, light)

        import plotly.express as px

        c1, c2, c3 = st.columns(3)

        if "speed_corr" in corrs:
            with c1:
                fig = px.bar(y=corrs["speed_corr"], title="Speed correlation",
                             labels={"y": "Spearman ρ", "x": "Superneuron"})
                fig.update_layout(height=250, margin=dict(l=40, r=20, t=40, b=30))
                st.plotly_chart(fig, use_container_width=True)

        if "hd_corr" in corrs:
            with c2:
                fig = px.bar(y=corrs["hd_corr"], title="HD correlation",
                             labels={"y": "|ρ| (sin/cos)", "x": "Superneuron"})
                fig.update_layout(height=250, margin=dict(l=40, r=20, t=40, b=30))
                st.plotly_chart(fig, use_container_width=True)

        if "light_mod" in corrs:
            with c3:
                fig = px.bar(y=corrs["light_mod"], title="Light modulation",
                             labels={"y": "Mod index", "x": "Superneuron"})
                fig.update_layout(height=250, margin=dict(l=40, r=20, t=40, b=30))
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Condition comparison ──
    with tabs[2]:
        st.subheader("Condition-Specific Rastermap")

        if "light_on" not in data or "speed_cm_s" not in data:
            st.warning("No behavioural data for condition comparison.")
        else:
            condition = st.radio("Compare", ["Light vs Dark", "Moving vs Stationary"],
                                 horizontal=True, key="rmap_cond")

            light = data["light_on"][:n_frames].astype(bool)
            speed = data["speed_cm_s"][:n_frames]
            bad = data.get("bad_behav", np.zeros(n_frames, dtype=bool))[:n_frames].astype(bool)
            valid = ~bad

            if condition == "Light vs Dark":
                mask_a = valid & light
                mask_b = valid & ~light
                label_a, label_b = "Light", "Dark"
            else:
                mask_a = valid & (speed >= 2.5)
                mask_b = valid & (speed < 2.5)
                label_a, label_b = "Moving", "Stationary"

            # Use same sorting for both conditions
            sorted_dff = dff_filt[isort]
            ds2 = max(1, n_frames // 2000)

            # Mean activity per sorted neuron in each condition
            mean_a = np.nanmean(sorted_dff[:, mask_a], axis=1) if mask_a.any() else np.zeros(sorted_dff.shape[0])
            mean_b = np.nanmean(sorted_dff[:, mask_b], axis=1) if mask_b.any() else np.zeros(sorted_dff.shape[0])

            fig = make_subplots(rows=1, cols=2, subplot_titles=[label_a, label_b])
            fig.add_trace(go.Bar(y=mean_a, name=label_a, marker_color="orange"), row=1, col=1)
            fig.add_trace(go.Bar(y=mean_b, name=label_b, marker_color="gray"), row=1, col=2)
            fig.update_layout(height=400, showlegend=False,
                              margin=dict(l=40, r=20, t=40, b=40))
            fig.update_yaxes(title_text="Mean dF/F0", row=1, col=1)
            fig.update_xaxes(title_text="Neuron (sorted)", row=1, col=1)
            fig.update_xaxes(title_text="Neuron (sorted)", row=1, col=2)
            st.plotly_chart(fig, use_container_width=True)

            # Difference
            diff = mean_a - mean_b
            fig_diff = go.Figure(go.Bar(y=diff, marker_color=np.where(diff > 0, "orange", "steelblue")))
            fig_diff.update_layout(height=250, title=f"Difference ({label_a} - {label_b})",
                                   yaxis_title="Δ mean dF/F0",
                                   margin=dict(l=40, r=20, t=40, b=30))
            st.plotly_chart(fig_diff, use_container_width=True)

    with st.expander("Methods & References"):
        st.markdown(
            "**Rastermap:** Stringer C et al. 2025. \"Rastermap: a discovery method for "
            "neural population recordings.\" Nature Neuroscience 28:201-212. "
            "doi:10.1038/s41593-024-01783-4\n\n"
            "https://github.com/MouseLand/rastermap\n\n"
            "Neurons are sorted along a 1D axis by optimizing an asymmetric similarity "
            "matrix (cross-correlation at non-negative time lags) to match a target "
            "combining power-law and sequential structure. Superneurons average groups "
            "of nearby neurons in the sorting for denoised visualization."
        )


try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is not None:
        _page()
except ImportError:
    _page()
