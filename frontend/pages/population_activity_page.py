"""Population Activity — ROI-free analysis of imaging data.

PCA of population activity, frame-to-frame correlation, movement
regression, and CASCADE vs dF/F comparison. No ROI detection required.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    download_s3_bytes,
    load_animals,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.population_activity")


def _page() -> None:
    st.title("Population Activity")
    st.caption(
        "Population-level neural signals without ROI detection: PCA of activity, "
        "frame-to-frame correlation, movement regression (Zagha et al. 2022), "
        "and CASCADE vs dF/F comparison (Rupprecht et al. 2021)."
    )

    experiments = load_experiments()
    animals = load_animals()
    animal_map = {a["animal_id"]: a for a in animals}

    exp_ids = [e["exp_id"] for e in experiments]
    selected = st.selectbox(
        "Session", exp_ids,
        format_func=lambda x: f"{x} ({animal_map.get(x.split('_')[-1], {}).get('celltype', '?')})",
        key="popact_session",
    )
    if not selected:
        st.stop()

    sub, ses = parse_session_id(selected)

    @st.cache_data(ttl=300)
    def _load(sub: str, ses: str) -> dict | None:
        import h5py
        # Load ca.h5
        ca_data = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
        if ca_data is None:
            return None
        result = {}
        with h5py.File(io.BytesIO(ca_data), "r") as f:
            result["dff"] = f["dff"][:]
            result["fps"] = float(f.attrs.get("fps_imaging", 9.8))
            for key in ("spikes", "deconv_norm", "event_masks", "event_masks_sd", "roi_types"):
                if key in f:
                    result[key] = f[key][:]
        # Load sync.h5 for behaviour (optional)
        sync_data = download_s3_bytes(DERIVATIVES_BUCKET, f"sync/{sub}/{ses}/sync.h5")
        if sync_data:
            with h5py.File(io.BytesIO(sync_data), "r") as f:
                for key in ("speed_cm_s", "ahv_deg_s", "hd_deg", "light_on", "active"):
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
    has_spikes = "spikes" in data
    has_behaviour = "speed_cm_s" in data

    st.markdown(f"**{n_rois} ROIs**, {n_frames / fps:.0f}s, {fps:.1f} Hz")

    # ── Tabs ──
    tabs = st.tabs(["PCA", "Frame Correlation", "Movement Regression", "CASCADE vs dF/F"])

    # ── Tab 1: PCA ──
    with tabs[0]:
        st.subheader("Population PCA")
        from hm2p.calcium.population import compute_population_signals

        n_pcs = st.slider("Number of PCs", 3, 20, 10, key="popact_npcs")
        pop = compute_population_signals(dff, n_components=n_pcs)

        # Variance explained
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(len(pop["explained_variance_ratio"]))],
            y=pop["explained_variance_ratio"] * 100,
        ))
        fig_var.update_layout(
            title="Variance explained per PC",
            yaxis_title="% variance", height=250,
            margin=dict(l=50, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_var, use_container_width=True)

        cumvar = np.cumsum(pop["explained_variance_ratio"]) * 100
        st.caption(f"Top {n_pcs} PCs explain {cumvar[-1]:.1f}% of total variance")

        # PC time courses
        n_show = min(5, n_pcs)
        fig_pcs = make_subplots(rows=n_show + 1, cols=1, shared_xaxes=True,
                                vertical_spacing=0.02,
                                subplot_titles=["Mean activity"] + [f"PC{i+1} ({pop['explained_variance_ratio'][i]*100:.1f}%)" for i in range(n_show)])

        ds = max(1, n_frames // 3000)
        fig_pcs.add_trace(go.Scatter(
            x=time_s[::ds], y=pop["mean_activity"][::ds],
            mode="lines", line=dict(color="black", width=0.8),
            showlegend=False,
        ), row=1, col=1)

        colors = px.colors.qualitative.Set2
        for i in range(n_show):
            fig_pcs.add_trace(go.Scatter(
                x=time_s[::ds], y=pop["components"][i][::ds],
                mode="lines", line=dict(color=colors[i % len(colors)], width=0.8),
                showlegend=False,
            ), row=i + 2, col=1)

        fig_pcs.update_layout(height=150 * (n_show + 1), margin=dict(l=50, r=20, t=30, b=40))
        fig_pcs.update_xaxes(title_text="Time (s)", row=n_show + 1, col=1)
        st.plotly_chart(fig_pcs, use_container_width=True)

    # ── Tab 2: Frame correlation ──
    with tabs[1]:
        st.subheader("Frame-to-Frame Correlation")
        st.caption(
            "Pearson correlation between population vectors of adjacent frames. "
            "Drops indicate state transitions or large movements."
        )
        from hm2p.calcium.population import frame_correlation

        corrs = frame_correlation(dff, lag=1)
        ds = max(1, len(corrs) // 3000)
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=time_s[:len(corrs)][::ds], y=corrs[::ds],
            mode="lines", line=dict(color="steelblue", width=0.8),
        ))
        fig_fc.update_layout(
            height=250, yaxis_title="Frame correlation",
            xaxis_title="Time (s)",
            margin=dict(l=50, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("Mean correlation", f"{np.nanmean(corrs):.3f}")
        c2.metric("Std", f"{np.nanstd(corrs):.3f}")

    # ── Tab 3: Movement regression ──
    with tabs[2]:
        st.subheader("Movement Regression")
        if not has_behaviour:
            st.warning("No behavioural data (sync.h5) available for this session.")
        else:
            st.caption(
                "How much neural variance is explained by movement variables? "
                "Following Zagha et al. 2022: include speed, AHV, and acceleration "
                "as regressors, then check R² per ROI."
            )
            from hm2p.calcium.population import regress_movement

            speed = data["speed_cm_s"][:n_frames]
            ahv = data.get("ahv_deg_s")
            if ahv is not None:
                ahv = ahv[:n_frames]
            # Compute acceleration from speed
            accel = np.gradient(speed) * fps if len(speed) > 1 else None

            with st.spinner("Regressing movement..."):
                reg = regress_movement(dff, speed, ahv=ahv, acceleration=accel)

            c1, c2 = st.columns(2)
            c1.metric("Mean R² (movement model)", f"{reg['mean_r_squared']:.3f}")
            c2.metric("Mean speed corr (ρ)", f"{np.nanmean(reg['speed_corr']):.3f}")

            fig_r2 = px.histogram(
                x=reg["r_squared"][np.isfinite(reg["r_squared"])],
                nbins=30, title="R² distribution (movement → neural)",
                labels={"x": "R²", "y": "ROI count"},
            )
            fig_r2.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_r2, use_container_width=True)

            if reg["ahv_corr"] is not None:
                fig_corr = make_subplots(rows=1, cols=2, subplot_titles=["Speed corr", "AHV corr"])
                fig_corr.add_trace(go.Histogram(x=reg["speed_corr"][np.isfinite(reg["speed_corr"])], nbinsx=30), row=1, col=1)
                fig_corr.add_trace(go.Histogram(x=reg["ahv_corr"][np.isfinite(reg["ahv_corr"])], nbinsx=30), row=1, col=2)
                fig_corr.update_layout(height=250, showlegend=False, margin=dict(l=40, r=20, t=40, b=40))
                st.plotly_chart(fig_corr, use_container_width=True)

            # Regress PCs against movement
            st.markdown("**PCA components vs movement:**")
            pop = compute_population_signals(dff, n_components=10)
            reg_pcs = regress_movement(pop["components"], speed, ahv=ahv, acceleration=accel)

            pc_df = pd.DataFrame({
                "PC": [f"PC{i+1}" for i in range(len(reg_pcs["r_squared"]))],
                "R²": reg_pcs["r_squared"],
                "Speed ρ": reg_pcs["speed_corr"],
            })
            st.dataframe(pc_df, use_container_width=True)

    # ── Tab 4: CASCADE vs dF/F ──
    with tabs[3]:
        st.subheader("CASCADE vs dF/F Comparison")
        if not has_spikes:
            st.warning("CASCADE spikes not available. Run CASCADE first.")
        else:
            from hm2p.calcium.population import compare_spikes_to_fluorescence

            deconv_norm = data.get("deconv_norm")
            with st.spinner("Computing comparisons..."):
                comp = compare_spikes_to_fluorescence(dff, data["spikes"], deconv_norm, fps)

            c1, c2, c3 = st.columns(3)
            c1.metric("Mean ρ (spikes vs dF/F)", f"{comp['mean_corr_dff']:.3f}")
            c2.metric("Mean ρ (spikes vs deconv)", f"{comp['mean_corr_deconv']:.3f}")
            c3.metric("Mean temporal lag", f"{comp['mean_lag_s']*1000:.0f} ms")

            # Correlation distributions
            fig_comp = make_subplots(rows=1, cols=2,
                                     subplot_titles=["Spikes vs dF/F", "Spikes vs Deconv"])
            valid_dff = comp["corr_dff_spikes"][np.isfinite(comp["corr_dff_spikes"])]
            valid_dc = comp["corr_deconv_spikes"][np.isfinite(comp["corr_deconv_spikes"])]
            fig_comp.add_trace(go.Histogram(x=valid_dff, nbinsx=30, marker_color="#d62728"), row=1, col=1)
            fig_comp.add_trace(go.Histogram(x=valid_dc, nbinsx=30, marker_color="steelblue"), row=1, col=2)
            fig_comp.update_layout(height=250, showlegend=False, margin=dict(l=40, r=20, t=40, b=40))
            fig_comp.update_xaxes(title_text="Spearman ρ", row=1, col=1)
            fig_comp.update_xaxes(title_text="Spearman ρ", row=1, col=2)
            st.plotly_chart(fig_comp, use_container_width=True)

            # Temporal lag distribution
            valid_lag = comp["peak_lag_seconds"][np.isfinite(comp["peak_lag_seconds"])]
            fig_lag = px.histogram(x=valid_lag * 1000, nbins=30,
                                   title="Temporal lag: spikes relative to dF/F",
                                   labels={"x": "Lag (ms)", "y": "ROI count"})
            fig_lag.add_vline(x=0, line_dash="dash", line_color="red")
            fig_lag.update_layout(height=250, margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_lag, use_container_width=True)

            # Event-triggered average
            if comp["event_triggered_avg"] is not None:
                st.markdown("**Event-triggered average dF/F aligned to CASCADE spike peaks:**")
                fig_eta = go.Figure()
                fig_eta.add_trace(go.Scatter(
                    x=comp["event_triggered_time"],
                    y=comp["event_triggered_avg"],
                    mode="lines", line=dict(color="black", width=2),
                ))
                fig_eta.add_vline(x=0, line_dash="dash", line_color="red",
                                  annotation_text="spike peak")
                fig_eta.update_layout(
                    height=300, xaxis_title="Time from spike peak (s)",
                    yaxis_title="Mean dF/F",
                    margin=dict(l=50, r=20, t=20, b=40),
                )
                st.plotly_chart(fig_eta, use_container_width=True)

    # Methods
    with st.expander("Methods & References"):
        st.markdown(
            "**PCA:** Principal components of the (ROIs × frames) fluorescence matrix. "
            "Captures dominant co-activation patterns without requiring single-cell identity.\n\n"
            "**Frame correlation:** Pearson r between population vectors of consecutive frames. "
            "Tracks brain state stability (Stringer et al. 2026).\n\n"
            "**Movement regression:** OLS regression of each ROI's activity against speed, "
            "|AHV|, and acceleration. R² quantifies movement-related variance. "
            "Zagha et al. 2022, J Neurosci. doi:10.1523/JNEUROSCI.1919-21.2021\n\n"
            "**CASCADE comparison:** Spearman correlation, cross-correlation lag, and "
            "event-triggered average between CASCADE spike rates and dF/F. "
            "Rupprecht et al. 2021, Nature Neuroscience. doi:10.1038/s41593-021-00895-5"
        )


try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is not None:
        _page()
except ImportError:
    _page()
