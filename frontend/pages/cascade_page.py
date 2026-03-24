"""CASCADE Spike Inference — explore calibrated spike rates from ca.h5.

Displays CASCADE spike rate data alongside dF/F0, deconvolved traces,
and event masks. Does not require sync.h5 or behavioural data — works
directly from ca.h5.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    download_s3_bytes,
    load_animals,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.cascade")


def _page() -> None:
    st.title("CASCADE Spike Inference")
    st.caption(
        "Calibrated spike rates (spikes/s) from CASCADE "
        "(Rupprecht et al. 2021, Nature Neuroscience). "
        "Compares CASCADE output with dF/F0, Suite2p deconvolution, "
        "and event detection methods."
    )

    experiments = load_experiments()
    animals = load_animals()
    animal_map = {a["animal_id"]: a for a in animals}

    exp_ids = [e["exp_id"] for e in experiments]
    selected = st.selectbox(
        "Session", exp_ids,
        format_func=lambda x: f"{x} ({animal_map.get(x.split('_')[-1], {}).get('celltype', '?')})",
        key="cascade_session",
    )

    if not selected:
        st.stop()

    sub, ses = parse_session_id(selected)

    @st.cache_data(ttl=300)
    def _load_ca(sub: str, ses: str) -> dict | None:
        import h5py
        data = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
        if data is None:
            return None
        with h5py.File(io.BytesIO(data), "r") as f:
            result = {
                "dff": f["dff"][:],
                "fps": float(f.attrs.get("fps_imaging", 9.8)),
            }
            for key in ("spikes", "deconv", "deconv_norm", "event_masks", "event_masks_sd",
                        "roi_types", "noise_probs"):
                if key in f:
                    result[key] = f[key][:]
        return result

    with st.spinner("Loading ca.h5..."):
        ca = _load_ca(sub, ses)

    if ca is None:
        st.warning("No ca.h5 found for this session.")
        st.stop()

    dff = ca["dff"]
    n_rois, n_frames = dff.shape
    fps = ca["fps"]
    has_spikes = "spikes" in ca
    has_deconv = "deconv" in ca
    time_s = np.arange(n_frames) / fps

    # --- Summary ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROIs", n_rois)
    col2.metric("Duration", f"{n_frames / fps:.0f}s")
    col3.metric("FPS", f"{fps:.1f}")
    col4.metric("CASCADE", "Available" if has_spikes else "Not run")

    if not has_spikes:
        st.warning(
            "CASCADE spike rates not available for this session. "
            "CASCADE needs to be run separately (requires Python 3.8 + TF 2.3). "
            "See scripts/run_cascade.py."
        )
        # Still show other signals
    else:
        spikes = ca["spikes"]

        # --- Population spike rate ---
        st.subheader("Population Spike Rate")

        mean_spike_rate = np.nanmean(spikes, axis=0)
        kernel = np.ones(max(1, int(fps))) / max(1, int(fps))
        smooth_rate = np.convolve(mean_spike_rate, kernel, mode="same")

        fig_pop = go.Figure()
        fig_pop.add_trace(go.Scatter(
            x=time_s, y=smooth_rate,
            mode="lines", name="Mean spike rate",
            line=dict(color="#d62728", width=1),
            fill="tozeroy", fillcolor="rgba(214, 39, 40, 0.15)",
        ))
        fig_pop.update_layout(
            height=250,
            xaxis_title="Time (s)", yaxis_title="Spikes/s (population mean)",
            margin=dict(l=50, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_pop, use_container_width=True)

        # --- Per-ROI spike rate stats ---
        st.subheader("Per-ROI Spike Statistics")

        mean_rates = np.nanmean(spikes, axis=1) * 60  # spikes/min
        max_rates = np.nanmax(spikes, axis=1)
        active_frac = np.mean(spikes > 0, axis=1)

        import plotly.express as px
        import pandas as pd

        roi_df = pd.DataFrame({
            "ROI": np.arange(n_rois),
            "Mean rate (spk/min)": mean_rates,
            "Max rate (spk/s)": max_rates,
            "Active fraction": active_frac,
        })

        if "roi_types" in ca:
            type_labels = {0: "soma", 1: "dendrite", 2: "artefact"}
            roi_df["Type"] = [type_labels.get(int(t), "?") for t in ca["roi_types"]]

        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(
                roi_df, x="Mean rate (spk/min)", nbins=30,
                color="Type" if "Type" in roi_df else None,
                title="Spike rate distribution",
            )
            fig_hist.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            fig_active = px.histogram(
                roi_df, x="Active fraction", nbins=30,
                color="Type" if "Type" in roi_df else None,
                title="Fraction of frames with spikes > 0",
            )
            fig_active.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_active, use_container_width=True)

    # --- ROI browser: compare all signals ---
    st.subheader("ROI Signal Comparison")
    st.caption("Compare dF/F0, CASCADE spikes, Suite2p deconv, and event masks for individual ROIs.")

    roi_idx = st.selectbox("ROI", range(n_rois),
                           format_func=lambda i: f"ROI {i}", key="cascade_roi")

    # Build comparison figure
    signals_available = ["dF/F0"]
    if has_spikes:
        signals_available.append("CASCADE spikes")
    if has_deconv:
        signals_available.append("Deconv (raw)")
    if "deconv_norm" in ca:
        signals_available.append("Deconv (normalized)")

    n_panels = len(signals_available)
    if "event_masks" in ca:
        n_panels += 1
        signals_available.append("Events (V&H)")
    if "event_masks_sd" in ca:
        n_panels += 1
        signals_available.append("Events (SD)")

    fig = make_subplots(
        rows=n_panels, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=signals_available,
    )

    panel = 1

    # dF/F0
    fig.add_trace(go.Scatter(
        x=time_s, y=dff[roi_idx],
        mode="lines", line=dict(color="black", width=0.8), name="dF/F0",
        showlegend=False,
    ), row=panel, col=1)
    panel += 1

    # CASCADE
    if has_spikes:
        fig.add_trace(go.Scatter(
            x=time_s, y=ca["spikes"][roi_idx],
            mode="lines", line=dict(color="#d62728", width=0.8), name="Spikes",
            showlegend=False,
        ), row=panel, col=1)
        panel += 1

    # Deconv raw
    if has_deconv:
        fig.add_trace(go.Scatter(
            x=time_s, y=ca["deconv"][roi_idx],
            mode="lines", line=dict(color="steelblue", width=0.8), name="Deconv",
            showlegend=False,
        ), row=panel, col=1)
        panel += 1

    # Deconv norm
    if "deconv_norm" in ca:
        fig.add_trace(go.Scatter(
            x=time_s, y=ca["deconv_norm"][roi_idx],
            mode="lines", line=dict(color="teal", width=0.8), name="Deconv norm",
            showlegend=False,
        ), row=panel, col=1)
        panel += 1

    # Events V&H
    if "event_masks" in ca:
        em = ca["event_masks"][roi_idx]
        trace_masked = dff[roi_idx].copy()
        trace_masked[em == 0] = np.nan
        fig.add_trace(go.Scatter(
            x=time_s, y=trace_masked,
            mode="lines", line=dict(color="red", width=1), name="V&H events",
            showlegend=False,
        ), row=panel, col=1)
        panel += 1

    # Events SD
    if "event_masks_sd" in ca:
        em_sd = ca["event_masks_sd"][roi_idx]
        trace_sd = dff[roi_idx].copy()
        trace_sd[em_sd == 0] = np.nan
        fig.add_trace(go.Scatter(
            x=time_s, y=trace_sd,
            mode="lines", line=dict(color="orange", width=1), name="SD events",
            showlegend=False,
        ), row=panel, col=1)

    fig.update_layout(
        height=150 * n_panels + 50,
        margin=dict(l=50, r=20, t=30, b=40),
    )
    fig.update_xaxes(title_text="Time (s)", row=n_panels, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # --- Correlation between signals ---
    if has_spikes and has_deconv:
        st.subheader("Signal Correlations")

        from scipy.stats import spearmanr

        sp = ca["spikes"][roi_idx]
        dc = ca["deconv"][roi_idx] if has_deconv else None
        df_trace = dff[roi_idx]

        corrs = {}
        if np.std(sp) > 0 and np.std(df_trace) > 0:
            corrs["CASCADE vs dF/F"] = float(spearmanr(sp, df_trace)[0])
        if dc is not None and np.std(sp) > 0 and np.std(dc) > 0:
            corrs["CASCADE vs Deconv"] = float(spearmanr(sp, dc)[0])
        if dc is not None and np.std(dc) > 0 and np.std(df_trace) > 0:
            corrs["Deconv vs dF/F"] = float(spearmanr(dc, df_trace)[0])

        cols = st.columns(len(corrs))
        for i, (label, r) in enumerate(corrs.items()):
            cols[i].metric(label, f"ρ = {r:.3f}")

    # --- Methods ---
    with st.expander("Methods & References"):
        st.markdown(
            "**CASCADE** (Calibrated spike inference):\n"
            "Rupprecht P et al. 2021. \"A database and deep learning toolbox for "
            "noise-optimized, generalized spike inference from calcium imaging.\" "
            "Nature Neuroscience 24:1324-1337. doi:10.1038/s41593-021-00895-5\n\n"
            "**Model used:** Global_EXC_7.5Hz_smoothing200ms "
            "(closest match to GCaMP7f at ~9.6 Hz imaging rate).\n\n"
            "**Event detection methods:**\n"
            "- V&H: Voigts & Harnett 2020, Neuron. Percentile-based noise model.\n"
            "- SD threshold: Zong et al. 2022, Cell. 2× noise SD with 0.3s minimum duration.\n\n"
            "**Deconvolution:** Suite2p OASIS algorithm (Pachitariu et al. 2016). "
            "Normalized version divides by per-ROI maximum."
        )


# Guard for test imports
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is not None:
        _page()
except ImportError:
    _page()
