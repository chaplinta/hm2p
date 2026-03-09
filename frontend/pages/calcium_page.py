"""Calcium data viewer — dF/F traces, events, neuropil, per-cell drill-down.

Displays calcium processing results (ca.h5) with interactive trace viewing,
event detection overlays, and per-cell quality metrics.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    download_s3_bytes,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.calcium")

st.title("Calcium Data Viewer")

# --- Session selector ---
experiments = load_experiments()
exp_ids = [e["exp_id"] for e in experiments]

default_idx = 0
if "selected_exp_id" in st.session_state:
    sel = st.session_state["selected_exp_id"]
    if sel in exp_ids:
        default_idx = exp_ids.index(sel)

selected_exp = st.selectbox("Session", exp_ids, index=default_idx, key="ca_exp")
sub, ses = parse_session_id(selected_exp)
st.caption(f"`{sub}/{ses}`")


@st.cache_data(ttl=300)
def load_ca_h5(bucket: str, key: str) -> dict | None:
    """Download and parse ca.h5 from S3."""
    import h5py

    data = download_s3_bytes(bucket, key)
    if data is None:
        return None
    try:
        f = h5py.File(io.BytesIO(data), "r")
        result = {}
        for k in f.keys():
            result[k] = f[k][:]
        for k, v in f.attrs.items():
            result[f"_attr_{k}"] = v
        f.close()
        return result
    except Exception:
        return None


ca_key = f"calcium/{sub}/{ses}/ca.h5"
ca = load_ca_h5(DERIVATIVES_BUCKET, ca_key)

if ca is None:
    st.warning(f"No ca.h5 found at `s3://{DERIVATIVES_BUCKET}/{ca_key}`")
    st.stop()

dff = ca.get("dff")
if dff is None:
    st.error("ca.h5 has no 'dff' dataset")
    st.stop()

n_rois, n_frames = dff.shape
fps = float(ca.get("_attr_fps_imaging", 9.8))
event_masks = ca.get("event_masks")
noise_probs = ca.get("noise_probs")
spks = ca.get("spks")

# --- Summary metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("ROIs", n_rois)
col2.metric("Frames", n_frames)
col3.metric("Duration", f"{n_frames / fps:.0f}s")
col4.metric("FPS", f"{fps:.1f} Hz")

# --- Tabs ---
tab_overview, tab_traces, tab_events, tab_cell = st.tabs([
    "Overview", "Trace Viewer", "Event Detection", "Cell Drill-down",
])


# --- Tab 1: Overview ---
with tab_overview:
    st.subheader("Population Overview")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Mean dF/F per ROI
    mean_dff = np.nanmean(dff, axis=1)
    max_dff = np.nanmax(dff, axis=1)
    std_dff = np.nanstd(dff, axis=1)

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "Mean dF/F per ROI", "Max dF/F per ROI",
        "Std dF/F per ROI", "Active fraction per ROI",
    ])

    fig.add_trace(go.Bar(x=list(range(n_rois)), y=mean_dff, name="Mean"), row=1, col=1)
    fig.add_trace(go.Bar(x=list(range(n_rois)), y=max_dff, name="Max"), row=1, col=2)
    fig.add_trace(go.Bar(x=list(range(n_rois)), y=std_dff, name="Std"), row=2, col=1)

    if event_masks is not None:
        active_frac = event_masks.mean(axis=1)
        fig.add_trace(
            go.Bar(x=list(range(n_rois)), y=active_frac, name="Active frac"),
            row=2, col=2,
        )

    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap of dF/F (downsampled for display)
    st.subheader("dF/F Heatmap (all ROIs)")
    # Downsample to max 1000 columns
    ds_factor = max(1, n_frames // 1000)
    dff_ds = dff[:, ::ds_factor]
    time_ds = np.arange(dff_ds.shape[1]) * ds_factor / fps

    fig_heat = go.Figure(data=go.Heatmap(
        z=dff_ds,
        x=time_ds,
        colorscale="RdBu_r",
        zmin=-np.percentile(np.abs(dff_ds), 95),
        zmax=np.percentile(np.abs(dff_ds), 95),
        colorbar=dict(title="dF/F"),
    ))
    fig_heat.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="ROI",
        height=400,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Correlation matrix
    if n_rois > 1 and n_rois <= 100:
        st.subheader("Pairwise Correlation Matrix")
        corr = np.corrcoef(dff)
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            colorbar=dict(title="r"),
        ))
        fig_corr.update_layout(
            height=400, width=500,
            xaxis_title="ROI", yaxis_title="ROI",
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# --- Tab 2: Trace viewer ---
with tab_traces:
    st.subheader("Interactive Trace Viewer")

    col1, col2 = st.columns([1, 3])
    with col1:
        n_show = st.slider("ROIs to show", 1, min(20, n_rois), min(5, n_rois), key="trace_n")
        start_roi = st.slider("Start ROI", 0, max(0, n_rois - n_show), 0, key="trace_start")
        show_events_overlay = st.checkbox("Show events", value=True, key="trace_events")
        show_deconv = st.checkbox("Show deconvolved", value=False, key="trace_deconv")
        time_range = st.slider(
            "Time range (s)",
            0.0, float(n_frames / fps),
            (0.0, min(60.0, float(n_frames / fps))),
            key="trace_time",
        )

    frame_start = int(time_range[0] * fps)
    frame_end = int(time_range[1] * fps)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rois = list(range(start_roi, start_roi + n_show))
    fig = make_subplots(
        rows=len(rois), cols=1, shared_xaxes=True,
        subplot_titles=[f"ROI {r}" for r in rois],
        vertical_spacing=0.02,
    )

    time_axis = np.arange(frame_start, frame_end) / fps

    for idx, roi in enumerate(rois, 1):
        trace = dff[roi, frame_start:frame_end]
        fig.add_trace(
            go.Scatter(x=time_axis, y=trace, mode="lines",
                       line=dict(width=0.8, color="black"), name=f"ROI {roi}"),
            row=idx, col=1,
        )

        if show_events_overlay and event_masks is not None:
            em = event_masks[roi, frame_start:frame_end].astype(bool)
            if em.any():
                event_trace = trace.copy()
                event_trace[~em] = np.nan
                fig.add_trace(
                    go.Scatter(x=time_axis, y=event_trace, mode="lines",
                               line=dict(width=1.5, color="red"), name="Events",
                               showlegend=(idx == 1)),
                    row=idx, col=1,
                )

        if show_deconv and spks is not None:
            sp = spks[roi, frame_start:frame_end]
            sp_scaled = sp / max(sp.max(), 1) * max(trace.max(), 0.1)
            fig.add_trace(
                go.Scatter(x=time_axis, y=sp_scaled, mode="lines",
                           line=dict(width=0.5, color="blue", dash="dot"),
                           name="Deconv", showlegend=(idx == 1)),
                row=idx, col=1,
            )

    fig.update_layout(
        height=200 * len(rois),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(title_text="Time (s)", row=len(rois), col=1)
    st.plotly_chart(fig, use_container_width=True)


# --- Tab 3: Event Detection ---
with tab_events:
    st.subheader("Event Detection Summary")

    if event_masks is None:
        st.info("No event_masks in ca.h5")
    else:
        import plotly.graph_objects as go

        # Per-ROI event statistics
        event_counts = []
        event_rates = []
        mean_amplitudes = []

        for i in range(n_rois):
            em = event_masks[i].astype(bool)
            # Count onsets
            onsets = np.flatnonzero(em[1:] & ~em[:-1]) + 1
            if em[0]:
                onsets = np.concatenate([[0], onsets])
            n_events = len(onsets)
            event_counts.append(n_events)
            event_rates.append(n_events / (n_frames / fps) * 60)  # events/min

            # Mean amplitude during events
            if em.any():
                mean_amplitudes.append(float(np.mean(dff[i, em])))
            else:
                mean_amplitudes.append(0.0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total events", sum(event_counts))
        col2.metric("Mean rate", f"{np.mean(event_rates):.1f} events/min")
        col3.metric("Active ROIs", sum(1 for c in event_counts if c > 0))

        fig = make_subplots(rows=1, cols=3, subplot_titles=[
            "Events per ROI", "Event rate (events/min)", "Mean event amplitude",
        ])
        fig.add_trace(go.Bar(x=list(range(n_rois)), y=event_counts), row=1, col=1)
        fig.add_trace(go.Bar(x=list(range(n_rois)), y=event_rates), row=1, col=2)
        fig.add_trace(go.Bar(x=list(range(n_rois)), y=mean_amplitudes), row=1, col=3)
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Noise probability viewer
        if noise_probs is not None:
            st.subheader("Noise Probability")
            roi_np = st.selectbox("ROI for noise prob", range(n_rois), key="np_roi")

            ds = max(1, n_frames // 2000)
            time_np = np.arange(0, n_frames, ds) / fps
            np_trace = noise_probs[roi_np, ::ds]
            dff_trace = dff[roi_np, ::ds]

            from plotly.subplots import make_subplots
            fig_np = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   subplot_titles=["dF/F", "Noise probability"])
            fig_np.add_trace(go.Scatter(x=time_np, y=dff_trace, line=dict(width=0.5)), row=1, col=1)
            fig_np.add_trace(go.Scatter(x=time_np, y=np_trace, line=dict(width=0.5, color="orange")), row=2, col=1)
            fig_np.add_hline(y=0.2, line_dash="dash", line_color="red", row=2, col=1,
                             annotation_text="Onset threshold")
            fig_np.add_hline(y=0.7, line_dash="dash", line_color="blue", row=2, col=1,
                             annotation_text="Offset threshold")
            fig_np.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_np, use_container_width=True)


# --- Tab 4: Cell Drill-down ---
with tab_cell:
    st.subheader("Single Cell Analysis")

    roi = st.selectbox("Select ROI", range(n_rois), key="cell_roi")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    trace = dff[roi]
    time_axis = np.arange(n_frames) / fps

    # Downsample for display
    ds = max(1, n_frames // 3000)
    t_ds = time_axis[::ds]
    trace_ds = trace[::ds]

    # Compute basic stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean dF/F", f"{np.nanmean(trace):.4f}")
    col2.metric("Max dF/F", f"{np.nanmax(trace):.3f}")
    col3.metric("Std dF/F", f"{np.nanstd(trace):.4f}")

    if event_masks is not None:
        em = event_masks[roi].astype(bool)
        onsets = np.flatnonzero(em[1:] & ~em[:-1]) + 1
        col4.metric("Events", len(onsets))
    else:
        col4.metric("Events", "N/A")

    # Full trace with events
    n_plots = 2 if spks is not None else 1
    fig = make_subplots(rows=n_plots, cols=1, shared_xaxes=True,
                        subplot_titles=["dF/F + Events"] + (["Deconvolved"] if spks is not None else []))

    fig.add_trace(
        go.Scatter(x=t_ds, y=trace_ds, mode="lines",
                   line=dict(width=0.5, color="black"), name="dF/F"),
        row=1, col=1,
    )

    if event_masks is not None:
        em_ds = event_masks[roi, ::ds].astype(bool)
        event_trace = trace_ds.copy()
        event_trace[~em_ds] = np.nan
        fig.add_trace(
            go.Scatter(x=t_ds, y=event_trace, mode="lines",
                       line=dict(width=1.5, color="red"), name="Events"),
            row=1, col=1,
        )

    if spks is not None:
        sp_ds = spks[roi, ::ds]
        fig.add_trace(
            go.Scatter(x=t_ds, y=sp_ds, mode="lines",
                       line=dict(width=0.5, color="blue"), name="Deconvolved"),
            row=2, col=1,
        )

    fig.update_layout(height=300 * n_plots, showlegend=True)
    fig.update_xaxes(title_text="Time (s)", row=n_plots, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # dF/F histogram
    st.subheader("dF/F Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=trace[~np.isnan(trace)], nbinsx=100, name="dF/F",
        ))
        fig_hist.update_layout(
            xaxis_title="dF/F", yaxis_title="Count",
            height=300, title="dF/F Histogram",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Event amplitude distribution
        if event_masks is not None and em.any():
            amplitudes = []
            onsets_list = np.flatnonzero(em[1:] & ~em[:-1]) + 1
            if em[0]:
                onsets_list = np.concatenate([[0], onsets_list])
            offsets_list = np.flatnonzero(~em[1:] & em[:-1]) + 1
            if em[-1]:
                offsets_list = np.concatenate([offsets_list, [n_frames]])

            for on, off in zip(onsets_list, offsets_list[:len(onsets_list)]):
                amplitudes.append(float(np.max(trace[on:off])))

            fig_amp = go.Figure()
            fig_amp.add_trace(go.Histogram(x=amplitudes, nbinsx=30, name="Amplitude"))
            fig_amp.update_layout(
                xaxis_title="Peak dF/F", yaxis_title="Count",
                height=300, title="Event Amplitude Distribution",
            )
            st.plotly_chart(fig_amp, use_container_width=True)
        else:
            st.info("No events detected for this ROI")
