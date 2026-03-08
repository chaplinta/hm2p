"""Trace Comparison — overlay multiple ROI traces for visual correlation analysis."""

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
    load_animals,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.trace_compare")

st.title("Trace Comparison")
st.caption("Overlay traces from multiple ROIs to visually inspect correlated activity.")

# --- Session selector ---
experiments = load_experiments()
animals = load_animals()
animal_map = {a["animal_id"]: a for a in animals}

exp_ids = [e["exp_id"] for e in experiments]
selected = st.selectbox(
    "Session",
    exp_ids,
    format_func=lambda x: f"{x} ({animal_map.get(x.split('_')[-1], {}).get('celltype', '?')})",
    key="tc_session",
)

if not selected:
    st.stop()

sub, ses = parse_session_id(selected)


@st.cache_data(ttl=300)
def load_trace_data(sub: str, ses: str) -> dict | None:
    """Load ca.h5 for trace comparison."""
    import h5py

    data = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
    if data is None:
        return None

    f = h5py.File(io.BytesIO(data), "r")
    result = {
        "dff": f["dff"][:],
        "fps": float(f.attrs.get("fps_imaging", 9.8)),
    }
    if "event_masks" in f:
        result["event_masks"] = f["event_masks"][:]
    if "spks" in f:
        result["spks"] = f["spks"][:]
    f.close()
    return result


with st.spinner("Loading traces..."):
    data = load_trace_data(sub, ses)

if data is None:
    st.warning("No calcium data found.")
    st.stop()

dff = data["dff"]
n_rois, n_frames = dff.shape
fps = data["fps"]
time_s = np.arange(n_frames) / fps
duration_s = n_frames / fps

# Compute SNR
snrs = []
for i in range(n_rois):
    trace = dff[i]
    baseline_std = np.std(trace[trace < np.percentile(trace, 50)])
    peak = np.percentile(trace, 95)
    snrs.append(peak / baseline_std if baseline_std > 0 else 0)
snrs = np.array(snrs)

# --- Controls ---
col1, col2, col3 = st.columns(3)

with col1:
    display_mode = st.selectbox("Display", ["Overlay", "Stacked", "Normalized"], key="tc_mode")

with col2:
    signal = st.selectbox("Signal", ["dff"] + (["deconv"] if "spks" in data else []) + (["events"] if "event_masks" in data else []), key="tc_signal")

with col3:
    show_events = st.checkbox("Highlight events", value=True, key="tc_events")

# ROI selection
st.markdown("**Select ROIs to compare:**")
col1, col2 = st.columns([1, 3])

with col1:
    min_snr = st.slider("Min SNR filter", 0.0, 15.0, 0.0, 0.5, key="tc_snr")
    good_rois = np.where(snrs >= min_snr)[0]

    # Quick select options
    select_mode = st.radio("Selection mode", ["Manual", "Top N by SNR", "Most correlated pair"], key="tc_select_mode")

    if select_mode == "Manual":
        selected_rois = st.multiselect(
            "ROIs",
            good_rois.tolist(),
            default=good_rois[:3].tolist() if len(good_rois) >= 3 else good_rois.tolist(),
            format_func=lambda i: f"ROI {i} (SNR={snrs[i]:.1f})",
            key="tc_rois",
        )
    elif select_mode == "Top N by SNR":
        n_top = st.slider("N", 2, min(10, len(good_rois)), 3, key="tc_top_n")
        top_idx = np.argsort(snrs[good_rois])[::-1][:n_top]
        selected_rois = good_rois[top_idx].tolist()
        st.markdown(f"Selected: {', '.join(f'ROI {r}' for r in selected_rois)}")
    else:
        # Find most correlated pair
        if len(good_rois) >= 2:
            corr_sub = np.corrcoef(dff[good_rois])
            np.fill_diagonal(corr_sub, -1)
            max_idx = np.unravel_index(np.argmax(corr_sub), corr_sub.shape)
            selected_rois = [good_rois[max_idx[0]], good_rois[max_idx[1]]]
            r_val = corr_sub[max_idx]
            st.markdown(f"Most correlated: ROI {selected_rois[0]} & ROI {selected_rois[1]} (r={r_val:.3f})")
        else:
            selected_rois = good_rois.tolist()

with col2:
    # Time range
    t_range = st.slider(
        "Time range (s)",
        0.0, float(duration_s),
        (0.0, min(60.0, float(duration_s))),
        key="tc_trange",
    )

if not selected_rois:
    st.info("Select at least one ROI.")
    st.stop()

# --- Get signal arrays ---
import plotly.graph_objects as go

frame_start = int(t_range[0] * fps)
frame_end = int(t_range[1] * fps)
t_window = time_s[frame_start:frame_end]

colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

fig = go.Figure()

for plot_idx, roi_idx in enumerate(selected_rois):
    color = colors[plot_idx % len(colors)]

    # Get signal
    if signal == "dff":
        trace = dff[roi_idx, frame_start:frame_end]
    elif signal == "deconv":
        trace = data["spks"][roi_idx, frame_start:frame_end]
    else:
        trace = data["event_masks"][roi_idx, frame_start:frame_end].astype(float)

    # Apply display mode
    if display_mode == "Stacked":
        offset = plot_idx * (np.nanmax(trace) - np.nanmin(trace) + 0.1) if np.ptp(trace) > 0 else plot_idx * 0.5
        trace_display = trace + offset
    elif display_mode == "Normalized":
        if trace.std() > 0:
            trace_display = (trace - trace.mean()) / trace.std()
        else:
            trace_display = trace - trace.mean()
    else:
        trace_display = trace

    fig.add_trace(go.Scatter(
        x=t_window,
        y=trace_display,
        mode="lines",
        name=f"ROI {roi_idx} (SNR={snrs[roi_idx]:.1f})",
        line=dict(color=color, width=1),
    ))

    # Event highlight
    if show_events and "event_masks" in data:
        em = data["event_masks"][roi_idx, frame_start:frame_end].astype(bool)
        event_trace = trace_display.copy()
        event_trace[~em] = np.nan
        fig.add_trace(go.Scatter(
            x=t_window,
            y=event_trace,
            mode="lines",
            name=f"ROI {roi_idx} events",
            line=dict(color=color, width=2.5),
            showlegend=False,
        ))

fig.update_layout(
    height=400,
    title=f"{display_mode} view — {signal} — {len(selected_rois)} ROIs",
    xaxis_title="Time (s)",
    yaxis_title=signal if display_mode != "Normalized" else "Z-score",
    margin=dict(l=50, r=20, t=40, b=30),
)
st.plotly_chart(fig, use_container_width=True)

# --- Pairwise correlations ---
if len(selected_rois) >= 2:
    st.subheader("Pairwise Correlations")

    selected_signals = []
    for roi_idx in selected_rois:
        if signal == "dff":
            selected_signals.append(dff[roi_idx])
        elif signal == "deconv":
            selected_signals.append(data["spks"][roi_idx])
        else:
            selected_signals.append(data["event_masks"][roi_idx].astype(float))

    corr = np.corrcoef(selected_signals)

    # Display as table
    import pandas as pd
    labels = [f"ROI {r}" for r in selected_rois]
    corr_df = pd.DataFrame(corr, index=labels, columns=labels).round(3)
    st.dataframe(corr_df, use_container_width=True)

    # Cross-correlation lag analysis
    if len(selected_rois) == 2:
        st.subheader("Cross-Correlation (lag analysis)")

        s1 = selected_signals[0]
        s2 = selected_signals[1]

        # Z-score
        s1 = (s1 - s1.mean()) / (s1.std() + 1e-10)
        s2 = (s2 - s2.mean()) / (s2.std() + 1e-10)

        max_lag = int(5 * fps)  # +/- 5 seconds
        lags = np.arange(-max_lag, max_lag + 1)
        xcorr = np.correlate(s1, s2, mode="full")
        xcorr = xcorr / len(s1)  # Normalize

        # Extract relevant lags
        center = len(xcorr) // 2
        start = center - max_lag
        end = center + max_lag + 1
        xcorr_window = xcorr[start:end]
        lag_time = lags / fps

        fig_xcorr = go.Figure()
        fig_xcorr.add_trace(go.Scatter(
            x=lag_time,
            y=xcorr_window,
            mode="lines",
            line=dict(color="steelblue"),
        ))
        fig_xcorr.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

        # Mark peak
        peak_idx = np.argmax(xcorr_window)
        peak_lag = lag_time[peak_idx]
        peak_val = xcorr_window[peak_idx]
        fig_xcorr.add_trace(go.Scatter(
            x=[peak_lag],
            y=[peak_val],
            mode="markers",
            marker=dict(color="red", size=10),
            name=f"Peak: lag={peak_lag:.2f}s, r={peak_val:.3f}",
        ))

        fig_xcorr.update_layout(
            height=300,
            title=f"Cross-correlation: ROI {selected_rois[0]} vs ROI {selected_rois[1]}",
            xaxis_title="Lag (s, positive = ROI2 leads)",
            yaxis_title="Cross-correlation",
            margin=dict(l=50, r=20, t=40, b=30),
        )
        st.plotly_chart(fig_xcorr, use_container_width=True)
