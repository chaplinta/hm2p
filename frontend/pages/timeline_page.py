"""Session Timeline — visual overview of light cycles, movement, and neural activity."""

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

log = logging.getLogger("hm2p.frontend.timeline")

st.title("Session Timeline")
st.caption("Temporal overview: light cycles, movement bouts, and population activity.")

# --- Session selector ---
experiments = load_experiments()
animals = load_animals()
animal_map = {a["animal_id"]: a for a in animals}

exp_ids = [e["exp_id"] for e in experiments]
selected = st.selectbox(
    "Session",
    exp_ids,
    format_func=lambda x: f"{x} ({animal_map.get(x.split('_')[-1], {}).get('celltype', '?')})",
    key="timeline_session",
)

if not selected:
    st.stop()

sub, ses = parse_session_id(selected)
animal_id = selected.split("_")[-1]
celltype = animal_map.get(animal_id, {}).get("celltype", "?")
st.markdown(f"**{sub} / {ses}** — celltype: **{celltype}**")


@st.cache_data(ttl=300)
def load_session_data(sub: str, ses: str) -> dict | None:
    """Load ca.h5 data for timeline visualization."""
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
    if "frame_times" in f:
        result["frame_times"] = f["frame_times"][:]

    # SNR for each ROI
    dff = result["dff"]
    snrs = []
    for i in range(dff.shape[0]):
        trace = dff[i]
        baseline_std = np.std(trace[trace < np.percentile(trace, 50)])
        peak = np.percentile(trace, 95)
        snrs.append(peak / baseline_std if baseline_std > 0 else 0)
    result["snr"] = np.array(snrs)

    f.close()
    return result


@st.cache_data(ttl=300)
def load_sync_data(sub: str, ses: str) -> dict | None:
    """Load sync.h5 for behavioural variables aligned to imaging."""
    import h5py

    data = download_s3_bytes(DERIVATIVES_BUCKET, f"sync/{sub}/{ses}/sync.h5")
    if data is None:
        return None

    f = h5py.File(io.BytesIO(data), "r")
    result = {}
    for key in ["hd", "speed", "x", "y", "light_on", "active"]:
        if key in f:
            result[key] = f[key][:]
    f.close()
    return result


@st.cache_data(ttl=300)
def load_timestamps(sub: str, ses: str) -> dict | None:
    """Load timestamps.h5 for light cycle info."""
    import h5py

    data = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
    if data is None:
        return None

    f = h5py.File(io.BytesIO(data), "r")
    result = {}
    if "frame_times" in f:
        result["frame_times"] = f["frame_times"][:]
    f.close()
    return result


with st.spinner("Loading session data..."):
    ca_data = load_session_data(sub, ses)
    sync_data = load_sync_data(sub, ses)

if ca_data is None:
    st.warning("No calcium data found for this session.")
    st.stop()

import plotly.graph_objects as go
from plotly.subplots import make_subplots

dff = ca_data["dff"]
n_rois, n_frames = dff.shape
fps = ca_data["fps"]
time_s = np.arange(n_frames) / fps

# --- Summary metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("ROIs", n_rois)
col2.metric("Duration", f"{n_frames / fps:.0f}s")
col3.metric("Frames", f"{n_frames:,}")
col4.metric("FPS", f"{fps:.1f}")

# --- Build timeline ---
st.subheader("Timeline")

# Determine number of subplot rows based on available data
has_sync = sync_data is not None and len(sync_data) > 0
n_rows = 3 if has_sync else 2
row_heights = [0.15, 0.6, 0.25] if has_sync else [0.3, 0.7]
subplot_titles = []
if has_sync:
    subplot_titles = ["Behaviour", "Population Activity (dF/F)", "Event Rate"]
else:
    subplot_titles = ["Population Activity (dF/F)", "Event Rate"]

fig = make_subplots(
    rows=n_rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=row_heights,
    subplot_titles=subplot_titles,
)

behaviour_row = 1 if has_sync else None
heatmap_row = 2 if has_sync else 1
event_row = 3 if has_sync else 2

# --- Row 1: Behavioural context (if sync data available) ---
if has_sync and sync_data:
    # Light on/off as background shading
    if "light_on" in sync_data:
        light = sync_data["light_on"]
        n_sync = min(len(light), n_frames)
        sync_time = time_s[:n_sync]

        # Find light-off epochs
        light_trimmed = light[:n_sync].astype(bool)
        dark_starts = []
        dark_ends = []
        in_dark = not light_trimmed[0]
        if in_dark:
            dark_starts.append(sync_time[0])
        for i in range(1, len(light_trimmed)):
            if not light_trimmed[i] and light_trimmed[i - 1]:
                dark_starts.append(sync_time[i])
            elif light_trimmed[i] and not light_trimmed[i - 1]:
                dark_ends.append(sync_time[i])
        if len(dark_starts) > len(dark_ends):
            dark_ends.append(sync_time[-1])

        for ds, de in zip(dark_starts, dark_ends):
            for row in range(1, n_rows + 1):
                fig.add_vrect(
                    x0=ds,
                    x1=de,
                    fillcolor="rgba(50,50,50,0.15)",
                    layer="below",
                    line_width=0,
                    row=row,
                    col=1,
                )

    # Speed trace
    if "speed" in sync_data:
        speed = sync_data["speed"]
        n_sync = min(len(speed), n_frames)
        fig.add_trace(
            go.Scatter(
                x=time_s[:n_sync],
                y=speed[:n_sync],
                mode="lines",
                name="Speed (cm/s)",
                line=dict(color="orange", width=1),
            ),
            row=behaviour_row,
            col=1,
        )
        # Movement threshold line
        fig.add_hline(
            y=2.5,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            row=behaviour_row,
            col=1,
        )

    # HD trace
    if "hd" in sync_data:
        hd = sync_data["hd"]
        n_sync = min(len(hd), n_frames)
        fig.add_trace(
            go.Scatter(
                x=time_s[:n_sync],
                y=np.degrees(hd[:n_sync]) if np.max(np.abs(hd[:n_sync])) < 10 else hd[:n_sync],
                mode="lines",
                name="Head direction (deg)",
                line=dict(color="purple", width=0.8),
                yaxis="y2",
                opacity=0.6,
            ),
            row=behaviour_row,
            col=1,
        )

# --- Row 2: Population dF/F heatmap ---
# Sort ROIs by SNR for better visualization
snr = ca_data["snr"]
sort_idx = np.argsort(snr)[::-1]  # High SNR first
dff_sorted = dff[sort_idx]

# Downsample for performance (max 2000 time points)
ds_factor = max(1, n_frames // 2000)
dff_ds = dff_sorted[:, ::ds_factor]
time_ds = time_s[::ds_factor]

# Clip for better contrast
vmax = np.percentile(dff_ds, 98)
dff_clipped = np.clip(dff_ds, 0, vmax)

fig.add_trace(
    go.Heatmap(
        z=dff_clipped,
        x=time_ds,
        y=np.arange(n_rois),
        colorscale="Hot",
        colorbar=dict(title="dF/F", len=0.3, y=0.5),
        hovertemplate="Time: %{x:.1f}s<br>ROI: %{y}<br>dF/F: %{z:.3f}<extra></extra>",
    ),
    row=heatmap_row,
    col=1,
)

# --- Row 3: Population event rate ---
if "event_masks" in ca_data:
    em = ca_data["event_masks"]
    # Compute population event rate (fraction of cells active per frame)
    pop_rate = em.mean(axis=0)
    # Smooth with 1s window
    kernel_size = max(1, int(fps))
    kernel = np.ones(kernel_size) / kernel_size
    pop_rate_smooth = np.convolve(pop_rate, kernel, mode="same")

    fig.add_trace(
        go.Scatter(
            x=time_s,
            y=pop_rate_smooth,
            mode="lines",
            name="Population event rate",
            line=dict(color="#2ca02c", width=1),
            fill="tozeroy",
            fillcolor="rgba(44,160,44,0.2)",
        ),
        row=event_row,
        col=1,
    )
else:
    # Use mean dF/F as proxy
    mean_dff = dff.mean(axis=0)
    kernel_size = max(1, int(fps))
    kernel = np.ones(kernel_size) / kernel_size
    mean_smooth = np.convolve(mean_dff, kernel, mode="same")

    fig.add_trace(
        go.Scatter(
            x=time_s,
            y=mean_smooth,
            mode="lines",
            name="Mean dF/F",
            line=dict(color="#2ca02c", width=1),
            fill="tozeroy",
            fillcolor="rgba(44,160,44,0.2)",
        ),
        row=event_row,
        col=1,
    )

# Layout
fig.update_layout(
    height=700 if has_sync else 550,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(title="Time (s)") if not has_sync else {},
    margin=dict(l=60, r=20, t=40, b=40),
)

# X-axis label on bottom only
fig.update_xaxes(title_text="Time (s)", row=n_rows, col=1)

# Y-axis labels
if has_sync:
    fig.update_yaxes(title_text="cm/s", row=behaviour_row, col=1)
fig.update_yaxes(title_text="ROI (by SNR)", row=heatmap_row, col=1)
fig.update_yaxes(title_text="Fraction active", row=event_row, col=1)

st.plotly_chart(fig, use_container_width=True)

# --- Light cycle summary ---
if has_sync and sync_data and "light_on" in sync_data:
    st.subheader("Light Cycle Summary")

    light = sync_data["light_on"][:n_frames].astype(bool)
    n_light = light.sum()
    n_dark = (~light).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Light ON frames", f"{n_light:,} ({n_light/n_frames*100:.1f}%)")
    col2.metric("Light OFF frames", f"{n_dark:,} ({n_dark/n_frames*100:.1f}%)")

    # Count transitions
    transitions = np.sum(np.diff(light.astype(int)) != 0)
    col3.metric("Light transitions", transitions)

    # Mean activity by light condition
    if "event_masks" in ca_data:
        em = ca_data["event_masks"][:, :n_frames]
        light_rate = em[:, light].mean()
        dark_rate = em[:, ~light].mean()
        st.markdown(
            f"**Population event rate:** Light ON = {light_rate:.4f}, "
            f"Light OFF = {dark_rate:.4f}, "
            f"ratio = {dark_rate/light_rate:.2f}x" if light_rate > 0 else ""
        )

# --- Individual ROI browser ---
st.markdown("---")
st.subheader("ROI Trace Browser")
st.caption("Click through individual ROIs to see their activity within the session context.")

col1, col2 = st.columns([1, 3])
with col1:
    snr_threshold = st.slider("Min SNR", 0.0, 15.0, 0.0, 0.5, key="tl_snr")
    good_rois = np.where(snr > snr_threshold)[0]
    st.markdown(f"**{len(good_rois)}** ROIs above threshold")

    if len(good_rois) == 0:
        st.warning("No ROIs pass SNR filter.")
        st.stop()

    roi_idx = st.selectbox(
        "ROI",
        good_rois.tolist(),
        format_func=lambda i: f"ROI {i} (SNR={snr[i]:.1f})",
        key="tl_roi",
    )

with col2:
    trace = dff[roi_idx]
    roi_fig = go.Figure()

    # Add light/dark shading if available
    if has_sync and sync_data and "light_on" in sync_data:
        light = sync_data["light_on"][:n_frames].astype(bool)
        light_trimmed = light
        dark_starts = []
        dark_ends = []
        in_dark = not light_trimmed[0]
        if in_dark:
            dark_starts.append(time_s[0])
        for i in range(1, len(light_trimmed)):
            if not light_trimmed[i] and light_trimmed[i - 1]:
                dark_starts.append(time_s[i])
            elif light_trimmed[i] and not light_trimmed[i - 1]:
                dark_ends.append(time_s[i])
        if len(dark_starts) > len(dark_ends):
            dark_ends.append(time_s[-1])
        for ds, de in zip(dark_starts, dark_ends):
            roi_fig.add_vrect(
                x0=ds, x1=de,
                fillcolor="rgba(50,50,50,0.12)",
                layer="below",
                line_width=0,
            )

    # dF/F trace
    roi_fig.add_trace(
        go.Scatter(
            x=time_s,
            y=trace,
            mode="lines",
            name="dF/F",
            line=dict(color="black", width=0.8),
        )
    )

    # Event overlay
    if "event_masks" in ca_data:
        em = ca_data["event_masks"][roi_idx].astype(bool)
        event_trace = trace.copy()
        event_trace[~em] = np.nan
        roi_fig.add_trace(
            go.Scatter(
                x=time_s,
                y=event_trace,
                mode="lines",
                name="Events",
                line=dict(color="red", width=1.5),
            )
        )

    roi_fig.update_layout(
        height=250,
        margin=dict(l=50, r=20, t=30, b=30),
        title=f"ROI {roi_idx} — SNR={snr[roi_idx]:.1f}",
        xaxis_title="Time (s)",
        yaxis_title="dF/F",
        showlegend=True,
    )
    st.plotly_chart(roi_fig, use_container_width=True)
