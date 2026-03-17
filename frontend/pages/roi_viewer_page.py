"""ROI Viewer — fast single-ROI inspection with arrow-key navigation.

Browse ROIs one at a time with keyboard arrow keys. Shows:
- Mean image + max image with ROI footprint overlay
- Interactive time series: raw F, F₀ baseline, dF/F₀, deconvolved
- ROI metadata (type, SNR, event rate)

Data is cached per session so switching between ROIs is instant.
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
    load_all_ca_data,
    load_all_suite2p_spatial,
)
from hm2p.constants import HEX_PENK, HEX_NONPENK

log = logging.getLogger("hm2p.frontend.roi_viewer")

ROI_TYPE_NAMES = {0: "Soma", 1: "Dendrite", 2: "Artefact"}
ROI_TYPE_COLORS = {0: "turquoise", 1: "darkorchid", 2: "gray"}

st.title("ROI Viewer")
st.caption("Fast single-ROI inspection. Use arrow keys on the ROI selector to browse.")


# ── Load and cache all session data ──────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner="Loading calcium data...")
def _load_ca():
    return load_all_ca_data()


@st.cache_data(ttl=1800, show_spinner="Loading Suite2p spatial data...")
def _load_spatial():
    return load_all_suite2p_spatial()


@st.cache_data(ttl=1800, show_spinner="Loading raw fluorescence...")
def _load_raw_f(sub: str, ses: str) -> dict | None:
    """Download F.npy and Fneu.npy from Suite2p on S3."""
    prefix = f"ca_extraction/{sub}/{ses}/suite2p/plane0"
    f_data = download_s3_bytes(DERIVATIVES_BUCKET, f"{prefix}/F.npy")
    fneu_data = download_s3_bytes(DERIVATIVES_BUCKET, f"{prefix}/Fneu.npy")
    iscell_data = download_s3_bytes(DERIVATIVES_BUCKET, f"{prefix}/iscell.npy")

    if f_data is None:
        return None

    F_all = np.load(io.BytesIO(f_data))
    Fneu_all = np.load(io.BytesIO(fneu_data)) if fneu_data else None

    # Filter to accepted cells (iscell[:, 0] == 1)
    if iscell_data is not None:
        iscell = np.load(io.BytesIO(iscell_data))
        mask = iscell[:, 0].astype(bool)
        F = F_all[mask]
        Fneu = Fneu_all[mask] if Fneu_all is not None else None
    else:
        F = F_all
        Fneu = Fneu_all

    return {"F": F, "Fneu": Fneu}


ca_sessions = _load_ca()
spatial_data = _load_spatial()

if not ca_sessions:
    st.warning("No calcium data available. Run Stage 4 first.")
    st.stop()


# ── Sidebar: session + ROI type filter ────────────────────────────────────

with st.sidebar:
    st.header("Session")
    exp_ids = [s["exp_id"] for s in ca_sessions]
    sel_idx = st.selectbox("Session", range(len(exp_ids)),
                           format_func=lambda i: f"{exp_ids[i]} ({ca_sessions[i]['celltype']})",
                           key="rv_ses")
    ses = ca_sessions[sel_idx]

    st.header("ROI Filter")
    roi_filter = st.radio("Show", ["Soma", "Dendrite", "All"], index=0, key="rv_filter")

# Get ROI indices based on filter
roi_types = ses.get("roi_types", np.zeros(ses["n_rois"], dtype=np.uint8))
if roi_filter == "Soma":
    valid_rois = np.where(roi_types == 0)[0]
elif roi_filter == "Dendrite":
    valid_rois = np.where(roi_types == 1)[0]
else:
    valid_rois = np.arange(ses["n_rois"])

if len(valid_rois) == 0:
    st.warning(f"No {roi_filter.lower()} ROIs in this session.")
    st.stop()

# ── ROI selector (arrow keys work when focused) ──────────────────────────

n_valid = len(valid_rois)
col_sel, col_info = st.columns([2, 4])
with col_sel:
    roi_pos = st.number_input(
        f"ROI ({n_valid} {roi_filter.lower()})",
        min_value=0, max_value=n_valid - 1, value=0, step=1,
        key="rv_roi",
        help="Use arrow keys (↑↓) to browse ROIs quickly",
    )

roi_idx = valid_rois[roi_pos]
roi_type = int(roi_types[roi_idx]) if roi_idx < len(roi_types) else 0

with col_info:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROI index", roi_idx)
    c2.metric("Type", ROI_TYPE_NAMES.get(roi_type, "?"))
    c3.metric("Session", ses["exp_id"].split("_")[-1])
    c4.metric("Celltype", "Penk+" if ses["celltype"] == "penk" else "Penk⁻CamKII+")


# ── Spatial images: mean + max with ROI overlay ──────────────────────────

spatial = spatial_data.get(ses["exp_id"], {})
mean_img = spatial.get("mean_img")
shape_features = spatial.get("shape_features", [])

col_mean, col_roi = st.columns(2)

if mean_img is not None:
    with col_mean:
        fig = go.Figure(data=go.Heatmap(
            z=mean_img, colorscale="gray", showscale=False,
        ))
        fig.update_layout(
            height=350, title="Mean Image",
            yaxis=dict(scaleanchor="x", autorange="reversed"),
            margin=dict(t=30, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True, key="rv_mean")

    with col_roi:
        # Draw ROI on mean image
        fig = go.Figure(data=go.Heatmap(
            z=mean_img, colorscale="gray", showscale=False,
        ))
        # Overlay ROI pixels
        if roi_idx < len(shape_features) and shape_features[roi_idx] is not None:
            sf = shape_features[roi_idx]
            ypix = sf.get("ypix", [])
            xpix = sf.get("xpix", [])
            if len(xpix) > 0:
                color = ROI_TYPE_COLORS.get(roi_type, "yellow")
                fig.add_trace(go.Scatter(
                    x=xpix, y=ypix, mode="markers",
                    marker=dict(size=2, color=color, opacity=0.7),
                    name=f"ROI {roi_idx}",
                ))
        fig.update_layout(
            height=350, title=f"ROI {roi_idx} ({ROI_TYPE_NAMES.get(roi_type, '?')})",
            yaxis=dict(scaleanchor="x", autorange="reversed"),
            margin=dict(t=30, b=10, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key="rv_roi_img")
else:
    st.info("No spatial data (mean image) available for this session.")


# ── Time series ──────────────────────────────────────────────────────────

st.subheader("Time Series")

dff = ses["dff"][roi_idx]
frame_times = ses.get("frame_times")
if frame_times is not None:
    t = frame_times - frame_times[0]  # seconds from start
else:
    fps = 9.8
    t = np.arange(len(dff)) / fps

# Try to load raw F for this session
raw = _load_raw_f(ses["sub"], ses["ses"])

# Build multi-panel trace figure
n_panels = 2  # dF/F + deconv always shown
has_raw = raw is not None and raw["F"] is not None and roi_idx < raw["F"].shape[0]
if has_raw:
    n_panels = 4  # raw F, F0, dF/F, deconv

fig = make_subplots(
    rows=n_panels, cols=1, shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[1] * n_panels,
)

row = 1

if has_raw:
    F_trace = raw["F"][roi_idx]
    Fneu_trace = raw["Fneu"][roi_idx] if raw["Fneu"] is not None and roi_idx < raw["Fneu"].shape[0] else None

    # Raw fluorescence
    fig.add_trace(go.Scattergl(
        x=t, y=F_trace, mode="lines",
        line=dict(color="gray", width=1),
        name="Raw F",
    ), row=row, col=1)
    if Fneu_trace is not None:
        fig.add_trace(go.Scattergl(
            x=t, y=Fneu_trace, mode="lines",
            line=dict(color="green", width=1, dash="dot"),
            name="Neuropil",
        ), row=row, col=1)
    fig.update_yaxes(title_text="F (a.u.)", row=row, col=1)
    row += 1

    # F0 baseline (reconstruct from dF/F: F0 = F_corrected / (dF/F + 1))
    neuropil_coeff = 0.7
    F_corrected = F_trace - neuropil_coeff * Fneu_trace if Fneu_trace is not None else F_trace
    # Avoid division by zero
    safe_dff = np.where(np.abs(dff) > 0.01, dff, 0.01)
    F0_approx = F_corrected / (safe_dff + 1.0)
    fig.add_trace(go.Scattergl(
        x=t, y=F_corrected, mode="lines",
        line=dict(color="steelblue", width=1),
        name="F corrected",
    ), row=row, col=1)
    fig.add_trace(go.Scattergl(
        x=t, y=F0_approx, mode="lines",
        line=dict(color="red", width=1.5),
        name="F₀ baseline",
    ), row=row, col=1)
    fig.update_yaxes(title_text="F / F₀", row=row, col=1)
    row += 1

# dF/F₀ (always shown)
fig.add_trace(go.Scattergl(
    x=t, y=dff, mode="lines",
    line=dict(color="royalblue", width=1),
    name="dF/F₀",
), row=row, col=1)

# Overlay events if available
event_masks = ses.get("event_masks")
if event_masks is not None and roi_idx < event_masks.shape[0]:
    events = event_masks[roi_idx].astype(bool)
    if events.any():
        fig.add_trace(go.Scattergl(
            x=t[events], y=dff[events],
            mode="markers",
            marker=dict(size=3, color="red", symbol="circle"),
            name="Events",
        ), row=row, col=1)
fig.update_yaxes(title_text="dF/F₀", row=row, col=1)
row += 1

# Deconvolved / spikes
deconv = ses.get("deconv")
spikes = ses.get("spikes")
trace_deconv = None
if deconv is not None and roi_idx < deconv.shape[0]:
    trace_deconv = deconv[roi_idx]
elif spikes is not None and roi_idx < spikes.shape[0]:
    trace_deconv = spikes[roi_idx]

if trace_deconv is not None:
    fig.add_trace(go.Scattergl(
        x=t, y=trace_deconv, mode="lines",
        line=dict(color="darkorange", width=1),
        name="Deconvolved",
    ), row=row, col=1)
    fig.update_yaxes(title_text="Deconv", row=row, col=1)
else:
    # Show dF/F again as placeholder
    fig.add_trace(go.Scattergl(
        x=t, y=dff, mode="lines",
        line=dict(color="royalblue", width=1, dash="dot"),
        name="dF/F₀ (no deconv)",
    ), row=row, col=1)
    fig.update_yaxes(title_text="(no deconv)", row=row, col=1)

fig.update_xaxes(title_text="Time (s)", row=n_panels, col=1)
fig.update_layout(
    height=150 * n_panels + 50,
    margin=dict(t=20, b=40, l=60, r=20),
    showlegend=True,
    legend=dict(orientation="h", y=-0.05),
)
st.plotly_chart(fig, use_container_width=True, key="rv_traces")


# ── ROI metrics ──────────────────────────────────────────────────────────

with st.expander("ROI Metrics"):
    baseline_std = float(np.nanstd(dff[dff < np.nanpercentile(dff, 25)])) if len(dff) > 10 else 0
    peak_dff = float(np.nanmax(dff))
    snr = peak_dff / baseline_std if baseline_std > 0 else 0
    n_events = int(event_masks[roi_idx].sum()) if event_masks is not None and roi_idx < event_masks.shape[0] else 0
    duration_s = float(t[-1] - t[0]) if len(t) > 1 else 1
    event_rate = n_events / duration_s if duration_s > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("SNR", f"{snr:.1f}")
    c2.metric("Peak dF/F₀", f"{peak_dff:.2f}")
    c3.metric("Baseline σ", f"{baseline_std:.4f}")
    c4.metric("Events", n_events)
    c5.metric("Event rate", f"{event_rate:.2f}/s")


# ── Footer ───────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Tip: Click on the ROI number input and use **↑↓ arrow keys** to quickly "
    "browse through ROIs. Data is cached per session — switching ROIs is instant."
)
