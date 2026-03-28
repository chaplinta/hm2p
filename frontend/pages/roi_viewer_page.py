"""ROI Viewer — fast single-ROI inspection across all sessions.

Browse ALL ROIs across ALL sessions with arrow-key navigation.
A flat global index cycles through every ROI in every session.
Data is cached per session so switching ROIs is instant.
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

log = logging.getLogger("hm2p.frontend.roi_viewer")

ROI_TYPE_NAMES = {0: "Soma", 1: "Dendrite", 2: "Non-cell"}
ROI_TYPE_COLORS = {0: "turquoise", 1: "darkorchid", 2: "gray"}

st.title("ROI Viewer")
st.caption("Browse all ROIs across all sessions. Click the index and use arrow keys.")


# ── Cached loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner="Loading calcium data...")
def _load_ca():
    return load_all_ca_data()


@st.cache_data(ttl=1800, show_spinner="Loading Suite2p spatial data...")
def _load_spatial():
    return load_all_suite2p_spatial()


@st.cache_data(ttl=1800, show_spinner="Loading raw fluorescence...")
def _load_raw_f(sub: str, ses: str) -> dict | None:
    prefix = f"ca_extraction/{sub}/{ses}/suite2p/plane0"
    f_data = download_s3_bytes(DERIVATIVES_BUCKET, f"{prefix}/F.npy")
    fneu_data = download_s3_bytes(DERIVATIVES_BUCKET, f"{prefix}/Fneu.npy")
    iscell_data = download_s3_bytes(DERIVATIVES_BUCKET, f"{prefix}/iscell.npy")
    if f_data is None:
        return None
    F_all = np.load(io.BytesIO(f_data))
    Fneu_all = np.load(io.BytesIO(fneu_data)) if fneu_data else None
    if iscell_data is not None:
        iscell = np.load(io.BytesIO(iscell_data))
        mask = iscell[:, 0].astype(bool)
        F_all = F_all[mask]
        Fneu_all = Fneu_all[mask] if Fneu_all is not None else None
    return {"F": F_all, "Fneu": Fneu_all}


ca_sessions = _load_ca()
spatial_data = _load_spatial()

if not ca_sessions:
    st.warning("No calcium data available. Run Stage 4 first.")
    st.stop()


# ── Sidebar filters ──────────────────────────────────────────────────────

fc1, fc2 = st.columns(2)
with fc1:
    roi_filter = st.radio(
        "ROI type", ["Soma only", "Non-soma only", "All"],
        index=0, key="rv_type", horizontal=True,
    )
with fc2:
    session_filter = st.radio(
        "Sessions", ["Primary (non-excluded)", "All sessions"],
        index=0, key="rv_ses_filter", horizontal=True,
    )


# ── Build flat global ROI list ───────────────────────────────────────────
# Each entry: (session_index, roi_index_within_session)

roi_list: list[tuple[int, int]] = []

for si, ses in enumerate(ca_sessions):
    # Session filter
    if session_filter == "Primary (non-excluded)":
        if str(ses.get("exclude", "0")).strip() == "1":
            continue

    roi_types = ses.get("roi_types", np.zeros(ses["n_rois"], dtype=np.uint8))

    for ri in range(ses["n_rois"]):
        rt = int(roi_types[ri]) if ri < len(roi_types) else 0
        if roi_filter == "Soma only" and rt != 0:
            continue
        if roi_filter == "Non-soma only" and rt == 0:
            continue
        roi_list.append((si, ri))

if not roi_list:
    st.warning("No ROIs match the current filters.")
    st.stop()

n_total = len(roi_list)


# ── Global ROI selector ──────────────────────────────────────────────────

roi_pos = st.number_input(
    f"ROI index (0–{n_total - 1}, {n_total} total)",
    min_value=0, max_value=n_total - 1, value=0, step=1,
    key="rv_global",
    help="Click here then use ↑↓ arrow keys to browse all ROIs across all sessions",
)

ses_idx, roi_idx = roi_list[roi_pos]
ses = ca_sessions[ses_idx]
roi_types = ses.get("roi_types", np.zeros(ses["n_rois"], dtype=np.uint8))
roi_type = int(roi_types[roi_idx]) if roi_idx < len(roi_types) else 0
dff = ses["dff"][roi_idx]
frame_times = ses.get("frame_times")
t = (frame_times - frame_times[0]) if frame_times is not None else np.arange(len(dff)) / 9.8
duration_s = float(t[-1] - t[0]) if len(t) > 1 else 1.0

event_masks = ses.get("event_masks")
n_events = int(event_masks[roi_idx].sum()) if event_masks is not None and roi_idx < event_masks.shape[0] else 0
event_rate = n_events / duration_s if duration_s > 0 else 0

baseline_vals = dff[dff < np.nanpercentile(dff, 25)] if len(dff) > 10 else dff
baseline_std = float(np.nanstd(baseline_vals)) if len(baseline_vals) > 0 else 0
peak_dff = float(np.nanmax(dff)) if len(dff) > 0 else 0
snr = peak_dff / baseline_std if baseline_std > 0 else 0
celltype_label = "Penk+" if ses["celltype"] == "penk" else "Penk\u207bCamKII+"


# ── Metrics bar (always visible at top) ──────────────────────────────────

c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
c1.metric("Global #", f"{roi_pos}")
c2.metric("Session", ses["exp_id"].split("_")[-1])
c3.metric("ROI", roi_idx)
c4.metric("Type", ROI_TYPE_NAMES.get(roi_type, "?"))
c5.metric("Celltype", celltype_label)
c6.metric("SNR", f"{snr:.1f}")
c7.metric("Events", f"{n_events}")
c8.metric("Rate", f"{event_rate:.2f}/s")

st.markdown(
    f"<small style='color:gray'>{ses['exp_id']} &mdash; "
    f"peak dF/F₀ = {peak_dff:.2f}, baseline σ = {baseline_std:.4f}, "
    f"duration = {duration_s:.0f}s, {ses['n_rois']} ROIs in session</small>",
    unsafe_allow_html=True,
)


# ── Spatial images ───────────────────────────────────────────────────────

spatial = spatial_data.get(ses["exp_id"], {})
mean_img = spatial.get("mean_img")
max_img = spatial.get("max_img")
shape_features = spatial.get("shape_features", [])

_img_choice = st.radio(
    "Background image", ["Max projection", "Mean"],
    horizontal=True, key="rv_img_type",
)
bg_img = max_img if _img_choice == "Max projection" and max_img is not None else mean_img

col_mean, col_roi = st.columns(2)

if bg_img is not None:
    with col_mean:
        fig = go.Figure(data=go.Heatmap(
            z=bg_img, colorscale="gray", showscale=False,
        ))
        fig.update_layout(
            height=300, title=_img_choice,
            yaxis=dict(scaleanchor="x", autorange="reversed"),
            margin=dict(t=30, b=5, l=5, r=5),
        )
        st.plotly_chart(fig, use_container_width=True, key="rv_mean")

    with col_roi:
        fig = go.Figure(data=go.Heatmap(
            z=bg_img, colorscale="gray", showscale=False,
        ))
        if roi_idx < len(shape_features) and shape_features[roi_idx] is not None:
            sf = shape_features[roi_idx]
            ypix = sf.get("ypix", [])
            xpix = sf.get("xpix", [])
            if len(xpix) > 0:
                color = ROI_TYPE_COLORS.get(roi_type, "yellow")
                fig.add_trace(go.Scatter(
                    x=xpix, y=ypix, mode="markers",
                    marker=dict(size=3, color=color, opacity=0.8),
                    name=f"ROI {roi_idx}",
                ))
        img_h, img_w = bg_img.shape[:2]
        fig.update_layout(
            height=300,
            title=f"ROI {roi_idx} ({ROI_TYPE_NAMES.get(roi_type, '?')})",
            xaxis=dict(range=[0, img_w], constrain="domain"),
            yaxis=dict(range=[img_h, 0], scaleanchor="x", constrain="domain"),
            margin=dict(t=30, b=5, l=5, r=5),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key="rv_roi_img")
else:
    st.info("No spatial data for this session.")


# ── Time series ──────────────────────────────────────────────────────────

raw = _load_raw_f(ses["sub"], ses["ses"])

n_panels = 2
has_raw = raw is not None and raw["F"] is not None and roi_idx < raw["F"].shape[0]
has_spikes = ses.get("spikes") is not None and roi_idx < (ses.get("spikes", np.empty((0,))).shape[0])
if has_raw:
    n_panels = 4
if has_spikes:
    n_panels += 1

fig = make_subplots(
    rows=n_panels, cols=1, shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[1] * n_panels,
)

row = 1

if has_raw:
    F_trace = raw["F"][roi_idx]
    Fneu_trace = raw["Fneu"][roi_idx] if raw["Fneu"] is not None and roi_idx < raw["Fneu"].shape[0] else None

    # Panel 1: Raw F + neuropil
    fig.add_trace(go.Scattergl(
        x=t, y=F_trace, mode="lines", line=dict(color="gray", width=1), name="Raw F",
    ), row=row, col=1)
    if Fneu_trace is not None:
        fig.add_trace(go.Scattergl(
            x=t, y=Fneu_trace, mode="lines",
            line=dict(color="green", width=1, dash="dot"), name="Neuropil",
        ), row=row, col=1)
    fig.update_yaxes(title_text="F (a.u.)", row=row, col=1)
    row += 1

    # Panel 2: F corrected + F0
    neuropil_coeff = 0.7
    F_corr = F_trace - neuropil_coeff * Fneu_trace if Fneu_trace is not None else F_trace
    safe_dff = np.where(np.abs(dff) > 0.01, dff, 0.01)
    F0_approx = F_corr / (safe_dff + 1.0)
    fig.add_trace(go.Scattergl(
        x=t, y=F_corr, mode="lines", line=dict(color="steelblue", width=1), name="F corrected",
    ), row=row, col=1)
    fig.add_trace(go.Scattergl(
        x=t, y=F0_approx, mode="lines", line=dict(color="red", width=1.5), name="F₀ baseline",
    ), row=row, col=1)
    fig.update_yaxes(title_text="F / F₀", row=row, col=1)
    row += 1

# Panel 3: dF/F₀ + events
fig.add_trace(go.Scattergl(
    x=t, y=dff, mode="lines", line=dict(color="royalblue", width=1), name="dF/F₀",
), row=row, col=1)
if event_masks is not None and roi_idx < event_masks.shape[0]:
    events = event_masks[roi_idx].astype(bool)
    if events.any():
        fig.add_trace(go.Scattergl(
            x=t[events], y=dff[events], mode="markers",
            marker=dict(size=3, color="red", symbol="circle"), name="Events",
        ), row=row, col=1)
fig.update_yaxes(title_text="dF/F₀", row=row, col=1)
row += 1

# Panel 4: Deconvolved (normalized)
deconv_norm = ses.get("deconv_norm")
if deconv_norm is not None and roi_idx < deconv_norm.shape[0]:
    fig.add_trace(go.Scattergl(
        x=t, y=deconv_norm[roi_idx], mode="lines",
        line=dict(color="darkorange", width=1), name="Deconv (norm)",
    ), row=row, col=1)
    fig.update_yaxes(title_text="Deconv", row=row, col=1)
else:
    fig.add_trace(go.Scattergl(
        x=t, y=np.zeros_like(dff), mode="lines",
        line=dict(color="lightgray", width=0.5), name="(no deconv)",
    ), row=row, col=1)
    fig.update_yaxes(title_text="(no deconv)", row=row, col=1)

# Panel: CASCADE spikes (if available)
row += 1
spikes = ses.get("spikes")
if has_spikes and spikes is not None and roi_idx < spikes.shape[0]:
    spike_trace = np.nan_to_num(spikes[roi_idx])
    fig.add_trace(go.Scattergl(
        x=t, y=spike_trace, mode="lines",
        line=dict(color="#d62728", width=1), name="CASCADE spikes",
    ), row=row, col=1)
    fig.update_yaxes(title_text="Spk/s", row=row, col=1)

fig.update_xaxes(title_text="Time (s)", row=n_panels, col=1)
fig.update_layout(
    height=130 * n_panels + 40,
    margin=dict(t=10, b=35, l=55, r=15),
    showlegend=True,
    legend=dict(orientation="h", y=-0.06),
)
st.plotly_chart(fig, use_container_width=True, key="rv_traces")


# ── Footer ───────────────────────────────────────────────────────────────

st.caption(
    "Click the ROI index input and use **↑↓ arrow keys** to cycle through all ROIs "
    "across all sessions. Session data is cached — switching is instant."
)
