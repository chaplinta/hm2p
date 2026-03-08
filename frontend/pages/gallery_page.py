"""ROI Gallery — grid view of all ROIs with mini traces for quick scanning."""

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

log = logging.getLogger("hm2p.frontend.gallery")

st.title("ROI Gallery")
st.caption("Grid view of all ROIs — quickly scan traces, SNR, and event patterns.")

# --- Session selector ---
experiments = load_experiments()
animals = load_animals()
animal_map = {a["animal_id"]: a for a in animals}

exp_ids = [e["exp_id"] for e in experiments]
selected = st.selectbox(
    "Session",
    exp_ids,
    format_func=lambda x: f"{x} ({animal_map.get(x.split('_')[-1], {}).get('celltype', '?')})",
    key="gallery_session",
)

if not selected:
    st.stop()

sub, ses = parse_session_id(selected)
animal_id = selected.split("_")[-1]
celltype = animal_map.get(animal_id, {}).get("celltype", "?")


@st.cache_data(ttl=300)
def load_gallery_data(sub: str, ses: str) -> dict | None:
    """Load ca.h5 data for gallery."""
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

    # ROI spatial info if available
    if "roi_x" in f:
        result["roi_x"] = f["roi_x"][:]
    if "roi_y" in f:
        result["roi_y"] = f["roi_y"][:]

    f.close()
    return result


with st.spinner("Loading calcium data..."):
    data = load_gallery_data(sub, ses)

if data is None:
    st.warning("No calcium data found.")
    st.stop()

dff = data["dff"]
n_rois, n_frames = dff.shape
fps = data["fps"]

# Compute per-ROI metrics
snrs = []
event_rates = []
mean_dffs = []
max_dffs = []
skewnesses = []

for i in range(n_rois):
    trace = dff[i]
    baseline_std = np.std(trace[trace < np.percentile(trace, 50)])
    peak = np.percentile(trace, 95)
    snrs.append(peak / baseline_std if baseline_std > 0 else 0)
    mean_dffs.append(float(np.nanmean(trace)))
    max_dffs.append(float(np.nanmax(trace)))

    if trace.std() > 0:
        skewnesses.append(float(((trace - trace.mean()) ** 3).mean() / trace.std() ** 3))
    else:
        skewnesses.append(0.0)

    if "event_masks" in data:
        em = data["event_masks"][i].astype(bool)
        onsets = np.flatnonzero(em[1:] & ~em[:-1])
        n_events = len(onsets) + (1 if em[0] else 0)
        event_rates.append(n_events / (n_frames / fps / 60))
    else:
        event_rates.append(0.0)

snrs = np.array(snrs)
event_rates = np.array(event_rates)

# --- Controls ---
st.markdown(f"**{sub} / {ses}** — {celltype} — **{n_rois} ROIs**")

col1, col2, col3, col4 = st.columns(4)
with col1:
    n_cols = st.selectbox("Columns", [3, 4, 5, 6], index=1, key="gal_cols")
with col2:
    sort_by = st.selectbox("Sort by", ["ROI index", "SNR (high first)", "Event rate", "Max dF/F"], key="gal_sort")
with col3:
    min_snr = st.slider("Min SNR", 0.0, 15.0, 0.0, 0.5, key="gal_snr")
with col4:
    max_rois = st.selectbox("Max ROIs shown", [12, 24, 48, 96, 200], index=1, key="gal_max")

# Filter
mask = snrs >= min_snr
valid_indices = np.where(mask)[0]

# Sort
if sort_by == "SNR (high first)":
    order = np.argsort(snrs[valid_indices])[::-1]
elif sort_by == "Event rate":
    order = np.argsort(event_rates[valid_indices])[::-1]
elif sort_by == "Max dF/F":
    order = np.argsort([max_dffs[i] for i in valid_indices])[::-1]
else:
    order = np.arange(len(valid_indices))

sorted_indices = valid_indices[order][:max_rois]

st.markdown(f"Showing **{len(sorted_indices)}** of {n_rois} ROIs (SNR >= {min_snr})")

# --- Gallery grid ---
import plotly.graph_objects as go

time_s = np.arange(n_frames) / fps

# Downsample for performance
ds = max(1, n_frames // 500)
time_ds = time_s[::ds]

for row_start in range(0, len(sorted_indices), n_cols):
    row_indices = sorted_indices[row_start : row_start + n_cols]
    cols = st.columns(n_cols)

    for col_idx, roi_idx in enumerate(row_indices):
        with cols[col_idx]:
            trace = dff[roi_idx]
            trace_ds = trace[::ds]

            # Mini trace plot
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=time_ds,
                    y=trace_ds,
                    mode="lines",
                    line=dict(color="black", width=0.5),
                    showlegend=False,
                )
            )

            # Event overlay
            if "event_masks" in data:
                em = data["event_masks"][roi_idx].astype(bool)
                event_trace = trace.copy()
                event_trace[~em] = np.nan
                event_ds = event_trace[::ds]
                fig.add_trace(
                    go.Scatter(
                        x=time_ds,
                        y=event_ds,
                        mode="lines",
                        line=dict(color="red", width=1),
                        showlegend=False,
                    )
                )

            fig.update_layout(
                height=120,
                margin=dict(l=25, r=5, t=20, b=15),
                title=dict(
                    text=f"ROI {roi_idx}",
                    font=dict(size=10),
                ),
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"gal_{roi_idx}")

            # Metrics below trace
            st.markdown(
                f"<div style='font-size:11px; line-height:1.3'>"
                f"SNR: <b>{snrs[roi_idx]:.1f}</b> &nbsp; "
                f"Events: <b>{event_rates[roi_idx]:.0f}</b>/min &nbsp; "
                f"Max: <b>{max_dffs[roi_idx]:.2f}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

# --- Summary statistics ---
st.markdown("---")
st.subheader("ROI Quality Distribution")

import plotly.express as px
import pandas as pd

roi_df = pd.DataFrame({
    "roi_idx": np.arange(n_rois),
    "snr": snrs,
    "event_rate": event_rates,
    "mean_dff": mean_dffs,
    "max_dff": max_dffs,
    "skewness": skewnesses,
})

col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(roi_df, x="snr", nbins=30, title="SNR Distribution")
    fig.add_vline(x=min_snr, line_dash="dash", line_color="red")
    fig.update_layout(height=250, margin=dict(l=40, r=20, t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        roi_df, x="snr", y="event_rate",
        title="SNR vs Event Rate",
        hover_data=["roi_idx"],
        opacity=0.6,
    )
    fig.update_layout(height=250, margin=dict(l=40, r=20, t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)
