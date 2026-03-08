"""Data Explorer — unified drill-down into any session's neural and behavioural data.

One-stop page for exploring a session's calcium traces, pose data, timestamps,
and (when available) synced kinematics. Navigate by session, view any data type.
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
    RAWDATA_BUCKET,
    download_s3_bytes,
    list_s3_session_files,
    load_animals,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.explorer")

st.title("Data Explorer")


@st.cache_data(ttl=600)
def _load_h5_from_s3(bucket: str, key: str) -> dict | None:
    """Download and parse an HDF5 file from S3."""
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


# --- Session selector ---
experiments = load_experiments()
animals = load_animals()
animal_map = {a["animal_id"]: a for a in animals}
exp_ids = [e["exp_id"] for e in experiments]

default_idx = 0
if "selected_exp_id" in st.session_state:
    sel = st.session_state["selected_exp_id"]
    if sel in exp_ids:
        default_idx = exp_ids.index(sel)

selected_exp = st.selectbox(
    "Session",
    exp_ids,
    index=default_idx,
    format_func=lambda x: f"{x} ({animal_map.get(x.split('_')[-1], {}).get('celltype', '?')})",
    key="explorer_session",
)
sub, ses = parse_session_id(selected_exp)
animal_id = selected_exp.split("_")[-1]
animal = animal_map.get(animal_id, {})

st.caption(
    f"`{sub}/{ses}` | {animal.get('celltype', '?')} | "
    f"{animal.get('strain', '?')} | {animal.get('gcamp', '?')}"
)

# --- Check available data ---
data_sources = {}


@st.cache_data(ttl=120)
def _check_data_availability(sub: str, ses: str) -> dict[str, bool]:
    """Check which data types are available for this session."""
    import boto3
    s3 = boto3.client("s3", region_name="ap-southeast-2")

    checks = {
        "timestamps": f"movement/{sub}/{ses}/timestamps.h5",
        "ca_extraction": f"ca_extraction/{sub}/{ses}/suite2p/plane0/F.npy",
        "calcium": f"calcium/{sub}/{ses}/ca.h5",
        "pose": f"pose/{sub}/{ses}/",
        "kinematics": f"movement/{sub}/{ses}/kinematics.h5",
        "sync": f"sync/{sub}/{ses}/sync.h5",
        "analysis": f"analysis/{sub}/{ses}/analysis.h5",
    }
    available = {}
    for name, prefix in checks.items():
        try:
            resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=prefix, MaxKeys=1)
            available[name] = resp.get("KeyCount", 0) > 0
        except Exception:
            available[name] = False
    return available


avail = _check_data_availability(sub, ses)

# Show availability status
status_cols = st.columns(7)
labels = ["Timestamps", "Suite2p", "Calcium", "Pose", "Kinematics", "Sync", "Analysis"]
keys = ["timestamps", "ca_extraction", "calcium", "pose", "kinematics", "sync", "analysis"]
for col, label, key in zip(status_cols, labels, keys):
    icon = ":white_check_mark:" if avail.get(key, False) else ":x:"
    col.markdown(f"{icon} {label}")

st.markdown("---")

# --- Data viewing tabs ---
available_tabs = ["Summary"]
if avail.get("calcium"):
    available_tabs.append("Calcium Traces")
if avail.get("timestamps"):
    available_tabs.append("Timestamps")
if avail.get("pose"):
    available_tabs.append("Pose Data")
available_tabs.append("Raw Files")

tabs = st.tabs(available_tabs)
tab_idx = 0

# --- Summary Tab ---
with tabs[tab_idx]:
    tab_idx += 1
    st.subheader("Session Summary")

    # Experiment metadata
    exp = next(e for e in experiments if e["exp_id"] == selected_exp)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Experiment:**")
        for key in ["lens", "orientation", "fibre", "primary_exp", "exclude", "Notes"]:
            val = exp.get(key, "")
            if val:
                st.text(f"  {key}: {val}")

    with col2:
        st.markdown("**Animal:**")
        for key in ["celltype", "strain", "gcamp", "virus_id", "hemisphere", "sex"]:
            st.text(f"  {key}: {animal.get(key, '')}")

    # Quick calcium stats
    if avail.get("calcium"):
        st.markdown("---")
        st.markdown("**Calcium data:**")
        ca = _load_h5_from_s3(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
        if ca and "dff" in ca:
            dff = ca["dff"]
            n_rois, n_frames = dff.shape
            fps = float(ca.get("_attr_fps_imaging", 9.8))

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("ROIs", n_rois)
            col2.metric("Frames", n_frames)
            col3.metric("Duration", f"{n_frames/fps:.0f}s")
            col4.metric("FPS", f"{fps:.1f}")

            if "event_masks" in ca:
                em = ca["event_masks"]
                total_events = 0
                for i in range(n_rois):
                    m = em[i].astype(bool)
                    onsets = np.flatnonzero(m[1:] & ~m[:-1])
                    total_events += len(onsets) + (1 if m[0] else 0)
                st.metric("Total events", total_events)

    # Quick timestamps stats
    if avail.get("timestamps"):
        st.markdown("---")
        st.markdown("**Timestamps:**")
        ts = _load_h5_from_s3(DERIVATIVES_BUCKET, f"movement/{sub}/{ses}/timestamps.h5")
        if ts:
            cam = ts.get("frame_times_camera")
            img = ts.get("frame_times_imaging")
            lon = ts.get("light_on_times")
            if cam is not None:
                st.text(f"  Camera frames: {len(cam)}, Duration: {cam[-1]:.1f}s")
            if img is not None:
                st.text(f"  Imaging frames: {len(img)}, Duration: {img[-1]:.1f}s")
            if lon is not None:
                st.text(f"  Light cycles: {len(lon)}")


# --- Calcium Traces Tab ---
if avail.get("calcium"):
    with tabs[tab_idx]:
        tab_idx += 1
        st.subheader("Calcium Traces")

        if ca and "dff" in ca:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            roi = st.selectbox("ROI", range(n_rois), key="exp_roi")

            # Time range
            max_time = n_frames / fps
            t_start, t_end = st.slider(
                "Time range (s)", 0.0, max_time,
                (0.0, min(120.0, max_time)),
                key="exp_time",
            )
            f_start = int(t_start * fps)
            f_end = int(t_end * fps)

            trace = dff[roi, f_start:f_end]
            time_ax = np.arange(f_start, f_end) / fps

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_ax, y=trace,
                mode="lines", line=dict(width=0.8, color="black"),
                name="dF/F",
            ))

            # Event overlay
            if "event_masks" in ca:
                em_roi = ca["event_masks"][roi, f_start:f_end].astype(bool)
                if em_roi.any():
                    event_trace = trace.copy()
                    event_trace[~em_roi] = np.nan
                    fig.add_trace(go.Scatter(
                        x=time_ax, y=event_trace,
                        mode="lines", line=dict(width=2, color="red"),
                        name="Events",
                    ))

            # Light on/off overlay (from timestamps)
            if avail.get("timestamps") and ts:
                lon = ts.get("light_on_times")
                loff = ts.get("light_off_times")
                if lon is not None and loff is not None:
                    for i in range(len(lon)):
                        off_t = loff[i] if i < len(loff) else max_time
                        if off_t > t_start and lon[i] < t_end:
                            fig.add_vrect(
                                x0=max(lon[i], t_start), x1=min(off_t, t_end),
                                fillcolor="yellow", opacity=0.15, line_width=0,
                            )

            fig.update_layout(
                title=f"ROI {roi} dF/F",
                xaxis_title="Time (s)",
                yaxis_title="dF/F",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Multi-ROI view
            st.subheader("Multi-ROI Heatmap")
            ds = max(1, (f_end - f_start) // 1000)
            dff_slice = dff[:, f_start:f_end:ds]
            time_ds = np.arange(dff_slice.shape[1]) * ds / fps + t_start

            fig_heat = go.Figure(data=go.Heatmap(
                z=dff_slice, x=time_ds,
                colorscale="RdBu_r",
                zmin=-np.percentile(np.abs(dff_slice), 95),
                zmax=np.percentile(np.abs(dff_slice), 95),
            ))
            fig_heat.update_layout(
                xaxis_title="Time (s)", yaxis_title="ROI",
                height=300,
            )
            st.plotly_chart(fig_heat, use_container_width=True)


# --- Timestamps Tab ---
if avail.get("timestamps"):
    with tabs[tab_idx]:
        tab_idx += 1
        st.subheader("Timing Data")

        if ts:
            cam = ts.get("frame_times_camera")
            img = ts.get("frame_times_imaging")

            import plotly.graph_objects as go

            if cam is not None and len(cam) > 1:
                intervals = np.diff(cam) * 1000  # ms
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=intervals, mode="lines", line=dict(width=0.5)))
                fig.update_layout(
                    title="Camera frame intervals",
                    yaxis_title="Interval (ms)",
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                col1.metric("Mean", f"{intervals.mean():.2f} ms")
                col2.metric("Std", f"{intervals.std():.3f} ms")
                col3.metric("Jitter", f"{intervals.std()/intervals.mean()*100:.2f}%")


# --- Pose Data Tab ---
if avail.get("pose"):
    with tabs[tab_idx]:
        tab_idx += 1
        st.subheader("Pose Data (DLC)")

        import json
        import pandas as pd

        meta_bytes = download_s3_bytes(DERIVATIVES_BUCKET, f"pose/{sub}/{ses}/dlc_meta.json")
        if meta_bytes:
            meta = json.loads(meta_bytes)
            st.text(f"Model: {meta.get('model', '?')} | FPS: {meta.get('tracking_fps', '?')}")

        # List pose files
        pose_files = list_s3_session_files(DERIVATIVES_BUCKET, f"pose/{sub}/{ses}/")
        h5_files = [f for f in pose_files if f["key"].endswith(".h5")]

        if h5_files:
            h5_data = download_s3_bytes(DERIVATIVES_BUCKET, h5_files[0]["key"])
            if h5_data:
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as tmp:
                        tmp.write(h5_data)
                        tmp.flush()
                        df = pd.read_hdf(tmp.name)
                    if isinstance(df.columns, pd.MultiIndex):
                        scorer = df.columns.get_level_values(0)[0]
                        bodyparts = df.columns.get_level_values(1).unique().tolist()
                        st.text(f"Body parts: {', '.join(bodyparts)}")
                        st.text(f"Frames: {len(df)}")

                        import plotly.graph_objects as go

                        bp = st.selectbox("Body part", bodyparts, key="exp_bp")
                        x = df[(scorer, bp, "x")].values
                        y = df[(scorer, bp, "y")].values

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=x[::5], y=y[::5],
                            mode="markers",
                            marker=dict(size=1, color=np.arange(0, len(x), 5)),
                        ))
                        fig.update_layout(
                            title=f"{bp} trajectory",
                            yaxis=dict(autorange="reversed"),
                            height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not load pose data: {e}")

# --- Raw Files Tab ---
with tabs[tab_idx]:
    st.subheader("Files on S3")

    # List derivatives
    for stage, label in [
        ("movement", "Timestamps"),
        ("ca_extraction", "Suite2p"),
        ("calcium", "Calcium"),
        ("pose", "Pose"),
        ("sync", "Sync"),
        ("analysis", "Analysis"),
    ]:
        files = list_s3_session_files(DERIVATIVES_BUCKET, f"{stage}/{sub}/{ses}/")
        if files:
            with st.expander(f"{label} ({len(files)} files)"):
                for f in files:
                    name = f["key"].split("/")[-1]
                    st.text(f"  {name} ({f['size_mb']:.1f} MB)")

    # Raw data
    raw_files = list_s3_session_files(RAWDATA_BUCKET, f"rawdata/{sub}/{ses}/")
    if raw_files:
        with st.expander(f"Raw data ({len(raw_files)} files)"):
            for f in raw_files:
                name = "/".join(f["key"].split("/")[-2:])
                st.text(f"  {name} ({f['size_mb']:.1f} MB)")
