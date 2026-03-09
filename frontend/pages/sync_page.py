"""Sync diagnostics page — timing pulse visualization and sync quality."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Ensure src is on path for validation module
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    RAWDATA_BUCKET,
    download_s3_bytes,
    download_s3_numpy,
    list_s3_session_files,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.sync")

st.title("Sync Diagnostics")

# --- Session selector ---
experiments = load_experiments()
exp_ids = [e["exp_id"] for e in experiments]

# Use session state from sessions page if available
default_idx = 0
if "selected_exp_id" in st.session_state:
    sel = st.session_state["selected_exp_id"]
    if sel in exp_ids:
        default_idx = exp_ids.index(sel)

selected_exp = st.selectbox("Session", exp_ids, index=default_idx, key="sync_exp")
sub, ses = parse_session_id(selected_exp)
st.caption(f"`{sub}/{ses}`")


# --- Helper to load HDF5 from S3 ---
@st.cache_data(ttl=120)
def load_h5_from_s3(bucket: str, key: str) -> dict | None:
    """Download an HDF5 file from S3 and return its datasets as a dict."""
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
        log.exception("Error reading HDF5 from s3://%s/%s", bucket, key)
        return None


# --- Check for timestamps.h5 ---
timestamps_key = f"movement/{sub}/{ses}/timestamps.h5"
timestamps = load_h5_from_s3(DERIVATIVES_BUCKET, timestamps_key)

# Also check rawdata for TDMS files
rawdata_prefix = f"rawdata/{sub}/{ses}/"
raw_files = list_s3_session_files(RAWDATA_BUCKET, rawdata_prefix)
tdms_files = [f for f in raw_files if f["key"].endswith(".tdms")]

# --- Tab layout ---
tab_timing, tab_quality, tab_batch, tab_raw = st.tabs(
    ["Timing Pulses", "Sync Quality", "Batch Validation", "Raw Files"]
)

with tab_timing:
    if timestamps is None:
        st.warning(
            f"No `timestamps.h5` found at `s3://{DERIVATIVES_BUCKET}/{timestamps_key}`.\n\n"
            "Run Stage 0 (DAQ parsing) first, or upload TDMS files to S3."
        )
        if tdms_files:
            st.info(f"Found {len(tdms_files)} TDMS file(s) in rawdata — Stage 0 can process these.")
            for f in tdms_files:
                st.text(f"  {f['key']} ({f['size_mb']:.1f} MB)")
        else:
            st.info("No TDMS files found in rawdata either. Upload raw data first.")
    else:
        st.success("timestamps.h5 loaded")

        # Show metadata
        fps_cam = timestamps.get("_attr_fps_camera", "?")
        fps_img = timestamps.get("_attr_fps_imaging", "?")
        st.markdown(f"**Camera FPS:** {fps_cam} | **Imaging FPS:** {fps_img}")

        cam_times = timestamps.get("frame_times_camera")
        img_times = timestamps.get("frame_times_imaging")
        light_on = timestamps.get("light_on_times")
        light_off = timestamps.get("light_off_times")

        if cam_times is not None:
            st.markdown(f"**Camera frames:** {len(cam_times)} | "
                        f"**Duration:** {cam_times[-1]:.1f}s")
        if img_times is not None:
            st.markdown(f"**Imaging frames:** {len(img_times)} | "
                        f"**Duration:** {img_times[-1]:.1f}s")
        if light_on is not None:
            st.markdown(f"**Light on pulses:** {len(light_on)} | "
                        f"**Light off pulses:** {len(light_off) if light_off is not None else '?'}")

        st.subheader("Camera Frame Intervals")
        if cam_times is not None and len(cam_times) > 1:
            cam_intervals = np.diff(cam_times) * 1000  # ms
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=cam_intervals,
                mode="lines",
                name="Camera frame interval",
                line=dict(width=0.5),
            ))
            nominal = 1000 / float(fps_cam) if fps_cam != "?" else None
            if nominal:
                fig.add_hline(y=nominal, line_dash="dash", line_color="green",
                              annotation_text=f"Nominal ({nominal:.1f} ms)")
            fig.update_layout(
                xaxis_title="Frame #",
                yaxis_title="Interval (ms)",
                height=300,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{cam_intervals.mean():.2f} ms")
            col2.metric("Std", f"{cam_intervals.std():.3f} ms")
            col3.metric("Min", f"{cam_intervals.min():.2f} ms")
            col4.metric("Max", f"{cam_intervals.max():.2f} ms")

            # Flag bad intervals
            if nominal:
                bad_mask = np.abs(cam_intervals - nominal) > 2.0  # >2ms off
                n_bad = bad_mask.sum()
                if n_bad > 0:
                    st.warning(f"{n_bad} camera frame intervals deviate >2ms from nominal")
                    # Show where the bad intervals are
                    bad_indices = np.flatnonzero(bad_mask)
                    if len(bad_indices) <= 20:
                        st.caption(f"Bad intervals at frames: {bad_indices.tolist()}")
                else:
                    st.success("All camera frame intervals within 2ms of nominal")

        st.subheader("Imaging Frame Intervals")
        if img_times is not None and len(img_times) > 1:
            img_intervals = np.diff(img_times) * 1000  # ms
            import plotly.graph_objects as go

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                y=img_intervals,
                mode="lines",
                name="Imaging frame interval",
                line=dict(width=0.5),
            ))
            nominal_img = 1000 / float(fps_img) if fps_img != "?" else None
            if nominal_img:
                fig2.add_hline(y=nominal_img, line_dash="dash", line_color="green",
                               annotation_text=f"Nominal ({nominal_img:.1f} ms)")
            fig2.update_layout(
                xaxis_title="Frame #",
                yaxis_title="Interval (ms)",
                height=300,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{img_intervals.mean():.2f} ms")
            col2.metric("Std", f"{img_intervals.std():.3f} ms")
            col3.metric("Min", f"{img_intervals.min():.2f} ms")
            col4.metric("Max", f"{img_intervals.max():.2f} ms")

            if nominal_img:
                bad_mask = np.abs(img_intervals - nominal_img) > 1.0  # >1ms off
                n_bad = bad_mask.sum()
                if n_bad > 0:
                    st.warning(f"{n_bad} imaging frame intervals deviate >1ms from nominal")
                else:
                    st.success("All imaging frame intervals within 1ms of nominal")

        st.subheader("Light On/Off Timeline")
        if light_on is not None and light_off is not None and len(light_on) > 0:
            import plotly.graph_objects as go

            fig3 = go.Figure()

            # Draw light-on periods as rectangles
            for i in range(len(light_on)):
                off_time = light_off[i] if i < len(light_off) else (cam_times[-1] if cam_times is not None else light_on[i] + 60)
                fig3.add_vrect(
                    x0=light_on[i], x1=off_time,
                    fillcolor="yellow", opacity=0.3,
                    line_width=0,
                )

            # Mark on/off edges
            fig3.add_trace(go.Scatter(
                x=light_on, y=[1] * len(light_on),
                mode="markers", name="Light ON",
                marker=dict(color="orange", size=6),
            ))
            fig3.add_trace(go.Scatter(
                x=light_off, y=[0] * len(light_off),
                mode="markers", name="Light OFF",
                marker=dict(color="blue", size=6),
            ))
            fig3.update_layout(
                xaxis_title="Time (s)",
                yaxis_title="Light state",
                yaxis=dict(range=[-0.5, 1.5], tickvals=[0, 1], ticktext=["OFF", "ON"]),
                height=250,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Light interval stats
            if len(light_on) > 1:
                on_intervals = np.diff(light_on)
                st.markdown(f"**Light cycle period:** {on_intervals.mean():.1f}s "
                            f"(expected ~120s for 1min on / 1min off)")


with tab_quality:
    st.subheader("Sync Quality Checks")

    if timestamps is None:
        st.info("No timestamps data available. Run Stage 0 first.")
    else:
        from hm2p.sync.validate import Status, validate_timestamps

        cam_times = timestamps.get("frame_times_camera")
        img_times = timestamps.get("frame_times_imaging")
        fps_cam = timestamps.get("_attr_fps_camera")
        fps_img = timestamps.get("_attr_fps_imaging")

        # Get Suite2p frame count if available
        n_tiff_frames = None
        s2p_ops_key = f"ca_extraction/{sub}/{ses}/suite2p/plane0/ops.npy"
        ops = download_s3_numpy(DERIVATIVES_BUCKET, s2p_ops_key, allow_pickle=True)
        if ops is not None:
            ops = ops.item() if hasattr(ops, "item") else ops
            n_tiff_frames = ops.get("nframes")

        # Build timestamps dict without attr keys
        ts_data = {k: v for k, v in timestamps.items() if not k.startswith("_attr_")}

        results = validate_timestamps(
            ts_data,
            fps_camera=float(fps_cam) if fps_cam is not None else None,
            fps_imaging=float(fps_img) if fps_img is not None else None,
            n_tiff_frames=n_tiff_frames,
        )

        for r in results:
            icon = {
                Status.OK: ":white_check_mark:",
                Status.WARN: ":warning:",
                Status.ERROR: ":x:",
                Status.SKIP: ":fast_forward:",
            }.get(r.status, "")

            with st.expander(f"{icon} **{r.name}** — {r.message}", expanded=(r.status != Status.OK)):
                if r.details:
                    for k, v in r.details.items():
                        if isinstance(v, float):
                            st.text(f"  {k}: {v:.4f}")
                        else:
                            st.text(f"  {k}: {v}")

                if r.status == Status.OK:
                    st.success(r.message)
                elif r.status == Status.WARN:
                    st.warning(r.message)
                elif r.status == Status.ERROR:
                    st.error(r.message)


with tab_batch:
    st.subheader("Batch Sync Validation")
    st.caption("Runs validation checks across all sessions with timestamps.h5")

    if st.button("Run batch validation", key="batch_validate"):
        from hm2p.sync.validate import Status, validate_timestamps

        all_results = {}
        progress = st.progress(0, text="Checking sessions...")

        for i, exp in enumerate(experiments):
            exp_id = exp["exp_id"]
            s, ss = parse_session_id(exp_id)
            ts_key = f"movement/{s}/{ss}/timestamps.h5"
            ts = load_h5_from_s3(DERIVATIVES_BUCKET, ts_key)

            progress.progress((i + 1) / len(experiments), text=f"Checking {exp_id}...")

            if ts is None:
                all_results[exp_id] = [{"name": "timestamps", "status": "missing", "message": "No timestamps.h5"}]
                continue

            ts_data = {k: v for k, v in ts.items() if not k.startswith("_attr_")}
            fps_c = ts.get("_attr_fps_camera")
            fps_i = ts.get("_attr_fps_imaging")

            # Get Suite2p frame count
            n_tiff = None
            ops_key = f"ca_extraction/{s}/{ss}/suite2p/plane0/ops.npy"
            ops_data = download_s3_numpy(DERIVATIVES_BUCKET, ops_key, allow_pickle=True)
            if ops_data is not None:
                ops_dict = ops_data.item() if hasattr(ops_data, "item") else ops_data
                n_tiff = ops_dict.get("nframes")

            results = validate_timestamps(
                ts_data,
                fps_camera=float(fps_c) if fps_c is not None else None,
                fps_imaging=float(fps_i) if fps_i is not None else None,
                n_tiff_frames=n_tiff,
            )
            all_results[exp_id] = [
                {"name": r.name, "status": r.status.value, "message": r.message}
                for r in results
            ]

        progress.empty()

        # Summary table
        st.markdown("### Results")

        # Count issues
        n_ok = 0
        n_warn = 0
        n_error = 0
        n_missing = 0

        for exp_id, checks in all_results.items():
            for c in checks:
                if c["status"] == "ok":
                    n_ok += 1
                elif c["status"] == "warn":
                    n_warn += 1
                elif c["status"] == "error":
                    n_error += 1
                elif c["status"] == "missing":
                    n_missing += 1

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("OK", n_ok)
        col2.metric("Warnings", n_warn)
        col3.metric("Errors", n_error)
        col4.metric("Missing", n_missing)

        # Show sessions with issues
        issues = {
            exp_id: [c for c in checks if c["status"] in ("warn", "error")]
            for exp_id, checks in all_results.items()
            if any(c["status"] in ("warn", "error") for c in checks)
        }

        missing = [
            exp_id for exp_id, checks in all_results.items()
            if any(c["status"] == "missing" for c in checks)
        ]

        if missing:
            st.warning(f"{len(missing)} sessions missing timestamps.h5")
            with st.expander("Missing sessions"):
                for exp_id in missing:
                    st.text(f"  {exp_id}")

        if issues:
            st.warning(f"{len(issues)} sessions with validation issues")
            for exp_id, checks in issues.items():
                with st.expander(f"{exp_id}"):
                    for c in checks:
                        icon = ":warning:" if c["status"] == "warn" else ":x:"
                        st.markdown(f"{icon} **{c['name']}**: {c['message']}")
        elif not missing:
            st.success("All sessions pass validation")


with tab_raw:
    st.subheader("Raw Data Files")

    if raw_files:
        st.markdown(f"**{len(raw_files)}** files in `s3://{RAWDATA_BUCKET}/{rawdata_prefix}`")

        # Categorize
        categories = {
            "TDMS (DAQ)": [f for f in raw_files if f["key"].endswith(".tdms")],
            "TIFF (2P)": [f for f in raw_files if f["key"].endswith((".tif", ".tiff"))],
            "Video": [f for f in raw_files if f["key"].endswith((".mp4", ".avi"))],
            "Metadata": [f for f in raw_files if f["key"].endswith((".txt", ".ini", ".csv"))],
            "Other": [f for f in raw_files if not f["key"].endswith((".tdms", ".tif", ".tiff", ".mp4", ".avi", ".txt", ".ini", ".csv"))],
        }

        for cat, files in categories.items():
            if files:
                with st.expander(f"{cat} ({len(files)} files)"):
                    for f in files:
                        name = f["key"].split("/")[-1]
                        st.text(f"  {name} ({f['size_mb']:.1f} MB)")
    else:
        st.info(f"No raw data found at `s3://{RAWDATA_BUCKET}/{rawdata_prefix}`")

    # Also show derivatives for this session
    st.subheader("Derivatives Files")

    # Check multiple derivative prefixes
    for stage, label in [
        ("movement", "Timestamps"),
        ("ca_extraction", "Suite2p output"),
        ("calcium", "Calcium (ca.h5)"),
        ("sync", "Sync"),
    ]:
        deriv_files = list_s3_session_files(DERIVATIVES_BUCKET, f"{stage}/{sub}/{ses}/")
        if deriv_files:
            with st.expander(f"{label} ({len(deriv_files)} files)"):
                for f in deriv_files:
                    name = f["key"].split("/")[-1]
                    st.text(f"  {name} ({f['size_mb']:.1f} MB)")
