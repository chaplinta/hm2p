"""DLC Viewer — labelled video playback + frame-by-frame inspection."""

from __future__ import annotations

import io
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    download_s3_bytes,
    load_animals,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.dlc_viewer")

BODYPARTS = [
    "nose_tip", "nose",  # nose_tip (finetuned) or nose (SuperAnimal)
    "left_ear", "right_ear",
    "implant_base_rear", "neck",
    "mid_back", "mouse_center", "tail_base",
]
BP_HEX = {
    "nose_tip": "#FF0000",
    "nose": "#FF0000",
    "left_ear": "#0000FF",
    "right_ear": "#00FFFF",
    "implant_base_rear": "#FFA500",
    "neck": "#800080",
    "mid_back": "#00CC00",
    "mouse_center": "#FFD700",
    "tail_base": "#FF00FF",
}
# Display names for the legend (skip "nose" alias if nose_tip exists)
BP_LEGEND = {
    "nose_tip": ("Nose tip", "#FF0000"),
    "left_ear": ("Left ear", "#0000FF"),
    "right_ear": ("Right ear", "#00FFFF"),
    "implant_base_rear": ("Implant base", "#FFA500"),
    "neck": ("Neck", "#800080"),
    "mid_back": ("Mid back", "#00CC00"),
    "mouse_center": ("Mouse centre", "#FFD700"),
    "tail_base": ("Tail base", "#FF00FF"),
}
VIDEO_FPS = 30

st.title("DLC Viewer")
st.caption("Labelled video playback + frame-by-frame inspection for QC.")

# Colour legend for bodyparts
_legend_html = " &nbsp; ".join(
    f'<span style="color:{color}; font-weight:bold;">●</span> {label}'
    for label, color in BP_LEGEND.values()
)
st.markdown(_legend_html, unsafe_allow_html=True)

# ── Cached loaders ───────────────────────────────────────────────────────


VIDEO_FILENAMES = {
    "DLC raw": "labelled_30fps.mp4",
    "DLC median filtered": "labelled_median_30fps.mp4",
    "Pipeline filtered": "labelled_pipeline_30fps.mp4",
}


@st.cache_data(ttl=3600, show_spinner="Downloading labelled video...")
def dl_video(sub: str, ses: str, mode: str = "DLC raw") -> bytes | None:
    fname = VIDEO_FILENAMES.get(mode, "labelled_30fps.mp4")
    data = download_s3_bytes(DERIVATIVES_BUCKET, f"pose/{sub}/{ses}/{fname}")
    if data is None and mode != "DLC raw":
        # Fall back to raw if filtered version doesn't exist yet
        data = download_s3_bytes(DERIVATIVES_BUCKET, f"pose/{sub}/{ses}/labelled_30fps.mp4")
    return data


@st.cache_data(ttl=3600, show_spinner="Downloading DLC .h5...")
def dl_dlc(sub: str, ses: str) -> dict | None:
    """Download DLC .h5 from S3 and load via movement.

    Returns a dict with keys:
        - position: np.ndarray (time, space, keypoints) — x/y coordinates
        - confidence: np.ndarray (time, keypoints) — likelihood values
        - keypoints: list[str] — bodypart names
        - n_frames: int — number of frames
    Or None if no DLC data found.
    """
    import boto3

    s3 = boto3.client("s3", region_name="ap-southeast-2")
    prefix = f"pose/{sub}/{ses}/"
    h5_key = None
    try:
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=DERIVATIVES_BUCKET, Prefix=prefix
        ):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                nm = k.split("/")[-1]
                if k.endswith(".h5") and "_single" not in nm and "_filtered" not in nm:
                    h5_key = k
                    break
    except Exception:
        return None
    if h5_key is None:
        return None
    data = download_s3_bytes(DERIVATIVES_BUCKET, h5_key)
    if data is None:
        return None

    # Write to temp file and load with movement
    from movement.io import load_poses

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        ds = load_poses.from_file(
            file=Path(tmp.name), source_software="DeepLabCut", fps=VIDEO_FPS,
        )

    # For multi-animal DLC, pick best individual per frame (highest mean confidence)
    individuals = ds.position.coords["individuals"].values.tolist()
    keypoints = ds.position.coords["keypoints"].values.tolist()
    n_time = ds.sizes["time"]

    if len(individuals) > 1:
        # confidence: (time, keypoints, individuals)
        conf = ds.confidence.values  # (time, keypoints, individuals)
        # Mean confidence per individual per frame: (time, individuals)
        mean_conf = np.nanmean(conf, axis=1)
        best_ind = np.argmax(mean_conf, axis=1)  # (time,)

        # position: (time, space, keypoints, individuals)
        pos = ds.position.values  # (time, space, keypoints, individuals)
        # Select best individual per frame
        pos_best = np.empty((n_time, 2, len(keypoints)), dtype=np.float64)
        conf_best = np.empty((n_time, len(keypoints)), dtype=np.float64)
        for t in range(n_time):
            j = best_ind[t]
            pos_best[t] = pos[t, :, :, j]
            conf_best[t] = conf[t, :, j]
    else:
        # Single individual — squeeze out individuals dim
        pos_best = ds.position.isel(individuals=0).values  # (time, space, keypoints)
        conf_best = ds.confidence.isel(individuals=0).values  # (time, keypoints)

    # Filter keypoints to known bodyparts (keep order)
    bp_avail = [b for b in BODYPARTS if b in keypoints]
    if not bp_avail:
        bp_avail = keypoints
    bp_indices = [keypoints.index(b) for b in bp_avail]

    return {
        "position": pos_best[:, :, bp_indices],  # (time, space=[x,y], keypoints)
        "confidence": conf_best[:, bp_indices],    # (time, keypoints)
        "keypoints": bp_avail,
        "n_frames": n_time,
        "ds": ds,  # Keep full Dataset for filtering
    }


@st.cache_data(ttl=3600, show_spinner="Applying movement median filter...")
def get_median_filtered(sub: str, ses: str) -> dict | None:
    """Apply movement's rolling median filter to DLC data.

    Returns same dict structure as dl_dlc() but with filtered positions,
    or None if DLC data unavailable.
    """
    dlc = dl_dlc(sub, ses)
    if dlc is None:
        return None

    from movement.filtering import filter_by_confidence, rolling_filter

    ds = dlc["ds"]
    # Filter low-confidence detections (set to NaN)
    filtered_pos = filter_by_confidence(
        data=ds.position, confidence=ds.confidence, threshold=0.5,
    )
    # Apply rolling median filter (window=5, matching pipeline)
    filtered_pos = rolling_filter(data=filtered_pos, window=5, statistic="median")

    # Extract same way as dl_dlc
    individuals = ds.position.coords["individuals"].values.tolist()
    keypoints = ds.position.coords["keypoints"].values.tolist()
    n_time = ds.sizes["time"]

    if len(individuals) > 1:
        conf = ds.confidence.values
        mean_conf = np.nanmean(conf, axis=1)
        best_ind = np.argmax(mean_conf, axis=1)

        pos = filtered_pos.values
        conf_vals = ds.confidence.values
        pos_best = np.empty((n_time, 2, len(keypoints)), dtype=np.float64)
        conf_best = np.empty((n_time, len(keypoints)), dtype=np.float64)
        for t in range(n_time):
            j = best_ind[t]
            pos_best[t] = pos[t, :, :, j]
            conf_best[t] = conf_vals[t, :, j]
    else:
        pos_best = filtered_pos.isel(individuals=0).values
        conf_best = ds.confidence.isel(individuals=0).values

    bp_avail = [b for b in BODYPARTS if b in keypoints]
    if not bp_avail:
        bp_avail = keypoints
    bp_indices = [keypoints.index(b) for b in bp_avail]

    return {
        "position": pos_best[:, :, bp_indices],
        "confidence": conf_best[:, bp_indices],
        "keypoints": bp_avail,
        "n_frames": n_time,
    }


@st.cache_data(ttl=3600, show_spinner="Loading kinematics...")
def dl_kinematics(sub: str, ses: str) -> dict | None:
    """Load kinematics.h5 from sync/ for filtered positions."""
    key = f"sync/{sub}/{ses}/sync.h5"
    data = download_s3_bytes(DERIVATIVES_BUCKET, key)
    if data is None:
        return None
    import h5py
    result = {}
    with h5py.File(io.BytesIO(data), "r") as f:
        for k in ["x_mm", "y_mm", "hd_deg", "speed_cm_s", "frame_times"]:
            if k in f:
                result[k] = f[k][:]
    return result if result else None


@st.cache_data(ttl=3600, show_spinner=False)
def _cache_video_path(vbytes: bytes) -> str:
    """Write video bytes to a persistent temp file (cached by content hash)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(vbytes)
    tmp.close()
    return tmp.name


def extract_frame(vbytes: bytes, idx: int):
    import cv2
    p = _cache_video_path(vbytes)
    cap = cv2.VideoCapture(p)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None


def get_xy(dlc_data: dict, bp: str):
    """Extract x, y, confidence arrays for a bodypart from movement-loaded data."""
    keypoints = dlc_data["keypoints"]
    if bp not in keypoints:
        return None, None, None
    ki = keypoints.index(bp)
    x = dlc_data["position"][:, 0, ki]  # (time,) — space dim 0 = x
    y = dlc_data["position"][:, 1, ki]  # (time,) — space dim 1 = y
    lk = dlc_data["confidence"][:, ki]   # (time,)
    return x, y, lk


# ── Reload button ────────────────────────────────────────────────────────

if st.button("Reload video from S3", key="dlcv_reload"):
    dl_video.clear()
    dl_dlc.clear()
    get_median_filtered.clear()
    dl_kinematics.clear()
    _cache_video_path.clear()
    st.cache_data.clear()
    st.rerun()

# ── Session selector ─────────────────────────────────────────────────────

experiments = load_experiments()
animals = load_animals()
amap = {a["animal_id"]: a for a in animals}

opts = []
for exp in experiments:
    eid = exp["exp_id"]
    aid = eid.split("_")[-1]
    ct = amap.get(aid, {}).get("celltype", "?")
    excl = str(exp.get("exclude", "0")).strip()
    lbl = f"{eid} [{ct}]"
    if excl == "1":
        lbl += " [excl]"
    opts.append((lbl, eid))

if not opts:
    st.warning("No sessions.")
    st.stop()

# Controls in main page body (not sidebar)
col_ses, col_mode, col_pos = st.columns([3, 1, 2])
with col_ses:
    sel = st.selectbox("Session", [o[0] for o in opts], key="dlcv_s")
with col_mode:
    mode = st.radio("Mode", ["Playback", "Inspect"], key="dlcv_m")
with col_pos:
    pos_source = st.radio(
        "Positions", ["DLC raw", "DLC median filtered", "Pipeline filtered"],
        index=0, key="dlcv_pos", horizontal=True,
    )

conf_thr = 0.05  # DLC 3.0 PyTorch outputs conservative confidences (~0.1-0.3)

eid = dict(opts)[sel]
sub, ses = parse_session_id(eid)
aid = eid.split("_")[-1]
ct = amap.get(aid, {}).get("celltype", "?")
ct_label = "Penk+" if ct == "penk" else "Penk\u207bCamKII+" if ct == "nonpenk" else ct
st.caption(f"Animal {aid} | {ct_label}")

# ── Load data ────────────────────────────────────────────────────────────

vbytes = dl_video(sub, ses, mode=pos_source)
dlc_data = dl_dlc(sub, ses)
dlc_filtered = (
    get_median_filtered(sub, ses)
    if pos_source == "DLC median filtered"
    else None
)
kin = dl_kinematics(sub, ses) if pos_source == "Pipeline filtered" else None

# Choose which DLC data dict to use for position display
active_dlc = dlc_filtered if dlc_filtered is not None else dlc_data

n_dlc = dlc_data["n_frames"] if dlc_data is not None else 0

# ── Time series builder ──────────────────────────────────────────────────


def make_ts_fig(vline_frame=None, ds_step=50):
    """Build position + confidence time series."""
    fig = go.Figure()
    if dlc_data is None:
        return fig

    n = n_dlc
    step = max(1, ds_step)
    idx = np.arange(0, n, step)
    t = idx / VIDEO_FPS

    # Position traces
    if pos_source == "Pipeline filtered" and kin is not None:
        # Pipeline-filtered positions (from sync.h5, at imaging rate ~9.6Hz)
        sync_t = kin.get("frame_times")
        dlc_t = np.arange(n) / VIDEO_FPS
        for key, label, color in [
            ("x_mm", "x (filtered, mm)", "#00FF00"),
            ("y_mm", "y (filtered, mm)", "#FF8800"),
        ]:
            vals = kin.get(key)
            if vals is not None and sync_t is not None:
                interp = np.interp(dlc_t, sync_t - sync_t[0], vals)
                fig.add_trace(go.Scattergl(
                    x=t, y=interp[idx], mode="lines",
                    line=dict(color=color, width=1.5),
                    name=label,
                ))
    else:
        # DLC raw or DLC median filtered — use active_dlc
        src = active_dlc if active_dlc is not None else dlc_data
        if src is not None:
            for bp in src["keypoints"]:
                x, y, lk = get_xy(src, bp)
                if x is None:
                    continue
                fig.add_trace(go.Scattergl(
                    x=t, y=x[idx], mode="lines",
                    line=dict(color=BP_HEX.get(bp, "gray"), width=1),
                    name=f"{bp} x", legendgroup=bp,
                    visible="legendonly" if bp != "nose" else True,
                ))
                fig.add_trace(go.Scattergl(
                    x=t, y=y[idx], mode="lines",
                    line=dict(color=BP_HEX.get(bp, "gray"), width=1, dash="dot"),
                    name=f"{bp} y", legendgroup=bp,
                    visible="legendonly" if bp != "nose" else True,
                ))

    # Mean confidence (always from raw DLC data)
    if dlc_data is not None:
        mean_lk = np.nanmean(dlc_data["confidence"], axis=1)  # (time,)
        fig.add_trace(go.Scattergl(
            x=t, y=mean_lk[idx] * 100, mode="lines",
            line=dict(color="white", width=2),
            name="confidence (%)",
            yaxis="y2",
        ))
        fig.add_hline(y=conf_thr * 100, line_dash="dash", line_color="red", line_width=1)

    # Vertical line at current frame
    if vline_frame is not None:
        vt = vline_frame / VIDEO_FPS
        fig.add_vline(x=vt, line_color="lime", line_width=2)

    pos_label = "Position (px)"
    if pos_source == "Pipeline filtered":
        pos_label = "Position (mm)"
    elif pos_source == "DLC median filtered":
        pos_label = "Position (px, filtered)"

    fig.update_layout(
        height=400,
        xaxis_title="Time (s)",
        yaxis_title=pos_label,
        yaxis2=dict(
            title="Confidence (%)", overlaying="y", side="right",
            range=[0, 105], showgrid=False,
        ),
        margin=dict(t=10, b=50, l=60, r=60),
        legend=dict(orientation="h", y=-0.2),
        template="plotly_dark",
    )
    return fig


# ── Playback mode ────────────────────────────────────────────────────────

if mode == "Playback":
    if vbytes is not None:
        col_vid, _ = st.columns([1, 1])
        with col_vid:
            st.video(vbytes, format="video/mp4")
    else:
        st.warning("No labelled video. Run `scripts/render_dlc_videos.py` first.")

    st.subheader("Position + Confidence")
    fig = make_ts_fig(ds_step=50)
    st.plotly_chart(fig, use_container_width=True, key="dlcv_ts_play")

    if n_dlc > 0 and dlc_data is not None:
        mean_c = np.nanmean(dlc_data["confidence"], axis=1)
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean confidence", f"{np.mean(mean_c):.3f}")
        c2.metric("Below threshold", f"{(mean_c < conf_thr).mean()*100:.1f}%")
        c3.metric("Frames", f"{n_dlc:,}")

# ── Inspect mode ─────────────────────────────────────────────────────────

if mode == "Inspect":
    if n_dlc == 0 and vbytes is None:
        st.warning("No data for this session.")
        st.stop()

    n_frames = n_dlc if n_dlc > 0 else 0
    if n_frames == 0:
        st.warning("No DLC frames.")
        st.stop()

    fi = st.number_input(
        f"Frame (0-{n_frames-1})", 0, n_frames - 1, 0, 1,
        key="dlcv_f", help="Arrow keys to step",
    )
    st.caption(f"Frame {fi} | t = {fi/VIDEO_FPS:.2f}s | {fi/n_frames*100:.1f}%")

    # Frame image — optionally overlay filtered markers
    if vbytes is not None:
        rgb = extract_frame(vbytes, fi)
        if rgb is not None:
            # If using filtered positions, draw filtered markers on top
            if pos_source != "DLC raw" and active_dlc is not None and fi < active_dlc["n_frames"]:
                import cv2
                frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                h, w = frame_bgr.shape[:2]
                for bp in active_dlc["keypoints"]:
                    x, y, lk = get_xy(active_dlc, bp)
                    if x is None:
                        continue
                    xv, yv, lkv = float(x[fi]), float(y[fi]), float(lk[fi])
                    if np.isnan(xv) or np.isnan(yv):
                        continue
                    # Scale from original DLC pixel coords to displayed frame size
                    # The labelled video is 416x304, DLC coords are in original resolution
                    sx = w / 832.0  # original video width
                    sy = h / 608.0  # original video height
                    px, py = int(xv * sx), int(yv * sy)
                    color_map = {
                        "nose_tip": (0,0,255), "nose": (0,0,255),
                        "left_ear": (255,0,0), "right_ear": (255,255,0),
                        "implant_base_rear": (0,165,255), "neck": (128,0,128),
                        "mid_back": (0,204,0), "mouse_center": (0,215,255),
                        "tail_base": (255,0,255),
                    }
                    bgr = color_map.get(bp, (255,255,255))
                    # Draw filtered marker as a larger ring (distinguishable from baked-in dots)
                    cv2.circle(frame_bgr, (px, py), 6, bgr, 2, cv2.LINE_AA)
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                label = "filtered" if pos_source == "DLC median filtered" else "pipeline"
                st.image(rgb, caption=f"Frame {fi} (rings = {label} positions)", use_container_width=True)
            else:
                st.image(rgb, caption=f"Frame {fi}", use_container_width=True)

    # Keypoint table (always from raw DLC data for QC)
    if dlc_data is not None and fi < dlc_data["n_frames"]:
        import pandas as pd

        rows = []
        for bp in dlc_data["keypoints"]:
            x, y, lk = get_xy(dlc_data, bp)
            if x is None:
                continue
            xv = float(x[fi])
            yv = float(y[fi])
            lkv = float(lk[fi])
            flag = "LOW" if (not np.isnan(lkv) and lkv < conf_thr) else ""
            rows.append({
                "bodypart": bp, "x": round(xv, 1), "y": round(yv, 1),
                "likelihood": round(lkv, 4), "flag": flag,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Time series with vertical line
    st.subheader("Position + Confidence")
    fig = make_ts_fig(vline_frame=fi, ds_step=50)
    st.plotly_chart(fig, use_container_width=True, key="dlcv_ts_insp")

# Footer
st.divider()
st.caption(
    "Playback: native video controls. Inspect: arrow keys on frame input. "
    "Toggle DLC raw / DLC median filtered / pipeline filtered positions in sidebar."
)
