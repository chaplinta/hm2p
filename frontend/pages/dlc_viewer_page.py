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
    check_stale_data_warning,
    download_s3_bytes,
    get_dlc_champion,
    get_mm_per_pix,
    get_s3_client,
    invalidate_session_cache,
    load_animals,
    load_experiments,
    parse_session_id,
    video_is_current,
)

log = logging.getLogger("hm2p.frontend.dlc_viewer")

BODYPARTS = [
    "nose_tip", "nose",  # nose_tip (finetuned) or nose (SuperAnimal)
    "left_ear", "right_ear",
    "head_midpoint", "implant_base_rear",  # head_midpoint preferred; implant_base_rear is legacy alias
    "neck",
    "mid_back", "mouse_center", "tail_base",
]
# DLC rainbow colormap hex values (matplotlib.cm.rainbow, 8 bodyparts)
BP_HEX = {
    "nose_tip": "#7F00FF",   # purple
    "nose": "#7F00FF",       # alias
    "left_ear": "#376DF8",   # blue
    "right_ear": "#12C7E5",  # cyan
    "head_midpoint": "#5AF8C7",  # aqua
    "implant_base_rear": "#5AF8C7",  # legacy alias
    "neck": "#A4F89E",       # green
    "mid_back": "#ECC76E",   # yellow
    "mouse_center": "#FF6D38",  # orange
    "tail_base": "#FF0000",  # red
}
BP_LEGEND = {
    "nose_tip": ("Nose tip", "#7F00FF"),
    "left_ear": ("Left ear", "#376DF8"),
    "right_ear": ("Right ear", "#12C7E5"),
    "head_midpoint": ("Head midpoint", "#5AF8C7"),
    "implant_base_rear": ("Head midpoint", "#5AF8C7"),  # legacy alias
    "neck": ("Neck", "#A4F89E"),
    "mid_back": ("Mid back", "#ECC76E"),
    "mouse_center": ("Mouse centre", "#FF6D38"),
    "tail_base": ("Tail base", "#FF0000"),
}
VIDEO_FPS = 30

st.title("DLC Viewer")
st.caption("Labelled video playback + frame-by-frame inspection for QC.")

# Colour legend for bodyparts (deduplicate aliases like implant_base_rear)
_seen_labels = set()
_legend_parts = []
for label, color in BP_LEGEND.values():
    if label not in _seen_labels:
        _legend_parts.append(f'<span style="color:{color}; font-weight:bold;">●</span> {label}')
        _seen_labels.add(label)
st.markdown(" &nbsp; ".join(_legend_parts), unsafe_allow_html=True)

# ── Cached loaders ───────────────────────────────────────────────────────


VIDEO_FILENAMES = {
    "DLC raw": "labelled_30fps.mp4",
    "DLC median filtered": "labelled_median_30fps.mp4",
    "Pipeline filtered": "labelled_pipeline_30fps.mp4",
}


@st.cache_data(ttl=3600, show_spinner="Generating video URL...")
def dl_video_url(sub: str, ses: str, mode: str = "DLC raw") -> str | None:
    """Generate a presigned S3 URL for the labelled video.

    Avoids downloading 300-700 MB into memory; lets st.video stream directly.
    """
    fname = VIDEO_FILENAMES.get(mode, "labelled_30fps.mp4")
    key = f"pose/{sub}/{ses}/{fname}"
    try:
        s3 = get_s3_client()
        # Check file exists first
        s3.head_object(Bucket=DERIVATIVES_BUCKET, Key=key)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": DERIVATIVES_BUCKET, "Key": key},
            ExpiresIn=3600,
        )
        return url
    except Exception:
        log.debug("Video not found: s3://%s/%s", DERIVATIVES_BUCKET, key)
        return None


@st.cache_data(ttl=3600, show_spinner="Downloading labelled video...")
def dl_video(sub: str, ses: str, mode: str = "DLC raw") -> bytes | None:
    """Download labelled video bytes (fallback for frame extraction)."""
    fname = VIDEO_FILENAMES.get(mode, "labelled_30fps.mp4")
    return download_s3_bytes(DERIVATIVES_BUCKET, f"pose/{sub}/{ses}/{fname}")


@st.cache_data(ttl=3600, show_spinner="Downloading DLC .h5...")
def dl_dlc(sub: str, ses: str) -> dict | None:
    """Download DLC .h5 from S3 and load via movement.

    Selects the best finetuned model output using
    :func:`hm2p.pose.select.select_best_dlc_h5`.  Falls back to the first
    available .h5 if no finetuned file is found.

    Returns a dict with keys:
        - position: np.ndarray (time, space, keypoints) — x/y coordinates
        - confidence: np.ndarray (time, keypoints) — likelihood values
        - keypoints: list[str] — bodypart names
        - n_frames: int — number of frames
    Or None if no DLC data found.
    """
    from frontend.data import list_s3_session_files  # noqa: PLC0415
    from hm2p.pose.select import select_best_dlc_h5  # noqa: PLC0415

    prefix = f"pose/{sub}/{ses}/"
    files = list_s3_session_files(DERIVATIVES_BUCKET, prefix)
    all_keys = [f["key"] for f in files]
    h5_key = select_best_dlc_h5(all_keys)
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
        # movement ≥0.1.0 renamed 'file' → 'file_path'. Support both.
        import inspect as _inspect
        _sig = _inspect.signature(load_poses.from_file)
        _file_kw = "file_path" if "file_path" in _sig.parameters else "file"
        ds = load_poses.from_file(
            **{_file_kw: Path(tmp.name)}, source_software="DeepLabCut", fps=VIDEO_FPS,
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
        "h5_key": h5_key,  # S3 key for provenance display
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
    # Filter low-confidence detections (set to NaN).
    # DLC 3.x HRNet outputs confidence in ~0.05-0.35 range, not 0-1.
    # Use 0.05 threshold to match the kinematics pipeline (compute.py).
    filtered_pos = filter_by_confidence(
        data=ds.position, confidence=ds.confidence, threshold=0.05,
    )
    # Apply rolling median filter (window=3 at 30fps ≈ 100ms, matching pipeline)
    filtered_pos = rolling_filter(data=filtered_pos, window=3, statistic="median")

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

_top_c1, _top_c2 = st.columns([1, 1])
with _top_c1:
    if st.button("Reload video from S3", key="dlcv_reload"):
        dl_video_url.clear()
        dl_video.clear()
        dl_dlc.clear()
        get_median_filtered.clear()
        dl_kinematics.clear()
        _cache_video_path.clear()
        st.cache_data.clear()
        invalidate_session_cache()
        st.rerun()
with _top_c2:
    if st.button("Sync all rendered videos to local cache", key="dlcv_sync_all"):
        # Bind-mounted from the Mac at ~/Neuro/hm2p-dlc-videos
        # (devcontainer.json). Writes here land directly on the Mac
        # filesystem outside the repo so the videos are easy to open in
        # Quicktime / VLC and don't pollute the project tree.
        _sync_dir = Path("/host-dlc-videos")
        if not _sync_dir.exists():
            st.error(
                f"Local cache directory `{_sync_dir}` is not mounted. "
                "Create `~/Neuro/hm2p-dlc-videos` on the Mac and rebuild the "
                "devcontainer (the bind mount is declared in "
                "`.devcontainer/devcontainer.json`)."
            )
            st.stop()
        _sync_dir.mkdir(parents=True, exist_ok=True)
        s3 = get_s3_client()
        paginator = s3.get_paginator("list_objects_v2")
        _VIDEO_SUFFIXES = (
            "labelled_30fps.mp4",
            "labelled_median_30fps.mp4",
            "labelled_pipeline_30fps.mp4",
        )
        keys: list[tuple[str, int]] = []
        for page in paginator.paginate(Bucket=DERIVATIVES_BUCKET, Prefix="pose/"):
            for obj in page.get("Contents", []):
                if any(obj["Key"].endswith(s) for s in _VIDEO_SUFFIXES):
                    keys.append((obj["Key"], obj["Size"]))
        if not keys:
            st.warning("No labelled video files found under s3://hm2p-derivatives/pose/.")
        else:
            # Filter to videos rendered with the current champion. All three
            # modes (raw, median, pipeline) are rendered together from the same
            # DLC model, so we check the labelled_30fps sidecar for all of them.
            champion = get_dlc_champion()
            # Pre-compute which sessions are current (one check per session)
            _session_current: dict[tuple[str, str], bool] = {}
            sync_keys: list[tuple[str, int]] = []
            stale_keys: list[str] = []
            for key, size in keys:
                # key format: pose/sub-XXX/ses-YYY/<filename>.mp4
                parts = key.split("/")
                _sub, _ses, _fname = parts[1], parts[2], parts[3]
                cache_key = (_sub, _ses)
                if cache_key not in _session_current:
                    _session_current[cache_key] = video_is_current(
                        _sub, _ses, "labelled_30fps.mp4", champion,
                    )
                if _session_current[cache_key]:
                    sync_keys.append((key, size))
                else:
                    stale_keys.append(key[len("pose/"):])
            if stale_keys:
                _examples = ", ".join(stale_keys[:5])
                _suffix = (
                    f" (and {len(stale_keys) - 5} more)"
                    if len(stale_keys) > 5 else ""
                )
                st.warning(
                    f"Skipping {len(stale_keys)} video(s) that don't match the "
                    f"current champion (no sidecar, or sidecar doesn't match): "
                    f"{_examples}{_suffix}.",
                    icon="⚠️",
                )
            if not sync_keys:
                st.error("No champion-current videos to sync.")
            else:
                progress = st.progress(0.0, text=f"Syncing 0/{len(sync_keys)}...")
                total_bytes = 0
                for i, (key, size) in enumerate(sync_keys):
                    # Flat layout: every file lands directly in _sync_dir.
                    # The S3 key is pose/{sub}/{ses}/labelled_30fps.mp4 — all
                    # videos share the same basename, so we rename to
                    # ``{sub}_{ses}_labelled_30fps.mp4`` to keep them
                    # uniquely identifiable in a single folder.
                    parts = key.split("/")
                    _sub, _ses, _fname = parts[1], parts[2], parts[3]
                    flat_name = f"{_sub}_{_ses}_{_fname}"
                    local = _sync_dir / flat_name
                    # Always overwrite — no skip-if-exists check.
                    s3.download_file(DERIVATIVES_BUCKET, key, str(local))
                    total_bytes += size
                    progress.progress(
                        (i + 1) / len(sync_keys),
                        text=f"Syncing {i + 1}/{len(sync_keys)} — {flat_name} ({size / 1024 / 1024:.1f} MB)",
                    )
                progress.empty()
                # The container sees /host-dlc-videos but the user-facing
                # path on macOS is the bind-mount source. Show that one so
                # they can open the folder in Finder directly.
                _mac_path = "~/Neuro/hm2p-dlc-videos"
                st.success(
                    f"Downloaded {len(sync_keys)} videos "
                    f"({total_bytes / 1024 / 1024 / 1024:.2f} GB total) to:\n\n"
                    f"**Mac:** `{_mac_path}`  \n"
                    f"**Container path:** `{_sync_dir}`"
                )

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
mm_per_pix = get_mm_per_pix(sub, ses)
st.caption(f"Animal {aid} | {ct_label}")

with st.expander("Position modes explained"):
    st.markdown(
        "**DLC raw** — Direct DeepLabCut output. Every frame has a prediction "
        "for every body part, regardless of confidence. No filtering or "
        "interpolation. Low-confidence predictions may be inaccurate (wrong "
        "body part, background clutter).\n\n"
        "**DLC median filtered** — A 3-frame rolling median filter is applied "
        "to x/y coordinates after setting low-confidence detections "
        "(likelihood < 0.05) to NaN. The 3-frame window at 30fps "
        "gives ~100ms temporal smoothing (the old 100fps pipeline used 5 "
        "frames = 50ms). This removes single-frame outliers but does not "
        "interpolate across NaN gaps. Frames where a body part was below "
        "threshold will show no position for that part.\n\n"
        "**Pipeline filtered** — The full kinematics pipeline output (from "
        "sync.h5). Processing steps: confidence threshold → linear "
        "interpolation of short gaps (up to 5 frames) → 3-frame rolling "
        "median → orientation rotation → perspective correction → conversion "
        "to mm. Positions are in physical coordinates (mm). Frames with "
        "extended low-confidence runs remain as NaN (not interpolated).\n\n"
        "In Inspect mode, **filled circles** indicate confident detections; "
        "**open circles** indicate the body part was detected but below the "
        "confidence threshold."
    )

# ── Load data ────────────────────────────────────────────────────────────

# Always load labelled_30fps.mp4 — it's the only labelled video that exists.
# The median/pipeline variants are not pre-rendered; position source only
# affects which keypoint data is overlaid in Inspect mode.
# Use presigned URL for playback (avoids 300-700 MB download into memory).
# Only download bytes for Inspect mode frame extraction.
video_url = dl_video_url(sub, ses, mode="DLC raw")
vbytes = None  # loaded lazily only for Inspect mode
dlc_data = dl_dlc(sub, ses)
dlc_filtered = (
    get_median_filtered(sub, ses)
    if pos_source == "DLC median filtered"
    else None
)
kin = dl_kinematics(sub, ses) if pos_source == "Pipeline filtered" else None

# Show staleness warning when displaying pipeline-filtered positions from sync.h5
if pos_source == "Pipeline filtered":
    check_stale_data_warning(stages=["sync", "kinematics"])

# Choose which DLC data dict to use for position display
active_dlc = dlc_filtered if dlc_filtered is not None else dlc_data

n_dlc = dlc_data["n_frames"] if dlc_data is not None else 0

# Show model provenance
if dlc_data is not None and "h5_key" in dlc_data:
    _h5_name = dlc_data["h5_key"].split("/")[-1]
    st.caption(f"**DLC model file:** `{_h5_name}`")

# ── Time series builder ──────────────────────────────────────────────────


def make_ts_fig(vline_frame=None, ds_step=50, mm_per_pix=None):
    """Build position + confidence time series.

    Parameters
    ----------
    vline_frame:
        Frame index at which to draw a vertical marker, or None.
    ds_step:
        Down-sampling step for display (every Nth frame).
    mm_per_pix:
        Scale factor (mm per pixel). When provided and pos_source is
        "DLC raw" or "DLC median filtered", pixel coordinates are
        converted to mm for display.
    """
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
        # DLC raw or DLC median filtered — convert to mm if scale is available
        src = active_dlc if active_dlc is not None else dlc_data
        scale = mm_per_pix if mm_per_pix is not None else 1.0
        if src is not None:
            for bp in src["keypoints"]:
                x, y, lk = get_xy(src, bp)
                if x is None:
                    continue
                fig.add_trace(go.Scattergl(
                    x=t, y=x[idx] * scale, mode="lines",
                    line=dict(color=BP_HEX.get(bp, "gray"), width=1),
                    name=f"{bp} x", legendgroup=bp,
                    visible="legendonly" if bp != "nose" else True,
                ))
                fig.add_trace(go.Scattergl(
                    x=t, y=y[idx] * scale, mode="lines",
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

    if pos_source == "Pipeline filtered":
        pos_label = "Position (mm)"
    elif mm_per_pix is not None:
        suffix = ", filtered" if pos_source == "DLC median filtered" else ""
        pos_label = f"Position (mm{suffix})"
    else:
        suffix = ", filtered" if pos_source == "DLC median filtered" else ""
        pos_label = f"Position (px{suffix})"

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
    if video_url is not None:
        col_vid, _ = st.columns([1, 1])
        with col_vid:
            st.video(video_url, format="video/mp4")
    else:
        st.warning("No labelled video. Run `scripts/render_dlc_videos.py` first.")

    st.subheader("Position + Confidence")
    fig = make_ts_fig(ds_step=50, mm_per_pix=mm_per_pix)
    st.plotly_chart(fig, use_container_width=True, key="dlcv_ts_play")

    if n_dlc > 0 and dlc_data is not None:
        mean_c = np.nanmean(dlc_data["confidence"], axis=1)
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean confidence", f"{np.mean(mean_c):.3f}")
        c2.metric("Below threshold", f"{(mean_c < conf_thr).mean()*100:.1f}%")
        c3.metric("Frames", f"{n_dlc:,}")

# ── Inspect mode ─────────────────────────────────────────────────────────

if mode == "Inspect":
    # Download video bytes for frame extraction (only in Inspect mode)
    if video_url is not None:
        vbytes = dl_video(sub, ses, mode="DLC raw")
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

    # Frame image — overlay body part markers
    if vbytes is not None:
        rgb = extract_frame(vbytes, fi)
        if rgb is not None:
            import cv2

            frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = frame_bgr.shape[:2]
            sx = w / 832.0  # original video width
            sy = h / 608.0  # original video height

            # DLC rainbow colormap (BGR for OpenCV)
            color_map = {
                "nose_tip": (255,0,127), "nose": (255,0,127),
                "left_ear": (248,109,55), "right_ear": (229,199,18),
                "head_midpoint": (199,248,90), "implant_base_rear": (199,248,90),
                "neck": (158,248,164),
                "mid_back": (110,199,236), "mouse_center": (56,109,255),
                "tail_base": (0,0,255),
            }

            # Always draw raw DLC positions: filled = confident, open = low confidence
            if dlc_data is not None and fi < dlc_data["n_frames"]:
                for bp in dlc_data["keypoints"]:
                    x, y, lk = get_xy(dlc_data, bp)
                    if x is None:
                        continue
                    xv, yv, lkv = float(x[fi]), float(y[fi]), float(lk[fi])
                    if np.isnan(xv) or np.isnan(yv):
                        continue
                    px, py = int(xv * sx), int(yv * sy)
                    bgr = color_map.get(bp, (255,255,255))
                    confident = not np.isnan(lkv) and lkv >= conf_thr
                    if confident:
                        cv2.circle(frame_bgr, (px, py), 5, bgr, -1, cv2.LINE_AA)  # filled
                    else:
                        cv2.circle(frame_bgr, (px, py), 5, bgr, 1, cv2.LINE_AA)   # open

            # If using filtered positions, draw them as larger rings on top
            if pos_source != "DLC raw" and active_dlc is not None and fi < active_dlc["n_frames"]:
                for bp in active_dlc["keypoints"]:
                    x, y, lk = get_xy(active_dlc, bp)
                    if x is None:
                        continue
                    xv, yv = float(x[fi]), float(y[fi])
                    if np.isnan(xv) or np.isnan(yv):
                        continue
                    px, py = int(xv * sx), int(yv * sy)
                    bgr = color_map.get(bp, (255,255,255))
                    cv2.circle(frame_bgr, (px, py), 8, bgr, 2, cv2.LINE_AA)

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if pos_source == "DLC raw":
                caption = f"Frame {fi} (filled = confident, open = below threshold)"
            else:
                label = "median filtered" if pos_source == "DLC median filtered" else "pipeline"
                caption = f"Frame {fi} (dots = raw, rings = {label})"
            st.image(rgb, caption=caption, use_container_width=True)

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
            row: dict = {
                "bodypart": bp,
                "x_px": round(xv, 1),
                "y_px": round(yv, 1),
            }
            if mm_per_pix is not None:
                row["x_mm"] = round(xv * mm_per_pix, 2)
                row["y_mm"] = round(yv * mm_per_pix, 2)
            row["likelihood"] = round(lkv, 4)
            row["flag"] = flag
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Time series with vertical line
    st.subheader("Position + Confidence")
    fig = make_ts_fig(vline_frame=fi, ds_step=50, mm_per_pix=mm_per_pix)
    st.plotly_chart(fig, use_container_width=True, key="dlcv_ts_insp")

# Footer
st.divider()
st.caption(
    "Playback: native video controls. Inspect: arrow keys on frame input. "
    "Toggle DLC raw / DLC median filtered / pipeline filtered positions in sidebar."
)
