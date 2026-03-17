"""DLC Viewer — playback labelled videos and inspect individual frames.

Two modes controlled by sidebar toggle:
  - Playback: stream the pre-rendered labelled MP4 via st.video, confidence time series below
  - Inspect: frame-by-frame navigation with arrow keys, keypoint overlay, per-frame table
"""

from __future__ import annotations

import io
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    RAWDATA_BUCKET,
    download_s3_bytes,
    load_animals,
    load_experiments,
    parse_session_id,
    sanitize_error,
)

log = logging.getLogger("hm2p.frontend.dlc_viewer")

# ── Constants ─────────────────────────────────────────────────────────────

BODYPARTS = ["nose", "left_ear", "right_ear", "mid_back", "mouse_center", "tail_base"]

# Colours matching the DLC renderer
BODYPART_COLORS = {
    "nose": "red",
    "left_ear": "blue",
    "right_ear": "cyan",
    "mid_back": "green",
    "mouse_center": "yellow",
    "tail_base": "magenta",
}

# Plotly-friendly hex equivalents
BODYPART_HEX = {
    "nose": "#FF0000",
    "left_ear": "#0000FF",
    "right_ear": "#00FFFF",
    "mid_back": "#00CC00",
    "mouse_center": "#FFD700",
    "tail_base": "#FF00FF",
}

# BGR for cv2 drawing (matching the hex colours above)
BODYPART_BGR = {
    "nose": (0, 0, 255),
    "left_ear": (255, 0, 0),
    "right_ear": (255, 255, 0),
    "mid_back": (0, 204, 0),
    "mouse_center": (0, 215, 255),
    "tail_base": (255, 0, 255),
}

DEFAULT_BGR = (200, 200, 200)
DEFAULT_HEX = "#C8C8C8"

VIDEO_FPS = 30  # labelled videos are subsampled to 30 fps


# ── Cached data loaders ──────────────────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner="Downloading labelled video...")
def _download_labelled_video(sub: str, ses: str) -> bytes | None:
    """Download labelled_30fps.mp4 from S3 as bytes for st.video()."""
    key = f"pose/{sub}/{ses}/labelled_30fps.mp4"
    return download_s3_bytes(DERIVATIVES_BUCKET, key)


@st.cache_data(ttl=3600, show_spinner="Downloading DLC .h5...")
def _download_dlc_h5(sub: str, ses: str) -> pd.DataFrame | None:
    """Download DLC .h5 from S3, convert maDLC to single-animal, return DataFrame.

    Multi-animal DLC format has 4-level columns (scorer/individuals/bodyparts/coords).
    We pick the best individual per frame (highest mean likelihood across target
    bodyparts) and return a standard 3-level DataFrame (scorer/bodyparts/coords).

    Uses the same conversion logic as scripts/run_kpms.py convert_madlc_to_single.
    """
    import boto3

    s3 = boto3.client("s3", region_name="ap-southeast-2")
    prefix = f"pose/{sub}/{ses}/"
    h5_key = None
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=DERIVATIVES_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                name = key.split("/")[-1]
                if key.endswith(".h5"):
                    # Prefer main DLC output, not _single.h5 or _filtered.h5
                    if "DLC" in name and "_single" not in name and "_filtered" not in name:
                        h5_key = key
                        break
                    if h5_key is None:
                        h5_key = key
    except Exception:
        log.exception("Error listing DLC .h5 for %s/%s", sub, ses)
        return None

    if h5_key is None:
        return None

    data = download_s3_bytes(DERIVATIVES_BUCKET, h5_key)
    if data is None:
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            df = pd.read_hdf(tmp.name)
    except Exception:
        log.exception("Failed to read DLC .h5 for %s/%s", sub, ses)
        return None

    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # Already single-animal (3 levels)
    if df.columns.nlevels == 3:
        return df

    if df.columns.nlevels != 4:
        log.error("Expected 3 or 4 column levels, got %d", df.columns.nlevels)
        return None

    # ── Convert maDLC (4-level) to single-animal (3-level) ───────────────
    scorer = df.columns.get_level_values("scorer")[0]
    individuals = df.columns.get_level_values("individuals").unique().tolist()
    available_bps = df.columns.get_level_values("bodyparts").unique().tolist()

    use_bps = [bp for bp in BODYPARTS if bp in available_bps]
    if not use_bps:
        use_bps = available_bps

    n_frames = len(df)

    # Build (n_frames, n_individuals) likelihood matrix
    ind_scores = np.full((n_frames, len(individuals)), -1.0)
    for j, ind in enumerate(individuals):
        lk_cols = []
        for bp in use_bps:
            try:
                lk_cols.append(df[(scorer, ind, bp, "likelihood")].values)
            except KeyError:
                pass
        if lk_cols:
            ind_scores[:, j] = np.nanmean(np.column_stack(lk_cols), axis=1)

    best_ind_idx = np.argmax(ind_scores, axis=1)

    # Build single-animal dataframe by gathering from best individual per frame
    coords = ["x", "y", "likelihood"]
    new_columns = pd.MultiIndex.from_tuples(
        [(scorer, bp, c) for bp in use_bps for c in coords],
        names=["scorer", "bodyparts", "coords"],
    )
    new_data = np.empty((n_frames, len(use_bps) * len(coords)))

    for i, bp in enumerate(use_bps):
        for k, coord in enumerate(coords):
            col_idx = i * len(coords) + k
            for j, ind in enumerate(individuals):
                mask = best_ind_idx == j
                if mask.any():
                    try:
                        new_data[mask, col_idx] = df.loc[
                            df.index[mask], (scorer, ind, bp, coord)
                        ].values
                    except KeyError:
                        new_data[mask, col_idx] = np.nan

    return pd.DataFrame(new_data, index=df.index, columns=new_columns)


# ── Video frame helpers ──────────────────────────────────────────────────


def _extract_frame(video_bytes: bytes, frame_idx: int) -> np.ndarray | None:
    """Extract a single frame from video bytes using cv2. Returns RGB array."""
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@st.cache_data(ttl=3600)
def _get_video_frame_count(video_bytes: bytes) -> int:
    """Get total frame count from video bytes."""
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── DLC data helpers ─────────────────────────────────────────────────────


def _get_confidence_df(dlc_df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-bodypart likelihood columns into a simple DataFrame with mean."""
    scorer = dlc_df.columns.get_level_values("scorer")[0]
    bps = dlc_df.columns.get_level_values("bodyparts").unique().tolist()
    use_bps = [bp for bp in BODYPARTS if bp in bps]

    conf = {}
    for bp in use_bps:
        try:
            conf[bp] = dlc_df[(scorer, bp, "likelihood")].values
        except KeyError:
            pass

    if not conf:
        return pd.DataFrame()

    df = pd.DataFrame(conf)
    df["mean"] = df.mean(axis=1)
    return df


def _get_keypoints_at_frame(dlc_df: pd.DataFrame, frame_idx: int) -> pd.DataFrame:
    """Get x, y, likelihood for each bodypart at a single frame."""
    scorer = dlc_df.columns.get_level_values("scorer")[0]
    bps = dlc_df.columns.get_level_values("bodyparts").unique().tolist()
    use_bps = [bp for bp in BODYPARTS if bp in bps]

    rows = []
    for bp in use_bps:
        try:
            x = float(dlc_df.iloc[frame_idx][(scorer, bp, "x")])
            y = float(dlc_df.iloc[frame_idx][(scorer, bp, "y")])
            lk = float(dlc_df.iloc[frame_idx][(scorer, bp, "likelihood")])
        except (KeyError, IndexError):
            x, y, lk = np.nan, np.nan, np.nan
        rows.append({
            "bodypart": bp,
            "x": round(x, 1) if not np.isnan(x) else np.nan,
            "y": round(y, 1) if not np.isnan(y) else np.nan,
            "likelihood": round(lk, 4) if not np.isnan(lk) else np.nan,
        })
    return pd.DataFrame(rows)


# ── Confidence time series figure ────────────────────────────────────────


def _make_confidence_fig(
    conf_df: pd.DataFrame,
    threshold: float,
    vline_frame: int | None = None,
    downsample: int = 100,
) -> go.Figure:
    """Build Plotly confidence time series. Downsample 100:1 for performance."""
    if conf_df.empty:
        return go.Figure()

    n = len(conf_df)
    step = max(1, downsample)
    idx = np.arange(0, n, step)
    t_sec = idx / VIDEO_FPS

    fig = go.Figure()

    # Per-bodypart traces
    for bp in [c for c in conf_df.columns if c != "mean"]:
        color = BODYPART_HEX.get(bp, DEFAULT_HEX)
        fig.add_trace(go.Scattergl(
            x=t_sec,
            y=conf_df[bp].values[idx],
            mode="lines",
            line=dict(color=color, width=1),
            name=bp,
            opacity=0.5,
        ))

    # Mean confidence (bold white)
    mean_vals = conf_df["mean"].values[idx]
    fig.add_trace(go.Scattergl(
        x=t_sec,
        y=mean_vals,
        mode="lines",
        line=dict(color="white", width=2),
        name="mean",
    ))

    # Threshold line
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="red",
        annotation_text=f"threshold={threshold}",
    )

    # Highlight below-threshold regions
    below = mean_vals < threshold
    if below.any():
        fig.add_trace(go.Scattergl(
            x=t_sec[below],
            y=mean_vals[below],
            mode="markers",
            marker=dict(size=3, color="red", opacity=0.6),
            name="below threshold",
        ))

    # Vertical line for inspect mode
    if vline_frame is not None:
        vline_t = vline_frame / VIDEO_FPS
        fig.add_vline(x=vline_t, line_dash="solid", line_color="lime", line_width=2)

    fig.update_layout(
        height=250,
        xaxis_title="Time (s)",
        yaxis_title="Likelihood",
        yaxis=dict(range=[0, 1.05]),
        margin=dict(t=10, b=40, l=55, r=15),
        legend=dict(orientation="h", y=-0.25),
        template="plotly_dark",
    )
    return fig


# ── Page layout ──────────────────────────────────────────────────────────

st.title("DLC Viewer")
st.caption("View pre-rendered DLC labelled videos and inspect individual frames.")

experiments = load_experiments()
animals = load_animals()
animal_map = {a["animal_id"]: a for a in animals}

# ── Sidebar ──────────────────────────────────────────────────────────────

# Build session options
session_options: list[tuple[str, str]] = []
for exp in experiments:
    exp_id = exp["exp_id"]
    animal_id = exp_id.split("_")[-1]
    celltype = animal_map.get(animal_id, {}).get("celltype", "?")
    exclude = str(exp.get("exclude", "0")).strip()
    label = f"{exp_id}  [{celltype}]"
    if exclude == "1":
        label += " [excluded]"
    session_options.append((label, exp_id))

if not session_options:
    st.warning("No sessions found in experiments.csv.")
    st.stop()

with st.sidebar:
    st.header("DLC Viewer")

    mode = st.radio("Mode", ["Playback", "Inspect"], index=0, key="dlcv_mode")

    selected_label = st.selectbox(
        "Session",
        [s[0] for s in session_options],
        index=0,
        key="dlcv_session",
    )

    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="dlcv_conf",
        help="Frames below this mean confidence are highlighted in red in the time series.",
    )

selected_exp_id = dict(session_options)[selected_label]
sub, ses = parse_session_id(selected_exp_id)

st.subheader(f"Session: {selected_exp_id}")
st.caption(f"`s3://{DERIVATIVES_BUCKET}/pose/{sub}/{ses}/`")

# ── Load data ────────────────────────────────────────────────────────────

video_bytes = _download_labelled_video(sub, ses)
dlc_df = _download_dlc_h5(sub, ses)


# ── Playback mode ────────────────────────────────────────────────────────

if mode == "Playback":
    if video_bytes is not None:
        st.video(video_bytes, format="video/mp4")
        size_mb = len(video_bytes) / (1024 * 1024)
        st.caption(
            f"Video size: {size_mb:.1f} MB | {VIDEO_FPS} fps | "
            "Use browser controls to play/pause/seek."
        )
    else:
        st.warning(
            f"No labelled video found at `pose/{sub}/{ses}/labelled_30fps.mp4`. "
            "Run the DLC labelling step first."
        )

    # Confidence time series below the video
    if dlc_df is not None:
        conf_df = _get_confidence_df(dlc_df)
        if not conf_df.empty:
            st.subheader("Mean confidence across keypoints")
            fig = _make_confidence_fig(conf_df, conf_threshold, downsample=100)
            st.plotly_chart(fig, use_container_width=True, key="dlcv_conf_playback")

            # Summary stats
            mean_conf = conf_df["mean"].mean()
            pct_below = (conf_df["mean"] < conf_threshold).mean() * 100
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean confidence", f"{mean_conf:.3f}")
            c2.metric("Frames below threshold", f"{pct_below:.1f}%")
            c3.metric("Total frames", f"{len(conf_df):,}")
    elif video_bytes is not None:
        st.info("No DLC .h5 file found for this session -- confidence plot unavailable.")


# ── Inspect mode ─────────────────────────────────────────────────────────

elif mode == "Inspect":
    if video_bytes is None and dlc_df is None:
        st.warning("No video or DLC data found for this session.")
        st.stop()

    # Determine frame count
    n_frames = 0
    if dlc_df is not None:
        n_frames = len(dlc_df)
    elif video_bytes is not None:
        try:
            n_frames = _get_video_frame_count(video_bytes)
        except Exception:
            st.error("Could not determine frame count from video.")
            st.stop()

    if n_frames == 0:
        st.warning("No frames available.")
        st.stop()

    frame_idx = st.number_input(
        f"Frame index (0 to {n_frames - 1})",
        min_value=0,
        max_value=n_frames - 1,
        value=0,
        step=1,
        key="dlcv_frame",
        help="Click here then use arrow keys to step through frames.",
    )

    time_s = frame_idx / VIDEO_FPS
    st.caption(f"Frame {frame_idx} / {n_frames - 1} | t = {time_s:.3f} s")

    # ── Frame display + keypoint table side by side ──────────────────────

    col_img, col_table = st.columns([2, 1])

    with col_img:
        if video_bytes is not None:
            try:
                frame_rgb = _extract_frame(video_bytes, frame_idx)
                if frame_rgb is not None:
                    st.image(
                        frame_rgb,
                        caption=f"Frame {frame_idx}",
                        use_container_width=True,
                    )
                else:
                    st.warning(f"Could not extract frame {frame_idx}.")
            except ImportError:
                st.error(
                    "cv2 (opencv-python) is required for inspect mode. "
                    "Install with: pip install opencv-python-headless"
                )
            except Exception as e:
                st.error(f"Error extracting frame: {sanitize_error(str(e))}")
        else:
            st.info("No labelled video available -- showing keypoint table only.")

    with col_table:
        if dlc_df is not None and frame_idx < len(dlc_df):
            kp_df = _get_keypoints_at_frame(dlc_df, frame_idx)
            st.markdown("**Keypoints at this frame:**")

            # Colour-code rows with low confidence
            def _highlight_low_conf(row: pd.Series) -> list[str]:
                lk = row.get("likelihood", 1.0)
                if pd.isna(lk) or lk < conf_threshold:
                    return ["background-color: #ff000030"] * len(row)
                return [""] * len(row)

            styled = kp_df.style.apply(_highlight_low_conf, axis=1).format(
                {"x": "{:.1f}", "y": "{:.1f}", "likelihood": "{:.4f}"},
                na_rep="\u2014",
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Mean confidence for this frame
            mean_lk = kp_df["likelihood"].mean()
            if pd.isna(mean_lk):
                st.caption("No likelihood data for this frame.")
            else:
                status = "below threshold" if mean_lk < conf_threshold else "OK"
                color = "red" if mean_lk < conf_threshold else "green"
                st.markdown(
                    f"Mean likelihood: <span style='color:{color}'>"
                    f"**{mean_lk:.4f}** ({status})</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No DLC .h5 file -- keypoint table unavailable.")

    # ── Confidence time series with vertical line at current frame ────────

    if dlc_df is not None:
        conf_df = _get_confidence_df(dlc_df)
        if not conf_df.empty:
            st.subheader("Confidence time series")
            fig = _make_confidence_fig(
                conf_df,
                conf_threshold,
                vline_frame=frame_idx,
                downsample=100,
            )
            st.plotly_chart(fig, use_container_width=True, key="dlcv_conf_inspect")


# ── Footer ───────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "**Playback mode:** native video player with play/pause/seek. "
    "**Inspect mode:** click the frame index input and use arrow keys to step through frames. "
    "Red highlights in the confidence plot indicate frames below the threshold."
)
