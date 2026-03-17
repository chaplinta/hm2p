"""DLC Viewer — labelled video playback + frame-by-frame inspection."""

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
    download_s3_bytes,
    load_animals,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.dlc_viewer")

BODYPARTS = ["nose", "left_ear", "right_ear", "mid_back", "mouse_center", "tail_base"]
BP_HEX = {
    "nose": "#FF0000", "left_ear": "#0000FF", "right_ear": "#00FFFF",
    "mid_back": "#00CC00", "mouse_center": "#FFD700", "tail_base": "#FF00FF",
}
VIDEO_FPS = 30

st.title("DLC Viewer")
st.caption("Labelled video playback + frame-by-frame inspection for QC.")

# ── Cached loaders ───────────────────────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner="Downloading labelled video...")
def dl_video(sub: str, ses: str) -> bytes | None:
    return download_s3_bytes(DERIVATIVES_BUCKET, f"pose/{sub}/{ses}/labelled_30fps.mp4")


@st.cache_data(ttl=3600, show_spinner="Downloading DLC .h5...")
def dl_dlc(sub: str, ses: str) -> pd.DataFrame | None:
    import boto3
    s3 = boto3.client("s3", region_name="ap-southeast-2")
    prefix = f"pose/{sub}/{ses}/"
    h5_key = None
    try:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=DERIVATIVES_BUCKET, Prefix=prefix):
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
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        df = pd.read_hdf(tmp.name)
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels == 3:
        return df
    if df.columns.nlevels != 4:
        return None
    return _convert_madlc(df)


def _convert_madlc(df: pd.DataFrame) -> pd.DataFrame:
    scorer = df.columns.get_level_values("scorer")[0]
    individuals = df.columns.get_level_values("individuals").unique().tolist()
    avail = df.columns.get_level_values("bodyparts").unique().tolist()
    bps = [b for b in BODYPARTS if b in avail]
    if not bps:
        bps = avail
    n = len(df)
    scores = np.full((n, len(individuals)), -1.0)
    for j, ind in enumerate(individuals):
        lk = []
        for bp in bps:
            try:
                lk.append(df[(scorer, ind, bp, "likelihood")].values)
            except KeyError:
                pass
        if lk:
            scores[:, j] = np.nanmean(np.column_stack(lk), axis=1)
    best = np.argmax(scores, axis=1)
    cols = pd.MultiIndex.from_tuples(
        [(scorer, bp, c) for bp in bps for c in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"],
    )
    out = np.empty((n, len(bps) * 3))
    for i, bp in enumerate(bps):
        for k, coord in enumerate(("x", "y", "likelihood")):
            ci = i * 3 + k
            for j, ind in enumerate(individuals):
                m = best == j
                if m.any():
                    try:
                        out[m, ci] = df.loc[df.index[m], (scorer, ind, bp, coord)].values
                    except KeyError:
                        out[m, ci] = np.nan
    return pd.DataFrame(out, index=df.index, columns=cols)


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


def extract_frame(vbytes: bytes, idx: int):
    import cv2
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(vbytes)
        p = tmp.name
    try:
        cap = cv2.VideoCapture(p)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None
    finally:
        Path(p).unlink(missing_ok=True)


def get_xy(dlc_df: pd.DataFrame, bp: str):
    """Extract x, y arrays for a bodypart from single-animal DLC DataFrame."""
    scorer = dlc_df.columns.get_level_values("scorer")[0]
    try:
        x = dlc_df[(scorer, bp, "x")].values
        y = dlc_df[(scorer, bp, "y")].values
        lk = dlc_df[(scorer, bp, "likelihood")].values
        return x, y, lk
    except KeyError:
        return None, None, None


# ── Sidebar ──────────────────────────────────────────────────────────────

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

with st.sidebar:
    mode = st.radio("Mode", ["Playback", "Inspect"], key="dlcv_m")
    sel = st.selectbox("Session", [o[0] for o in opts], key="dlcv_s")
    conf_thr = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05, key="dlcv_c")
    pos_source = st.radio(
        "Position data",
        ["DLC raw", "Pipeline filtered (sync.h5)"],
        index=0, key="dlcv_pos",
        help="DLC raw = unfiltered keypoint coords from DLC .h5. "
             "Pipeline filtered = median-filtered, confidence-gated, "
             "interpolated positions from the kinematics pipeline.",
    )

eid = dict(opts)[sel]
sub, ses = parse_session_id(eid)

# ── Load data ────────────────────────────────────────────────────────────

vbytes = dl_video(sub, ses)
dlc_df = dl_dlc(sub, ses)
kin = dl_kinematics(sub, ses) if pos_source == "Pipeline filtered (sync.h5)" else None

n_dlc = len(dlc_df) if dlc_df is not None else 0

# ── Time series builder ──────────────────────────────────────────────────


def make_ts_fig(vline_frame=None, ds=50):
    """Build position + confidence time series."""
    fig = go.Figure()
    if dlc_df is None:
        return fig

    n = n_dlc
    step = max(1, ds)
    idx = np.arange(0, n, step)
    t = idx / VIDEO_FPS

    # Position traces
    if pos_source == "DLC raw" or kin is None:
        # Raw DLC positions
        for bp in BODYPARTS:
            x, y, lk = get_xy(dlc_df, bp)
            if x is None:
                continue
            fig.add_trace(go.Scattergl(
                x=t, y=x[idx], mode="lines",
                line=dict(color=BP_HEX.get(bp, "gray"), width=1),
                name=f"{bp} x", legendgroup=bp, visible="legendonly" if bp != "nose" else True,
            ))
            fig.add_trace(go.Scattergl(
                x=t, y=y[idx], mode="lines",
                line=dict(color=BP_HEX.get(bp, "gray"), width=1, dash="dot"),
                name=f"{bp} y", legendgroup=bp, visible="legendonly" if bp != "nose" else True,
            ))
    else:
        # Pipeline-filtered positions (from sync.h5, at imaging rate ~9.6Hz)
        # These are at a different frame rate — resample to 30fps for alignment
        sync_t = kin.get("frame_times")
        dlc_t = np.arange(n) / VIDEO_FPS
        for key, label, color in [
            ("x_mm", "x (filtered, mm)", "#00FF00"),
            ("y_mm", "y (filtered, mm)", "#FF8800"),
        ]:
            vals = kin.get(key)
            if vals is not None and sync_t is not None:
                # Interpolate sync rate → DLC rate for display
                interp = np.interp(dlc_t, sync_t - sync_t[0], vals)
                fig.add_trace(go.Scattergl(
                    x=t, y=interp[idx], mode="lines",
                    line=dict(color=color, width=1.5),
                    name=label,
                ))

    # Mean confidence
    scorer = dlc_df.columns.get_level_values("scorer")[0]
    bps_avail = dlc_df.columns.get_level_values("bodyparts").unique().tolist()
    lk_cols = []
    for bp in [b for b in BODYPARTS if b in bps_avail]:
        try:
            lk_cols.append(dlc_df[(scorer, bp, "likelihood")].values)
        except KeyError:
            pass
    if lk_cols:
        mean_lk = np.nanmean(np.column_stack(lk_cols), axis=1)
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

    fig.update_layout(
        height=400,
        xaxis_title="Time (s)",
        yaxis_title="Position (px)" if pos_source == "DLC raw" else "Position (mm)",
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
        st.video(vbytes, format="video/mp4")
    else:
        st.warning("No labelled video. Run `scripts/render_dlc_videos.py` first.")

    st.subheader("Position + Confidence")
    fig = make_ts_fig(ds=50)
    st.plotly_chart(fig, use_container_width=True, key="dlcv_ts_play")

    if n_dlc > 0 and dlc_df is not None:
        scorer = dlc_df.columns.get_level_values("scorer")[0]
        bps_a = dlc_df.columns.get_level_values("bodyparts").unique().tolist()
        lk_all = []
        for bp in [b for b in BODYPARTS if b in bps_a]:
            try:
                lk_all.append(dlc_df[(scorer, bp, "likelihood")].values)
            except KeyError:
                pass
        if lk_all:
            mean_c = np.nanmean(np.column_stack(lk_all), axis=1)
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
        f"Frame (0–{n_frames-1})", 0, n_frames - 1, 0, 1,
        key="dlcv_f", help="Arrow keys to step",
    )
    st.caption(f"Frame {fi} | t = {fi/VIDEO_FPS:.2f}s | {fi/n_frames*100:.1f}%")

    # Frame image
    if vbytes is not None:
        rgb = extract_frame(vbytes, fi)
        if rgb is not None:
            st.image(rgb, caption=f"Frame {fi}", use_container_width=True)

    # Keypoint table
    if dlc_df is not None and fi < len(dlc_df):
        scorer = dlc_df.columns.get_level_values("scorer")[0]
        bps_a = dlc_df.columns.get_level_values("bodyparts").unique().tolist()
        rows = []
        for bp in [b for b in BODYPARTS if b in bps_a]:
            try:
                x = float(dlc_df.iloc[fi][(scorer, bp, "x")])
                y = float(dlc_df.iloc[fi][(scorer, bp, "y")])
                lk = float(dlc_df.iloc[fi][(scorer, bp, "likelihood")])
            except (KeyError, IndexError):
                x, y, lk = np.nan, np.nan, np.nan
            flag = "LOW" if (not np.isnan(lk) and lk < conf_thr) else ""
            rows.append({"bodypart": bp, "x": round(x, 1), "y": round(y, 1),
                         "likelihood": round(lk, 4), "flag": flag})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Time series with vertical line
    st.subheader("Position + Confidence")
    fig = make_ts_fig(vline_frame=fi, ds=50)
    st.plotly_chart(fig, use_container_width=True, key="dlcv_ts_insp")

# Footer
st.divider()
st.caption(
    "Playback: native video controls. Inspect: arrow keys on frame input. "
    "Toggle DLC raw vs pipeline filtered positions in sidebar."
)
