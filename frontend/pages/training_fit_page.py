"""Training Fit page — compare DLC model predictions against ground-truth labels.

For each labeled training frame, computes Euclidean pixel error between the
human-labeled position (CollectedData_tristan.h5) and the model's prediction
(pose .h5 on S3).  High error indicates either a bad label or a pose the model
finds difficult.
"""

from __future__ import annotations

import contextlib
import logging
import re
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
    get_mm_per_pix,
    get_s3_client,
)

log = logging.getLogger("hm2p.frontend.training_fit")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DLC_PROJECT = Path("/workspace/sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20")
LABELED_DATA_DIR = DLC_PROJECT / "labeled-data"
RETRAIN_FRAMES_DIR = Path("/workspace/metadata/retrain_frames")

BODYPARTS = [
    "nose_tip",
    "left_ear",
    "right_ear",
    "head_midpoint",
    "neck",
    "mid_back",
    "mouse_center",
    "tail_base",
]

BP_HEX: dict[str, str] = {
    "nose_tip": "#FF0000",
    "left_ear": "#0000FF",
    "right_ear": "#00FFFF",
    "head_midpoint": "#FFA500",
    "neck": "#800080",
    "mid_back": "#00CC00",
    "mouse_center": "#FFD700",
    "tail_base": "#FF00FF",
}

# Error threshold in pixels — cells above this are highlighted
ERROR_THRESHOLD_PX = 20

# ---------------------------------------------------------------------------
# Data loading helpers — ground-truth labels
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def _load_all_labeled_data() -> list[dict]:
    """Load all CollectedData_tristan.h5 files from labeled-data directories.

    Returns
    -------
    list[dict]
        Each entry has keys: ``clip``, ``df``, ``scorer``, ``bodyparts``,
        ``n_labeled``.  Only sessions with at least one labeled frame are
        included.
    """
    if not LABELED_DATA_DIR.exists():
        return []

    records = []
    for clip_dir in sorted(LABELED_DATA_DIR.iterdir()):
        if not clip_dir.is_dir():
            continue
        h5_path = clip_dir / "CollectedData_tristan.h5"
        if not h5_path.exists():
            continue
        try:
            df = pd.read_hdf(h5_path)
        except Exception as exc:
            log.warning("Could not load %s: %s", h5_path, exc)
            continue
        if len(df) == 0 or df.columns.nlevels != 3:
            continue
        scorer = df.columns.get_level_values(0)[0]
        bps = df.columns.get_level_values(1).unique().tolist()
        any_labeled = df.notna().any(axis=1)
        n_labeled = int(any_labeled.sum())
        if n_labeled == 0:
            continue
        records.append(
            {
                "clip": clip_dir.name,
                "df": df,
                "scorer": scorer,
                "bodyparts": bps,
                "n_labeled": n_labeled,
            }
        )
    return records


def _frame_numbers(df: pd.DataFrame) -> np.ndarray:
    """Extract integer frame indices from DLC multi-index level 2.

    Index entries look like ``'frame_000606.png'``; this returns the integer
    frame number (e.g. 606) for each row.  Returns NaN where the pattern does
    not match.
    """
    frames = []
    for idx in df.index:
        frame_file = idx[2] if isinstance(idx, tuple) else str(idx)
        m = re.match(r"frame_(\d+)\.png", frame_file)
        frames.append(int(m.group(1)) if m else np.nan)
    return np.array(frames, dtype=float)


def _clip_to_sub_ses(clip: str) -> tuple[str, str] | None:
    """Map a labeled-data clip directory name to (sub, ses).

    Matches on date + animal ID against metadata/retrain_frames/*.json.
    """
    parts = clip.split("_")
    if len(parts) < 5:
        return None
    date = parts[0]
    animal = parts[4].split("-")[0]
    for f in RETRAIN_FRAMES_DIR.glob("*.json"):
        fp = f.stem.split("_")  # ['sub-1114353', 'ses-20210823T165950']
        if len(fp) < 2:
            continue
        f_animal = fp[0].replace("sub-", "")
        f_date = fp[1].replace("ses-", "")[:8]
        if f_animal == animal and f_date == date:
            return fp[0], fp[1]
    return None


def _short_session(clip: str) -> str:
    """Shorten clip name to '<date>_<animal_id>' for display."""
    parts = clip.split("_")
    if len(parts) >= 5:
        return f"{parts[0]}_{parts[4]}"
    return clip[:30]


# ---------------------------------------------------------------------------
# Data loading helpers — DLC predictions from S3
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner="Downloading DLC pose file...")
def _load_dlc_predictions(sub: str, ses: str) -> pd.DataFrame | None:
    """Download and parse DLC pose .h5 from S3.

    Selects the finetuned model output (skips ``_single`` and ``_filtered``
    files, which are multi-animal and post-processed variants respectively).
    For multi-animal outputs, picks the best individual per frame by mean
    likelihood before returning.

    Parameters
    ----------
    sub:
        Subject identifier, e.g. ``"sub-1114353"``.
    ses:
        Session identifier, e.g. ``"ses-20210823T165950"``.

    Returns
    -------
    pd.DataFrame | None
        DataFrame with 3-level MultiIndex columns
        ``(scorer, bodypart, coord)`` and integer positional index
        (0 = frame 0 of the 30 fps subsampled video).  Returns ``None`` if
        no pose file is found on S3.
    """
    s3 = get_s3_client()
    prefix = f"pose/{sub}/{ses}/"
    h5_key: str | None = None
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=DERIVATIVES_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                nm = k.split("/")[-1]
                if k.endswith(".h5") and "_single" not in nm and "_filtered" not in nm:
                    h5_key = k
                    break
            if h5_key:
                break
    except Exception:
        log.exception("Error listing S3 prefix %s", prefix)
        return None

    if h5_key is None:
        log.info("No pose .h5 found for %s/%s", sub, ses)
        return None

    data = download_s3_bytes(DERIVATIVES_BUCKET, h5_key)
    if data is None:
        return None

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        df = pd.read_hdf(tmp.name)

    if not hasattr(df.columns, "get_level_values"):
        return None

    # Handle multi-animal format (4 levels: scorer/individuals/bodyparts/coords)
    if df.columns.nlevels == 4:
        scorer = df.columns.get_level_values(0)[0]
        individuals = df.columns.get_level_values(1).unique().tolist()
        bodyparts = df.columns.get_level_values(2).unique().tolist()
        coords_list = df.columns.get_level_values(3).unique().tolist()

        if "likelihood" in coords_list and len(individuals) > 1:
            lik_stack = []
            for ind in individuals:
                lik_vals = []
                for bp in bodyparts:
                    with contextlib.suppress(KeyError):
                        lik_vals.append(df[(scorer, ind, bp, "likelihood")].values)
                if lik_vals:
                    lik_stack.append(np.nanmean(np.column_stack(lik_vals), axis=1))
                else:
                    lik_stack.append(np.zeros(len(df)))
            best_idx = np.argmax(np.column_stack(lik_stack), axis=1)
        else:
            best_idx = np.zeros(len(df), dtype=int)

        new_data: dict = {}
        for bp in bodyparts:
            for coord in coords_list:
                vals = np.empty(len(df))
                for fi in range(len(df)):
                    try:
                        vals[fi] = df.iloc[fi][(scorer, individuals[best_idx[fi]], bp, coord)]
                    except (KeyError, IndexError):
                        vals[fi] = np.nan
                new_data[(scorer, bp, coord)] = vals
        df = pd.DataFrame(new_data, index=range(len(df)))
        df.columns = pd.MultiIndex.from_tuples(df.columns)
    else:
        # Reset to integer index so iloc == frame number
        df = df.reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Error computation
# ---------------------------------------------------------------------------


def _compute_errors(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    scorer: str,
    bodyparts: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute per-frame, per-bodypart Euclidean pixel error.

    Parameters
    ----------
    gt_df:
        Ground-truth label DataFrame with 3-level MultiIndex columns.
    pred_df:
        DLC prediction DataFrame with integer positional index (frame numbers).
    scorer:
        Scorer name used in gt_df columns.
    bodyparts:
        Body parts to evaluate.

    Returns
    -------
    error_df : pd.DataFrame
        Shape ``(n_labeled_frames, n_bodyparts)`` with pixel error values.
        Index is the integer frame number.
    frame_nums : np.ndarray
        Integer frame numbers corresponding to each row.
    """
    frame_nums = _frame_numbers(gt_df).astype(int)

    pred_scorer = pred_df.columns.get_level_values(0)[0]
    pred_bps = pred_df.columns.get_level_values(1).unique().tolist()

    # Frame indices in labels are from the raw ~100fps video.
    # DLC predictions are on the 30fps subsampled video.
    _RAW_FPS = 100.0
    _DLC_FPS = 30.0

    error_rows = []
    valid_frame_nums = []

    for i, frame_num in enumerate(frame_nums):
        # Convert raw video frame index to DLC 30fps frame index
        dlc_frame = round(frame_num * _DLC_FPS / _RAW_FPS)
        if dlc_frame >= len(pred_df):
            continue

        row: dict[str, float] = {}
        any_valid = False
        for bp in bodyparts:
            # Ground truth
            try:
                gt_x = float(gt_df.iloc[i][(scorer, bp, "x")])
                gt_y = float(gt_df.iloc[i][(scorer, bp, "y")])
            except KeyError:
                row[bp] = np.nan
                continue
            if np.isnan(gt_x) or np.isnan(gt_y):
                row[bp] = np.nan
                continue

            # Prediction — look up by integer frame index
            if bp not in pred_bps:
                row[bp] = np.nan
                continue
            try:
                pred_x = float(pred_df.iloc[dlc_frame][(pred_scorer, bp, "x")])
                pred_y = float(pred_df.iloc[dlc_frame][(pred_scorer, bp, "y")])
            except (KeyError, IndexError):
                row[bp] = np.nan
                continue

            if np.isnan(pred_x) or np.isnan(pred_y):
                row[bp] = np.nan
                continue

            error = float(np.sqrt((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2))
            row[bp] = error
            any_valid = True

        if any_valid:
            error_rows.append(row)
            valid_frame_nums.append(frame_num)

    if not error_rows:
        return pd.DataFrame(columns=bodyparts), np.array([], dtype=int)

    error_df = pd.DataFrame(error_rows, index=valid_frame_nums, columns=bodyparts)
    return error_df, np.array(valid_frame_nums, dtype=int)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("DLC Training Fit")
st.caption(
    "Pixel error between ground-truth labels and DLC model predictions on "
    "training frames. High error may indicate a bad label or a difficult pose."
)

with st.expander("Interpreting errors"):
    st.markdown(
        "**High error for one bodypart** in a frame → likely a labelling "
        "error for that part. Check the per-frame detail below.\n\n"
        "**High error across all bodyparts** → unusual pose the model "
        "hasn't learned, or the frame is ambiguous.\n\n"
        "**Note on occluded labels:** labels follow the SuperAnimal "
        "TopViewMouse convention. Occluded body parts are labelled when "
        "the position can be inferred from the visible anatomy. The model "
        "may show higher error on these frames because it hasn't seen "
        "enough occlusion examples — this is expected and not necessarily "
        "a labelling error."
    )

# Load ground-truth labels
records = _load_all_labeled_data()

if not records:
    st.info(f"No labeled data found. Expected .h5 files in `{LABELED_DATA_DIR}`.")
    st.stop()

# ---------------------------------------------------------------------------
# Session selector
# ---------------------------------------------------------------------------

clip_options = [r["clip"] for r in records]
display_names = {r["clip"]: _short_session(r["clip"]) for r in records}

col_sel, col_thresh = st.columns([3, 1])
with col_sel:
    selected_clips = st.multiselect(
        "Sessions",
        options=clip_options,
        default=clip_options,
        format_func=lambda c: display_names[c],
        help="Select which labeled sessions to include in the analysis.",
    )
with col_thresh:
    threshold_px = st.number_input(
        "Error threshold (px)",
        min_value=1,
        max_value=200,
        value=ERROR_THRESHOLD_PX,
        step=1,
        help="Cells with error above this value are highlighted in the heatmap.",
    )

if not selected_clips:
    st.warning("Select at least one session.")
    st.stop()

# ---------------------------------------------------------------------------
# Compute errors — load pose predictions and compare to GT
# ---------------------------------------------------------------------------

cache_key = "training_fit_errors"

if cache_key not in st.session_state or st.button("Reload from S3"):
    all_error_dfs: list[dict] = []
    progress = st.progress(0.0, text="Loading pose predictions...")
    n = len(selected_clips)
    for i, clip in enumerate(selected_clips):
        progress.progress((i + 0.5) / n, text=f"Loading {display_names[clip]}...")
        result = _clip_to_sub_ses(clip)
        if result is None:
            log.warning("Could not map clip to sub/ses: %s", clip)
            continue
        sub, ses = result
        record = next((r for r in records if r["clip"] == clip), None)
        if record is None:
            continue

        pred_df = _load_dlc_predictions(sub, ses)
        if pred_df is None:
            log.info("No predictions found for %s/%s", sub, ses)
            continue

        mm_per_pix = get_mm_per_pix(sub, ses)
        error_df, frame_nums = _compute_errors(
            gt_df=record["df"],
            pred_df=pred_df,
            scorer=record["scorer"],
            bodyparts=[bp for bp in BODYPARTS if bp in record["bodyparts"]],
        )
        if len(error_df) == 0:
            continue

        all_error_dfs.append(
            {
                "clip": clip,
                "short": display_names[clip],
                "sub": sub,
                "ses": ses,
                "error_df": error_df,
                "frame_nums": frame_nums,
                "mm_per_pix": mm_per_pix,
                "gt_df": record["df"],
                "gt_scorer": record["scorer"],
                "pred_df": pred_df,
            }
        )
        progress.progress((i + 1) / n, text=f"Loaded {display_names[clip]}")

    progress.empty()
    st.session_state[cache_key] = all_error_dfs

all_error_dfs = st.session_state.get(cache_key, [])

# Filter to selected clips
all_error_dfs = [d for d in all_error_dfs if d["clip"] in selected_clips]

if not all_error_dfs:
    st.warning(
        "No predictions found for the selected sessions. "
        "Ensure DLC inference has run and pose .h5 files are on S3."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

st.header("Summary")

# Pool all errors
all_errors_concat = pd.concat([d["error_df"] for d in all_error_dfs], axis=0)
present_bps = [bp for bp in BODYPARTS if bp in all_errors_concat.columns]

total_frames = sum(len(d["error_df"]) for d in all_error_dfs)
frames_above = int((all_errors_concat[present_bps] > threshold_px).any(axis=1).sum())

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Sessions analyzed", len(all_error_dfs))
col_b.metric("Total labeled frames", total_frames)
col_c.metric(f"Frames with any error > {threshold_px}px", frames_above)
col_d.metric(
    "Mean error (all BPs)",
    f"{float(all_errors_concat[present_bps].stack().mean()):.1f} px",
)

# Bar chart — mean error per bodypart
mean_per_bp = {bp: float(all_errors_concat[bp].mean()) for bp in present_bps}
std_per_bp = {bp: float(all_errors_concat[bp].std()) for bp in present_bps}

fig_bar = go.Figure()
fig_bar.add_trace(
    go.Bar(
        x=list(mean_per_bp.keys()),
        y=list(mean_per_bp.values()),
        error_y={"type": "data", "array": list(std_per_bp.values()), "visible": True},
        marker_color=[BP_HEX.get(bp, "#888888") for bp in present_bps],
        text=[f"{v:.1f}" for v in mean_per_bp.values()],
        textposition="outside",
    )
)
fig_bar.add_hline(
    y=threshold_px,
    line_dash="dot",
    line_color="red",
    annotation_text=f"Threshold ({threshold_px} px)",
    annotation_position="right",
)
fig_bar.update_layout(
    title="Mean pixel error per body part (all sessions)",
    xaxis_title="Body part",
    yaxis_title="Mean error (px)",
    showlegend=False,
    height=350,
)
st.plotly_chart(fig_bar, use_container_width=True)

# Worst frames overall
all_rows: list[dict] = []
for d in all_error_dfs:
    err = d["error_df"].copy()
    err["mean_error_px"] = err[present_bps].mean(axis=1)
    err["session"] = d["short"]
    err["frame"] = [f"#{i+1}" for i in range(len(d["frame_nums"]))]
    all_rows.append(err)

worst_df = pd.concat(all_rows, axis=0).sort_values("mean_error_px", ascending=False)
worst_df = worst_df[["session", "frame", "mean_error_px"] + present_bps].head(20)

with st.expander("Worst 20 frames (highest mean error)", expanded=False):
    st.dataframe(
        worst_df.style.format(
            {col: "{:.1f}" for col in ["mean_error_px"] + present_bps}
        ).background_gradient(subset=["mean_error_px"], cmap="YlOrRd"),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# Per-session heatmaps
# ---------------------------------------------------------------------------

st.header("Per-session error heatmaps")
st.caption(
    f"Rows = labeled frames, columns = body parts. "
    f"Color = pixel error. Red cells exceed the threshold of {threshold_px} px."
)

for d in all_error_dfs:
    err = d["error_df"][present_bps]
    frame_labels = [f"#{i+1}" for i in range(len(d["frame_nums"]))]

    # Build colorscale: white → yellow → orange → red
    colorscale = [
        [0.0, "#FFFFFF"],
        [0.3, "#FFFACD"],
        [0.6, "#FFA500"],
        [1.0, "#CC0000"],
    ]

    fig_hm = go.Figure(
        go.Heatmap(
            z=err.values,
            x=present_bps,
            y=frame_labels,
            colorscale=colorscale,
            zmin=0,
            zmax=max(
                threshold_px * 2,
                float(err.max().max()) if not err.empty else threshold_px * 2,
            ),
            colorbar={"title": "Error (px)"},
            hovertemplate="Frame: %{y}<br>Body part: %{x}<br>Error: %{z:.1f} px<extra></extra>",
        )
    )
    fig_hm.update_layout(
        title=d["short"],
        xaxis_title="Body part",
        yaxis_title="Frame",
        height=max(250, len(frame_labels) * 22 + 80),
        margin={"t": 50, "b": 40},
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # Small stats row beneath each heatmap
    n_bad = int((err > threshold_px).any(axis=1).sum())
    mean_err = float(err.stack().mean())
    st.caption(
        f"{d['short']} — {len(err)} frames, "
        f"mean error {mean_err:.1f} px, "
        f"{n_bad} frame(s) with any body part > {threshold_px} px."
    )

# ---------------------------------------------------------------------------
# Per-frame detail
# ---------------------------------------------------------------------------

st.header("Per-frame detail")
st.caption("Select a session and frame to see ground-truth vs predicted positions.")

session_labels = [d["short"] for d in all_error_dfs]
detail_col1, detail_col2 = st.columns(2)

with detail_col1:
    detail_ses_label = st.selectbox(
        "Session",
        options=session_labels,
        key="fit_detail_session",
    )
detail_d = next((d for d in all_error_dfs if d["short"] == detail_ses_label), None)

with detail_col2:
    if detail_d is not None:
        n_frames = len(detail_d["frame_nums"])
        frame_labels_detail = [f"#{i+1}" for i in range(n_frames)]
        detail_frame_label = st.selectbox(
            "Frame",
            options=frame_labels_detail,
            key="fit_detail_frame",
        )
        detail_frame_idx = frame_labels_detail.index(detail_frame_label)
        detail_frame = int(detail_d["frame_nums"][detail_frame_idx])
    else:
        detail_frame = None

if detail_d is not None and detail_frame is not None:
    gt_df = detail_d["gt_df"]
    gt_scorer = detail_d["gt_scorer"]
    pred_df = detail_d["pred_df"]
    pred_scorer = pred_df.columns.get_level_values(0)[0]
    pred_bps_avail = pred_df.columns.get_level_values(1).unique().tolist()
    mm_per_pix = detail_d["mm_per_pix"]

    # Find the row index in gt_df corresponding to this frame number
    all_fn = _frame_numbers(gt_df).astype(int)
    row_indices = np.where(all_fn == detail_frame)[0]

    if len(row_indices) == 0:
        st.warning(f"Frame {detail_frame} not found in ground-truth labels for this session.")
    else:
        row_i = int(row_indices[0])

        fig_detail = go.Figure()

        gt_points: list[dict] = []
        pred_points: list[dict] = []
        vec_data: list[dict] = []

        for bp in present_bps:
            color = BP_HEX.get(bp, "#888888")
            # GT
            try:
                gx = float(gt_df.iloc[row_i][(gt_scorer, bp, "x")])
                gy = float(gt_df.iloc[row_i][(gt_scorer, bp, "y")])
            except KeyError:
                gx, gy = np.nan, np.nan

            # Prediction — convert raw frame index to DLC 30fps index
            _detail_dlc = round(detail_frame * 30.0 / 100.0)
            if bp in pred_bps_avail and _detail_dlc < len(pred_df):
                try:
                    px_ = float(pred_df.iloc[_detail_dlc][(pred_scorer, bp, "x")])
                    py_ = float(pred_df.iloc[_detail_dlc][(pred_scorer, bp, "y")])
                except (KeyError, IndexError):
                    px_, py_ = np.nan, np.nan
            else:
                px_, py_ = np.nan, np.nan

            if not (np.isnan(gx) or np.isnan(gy)):
                gt_points.append({"bp": bp, "x": gx, "y": gy, "color": color})
            if not (np.isnan(px_) or np.isnan(py_)):
                pred_points.append({"bp": bp, "x": px_, "y": py_, "color": color})

            if not any(np.isnan([gx, gy, px_, py_])):
                vec_data.append(
                    {
                        "bp": bp,
                        "color": color,
                        "gx": gx,
                        "gy": gy,
                        "px": px_,
                        "py": py_,
                        "err": np.sqrt((gx - px_) ** 2 + (gy - py_) ** 2),
                    }
                )

        # Plot error vectors
        for v in vec_data:
            fig_detail.add_trace(
                go.Scatter(
                    x=[v["gx"], v["px"]],
                    y=[v["gy"], v["py"]],
                    mode="lines",
                    line={"color": v["color"], "width": 1.5, "dash": "dot"},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # GT markers (filled circle)
        if gt_points:
            fig_detail.add_trace(
                go.Scatter(
                    x=[p["x"] for p in gt_points],
                    y=[p["y"] for p in gt_points],
                    mode="markers",
                    marker={
                        "symbol": "circle",
                        "size": 12,
                        "color": [p["color"] for p in gt_points],
                        "line": {"width": 2, "color": "black"},
                    },
                    text=[p["bp"] for p in gt_points],
                    hovertemplate="%{text}<br>GT: (%{x:.1f}, %{y:.1f})<extra>GT</extra>",
                    name="Ground truth",
                )
            )

        # Predicted markers (X symbol)
        if pred_points:
            fig_detail.add_trace(
                go.Scatter(
                    x=[p["x"] for p in pred_points],
                    y=[p["y"] for p in pred_points],
                    mode="markers",
                    marker={
                        "symbol": "x",
                        "size": 12,
                        "color": [p["color"] for p in pred_points],
                        "line": {"width": 2},
                    },
                    text=[p["bp"] for p in pred_points],
                    hovertemplate="%{text}<br>Pred: (%{x:.1f}, %{y:.1f})<extra>Pred</extra>",
                    name="Predicted",
                )
            )

        fig_detail.update_layout(
            title=f"{detail_ses_label} — frame {detail_frame}",
            xaxis_title="x (px)",
            yaxis_title="y (px)",
            yaxis_autorange="reversed",
            height=450,
            legend={"orientation": "h"},
        )
        st.plotly_chart(fig_detail, use_container_width=True)

        # Error table for this frame
        if vec_data:
            err_table = pd.DataFrame(
                [
                    {
                        "Body part": v["bp"],
                        "GT x (px)": round(v["gx"], 1),
                        "GT y (px)": round(v["gy"], 1),
                        "Pred x (px)": round(v["px"], 1),
                        "Pred y (px)": round(v["py"], 1),
                        "Error (px)": round(v["err"], 1),
                        **(
                            {"Error (mm)": round(v["err"] * mm_per_pix, 2)}
                            if mm_per_pix is not None
                            else {}
                        ),
                    }
                    for v in vec_data
                ]
            )
            st.dataframe(
                err_table.style.background_gradient(subset=["Error (px)"], cmap="YlOrRd"),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No body parts have both ground-truth and predicted positions for this frame.")
