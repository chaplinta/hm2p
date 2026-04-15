"""Training QC page — quality checks on DLC labeled/training data.

Loads all CollectedData_tristan.h5 files from the local DLC labeled-data
directory and runs anatomical consistency checks on the human-labeled keypoints.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from hm2p.pose.quality import (
    body_length_consistency,
    detect_anterior_posterior_violations,
    detect_ear_asymmetry,
    detect_ear_distance_outliers,
    detect_ear_swaps,
    detect_head_midpoint_outside_triangle,
    detect_neck_inside_triangle,
)

log = logging.getLogger("hm2p.frontend.training_qc")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DLC_PROJECT = Path("/workspace/sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20")
LABELED_DATA_DIR = DLC_PROJECT / "labeled-data"
MANIFEST_PATH = DLC_PROJECT / "_retrain_manifest.json"
RETRAIN_FRAMES_DIR = Path("/workspace/metadata/retrain_frames")
VIDEO_META_DIR = Path("/workspace/metadata/video_meta")

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

# Colors matched to dlc_viewer_page.py BP_HEX
# Body axis keypoint pairs for ear swap detection, ordered by preference.
# Anterior → posterior. Falls back through the list until a pair is found
# where both keypoints are available.
BODY_AXIS_PAIRS: list[tuple[str, str]] = [
    ("nose_tip", "head_midpoint"),
    ("nose_tip", "neck"),
    ("nose_tip", "mid_back"),
    ("nose_tip", "mouse_center"),
    ("head_midpoint", "mid_back"),
    ("head_midpoint", "mouse_center"),
    ("head_midpoint", "tail_base"),
    ("neck", "mid_back"),
    ("neck", "mouse_center"),
    ("neck", "tail_base"),
    ("mid_back", "mouse_center"),
    ("mid_back", "tail_base"),
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

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


RETRAIN_FRAMES_DIR = Path("/workspace/metadata/retrain_frames")


@st.cache_data(ttl=60)
def _load_manifest() -> dict:
    """Load _retrain_manifest.json. Returns {} if not found."""
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


@st.cache_data(ttl=60)
def _load_retrain_frame_counts() -> dict[str, int]:
    """Load per-session frame counts from metadata/retrain_frames/*.json.

    Returns mapping from session name (e.g. 'sub-1114353_ses-20210823T165950')
    to the number of frames selected for labeling.
    """
    if not RETRAIN_FRAMES_DIR.exists():
        return {}
    counts = {}
    for f in sorted(RETRAIN_FRAMES_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        frames = data.get("frames", data.get("frame_indices", []))
        counts[f.stem] = len(frames)
    return counts


@st.cache_data(ttl=60)
def _load_all_labeled_data() -> list[dict]:
    """Load all CollectedData_tristan.h5 files from labeled-data directories.

    Returns
    -------
    list[dict]
        Each entry has keys: ``clip``, ``session_id``, ``df``, ``scorer``,
        ``bodyparts``, ``n_rows``, ``n_labeled``.
        Only sessions with at least one labeled frame are included.
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

        if len(df) == 0:
            continue

        if df.columns.nlevels != 3:
            log.warning("Unexpected column levels in %s, skipping", h5_path)
            continue

        scorer = df.columns.get_level_values(0)[0]
        bps = df.columns.get_level_values(1).unique().tolist()

        # Count rows that have at least one non-NaN value
        any_labeled = df.notna().any(axis=1)
        n_labeled = int(any_labeled.sum())
        if n_labeled == 0:
            continue

        records.append(
            {
                "clip": clip_dir.name,
                "session_id": clip_dir.name,  # full folder name
                "df": df,
                "scorer": scorer,
                "bodyparts": bps,
                "n_rows": len(df),
                "n_labeled": n_labeled,
                "mm_per_pix": _get_mm_per_pix_for_clip(clip_dir.name),
            }
        )

    return records


def _extract_xy(
    df: pd.DataFrame, scorer: str, bp: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) arrays for a body part; NaN where unlabeled."""
    try:
        x = df[(scorer, bp, "x")].values.astype(np.float64)
        y = df[(scorer, bp, "y")].values.astype(np.float64)
    except KeyError:
        n = len(df)
        x = np.full(n, np.nan)
        y = np.full(n, np.nan)
    return x, y


def _frame_numbers(df: pd.DataFrame) -> np.ndarray:
    """Extract frame indices from the MultiIndex level 2 ('frame_XXXXXX.png')."""
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
        # e.g. sub-1114353_ses-20210823T165950
        fp = f.stem.split("_")  # ['sub-1114353', 'ses-20210823T165950']
        if len(fp) < 2:
            continue
        f_animal = fp[0].replace("sub-", "")
        f_date = fp[1].replace("ses-", "")[:8]
        if f_animal == animal and f_date == date:
            return fp[0], fp[1]
    return None


def _get_mm_per_pix_for_clip(clip: str) -> float | None:
    """Get mm_per_pix for a labeled-data clip by mapping to its video meta."""
    import configparser

    result = _clip_to_sub_ses(clip)
    if result is None:
        return None
    sub, ses = result
    meta_path = VIDEO_META_DIR / f"{sub}_{ses}_meta.txt"
    if not meta_path.exists():
        return None
    cfg = configparser.ConfigParser()
    cfg.read(meta_path)
    try:
        return float(cfg["scale"]["mm_per_pix"])
    except (KeyError, ValueError):
        return None


def _short_session(clip: str) -> str:
    """Shorten clip name to '<date>_<animal_id>' for display."""
    parts = clip.split("_")
    if len(parts) >= 5:
        return f"{parts[0]}_{parts[4]}"
    return clip[:30]


def _format_flagged(labels: list[str]) -> str:
    """Group flagged frame labels by session into a readable markdown table."""
    from collections import defaultdict
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for lbl in labels:
        parts = lbl.rsplit(" #", 1)
        session = parts[0] if len(parts) == 2 else "unknown"
        frame = f"#{parts[1]}" if len(parts) == 2 else lbl
        grouped[session].append(frame)
    lines = ["| Session | Frames |", "|---------|--------|"]
    for ses in sorted(grouped):
        frames_str = ", ".join(grouped[ses])
        lines.append(f"| {ses} | {frames_str} |")
    return "\n".join(lines)


def _px_to_mm(val_px: float, scale: float | None) -> tuple[float, str]:
    """Convert a pixel value to mm if scale is available.

    Returns (value, unit_label).
    """
    if scale is not None:
        return val_px * scale, "mm"
    return val_px, "px"


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("DLC Training Data QC")
st.caption(
    "Quality checks on human-labeled keypoints in the DLC training set. "
    "Detects anatomical inconsistencies that indicate labeling errors."
)

records = _load_all_labeled_data()

if not records:
    st.info(
        "No labeled data found. Expected .h5 files in "
        f"`{LABELED_DATA_DIR}`."
    )
    st.stop()

manifest = _load_manifest()
retrain_counts = _load_retrain_frame_counts()

# Compute mean mm_per_pix across all sessions for display
_all_scales = [r["mm_per_pix"] for r in records if r["mm_per_pix"] is not None]
pooled_scale: float | None = float(np.mean(_all_scales)) if _all_scales else None

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

st.header("Labeled Frame Overview")

total_selected_sessions = len(retrain_counts)
total_selected_frames = sum(retrain_counts.values())
total_labeled = sum(r["n_labeled"] for r in records)
sessions_with_labels = len(records)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Sessions selected", total_selected_sessions)
col_b.metric("Frames selected", total_selected_frames)
col_c.metric("Sessions labeled", sessions_with_labels)
col_d.metric("Frames labeled", total_labeled)

if sessions_with_labels < total_selected_sessions:
    st.warning(
        f"{total_selected_sessions - sessions_with_labels} sessions have frames "
        f"selected but no labels yet ({total_selected_frames - total_labeled} "
        f"frames remaining)."
    )

# Build summary table from retrain_frames (all sessions), not just labeled ones
summary_rows = []
# Map retrain_frames keys (sub-xxx_ses-xxx) to records using date+animal matching
record_by_retrain_key = {}
for r in records:
    result = _clip_to_sub_ses(r["clip"])
    if result is not None:
        sub, ses = result
        record_by_retrain_key[f"{sub}_{ses}"] = r

for ses_key in sorted(retrain_counts.keys()):
    n_selected = retrain_counts[ses_key]
    r = record_by_retrain_key.get(ses_key)
    n_labeled = r["n_labeled"] if r else 0
    row: dict = {
        "Session": ses_key,
        "Selected": n_selected,
        "Labeled": n_labeled,
        "Status": "Done" if n_labeled >= n_selected else (
            "Partial" if n_labeled > 0 else "Not started"
        ),
    }
    if r:
        df = r["df"]
        scorer = r["scorer"]
        for bp in BODYPARTS:
            try:
                nan_count = int(df[(scorer, bp, "x")].isna().sum())
                row[f"NaN — {bp}"] = nan_count
            except KeyError:
                row[f"NaN — {bp}"] = r["n_rows"]
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Missing labels per body part (aggregate bar chart)
# ---------------------------------------------------------------------------

st.header("Missing Labels by Body Part")

bp_nan_totals: dict[str, int] = {bp: 0 for bp in BODYPARTS}
bp_total_frames: dict[str, int] = {bp: 0 for bp in BODYPARTS}

for r in records:
    df = r["df"]
    scorer = r["scorer"]
    for bp in BODYPARTS:
        if bp not in r["bodyparts"]:
            continue
        try:
            nan_count = int(df[(scorer, bp, "x")].isna().sum())
        except KeyError:
            nan_count = r["n_rows"]
        bp_nan_totals[bp] += nan_count
        bp_total_frames[bp] += r["n_rows"]

bp_pct_missing = {
    bp: (bp_nan_totals[bp] / bp_total_frames[bp] * 100) if bp_total_frames[bp] > 0 else 0.0
    for bp in BODYPARTS
}

fig_missing = go.Figure(
    go.Bar(
        x=list(bp_pct_missing.keys()),
        y=list(bp_pct_missing.values()),
        marker_color=[BP_HEX.get(bp, "#888888") for bp in BODYPARTS],
        text=[f"{v:.1f}%" for v in bp_pct_missing.values()],
        textposition="outside",
    )
)
fig_missing.update_layout(
    title="Percentage of frames with missing label per body part (all sessions)",
    xaxis_title="Body part",
    yaxis_title="Missing (%)",
    yaxis_range=[0, 105],
    height=350,
    margin=dict(t=50, b=40),
)
st.plotly_chart(fig_missing, use_container_width=True, key="missing_bp_chart")

# ---------------------------------------------------------------------------
# Spatial coverage scatter
# ---------------------------------------------------------------------------

st.header("Spatial Coverage")

st.caption(
    "All labeled positions for each body part across all sessions. "
    "Gaps indicate arena regions not represented in the training set."
)

fig_cov = go.Figure()
for bp in BODYPARTS:
    all_x, all_y, all_hover = [], [], []
    for r in records:
        if bp not in r["bodyparts"]:
            continue
        df_r = r["df"]
        any_lab = df_r.notna().any(axis=1)
        df_lab = df_r[any_lab]
        x, y = _extract_xy(df_lab, r["scorer"], bp)
        valid = np.isfinite(x) & np.isfinite(y)
        s = r.get("mm_per_pix")
        short = _short_session(r["clip"])
        frames = _frame_numbers(df_lab)
        for i in np.where(valid)[0]:
            f_label = f"{short} #{i+1}"
            all_hover.append(f_label)
            all_x.append(float(x[i] * s) if s else float(x[i]))
            all_y.append(float(y[i] * s) if s else float(y[i]))

    if not all_x:
        continue

    fig_cov.add_trace(
        go.Scatter(
            x=all_x,
            y=all_y,
            mode="markers",
            marker=dict(size=6, color=BP_HEX.get(bp, "#888888"), opacity=0.7),
            text=all_hover, hoverinfo="text+x+y",
            name=bp,
        )
    )

cov_unit = "mm" if pooled_scale else "px"
fig_cov.update_layout(
    xaxis_title=f"x ({cov_unit})",
    yaxis_title=f"y ({cov_unit})",
    yaxis_autorange="reversed",  # image coordinates: y increases downward
    height=500,
    legend=dict(itemsizing="constant"),
    margin=dict(t=20),
)
st.plotly_chart(fig_cov, use_container_width=True, key="spatial_coverage_chart")

# ---------------------------------------------------------------------------
# Anatomical quality checks — single session
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("All-Session Quality Checks")

st.caption(
    "Anatomical consistency checks pooled across all labeled frames from all sessions."
)

# Pool all labeled data, tracking session + frame for hover labels
all_lx, all_ly, all_rx, all_ry = [], [], [], []
all_ax1x, all_ax1y, all_ax2x, all_ax2y = [], [], [], []
all_hx, all_hy, all_tx, all_ty = [], [], [], []
all_ear_labels: list[str] = []  # hover text for ear data
all_axis_labels: list[str] = []
all_body_labels: list[str] = []
has_all_ears = False
has_all_axis = False
has_all_body = False


for r in records:
    df_r = r["df"]
    sc = r["scorer"]
    bps_r = r["bodyparts"]
    any_lab = df_r.notna().any(axis=1)
    df_lab = df_r[any_lab]
    if len(df_lab) == 0:
        continue

    short = _short_session(r["clip"])
    frames = _frame_numbers(df_lab)
    frame_labels = [f"{short} #{i+1}" for i, f in enumerate(frames)]

    if "left_ear" in bps_r and "right_ear" in bps_r:
        lx, ly = _extract_xy(df_lab, sc, "left_ear")
        rx, ry = _extract_xy(df_lab, sc, "right_ear")
        all_lx.append(lx); all_ly.append(ly)
        all_rx.append(rx); all_ry.append(ry)
        all_ear_labels.extend(frame_labels)
        has_all_ears = True

        # Axis for ear swap
        for bp1, bp2 in BODY_AXIS_PAIRS:
            if bp1 in bps_r and bp2 in bps_r:
                a1x, a1y = _extract_xy(df_lab, sc, bp1)
                a2x, a2y = _extract_xy(df_lab, sc, bp2)
                all_ax1x.append(a1x); all_ax1y.append(a1y)
                all_ax2x.append(a2x); all_ax2y.append(a2y)
                all_axis_labels.extend(frame_labels)
                has_all_axis = True
                break

    if "nose_tip" in bps_r and "tail_base" in bps_r:
        hx, hy = _extract_xy(df_lab, sc, "nose_tip")
        tx, ty = _extract_xy(df_lab, sc, "tail_base")
        all_hx.append(hx); all_hy.append(hy)
        all_tx.append(tx); all_ty.append(ty)
        all_body_labels.extend(frame_labels)
        has_all_body = True

tab_all_ears, tab_all_swap, tab_all_body, tab_all_triangle, tab_all_neck, tab_all_order, tab_all_symmetry = st.tabs(
    ["Ear Distance", "Ear Swap", "Body Length", "Head in Triangle", "Neck in Triangle", "Body Order", "Ear Symmetry"]
)

with tab_all_ears:
    if has_all_ears:
        pool_lx = np.concatenate(all_lx)
        pool_ly = np.concatenate(all_ly)
        pool_rx = np.concatenate(all_rx)
        pool_ry = np.concatenate(all_ry)
        ear_all = detect_ear_distance_outliers(pool_lx, pool_ly, pool_rx, pool_ry)
        med_v, med_u = _px_to_mm(ear_all["median"], pooled_scale)
        mad_v, mad_u = _px_to_mm(ear_all["mad"], pooled_scale)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Median ear distance ({med_u})", f"{med_v:.1f}")
        c2.metric(f"MAD ({mad_u})", f"{mad_v:.1f}")
        c3.metric("Outlier frames", ear_all["n_outliers"])

        dist = ear_all["distance"]
        if pooled_scale is not None:
            dist = dist * pooled_scale
        valid = np.isfinite(dist)
        dist_unit = "mm" if pooled_scale else "px"
        hover_ear = [all_ear_labels[i] for i in range(len(dist)) if valid[i]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=dist[valid].tolist(), mode="markers",
            marker=dict(size=6, color=[
                "#FF0000" if o else "#1f77b4"
                for o in ear_all["is_outlier"][valid]
            ]),
            text=hover_ear, hoverinfo="text+y",
            name="Ear distance",
        ))
        if np.isfinite(ear_all["median"]):
            fig.add_hline(y=med_v, line_dash="dash", line_color="green",
                          annotation_text="Median")
        fig.update_layout(
            xaxis_title="Labeled frame", yaxis_title=f"Distance ({dist_unit})", height=300,
        )
        st.plotly_chart(fig, use_container_width=True, key="ear_dist_all")
    else:
        st.info("No sessions have both left_ear and right_ear labeled.")

with tab_all_swap:
    if has_all_ears and has_all_axis:
        pool_lx = np.concatenate(all_lx)
        pool_ly = np.concatenate(all_ly)
        pool_rx = np.concatenate(all_rx)
        pool_ry = np.concatenate(all_ry)
        pool_a1x = np.concatenate(all_ax1x)
        pool_a1y = np.concatenate(all_ax1y)
        pool_a2x = np.concatenate(all_ax2x)
        pool_a2y = np.concatenate(all_ax2y)
        swap_all = detect_ear_swaps(pool_lx, pool_ly, pool_rx, pool_ry,
                                    pool_a1x, pool_a1y, pool_a2x, pool_a2y)
        c1, c2 = st.columns(2)
        c1.metric("Swapped frames", swap_all["n_swapped"])
        c2.metric("% swapped", f"{swap_all['pct_swapped']*100:.1f}%")

        if swap_all["n_swapped"] == 0:
            st.success("No ear swaps detected across all labeled frames.")
        else:
            flagged_swap = [all_axis_labels[i] for i in range(len(swap_all["is_swapped"])) if swap_all["is_swapped"][i]]
            st.error(f"{swap_all['n_swapped']} frame(s) with swapped ears:")
            st.markdown(_format_flagged(flagged_swap))

        left_s = swap_all["left_sign"]
        valid = np.isfinite(left_s)
        hover_swap = [all_axis_labels[i] for i in range(len(left_s)) if valid[i]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=left_s[valid].tolist(),
            marker_color=["#FF0000" if s else "#1f77b4" for s in swap_all["is_swapped"][valid]],
            text=hover_swap, hoverinfo="text+y",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(
            xaxis_title="Labeled frame", yaxis_title="Cross-product", height=280,
        )
        st.plotly_chart(fig, use_container_width=True, key="ear_swap_all")
    else:
        st.info("Need ears + midline keypoints across labeled sessions.")

with tab_all_body:
    if has_all_body:
        pool_hx = np.concatenate(all_hx)
        pool_hy = np.concatenate(all_hy)
        pool_tx = np.concatenate(all_tx)
        pool_ty = np.concatenate(all_ty)
        bl_all = body_length_consistency(pool_hx, pool_hy, pool_tx, pool_ty)
        bl_med_v, bl_med_u = _px_to_mm(bl_all["median"], pooled_scale)
        bl_mad_v, bl_mad_u = _px_to_mm(bl_all["mad"], pooled_scale)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Median body length ({bl_med_u})", f"{bl_med_v:.1f}")
        c2.metric(f"MAD ({bl_mad_u})", f"{bl_mad_v:.1f}")
        c3.metric("Outlier frames", bl_all["n_outliers"])

        bl_dist = bl_all["length"]
        if pooled_scale is not None:
            bl_dist = bl_dist * pooled_scale
        valid = np.isfinite(bl_dist)
        bl_unit = "mm" if pooled_scale else "px"
        hover_bl = [all_body_labels[i] for i in range(len(bl_dist)) if valid[i]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=bl_dist[valid].tolist(), mode="markers",
            marker=dict(size=6, color=[
                "#FF0000" if o else "#2ca02c"
                for o in bl_all["is_outlier"][valid]
            ]),
            text=hover_bl, hoverinfo="text+y",
            name="Body length",
        ))
        if np.isfinite(bl_all["median"]):
            fig.add_hline(y=bl_med_v, line_dash="dash", line_color="green",
                          annotation_text="Median")
        fig.update_layout(
            xaxis_title="Labeled frame", yaxis_title=f"nose_tip → tail_base ({bl_unit})", height=300,
        )
        st.plotly_chart(fig, use_container_width=True, key="body_len_all")
    else:
        st.info("No sessions have both nose_tip and tail_base labeled.")

with tab_all_triangle:
    st.caption(
        "head_midpoint should lie within the triangle formed by nose_tip, "
        "left_ear, and right_ear. Frames where it falls outside suggest "
        "the head_midpoint label is misplaced."
    )
    # Pool triangle data
    pool_tri_nose_x, pool_tri_nose_y = [], []
    pool_tri_lx, pool_tri_ly = [], []
    pool_tri_rx, pool_tri_ry = [], []
    pool_tri_mx, pool_tri_my = [], []
    pool_tri_labels: list[str] = []
    for r in records:
        sc = r["scorer"]
        bps_r = r["bodyparts"]
        df_r = r["df"]
        any_lab = df_r.notna().any(axis=1)
        df_lab = df_r[any_lab]
        if len(df_lab) == 0:
            continue
        needed = ["nose_tip", "left_ear", "right_ear"]
        midpoint_bp = "head_midpoint" if "head_midpoint" in bps_r else (
            "implant_base_rear" if "implant_base_rear" in bps_r else None
        )
        if not all(bp in bps_r for bp in needed) or midpoint_bp is None:
            continue
        nx, ny = _extract_xy(df_lab, sc, "nose_tip")
        lx, ly = _extract_xy(df_lab, sc, "left_ear")
        rx, ry = _extract_xy(df_lab, sc, "right_ear")
        mx, my = _extract_xy(df_lab, sc, midpoint_bp)
        pool_tri_nose_x.append(nx); pool_tri_nose_y.append(ny)
        pool_tri_lx.append(lx); pool_tri_ly.append(ly)
        pool_tri_rx.append(rx); pool_tri_ry.append(ry)
        pool_tri_mx.append(mx); pool_tri_my.append(my)
        short = _short_session(r["clip"])
        pool_tri_labels.extend([f"{short} #{i+1}" for i in range(len(df_lab))])

    if pool_tri_nose_x:
        tri_result = detect_head_midpoint_outside_triangle(
            np.concatenate(pool_tri_nose_x), np.concatenate(pool_tri_nose_y),
            np.concatenate(pool_tri_lx), np.concatenate(pool_tri_ly),
            np.concatenate(pool_tri_rx), np.concatenate(pool_tri_ry),
            np.concatenate(pool_tri_mx), np.concatenate(pool_tri_my),
        )
        c1, c2 = st.columns(2)
        c1.metric("Outside triangle", tri_result["n_outside"])
        c2.metric("% outside", f"{tri_result['pct_outside']*100:.1f}%")
        if tri_result["n_outside"] == 0:
            st.success("head_midpoint is inside the nose-ears triangle for all labeled frames.")
        else:
            flagged = [pool_tri_labels[i] for i in range(len(tri_result["is_outside"])) if tri_result["is_outside"][i]]
            st.error(f"head_midpoint outside triangle in {tri_result['n_outside']} frame(s):")
            st.markdown(_format_flagged(flagged))
    else:
        st.info("Need nose_tip, left_ear, right_ear, and head_midpoint for this check.")

with tab_all_neck:
    st.caption(
        "The neck should be posterior to the ears — outside the triangle "
        "formed by nose_tip, left_ear, right_ear. If it falls inside, "
        "the neck label may be confused with head_midpoint."
    )
    pool_neck_x, pool_neck_y = [], []
    pool_neck_labels: list[str] = []
    has_neck_data = False
    for r in records:
        sc = r["scorer"]
        bps_r = r["bodyparts"]
        df_r = r["df"]
        any_lab = df_r.notna().any(axis=1)
        df_lab = df_r[any_lab]
        if len(df_lab) == 0:
            continue
        needed = ["nose_tip", "left_ear", "right_ear", "neck"]
        if not all(bp in bps_r for bp in needed):
            continue
        nx, ny = _extract_xy(df_lab, sc, "neck")
        pool_neck_x.append(nx)
        pool_neck_y.append(ny)
        short = _short_session(r["clip"])
        pool_neck_labels.extend([f"{short} #{i+1}" for i in range(len(df_lab))])
        has_neck_data = True

    if has_neck_data and pool_tri_nose_x:
        neck_result = detect_neck_inside_triangle(
            np.concatenate(pool_tri_nose_x), np.concatenate(pool_tri_nose_y),
            np.concatenate(pool_tri_lx), np.concatenate(pool_tri_ly),
            np.concatenate(pool_tri_rx), np.concatenate(pool_tri_ry),
            np.concatenate(pool_neck_x), np.concatenate(pool_neck_y),
        )
        c1, c2 = st.columns(2)
        c1.metric("Neck inside triangle", neck_result["n_inside"])
        c2.metric("% inside", f"{neck_result['pct_inside']*100:.1f}%")
        if neck_result["n_inside"] == 0:
            st.success("Neck is outside the nose-ears triangle for all labeled frames.")
        else:
            flagged_neck = [pool_neck_labels[i] for i in range(len(neck_result["is_inside"])) if neck_result["is_inside"][i]]
            st.error(f"Neck inside nose-ears triangle in {neck_result['n_inside']} frame(s):")
            st.markdown(_format_flagged(flagged_neck))
    else:
        st.info("Need nose_tip, left_ear, right_ear, and neck for this check.")

with tab_all_order:
    st.caption(
        "Checks that body parts are in the correct anterior→posterior order "
        "along the body axis: nose_tip → head_midpoint → neck → mid_back → "
        "mouse_center → tail_base. Violations indicate a body part is labeled "
        "on the wrong side of its neighbour."
    )
    # Pool order data
    pool_order_kps: dict[str, list] = {}
    pool_order_labels: list[str] = []
    order_list = ["nose_tip", "head_midpoint", "neck", "mid_back", "mouse_center", "tail_base"]
    for r in records:
        sc = r["scorer"]
        bps_r = r["bodyparts"]
        df_r = r["df"]
        any_lab = df_r.notna().any(axis=1)
        df_lab = df_r[any_lab]
        if len(df_lab) == 0:
            continue
        avail = [bp for bp in order_list if bp in bps_r]
        if len(avail) < 2:
            continue
        short = _short_session(r["clip"])
        for bp in avail:
            x, y = _extract_xy(df_lab, sc, bp)
            pool_order_kps.setdefault(bp, ([], []))
            pool_order_kps[bp][0].append(x)
            pool_order_kps[bp][1].append(y)
        pool_order_labels.extend([f"{short} #{i+1}" for i in range(len(df_lab))])

    if pool_order_kps:
        concat_kps = {
            bp: (np.concatenate(xs), np.concatenate(ys))
            for bp, (xs, ys) in pool_order_kps.items()
        }
        order_result = detect_anterior_posterior_violations(concat_kps, order=order_list)
        c1, c2 = st.columns(2)
        c1.metric("Frames with ordering violations", order_result["n_violated"])
        c2.metric("% violated", f"{order_result['pct_violated']*100:.1f}%")
        if order_result["violations_per_pair"]:
            st.markdown("**Violations by pair:**")
            for pair, count in sorted(order_result["violations_per_pair"].items(), key=lambda x: -x[1]):
                st.markdown(f"- `{pair}`: {count} frames")
            flagged = [pool_order_labels[i] for i in range(len(order_result["is_violated"])) if order_result["is_violated"][i]]
            st.markdown(_format_flagged(flagged))
        else:
            st.success("All body parts are in correct anterior→posterior order.")
    else:
        st.info("Need at least 2 body parts from the ordering sequence.")

with tab_all_symmetry:
    st.caption(
        "Checks that left and right ears are roughly equidistant from the "
        "body axis. A ratio > 3 (one ear 3x further from the axis than the "
        "other) suggests one ear is misplaced."
    )
    if has_all_ears and has_all_axis:
        pool_lx_s = np.concatenate(all_lx)
        pool_ly_s = np.concatenate(all_ly)
        pool_rx_s = np.concatenate(all_rx)
        pool_ry_s = np.concatenate(all_ry)
        pool_a1x_s = np.concatenate(all_ax1x)
        pool_a1y_s = np.concatenate(all_ax1y)
        pool_a2x_s = np.concatenate(all_ax2x)
        pool_a2y_s = np.concatenate(all_ax2y)
        sym_result = detect_ear_asymmetry(
            pool_lx_s, pool_ly_s, pool_rx_s, pool_ry_s,
            pool_a1x_s, pool_a1y_s, pool_a2x_s, pool_a2y_s,
        )
        c1, c2 = st.columns(2)
        c1.metric("Asymmetric frames", sym_result["n_asymmetric"])
        c2.metric("% asymmetric", f"{sym_result['n_asymmetric'] / max(1, len(pool_lx_s)) * 100:.1f}%")

        ratio = sym_result["ratio"]
        valid = np.isfinite(ratio)
        if valid.any():
            hover_sym = [all_ear_labels[i] for i in range(len(ratio)) if valid[i]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=ratio[valid].tolist(), mode="markers",
                marker=dict(size=6, color=[
                    "#FF0000" if a else "#1f77b4"
                    for a in sym_result["is_asymmetric"][valid]
                ]),
                text=hover_sym, hoverinfo="text+y",
                name="Distance ratio",
            ))
            fig.add_hline(y=3.0, line_dash="dash", line_color="red",
                          annotation_text="Threshold (3x)")
            fig.update_layout(
                xaxis_title="Labeled frame",
                yaxis_title="max(d_left, d_right) / min(...)",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True, key="ear_sym_all")

        if sym_result["n_asymmetric"] == 0:
            st.success("Ears are symmetrically placed for all labeled frames.")
        else:
            flagged = [all_ear_labels[i] for i in range(len(ratio)) if sym_result["is_asymmetric"][i]]
            st.error(f"Asymmetric ears in {sym_result['n_asymmetric']} frame(s):")
            st.markdown(_format_flagged(flagged))
    else:
        st.info("Need ears + midline keypoints for symmetry check.")

st.markdown("---")
st.header("Per-Session Quality Checks")

session_labels = [_short_session(r["clip"]) for r in records]
selected_label = st.selectbox(
    "Select session",
    session_labels,
    key="tqc_session_select",
)
sel_idx = session_labels.index(selected_label)
sel = records[sel_idx]
df_sel = sel["df"]
scorer_sel = sel["scorer"]
bps_sel = sel["bodyparts"]

n_frames_sel = sel["n_labeled"]
st.caption(
    f"Clip: `{sel['clip']}` | Labeled frames: {n_frames_sel} | "
    f"Body parts: {len(bps_sel)}"
)

# Only use labeled (non-NaN) rows for quality checks
any_labeled_mask = df_sel.notna().any(axis=1)
df_labeled = df_sel[any_labeled_mask]

tab_ear_dist, tab_ear_swap, tab_body_len, tab_scatter = st.tabs(
    ["Ear Distance", "Ear Swap", "Body Length", "Frame Scatter"]
)

# ---- Ear distance -------------------------------------------------------
with tab_ear_dist:
    st.subheader("Inter-ear distance consistency")
    st.caption(
        "Ear separation should be near-constant for a rigid mouse head. "
        "Large deviations indicate one ear was misplaced."
    )
    if "left_ear" in bps_sel and "right_ear" in bps_sel:
        lx, ly = _extract_xy(df_labeled, scorer_sel, "left_ear")
        rx, ry = _extract_xy(df_labeled, scorer_sel, "right_ear")
        result = detect_ear_distance_outliers(lx, ly, rx, ry)
        s_scale = sel.get("mm_per_pix")
        ed_med_v, ed_med_u = _px_to_mm(result["median"], s_scale)
        ed_mad_v, ed_mad_u = _px_to_mm(result["mad"], s_scale)

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Median ear distance ({ed_med_u})", f"{ed_med_v:.1f}")
        c2.metric(f"MAD ({ed_mad_u})", f"{ed_mad_v:.1f}")
        c3.metric("Outlier frames", result["n_outliers"])

        frames = _frame_numbers(df_labeled)
        dist = result["distance"]
        if s_scale is not None:
            dist = dist * s_scale
        valid = np.isfinite(dist)
        fig_ed = go.Figure()
        fig_ed.add_trace(
            go.Scatter(
                x=frames[valid].tolist(),
                y=dist[valid].tolist(),
                mode="markers+lines",
                marker=dict(
                    size=8,
                    color=[
                        "#FF0000" if o else "#1f77b4"
                        for o in result["is_outlier"][valid]
                    ],
                ),
                line=dict(width=1, color="#aaaaaa"),
                name="Ear distance",
            )
        )
        if np.isfinite(result["median"]):
            fig_ed.add_hline(
                y=ed_med_v,
                line_dash="dash",
                line_color="green",
                annotation_text="Median",
            )
        ed_unit = "mm" if s_scale else "px"
        fig_ed.update_layout(
            xaxis_title="Frame number",
            yaxis_title=f"Distance ({ed_unit})",
            height=300,
            margin=dict(t=20),
        )
        st.plotly_chart(fig_ed, use_container_width=True, key="ear_dist_sel")
        if result["n_outliers"] > 0:
            outlier_frames = frames[result["is_outlier"]].tolist()
            st.warning(
                f"Outlier frames: {[int(f) for f in outlier_frames if np.isfinite(f)]}"
            )
    else:
        st.info("left_ear and/or right_ear not labeled in this session.")

# ---- Ear swap -----------------------------------------------------------
with tab_ear_swap:
    st.subheader("Ear swap detection")
    st.caption(
        "Checks whether the left ear is on the correct side of the nose-to-implant "
        "body axis. A negative cross-product indicates the ears may be swapped."
    )
    has_ears = "left_ear" in bps_sel and "right_ear" in bps_sel
    axis_bp1, axis_bp2 = None, None
    for bp1, bp2 in BODY_AXIS_PAIRS:
        if bp1 in bps_sel and bp2 in bps_sel:
            axis_bp1, axis_bp2 = bp1, bp2
            break

    if has_ears and axis_bp1 is not None:
        lx, ly = _extract_xy(df_labeled, scorer_sel, "left_ear")
        rx, ry = _extract_xy(df_labeled, scorer_sel, "right_ear")
        ax1x, ax1y = _extract_xy(df_labeled, scorer_sel, axis_bp1)
        ax2x, ax2y = _extract_xy(df_labeled, scorer_sel, axis_bp2)

        swap_result = detect_ear_swaps(lx, ly, rx, ry, ax1x, ax1y, ax2x, ax2y)

        c1, c2, c3 = st.columns(3)
        c1.metric("Swapped frames", swap_result["n_swapped"])
        c2.metric("% swapped", f"{swap_result['pct_swapped'] * 100:.1f}%")
        c3.metric("Body axis", f"{axis_bp1} → {axis_bp2}")

        if swap_result["n_swapped"] == 0:
            st.success("No ear swaps detected in labeled frames.")
        else:
            frames = _frame_numbers(df_labeled)
            swapped_frames = frames[swap_result["is_swapped"]].tolist()
            st.error(
                f"Ear swap detected in {swap_result['n_swapped']} frame(s): "
                f"{[int(f) for f in swapped_frames if np.isfinite(f)]}"
            )
            st.caption(
                "Verify these frames in napari — left/right ears may be reversed. "
                "Correct with `uv run python scripts/interactive_label.py`."
            )

        left_s = swap_result["left_sign"]
        frames = _frame_numbers(df_labeled)
        valid = np.isfinite(left_s)
        fig_swap = go.Figure()
        fig_swap.add_trace(
            go.Bar(
                x=frames[valid].tolist(),
                y=left_s[valid].tolist(),
                marker_color=[
                    "#FF0000" if s else "#1f77b4"
                    for s in swap_result["is_swapped"][valid]
                ],
                name="Left ear cross-product",
            )
        )
        fig_swap.add_hline(y=0, line_dash="dash", line_color="black")
        fig_swap.update_layout(
            xaxis_title="Frame number",
            yaxis_title="Cross-product (+ = left of axis, − = right)",
            height=280,
            margin=dict(t=20),
        )
        st.plotly_chart(fig_swap, use_container_width=True, key="ear_swap_sel")
    else:
        missing = []
        if not has_ears:
            missing.append("left_ear / right_ear")
        if axis_bp1 is None:
            missing.append("midline keypoints (nose_tip, head_midpoint, neck)")
        st.info(f"Ear swap check requires: {', '.join(missing)}.")

# ---- Body length consistency --------------------------------------------
with tab_body_len:
    st.subheader("Body length consistency")
    st.caption(
        "nose_tip → tail_base distance should be roughly constant. "
        "Outliers suggest one of these keypoints is misplaced."
    )
    head_bp = "nose_tip" if "nose_tip" in bps_sel else None
    tail_bp = "tail_base" if "tail_base" in bps_sel else None

    if head_bp and tail_bp:
        hx, hy = _extract_xy(df_labeled, scorer_sel, head_bp)
        tx, ty = _extract_xy(df_labeled, scorer_sel, tail_bp)
        bl_result = body_length_consistency(hx, hy, tx, ty)
        s_scale = sel.get("mm_per_pix")
        bl_s_med_v, bl_s_med_u = _px_to_mm(bl_result["median"], s_scale)
        bl_s_mad_v, bl_s_mad_u = _px_to_mm(bl_result["mad"], s_scale)

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Median body length ({bl_s_med_u})", f"{bl_s_med_v:.1f}")
        c2.metric(f"MAD ({bl_s_mad_u})", f"{bl_s_mad_v:.1f}")
        c3.metric("Outlier frames", bl_result["n_outliers"])

        frames = _frame_numbers(df_labeled)
        bl_dist = bl_result["length"]
        if s_scale is not None:
            bl_dist = bl_dist * s_scale
        valid = np.isfinite(bl_dist)

        fig_bl = go.Figure()
        fig_bl.add_trace(
            go.Scatter(
                x=frames[valid].tolist(),
                y=bl_dist[valid].tolist(),
                mode="markers+lines",
                marker=dict(
                    size=8,
                    color=[
                        "#FF0000" if o else "#2ca02c"
                        for o in bl_result["is_outlier"][valid]
                    ],
                ),
                line=dict(width=1, color="#aaaaaa"),
                name="Body length",
            )
        )
        if np.isfinite(bl_result["median"]):
            fig_bl.add_hline(
                y=bl_s_med_v,
                line_dash="dash",
                line_color="green",
                annotation_text="Median",
            )
        bl_s_unit = "mm" if s_scale else "px"
        fig_bl.update_layout(
            xaxis_title="Frame number",
            yaxis_title=f"nose_tip → tail_base ({bl_s_unit})",
            height=300,
            margin=dict(t=20),
        )
        st.plotly_chart(fig_bl, use_container_width=True, key="body_len_sel")

        if bl_result["n_outliers"] > 0:
            outlier_frames = frames[bl_result["is_outlier"]].tolist()
            st.warning(
                f"Outlier frames: {[int(f) for f in outlier_frames if np.isfinite(f)]}"
            )
    else:
        missing = []
        if not head_bp:
            missing.append("nose_tip")
        if not tail_bp:
            missing.append("tail_base")
        st.info(f"Body length check requires: {', '.join(missing)}.")

# ---- Frame scatter for selected session ---------------------------------
with tab_scatter:
    st.subheader("Labeled keypoint positions")
    st.caption(
        "Positions of all labeled keypoints for this session, "
        "color-coded by body part."
    )
    fig_sc = go.Figure()
    frames = _frame_numbers(df_labeled)

    for bp in BODYPARTS:
        if bp not in bps_sel:
            continue
        x, y = _extract_xy(df_labeled, scorer_sel, bp)
        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            continue
        fig_sc.add_trace(
            go.Scatter(
                x=x[valid].tolist(),
                y=y[valid].tolist(),
                mode="markers+text",
                text=[str(int(f)) for f in frames[valid].tolist()],
                textposition="top center",
                textfont=dict(size=8),
                marker=dict(size=10, color=BP_HEX.get(bp, "#888888")),
                name=bp,
            )
        )
    fig_sc.update_layout(
        xaxis_title="x (pixels)",
        yaxis_title="y (pixels)",
        yaxis_autorange="reversed",
        height=500,
        legend=dict(itemsizing="constant"),
        margin=dict(t=20),
    )
    st.plotly_chart(fig_sc, use_container_width=True, key="frame_scatter_sel")

# ---------------------------------------------------------------------------
# Aggregate quality summary across all sessions
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Aggregate Quality Summary")

st.caption(
    "Ear swap and body length outlier counts pooled across all labeled sessions."
)

agg_rows = []
for r in records:
    df_r = r["df"]
    sc = r["scorer"]
    bps_r = r["bodyparts"]

    any_lab = df_r.notna().any(axis=1)
    df_r_lab = df_r[any_lab]
    if len(df_r_lab) == 0:
        continue

    # Ear swap
    has_ears_r = "left_ear" in bps_r and "right_ear" in bps_r
    axis_r1, axis_r2 = None, None
    for bp1, bp2 in BODY_AXIS_PAIRS:
        if bp1 in bps_r and bp2 in bps_r:
            axis_r1, axis_r2 = bp1, bp2
            break

    n_swapped = 0
    if has_ears_r and axis_r1 is not None:
        lx, ly = _extract_xy(df_r_lab, sc, "left_ear")
        rx, ry = _extract_xy(df_r_lab, sc, "right_ear")
        ax1x, ax1y = _extract_xy(df_r_lab, sc, axis_r1)
        ax2x, ax2y = _extract_xy(df_r_lab, sc, axis_r2)
        sw = detect_ear_swaps(lx, ly, rx, ry, ax1x, ax1y, ax2x, ax2y)
        n_swapped = sw["n_swapped"]

    # Body length outliers
    n_bl_outliers = 0
    if "nose_tip" in bps_r and "tail_base" in bps_r:
        hx, hy = _extract_xy(df_r_lab, sc, "nose_tip")
        tx, ty = _extract_xy(df_r_lab, sc, "tail_base")
        bl = body_length_consistency(hx, hy, tx, ty)
        n_bl_outliers = bl["n_outliers"]

    # Ear distance outliers
    n_ear_outliers = 0
    if "left_ear" in bps_r and "right_ear" in bps_r:
        lx, ly = _extract_xy(df_r_lab, sc, "left_ear")
        rx, ry = _extract_xy(df_r_lab, sc, "right_ear")
        ed = detect_ear_distance_outliers(lx, ly, rx, ry)
        n_ear_outliers = ed["n_outliers"]

    agg_rows.append(
        {
            "Session": _short_session(r["clip"]),
            "Labeled frames": r["n_labeled"],
            "Ear swaps": n_swapped,
            "Ear dist. outliers": n_ear_outliers,
            "Body len. outliers": n_bl_outliers,
            "Any issues": n_swapped + n_ear_outliers + n_bl_outliers,
        }
    )

if agg_rows:
    agg_df = pd.DataFrame(agg_rows)

    # Highlight rows with issues
    def _highlight_issues(row: pd.Series) -> list[str]:
        color = "background-color: #ffcccc" if row["Any issues"] > 0 else ""
        return [color] * len(row)

    st.dataframe(
        agg_df.style.apply(_highlight_issues, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    total_issues = int(agg_df["Any issues"].sum())
    if total_issues == 0:
        st.success("No anatomical inconsistencies detected across all labeled sessions.")
    else:
        st.warning(
            f"{total_issues} potential labeling issue(s) detected across "
            f"{int((agg_df['Any issues'] > 0).sum())} session(s). "
            "Review flagged sessions using the per-session tabs above or the "
            "interactive labeller."
        )
        st.markdown(
            "**To correct labels:** `uv run python scripts/interactive_label.py`"
        )

st.markdown("---")
st.caption("Training Data QC | hm2p v2")
