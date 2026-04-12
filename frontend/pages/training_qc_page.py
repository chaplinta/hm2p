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
    detect_ear_distance_outliers,
    detect_ear_swaps,
)

log = logging.getLogger("hm2p.frontend.training_qc")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DLC_PROJECT = Path("/workspace/sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20")
LABELED_DATA_DIR = DLC_PROJECT / "labeled-data"
MANIFEST_PATH = DLC_PROJECT / "_retrain_manifest.json"

BODYPARTS = [
    "nose_tip",
    "left_ear",
    "right_ear",
    "implant_base_rear",
    "neck",
    "mid_back",
    "mouse_center",
    "tail_base",
]

# Colors matched to dlc_viewer_page.py BP_HEX
BP_HEX: dict[str, str] = {
    "nose_tip": "#FF0000",
    "left_ear": "#0000FF",
    "right_ear": "#00FFFF",
    "implant_base_rear": "#FFA500",
    "neck": "#800080",
    "mid_back": "#00CC00",
    "mouse_center": "#FFD700",
    "tail_base": "#FF00FF",
}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def _load_manifest() -> dict:
    """Load _retrain_manifest.json. Returns {} if not found."""
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


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


def _short_session(clip: str) -> str:
    """Shorten clip name to '<date>_<animal_id>' for display."""
    parts = clip.split("_")
    if len(parts) >= 5:
        return f"{parts[0]}_{parts[4]}"
    return clip[:30]


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

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

st.header("Labeled Frame Overview")

summary_rows = []
for r in records:
    df = r["df"]
    scorer = r["scorer"]
    bps = r["bodyparts"]
    nan_per_bp = {}
    for bp in bps:
        try:
            nan_per_bp[bp] = int(df[(scorer, bp, "x")].isna().sum())
        except KeyError:
            nan_per_bp[bp] = r["n_rows"]
    manifest_entry = manifest.get(r["clip"], {})
    manifest_n = manifest_entry.get("n_frames", "—")
    summary_rows.append(
        {
            "Clip": _short_session(r["clip"]),
            "Labeled frames": r["n_labeled"],
            "Manifest frames": manifest_n,
            "Body parts": len(bps),
            **{f"NaN — {bp}": nan_per_bp.get(bp, "—") for bp in BODYPARTS},
        }
    )

summary_df = pd.DataFrame(summary_rows)

# Highlight rows where labeled < manifest (incompletely labeled sessions)
total_labeled = sum(r["n_labeled"] for r in records)
total_sessions = len(records)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Sessions with labels", total_sessions)
col_b.metric("Total labeled frames", total_labeled)
col_c.metric("Sessions in manifest", len(manifest))

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
    all_x, all_y = [], []
    for r in records:
        if bp not in r["bodyparts"]:
            continue
        x, y = _extract_xy(r["df"], r["scorer"], bp)
        valid = np.isfinite(x) & np.isfinite(y)
        all_x.extend(x[valid].tolist())
        all_y.extend(y[valid].tolist())

    if not all_x:
        continue

    fig_cov.add_trace(
        go.Scatter(
            x=all_x,
            y=all_y,
            mode="markers",
            marker=dict(size=6, color=BP_HEX.get(bp, "#888888"), opacity=0.7),
            name=bp,
        )
    )

fig_cov.update_layout(
    xaxis_title="x (pixels)",
    yaxis_title="y (pixels)",
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

        c1, c2, c3 = st.columns(3)
        c1.metric("Median ear distance (px)", f"{result['median']:.1f}")
        c2.metric("MAD (px)", f"{result['mad']:.1f}")
        c3.metric("Outlier frames", result["n_outliers"])

        frames = _frame_numbers(df_labeled)
        dist = result["distance"]
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
                y=result["median"],
                line_dash="dash",
                line_color="green",
                annotation_text="Median",
            )
        fig_ed.update_layout(
            xaxis_title="Frame number",
            yaxis_title="Distance (px)",
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
    axis_pairs = [
        ("nose_tip", "implant_base_rear"),
        ("nose_tip", "neck"),
        ("implant_base_rear", "tail_base"),
        ("neck", "tail_base"),
    ]
    axis_bp1, axis_bp2 = None, None
    for bp1, bp2 in axis_pairs:
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
            missing.append("midline keypoints (nose_tip, implant_base_rear, neck)")
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

        c1, c2, c3 = st.columns(3)
        c1.metric("Median body length (px)", f"{bl_result['median']:.1f}")
        c2.metric("MAD (px)", f"{bl_result['mad']:.1f}")
        c3.metric("Outlier frames", bl_result["n_outliers"])

        frames = _frame_numbers(df_labeled)
        bl_dist = bl_result["distance"]
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
                y=bl_result["median"],
                line_dash="dash",
                line_color="green",
                annotation_text="Median",
            )
        fig_bl.update_layout(
            xaxis_title="Frame number",
            yaxis_title="nose_tip → tail_base (px)",
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
    for bp1, bp2 in [("nose_tip", "implant_base_rear"), ("nose_tip", "neck"),
                     ("implant_base_rear", "tail_base"), ("neck", "tail_base")]:
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
