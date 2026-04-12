"""Tracking Quality page — detect poor pose tracking and prepare retraining data.

Provides diagnostic tools to identify sessions with tracking issues,
visualize problem frames, and extract frames for DLC retraining.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    download_s3_bytes,
    get_mm_per_pix,
    list_s3_session_files,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.tracking_quality")

st.title("Tracking Quality & Retraining")

# --- Load experiments ---
experiments = load_experiments()
if not experiments:
    st.warning("No experiments found.")
    st.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _load_dlc_data(sub: str, ses: str) -> tuple:
    """Load DLC h5 + meta from S3. Returns (df, meta_dict, bodyparts, scorer)."""
    files = list_s3_session_files(DERIVATIVES_BUCKET, f"pose/{sub}/{ses}/")
    h5_files = [f for f in files if f["key"].endswith(".h5")]
    if not h5_files:
        return None, None, None, None

    h5_data = download_s3_bytes(DERIVATIVES_BUCKET, h5_files[0]["key"])
    if not h5_data:
        return None, None, None, None

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as tmp:
        tmp.write(h5_data)
        tmp.flush()
        df = pd.read_hdf(tmp.name)
    if not hasattr(df.columns, "get_level_values"):
        return None, None, None, None

    # Handle multi-animal DLC format (4 levels: scorer/individuals/bodyparts/coords)
    if df.columns.nlevels == 4:
        scorer = df.columns.get_level_values(0)[0]
        individuals = df.columns.get_level_values(1).unique().tolist()
        bodyparts = df.columns.get_level_values(2).unique().tolist()
        coords_list = df.columns.get_level_values(3).unique().tolist()

        # Pick best individual per frame by mean likelihood
        if "likelihood" in coords_list and len(individuals) > 1:
            lik_arrays = []
            for ind in individuals:
                lik_vals = []
                for bp in bodyparts:
                    try:
                        lik_vals.append(df[(scorer, ind, bp, "likelihood")].values)
                    except KeyError:
                        pass
                if lik_vals:
                    lik_arrays.append(np.nanmean(np.column_stack(lik_vals), axis=1))
                else:
                    lik_arrays.append(np.zeros(len(df)))
            best_idx = np.argmax(np.column_stack(lik_arrays), axis=1)
        else:
            best_idx = np.zeros(len(df), dtype=int)

        # Reconstruct single-animal DataFrame
        new_data = {}
        for bp in bodyparts:
            for coord in coords_list:
                vals = np.empty(len(df))
                for frame_idx in range(len(df)):
                    try:
                        vals[frame_idx] = df.iloc[frame_idx][(scorer, individuals[best_idx[frame_idx]], bp, coord)]
                    except (KeyError, IndexError):
                        vals[frame_idx] = np.nan
                new_data[(scorer, bp, coord)] = vals
        df = pd.DataFrame(new_data, index=df.index)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
    else:
        scorer = df.columns.get_level_values(0)[0]
        bodyparts = df.columns.get_level_values(1).unique().tolist()

    meta_bytes = download_s3_bytes(DERIVATIVES_BUCKET, f"pose/{sub}/{ses}/dlc_meta.json")
    meta = json.loads(meta_bytes) if meta_bytes else {}

    return df, meta, bodyparts, scorer


def _extract_keypoint_data(df, scorer, bodyparts):
    """Extract {bodypart: {x, y, likelihood}} from DLC DataFrame."""
    data = {}
    for bp in bodyparts:
        try:
            data[bp] = {
                "x": df[(scorer, bp, "x")].values.astype(np.float64),
                "y": df[(scorer, bp, "y")].values.astype(np.float64),
                "likelihood": df[(scorer, bp, "likelihood")].values.astype(np.float64),
            }
        except KeyError:
            pass
    return data


# ---------------------------------------------------------------------------
# Session-level quality overview
# ---------------------------------------------------------------------------

st.header("Session Quality Overview")

# Build session list with pose data
@st.cache_data(ttl=120)
def _list_pose_sessions():
    from frontend.data import get_s3_client
    s3 = get_s3_client()
    sessions = []
    for exp in experiments:
        exp_id = exp["exp_id"]
        sub, ses = parse_session_id(exp_id)
        prefix = f"pose/{sub}/{ses}/"
        try:
            resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=prefix, MaxKeys=1)
            if resp.get("KeyCount", 0) > 0:
                sessions.append(f"{sub}/{ses}")
        except Exception as e:
            st.warning(f"Could not check pose data for {exp_id}: {e}")
    return sessions


pose_sessions = _list_pose_sessions()

if not pose_sessions:
    st.info("No pose data available yet. Run DLC first.")
    st.stop()

# --- Quality scan across sessions ---
with st.expander("Scan all sessions for quality issues", expanded=False):
    if st.button("Run quality scan", key="scan_btn"):
        from hm2p.pose.quality import session_quality_report

        results = []
        progress = st.progress(0, text="Scanning...")

        for i, sess_key in enumerate(pose_sessions):
            sub, ses = sess_key.split("/")
            df, meta, bodyparts, scorer = _load_dlc_data(sub, ses)
            progress.progress((i + 1) / len(pose_sessions), text=f"Scanning {sess_key}...")

            if df is None:
                results.append({"session": sess_key, "score": None, "issues": ["No data"]})
                continue

            kp_data = _extract_keypoint_data(df, scorer, bodyparts)
            fps = meta.get("tracking_fps", 30)
            report = session_quality_report(kp_data, fps=fps)
            results.append({
                "session": sess_key,
                "score": report["overall_score"],
                "pct_good": report["pct_good"],
                "n_frames": report["n_frames"],
                "issues": report["issues"],
            })

        progress.empty()

        # Display results sorted by score (worst first)
        results.sort(key=lambda r: r["score"] if r["score"] is not None else -1)

        for r in results:
            score = r["score"]
            if score is None:
                st.markdown(f"**{r['session']}** — :red[No data]")
                continue

            if score >= 80:
                color = "green"
                label = "Good"
            elif score >= 60:
                color = "orange"
                label = "Fair"
            else:
                color = "red"
                label = "Poor"

            st.markdown(
                f"**{r['session']}** — :{color}[{label} ({score:.0f}/100)] "
                f"| {r['pct_good']*100:.1f}% clean frames | {r['n_frames']} frames"
            )
            if r["issues"]:
                for issue in r["issues"]:
                    st.caption(f"  - {issue}")


# ---------------------------------------------------------------------------
# Single session diagnostics
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Session Diagnostics")

selected = st.selectbox("Select session", pose_sessions, key="tq_session")
sub, ses = selected.split("/")
mm_per_pix = get_mm_per_pix(sub, ses)

# Clear stale retrain state when session changes
if st.session_state.get("retrain_session") != selected:
    st.session_state.pop("retrain_frames", None)
    st.session_state.pop("retrain_session", None)

df, meta, bodyparts, scorer = _load_dlc_data(sub, ses)
if df is None:
    st.warning("Could not load pose data for this session.")
    st.stop()

st.caption(
    f"**Scorer:** `{scorer}` | **Body parts:** {len(bodyparts)} | "
    f"**Frames:** {len(df)} | **FPS:** {meta.get('tracking_fps', '?')}"
)

kp_data = _extract_keypoint_data(df, scorer, bodyparts)

# --- Quality report ---
from hm2p.pose.quality import (
    body_length_consistency,
    detect_ear_distance_outliers,
    detect_ear_swaps,
    detect_frozen_keypoint,
    detect_jumps,
    session_quality_report,
)

fps = meta.get("tracking_fps", 30)
report = session_quality_report(kp_data, fps=fps)

col1, col2, col3 = st.columns(3)
col1.metric("Quality Score", f"{report['overall_score']:.0f}/100")
col2.metric("Clean Frames", f"{report['pct_good']*100:.1f}%")
col3.metric("Issues", len(report["issues"]))

if report["issues"]:
    st.warning("Issues detected:")
    for issue in report["issues"]:
        st.markdown(f"- {issue}")

# --- Jump detection ---
st.subheader("Jump Detection")

if mm_per_pix is not None:
    # Slider in mm/frame; convert to px/frame for the detection function
    jump_thr_mm = st.slider(
        "Jump threshold (mm/frame)", 1.0, 150.0, round(50.0 * mm_per_pix, 1), 0.5,
        key="jump_thresh",
    )
    jump_threshold = jump_thr_mm / mm_per_pix
else:
    jump_threshold = st.slider(
        "Jump threshold (pixels/frame)", 10.0, 200.0, 50.0, 5.0, key="jump_thresh"
    )

bp_select = st.selectbox("Body part for diagnostics", bodyparts, key="diag_bp")

if bp_select in kp_data:
    x = kp_data[bp_select]["x"]
    y = kp_data[bp_select]["y"]
    lik = kp_data[bp_select]["likelihood"]

    jumps = detect_jumps(x, y, threshold_px=jump_threshold)
    n_jumps = int(jumps.sum())

    st.metric(f"Jump frames ({bp_select})", f"{n_jumps} ({n_jumps/len(x)*100:.2f}%)")

    if n_jumps > 0:
        import plotly.graph_objects as go

        # Show displacement plot with jumps highlighted
        dx = np.diff(x)
        dy = np.diff(y)
        displacement = np.sqrt(dx**2 + dy**2)

        if mm_per_pix is not None:
            disp_display = displacement * mm_per_pix
            thr_display = jump_threshold * mm_per_pix
            y_label = "Displacement (mm)"
            thr_text = f"Threshold: {thr_display:.1f} mm"
        else:
            disp_display = displacement
            thr_display = jump_threshold
            y_label = "Displacement (px)"
            thr_text = f"Threshold: {thr_display:.0f} px"

        ds = max(1, len(disp_display) // 3000)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=disp_display[::ds], mode="lines",
            line=dict(width=0.5, color="steelblue"), name="Displacement",
        ))
        fig.add_hline(y=thr_display, line_dash="dash", line_color="red",
                       annotation_text=thr_text)
        fig.update_layout(
            title=f"Frame-to-frame displacement — {bp_select}",
            xaxis_title="Frame", yaxis_title=y_label,
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True, key="jump_plot")

# --- Anatomical constraints ---
st.subheader("Anatomical Constraints")

tab_ears, tab_swap, tab_body, tab_frozen = st.tabs(["Ear Distance", "Ear Swap", "Body Length", "Frozen Keypoints"])

with tab_ears:
    if "left_ear" in kp_data and "right_ear" in kp_data:
        ear_result = detect_ear_distance_outliers(
            kp_data["left_ear"]["x"], kp_data["left_ear"]["y"],
            kp_data["right_ear"]["x"], kp_data["right_ear"]["y"],
        )
        c1, c2, c3 = st.columns(3)
        if mm_per_pix is not None:
            c1.metric("Median ear distance", f"{ear_result['median'] * mm_per_pix:.1f} mm")
            c2.metric("MAD", f"{ear_result['mad'] * mm_per_pix:.1f} mm")
        else:
            c1.metric("Median ear distance", f"{ear_result['median']:.1f} px")
            c2.metric("MAD", f"{ear_result['mad']:.1f} px")
        c3.metric("Outlier frames", ear_result["n_outliers"])

        if ear_result["n_outliers"] > 0:
            import plotly.graph_objects as go

            dist = ear_result["distance"]
            if mm_per_pix is not None:
                dist_display = dist * mm_per_pix
                median_display = ear_result["median"] * mm_per_pix
                ear_y_label = "Distance (mm)"
            else:
                dist_display = dist
                median_display = ear_result["median"]
                ear_y_label = "Distance (px)"
            ds = max(1, len(dist_display) // 3000)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=dist_display[::ds], mode="lines",
                line=dict(width=0.5), name="Ear distance",
            ))
            fig.add_hline(y=median_display, line_dash="dash", line_color="green",
                           annotation_text="Median")
            fig.update_layout(
                title="Inter-ear distance over time",
                xaxis_title="Frame", yaxis_title=ear_y_label,
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True, key="ear_dist_plot")
    else:
        st.info("Left/right ear keypoints not found.")

with tab_swap:
    # Need ears + at least one midline keypoint pair for the body axis
    has_ears = "left_ear" in kp_data and "right_ear" in kp_data
    # Pick best available midline pair: nose→implant preferred, fallback to others
    axis_pairs = [
        ("nose_tip", "implant_base_rear"),
        ("nose", "implant_base_rear"),
        ("nose_tip", "neck"),
        ("nose", "neck"),
        ("implant_base_rear", "tail_base"),
        ("neck", "tail_base"),
    ]
    axis_bp1, axis_bp2 = None, None
    for bp1, bp2 in axis_pairs:
        if bp1 in kp_data and bp2 in kp_data:
            axis_bp1, axis_bp2 = bp1, bp2
            break

    if has_ears and axis_bp1 is not None:
        swap_result = detect_ear_swaps(
            kp_data["left_ear"]["x"], kp_data["left_ear"]["y"],
            kp_data["right_ear"]["x"], kp_data["right_ear"]["y"],
            kp_data[axis_bp1]["x"], kp_data[axis_bp1]["y"],
            kp_data[axis_bp2]["x"], kp_data[axis_bp2]["y"],
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Swapped frames", f"{swap_result['n_swapped']:,}")
        c2.metric("% swapped", f"{swap_result['pct_swapped']*100:.1f}%")
        c3.metric("Body axis", f"{axis_bp1} → {axis_bp2}")

        if swap_result["n_swapped"] > 0:
            st.caption(
                "Frames where the left ear is on the right side of the body "
                "axis (or vice versa), indicating DLC swapped the ear labels. "
                f"Body axis defined by {axis_bp1} → {axis_bp2}."
            )

            import plotly.graph_objects as go
            left_s = swap_result["left_sign"]
            ds = max(1, len(left_s) // 3000)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=left_s[::ds], mode="lines",
                line=dict(width=0.5, color="steelblue"),
                name="Left ear side (+ = left of axis)",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            # Highlight swapped regions
            swapped_idx = np.where(swap_result["is_swapped"])[0]
            if len(swapped_idx) > 0:
                fig.add_trace(go.Scatter(
                    x=swapped_idx[::ds] if len(swapped_idx) > 3000 else swapped_idx,
                    y=left_s[swapped_idx][::ds] if len(swapped_idx) > 3000 else left_s[swapped_idx],
                    mode="markers",
                    marker=dict(size=3, color="red"),
                    name="Swapped",
                ))
            fig.update_layout(
                title="Left ear signed side relative to body axis",
                xaxis_title="Frame",
                yaxis_title="Cross-product (+ = left, − = right)",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True, key="ear_swap_plot")
        else:
            st.success("No ear swaps detected.")
    else:
        missing = []
        if not has_ears:
            missing.append("left_ear / right_ear")
        if axis_bp1 is None:
            missing.append("midline keypoints (nose, implant, neck, tail)")
        st.info(f"Need {', '.join(missing)} for ear swap detection.")

with tab_body:
    head_bp = "mouse_center" if "mouse_center" in kp_data else (
        "mid_back" if "mid_back" in kp_data else None
    )
    tail_bp = "tail_base" if "tail_base" in kp_data else None

    if head_bp and tail_bp:
        body_result = body_length_consistency(
            kp_data[head_bp]["x"], kp_data[head_bp]["y"],
            kp_data[tail_bp]["x"], kp_data[tail_bp]["y"],
        )
        c1, c2, c3 = st.columns(3)
        if mm_per_pix is not None:
            c1.metric("Median body length", f"{body_result['median'] * mm_per_pix:.1f} mm")
            c2.metric("MAD", f"{body_result['mad'] * mm_per_pix:.1f} mm")
        else:
            c1.metric("Median body length", f"{body_result['median']:.1f} px")
            c2.metric("MAD", f"{body_result['mad']:.1f} px")
        c3.metric("Outlier frames", body_result["n_outliers"])
    else:
        st.info(f"Need head ({head_bp}) and tail ({tail_bp}) keypoints.")

with tab_frozen:
    if bp_select in kp_data:
        frozen = detect_frozen_keypoint(
            kp_data[bp_select]["x"], kp_data[bp_select]["y"],
        )
        n_frozen = int(frozen.sum())
        st.metric(
            f"Frozen frames ({bp_select})",
            f"{n_frozen} ({n_frozen/max(len(frozen), 1)*100:.2f}%)",
        )
        if n_frozen > 0:
            st.caption(
                "Frozen keypoints move < 0.5px over 30 consecutive frames. "
                "This often indicates the detector locked onto a fixed point."
            )


# ---------------------------------------------------------------------------
# Frame selection for retraining
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Retraining Frame Selection")

st.markdown(
    "Select poorly-tracked frames for manual labeling in DLC. "
    "Frames are chosen to maximize coverage of failure modes."
)

method = st.radio(
    "Selection method",
    ["Stratified (recommended)", "Worst frames only"],
    key="retrain_method",
)

n_frames = st.slider("Number of frames to select", 5, 100, 20, 5, key="retrain_n")
min_spacing = st.slider("Minimum frame spacing", 10, 100, 30, 5, key="retrain_spacing")

if st.button("Select frames", key="select_frames_btn"):
    from hm2p.pose.quality import stratified_frame_selection, worst_frames

    # Build likelihood matrix (n_frames, n_keypoints)
    n_total = len(df)
    lik_cols = [kp_data[bp]["likelihood"] for bp in bodyparts if bp in kp_data]
    if not lik_cols:
        st.warning("No likelihood data available for selected bodyparts.")
        st.stop()
    lik_matrix = np.column_stack(lik_cols)

    # Build position matrix for duplicate detection
    pos_cols = []
    for bp in bodyparts:
        if bp in kp_data:
            pos_cols.append(kp_data[bp]["x"])
            pos_cols.append(kp_data[bp]["y"])
    pos_matrix = np.column_stack(pos_cols) if pos_cols else None

    if method.startswith("Stratified"):
        result = stratified_frame_selection(
            lik_matrix, n_per_bin=max(1, n_frames // 4), min_spacing=min_spacing,
            positions=pos_matrix,
        )
        selected_indices = result["indices"]

        st.success(f"Selected {len(selected_indices)} frames across quality bins")

        for label, bin_idx in result["bins"]:
            st.caption(f"**{label.title()}** ({len(bin_idx)} frames): {bin_idx.tolist()}")
    else:
        selected_indices = worst_frames(lik_matrix, n_frames=n_frames, min_spacing=min_spacing)
        st.success(f"Selected {len(selected_indices)} worst frames")

    # Show selected frame details
    st.subheader("Selected Frames")

    mean_lik = np.nanmean(lik_matrix, axis=1)
    for idx in selected_indices[:20]:  # Show first 20
        frame_lik = mean_lik[idx]
        color = "red" if frame_lik < 0.5 else ("orange" if frame_lik < 0.9 else "green")
        st.markdown(
            f"Frame **{idx}** — :{color}[likelihood: {frame_lik:.3f}]"
        )

    if len(selected_indices) > 20:
        st.caption(f"... and {len(selected_indices) - 20} more frames")

    # Store selection in session state for export
    st.session_state["retrain_frames"] = selected_indices
    st.session_state["retrain_session"] = selected


# --- Export instructions ---
if "retrain_frames" in st.session_state:
    st.subheader("Export for Labeling")

    _rt_sub, _rt_ses = st.session_state["retrain_session"].split("/")
    _rt_frames = st.session_state["retrain_frames"].tolist()
    _rt_s3_video = f"s3://hm2p-rawdata/rawdata/{_rt_sub}/{_rt_ses}/behav/"
    _rt_output = f"retrain_frames/{_rt_sub}_{_rt_ses}"
    _rt_frames_str = str(_rt_frames[:20])
    if len(_rt_frames) > 20:
        _rt_frames_str = _rt_frames_str[:-1] + ", ...]"

    _frames_arg = " ".join(str(f) for f in _rt_frames)

    st.markdown(f"**Session:** `{_rt_sub}/{_rt_ses}` | **Frames:** {len(_rt_frames)}")

    st.markdown("**Run this on your Mac** (downloads video, extracts frames, sets up DLC project):")
    st.code(
        f"# Install DLC GUI (only needed once)\n"
        f"uv pip install --pre 'deeplabcut[gui]' --python ~/.venv-hm2p/bin/python\n\n"
        f"# Prepare frames + DLC project\n"
        f"cd ~/Neuro/hm2p-v2\n"
        f"uv run python scripts/prepare_retrain_frames.py {_rt_sub}/{_rt_ses} {_frames_arg}",
        language="bash",
    )

    with st.expander("How to label frames in napari", expanded=True):
        st.markdown(
            "The script opens a **napari** window. For each frame, place 8 keypoints:\n\n"
            "| Keypoint | Where to click |\n"
            "|----------|---------------|\n"
            "| **nose_tip** | Tip of the snout (skip if hidden behind implant) |\n"
            "| **left_ear** | Centre of the left ear (mouse's left, your right from above) |\n"
            "| **right_ear** | Centre of the right ear |\n"
            "| **implant_base_rear** | Rear edge of the 2P headstage base |\n"
            "| **neck** | Base of skull, between ears and mid_back |\n"
            "| **mid_back** | Dorsal midline just behind the shoulders |\n"
            "| **mouse_center** | Geometric centre of the body |\n"
            "| **tail_base** | Where the tail meets the body |\n\n"
            "**Steps:**\n"
            "1. Select a bodypart from the **Points layer** dropdown (left panel).\n"
            "2. Click **Add Points** mode (the + icon in the layer controls).\n"
            "3. Click on the image to place the keypoint.\n"
            "4. Repeat for all 8 bodyparts on this frame.\n"
            "5. Use the **slider at the bottom** to move to the next frame.\n"
            "6. When done, **close the napari window** (Cmd+Q). Labels save automatically.\n\n"
            "**Tips:**\n"
            "- If a bodypart is **occluded** (hidden by the headstage, another body part, "
            "or the maze wall), **skip it** — don't guess. DLC handles missing labels.\n"
            "- If the mouse is **out of frame** or the frame is very blurry, skip the entire frame.\n"
            "- Label the **centre** of each body part, not the edge.\n"
            "- Zoom in (scroll wheel) for precise placement on small features like ears.\n"
            "- You can **drag** a placed point to adjust its position.\n"
            "- To **delete** a misplaced point, select it and press Delete.\n\n"
            "**How many frames to label:**\n"
            "- **Minimum for a first retrain:** 50 frames across 3-5 sessions.\n"
            "- **Recommended:** 100-200 frames across 5-10 sessions, focusing on sessions "
            "with the worst tracking quality (lowest scores above).\n"
            "- Include frames from **different conditions**: light on, light off, "
            "mouse near walls, mouse in open corridor, mouse turning, mouse stationary.\n"
            "- Run the script multiple times with different sessions to accumulate "
            "frames in the same DLC project.\n"
            "- After the first retrain, review tracking quality again and add more "
            "frames from remaining problem areas."
        )

    st.markdown("After labeling and closing napari:")
    st.code(
        f"# Upload labels and launch training on AWS (GPU)\n"
        f"uv run python scripts/upload_dlc_labels.py\n"
        f"uv run python scripts/launch_dlc_finetune_ec2.py",
        language="bash",
    )

st.markdown("---")

# ── Auto frame selection across all sessions ─────────────────────────────
st.header("Auto-Select Frames Across All Sessions")

st.markdown(
    "Automatically find the best frames to label across **all 26 sessions** "
    "by scoring every frame for:\n"
    "- **Low confidence** — model is uncertain about bodypart positions\n"
    "- **Temporal jumps** — predictions are inconsistent between consecutive frames\n"
    "- **Unusual poses** — bodypart spread deviates from the session median\n\n"
    "Frames are allocated across sessions (worst-tracked sessions get more), "
    "with constraints to avoid near-duplicates and already-labelled frames."
)

n_auto = st.slider("Number of frames to select", 20, 200, 60, 10, key="auto_n")

st.markdown("**Run on your Mac:**")
st.code(
    f"# Preview selection (no files created)\n"
    f"uv run python scripts/select_labelling_frames.py --n {n_auto} --dry-run\n\n"
    f"# Select, extract, and label all frames (opens napari per session)\n"
    f"uv run python scripts/select_labelling_frames.py --n {n_auto} --label",
    language="bash",
)

with st.expander("How it works"):
    st.markdown(
        "The script downloads pose `.h5` files from S3 for all sessions and scores "
        "every frame (0-1, higher = worse tracking). The score combines:\n\n"
        "- **Confidence score (50%):** inverted mean DLC likelihood across bodyparts\n"
        "- **Jump score (30%):** frame-to-frame bodypart displacement normalised by "
        "median displacement — catches sudden tracking failures\n"
        "- **Pose score (20%):** deviation of bodypart spread from median — catches "
        "unusual postures (grooming, rearing, against walls)\n\n"
        "Frames are then selected per session with:\n"
        "- 2-8 frames per session (worst sessions get more)\n"
        "- Minimum 30-frame spacing (no temporal neighbours)\n"
        "- Position similarity rejection (no visual near-duplicates)\n"
        "- Already-labelled frames excluded\n\n"
        "The output is a list of `prepare_retrain_frames.py` commands — "
        "run each one to extract frames and open napari for labelling."
    )

st.markdown("---")

# ── Interactive labeller ─────────────────────────────────────────────────
st.header("Interactive Labeller")

st.markdown(
    "Open a custom napari labeller that shows all sessions with existing "
    "labels. Pick a session from a menu, label/edit bodyparts, close napari "
    "to save, then pick the next session."
)

st.markdown("**Run on your Mac:**")
st.code(
    "# Interactive menu — pick sessions, label in napari\n"
    "uv run python scripts/interactive_label.py",
    language="bash",
)

with st.expander("How it works"):
    st.markdown(
        "Opens a terminal menu showing all sessions with frame counts and "
        "label status. Enter a session number to open napari.\n\n"
        "**In napari:**\n"
        "- All bodyparts are in a **single layer**, colour-coded\n"
        "- Press **1-8** to select which bodypart to place:\n"
        "  1. nose_tip (red)\n"
        "  2. left_ear (blue)\n"
        "  3. right_ear (cyan)\n"
        "  4. implant_base_rear (orange)\n"
        "  5. neck (purple)\n"
        "  6. mid_back (green)\n"
        "  7. mouse_center (gold)\n"
        "  8. tail_base (magenta)\n"
        "- **Click** on the image to place a point\n"
        "- **Drag** an existing point to move it\n"
        "- **Delete** key to remove a selected point\n"
        "- **Close napari** to save and return to the menu\n"
        "- Enter **'a'** to label all sessions sequentially\n"
        "- Enter **'q'** to quit\n\n"
        "Labels are saved to `CollectedData_tristan.csv` + `.h5` in DLC format. "
        "Existing labels are loaded and editable.\n\n"
        "**Labelling tips:**\n"
        "- Label the **centre** of each bodypart, not the edge\n"
        "- If a bodypart is **occluded but you can infer its position**, label it\n"
        "- If you **cannot guess** where it is, skip it (leave unlabelled)\n"
        "- Be **consistent** — always guess occluded parts the same way\n"
        "- The mouse is sometimes behind **transparent acrylic** walls — "
        "still label bodyparts if visible through the acrylic"
    )

st.markdown("**After labelling:**")
st.code(
    "# Upload labels to S3 and retrain\n"
    "uv run python scripts/upload_dlc_labels.py\n"
    "uv run python scripts/launch_dlc_finetune_ec2.py",
    language="bash",
)

st.markdown("---")
st.caption("Tracking Quality & Retraining | hm2p v2")
