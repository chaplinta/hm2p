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
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    download_s3_bytes,
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
    import pandas as pd

    files = list_s3_session_files(DERIVATIVES_BUCKET, f"pose/{sub}/{ses}/")
    h5_files = [f for f in files if f["key"].endswith(".h5")]
    if not h5_files:
        return None, None, None, None

    h5_data = download_s3_bytes(DERIVATIVES_BUCKET, h5_files[0]["key"])
    if not h5_data:
        return None, None, None, None

    df = pd.read_hdf(io.BytesIO(h5_data))
    if not hasattr(df.columns, "get_level_values"):
        return None, None, None, None

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
        except Exception:
            pass
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

        ds = max(1, len(displacement) // 3000)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=displacement[::ds], mode="lines",
            line=dict(width=0.5, color="steelblue"), name="Displacement",
        ))
        fig.add_hline(y=jump_threshold, line_dash="dash", line_color="red",
                       annotation_text=f"Threshold: {jump_threshold}px")
        fig.update_layout(
            title=f"Frame-to-frame displacement — {bp_select}",
            xaxis_title="Frame", yaxis_title="Pixels",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True, key="jump_plot")

# --- Anatomical constraints ---
st.subheader("Anatomical Constraints")

tab_ears, tab_body, tab_frozen = st.tabs(["Ear Distance", "Body Length", "Frozen Keypoints"])

with tab_ears:
    if "left_ear" in kp_data and "right_ear" in kp_data:
        ear_result = detect_ear_distance_outliers(
            kp_data["left_ear"]["x"], kp_data["left_ear"]["y"],
            kp_data["right_ear"]["x"], kp_data["right_ear"]["y"],
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Median ear distance", f"{ear_result['median']:.1f} px")
        c2.metric("MAD", f"{ear_result['mad']:.1f} px")
        c3.metric("Outlier frames", ear_result["n_outliers"])

        if ear_result["n_outliers"] > 0:
            import plotly.graph_objects as go

            dist = ear_result["distance"]
            ds = max(1, len(dist) // 3000)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=dist[::ds], mode="lines",
                line=dict(width=0.5), name="Ear distance",
            ))
            fig.add_hline(y=ear_result["median"], line_dash="dash", line_color="green",
                           annotation_text="Median")
            fig.update_layout(
                title="Inter-ear distance over time",
                xaxis_title="Frame", yaxis_title="Distance (px)",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True, key="ear_dist_plot")
    else:
        st.info("Left/right ear keypoints not found.")

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
    lik_matrix = np.column_stack([
        kp_data[bp]["likelihood"] for bp in bodyparts if bp in kp_data
    ])

    if method.startswith("Stratified"):
        result = stratified_frame_selection(
            lik_matrix, n_per_bin=max(1, n_frames // 4), min_spacing=min_spacing,
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
    st.markdown(
        "To extract these frames for DLC labeling:\n\n"
        "```python\n"
        "from hm2p.pose.retrain import extract_frames_from_video\n"
        "from pathlib import Path\n\n"
        "frames = extract_frames_from_video(\n"
        "    video_path=Path('path/to/video.mp4'),\n"
        f"    frame_indices=np.array({st.session_state['retrain_frames'].tolist()[:10]}...),\n"
        "    output_dir=Path('retrain_frames/'),\n"
        ")\n"
        "```\n\n"
        "Then open the DLC labeling GUI to annotate the extracted frames."
    )


st.markdown("---")
st.caption("Tracking Quality & Retraining | hm2p v2")
