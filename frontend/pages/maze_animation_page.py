"""Maze Animation — animated replay of mouse trajectory through the Rosenberg maze.

Shows the mouse position and head direction arrow as it moves through the maze,
with a trail showing recent trajectory. Uses an HTML5 Canvas component for
smooth 60 fps playback without Streamlit reruns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.components.maze_canvas import render_maze_canvas
from frontend.data import load_all_sync_data, render_tracker_provenance
from frontend.data import session_filter_controls as session_filter_sidebar

# ── DLC skeleton and rainbow colormap ────────────────────────────────
_SKELETON = [
    ("nose_tip", "head_midpoint"),
    ("nose_tip", "left_ear"),
    ("nose_tip", "right_ear"),
    ("left_ear", "head_midpoint"),
    ("right_ear", "head_midpoint"),
    ("left_ear", "right_ear"),
    ("head_midpoint", "neck"),
    ("neck", "mid_back"),
    ("mid_back", "mouse_center"),
    ("mouse_center", "tail_base"),
]

# Bodypart colours: matplotlib.cm.rainbow / DLC-native palette, mirrored
# from frontend/pages/dlc_viewer_page.py (BP_HEX). This is the canonical
# DLC-rainbow palette used across the frontend (also in training_qc_page
# and training_fit_page). The legend rendered in dlc_viewer_page uses
# these colours, so the maze animation must match for cross-page
# consistency. ``render_dlc_videos.KEYPOINT_COLORS`` uses a different
# (BGR-bright) palette and is the inconsistent one.
_BP_COLORS = {
    "nose_tip": "#7F00FF",  # purple
    "nose": "#7F00FF",  # SuperAnimal alias
    "left_ear": "#376DF8",  # blue
    "right_ear": "#12C7E5",  # cyan
    "head_midpoint": "#5AF8C7",  # aqua
    "implant_base_rear": "#5AF8C7",  # legacy DLC alias
    "neck": "#A4F89E",  # green
    "mid_back": "#ECC76E",  # yellow
    "mouse_center": "#FF6D38",  # orange
    "tail_base": "#FF0000",  # red
}

# ── Maze boundary polygon (7x5 Rosenberg maze) ──────────────────────
# This traces the outer wall of the 23 accessible cells.
# Imported by perspective_compare_page — keep these at module level.
_MAZE_WALLS_X = [
    0,
    3,
    3,
    2,
    2,
    5,
    5,
    4,
    4,
    7,
    7,
    6,
    6,
    7,
    7,
    4,
    4,
    5,
    5,
    4,
    4,
    3,
    3,
    2,
    2,
    3,
    3,
    0,
    0,
    1,
    1,
    0,
    0,
]
_MAZE_WALLS_Y = [
    0,
    0,
    1,
    1,
    2,
    2,
    1,
    1,
    0,
    0,
    1,
    1,
    4,
    4,
    5,
    5,
    4,
    4,
    3,
    3,
    5,
    5,
    3,
    3,
    4,
    4,
    5,
    5,
    4,
    4,
    1,
    1,
    0,
]


def _draw_maze(fig: go.Figure) -> None:
    """Add the Rosenberg maze boundary walls to a Plotly figure."""
    fig.add_trace(
        go.Scatter(
            x=_MAZE_WALLS_X,
            y=_MAZE_WALLS_Y,
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
            hoverinfo="skip",
            name="walls",
        )
    )


def _subsample(arr: np.ndarray, step: int) -> np.ndarray:
    """Take every ``step``-th element."""
    return arr[::step]


def _nan_to_none(arr: np.ndarray) -> list:
    """Convert numpy array to list, replacing NaN/inf with None for JSON.

    JSON does not support NaN or Infinity, so these must be converted to
    null (Python None) for safe serialization.
    """
    result = []
    for v in arr:
        if np.isfinite(v):
            result.append(float(v))
        else:
            result.append(None)
    return result


def _build_canvas_payload(
    x_maze: np.ndarray,
    y_maze: np.ndarray,
    hd_deg: np.ndarray,
    speed: np.ndarray,
    light_on: np.ndarray,
    frame_times: np.ndarray,
    trail_seconds: float,
    step: int,
    arrow_length: float,
    bp_maze: dict | None = None,
    show_position: bool = True,
    show_skeleton: bool = True,
) -> dict:
    """Build the JSON-serializable payload for the canvas component.

    Parameters
    ----------
    x_maze, y_maze : np.ndarray
        Body centroid in maze coordinates (0-7 x, 0-5 y).
    hd_deg : np.ndarray
        Head direction in degrees (unwrapped).
    speed : np.ndarray
        Speed in cm/s.
    light_on : np.ndarray
        Boolean array — True when room lights are on.
    frame_times : np.ndarray
        Timestamps (seconds) for each frame.
    trail_seconds : float
        Duration of the fading position trail.
    step : int
        Subsample factor (take every N-th frame).
    arrow_length : float
        Length of the HD arrow in maze units.
    bp_maze : dict or None
        Per-bodypart maze coordinates. Keys are bodypart names, values are
        dicts with ``"x"`` and ``"y"`` arrays.
    show_position : bool
        Whether to show the position dot and HD arrow.
    show_skeleton : bool
        Whether to show the DLC skeleton.

    Returns
    -------
    dict
        JSON-serializable payload for ``build_maze_canvas_html``.
    """
    # Subsample all arrays
    x_sub = _subsample(x_maze, step)
    y_sub = _subsample(y_maze, step)
    hd_sub = _subsample(hd_deg, step)
    speed_sub = _subsample(speed, step)
    light_sub = _subsample(light_on, step)
    ft_sub = _subsample(frame_times, step)

    n = len(x_sub)

    # Determine which bodyparts are present
    bp_names_ordered = [
        "nose_tip",
        "nose",
        "left_ear",
        "right_ear",
        "head_midpoint",
        "implant_base_rear",
        "neck",
        "mid_back",
        "mouse_center",
        "tail_base",
    ]

    bp_present = []
    bp_x_data: dict[str, list] = {}
    bp_y_data: dict[str, list] = {}

    if bp_maze:
        for bp_name in bp_names_ordered:
            if bp_name in bp_maze:
                bp_present.append(bp_name)
                bp_x_arr = _subsample(bp_maze[bp_name]["x"], step)
                bp_y_arr = _subsample(bp_maze[bp_name]["y"], step)
                bp_x_data[bp_name] = _nan_to_none(bp_x_arr)
                bp_y_data[bp_name] = _nan_to_none(bp_y_arr)

    # If no bodypart data, synthesize from centroid for trail drawing
    if not bp_present:
        bp_present = ["mouse_center"]
        bp_x_data["mouse_center"] = _nan_to_none(x_sub)
        bp_y_data["mouse_center"] = _nan_to_none(y_sub)

    # Filter skeleton connections to only those with both endpoints present
    skeleton_filtered = [
        [bp1, bp2] for bp1, bp2 in _SKELETON if bp1 in bp_present and bp2 in bp_present
    ]

    # Filter colours to present bodyparts only
    bp_colors_filtered = {bp: _BP_COLORS.get(bp, "#888888") for bp in bp_present}

    return {
        "n_frames": n,
        "bp_names": bp_present,
        "skeleton": skeleton_filtered,
        "bp_colors": bp_colors_filtered,
        "maze_walls_x": _MAZE_WALLS_X,
        "maze_walls_y": _MAZE_WALLS_Y,
        "bp_x": bp_x_data,
        "bp_y": bp_y_data,
        "hd_deg": _nan_to_none(hd_sub),
        "speed": _nan_to_none(speed_sub),
        "light_on": [int(bool(v)) for v in light_sub],
        "frame_times": [float(v) for v in ft_sub],
        "arrow_length": float(arrow_length),
        "trail_seconds": float(trail_seconds),
        "show_position": bool(show_position),
        "show_skeleton": bool(show_skeleton),
    }


# ── Page ──────────────────────────────────────────────────────────────


def _page() -> None:
    """Render the maze animation page (called by Streamlit runner)."""
    st.title("Maze Animation")
    st.caption(
        "Animated replay of mouse trajectory through the Rosenberg maze. "
        "Shows head position, facing direction (purple arrow), and recent trail. "
        "Colour indicates light state (orange = lights on, grey = dark)."
    )

    with st.expander("How this works"):
        st.markdown(
            "**Position** is computed from the DLC `mouse_center` keypoint, "
            "converted from pixels to maze coordinates (0\u20137 x, 0\u20135 y) "
            "via camera calibration and perspective correction.\n\n"
            "**Head direction** is a confidence-weighted circular mean of 4 "
            "estimates: (1) perpendicular to the ear\u2013ear line, (2) nose\u2192head "
            "midpoint, (3) nose\u2192neck, (4) head midpoint\u2192neck. Each estimate "
            "is weighted by the mean DLC confidence of its keypoints. Shown as "
            "a purple arrow.\n\n"
            "**Raw** shows all imaging frames. Where DLC confidence was below "
            "threshold (position or ears), gaps are filled with linear "
            "interpolation between the nearest confident frames. This gives "
            "continuous playback but interpolated segments may not reflect "
            "true motion.\n\n"
            "**Cleaned** shows only frames where the DLC confidence-filtered, "
            "gap-filled (short gaps only), and median-smoothed position passed "
            "all quality checks. Frames with low-confidence predictions are "
            "excluded (shown as NaN in the data).\n\n"
            "**Playback controls** (play/pause, speed, scrubber) are in the "
            "animation itself and do not trigger a page reload. Streamlit "
            "controls (session, time range, etc.) above the animation do "
            "trigger a data reload when changed."
        )

    from frontend.data import check_stale_data_warning

    check_stale_data_warning(stages=["kinematics", "sync"], block=True)

    with st.spinner("Loading sync data..."):
        all_data = load_all_sync_data()

    if all_data["n_sessions"] == 0:
        st.warning("No sync.h5 data available. This page requires completed pipeline stages 0-5.")
        st.stop()

    sessions = session_filter_sidebar(
        all_data["sessions"], show_roi_filter=False, key_prefix="maze_anim"
    )

    if not sessions:
        st.warning("No sessions match the current filters.")
        st.stop()
    render_tracker_provenance(sessions)

    sessions_with_pos = [
        s for s in sessions if s.get("x_maze") is not None and s.get("y_maze") is not None
    ]

    if not sessions_with_pos:
        st.warning("No sessions have position data (kinematics.h5 not yet generated).")
        st.stop()

    col_sel, col_opts = st.columns([2, 3])

    with col_sel:
        session_labels = [f"{s['exp_id']} ({s['celltype']})" for s in sessions_with_pos]
        selected_idx = st.selectbox(
            "Session",
            range(len(session_labels)),
            format_func=lambda i: session_labels[i],
            key="maze_anim_session",
        )

    with col_opts:
        # Row 1
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            pos_mode = st.radio(
                "Position data",
                ["Raw", "Cleaned"],
                index=0,
                key="maze_anim_pos_mode",
                help=(
                    "**Raw:** unfiltered DLC pose — no confidence filter, "
                    "no interpolation, no median smoothing, no confidence "
                    "weighting in the body centroid. **Cleaned:** confidence-"
                    "filtered, gap-interpolated, median-smoothed pose."
                ),
            )
        with r1c2:
            subsample = st.slider(
                "Subsample (every N frames)",
                1,
                30,
                1,
                1,
                key="maze_anim_sub",
                help="Reduce frame count for browser performance (does not change playback time-base).",
            )

        # Row 2
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            trail_s = st.slider("Trail (s)", 1.0, 30.0, 10.0, 1.0, key="maze_anim_trail")
        with r2c2:
            arrow_len = st.slider("Arrow length", 0.1, 1.5, 0.5, 0.1, key="maze_anim_arrow")
        with r2c3:
            show_position = st.checkbox(
                "Show position + HD",
                value=True,
                key="maze_anim_show_pos",
                help="Display the body-centroid circle and the head-direction arrow.",
            )
        with r2c4:
            show_skeleton = st.checkbox(
                "Show skeleton",
                value=True,
                key="maze_anim_skel",
                help="Draw DLC bodypart skeleton (requires per-bodypart maze coordinates in sync.h5).",
            )

    ses = sessions_with_pos[selected_idx]

    hd_deg = ses["hd_deg"]
    speed = ses["speed_cm_s"]
    light_on = ses["light_on"]
    frame_times = ses["frame_times"]

    if pos_mode == "Raw":
        x_maze_raw = ses.get("x_maze_raw")
        y_maze_raw = ses.get("y_maze_raw")
        if x_maze_raw is not None and y_maze_raw is not None:
            x_maze = x_maze_raw
            y_maze = y_maze_raw
            ses_bp_maze = ses.get("bp_maze_raw") or ses.get("bp_maze")
        else:
            st.warning(
                "Raw position fields are not present in this session's "
                "sync.h5 — it predates the raw-fields commit. Showing "
                "cleaned position instead. Re-run the pipeline to populate "
                "x_maze_raw / y_maze_raw."
            )
            x_maze = ses["x_maze"]
            y_maze = ses["y_maze"]
            ses_bp_maze = ses.get("bp_maze")
        valid = np.isfinite(x_maze) & np.isfinite(y_maze) & ~ses["bad_behav"]
    else:
        x_maze = ses["x_maze"]
        y_maze = ses["y_maze"]
        ses_bp_maze = ses.get("bp_maze")
        valid = np.isfinite(x_maze) & np.isfinite(y_maze) & ~ses["bad_behav"]

    if valid.sum() < 10:
        st.warning("Not enough valid position data for this session.")
        st.stop()

    n_total = len(frame_times)
    n_confident = (
        np.isfinite(ses["x_maze"]) & np.isfinite(ses["y_maze"]) & ~ses["bad_behav"]
    ).sum()
    n_valid = valid.sum()
    if pos_mode == "Raw":
        suffix = "(raw, all finite frames)"
    else:
        suffix = "(cleaned)"
    st.markdown(f"**Frames:** {n_valid}/{n_total} ({n_confident} cleaned-confident) {suffix}")

    total_dur_s = frame_times[-1] - frame_times[0]
    total_dur_min = total_dur_s / 60.0

    time_range = st.slider(
        "Time range (minutes)",
        0.0,
        float(np.ceil(total_dur_min)),
        (0.0, float(np.ceil(total_dur_min))),
        0.1,
        key="maze_anim_time",
        help="Select a time window to animate. Shorter windows render faster.",
    )

    t0 = frame_times[0]
    time_mask = valid.copy()
    time_mask &= (frame_times >= t0 + time_range[0] * 60) & (
        frame_times <= t0 + time_range[1] * 60
    )

    if time_mask.sum() < 5:
        st.warning("Selected time range has too few frames.")
        st.stop()

    x_sel = x_maze[time_mask]
    y_sel = y_maze[time_mask]
    hd_sel = hd_deg[time_mask]
    speed_sel = speed[time_mask]
    light_sel = light_on[time_mask]
    ft_sel = frame_times[time_mask]

    # Bodypart positions for skeleton — match raw vs cleaned to position mode.
    bp_maze_data = ses_bp_maze if show_skeleton else None
    bp_sel = None
    if bp_maze_data:
        bp_sel = {}
        for bp_name, bp_d in bp_maze_data.items():
            bp_sel[bp_name] = {
                "x": bp_d["x"][time_mask],
                "y": bp_d["y"][time_mask],
            }

    n_frames_anim = len(x_sel) // subsample
    st.caption(f"Rendering {n_frames_anim} frames ({len(x_sel)} total, subsample {subsample})")

    if n_frames_anim > 30000:
        st.info(
            f"Large frame count ({n_frames_anim}). The canvas handles this "
            "well but the JSON payload will be large. Consider raising the "
            "Subsample slider or narrowing the time range."
        )

    with st.spinner("Building animation data..."):
        payload = _build_canvas_payload(
            x_sel,
            y_sel,
            hd_sel,
            speed_sel,
            light_sel,
            ft_sel,
            trail_seconds=trail_s,
            step=subsample,
            arrow_length=arrow_len,
            bp_maze=bp_sel,
            show_position=show_position,
            show_skeleton=show_skeleton,
        )

    render_maze_canvas(payload, height=780)

    # ── Static trajectory plot (Plotly, unchanged) ───────────────────
    with st.expander("Full trajectory (static)"):
        fig_static = go.Figure()
        _draw_maze(fig_static)

        fig_static.add_trace(
            go.Scatter(
                x=x_sel,
                y=y_sel,
                mode="markers",
                marker=dict(
                    size=2,
                    color=ft_sel - ft_sel[0],
                    colorscale="Viridis",
                    colorbar=dict(title="Time (s)"),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_static.update_layout(
            xaxis=dict(
                range=[-0.5, 7.5], scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False
            ),
            yaxis=dict(range=[-0.5, 5.5], showgrid=False, zeroline=False),
            width=700,
            height=540,
            margin=dict(l=40, r=40, t=20, b=40),
            title="Trajectory coloured by time",
        )
        st.plotly_chart(fig_static, use_container_width=False)


_page()
