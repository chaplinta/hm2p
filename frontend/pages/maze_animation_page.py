"""Maze Animation — animated replay of mouse trajectory through the q-rose maze.

Shows the mouse position and head direction arrow as it moves through the maze,
with a trail showing recent trajectory. Uses Plotly animation frames for smooth
playback without video files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import load_all_sync_data, render_tracker_provenance, session_filter_controls as session_filter_sidebar

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
    "nose_tip": "#7F00FF",          # purple
    "nose": "#7F00FF",              # SuperAnimal alias
    "left_ear": "#376DF8",          # blue
    "right_ear": "#12C7E5",         # cyan
    "head_midpoint": "#5AF8C7",     # aqua
    "implant_base_rear": "#5AF8C7", # legacy DLC alias
    "neck": "#A4F89E",              # green
    "mid_back": "#ECC76E",          # yellow
    "mouse_center": "#FF6D38",      # orange
    "tail_base": "#FF0000",         # red
}

# ── Maze boundary polygon (7×5 q-rose maze) ───────────────────────────
# This traces the outer wall of the 23 accessible cells.
_MAZE_WALLS_X = [0, 3, 3, 2, 2, 5, 5, 4, 4, 7, 7, 6, 6, 7, 7, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 3, 3, 0, 0, 1, 1, 0, 0]
_MAZE_WALLS_Y = [0, 0, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1, 4, 4, 5, 5, 4, 4, 3, 3, 5, 5, 3, 3, 4, 4, 5, 5, 4, 4, 1, 1, 0]


def _draw_maze(fig: go.Figure) -> None:
    """Add the q-rose maze boundary walls to a Plotly figure."""
    fig.add_trace(go.Scatter(
        x=_MAZE_WALLS_X, y=_MAZE_WALLS_Y,
        mode="lines",
        line=dict(color="black", width=2),
        showlegend=False,
        hoverinfo="skip",
        name="walls",
    ))


def _subsample(arr: np.ndarray, step: int) -> np.ndarray:
    """Take every `step`-th element."""
    return arr[::step]


def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Fill NaN gaps with linear interpolation; extrapolate edges with nearest."""
    out = arr.copy()
    finite = np.isfinite(out)
    if finite.all() or not finite.any():
        return out
    idx = np.arange(len(out))
    out[~finite] = np.interp(idx[~finite], idx[finite], out[finite])
    return out


def _build_animation_figure(
    x: np.ndarray,
    y: np.ndarray,
    hd: np.ndarray,
    speed: np.ndarray,
    light_on: np.ndarray,
    frame_times: np.ndarray,
    trail_seconds: float,
    step: int,
    arrow_length: float,
    bp_maze: dict | None = None,
    playback_speed: float = 1.0,
    show_position: bool = True,
) -> go.Figure:
    """Build a Plotly figure with animation frames for mouse trajectory.

    Each animation frame shows:
    - The full maze walls
    - A fading trail of recent positions
    - The current position as a filled circle
    - An arrow indicating head direction
    - Colour indicates light state (orange = light on, grey = dark)
    """
    # Subsample to reduce frame count
    x = _subsample(x, step)
    y = _subsample(y, step)
    hd = _subsample(hd, step)
    speed = _subsample(speed, step)
    light_on = _subsample(light_on, step)
    frame_times = _subsample(frame_times, step)

    # Subsample bodypart positions too
    bp_sub = None
    if bp_maze:
        bp_sub = {}
        for bp_name, bp_data in bp_maze.items():
            bp_sub[bp_name] = {
                "x": _subsample(bp_data["x"], step),
                "y": _subsample(bp_data["y"], step),
            }

    n = len(x)
    if n == 0:
        return go.Figure()

    # Compute trail length in subsampled frames
    dt = np.median(np.diff(frame_times)) if n > 1 else 1.0
    trail_frames = max(1, int(trail_seconds / dt))

    # Precompute HD arrow endpoints.
    # The kinematics module computes HD as ``atan2(ear_left_x - ear_right_x,
    # ear_left_y - ear_right_y) + 180`` (compute.py:202–205) — note dx, dy
    # argument order. That convention gives HD=0 → heading +y and
    # HD=90 → heading +x. The line endpoints therefore use sin/cos in that
    # swapped order so the line direction matches the HD value (and matches
    # the plotly arrow-marker ``angle`` parameter which is also y-aligned).
    hd_rad = np.deg2rad(hd)
    dx = arrow_length * np.sin(hd_rad)
    dy = arrow_length * np.cos(hd_rad)

    # Build frames — every (post-subsample) input frame becomes one
    # animation frame. No internal decimation: the user controls frame
    # count via the Subsample slider on the page.
    frames = []
    frame_indices = list(range(0, n, 1))

    # Surround rectangle (covers entire plot area) — visible only during dark
    # We draw it as a filled scatter polygon that sits behind everything.
    # During light-on frames it's transparent; during dark it's grey.
    _SURROUND_X = [-0.5, 7.5, 7.5, -0.5, -0.5]
    _SURROUND_Y = [-0.5, -0.5, 5.5, 5.5, -0.5]

    for i in frame_indices:
        trail_start = max(0, i - trail_frames)
        trail_x = x[trail_start:i + 1]
        trail_y = y[trail_start:i + 1]

        # Trail opacity: fades from transparent to solid
        n_trail = len(trail_x)
        if n_trail > 1:
            opacities = np.linspace(0.1, 0.8, n_trail)
        else:
            opacities = np.array([0.8])

        # Trail colour: mouse_center from DLC rainbow (#FF6D38), dimmed in dark
        light_color = "rgba(255, 109, 56, {a})" if light_on[i] else "rgba(140, 140, 140, {a})"
        trail_colors = [light_color.format(a=f"{op:.2f}") for op in opacities]

        # Head position marker — mouse_center color from DLC rainbow
        head_color = "#FF6D38" if light_on[i] else "#8C8C8C"

        # Surround fill: grey when dark, transparent when light
        surround_fill = "rgba(60, 60, 60, 0.55)" if not light_on[i] else "rgba(0, 0, 0, 0)"
        wall_color = "black" if light_on[i] else "rgba(200, 200, 200, 0.8)"

        # Arrow line (head position → arrow tip) — only if HD is valid
        has_hd = np.isfinite(hd[i])
        if has_hd:
            ax = x[i] + dx[i]
            ay = y[i] + dy[i]

        t_s = frame_times[i] - frame_times[0]
        t_min = t_s / 60.0

        # Wrap unwrapped HD (cumulative) into [0, 360) for display.
        hd_wrapped = (hd[i] % 360.0) if has_hd else float("nan")
        hd_text = f"HD={hd_wrapped:.0f}°, " if has_hd else ""
        spd_text = f"speed={speed[i]:.1f} cm/s" if np.isfinite(speed[i]) else ""

        frame_data = [
            # Dark surround (grey fill covering area outside maze)
            go.Scatter(
                x=_SURROUND_X, y=_SURROUND_Y,
                mode="lines",
                fill="toself",
                fillcolor=surround_fill,
                line=dict(color="rgba(0,0,0,0)", width=0),
                showlegend=False, hoverinfo="skip",
            ),
            # Maze walls
            go.Scatter(
                x=_MAZE_WALLS_X, y=_MAZE_WALLS_Y,
                mode="lines",
                line=dict(color=wall_color, width=2),
                showlegend=False, hoverinfo="skip",
            ),
            # Trail
            go.Scatter(
                x=trail_x.tolist(), y=trail_y.tolist(),
                mode="markers",
                marker=dict(size=3, color=trail_colors),
                showlegend=False, hoverinfo="skip",
            ),
        ]

        # Skeleton: all lines as one trace, all dots as one trace
        # (Plotly animation needs consistent trace count per frame)
        skel_line_x, skel_line_y = [], []
        skel_dot_x, skel_dot_y, skel_dot_colors = [], [], []
        if bp_sub:
            for bp1, bp2 in _SKELETON:
                if bp1 not in bp_sub or bp2 not in bp_sub:
                    continue
                x1 = float(bp_sub[bp1]["x"][i])
                y1 = float(bp_sub[bp1]["y"][i])
                x2 = float(bp_sub[bp2]["x"][i])
                y2 = float(bp_sub[bp2]["y"][i])
                if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                    continue
                skel_line_x.extend([x1, x2, None])
                skel_line_y.extend([y1, y2, None])
            for bp_name, bp_d in bp_sub.items():
                bx = float(bp_d["x"][i])
                by = float(bp_d["y"][i])
                if np.isnan(bx) or np.isnan(by):
                    continue
                skel_dot_x.append(bx)
                skel_dot_y.append(by)
                skel_dot_colors.append(_BP_COLORS.get(bp_name, "#888888"))

        # Position circle + HD arrow are gated by show_position so the user
        # can hide them and watch only the trail + skeleton. Trace count
        # must stay constant across animation frames, so when hidden we
        # keep the trace structure with empty x/y arrays.
        pos_x = [x[i]] if show_position else []
        pos_y = [y[i]] if show_position else []
        arrow_x = [x[i], ax] if (show_position and has_hd) else []
        arrow_y = [y[i], ay] if (show_position and has_hd) else []
        head_x = [ax] if (show_position and has_hd) else []
        head_y = [ay] if (show_position and has_hd) else []

        frame_data += [
            # Skeleton lines (single trace with None separators)
            go.Scatter(
                x=skel_line_x, y=skel_line_y,
                mode="lines",
                line=dict(color="rgba(180,180,180,0.7)", width=1.5),
                showlegend=False, hoverinfo="skip",
            ),
            # Skeleton keypoint dots (single trace)
            go.Scatter(
                x=skel_dot_x, y=skel_dot_y,
                mode="markers",
                marker=dict(size=6, color=skel_dot_colors if skel_dot_colors else ["#888"],
                            line=dict(color="black", width=0.5)),
                showlegend=False, hoverinfo="skip",
            ),
            # Current position (centroid — smaller when skeleton is shown)
            go.Scatter(
                x=pos_x, y=pos_y,
                mode="markers",
                marker=dict(size=10 if not bp_sub else 4, color=head_color,
                            line=dict(color="black" if light_on[i] else "white", width=1)),
                showlegend=False,
                hovertext=f"t={t_min:.1f} min, {hd_text}{spd_text}",
                hoverinfo="text",
            ),
            # HD arrow line — nose_tip color from DLC rainbow (#7F00FF purple)
            go.Scatter(
                x=arrow_x, y=arrow_y,
                mode="lines",
                line=dict(color="#7F00FF" if light_on[i] else "#A080D0", width=2),
                showlegend=False, hoverinfo="skip",
            ),
            # Arrowhead. Plotly arrow marker uses the same y-up convention
            # as our HD (HD=0 → up), so angle = -hd_wrapped points it the
            # same direction as the line.
            go.Scatter(
                x=head_x, y=head_y,
                mode="markers",
                marker=dict(
                    size=8,
                    color="#7F00FF" if light_on[i] else "#A080D0",
                    symbol="arrow",
                    angle=-hd_wrapped if has_hd else 0,
                ),
                showlegend=False, hoverinfo="skip",
            ),
        ]
        frames.append(go.Frame(
            data=frame_data,
            name=str(i),
            layout=go.Layout(title_text=f"t = {t_min:.1f} min | {f'HD = {hd_wrapped:.0f}° | ' if has_hd else ''}{f'{speed[i]:.1f} cm/s | ' if np.isfinite(speed[i]) else ''}{'Light' if light_on[i] else 'Dark'}"),
        ))

    # Initial frame
    first = frames[0] if frames else None

    # Static legend traces — one per skeleton bodypart present in the data.
    # These sit AFTER the animated traces in fig.data; frames update only the
    # first len(frame.data) traces, so these stay constant and just populate
    # the legend with the matching DLC colour. Order matches the skeleton's
    # head-to-tail traversal so the legend reads sensibly.
    _BP_LEGEND_ORDER = [
        "nose_tip", "nose", "left_ear", "right_ear", "head_midpoint",
        "implant_base_rear", "neck", "mid_back", "mouse_center", "tail_base",
    ]
    legend_traces: list[go.Scatter] = []
    if bp_sub:
        for bp_name in _BP_LEGEND_ORDER:
            if bp_name not in bp_sub:
                continue
            legend_traces.append(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=8, color=_BP_COLORS.get(bp_name, "#888888"),
                            line=dict(color="black", width=0.5)),
                name=bp_name,
                showlegend=True,
                hoverinfo="skip",
            ))

    fig = go.Figure(
        data=(list(first.data) + legend_traces) if first else legend_traces,
        layout=go.Layout(
            xaxis=dict(range=[-0.5, 7.5], scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, title="x (maze units)"),
            yaxis=dict(range=[-0.5, 5.5], showgrid=False, zeroline=False, title="y (maze units)"),
            width=900 if legend_traces else 800,
            height=620,
            margin=dict(l=40, r=40, t=60, b=40),
            title=first.layout.title if first else None,
            showlegend=bool(legend_traces),
            legend=dict(
                title="Bodyparts",
                x=1.02, y=1.0, xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.2)", borderwidth=1,
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.12, x=0.5, xanchor="center",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=max(20, int(round(dt * 1000 / playback_speed))), redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]),
                ],
            )],
            sliders=[dict(
                active=0,
                yanchor="top", xanchor="left",
                currentvalue=dict(prefix="Frame: ", visible=True),
                transition=dict(duration=0),
                pad=dict(b=10, t=40),
                len=0.9, x=0.05,
                steps=[dict(args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))],
                            label=f"{(frame_times[i] - frame_times[0]) / 60:.1f}m",
                            method="animate")
                       for i in frame_indices],
            )],
        ),
        frames=frames,
    )

    return fig


# ── Page ──────────────────────────────────────────────────────────────────


def _page() -> None:
    """Render the maze animation page (called by Streamlit runner)."""
    st.title("Maze Animation")
    st.caption(
        "Animated replay of mouse trajectory through the Rosenberg maze. "
        "Shows head position, facing direction (blue arrow), and recent trail. "
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
            "excluded (shown as NaN in the data)."
        )

    from frontend.data import check_stale_data_warning
    check_stale_data_warning(stages=["kinematics", "sync"], block=True)

    with st.spinner("Loading sync data..."):
        all_data = load_all_sync_data()

    if all_data["n_sessions"] == 0:
        st.warning(
            "No sync.h5 data available. This page requires completed pipeline stages 0-5."
        )
        st.stop()

    sessions = session_filter_sidebar(
        all_data["sessions"], show_roi_filter=False, key_prefix="maze_anim"
    )

    if not sessions:
        st.warning("No sessions match the current filters.")
        st.stop()
    render_tracker_provenance(sessions)

    sessions_with_pos = [s for s in sessions if s.get("x_maze") is not None and s.get("y_maze") is not None]

    if not sessions_with_pos:
        st.warning("No sessions have position data (kinematics.h5 not yet generated).")
        st.stop()

    col_sel, col_opts = st.columns([2, 3])

    with col_sel:
        session_labels = [f"{s['exp_id']} ({s['celltype']})" for s in sessions_with_pos]
        selected_idx = st.selectbox("Session", range(len(session_labels)), format_func=lambda i: session_labels[i], key="maze_anim_session")

    with col_opts:
        # Row 1
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
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
            playback_label = st.selectbox(
                "Playback speed",
                ["0.25× realtime", "0.5× realtime", "1× realtime", "2× realtime", "4× realtime"],
                index=2,
                key="maze_anim_speed",
                help="Wall-clock speed relative to recorded time. 1× = real-time replay.",
            )
            playback_speed = float(playback_label.split("×")[0])
        with r1c3:
            subsample = st.slider("Subsample (every N frames)", 1, 30, 1, 1, key="maze_anim_sub",
                                  help="Reduce frame count for browser performance (does not change playback time-base).")
        with r1c4:
            trail_s = st.slider("Trail length (s)", 1.0, 30.0, 10.0, 1.0, key="maze_anim_trail")

        # Row 2
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            show_position = st.checkbox(
                "Show position + HD arrow", value=True, key="maze_anim_show_pos",
                help="Display the body-centroid circle and the head-direction arrow.",
            )
        with r2c2:
            show_skeleton = st.checkbox(
                "Show skeleton", value=True, key="maze_anim_skel",
                help="Draw DLC bodypart skeleton (requires per-bodypart maze coordinates in sync.h5).",
            )
        with r2c3:
            arrow_len = st.slider("Arrow length", 0.1, 1.5, 0.5, 0.1, key="maze_anim_arrow")
        with r2c4:
            st.empty()

    ses = sessions_with_pos[selected_idx]

    hd_deg = ses["hd_deg"]
    speed = ses["speed_cm_s"]
    light_on = ses["light_on"]
    frame_times = ses["frame_times"]

    if pos_mode == "Raw":
        # Truly raw: load x_maze_raw / y_maze_raw written by the kinematics
        # stage from the unfiltered pose. Older sync.h5 files do not have
        # these fields — fall back to the cleaned position with a notice.
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
        # Raw position has no NaN gaps in the kinematics fields, so no
        # interpolation is applied. HD and speed remain from the cleaned
        # pipeline (these are derived signals; raw analogues would be
        # extremely noisy and not meaningful for visualisation).
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
    n_confident = (np.isfinite(ses["x_maze"]) & np.isfinite(ses["y_maze"]) & ~ses["bad_behav"]).sum()
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
        0.0, float(np.ceil(total_dur_min)),
        (0.0, float(np.ceil(total_dur_min))),
        0.1,
        key="maze_anim_time",
        help="Select a time window to animate. Shorter windows render faster.",
    )

    t0 = frame_times[0]
    time_mask = valid.copy()
    time_mask &= (frame_times >= t0 + time_range[0] * 60) & (frame_times <= t0 + time_range[1] * 60)

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
    st.caption(f"Generating {n_frames_anim} animation frames...")

    if n_frames_anim > 2000:
        st.info(
            f"Large frame count ({n_frames_anim}). The animation shows every "
            "frame — no internal decimation — so playback may be slow or "
            "stutter in the browser. Raise the Subsample slider or narrow "
            "the time range to lighten the load."
        )

    with st.spinner("Building animation..."):
        fig = _build_animation_figure(
            x_sel, y_sel, hd_sel, speed_sel, light_sel, ft_sel,
            trail_seconds=trail_s,
            step=subsample,
            arrow_length=arrow_len,
            bp_maze=bp_sel,
            playback_speed=playback_speed,
            show_position=show_position,
        )

    st.plotly_chart(fig, use_container_width=False)

    with st.expander("Full trajectory (static)"):
        fig_static = go.Figure()
        _draw_maze(fig_static)

        fig_static.add_trace(go.Scatter(
            x=x_sel, y=y_sel,
            mode="markers",
            marker=dict(
                size=2,
                color=ft_sel - ft_sel[0],
                colorscale="Viridis",
                colorbar=dict(title="Time (s)"),
            ),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig_static.update_layout(
            xaxis=dict(range=[-0.5, 7.5], scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False),
            yaxis=dict(range=[-0.5, 5.5], showgrid=False, zeroline=False),
            width=700, height=540,
            margin=dict(l=40, r=40, t=20, b=40),
            title="Trajectory coloured by time",
        )
        st.plotly_chart(fig_static, use_container_width=False)


_page()
