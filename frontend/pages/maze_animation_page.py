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

# DLC rainbow colormap hex (matplotlib.cm.rainbow, 8 bodyparts)
_BP_COLORS = {
    "nose_tip": "#7F00FF",
    "left_ear": "#376DF8",
    "right_ear": "#12C7E5",
    "head_midpoint": "#5AF8C7",
    "neck": "#A4F89E",
    "mid_back": "#ECC76E",
    "mouse_center": "#FF6D38",
    "tail_base": "#FF0000",
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

    # Precompute HD arrow endpoints
    hd_rad = np.deg2rad(hd)
    dx = arrow_length * np.cos(hd_rad)
    dy = arrow_length * np.sin(hd_rad)

    # Build frames
    frames = []
    # We generate frames at intervals to keep total count manageable
    frame_indices = list(range(0, n, 1))
    if len(frame_indices) > 500:
        # Cap at 500 animation frames for browser performance.
        # Each frame has 6 traces; >500 frames overwhelms the browser.
        skip = len(frame_indices) // 500
        frame_indices = frame_indices[::skip]

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

        hd_text = f"HD={hd[i]:.0f}°, " if has_hd else ""
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
                x=[x[i]], y=[y[i]],
                mode="markers",
                marker=dict(size=10 if not bp_sub else 4, color=head_color,
                            line=dict(color="black" if light_on[i] else "white", width=1)),
                showlegend=False,
                hovertext=f"t={t_min:.1f} min, {hd_text}{spd_text}",
                hoverinfo="text",
            ),
            # HD arrow — nose_tip color from DLC rainbow (#7F00FF purple)
            go.Scatter(
                x=[x[i], ax] if has_hd else [], y=[y[i], ay] if has_hd else [],
                mode="lines",
                line=dict(color="#7F00FF" if light_on[i] else "#A080D0", width=2),
                showlegend=False, hoverinfo="skip",
            ),
            # Arrowhead
            go.Scatter(
                x=[ax] if has_hd else [], y=[ay] if has_hd else [],
                mode="markers",
                marker=dict(
                    size=8,
                    color="#7F00FF" if light_on[i] else "#A080D0",
                    symbol="arrow",
                    angle=-(hd[i] % 360) if has_hd else 0,
                ),
                showlegend=False, hoverinfo="skip",
            ),
        ]
        frames.append(go.Frame(
            data=frame_data,
            name=str(i),
            layout=go.Layout(title_text=f"t = {t_min:.1f} min | {f'HD = {hd[i]:.0f}° | ' if has_hd else ''}{f'{speed[i]:.1f} cm/s | ' if np.isfinite(speed[i]) else ''}{'Light' if light_on[i] else 'Dark'}"),
        ))

    # Initial frame
    first = frames[0] if frames else None
    fig = go.Figure(
        data=first.data if first else [],
        layout=go.Layout(
            xaxis=dict(range=[-0.5, 7.5], scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, title="x (maze units)"),
            yaxis=dict(range=[-0.5, 5.5], showgrid=False, zeroline=False, title="y (maze units)"),
            width=800,
            height=620,
            margin=dict(l=40, r=40, t=60, b=40),
            title=first.layout.title if first else None,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.12, x=0.5, xanchor="center",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
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
            "**Head direction** is the angle of the ear-to-ear vector "
            "(`left_ear` \u2192 `right_ear`), shown as a blue arrow.\n\n"
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
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pos_mode = st.radio(
                "Position data",
                ["Raw", "Cleaned"],
                index=0,
                key="maze_anim_pos_mode",
                help=(
                    "**Raw:** all frames shown; low-confidence gaps filled with "
                    "linear interpolation. **Cleaned:** only frames that passed "
                    "confidence filtering and median smoothing."
                ),
            )
        with c2:
            trail_s = st.slider("Trail length (s)", 1.0, 30.0, 10.0, 1.0, key="maze_anim_trail")
        with c3:
            subsample = st.slider("Subsample (every N frames)", 1, 30, 1, 1, key="maze_anim_sub",
                                  help="Higher = faster animation, fewer frames. At ~9.6 Hz imaging, step=10 gives ~1 Hz playback.")
        with c4:
            arrow_len = st.slider("Arrow length", 0.1, 1.5, 0.5, 0.1, key="maze_anim_arrow")

    ses = sessions_with_pos[selected_idx]

    x_maze = ses["x_maze"]
    y_maze = ses["y_maze"]
    hd_deg = ses["hd_deg"]
    speed = ses["speed_cm_s"]
    light_on = ses["light_on"]
    frame_times = ses["frame_times"]

    use_interp = pos_mode == "Raw"

    if use_interp:
        # Interpolate NaN gaps for continuous playback
        x_maze = _interpolate_nans(x_maze)
        y_maze = _interpolate_nans(y_maze)
        hd_deg = _interpolate_nans(hd_deg)
        speed = _interpolate_nans(speed)
        # All non-bad-behav frames are valid
        valid = ~ses["bad_behav"]
    else:
        valid = np.isfinite(x_maze) & np.isfinite(y_maze) & ~ses["bad_behav"]

    if valid.sum() < 10:
        st.warning("Not enough valid position data for this session.")
        st.stop()

    n_total = len(frame_times)
    n_confident = (np.isfinite(ses["x_maze"]) & np.isfinite(ses["y_maze"]) & ~ses["bad_behav"]).sum()
    n_valid = valid.sum()

    st.markdown(
        f"**Frames:** {n_valid}/{n_total} "
        f"({n_confident} confident, {n_total - n_confident} interpolated)"
        if use_interp else
        f"**Frames:** {n_valid}/{n_total} (cleaned only)"
    )

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

    # Bodypart positions for skeleton
    bp_maze_data = ses.get("bp_maze")
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
            f"Large number of frames ({n_frames_anim}). "
            "Consider increasing subsample or narrowing the time range for smoother playback."
        )

    with st.spinner("Building animation..."):
        fig = _build_animation_figure(
            x_sel, y_sel, hd_sel, speed_sel, light_sel, ft_sel,
            trail_seconds=trail_s,
            step=subsample,
            arrow_length=arrow_len,
            bp_maze=bp_sel,
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
