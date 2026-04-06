"""Maze Animation — animated replay of mouse trajectory through the Rosenberg maze.

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

from frontend.data import load_all_sync_data, session_filter_sidebar

# ── Maze boundary polygon (7×5 Rosenberg maze) ───────────────────────────
# This traces the outer wall of the 23 accessible cells.
_MAZE_WALLS_X = [0, 3, 3, 2, 2, 5, 5, 4, 4, 7, 7, 6, 6, 7, 7, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 3, 3, 0, 0, 1, 1, 0, 0]
_MAZE_WALLS_Y = [0, 0, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1, 4, 4, 5, 5, 4, 4, 3, 3, 5, 5, 3, 3, 4, 4, 5, 5, 4, 4, 1, 1, 0]


def _draw_maze(fig: go.Figure) -> None:
    """Add the Rosenberg maze boundary walls to a Plotly figure."""
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

        # Trail colour based on light state
        light_color = "rgba(255, 165, 0, {a})" if light_on[i] else "rgba(100, 100, 100, {a})"
        trail_colors = [light_color.format(a=f"{op:.2f}") for op in opacities]

        # Head position marker
        head_color = "orange" if light_on[i] else "dimgrey"

        # Surround fill: grey when dark, transparent when light
        surround_fill = "rgba(60, 60, 60, 0.55)" if not light_on[i] else "rgba(0, 0, 0, 0)"
        wall_color = "black" if light_on[i] else "rgba(200, 200, 200, 0.8)"

        # Arrow line (head position → arrow tip)
        ax = x[i] + dx[i]
        ay = y[i] + dy[i]

        t_s = frame_times[i] - frame_times[0]
        t_min = t_s / 60.0

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
            # Current position (head)
            go.Scatter(
                x=[x[i]], y=[y[i]],
                mode="markers",
                marker=dict(size=10, color=head_color, line=dict(color="black" if light_on[i] else "white", width=1)),
                showlegend=False,
                hovertext=f"t={t_min:.1f} min, HD={hd[i]:.0f}°, speed={speed[i]:.1f} cm/s",
                hoverinfo="text",
            ),
            # HD arrow (line from head to arrow tip)
            go.Scatter(
                x=[x[i], ax], y=[y[i], ay],
                mode="lines",
                line=dict(color="deepskyblue" if not light_on[i] else "blue", width=2),
                showlegend=False, hoverinfo="skip",
            ),
            # Arrowhead (small triangle marker at tip)
            go.Scatter(
                x=[ax], y=[ay],
                mode="markers",
                marker=dict(
                    size=8,
                    color="deepskyblue" if not light_on[i] else "blue",
                    symbol="arrow",
                    angle=-(hd[i] % 360),  # Plotly arrow angles are CCW from right
                ),
                showlegend=False, hoverinfo="skip",
            ),
        ]
        frames.append(go.Frame(
            data=frame_data,
            name=str(i),
            layout=go.Layout(title_text=f"t = {t_min:.1f} min | HD = {hd[i]:.0f}° | {speed[i]:.1f} cm/s | {'Light' if light_on[i] else 'Dark'}"),
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

    sessions_with_pos = [s for s in sessions if s.get("x_maze") is not None and s.get("y_maze") is not None]

    if not sessions_with_pos:
        st.warning("No sessions have position data (kinematics.h5 not yet generated).")
        st.stop()

    col_sel, col_opts = st.columns([2, 3])

    with col_sel:
        session_labels = [f"{s['exp_id']} ({s['celltype']})" for s in sessions_with_pos]
        selected_idx = st.selectbox("Session", range(len(session_labels)), format_func=lambda i: session_labels[i], key="maze_anim_session")

    with col_opts:
        c1, c2, c3 = st.columns(3)
        with c1:
            trail_s = st.slider("Trail length (s)", 1.0, 30.0, 10.0, 1.0, key="maze_anim_trail")
        with c2:
            subsample = st.slider("Subsample (every N frames)", 1, 30, 10, 1, key="maze_anim_sub",
                                  help="Higher = faster animation, fewer frames. At ~9.6 Hz imaging, step=10 gives ~1 Hz playback.")
        with c3:
            arrow_len = st.slider("Arrow length", 0.1, 1.5, 0.5, 0.1, key="maze_anim_arrow")

    ses = sessions_with_pos[selected_idx]

    x_maze = ses["x_maze"]
    y_maze = ses["y_maze"]
    hd_deg = ses["hd_deg"]
    speed = ses["speed_cm_s"]
    light_on = ses["light_on"]
    frame_times = ses["frame_times"]

    valid = np.isfinite(x_maze) & np.isfinite(y_maze) & np.isfinite(hd_deg) & ~ses["bad_behav"]
    if valid.sum() < 10:
        st.warning("Not enough valid position data for this session.")
        st.stop()

    total_dur_s = frame_times[valid][-1] - frame_times[valid][0]
    total_dur_min = total_dur_s / 60.0

    st.markdown(f"**Session duration:** {total_dur_min:.1f} min ({valid.sum()} valid frames)")

    time_range = st.slider(
        "Time range (minutes)",
        0.0, float(np.ceil(total_dur_min)),
        (0.0, min(2.0, float(np.ceil(total_dur_min)))),
        0.1,
        key="maze_anim_time",
        help="Select a time window to animate. Shorter windows render faster.",
    )

    t0 = frame_times[valid][0]
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


# Guard: only run when executed by Streamlit, not when imported by tests.
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is not None:
        _page()
except ImportError:
    _page()
