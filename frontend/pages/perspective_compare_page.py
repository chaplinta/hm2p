"""Perspective Correction Comparison — side-by-side uncorrected vs corrected trajectories.

Shows two maze plots: the left has raw positions, the right has positions after
perspective correction. This lets you see the effect of projecting bodypart
heights to the ground plane (removing parallax from the off-axis camera).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import load_all_sync_data, session_filter_sidebar
from frontend.pages.maze_animation_page import _MAZE_WALLS_X, _MAZE_WALLS_Y

# ── Perspective correction (operates on pixel-level data, before mm/maze conversion)
# Since sync.h5 stores maze-coordinate positions (already converted), we reverse
# the correction direction: we show the CURRENT data as "corrected" and compute
# what the uncorrected version would look like by applying the INVERSE transform.
#
# However, if the pipeline was run WITHOUT perspective correction (camera_center_px=None),
# the current data IS uncorrected. In that case we apply the forward correction.
#
# For this comparison page we work entirely in maze coordinates and apply the
# perspective correction formula in maze-unit space. The correction is a simple
# radial scaling toward/away from the camera centre, which is linear and works
# in any coordinate system.


def _apply_perspective_in_maze_coords(
    x_maze: np.ndarray,
    y_maze: np.ndarray,
    camera_center_maze: tuple[float, float],
    camera_height_mm: float,
    bodypart_height_mm: float,
    inverse: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply (or invert) perspective correction in maze coordinates.

    Args:
        x_maze, y_maze: Positions in maze units (0-7, 0-5).
        camera_center_maze: Camera optical centre in maze units.
        camera_height_mm: Camera height above floor.
        bodypart_height_mm: Bodypart height above floor.
        inverse: If True, apply the inverse (uncorrect).

    Returns:
        (x_corrected, y_corrected) in maze units.
    """
    if bodypart_height_mm == 0:
        return x_maze.copy(), y_maze.copy()

    cx, cy = camera_center_maze
    scale = camera_height_mm / (camera_height_mm - bodypart_height_mm)

    if inverse:
        # Inverse: multiply displacement by scale (undo the correction)
        x_out = cx + (x_maze - cx) * scale
        y_out = cy + (y_maze - cy) * scale
    else:
        # Forward: divide displacement by scale
        x_out = cx + (x_maze - cx) / scale
        y_out = cy + (y_maze - cy) / scale

    return x_out, y_out


def _estimate_camera_center_maze(
    maze_corners_px: np.ndarray | None,
    scale_mm_per_px: float | None,
) -> tuple[float, float]:
    """Estimate camera centre in maze coordinates.

    Uses a typical crop offset of (108, 261) for sessions with f4mm lens.
    The camera centre in the cropped frame is (532, 251), which maps to
    approximately (3.8, 2.1) in maze units for a typical session.

    For a more accurate estimate we'd need the per-session meta.txt, but
    for visualization purposes this approximation is sufficient.
    """
    # Typical camera centre in cropped frame pixels
    # (1280/2 - 108, 1024/2 - 261) = (532, 251)
    # In a typical session, maze spans roughly x=[149,764] y=[72,509] in pixels
    # So camera centre in maze fraction: (532-149)/(764-149) ≈ 0.62, (251-72)/(509-72) ≈ 0.41
    # In maze units: 0.62 * 7 ≈ 4.3, 0.41 * 5 ≈ 2.1
    return (4.3, 2.1)


def _build_comparison_figure(
    x_maze: np.ndarray,
    y_maze: np.ndarray,
    hd_deg: np.ndarray,
    light_on: np.ndarray,
    camera_center_maze: tuple[float, float],
    bodypart_height_mm: float,
    camera_height_mm: float = 700.0,
) -> go.Figure:
    """Build side-by-side Plotly figure: uncorrected vs corrected."""

    # Assume current data is uncorrected (pipeline hasn't re-run yet).
    # Apply forward correction to get the corrected version.
    x_corr, y_corr = _apply_perspective_in_maze_coords(
        x_maze, y_maze, camera_center_maze,
        camera_height_mm, bodypart_height_mm, inverse=False,
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Uncorrected (raw)", "Perspective-corrected"],
        horizontal_spacing=0.08,
    )

    # Colour array: orange for light, grey for dark
    colors = np.where(light_on, "rgba(255,165,0,0.3)", "rgba(100,100,100,0.3)")

    for col, (xd, yd, title) in enumerate([
        (x_maze, y_maze, "Raw"),
        (x_corr, y_corr, "Corrected"),
    ], start=1):
        # Maze walls
        fig.add_trace(go.Scatter(
            x=_MAZE_WALLS_X, y=_MAZE_WALLS_Y,
            mode="lines", line=dict(color="black", width=2),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=col)

        # Trajectory
        fig.add_trace(go.Scatter(
            x=xd, y=yd,
            mode="markers",
            marker=dict(size=1.5, color=colors.tolist()),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=col)

        # Camera centre
        fig.add_trace(go.Scatter(
            x=[camera_center_maze[0]], y=[camera_center_maze[1]],
            mode="markers",
            marker=dict(size=8, color="red", symbol="x"),
            showlegend=False, name="Camera centre",
            hovertext="Camera optical centre",
            hoverinfo="text",
        ), row=1, col=col)

    # Layout
    for col in [1, 2]:
        fig.update_xaxes(range=[-0.5, 7.5], scaleanchor=f"y{col if col > 1 else ''}", scaleratio=1,
                         showgrid=False, zeroline=False, title="x (maze units)", row=1, col=col)
        fig.update_yaxes(range=[-0.5, 5.5], showgrid=False, zeroline=False,
                         title="y (maze units)" if col == 1 else "", row=1, col=col)

    fig.update_layout(
        width=1100, height=500,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig, x_corr, y_corr


def _out_of_bounds_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Count positions outside the 0-7 x 0-5 maze bounding box."""
    valid = np.isfinite(x) & np.isfinite(y)
    n_valid = valid.sum()
    if n_valid == 0:
        return {"n_valid": 0, "n_oob": 0, "pct_oob": 0.0}
    oob = (x[valid] < 0) | (x[valid] > 7) | (y[valid] < 0) | (y[valid] > 5)
    n_oob = oob.sum()
    return {"n_valid": int(n_valid), "n_oob": int(n_oob), "pct_oob": 100.0 * n_oob / n_valid}


# ── Page ──────────────────────────────────────────────────────────────────


def _page() -> None:
    """Render the perspective correction comparison page."""
    st.title("Perspective Correction")
    st.caption(
        "Side-by-side comparison of mouse trajectories before and after perspective "
        "correction. The correction projects bodypart positions from their height "
        "above the maze floor to the ground plane, removing parallax displacement "
        "from the off-axis overhead camera."
    )

    # Check for stale data before starting the slow download
    from frontend.data import check_stale_data_warning
    check_stale_data_warning(stages=["kinematics", "sync"], block=True)

    with st.spinner("Loading sync data..."):
        all_data = load_all_sync_data()

    if all_data["n_sessions"] == 0:
        st.warning("No sync.h5 data available.")
        st.stop()

    sessions = session_filter_sidebar(
        all_data["sessions"], show_roi_filter=False, key_prefix="persp"
    )

    sessions_with_pos = [s for s in sessions if s.get("x_maze") is not None]

    if not sessions_with_pos:
        st.warning("No sessions have position data.")
        st.stop()

    col1, col2 = st.columns([2, 3])

    with col1:
        labels = [f"{s['exp_id']} ({s['celltype']})" for s in sessions_with_pos]
        sel_idx = st.selectbox("Session", range(len(labels)), format_func=lambda i: labels[i], key="persp_ses")

    with col2:
        c1, c2 = st.columns(2)
        with c1:
            height_mm = st.slider(
                "Bodypart height (mm)", 0, 80, 30, 5,
                key="persp_height",
                help="Estimated height of the body centroid above the maze floor. "
                     "Mouse body ~20mm, ears with 2P implant ~40mm. "
                     "This slider lets you see the effect at different heights.",
            )
        with c2:
            cam_height = st.number_input(
                "Camera height (mm)", 500, 1000, 700, 50,
                key="persp_cam",
                help="Distance from camera sensor to maze floor.",
            )

    ses = sessions_with_pos[sel_idx]
    x_maze = ses["x_maze"]
    y_maze = ses["y_maze"]
    hd_deg = ses["hd_deg"]
    light_on = ses["light_on"]

    valid = np.isfinite(x_maze) & np.isfinite(y_maze) & ~ses["bad_behav"]
    x_v = x_maze[valid]
    y_v = y_maze[valid]
    hd_v = hd_deg[valid]
    light_v = light_on[valid]

    camera_center_maze = _estimate_camera_center_maze(None, None)

    fig, x_corr, y_corr = _build_comparison_figure(
        x_v, y_v, hd_v, light_v,
        camera_center_maze, float(height_mm), float(cam_height),
    )
    st.plotly_chart(fig, use_container_width=False)

    raw_stats = _out_of_bounds_stats(x_v, y_v)
    corr_stats = _out_of_bounds_stats(x_corr, y_corr)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Out-of-bounds (raw)", f"{raw_stats['pct_oob']:.1f}%", help="Positions outside the 7×5 maze bounding box")
    with col_b:
        st.metric("Out-of-bounds (corrected)", f"{corr_stats['pct_oob']:.1f}%",
                  delta=f"{corr_stats['pct_oob'] - raw_stats['pct_oob']:.1f}%",
                  delta_color="inverse")
    with col_c:
        disp = np.sqrt((x_corr - x_v)**2 + (y_corr - y_v)**2)
        st.metric("Mean correction", f"{np.nanmean(disp):.3f} maze units",
                  help="Average distance each point moved due to correction")

    with st.expander("Correction displacement map"):
        st.markdown(
            "Arrows show the direction and magnitude of the correction at each point. "
            "Points near the camera centre (red x) move less; points at the edges move more."
        )
        step = max(1, len(x_v) // 500)
        xs, ys = x_v[::step], y_v[::step]
        xc, yc = x_corr[::step], y_corr[::step]

        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(
            x=_MAZE_WALLS_X, y=_MAZE_WALLS_Y,
            mode="lines", line=dict(color="black", width=2),
            showlegend=False, hoverinfo="skip",
        ))

        for j in range(len(xs)):
            if np.isfinite(xs[j]) and np.isfinite(xc[j]):
                fig_q.add_annotation(
                    x=xc[j], y=yc[j], ax=xs[j], ay=ys[j],
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=1,
                    arrowcolor="rgba(255,0,0,0.3)",
                )

        fig_q.add_trace(go.Scatter(
            x=[camera_center_maze[0]], y=[camera_center_maze[1]],
            mode="markers", marker=dict(size=10, color="red", symbol="x"),
            showlegend=False, hovertext="Camera centre", hoverinfo="text",
        ))
        fig_q.update_layout(
            xaxis=dict(range=[-0.5, 7.5], scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False),
            yaxis=dict(range=[-0.5, 5.5], showgrid=False, zeroline=False),
            width=700, height=540, margin=dict(l=40, r=40, t=20, b=40),
        )
        st.plotly_chart(fig_q, use_container_width=False)


# Guard: only run when executed by Streamlit, not when imported by tests.
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is not None:
        _page()
except ImportError:
    _page()
