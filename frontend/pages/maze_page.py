"""Maze Analysis — rose maze topology, exploration, and navigation metrics.

Inspired by Rosenberg et al. (2021) eLife.
Visualizes maze structure, occupancy, exploration efficiency, turn bias,
and goal-directed behaviour.

Requires real kinematics data from S3 for exploration/turn/cross-session tabs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import load_all_sync_data, session_filter_sidebar
from hm2p.maze.analysis import (
    cell_occupancy,
    dead_end_visits,
    exploration_efficiency,
    maze_exploration_summary,
    occupancy_fraction,
    per_junction_turn_bias,
    sequence_entropy,
    transition_matrix,
    turn_bias,
)
from hm2p.maze.discretize import cell_sequence, discretize_position_fast, node_sequence
from hm2p.maze.topology import build_rose_maze

log = logging.getLogger("hm2p.frontend.maze")

CELLTYPE_COLORS = {"penk": "#1f77b4", "nonpenk": "#ff7f0e", "unknown": "#999999"}

st.title("Maze Analysis")
st.caption(
    "Rose maze topology, exploration efficiency, and navigation metrics. "
    "Inspired by Rosenberg et al. (2021) eLife."
)

# Build maze
maze = build_rose_maze()

# --- Maze topology visualization ---
tab_topo, tab_explore, tab_turns, tab_compare = st.tabs([
    "Topology", "Exploration", "Turn Bias", "Cross-Session",
])

with tab_topo:
    st.subheader("Rose Maze Structure")
    st.markdown(
        "The rose maze is a **7x5 unit grid** with internal walls creating corridors. "
        f"It has **{maze.n_cells} accessible cells**, **{len(maze.junctions)} T-junctions**, "
        f"and **{len(maze.dead_ends)} dead ends**."
    )

    fig = go.Figure()

    # Draw all cells
    for cell in maze.cells:
        col, row = cell
        color = {
            "dead_end": "rgba(255, 99, 71, 0.6)",
            "corridor": "rgba(200, 200, 200, 0.4)",
            "t_junction": "rgba(65, 105, 225, 0.6)",
            "crossroads": "rgba(255, 215, 0, 0.6)",
        }[maze.node_types[cell]]
        fig.add_shape(
            type="rect",
            x0=col, y0=row, x1=col + 1, y1=row + 1,
            fillcolor=color,
            line=dict(color="black", width=1),
        )
        # Label
        label = maze.node_types[cell][0].upper()
        if maze.node_types[cell] == "t_junction":
            label = "T"
        elif maze.node_types[cell] == "dead_end":
            label = "D"
        elif maze.node_types[cell] == "corridor":
            label = ""
        fig.add_annotation(
            x=col + 0.5, y=row + 0.5, text=label,
            showarrow=False, font=dict(size=12, color="black"),
        )

    # Draw edges
    for cell, nbs in maze.adj.items():
        for nb in nbs:
            if (cell[0] < nb[0]) or (cell[0] == nb[0] and cell[1] < nb[1]):
                fig.add_shape(
                    type="line",
                    x0=cell[0] + 0.5, y0=cell[1] + 0.5,
                    x1=nb[0] + 0.5, y1=nb[1] + 0.5,
                    line=dict(color="gray", width=2),
                )

    fig.update_layout(
        height=400, width=600,
        xaxis=dict(range=[-0.2, 7.2], scaleanchor="y", title="x (maze units)"),
        yaxis=dict(range=[-0.2, 5.2], title="y (maze units)"),
        title="Maze Topology (T=T-junction, D=dead end)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="maze_topo_fig")

    # Distance matrix
    with st.expander("Distance Matrix"):
        st.markdown("Shortest-path distances between all cells:")
        labels = [f"({c[0]},{c[1]})" for c in maze.cell_list]
        df = pd.DataFrame(maze.dist, index=labels, columns=labels)
        st.dataframe(df, height=400)

    # Key statistics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total cells", maze.n_cells)
    col2.metric("T-junctions", len(maze.junctions))
    col3.metric("Dead ends", len(maze.dead_ends))

    max_d = int(maze.dist.max())
    mean_d = float(maze.dist.mean())
    col1, col2, col3 = st.columns(3)
    col1.metric("Max distance", max_d)
    col2.metric("Mean distance", f"{mean_d:.1f}")
    col3.metric("Corridors", len(maze.corridors))

    st.markdown("---")
    st.markdown(
        "**Dead ends:** "
        + ", ".join(f"({c[0]},{c[1]})" for c in sorted(maze.dead_ends))
    )
    st.markdown(
        "**T-junctions:** "
        + ", ".join(f"({c[0]},{c[1]})" for c in sorted(maze.junctions))
    )


# ── Helper: filter sessions with maze position data ─────────────────────
def _sessions_with_position(sessions: list[dict]) -> list[dict]:
    """Return only sessions that have x_maze and y_maze arrays."""
    out = []
    for s in sessions:
        if s.get("x_maze") is not None and s.get("y_maze") is not None:
            x = s["x_maze"]
            y = s["y_maze"]
            if len(x) > 0 and np.any(np.isfinite(x)):
                out.append(s)
    return out


def _discretize_session(s: dict) -> np.ndarray:
    """Discretize a session's maze positions to cell indices."""
    x = np.asarray(s["x_maze"], dtype=np.float64)
    y = np.asarray(s["y_maze"], dtype=np.float64)
    # Exclude bad_behav frames by setting them to NaN
    bad = s.get("bad_behav")
    if bad is not None:
        bad = np.asarray(bad, dtype=bool)
        x = x.copy()
        y = y.copy()
        x[bad] = np.nan
        y[bad] = np.nan
    return discretize_position_fast(x, y, maze)


def _build_occupancy_grid(occ_frac: np.ndarray) -> np.ndarray:
    """Build a 7x5 grid from occupancy fraction array for heatmap display."""
    grid = np.full((5, 7), np.nan)  # rows x cols, rows=y, cols=x
    for idx, cell in enumerate(maze.cell_list):
        col, row = cell
        grid[row, col] = occ_frac[idx]
    return grid


# ── Load data for analysis tabs ──────────────────────────────────────────
with st.spinner("Loading sync data for all sessions..."):
    all_data = load_all_sync_data()

if all_data["n_sessions"] > 0:
    sessions = session_filter_sidebar(
        all_data["sessions"], show_roi_filter=False, key_prefix="maze"
    )
else:
    sessions = []

pos_sessions = _sessions_with_position(sessions) if sessions else []

# --- Exploration analysis ---
with tab_explore:
    st.subheader("Exploration Efficiency")

    if not pos_sessions:
        st.warning(
            "No sessions with maze position data available. "
            "This tab will populate when sync.h5 files contain x_maze / y_maze "
            "(from Stage 3 kinematics with maze coordinate transform)."
        )
    else:
        n_with = len(pos_sessions)
        n_total = len(sessions) if sessions else 0
        if n_with < n_total:
            st.info(
                f"{n_with} of {n_total} filtered sessions have maze position data. "
                f"Showing analysis for available sessions only."
            )
        st.markdown(f"**{n_with} sessions** with maze position data")

        # ── Pooled occupancy heatmap ────────────────────────────────────
        st.markdown("### Occupancy Heatmap")
        st.markdown(
            "Fraction of time spent in each maze cell, pooled across sessions."
        )

        # Compute per-celltype occupancy
        celltype_occ: dict[str, np.ndarray] = {}
        celltype_counts: dict[str, int] = {}
        for s in pos_sessions:
            ct = s.get("celltype", "unknown")
            cell_idx = _discretize_session(s)
            occ = occupancy_fraction(cell_idx, maze.n_cells)
            if ct not in celltype_occ:
                celltype_occ[ct] = np.zeros(maze.n_cells)
                celltype_counts[ct] = 0
            celltype_occ[ct] += occ
            celltype_counts[ct] += 1

        # Normalize per celltype (average across sessions)
        for ct in celltype_occ:
            if celltype_counts[ct] > 0:
                celltype_occ[ct] /= celltype_counts[ct]

        # Show side-by-side heatmaps if both celltypes present
        celltypes_present = sorted(celltype_occ.keys())
        heatmap_cols = st.columns(max(len(celltypes_present), 1))

        for i, ct in enumerate(celltypes_present):
            with heatmap_cols[i]:
                grid = _build_occupancy_grid(celltype_occ[ct])
                fig_occ = go.Figure(data=go.Heatmap(
                    z=grid,
                    x=[str(c) for c in range(7)],
                    y=[str(r) for r in range(5)],
                    colorscale="YlOrRd",
                    colorbar=dict(title="Frac."),
                    hoverongaps=False,
                    text=np.where(np.isnan(grid), "wall", ""),
                    hovertemplate="Col %{x}, Row %{y}<br>Occupancy: %{z:.3f}<extra></extra>",
                ))
                fig_occ.update_layout(
                    title=f"Occupancy — {ct} (n={celltype_counts[ct]})",
                    xaxis=dict(title="Column", dtick=1),
                    yaxis=dict(title="Row", dtick=1, scaleanchor="x"),
                    height=350,
                )
                st.plotly_chart(fig_occ, use_container_width=True, key=f"maze_occ_{ct}")

        # ── Coverage over time ──────────────────────────────────────────
        st.markdown("### Exploration Coverage Over Time")
        st.markdown(
            "Fraction of maze cells visited as a function of time within each session. "
            "Steeper curves indicate faster exploration."
        )

        coverage_records = []
        for s in pos_sessions:
            cell_idx = _discretize_session(s)
            ct = s.get("celltype", "unknown")
            exp_id = s.get("exp_id", "unknown")
            ft = s.get("frame_times")

            # Compute cumulative unique cells visited
            seen: set[int] = set()
            # Sample at ~100 points to keep the plot manageable
            n_frames = len(cell_idx)
            step = max(1, n_frames // 100)
            for frame_i in range(0, n_frames, step):
                c = int(cell_idx[frame_i])
                if c >= 0:
                    seen.add(c)
                # Time in seconds from start
                if ft is not None and len(ft) > frame_i:
                    t = float(ft[frame_i] - ft[0])
                else:
                    t = float(frame_i)
                coverage_records.append({
                    "session": exp_id,
                    "celltype": ct,
                    "time_s": t,
                    "coverage": len(seen) / maze.n_cells,
                })

        if coverage_records:
            df_cov = pd.DataFrame(coverage_records)
            fig_cov = px.line(
                df_cov,
                x="time_s",
                y="coverage",
                color="celltype",
                line_group="session",
                color_discrete_map=CELLTYPE_COLORS,
                labels={"time_s": "Time (s)", "coverage": "Fraction of cells visited"},
            )
            fig_cov.update_layout(
                title="Exploration Coverage Over Time",
                yaxis=dict(range=[0, 1.05]),
                height=400,
            )
            st.plotly_chart(fig_cov, use_container_width=True, key="maze_coverage_time")

        # ── Per-session summary metrics ─────────────────────────────────
        st.markdown("### Per-Session Exploration Summary")

        summary_records = []
        for s in pos_sessions:
            cell_idx = _discretize_session(s)
            ct = s.get("celltype", "unknown")
            exp_id = s.get("exp_id", "unknown")
            animal = s.get("animal_id", "unknown")
            ft = s.get("frame_times")
            fps = 1.0 / np.median(np.diff(ft)) if ft is not None and len(ft) > 1 else 9.6

            summ = maze_exploration_summary(cell_idx, maze, fps=fps)
            summary_records.append({
                "Session": exp_id,
                "Animal": animal,
                "Cell type": ct,
                "Duration (s)": round(summ["duration_s"], 1),
                "Cells visited": summ["unique_cells_visited"],
                "Coverage": round(summ["coverage_frac"], 2),
                "Transitions": summ["n_cell_transitions"],
                "Occupancy entropy": round(summ["occupancy_entropy"], 2),
                "Left turn frac": round(summ["turn_bias"]["left_frac"], 2),
            })

        if summary_records:
            df_summ = pd.DataFrame(summary_records)
            st.dataframe(df_summ, use_container_width=True)

            # Coverage by celltype box plot
            fig_box = px.box(
                df_summ,
                x="Cell type",
                y="Coverage",
                color="Cell type",
                color_discrete_map=CELLTYPE_COLORS,
                points="all",
                title="Exploration Coverage by Cell Type",
            )
            fig_box.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True, key="maze_coverage_box")

        # ── Dead-end visits ─────────────────────────────────────────────
        with st.expander("Dead-End Visit Analysis"):
            de_records = []
            for s in pos_sessions:
                cell_idx = _discretize_session(s)
                cells_visited, _ = cell_sequence(cell_idx)
                ct = s.get("celltype", "unknown")
                exp_id = s.get("exp_id", "unknown")

                if len(cells_visited) > 0:
                    de = dead_end_visits(cells_visited, maze)
                    for de_cell, info in de.items():
                        de_records.append({
                            "Session": exp_id,
                            "Cell type": ct,
                            "Dead end": f"({de_cell[0]},{de_cell[1]})",
                            "Visits": info["visits"],
                            "Mean dwell": round(info["mean_dwell"], 1),
                        })

            if de_records:
                df_de = pd.DataFrame(de_records)
                # Aggregate across sessions per celltype
                df_de_agg = df_de.groupby(["Cell type", "Dead end"]).agg(
                    Total_visits=("Visits", "sum"),
                    Mean_visits=("Visits", "mean"),
                    Mean_dwell=("Mean dwell", "mean"),
                ).reset_index()
                st.dataframe(df_de_agg, use_container_width=True)
            else:
                st.info("No dead-end visit data available.")


# --- Turn Bias analysis ---
with tab_turns:
    st.subheader("Per-Junction Turn Bias")

    if not pos_sessions:
        st.warning(
            "No sessions with maze position data available. "
            "This tab will populate when sync.h5 files contain x_maze / y_maze."
        )
    else:
        st.markdown(
            "At each T-junction, turns are classified as left, right, forward, or back "
            "relative to the direction of approach. The bar chart shows the fraction of "
            "left vs right turns at each junction."
        )

        # ── Global turn bias by celltype ────────────────────────────────
        st.markdown("### Global Turn Bias")

        global_bias_records = []
        for s in pos_sessions:
            cell_idx = _discretize_session(s)
            cells_visited, _ = cell_sequence(cell_idx)
            ct = s.get("celltype", "unknown")
            exp_id = s.get("exp_id", "unknown")

            if len(cells_visited) > 2:
                tb = turn_bias(cells_visited, maze)
                total = tb["left"] + tb["right"] + tb["back"] + tb["forward"]
                global_bias_records.append({
                    "Session": exp_id,
                    "Cell type": ct,
                    "Left": tb["left"],
                    "Right": tb["right"],
                    "Back": tb["back"],
                    "Forward": tb["forward"],
                    "Left fraction": tb["left_frac"],
                    "Total turns": total,
                })

        if global_bias_records:
            df_bias = pd.DataFrame(global_bias_records)

            # Box plot: left fraction by celltype
            fig_global = px.box(
                df_bias,
                x="Cell type",
                y="Left fraction",
                color="Cell type",
                color_discrete_map=CELLTYPE_COLORS,
                points="all",
                title="Left Turn Fraction by Cell Type (0.5 = no bias)",
            )
            fig_global.add_hline(
                y=0.5, line_dash="dash", line_color="gray",
                annotation_text="No bias",
            )
            fig_global.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_global, use_container_width=True, key="maze_global_bias")

            # Summary table
            with st.expander("Per-session turn counts"):
                st.dataframe(df_bias, use_container_width=True)

        # ── Per-junction turn bias ──────────────────────────────────────
        st.markdown("### Per-Junction Turn Bias")

        # Accumulate per-junction counts by celltype
        junction_data: dict[str, dict[tuple[int, int], dict[str, int]]] = {}
        for s in pos_sessions:
            cell_idx = _discretize_session(s)
            cells_visited, _ = cell_sequence(cell_idx)
            ct = s.get("celltype", "unknown")

            if len(cells_visited) > 2:
                pj = per_junction_turn_bias(cells_visited, maze)
                if ct not in junction_data:
                    junction_data[ct] = {
                        j: {"left": 0, "right": 0, "back": 0, "forward": 0}
                        for j in maze.junctions
                    }
                for j, counts in pj.items():
                    for direction, n in counts.items():
                        junction_data[ct][j][direction] += n

        if junction_data:
            # Build bar chart data
            bar_records = []
            for ct in sorted(junction_data.keys()):
                for j in sorted(maze.junctions):
                    counts = junction_data[ct][j]
                    total_lr = counts["left"] + counts["right"]
                    left_frac = counts["left"] / total_lr if total_lr > 0 else 0.5
                    bar_records.append({
                        "Junction": f"({j[0]},{j[1]})",
                        "Cell type": ct,
                        "Left fraction": left_frac,
                        "Left": counts["left"],
                        "Right": counts["right"],
                        "Total L+R": total_lr,
                    })

            df_junc = pd.DataFrame(bar_records)

            fig_junc = px.bar(
                df_junc,
                x="Junction",
                y="Left fraction",
                color="Cell type",
                color_discrete_map=CELLTYPE_COLORS,
                barmode="group",
                title="Left Turn Fraction at Each T-Junction",
                hover_data=["Left", "Right", "Total L+R"],
            )
            fig_junc.add_hline(
                y=0.5, line_dash="dash", line_color="gray",
                annotation_text="No bias",
            )
            fig_junc.update_layout(
                height=400,
                yaxis=dict(range=[0, 1], title="Left turn fraction"),
            )
            st.plotly_chart(fig_junc, use_container_width=True, key="maze_junc_bias")

            # Raw counts table
            with st.expander("Per-junction turn counts (pooled across sessions)"):
                st.dataframe(df_junc, use_container_width=True)
        else:
            st.info("Insufficient trajectory data to compute turn bias.")


# --- Cross-Session Comparison ---
with tab_compare:
    st.subheader("Cross-Session Comparison")

    if not pos_sessions:
        st.warning(
            "No sessions with maze position data available. "
            "This tab will populate when sync.h5 files contain x_maze / y_maze."
        )
    else:
        st.markdown(
            "Compare exploration patterns across sessions, examining occupancy "
            "similarity, navigation entropy, and celltype differences."
        )

        # ── Per-session occupancy vectors ───────────────────────────────
        session_occ: dict[str, np.ndarray] = {}
        session_meta: dict[str, dict] = {}
        for s in pos_sessions:
            cell_idx = _discretize_session(s)
            exp_id = s.get("exp_id", "unknown")
            occ = occupancy_fraction(cell_idx, maze.n_cells)
            session_occ[exp_id] = occ
            session_meta[exp_id] = {
                "celltype": s.get("celltype", "unknown"),
                "animal_id": s.get("animal_id", "unknown"),
            }

        # ── Occupancy correlation matrix ────────────────────────────────
        st.markdown("### Occupancy Correlation Between Sessions")
        st.markdown(
            "Pearson correlation of occupancy vectors between all session pairs. "
            "High correlation means mice spent time in similar areas of the maze."
        )

        exp_ids = sorted(session_occ.keys())
        n_sess = len(exp_ids)

        if n_sess >= 2:
            occ_matrix = np.array([session_occ[eid] for eid in exp_ids])
            # Pearson correlation
            corr = np.corrcoef(occ_matrix)
            # Handle NaN (e.g. if a session has zero variance)
            corr = np.nan_to_num(corr, nan=0.0)

            # Label by celltype
            corr_labels = [
                f"{eid[:8]}..({session_meta[eid]['celltype'][:4]})"
                for eid in exp_ids
            ]

            fig_corr = go.Figure(data=go.Heatmap(
                z=corr,
                x=corr_labels,
                y=corr_labels,
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,
                zmax=1,
                colorbar=dict(title="r"),
                hovertemplate="Session %{x} vs %{y}<br>r = %{z:.3f}<extra></extra>",
            ))
            fig_corr.update_layout(
                title="Occupancy Correlation Matrix",
                height=500,
                xaxis=dict(tickangle=45),
            )
            st.plotly_chart(fig_corr, use_container_width=True, key="maze_occ_corr")

            # ── Within vs between celltype correlations ─────────────────
            within_corrs = []
            between_corrs = []
            for i in range(n_sess):
                for j in range(i + 1, n_sess):
                    r = corr[i, j]
                    ct_i = session_meta[exp_ids[i]]["celltype"]
                    ct_j = session_meta[exp_ids[j]]["celltype"]
                    if ct_i == ct_j:
                        within_corrs.append({"r": r, "comparison": "Within celltype"})
                    else:
                        between_corrs.append({"r": r, "comparison": "Between celltypes"})

            corr_comparison = within_corrs + between_corrs
            if corr_comparison:
                df_corr_comp = pd.DataFrame(corr_comparison)
                fig_corr_comp = px.box(
                    df_corr_comp,
                    x="comparison",
                    y="r",
                    color="comparison",
                    points="all",
                    title="Occupancy Correlation: Within vs Between Cell Types",
                )
                fig_corr_comp.update_layout(
                    height=350,
                    showlegend=False,
                    yaxis=dict(title="Pearson r"),
                )
                st.plotly_chart(
                    fig_corr_comp, use_container_width=True,
                    key="maze_corr_within_between",
                )

                n_within = len(within_corrs)
                n_between = len(between_corrs)
                mean_within = np.mean([c["r"] for c in within_corrs]) if within_corrs else 0
                mean_between = np.mean([c["r"] for c in between_corrs]) if between_corrs else 0
                c1, c2 = st.columns(2)
                c1.metric(
                    "Mean within-celltype r",
                    f"{mean_within:.3f}",
                    help=f"n={n_within} pairs",
                )
                c2.metric(
                    "Mean between-celltype r",
                    f"{mean_between:.3f}",
                    help=f"n={n_between} pairs",
                )
        else:
            st.info("Need at least 2 sessions with position data for correlation analysis.")

        # ── Navigation entropy comparison ───────────────────────────────
        st.markdown("### Navigation Entropy by Cell Type")
        st.markdown(
            "Conditional entropy of the node visit sequence. Lower entropy indicates "
            "more predictable (potentially more goal-directed) navigation."
        )

        entropy_records = []
        for s in pos_sessions:
            cell_idx = _discretize_session(s)
            nodes, node_times = node_sequence(cell_idx, maze)
            ct = s.get("celltype", "unknown")
            exp_id = s.get("exp_id", "unknown")

            if len(nodes) > 5:
                ctx_lens, cond_ent = sequence_entropy(nodes, max_context=5)
                # Use context=1 (first-order) entropy as summary metric
                h1 = float(cond_ent[0]) if len(cond_ent) > 0 else 0.0
                entropy_records.append({
                    "Session": exp_id,
                    "Cell type": ct,
                    "Entropy (bits)": round(h1, 3),
                    "Node transitions": len(nodes),
                })

        if entropy_records:
            df_ent = pd.DataFrame(entropy_records)

            fig_ent = px.box(
                df_ent,
                x="Cell type",
                y="Entropy (bits)",
                color="Cell type",
                color_discrete_map=CELLTYPE_COLORS,
                points="all",
                title="First-Order Navigation Entropy by Cell Type",
            )
            fig_ent.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_ent, use_container_width=True, key="maze_entropy_box")
        else:
            st.info("Insufficient node transitions to compute entropy.")

        # ── Exploration metrics scatter ──────────────────────────────────
        st.markdown("### Exploration Metrics Across Sessions")

        if summary_records:
            # summary_records was built in the Exploration tab; reuse if available
            df_cross = pd.DataFrame(summary_records)

            metric_y = st.selectbox(
                "Metric to compare",
                ["Coverage", "Transitions", "Occupancy entropy", "Left turn frac"],
                key="maze_cross_metric",
            )

            fig_cross = px.strip(
                df_cross,
                x="Cell type",
                y=metric_y,
                color="Cell type",
                color_discrete_map=CELLTYPE_COLORS,
                hover_data=["Session", "Animal"],
                title=f"{metric_y} by Cell Type",
            )
            fig_cross.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_cross, use_container_width=True, key="maze_cross_strip")

        # ── Methods reference ───────────────────────────────────────────
        with st.expander("Methods & References"):
            st.markdown(
                "**Maze exploration analysis** adapted from:\n\n"
                'Rosenberg, Zhang, Perona & Meister (2021). "Mice in a labyrinth show '
                'rapid learning, sudden insight, and efficient exploration." '
                "*eLife* 10, e66175. doi:10.7554/eLife.66175\n\n"
                "**Metrics:**\n"
                "- **Occupancy** — fraction of total time in each cell\n"
                "- **Coverage** — fraction of accessible cells visited at least once\n"
                "- **Turn bias** — left/right classification at T-junctions using "
                "cross-product of approach and departure vectors\n"
                "- **Navigation entropy** — conditional entropy of node visit sequence "
                "(lower = more predictable)\n"
                "- **Occupancy correlation** — Pearson r between occupancy vectors"
            )


# --- Reference ---
st.markdown("---")
st.caption(
    "Analysis methods adapted from: Rosenberg, Zhang, Perona & Meister (2021). "
    '"Mice in a labyrinth show rapid learning, sudden insight, and efficient exploration." '
    "eLife 10, e66175."
)
