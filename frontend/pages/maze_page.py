"""Maze Analysis — rose maze topology, exploration, and navigation metrics.

Inspired by Rosenberg et al. (2021) eLife.
Visualizes maze structure, occupancy, exploration efficiency, turn bias,
and goal-directed behaviour.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from hm2p.maze.topology import build_rose_maze

log = logging.getLogger("hm2p.frontend.maze")

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
        "The rose maze is a **7×5 unit grid** with internal walls creating corridors. "
        f"It has **{maze.n_cells} accessible cells**, **{len(maze.junctions)} T-junctions**, "
        f"and **{len(maze.dead_ends)} dead ends**."
    )

    # Draw maze as a plotly heatmap
    import plotly.graph_objects as go

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
    st.plotly_chart(fig, use_container_width=True)

    # Distance matrix
    with st.expander("Distance Matrix"):
        st.markdown("Shortest-path distances between all cells:")
        import pandas as pd
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

# --- Exploration analysis (requires kinematics data) ---
with tab_explore:
    st.subheader("Exploration Efficiency")
    st.info(
        "Exploration metrics require kinematics data (Stage 3 — position in maze coordinates). "
        "This will be available after DLC pose estimation and kinematics processing complete. "
        "Below is a synthetic demo."
    )

    # Demo with synthetic trajectory
    with st.expander("Synthetic Demo", expanded=True):
        st.markdown("Simulating a random walk through the maze to demonstrate metrics.")

        n_steps = st.slider("Number of steps", 100, 5000, 1000, 100, key="maze_demo_steps")
        seed = st.number_input("Random seed", 0, 1000, 42, key="maze_demo_seed")

        rng = np.random.default_rng(seed)
        # Random walk on maze graph
        cell_list = maze.cell_list
        current = rng.choice(len(cell_list))
        trajectory = [current]
        for _ in range(n_steps - 1):
            cell = cell_list[current]
            nbs = maze.adj[cell]
            next_cell = nbs[rng.integers(len(nbs))]
            current = maze.cell_to_idx[next_cell]
            trajectory.append(current)
        traj = np.array(trajectory, dtype=np.int32)

        from hm2p.maze.analysis import (
            cell_occupancy,
            exploration_efficiency,
            maze_exploration_summary,
            sequence_entropy,
            turn_bias,
        )
        from hm2p.maze.discretize import cell_sequence, node_sequence

        summary = maze_exploration_summary(traj, maze, fps=30.0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Unique cells", summary["unique_cells_visited"])
        col2.metric("Coverage", f"{summary['coverage_frac']:.0%}")
        col3.metric("Cell transitions", summary["n_cell_transitions"])
        col4.metric("Occupancy entropy", f"{summary['occupancy_entropy']:.2f} bits")

        # Occupancy heatmap
        occ = cell_occupancy(traj, maze.n_cells)
        grid = np.full((5, 7), np.nan)
        for i, cell in enumerate(cell_list):
            grid[cell[1], cell[0]] = occ[i]

        import plotly.express as px

        fig = px.imshow(
            grid[::-1],
            labels=dict(x="x", y="y", color="Frames"),
            x=[str(i) for i in range(7)],
            y=[str(4 - i) for i in range(5)],
            color_continuous_scale="Viridis",
            title="Occupancy Heatmap (frames per cell)",
            aspect="equal",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Exploration efficiency curve
        nodes, ntimes = node_sequence(traj, maze)
        if len(nodes) > 2:
            ws, nn = exploration_efficiency(nodes)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ws, y=nn, mode="lines+markers", name="Observed"))
            fig.update_layout(
                height=300,
                title="Exploration Efficiency (distinct nodes per window)",
                xaxis_title="Window size (node visits)",
                yaxis_title="Mean distinct nodes",
                xaxis_type="log",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Sequence entropy
        if len(nodes) > 5:
            ctx, ent = sequence_entropy(nodes, max_context=8)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ctx, y=ent, mode="lines+markers"))
            fig.update_layout(
                height=300,
                title="Navigation Entropy (bits per node visit)",
                xaxis_title="Context length",
                yaxis_title="Conditional entropy (bits)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Turn bias
        cells_seq, _ = cell_sequence(traj)
        tb = turn_bias(cells_seq, maze)
        fig = go.Figure(data=[go.Bar(
            x=["Left", "Right", "Back", "Forward"],
            y=[tb["left"], tb["right"], tb["back"], tb["forward"]],
            marker_color=["#4169E1", "#E14169", "#999", "#41E169"],
        )])
        fig.update_layout(height=300, title=f"Turn Bias (left frac: {tb['left_frac']:.2f})")
        st.plotly_chart(fig, use_container_width=True)


with tab_turns:
    st.subheader("Per-Junction Turn Bias")
    st.info("Requires kinematics data. Showing synthetic demo below.")

    with st.expander("Per-Junction Demo", expanded=True):
        from hm2p.maze.analysis import per_junction_turn_bias

        # Reuse synthetic trajectory from above if available
        n_steps_turn = 2000
        rng2 = np.random.default_rng(42)
        current2 = rng2.choice(len(maze.cell_list))
        traj2 = [current2]
        for _ in range(n_steps_turn - 1):
            cell = maze.cell_list[current2]
            nbs = maze.adj[cell]
            next_cell = nbs[rng2.integers(len(nbs))]
            current2 = maze.cell_to_idx[next_cell]
            traj2.append(current2)
        traj2 = np.array(traj2, dtype=np.int32)

        cells_seq2, _ = cell_sequence(traj2)
        pj = per_junction_turn_bias(cells_seq2, maze)

        # Draw maze with turn bias overlay
        fig = go.Figure()
        for cell in maze.cells:
            col, row = cell
            if cell in pj:
                counts = pj[cell]
                total = sum(counts.values())
                if total > 0:
                    left_f = counts["left"] / total
                    right_f = counts["right"] / total
                    back_f = counts["back"] / total
                    # Color by left-right bias
                    bias = left_f - right_f
                    r = int(max(0, -bias * 255))
                    b = int(max(0, bias * 255))
                    color = f"rgba({r}, 100, {b}, 0.7)"
                else:
                    color = "rgba(200, 200, 200, 0.3)"
            else:
                color = "rgba(200, 200, 200, 0.3)"

            fig.add_shape(
                type="rect",
                x0=col, y0=row, x1=col + 1, y1=row + 1,
                fillcolor=color,
                line=dict(color="black", width=1),
            )

            if cell in pj:
                counts = pj[cell]
                total = sum(counts.values())
                if total > 0:
                    fig.add_annotation(
                        x=col + 0.5, y=row + 0.5,
                        text=f"L:{counts['left']}<br>R:{counts['right']}",
                        showarrow=False, font=dict(size=8),
                    )

        fig.update_layout(
            height=400, width=600,
            xaxis=dict(range=[-0.2, 7.2], scaleanchor="y"),
            yaxis=dict(range=[-0.2, 5.2]),
            title="Per-Junction Turn Counts (blue=left bias, red=right bias)",
        )
        st.plotly_chart(fig, use_container_width=True)


with tab_compare:
    st.subheader("Cross-Session Comparison")
    st.info(
        "Cross-session maze analysis will compare exploration patterns across all 26 sessions "
        "and between Penk vs non-Penk mice. Requires Stage 3 kinematics data."
    )

    st.markdown("""
    **Planned analyses (pending kinematics data):**
    - Exploration coverage over time by cell type
    - Turn bias: Penk vs non-Penk comparison
    - Path efficiency learning curves across sessions
    - Navigation entropy by cell type
    - Dead-end visit frequency by cell type
    - Occupancy heatmap comparison (Penk vs non-Penk)
    """)

# --- Reference ---
st.markdown("---")
st.caption(
    "Analysis methods adapted from: Rosenberg, Zhang, Perona & Meister (2021). "
    '"Mice in a labyrinth show rapid learning, sudden insight, and efficient exploration." '
    "eLife 10, e66175."
)
