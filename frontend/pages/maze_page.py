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
        "The rose maze is a **7x5 unit grid** with internal walls creating corridors. "
        f"It has **{maze.n_cells} accessible cells**, **{len(maze.junctions)} T-junctions**, "
        f"and **{len(maze.dead_ends)} dead ends**."
    )

    import plotly.express as px
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
    st.warning(
        "No data available yet. This page will populate when the relevant "
        "pipeline stage completes (Stage 3 kinematics -- position in maze coordinates)."
    )

with tab_turns:
    st.subheader("Per-Junction Turn Bias")
    st.warning(
        "No data available yet. This page will populate when the relevant "
        "pipeline stage completes (Stage 3 kinematics)."
    )

with tab_compare:
    st.subheader("Cross-Session Comparison")
    st.warning(
        "No data available yet. This page will populate when the relevant "
        "pipeline stage completes (Stage 3 kinematics)."
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
