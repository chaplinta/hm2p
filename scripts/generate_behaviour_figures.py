#!/usr/bin/env python3
"""Generate publication-quality figures for the behavioural manuscript.

Reads pre-computed analysis results from behaviour-results.json and produces
Figures 1-5 as PDF and PNG files.

Usage:
    python scripts/generate_behaviour_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_PATH = Path("/workspace/docs/manuscripts/behaviour-results.json")
FIGURES_DIR = Path("/workspace/docs/manuscripts/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------
# Publication-quality defaults
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "lines.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,  # TrueType in PDF (editable text)
        "ps.fonttype": 42,
    }
)

# Colour palette
COL_LIGHT = "#E8941A"  # warm orange for light condition
COL_DARK = "#5B7BA5"  # steel blue for dark condition
COL_LIGHT_FILL = "#F5D59A"  # light orange fill
COL_DARK_FILL = "#A8BDD4"  # light blue fill
COL_DOTS = "#888888"  # individual session dots
COL_JUNCTION = "#D64550"  # red for junctions
COL_CORRIDOR = "#4A90D9"  # blue for corridors
COL_DEAD_END = "#6BBF6B"  # green for dead ends

SINGLE_COL = 3.5  # inches
DOUBLE_COL = 7.0  # inches


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_results() -> dict[str, Any]:
    """Load the pre-computed results JSON."""
    with open(RESULTS_PATH) as f:
        return json.load(f)


def get_usable_sessions(results: dict) -> list[dict]:
    """Return list of usable (non-excluded) per-session dicts."""
    return [s for s in results["per_session"] if not s["exclude"] and s.get("status", "ok") == "ok"]


def extract_paired_arrays(
    sessions: list[dict], key_light: str, key_dark: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extract matched light/dark arrays from per-session data.

    Skips sessions where either value is None or NaN.
    """
    light_vals, dark_vals = [], []
    for s in sessions:
        vl = s.get(key_light)
        vd = s.get(key_dark)
        if vl is not None and vd is not None:
            if not (np.isnan(vl) or np.isnan(vd)):
                light_vals.append(vl)
                dark_vals.append(vd)
    return np.array(light_vals), np.array(dark_vals)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def add_panel_label(ax: plt.Axes, label: str, x: float = -0.15, y: float = 1.08) -> None:
    """Add bold panel label (A, B, C, ...) to axes."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def paired_dot_plot(
    ax: plt.Axes,
    light: np.ndarray,
    dark: np.ndarray,
    ylabel: str,
    test_result: dict | None = None,
    xlabels: tuple[str, str] = ("Light", "Dark"),
    jitter: float = 0.05,
) -> None:
    """Plot paired dot plot with connecting lines.

    Parameters
    ----------
    ax : Axes
    light, dark : arrays of matched values
    ylabel : y-axis label
    test_result : dict with 'stat', 'p', 'r' keys for annotation
    xlabels : x-axis tick labels
    jitter : random horizontal jitter magnitude
    """
    n = len(light)
    rng = np.random.default_rng(42)
    jitter_l = rng.uniform(-jitter, jitter, n)
    jitter_d = rng.uniform(-jitter, jitter, n)

    x_light = np.zeros(n) + jitter_l
    x_dark = np.ones(n) + jitter_d

    # Connecting lines
    for i in range(n):
        ax.plot(
            [x_light[i], x_dark[i]],
            [light[i], dark[i]],
            color="#CCCCCC",
            linewidth=0.5,
            zorder=1,
        )

    # Dots
    ax.scatter(x_light, light, color=COL_LIGHT, s=18, zorder=2, edgecolors="none", alpha=0.8)
    ax.scatter(x_dark, dark, color=COL_DARK, s=18, zorder=2, edgecolors="none", alpha=0.8)

    # Medians
    ax.plot([-0.15, 0.15], [np.median(light)] * 2, color=COL_LIGHT, linewidth=2, zorder=3)
    ax.plot([0.85, 1.15], [np.median(dark)] * 2, color=COL_DARK, linewidth=2, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(xlabels)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(ylabel)

    # Statistical annotation
    if test_result is not None:
        p = test_result.get("p", 1.0)
        r = test_result.get("r", 0.0)
        n_test = test_result.get("n", n)
        p_str = _format_p(p)
        ann = f"{p_str}\nr = {r:.2f}, N = {n_test}"
        y_max = max(np.max(light), np.max(dark))
        y_min = min(np.min(light), np.min(dark))
        y_range = y_max - y_min
        bar_y = y_max + 0.04 * y_range
        tick_y = y_max + 0.02 * y_range
        # Bracket: two vertical ticks connected by a horizontal bar
        ax.plot([0, 0], [tick_y, bar_y], color="#444444", linewidth=0.5)
        ax.plot([1, 1], [tick_y, bar_y], color="#444444", linewidth=0.5)
        ax.plot([0, 1], [bar_y, bar_y], color="#444444", linewidth=0.5)
        ax.text(
            0.5,
            bar_y + 0.02 * y_range,
            ann,
            ha="center",
            va="bottom",
            fontsize=6,
            color="#444444",
        )
        # Pad y-axis to fit annotation
        ax.set_ylim(y_min - 0.05 * y_range, bar_y + 0.22 * y_range)


def _format_p(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return f"p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.05:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure as PDF and PNG."""
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=300)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Maze structure and exploration overview
# ---------------------------------------------------------------------------
def figure1(results: dict) -> None:
    """Maze topology, example trajectory, and mean coverage."""
    print("Generating Figure 1...")

    # Import maze topology
    sys.path.insert(0, "/workspace/src")
    from hm2p.maze.topology import build_rose_maze

    maze = build_rose_maze()

    fig = plt.figure(figsize=(DOUBLE_COL, 2.6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.9], wspace=0.35)

    # --- Panel A: Maze topology graph ---
    ax_a = fig.add_subplot(gs[0])
    _draw_maze_topology(ax_a, maze)
    add_panel_label(ax_a, "A")

    # --- Panel B: Example trajectory ---
    ax_b = fig.add_subplot(gs[1])
    _draw_example_trajectory(ax_b, maze, results)
    add_panel_label(ax_b, "B")

    # --- Panel C: Mean coverage per epoch ---
    ax_c = fig.add_subplot(gs[2])
    sessions = get_usable_sessions(results)
    light_cov, dark_cov = extract_paired_arrays(
        sessions, "mean_epoch_coverage_light", "mean_epoch_coverage_dark"
    )
    # Convert fraction to percentage
    light_cov_pct = light_cov * 100
    dark_cov_pct = dark_cov * 100
    test = results["cross_session"]["figure3"]["epoch_coverage_light_vs_dark"]["test"]
    paired_dot_plot(ax_c, light_cov_pct, dark_cov_pct,
                    "Coverage per epoch (%)", test_result=test)
    add_panel_label(ax_c, "C")

    save_figure(fig, "figure1_maze_exploration")


def _draw_maze_topology(ax: plt.Axes, maze) -> None:
    """Draw maze graph with nodes coloured by type."""
    type_colors = {
        "dead_end": COL_DEAD_END,
        "corridor": COL_CORRIDOR,
        "t_junction": COL_JUNCTION,
        "crossroads": "#9B59B6",
    }
    type_markers = {
        "dead_end": "s",
        "corridor": "o",
        "t_junction": "D",
        "crossroads": "h",
    }

    # Draw edges
    for cell in sorted(maze.adj.keys()):
        cx, cy = cell
        for nb in maze.adj[cell]:
            nx, ny = nb
            if cell < nb:
                ax.plot(
                    [cx + 0.5, nx + 0.5],
                    [cy + 0.5, ny + 0.5],
                    color="#BBBBBB",
                    linewidth=1.5,
                    zorder=1,
                )

    # Draw nodes
    for cell in sorted(maze.cells):
        cx, cy = cell
        ntype = maze.node_types[cell]
        ax.scatter(
            cx + 0.5,
            cy + 0.5,
            color=type_colors[ntype],
            marker=type_markers[ntype],
            s=80,
            zorder=2,
            edgecolors="white",
            linewidth=0.5,
        )

    # Internal walls — draw as thick wall indicators
    # Walls between (2,4)-(3,4) and (3,4)-(4,4)
    for (c1, c2) in [((2, 4), (3, 4)), ((3, 4), (4, 4))]:
        mid_x = (c1[0] + c2[0]) / 2.0 + 0.5
        mid_y = (c1[1] + c2[1]) / 2.0 + 0.5
        ax.plot(mid_x, mid_y, "x", color="#CC0000", markersize=5, zorder=3, markeredgewidth=1.0)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COL_DEAD_END,
               markersize=6, label="Dead end"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_CORRIDOR,
               markersize=6, label="Corridor"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COL_JUNCTION,
               markersize=6, label="T-junction"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=5.5,
              frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
              handletextpad=0.3, borderpad=0.3)

    ax.set_xlim(-0.2, 7.2)
    ax.set_ylim(-0.3, 5.3)
    ax.set_aspect("equal")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(np.arange(0, 7) + 0.5)
    ax.set_xticklabels(np.arange(0, 7))
    ax.set_yticks(np.arange(0, 5) + 0.5)
    ax.set_yticklabels(np.arange(0, 5))


def _try_load_real_trajectory():
    """Attempt to download a real trajectory from sync.h5 on S3.

    Tries sub-1114353/ses-20210823T165950 first (first usable session).
    Returns (x_maze, y_maze, light_on) arrays or None if unavailable.
    """
    try:
        import boto3
        import h5py
        import tempfile
        import os

        s3 = boto3.client("s3", region_name="ap-southeast-2")
        key = "sync/sub-1114353/ses-20210823T165950/sync.h5"
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmppath = tmp.name
        try:
            s3.download_file("hm2p-derivatives", key, tmppath)
            with h5py.File(tmppath, "r") as f:
                if "x_maze" in f and "y_maze" in f and "light_on" in f:
                    return (
                        f["x_maze"][:],
                        f["y_maze"][:],
                        f["light_on"][:].astype(bool),
                    )
        except Exception:
            pass
        finally:
            if os.path.exists(tmppath):
                os.unlink(tmppath)
    except Exception:
        pass
    return None


def _draw_example_trajectory(ax: plt.Axes, maze, results: dict) -> None:
    """Draw example trajectory on the maze.

    Attempts to load a real trajectory from sync.h5 on S3.  If
    unavailable (pipeline not yet re-run), falls back to a schematic
    graph-constrained walk and labels it clearly.
    """
    adj = maze.adj

    # Draw maze cell outlines first (background)
    for cell in maze.cells:
        rect = plt.Rectangle(
            (cell[0], cell[1]), 1, 1,
            facecolor="#F0F0F0",
            edgecolor="#CCCCCC",
            linewidth=0.3,
            zorder=0,
        )
        ax.add_patch(rect)

    # Internal walls
    for (c1, c2) in [((2, 4), (3, 4)), ((3, 4), (4, 4))]:
        wall_x = max(c1[0], c2[0])
        ax.plot([wall_x, wall_x], [c1[1], c1[1] + 1],
                color="#666666", linewidth=2, zorder=1)

    real_data = _try_load_real_trajectory()

    if real_data is not None:
        x_maze, y_maze, light_on = real_data
        valid = np.isfinite(x_maze) & np.isfinite(y_maze)
        # Subsample for visual clarity (every 3rd frame)
        step = 3
        for mask, col, label in [
            (valid & light_on, COL_LIGHT, "Light"),
            (valid & ~light_on, COL_DARK, "Dark"),
        ]:
            idx = np.where(mask)[0][::step]
            ax.plot(x_maze[idx], y_maze[idx], color=col, linewidth=0.25,
                    alpha=0.5, label=label, zorder=2)
        title_suffix = ""
    else:
        # DEFERRED: sync.h5 not yet on S3, use graph-constrained schematic
        rng = np.random.default_rng(12345)
        walk = [(0, 0)]
        visited = {(0, 0)}
        current = (0, 0)
        for _ in range(250):
            neighbours = adj[current]
            unvisited = [n for n in neighbours if n not in visited]
            if unvisited:
                current = unvisited[rng.integers(len(unvisited))]
            else:
                current = neighbours[rng.integers(len(neighbours))]
            walk.append(current)
            visited.add(current)

        n_steps = len(walk) - 1
        mid = n_steps // 2

        for condition, (start, end, col) in enumerate([
            (0, mid, COL_LIGHT),
            (mid, n_steps, COL_DARK),
        ]):
            xs, ys = [], []
            for i in range(start, end + 1):
                cx, cy = walk[i]
                jx = rng.normal(0, 0.12)
                jy = rng.normal(0, 0.12)
                xs.append(cx + 0.5 + np.clip(jx, -0.35, 0.35))
                ys.append(cy + 0.5 + np.clip(jy, -0.35, 0.35))
            label = "Light" if condition == 0 else "Dark"
            ax.plot(xs, ys, color=col, linewidth=0.35, alpha=0.65,
                    label=label, zorder=2)
        title_suffix = " (schematic)"

    ax.legend(fontsize=5.5, loc="lower right", frameon=True, framealpha=0.9,
              edgecolor="#CCCCCC", handletextpad=0.3, borderpad=0.3)
    ax.set_xlim(-0.2, 7.2)
    ax.set_ylim(-0.3, 5.3)
    ax.set_aspect("equal")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(np.arange(0, 7) + 0.5)
    ax.set_xticklabels(np.arange(0, 7))
    ax.set_yticks(np.arange(0, 5) + 0.5)
    ax.set_yticklabels(np.arange(0, 5))
    if title_suffix:
        ax.set_title(f"Example trajectory{title_suffix}", fontsize=7)


# ---------------------------------------------------------------------------
# Figure 2: Turn behaviour at junctions
# ---------------------------------------------------------------------------
def figure2(results: dict) -> None:
    """Per-junction bias, lag-1 autocorrelation, light vs dark."""
    print("Generating Figure 2...")
    sessions = get_usable_sessions(results)
    cs = results["cross_session"]["figure4"]

    fig = plt.figure(figsize=(DOUBLE_COL, 2.6))
    gs = gridspec.GridSpec(1, 3, wspace=0.45)

    # --- Panel A: Per-junction left/right fraction ---
    ax_a = fig.add_subplot(gs[0])
    _draw_junction_bias(ax_a, cs["per_junction_bias"])
    add_panel_label(ax_a, "A")

    # --- Panel B: Lag-1 turn autocorrelation distribution ---
    ax_b = fig.add_subplot(gs[1])
    autocorrs = np.array([s["turn_autocorr_all"] for s in sessions
                          if s.get("turn_autocorr_all") is not None
                          and not np.isnan(s["turn_autocorr_all"])])
    ax_b.hist(autocorrs, bins=12, color="#888888", edgecolor="white",
              linewidth=0.5, alpha=0.8)
    ax_b.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax_b.axvline(np.median(autocorrs), color=COL_JUNCTION, linewidth=1.2)
    ax_b.set_xlabel("Lag-1 autocorrelation")
    ax_b.set_ylabel("Sessions")

    # Annotate with one-sample test
    test_zero = cs["turn_autocorr_vs_zero"]["test"]
    p_str = _format_p(test_zero["p"])
    ax_b.text(0.97, 0.95,
              f"median = {np.median(autocorrs):.3f}\n{p_str}, r = {test_zero['r']:.2f}",
              transform=ax_b.transAxes, fontsize=6, va="top", ha="right",
              color="#444444")
    add_panel_label(ax_b, "B")

    # --- Panel C: Turn autocorrelation light vs dark ---
    ax_c = fig.add_subplot(gs[2])
    light_ac, dark_ac = extract_paired_arrays(
        sessions, "turn_autocorr_light", "turn_autocorr_dark"
    )
    test_ld = cs["turn_autocorr_light_vs_dark"]["test_light_vs_dark"]
    paired_dot_plot(ax_c, light_ac, dark_ac,
                    "Lag-1 autocorrelation", test_result=test_ld)
    add_panel_label(ax_c, "C")

    save_figure(fig, "figure2_turn_behaviour")


def _draw_junction_bias(ax: plt.Axes, per_junction: dict) -> None:
    """Bar plot of left fraction per junction."""
    junctions = sorted(per_junction.keys())
    left_fracs = [per_junction[j]["left_frac"] for j in junctions]
    totals = [per_junction[j]["total"] for j in junctions]
    adj_ps = [per_junction[j]["binomial_p_adjusted"] for j in junctions]

    x = np.arange(len(junctions))
    bars = ax.bar(x, left_fracs, color="#6BAED6", edgecolor="white", linewidth=0.5, width=0.65)

    # Reference line at 0.5
    ax.axhline(0.5, color="black", linewidth=0.5, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels([j.replace("(", "").replace(")", "") for j in junctions],
                       fontsize=5.5, rotation=45, ha="right")
    ax.set_ylabel("Left fraction")
    ax.set_xlabel("Junction (col, row)")
    ax.set_ylim(0.35, 0.65)

    # Significance markers
    for i, p_adj in enumerate(adj_ps):
        if p_adj < 0.05:
            ax.text(x[i], left_fracs[i] + 0.01, "*", ha="center", fontsize=8)

    # Sample sizes inside bars near the base
    for i, n_total in enumerate(totals):
        ax.text(x[i], 0.365, str(n_total), ha="center",
                fontsize=4.5, color="#555555", rotation=90, va="bottom")


# ---------------------------------------------------------------------------
# Figure 3: Speed and movement
# ---------------------------------------------------------------------------
def figure3(results: dict) -> None:
    """Speed light vs dark, speed by node type, fraction active."""
    print("Generating Figure 3...")
    sessions = get_usable_sessions(results)
    cs = results["cross_session"]

    fig = plt.figure(figsize=(DOUBLE_COL, 2.6))
    gs = gridspec.GridSpec(1, 3, wspace=0.45)

    # --- Panel A: Speed distributions light vs dark (violin-style) ---
    ax_a = fig.add_subplot(gs[0])
    light_speed, dark_speed = extract_paired_arrays(
        sessions, "median_speed_light", "median_speed_dark"
    )
    test_speed = cs["figure2"]["speed_light_vs_dark"]["test"]
    paired_dot_plot(ax_a, light_speed, dark_speed,
                    "Median speed (cm/s)", test_result=test_speed)
    add_panel_label(ax_a, "A")

    # --- Panel B: Speed by node type ---
    ax_b = fig.add_subplot(gs[1])
    _draw_speed_by_node_type(ax_b, sessions, cs["figure6"]["speed_by_node_type"])
    add_panel_label(ax_b, "B")

    # --- Panel C: Fraction active light vs dark ---
    ax_c = fig.add_subplot(gs[2])
    light_frac, dark_frac = extract_paired_arrays(
        sessions, "frac_active_light", "frac_active_dark"
    )
    # Convert to percentage
    light_frac_pct = light_frac * 100
    dark_frac_pct = dark_frac * 100
    test_frac = cs["figure2"]["frac_active_light_vs_dark"]["test"]
    paired_dot_plot(ax_c, light_frac_pct, dark_frac_pct,
                    "Time active (%)", test_result=test_frac)
    add_panel_label(ax_c, "C")

    save_figure(fig, "figure3_speed_movement")


def _draw_speed_by_node_type(ax: plt.Axes, sessions: list[dict], stats: dict) -> None:
    """Box/strip plot of speed by node type."""
    junction_speeds = np.array([s["speed_junction"] for s in sessions])
    corridor_speeds = np.array([s["speed_corridor"] for s in sessions])
    dead_end_speeds = np.array([s["speed_dead_end"] for s in sessions])

    data = [junction_speeds, corridor_speeds, dead_end_speeds]
    colors = [COL_JUNCTION, COL_CORRIDOR, COL_DEAD_END]
    labels = ["Junction", "Corridor", "Dead end"]

    positions = [0, 1, 2]
    rng = np.random.default_rng(42)

    for i, (d, col) in enumerate(zip(data, colors)):
        bp = ax.boxplot(
            d,
            positions=[positions[i]],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=col, alpha=0.3, edgecolor=col, linewidth=0.6),
            whiskerprops=dict(color=col, linewidth=0.6),
            capprops=dict(color=col, linewidth=0.6),
            medianprops=dict(color=col, linewidth=1.2),
        )
        jitter = rng.uniform(-0.12, 0.12, len(d))
        ax.scatter(
            positions[i] + jitter,
            d,
            color=col,
            s=12,
            alpha=0.7,
            edgecolors="none",
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Mean speed (cm/s)")

    # Post-hoc significance brackets — use 75th percentile to anchor
    # bracket base so outliers don't push everything off screen
    q75 = max(np.percentile(junction_speeds, 90),
              np.percentile(corridor_speeds, 90),
              np.percentile(dead_end_speeds, 90))
    y_range = q75 - min(np.min(junction_speeds), np.min(corridor_speeds),
                        np.min(dead_end_speeds))
    bracket_h = 0.03 * y_range

    posthoc_pairs = [
        (0, 1, stats["posthoc_adjusted_p"]["junc_vs_corr"]),
        (1, 2, stats["posthoc_adjusted_p"]["corr_vs_de"]),
        (0, 2, stats["posthoc_adjusted_p"]["junc_vs_de"]),
    ]
    for idx, (i, j, p_adj) in enumerate(posthoc_pairs):
        y_bar = q75 + (idx + 1) * 0.10 * y_range
        sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "n.s."
        ax.plot([positions[i], positions[j]], [y_bar, y_bar],
                color="#444444", linewidth=0.5)
        ax.plot([positions[i], positions[i]], [y_bar - bracket_h, y_bar],
                color="#444444", linewidth=0.5)
        ax.plot([positions[j], positions[j]], [y_bar - bracket_h, y_bar],
                color="#444444", linewidth=0.5)
        ax.text((positions[i] + positions[j]) / 2, y_bar + 0.005 * y_range,
                sig, ha="center", fontsize=5.5, color="#444444")

    # Friedman test annotation — below the brackets
    friedman = stats["friedman"]
    p_str = _format_p(friedman["p"])
    top_bracket = q75 + 3.5 * 0.10 * y_range
    ax.text(1.0, top_bracket, f"Friedman {p_str}, N = {friedman['n']}",
            fontsize=5.5, ha="center", va="bottom", color="#444444")

    # Set ylim to accommodate brackets but not let outliers dominate
    ax.set_ylim(None, top_bracket + 0.15 * y_range)


# ---------------------------------------------------------------------------
# Figure 4: Exploration strategy in light vs dark
# ---------------------------------------------------------------------------
def figure4(results: dict) -> None:
    """Coverage, dead-end rate, backtracking rate in light vs dark."""
    print("Generating Figure 4...")
    sessions = get_usable_sessions(results)
    cs = results["cross_session"]

    fig = plt.figure(figsize=(DOUBLE_COL, 2.6))
    gs = gridspec.GridSpec(1, 3, wspace=0.45)

    # --- Panel A: Coverage light vs dark ---
    ax_a = fig.add_subplot(gs[0])
    light_cov, dark_cov = extract_paired_arrays(
        sessions, "mean_epoch_coverage_light", "mean_epoch_coverage_dark"
    )
    light_cov_pct = light_cov * 100
    dark_cov_pct = dark_cov * 100
    test_cov = cs["figure3"]["epoch_coverage_light_vs_dark"]["test"]
    paired_dot_plot(ax_a, light_cov_pct, dark_cov_pct,
                    "Coverage per epoch (%)", test_result=test_cov)
    add_panel_label(ax_a, "A")

    # --- Panel B: Dead-end visit rate ---
    ax_b = fig.add_subplot(gs[1])
    light_de, dark_de = extract_paired_arrays(
        sessions, "dead_end_rate_light", "dead_end_rate_dark"
    )
    test_de = cs["figure3"]["dead_end_rate_light_vs_dark"]["test"]
    paired_dot_plot(ax_b, light_de, dark_de,
                    "Dead-end visits / min", test_result=test_de)
    add_panel_label(ax_b, "B")

    # --- Panel C: Backtracking rate ---
    ax_c = fig.add_subplot(gs[2])
    light_bt, dark_bt = extract_paired_arrays(
        sessions, "back_rate_light", "back_rate_dark"
    )
    # Convert to percentage
    light_bt_pct = light_bt * 100
    dark_bt_pct = dark_bt * 100
    test_bt = cs["figure4"]["back_rate_light_vs_dark"]["test"]
    paired_dot_plot(ax_c, light_bt_pct, dark_bt_pct,
                    "Backtracking rate (%)", test_result=test_bt)
    add_panel_label(ax_c, "C")

    save_figure(fig, "figure4_exploration_strategy")


# ---------------------------------------------------------------------------
# Figure 5: Head direction
# ---------------------------------------------------------------------------
def figure5(results: dict) -> None:
    """HD polar histogram, MRL, AHV."""
    print("Generating Figure 5...")
    sessions = get_usable_sessions(results)
    cs = results["cross_session"]["figure5"]

    fig = plt.figure(figsize=(DOUBLE_COL, 2.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 0.9, 0.9], wspace=0.40)

    # --- Panel A: HD distribution polar histogram (schematic) ---
    ax_a = fig.add_subplot(gs[0], projection="polar")
    _draw_hd_polar(ax_a, sessions)
    add_panel_label(ax_a, "A", x=-0.1, y=1.12)

    # --- Panel B: MRL light vs dark ---
    ax_b = fig.add_subplot(gs[1])
    light_mrl, dark_mrl = extract_paired_arrays(
        sessions, "hd_mrl_light", "hd_mrl_dark"
    )
    test_mrl = cs["hd_mrl_light_vs_dark"]["test"]
    paired_dot_plot(ax_b, light_mrl, dark_mrl,
                    "Mean resultant length", test_result=test_mrl)
    add_panel_label(ax_b, "B")

    # --- Panel C: |AHV| light vs dark ---
    ax_c = fig.add_subplot(gs[2])
    light_ahv, dark_ahv = extract_paired_arrays(
        sessions, "median_ahv_light", "median_ahv_dark"
    )
    test_ahv = cs["ahv_light_vs_dark"]["test"]
    paired_dot_plot(ax_c, light_ahv, dark_ahv,
                    "Median |AHV| (deg/s)", test_result=test_ahv)
    add_panel_label(ax_c, "C")

    save_figure(fig, "figure5_head_direction")


def _draw_hd_polar(ax: plt.Axes, sessions: list[dict]) -> None:
    """Draw MRL summary on polar axes using real per-session data.

    Displays real per-session MRL magnitudes as radial markers
    distributed uniformly around the circle (for visual separation
    only -- angular position is arbitrary since per-session preferred
    directions are not stored in the results JSON).  Mean MRL for
    each condition is shown as a concentric circle.

    All MRL values are real, computed from active-only frames in the
    main analysis.
    """
    light_mrls = np.array([s["hd_mrl_light"] for s in sessions
                           if s.get("hd_mrl_light") is not None])
    dark_mrls = np.array([s["hd_mrl_dark"] for s in sessions
                          if s.get("hd_mrl_dark") is not None])
    n = len(light_mrls)

    # Distribute sessions uniformly around the circle for display
    # (angular position is for visual separation, not data)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Per-session MRL dots
    ax.scatter(theta, light_mrls, color=COL_LIGHT, s=18, alpha=0.7,
               edgecolors="none", zorder=3, label=None)
    ax.scatter(theta + 0.08, dark_mrls, color=COL_DARK, s=18, alpha=0.7,
               edgecolors="none", zorder=3, label=None)

    # Connecting lines between light and dark for each session
    for i in range(n):
        ax.plot([theta[i], theta[i] + 0.08],
                [light_mrls[i], dark_mrls[i]],
                color="#CCCCCC", linewidth=0.4, zorder=2)

    # Grand mean MRL as concentric circles
    mean_mrl_light = float(np.mean(light_mrls))
    mean_mrl_dark = float(np.mean(dark_mrls))
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax.plot(theta_circle, [mean_mrl_light] * 100, color=COL_LIGHT,
            linewidth=1.5, linestyle="-",
            label=f"Light (MRL={mean_mrl_light:.3f})")
    ax.plot(theta_circle, [mean_mrl_dark] * 100, color=COL_DARK,
            linewidth=1.5, linestyle="--",
            label=f"Dark (MRL={mean_mrl_dark:.3f})")

    ax.set_ylim(0, 0.6)
    ax.set_yticks([0.2, 0.4])
    ax.set_yticklabels(["0.2", "0.4"], fontsize=5)
    ax.legend(fontsize=5, loc="lower left", bbox_to_anchor=(-0.15, -0.18),
              frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
    ax.set_title("Per-session MRL", fontsize=7, pad=12)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Generate all manuscript figures."""
    results = load_results()
    print(f"Loaded results: {len(results['per_session'])} sessions, "
          f"{len(get_usable_sessions(results))} usable")

    figure1(results)
    figure2(results)
    figure3(results)
    figure4(results)
    figure5(results)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
