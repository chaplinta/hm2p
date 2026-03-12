"""Standardized comparison plots.

Two canonical comparison patterns used throughout the analysis:
1. Between-group box plots (Penk+ vs CamKII+) with unpaired Wilcoxon (Mann-Whitney U)
2. Within-cell scatter plots (2 conditions) with paired Wilcoxon signed-rank test
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from hm2p.constants import CELLTYPE_LABEL, HEX_NONPENK, HEX_PENK


def format_pvalue(p: float) -> str:
    """Format a p-value for display on a plot.

    Parameters
    ----------
    p : float
        The p-value to format.

    Returns
    -------
    str
        Formatted string, e.g. "p = 0.003" or "p < 0.001".
    """
    if np.isnan(p):
        return "p = NaN"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def celltype_comparison_box(
    penk_values: np.ndarray,
    nonpenk_values: np.ndarray,
    measure_name: str,
    *,
    title: str | None = None,
    penk_color: str = HEX_PENK,
    nonpenk_color: str = HEX_NONPENK,
    penk_label: str | None = None,
    nonpenk_label: str | None = None,
    height: int = 500,
    width: int = 450,
) -> tuple[go.Figure, dict[str, Any]]:
    """Between-group box plot comparing Penk+ vs CamKII+ for a single measure.

    Creates a Plotly box plot with individual data points (jittered) and runs
    an unpaired two-sided Mann-Whitney U test.

    Parameters
    ----------
    penk_values : array-like
        Values for the Penk+ group.
    nonpenk_values : array-like
        Values for the CamKII+ group.
    measure_name : str
        Label for the y-axis (the quantity being compared).
    title : str, optional
        Plot title. Defaults to the measure name.
    penk_color : str
        Hex colour for the Penk+ group.
    nonpenk_color : str
        Hex colour for the CamKII+ group.
    penk_label : str, optional
        Display label for the Penk+ group. Defaults to ``CELLTYPE_LABEL["penk"]``.
    nonpenk_label : str, optional
        Display label for the CamKII+ group. Defaults to ``CELLTYPE_LABEL["nonpenk"]``.
    height : int
        Figure height in pixels.
    width : int
        Figure width in pixels.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The box plot figure.
    stat_result : dict
        Keys: ``test``, ``U``, ``p``, ``n_penk``, ``n_nonpenk``, ``measure``.
    """
    penk_arr = np.asarray(penk_values, dtype=float)
    nonpenk_arr = np.asarray(nonpenk_values, dtype=float)

    # Remove NaNs
    penk_arr = penk_arr[~np.isnan(penk_arr)]
    nonpenk_arr = nonpenk_arr[~np.isnan(nonpenk_arr)]

    if penk_label is None:
        penk_label = CELLTYPE_LABEL["penk"]
    if nonpenk_label is None:
        nonpenk_label = CELLTYPE_LABEL["nonpenk"]
    if title is None:
        title = measure_name

    # --- Statistical test ---
    if len(penk_arr) > 0 and len(nonpenk_arr) > 0:
        u_stat, p_val = stats.mannwhitneyu(
            penk_arr, nonpenk_arr, alternative="two-sided"
        )
    else:
        u_stat = float("nan")
        p_val = float("nan")

    stat_result = {
        "test": "Mann-Whitney U",
        "U": float(u_stat),
        "p": float(p_val),
        "n_penk": len(penk_arr),
        "n_nonpenk": len(nonpenk_arr),
        "measure": measure_name,
    }

    # --- Build figure ---
    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=penk_arr,
            name=penk_label,
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(color=penk_color),
            line=dict(color=penk_color),
            fillcolor=_lighten_hex(penk_color, 0.3),
        )
    )

    fig.add_trace(
        go.Box(
            y=nonpenk_arr,
            name=nonpenk_label,
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(color=nonpenk_color),
            line=dict(color=nonpenk_color),
            fillcolor=_lighten_hex(nonpenk_color, 0.3),
        )
    )

    # --- P-value annotation bracket ---
    all_vals = np.concatenate([penk_arr, nonpenk_arr]) if (
        len(penk_arr) > 0 and len(nonpenk_arr) > 0
    ) else (penk_arr if len(penk_arr) > 0 else nonpenk_arr)

    if len(all_vals) > 0:
        y_max = float(np.nanmax(all_vals))
        y_range = float(np.nanmax(all_vals) - np.nanmin(all_vals))
        bracket_y = y_max + 0.05 * max(y_range, abs(y_max) * 0.1)
        text_y = bracket_y + 0.03 * max(y_range, abs(y_max) * 0.1)
    else:
        bracket_y = 1.0
        text_y = 1.1

    # Bracket lines
    fig.add_shape(
        type="line", x0=0, x1=0, y0=bracket_y, y1=bracket_y * 0.98,
        xref="x", yref="y", line=dict(color="black", width=1.5),
    )
    fig.add_shape(
        type="line", x0=0, x1=1, y0=bracket_y, y1=bracket_y,
        xref="x", yref="y", line=dict(color="black", width=1.5),
    )
    fig.add_shape(
        type="line", x0=1, x1=1, y0=bracket_y, y1=bracket_y * 0.98,
        xref="x", yref="y", line=dict(color="black", width=1.5),
    )

    # P-value text
    fig.add_annotation(
        x=0.5, y=text_y, xref="x", yref="y",
        text=format_pvalue(p_val),
        showarrow=False, font=dict(size=13),
    )

    fig.update_layout(
        title=title,
        yaxis_title=measure_name,
        showlegend=False,
        height=height,
        width=width,
        template="plotly_white",
    )

    return fig, stat_result


def paired_condition_scatter(
    values_cond1: np.ndarray,
    values_cond2: np.ndarray,
    label1: str,
    label2: str,
    measure_name: str,
    *,
    title: str | None = None,
    marker_color: str = "steelblue",
    marker_size: int = 8,
    height: int = 500,
    width: int = 500,
) -> tuple[go.Figure, dict[str, Any]]:
    """Within-cell scatter plot comparing the same cells across two conditions.

    Creates a square-aspect scatter plot with a line of unity (dashed gray)
    and runs a paired two-sided Wilcoxon signed-rank test.

    Parameters
    ----------
    values_cond1 : array-like
        Values for condition 1 (x-axis).
    values_cond2 : array-like
        Values for condition 2 (y-axis).
    label1 : str
        Name of condition 1 (e.g. "Light").
    label2 : str
        Name of condition 2 (e.g. "Dark").
    measure_name : str
        The quantity being compared (used in axis labels).
    title : str, optional
        Plot title. Defaults to ``"{measure_name}: {label1} vs {label2}"``.
    marker_color : str
        Colour for all scatter points.
    marker_size : int
        Marker size in pixels.
    height : int
        Figure height in pixels.
    width : int
        Figure width in pixels.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The scatter plot figure.
    stat_result : dict
        Keys: ``test``, ``W``, ``p``, ``n``, ``measure``.
    """
    arr1 = np.asarray(values_cond1, dtype=float)
    arr2 = np.asarray(values_cond2, dtype=float)

    # Remove paired NaNs
    valid = ~(np.isnan(arr1) | np.isnan(arr2))
    arr1 = arr1[valid]
    arr2 = arr2[valid]

    if title is None:
        title = f"{measure_name}: {label1} vs {label2}"

    # --- Statistical test ---
    if len(arr1) == 0 or np.all(arr1 == arr2):
        w_stat = float("nan")
        p_val = float("nan")
    else:
        try:
            w_stat, p_val = stats.wilcoxon(arr1, arr2, alternative="two-sided")
        except ValueError:
            # e.g. all differences are zero after rounding
            w_stat = float("nan")
            p_val = float("nan")

    stat_result = {
        "test": "Wilcoxon signed-rank",
        "W": float(w_stat),
        "p": float(p_val),
        "n": len(arr1),
        "measure": measure_name,
    }

    # --- Build figure ---
    fig = go.Figure()

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=arr1,
            y=arr2,
            mode="markers",
            marker=dict(color=marker_color, size=marker_size, opacity=0.7),
            name="cells",
        )
    )

    # Line of unity
    if len(arr1) > 0:
        all_vals = np.concatenate([arr1, arr2])
        lo = float(np.nanmin(all_vals))
        hi = float(np.nanmax(all_vals))
        pad = 0.05 * max(hi - lo, abs(hi) * 0.01, 1e-6)
        unity_range = [lo - pad, hi + pad]
    else:
        unity_range = [0, 1]

    fig.add_trace(
        go.Scatter(
            x=unity_range,
            y=unity_range,
            mode="lines",
            line=dict(color="gray", dash="dash", width=1.5),
            name="unity",
            showlegend=False,
        )
    )

    # P-value annotation
    fig.add_annotation(
        x=0.05, y=0.95, xref="paper", yref="paper",
        text=format_pvalue(p_val),
        showarrow=False, font=dict(size=13),
        xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray", borderwidth=1, borderpad=4,
    )

    fig.update_layout(
        title=title,
        xaxis_title=f"{measure_name} ({label1})",
        yaxis_title=f"{measure_name} ({label2})",
        height=height,
        width=width,
        template="plotly_white",
        showlegend=False,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    return fig, stat_result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lighten_hex(hex_color: str, alpha: float = 0.3) -> str:
    """Blend a hex colour toward white to create a lighter fill.

    Parameters
    ----------
    hex_color : str
        Hex colour string (e.g. "#0000FF").
    alpha : float
        Blend factor (0 = original, 1 = white).

    Returns
    -------
    str
        Lightened hex colour string.
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(r + (255 - r) * alpha)
    g = int(g + (255 - g) * alpha)
    b = int(b + (255 - b) * alpha)
    return f"#{r:02x}{g:02x}{b:02x}"
