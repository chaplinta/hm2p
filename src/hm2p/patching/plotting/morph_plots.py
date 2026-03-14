"""Morphology visualization: .mat loading, 2D projections, Sholl, and metrics.

Loads TREES-toolbox morph_data.mat files produced by the MATLAB patching
pipeline.  Each file contains tree structures (sparse adjacency ``dA``,
coordinates ``X/Y/Z``, diameter ``D``, region ``R``), pre-computed global
stats (``gstats``), distributions (``dstats``), and Sholl profiles
(``dsholl``).

Citations
---------
TREES toolbox:
    Cuntz et al. 2010. "One rule to grow them all: a general theory of
    neuronal branching and its practical application." PLoS Comput Biol.
    doi:10.1371/journal.pcbi.1000877
    GitHub: https://github.com/cuntzlab/treestoolbox

Sholl analysis:
    Sholl 1953. "Dendritic organization in the neurons of the visual and
    motor cortices of the cat." J Anat 87(4):387-406.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

#: Default processed directory (read-only bind mount).
DEFAULT_PROCESSED_DIR = Path("/data/patching/patching/processed")

#: Colour scheme for compartments.
COMPARTMENT_COLOURS: dict[str, str] = {
    "apical": "#1f77b4",   # blue
    "basal": "#d62728",    # red
    "soma": "#111111",     # near-black
}

#: Colour scheme for cell types.
CELLTYPE_COLOURS: dict[str, str] = {
    "penkpos": "#1f77b4",  # blue
    "penkneg": "#ff7f0e",  # orange
}

#: Human-readable gstats metric labels and units.
GSTATS_LABELS: dict[str, tuple[str, str]] = {
    "len": ("Total length", "um"),
    "max_plen": ("Max path length", "um"),
    "bpoints": ("Branch points", "count"),
    "mpeucl": ("Mean path/Euclidean ratio", ""),
    "maxbo": ("Max branch order", ""),
    "mangleB": ("Mean branch angle", "rad"),
    "mblen": ("Mean branch length", "um"),
    "mplen": ("Mean path length", "um"),
    "mbo": ("Mean branch order", ""),
    "width": ("Width", "um"),
    "height": ("Height", "um"),
    "depth": ("Depth", "um"),
    "wh": ("Width/height ratio", ""),
    "wd": ("Width/depth ratio", ""),
    "hull": ("Convex hull volume", "um^3"),
    "masym": ("Mean asymmetry", ""),
    "mparea": ("Mean partition area", ""),
}


# ============================================================================
# Discovery
# ============================================================================


def discover_morph_cells(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> list[str]:
    """Return sorted list of cell_id directory names that contain morph_data.mat.

    Parameters
    ----------
    processed_dir : Path
        Root of the per-cell processed directory tree.

    Returns
    -------
    list[str]
        Cell directory names (e.g. ``"015-CAA-1116461-S2-1"``), sorted.
    """
    if not processed_dir.is_dir():
        return []
    cells = []
    for d in sorted(processed_dir.iterdir()):
        if d.is_dir() and (d / "morph_data.mat").exists():
            cells.append(d.name)
    return cells


def cell_index_from_dirname(dirname: str) -> int:
    """Extract the integer cell index from a directory name like ``'015-CAA-...'``."""
    return int(dirname.split("-")[0])


# ============================================================================
# MAT loading
# ============================================================================


def load_morph_mat(mat_path: Path | str) -> dict[str, Any]:
    """Load a morph_data.mat file into a Python dict.

    Parameters
    ----------
    mat_path : Path or str
        Path to ``morph_data.mat``.

    Returns
    -------
    dict
        Keys:

        - ``"trees"`` -- list of dicts, each with ``name``, ``X``, ``Y``,
          ``Z``, ``D``, ``edges`` (``(E,2)`` array from sparse ``dA``).
        - ``"soma_center"`` -- ``(3,)`` array ``[mx, my, mz]``.
        - ``"apical_gstats"`` -- dict of global stats for apical tree.
        - ``"basal_gstats"`` -- dict of global stats for basal tree.
        - ``"apical_dsholl"`` -- 1-D array of Sholl intersection counts.
        - ``"basal_dsholl"`` -- 1-D array of Sholl intersection counts.
        - ``"apical_dstats"`` -- dict of distribution arrays.
        - ``"basal_dstats"`` -- dict of distribution arrays.
        - ``"surface_stats"`` -- dict with ``dist_soma``, ``angle_soma_deg``.
    """
    import scipy.io as sio

    mat_path = Path(mat_path)
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    md = mat["morphData"]

    # --- Trees ---
    trees = []
    traces = md.traces
    # Handle case where traces is a single struct (not an array)
    if not hasattr(traces, "__len__"):
        traces = [traces]
    for t in traces:
        dA = t.dA
        if issparse(dA):
            rows, cols = dA.nonzero()
            edges = np.column_stack([rows, cols])
        else:
            rows, cols = np.nonzero(dA)
            edges = np.column_stack([rows, cols]) if len(rows) > 0 else np.empty((0, 2), dtype=int)
        trees.append({
            "name": str(t.name),
            "X": np.asarray(t.X, dtype=float),
            "Y": np.asarray(t.Y, dtype=float),
            "Z": np.asarray(t.Z, dtype=float),
            "D": np.asarray(t.D, dtype=float),
            "edges": edges,
        })

    # --- Soma center ---
    ss = md.soma_stats
    soma_center = np.array([float(ss.mx), float(ss.my), float(ss.mz)])

    # --- Gstats ---
    apical_gstats = _extract_gstats(md.apical_stats.gstats)
    basal_gstats = _extract_gstats(md.basal_stats.gstats)

    # --- Sholl ---
    apical_dsholl = np.asarray(md.apical_stats.dsholl, dtype=float)
    basal_dsholl = np.asarray(md.basal_stats.dsholl, dtype=float)

    # --- Dstats ---
    apical_dstats = _extract_dstats(md.apical_stats.dstats)
    basal_dstats = _extract_dstats(md.basal_stats.dstats)

    # --- Surface stats ---
    su = md.surface_stats
    surface_stats = {
        "dist_soma": float(su.dist_soma),
        "angle_soma_deg": float(su.angle_soma_deg),
    }

    return {
        "trees": trees,
        "soma_center": soma_center,
        "apical_gstats": apical_gstats,
        "basal_gstats": basal_gstats,
        "apical_dsholl": apical_dsholl,
        "basal_dsholl": basal_dsholl,
        "apical_dstats": apical_dstats,
        "basal_dstats": basal_dstats,
        "surface_stats": surface_stats,
    }


def _extract_gstats(gs: Any) -> dict[str, float]:
    """Pull all scalar fields from a gstats struct into a plain dict."""
    result: dict[str, float] = {}
    if not hasattr(gs, "_fieldnames"):
        return result
    for fn in gs._fieldnames:
        val = getattr(gs, fn)
        try:
            val = float(val)
            result[fn] = val
        except (TypeError, ValueError):
            # Skip array-valued or non-numeric fields (e.g. chullx/y/z)
            if hasattr(val, "shape") and val.ndim == 0:
                result[fn] = float(val)
    return result


def _extract_dstats(ds: Any) -> dict[str, np.ndarray]:
    """Pull all array fields from a dstats struct."""
    result: dict[str, np.ndarray] = {}
    if not hasattr(ds, "_fieldnames"):
        return result
    for fn in ds._fieldnames:
        val = getattr(ds, fn)
        if hasattr(val, "shape") and val.ndim >= 1:
            result[fn] = np.asarray(val, dtype=float)
    return result


# ============================================================================
# Plotting: single morphology 2D
# ============================================================================


def plot_single_morphology_2d(
    morph_data: dict[str, Any],
    *,
    title: str = "",
    show_soma: bool = True,
    width: int = 700,
    height: int = 700,
) -> go.Figure:
    """Plot a 2D X-Y projection of apical + basal + soma trees.

    Parameters
    ----------
    morph_data : dict
        Output of :func:`load_morph_mat`.
    title : str
        Figure title.
    show_soma : bool
        Whether to draw the soma as a filled marker.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    for tree in morph_data["trees"]:
        name_lower = tree["name"].lower()
        if "apical" in name_lower:
            colour = COMPARTMENT_COLOURS["apical"]
            label = "Apical"
        elif "basal" in name_lower:
            colour = COMPARTMENT_COLOURS["basal"]
            label = "Basal"
        elif "soma" in name_lower:
            if show_soma:
                # Draw soma outline as a trace
                colour = COMPARTMENT_COLOURS["soma"]
                label = "Soma"
            else:
                continue
        else:
            colour = "#888888"
            label = tree["name"]

        X = tree["X"]
        Y = tree["Y"]
        edges = tree["edges"]

        # Draw edges as line segments (using None separators for efficiency)
        if len(edges) > 0:
            x_lines: list[float | None] = []
            y_lines: list[float | None] = []
            for r, c in edges:
                x_lines.extend([X[r], X[c], None])
                y_lines.extend([Y[r], Y[c], None])

            fig.add_trace(go.Scatter(
                x=x_lines,
                y=y_lines,
                mode="lines",
                line=dict(color=colour, width=1),
                name=label,
                hoverinfo="name",
            ))

    # Draw soma as a dot at the soma center
    if show_soma:
        sc = morph_data["soma_center"]
        fig.add_trace(go.Scatter(
            x=[sc[0]],
            y=[sc[1]],
            mode="markers",
            marker=dict(color=COMPARTMENT_COLOURS["soma"], size=8, symbol="circle"),
            name="Soma center",
            hoverinfo="name",
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="X (um)", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y (um)"),
        width=width,
        height=height,
        showlegend=True,
        template="plotly_white",
    )
    return fig


# ============================================================================
# Plotting: population overlay
# ============================================================================


def plot_population_overlay(
    all_morph: dict[str, dict[str, Any]],
    *,
    width: int = 800,
    height: int = 800,
    alpha: float = 0.15,
) -> go.Figure:
    """Overlay all cell morphologies centred on soma, with low opacity.

    Parameters
    ----------
    all_morph : dict
        ``{cell_id: morph_data}`` dict.
    alpha, width, height
        Display parameters.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    for i, (cell_id, md) in enumerate(all_morph.items()):
        soma = md["soma_center"]
        for tree in md["trees"]:
            name_lower = tree["name"].lower()
            if "soma" in name_lower:
                continue
            if "apical" in name_lower:
                colour = COMPARTMENT_COLOURS["apical"]
            elif "basal" in name_lower:
                colour = COMPARTMENT_COLOURS["basal"]
            else:
                colour = "#888888"

            X = tree["X"] - soma[0]
            Y = tree["Y"] - soma[1]
            edges = tree["edges"]

            if len(edges) > 0:
                x_lines: list[float | None] = []
                y_lines: list[float | None] = []
                for r, c in edges:
                    x_lines.extend([X[r], X[c], None])
                    y_lines.extend([Y[r], Y[c], None])

                fig.add_trace(go.Scatter(
                    x=x_lines,
                    y=y_lines,
                    mode="lines",
                    line=dict(color=colour, width=0.5),
                    opacity=alpha,
                    name=cell_id,
                    showlegend=(i == 0 and "apical" in name_lower) or (i == 0 and "basal" in name_lower),
                    hoverinfo="name",
                ))

    # Soma at origin
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers",
        marker=dict(color="black", size=6),
        name="Soma (origin)",
    ))

    fig.update_layout(
        title="Population overlay (soma-centred)",
        xaxis=dict(title="X (um)", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y (um)"),
        width=width,
        height=height,
        showlegend=True,
        template="plotly_white",
    )
    return fig


def plot_density_heatmap(
    all_morph: dict[str, dict[str, Any]],
    *,
    compartment: str = "apical",
    bin_size: float = 10.0,
    width: int = 700,
    height: int = 700,
) -> go.Figure:
    """2D histogram (density) of all node positions, soma-centred.

    Parameters
    ----------
    all_morph : dict
        ``{cell_id: morph_data}`` dict.
    compartment : str
        ``"apical"`` or ``"basal"``.
    bin_size : float
        Bin width in um.
    width, height : int
        Figure dimensions.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    all_x: list[float] = []
    all_y: list[float] = []

    for md in all_morph.values():
        soma = md["soma_center"]
        for tree in md["trees"]:
            if compartment.lower() in tree["name"].lower():
                all_x.extend((tree["X"] - soma[0]).tolist())
                all_y.extend((tree["Y"] - soma[1]).tolist())

    fig = go.Figure()
    if all_x:
        fig.add_trace(go.Histogram2d(
            x=all_x,
            y=all_y,
            xbins=dict(size=bin_size),
            ybins=dict(size=bin_size),
            colorscale="Hot",
            reversescale=True,
            colorbar=dict(title="Node count"),
        ))

    fig.update_layout(
        title=f"{compartment.capitalize()} density (soma-centred, {bin_size} um bins)",
        xaxis=dict(title="X (um)", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y (um)"),
        width=width,
        height=height,
        template="plotly_white",
    )
    return fig


# ============================================================================
# Plotting: Sholl
# ============================================================================


def plot_sholl_profile(
    morph_data: dict[str, Any],
    *,
    title: str = "",
    width: int = 700,
    height: int = 400,
) -> go.Figure:
    """Plot Sholl intersection profile for a single cell (apical + basal).

    Parameters
    ----------
    morph_data : dict
        Output of :func:`load_morph_mat`.
    title : str
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    for compartment, key in [("Apical", "apical_dsholl"), ("Basal", "basal_dsholl")]:
        profile = morph_data[key]
        if len(profile) == 0:
            continue
        radii = np.arange(1, len(profile) + 1)
        colour = COMPARTMENT_COLOURS[compartment.lower()]
        fig.add_trace(go.Scatter(
            x=radii,
            y=profile,
            mode="lines",
            name=compartment,
            line=dict(color=colour),
        ))

    fig.update_layout(
        title=title or "Sholl analysis",
        xaxis=dict(title="Distance from soma (um)"),
        yaxis=dict(title="Intersections"),
        width=width,
        height=height,
        template="plotly_white",
    )
    return fig


def compute_population_sholl(
    all_morph: dict[str, dict[str, Any]],
    compartment: str = "apical",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute population mean +/- SEM Sholl profile.

    Parameters
    ----------
    all_morph : dict
        ``{cell_id: morph_data}`` dict.
    compartment : str
        ``"apical"`` or ``"basal"``.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        ``(radii, mean_profile, sem_profile)``
    """
    key = f"{compartment}_dsholl"
    profiles = []
    for md in all_morph.values():
        p = md[key]
        if len(p) > 0:
            profiles.append(p)

    if not profiles:
        return np.array([]), np.array([]), np.array([])

    # Pad to same length
    max_len = max(len(p) for p in profiles)
    padded = np.zeros((len(profiles), max_len))
    for i, p in enumerate(profiles):
        padded[i, : len(p)] = p

    radii = np.arange(1, max_len + 1)
    mean_profile = np.mean(padded, axis=0)
    sem_profile = np.std(padded, axis=0, ddof=1) / np.sqrt(len(profiles))
    return radii, mean_profile, sem_profile


def _hex_to_rgba(hex_colour: str, alpha: float = 1.0) -> str:
    """Convert a hex colour like ``'#1f77b4'`` to ``'rgba(31,119,180,0.2)'``."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_population_sholl(
    all_morph: dict[str, dict[str, Any]],
    *,
    compartment: str = "apical",
    width: int = 700,
    height: int = 400,
) -> go.Figure:
    """Mean +/- SEM Sholl profile across all cells.

    Parameters
    ----------
    all_morph : dict
        ``{cell_id: morph_data}`` dict.
    compartment : str
        ``"apical"`` or ``"basal"``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    radii, mean_p, sem_p = compute_population_sholl(all_morph, compartment)
    colour = COMPARTMENT_COLOURS.get(compartment, "#888888")

    fig = go.Figure()
    if len(radii) > 0:
        # SEM band — convert hex colour to rgba for transparency
        fill_rgba = _hex_to_rgba(colour, 0.2)
        fig.add_trace(go.Scatter(
            x=np.concatenate([radii, radii[::-1]]),
            y=np.concatenate([mean_p + sem_p, (mean_p - sem_p)[::-1]]),
            fill="toself",
            fillcolor=fill_rgba,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Mean line
        fig.add_trace(go.Scatter(
            x=radii,
            y=mean_p,
            mode="lines",
            name=f"{compartment.capitalize()} (n={len(all_morph)})",
            line=dict(color=colour, width=2),
        ))

    fig.update_layout(
        title=f"Population Sholl -- {compartment.capitalize()} (mean +/- SEM)",
        xaxis=dict(title="Distance from soma (um)"),
        yaxis=dict(title="Intersections"),
        width=width,
        height=height,
        template="plotly_white",
    )
    return fig


# ============================================================================
# Plotting: metric comparison
# ============================================================================


def build_metrics_dataframe(
    all_morph: dict[str, dict[str, Any]],
    cell_types: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a tidy DataFrame of gstats metrics for all cells.

    Parameters
    ----------
    all_morph : dict
        ``{cell_id: morph_data}`` dict.
    cell_types : dict, optional
        ``{cell_id: cell_type}`` mapping. If *None*, ``cell_type`` column
        is set to ``"unknown"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``cell_id``, ``compartment``, ``metric``, ``value``,
        ``cell_type``.
    """
    rows: list[dict[str, Any]] = []
    for cell_id, md in all_morph.items():
        ct = cell_types.get(cell_id, "unknown") if cell_types else "unknown"
        for compartment, key in [("apical", "apical_gstats"), ("basal", "basal_gstats")]:
            gs = md[key]
            for metric, val in gs.items():
                if metric in GSTATS_LABELS:
                    rows.append({
                        "cell_id": cell_id,
                        "compartment": compartment,
                        "metric": metric,
                        "value": val,
                        "cell_type": ct,
                    })
    return pd.DataFrame(rows)


def plot_metric_comparison(
    metrics_df: pd.DataFrame,
    metric_name: str,
    *,
    compartment: str = "apical",
    width: int = 500,
    height: int = 400,
) -> go.Figure:
    """Box + strip plot comparing a metric between cell types.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of :func:`build_metrics_dataframe`.
    metric_name : str
        The gstats key to plot (e.g. ``"len"``, ``"bpoints"``).
    compartment : str
        ``"apical"`` or ``"basal"``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = metrics_df[
        (metrics_df["metric"] == metric_name) & (metrics_df["compartment"] == compartment)
    ].copy()

    label, unit = GSTATS_LABELS.get(metric_name, (metric_name, ""))
    y_label = f"{label} ({unit})" if unit else label

    fig = go.Figure()

    for ct in sorted(df["cell_type"].unique()):
        subset = df[df["cell_type"] == ct]
        colour = CELLTYPE_COLOURS.get(ct, "#888888")
        fig.add_trace(go.Box(
            y=subset["value"],
            name=ct,
            marker_color=colour,
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
        ))

    fig.update_layout(
        title=f"{compartment.capitalize()} -- {label}",
        yaxis=dict(title=y_label),
        width=width,
        height=height,
        template="plotly_white",
        showlegend=False,
    )
    return fig


# ============================================================================
# Stats table
# ============================================================================


def format_stats_table(morph_data: dict[str, Any]) -> pd.DataFrame:
    """Format apical + basal gstats into a display-ready DataFrame.

    Parameters
    ----------
    morph_data : dict
        Output of :func:`load_morph_mat`.

    Returns
    -------
    pd.DataFrame
        Columns: ``Metric``, ``Apical``, ``Basal``, ``Unit``.
    """
    rows = []
    for key in GSTATS_LABELS:
        label, unit = GSTATS_LABELS[key]
        api_val = morph_data["apical_gstats"].get(key, np.nan)
        bas_val = morph_data["basal_gstats"].get(key, np.nan)
        rows.append({
            "Metric": label,
            "Apical": round(api_val, 3) if np.isfinite(api_val) else "-",
            "Basal": round(bas_val, 3) if np.isfinite(bas_val) else "-",
            "Unit": unit,
        })
    return pd.DataFrame(rows)
