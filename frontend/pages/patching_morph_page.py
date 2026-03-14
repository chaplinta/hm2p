"""Patching morphology visualization page.

Displays per-cell and population-level morphological data loaded from
TREES-toolbox morph_data.mat files in the patching processed directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from hm2p.patching.plotting.morph_plots import (
    CELLTYPE_COLOURS,
    COMPARTMENT_COLOURS,
    DEFAULT_PROCESSED_DIR,
    GSTATS_LABELS,
    build_metrics_dataframe,
    cell_index_from_dirname,
    discover_morph_cells,
    format_stats_table,
    load_morph_mat,
    plot_density_heatmap,
    plot_metric_comparison,
    plot_population_overlay,
    plot_population_sholl,
    plot_sholl_profile,
    plot_single_morphology_2d,
)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "patching" / "analysis"


@st.cache_data(ttl=600)
def _load_metrics_csv() -> pd.DataFrame | None:
    """Load patching metrics.csv for cell type info."""
    path = RESULTS_DIR / "metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=600)
def _discover_cells() -> list[str]:
    """Discover cell IDs with morph_data.mat."""
    return discover_morph_cells(DEFAULT_PROCESSED_DIR)


@st.cache_data(ttl=600)
def _load_morph(cell_id: str) -> dict | None:
    """Load morph data for a single cell."""
    mat_path = DEFAULT_PROCESSED_DIR / cell_id / "morph_data.mat"
    if not mat_path.exists():
        return None
    try:
        return load_morph_mat(mat_path)
    except Exception as e:
        st.error(f"Failed to load {cell_id}: {e}")
        return None


@st.cache_data(ttl=600)
def _load_all_morph(cell_ids: tuple[str, ...]) -> dict[str, dict]:
    """Load morph data for all cells (cached)."""
    result = {}
    for cid in cell_ids:
        md = _load_morph(cid)
        if md is not None:
            result[cid] = md
    return result


def _get_cell_types(cell_ids: list[str]) -> dict[str, str]:
    """Map cell_id -> cell_type using metrics.csv."""
    metrics = _load_metrics_csv()
    if metrics is None:
        return {}
    ct_map = {}
    for cid in cell_ids:
        idx = cell_index_from_dirname(cid)
        row = metrics[metrics["cell_index"] == idx]
        if len(row) > 0:
            ct_map[cid] = row.iloc[0]["cell_type"]
    return ct_map


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Patching Morphology")

if not DEFAULT_PROCESSED_DIR.is_dir():
    st.warning(
        "Patching data directory not available. "
        "The container may need to be rebuilt with the `/data/patching` bind mount."
    )
    st.stop()

cell_ids = _discover_cells()
if not cell_ids:
    st.warning(
        "No morph_data.mat files found in "
        f"`{DEFAULT_PROCESSED_DIR}`. No morphology data available."
    )
    st.stop()

st.caption(f"Found **{len(cell_ids)}** cells with morphology data.")

# Pre-load cell type mapping
cell_type_map = _get_cell_types(cell_ids)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_single, tab_pop, tab_sholl, tab_metrics, tab_methods = st.tabs([
    "Single Cell",
    "Population",
    "Sholl Analysis",
    "Metrics",
    "Methods & References",
])


# ===== Single Cell =====
with tab_single:
    st.header("Single cell morphology")

    # Build display names with cell type
    display_names = []
    for cid in cell_ids:
        ct = cell_type_map.get(cid, "")
        suffix = f"  [{ct}]" if ct else ""
        display_names.append(f"{cid}{suffix}")

    selected_display = st.selectbox(
        "Select cell",
        display_names,
        key="morph_single_cell_select",
    )
    selected_cell = cell_ids[display_names.index(selected_display)]

    md = _load_morph(selected_cell)
    if md is None:
        st.error(f"Could not load morphology for {selected_cell}.")
    else:
        col_plot, col_stats = st.columns([2, 1])

        with col_plot:
            fig = plot_single_morphology_2d(
                md,
                title=f"{selected_cell}",
                width=600,
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True, key="morph_single_2d")

        with col_stats:
            st.subheader("Morphological statistics")
            stats_df = format_stats_table(md)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

            st.subheader("Surface stats")
            su = md["surface_stats"]
            st.metric("Soma-to-surface distance", f"{su['dist_soma']:.1f} um")
            st.metric("Soma angle from surface", f"{su['angle_soma_deg']:.1f} deg")

        # Sholl for this cell
        st.subheader("Sholl profile")
        fig_sholl = plot_sholl_profile(md, title=selected_cell, width=700, height=350)
        st.plotly_chart(fig_sholl, use_container_width=True, key="morph_single_sholl")


# ===== Population =====
with tab_pop:
    st.header("Population morphology")

    all_morph = _load_all_morph(tuple(cell_ids))
    if not all_morph:
        st.warning("No morphology data could be loaded.")
    else:
        st.subheader("Overlay")
        fig_overlay = plot_population_overlay(all_morph, width=700, height=700)
        st.plotly_chart(fig_overlay, use_container_width=True, key="morph_pop_overlay")

        st.subheader("Density heatmap")
        comp_choice = st.radio(
            "Compartment",
            ["apical", "basal"],
            horizontal=True,
            key="morph_density_compartment",
        )
        bin_size = st.slider(
            "Bin size (um)", 5, 50, 10, key="morph_density_bin"
        )
        fig_density = plot_density_heatmap(
            all_morph, compartment=comp_choice, bin_size=float(bin_size),
            width=700, height=700,
        )
        st.plotly_chart(fig_density, use_container_width=True, key="morph_pop_density")


# ===== Sholl Analysis =====
with tab_sholl:
    st.header("Sholl analysis")

    all_morph_sholl = _load_all_morph(tuple(cell_ids))
    if not all_morph_sholl:
        st.warning("No morphology data could be loaded.")
    else:
        # Per-cell Sholl selector
        st.subheader("Per-cell Sholl profiles")
        sholl_comp = st.radio(
            "Compartment",
            ["apical", "basal"],
            horizontal=True,
            key="morph_sholl_comp",
        )

        import plotly.graph_objects as go

        fig_all_sholl = go.Figure()
        key_name = f"{sholl_comp}_dsholl"
        for cid, md_s in all_morph_sholl.items():
            profile = md_s[key_name]
            if len(profile) == 0:
                continue
            radii = np.arange(1, len(profile) + 1)
            ct = cell_type_map.get(cid, "unknown")
            colour = CELLTYPE_COLOURS.get(ct, "#888888")
            fig_all_sholl.add_trace(go.Scatter(
                x=radii,
                y=profile,
                mode="lines",
                name=cid,
                line=dict(color=colour, width=1),
                opacity=0.5,
            ))
        fig_all_sholl.update_layout(
            title=f"All {sholl_comp} Sholl profiles (colour = cell type)",
            xaxis=dict(title="Distance from soma (um)"),
            yaxis=dict(title="Intersections"),
            height=450,
            template="plotly_white",
        )
        st.plotly_chart(fig_all_sholl, use_container_width=True, key="morph_sholl_all")

        # Population mean
        st.subheader("Population mean +/- SEM")
        fig_pop_sholl = plot_population_sholl(
            all_morph_sholl, compartment=sholl_comp, width=700, height=400,
        )
        st.plotly_chart(fig_pop_sholl, use_container_width=True, key="morph_sholl_pop")


# ===== Metrics =====
with tab_metrics:
    st.header("Morphological metrics comparison")

    all_morph_met = _load_all_morph(tuple(cell_ids))
    if not all_morph_met:
        st.warning("No morphology data could be loaded.")
    else:
        metrics_df = build_metrics_dataframe(all_morph_met, cell_type_map)

        if metrics_df.empty:
            st.warning("No metrics extracted.")
        else:
            met_comp = st.radio(
                "Compartment",
                ["apical", "basal"],
                horizontal=True,
                key="morph_metrics_comp",
            )

            # Get available metrics for this compartment
            available = metrics_df[metrics_df["compartment"] == met_comp]["metric"].unique()
            available_sorted = sorted(available, key=lambda m: list(GSTATS_LABELS.keys()).index(m) if m in GSTATS_LABELS else 999)

            # Display in a grid
            n_cols = 3
            for row_start in range(0, len(available_sorted), n_cols):
                cols = st.columns(n_cols)
                for j, col in enumerate(cols):
                    idx = row_start + j
                    if idx >= len(available_sorted):
                        break
                    metric_name = available_sorted[idx]
                    with col:
                        fig_box = plot_metric_comparison(
                            metrics_df,
                            metric_name,
                            compartment=met_comp,
                            width=350,
                            height=350,
                        )
                        st.plotly_chart(
                            fig_box,
                            use_container_width=True,
                            key=f"morph_metric_{met_comp}_{metric_name}",
                        )

            # Summary table
            st.subheader("Summary table")
            pivot = metrics_df[metrics_df["compartment"] == met_comp].copy()
            if not pivot.empty:
                summary_rows = []
                for m in available_sorted:
                    m_df = pivot[pivot["metric"] == m]
                    label, unit = GSTATS_LABELS.get(m, (m, ""))
                    row_data: dict = {"Metric": label, "Unit": unit}
                    for ct in sorted(m_df["cell_type"].unique()):
                        vals = m_df[m_df["cell_type"] == ct]["value"]
                        row_data[f"{ct} mean"] = f"{vals.mean():.2f}"
                        row_data[f"{ct} std"] = f"{vals.std():.2f}"
                        row_data[f"{ct} n"] = len(vals)
                    summary_rows.append(row_data)
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ===== Methods & References =====
with tab_methods:
    st.header("Methods & References")

    st.markdown("""
### Morphological reconstruction

Neurons were filled with biocytin during whole-cell patch-clamp recording,
then processed for confocal imaging.  Dendritic trees were traced and
reconstructed using the **TREES toolbox** in MATLAB.  Reconstructions
were soma-subtracted (coordinates centred on soma), rotated to align the
pial surface horizontally, and split into apical and basal compartments.

### Sholl analysis

Sholl analysis counts the number of dendritic branches intersecting
concentric spheres of increasing radius centred on the soma.  The Sholl
profile characterises dendritic complexity as a function of distance from
the cell body.

### Metrics

Global tree statistics (total length, branch points, branch order, spatial
extent, convex hull, asymmetry, etc.) were computed per compartment using
the TREES toolbox ``stats_tree`` functions.

### References

- Cuntz et al. 2010. "One rule to grow them all: a general theory of
  neuronal branching and its practical application." *PLoS Comput Biol.*
  doi:[10.1371/journal.pcbi.1000877](https://doi.org/10.1371/journal.pcbi.1000877)
  GitHub: [cuntzlab/treestoolbox](https://github.com/cuntzlab/treestoolbox)

- Sholl 1953. "Dendritic organization in the neurons of the visual and
  motor cortices of the cat." *J Anat* 87(4):387-406.
""")
