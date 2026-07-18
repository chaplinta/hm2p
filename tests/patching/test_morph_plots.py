"""Tests for hm2p.patching.plotting.morph_plots — morphology visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from hm2p.patching.plotting.morph_plots import (
    GSTATS_LABELS,
    _hex_to_rgba,
    build_metrics_dataframe,
    cell_index_from_dirname,
    compute_population_sholl,
    discover_morph_cells,
    format_stats_table,
    plot_density_heatmap,
    plot_metric_comparison,
    plot_population_overlay,
    plot_population_sholl,
    plot_sholl_profile,
    plot_single_morphology_2d,
)

# ---------------------------------------------------------------------------
# Helpers: synthetic morph_data dicts mimicking load_morph_mat output
# ---------------------------------------------------------------------------


def _make_tree(name: str, n_nodes: int = 20, seed: int = 0) -> dict:
    """Create a synthetic tree dict with random coordinates and edges."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n_nodes) * 50
    Y = rng.standard_normal(n_nodes) * 50
    Z = rng.standard_normal(n_nodes) * 10
    D = rng.uniform(0.3, 2.0, n_nodes)
    # Simple chain edges: 0->1, 1->2, ...
    edges = np.column_stack([np.arange(n_nodes - 1), np.arange(1, n_nodes)])
    return {"name": name, "X": X, "Y": Y, "Z": Z, "D": D, "edges": edges}


def _make_gstats(**overrides: float) -> dict[str, float]:
    """Create synthetic gstats dict with default values for all known keys."""
    defaults = {
        "len": 500.0,
        "max_plen": 200.0,
        "bpoints": 5.0,
        "mpeucl": 1.2,
        "maxbo": 3.0,
        "mangleB": 0.8,
        "mblen": 40.0,
        "mplen": 80.0,
        "mbo": 1.5,
        "width": 150.0,
        "height": 200.0,
        "depth": 30.0,
        "wh": 0.75,
        "wd": 5.0,
        "hull": 120000.0,
        "masym": 0.4,
        "mparea": 0.6,
    }
    defaults.update(overrides)
    return defaults


def _make_morph_data(
    *,
    include_apical: bool = True,
    include_basal: bool = True,
    n_nodes: int = 20,
    sholl_len: int = 10,
    seed: int = 0,
) -> dict:
    """Create a synthetic morph_data dict mimicking load_morph_mat output."""
    rng = np.random.default_rng(seed)
    trees = []
    if include_apical:
        trees.append(_make_tree("apical", n_nodes=n_nodes, seed=seed))
    if include_basal:
        trees.append(_make_tree("basal", n_nodes=n_nodes, seed=seed + 1))
    trees.append(_make_tree("soma", n_nodes=5, seed=seed + 2))

    return {
        "trees": trees,
        "soma_center": np.array([10.0, 20.0, 5.0]),
        "apical_gstats": _make_gstats(len=500.0) if include_apical else {},
        "basal_gstats": _make_gstats(len=300.0) if include_basal else {},
        "apical_dsholl": rng.integers(0, 20, size=sholl_len).astype(float)
        if include_apical
        else np.array([]),
        "basal_dsholl": rng.integers(0, 15, size=sholl_len).astype(float)
        if include_basal
        else np.array([]),
        "apical_dstats": {"blen": rng.standard_normal(15)} if include_apical else {},
        "basal_dstats": {"blen": rng.standard_normal(10)} if include_basal else {},
        "surface_stats": {"dist_soma": 120.0, "angle_soma_deg": 45.0},
    }


@pytest.fixture()
def morph_data() -> dict:
    return _make_morph_data()


@pytest.fixture()
def all_morph() -> dict[str, dict]:
    """Three synthetic cells for population-level tests."""
    return {
        "001-CAA-111-S1-1": _make_morph_data(seed=0),
        "002-CAA-222-S1-1": _make_morph_data(seed=10),
        "003-CAA-333-S1-1": _make_morph_data(seed=20),
    }


# ---------------------------------------------------------------------------
# Tests: discover_morph_cells
# ---------------------------------------------------------------------------


class TestDiscoverMorphCells:
    def test_finds_cells_with_morph_data(self, tmp_path: Path) -> None:
        """Directories containing morph_data.mat are returned, sorted."""
        (tmp_path / "015-CAA-111-S2-1").mkdir()
        (tmp_path / "015-CAA-111-S2-1" / "morph_data.mat").touch()
        (tmp_path / "003-CAA-222-S1-1").mkdir()
        (tmp_path / "003-CAA-222-S1-1" / "morph_data.mat").touch()
        # Directory without .mat should be excluded
        (tmp_path / "099-CAA-333-S1-1").mkdir()

        result = discover_morph_cells(tmp_path)
        assert result == ["003-CAA-222-S1-1", "015-CAA-111-S2-1"]

    def test_returns_empty_for_nonexistent_dir(self, tmp_path: Path) -> None:
        result = discover_morph_cells(tmp_path / "does_not_exist")
        assert result == []

    def test_returns_empty_for_empty_dir(self, tmp_path: Path) -> None:
        result = discover_morph_cells(tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: cell_index_from_dirname
# ---------------------------------------------------------------------------


class TestCellIndex:
    def test_extracts_index(self) -> None:
        assert cell_index_from_dirname("015-CAA-1116461-S2-1") == 15

    def test_extracts_single_digit(self) -> None:
        assert cell_index_from_dirname("3-XYZ") == 3


# ---------------------------------------------------------------------------
# Tests: plot_single_morphology_2d
# ---------------------------------------------------------------------------


class TestPlotSingleMorphology2D:
    def test_returns_figure(self, morph_data: dict) -> None:
        fig = plot_single_morphology_2d(morph_data, title="Test cell")
        assert isinstance(fig, go.Figure)

    def test_has_traces(self, morph_data: dict) -> None:
        fig = plot_single_morphology_2d(morph_data)
        # At least apical + basal edges + soma center + soma tree
        assert len(fig.data) >= 3

    def test_no_soma_flag(self, morph_data: dict) -> None:
        fig = plot_single_morphology_2d(morph_data, show_soma=False)
        trace_names = [t.name for t in fig.data]
        assert "Soma center" not in trace_names
        assert "Soma" not in trace_names

    def test_empty_trees(self) -> None:
        """morph_data with no trees should still return a valid figure."""
        md = _make_morph_data()
        md["trees"] = []
        fig = plot_single_morphology_2d(md)
        assert isinstance(fig, go.Figure)

    def test_tree_with_no_edges(self) -> None:
        """A tree with zero edges should not crash."""
        md = _make_morph_data(include_basal=False)
        md["trees"][0]["edges"] = np.empty((0, 2), dtype=int)
        fig = plot_single_morphology_2d(md)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Tests: plot_population_overlay
# ---------------------------------------------------------------------------


class TestPlotPopulationOverlay:
    def test_returns_figure(self, all_morph: dict) -> None:
        fig = plot_population_overlay(all_morph)
        assert isinstance(fig, go.Figure)

    def test_empty_input(self) -> None:
        fig = plot_population_overlay({})
        assert isinstance(fig, go.Figure)
        # Should still have the soma origin marker
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# Tests: plot_density_heatmap
# ---------------------------------------------------------------------------


class TestPlotDensityHeatmap:
    def test_returns_figure(self, all_morph: dict) -> None:
        fig = plot_density_heatmap(all_morph, compartment="apical")
        assert isinstance(fig, go.Figure)

    def test_no_matching_compartment(self, all_morph: dict) -> None:
        """Requesting a compartment that doesn't exist yields an empty figure."""
        fig = plot_density_heatmap(all_morph, compartment="axon")
        assert isinstance(fig, go.Figure)
        # No histogram trace should be added
        assert len(fig.data) == 0


# ---------------------------------------------------------------------------
# Tests: Sholl
# ---------------------------------------------------------------------------


class TestShollProfile:
    def test_returns_figure(self, morph_data: dict) -> None:
        fig = plot_sholl_profile(morph_data, title="Test Sholl")
        assert isinstance(fig, go.Figure)

    def test_empty_sholl(self) -> None:
        """Both compartments with empty Sholl arrays should still produce a figure."""
        md = _make_morph_data(include_apical=False, include_basal=False)
        fig = plot_sholl_profile(md)
        assert isinstance(fig, go.Figure)
        # No data traces (only layout)
        assert len(fig.data) == 0


class TestComputePopulationSholl:
    def test_returns_correct_shapes(self, all_morph: dict) -> None:
        radii, mean_p, sem_p = compute_population_sholl(all_morph, "apical")
        assert len(radii) == len(mean_p) == len(sem_p)
        assert len(radii) > 0

    def test_mean_is_reasonable(self, all_morph: dict) -> None:
        radii, mean_p, sem_p = compute_population_sholl(all_morph, "apical")
        assert np.all(np.isfinite(mean_p))
        assert np.all(sem_p >= 0)

    def test_empty_input(self) -> None:
        radii, mean_p, sem_p = compute_population_sholl({}, "apical")
        assert len(radii) == 0

    def test_unequal_lengths_padded(self) -> None:
        """Profiles of different lengths should be zero-padded."""
        md1 = _make_morph_data(sholl_len=5, seed=0)
        md2 = _make_morph_data(sholl_len=15, seed=1)
        radii, mean_p, _ = compute_population_sholl({"a": md1, "b": md2}, "apical")
        assert len(radii) == 15


class TestPlotPopulationSholl:
    def test_returns_figure(self, all_morph: dict) -> None:
        fig = plot_population_sholl(all_morph, compartment="apical")
        assert isinstance(fig, go.Figure)

    def test_empty_input(self) -> None:
        fig = plot_population_sholl({}, compartment="basal")
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Tests: metrics DataFrame and comparison plot
# ---------------------------------------------------------------------------


class TestBuildMetricsDataframe:
    def test_returns_dataframe(self, all_morph: dict) -> None:
        df = build_metrics_dataframe(all_morph)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"cell_id", "compartment", "metric", "value", "cell_type"}

    def test_cell_types_assigned(self, all_morph: dict) -> None:
        ct = {"001-CAA-111-S1-1": "penkpos", "002-CAA-222-S1-1": "penkneg"}
        df = build_metrics_dataframe(all_morph, cell_types=ct)
        row = df[df["cell_id"] == "001-CAA-111-S1-1"].iloc[0]
        assert row["cell_type"] == "penkpos"
        # Missing cell_type defaults to "unknown"
        row3 = df[df["cell_id"] == "003-CAA-333-S1-1"].iloc[0]
        assert row3["cell_type"] == "unknown"

    def test_empty_gstats(self) -> None:
        md = _make_morph_data(include_apical=False, include_basal=False)
        df = build_metrics_dataframe({"x": md})
        assert len(df) == 0


class TestPlotMetricComparison:
    def test_returns_figure(self, all_morph: dict) -> None:
        ct = {
            "001-CAA-111-S1-1": "penkpos",
            "002-CAA-222-S1-1": "penkneg",
            "003-CAA-333-S1-1": "penkpos",
        }
        df = build_metrics_dataframe(all_morph, cell_types=ct)
        fig = plot_metric_comparison(df, "len", compartment="apical")
        assert isinstance(fig, go.Figure)

    def test_unknown_metric(self, all_morph: dict) -> None:
        """A metric not in GSTATS_LABELS should not crash — just empty."""
        df = build_metrics_dataframe(all_morph)
        fig = plot_metric_comparison(df, "nonexistent_metric")
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Tests: format_stats_table
# ---------------------------------------------------------------------------


class TestFormatStatsTable:
    def test_returns_dataframe(self, morph_data: dict) -> None:
        df = format_stats_table(morph_data)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["Metric", "Apical", "Basal", "Unit"]
        assert len(df) == len(GSTATS_LABELS)

    def test_missing_keys_show_dash(self) -> None:
        """When gstats is missing a key, the table should show '-'."""
        md = _make_morph_data()
        md["apical_gstats"] = {"len": 100.0}  # only one key
        md["basal_gstats"] = {}
        df = format_stats_table(md)
        # Basal column should be all dashes
        basal_vals = df["Basal"].tolist()
        assert all(v == "-" for v in basal_vals)
        # Apical should have 100.0 for 'len' and '-' for others
        len_row = df[df["Metric"] == "Total length"]
        assert len_row["Apical"].iloc[0] == 100.0

    def test_nan_values_show_dash(self) -> None:
        md = _make_morph_data()
        md["apical_gstats"]["len"] = np.nan
        df = format_stats_table(md)
        len_row = df[df["Metric"] == "Total length"]
        assert len_row["Apical"].iloc[0] == "-"


# ---------------------------------------------------------------------------
# Tests: _hex_to_rgba helper
# ---------------------------------------------------------------------------


class TestHexToRgba:
    def test_basic_conversion(self) -> None:
        assert _hex_to_rgba("#1f77b4", 0.5) == "rgba(31,119,180,0.5)"

    def test_full_opacity(self) -> None:
        assert _hex_to_rgba("#000000") == "rgba(0,0,0,1.0)"

    def test_white(self) -> None:
        assert _hex_to_rgba("#ffffff", 0.0) == "rgba(255,255,255,0.0)"
