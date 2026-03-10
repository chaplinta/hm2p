"""Tests for hm2p.patching.metrics — cell metrics assembly and table building."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hm2p.patching.metrics import (
    ALL_METRIC_COLS,
    build_cell_metrics,
    build_metrics_table,
    compute_derived_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cell_info(**overrides: object) -> dict:
    defaults = {
        "cell_index": 1,
        "animal_id": "M001",
        "slice_id": "S1",
        "cell_slice_id": "1",
        "hemisphere": "R",
        "cell_type": "penkpos",
        "depth_slice": 200.0,
        "depth_pial": 150.0,
        "area": "RSP",
        "layer": "L5",
    }
    defaults.update(overrides)
    return defaults


def _make_ephys_data(**overrides: object) -> dict:
    defaults = {
        "passive": {
            "RMP": -65.0,
            "rin": 200.0,
            "tau": 20.0,
            "sag": 10.0,
        },
        "active": {
            "minVm": -70.0,
            "peakVm": 30.0,
            "maxVmSlope": 150.0,
            "halfVm": -20.0,
            "amplitude": 100.0,
            "maxAHP": -55.0,
            "halfWidth": 0.8,
        },
        "rheobase": 100.0,
        "max_spike_rate": 40.0,
    }
    defaults.update(overrides)
    return defaults


def _make_morph_data(**overrides: object) -> dict:
    defaults = {
        "apical_stats": {
            "total_length": 500.0,
            "max_path_length": 200.0,
            "n_branch_points": 5,
            "mean_path_eucl_ratio": 1.3,
            "max_branch_order": 3,
            "mean_branch_length": 50.0,
            "mean_path_length": 100.0,
            "mean_branch_order": 1.5,
            "width": 300.0,
            "height": 400.0,
            "depth": 50.0,
            "width_height_ratio": 0.75,
            "width_depth_ratio": 6.0,
        },
        "basal_stats": {
            "total_length": 300.0,
            "max_path_length": 150.0,
            "n_branch_points": 3,
            "mean_path_eucl_ratio": 1.2,
            "max_branch_order": 2,
            "mean_branch_length": 40.0,
            "mean_path_length": 80.0,
            "mean_branch_order": 1.0,
            "width": 200.0,
            "height": 150.0,
            "depth": 30.0,
            "width_height_ratio": 1.33,
            "width_depth_ratio": 6.67,
        },
        "apical_sholl": {"peak_crossings": 8, "peak_distance": 120.0},
        "basal_sholl": {"peak_crossings": 5, "peak_distance": 80.0},
        "apical_surface_dist": {"dist_superficial": 10.0, "dist_deep": 200.0},
        "basal_surface_dist": {"dist_superficial": 15.0, "dist_deep": 100.0},
        "n_basal_trees": 3,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Tests: build_cell_metrics
# ---------------------------------------------------------------------------


class TestBuildCellMetrics:
    def test_full_data(self) -> None:
        m = build_cell_metrics(
            ephys_data=_make_ephys_data(),
            morph_data=_make_morph_data(),
            cell_info=_make_cell_info(),
        )
        # Metadata
        assert m["cell_index"] == 1
        assert m["cell_type"] == "penkpos"
        # Passive ephys
        assert m["ephys_passive_RMP"] == -65.0
        assert m["ephys_passive_rin"] == 200.0
        assert m["ephys_passive_tau"] == 20.0
        assert m["ephys_passive_incap"] == pytest.approx(20.0 / 200.0)
        assert m["ephys_passive_sag"] == 10.0
        assert m["ephys_passive_rhreo"] == 100.0
        assert m["ephys_passive_maxsp"] == 40.0
        # Active ephys
        assert m["ephys_active_peakVm"] == 30.0
        assert m["ephys_active_halfWidth"] == 0.8
        # Apical morph
        assert m["morph_api_len"] == 500.0
        assert m["morph_api_bpoints"] == 5
        assert m["morph_api_shlpeakcr"] == 8
        assert m["morph_api_ext_super"] == 10.0
        # Basal morph
        assert m["morph_bas_len"] == 300.0
        assert m["morph_bas_ntrees"] == 3

    def test_no_ephys(self) -> None:
        m = build_cell_metrics(
            ephys_data=None,
            morph_data=_make_morph_data(),
            cell_info=_make_cell_info(),
        )
        assert math.isnan(m["ephys_passive_RMP"])
        assert math.isnan(m["ephys_active_peakVm"])
        # Morph should still be present
        assert m["morph_api_len"] == 500.0

    def test_no_morph(self) -> None:
        m = build_cell_metrics(
            ephys_data=_make_ephys_data(),
            morph_data=None,
            cell_info=_make_cell_info(),
        )
        assert math.isnan(m["morph_api_len"])
        assert math.isnan(m["morph_bas_ntrees"])
        # Ephys should still be present
        assert m["ephys_passive_RMP"] == -65.0

    def test_no_ephys_no_morph(self) -> None:
        m = build_cell_metrics(
            ephys_data=None, morph_data=None, cell_info=_make_cell_info()
        )
        for col in ALL_METRIC_COLS:
            if col in (
                "cell_index", "animal_id", "slice_id", "cell_slice_id",
                "hemisphere", "cell_type", "depth_slice", "depth_pial",
                "area", "layer",
            ):
                continue
            assert math.isnan(m[col]), f"{col} should be NaN"

    def test_incap_zero_rin(self) -> None:
        ephys = _make_ephys_data()
        ephys["passive"]["rin"] = 0.0
        m = build_cell_metrics(
            ephys_data=ephys, morph_data=None, cell_info=_make_cell_info()
        )
        assert math.isnan(m["ephys_passive_incap"])


# ---------------------------------------------------------------------------
# Tests: build_metrics_table
# ---------------------------------------------------------------------------


class TestBuildMetricsTable:
    def test_empty_list(self) -> None:
        df = build_metrics_table([])
        assert len(df) == 0
        assert list(df.columns) == ALL_METRIC_COLS

    def test_single_cell(self) -> None:
        m = build_cell_metrics(
            _make_ephys_data(), _make_morph_data(), _make_cell_info()
        )
        df = build_metrics_table([m])
        assert len(df) == 1
        assert df.iloc[0]["cell_index"] == 1

    def test_multiple_cells(self) -> None:
        cells = [
            build_cell_metrics(
                _make_ephys_data(), _make_morph_data(),
                _make_cell_info(cell_index=i, cell_type="penkpos" if i % 2 == 0 else "penkneg"),
            )
            for i in range(5)
        ]
        df = build_metrics_table(cells)
        assert len(df) == 5
        assert set(df["cell_type"].unique()) == {"penkpos", "penkneg"}

    def test_column_order(self) -> None:
        m = build_cell_metrics(
            _make_ephys_data(), _make_morph_data(), _make_cell_info()
        )
        df = build_metrics_table([m])
        # Standard columns should appear in the expected order
        standard_cols_in_df = [c for c in df.columns if c in ALL_METRIC_COLS]
        expected_order = [c for c in ALL_METRIC_COLS if c in standard_cols_in_df]
        assert standard_cols_in_df == expected_order


# ---------------------------------------------------------------------------
# Tests: compute_derived_metrics
# ---------------------------------------------------------------------------


class TestComputeDerivedMetrics:
    def test_incap_recomputed(self) -> None:
        cells = [
            build_cell_metrics(
                _make_ephys_data(), _make_morph_data(), _make_cell_info()
            )
        ]
        df = build_metrics_table(cells)
        # Manually set incap to something wrong
        df.loc[0, "ephys_passive_incap"] = 999.0
        result = compute_derived_metrics(df)
        expected = 20.0 / 200.0  # tau / rin
        assert result.loc[0, "ephys_passive_incap"] == pytest.approx(expected)

    def test_incap_nan_when_rin_zero(self) -> None:
        cells = [
            build_cell_metrics(
                _make_ephys_data(), _make_morph_data(), _make_cell_info()
            )
        ]
        df = build_metrics_table(cells)
        df.loc[0, "ephys_passive_rin"] = 0.0
        result = compute_derived_metrics(df)
        assert math.isnan(result.loc[0, "ephys_passive_incap"])

    def test_wh_ratio_recomputed(self) -> None:
        cells = [
            build_cell_metrics(
                _make_ephys_data(), _make_morph_data(), _make_cell_info()
            )
        ]
        df = build_metrics_table(cells)
        result = compute_derived_metrics(df)
        expected_wh = 300.0 / 400.0
        assert result.loc[0, "morph_api_wh"] == pytest.approx(expected_wh)

    def test_wd_ratio_zero_depth(self) -> None:
        cells = [
            build_cell_metrics(
                _make_ephys_data(), _make_morph_data(), _make_cell_info()
            )
        ]
        df = build_metrics_table(cells)
        df.loc[0, "morph_api_depth"] = 0.0
        result = compute_derived_metrics(df)
        assert result.loc[0, "morph_api_wd"] == 0.0

    def test_does_not_mutate_input(self) -> None:
        cells = [
            build_cell_metrics(
                _make_ephys_data(), _make_morph_data(), _make_cell_info()
            )
        ]
        df = build_metrics_table(cells)
        original_val = df.loc[0, "ephys_passive_incap"]
        df.loc[0, "ephys_passive_incap"] = 999.0
        _ = compute_derived_metrics(df)
        # Original df should still have 999
        assert df.loc[0, "ephys_passive_incap"] == 999.0
