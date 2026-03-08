"""Tests for analysis/save.py — save and load analysis results to/from HDF5."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from hm2p.analysis.run import AnalysisParams, CellResult
from hm2p.analysis.save import load_analysis_results, save_analysis_results


def _make_cell_result(roi_idx: int, n_bins: int = 36) -> CellResult:
    """Create a synthetic CellResult with realistic-looking data."""
    rng = np.random.default_rng(roi_idx)
    centers = np.linspace(5, 355, n_bins)

    def _make_hd(sig: bool = True):
        tc = rng.random(n_bins).astype(np.float64)
        return {
            "tuning_curve": tc,
            "bin_centers": centers,
            "mvl": float(rng.random() * 0.5),
            "preferred_direction": float(rng.random() * 360),
            "tuning_width": float(rng.random() * 90 + 30),
            "p_value": float(rng.random() * 0.1 if sig else rng.random()),
            "significant": sig,
        }

    def _make_place():
        return {
            "rate_map": rng.random((10, 10)),
            "occupancy_map": rng.random((10, 10)),
            "spatial_info": float(rng.random()),
            "spatial_coherence": float(rng.random()),
            "sparsity": float(rng.random()),
            "p_value": float(rng.random()),
            "significant": bool(rng.random() > 0.5),
        }

    return CellResult(
        roi_idx=roi_idx,
        activity={
            "moving_light_event_rate": float(rng.random()),
            "moving_dark_event_rate": float(rng.random()),
            "stationary_light_event_rate": float(rng.random()),
            "stationary_dark_event_rate": float(rng.random()),
            "moving_light_mean_signal": float(rng.random()),
            "moving_dark_mean_signal": float(rng.random()),
            "stationary_light_mean_signal": float(rng.random()),
            "stationary_dark_mean_signal": float(rng.random()),
            "moving_light_mean_amplitude": float(rng.random()),
            "moving_dark_mean_amplitude": float(rng.random()),
            "stationary_light_mean_amplitude": float(rng.random()),
            "stationary_dark_mean_amplitude": float(rng.random()),
            "movement_modulation": float(rng.random() * 2 - 1),
            "light_modulation": float(rng.random() * 2 - 1),
        },
        hd_all=_make_hd(True),
        hd_light=_make_hd(True),
        hd_dark=_make_hd(False),
        place_all=_make_place(),
        place_light=_make_place(),
        place_dark=_make_place(),
        hd_comparison={
            "correlation": float(rng.random()),
            "pd_shift": float(rng.random() * 30 - 15),
            "mvl_ratio_dark_over_light": float(rng.random() * 2),
        },
        place_comparison={
            "correlation": float(rng.random()),
        },
    )


class TestSaveLoad:
    """Test round-trip save and load of analysis results."""

    def test_save_creates_file(self, tmp_path: Path):
        out = tmp_path / "analysis.h5"
        results = [_make_cell_result(i) for i in range(5)]
        params = AnalysisParams()

        save_analysis_results(
            out,
            {"dff": results},
            params,
            session_id="test_session",
            n_rois=5,
            n_frames=1000,
            fps=30.0,
            signal_types_available=["dff"],
        )

        assert out.exists()
        assert out.stat().st_size > 0

    def test_round_trip(self, tmp_path: Path):
        out = tmp_path / "analysis.h5"
        n_rois = 3
        results_dff = [_make_cell_result(i) for i in range(n_rois)]
        results_events = [_make_cell_result(i + 100) for i in range(n_rois)]
        params = AnalysisParams(n_shuffles=200, hd_n_bins=36)

        save_analysis_results(
            out,
            {"dff": results_dff, "events": results_events},
            params,
            session_id="test_session",
            n_rois=n_rois,
            n_frames=500,
            fps=9.8,
            signal_types_available=["dff", "events"],
        )

        loaded = load_analysis_results(out)

        # Check metadata
        assert loaded["meta"]["session_id"] == "test_session"
        assert loaded["meta"]["n_rois"] == n_rois
        assert loaded["meta"]["fps"] == 9.8

        # Check params
        assert loaded["params"]["n_shuffles"] == 200
        assert loaded["params"]["hd_n_bins"] == 36

        # Check signal type data exists
        assert "dff" in loaded
        assert "events" in loaded

        # Check HD data
        dff_data = loaded["dff"]
        assert "hd" in dff_data
        assert "all" in dff_data["hd"]
        assert "mvl" in dff_data["hd"]["all"]
        assert len(dff_data["hd"]["all"]["mvl"]) == n_rois

        # Check activity data
        assert "activity" in dff_data
        assert "moving_light_event_rate" in dff_data["activity"]
        assert len(dff_data["activity"]["moving_light_event_rate"]) == n_rois

        # Check HD comparison
        assert "comparison" in dff_data["hd"]
        assert "correlation" in dff_data["hd"]["comparison"]
        assert len(dff_data["hd"]["comparison"]["correlation"]) == n_rois

    def test_empty_conditions(self, tmp_path: Path):
        """Test saving results where some conditions have no data."""
        out = tmp_path / "analysis.h5"
        result = CellResult(roi_idx=0, activity={}, hd_all={}, hd_light={}, hd_dark={})
        params = AnalysisParams()

        save_analysis_results(
            out,
            {"dff": [result]},
            params,
            session_id="empty_test",
            n_rois=1,
            n_frames=100,
            fps=30.0,
            signal_types_available=["dff"],
        )

        loaded = load_analysis_results(out)
        assert loaded["meta"]["n_rois"] == 1
        assert "dff" in loaded

    def test_load_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_analysis_results(tmp_path / "nonexistent.h5")

    def test_multiple_signal_types(self, tmp_path: Path):
        """Test saving with all three signal types."""
        out = tmp_path / "analysis.h5"
        n = 4
        params = AnalysisParams()

        save_analysis_results(
            out,
            {
                "dff": [_make_cell_result(i) for i in range(n)],
                "deconv": [_make_cell_result(i + 10) for i in range(n)],
                "events": [_make_cell_result(i + 20) for i in range(n)],
            },
            params,
            session_id="multi_test",
            n_rois=n,
            n_frames=1000,
            fps=30.0,
            signal_types_available=["dff", "deconv", "events"],
        )

        loaded = load_analysis_results(out)
        assert "dff" in loaded
        assert "deconv" in loaded
        assert "events" in loaded

        # Each should have full structure
        for sig in ["dff", "deconv", "events"]:
            assert "hd" in loaded[sig]
            assert "place" in loaded[sig]
            assert "activity" in loaded[sig]
