"""Tests for hm2p.patching.run — pipeline orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from hm2p.patching.config import PatchConfig
from hm2p.patching.run import (
    _build_cell_info,
    _build_ephys_data,
    _build_morph_data,
    _extract_active_features,
    load_metadata,
    process_cell,
    run_pca_analysis,
    run_pipeline,
    run_statistics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_config(tmp_path: Path) -> PatchConfig:
    """Return a PatchConfig pointing at tmp_path subdirectories."""
    for subdir in ("metadata", "ephys", "morph", "processed", "analysis"):
        (tmp_path / subdir).mkdir()
    return PatchConfig(
        metadata_dir=tmp_path / "metadata",
        ephys_dir=tmp_path / "ephys",
        morph_dir=tmp_path / "morph",
        processed_dir=tmp_path / "processed",
        analysis_dir=tmp_path / "analysis",
    )


@pytest.fixture()
def animals_csv(tmp_config: PatchConfig) -> Path:
    """Write a minimal animals.csv and return its path."""
    df = pd.DataFrame(
        {
            "animal_id": ["A001", "A002"],
            "cell_type": ["penk", "nonpenk"],
        }
    )
    p = tmp_config.metadata_dir / "animals.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture()
def cells_csv(tmp_config: PatchConfig) -> Path:
    """Write a minimal cells.csv and return its path."""
    df = pd.DataFrame(
        {
            "cell_index": [1, 2, 3],
            "animal_id": ["A001", "A001", "A002"],
            "ephys_id": ["e001", "", "e003"],
            "good_morph": [True, False, True],
            "morph_id": ["m001", "", "m003"],
            "slice_id": ["s1", "s1", "s2"],
            "cell_slice_id": ["cs1", "cs2", "cs3"],
            "hemisphere": ["L", "L", "R"],
            "depth_slice": [100, 200, 150],
            "depth_pial": [50, 80, 60],
            "area": ["RSP", "RSP", "RSP"],
            "layer": ["5", "5", "2/3"],
        }
    )
    p = tmp_config.metadata_dir / "cells.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# load_metadata
# ---------------------------------------------------------------------------


class TestLoadMetadata:
    def test_merges_on_animal_id(
        self, tmp_config: PatchConfig, animals_csv: Path, cells_csv: Path
    ):
        df = load_metadata(tmp_config)
        assert len(df) == 3
        # cell_type should come from animals merge
        assert "cell_type" in df.columns
        assert df.loc[df["animal_id"] == "A001", "cell_type"].iloc[0] == "penk"

    def test_missing_animals_csv(self, tmp_config: PatchConfig, cells_csv: Path):
        with pytest.raises(FileNotFoundError, match="animals.csv"):
            load_metadata(tmp_config)

    def test_missing_cells_csv(self, tmp_config: PatchConfig, animals_csv: Path):
        with pytest.raises(FileNotFoundError, match="cells.csv"):
            load_metadata(tmp_config)

    def test_empty_csvs(self, tmp_config: PatchConfig):
        pd.DataFrame({"animal_id": []}).to_csv(
            tmp_config.metadata_dir / "animals.csv", index=False
        )
        pd.DataFrame({"animal_id": [], "cell_index": [], "ephys_id": []}).to_csv(
            tmp_config.metadata_dir / "cells.csv", index=False
        )
        df = load_metadata(tmp_config)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# _build_cell_info
# ---------------------------------------------------------------------------


class TestBuildCellInfo:
    def test_extracts_metadata_keys(self):
        row = pd.Series(
            {
                "cell_index": 1,
                "animal_id": "A001",
                "slice_id": "s1",
                "cell_slice_id": "cs1",
                "hemisphere": "L",
                "cell_type": "penk",
                "depth_slice": 100,
                "depth_pial": 50,
                "area": "RSP",
                "layer": "5",
                "extra_col": "ignored",
            }
        )
        info = _build_cell_info(row)
        assert info["cell_index"] == 1
        assert info["animal_id"] == "A001"
        assert "extra_col" not in info

    def test_missing_keys_become_nan(self):
        row = pd.Series({"cell_index": 1})
        info = _build_cell_info(row)
        assert np.isnan(info["animal_id"])


# ---------------------------------------------------------------------------
# _build_ephys_data
# ---------------------------------------------------------------------------


class TestBuildEphysData:
    def test_with_all_protocols(self):
        iv_result = MagicMock()
        iv_result.rmp = -65.0
        iv_result.spike_counts = np.array([0, 1, 3, 5])

        passive_result = MagicMock()
        passive_result.rmp = -64.0
        passive_result.rin = 200.0
        passive_result.tau = np.array([15.0, 16.0, np.nan])

        rheo_result = MagicMock()
        rheo_result.rheo_current = 0.05

        sag_result = MagicMock()
        sag_result.sag_ratio = np.array([10.0, 12.0])

        protocols = {
            "iv": iv_result,
            "passive": passive_result,
            "rheobase": rheo_result,
            "sag": sag_result,
        }

        data = _build_ephys_data(protocols)
        assert data["passive"]["RMP"] == -64.0
        assert data["passive"]["rin"] == 200.0
        assert data["rheobase"] == 0.05
        assert data["max_spike_rate"] == 5.0
        assert data["passive"]["sag"] == pytest.approx(11.0)

    def test_iv_only(self):
        iv_result = MagicMock()
        iv_result.rmp = -65.0
        iv_result.spike_counts = np.array([0, 2])

        data = _build_ephys_data({"iv": iv_result})
        assert data["passive"]["RMP"] == -65.0
        assert np.isnan(data["rheobase"])
        assert data["max_spike_rate"] == 2.0

    def test_empty_protocols(self):
        data = _build_ephys_data({})
        assert np.isnan(data["rheobase"])
        assert np.isnan(data["max_spike_rate"])


# ---------------------------------------------------------------------------
# _extract_active_features
# ---------------------------------------------------------------------------


class TestExtractActiveFeatures:
    def test_no_rheobase_returns_none(self):
        assert _extract_active_features({}) is None

    def test_no_spikes_returns_none(self):
        rheo = MagicMock()
        rheo.spike_counts = np.array([0, 0, 0])
        assert _extract_active_features({"rheobase": rheo}) is None

    @patch("hm2p.patching.run.extract_spike_features")
    def test_returns_active_dict(self, mock_esf):
        mock_esf.return_value = {
            "min_vm": -70.0,
            "peak_vm": 30.0,
            "max_vm_slope": 200.0,
            "half_vm": -10.0,
            "amplitude": 90.0,
            "max_ahp": -75.0,
            "half_width": 0.8,
            "spike_time": 100.0,
        }

        rheo = MagicMock()
        rheo.spike_counts = np.array([0, 2, 3])
        rheo.traces = np.random.randn(4000, 3)

        result = _extract_active_features({"rheobase": rheo})
        assert result is not None
        assert result["peakVm"] == 30.0
        assert result["halfWidth"] == 0.8

    @patch("hm2p.patching.run.extract_spike_features", return_value=None)
    def test_no_features_returns_none(self, mock_esf):
        rheo = MagicMock()
        rheo.spike_counts = np.array([1])
        rheo.traces = np.random.randn(4000, 1)
        assert _extract_active_features({"rheobase": rheo}) is None


# ---------------------------------------------------------------------------
# _build_morph_data
# ---------------------------------------------------------------------------


class TestBuildMorphData:
    def test_nonexistent_dir_returns_none(self, tmp_path: Path):
        result = _build_morph_data(tmp_path / "nonexistent")
        assert result is None

    @patch("hm2p.patching.run.load_morphology")
    def test_no_soma_returns_none(self, mock_load, tmp_path: Path):
        mock_load.return_value = {"apical": MagicMock()}
        result = _build_morph_data(tmp_path)
        assert result is None

    @patch("hm2p.patching.run.compute_surface_distance")
    @patch("hm2p.patching.run.compute_sholl")
    @patch("hm2p.patching.run.compute_tree_stats")
    @patch("hm2p.patching.run.soma_subtract")
    @patch("hm2p.patching.run.load_morphology")
    def test_with_soma_and_apical(
        self, mock_load, mock_sub, mock_stats, mock_sholl, mock_surf, tmp_path: Path
    ):
        nodes = pd.DataFrame(
            {
                "id": [1, 2],
                "type": [1, 3],
                "x": [0.0, 10.0],
                "y": [0.0, 5.0],
                "z": [0.0, 0.0],
                "radius": [1.0, 0.5],
                "parent_id": [-1, 1],
            }
        )
        edges = np.array([[1, 2]])
        tree = {"nodes": nodes, "edges": edges}

        mock_load.return_value = {"soma": tree, "apical": tree}
        mock_sub.return_value = {"soma": tree, "apical": tree}
        mock_stats.return_value = {"total_length": 100.0}
        mock_sholl.return_value = np.array([0, 3, 5, 2, 1])
        # No surface — surface_dist shouldn't be called

        result = _build_morph_data(tmp_path)
        assert result is not None
        assert result["apical_stats"]["total_length"] == 100.0
        assert result["apical_sholl"]["peak_crossings"] == 5
        assert result["n_basal_trees"] == 0  # no Basal*.swc files


# ---------------------------------------------------------------------------
# process_cell
# ---------------------------------------------------------------------------


class TestProcessCell:
    def test_no_ephys_no_morph_returns_none(self, tmp_config: PatchConfig):
        row = pd.Series(
            {
                "cell_index": 1,
                "ephys_id": "",
                "good_morph": False,
            }
        )
        assert process_cell(row, tmp_config) is None

    def test_nan_ephys_id_no_morph(self, tmp_config: PatchConfig):
        row = pd.Series(
            {
                "cell_index": 2,
                "ephys_id": np.nan,
                "good_morph": False,
            }
        )
        assert process_cell(row, tmp_config) is None

    @patch("hm2p.patching.run.process_all_protocols")
    def test_ephys_only(self, mock_protocols, tmp_config: PatchConfig):
        iv = MagicMock()
        iv.rmp = -65.0
        iv.spike_counts = np.array([0, 1])
        mock_protocols.return_value = {"iv": iv}

        row = pd.Series(
            {
                "cell_index": 3,
                "ephys_id": "e003",
                "good_morph": False,
                "animal_id": "A001",
                "slice_id": "s1",
                "cell_slice_id": "cs1",
                "hemisphere": "L",
                "cell_type": "penk",
                "depth_slice": 100,
                "depth_pial": 50,
                "area": "RSP",
                "layer": "5",
            }
        )
        result = process_cell(row, tmp_config)
        assert result is not None
        assert result["ephys_passive_RMP"] == -65.0

    @patch("hm2p.patching.run._build_morph_data")
    def test_morph_only(self, mock_morph, tmp_config: PatchConfig):
        mock_morph.return_value = {
            "apical_stats": {"total_length": 500.0},
            "basal_stats": {},
            "apical_sholl": {"peak_crossings": 10, "peak_distance": 100.0},
            "basal_sholl": {},
            "apical_surface_dist": {},
            "basal_surface_dist": {},
            "n_basal_trees": 0,
        }
        row = pd.Series(
            {
                "cell_index": 4,
                "ephys_id": "",
                "good_morph": True,
                "morph_id": "m004",
                "animal_id": "A002",
                "slice_id": "s2",
                "cell_slice_id": "cs4",
                "hemisphere": "R",
                "cell_type": "nonpenk",
                "depth_slice": 150,
                "depth_pial": 60,
                "area": "RSP",
                "layer": "2/3",
            }
        )
        result = process_cell(row, tmp_config)
        assert result is not None
        assert result["morph_api_shlpeakcr"] == 10

    @patch("hm2p.patching.run.process_all_protocols")
    def test_ephys_exception_doesnt_crash(
        self, mock_protocols, tmp_config: PatchConfig, caplog
    ):
        mock_protocols.side_effect = RuntimeError("disk failure")
        row = pd.Series(
            {
                "cell_index": 5,
                "ephys_id": "e005",
                "good_morph": False,
                "animal_id": "A001",
            }
        )
        with caplog.at_level(logging.ERROR):
            result = process_cell(row, tmp_config)
        # No data extracted, returns None
        assert result is None

    @patch("hm2p.patching.run._build_morph_data")
    def test_morph_exception_doesnt_crash(
        self, mock_morph, tmp_config: PatchConfig, caplog
    ):
        mock_morph.side_effect = ValueError("corrupt SWC")
        row = pd.Series(
            {
                "cell_index": 6,
                "ephys_id": "",
                "good_morph": True,
                "morph_id": "m006",
                "animal_id": "A001",
            }
        )
        with caplog.at_level(logging.ERROR):
            result = process_cell(row, tmp_config)
        assert result is None


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:
    @patch("hm2p.patching.run.process_cell")
    @patch("hm2p.patching.run.load_metadata")
    def test_full_orchestration(
        self, mock_meta, mock_process, tmp_config: PatchConfig
    ):
        mock_meta.return_value = pd.DataFrame(
            {
                "cell_index": [1, 2],
                "animal_id": ["A001", "A002"],
            }
        )
        # First cell returns metrics, second returns None
        mock_process.side_effect = [
            {"cell_index": 1, "animal_id": "A001", "ephys_passive_RMP": -65.0},
            None,
        ]

        df = run_pipeline(tmp_config)
        assert len(df) == 1
        assert df.iloc[0]["ephys_passive_RMP"] == -65.0
        assert (tmp_config.analysis_dir / "metrics.csv").exists()

    @patch("hm2p.patching.run.process_cell")
    @patch("hm2p.patching.run.load_metadata")
    def test_empty_metadata(self, mock_meta, mock_process, tmp_config: PatchConfig):
        mock_meta.return_value = pd.DataFrame(columns=["cell_index", "animal_id"])
        df = run_pipeline(tmp_config)
        assert len(df) == 0
        mock_process.assert_not_called()

    @patch("hm2p.patching.run.process_cell")
    @patch("hm2p.patching.run.load_metadata")
    def test_exception_in_one_cell_doesnt_crash(
        self, mock_meta, mock_process, tmp_config: PatchConfig
    ):
        mock_meta.return_value = pd.DataFrame(
            {
                "cell_index": [1, 2, 3],
                "animal_id": ["A001", "A002", "A003"],
            }
        )
        mock_process.side_effect = [
            {"cell_index": 1, "ephys_passive_RMP": -65.0},
            RuntimeError("boom"),
            {"cell_index": 3, "ephys_passive_RMP": -70.0},
        ]

        df = run_pipeline(tmp_config)
        # Cell 2 failed but 1 and 3 should be in the table
        assert len(df) == 2

    @patch("hm2p.patching.run.process_cell")
    @patch("hm2p.patching.run.load_metadata")
    def test_all_cells_return_none(
        self, mock_meta, mock_process, tmp_config: PatchConfig
    ):
        mock_meta.return_value = pd.DataFrame(
            {"cell_index": [1, 2], "animal_id": ["A001", "A002"]}
        )
        mock_process.return_value = None
        df = run_pipeline(tmp_config)
        assert len(df) == 0

    @patch("hm2p.patching.run.process_cell")
    @patch("hm2p.patching.run.load_metadata")
    def test_saves_csv(self, mock_meta, mock_process, tmp_config: PatchConfig):
        mock_meta.return_value = pd.DataFrame(
            {"cell_index": [1], "animal_id": ["A001"]}
        )
        mock_process.return_value = {
            "cell_index": 1,
            "animal_id": "A001",
            "ephys_passive_RMP": -65.0,
        }
        run_pipeline(tmp_config)
        csv_path = tmp_config.analysis_dir / "metrics.csv"
        assert csv_path.exists()
        saved = pd.read_csv(csv_path)
        assert len(saved) == 1


# ---------------------------------------------------------------------------
# run_statistics
# ---------------------------------------------------------------------------


class TestRunStatistics:
    @patch("hm2p.patching.statistics", create=True)
    def test_saves_both_files(self, mock_stats_module, tmp_config: PatchConfig):
        mock_summary = pd.DataFrame({"metric": ["RMP"], "mean": [-65.0]})
        mock_mw = pd.DataFrame({"metric": ["RMP"], "pvalue": [0.05]})
        mock_stats_module.compute_summary_stats = MagicMock(return_value=mock_summary)
        mock_stats_module.compute_mannwhitney = MagicMock(return_value=mock_mw)

        metrics_df = pd.DataFrame({"ephys_passive_RMP": [-65.0, -70.0]})

        with patch.dict(
            "sys.modules",
            {"hm2p.patching.statistics": mock_stats_module},
        ):
            run_statistics(metrics_df, tmp_config)

        assert (tmp_config.analysis_dir / "summary_stats.csv").exists()
        assert (tmp_config.analysis_dir / "mannwhitney.csv").exists()


# ---------------------------------------------------------------------------
# run_pca_analysis
# ---------------------------------------------------------------------------


class TestRunPcaAnalysis:
    @patch("hm2p.patching.pca", create=True)
    def test_runs_all_subsets(self, mock_pca_module, tmp_config: PatchConfig):
        mock_result = pd.DataFrame({"PC1": [0.1], "PC2": [0.2]})
        mock_pca_module.run_pca = MagicMock(return_value=mock_result)

        metrics_df = pd.DataFrame({"ephys_passive_RMP": [-65.0]})

        with patch.dict(
            "sys.modules",
            {"hm2p.patching.pca": mock_pca_module},
        ):
            run_pca_analysis(metrics_df, tmp_config)

        pca_dir = tmp_config.analysis_dir / "pca"
        assert (pca_dir / "pca_ephys.csv").exists()
        assert (pca_dir / "pca_morph.csv").exists()
        assert (pca_dir / "pca_all.csv").exists()
        assert mock_pca_module.run_pca.call_count == 3

    @patch("hm2p.patching.pca", create=True)
    def test_one_subset_failure_doesnt_stop_others(
        self, mock_pca_module, tmp_config: PatchConfig
    ):
        ok_result = pd.DataFrame({"PC1": [0.1]})

        def side_effect(df, subset):
            if subset == "morph":
                raise ValueError("not enough data")
            return ok_result

        mock_pca_module.run_pca = MagicMock(side_effect=side_effect)
        metrics_df = pd.DataFrame({"ephys_passive_RMP": [-65.0]})

        with patch.dict(
            "sys.modules",
            {"hm2p.patching.pca": mock_pca_module},
        ):
            run_pca_analysis(metrics_df, tmp_config)

        pca_dir = tmp_config.analysis_dir / "pca"
        assert (pca_dir / "pca_ephys.csv").exists()
        assert not (pca_dir / "pca_morph.csv").exists()  # failed
        assert (pca_dir / "pca_all.csv").exists()
