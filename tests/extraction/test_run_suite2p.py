"""Tests for extraction/run_suite2p.py — Suite2p execution wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hm2p.extraction.run_suite2p import default_ops, run_suite2p

# ---------------------------------------------------------------------------
# default_ops
# ---------------------------------------------------------------------------


class TestDefaultOps:
    def test_returns_dict(self):
        ops = default_ops()
        assert isinstance(ops, dict)

    def test_fs_matches_arg(self):
        ops = default_ops(fps=15.0)
        assert ops["fs"] == 15.0

    def test_default_fs(self):
        ops = default_ops()
        assert ops["fs"] == 29.97

    def test_single_plane(self):
        ops = default_ops()
        assert ops["nplanes"] == 1

    def test_spike_detect_off(self):
        ops = default_ops()
        assert ops["spikedetect"] is False


# ---------------------------------------------------------------------------
# run_suite2p
# ---------------------------------------------------------------------------


class TestRunSuite2p:
    def test_missing_tiff_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="TIFF directory"):
            run_suite2p(tmp_path / "nonexistent", tmp_path / "output")

    def test_empty_tiff_dir_raises(self, tmp_path):
        tiff_dir = tmp_path / "tiffs"
        tiff_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No TIFF"):
            run_suite2p(tiff_dir, tmp_path / "output")

    def test_importerror_without_suite2p(self, tmp_path):
        tiff_dir = tmp_path / "tiffs"
        tiff_dir.mkdir()
        (tiff_dir / "data_XYT.tif").write_bytes(b"\x00")

        with (
            patch.dict("sys.modules", {"suite2p": None}),
            pytest.raises(ImportError, match="suite2p"),
        ):
            run_suite2p(tiff_dir, tmp_path / "output")

    def test_successful_run_with_mock(self, tmp_path):
        tiff_dir = tmp_path / "tiffs"
        tiff_dir.mkdir()
        (tiff_dir / "data_XYT.tif").write_bytes(b"\x00")
        output_dir = tmp_path / "output"

        mock_suite2p = MagicMock()

        def fake_run_s2p(ops):
            """Simulate Suite2p creating plane0/ with required files."""
            s2p_dir = Path(ops["save_path0"]) / "suite2p" / "plane0"
            s2p_dir.mkdir(parents=True)
            np.save(s2p_dir / "F.npy", np.zeros((5, 100)))
            np.save(s2p_dir / "Fneu.npy", np.zeros((5, 100)))
            np.save(s2p_dir / "iscell.npy", np.ones((5, 2)))
            np.save(
                s2p_dir / "stat.npy",
                np.array([{"ypix": np.array([0]), "xpix": np.array([0])}] * 5, dtype=object),
                allow_pickle=True,
            )
            np.save(s2p_dir / "ops.npy", {"fs": 29.97, "Ly": 64, "Lx": 64})
            return ops

        mock_suite2p.run_s2p = fake_run_s2p

        with patch.dict("sys.modules", {"suite2p": mock_suite2p}):
            result = run_suite2p(tiff_dir, output_dir)

        assert result == output_dir / "suite2p"
        assert (result / "plane0" / "F.npy").exists()
        assert (result / "plane0" / "Fneu.npy").exists()
        assert (result / "plane0" / "iscell.npy").exists()

    def test_ops_overrides(self, tmp_path):
        tiff_dir = tmp_path / "tiffs"
        tiff_dir.mkdir()
        (tiff_dir / "image.tif").write_bytes(b"\x00")
        output_dir = tmp_path / "output"

        mock_suite2p = MagicMock()
        captured_ops = {}

        def fake_run_s2p(ops):
            captured_ops.update(ops)
            s2p_dir = Path(ops["save_path0"]) / "suite2p" / "plane0"
            s2p_dir.mkdir(parents=True)
            for name in ("F.npy", "Fneu.npy", "iscell.npy", "stat.npy", "ops.npy"):
                np.save(s2p_dir / name, np.zeros(1))
            return ops

        mock_suite2p.run_s2p = fake_run_s2p

        with patch.dict("sys.modules", {"suite2p": mock_suite2p}):
            run_suite2p(tiff_dir, output_dir, ops_overrides={"tau": 2.0, "batch_size": 200})

        assert captured_ops["tau"] == 2.0
        assert captured_ops["batch_size"] == 200
        assert captured_ops["fs"] == 29.97  # default preserved

    def test_missing_plane0_raises_runtime(self, tmp_path):
        tiff_dir = tmp_path / "tiffs"
        tiff_dir.mkdir()
        (tiff_dir / "data.tif").write_bytes(b"\x00")
        output_dir = tmp_path / "output"

        mock_suite2p = MagicMock()
        mock_suite2p.run_s2p.return_value = {}  # doesn't create plane0

        with (
            patch.dict("sys.modules", {"suite2p": mock_suite2p}),
            pytest.raises(RuntimeError, match="plane0"),
        ):
            run_suite2p(tiff_dir, output_dir)

    def test_missing_output_file_raises_runtime(self, tmp_path):
        tiff_dir = tmp_path / "tiffs"
        tiff_dir.mkdir()
        (tiff_dir / "data.tif").write_bytes(b"\x00")
        output_dir = tmp_path / "output"

        mock_suite2p = MagicMock()

        def fake_run_s2p(ops):
            # Create plane0 but only some files
            s2p_dir = Path(ops["save_path0"]) / "suite2p" / "plane0"
            s2p_dir.mkdir(parents=True)
            np.save(s2p_dir / "F.npy", np.zeros(1))
            # Missing: Fneu.npy, iscell.npy, stat.npy, ops.npy
            return ops

        mock_suite2p.run_s2p = fake_run_s2p

        with (
            patch.dict("sys.modules", {"suite2p": mock_suite2p}),
            pytest.raises(RuntimeError, match="missing"),
        ):
            run_suite2p(tiff_dir, output_dir)

    def test_tiff_and_tiff_extension(self, tmp_path):
        tiff_dir = tmp_path / "tiffs"
        tiff_dir.mkdir()
        (tiff_dir / "image.tiff").write_bytes(b"\x00")  # .tiff not .tif
        output_dir = tmp_path / "output"

        mock_suite2p = MagicMock()

        def fake_run_s2p(ops):
            s2p_dir = Path(ops["save_path0"]) / "suite2p" / "plane0"
            s2p_dir.mkdir(parents=True)
            for name in ("F.npy", "Fneu.npy", "iscell.npy", "stat.npy", "ops.npy"):
                np.save(s2p_dir / name, np.zeros(1))
            return ops

        mock_suite2p.run_s2p = fake_run_s2p

        with patch.dict("sys.modules", {"suite2p": mock_suite2p}):
            result = run_suite2p(tiff_dir, output_dir)
        assert result.exists()
