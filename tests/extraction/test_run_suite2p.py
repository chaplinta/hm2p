"""Tests for extraction/run_suite2p.py — Suite2p execution wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hm2p.extraction.run_suite2p import (
    _deep_update,
    default_ops,
    default_settings,
    run_suite2p,
)

# ---------------------------------------------------------------------------
# default_settings / default_ops
# ---------------------------------------------------------------------------


_suite2p_available = False
try:
    import suite2p  # noqa: F401
    _suite2p_available = True
except ImportError:
    pass


class TestDefaultSettings:
    """Tests for the Suite2p 1.0 API default_settings."""

    def test_returns_dict(self):
        settings = default_settings()
        assert isinstance(settings, dict)

    def test_fs_matches_arg(self):
        settings = default_settings(fps=15.0)
        assert settings["fs"] == 15.0

    def test_default_fs(self):
        settings = default_settings()
        assert settings["fs"] == 29.97

    def test_tau_gcamg7f(self):
        settings = default_settings()
        assert settings["tau"] == 1.0

    @pytest.mark.skipif(not _suite2p_available, reason="suite2p not installed")
    def test_deconvolution_off(self):
        """CASCADE handles spikes — Suite2p deconvolution should be off."""
        settings = default_settings()
        assert settings["run"]["do_deconvolution"] is False

    @pytest.mark.skipif(not _suite2p_available, reason="suite2p not installed")
    def test_nonrigid_registration(self):
        settings = default_settings()
        assert settings["registration"]["nonrigid"] is True

    @pytest.mark.skipif(not _suite2p_available, reason="suite2p not installed")
    def test_delete_bin_true(self):
        settings = default_settings()
        assert settings["io"]["delete_bin"] is True


class TestDefaultOps:
    """Tests for the backward-compatible default_ops alias."""

    def test_returns_dict(self):
        ops = default_ops()
        assert isinstance(ops, dict)

    def test_fs_matches_arg(self):
        ops = default_ops(fps=15.0)
        assert ops["fs"] == 15.0

    def test_default_fs(self):
        ops = default_ops()
        assert ops["fs"] == 29.97


# ---------------------------------------------------------------------------
# _deep_update
# ---------------------------------------------------------------------------


class TestDeepUpdate:
    def test_flat_update(self):
        base = {"a": 1, "b": 2}
        result = _deep_update(base, {"b": 3, "c": 4})
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_update(self):
        base = {"x": {"y": 1, "z": 2}, "a": 10}
        result = _deep_update(base, {"x": {"z": 99}})
        assert result["x"]["y"] == 1
        assert result["x"]["z"] == 99
        assert result["a"] == 10

    def test_nested_override_with_non_dict(self):
        base = {"x": {"y": 1}}
        result = _deep_update(base, {"x": 42})
        assert result["x"] == 42


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
        """Mocked run_s2p creates plane0 with required files."""
        tiff_dir = tmp_path / "tiffs"
        tiff_dir.mkdir()
        (tiff_dir / "data_XYT.tif").write_bytes(b"\x00")
        output_dir = tmp_path / "output"

        mock_suite2p = MagicMock()

        def fake_run_s2p(db, settings):
            """Simulate Suite2p 1.0 creating plane0/ with required files."""
            s2p_dir = Path(db["save_path0"]) / "suite2p" / "plane0"
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

        mock_suite2p.run_s2p = fake_run_s2p

        with patch.dict("sys.modules", {"suite2p": mock_suite2p}):
            result = run_suite2p(tiff_dir, output_dir)

        assert result == output_dir / "suite2p"
        assert (result / "plane0" / "F.npy").exists()
        assert (result / "plane0" / "Fneu.npy").exists()
        assert (result / "plane0" / "iscell.npy").exists()

    def test_ops_overrides(self, tmp_path):
        """ops_overrides are deep-merged into settings."""
        tiff_dir = tmp_path / "tiffs"
        tiff_dir.mkdir()
        (tiff_dir / "image.tif").write_bytes(b"\x00")
        output_dir = tmp_path / "output"

        mock_suite2p = MagicMock()
        # default_settings returns a real nested dict so deep_update works
        mock_suite2p.default_settings.return_value = {
            "fs": 29.97, "tau": 1.0,
            "run": {"do_deconvolution": False},
            "io": {"delete_bin": True},
            "registration": {"nonrigid": True, "block_size": (128, 128),
                             "batch_size": 100, "maxregshift": 0.1,
                             "smooth_sigma": 1.15, "th_badframes": 1.0, "subpixel": 10},
            "detection": {"threshold_scaling": 1.0, "max_overlap": 0.75,
                          "sparsery_settings": {"highpass_neuropil": 25}},
            "extraction": {"batch_size": 500, "neuropil_extract": True,
                           "neuropil_coefficient": 0.7, "inner_neuropil_radius": 2,
                           "min_neuropil_pixels": 350, "allow_overlap": False},
            "classification": {"use_builtin_classifier": True},
        }
        captured = {}

        def fake_run_s2p(db, settings):
            captured["db"] = db
            captured["settings"] = settings
            s2p_dir = Path(db["save_path0"]) / "suite2p" / "plane0"
            s2p_dir.mkdir(parents=True)
            for name in ("F.npy", "Fneu.npy", "iscell.npy", "stat.npy", "ops.npy"):
                np.save(s2p_dir / name, np.zeros(1))

        mock_suite2p.run_s2p = fake_run_s2p

        with patch.dict("sys.modules", {"suite2p": mock_suite2p}):
            run_suite2p(tiff_dir, output_dir, ops_overrides={"tau": 2.0})

        assert captured["settings"]["tau"] == 2.0
        assert captured["settings"]["fs"] == 29.97  # default preserved

    def test_db_contains_paths(self, tmp_path):
        """db dict passed to run_s2p contains the right paths."""
        tiff_dir = tmp_path / "tiffs"
        tiff_dir.mkdir()
        (tiff_dir / "image.tif").write_bytes(b"\x00")
        output_dir = tmp_path / "output"

        mock_suite2p = MagicMock()
        captured = {}

        def fake_run_s2p(db, settings):
            captured["db"] = db
            s2p_dir = Path(db["save_path0"]) / "suite2p" / "plane0"
            s2p_dir.mkdir(parents=True)
            for name in ("F.npy", "Fneu.npy", "iscell.npy", "stat.npy", "ops.npy"):
                np.save(s2p_dir / name, np.zeros(1))

        mock_suite2p.run_s2p = fake_run_s2p

        with patch.dict("sys.modules", {"suite2p": mock_suite2p}):
            run_suite2p(tiff_dir, output_dir)

        assert str(tiff_dir) in captured["db"]["data_path"]
        assert captured["db"]["save_path0"] == str(output_dir)
        assert captured["db"]["nplanes"] == 1
        assert captured["db"]["nchannels"] == 1

    def test_missing_plane0_raises_runtime(self, tmp_path):
        tiff_dir = tmp_path / "tiffs"
        tiff_dir.mkdir()
        (tiff_dir / "data.tif").write_bytes(b"\x00")
        output_dir = tmp_path / "output"

        mock_suite2p = MagicMock()
        mock_suite2p.run_s2p.return_value = None  # doesn't create plane0

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

        def fake_run_s2p(db, settings):
            # Create plane0 but only some files
            s2p_dir = Path(db["save_path0"]) / "suite2p" / "plane0"
            s2p_dir.mkdir(parents=True)
            np.save(s2p_dir / "F.npy", np.zeros(1))
            # Missing: Fneu.npy, iscell.npy, stat.npy, ops.npy

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

        def fake_run_s2p(db, settings):
            s2p_dir = Path(db["save_path0"]) / "suite2p" / "plane0"
            s2p_dir.mkdir(parents=True)
            for name in ("F.npy", "Fneu.npy", "iscell.npy", "stat.npy", "ops.npy"):
                np.save(s2p_dir / name, np.zeros(1))

        mock_suite2p.run_s2p = fake_run_s2p

        with patch.dict("sys.modules", {"suite2p": mock_suite2p}):
            result = run_suite2p(tiff_dir, output_dir)
        assert result.exists()

    def test_custom_fps(self, tmp_path):
        """Custom fps parameter is reflected in settings."""
        tiff_dir = tmp_path / "tiffs"
        tiff_dir.mkdir()
        (tiff_dir / "data.tif").write_bytes(b"\x00")
        output_dir = tmp_path / "output"

        mock_suite2p = MagicMock()
        mock_suite2p.default_settings.return_value = {
            "fs": 29.97, "tau": 1.0,
            "run": {"do_deconvolution": False},
            "io": {"delete_bin": True},
            "registration": {"nonrigid": True, "block_size": (128, 128),
                             "batch_size": 100, "maxregshift": 0.1,
                             "smooth_sigma": 1.15, "th_badframes": 1.0, "subpixel": 10},
            "detection": {"threshold_scaling": 1.0, "max_overlap": 0.75,
                          "sparsery_settings": {"highpass_neuropil": 25}},
            "extraction": {"batch_size": 500, "neuropil_extract": True,
                           "neuropil_coefficient": 0.7, "inner_neuropil_radius": 2,
                           "min_neuropil_pixels": 350, "allow_overlap": False},
            "classification": {"use_builtin_classifier": True},
        }
        captured = {}

        def fake_run_s2p(db, settings):
            captured["settings"] = settings
            s2p_dir = Path(db["save_path0"]) / "suite2p" / "plane0"
            s2p_dir.mkdir(parents=True)
            for name in ("F.npy", "Fneu.npy", "iscell.npy", "stat.npy", "ops.npy"):
                np.save(s2p_dir / name, np.zeros(1))

        mock_suite2p.run_s2p = fake_run_s2p

        with patch.dict("sys.modules", {"suite2p": mock_suite2p}):
            run_suite2p(tiff_dir, output_dir, fps=15.0)

        assert captured["settings"]["fs"] == 15.0
