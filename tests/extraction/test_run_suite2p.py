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
    fps_from_timestamps,
    run_suite2p,
    tau_for_indicator,
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
        assert settings["fs"] == 9.6

    def test_tau_default(self):
        # Default tau is 1.0 s (_INDICATOR_TAU_DEFAULT).
        settings = default_settings()
        assert settings["tau"] == 1.0

    def test_tau_from_arg(self):
        settings = default_settings(fps=9.6, tau=0.4)
        assert settings["tau"] == 0.4

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
        assert ops["fs"] == 9.6


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
            np.save(s2p_dir / "ops.npy", {"fs": 9.6, "Ly": 64, "Lx": 64})

        mock_suite2p.run_s2p = fake_run_s2p

        with patch.dict("sys.modules", {"suite2p": mock_suite2p}):
            # Pass fps explicitly so the test does not depend on timestamps.h5.
            # anatomical_only=0 avoids importing cellpose (not under test here).
            result = run_suite2p(tiff_dir, output_dir, fps=9.6, anatomical_only=0)

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
            "fs": 29.97,
            "tau": 1.0,
            "run": {"do_deconvolution": False},
            "io": {"delete_bin": True},
            "registration": {
                "nonrigid": True,
                "block_size": (128, 128),
                "batch_size": 100,
                "maxregshift": 0.1,
                "smooth_sigma": 1.15,
                "th_badframes": 1.0,
                "subpixel": 10,
            },
            "detection": {
                "threshold_scaling": 1.0,
                "max_overlap": 0.75,
                "sparsery_settings": {"highpass_neuropil": 25},
            },
            "extraction": {
                "batch_size": 500,
                "neuropil_extract": True,
                "neuropil_coefficient": 0.7,
                "inner_neuropil_radius": 2,
                "min_neuropil_pixels": 350,
                "allow_overlap": False,
            },
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
            # anatomical_only=0 avoids importing cellpose (not under test here).
            run_suite2p(
                tiff_dir, output_dir, ops_overrides={"tau": 2.0}, fps=9.6, anatomical_only=0
            )

        assert captured["settings"]["tau"] == 2.0
        assert captured["settings"]["fs"] == 9.6  # explicit fps preserved

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
            # anatomical_only=0 avoids importing cellpose (not under test here).
            run_suite2p(tiff_dir, output_dir, fps=9.6, anatomical_only=0)

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
            # anatomical_only=0 avoids importing cellpose (not under test here).
            run_suite2p(tiff_dir, output_dir, fps=9.6, anatomical_only=0)

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
            # anatomical_only=0 avoids importing cellpose (not under test here).
            run_suite2p(tiff_dir, output_dir, fps=9.6, anatomical_only=0)

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
            # anatomical_only=0 avoids importing cellpose (not under test here).
            result = run_suite2p(tiff_dir, output_dir, fps=9.6, anatomical_only=0)
        assert result.exists()

    def test_custom_fps(self, tmp_path):
        """Custom fps parameter is reflected in settings."""
        tiff_dir = tmp_path / "tiffs"
        tiff_dir.mkdir()
        (tiff_dir / "data.tif").write_bytes(b"\x00")
        output_dir = tmp_path / "output"

        mock_suite2p = MagicMock()
        mock_suite2p.default_settings.return_value = {
            "fs": 29.97,
            "tau": 1.0,
            "run": {"do_deconvolution": False},
            "io": {"delete_bin": True},
            "registration": {
                "nonrigid": True,
                "block_size": (128, 128),
                "batch_size": 100,
                "maxregshift": 0.1,
                "smooth_sigma": 1.15,
                "th_badframes": 1.0,
                "subpixel": 10,
            },
            "detection": {
                "threshold_scaling": 1.0,
                "max_overlap": 0.75,
                "sparsery_settings": {"highpass_neuropil": 25},
            },
            "extraction": {
                "batch_size": 500,
                "neuropil_extract": True,
                "neuropil_coefficient": 0.7,
                "inner_neuropil_radius": 2,
                "min_neuropil_pixels": 350,
                "allow_overlap": False,
            },
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
            # anatomical_only=0 avoids importing cellpose (not under test here).
            run_suite2p(tiff_dir, output_dir, fps=15.0, anatomical_only=0)

        assert captured["settings"]["fs"] == 15.0


# ---------------------------------------------------------------------------
# tau_for_indicator
# ---------------------------------------------------------------------------


class TestTauForIndicator:
    def test_known_gcamps(self):
        from hm2p.extraction.run_suite2p import INDICATOR_TAU

        for name, expected in INDICATOR_TAU.items():
            assert tau_for_indicator(name) == expected

    def test_gcamps6s_value(self):
        assert tau_for_indicator("GCaMP6s") == 1.5

    def test_gcamps6f_value(self):
        assert tau_for_indicator("GCaMP6f") == 0.4

    def test_gcamps8f_value(self):
        assert tau_for_indicator("GCaMP8f") == 0.2

    def test_unknown_returns_default(self):
        result = tau_for_indicator("SomeNewIndicator")
        assert result == 1.0  # _INDICATOR_TAU_DEFAULT

    def test_unknown_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="hm2p.extraction.run_suite2p"):
            tau_for_indicator("FakeGCaMP99x")
        assert "FakeGCaMP99x" in caplog.text
        assert "default tau" in caplog.text.lower() or "using default" in caplog.text.lower()

    def test_empty_string_returns_default(self):
        result = tau_for_indicator("")
        assert result == 1.0

    def test_case_sensitive(self):
        # Lookup is case-sensitive; lowercase should fall back to default.
        result = tau_for_indicator("gcamps6s")
        assert result == 1.0


# ---------------------------------------------------------------------------
# fps_from_timestamps
# ---------------------------------------------------------------------------


def _write_timestamps_h5(path: Path, frame_times: np.ndarray) -> None:
    """Write a minimal timestamps.h5 for testing."""
    from hm2p.io.hdf5 import write_h5

    write_h5(path, {"frame_times_imaging": frame_times}, attrs={})


class TestFpsFromTimestamps:
    def test_uniform_10hz(self, tmp_path):
        """10 Hz uniform timestamps → fps ~ 10.0."""
        ts_path = tmp_path / "timestamps.h5"
        frame_times = np.arange(100) / 10.0  # 100 frames at 10 Hz
        _write_timestamps_h5(ts_path, frame_times)
        fps = fps_from_timestamps(ts_path)
        np.testing.assert_allclose(fps, 10.0, rtol=1e-4)

    def test_uniform_9p6hz(self, tmp_path):
        """9.6 Hz uniform timestamps → fps ~ 9.6."""
        ts_path = tmp_path / "timestamps.h5"
        frame_times = np.arange(200) / 9.6
        _write_timestamps_h5(ts_path, frame_times)
        fps = fps_from_timestamps(ts_path)
        np.testing.assert_allclose(fps, 9.6, rtol=1e-3)

    def test_missing_file_returns_fallback(self, tmp_path):
        """Missing timestamps.h5 → fallback 29.97 Hz with warning."""
        ts_path = tmp_path / "nonexistent.h5"
        fps = fps_from_timestamps(ts_path)
        assert fps == pytest.approx(29.97, rel=1e-3)

    def test_missing_file_logs_warning(self, tmp_path, caplog):
        import logging

        ts_path = tmp_path / "nonexistent.h5"
        with caplog.at_level(logging.WARNING, logger="hm2p.extraction.run_suite2p"):
            fps_from_timestamps(ts_path)
        assert "fallback" in caplog.text.lower() or "not found" in caplog.text.lower()

    def test_fewer_than_2_frames_returns_fallback(self, tmp_path):
        """Single frame in timestamps → fallback 29.97 Hz."""
        ts_path = tmp_path / "timestamps.h5"
        _write_timestamps_h5(ts_path, np.array([0.0]))
        fps = fps_from_timestamps(ts_path)
        assert fps == pytest.approx(29.97, rel=1e-3)

    def test_returns_float(self, tmp_path):
        ts_path = tmp_path / "timestamps.h5"
        _write_timestamps_h5(ts_path, np.linspace(0, 10, 100))
        fps = fps_from_timestamps(ts_path)
        assert isinstance(fps, float)

    def test_jittered_timestamps(self, tmp_path):
        """Slightly jittered timestamps: mean fps should be close to nominal."""
        rng = np.random.default_rng(0)
        nominal_dt = 1.0 / 9.6
        dt = nominal_dt + rng.standard_normal(199) * 0.0001  # 0.1 ms jitter
        frame_times = np.concatenate([[0.0], np.cumsum(dt)])
        ts_path = tmp_path / "timestamps.h5"
        _write_timestamps_h5(ts_path, frame_times)
        fps = fps_from_timestamps(ts_path)
        np.testing.assert_allclose(fps, 9.6, rtol=0.01)

    def test_robust_to_single_dropped_frame(self, tmp_path):
        """QA 1.8: median ISI estimator is unbiased by a single dropped frame.

        Construct a 9.6 Hz train with one dropped frame in the middle (so
        one ISI is doubled). The previous mean-of-reciprocals estimator
        was biased high by Jensen's inequality plus the heavy-tailed ISI
        distribution that dropped frames produce. Median ISI is robust
        to that outlier.
        """
        nominal_dt = 1.0 / 9.6
        n = 200
        dt = np.full(n - 1, nominal_dt)
        # Drop one frame: double the ISI at index 100.
        dt[100] = 2.0 * nominal_dt
        frame_times = np.concatenate([[0.0], np.cumsum(dt)])
        ts_path = tmp_path / "timestamps.h5"
        _write_timestamps_h5(ts_path, frame_times)
        fps = fps_from_timestamps(ts_path)
        # Median ISI keeps 9.6 Hz exactly; the dropped frame is a single
        # outlier in 199 ISIs.
        np.testing.assert_allclose(fps, 9.6, rtol=1e-6)

    def test_estimator_matches_calcium_run_convention(self, tmp_path):
        """fps_from_timestamps must use the same estimator as calcium/run.py.

        Both pipeline stages must agree on the per-session fps. ``calcium/run``
        uses ``1 / median(diff(frame_times))``; this regression check pins
        the same convention here.
        """
        rng = np.random.default_rng(1)
        nominal_dt = 1.0 / 9.6
        # 1 ms jitter — Jensen bias on mean-of-reciprocals is then
        # noticeable at 4 dp, larger than the median estimator's noise.
        dt = nominal_dt + rng.standard_normal(499) * 0.001
        dt = np.clip(dt, 0.001, None)  # keep positive
        frame_times = np.concatenate([[0.0], np.cumsum(dt)])
        ts_path = tmp_path / "timestamps.h5"
        _write_timestamps_h5(ts_path, frame_times)
        fps = fps_from_timestamps(ts_path)
        # Both stages share this convention.
        expected = 1.0 / float(np.median(np.diff(frame_times)))
        np.testing.assert_allclose(fps, expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# run_suite2p — indicator + timestamps_h5 wiring
# ---------------------------------------------------------------------------


class TestRunSuite2pWiring:
    """Tests for per-session fps/tau resolution inside run_suite2p."""

    def _make_tiffs(self, tmp_path):
        tiff_dir = tmp_path / "tiffs"
        tiff_dir.mkdir()
        (tiff_dir / "data.tif").write_bytes(b"\x00")
        return tiff_dir

    def _make_mock_suite2p(self, captured: dict):
        mock = MagicMock()
        # Return a real dict so deep_update and key access work correctly.
        mock.default_settings.return_value = {
            "fs": 9.6,
            "tau": 1.0,
            "run": {"do_deconvolution": False},
            "io": {"delete_bin": True},
            "registration": {
                "nonrigid": True,
                "block_size": (96, 96),
                "batch_size": 100,
                "maxregshift": 0.15,
                "smooth_sigma": 1.15,
                "th_badframes": 1.0,
                "subpixel": 10,
            },
            "detection": {
                "threshold_scaling": 1.0,
                "max_overlap": 0.75,
                "sparsery_settings": {"highpass_neuropil": 25},
            },
            "extraction": {
                "batch_size": 500,
                "neuropil_extract": True,
                "neuropil_coefficient": 0.7,
                "inner_neuropil_radius": 2,
                "min_neuropil_pixels": 350,
                "allow_overlap": False,
            },
            "classification": {"use_builtin_classifier": True},
        }

        def fake_run_s2p(db, settings):
            captured["settings"] = settings
            s2p_dir = Path(db["save_path0"]) / "suite2p" / "plane0"
            s2p_dir.mkdir(parents=True)
            for name in ("F.npy", "Fneu.npy", "iscell.npy", "stat.npy", "ops.npy"):
                np.save(s2p_dir / name, np.zeros(1))

        mock.run_s2p = fake_run_s2p
        return mock

    def test_fps_read_from_timestamps_h5(self, tmp_path):
        """fps is read from timestamps.h5 when not supplied explicitly."""
        tiff_dir = self._make_tiffs(tmp_path)
        ts_path = tmp_path / "timestamps.h5"
        # Write 9.6 Hz uniform timestamps.
        _write_timestamps_h5(ts_path, np.arange(500) / 9.6)

        captured: dict = {}
        mock = self._make_mock_suite2p(captured)

        with patch.dict("sys.modules", {"suite2p": mock}):
            # anatomical_only=0 avoids importing cellpose (not under test here).
            run_suite2p(tiff_dir, tmp_path / "out", timestamps_h5=ts_path, anatomical_only=0)

        np.testing.assert_allclose(captured["settings"]["fs"], 9.6, rtol=0.01)

    def test_explicit_fps_overrides_timestamps(self, tmp_path):
        """Explicit fps= overrides reading from timestamps.h5."""
        tiff_dir = self._make_tiffs(tmp_path)
        ts_path = tmp_path / "timestamps.h5"
        _write_timestamps_h5(ts_path, np.arange(500) / 9.6)

        captured: dict = {}
        mock = self._make_mock_suite2p(captured)

        with patch.dict("sys.modules", {"suite2p": mock}):
            # anatomical_only=0 avoids importing cellpose (not under test here).
            run_suite2p(
                tiff_dir, tmp_path / "out", fps=15.0, timestamps_h5=ts_path, anatomical_only=0
            )

        assert captured["settings"]["fs"] == 15.0

    def test_indicator_sets_tau(self, tmp_path):
        """indicator='GCaMP6f' → tau=0.4 in settings."""
        tiff_dir = self._make_tiffs(tmp_path)
        captured: dict = {}
        mock = self._make_mock_suite2p(captured)

        with patch.dict("sys.modules", {"suite2p": mock}):
            # anatomical_only=0 avoids importing cellpose (not under test here).
            run_suite2p(
                tiff_dir, tmp_path / "out", fps=9.6, indicator="GCaMP6f", anatomical_only=0
            )

        assert captured["settings"]["tau"] == pytest.approx(0.4)

    def test_default_indicator_gcamps6s(self, tmp_path):
        """Default indicator='GCaMP6s' → tau=1.5."""
        tiff_dir = self._make_tiffs(tmp_path)
        captured: dict = {}
        mock = self._make_mock_suite2p(captured)

        with patch.dict("sys.modules", {"suite2p": mock}):
            # anatomical_only=0 avoids importing cellpose (not under test here).
            run_suite2p(tiff_dir, tmp_path / "out", fps=9.6, anatomical_only=0)

        # GCaMP6s tau=1.5
        assert captured["settings"]["tau"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Item 5 — Cellpose 3 anatomical prior (anatomical_only parameter)
# ---------------------------------------------------------------------------


class TestDefaultSettingsAnatomicalOnly:
    """Tests that anatomical_only propagates through default_settings."""

    def test_anatomical_only_default_is_2(self):
        """Default anatomical_only is 2 (Cellpose seeds + activity refinement)."""
        settings = default_settings()
        assert settings.get("anatomical_only") == 2

    def test_anatomical_only_propagates_0(self):
        """anatomical_only=0 disables Cellpose (activity-only mode)."""
        settings = default_settings(anatomical_only=0)
        assert settings.get("anatomical_only") == 0

    def test_anatomical_only_propagates_1(self):
        settings = default_settings(anatomical_only=1)
        assert settings.get("anatomical_only") == 1

    def test_anatomical_only_propagates_2(self):
        settings = default_settings(anatomical_only=2)
        assert settings.get("anatomical_only") == 2

    def test_anatomical_only_propagates_3(self):
        settings = default_settings(anatomical_only=3)
        assert settings.get("anatomical_only") == 3

    @pytest.mark.skipif(not _suite2p_available, reason="suite2p not installed")
    def test_anatomical_only_in_detection_block(self):
        """anatomical_only is stored under the detection sub-dict (suite2p API)."""
        settings = default_settings(anatomical_only=2)
        # Must appear in the detection block so Suite2p reads it correctly
        assert "detection" in settings
        assert settings["detection"].get("anatomical_only") == 2

    def test_anatomical_only_in_top_level_when_suite2p_absent(self):
        """Without suite2p, fallback dict carries anatomical_only at top level."""
        with patch.dict("sys.modules", {"suite2p": None}):
            # Import the module afresh to pick up the patched suite2p
            import importlib

            import hm2p.extraction.run_suite2p as m

            importlib.reload(m)
            settings = m.default_settings(anatomical_only=3)
        assert settings.get("anatomical_only") == 3
        # Reload back to normal
        import importlib

        import hm2p.extraction.run_suite2p

        importlib.reload(hm2p.extraction.run_suite2p)


class TestRunSuite2pCellposeCheck:
    """Tests for the Cellpose pre-flight ImportError in run_suite2p."""

    def _make_tiffs(self, tmp_path):
        tiff_dir = tmp_path / "tiffs"
        tiff_dir.mkdir()
        (tiff_dir / "data.tif").write_bytes(b"\x00")
        return tiff_dir

    def _make_mock_suite2p(self):
        mock = MagicMock()
        mock.default_settings.return_value = {
            "fs": 9.6,
            "tau": 1.0,
            "run": {"do_deconvolution": False},
            "io": {"delete_bin": True},
            "registration": {
                "nonrigid": True,
                "block_size": (96, 96),
                "batch_size": 100,
                "maxregshift": 0.15,
                "smooth_sigma": 1.15,
                "th_badframes": 1.0,
                "subpixel": 10,
            },
            "detection": {
                "threshold_scaling": 1.0,
                "max_overlap": 0.75,
                "sparsery_settings": {"highpass_neuropil": 25},
            },
            "extraction": {
                "batch_size": 500,
                "neuropil_extract": True,
                "neuropil_coefficient": 0.7,
                "inner_neuropil_radius": 2,
                "min_neuropil_pixels": 350,
                "allow_overlap": False,
            },
            "classification": {"use_builtin_classifier": True},
        }

        def fake_run_s2p(db, settings):
            s2p_dir = Path(db["save_path0"]) / "suite2p" / "plane0"
            s2p_dir.mkdir(parents=True)
            for name in ("F.npy", "Fneu.npy", "iscell.npy", "stat.npy", "ops.npy"):
                np.save(s2p_dir / name, np.zeros(1))

        mock.run_s2p = fake_run_s2p
        return mock

    def test_anatomical_only_1_raises_if_cellpose_missing(self, tmp_path):
        """anatomical_only >= 1 must raise ImportError when cellpose is absent."""
        tiff_dir = self._make_tiffs(tmp_path)
        mock_s2p = self._make_mock_suite2p()

        with (
            patch.dict("sys.modules", {"suite2p": mock_s2p, "cellpose": None}),
            pytest.raises(ImportError, match="cellpose"),
        ):
            run_suite2p(tiff_dir, tmp_path / "out", fps=9.6, anatomical_only=1)

    def test_anatomical_only_2_raises_if_cellpose_missing(self, tmp_path):
        """Default anatomical_only=2 raises ImportError when cellpose absent."""
        tiff_dir = self._make_tiffs(tmp_path)
        mock_s2p = self._make_mock_suite2p()

        with (
            patch.dict("sys.modules", {"suite2p": mock_s2p, "cellpose": None}),
            pytest.raises(ImportError, match="anatomical_only=2"),
        ):
            run_suite2p(tiff_dir, tmp_path / "out", fps=9.6, anatomical_only=2)

    def test_anatomical_only_0_does_not_require_cellpose(self, tmp_path):
        """anatomical_only=0 (activity-only) must NOT import cellpose."""
        tiff_dir = self._make_tiffs(tmp_path)
        mock_s2p = self._make_mock_suite2p()

        # Remove cellpose from sys.modules entirely to simulate absence.
        with patch.dict("sys.modules", {"suite2p": mock_s2p, "cellpose": None}):
            # Should not raise — cellpose is not required for anatomical_only=0.
            result = run_suite2p(tiff_dir, tmp_path / "out", fps=9.6, anatomical_only=0)

        assert result.exists()

    def test_error_message_mentions_install_instructions(self, tmp_path):
        """ImportError message must reference the install command."""
        tiff_dir = self._make_tiffs(tmp_path)
        mock_s2p = self._make_mock_suite2p()

        with (
            patch.dict("sys.modules", {"suite2p": mock_s2p, "cellpose": None}),
            pytest.raises(ImportError) as exc_info,
        ):
            run_suite2p(tiff_dir, tmp_path / "out", fps=9.6, anatomical_only=1)

        msg = str(exc_info.value)
        assert "cellpose" in msg.lower()
        # Should mention how to install
        assert "pip install" in msg or "install" in msg.lower()

    def test_anatomical_only_wired_into_settings(self, tmp_path):
        """anatomical_only value is passed through to Suite2p settings dict."""
        tiff_dir = self._make_tiffs(tmp_path)
        mock_s2p = self._make_mock_suite2p()
        captured: dict = {}

        original_run_s2p = mock_s2p.run_s2p

        def capturing_run_s2p(db, settings):
            captured["settings"] = settings
            return original_run_s2p(db, settings)

        mock_s2p.run_s2p = capturing_run_s2p

        mock_cellpose = MagicMock()

        with patch.dict("sys.modules", {"suite2p": mock_s2p, "cellpose": mock_cellpose}):
            run_suite2p(tiff_dir, tmp_path / "out", fps=9.6, anatomical_only=2)

        assert captured["settings"].get("detection", {}).get("anatomical_only") == 2
