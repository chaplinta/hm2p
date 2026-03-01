"""Tests for calcium/run.py — Stage 4 end-to-end pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hm2p.calcium.run import load_suite2p, run


# ---------------------------------------------------------------------------
# Helpers: synthetic Suite2p plane0 directory + timestamps.h5
# ---------------------------------------------------------------------------


def _write_suite2p_plane0(
    suite2p_dir: Path,
    n_rois: int = 12,
    n_cells: int = 8,
    n_frames: int = 300,
    rng: np.random.Generator | None = None,
) -> None:
    """Write minimal synthetic Suite2p plane0 numpy files."""
    if rng is None:
        rng = np.random.default_rng(0)
    plane = suite2p_dir / "plane0"
    plane.mkdir(parents=True)

    F = rng.uniform(100, 500, (n_rois, n_frames)).astype(np.float32)
    Fneu = rng.uniform(50, 200, (n_rois, n_frames)).astype(np.float32)

    # First n_cells rows are cells; rest are non-cells
    iscell = np.zeros((n_rois, 2), dtype=np.float32)
    iscell[:n_cells, 0] = 1.0

    np.save(plane / "F.npy", F)
    np.save(plane / "Fneu.npy", Fneu)
    np.save(plane / "iscell.npy", iscell)


def _write_timestamps(path: Path, n_frames: int = 300, fps: float = 30.0) -> None:
    """Write minimal synthetic timestamps.h5."""
    from hm2p.io.hdf5 import write_h5

    frame_times_camera = np.linspace(0.0, n_frames / 100.0, n_frames * 3, dtype=np.float64)
    frame_times_imaging = np.linspace(0.0, n_frames / fps, n_frames, dtype=np.float64)
    write_h5(
        path,
        {
            "frame_times_camera": frame_times_camera,
            "frame_times_imaging": frame_times_imaging,
            "light_on_times": np.array([0.0, 60.0], dtype=np.float64),
            "light_off_times": np.array([60.0, 120.0], dtype=np.float64),
        },
        attrs={"session_id": "test", "fps_camera": 100.0, "fps_imaging": fps},
    )


# ---------------------------------------------------------------------------
# load_suite2p
# ---------------------------------------------------------------------------


class TestLoadSuite2p:
    def test_returns_correct_shapes(self, tmp_path: Path) -> None:
        """F and Fneu have shape (n_rois_all, n_frames); cell_mask is 1D bool."""
        _write_suite2p_plane0(tmp_path / "suite2p", n_rois=12, n_cells=8, n_frames=300)
        F, Fneu, cell_mask = load_suite2p(tmp_path / "suite2p")
        assert F.shape == (12, 300)
        assert Fneu.shape == (12, 300)
        assert cell_mask.shape == (12,)
        assert cell_mask.dtype == bool

    def test_cell_mask_correct_count(self, tmp_path: Path) -> None:
        """cell_mask has exactly n_cells True values."""
        _write_suite2p_plane0(tmp_path / "suite2p", n_rois=15, n_cells=10, n_frames=100)
        _, _, cell_mask = load_suite2p(tmp_path / "suite2p")
        assert cell_mask.sum() == 10

    def test_dtypes_float32(self, tmp_path: Path) -> None:
        """F and Fneu are cast to float32."""
        _write_suite2p_plane0(tmp_path / "suite2p")
        F, Fneu, _ = load_suite2p(tmp_path / "suite2p")
        assert F.dtype == np.float32
        assert Fneu.dtype == np.float32

    def test_missing_plane0_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError if plane0/ does not exist."""
        (tmp_path / "suite2p").mkdir()
        with pytest.raises(FileNotFoundError, match="plane0"):
            load_suite2p(tmp_path / "suite2p")

    def test_missing_npy_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError if a required .npy file is absent."""
        plane = tmp_path / "suite2p" / "plane0"
        plane.mkdir(parents=True)
        np.save(plane / "F.npy", np.zeros((5, 100), dtype=np.float32))
        # Fneu.npy and iscell.npy missing
        with pytest.raises(FileNotFoundError):
            load_suite2p(tmp_path / "suite2p")


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestCalciumRun:
    def test_creates_output_file(self, tmp_path: Path) -> None:
        """run() creates ca.h5 at output_path."""
        suite2p_dir = tmp_path / "suite2p"
        _write_suite2p_plane0(suite2p_dir, n_rois=10, n_cells=6, n_frames=300)
        ts_h5 = tmp_path / "timestamps.h5"
        _write_timestamps(ts_h5, n_frames=300)
        out = tmp_path / "ca.h5"
        run(suite2p_dir, ts_h5, session_id="test", output_path=out)
        assert out.exists()

    def test_dff_shape(self, tmp_path: Path) -> None:
        """dff in ca.h5 has shape (n_cells, n_imaging_frames)."""
        from hm2p.io.hdf5 import read_h5

        suite2p_dir = tmp_path / "suite2p"
        _write_suite2p_plane0(suite2p_dir, n_rois=10, n_cells=6, n_frames=300)
        ts_h5 = tmp_path / "timestamps.h5"
        _write_timestamps(ts_h5, n_frames=300)
        out = tmp_path / "ca.h5"
        run(suite2p_dir, ts_h5, session_id="test", output_path=out)

        ca = read_h5(out)
        assert ca["dff"].shape == (6, 300)

    def test_dff_dtype_float32(self, tmp_path: Path) -> None:
        """dff is stored as float32."""
        from hm2p.io.hdf5 import read_h5

        suite2p_dir = tmp_path / "suite2p"
        _write_suite2p_plane0(suite2p_dir, n_rois=8, n_cells=5, n_frames=200)
        ts_h5 = tmp_path / "timestamps.h5"
        _write_timestamps(ts_h5, n_frames=200)
        out = tmp_path / "ca.h5"
        run(suite2p_dir, ts_h5, session_id="test", output_path=out)

        ca = read_h5(out)
        assert ca["dff"].dtype == np.float32

    def test_frame_times_in_output(self, tmp_path: Path) -> None:
        """frame_times in ca.h5 matches imaging frame times from timestamps.h5."""
        from hm2p.io.hdf5 import read_h5

        suite2p_dir = tmp_path / "suite2p"
        _write_suite2p_plane0(suite2p_dir, n_rois=6, n_cells=4, n_frames=200)
        ts_h5 = tmp_path / "timestamps.h5"
        _write_timestamps(ts_h5, n_frames=200, fps=30.0)
        out = tmp_path / "ca.h5"
        run(suite2p_dir, ts_h5, session_id="test", output_path=out)

        ca = read_h5(out)
        ts = read_h5(ts_h5)
        np.testing.assert_array_equal(ca["frame_times"], ts["frame_times_imaging"])

    def test_attrs_session_id(self, tmp_path: Path) -> None:
        """session_id attribute in ca.h5 matches argument."""
        from hm2p.io.hdf5 import read_attrs

        suite2p_dir = tmp_path / "suite2p"
        _write_suite2p_plane0(suite2p_dir, n_rois=5, n_cells=3, n_frames=150)
        ts_h5 = tmp_path / "timestamps.h5"
        _write_timestamps(ts_h5, n_frames=150)
        out = tmp_path / "ca.h5"
        run(suite2p_dir, ts_h5, session_id="20220804_13_52_02_1117646", output_path=out)

        attrs = read_attrs(out)
        assert attrs["session_id"] == "20220804_13_52_02_1117646"

    def test_no_spikes_by_default(self, tmp_path: Path) -> None:
        """'spikes' key is absent in ca.h5 when run_cascade=False (default)."""
        from hm2p.io.hdf5 import read_h5

        suite2p_dir = tmp_path / "suite2p"
        _write_suite2p_plane0(suite2p_dir, n_rois=5, n_cells=3, n_frames=150)
        ts_h5 = tmp_path / "timestamps.h5"
        _write_timestamps(ts_h5, n_frames=150)
        out = tmp_path / "ca.h5"
        run(suite2p_dir, ts_h5, session_id="test", output_path=out)

        ca = read_h5(out)
        assert "spikes" not in ca

    def test_neuropil_coefficient_applied(self, tmp_path: Path) -> None:
        """Different neuropil coefficients produce different dff values."""
        from hm2p.io.hdf5 import read_h5

        rng = np.random.default_rng(7)
        suite2p_dir = tmp_path / "s2p"
        _write_suite2p_plane0(suite2p_dir, n_rois=5, n_cells=5, n_frames=200, rng=rng)
        ts_h5 = tmp_path / "ts.h5"
        _write_timestamps(ts_h5, n_frames=200)

        out_a = tmp_path / "ca_a.h5"
        out_b = tmp_path / "ca_b.h5"
        run(suite2p_dir, ts_h5, session_id="test", output_path=out_a, neuropil_coefficient=0.3)
        run(suite2p_dir, ts_h5, session_id="test", output_path=out_b, neuropil_coefficient=0.9)

        dff_a = read_h5(out_a)["dff"]
        dff_b = read_h5(out_b)["dff"]
        assert not np.allclose(dff_a, dff_b)

    def test_output_validates_ca_schema(self, tmp_path: Path) -> None:
        """ca.h5 output passes the pandera schema validator."""
        from hm2p.io.hdf5 import read_h5, validate_ca_h5

        suite2p_dir = tmp_path / "suite2p"
        _write_suite2p_plane0(suite2p_dir, n_rois=8, n_cells=5, n_frames=300)
        ts_h5 = tmp_path / "timestamps.h5"
        _write_timestamps(ts_h5, n_frames=300)
        out = tmp_path / "ca.h5"
        run(suite2p_dir, ts_h5, session_id="test", output_path=out)

        validate_ca_h5(read_h5(out))  # should not raise
