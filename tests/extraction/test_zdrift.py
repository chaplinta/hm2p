"""Tests for extraction/zdrift.py — z-drift estimation from z-stacks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tifffile

from hm2p.extraction.zdrift import (
    _phase_correlate_2d,
    _register_to_zstack_fallback,
    compute_zdrift,
    compute_zdrift_from_meanimg,
    load_zdrift,
    load_zdrift_meanimg,
    load_zstack,
    save_zdrift,
    save_zdrift_meanimg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_zstack_tiff(path: Path, n_zplanes: int = 5, ly: int = 32, lx: int = 32) -> Path:
    """Write a small synthetic z-stack TIFF and return the path."""
    rng = np.random.default_rng(42)
    zstack = rng.integers(0, 255, (n_zplanes, ly, lx), dtype=np.uint16)
    tifffile.imwrite(str(path), zstack)
    return path


def _make_suite2p_dir(
    tmp_path: Path,
    n_frames: int = 20,
    ly: int = 32,
    lx: int = 32,
) -> Path:
    """Create a minimal suite2p plane0/ directory with ops.npy and data.bin."""
    plane_dir = tmp_path / "suite2p" / "plane0"
    plane_dir.mkdir(parents=True)

    ops = {
        "nframes": n_frames,
        "Ly": ly,
        "Lx": lx,
    }
    np.save(plane_dir / "ops.npy", ops)

    # Write a binary file matching Suite2p's int16 format
    rng = np.random.default_rng(99)
    frames = rng.integers(-100, 500, (n_frames, ly, lx), dtype=np.int16)
    frames.tofile(str(plane_dir / "data.bin"))

    return plane_dir


# ---------------------------------------------------------------------------
# test_load_zstack
# ---------------------------------------------------------------------------


class TestLoadZstack:
    """Tests for load_zstack."""

    def test_basic(self, tmp_path: Path) -> None:
        tiff_path = _make_zstack_tiff(tmp_path / "zstack.tif", n_zplanes=5, ly=16, lx=16)
        zstack = load_zstack(tiff_path)
        assert zstack.shape == (5, 16, 16)
        assert zstack.dtype == np.float32

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Z-stack TIFF not found"):
            load_zstack(tmp_path / "nonexistent.tif")

    def test_wrong_ndim(self, tmp_path: Path) -> None:
        # Write a 2-D TIFF
        tifffile.imwrite(str(tmp_path / "flat.tif"), np.zeros((32, 32), dtype=np.uint16))
        with pytest.raises(ValueError, match="Expected 3-D z-stack"):
            load_zstack(tmp_path / "flat.tif")


# ---------------------------------------------------------------------------
# test_phase_correlate_2d
# ---------------------------------------------------------------------------


class TestPhaseCorrelate:
    """Tests for the fallback phase-correlation function."""

    def test_identical_images(self) -> None:
        rng = np.random.default_rng(0)
        img = rng.random((16, 16))
        corr = _phase_correlate_2d(img, img)
        # Identical images should give correlation close to 1.0
        assert corr > 0.9

    def test_different_images(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.random((16, 16))
        b = rng.random((16, 16)) * 0.01  # very different
        corr_same = _phase_correlate_2d(a, a)
        corr_diff = _phase_correlate_2d(a, b)
        # Same should be higher than different
        assert corr_same > corr_diff


# ---------------------------------------------------------------------------
# test_register_to_zstack_fallback
# ---------------------------------------------------------------------------


class TestFallbackRegistration:
    """Tests for _register_to_zstack_fallback."""

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(1)
        frames = rng.random((3, 16, 16)).astype(np.float32)
        zstack = rng.random((5, 16, 16)).astype(np.float32)
        zcorr = _register_to_zstack_fallback(frames, zstack)
        assert zcorr.shape == (3, 5)

    def test_best_match_at_correct_plane(self) -> None:
        """Frame copied from z-plane 2 should correlate best with plane 2."""
        rng = np.random.default_rng(2)
        zstack = rng.random((5, 16, 16)).astype(np.float32)
        # Frame is exactly z-plane 2
        frames = zstack[2:3].copy()
        zcorr = _register_to_zstack_fallback(frames, zstack)
        assert np.argmax(zcorr[0]) == 2


# ---------------------------------------------------------------------------
# test_compute_zdrift (mocked Suite2p)
# ---------------------------------------------------------------------------


class TestComputeZdrift:
    """Tests for compute_zdrift output shape and content."""

    def test_output_shape_fallback(self, tmp_path: Path) -> None:
        """Test compute_zdrift with the fallback (no Suite2p)."""
        n_frames = 10
        n_zplanes = 4
        ly, lx = 16, 16

        zstack_path = _make_zstack_tiff(
            tmp_path / "zstack.tif", n_zplanes=n_zplanes, ly=ly, lx=lx
        )
        plane_dir = _make_suite2p_dir(
            tmp_path, n_frames=n_frames, ly=ly, lx=lx
        )

        # Force fallback path
        with patch("hm2p.extraction.zdrift._HAS_SUITE2P", False):
            result = compute_zdrift(
                suite2p_dir=plane_dir,
                zstack_path=zstack_path,
                sigma=1.0,
                batch_size=5,
            )

        assert result["zpos"].shape == (n_frames,)
        assert result["zpos"].dtype == np.int32
        assert result["zcorr"].shape == (n_frames, n_zplanes)
        assert result["zcorr"].dtype == np.float32
        assert result["zpos_smooth"].shape == (n_frames,)
        assert result["n_zplanes"] == n_zplanes
        assert result["zstack_path"] == str(zstack_path)
        # zpos values should be valid plane indices
        assert np.all(result["zpos"] >= 0)
        assert np.all(result["zpos"] < n_zplanes)

    def test_missing_suite2p_dir(self, tmp_path: Path) -> None:
        zstack_path = _make_zstack_tiff(tmp_path / "zstack.tif")
        with pytest.raises(FileNotFoundError, match="Suite2p directory not found"):
            compute_zdrift(tmp_path / "nope", zstack_path)

    def test_missing_ops(self, tmp_path: Path) -> None:
        zstack_path = _make_zstack_tiff(tmp_path / "zstack.tif")
        plane_dir = tmp_path / "plane0"
        plane_dir.mkdir(parents=True)
        # Write data.bin but not ops.npy
        np.zeros(10, dtype=np.int16).tofile(str(plane_dir / "data.bin"))
        with pytest.raises(FileNotFoundError, match="ops.npy not found"):
            compute_zdrift(plane_dir, zstack_path)

    def test_missing_data_bin(self, tmp_path: Path) -> None:
        zstack_path = _make_zstack_tiff(tmp_path / "zstack.tif")
        plane_dir = tmp_path / "plane0"
        plane_dir.mkdir(parents=True)
        np.save(plane_dir / "ops.npy", {"nframes": 10, "Ly": 32, "Lx": 32})
        with pytest.raises(FileNotFoundError, match="data.bin not found"):
            compute_zdrift(plane_dir, zstack_path)

    def test_batching_produces_same_result(self, tmp_path: Path) -> None:
        """Different batch_size values should give the same result."""
        n_frames = 12
        ly, lx = 16, 16
        zstack_path = _make_zstack_tiff(tmp_path / "zstack.tif", n_zplanes=3, ly=ly, lx=lx)
        plane_dir = _make_suite2p_dir(tmp_path, n_frames=n_frames, ly=ly, lx=lx)

        with patch("hm2p.extraction.zdrift._HAS_SUITE2P", False):
            r1 = compute_zdrift(plane_dir, zstack_path, sigma=1.0, batch_size=4)
            r2 = compute_zdrift(plane_dir, zstack_path, sigma=1.0, batch_size=100)

        np.testing.assert_array_equal(r1["zpos"], r2["zpos"])
        np.testing.assert_allclose(r1["zcorr"], r2["zcorr"], atol=1e-6)


# ---------------------------------------------------------------------------
# test_save_load_zdrift roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadZdrift:
    """Tests for save_zdrift / load_zdrift roundtrip."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(7)
        n_frames, n_zplanes = 50, 8
        zdrift = {
            "zpos": rng.integers(0, n_zplanes, n_frames).astype(np.int32),
            "zcorr": rng.random((n_frames, n_zplanes)).astype(np.float32),
            "zpos_smooth": rng.random(n_frames).astype(np.float64),
            "n_zplanes": n_zplanes,
            "zstack_path": "/data/zstack.tif",
        }
        h5_path = tmp_path / "zdrift.h5"
        save_zdrift(zdrift, h5_path)
        assert h5_path.exists()

        loaded = load_zdrift(h5_path)
        np.testing.assert_array_equal(loaded["zpos"], zdrift["zpos"])
        np.testing.assert_allclose(loaded["zcorr"], zdrift["zcorr"], atol=1e-6)
        np.testing.assert_allclose(loaded["zpos_smooth"], zdrift["zpos_smooth"], atol=1e-10)
        assert loaded["n_zplanes"] == n_zplanes
        assert loaded["zstack_path"] == "/data/zstack.tif"

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_zdrift(tmp_path / "nope.h5")


# ---------------------------------------------------------------------------
# test_zdrift_smooth_reduces_noise
# ---------------------------------------------------------------------------


class TestSmoothReducesNoise:
    """Test that Gaussian smoothing reduces noise in z-position estimates."""

    def test_smooth_reduces_noise(self, tmp_path: Path) -> None:
        """Higher sigma should produce a smoother zpos_smooth trace."""
        n_frames = 50
        ly, lx = 16, 16
        n_zplanes = 5
        zstack_path = _make_zstack_tiff(
            tmp_path / "zstack.tif", n_zplanes=n_zplanes, ly=ly, lx=lx
        )
        plane_dir = _make_suite2p_dir(tmp_path, n_frames=n_frames, ly=ly, lx=lx)

        with patch("hm2p.extraction.zdrift._HAS_SUITE2P", False):
            r_low = compute_zdrift(plane_dir, zstack_path, sigma=0.5, batch_size=50)
            r_high = compute_zdrift(plane_dir, zstack_path, sigma=5.0, batch_size=50)

        # Measure roughness as sum of absolute second differences
        def roughness(arr: np.ndarray) -> float:
            return float(np.sum(np.abs(np.diff(arr, n=2))))

        rough_low = roughness(r_low["zpos_smooth"])
        rough_high = roughness(r_high["zpos_smooth"])
        # Higher sigma should give smoother (less rough) trace
        assert rough_high <= rough_low


# ---------------------------------------------------------------------------
# Helpers for mean-image tests
# ---------------------------------------------------------------------------


def _make_ops_npy(
    path: Path,
    ly: int = 32,
    lx: int = 32,
    seed: int = 42,
) -> Path:
    """Create a minimal ops.npy with a meanImg and return the path."""
    rng = np.random.default_rng(seed)
    ops = {
        "nframes": 100,
        "Ly": ly,
        "Lx": lx,
        "meanImg": rng.random((ly, lx)).astype(np.float32),
    }
    np.save(path, ops)
    return path


# ---------------------------------------------------------------------------
# test_compute_zdrift_from_meanimg
# ---------------------------------------------------------------------------


class TestComputeZdriftFromMeanimg:
    """Tests for compute_zdrift_from_meanimg."""

    def test_output_keys_and_shapes(self, tmp_path: Path) -> None:
        n_zplanes = 5
        ly, lx = 16, 16
        ops_path = _make_ops_npy(tmp_path / "ops.npy", ly=ly, lx=lx)
        zstack_path = _make_zstack_tiff(
            tmp_path / "zstack.tif", n_zplanes=n_zplanes, ly=ly, lx=lx
        )
        result = compute_zdrift_from_meanimg(ops_path, zstack_path)

        assert "zpos_mean" in result
        assert "zcorr_mean" in result
        assert "max_corr" in result
        assert "n_zplanes" in result
        assert "zstack_path" in result

        assert isinstance(result["zpos_mean"], int)
        assert 0 <= result["zpos_mean"] < n_zplanes
        assert result["zcorr_mean"].shape == (n_zplanes,)
        assert result["zcorr_mean"].dtype == np.float64
        assert isinstance(result["max_corr"], float)
        assert result["n_zplanes"] == n_zplanes
        assert result["zstack_path"] == str(zstack_path)

    def test_best_match_correct_plane(self, tmp_path: Path) -> None:
        """meanImg copied from z-plane 3 should match plane 3."""
        ly, lx = 16, 16
        n_zplanes = 6
        rng = np.random.default_rng(10)
        zstack = rng.random((n_zplanes, ly, lx)).astype(np.float32)
        tifffile.imwrite(str(tmp_path / "zstack.tif"), zstack)

        # Make ops with meanImg = z-plane 3
        ops = {"nframes": 50, "Ly": ly, "Lx": lx, "meanImg": zstack[3].copy()}
        np.save(tmp_path / "ops.npy", ops)

        result = compute_zdrift_from_meanimg(
            tmp_path / "ops.npy", tmp_path / "zstack.tif"
        )
        assert result["zpos_mean"] == 3
        assert result["max_corr"] > 0.9

    def test_mismatched_dimensions_cropped(self, tmp_path: Path) -> None:
        """Handles z-stack and meanImg with different dimensions."""
        # meanImg is 20x20, z-stack is 16x16 — should crop to min
        ops = {
            "nframes": 10,
            "Ly": 20,
            "Lx": 20,
            "meanImg": np.random.default_rng(1).random((20, 20)).astype(np.float32),
        }
        np.save(tmp_path / "ops.npy", ops)
        _make_zstack_tiff(tmp_path / "zstack.tif", n_zplanes=4, ly=16, lx=16)

        result = compute_zdrift_from_meanimg(
            tmp_path / "ops.npy", tmp_path / "zstack.tif"
        )
        assert result["zcorr_mean"].shape == (4,)
        assert 0 <= result["zpos_mean"] < 4

    def test_missing_ops(self, tmp_path: Path) -> None:
        zstack_path = _make_zstack_tiff(tmp_path / "zstack.tif")
        with pytest.raises(FileNotFoundError, match="ops.npy not found"):
            compute_zdrift_from_meanimg(tmp_path / "nope.npy", zstack_path)

    def test_missing_zstack(self, tmp_path: Path) -> None:
        ops_path = _make_ops_npy(tmp_path / "ops.npy")
        with pytest.raises(FileNotFoundError, match="Z-stack TIFF not found"):
            compute_zdrift_from_meanimg(ops_path, tmp_path / "nope.tif")

    def test_no_meanimg_in_ops(self, tmp_path: Path) -> None:
        """ops.npy without meanImg should raise ValueError."""
        ops = {"nframes": 10, "Ly": 16, "Lx": 16}
        np.save(tmp_path / "ops.npy", ops)
        zstack_path = _make_zstack_tiff(tmp_path / "zstack.tif")
        with pytest.raises(ValueError, match="does not contain 'meanImg'"):
            compute_zdrift_from_meanimg(tmp_path / "ops.npy", zstack_path)

    def test_max_corr_equals_peak(self, tmp_path: Path) -> None:
        """max_corr should equal the peak of zcorr_mean."""
        ops_path = _make_ops_npy(tmp_path / "ops.npy", ly=16, lx=16)
        zstack_path = _make_zstack_tiff(
            tmp_path / "zstack.tif", n_zplanes=4, ly=16, lx=16
        )
        result = compute_zdrift_from_meanimg(ops_path, zstack_path)
        assert result["max_corr"] == pytest.approx(
            result["zcorr_mean"].max(), abs=1e-10
        )


# ---------------------------------------------------------------------------
# test_save_load_zdrift_meanimg roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadZdriftMeanimg:
    """Tests for save_zdrift_meanimg / load_zdrift_meanimg roundtrip."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(55)
        n_zplanes = 6
        zdrift = {
            "zpos_mean": 3,
            "zcorr_mean": rng.random(n_zplanes).astype(np.float64),
            "max_corr": 0.87,
            "n_zplanes": n_zplanes,
            "zstack_path": "/data/zstack.tif",
        }
        h5_path = tmp_path / "zdrift_meanimg.h5"
        save_zdrift_meanimg(zdrift, h5_path)
        assert h5_path.exists()

        loaded = load_zdrift_meanimg(h5_path)
        assert loaded["zpos_mean"] == 3
        np.testing.assert_allclose(
            loaded["zcorr_mean"], zdrift["zcorr_mean"], atol=1e-10
        )
        assert loaded["max_corr"] == pytest.approx(0.87, abs=1e-6)
        assert loaded["n_zplanes"] == n_zplanes
        assert loaded["zstack_path"] == "/data/zstack.tif"

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_zdrift_meanimg(tmp_path / "nope.h5")

    def test_end_to_end_compute_save_load(self, tmp_path: Path) -> None:
        """Full pipeline: compute → save → load."""
        ly, lx = 16, 16
        ops_path = _make_ops_npy(tmp_path / "ops.npy", ly=ly, lx=lx)
        zstack_path = _make_zstack_tiff(
            tmp_path / "zstack.tif", n_zplanes=5, ly=ly, lx=lx
        )
        result = compute_zdrift_from_meanimg(ops_path, zstack_path)

        h5_path = tmp_path / "zdrift_meanimg.h5"
        save_zdrift_meanimg(result, h5_path)
        loaded = load_zdrift_meanimg(h5_path)

        assert loaded["zpos_mean"] == result["zpos_mean"]
        np.testing.assert_allclose(
            loaded["zcorr_mean"], result["zcorr_mean"], atol=1e-10
        )
        assert loaded["max_corr"] == pytest.approx(result["max_corr"], abs=1e-10)
        assert loaded["n_zplanes"] == result["n_zplanes"]
