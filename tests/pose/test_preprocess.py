"""Tests for pose/preprocess.py — video pre-processing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hm2p.pose.preprocess import (
    crop_to_maze_roi,
    load_calibration,
    load_meta,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_meta_txt(path: Path, **overrides: object) -> None:
    """Write a minimal meta.txt with default values, overridable per-key."""
    defaults = {
        "crop_x": 108, "crop_y": 261, "crop_w": 832, "crop_h": 608,
        "mm_per_pix": 0.8113483203691485,
        "x1": 149.0, "y1": 72.0,
        "x2": 764.0, "y2": 82.0,
        "x3": 757.0, "y3": 509.0,
        "x4": 143.0, "y4": 500.0,
    }
    defaults.update(overrides)
    d = defaults
    path.write_text(
        f"[crop]\n"
        f"x = {d['crop_x']}\n"
        f"y = {d['crop_y']}\n"
        f"width = {d['crop_w']}\n"
        f"height = {d['crop_h']}\n"
        f"\n"
        f"[scale]\n"
        f"mm_per_pix = {d['mm_per_pix']}\n"
        f"\n"
        f"[roi]\n"
        f"x1 = {d['x1']}\n"
        f"y1 = {d['y1']}\n"
        f"x2 = {d['x2']}\n"
        f"y2 = {d['y2']}\n"
        f"x3 = {d['x3']}\n"
        f"y3 = {d['y3']}\n"
        f"x4 = {d['x4']}\n"
        f"y4 = {d['y4']}\n"
    )


def _write_calib_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Write a minimal calibration .npz and return (mtx, dist)."""
    mtx = np.array(
        [[854.0, 0.0, 673.0], [0.0, 854.0, 519.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.array([[-0.249, 0.118, 0.001, -0.0003, -0.033]], dtype=np.float64)
    np.savez(path, mtx=mtx, dist=dist)
    return mtx, dist


# ---------------------------------------------------------------------------
# crop_to_maze_roi
# ---------------------------------------------------------------------------

def test_crop_to_maze_roi_shape() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    roi = (100, 50, 200, 150)
    cropped = crop_to_maze_roi(frame, roi)
    assert cropped.shape == (150, 200, 3)


def test_crop_to_maze_roi_content() -> None:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    roi = (100, 50, 200, 150)
    cropped = crop_to_maze_roi(frame, roi)
    np.testing.assert_array_equal(cropped, frame[50:200, 100:300])


def test_crop_to_maze_roi_origin() -> None:
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    cropped = crop_to_maze_roi(frame, (0, 0, 300, 200))
    np.testing.assert_array_equal(cropped, frame)


# ---------------------------------------------------------------------------
# load_meta
# ---------------------------------------------------------------------------

class TestLoadMeta:
    def test_roi_tuple(self, tmp_path: Path) -> None:
        p = tmp_path / "meta.txt"
        _write_meta_txt(p, crop_x=10, crop_y=20, crop_w=800, crop_h=600)
        result = load_meta(p)
        assert result["roi"] == (10, 20, 800, 600)

    def test_scale(self, tmp_path: Path) -> None:
        p = tmp_path / "meta.txt"
        _write_meta_txt(p, mm_per_pix=0.8113)
        result = load_meta(p)
        assert pytest.approx(result["scale_mm_per_px"]) == 0.8113

    def test_maze_corners_shape(self, tmp_path: Path) -> None:
        p = tmp_path / "meta.txt"
        _write_meta_txt(p)
        result = load_meta(p)
        corners = result["maze_corners"]
        assert isinstance(corners, np.ndarray)
        assert corners.shape == (4, 2)
        assert corners.dtype == np.float64

    def test_maze_corners_values(self, tmp_path: Path) -> None:
        p = tmp_path / "meta.txt"
        _write_meta_txt(p, x1=10.0, y1=20.0, x2=100.0, y2=25.0,
                        x3=98.0, y3=120.0, x4=8.0, y4=118.0)
        corners = load_meta(p)["maze_corners"]
        np.testing.assert_array_equal(
            corners,
            [[10.0, 20.0], [100.0, 25.0], [98.0, 120.0], [8.0, 118.0]],
        )

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_meta(tmp_path / "nonexistent.txt")

    def test_missing_section_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "meta.txt"
        p.write_text("[crop]\nx = 0\n")  # missing [scale] and [roi]
        with pytest.raises(KeyError):
            load_meta(p)

    def test_real_values(self, tmp_path: Path) -> None:
        """Smoke test with values from an actual session meta.txt."""
        p = tmp_path / "meta.txt"
        _write_meta_txt(
            p,
            crop_x=108, crop_y=261, crop_w=832, crop_h=608,
            mm_per_pix=0.8113483203691485,
            x1=149.0, y1=72.0, x2=764.0, y2=82.0,
            x3=757.0, y3=509.0, x4=143.0, y4=500.0,
        )
        result = load_meta(p)
        assert result["roi"] == (108, 261, 832, 608)
        assert pytest.approx(result["scale_mm_per_px"], rel=1e-6) == 0.8113483203691485
        assert result["maze_corners"].shape == (4, 2)


# ---------------------------------------------------------------------------
# load_calibration
# ---------------------------------------------------------------------------

class TestLoadCalibration:
    def test_returns_correct_keys(self, tmp_path: Path) -> None:
        p = tmp_path / "calib.npz"
        _write_calib_npz(p)
        result = load_calibration(p)
        assert "camera_matrix" in result
        assert "dist_coeffs" in result

    def test_matrix_shape(self, tmp_path: Path) -> None:
        p = tmp_path / "calib.npz"
        _write_calib_npz(p)
        result = load_calibration(p)
        assert result["camera_matrix"].shape == (3, 3)
        assert result["dist_coeffs"].shape == (1, 5)

    def test_dtype_float64(self, tmp_path: Path) -> None:
        p = tmp_path / "calib.npz"
        _write_calib_npz(p)
        result = load_calibration(p)
        assert result["camera_matrix"].dtype == np.float64
        assert result["dist_coeffs"].dtype == np.float64

    def test_values_preserved(self, tmp_path: Path) -> None:
        p = tmp_path / "calib.npz"
        mtx, dist = _write_calib_npz(p)
        result = load_calibration(p)
        np.testing.assert_array_almost_equal(result["camera_matrix"], mtx)
        np.testing.assert_array_almost_equal(result["dist_coeffs"], dist)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_calibration(tmp_path / "missing.npz")

    def test_missing_keys_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.npz"
        np.savez(p, wrong_key=np.zeros((3, 3)))
        with pytest.raises(KeyError):
            load_calibration(p)

    def test_extra_keys_ignored(self, tmp_path: Path) -> None:
        """Extra .npz keys (rvecs, tvecs) do not cause errors."""
        p = tmp_path / "full.npz"
        mtx = np.eye(3, dtype=np.float64)
        dist = np.zeros((1, 5), dtype=np.float64)
        np.savez(p, mtx=mtx, dist=dist, rvecs=np.zeros((1, 3)), tvecs=np.zeros((1, 3)))
        result = load_calibration(p)
        assert result["camera_matrix"].shape == (3, 3)
