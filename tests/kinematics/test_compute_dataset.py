"""Tests for kinematics/compute.py — dataset-level functions.

These tests use synthetic xarray Datasets that mirror the movement library's
output format, without requiring movement to be installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from hm2p.kinematics.compute import (
    apply_orientation_rotation,
    compute_head_direction,
    compute_position_mm,
    run,
)

KEYPOINTS = ["left_ear", "right_ear", "mid_back", "mouse_center", "tail_base"]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def _make_ds(
    n_frames: int = 50,
    pos_data: np.ndarray | None = None,
    conf_data: np.ndarray | None = None,
) -> xr.Dataset:
    """Build a minimal movement-style xarray Dataset."""
    n_kp = len(KEYPOINTS)
    if pos_data is None:
        pos_data = np.ones((n_frames, 2, n_kp, 1), dtype=np.float64)
    if conf_data is None:
        conf_data = np.ones((n_frames, n_kp, 1), dtype=np.float64)

    position = xr.DataArray(
        pos_data,
        dims=["time", "space", "keypoints", "individuals"],
        coords={
            "time": np.arange(n_frames, dtype=float),
            "space": ["x", "y"],
            "keypoints": KEYPOINTS,
            "individuals": ["mouse"],
        },
    )
    confidence = xr.DataArray(
        conf_data,
        dims=["time", "keypoints", "individuals"],
        coords={
            "time": np.arange(n_frames, dtype=float),
            "keypoints": KEYPOINTS,
            "individuals": ["mouse"],
        },
    )
    return xr.Dataset({"position": position, "confidence": confidence})


# ---------------------------------------------------------------------------
# Mock movement modules for deferred imports
# ---------------------------------------------------------------------------


def _install_mock_movement() -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Install mock movement modules into sys.modules so deferred imports work.

    Returns (mock_movement, mock_io, mock_load_poses, mock_filtering).
    The actual code does ``from movement.io import load_poses`` then
    ``load_poses.from_file(file=..., source_software=...)``.
    """
    mock_movement = MagicMock()
    mock_io = MagicMock()
    mock_load_poses = MagicMock()
    mock_filtering = MagicMock()
    mock_movement.io = mock_io
    mock_movement.io.load_poses = mock_load_poses
    mock_movement.filtering = mock_filtering
    sys.modules["movement"] = mock_movement
    sys.modules["movement.io"] = mock_io
    sys.modules["movement.io.load_poses"] = mock_load_poses
    sys.modules["movement.filtering"] = mock_filtering
    return mock_movement, mock_io, mock_load_poses, mock_filtering


def _remove_mock_movement() -> None:
    """Remove mock movement modules from sys.modules."""
    for key in [
        "movement", "movement.io", "movement.io.load_poses", "movement.filtering",
    ]:
        sys.modules.pop(key, None)


# ---------------------------------------------------------------------------
# load_pose_dataset
# ---------------------------------------------------------------------------


class TestLoadPoseDataset:
    def test_unknown_tracker_raises(self, tmp_path: Path) -> None:
        from hm2p.kinematics.compute import load_pose_dataset

        _, _, mock_lp, _ = _install_mock_movement()
        try:
            with pytest.raises(ValueError, match="Unknown tracker"):
                load_pose_dataset(tmp_path / "fake.h5", "nonexistent")
        finally:
            _remove_mock_movement()

    def test_known_tracker_calls_movement(self, tmp_path: Path) -> None:
        from hm2p.kinematics.compute import load_pose_dataset

        mock_ds = _make_ds(10)
        _, _, mock_lp, _ = _install_mock_movement()
        try:
            mock_lp.from_file.return_value = mock_ds
            result = load_pose_dataset(tmp_path / "pose.h5", "dlc")
            mock_lp.from_file.assert_called_once_with(
                file=tmp_path / "pose.h5", source_software="DeepLabCut"
            )
            assert result is mock_ds
        finally:
            _remove_mock_movement()

    def test_sleap_tracker_mapping(self, tmp_path: Path) -> None:
        from hm2p.kinematics.compute import load_pose_dataset

        mock_ds = _make_ds(10)
        _, _, mock_lp, _ = _install_mock_movement()
        try:
            mock_lp.from_file.return_value = mock_ds
            load_pose_dataset(tmp_path / "pose.h5", "sleap")
            mock_lp.from_file.assert_called_once_with(
                file=tmp_path / "pose.h5", source_software="SLEAP"
            )
        finally:
            _remove_mock_movement()

    def test_lp_tracker_mapping(self, tmp_path: Path) -> None:
        from hm2p.kinematics.compute import load_pose_dataset

        mock_ds = _make_ds(10)
        _, _, mock_lp, _ = _install_mock_movement()
        try:
            mock_lp.from_file.return_value = mock_ds
            load_pose_dataset(tmp_path / "pose.csv", "lp")
            mock_lp.from_file.assert_called_once_with(
                file=tmp_path / "pose.csv", source_software="LightningPose"
            )
        finally:
            _remove_mock_movement()


# ---------------------------------------------------------------------------
# filter_low_confidence
# ---------------------------------------------------------------------------


class TestFilterLowConfidence:
    def test_calls_movement_filter(self) -> None:
        from hm2p.kinematics.compute import filter_low_confidence

        ds = _make_ds(10)
        _, _, _, mock_filtering = _install_mock_movement()
        try:
            mock_filtering.filter_by_confidence.return_value = ds.position
            result = filter_low_confidence(ds, threshold=0.9)
            mock_filtering.filter_by_confidence.assert_called_once()
            assert "position" in result.data_vars
        finally:
            _remove_mock_movement()

    def test_threshold_passed(self) -> None:
        from hm2p.kinematics.compute import filter_low_confidence

        ds = _make_ds(10)
        _, _, _, mock_filtering = _install_mock_movement()
        try:
            mock_filtering.filter_by_confidence.return_value = ds.position
            filter_low_confidence(ds, threshold=0.5)
            _, kwargs = mock_filtering.filter_by_confidence.call_args
            assert kwargs["threshold"] == 0.5
        finally:
            _remove_mock_movement()


# ---------------------------------------------------------------------------
# interpolate_gaps
# ---------------------------------------------------------------------------


class TestInterpolateGaps:
    def test_calls_movement_interpolation(self) -> None:
        from hm2p.kinematics.compute import interpolate_gaps

        ds = _make_ds(10)
        _, _, _, mock_filtering = _install_mock_movement()
        try:
            mock_filtering.interpolate_over_time.return_value = ds.position
            result = interpolate_gaps(ds, max_gap_frames=5)
            mock_filtering.interpolate_over_time.assert_called_once()
            _, kwargs = mock_filtering.interpolate_over_time.call_args
            assert kwargs["method"] == "linear"
            assert kwargs["max_gap"] == 5
            assert "position" in result.data_vars
        finally:
            _remove_mock_movement()


# ---------------------------------------------------------------------------
# apply_orientation_rotation
# ---------------------------------------------------------------------------


class TestApplyOrientationRotation:
    def test_zero_rotation_unchanged(self) -> None:
        ds = _make_ds(20)
        result = apply_orientation_rotation(ds, 0.0)
        xr.testing.assert_identical(result, ds)

    def test_rotation_changes_positions(self) -> None:
        ds = _make_ds(10)
        pos = ds.position.values.copy()
        pos[:, 0, :, :] = np.linspace(1, 10, 10)[:, None, None]
        pos[:, 1, :, :] = np.linspace(1, 5, 10)[:, None, None]
        ds = ds.assign(
            position=xr.DataArray(pos, dims=ds.position.dims, coords=ds.position.coords)
        )
        rotated = apply_orientation_rotation(ds, 45.0)
        assert not np.allclose(rotated.position.values, ds.position.values)

    def test_360_rotation_roundtrip(self) -> None:
        ds = _make_ds(10)
        pos = ds.position.values.copy()
        pos[:, 0, :, :] = np.linspace(1, 10, 10)[:, None, None]
        pos[:, 1, :, :] = np.linspace(1, 5, 10)[:, None, None]
        ds = ds.assign(
            position=xr.DataArray(pos, dims=ds.position.dims, coords=ds.position.coords)
        )
        result = apply_orientation_rotation(ds, 360.0)
        np.testing.assert_allclose(result.position.values, ds.position.values, atol=1e-10)


# ---------------------------------------------------------------------------
# compute_head_direction (dataset-level)
# ---------------------------------------------------------------------------


class TestComputeHeadDirectionDataset:
    def test_shape_matches_time(self) -> None:
        ds = _make_ds(30)
        hd = compute_head_direction(ds)
        assert hd.shape == (30,)

    def test_dtype_float32(self) -> None:
        ds = _make_ds(10)
        hd = compute_head_direction(ds)
        assert hd.dtype == np.float32


# ---------------------------------------------------------------------------
# compute_position_mm (dataset-level)
# ---------------------------------------------------------------------------


class TestComputePositionMmDataset:
    def test_shape(self) -> None:
        ds = _make_ds(25)
        x, y = compute_position_mm(ds, scale_mm_per_px=0.8)
        assert x.shape == (25,)
        assert y.shape == (25,)

    def test_dtype(self) -> None:
        ds = _make_ds(10)
        x, y = compute_position_mm(ds, scale_mm_per_px=1.0)
        assert x.dtype == np.float32
        assert y.dtype == np.float32


# ---------------------------------------------------------------------------
# run() — end-to-end Stage 3 (mocked movement)
# ---------------------------------------------------------------------------


class TestKinematicsRun:
    def _write_timestamps(self, path: Path, n_cam: int = 500) -> None:
        from hm2p.io.hdf5 import write_h5

        duration = n_cam / 100.0
        write_h5(
            path,
            {
                "frame_times_camera": np.linspace(0, duration, n_cam, dtype=np.float64),
                "frame_times_imaging": np.linspace(0, duration, 150, dtype=np.float64),
                "light_on_times": np.array([0.0, 2.0], dtype=np.float64),
                "light_off_times": np.array([1.0, 3.0], dtype=np.float64),
            },
        )

    def _make_pose_ds(self, n_frames: int = 500) -> xr.Dataset:
        pos_data = np.zeros((n_frames, 2, len(KEYPOINTS), 1), dtype=np.float64)
        kp_idx = {k: i for i, k in enumerate(KEYPOINTS)}

        t = np.linspace(0, 4 * np.pi, n_frames)
        pos_data[:, 0, kp_idx["left_ear"], 0] = 400 + 10 * np.cos(t)
        pos_data[:, 1, kp_idx["left_ear"], 0] = 300 + 10 * np.sin(t)
        pos_data[:, 0, kp_idx["right_ear"], 0] = 400 - 10 * np.cos(t)
        pos_data[:, 1, kp_idx["right_ear"], 0] = 300 - 10 * np.sin(t)
        for kp in ["mid_back", "mouse_center", "tail_base"]:
            pos_data[:, 0, kp_idx[kp], 0] = np.linspace(200, 600, n_frames)
            pos_data[:, 1, kp_idx[kp], 0] = np.linspace(100, 400, n_frames)

        conf_data = np.ones((n_frames, len(KEYPOINTS), 1), dtype=np.float64)
        return _make_ds(n_frames, pos_data, conf_data)

    def _setup_mocks(self, pose_ds: xr.Dataset) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
        _, mock_io, mock_lp, mock_filtering = _install_mock_movement()
        mock_lp.from_file.return_value = pose_ds
        mock_filtering.filter_by_confidence.return_value = pose_ds.position
        mock_filtering.interpolate_over_time.return_value = pose_ds.position
        return _, mock_io, mock_lp, mock_filtering

    def test_run_creates_output(self, tmp_path: Path) -> None:
        ts_h5 = tmp_path / "timestamps.h5"
        self._write_timestamps(ts_h5, n_cam=500)
        out_h5 = tmp_path / "kinematics.h5"
        pose_ds = self._make_pose_ds(500)
        corners = np.array([[149, 72], [764, 82], [757, 509], [143, 500]], dtype=np.float64)

        self._setup_mocks(pose_ds)
        try:
            run(
                pose_path=tmp_path / "fake_pose.h5",
                timestamps_h5=ts_h5,
                session_id="test_session",
                tracker="dlc",
                orientation_deg=0.0,
                scale_mm_per_px=0.811,
                maze_corners_px=corners,
                bad_behav_intervals=[(1.0, 2.0)],
                output_path=out_h5,
            )
            assert out_h5.exists()
        finally:
            _remove_mock_movement()

    def test_run_output_keys(self, tmp_path: Path) -> None:
        from hm2p.io.hdf5 import read_attrs, read_h5

        ts_h5 = tmp_path / "timestamps.h5"
        self._write_timestamps(ts_h5, n_cam=500)
        out_h5 = tmp_path / "kinematics.h5"
        pose_ds = self._make_pose_ds(500)
        corners = np.array([[149, 72], [764, 82], [757, 509], [143, 500]], dtype=np.float64)

        self._setup_mocks(pose_ds)
        try:
            run(
                pose_path=tmp_path / "fake_pose.h5",
                timestamps_h5=ts_h5,
                session_id="test_session",
                tracker="dlc",
                orientation_deg=15.0,
                scale_mm_per_px=0.811,
                maze_corners_px=corners,
                bad_behav_intervals=[],
                output_path=out_h5,
            )

            data = read_h5(out_h5)
            expected_keys = {
                "frame_times",
                "hd_deg",
                "x_mm",
                "y_mm",
                "x_maze",
                "y_maze",
                "speed_cm_s",
                "ahv_deg_s",
                "active",
                "light_on",
                "bad_behav",
            }
            assert expected_keys.issubset(set(data.keys()))

            attrs = read_attrs(out_h5)
            assert attrs["session_id"] == "test_session"
            assert attrs["tracker"] == "dlc"
        finally:
            _remove_mock_movement()

    def test_run_output_lengths(self, tmp_path: Path) -> None:
        from hm2p.io.hdf5 import read_h5

        n_cam = 500
        ts_h5 = tmp_path / "timestamps.h5"
        self._write_timestamps(ts_h5, n_cam=n_cam)
        out_h5 = tmp_path / "kinematics.h5"
        pose_ds = self._make_pose_ds(n_cam)
        corners = np.array([[149, 72], [764, 82], [757, 509], [143, 500]], dtype=np.float64)

        self._setup_mocks(pose_ds)
        try:
            run(
                pose_path=tmp_path / "fake_pose.h5",
                timestamps_h5=ts_h5,
                session_id="test",
                tracker="dlc",
                orientation_deg=0.0,
                scale_mm_per_px=0.811,
                maze_corners_px=corners,
                bad_behav_intervals=[],
                output_path=out_h5,
            )

            data = read_h5(out_h5)
            assert len(data["hd_deg"]) == n_cam
            assert len(data["speed_cm_s"]) == n_cam
            assert len(data["active"]) == n_cam
            assert len(data["light_on"]) == n_cam
        finally:
            _remove_mock_movement()
