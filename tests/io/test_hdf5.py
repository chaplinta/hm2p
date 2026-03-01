"""Tests for io/hdf5.py — HDF5 read/write and schema validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hm2p.io.hdf5 import (
    read_attrs,
    read_h5,
    validate_ca_h5,
    validate_kinematics_h5,
    validate_sync_h5,
    validate_timestamps_h5,
    write_h5,
)

# ---------------------------------------------------------------------------
# write_h5 / read_h5
# ---------------------------------------------------------------------------


def test_write_and_read_roundtrip(tmp_path: Path, rng: np.random.Generator) -> None:
    """Arrays written by write_h5 are read back identically by read_h5."""
    arrays = {
        "frame_times": rng.random(1000).astype(np.float64),
        "hd": rng.standard_normal(1000).astype(np.float32),
        "light_on": rng.integers(0, 2, 1000).astype(bool),
    }
    path = tmp_path / "test.h5"
    write_h5(path, arrays)
    loaded = read_h5(path)
    for key in arrays:
        np.testing.assert_array_equal(loaded[key], arrays[key])


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    """write_h5 creates parent directories if they don't exist."""
    path = tmp_path / "deep" / "nested" / "output.h5"
    write_h5(path, {"x": np.array([1.0, 2.0])})
    assert path.exists()


def test_read_h5_file_not_found(tmp_path: Path) -> None:
    """read_h5 raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        read_h5(tmp_path / "nonexistent.h5")


def test_read_h5_selected_keys(tmp_path: Path, rng: np.random.Generator) -> None:
    """read_h5 returns only requested keys when keys= is specified."""
    arrays = {
        "a": rng.random(100),
        "b": rng.random(100),
        "c": rng.random(100),
    }
    path = tmp_path / "test.h5"
    write_h5(path, arrays)
    loaded = read_h5(path, keys=["a", "c"])
    assert set(loaded.keys()) == {"a", "c"}
    assert "b" not in loaded


def test_attrs_roundtrip(tmp_path: Path) -> None:
    """Root-level HDF5 attributes are written and read back correctly."""
    path = tmp_path / "test.h5"
    write_h5(
        path,
        arrays={"x": np.array([1.0])},
        attrs={"session_id": "20220804_13_52_02_1117646", "fps_camera": 100.0},
    )
    attrs = read_attrs(path)
    assert attrs["session_id"] == "20220804_13_52_02_1117646"
    assert attrs["fps_camera"] == pytest.approx(100.0)


def test_write_h5_overwrites_existing(tmp_path: Path) -> None:
    """write_h5 silently overwrites an existing file."""
    path = tmp_path / "test.h5"
    write_h5(path, {"x": np.array([1.0, 2.0, 3.0])})
    write_h5(path, {"y": np.array([7.0, 8.0])})
    loaded = read_h5(path)
    assert "y" in loaded
    assert "x" not in loaded  # old content gone


# ---------------------------------------------------------------------------
# Helpers to build valid synthetic arrays for each HDF5 schema
# ---------------------------------------------------------------------------


def _valid_timestamps(n_cam: int = 600, n_img: int = 180) -> dict:
    """Return a dict of valid timestamps.h5 arrays."""
    return {
        "frame_times_camera": np.linspace(0.0, 6.0, n_cam, dtype=np.float64),
        "frame_times_imaging": np.linspace(0.0, 6.0, n_img, dtype=np.float64),
        "light_on_times": np.array([0.0, 60.0, 120.0], dtype=np.float64),
        "light_off_times": np.array([60.0, 120.0, 180.0], dtype=np.float64),
    }


def _valid_kinematics(T: int = 180) -> dict:
    """Return a dict of valid kinematics.h5 arrays."""
    return {
        "frame_times": np.linspace(0.0, 6.0, T, dtype=np.float64),
        "hd_deg": np.zeros(T, dtype=np.float32),
        "x_mm": np.zeros(T, dtype=np.float32),
        "y_mm": np.zeros(T, dtype=np.float32),
        "speed_cm_s": np.ones(T, dtype=np.float32) * 2.0,
        "ahv_deg_s": np.zeros(T, dtype=np.float32),
        "active": np.ones(T, dtype=bool),
        "light_on": np.zeros(T, dtype=bool),
        "bad_behav": np.zeros(T, dtype=bool),
    }


def _valid_ca(n_rois: int = 10, T: int = 180) -> dict:
    """Return a dict of valid ca.h5 arrays."""
    return {
        "frame_times": np.linspace(0.0, 6.0, T, dtype=np.float64),
        "dff": np.zeros((n_rois, T), dtype=np.float32),
    }


def _valid_sync(n_rois: int = 10, T: int = 180) -> dict:
    """Return a dict of valid sync.h5 arrays (kinematics + ca merged)."""
    arrays = _valid_kinematics(T)
    arrays["dff"] = np.zeros((n_rois, T), dtype=np.float32)
    return arrays


# ---------------------------------------------------------------------------
# validate_timestamps_h5
# ---------------------------------------------------------------------------


class TestValidateTimestampsH5:
    def test_valid_passes(self) -> None:
        """Valid timestamps.h5 dict raises no error."""
        validate_timestamps_h5(_valid_timestamps())

    def test_missing_key_raises(self) -> None:
        """Missing required key raises SchemaError."""
        from pandera.errors import SchemaError

        arrays = _valid_timestamps()
        del arrays["frame_times_camera"]
        with pytest.raises(SchemaError, match="frame_times_camera"):
            validate_timestamps_h5(arrays)

    def test_wrong_dtype_raises(self) -> None:
        """float32 timestamps raises SchemaError (must be float64)."""
        from pandera.errors import SchemaError

        arrays = _valid_timestamps()
        arrays["frame_times_camera"] = arrays["frame_times_camera"].astype(np.float32)
        with pytest.raises(SchemaError, match="float64"):
            validate_timestamps_h5(arrays)

    def test_non_monotonic_camera_raises(self) -> None:
        """Non-monotonic frame_times_camera raises SchemaError."""
        from pandera.errors import SchemaError

        arrays = _valid_timestamps()
        ft = arrays["frame_times_camera"].copy()
        ft[10] = ft[5]  # duplicate → not strictly increasing
        arrays["frame_times_camera"] = ft
        with pytest.raises(SchemaError, match="strictly increasing"):
            validate_timestamps_h5(arrays)

    def test_non_monotonic_imaging_raises(self) -> None:
        """Non-monotonic frame_times_imaging raises SchemaError."""
        from pandera.errors import SchemaError

        arrays = _valid_timestamps()
        ft = arrays["frame_times_imaging"].copy()
        ft[-1] = ft[-2]
        arrays["frame_times_imaging"] = ft
        with pytest.raises(SchemaError, match="strictly increasing"):
            validate_timestamps_h5(arrays)

    def test_light_times_not_required_to_be_monotonic(self) -> None:
        """light_on_times with any order is accepted (events can be unsorted)."""
        arrays = _valid_timestamps()
        arrays["light_on_times"] = np.array([120.0, 0.0, 60.0], dtype=np.float64)
        validate_timestamps_h5(arrays)  # should not raise


# ---------------------------------------------------------------------------
# validate_kinematics_h5
# ---------------------------------------------------------------------------


class TestValidateKinematicsH5:
    def test_valid_passes(self) -> None:
        """Valid kinematics.h5 dict raises no error."""
        validate_kinematics_h5(_valid_kinematics())

    def test_missing_hd_deg_raises(self) -> None:
        from pandera.errors import SchemaError

        arrays = _valid_kinematics()
        del arrays["hd_deg"]
        with pytest.raises(SchemaError, match="hd_deg"):
            validate_kinematics_h5(arrays)

    def test_wrong_dtype_speed_raises(self) -> None:
        from pandera.errors import SchemaError

        arrays = _valid_kinematics()
        arrays["speed_cm_s"] = arrays["speed_cm_s"].astype(np.float64)
        with pytest.raises(SchemaError, match="float32"):
            validate_kinematics_h5(arrays)

    def test_negative_speed_raises(self) -> None:
        from pandera.errors import SchemaError

        arrays = _valid_kinematics()
        arrays["speed_cm_s"][5] = -1.0
        with pytest.raises(SchemaError, match="speed_cm_s"):
            validate_kinematics_h5(arrays)

    def test_wrong_length_raises(self) -> None:
        """Array with length != len(frame_times) raises SchemaError."""
        from pandera.errors import SchemaError

        arrays = _valid_kinematics(T=180)
        arrays["x_mm"] = np.zeros(100, dtype=np.float32)  # wrong length
        with pytest.raises(SchemaError, match="x_mm"):
            validate_kinematics_h5(arrays)

    def test_bool_as_int_raises(self) -> None:
        """active as int8 (not bool) raises SchemaError."""
        from pandera.errors import SchemaError

        arrays = _valid_kinematics()
        arrays["active"] = arrays["active"].astype(np.int8)
        with pytest.raises(SchemaError, match="bool"):
            validate_kinematics_h5(arrays)

    def test_non_monotonic_frame_times_raises(self) -> None:
        from pandera.errors import SchemaError

        arrays = _valid_kinematics()
        ft = arrays["frame_times"].copy()
        ft[50] = ft[49]
        arrays["frame_times"] = ft
        with pytest.raises(SchemaError, match="strictly increasing"):
            validate_kinematics_h5(arrays)


# ---------------------------------------------------------------------------
# validate_ca_h5
# ---------------------------------------------------------------------------


class TestValidateCaH5:
    def test_valid_passes(self) -> None:
        """Valid ca.h5 dict raises no error."""
        validate_ca_h5(_valid_ca())

    def test_missing_dff_raises(self) -> None:
        from pandera.errors import SchemaError

        arrays = _valid_ca()
        del arrays["dff"]
        with pytest.raises(SchemaError, match="dff"):
            validate_ca_h5(arrays)

    def test_dff_wrong_dtype_raises(self) -> None:
        from pandera.errors import SchemaError

        arrays = _valid_ca()
        arrays["dff"] = arrays["dff"].astype(np.float64)
        with pytest.raises(SchemaError, match="float32"):
            validate_ca_h5(arrays)

    def test_dff_wrong_ndim_raises(self) -> None:
        """1D dff raises SchemaError (must be 2D)."""
        from pandera.errors import SchemaError

        arrays = _valid_ca(T=180)
        arrays["dff"] = np.zeros(180, dtype=np.float32)
        with pytest.raises(SchemaError, match="2D"):
            validate_ca_h5(arrays)

    def test_dff_wrong_n_frames_raises(self) -> None:
        """dff with n_frames != len(frame_times) raises SchemaError."""
        from pandera.errors import SchemaError

        arrays = _valid_ca(n_rois=5, T=180)
        arrays["dff"] = np.zeros((5, 100), dtype=np.float32)  # wrong n_frames
        with pytest.raises(SchemaError):
            validate_ca_h5(arrays)

    def test_optional_spikes_valid(self) -> None:
        """Optional 'spikes' array with correct dtype/shape passes."""
        arrays = _valid_ca(n_rois=5, T=180)
        arrays["spikes"] = np.zeros((5, 180), dtype=np.float32)
        validate_ca_h5(arrays)  # should not raise

    def test_optional_spikes_wrong_shape_raises(self) -> None:
        """'spikes' with wrong shape raises SchemaError."""
        from pandera.errors import SchemaError

        arrays = _valid_ca(n_rois=5, T=180)
        arrays["spikes"] = np.zeros((3, 180), dtype=np.float32)  # wrong n_rois
        with pytest.raises(SchemaError):
            validate_ca_h5(arrays)


# ---------------------------------------------------------------------------
# validate_sync_h5
# ---------------------------------------------------------------------------


class TestValidateSyncH5:
    def test_valid_passes(self) -> None:
        """Valid sync.h5 dict raises no error."""
        validate_sync_h5(_valid_sync())

    def test_missing_dff_raises(self) -> None:
        from pandera.errors import SchemaError

        arrays = _valid_sync()
        del arrays["dff"]
        with pytest.raises(SchemaError, match="dff"):
            validate_sync_h5(arrays)

    def test_missing_kinematics_key_raises(self) -> None:
        """Missing kinematics key (e.g. bad_behav) raises SchemaError."""
        from pandera.errors import SchemaError

        arrays = _valid_sync()
        del arrays["bad_behav"]
        with pytest.raises(SchemaError, match="bad_behav"):
            validate_sync_h5(arrays)

    def test_dff_wrong_n_frames_raises(self) -> None:
        from pandera.errors import SchemaError

        arrays = _valid_sync(n_rois=5, T=180)
        arrays["dff"] = np.zeros((5, 50), dtype=np.float32)  # wrong T
        with pytest.raises(SchemaError):
            validate_sync_h5(arrays)
