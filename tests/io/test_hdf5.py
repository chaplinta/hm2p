"""Tests for io/hdf5.py — HDF5 read/write and schema validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hm2p.io.hdf5 import read_attrs, read_h5, write_h5

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
