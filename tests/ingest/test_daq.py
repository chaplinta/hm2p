"""Tests for ingest/daq.py — TDMS parsing and timestamps.h5 writing.

All tests use synthetic TDMS data generated without nptdms file I/O.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hm2p.ingest.daq import write_timestamps_h5
from hm2p.io.hdf5 import read_attrs, read_h5


def make_synthetic_timing(
    n_camera_frames: int = 6000,
    n_imaging_frames: int = 1800,
    n_light_pulses: int = 5,
    fps_camera: float = 100.0,
    fps_imaging: float = 30.0,
) -> dict[str, np.ndarray]:
    """Generate synthetic timing arrays without TDMS file I/O."""
    duration = n_camera_frames / fps_camera
    return {
        "frame_times_camera": np.linspace(0, duration, n_camera_frames, dtype=np.float64),
        "frame_times_imaging": np.linspace(0, duration, n_imaging_frames, dtype=np.float64),
        "light_on_times": np.arange(n_light_pulses, dtype=np.float64) * 120.0,
        "light_off_times": np.arange(n_light_pulses, dtype=np.float64) * 120.0 + 60.0,
        "fps_camera": np.float64(fps_camera),
        "fps_imaging": np.float64(fps_imaging),
    }


def test_write_timestamps_h5_creates_file(tmp_path: Path) -> None:
    """write_timestamps_h5 creates an HDF5 file with all expected keys."""
    arrays = make_synthetic_timing()
    output = tmp_path / "timestamps.h5"
    write_timestamps_h5(arrays, session_id="20220804_13_52_02_1117646", output_path=output)
    assert output.exists()


def test_write_timestamps_h5_shapes(tmp_path: Path) -> None:
    """Shapes in timestamps.h5 match input arrays."""
    arrays = make_synthetic_timing(n_camera_frames=6000, n_imaging_frames=1800)
    output = tmp_path / "timestamps.h5"
    write_timestamps_h5(arrays, session_id="test_session", output_path=output)

    loaded = read_h5(output)
    assert loaded["frame_times_camera"].shape == (6000,)
    assert loaded["frame_times_imaging"].shape == (1800,)


def test_write_timestamps_h5_session_id_attr(tmp_path: Path) -> None:
    """session_id is stored as a root-level HDF5 attribute."""
    arrays = make_synthetic_timing()
    output = tmp_path / "timestamps.h5"
    write_timestamps_h5(arrays, session_id="20220804_13_52_02_1117646", output_path=output)

    attrs = read_attrs(output)
    assert attrs["session_id"] == "20220804_13_52_02_1117646"


def test_write_timestamps_h5_monotonic(tmp_path: Path) -> None:
    """Camera and imaging frame timestamps are monotonically increasing."""
    arrays = make_synthetic_timing()
    output = tmp_path / "timestamps.h5"
    write_timestamps_h5(arrays, session_id="test", output_path=output)

    loaded = read_h5(output)
    assert np.all(np.diff(loaded["frame_times_camera"]) > 0)
    assert np.all(np.diff(loaded["frame_times_imaging"]) > 0)


def test_light_times_count(tmp_path: Path) -> None:
    """light_on_times and light_off_times have the same length."""
    arrays = make_synthetic_timing(n_light_pulses=5)
    output = tmp_path / "timestamps.h5"
    write_timestamps_h5(arrays, session_id="test", output_path=output)

    loaded = read_h5(output)
    assert len(loaded["light_on_times"]) == len(loaded["light_off_times"])
