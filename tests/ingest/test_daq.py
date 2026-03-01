"""Tests for ingest/daq.py — TDMS parsing and timestamps.h5 writing.

All tests use synthetic data generated without TDMS file I/O.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.ingest.daq import (
    _frame_times_from_line_clock,
    _meta_txt_path,
    _rising_edges,
    write_timestamps_h5,
)
from hm2p.io.hdf5 import read_attrs, read_h5


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

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


def _make_pulse_signal(
    n_samples: int,
    pulse_starts: np.ndarray,
    pulse_width: int = 3,
) -> np.ndarray:
    """Build a 0/1 pulse train for testing rising edge detection."""
    sig = np.zeros(n_samples, dtype=float)
    for s in pulse_starts:
        sig[int(s) : int(s) + pulse_width] = 1.0
    return sig


# ---------------------------------------------------------------------------
# _rising_edges
# ---------------------------------------------------------------------------

class TestRisingEdges:
    def test_single_pulse(self) -> None:
        sig = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        idxs = _rising_edges(sig, 0.5)
        assert list(idxs) == [2]

    def test_no_pulses(self) -> None:
        sig = np.zeros(100)
        assert _rising_edges(sig, 0.5).size == 0

    def test_constant_high(self) -> None:
        sig = np.ones(50)
        assert _rising_edges(sig, 0.5).size == 0

    def test_multiple_pulses(self) -> None:
        pulse_starts = np.array([10, 30, 50])
        sig = _make_pulse_signal(70, pulse_starts, pulse_width=5)
        idxs = _rising_edges(sig, 0.5)
        np.testing.assert_array_equal(idxs, pulse_starts)

    def test_threshold_0_9_ignores_low(self) -> None:
        """Threshold=0.9 should not trigger on 0→0.5 transitions."""
        sig = np.array([0.0, 0.5, 0.5, 1.0, 0.0])
        idxs = _rising_edges(sig, 0.9)
        assert list(idxs) == [3]

    def test_returns_int_array(self) -> None:
        sig = np.array([0.0, 1.0, 0.0])
        idxs = _rising_edges(sig, 0.5)
        assert idxs.dtype.kind == "i" or idxs.dtype.kind == "u"

    @given(
        n=st.integers(min_value=2, max_value=500),
        threshold=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=100)
    def test_indices_in_bounds(self, n: int, threshold: float) -> None:
        rng = np.random.default_rng(42)
        sig = rng.integers(0, 2, size=n).astype(float)
        idxs = _rising_edges(sig, threshold)
        assert np.all(idxs >= 1)
        assert np.all(idxs < n)

    @given(n=st.integers(min_value=2, max_value=300))
    def test_alternating_all_rising(self, n: int) -> None:
        """0,1,0,1,... signal has exactly floor(n/2) rising edges."""
        sig = np.tile([0.0, 1.0], n)[:n]
        idxs = _rising_edges(sig, 0.5)
        expected = n // 2
        assert len(idxs) == expected


# ---------------------------------------------------------------------------
# _frame_times_from_line_clock
# ---------------------------------------------------------------------------

class TestFrameTimesFromLineClock:
    def test_basic(self) -> None:
        line_times = np.arange(12, dtype=float)
        frames = _frame_times_from_line_clock(line_times, y_pix=4)
        np.testing.assert_array_equal(frames, [3.0, 7.0, 11.0])

    def test_single_frame(self) -> None:
        line_times = np.arange(10, dtype=float)
        frames = _frame_times_from_line_clock(line_times, y_pix=10)
        assert len(frames) == 1
        assert frames[0] == 9.0

    def test_remainder_lines_ignored(self) -> None:
        """Extra lines at the end (incomplete frame) are ignored."""
        line_times = np.arange(11, dtype=float)  # 11 lines, y_pix=4 → 2 full frames
        frames = _frame_times_from_line_clock(line_times, y_pix=4)
        assert len(frames) == 2

    def test_realistic_rates(self) -> None:
        """At 10 kHz DAQ, 162 y_pix, ~9.6 fps → frame interval ~104 ms."""
        sf = 10_000  # 10 kHz
        y_pix = 162
        fps_imaging = 9.645
        n_frames = 100
        n_lines = n_frames * y_pix
        line_period = 1.0 / (fps_imaging * y_pix)
        line_times = np.arange(n_lines) * line_period
        frames = _frame_times_from_line_clock(line_times, y_pix=y_pix)
        assert len(frames) == n_frames
        expected_interval = 1.0 / fps_imaging
        intervals = np.diff(frames)
        np.testing.assert_allclose(intervals, expected_interval, rtol=1e-6)

    @given(
        n_frames=st.integers(min_value=1, max_value=200),
        y_pix=st.integers(min_value=16, max_value=512),
    )
    def test_frame_count_property(self, n_frames: int, y_pix: int) -> None:
        line_times = np.arange(n_frames * y_pix, dtype=float)
        frames = _frame_times_from_line_clock(line_times, y_pix=y_pix)
        assert len(frames) == n_frames


# ---------------------------------------------------------------------------
# _meta_txt_path
# ---------------------------------------------------------------------------

class TestMetaTxtPath:
    def test_di_suffix(self) -> None:
        p = Path("/data/ses/20210823_17_00_04_1114353_maze-rose-di.tdms")
        expected = Path("/data/ses/20210823_17_00_04_1114353_maze-rose.meta.txt")
        assert _meta_txt_path(p) == expected

    def test_non_di_suffix(self) -> None:
        p = Path("/data/ses/some_file.tdms")
        result = _meta_txt_path(p)
        assert result.suffix == ".txt"
        assert result.name.endswith(".meta.txt")

    def test_same_directory(self) -> None:
        p = Path("/a/b/c/file-di.tdms")
        assert _meta_txt_path(p).parent == Path("/a/b/c")


# ---------------------------------------------------------------------------
# write_timestamps_h5
# ---------------------------------------------------------------------------

def test_write_timestamps_h5_creates_file(tmp_path: Path) -> None:
    arrays = make_synthetic_timing()
    output = tmp_path / "timestamps.h5"
    write_timestamps_h5(arrays, session_id="20220804_13_52_02_1117646", output_path=output)
    assert output.exists()


def test_write_timestamps_h5_shapes(tmp_path: Path) -> None:
    arrays = make_synthetic_timing(n_camera_frames=6000, n_imaging_frames=1800)
    output = tmp_path / "timestamps.h5"
    write_timestamps_h5(arrays, session_id="test_session", output_path=output)

    loaded = read_h5(output)
    assert loaded["frame_times_camera"].shape == (6000,)
    assert loaded["frame_times_imaging"].shape == (1800,)


def test_write_timestamps_h5_session_id_attr(tmp_path: Path) -> None:
    arrays = make_synthetic_timing()
    output = tmp_path / "timestamps.h5"
    write_timestamps_h5(arrays, session_id="20220804_13_52_02_1117646", output_path=output)

    attrs = read_attrs(output)
    assert attrs["session_id"] == "20220804_13_52_02_1117646"


def test_write_timestamps_h5_monotonic(tmp_path: Path) -> None:
    arrays = make_synthetic_timing()
    output = tmp_path / "timestamps.h5"
    write_timestamps_h5(arrays, session_id="test", output_path=output)

    loaded = read_h5(output)
    assert np.all(np.diff(loaded["frame_times_camera"]) > 0)
    assert np.all(np.diff(loaded["frame_times_imaging"]) > 0)


def test_light_times_count(tmp_path: Path) -> None:
    arrays = make_synthetic_timing(n_light_pulses=5)
    output = tmp_path / "timestamps.h5"
    write_timestamps_h5(arrays, session_id="test", output_path=output)

    loaded = read_h5(output)
    assert len(loaded["light_on_times"]) == len(loaded["light_off_times"])


def test_fps_stored_as_attrs(tmp_path: Path) -> None:
    """fps_camera and fps_imaging are stored as HDF5 attributes, not datasets."""
    arrays = make_synthetic_timing(fps_camera=100.0, fps_imaging=9.645)
    output = tmp_path / "timestamps.h5"
    write_timestamps_h5(arrays, session_id="test", output_path=output)

    attrs = read_attrs(output)
    assert pytest.approx(attrs["fps_camera"]) == 100.0
    assert pytest.approx(attrs["fps_imaging"]) == 9.645

    loaded = read_h5(output)
    assert "fps_camera" not in loaded
    assert "fps_imaging" not in loaded
