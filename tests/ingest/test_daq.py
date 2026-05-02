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


# ---------------------------------------------------------------------------
# tdms_diag / line_clock_times — sync-pipeline diagnostics rollout
# ---------------------------------------------------------------------------


def make_synthetic_timing_with_diag(
    n_camera_frames: int = 600,
    n_imaging_frames: int = 180,
    y_pix: int = 162,
    sci_lines_truncated_n: int = 0,
    fps_camera: float = 100.0,
    fps_imaging: float = 30.0,
    tdms_sample_rate_hz: float = 10000.0,
) -> dict[str, np.ndarray]:
    """Synthetic timing arrays mirroring the new schema in design §2.1."""
    arrays = make_synthetic_timing(
        n_camera_frames=n_camera_frames,
        n_imaging_frames=n_imaging_frames,
        n_light_pulses=2,
        fps_camera=fps_camera,
        fps_imaging=fps_imaging,
    )
    n_lines = n_imaging_frames * y_pix + sci_lines_truncated_n
    arrays["line_clock_times"] = np.linspace(
        0, n_camera_frames / fps_camera, n_lines, dtype=np.float64
    )
    arrays["tdms_diag"] = {
        "cam_min": 0.0,
        "cam_max": 1.0,
        "sci_min": 0.0,
        "sci_max": 1.0,
        "light_min": 0.0,
        "light_max": 1.0,
        "sci_lines_truncated_n": sci_lines_truncated_n,
        "tdms_sample_rate_hz": tdms_sample_rate_hz,
        "y_pix": y_pix,
    }
    return arrays


def test_line_clock_dataset_written(tmp_path: Path) -> None:
    """line_clock_times is written as a float64 1D dataset."""
    arrays = make_synthetic_timing_with_diag()
    output = tmp_path / "timestamps.h5"
    write_timestamps_h5(arrays, session_id="test", output_path=output)
    loaded = read_h5(output, keys=["line_clock_times"])
    assert loaded["line_clock_times"].dtype == np.float64
    assert loaded["line_clock_times"].ndim == 1


def test_line_clock_length_matches_y_pix(tmp_path: Path) -> None:
    """line_clock_times has y_pix * n_imaging_frames + sci_lines_truncated_n entries."""
    arrays = make_synthetic_timing_with_diag(n_imaging_frames=10, y_pix=4, sci_lines_truncated_n=3)
    output = tmp_path / "timestamps.h5"
    write_timestamps_h5(arrays, session_id="test", output_path=output)
    loaded = read_h5(output, keys=["line_clock_times"])
    attrs = read_attrs(output)
    expected_lines = 10 * 4 + 3
    assert loaded["line_clock_times"].shape == (expected_lines,)
    assert int(attrs["tdms_diag/sci_lines_truncated_n"]) == 3


def test_tdms_diag_attrs_populated(tmp_path: Path) -> None:
    """All tdms_diag/* attrs from design §2.1 are written."""
    arrays = make_synthetic_timing_with_diag()
    output = tmp_path / "timestamps.h5"
    write_timestamps_h5(arrays, session_id="test", output_path=output)
    attrs = read_attrs(output)
    for k in (
        "tdms_diag/cam_min",
        "tdms_diag/cam_max",
        "tdms_diag/sci_min",
        "tdms_diag/sci_max",
        "tdms_diag/light_min",
        "tdms_diag/light_max",
        "tdms_diag/sci_lines_truncated_n",
        "tdms_diag/tdms_sample_rate_hz",
        "tdms_diag/y_pix",
    ):
        assert k in attrs, f"missing {k}"
        assert np.isfinite(float(attrs[k])), k


def test_tdms_sample_rate_recorded(tmp_path: Path) -> None:
    """tdms_sample_rate_hz round-trips correctly."""
    arrays = make_synthetic_timing_with_diag(tdms_sample_rate_hz=12345.0)
    output = tmp_path / "timestamps.h5"
    write_timestamps_h5(arrays, session_id="test", output_path=output)
    attrs = read_attrs(output)
    assert float(attrs["tdms_diag/tdms_sample_rate_hz"]) == pytest.approx(12345.0)


def test_truncated_lines_zero_when_divisible(tmp_path: Path) -> None:
    """When line count is divisible by y_pix, sci_lines_truncated_n == 0."""
    arrays = make_synthetic_timing_with_diag(sci_lines_truncated_n=0)
    output = tmp_path / "timestamps.h5"
    write_timestamps_h5(arrays, session_id="test", output_path=output)
    attrs = read_attrs(output)
    assert int(attrs["tdms_diag/sci_lines_truncated_n"]) == 0


def test_validates_against_diagnostic_schema(tmp_path: Path) -> None:
    """Output passes validate_timestamps_h5(require_diagnostics=True)."""
    from hm2p.io.hdf5 import validate_timestamps_h5

    arrays = make_synthetic_timing_with_diag()
    output = tmp_path / "timestamps.h5"
    write_timestamps_h5(arrays, session_id="test", output_path=output)
    loaded = read_h5(output)
    attrs = read_attrs(output)
    validate_timestamps_h5(loaded, attrs=attrs, require_diagnostics=True)


# ---------------------------------------------------------------------------
# parse_tdms fail-closed behaviour — using stubbed nptdms
# ---------------------------------------------------------------------------


class _FakeChan:
    def __init__(self, data: np.ndarray, dt: float = 1e-4):
        self.data = data
        self._dt = dt
        self.properties = {"wf_increment": dt}

    def time_track(self) -> np.ndarray:
        return np.arange(self.data.size, dtype=np.float64) * self._dt


class _FakeTdms:
    """Minimal nptdms.TdmsFile stand-in for parse_tdms()."""

    def __init__(self, group_name: str, channels: dict[str, _FakeChan]) -> None:
        self._group_name = group_name
        self._channels = channels

    def __contains__(self, key: str) -> bool:
        return key in self._channels

    def __getitem__(self, key: str) -> _FakeTdms._Group:
        return _FakeTdms._Group(self._channels[key])

    def groups(self) -> list:
        return [_FakeTdms._GroupName(name) for name in self._channels]

    @classmethod
    def read(cls, path: Path) -> _FakeTdms:  # noqa: D401
        return cls.read_static  # type: ignore[attr-defined]

    def __enter__(self) -> _FakeTdms:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    class _GroupName:
        def __init__(self, name: str):
            self.name = name

    class _Group:
        def __init__(self, chan: _FakeChan):
            self._chan = chan

        def channels(self) -> list:
            return [self._chan]


def _build_fake_tdms(
    *,
    cam_data: np.ndarray,
    sci_data: np.ndarray,
    light_data: np.ndarray,
    group_name: str = "maze-rose",
    cam_chan_name: str = "cam_trigger",
    sci_chan_name: str = "sci_sync",
    lights_chan_name: str = "lights",
    dt: float = 1e-4,
) -> _FakeTdms:
    chans = {
        f"{group_name} - {cam_chan_name}": _FakeChan(cam_data, dt=dt),
        f"{group_name} - {sci_chan_name}": _FakeChan(sci_data, dt=dt),
        f"{group_name} - {lights_chan_name}": _FakeChan(light_data, dt=dt),
    }
    return _FakeTdms(group_name, chans)


def _make_pulses(n_samples: int, edge_idxs: list[int], width: int = 3) -> np.ndarray:
    sig = np.zeros(n_samples, dtype=float)
    for s in edge_idxs:
        sig[s : s + width] = 1.0
    return sig


def _write_meta_files(session_dir: Path, group_name: str = "maze-rose", y_pix: int = 4) -> None:
    """Write meta.txt and *_XYT.ini files alongside a TDMS path."""
    meta = session_dir / "test_maze-rose.meta.txt"
    meta.write_text(
        f"[DAQ]\ngroupname = {group_name}\n"
        f"cameratriggerchanname = cam_trigger\n"
        f"sciscanchanname = sci_sync\n"
        f"lightschanname = lights\n"
        "[Video]\nfps = 100.0\n"
        "[SciScan]\ninifile = test_XYT.ini\n"
    )
    ini = session_dir / "test_XYT.ini"
    ini.write_text(f"[_]\ny.pixels = {y_pix}\nframes.p.sec = 9.645\n")


def test_parse_tdms_emits_diag(monkeypatch, tmp_path: Path) -> None:
    """parse_tdms() returns line_clock_times + tdms_diag dict."""
    session_dir = tmp_path
    tdms_path = session_dir / "test_maze-rose-di.tdms"
    tdms_path.write_bytes(b"")  # existence is the only requirement
    _write_meta_files(session_dir, y_pix=4)

    cam_data = _make_pulses(1000, [10, 20, 30, 40, 50, 60, 70, 80])
    sci_data = _make_pulses(1000, list(range(100, 196, 4)))  # 24 lines = 6 frames
    light_data = _make_pulses(1000, [200, 600])

    fake = _build_fake_tdms(cam_data=cam_data, sci_data=sci_data, light_data=light_data)
    fake_module = type("nptdms", (), {})()
    fake_module.TdmsFile = type("F", (), {"read": staticmethod(lambda p: fake)})

    monkeypatch.setitem(__import__("sys").modules, "nptdms", fake_module)

    from hm2p.ingest.daq import parse_tdms

    out = parse_tdms(tdms_path)
    assert "line_clock_times" in out
    assert out["line_clock_times"].dtype == np.float64
    assert out["tdms_diag"]["sci_lines_truncated_n"] == 0
    assert out["tdms_diag"]["y_pix"] == 4
    assert out["tdms_diag"]["tdms_sample_rate_hz"] == pytest.approx(1.0 / 1e-4)


def test_parse_tdms_empty_line_clock_raises(monkeypatch, tmp_path: Path) -> None:
    """Zero SciScan line-clock pulses is now a hard failure."""
    session_dir = tmp_path
    tdms_path = session_dir / "test_maze-rose-di.tdms"
    tdms_path.write_bytes(b"")
    _write_meta_files(session_dir, y_pix=4)

    cam_data = _make_pulses(1000, [10, 20, 30])
    sci_data = np.zeros(1000)  # NO pulses
    light_data = np.zeros(1000)

    fake = _build_fake_tdms(cam_data=cam_data, sci_data=sci_data, light_data=light_data)
    fake_module = type("nptdms", (), {})()
    fake_module.TdmsFile = type("F", (), {"read": staticmethod(lambda p: fake)})
    monkeypatch.setitem(__import__("sys").modules, "nptdms", fake_module)

    from hm2p.ingest.daq import parse_tdms

    with pytest.raises(ValueError, match="line-clock"):
        parse_tdms(tdms_path)


def test_parse_tdms_empty_cam_raises(monkeypatch, tmp_path: Path) -> None:
    """Zero camera trigger pulses still raises (legacy behaviour preserved)."""
    session_dir = tmp_path
    tdms_path = session_dir / "test_maze-rose-di.tdms"
    tdms_path.write_bytes(b"")
    _write_meta_files(session_dir, y_pix=4)

    cam_data = np.zeros(1000)
    sci_data = _make_pulses(1000, list(range(100, 196, 4)))
    light_data = np.zeros(1000)

    fake = _build_fake_tdms(cam_data=cam_data, sci_data=sci_data, light_data=light_data)
    fake_module = type("nptdms", (), {})()
    fake_module.TdmsFile = type("F", (), {"read": staticmethod(lambda p: fake)})
    monkeypatch.setitem(__import__("sys").modules, "nptdms", fake_module)

    from hm2p.ingest.daq import parse_tdms

    with pytest.raises(ValueError, match="camera"):
        parse_tdms(tdms_path)


def test_parse_tdms_truncated_lines_recorded(monkeypatch, tmp_path: Path) -> None:
    """When line count % y_pix != 0, the residual is recorded."""
    session_dir = tmp_path
    tdms_path = session_dir / "test_maze-rose-di.tdms"
    tdms_path.write_bytes(b"")
    _write_meta_files(session_dir, y_pix=4)

    cam_data = _make_pulses(1000, [10, 20, 30])
    # 26 line pulses with y_pix=4 → 6 full frames (24 lines), 2 truncated
    sci_data = _make_pulses(1000, list(range(100, 100 + 26 * 4, 4)))
    light_data = np.zeros(1000)

    fake = _build_fake_tdms(cam_data=cam_data, sci_data=sci_data, light_data=light_data)
    fake_module = type("nptdms", (), {})()
    fake_module.TdmsFile = type("F", (), {"read": staticmethod(lambda p: fake)})
    monkeypatch.setitem(__import__("sys").modules, "nptdms", fake_module)

    from hm2p.ingest.daq import parse_tdms

    out = parse_tdms(tdms_path)
    assert out["tdms_diag"]["sci_lines_truncated_n"] == 2
    assert out["frame_times_imaging"].size == 6
    assert out["line_clock_times"].size == 26


def test_parse_tdms_light_count_mismatch_tolerated(monkeypatch, tmp_path: Path) -> None:
    """light_on != light_off counts no longer raises in ingest."""
    session_dir = tmp_path
    tdms_path = session_dir / "test_maze-rose-di.tdms"
    tdms_path.write_bytes(b"")
    _write_meta_files(session_dir, y_pix=4)

    cam_data = _make_pulses(1000, [10, 20, 30])
    sci_data = _make_pulses(1000, list(range(100, 196, 4)))
    # Two on edges, one off edge → mismatched
    light_data = _make_pulses(1000, [100, 700])

    fake = _build_fake_tdms(cam_data=cam_data, sci_data=sci_data, light_data=light_data)
    fake_module = type("nptdms", (), {})()
    fake_module.TdmsFile = type("F", (), {"read": staticmethod(lambda p: fake)})
    monkeypatch.setitem(__import__("sys").modules, "nptdms", fake_module)

    from hm2p.ingest.daq import parse_tdms

    out = parse_tdms(tdms_path)  # should not raise
    assert "light_on_times" in out
    assert "light_off_times" in out


def test_parse_tdms_records_channel_min_max(monkeypatch, tmp_path: Path) -> None:
    """tdms_diag records the raw channel min/max values."""
    session_dir = tmp_path
    tdms_path = session_dir / "test_maze-rose-di.tdms"
    tdms_path.write_bytes(b"")
    _write_meta_files(session_dir, y_pix=4)

    # cam pulses must cross 0.9 to be detected, but the digital level can
    # still be sub-1.0 to test min/max recording.
    cam_data = _make_pulses(1000, [10, 20, 30])
    cam_data[cam_data > 0] = 0.95
    sci_data = _make_pulses(1000, list(range(100, 196, 4)))
    light_data = np.zeros(1000)

    fake = _build_fake_tdms(cam_data=cam_data, sci_data=sci_data, light_data=light_data)
    fake_module = type("nptdms", (), {})()
    fake_module.TdmsFile = type("F", (), {"read": staticmethod(lambda p: fake)})
    monkeypatch.setitem(__import__("sys").modules, "nptdms", fake_module)

    from hm2p.ingest.daq import parse_tdms

    out = parse_tdms(tdms_path)
    diag = out["tdms_diag"]
    assert diag["cam_max"] == pytest.approx(0.95, abs=1e-6)
    assert diag["cam_min"] == 0.0
