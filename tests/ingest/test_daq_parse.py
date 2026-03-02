"""Tests for ingest/daq.py — parse_tdms with mocked nptdms.

These tests mock the nptdms library to test parse_tdms() end-to-end
without requiring actual TDMS files.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hm2p.ingest.daq import parse_tdms, run

# ---------------------------------------------------------------------------
# Helpers: mock TDMS infrastructure
# ---------------------------------------------------------------------------


def _write_meta_txt(
    directory: Path,
    session_name: str = "20210823_17_00_04_1114353_maze-rose",
) -> Path:
    """Write a minimal meta.txt file."""
    meta_path = directory / f"{session_name}.meta.txt"
    meta_path.write_text(
        "[DAQ]\n"
        "groupname = maze-rose\n"
        "cameratriggerchanname = cam_trigger\n"
        "sciscanchanname = sci_sync\n"
        "lightschanname = lights\n"
        "\n"
        "[Video]\n"
        "fps = 100.0\n"
        "\n"
        "[SciScan]\n"
        f"inifile = {session_name}_XYT.ini\n"
    )
    return meta_path


def _write_ini_file(
    directory: Path,
    session_name: str = "20210823_17_00_04_1114353_maze-rose",
    y_pixels: int = 162,
    fps: float = 9.645,
) -> Path:
    """Write a minimal SciScan .ini file."""
    ini_path = directory / f"{session_name}_XYT.ini"
    ini_path.write_text(f"[_]\ny.pixels = {y_pixels}\nframes.p.sec = {fps}\n")
    return ini_path


def _mock_tdms_channel(data: np.ndarray, sf: float = 10000.0) -> MagicMock:
    """Create a mock nptdms channel."""
    chan = MagicMock()
    chan.data = data
    chan.time_track.return_value = np.arange(len(data)) / sf
    return chan


def _mock_tdms_file(
    n_cam_pulses: int = 100,
    n_sci_pulses: int = 500,
    sf: float = 10000.0,
) -> MagicMock:
    """Build a mock nptdms.TdmsFile with realistic pulse data."""
    total_samples = int(sf * 10)  # 10 seconds of data

    # Camera trigger: pulses at ~100 Hz
    cam_data = np.zeros(total_samples)
    cam_interval = int(sf / 100)
    for i in range(min(n_cam_pulses, total_samples // cam_interval)):
        start = i * cam_interval
        cam_data[start : start + 3] = 1.0

    # SciScan line clock: pulses at ~1600 Hz
    sci_data = np.zeros(total_samples)
    sci_interval = max(1, int(sf / 1600))
    for i in range(min(n_sci_pulses, total_samples // sci_interval)):
        start = i * sci_interval
        sci_data[start : start + 2] = 1.0

    # Light: on for 5s, off for 5s
    light_data = np.zeros(total_samples)
    light_data[: int(sf * 5)] = 1.0

    cam_chan = _mock_tdms_channel(cam_data, sf)
    sci_chan = _mock_tdms_channel(sci_data, sf)
    light_chan = _mock_tdms_channel(light_data, sf)

    groups = {
        "maze-rose - cam_trigger": MagicMock(channels=MagicMock(return_value=[cam_chan])),
        "maze-rose - sci_sync": MagicMock(channels=MagicMock(return_value=[sci_chan])),
        "maze-rose - lights": MagicMock(channels=MagicMock(return_value=[light_chan])),
    }

    mock_file = MagicMock()
    mock_file.__contains__ = lambda self, key: key in groups
    mock_file.__getitem__ = lambda self, key: groups[key]
    mock_file.groups.return_value = [MagicMock(name=k) for k in groups]
    mock_file.__enter__ = MagicMock(return_value=mock_file)
    mock_file.__exit__ = MagicMock(return_value=False)

    return mock_file


# ---------------------------------------------------------------------------
# parse_tdms tests
# ---------------------------------------------------------------------------


class TestParseTdms:
    def test_missing_tdms_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="TDMS"):
            parse_tdms(tmp_path / "nonexistent-di.tdms")

    def test_missing_meta_txt_raises(self, tmp_path: Path) -> None:
        tdms_path = tmp_path / "20210823_17_00_04_1114353_maze-rose-di.tdms"
        tdms_path.write_text("")
        with pytest.raises(FileNotFoundError, match="meta.txt"):
            parse_tdms(tdms_path)

    def test_missing_ini_raises(self, tmp_path: Path) -> None:
        """If ini file referenced in meta.txt is missing, raises FileNotFoundError."""
        session_name = "20210823_17_00_04_1114353_maze-rose"
        tdms_path = tmp_path / f"{session_name}-di.tdms"
        tdms_path.write_text("")
        _write_meta_txt(tmp_path, session_name)

        with pytest.raises(FileNotFoundError, match="XYT.ini"):
            parse_tdms(tdms_path)

    def test_returns_expected_keys(self, tmp_path: Path) -> None:
        session_name = "20210823_17_00_04_1114353_maze-rose"
        tdms_path = tmp_path / f"{session_name}-di.tdms"
        tdms_path.write_text("")
        _write_meta_txt(tmp_path, session_name)
        _write_ini_file(tmp_path, session_name, y_pixels=162, fps=9.645)

        mock_file = _mock_tdms_file()

        with patch("nptdms.TdmsFile.read", return_value=mock_file):
            result = parse_tdms(tdms_path)

        expected_keys = {
            "frame_times_camera",
            "frame_times_imaging",
            "light_on_times",
            "light_off_times",
            "fps_camera",
            "fps_imaging",
        }
        assert expected_keys == set(result.keys())

    def test_frame_times_camera_dtype(self, tmp_path: Path) -> None:
        session_name = "20210823_17_00_04_1114353_maze-rose"
        tdms_path = tmp_path / f"{session_name}-di.tdms"
        tdms_path.write_text("")
        _write_meta_txt(tmp_path, session_name)
        _write_ini_file(tmp_path, session_name)

        with patch("nptdms.TdmsFile.read", return_value=_mock_tdms_file()):
            result = parse_tdms(tdms_path)

        assert result["frame_times_camera"].dtype == np.float64
        assert result["frame_times_imaging"].dtype == np.float64

    def test_frame_times_start_at_zero(self, tmp_path: Path) -> None:
        session_name = "20210823_17_00_04_1114353_maze-rose"
        tdms_path = tmp_path / f"{session_name}-di.tdms"
        tdms_path.write_text("")
        _write_meta_txt(tmp_path, session_name)
        _write_ini_file(tmp_path, session_name)

        with patch("nptdms.TdmsFile.read", return_value=_mock_tdms_file()):
            result = parse_tdms(tdms_path)

        assert result["frame_times_camera"][0] == pytest.approx(0.0)

    def test_fps_values(self, tmp_path: Path) -> None:
        session_name = "20210823_17_00_04_1114353_maze-rose"
        tdms_path = tmp_path / f"{session_name}-di.tdms"
        tdms_path.write_text("")
        _write_meta_txt(tmp_path, session_name)
        _write_ini_file(tmp_path, session_name, fps=9.645)

        with patch("nptdms.TdmsFile.read", return_value=_mock_tdms_file()):
            result = parse_tdms(tdms_path)

        assert pytest.approx(float(result["fps_camera"])) == 100.0
        assert pytest.approx(float(result["fps_imaging"])) == 9.645

    def test_monotonic_camera_times(self, tmp_path: Path) -> None:
        session_name = "20210823_17_00_04_1114353_maze-rose"
        tdms_path = tmp_path / f"{session_name}-di.tdms"
        tdms_path.write_text("")
        _write_meta_txt(tmp_path, session_name)
        _write_ini_file(tmp_path, session_name)

        with patch("nptdms.TdmsFile.read", return_value=_mock_tdms_file()):
            result = parse_tdms(tdms_path)

        assert np.all(np.diff(result["frame_times_camera"]) > 0)


# ---------------------------------------------------------------------------
# run() — end-to-end Stage 0
# ---------------------------------------------------------------------------


class TestParseTdmsErrorHandling:
    """Test error-handling branches in parse_tdms and helpers."""

    def test_missing_meta_key_raises_valueerror(self, tmp_path: Path) -> None:
        """Missing DAQ key in meta.txt → ValueError."""
        session_name = "20210823_17_00_04_1114353_maze-rose"
        tdms_path = tmp_path / f"{session_name}-di.tdms"
        tdms_path.write_text("")
        # Write meta.txt missing the 'lightschanname' key
        meta_path = tmp_path / f"{session_name}.meta.txt"
        meta_path.write_text(
            "[DAQ]\n"
            "groupname = maze-rose\n"
            "cameratriggerchanname = cam_trigger\n"
            "sciscanchanname = sci_sync\n"
            "\n"
            "[Video]\n"
            "fps = 100.0\n"
            "\n"
            "[SciScan]\n"
            f"inifile = {session_name}_XYT.ini\n"
        )
        _write_ini_file(tmp_path, session_name)

        with pytest.raises(ValueError, match="Required key missing"):
            parse_tdms(tdms_path)

    def test_missing_ini_key_raises_valueerror(self, tmp_path: Path) -> None:
        """Missing key in .ini file → ValueError."""
        session_name = "20210823_17_00_04_1114353_maze-rose"
        tdms_path = tmp_path / f"{session_name}-di.tdms"
        tdms_path.write_text("")
        _write_meta_txt(tmp_path, session_name)
        # Write ini file missing 'frames.p.sec'
        ini_path = tmp_path / f"{session_name}_XYT.ini"
        ini_path.write_text("[_]\ny.pixels = 162\n")

        with pytest.raises(ValueError, match="Required key missing"):
            parse_tdms(tdms_path)

    def test_ini_fallback_glob(self, tmp_path: Path) -> None:
        """If exact ini file missing but glob matches *_XYT.ini, uses fallback."""
        session_name = "20210823_17_00_04_1114353_maze-rose"
        tdms_path = tmp_path / f"{session_name}-di.tdms"
        tdms_path.write_text("")
        _write_meta_txt(tmp_path, session_name)
        # Write ini with a different filename that matches *_XYT.ini glob
        alt_ini = tmp_path / "other_session_XYT.ini"
        alt_ini.write_text("[_]\ny.pixels = 162\nframes.p.sec = 9.645\n")

        mock_file = _mock_tdms_file()
        with patch("nptdms.TdmsFile.read", return_value=mock_file):
            result = parse_tdms(tdms_path)
        assert "frame_times_camera" in result

    def test_missing_channel_group_raises_valueerror(self, tmp_path: Path) -> None:
        """Missing TDMS channel group → ValueError from _get_di_channel."""
        session_name = "20210823_17_00_04_1114353_maze-rose"
        tdms_path = tmp_path / f"{session_name}-di.tdms"
        tdms_path.write_text("")
        _write_meta_txt(tmp_path, session_name)
        _write_ini_file(tmp_path, session_name)

        # Build a mock TDMS file that only has cam and sci groups (lights missing)
        groups = {
            "maze-rose - cam_trigger": MagicMock(
                channels=MagicMock(return_value=[_mock_tdms_channel(np.ones(100))])
            ),
            "maze-rose - sci_sync": MagicMock(
                channels=MagicMock(return_value=[_mock_tdms_channel(np.ones(100))])
            ),
        }
        mock_file = MagicMock()
        mock_file.__contains__ = lambda self, key: key in groups
        mock_file.__getitem__ = lambda self, key: groups[key]
        mock_file.groups.return_value = [MagicMock(name=k) for k in groups]
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)

        with (
            patch("nptdms.TdmsFile.read", return_value=mock_file),
            pytest.raises(ValueError, match="not found in file"),
        ):
            parse_tdms(tdms_path)

    def test_empty_channels_raises_valueerror(self, tmp_path: Path) -> None:
        """Empty channels list in TDMS group → ValueError from _get_di_channel."""
        session_name = "20210823_17_00_04_1114353_maze-rose"
        tdms_path = tmp_path / f"{session_name}-di.tdms"
        tdms_path.write_text("")
        _write_meta_txt(tmp_path, session_name)
        _write_ini_file(tmp_path, session_name)

        # Build mock where cam_trigger group exists but has empty channels
        cam_group = MagicMock(channels=MagicMock(return_value=[]))
        sci_chan = _mock_tdms_channel(np.ones(100))
        light_chan = _mock_tdms_channel(np.ones(100))
        groups = {
            "maze-rose - cam_trigger": cam_group,
            "maze-rose - sci_sync": MagicMock(channels=MagicMock(return_value=[sci_chan])),
            "maze-rose - lights": MagicMock(channels=MagicMock(return_value=[light_chan])),
        }
        mock_file = MagicMock()
        mock_file.__contains__ = lambda self, key: key in groups
        mock_file.__getitem__ = lambda self, key: groups[key]
        mock_file.groups.return_value = [MagicMock(name=k) for k in groups]
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)

        with (
            patch("nptdms.TdmsFile.read", return_value=mock_file),
            pytest.raises(ValueError, match="No channels"),
        ):
            parse_tdms(tdms_path)

    def test_no_camera_pulses_raises_valueerror(self, tmp_path: Path) -> None:
        """All-zero camera signal (no pulses) → ValueError."""
        session_name = "20210823_17_00_04_1114353_maze-rose"
        tdms_path = tmp_path / f"{session_name}-di.tdms"
        tdms_path.write_text("")
        _write_meta_txt(tmp_path, session_name)
        _write_ini_file(tmp_path, session_name)

        # Build mock with zero camera data (no pulses)
        mock_file = _mock_tdms_file(n_cam_pulses=0)

        with (
            patch("nptdms.TdmsFile.read", return_value=mock_file),
            pytest.raises(ValueError, match="No camera trigger pulses"),
        ):
            parse_tdms(tdms_path)


class TestDaqRun:
    def test_run_creates_file(self, tmp_path: Path) -> None:
        session_name = "20210823_17_00_04_1114353_maze-rose"
        tdms_path = tmp_path / f"{session_name}-di.tdms"
        tdms_path.write_text("")
        _write_meta_txt(tmp_path, session_name)
        _write_ini_file(tmp_path, session_name)

        out_h5 = tmp_path / "timestamps.h5"

        with patch("nptdms.TdmsFile.read", return_value=_mock_tdms_file()):
            run(tdms_path, session_id="test_session", output_path=out_h5)

        assert out_h5.exists()

    def test_run_output_readable(self, tmp_path: Path) -> None:
        from hm2p.io.hdf5 import read_attrs, read_h5

        session_name = "20210823_17_00_04_1114353_maze-rose"
        tdms_path = tmp_path / f"{session_name}-di.tdms"
        tdms_path.write_text("")
        _write_meta_txt(tmp_path, session_name)
        _write_ini_file(tmp_path, session_name)

        out_h5 = tmp_path / "timestamps.h5"

        with patch("nptdms.TdmsFile.read", return_value=_mock_tdms_file()):
            run(tdms_path, session_id="ses_123", output_path=out_h5)

        data = read_h5(out_h5)
        assert "frame_times_camera" in data
        assert "frame_times_imaging" in data

        attrs = read_attrs(out_h5)
        assert attrs["session_id"] == "ses_123"
