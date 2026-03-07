"""Tests for helper functions in scripts/run_stage3_kinematics.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add scripts to path so we can import the module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from run_stage3_kinematics import parse_bad_behav_times, parse_meta_txt


# ---------------------------------------------------------------------------
# parse_bad_behav_times
# ---------------------------------------------------------------------------


class TestParseBadBehavTimes:
    def test_empty_string(self):
        assert parse_bad_behav_times("") == []

    def test_whitespace_only(self):
        assert parse_bad_behav_times("   ") == []

    def test_question_mark(self):
        assert parse_bad_behav_times("?") == []

    def test_question_mark_with_whitespace(self):
        assert parse_bad_behav_times("  ?  ") == []

    def test_none_input(self):
        assert parse_bad_behav_times(None) == []

    def test_single_interval(self):
        result = parse_bad_behav_times("11:10-11:30")
        assert result == [(670.0, 690.0)]

    def test_multiple_intervals(self):
        result = parse_bad_behav_times("11:10-11:30;13:20-21:00")
        assert len(result) == 2
        assert result[0] == (670.0, 690.0)
        assert result[1] == (800.0, 1260.0)

    def test_end_keyword(self):
        result = parse_bad_behav_times("27:00-end")
        assert len(result) == 1
        assert result[0] == (1620.0, 999999.0)

    def test_mixed_intervals_and_end(self):
        result = parse_bad_behav_times("11:10-11:30;13:20-21:00;27:00-end")
        assert len(result) == 3
        assert result[0] == (670.0, 690.0)
        assert result[1] == (800.0, 1260.0)
        assert result[2] == (1620.0, 999999.0)

    def test_zero_time(self):
        result = parse_bad_behav_times("0:00-1:00")
        assert result == [(0.0, 60.0)]

    def test_large_minutes(self):
        result = parse_bad_behav_times("120:30-125:45")
        assert result == [(7230.0, 7545.0)]

    def test_whitespace_around_segments(self):
        result = parse_bad_behav_times("  1:00-2:00 ; 3:00-4:00  ")
        assert len(result) == 2
        assert result[0] == (60.0, 120.0)
        assert result[1] == (180.0, 240.0)

    def test_whitespace_around_dash(self):
        result = parse_bad_behav_times("1:00 - 2:00")
        assert result == [(60.0, 120.0)]

    def test_trailing_semicolon(self):
        result = parse_bad_behav_times("1:00-2:00;")
        assert len(result) == 1
        assert result[0] == (60.0, 120.0)

    def test_return_types_are_float(self):
        result = parse_bad_behav_times("5:30-10:15")
        assert isinstance(result[0][0], float)
        assert isinstance(result[0][1], float)

    def test_invalid_segment_skipped(self, capsys):
        result = parse_bad_behav_times("garbage")
        assert result == []
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_partial_invalid_keeps_valid(self, capsys):
        result = parse_bad_behav_times("1:00-2:00;bad;3:00-4:00")
        assert len(result) == 2
        assert result[0] == (60.0, 120.0)
        assert result[1] == (180.0, 240.0)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_seconds_conversion_accuracy(self):
        # 5 minutes 30 seconds = 330 seconds
        result = parse_bad_behav_times("5:30-6:00")
        assert result[0][0] == 330.0
        assert result[0][1] == 360.0

    def test_single_second_values(self):
        result = parse_bad_behav_times("0:01-0:02")
        assert result == [(1.0, 2.0)]

    def test_same_start_end(self):
        result = parse_bad_behav_times("5:00-5:00")
        assert result == [(300.0, 300.0)]


# ---------------------------------------------------------------------------
# parse_meta_txt
# ---------------------------------------------------------------------------


class TestParseMetaTxt:
    def _write_meta(self, path: Path, mm_per_pix: float, corners: list[list[float]]):
        lines = [
            "[scale]",
            f"mm_per_pix = {mm_per_pix}",
            "",
            "[roi]",
        ]
        for i, (x, y) in enumerate(corners, start=1):
            lines.append(f"x{i} = {x}")
            lines.append(f"y{i} = {y}")
        path.write_text("\n".join(lines))

    def test_basic_parsing(self, tmp_path):
        meta = tmp_path / "meta.txt"
        corners = [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]]
        self._write_meta(meta, 0.1234, corners)
        mm_per_pix, result_corners = parse_meta_txt(meta)
        assert mm_per_pix == pytest.approx(0.1234)
        expected = np.array(corners)
        np.testing.assert_allclose(result_corners, expected)

    def test_returns_float_and_ndarray(self, tmp_path):
        meta = tmp_path / "meta.txt"
        corners = [[1, 2], [3, 4], [5, 6], [7, 8]]
        self._write_meta(meta, 0.5, corners)
        mm_per_pix, result_corners = parse_meta_txt(meta)
        assert isinstance(mm_per_pix, float)
        assert isinstance(result_corners, np.ndarray)

    def test_corners_shape(self, tmp_path):
        meta = tmp_path / "meta.txt"
        corners = [[100, 200], [300, 400], [500, 600], [700, 800]]
        self._write_meta(meta, 1.0, corners)
        _, result_corners = parse_meta_txt(meta)
        assert result_corners.shape == (4, 2)

    def test_fractional_corners(self, tmp_path):
        meta = tmp_path / "meta.txt"
        corners = [[10.5, 20.3], [30.7, 40.1], [50.9, 60.2], [70.4, 80.6]]
        self._write_meta(meta, 0.0567, corners)
        mm_per_pix, result_corners = parse_meta_txt(meta)
        assert mm_per_pix == pytest.approx(0.0567)
        np.testing.assert_allclose(result_corners, np.array(corners), atol=1e-10)

    def test_zero_scale(self, tmp_path):
        meta = tmp_path / "meta.txt"
        corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self._write_meta(meta, 0.0, corners)
        mm_per_pix, _ = parse_meta_txt(meta)
        assert mm_per_pix == 0.0

    def test_large_coordinates(self, tmp_path):
        meta = tmp_path / "meta.txt"
        corners = [[1920, 1080], [0, 1080], [0, 0], [1920, 0]]
        self._write_meta(meta, 0.08, corners)
        _, result_corners = parse_meta_txt(meta)
        np.testing.assert_allclose(result_corners[0], [1920, 1080])

    def test_missing_scale_section_raises(self, tmp_path):
        meta = tmp_path / "meta.txt"
        meta.write_text("[roi]\nx1 = 10\ny1 = 20\nx2 = 30\ny2 = 40\nx3 = 50\ny3 = 60\nx4 = 70\ny4 = 80\n")
        with pytest.raises(KeyError):
            parse_meta_txt(meta)

    def test_missing_roi_section_raises(self, tmp_path):
        meta = tmp_path / "meta.txt"
        meta.write_text("[scale]\nmm_per_pix = 0.1\n")
        with pytest.raises(KeyError):
            parse_meta_txt(meta)

    def test_missing_corner_key_raises(self, tmp_path):
        meta = tmp_path / "meta.txt"
        # Only 3 corners instead of 4
        content = "[scale]\nmm_per_pix = 0.1\n[roi]\nx1 = 10\ny1 = 20\nx2 = 30\ny2 = 40\nx3 = 50\ny3 = 60\n"
        meta.write_text(content)
        with pytest.raises((KeyError, configparser.NoOptionError)):
            parse_meta_txt(meta)

    def test_nonexistent_file_raises(self, tmp_path):
        meta = tmp_path / "nonexistent.txt"
        # configparser.read silently ignores missing files, so it will raise KeyError
        with pytest.raises(KeyError):
            parse_meta_txt(meta)

    def test_extra_sections_ignored(self, tmp_path):
        meta = tmp_path / "meta.txt"
        content = (
            "[scale]\nmm_per_pix = 0.25\n"
            "[roi]\nx1=1\ny1=2\nx2=3\ny2=4\nx3=5\ny3=6\nx4=7\ny4=8\n"
            "[extra]\nfoo = bar\n"
        )
        meta.write_text(content)
        mm_per_pix, corners = parse_meta_txt(meta)
        assert mm_per_pix == pytest.approx(0.25)
        assert corners.shape == (4, 2)


# Need configparser for the raises test
import configparser
