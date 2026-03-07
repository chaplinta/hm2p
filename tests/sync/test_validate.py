"""Tests for sync validation checks."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.sync.validate import (
    Status,
    check_camera_interval_jitter,
    check_frame_count_match,
    check_imaging_interval_jitter,
    check_light_cycle,
    check_temporal_overlap,
    validate_timestamps,
)


class TestCameraJitter:
    def test_perfect_intervals(self):
        fps = 100.0
        times = np.arange(1000) / fps
        result = check_camera_interval_jitter(times, fps)
        assert result.status == Status.OK
        assert result.details["n_bad"] == 0

    def test_jittery_intervals(self):
        fps = 100.0
        times = np.arange(1000) / fps
        # Add large jitter to a few frames
        times[500] += 0.005  # 5ms off
        times[600] += 0.010  # 10ms off
        result = check_camera_interval_jitter(times, fps)
        assert result.status in (Status.WARN, Status.ERROR)
        assert result.details["n_bad"] > 0

    def test_too_few_frames(self):
        result = check_camera_interval_jitter(np.array([0.0]), 100.0)
        assert result.status == Status.SKIP


class TestImagingJitter:
    def test_perfect_intervals(self):
        fps = 9.8
        times = np.arange(18000) / fps
        result = check_imaging_interval_jitter(times, fps)
        assert result.status == Status.OK

    def test_with_jitter(self):
        fps = 9.8
        times = np.arange(100) / fps
        times[50] += 0.005  # 5ms off
        result = check_imaging_interval_jitter(times, fps)
        assert result.status == Status.WARN
        assert result.details["n_bad"] > 0


class TestTemporalOverlap:
    def test_matching_durations(self):
        cam = np.arange(10000) / 100.0  # 100s at 100Hz
        img = np.arange(980) / 9.8  # 100s at 9.8Hz
        result = check_temporal_overlap(cam, img)
        assert result.status == Status.OK

    def test_mismatched_durations(self):
        cam = np.arange(10000) / 100.0  # 100s
        img = np.arange(1500) / 9.8  # 153s — much longer
        result = check_temporal_overlap(cam, img)
        assert result.status == Status.WARN
        assert result.details["duration_diff_s"] > 5.0


class TestFrameCountMatch:
    def test_exact_match(self):
        result = check_frame_count_match(18000, 18000)
        assert result.status == Status.OK

    def test_off_by_one(self):
        result = check_frame_count_match(18000, 18001)
        assert result.status == Status.OK
        assert "edge case" in result.message.lower() or "off by" in result.message.lower()

    def test_mismatch(self):
        result = check_frame_count_match(18000, 17950)
        assert result.status == Status.ERROR

    def test_no_tiff(self):
        result = check_frame_count_match(18000, None)
        assert result.status == Status.SKIP


class TestLightCycle:
    def test_regular_cycle(self):
        on_times = np.array([10.0, 130.0, 250.0, 370.0])  # ~120s period
        off_times = np.array([70.0, 190.0, 310.0, 430.0])
        result = check_light_cycle(on_times, off_times)
        assert result.status == Status.OK

    def test_irregular_cycle(self):
        on_times = np.array([10.0, 200.0, 400.0])  # ~190s period
        off_times = np.array([100.0, 300.0, 500.0])
        result = check_light_cycle(on_times, off_times)
        assert result.status == Status.WARN

    def test_single_event(self):
        result = check_light_cycle(np.array([10.0]), np.array([70.0]))
        assert result.status == Status.SKIP


class TestValidateTimestamps:
    def test_all_checks_run(self):
        fps = 100.0
        cam_times = np.arange(10000) / fps
        img_times = np.arange(980) / 9.8
        light_on = np.array([10.0, 130.0, 250.0])
        light_off = np.array([70.0, 190.0, 310.0])

        timestamps = {
            "frame_times_camera": cam_times,
            "frame_times_imaging": img_times,
            "light_on_times": light_on,
            "light_off_times": light_off,
        }
        results = validate_timestamps(timestamps)
        names = {r.name for r in results}
        assert "camera_jitter" in names
        assert "imaging_jitter" in names
        assert "temporal_overlap" in names
        assert "light_cycle" in names

    def test_with_tiff_frames(self):
        timestamps = {
            "frame_times_camera": np.arange(100) / 100.0,
            "frame_times_imaging": np.arange(10) / 9.8,
            "light_on_times": np.array([1.0]),
            "light_off_times": np.array([2.0]),
        }
        results = validate_timestamps(timestamps, n_tiff_frames=10)
        frame_check = [r for r in results if r.name == "frame_count"]
        assert len(frame_check) == 1
        assert frame_check[0].status == Status.OK
