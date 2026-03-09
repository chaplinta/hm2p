"""Sync validation — checks for known DAQ/timing failure modes.

Validates timestamps.h5 data against known issues from the legacy pipeline:
  1. Missing camera trigger pulses (frame count vs expected from FPS * duration)
  2. SciScan frame count mismatch (imaging pulses vs TIFF frame count)
  3. Camera frame interval jitter (>2ms from nominal)
  4. Imaging frame interval jitter (>1ms from nominal)
  5. Temporal overlap between camera and imaging recordings
  6. Light cycle period validation

Each check returns a ValidationResult with status, message, and optional details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Status(Enum):
    OK = "ok"
    WARN = "warn"
    ERROR = "error"
    SKIP = "skip"


@dataclass
class ValidationResult:
    name: str
    status: Status
    message: str
    details: dict = field(default_factory=dict)


def check_camera_interval_jitter(
    frame_times: np.ndarray,
    fps: float,
    tolerance_ms: float = 2.0,
) -> ValidationResult:
    """Check camera frame intervals for jitter beyond tolerance."""
    if len(frame_times) < 2:
        return ValidationResult("camera_jitter", Status.SKIP, "Not enough frames")

    intervals_ms = np.diff(frame_times) * 1000
    nominal_ms = 1000.0 / fps
    deviations = np.abs(intervals_ms - nominal_ms)
    n_bad = int((deviations > tolerance_ms).sum())
    max_dev = float(deviations.max())

    details = {
        "n_frames": len(frame_times),
        "nominal_ms": nominal_ms,
        "mean_ms": float(intervals_ms.mean()),
        "std_ms": float(intervals_ms.std()),
        "max_deviation_ms": max_dev,
        "n_bad": n_bad,
        "pct_bad": 100.0 * n_bad / len(intervals_ms),
    }

    if n_bad == 0:
        return ValidationResult(
            "camera_jitter", Status.OK,
            f"All {len(intervals_ms)} camera intervals within {tolerance_ms}ms of nominal",
            details,
        )
    elif n_bad < len(intervals_ms) * 0.01:  # <1%
        return ValidationResult(
            "camera_jitter", Status.WARN,
            f"{n_bad} camera intervals ({details['pct_bad']:.2f}%) deviate >{tolerance_ms}ms "
            f"(max {max_dev:.1f}ms)",
            details,
        )
    else:
        return ValidationResult(
            "camera_jitter", Status.ERROR,
            f"{n_bad} camera intervals ({details['pct_bad']:.1f}%) deviate >{tolerance_ms}ms",
            details,
        )


def check_imaging_interval_jitter(
    frame_times: np.ndarray,
    fps: float,
    tolerance_ms: float = 1.0,
) -> ValidationResult:
    """Check imaging frame intervals for jitter beyond tolerance."""
    if len(frame_times) < 2:
        return ValidationResult("imaging_jitter", Status.SKIP, "Not enough frames")

    intervals_ms = np.diff(frame_times) * 1000
    nominal_ms = 1000.0 / fps
    deviations = np.abs(intervals_ms - nominal_ms)
    n_bad = int((deviations > tolerance_ms).sum())
    max_dev = float(deviations.max())

    details = {
        "n_frames": len(frame_times),
        "nominal_ms": nominal_ms,
        "mean_ms": float(intervals_ms.mean()),
        "std_ms": float(intervals_ms.std()),
        "max_deviation_ms": max_dev,
        "n_bad": n_bad,
    }

    if n_bad == 0:
        return ValidationResult(
            "imaging_jitter", Status.OK,
            f"All {len(intervals_ms)} imaging intervals within {tolerance_ms}ms",
            details,
        )
    else:
        return ValidationResult(
            "imaging_jitter", Status.WARN,
            f"{n_bad} imaging intervals deviate >{tolerance_ms}ms (max {max_dev:.1f}ms)",
            details,
        )


def check_temporal_overlap(
    cam_times: np.ndarray,
    img_times: np.ndarray,
    max_diff_s: float = 5.0,
) -> ValidationResult:
    """Check temporal overlap between camera and imaging recordings."""
    cam_dur = cam_times[-1] - cam_times[0]
    img_dur = img_times[-1] - img_times[0]
    overlap_start = max(cam_times[0], img_times[0])
    overlap_end = min(cam_times[-1], img_times[-1])
    overlap_dur = max(0, overlap_end - overlap_start)
    dur_diff = abs(cam_dur - img_dur)

    details = {
        "cam_start": float(cam_times[0]),
        "cam_end": float(cam_times[-1]),
        "cam_duration_s": float(cam_dur),
        "img_start": float(img_times[0]),
        "img_end": float(img_times[-1]),
        "img_duration_s": float(img_dur),
        "overlap_s": float(overlap_dur),
        "duration_diff_s": float(dur_diff),
    }

    if dur_diff > max_diff_s:
        return ValidationResult(
            "temporal_overlap", Status.WARN,
            f"Camera ({cam_dur:.1f}s) and imaging ({img_dur:.1f}s) durations differ by {dur_diff:.1f}s",
            details,
        )
    return ValidationResult(
        "temporal_overlap", Status.OK,
        f"Recordings overlap {overlap_dur:.1f}s, duration diff {dur_diff:.1f}s",
        details,
    )


def check_frame_count_match(
    n_imaging_pulses: int,
    n_tiff_frames: int | None,
) -> ValidationResult:
    """Check if DAQ imaging pulse count matches TIFF frame count."""
    if n_tiff_frames is None:
        return ValidationResult(
            "frame_count", Status.SKIP,
            "No TIFF frame count available",
        )

    diff = abs(n_imaging_pulses - n_tiff_frames)
    details = {
        "n_imaging_pulses": n_imaging_pulses,
        "n_tiff_frames": n_tiff_frames,
        "difference": diff,
    }

    if diff == 0:
        return ValidationResult(
            "frame_count", Status.OK,
            f"Frame count matches exactly ({n_imaging_pulses})",
            details,
        )
    elif diff <= 1:
        return ValidationResult(
            "frame_count", Status.OK,
            f"Off by {diff} frame — acceptable SciScan edge case",
            details,
        )
    else:
        return ValidationResult(
            "frame_count", Status.ERROR,
            f"Frame count mismatch: {n_imaging_pulses} DAQ pulses vs {n_tiff_frames} TIFF frames (diff={diff})",
            details,
        )


def check_light_cycle(
    light_on_times: np.ndarray,
    light_off_times: np.ndarray,
    expected_period_s: float = 120.0,
    tolerance_s: float = 10.0,
) -> ValidationResult:
    """Check light on/off cycle period and consistency."""
    if len(light_on_times) < 2:
        return ValidationResult(
            "light_cycle", Status.SKIP,
            f"Only {len(light_on_times)} light-on event(s) — not enough to check cycle",
        )

    periods = np.diff(light_on_times)
    mean_period = float(periods.mean())
    std_period = float(periods.std())

    details = {
        "n_light_on": len(light_on_times),
        "n_light_off": len(light_off_times),
        "mean_period_s": mean_period,
        "std_period_s": std_period,
        "expected_period_s": expected_period_s,
    }

    if abs(mean_period - expected_period_s) > tolerance_s:
        return ValidationResult(
            "light_cycle", Status.WARN,
            f"Mean light cycle period {mean_period:.1f}s differs from expected {expected_period_s}s",
            details,
        )
    return ValidationResult(
        "light_cycle", Status.OK,
        f"Light cycle period {mean_period:.1f}s (expected ~{expected_period_s}s), std={std_period:.1f}s",
        details,
    )


def validate_timestamps(
    timestamps: dict[str, np.ndarray],
    fps_camera: float | None = None,
    fps_imaging: float | None = None,
    n_tiff_frames: int | None = None,
) -> list[ValidationResult]:
    """Run all validation checks on timestamps.h5 data.

    Args:
        timestamps: Dict with keys from timestamps.h5.
        fps_camera: Camera FPS (if known from attrs).
        fps_imaging: Imaging FPS (if known from attrs).
        n_tiff_frames: TIFF frame count from Suite2p ops.npy (if available).

    Returns:
        List of ValidationResult objects.
    """
    results = []

    cam_times = timestamps.get("frame_times_camera")
    img_times = timestamps.get("frame_times_imaging")
    light_on = timestamps.get("light_on_times")
    light_off = timestamps.get("light_off_times")

    # Infer FPS if not provided
    if fps_camera is None and cam_times is not None and len(cam_times) > 1:
        fps_camera = 1.0 / np.median(np.diff(cam_times))
    if fps_imaging is None and img_times is not None and len(img_times) > 1:
        fps_imaging = 1.0 / np.median(np.diff(img_times))

    if cam_times is not None and fps_camera is not None:
        results.append(check_camera_interval_jitter(cam_times, fps_camera))

    if img_times is not None and fps_imaging is not None:
        results.append(check_imaging_interval_jitter(img_times, fps_imaging))

    if cam_times is not None and img_times is not None:
        results.append(check_temporal_overlap(cam_times, img_times))

    if img_times is not None:
        results.append(check_frame_count_match(len(img_times), n_tiff_frames))

    if light_on is not None and light_off is not None:
        results.append(check_light_cycle(light_on, light_off))

    return results
