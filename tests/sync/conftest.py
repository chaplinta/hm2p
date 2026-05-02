"""Shared synthetic fixtures for sync-diagnostics tests.

Per CLAUDE.md, these helpers are unit-test-only — they never touch real
session data. Each builder uses ``np.random.default_rng(seed)`` for
reproducibility. See ``docs/sync-pipeline-design.md`` §6.1 for the
contract these fixtures satisfy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def synthetic_clean_pulse_train(
    rng: np.random.Generator,
    fps: float,
    duration_s: float,
    jitter_ms: float = 0.5,
) -> np.ndarray:
    """Return pulse times approximating a real session.

    Pulses are nominally evenly spaced at ``1/fps``; optional Gaussian
    jitter with standard deviation ``jitter_ms`` is added per inter-pulse
    interval (cumulative). Median ISI ≈ ``1/fps``, MAD ≈ ``jitter_ms``.
    """
    n = int(round(fps * duration_s)) + 1
    isis = np.full(n - 1, 1.0 / fps)
    if jitter_ms > 0:
        isis = isis + rng.normal(0.0, jitter_ms / 1000.0, size=isis.size)
        isis = np.clip(isis, 1e-6, None)
    times = np.concatenate([[0.0], np.cumsum(isis)])
    return times.astype(np.float64)


def synthetic_drifted_pulse_train(
    rng: np.random.Generator,
    fps: float,
    duration_s: float,
    drift_ppm: float,
) -> np.ndarray:
    """Apply linear drift to a clean train.

    Each ISI is set to ``(1 + drift_ppm × 1e-6) / fps`` so the regression
    slope of pulse index → time deviates from the nominal ``1/fps`` by
    exactly ``drift_ppm`` parts per million. ``drift_slope`` recovers
    this value when called with ``fps_nominal=fps``.
    """
    del rng
    n = int(round(fps * duration_s)) + 1
    if n < 2:
        return np.array([0.0], dtype=np.float64)
    isi = (1.0 + drift_ppm * 1e-6) / fps
    times = np.arange(n, dtype=np.float64) * isi
    return times


def synthetic_corrupted_pulse_train(
    rng: np.random.Generator,
    fps: float,
    duration_s: float,
    *,
    missing_idxs: tuple[int, ...] = (),
    duplicate_idxs: tuple[int, ...] = (),
) -> np.ndarray:
    """Build a clean train with deletions and/or duplicate insertions.

    ``missing_idxs`` are indices to drop (in pre-corruption coordinates);
    ``duplicate_idxs`` are indices to duplicate at a near-zero offset.
    """
    times = synthetic_clean_pulse_train(rng, fps, duration_s, jitter_ms=0.0)
    if missing_idxs:
        keep = np.ones(times.size, dtype=bool)
        for i in missing_idxs:
            if 0 <= i < keep.size:
                keep[i] = False
        times = times[keep]
    if duplicate_idxs:
        extras = []
        for i in duplicate_idxs:
            if 0 <= i < times.size:
                # Insert a near-duplicate ~1 microsecond later.
                extras.append(times[i] + 1e-6)
        if extras:
            times = np.sort(np.concatenate([times, np.array(extras, dtype=np.float64)]))
    return times.astype(np.float64)


def write_synthetic_timestamps_h5(
    path: Path,
    *,
    cam_times: np.ndarray | None = None,
    img_times: np.ndarray | None = None,
    line_times: np.ndarray | None = None,
    light_on: np.ndarray | None = None,
    light_off: np.ndarray | None = None,
    fps_camera: float = 100.0,
    fps_imaging: float = 30.0,
    session_id: str = "test",
    tdms_diag: dict[str, float] | None = None,
) -> None:
    """Write a minimal but valid timestamps.h5 fixture.

    All arrays default to a 6-second clean session at the conventional
    rates. ``tdms_diag`` defaults to a saturated digital channel.
    """
    from hm2p.io.hdf5 import write_h5

    if cam_times is None:
        cam_times = np.linspace(0.0, 6.0, 600, dtype=np.float64)
    if img_times is None:
        img_times = np.linspace(0.0, 6.0, 180, dtype=np.float64)
    if line_times is None:
        # 162 lines per frame is the canonical SciScan setting.
        line_times = np.linspace(0.0, 6.0, 180 * 162, dtype=np.float64)
    if light_on is None:
        light_on = np.array([0.0], dtype=np.float64)
    if light_off is None:
        light_off = np.array([3.0], dtype=np.float64)

    arrays = {
        "frame_times_camera": cam_times.astype(np.float64),
        "frame_times_imaging": img_times.astype(np.float64),
        "line_clock_times": line_times.astype(np.float64),
        "light_on_times": light_on.astype(np.float64),
        "light_off_times": light_off.astype(np.float64),
    }
    base_diag = {
        "tdms_diag/cam_min": 0.0,
        "tdms_diag/cam_max": 1.0,
        "tdms_diag/sci_min": 0.0,
        "tdms_diag/sci_max": 1.0,
        "tdms_diag/light_min": 0.0,
        "tdms_diag/light_max": 1.0,
        "tdms_diag/sci_lines_truncated_n": 0,
        "tdms_diag/tdms_sample_rate_hz": 10000.0,
        "tdms_diag/y_pix": 162,
    }
    if tdms_diag:
        for k, v in tdms_diag.items():
            base_diag[f"tdms_diag/{k}"] = v
    attrs = {
        "session_id": session_id,
        "fps_camera": fps_camera,
        "fps_imaging": fps_imaging,
        **base_diag,
    }
    write_h5(path, arrays, attrs=attrs)


def write_synthetic_kinematics_h5(
    path: Path,
    n: int = 600,
    *,
    bad_behav: np.ndarray | None = None,
    decimation_uniform: bool = True,
) -> None:
    """Write a minimal kinematics.h5 fixture (camera-rate)."""
    del decimation_uniform  # not yet emitted by Stage 3; reserved
    from hm2p.io.hdf5 import write_h5

    frame_times = np.linspace(0.0, 6.0, n, dtype=np.float64)
    if bad_behav is None:
        bad_behav = np.zeros(n, dtype=bool)
    write_h5(
        path,
        arrays={
            "frame_times": frame_times,
            "hd_deg": np.sin(frame_times).astype(np.float32),
            "x_mm": np.linspace(0, 50, n, dtype=np.float32),
            "y_mm": np.linspace(0, 30, n, dtype=np.float32),
            "speed_cm_s": np.ones(n, dtype=np.float32) * 5.0,
            "ahv_deg_s": np.zeros(n, dtype=np.float32),
            "active": np.ones(n, dtype=bool),
            "light_on": np.tile([True, False], n // 2).astype(bool),
            "bad_behav": bad_behav,
        },
        attrs={"session_id": "test", "fps_camera": 100.0},
    )


def write_synthetic_ca_h5(
    path: Path,
    t: int = 180,
    n_rois: int = 10,
    *,
    include_events: bool = False,
    include_spikes: bool = False,
    include_bad_imaging: bool = False,
) -> None:
    """Write a minimal ca.h5 fixture (imaging-rate)."""
    from hm2p.io.hdf5 import write_h5

    rng = np.random.default_rng(5)
    frame_times = np.linspace(0.0, 6.0, t, dtype=np.float64)
    arrays: dict[str, np.ndarray] = {
        "frame_times": frame_times,
        "dff": rng.standard_normal((n_rois, t)).astype(np.float32),
    }
    if include_events:
        arrays["event_masks"] = (rng.random((n_rois, t)) > 0.8).astype(bool)
    if include_spikes:
        arrays["spikes"] = rng.random((n_rois, t)).astype(np.float32)
    if include_bad_imaging:
        bad = np.zeros(t, dtype=bool)
        bad[5] = True
        arrays["bad_imaging_frames"] = bad
    attrs = {
        "session_id": "test",
        "fps_imaging": 30.0,
        "extractor": "suite2p",
    }
    write_h5(path, arrays, attrs=attrs)


def write_synthetic_sync_h5(
    path: Path,
    *,
    sync_status: str = "OK",
    sync_diag: dict | None = None,
    warnings: list[str] | None = None,
    failures: list[str] | None = None,
    payload: dict | None = None,
    extra_attrs: dict | None = None,
) -> None:
    """Write a synthetic sync.h5 with diagnostic attrs.

    Used by ``tests/sync/test_report.py`` and the frontend smoke tests.
    For OK / OK_WITH_WARNINGS the caller may pass a payload dict of
    resampled signals; FAILED_* sessions write only the diag attrs.
    """
    import json as _json

    from hm2p.io.hdf5 import write_h5

    if warnings is None:
        warnings = []
    if failures is None:
        failures = []
    diag = sync_diag or {}
    arrays = dict(payload) if payload is not None else {}
    attrs: dict = {
        "session_id": "test",
        "sync_status": sync_status,
        "sync_status_version": "1.0",
        "sync_warnings": _json.dumps(warnings),
        "sync_failures": _json.dumps(failures),
    }
    for k, v in diag.items():
        attrs[f"sync_diag/{k}"] = v
    if extra_attrs:
        attrs.update(extra_attrs)
    write_h5(path, arrays, attrs=attrs)
