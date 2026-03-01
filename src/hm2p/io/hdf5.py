"""HDF5 read/write utilities with pandera schema validation.

All pipeline HDF5 files (timestamps.h5, kinematics.h5, ca.h5, sync.h5) are
written and read through this module. Schema validation runs on every write,
catching shape/dtype/range errors before they propagate downstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def write_h5(
    path: Path,
    arrays: dict[str, np.ndarray],
    attrs: dict[str, Any] | None = None,
) -> None:
    """Write arrays and optional root-level attributes to an HDF5 file.

    The file is created (or overwritten) atomically via a temp file.

    Args:
        path: Destination file path.
        arrays: Dict mapping dataset name → numpy array.
        attrs: Optional dict of root-level HDF5 attributes (session_id, fps_*, etc.).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, arr in arrays.items():
            f.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
        if attrs:
            for key, val in attrs.items():
                f.attrs[key] = val


def read_h5(
    path: Path,
    keys: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Read arrays from an HDF5 file.

    Args:
        path: Path to the HDF5 file.
        keys: List of dataset names to read. If None, reads all datasets.

    Returns:
        Dict mapping dataset name → numpy array.

    Raises:
        FileNotFoundError: If path does not exist.
        KeyError: If a requested key is not present in the file.
    """
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    with h5py.File(path, "r") as f:
        _keys = keys if keys is not None else list(f.keys())
        return {k: f[k][:] for k in _keys}


def read_attrs(path: Path) -> dict[str, Any]:
    """Read root-level HDF5 attributes from a file.

    Args:
        path: Path to the HDF5 file.

    Returns:
        Dict of attribute name → value.
    """
    with h5py.File(path, "r") as f:
        return dict(f.attrs)


# ---------------------------------------------------------------------------
# Schema validation helpers
# ---------------------------------------------------------------------------


def _schema_error(message: str) -> None:
    """Raise pandera.errors.SchemaError with a descriptive message."""
    from pandera.errors import SchemaError

    raise SchemaError(None, None, message)


def _check_key(arrays: dict[str, np.ndarray], key: str, context: str) -> np.ndarray:
    """Assert that *key* is present in *arrays*; raise SchemaError if not."""
    if key not in arrays:
        _schema_error(f"{context}: missing required key '{key}'")
    return arrays[key]


def _check_dtype(arr: np.ndarray, expected: np.dtype, key: str, context: str) -> None:
    if arr.dtype != expected:
        _schema_error(
            f"{context}: '{key}' must be {expected}, got {arr.dtype}"
        )


def _check_ndim(arr: np.ndarray, expected: int, key: str, context: str) -> None:
    if arr.ndim != expected:
        _schema_error(
            f"{context}: '{key}' must be {expected}D, got ndim={arr.ndim}"
        )


def _check_monotonic(arr: np.ndarray, key: str, context: str) -> None:
    if arr.size > 1 and not np.all(np.diff(arr) > 0):
        _schema_error(f"{context}: '{key}' must be strictly increasing")


def _check_nonneg(arr: np.ndarray, key: str, context: str) -> None:
    if np.any(arr < 0):
        _schema_error(f"{context}: '{key}' must be ≥ 0, found negatives")


def _check_length(arr: np.ndarray, expected_len: int, key: str, context: str) -> None:
    if len(arr) != expected_len:
        _schema_error(
            f"{context}: '{key}' length {len(arr)} != frame_times length {expected_len}"
        )


# ---------------------------------------------------------------------------
# Schema validation (pandera SchemaError interface)
# ---------------------------------------------------------------------------


def validate_timestamps_h5(arrays: dict[str, np.ndarray]) -> None:
    """Validate arrays against the timestamps.h5 schema.

    Required keys: frame_times_camera, frame_times_imaging,
    light_on_times, light_off_times — all float64, 1D.
    frame_times_camera and frame_times_imaging must be strictly increasing.

    Raises:
        pandera.errors.SchemaError: If any validation constraint fails.
    """
    ctx = "timestamps.h5"
    for key in ("frame_times_camera", "frame_times_imaging", "light_on_times", "light_off_times"):
        arr = _check_key(arrays, key, ctx)
        _check_dtype(arr, np.dtype("float64"), key, ctx)
        _check_ndim(arr, 1, key, ctx)
    _check_monotonic(arrays["frame_times_camera"], "frame_times_camera", ctx)
    _check_monotonic(arrays["frame_times_imaging"], "frame_times_imaging", ctx)


def validate_kinematics_h5(arrays: dict[str, np.ndarray]) -> None:
    """Validate arrays against the kinematics.h5 schema.

    Required keys and dtypes:
      frame_times  float64  1D  strictly increasing
      hd_deg       float32  1D
      x_mm         float32  1D
      y_mm         float32  1D
      speed_cm_s   float32  1D  ≥ 0
      ahv_deg_s    float32  1D
      active       bool     1D
      light_on     bool     1D
      bad_behav    bool     1D

    All 1D arrays must have the same length as frame_times.

    Raises:
        pandera.errors.SchemaError: If any validation constraint fails.
    """
    ctx = "kinematics.h5"
    ft = _check_key(arrays, "frame_times", ctx)
    _check_dtype(ft, np.dtype("float64"), "frame_times", ctx)
    _check_ndim(ft, 1, "frame_times", ctx)
    _check_monotonic(ft, "frame_times", ctx)
    T = len(ft)

    float32_keys = ("hd_deg", "x_mm", "y_mm", "speed_cm_s", "ahv_deg_s")
    for key in float32_keys:
        arr = _check_key(arrays, key, ctx)
        _check_dtype(arr, np.dtype("float32"), key, ctx)
        _check_ndim(arr, 1, key, ctx)
        _check_length(arr, T, key, ctx)
    _check_nonneg(
        arrays["speed_cm_s"][~np.isnan(arrays["speed_cm_s"])],
        "speed_cm_s",
        ctx,
    )

    for key in ("active", "light_on", "bad_behav"):
        arr = _check_key(arrays, key, ctx)
        if arr.dtype != np.dtype("bool"):
            _schema_error(f"{ctx}: '{key}' must be bool, got {arr.dtype}")
        _check_ndim(arr, 1, key, ctx)
        _check_length(arr, T, key, ctx)


def validate_ca_h5(arrays: dict[str, np.ndarray]) -> None:
    """Validate arrays against the ca.h5 schema.

    Required keys:
      frame_times  float64  1D  strictly increasing
      dff          float32  2D  shape (n_rois, n_frames)

    If 'spikes' is present it must be float32 2D with the same shape as dff.

    Raises:
        pandera.errors.SchemaError: If any validation constraint fails.
    """
    ctx = "ca.h5"
    ft = _check_key(arrays, "frame_times", ctx)
    _check_dtype(ft, np.dtype("float64"), "frame_times", ctx)
    _check_ndim(ft, 1, "frame_times", ctx)
    _check_monotonic(ft, "frame_times", ctx)
    T = len(ft)

    dff = _check_key(arrays, "dff", ctx)
    _check_dtype(dff, np.dtype("float32"), "dff", ctx)
    _check_ndim(dff, 2, "dff", ctx)
    if dff.shape[1] != T:
        _schema_error(
            f"{ctx}: 'dff' shape {dff.shape} — second dim {dff.shape[1]} != "
            f"len(frame_times) {T}"
        )

    if "spikes" in arrays:
        spikes = arrays["spikes"]
        _check_dtype(spikes, np.dtype("float32"), "spikes", ctx)
        _check_ndim(spikes, 2, "spikes", ctx)
        if spikes.shape != dff.shape:
            _schema_error(
                f"{ctx}: 'spikes' shape {spikes.shape} != 'dff' shape {dff.shape}"
            )


def validate_sync_h5(arrays: dict[str, np.ndarray]) -> None:
    """Validate arrays against the sync.h5 schema.

    sync.h5 merges kinematics and calcium arrays at the imaging frame rate.
    All kinematics constraints apply (with frame_times replacing camera times);
    dff must be float32 2D with n_frames == len(frame_times).

    Raises:
        pandera.errors.SchemaError: If any validation constraint fails.
    """
    # Reuse kinematics validator for the shared keys
    validate_kinematics_h5(arrays)
    # Additionally require dff
    ctx = "sync.h5"
    T = len(arrays["frame_times"])
    dff = _check_key(arrays, "dff", ctx)
    _check_dtype(dff, np.dtype("float32"), "dff", ctx)
    _check_ndim(dff, 2, "dff", ctx)
    if dff.shape[1] != T:
        _schema_error(
            f"{ctx}: 'dff' shape {dff.shape} — second dim {dff.shape[1]} != "
            f"len(frame_times) {T}"
        )
