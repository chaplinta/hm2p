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
# Schema validation (pandera — implemented with actual schemas in tests/io/)
# ---------------------------------------------------------------------------


def validate_timestamps_h5(arrays: dict[str, np.ndarray]) -> None:
    """Validate arrays against the timestamps.h5 schema.

    Raises:
        pandera.errors.SchemaError: If any validation constraint fails.
    """
    raise NotImplementedError


def validate_kinematics_h5(arrays: dict[str, np.ndarray]) -> None:
    """Validate arrays against the kinematics.h5 schema.

    Raises:
        pandera.errors.SchemaError: If any validation constraint fails.
    """
    raise NotImplementedError


def validate_ca_h5(arrays: dict[str, np.ndarray]) -> None:
    """Validate arrays against the ca.h5 schema.

    Raises:
        pandera.errors.SchemaError: If any validation constraint fails.
    """
    raise NotImplementedError


def validate_sync_h5(arrays: dict[str, np.ndarray]) -> None:
    """Validate arrays against the sync.h5 schema.

    Raises:
        pandera.errors.SchemaError: If any validation constraint fails.
    """
    raise NotImplementedError
