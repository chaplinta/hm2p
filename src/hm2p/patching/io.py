"""File I/O for WaveSurfer HDF5 electrophysiology files and SWC morphology.

Reimplements the MATLAB ``loadDataFile_wavesurfer.m`` HDF5 reader and the
``getTracingFiles.m`` SWC file finder.

WaveSurfer H5 scaling
---------------------
Raw ADC counts are converted to physical units via a polynomial evaluated
using Horner's method, then divided by per-channel scale factors. The scaling
coefficients matrix has shape ``(n_coefficients, n_channels)`` and the
polynomial is evaluated highest-order-first:

    voltage = coeff[K-1, j]
    for k in range(K-2, -1, -1):
        voltage = coeff[k, j] + raw * voltage
    scaled = voltage / channel_scale[j]

This matches ``ws.scaledDoubleAnalogDataFromRaw`` in WaveSurfer 1.0.6.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# WaveSurfer HDF5 loading
# ---------------------------------------------------------------------------


def _crawl_h5_group(group: h5py.Group) -> dict[str, Any]:
    """Recursively read an HDF5 group into a nested dict.

    Datasets are loaded into memory as numpy arrays or Python scalars.
    Sub-groups become nested dicts.
    """
    result: dict[str, Any] = {}
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Group):
            result[key] = _crawl_h5_group(item)
        elif isinstance(item, h5py.Dataset):
            data = item[()]
            # Decode byte strings to str
            if isinstance(data, bytes):
                data = data.decode("utf-8", errors="replace")
            elif isinstance(data, np.ndarray) and data.dtype.kind == "S":
                data = data.astype(str)
            # Unbox scalar arrays
            if isinstance(data, np.ndarray) and data.ndim == 0:
                data = data.item()
            result[key] = data
    return result


def _apply_scaling(
    raw: np.ndarray,
    channel_scales: np.ndarray,
    scaling_coefficients: np.ndarray,
) -> np.ndarray:
    """Scale raw int16 ADC counts to physical units.

    Parameters
    ----------
    raw : np.ndarray
        (n_scans, n_channels) int16 array of ADC counts.
    channel_scales : np.ndarray
        (n_channels,) per-channel scale factors (V per native unit).
    scaling_coefficients : np.ndarray
        (n_coefficients, n_channels) polynomial coefficients. Evaluated
        via Horner's method from highest to lowest order.

    Returns
    -------
    np.ndarray
        (n_scans, n_channels) float64 scaled data.
    """
    raw_float = raw.astype(np.float64)
    n_coeff = scaling_coefficients.shape[0]

    if raw_float.ndim == 1:
        raw_float = raw_float[:, np.newaxis]

    n_scans, n_channels = raw_float.shape
    scaled = np.zeros_like(raw_float)

    for j in range(n_channels):
        # Horner's method — start at highest order coefficient
        voltage = np.full(n_scans, scaling_coefficients[n_coeff - 1, j])
        for k in range(n_coeff - 2, -1, -1):
            voltage = scaling_coefficients[k, j] + raw_float[:, j] * voltage
        scaled[:, j] = voltage / channel_scales[j]

    return scaled


def load_wavesurfer(path: Path) -> dict[str, Any]:
    """Load a WaveSurfer ``.h5`` file and return scaled sweep data.

    Parameters
    ----------
    path : Path
        Path to a WaveSurfer HDF5 file.

    Returns
    -------
    dict
        Nested dict mirroring the HDF5 group hierarchy. Sweep analog data
        is scaled from raw ADC counts to physical units. Top-level keys
        include ``"header"`` and ``"sweep_NNNN"`` entries.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file lacks required scaling metadata.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"WaveSurfer file not found: {path}")

    with h5py.File(path, "r") as f:
        data = _crawl_h5_group(f)

    # --- Retrieve scaling info from header ---
    header = data.get("header", {})

    # Channel scales
    channel_scales = _get_nested(
        header,
        ("AIChannelScales",),
        fallback_keys=("Acquisition", "AnalogChannelScales"),
    )
    if channel_scales is None:
        # No AI channels — return as-is
        return data

    channel_scales = np.atleast_1d(np.asarray(channel_scales, dtype=np.float64))

    # Active channel mask (optional — if absent, assume all active)
    is_active = _get_nested(
        header,
        ("IsAIChannelActive",),
        fallback_keys=("Acquisition", "IsAnalogChannelActive"),
    )
    if is_active is not None:
        is_active = np.atleast_1d(np.asarray(is_active, dtype=bool))
        channel_scales = channel_scales[is_active]

    # Scaling coefficients
    scaling_coefficients = _get_nested(
        header,
        ("AIScalingCoefficients",),
        fallback_keys=("Acquisition", "AnalogScalingCoefficients"),
    )
    if scaling_coefficients is None:
        return data

    scaling_coefficients = np.asarray(scaling_coefficients, dtype=np.float64)
    if scaling_coefficients.ndim == 1:
        scaling_coefficients = scaling_coefficients[:, np.newaxis]

    # --- Scale sweep analog data ---
    for key in list(data.keys()):
        if key.startswith("sweep_") or key.startswith("trial_"):
            sweep = data[key]
            if isinstance(sweep, dict) and "analogScans" in sweep:
                raw = np.asarray(sweep["analogScans"])
                if raw.ndim == 1:
                    raw = raw[:, np.newaxis]
                sweep["analogScans"] = _apply_scaling(
                    raw, channel_scales, scaling_coefficients
                )

    return data


def _get_nested(
    d: dict[str, Any],
    primary_keys: tuple[str, ...],
    fallback_keys: tuple[str, ...] | None = None,
) -> Any:
    """Look up a value in a nested dict, trying primary then fallback paths."""
    # Try primary (single key)
    val = d.get(primary_keys[0]) if len(primary_keys) == 1 else None
    if val is not None:
        return val

    # Try fallback path (e.g. ("Acquisition", "AnalogChannelScales"))
    if fallback_keys is not None:
        node = d
        for k in fallback_keys:
            if isinstance(node, dict):
                node = node.get(k)
            else:
                return None
        return node

    return None


# ---------------------------------------------------------------------------
# Sweep extraction
# ---------------------------------------------------------------------------


def get_sweep_traces(ws_data: dict[str, Any], sweep_idx: int) -> np.ndarray:
    """Extract the voltage channel trace from a specific sweep.

    Parameters
    ----------
    ws_data : dict
        Data dict returned by :func:`load_wavesurfer`.
    sweep_idx : int
        1-based sweep index (matching WaveSurfer naming ``sweep_0001`` etc.).

    Returns
    -------
    np.ndarray
        1-D float64 array of the first analog channel (voltage).

    Raises
    ------
    KeyError
        If the requested sweep does not exist.
    """
    sweep_key = f"sweep_{sweep_idx:04d}"
    if sweep_key not in ws_data:
        raise KeyError(
            f"Sweep '{sweep_key}' not found. "
            f"Available: {[k for k in ws_data if k.startswith('sweep_')]}"
        )
    analog = ws_data[sweep_key]["analogScans"]
    analog = np.asarray(analog, dtype=np.float64)
    if analog.ndim == 2:
        return analog[:, 0]
    return analog


# ---------------------------------------------------------------------------
# SWC morphology file loading
# ---------------------------------------------------------------------------


def load_swc_files(tracing_path: Path) -> dict[str, Path | list[Path]]:
    """Find SWC morphology files in a tracing directory.

    Looks for the standard file names used in the patching pipeline:
    ``Soma.swc``, ``Apical_tree.swc``, ``Basal*.swc``, ``Surface.swc``,
    ``Axon.swc``.

    Parameters
    ----------
    tracing_path : Path
        Directory containing SWC files for a single cell.

    Returns
    -------
    dict
        Keys are component types: ``"soma"``, ``"apical"``, ``"basal"``
        (list), ``"surface"`` (optional), ``"axon"`` (optional). Values
        are :class:`Path` objects (or list of Paths for basal trees).

    Raises
    ------
    FileNotFoundError
        If *tracing_path* does not exist or required files are missing.
    """
    tracing_path = Path(tracing_path)
    if not tracing_path.is_dir():
        raise FileNotFoundError(f"Tracing directory not found: {tracing_path}")

    result: dict[str, Path | list[Path]] = {}

    # Required files
    soma = tracing_path / "Soma.swc"
    if not soma.exists():
        raise FileNotFoundError(f"Soma.swc not found in {tracing_path}")
    result["soma"] = soma

    apical = tracing_path / "Apical_tree.swc"
    if not apical.exists():
        raise FileNotFoundError(f"Apical_tree.swc not found in {tracing_path}")
    result["apical"] = apical

    # Basal trees (Basal*.swc — may be multiple)
    basal_files = sorted(tracing_path.glob("Basal*.swc"))
    result["basal"] = basal_files

    # Optional files
    surface = tracing_path / "Surface.swc"
    if surface.exists():
        result["surface"] = surface

    axon = tracing_path / "Axon.swc"
    if axon.exists():
        result["axon"] = axon

    return result
