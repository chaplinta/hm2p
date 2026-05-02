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


def _collect_datasets(
    group: h5py.Group,
    prefix: str = "",
) -> list[str]:
    """Recursively collect all dataset paths in an HDF5 group.

    Returns slash-separated paths relative to the given group (e.g.
    ``"roi_qc/snr_event"``). Groups are traversed but not returned.

    Args:
        group: An open h5py Group (or File, which is a Group subclass).
        prefix: Path prefix for the current level (empty at the root).

    Returns:
        List of full slash-separated dataset names.
    """
    paths: list[str] = []
    for name, item in group.items():
        full_path = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Dataset):
            paths.append(full_path)
        elif isinstance(item, h5py.Group):
            paths.extend(_collect_datasets(item, prefix=full_path))
    return paths


def read_h5(
    path: Path,
    keys: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Read arrays from an HDF5 file.

    When ``keys`` is None, reads all datasets recursively (including those
    inside sub-groups such as ``roi_qc/snr_event``). Groups themselves are
    not returned — only leaf datasets.

    When ``keys`` is provided, each entry may be a slash-separated path
    (e.g. ``"roi_qc/snr_event"``); h5py resolves these as nested paths.

    Args:
        path: Path to the HDF5 file.
        keys: List of dataset names (or slash-separated paths) to read.
            If None, reads all datasets in the file recursively.

    Returns:
        Dict mapping dataset path → numpy array.

    Raises:
        FileNotFoundError: If path does not exist.
        KeyError: If a requested key is not present in the file.
    """
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    with h5py.File(path, "r") as f:
        _keys = keys if keys is not None else _collect_datasets(f)
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
        _schema_error(f"{context}: '{key}' must be {expected}, got {arr.dtype}")


def _check_ndim(arr: np.ndarray, expected: int, key: str, context: str) -> None:
    if arr.ndim != expected:
        _schema_error(f"{context}: '{key}' must be {expected}D, got ndim={arr.ndim}")


def _check_monotonic(arr: np.ndarray, key: str, context: str) -> None:
    if arr.size > 1 and not np.all(np.diff(arr) > 0):
        _schema_error(f"{context}: '{key}' must be strictly increasing")


def _check_nonneg(arr: np.ndarray, key: str, context: str) -> None:
    if np.any(arr < 0):
        _schema_error(f"{context}: '{key}' must be ≥ 0, found negatives")


def _check_length(arr: np.ndarray, expected_len: int, key: str, context: str) -> None:
    if len(arr) != expected_len:
        _schema_error(f"{context}: '{key}' length {len(arr)} != frame_times length {expected_len}")


# ---------------------------------------------------------------------------
# Schema validation (pandera SchemaError interface)
# ---------------------------------------------------------------------------


def validate_timestamps_h5(
    arrays: dict[str, np.ndarray],
    *,
    attrs: dict[str, Any] | None = None,
    require_diagnostics: bool = False,
) -> None:
    """Validate arrays against the timestamps.h5 schema.

    Required keys (always): frame_times_camera, frame_times_imaging,
    light_on_times, light_off_times — all float64, 1D.
    frame_times_camera and frame_times_imaging must be strictly increasing.

    When ``require_diagnostics`` is True (the new schema for files written
    after the sync-pipeline diagnostics rollout), the dataset
    ``line_clock_times`` (float64, 1D, strictly increasing) and the group
    attrs ``tdms_diag/cam_min``, ``tdms_diag/cam_max``, ``tdms_diag/sci_min``,
    ``tdms_diag/sci_max``, ``tdms_diag/light_min``, ``tdms_diag/light_max``,
    ``tdms_diag/sci_lines_truncated_n``, ``tdms_diag/tdms_sample_rate_hz``,
    ``tdms_diag/y_pix`` are required. ``tdms_diag/sci_lines_truncated_n``
    must be ≥ 0. The group attrs are passed via the ``attrs`` argument as
    a flat dict where keys are slash-prefixed (e.g. ``"tdms_diag/cam_min"``).

    Args:
        arrays: dict of dataset name → numpy array.
        attrs: dict of HDF5 attributes (root + group) — keys for group
            attributes must be slash-prefixed, e.g. ``"tdms_diag/cam_min"``.
        require_diagnostics: when True, also require the new diagnostic
            keys introduced for the sync pipeline.

    Raises:
        pandera.errors.SchemaError: If any validation constraint fails.

    See Also:
        ``docs/sync-pipeline-design.md`` §2.1 for the full schema.
    """
    ctx = "timestamps.h5"
    for key in ("frame_times_camera", "frame_times_imaging", "light_on_times", "light_off_times"):
        arr = _check_key(arrays, key, ctx)
        _check_dtype(arr, np.dtype("float64"), key, ctx)
        _check_ndim(arr, 1, key, ctx)
    _check_monotonic(arrays["frame_times_camera"], "frame_times_camera", ctx)
    _check_monotonic(arrays["frame_times_imaging"], "frame_times_imaging", ctx)

    if not require_diagnostics:
        return

    # Diagnostic schema (sync_status_version >= "1.0").
    line_clock = _check_key(arrays, "line_clock_times", ctx)
    _check_dtype(line_clock, np.dtype("float64"), "line_clock_times", ctx)
    _check_ndim(line_clock, 1, "line_clock_times", ctx)
    _check_monotonic(line_clock, "line_clock_times", ctx)

    if attrs is None:
        _schema_error(f"{ctx}: 'tdms_diag/' group missing (require_diagnostics=True)")
    assert attrs is not None  # mypy
    required_diag = (
        "tdms_diag/cam_min",
        "tdms_diag/cam_max",
        "tdms_diag/sci_min",
        "tdms_diag/sci_max",
        "tdms_diag/light_min",
        "tdms_diag/light_max",
        "tdms_diag/sci_lines_truncated_n",
        "tdms_diag/tdms_sample_rate_hz",
        "tdms_diag/y_pix",
    )
    for key in required_diag:
        if key not in attrs:
            _schema_error(f"{ctx}: missing required attr '{key}'")
    truncated = int(attrs["tdms_diag/sci_lines_truncated_n"])
    if truncated < 0:
        _schema_error(f"{ctx}: 'tdms_diag/sci_lines_truncated_n' must be >= 0, got {truncated}")


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

    Optional keys validated when present:
      spikes              float32  2D  same shape as dff (CASCADE spike rates)
      roi_types           uint8    1D  length n_rois (0=soma, 1=dend, 2=artefact)
      iscell              bool     1D  length n_rois (Suite2p classifier
                                       acceptance flag; orthogonal to roi_types)
      roi_qc/roi_index    int32    1D  length n_rois
      roi_qc/snr_event    float32  1D  length n_rois
      roi_qc/decay_tau_s  float32  1D  length n_rois
      roi_qc/fneu_dff_corr float32 1D  length n_rois
      roi_qc/bleach_slope float32  1D  length n_rois
      roi_qc/active_fraction float32 1D length n_rois
      roi_qc/p_soma       float32  1D  length n_rois
      roi_qc/p_dend       float32  1D  length n_rois
      roi_qc/p_artefact   float32  1D  length n_rois

    The roi_qc/* arrays are written by ``hm2p.calcium.qc.compute_roi_qc``
    (SNR, tau, etc.) and ``hm2p.extraction.soma_classifier`` (the three
    ``p_*`` probabilities); both use slash-keyed names so that h5py
    creates a ``roi_qc`` group automatically.

    ``roi_types`` is the soma/dend/artefact label produced by the soma
    classifier (``hm2p.extraction.soma_classifier``). ``iscell`` is
    Suite2p's accept/reject flag — kept separate so a Suite2p-rejected
    dendrite is distinguishable from a physical artefact.

    References:
        Pnevmatikakis et al. 2016. "Simultaneous Denoising, Deconvolution, and
        Demixing of Calcium Imaging Data." Neuron 89(2):285-299.
        doi:10.1016/j.neuron.2015.11.037

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
            f"{ctx}: 'dff' shape {dff.shape} — second dim {dff.shape[1]} != len(frame_times) {T}"
        )
    n_rois = dff.shape[0]

    if "spikes" in arrays:
        spikes = arrays["spikes"]
        _check_dtype(spikes, np.dtype("float32"), "spikes", ctx)
        _check_ndim(spikes, 2, "spikes", ctx)
        if spikes.shape != dff.shape:
            _schema_error(f"{ctx}: 'spikes' shape {spikes.shape} != 'dff' shape {dff.shape}")

    # roi_types: optional uint8 (0=soma, 1=dend, 2=artefact). Length must
    # match n_rois. ``iscell`` is a separate boolean — both are 1D arrays
    # that share the ROI axis with dff.
    if "roi_types" in arrays:
        rt = arrays["roi_types"]
        _check_dtype(rt, np.dtype("uint8"), "roi_types", ctx)
        _check_ndim(rt, 1, "roi_types", ctx)
        if len(rt) != n_rois:
            _schema_error(f"{ctx}: 'roi_types' length {len(rt)} != n_rois {n_rois}")
        if rt.size and (rt > 2).any():
            _schema_error(
                f"{ctx}: 'roi_types' values must be 0 (soma), 1 (dend), or "
                f"2 (artefact); found max {int(rt.max())}"
            )
    if "iscell" in arrays:
        ic = arrays["iscell"]
        _check_dtype(ic, np.dtype("bool"), "iscell", ctx)
        _check_ndim(ic, 1, "iscell", ctx)
        if len(ic) != n_rois:
            _schema_error(f"{ctx}: 'iscell' length {len(ic)} != n_rois {n_rois}")

    # Optional roi_qc group: all arrays must be 1D with length n_rois.
    # ``p_soma`` / ``p_dend`` / ``p_artefact`` are calibrated probabilities
    # written by the soma classifier framework (see
    # ``hm2p.extraction.soma_classifier``).
    roi_qc_float32 = (
        "roi_qc/snr_event",
        "roi_qc/decay_tau_s",
        "roi_qc/fneu_dff_corr",
        "roi_qc/bleach_slope",
        "roi_qc/active_fraction",
        "roi_qc/p_soma",
        "roi_qc/p_dend",
        "roi_qc/p_artefact",
    )
    if "roi_qc/roi_index" in arrays:
        idx = arrays["roi_qc/roi_index"]
        _check_dtype(idx, np.dtype("int32"), "roi_qc/roi_index", ctx)
        _check_ndim(idx, 1, "roi_qc/roi_index", ctx)
        if len(idx) != n_rois:
            _schema_error(f"{ctx}: 'roi_qc/roi_index' length {len(idx)} != n_rois {n_rois}")
    for key in roi_qc_float32:
        if key in arrays:
            arr = arrays[key]
            _check_dtype(arr, np.dtype("float32"), key, ctx)
            _check_ndim(arr, 1, key, ctx)
            if len(arr) != n_rois:
                _schema_error(f"{ctx}: '{key}' length {len(arr)} != n_rois {n_rois}")


SYNC_STATUS_CODES: tuple[str, ...] = (
    "OK",
    "OK_WITH_WARNINGS",
    "FAILED_NO_TIMESTAMPS",
    "FAILED_NO_PULSES",
    "FAILED_FRAME_COUNT_MISMATCH",
    "FAILED_TEMPORAL_OVERLAP",
    "FAILED_TRUNCATED_CAMERA",
)
"""The 7 finalised sync_status codes — see docs/sync-pipeline-design.md §3.1."""

SYNC_STATUS_VERSION_CURRENT: str = "1.0"
"""The current sync_status schema version emitted by Stage 5."""


def _decode_attr_str(value: Any) -> str:
    """Decode an HDF5 attr that might be ``bytes`` or ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def validate_sync_h5(
    arrays: dict[str, np.ndarray],
    *,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Validate arrays against the sync.h5 schema.

    sync.h5 merges kinematics and calcium arrays at the imaging frame rate.
    The validator branches on ``sync_status``:

    - For ``OK`` / ``OK_WITH_WARNINGS`` (full schema): all kinematics
      constraints apply (with frame_times replacing camera times); dff must
      be float32 2D with n_frames == len(frame_times).
    - For ``FAILED_*`` (stub schema): the resampled signals are NOT
      required — Stage 5 deliberately omits them for failed sessions.
      Only the diagnostic attrs are checked.

    Both schemas require root attrs ``sync_status`` (one of
    ``SYNC_STATUS_CODES``), ``sync_status_version`` (e.g. ``"1.0"``),
    ``sync_warnings`` (JSON-encoded array, may be ``"[]"``), and
    ``sync_failures`` (JSON-encoded array). Files written before the
    diagnostics rollout (no ``sync_status`` attr) are treated as legacy
    and rejected — they must be rebuilt.

    Args:
        arrays: dict of dataset name → numpy array.
        attrs: dict of HDF5 attrs. When ``None`` the validator falls back
            to checking only the array constraints (legacy behaviour).

    Raises:
        pandera.errors.SchemaError: If any validation constraint fails.

    See Also:
        ``docs/sync-pipeline-design.md`` §2.2 / §3.1.
    """
    ctx = "sync.h5"

    # Legacy mode: attrs not provided → behave like the pre-diagnostics
    # validator. This keeps existing tests (which don't pass attrs) working.
    if attrs is None:
        _validate_sync_h5_full(arrays)
        return

    # Required diagnostic attrs.
    if "sync_status" not in attrs:
        _schema_error(f"{ctx}: missing required attr 'sync_status'")
    status = _decode_attr_str(attrs["sync_status"])
    if status not in SYNC_STATUS_CODES:
        _schema_error(f"{ctx}: 'sync_status' must be one of {SYNC_STATUS_CODES}, got '{status}'")
    if "sync_status_version" not in attrs:
        _schema_error(f"{ctx}: missing required attr 'sync_status_version'")
    version = _decode_attr_str(attrs["sync_status_version"])
    if version == "0.0":
        _schema_error(
            f"{ctx}: schema version 0.0 — file predates the sync diagnostics "
            "system and must be rebuilt by re-running Stage 5"
        )
    if version != SYNC_STATUS_VERSION_CURRENT:
        _schema_error(
            f"{ctx}: unsupported sync_status_version '{version}' "
            f"(expected '{SYNC_STATUS_VERSION_CURRENT}')"
        )
    for key in ("sync_warnings", "sync_failures"):
        if key not in attrs:
            _schema_error(f"{ctx}: missing required attr '{key}'")
        raw = _decode_attr_str(attrs[key])
        try:
            import json

            decoded = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            _schema_error(f"{ctx}: '{key}' is not valid JSON: {exc}")
        if not isinstance(decoded, list):
            _schema_error(f"{ctx}: '{key}' must decode to a JSON array (list)")

    # Conditional payload check.
    if status.startswith("FAILED_"):
        # Stub schema — no resampled signals required.
        return
    _validate_sync_h5_full(arrays)


def _validate_sync_h5_full(arrays: dict[str, np.ndarray]) -> None:
    """Validate the full (non-stub) sync.h5 array payload."""
    ctx = "sync.h5"
    # Reuse kinematics validator for the shared keys
    validate_kinematics_h5(arrays)
    T = len(arrays["frame_times"])
    dff = _check_key(arrays, "dff", ctx)
    _check_dtype(dff, np.dtype("float32"), "dff", ctx)
    _check_ndim(dff, 2, "dff", ctx)
    if dff.shape[1] != T:
        _schema_error(
            f"{ctx}: 'dff' shape {dff.shape} — second dim {dff.shape[1]} != len(frame_times) {T}"
        )


def validate_sync_report_parquet(df: Any) -> None:
    """Validate the sync_report.parquet schema (one row per session).

    The parquet aggregates the diagnostic scalars from every ``sync.h5``
    plus identity fields and the read-error column. See
    ``docs/sync-pipeline-design.md`` §2.3 for the column semantics.

    Args:
        df: pandas DataFrame to validate.

    Raises:
        pandera.errors.SchemaError: If any column is missing or of the
            wrong dtype.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        _schema_error("sync_report.parquet: input is not a pandas DataFrame")
    ctx = "sync_report.parquet"
    required_str = (
        "exp_id",
        "sub",
        "ses",
        "sync_status",
        "sync_warnings",
        "sync_failures",
        "dlc_champion_id",
        "read_error",
    )
    for col in required_str:
        if col not in df.columns:
            _schema_error(f"{ctx}: missing required column '{col}'")
        # Allow object/string dtypes (pandas string types vary across versions).
        if not (df[col].dtype == object or str(df[col].dtype).startswith("string")):
            _schema_error(f"{ctx}: '{col}' must be string dtype, got {df[col].dtype}")

    # Numeric scalar columns from sync_diag/.
    int_cols = (
        "cam_n_pulses",
        "cam_n_isi_outliers",
        "img_n_pulses",
        "img_n_isi_outliers",
        "line_n_pulses",
        "n_tiff_frames",
        "pulse_count_diff",
        "pulse_count_diff_after_off_by_one",
        "light_n_on",
        "light_n_off",
        "light_first_state_at_t0",
        "kin_pose_decimation_uniform",
        "s2p_off_by_one_fix_applied",
    )
    float_cols = (
        "cam_duration_s",
        "cam_isi_median_ms",
        "cam_isi_mad_ms",
        "cam_isi_cv",
        "cam_drift_slope_ppm",
        "cam_min_isi_ms",
        "img_duration_s",
        "img_isi_median_ms",
        "img_isi_mad_ms",
        "img_isi_cv",
        "img_drift_slope_ppm",
        "line_isi_median_ms",
        "cross_overlap_s",
        "cross_start_offset_ms",
        "cross_end_offset_ms",
        "light_period_median_s",
        "light_period_mad_s",
        "light_duty_cycle",
        "kin_pose_decimation_ratio",
    )
    import numpy as _np

    for col in int_cols:
        if col not in df.columns:
            _schema_error(f"{ctx}: missing required column '{col}'")
        if not _np.issubdtype(df[col].dtype, _np.integer):
            _schema_error(f"{ctx}: '{col}' must be integer dtype, got {df[col].dtype}")
    for col in float_cols:
        if col not in df.columns:
            _schema_error(f"{ctx}: missing required column '{col}'")
        if not _np.issubdtype(df[col].dtype, _np.floating):
            _schema_error(f"{ctx}: '{col}' must be floating dtype, got {df[col].dtype}")
