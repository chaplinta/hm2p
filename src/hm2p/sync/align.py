"""Stage 5 — neural–behavioural synchronisation + sync diagnostics.

Resamples behavioural kinematics from camera rate (~100 Hz) to imaging rate
(~30 Hz) by linear interpolation at each imaging frame timestamp. Merges
calcium signals and resampled behaviour into sync.h5. Computes diagnostic
scalars (median/MAD ISI, drift slope, overlap) and classifies each session
into a ``sync_status`` tier (7 codes, first match wins) per
``docs/sync-pipeline-design.md`` §3.

Input:  kinematics.h5     (camera rate, N frames)
        ca.h5             (imaging rate, T frames)
        timestamps.h5     (Stage 0 — line clock, light pulses, tdms_diag)
        config/sync.yaml  (thresholds — falls back to packaged defaults)
Output: sync.h5           (imaging rate, T frames — all signals aligned + diag)

For ``OK`` / ``OK_WITH_WARNINGS`` sessions, sync.h5 contains the full
resampled payload plus the sync_diag/ group and JSON-encoded warnings/
failures lists. For ``FAILED_*`` sessions, sync.h5 is a stub: only the
classification + diag attrs are written; downstream Stage 6 refuses to
consume it unless ``include_failed_sync=True`` is passed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Keys in kinematics.h5 that are boolean (use nearest-neighbour resampling)
_BOOL_KEYS: frozenset[str] = frozenset({"light_on", "bad_behav", "active"})

# Keys that are categorical integers (use nearest-neighbour, keep dtype)
_CATEGORICAL_KEYS: frozenset[str] = frozenset({"syllable_id"})


def resample_to_imaging_rate(
    values: np.ndarray,
    src_times: np.ndarray,
    dst_times: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Resample a 1D signal from src_times to dst_times via interpolation.

    Args:
        values: (N,) float array — signal at camera rate.
        src_times: (N,) float64 — source timestamps (seconds).
        dst_times: (T,) float64 — destination timestamps (imaging frame times).
        method: Interpolation method: 'linear' (default) or 'nearest'.

    Returns:
        (T,) float — signal resampled to dst_times.
    """
    if method == "nearest":
        indices = np.searchsorted(src_times, dst_times, side="left")
        indices = np.clip(indices, 0, len(values) - 1)
        return values[indices].astype(float)
    return np.interp(dst_times, src_times, values)


def resample_bool_to_imaging_rate(
    mask: np.ndarray,
    src_times: np.ndarray,
    dst_times: np.ndarray,
) -> np.ndarray:
    """Resample a boolean mask using true nearest-neighbour interpolation.

    Args:
        mask: (N,) bool — boolean signal at camera rate.
        src_times: (N,) float64 — source timestamps.
        dst_times: (T,) float64 — imaging frame timestamps.

    Returns:
        (T,) bool — mask resampled to imaging rate.
    """
    # Find true nearest neighbour (not just next-left).
    # searchsorted("left") gives the first src >= dst, but the previous
    # src might be closer. Compare both and pick the nearer one.
    idx_right = np.searchsorted(src_times, dst_times, side="left")
    idx_right = np.clip(idx_right, 0, len(mask) - 1)
    idx_left = np.clip(idx_right - 1, 0, len(mask) - 1)

    dist_right = np.abs(src_times[idx_right] - dst_times)
    dist_left = np.abs(src_times[idx_left] - dst_times)
    indices = np.where(dist_left <= dist_right, idx_left, idx_right)
    return mask[indices]


# Provenance attributes copied from kinematics.h5 onto sync.h5 (full *or*
# stub). ``dlc_champion_id`` in particular is required for the staleness
# contract documented in ``docs/dlc-champion-model.md`` — every derivative
# produced from DLC pose data must be stamped with the champion id, and
# that holds for FAILED_* stubs too. Without it, the report parquet cannot
# distinguish a stale-but-failed session from a current-but-failed one.
_KIN_PROVENANCE_KEYS: tuple[str, ...] = (
    "tracker",
    "dlc_model_name",
    "dlc_snapshot",
    "dlc_champion_id",
    "confidence_threshold",
    "orientation_deg",
    "scale_mm_per_px",
)


def _stub_attrs(
    session_id: str,
    status: str,
    warnings: list[str],
    failures: list[str],
    diag_attrs: dict | None = None,
    provenance_attrs: dict | None = None,
) -> dict:
    """Build the diag-only root attrs for a sync.h5 (stub or full).

    ``provenance_attrs`` carries the kinematics.h5 provenance keys
    (``dlc_champion_id`` etc.) that must be stamped on every sync.h5,
    including FAILED_* stubs, per the staleness contract in
    ``docs/dlc-champion-model.md``.
    """
    from hm2p.io.hdf5 import SYNC_STATUS_VERSION_CURRENT
    from hm2p.sync.diagnostics import encode_codes_json

    attrs: dict = {
        "session_id": session_id,
        "sync_status": status,
        "sync_status_version": SYNC_STATUS_VERSION_CURRENT,
        "sync_warnings": encode_codes_json(warnings),
        "sync_failures": encode_codes_json(failures),
    }
    if provenance_attrs:
        for k, v in provenance_attrs.items():
            if v is not None:
                attrs[k] = v
    if diag_attrs:
        for k, v in diag_attrs.items():
            attrs[f"sync_diag/{k}"] = v
    return attrs


def _write_stub(
    output_path: Path,
    session_id: str,
    status: str,
    warnings: list[str],
    failures: list[str],
    diag_attrs: dict | None = None,
    provenance_attrs: dict | None = None,
) -> None:
    """Write a stub sync.h5 — diag + provenance attrs only, no resampled signals."""
    from hm2p.io.hdf5 import write_h5

    attrs = _stub_attrs(
        session_id,
        status,
        warnings,
        failures,
        diag_attrs=diag_attrs,
        provenance_attrs=provenance_attrs,
    )
    log.info(
        "sync stub written: session=%s status=%s warnings=%s failures=%s",
        session_id,
        status,
        warnings,
        failures,
    )
    write_h5(output_path, arrays={}, attrs=attrs)


def _read_kin_provenance(kinematics_h5: Path) -> dict:
    """Read kinematics.h5 provenance attrs (or empty dict if file is missing).

    Reads the keys listed in :data:`_KIN_PROVENANCE_KEYS`. Missing keys are
    silently omitted from the returned dict. This is called early in
    ``run()`` so that even FAILED_* sessions stamp ``dlc_champion_id`` (and
    related provenance) onto the stub sync.h5 — required by the staleness
    contract in ``docs/dlc-champion-model.md``.
    """
    from hm2p.io.hdf5 import read_attrs

    if not Path(kinematics_h5).exists():
        return {}
    try:
        kin_attrs = read_attrs(kinematics_h5)
    except Exception as exc:  # pragma: no cover — defensive path
        log.warning("Failed to read kinematics.h5 attrs at %s: %s", kinematics_h5, exc)
        return {}
    return {k: kin_attrs[k] for k in _KIN_PROVENANCE_KEYS if k in kin_attrs}


def _build_diagnostics(
    *,
    timestamps_present: bool,
    timestamps_arrays: dict | None,
    timestamps_attrs: dict | None,
    kin: dict | None,
    ca: dict | None,
    cfg: dict,
    s2p_off_by_one_fix_applied: int,
) -> tuple[object, str, list[str], list[str], dict]:
    """Run the diagnostics module against a session's loaded data.

    Returns ``(scalars, status, warnings, failures, diag_attrs)``.
    """
    from hm2p.sync.diagnostics import (
        build_scalars,
        classify,
        scalars_to_diag_attrs,
    )

    cam_times = (timestamps_arrays or {}).get("frame_times_camera") if timestamps_present else None
    img_times = (
        (timestamps_arrays or {}).get("frame_times_imaging") if timestamps_present else None
    )
    line_times = (timestamps_arrays or {}).get("line_clock_times") if timestamps_present else None
    light_on = (timestamps_arrays or {}).get("light_on_times") if timestamps_present else None
    light_off = (timestamps_arrays or {}).get("light_off_times") if timestamps_present else None

    fps_camera = float((timestamps_attrs or {}).get("fps_camera", 100.0))
    fps_imaging = float((timestamps_attrs or {}).get("fps_imaging", 30.0))

    # tdms_diag: peel out the slash-prefixed keys.
    tdms_diag: dict[str, float] = {}
    if timestamps_attrs:
        for k, v in timestamps_attrs.items():
            if isinstance(k, str) and k.startswith("tdms_diag/"):
                key = k[len("tdms_diag/") :]
                try:
                    tdms_diag[key] = float(v)
                except (TypeError, ValueError):
                    continue

    # n_tiff_frames from ca.h5 (Suite2p ops would be ideal but ca.h5 is what
    # we have at this stage; use dff column count which equals nframes).
    n_tiff_frames = -1
    if ca is not None and "dff" in ca:
        n_tiff_frames = int(ca["dff"].shape[1])

    # Pose decimation diagnostics from kinematics.h5 attrs (if present).
    kin_pose_decimation_ratio = 1.0
    kin_pose_decimation_uniform = 1
    # Future kinematics.h5 versions can record these as attrs; for now,
    # they default to "uniform decimation" (no warning).

    scalars = build_scalars(
        timestamps_present=timestamps_present,
        cam_times=cam_times,
        img_times=img_times,
        line_times=line_times,
        light_on=light_on,
        light_off=light_off,
        fps_camera=fps_camera,
        fps_imaging=fps_imaging,
        n_tiff_frames=n_tiff_frames,
        s2p_off_by_one_fix_applied=s2p_off_by_one_fix_applied,
        kin_pose_decimation_ratio=kin_pose_decimation_ratio,
        kin_pose_decimation_uniform=kin_pose_decimation_uniform,
        tdms_diag=tdms_diag,
        cfg=cfg,
    )
    status, warnings, failures = classify(scalars, cfg=cfg)
    diag_attrs = scalars_to_diag_attrs(scalars)
    return scalars, status, warnings, failures, diag_attrs


def run(
    kinematics_h5: Path,
    ca_h5: Path,
    session_id: str,
    output_path: Path,
    *,
    timestamps_h5: Path | None = None,
    config_path: Path | str | None = None,
) -> None:
    """End-to-end Stage 5: kinematics.h5 + ca.h5 + timestamps.h5 → sync.h5.

    Resamples kinematics from camera rate to imaging rate by linear
    interpolation (continuous signals) or nearest-neighbour (booleans).
    Combines with calcium arrays (already at imaging rate). Computes
    sync diagnostics from the timestamps.h5 pulse trains and classifies
    the session into a ``sync_status`` tier per
    ``docs/sync-pipeline-design.md`` §3.

    Failure-closed: if ``sync_status`` starts with ``FAILED_``, the
    output sync.h5 is a stub with only the classification + diag attrs
    (no resampled signals). Stage 6 refuses to consume the stub by
    default.

    Args:
        kinematics_h5: Stage 3 kinematics output.
        ca_h5: Stage 4 calcium output.
        session_id: Canonical session identifier.
        output_path: Destination sync.h5 file path.
        timestamps_h5: Stage 0 timestamps file. When ``None``, the path is
            inferred from the kinematics_h5 path (replacing
            ``derivatives/movement/`` with ``derivatives/timestamps/``).
            If the file is missing, sync_status is FAILED_NO_TIMESTAMPS
            and a stub is written.
        config_path: Path to ``config/sync.yaml``; defaults to packaged
            thresholds when missing.
    """
    from hm2p.io.hdf5 import read_attrs, read_h5, write_h5
    from hm2p.sync.diagnostics import load_config

    cfg = load_config(config_path)

    # --- Read kinematics provenance early so it is available to FAILED_*
    #     stubs (champion-id staleness contract — see
    #     docs/dlc-champion-model.md). Empty dict if kinematics.h5 absent. ---
    provenance_attrs = _read_kin_provenance(Path(kinematics_h5))

    # --- Resolve timestamps.h5 path ---
    ts_path = timestamps_h5
    if ts_path is None:
        # Convention: derivatives/movement/<sub>/<ses>/kinematics.h5
        #          → derivatives/timestamps/<sub>/<ses>/timestamps.h5
        kin_p = Path(kinematics_h5)
        candidate = (
            kin_p.parent.parent.parent.parent
            / "timestamps"
            / kin_p.parent.parent.name
            / kin_p.parent.name
            / "timestamps.h5"
        )
        ts_path = candidate

    timestamps_present = Path(ts_path).exists() if ts_path is not None else False

    # --- Read inputs (graceful on absence) ---
    kin: dict | None = None
    ca: dict | None = None
    timestamps_arrays: dict | None = None
    timestamps_attrs: dict | None = None
    try:
        if timestamps_present:
            timestamps_arrays = read_h5(ts_path)
            timestamps_attrs = read_attrs(ts_path)
    except Exception as exc:  # pragma: no cover — defensive path
        log.warning("Failed to read timestamps.h5 at %s: %s", ts_path, exc)
        timestamps_present = False

    if Path(kinematics_h5).exists():
        kin = read_h5(kinematics_h5)
    if Path(ca_h5).exists():
        ca = read_h5(ca_h5)

    # --- Apply Suite2p off-by-one trim (with explicit logging) ---
    s2p_off_by_one_fix_applied = 0
    if kin is not None and ca is not None:
        src_times = kin["frame_times"]
        dst_times = ca["frame_times"]
        if "dff" in ca:
            n_imaging = ca["dff"].shape[1]
            if len(dst_times) == n_imaging + 1:
                log.info(
                    "Suite2p off-by-one trim: dst_times had %d entries for %d dF/F columns",
                    len(dst_times),
                    n_imaging,
                )
                dst_times = dst_times[:n_imaging]
                s2p_off_by_one_fix_applied = 1
    else:
        src_times = None
        dst_times = None

    # --- Diagnostics + classification ---
    scalars, status, warnings, failures, diag_attrs = _build_diagnostics(
        timestamps_present=timestamps_present,
        timestamps_arrays=timestamps_arrays,
        timestamps_attrs=timestamps_attrs,
        kin=kin,
        ca=ca,
        cfg=cfg,
        s2p_off_by_one_fix_applied=s2p_off_by_one_fix_applied,
    )

    # --- Stub path: FAILED_* sessions get only the classification ---
    if status.startswith("FAILED_"):
        _write_stub(
            output_path,
            session_id,
            status,
            warnings,
            failures,
            diag_attrs=diag_attrs,
            provenance_attrs=provenance_attrs,
        )
        return

    # --- Full payload required: resample + write ---
    # If kin / ca are somehow missing here we fall back to a stub.
    if kin is None or ca is None or src_times is None or dst_times is None:
        log.warning("Stage 5: kin or ca missing despite OK status — falling back to stub")
        _write_stub(
            output_path,
            session_id,
            "FAILED_NO_TIMESTAMPS",
            warnings,
            ["no_pulses: kinematics or ca arrays missing"],
            diag_attrs=diag_attrs,
            provenance_attrs=provenance_attrs,
        )
        return

    datasets: dict[str, np.ndarray] = {}

    # Resample kinematics to imaging rate
    for key, arr in kin.items():
        if key == "frame_times":
            continue
        if key in _BOOL_KEYS:
            datasets[key] = resample_bool_to_imaging_rate(arr, src_times, dst_times)
        elif key in _CATEGORICAL_KEYS:
            # True nearest-neighbour for integer categories (syllable_id).
            idx_right = np.searchsorted(src_times, dst_times, side="left")
            idx_right = np.clip(idx_right, 0, len(arr) - 1)
            idx_left = np.clip(idx_right - 1, 0, len(arr) - 1)
            dist_right = np.abs(src_times[idx_right] - dst_times)
            dist_left = np.abs(src_times[idx_left] - dst_times)
            indices = np.where(dist_left <= dist_right, idx_left, idx_right)
            datasets[key] = arr[indices]
        elif key == "syllable_prob" and arr.ndim == 2:
            resampled = np.empty((len(dst_times), arr.shape[1]), dtype=np.float32)
            for col in range(arr.shape[1]):
                resampled[:, col] = resample_to_imaging_rate(arr[:, col], src_times, dst_times)
            datasets[key] = resampled
        else:
            datasets[key] = resample_to_imaging_rate(arr, src_times, dst_times).astype(np.float32)

    # Copy calcium arrays (already at imaging rate).
    for key, arr in ca.items():
        if key == "frame_times":
            datasets["frame_times"] = dst_times
        else:
            datasets[key] = arr

    # Build combined bad-frame mask: bad_frames = bad_imaging_frames | bad_behav.
    n_frames = len(dst_times)
    bad_imaging = datasets.get("bad_imaging_frames")
    bad_behav = datasets.get("bad_behav")
    if bad_imaging is not None and bad_behav is not None:
        bad_imaging_t = np.asarray(bad_imaging[:n_frames], dtype=bool)
        bad_behav_t = np.asarray(bad_behav[:n_frames], dtype=bool)
        datasets["bad_frames"] = bad_imaging_t | bad_behav_t
    elif bad_imaging is not None:
        datasets["bad_frames"] = np.asarray(bad_imaging[:n_frames], dtype=bool)
    elif bad_behav is not None:
        datasets["bad_frames"] = np.asarray(bad_behav[:n_frames], dtype=bool)

    # Build root attrs: start from ca.h5, overlay kinematics provenance
    # (read once at the top of run() so the same dict is used for full and
    # stub paths), then sync_status + diag attrs.
    from hm2p.io.hdf5 import SYNC_STATUS_VERSION_CURRENT
    from hm2p.sync.diagnostics import encode_codes_json

    attrs = dict(read_attrs(ca_h5))
    for key, value in provenance_attrs.items():
        if value is not None:
            attrs[key] = value
    attrs["session_id"] = session_id
    attrs["sync_status"] = status
    attrs["sync_status_version"] = SYNC_STATUS_VERSION_CURRENT
    attrs["sync_warnings"] = encode_codes_json(warnings)
    attrs["sync_failures"] = encode_codes_json(failures)
    for k, v in diag_attrs.items():
        attrs[f"sync_diag/{k}"] = v

    log.info("sync written: session=%s status=%s n_warnings=%d", session_id, status, len(warnings))
    write_h5(output_path, datasets, attrs=attrs)
