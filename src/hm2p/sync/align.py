"""Stage 5 — neural–behavioural synchronisation.

Resamples behavioural kinematics from camera rate (~100 Hz) to imaging rate
(~30 Hz) by linear interpolation at each imaging frame timestamp. Merges
calcium signals and resampled behaviour into sync.h5.

Input:  kinematics.h5  (camera rate, N frames)
        ca.h5          (imaging rate, T frames)
Output: sync.h5        (imaging rate, T frames — all signals aligned)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Keys in kinematics.h5 that are boolean (use nearest-neighbour resampling)
_BOOL_KEYS: frozenset[str] = frozenset({"light_on", "bad_behav", "active"})


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
    """Resample a boolean mask using nearest-neighbour interpolation.

    Args:
        mask: (N,) bool — boolean signal at camera rate.
        src_times: (N,) float64 — source timestamps.
        dst_times: (T,) float64 — imaging frame timestamps.

    Returns:
        (T,) bool — mask resampled to imaging rate.
    """
    indices = np.searchsorted(src_times, dst_times, side="left")
    indices = np.clip(indices, 0, len(mask) - 1)
    return mask[indices]


def run(
    kinematics_h5: Path,
    ca_h5: Path,
    session_id: str,
    output_path: Path,
) -> None:
    """End-to-end Stage 5: kinematics.h5 + ca.h5 → sync.h5.

    Resamples all kinematics signals from camera rate to imaging rate by
    linear interpolation (continuous signals) or nearest-neighbour (booleans).
    Combines with calcium arrays (already at imaging rate) and writes sync.h5.

    Args:
        kinematics_h5: Stage 3 kinematics output.
        ca_h5: Stage 4 calcium output.
        session_id: Canonical session identifier.
        output_path: Destination sync.h5 file path.
    """
    from hm2p.io.hdf5 import read_attrs, read_h5, write_h5

    kin = read_h5(kinematics_h5)
    ca = read_h5(ca_h5)

    src_times = kin["frame_times"]  # camera rate timestamps
    dst_times = ca["frame_times"]  # imaging rate timestamps (target grid)

    datasets: dict[str, np.ndarray] = {}

    # Resample kinematics to imaging rate
    for key, arr in kin.items():
        if key == "frame_times":
            continue
        if key in _BOOL_KEYS:
            datasets[key] = resample_bool_to_imaging_rate(arr, src_times, dst_times)
        else:
            datasets[key] = resample_to_imaging_rate(arr, src_times, dst_times).astype(np.float32)

    # Copy calcium arrays (already at imaging rate)
    for key, arr in ca.items():
        datasets[key] = arr  # includes frame_times, dff, spikes, etc.

    # Inherit ca.h5 root attrs, override session_id
    attrs = dict(read_attrs(ca_h5))
    attrs["session_id"] = session_id

    write_h5(output_path, datasets, attrs=attrs)
