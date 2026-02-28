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
    raise NotImplementedError


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
    raise NotImplementedError


def run(
    kinematics_h5: Path,
    ca_h5: Path,
    session_id: str,
    output_path: Path,
) -> None:
    """End-to-end Stage 5: kinematics.h5 + ca.h5 → sync.h5.

    Args:
        kinematics_h5: Stage 3 kinematics output.
        ca_h5: Stage 4 calcium output.
        session_id: Canonical session identifier.
        output_path: Destination sync.h5 file path.
    """
    raise NotImplementedError
