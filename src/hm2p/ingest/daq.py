"""Stage 0 — parse TDMS DAQ files → timestamps.h5.

Extracts per-session timing from National Instruments TDMS files written
during acquisition (SciScan + Basler camera):

  - Camera trigger times  → frame_times_camera  (N,) float64 s
  - SciScan line clock    → frame_times_imaging  (T,) float64 s
  - Lighting pulse times  → light_on_times / light_off_times  (L,) float64 s

All timestamps are in seconds since session start (first camera trigger = 0).

Output written to derivatives/movement/<sub>/<ses>/timestamps.h5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_tdms(tdms_path: Path) -> dict[str, np.ndarray]:
    """Parse a SciScan TDMS file and return timing arrays.

    Args:
        tdms_path: Path to the .tdms DAQ file.

    Returns:
        Dict with keys:
            'frame_times_camera'  (N,) float64 — camera frame timestamps (s)
            'frame_times_imaging' (T,) float64 — imaging frame timestamps (s)
            'light_on_times'      (L,) float64 — light-on pulse timestamps (s)
            'light_off_times'     (L,) float64 — light-off pulse timestamps (s)
            'fps_camera'          float — nominal camera frame rate
            'fps_imaging'         float — nominal imaging frame rate

    Raises:
        FileNotFoundError: If tdms_path does not exist.
        ValueError: If required channels are absent from the TDMS file.
    """
    raise NotImplementedError


def write_timestamps_h5(
    arrays: dict[str, np.ndarray],
    session_id: str,
    output_path: Path,
) -> None:
    """Write parsed timing arrays to timestamps.h5.

    Args:
        arrays: Output of parse_tdms().
        session_id: Canonical session identifier stored as HDF5 attribute.
        output_path: Destination file path (created or overwritten).
    """
    raise NotImplementedError


def run(tdms_path: Path, session_id: str, output_path: Path) -> None:
    """End-to-end Stage 0 DAQ parsing: TDMS → timestamps.h5.

    Args:
        tdms_path: Path to the TDMS file.
        session_id: Canonical session identifier.
        output_path: Destination HDF5 file path.
    """
    arrays = parse_tdms(tdms_path)
    write_timestamps_h5(arrays, session_id, output_path)
