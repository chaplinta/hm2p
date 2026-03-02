"""Stage 0 — parse TDMS DAQ files → timestamps.h5.

Extracts per-session timing from National Instruments TDMS files written
during acquisition (SciScan + Basler camera):

  - Camera trigger times  → frame_times_camera  (N,) float64 s
  - SciScan line clock    → frame_times_imaging  (T,) float64 s
  - Lighting pulse times  → light_on_times / light_off_times  (L,) float64 s

All timestamps are in seconds since session start (first camera trigger = 0).

Output written to derivatives/movement/<sub>/<ses>/timestamps.h5.

TDMS channel layout (from meta.txt [DAQ] section)
--------------------------------------------------
Group naming convention: "{groupname} - {channame}"  e.g. "maze-rose - cam_trigger"

  cam_trigger   : camera trigger pulses (uint8 0/1, ~100 Hz)
  sci_sync      : SciScan line-clock pulses (uint8 0/1, ~9.6 Hz × y_pix lines)
  lights        : overhead light control (uint8 0/1)

Rising edges detected via _rising_edges(); frame times derived from line
clock by taking every y_pix-th pulse (from SciScan .ini y.pixels field).
"""

from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np

from hm2p.io.hdf5 import write_h5

# Keys in the arrays dict that are scalars (stored as HDF5 attrs, not datasets)
_SCALAR_KEYS: frozenset[str] = frozenset({"fps_camera", "fps_imaging"})


# ---------------------------------------------------------------------------
# Pure helper functions (no I/O — fully unit-testable)
# ---------------------------------------------------------------------------


def _rising_edges(data: np.ndarray, threshold: float) -> np.ndarray:
    """Return sample indices of upward threshold crossings.

    A crossing occurs at index i when data[i-1] <= threshold < data[i].

    Args:
        data: 1D numeric array.
        threshold: Crossing threshold value.

    Returns:
        1D int array of crossing indices (length ≤ len(data) - 1).
    """
    return np.flatnonzero((data[:-1] <= threshold) & (data[1:] > threshold)) + 1


def _frame_times_from_line_clock(
    line_times: np.ndarray,
    y_pix: int,
) -> np.ndarray:
    """Derive imaging frame timestamps from SciScan line-clock pulse times.

    SciScan emits one DAQ pulse per scan line. A frame completes after
    y_pix lines have been scanned. Frame timestamps are taken at the
    last line of each frame (frame-end convention, matching legacy code).

    Args:
        line_times: (M,) float64 — timestamps of each line-clock pulse (s).
        y_pix: Number of scan lines per imaging frame.

    Returns:
        (T,) float64 — imaging frame timestamps, where T = M // y_pix.
    """
    return line_times[y_pix - 1 :: y_pix]


def _meta_txt_path(tdms_path: Path) -> Path:
    """Derive the meta.txt path from a -di.tdms path.

    '…/20210823_17_00_04_1114353_maze-rose-di.tdms'
    → '…/20210823_17_00_04_1114353_maze-rose.meta.txt'
    """
    name = tdms_path.name
    if name.endswith("-di.tdms"):
        base = name[: -len("-di.tdms")]
    else:
        base = name.rsplit(".", 1)[0]
    return tdms_path.parent / (base + ".meta.txt")


# ---------------------------------------------------------------------------
# TDMS I/O helpers
# ---------------------------------------------------------------------------


def _get_di_channel(di_file: object, group_name: str, chan_name: str) -> object:
    """Return the first channel from a DI TDMS group.

    DI groups are named '{group_name} - {chan_name}' (SciScan convention).

    Args:
        di_file: Open nptdms.TdmsFile.
        group_name: Base group name from meta.txt (e.g. 'maze-rose').
        chan_name: Channel name from meta.txt (e.g. 'cam_trigger').

    Returns:
        nptdms.TdmsChannel — the first (only) channel in the group.

    Raises:
        ValueError: If the group or any channel is absent.
    """
    full_name = f"{group_name} - {chan_name}"
    if full_name not in di_file:
        raise ValueError(
            f"Required TDMS channel group '{full_name}' not found in file. "
            f"Available groups: {[g.name for g in di_file.groups()]}"
        )
    channels = di_file[full_name].channels()
    if not channels:
        raise ValueError(f"No channels in TDMS group '{full_name}'")
    return channels[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_tdms(tdms_path: Path) -> dict[str, np.ndarray]:
    """Parse a SciScan TDMS file and return timing arrays.

    Reads the companion meta.txt and .ini files (in the same directory)
    to determine channel names, y_pix, and nominal frame rates.

    Args:
        tdms_path: Path to the -di.tdms DAQ file.

    Returns:
        Dict with keys:
            'frame_times_camera'  (N,) float64 — camera frame timestamps (s)
            'frame_times_imaging' (T,) float64 — imaging frame timestamps (s)
            'light_on_times'      (L,) float64 — light-on pulse timestamps (s)
            'light_off_times'     (L,) float64 — light-off pulse timestamps (s)
            'fps_camera'          float — nominal camera frame rate
            'fps_imaging'         float — nominal imaging frame rate

        All timestamps are relative to the first camera trigger (t=0).

    Raises:
        FileNotFoundError: If tdms_path or companion files do not exist.
        ValueError: If required channels or config keys are absent.
    """
    import nptdms  # heavy import — defer to call-site

    if not tdms_path.exists():
        raise FileNotFoundError(f"TDMS file not found: {tdms_path}")

    session_dir = tdms_path.parent

    # --- meta.txt (channel names + fps) ---
    meta_path = _meta_txt_path(tdms_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.txt not found: {meta_path}")

    cfg = configparser.ConfigParser()
    cfg.read(meta_path)
    try:
        group_name = cfg["DAQ"]["groupname"]
        fps_camera = float(cfg["Video"]["fps"])
        cam_chan_name = cfg["DAQ"]["cameratriggerchanname"]
        sci_chan_name = cfg["DAQ"]["sciscanchanname"]
        lights_chan_name = cfg["DAQ"]["lightschanname"]
        ini_filename = Path(cfg["SciScan"]["inifile"].replace("\\", "/")).name
    except KeyError as exc:
        raise ValueError(f"Required key missing in {meta_path}: {exc}") from exc

    # --- SciScan .ini (y_pix + fps_imaging) ---
    ini_path = session_dir / ini_filename
    if not ini_path.exists():
        matches = list(session_dir.glob("*_XYT.ini"))
        if not matches:
            raise FileNotFoundError(f"No *_XYT.ini in {session_dir}")
        ini_path = matches[0]

    ini = configparser.ConfigParser()
    ini.read(ini_path)
    try:
        y_pix = int(float(ini["_"]["y.pixels"]))
        fps_imaging = float(ini["_"]["frames.p.sec"])
    except KeyError as exc:
        raise ValueError(f"Required key missing in {ini_path}: {exc}") from exc

    # --- Read TDMS channels ---
    with nptdms.TdmsFile.read(tdms_path) as di_file:
        cam_chan = _get_di_channel(di_file, group_name, cam_chan_name)
        sci_chan = _get_di_channel(di_file, group_name, sci_chan_name)
        light_chan = _get_di_channel(di_file, group_name, lights_chan_name)

        cam_data = np.asarray(cam_chan.data, dtype=float)
        cam_time = np.asarray(cam_chan.time_track(), dtype=np.float64)

        sci_data = np.asarray(sci_chan.data, dtype=float)
        sci_time = np.asarray(sci_chan.time_track(), dtype=np.float64)

        light_data = np.asarray(light_chan.data, dtype=float)
        light_time = np.asarray(light_chan.time_track(), dtype=np.float64)

    # --- Detect events ---
    cam_idxs = _rising_edges(cam_data, 0.9)
    if cam_idxs.size == 0:
        raise ValueError("No camera trigger pulses found in TDMS file")

    sci_line_idxs = _rising_edges(sci_data, 0.5)
    sci_frame_idxs = sci_line_idxs[y_pix - 1 :: y_pix]

    light_on_idxs = _rising_edges(light_data, 0.9)
    light_off_idxs = _rising_edges(1.0 - light_data, 0.9)

    # --- Zero to first camera trigger ---
    t0 = cam_time[cam_idxs[0]]

    return {
        "frame_times_camera": (cam_time[cam_idxs] - t0).astype(np.float64),
        "frame_times_imaging": (sci_time[sci_frame_idxs] - t0).astype(np.float64),
        "light_on_times": (light_time[light_on_idxs] - t0).astype(np.float64),
        "light_off_times": (light_time[light_off_idxs] - t0).astype(np.float64),
        "fps_camera": np.float64(fps_camera),
        "fps_imaging": np.float64(fps_imaging),
    }


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
    datasets = {k: v for k, v in arrays.items() if k not in _SCALAR_KEYS}
    attrs: dict[str, object] = {"session_id": session_id}
    for key in _SCALAR_KEYS:
        if key in arrays:
            attrs[key] = float(arrays[key])
    write_h5(output_path, datasets, attrs=attrs)


def run(tdms_path: Path, session_id: str, output_path: Path) -> None:
    """End-to-end Stage 0 DAQ parsing: TDMS → timestamps.h5.

    Args:
        tdms_path: Path to the TDMS file.
        session_id: Canonical session identifier.
        output_path: Destination HDF5 file path.
    """
    arrays = parse_tdms(tdms_path)
    write_timestamps_h5(arrays, session_id, output_path)
