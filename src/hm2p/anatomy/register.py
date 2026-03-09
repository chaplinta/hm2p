"""Brain registration to Allen CCFv3 atlas via brainreg.

Registers serial2p whole-brain volumes to the Allen Mouse Brain Common
Coordinate Framework v3 (CCFv3) using the BrainGlobe brainreg tool.

References
----------
Tyson et al. 2022. "Accurate determination of marker location within
whole-brain microscopy images." Scientific Reports.
doi:10.1038/s41598-021-04676-9

Wang et al. 2020. "The Allen Mouse Brain Common Coordinate Framework."
Cell. doi:10.1016/j.cell.2020.04.007
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


def find_signal_channel(brain_dir: Path) -> Path:
    """Find the green (signal) channel TIFF in a brain directory.

    Searches for files containing ``'green'`` (case-insensitive) in
    the filename within *brain_dir* and its subdirectories.

    Args:
        brain_dir: Directory containing whole-brain TIFF volumes.

    Returns:
        Path to the green-channel TIFF file.

    Raises:
        FileNotFoundError: If no file with 'green' in the name is found.
        RuntimeError: If multiple green-channel files are found.
    """
    matches: list[Path] = []
    for p in brain_dir.rglob("*"):
        if p.is_file() and "green" in p.name.lower():
            matches.append(p)

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No file with 'green' in the name found under {brain_dir}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple green-channel files found under {brain_dir}: {matches}"
        )

    logger.info("found_signal_channel", path=str(matches[0]))
    return matches[0]


def run_brainreg(
    signal_path: Path,
    output_dir: Path,
    voxel_size: tuple[float, float, float] = (25.0, 25.0, 25.0),
    orientation: str = "psl",
    atlas: str = "allen_mouse_25um",
) -> Path:
    """Run brainreg registration on a signal channel TIFF.

    Calls the ``brainreg`` CLI via :func:`subprocess.run`.  The output
    directory is created if it does not exist.

    Args:
        signal_path: Path to the signal-channel TIFF volume.
        output_dir: Directory where brainreg writes its output.
        voxel_size: Isotropic voxel dimensions in micrometres (X, Y, Z).
        orientation: Three-letter BrainGlobe orientation code.
        atlas: BrainGlobe atlas name.

    Returns:
        The *output_dir* path (same as the input argument).

    Raises:
        FileNotFoundError: If *signal_path* does not exist.
        subprocess.CalledProcessError: If brainreg exits with non-zero status.
    """
    if not signal_path.exists():
        raise FileNotFoundError(f"Signal file not found: {signal_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    vx, vy, vz = voxel_size
    cmd: list[str] = [
        "brainreg",
        str(signal_path),
        str(output_dir),
        "-v",
        str(vx),
        str(vy),
        str(vz),
        "--orientation",
        orientation,
        "--atlas",
        atlas,
        "--save-original-orientation",
    ]

    logger.info(
        "running_brainreg",
        signal=str(signal_path),
        output=str(output_dir),
        voxel_size=voxel_size,
        orientation=orientation,
        atlas=atlas,
    )

    subprocess.run(cmd, check=True)

    logger.info("brainreg_complete", output=str(output_dir))
    return output_dir


def batch_register(
    brains_dir: Path,
    output_dir: Path,
    voxel_size: tuple[float, float, float] = (25.0, 25.0, 25.0),
    orientation: str = "psl",
) -> list[dict]:
    """Register all brains in a directory.

    Iterates over subdirectories of *brains_dir*, finds the green
    signal channel in each, and runs brainreg.  Each brain's output
    is placed in a subdirectory of *output_dir* named after the signal
    file (without extension).

    Args:
        brains_dir: Parent directory containing one subdirectory per brain.
        output_dir: Parent directory for all registration outputs.
        voxel_size: Isotropic voxel dimensions in micrometres.
        orientation: Three-letter BrainGlobe orientation code.

    Returns:
        List of dicts with keys ``brain_dir``, ``signal_path``,
        ``output_dir``, ``status`` (``"success"`` or ``"error"``),
        and ``error`` (error message string, only on failure).
    """
    results: list[dict] = []

    subdirs = sorted(
        p for p in brains_dir.iterdir() if p.is_dir()
    )

    if not subdirs:
        logger.warning("no_brain_directories", brains_dir=str(brains_dir))
        return results

    for brain_subdir in subdirs:
        result: dict = {"brain_dir": str(brain_subdir)}
        try:
            signal_path = find_signal_channel(brain_subdir)
            result["signal_path"] = str(signal_path)

            base_name = signal_path.stem
            out = output_dir / base_name
            result["output_dir"] = str(out)

            run_brainreg(
                signal_path,
                out,
                voxel_size=voxel_size,
                orientation=orientation,
            )
            result["status"] = "success"

        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            logger.error(
                "batch_register_error",
                brain_dir=str(brain_subdir),
                error=str(exc),
            )

        results.append(result)

    logger.info(
        "batch_register_complete",
        total=len(results),
        success=sum(1 for r in results if r["status"] == "success"),
        errors=sum(1 for r in results if r["status"] == "error"),
    )
    return results
