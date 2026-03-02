"""Stage 1 — run Suite2p on raw TIFF stacks.

Wraps ``suite2p.run_s2p()`` to produce the standard plane0/ output directory
containing F.npy, Fneu.npy, iscell.npy, stat.npy, and ops.npy.

Suite2p is an optional dependency (GPU recommended). Install via:
    pip install suite2p
    # or conda install -c conda-forge suite2p
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def default_ops(fps: float = 29.97) -> dict[str, Any]:
    """Return sensible default Suite2p ops for hm2p single-plane GCaMP imaging.

    These defaults are tuned for:
    - Single-plane recordings (~30 Hz)
    - GCaMP7f / GCaMP8f indicators
    - ~512×512 FOV
    - RSP cortex (moderate cell density)

    Args:
        fps: Imaging frame rate in Hz.

    Returns:
        Dict of Suite2p ops suitable for passing to ``suite2p.run_s2p()``.
    """
    return {
        "fs": fps,
        "nplanes": 1,
        "nchannels": 1,
        "tau": 1.0,  # GCaMP7f decay time ~1s
        "do_registration": True,
        "nonrigid": True,
        "block_size": [128, 128],
        "keep_movie_raw": False,
        "delete_bin": True,
        "roidetect": True,
        "spikedetect": False,  # CASCADE handles spike inference in Stage 4
        "batch_size": 500,
    }


def run_suite2p(
    tiff_dir: Path,
    output_dir: Path,
    ops_overrides: dict[str, Any] | None = None,
    fps: float = 29.97,
) -> Path:
    """Run Suite2p on a directory of TIFF stacks.

    Args:
        tiff_dir: Directory containing raw TIFF imaging files (*_XYT.tif).
        output_dir: Directory where Suite2p output (plane0/) will be written.
            Suite2p creates its own subdirectories under this path.
        ops_overrides: Optional dict of Suite2p ops to override defaults.
        fps: Imaging frame rate (Hz). Used to set ``ops["fs"]``.

    Returns:
        Path to the suite2p output directory containing plane0/.

    Raises:
        ImportError: If suite2p is not installed.
        FileNotFoundError: If ``tiff_dir`` does not exist or has no TIFFs.
        RuntimeError: If Suite2p fails during processing.
    """
    if not tiff_dir.exists():
        raise FileNotFoundError(f"TIFF directory not found: {tiff_dir}")

    tiff_files = sorted(tiff_dir.glob("*.tif")) + sorted(tiff_dir.glob("*.tiff"))
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in {tiff_dir}")

    try:
        import suite2p
    except ImportError as exc:
        raise ImportError(
            "suite2p is not installed. "
            "Install via: pip install suite2p (GPU recommended)\n"
            "See: https://suite2p.readthedocs.io/"
        ) from exc

    log.info("Found %d TIFF file(s) in %s", len(tiff_files), tiff_dir)

    # Build ops
    ops = default_ops(fps=fps)
    if ops_overrides:
        ops.update(ops_overrides)

    # Suite2p data paths
    ops["data_path"] = [str(tiff_dir)]
    ops["save_path0"] = str(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Running Suite2p (fs=%.2f Hz, %d TIFFs)...", fps, len(tiff_files))
    suite2p.run_s2p(ops=ops)

    # Suite2p writes output to save_path0/suite2p/plane0/
    suite2p_dir = output_dir / "suite2p"
    plane0 = suite2p_dir / "plane0"

    if not plane0.exists():
        raise RuntimeError(
            f"Suite2p completed but plane0 directory not found at {plane0}. "
            "Check Suite2p logs for errors."
        )

    # Verify required output files
    for name in ("F.npy", "Fneu.npy", "iscell.npy", "stat.npy", "ops.npy"):
        if not (plane0 / name).exists():
            raise RuntimeError(f"Suite2p output file missing: {plane0 / name}")

    log.info("Suite2p complete. Output: %s", suite2p_dir)
    return suite2p_dir
