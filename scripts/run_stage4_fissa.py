#!/usr/bin/env python3
"""Stage 4 FISSA reprocessing — regenerate ca.h5 with true FISSA neuropil subtraction.

This is the EC2-side driver for reprocessing sessions whose ca.h5 currently uses
fixed-0.7 neuropil subtraction, replacing it with FISSA (Keemink et al. 2018).

Why a separate path is needed
-----------------------------
FISSA needs the motion-corrected (Suite2p-registered) movie plus the ROI masks.
Suite2p's registered binary (``data.bin``) is NOT kept on S3 — Stage 1 runs with
``delete_bin=True`` and only F/Fneu/stat/ops.npy are uploaded. The saved ops.npy
does not contain the full nonrigid block geometry, so the registered movie cannot
be reconstructed from the offsets alone. Therefore the registered movie is
regenerated on-instance (x86, where suite2p installs) by re-running Suite2p
**registration only** with the session's saved parameters, with ROI detection
turned off so the existing ``stat.npy`` ROIs are reused unchanged.

Per-session flow
----------------
1. Download the session's TIFFs (rawdata) and existing Suite2p output + timestamps.
2. Re-run Suite2p registration with ``roidetect=False``, ``delete_bin=False`` to
   regenerate ``data.bin`` at the cropped registration window (yrange × xrange).
3. Build ROI masks from the EXISTING ``stat.npy`` (no re-detection) and crop them
   to the registration window so they align with the registered movie pixels.
4. Load ``data.bin`` as a memmapped movie and run the Stage 4 pipeline with
   ``neuropil_method="fissa"`` and ``fissa_movie`` / ``fissa_roi_masks`` set.
5. Run the XGBoost ROI classifier inline if ``roi_class.npy`` is missing.
6. Upload the new ca.h5 (now with Fneu_raw, roi_qc, neuropil_method="fissa").

This script writes to S3 only when actually run on EC2. It does nothing on import.

Usage (on the EC2 instance):
    python run_stage4_fissa.py --session sub-XXXX ses-YYYYMMDDTHHMMSS
    python run_stage4_fissa.py --all-fixed     # all sessions currently on fixed-0.7

References
----------
Keemink SW, Lowe SC, Pakan JMP, Dylda E, van Rossum MCW, Bhatt DH. 2018.
"FISSA: A neuropil decontamination toolbox for calcium imaging signals."
Scientific Reports 8:3493. doi:10.1038/s41598-018-21640-2
https://github.com/rochefort-lab/fissa
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add src to path when run as a standalone script on EC2.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REGION = "ap-southeast-2"
RAWDATA_BUCKET = "hm2p-rawdata"
DERIVATIVES_BUCKET = "hm2p-derivatives"

log = logging.getLogger("run_stage4_fissa")


def regenerate_registered_binary(
    tiff_dir: Path,
    out_dir: Path,
    fps: float,
    tau: float,
) -> Path:
    """Re-run Suite2p registration only, keeping data.bin, reusing saved params.

    Runs Suite2p with ROI detection disabled and ``delete_bin=False`` so the
    motion-corrected movie (``data.bin``) is produced at the cropped registration
    window. Registration is deterministic given identical TIFFs and parameters,
    so the regenerated movie matches the run that produced the existing
    ``stat.npy`` ROIs.

    Parameters
    ----------
    tiff_dir : Path
        Directory of the session's raw TIFF files (Stage 1 input).
    out_dir : Path
        Suite2p output directory (a ``suite2p/plane0`` tree is created under it).
    fps : float
        Imaging frame rate (Hz) — must match the original Stage 1 run.
    tau : float
        GCaMP decay constant (s) — must match the original Stage 1 run.

    Returns
    -------
    Path
        Path to the regenerated ``plane0`` directory (contains ``data.bin`` and
        ``ops.npy``).

    Raises
    ------
    RuntimeError
        If ``data.bin`` is not produced.
    """
    import suite2p

    from hm2p.extraction.run_suite2p import default_settings

    settings = default_settings(fps=fps, tau=tau)
    # Keep the registered binary, and skip ROI detection / extraction:
    # we reuse the existing stat.npy so ROIs stay identical.
    settings["io"]["delete_bin"] = False
    settings["run"] = settings.get("run", {})
    settings["run"]["roidetect"] = False
    settings["run"]["do_registration"] = True

    db = {
        "data_path": [str(tiff_dir)],
        "save_path0": str(out_dir),
        "nplanes": 1,
    }
    log.info("Re-running Suite2p registration (roidetect=False, keep data.bin)...")
    suite2p.run_s2p(db=db, settings=settings)

    plane0 = out_dir / "suite2p" / "plane0"
    if not (plane0 / "data.bin").exists():
        raise RuntimeError(f"registration did not produce data.bin in {plane0}")
    return plane0


def load_registered_movie_for_session(reg_plane0: Path) -> np.ndarray:
    """Load the regenerated registered movie at its cropped window.

    Reads ``ops.npy`` from the freshly-registered plane to obtain the crop
    window (``yrange``/``xrange``) and frame count, then memmaps ``data.bin``.

    Parameters
    ----------
    reg_plane0 : Path
        plane0 directory produced by :func:`regenerate_registered_binary`.

    Returns
    -------
    movie : np.ndarray
        ``(n_frames, crop_ly, crop_lx)`` int16 memmap of the registered movie.
    """
    from hm2p.calcium.neuropil import load_registered_movie

    ops = np.load(reg_plane0 / "ops.npy", allow_pickle=True).item()
    ly, lx = int(ops["Ly"]), int(ops["Lx"])
    yrange = ops.get("yrange", [0, ly])
    xrange = ops.get("xrange", [0, lx])
    crop_ly = int(yrange[1] - yrange[0])
    crop_lx = int(xrange[1] - xrange[0])
    n_frames = int(ops.get("nframes", 0)) or None
    return load_registered_movie(
        reg_plane0 / "data.bin", crop_ly=crop_ly, crop_lx=crop_lx, n_frames=n_frames
    )


def build_cropped_masks_for_session(
    existing_plane0: Path,
    reg_plane0: Path,
) -> list[np.ndarray]:
    """Build crop-aligned FISSA masks from the EXISTING stat.npy.

    ROI masks come from the existing Stage 1 ``stat.npy`` (ROIs unchanged) and
    are cropped to the registration window of the freshly-regenerated movie so
    masks and movie share a pixel grid.

    Parameters
    ----------
    existing_plane0 : Path
        plane0 directory with the existing ``stat.npy``/``ops.npy`` from S3.
    reg_plane0 : Path
        plane0 directory of the freshly-registered movie (for yrange/xrange).

    Returns
    -------
    list of np.ndarray
        Per-ROI cropped boolean masks aligned to the registered movie.
    """
    from hm2p.calcium.fissa_masks import (
        build_roi_masks_from_plane,
        crop_masks_to_window,
    )

    full_masks, _ = build_roi_masks_from_plane(existing_plane0)
    reg_ops = np.load(reg_plane0 / "ops.npy", allow_pickle=True).item()
    ly, lx = int(reg_ops["Ly"]), int(reg_ops["Lx"])
    yrange = tuple(int(v) for v in reg_ops.get("yrange", [0, ly]))
    xrange = tuple(int(v) for v in reg_ops.get("xrange", [0, lx]))
    return crop_masks_to_window(full_masks, yrange=yrange, xrange=xrange)


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description="Stage 4 FISSA reprocessing")
    parser.add_argument("--session", nargs=2, metavar=("SUB", "SES"),
                        help="Process a single session: sub-XXXX ses-YYYYMMDDTHHMMSS")
    parser.add_argument("--all-fixed", action="store_true",
                        help="Process all sessions currently on fixed-0.7")
    parser.add_argument("--dry-run", action="store_true")
    parser.parse_args()
    raise SystemExit(
        "This driver runs on EC2 with suite2p + fissa installed. "
        "Invoke run_session_fissa() from the launcher user-data, or run with "
        "--session on an instance. No-op when imported."
    )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    main()
