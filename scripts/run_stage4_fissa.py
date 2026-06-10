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
import sys
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


def validate_movie_alignment(
    movie: np.ndarray,
    cropped_masks: list[np.ndarray],
    stored_F: np.ndarray,
) -> dict:
    """Check that the regenerated movie aligns with the existing ROIs.

    The FISSA reprocessing path depends on the re-registered movie reproducing
    the pixel grid of the Stage 1 registration that produced ``stat.npy``. If
    registration is deterministic, re-extracting each ROI's raw fluorescence
    (mean over its mask pixels) from the regenerated movie reproduces the stored
    Suite2p ``F.npy`` trace up to a near-perfect rank correlation. A low
    correlation means the movie and the masks no longer share a pixel grid, so
    the resulting FISSA traces would be meaningless — the caller must abort.

    The comparison uses Spearman rank correlation per ROI. Suite2p's ``F.npy`` is
    a ``lam``-weighted mean over the ROI footprint whereas this re-extraction is
    an unweighted mean over the boolean mask, so even a perfectly aligned movie
    does not give correlation exactly 1.0; aligned ROIs sit close to 1.0 and
    misaligned ROIs collapse toward 0.

    Parameters
    ----------
    movie : np.ndarray
        ``(n_frames, crop_ly, crop_lx)`` regenerated registered movie.
    cropped_masks : list of np.ndarray
        Per-ROI ``(crop_ly, crop_lx)`` boolean masks aligned to ``movie`` (output
        of :func:`crop_masks_to_window`), in the same order as ``stored_F`` rows.
    stored_F : np.ndarray
        ``(n_rois, n_frames)`` Suite2p ``F.npy`` from the existing Stage 1 output.

    Returns
    -------
    dict
        ``median_spearman`` and ``min_spearman`` over ROIs with at least one mask
        pixel and non-constant traces, ``n_rois``, ``n_evaluated`` (ROIs that
        yielded a finite correlation), and ``per_roi`` (the finite correlations).

    Raises
    ------
    ValueError
        If the ROI counts or frame counts of ``cropped_masks``, ``movie`` and
        ``stored_F`` do not match.
    """
    from scipy.stats import spearmanr

    n_rois = len(cropped_masks)
    if stored_F.shape[0] != n_rois:
        raise ValueError(
            f"stored_F has {stored_F.shape[0]} ROIs but {n_rois} masks given"
        )
    if movie.shape[0] != stored_F.shape[1]:
        raise ValueError(
            f"movie has {movie.shape[0]} frames but stored_F has {stored_F.shape[1]}"
        )

    correlations: list[float] = []
    for i, mask in enumerate(cropped_masks):
        if not mask.any():
            continue
        # Mean over mask pixels per frame -> (n_frames,) re-extracted trace.
        reextracted = movie[:, mask].mean(axis=1).astype(np.float64)
        stored = np.asarray(stored_F[i], dtype=np.float64)
        if reextracted.std() == 0 or stored.std() == 0:
            continue
        rho, _ = spearmanr(reextracted, stored)
        if np.isfinite(rho):
            correlations.append(float(rho))

    if not correlations:
        return {
            "median_spearman": float("nan"),
            "min_spearman": float("nan"),
            "n_rois": n_rois,
            "n_evaluated": 0,
            "per_roi": [],
        }

    arr = np.asarray(correlations)
    return {
        "median_spearman": float(np.median(arr)),
        "min_spearman": float(arr.min()),
        "n_rois": n_rois,
        "n_evaluated": int(arr.size),
        "per_roi": correlations,
    }


def run_session_fissa(  # pragma: no cover - EC2 I/O + subprocess orchestration
    sub: str,
    ses: str,
    work_dir: Path,
    fissa_python: str,
    *,
    validate_only: bool = False,
    alignment_threshold: float = 0.9,
) -> dict:
    """Reprocess one session's ca.h5 with FISSA neuropil subtraction.

    Orchestrates the full per-session flow on an EC2 instance: download inputs,
    regenerate the registered movie, validate ROI/movie alignment, run FISSA in
    the isolated environment, and re-run the Stage 4 pipeline with the FISSA
    corrected traces handed back via ``precomputed_F_corr``.

    Parameters
    ----------
    sub : str
        Subject label, e.g. ``sub-1117646``.
    ses : str
        Session label, e.g. ``ses-20220804T135202``.
    work_dir : Path
        Scratch directory for downloads and intermediates.
    fissa_python : str
        Path to the isolated FISSA-environment Python interpreter (scikit-learn
        < 1.2) used to run :mod:`fissa_bridge`.
    validate_only : bool
        If True, stop after the alignment check and return the report without
        running FISSA, the pipeline, or any upload. Used for the single-session
        validation gate before the full batch.
    alignment_threshold : float
        Minimum acceptable ``median_spearman`` from
        :func:`validate_movie_alignment`. Below this the session is rejected and
        no ca.h5 is written.

    Returns
    -------
    dict
        ``{"sub", "ses", "status", "alignment": <report>}`` where ``status`` is
        one of ``"validated"`` (validate_only), ``"done"``, or ``"rejected"``.
    """
    import subprocess

    from hm2p.calcium.run import run

    sess_dir = work_dir / sub / ses
    tiff_dir = sess_dir / "funcimg"
    existing_plane0 = sess_dir / "existing" / "suite2p" / "plane0"
    reg_out = sess_dir / "reg"
    ts_path = sess_dir / "timestamps.h5"
    for d in (tiff_dir, existing_plane0.parent.parent, reg_out):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Download inputs: TIFFs, existing Suite2p output, timestamps.
    _s3_sync(
        f"s3://{RAWDATA_BUCKET}/rawdata/{sub}/{ses}/funcimg/",
        tiff_dir,
        include=("*.tif", "*.tiff"),
    )
    _s3_sync(
        f"s3://{DERIVATIVES_BUCKET}/ca_extraction/{sub}/{ses}/suite2p/",
        existing_plane0.parent.parent,
    )
    _s3_cp(
        f"s3://{DERIVATIVES_BUCKET}/timestamps/{sub}/{ses}/timestamps.h5", ts_path
    )

    # 2. Regenerate the registered binary, matching the original fps/tau.
    existing_ops = np.load(existing_plane0 / "ops.npy", allow_pickle=True).item()
    fps = float(existing_ops.get("fs", 9.6))
    tau = float(existing_ops.get("tau", 1.0))
    reg_plane0 = regenerate_registered_binary(tiff_dir, reg_out, fps=fps, tau=tau)

    # 3. Geometry gate: the regenerated crop window must match the original.
    reg_ops = np.load(reg_plane0 / "ops.npy", allow_pickle=True).item()
    for key in ("yrange", "xrange"):
        if list(existing_ops.get(key, [])) != list(reg_ops.get(key, [])):
            return {
                "sub": sub, "ses": ses, "status": "rejected",
                "alignment": {"reason": f"{key} mismatch "
                              f"{existing_ops.get(key)} != {reg_ops.get(key)}"},
            }

    # 4-5. Build crop-aligned masks and load the regenerated movie.
    masks = build_cropped_masks_for_session(existing_plane0, reg_plane0)
    movie = load_registered_movie_for_session(reg_plane0)

    # 6. Alignment validation against the stored F.npy.
    stored_F = np.load(existing_plane0 / "F.npy")
    report = validate_movie_alignment(movie, masks, stored_F)
    log.info("alignment median_spearman=%.4f (min=%.4f, n=%d/%d)",
             report["median_spearman"], report["min_spearman"],
             report["n_evaluated"], report["n_rois"])

    if not (report["median_spearman"] >= alignment_threshold):
        return {"sub": sub, "ses": ses, "status": "rejected", "alignment": report}
    if validate_only:
        return {"sub": sub, "ses": ses, "status": "validated", "alignment": report}

    # 7. Hand the movie + masks to FISSA in the isolated environment.
    movie_npy = sess_dir / "movie.npy"
    masks_npz = sess_dir / "masks.npz"
    fcorr_npy = sess_dir / "F_corr.npy"
    np.save(movie_npy, np.asarray(movie))
    np.savez(masks_npz, *[m.astype(bool) for m in masks])
    bridge = Path(__file__).resolve().parent / "fissa_bridge.py"
    subprocess.run(
        [fissa_python, str(bridge), "--movie", str(movie_npy),
         "--masks", str(masks_npz), "--out", str(fcorr_npy)],
        check=True,
    )
    F_corr = np.load(fcorr_npy)

    # 8. Re-run Stage 4 with the FISSA-corrected traces.
    ca_h5 = sess_dir / "ca.h5"
    run(
        suite2p_dir=existing_plane0.parent,
        timestamps_h5=ts_path,
        session_id=f"{sub}/{ses}",
        output_path=ca_h5,
        neuropil_method="fissa",
        precomputed_F_corr=F_corr,
    )

    # 9. Upload the regenerated ca.h5.
    _s3_cp(ca_h5, f"s3://{DERIVATIVES_BUCKET}/calcium/{sub}/{ses}/ca.h5")
    return {"sub": sub, "ses": ses, "status": "done", "alignment": report}


def _s3_sync(  # pragma: no cover - thin aws cli wrapper
    src: str, dst: Path, include: tuple[str, ...] = ()
) -> None:
    import subprocess

    cmd = ["aws", "s3", "sync", src, str(dst)]
    if include:
        cmd += ["--exclude", "*"]
        for pat in include:
            cmd += ["--include", pat]
    subprocess.run(cmd, check=True)


def _s3_cp(src, dst) -> None:  # pragma: no cover - thin aws cli wrapper
    import subprocess

    subprocess.run(["aws", "s3", "cp", str(src), str(dst)], check=True)


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description="Stage 4 FISSA reprocessing")
    parser.add_argument("--session", nargs=2, metavar=("SUB", "SES"),
                        help="Process a single session: sub-XXXX ses-YYYYMMDDTHHMMSS")
    parser.add_argument("--all-fixed", action="store_true",
                        help="Process all sessions currently on fixed-0.7")
    parser.add_argument("--fissa-python", default="/opt/fissa/bin/python",
                        help="Interpreter for the isolated FISSA env (sklearn<1.2)")
    parser.add_argument("--work-dir", default="/tmp/hm2p-fissa")
    parser.add_argument("--validate-only", action="store_true",
                        help="Stop after the alignment check; write no ca.h5")
    parser.add_argument("--alignment-threshold", type=float, default=0.9)
    args = parser.parse_args()

    if not args.session:
        raise SystemExit(
            "Provide --session SUB SES to process one session on the instance. "
            "--all-fixed batch iteration is driven by the launcher user-data."
        )

    sub, ses = args.session
    result = run_session_fissa(
        sub, ses,
        work_dir=Path(args.work_dir),
        fissa_python=args.fissa_python,
        validate_only=args.validate_only,
        alignment_threshold=args.alignment_threshold,
    )
    import json

    print(json.dumps({k: v for k, v in result.items() if k != "alignment"
                      } | {"alignment": {k: v for k, v in result["alignment"].items()
                                         if k != "per_roi"}}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    main()
