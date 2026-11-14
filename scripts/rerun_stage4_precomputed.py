#!/usr/bin/env python3
"""Re-run Stage 4 from FISSA traces already stored in ca.h5 (no re-registration).

For sessions whose ca.h5 ``F_corr`` is genuine FISSA output (the 17 sessions
reprocessed through the EC2 FISSA bridge in 2026-06), this re-derives
everything downstream of neuropil correction with the current code:

1. download Suite2p plane0 (stat, F, Fneu, iscell, ops) and timestamps.h5;
2. classify ROIs with the imaging frame rate from timestamps.h5
   (the stored labels were computed at 29.97 Hz);
3. ``hm2p.calcium.run.run(neuropil_method="fissa", precomputed_F_corr=F_corr)``
   — dF/F relative to the raw-trace baseline, events, roi_qc;
4. upload ca.h5 (and roi_class.npy / roi_class_prob.npy to the Suite2p
   prefix). CASCADE ``spikes`` are not in the new file; re-run
   ``scripts/launch_cascade_ec2.py --overwrite`` afterwards.

A session is refused if its stored F_corr is coefficient subtraction
(``F_raw - F_corr`` an exact linear function of ``Fneu``), so such sessions
are not relabelled as FISSA. Back up ``calcium/`` before running (uploads overwrite).

Usage
-----
    python scripts/rerun_stage4_precomputed.py --sessions EXP_ID [EXP_ID ...]
    python scripts/rerun_stage4_precomputed.py --sessions ... --no-upload   # local only
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

log = logging.getLogger("rerun_stage4")
BUCKET = "hm2p-derivatives"
PLANE_FILES = ("stat.npy", "F.npy", "Fneu.npy", "iscell.npy", "ops.npy")
LINEAR_R2_MIN = 0.99  # coefficient subtraction gives R^2 = 1 for every ROI


def parse_exp_id(exp_id: str) -> tuple[str, str]:
    p = exp_id.split("_")
    return f"sub-{p[-1]}", f"ses-{p[0]}T{p[1]}{p[2]}{p[3]}"


def neuropil_linear_r2(F_corr: np.ndarray, F_raw: np.ndarray, Fneu: np.ndarray) -> np.ndarray:
    """Per-ROI R^2 of ``F_raw - F_corr = a * Fneu + b`` (least squares)."""
    out = np.empty(F_corr.shape[0])
    for i in range(F_corr.shape[0]):
        y = (F_raw[i] - F_corr[i]).astype(np.float64)
        A = np.column_stack([Fneu[i].astype(np.float64), np.ones(y.size)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        var = y.var()
        out[i] = 1.0 - (y - A @ coef).var() / var if var > 0 else 1.0
    return out


def looks_like_fissa(F_corr: np.ndarray, F_raw: np.ndarray, Fneu: np.ndarray) -> bool:
    """False if F_corr is coefficient subtraction (exactly linear in Fneu for most ROIs).

    FISSA's separated signal is not a linear function of the neuropil trace;
    ``F_raw - a * Fneu`` gives R^2 = 1 (the 9 original sessions: 1.000).
    """
    return bool(np.median(neuropil_linear_r2(F_corr, F_raw, Fneu)) < LINEAR_R2_MIN)


def process(s3, exp_id: str, work: Path, upload: bool) -> dict:  # pragma: no cover - S3 I/O
    from hm2p.calcium.run import run
    from hm2p.extraction.roi_classify import classify_session
    from hm2p.extraction.run_suite2p import fps_from_timestamps

    sub, ses = parse_exp_id(exp_id)
    plane = work / exp_id / "suite2p" / "plane0"
    plane.mkdir(parents=True, exist_ok=True)
    for f in PLANE_FILES:
        s3.download_file(BUCKET, f"ca_extraction/{sub}/{ses}/suite2p/plane0/{f}", str(plane / f))
    ts = work / exp_id / "timestamps.h5"
    s3.download_file(BUCKET, f"movement/{sub}/{ses}/timestamps.h5", str(ts))
    body = s3.get_object(Bucket=BUCKET, Key=f"calcium/{sub}/{ses}/ca.h5")["Body"].read()
    with h5py.File(io.BytesIO(body), "r") as f:
        F_corr = f["F_corr"][:].astype(np.float32)
        F_raw = f["F_raw"][:]
        Fneu = f["Fneu_raw"][:]
    if not looks_like_fissa(F_corr, F_raw, Fneu):
        raise ValueError(f"{exp_id}: stored F_corr is not FISSA-like; refusing to relabel")
    fps = fps_from_timestamps(ts)
    cls = classify_session(plane, fps=fps)
    out = work / exp_id / "ca.h5"
    run(
        suite2p_dir=plane.parent,
        timestamps_h5=ts,
        session_id=f"{sub}/{ses}",
        output_path=out,
        neuropil_method="fissa",
        precomputed_F_corr=F_corr,
    )
    if upload:
        s3.upload_file(str(out), BUCKET, f"calcium/{sub}/{ses}/ca.h5")
        for f in ("roi_class.npy", "roi_class_prob.npy"):
            s3.upload_file(str(plane / f), BUCKET, f"ca_extraction/{sub}/{ses}/suite2p/plane0/{f}")
    return {
        "exp_id": exp_id,
        "fps": fps,
        "n_soma": cls["n_soma"],
        "n_dend": cls["n_dend"],
        "uploaded": upload,
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - S3 entry point
    import boto3

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sessions", nargs="+", required=True, metavar="EXP_ID")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--profile", default="hm2p-agent")
    ap.add_argument(
        "--work-dir", type=Path, help="keep intermediate files here (default: temp dir)"
    )
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s3 = boto3.Session(profile_name=a.profile).client("s3")
    failed = []
    with tempfile.TemporaryDirectory(dir=a.work_dir) as td:
        for exp_id in a.sessions:
            try:
                r = process(s3, exp_id, a.work_dir or Path(td), upload=not a.no_upload)
                log.info("done %s", r)
            except Exception as exc:  # noqa: BLE001 - report and continue
                log.error("FAILED %s: %s", exp_id, exc)
                failed.append(exp_id)
    log.info("finished: %d ok, %d failed %s", len(a.sessions) - len(failed), len(failed), failed)
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
