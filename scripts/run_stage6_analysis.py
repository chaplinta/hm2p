#!/usr/bin/env python3
"""Stage 6 — Run analysis for all sessions and signal types, save results.

Downloads ca.h5, kinematics.h5, and timestamps.h5 from S3, runs the full
analysis pipeline for each available signal type (dff, deconv, events),
and uploads analysis.h5 to S3.

Usage:
    python scripts/run_stage6_analysis.py [--session EXP_ID] [--n-shuffles 500]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hm2p.analysis.run import AnalysisParams, analyze_cell
from hm2p.analysis.save import save_analysis_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage6")

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
METADATA_DIR = Path(__file__).resolve().parent.parent / "metadata"


def parse_session_id(exp_id: str) -> tuple[str, str]:
    """Convert exp_id to (sub, ses) NeuroBlueprint names."""
    parts = exp_id.split("_")
    animal = parts[-1]
    sub = f"sub-{animal}"
    ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
    return sub, ses


def download_h5(s3, bucket: str, key: str, local_path: Path) -> bool:
    """Download an HDF5 file from S3. Returns True on success."""
    try:
        s3.download_file(bucket, key, str(local_path))
        return True
    except Exception as e:
        log.warning("Could not download s3://%s/%s: %s", bucket, key, e)
        return False


def load_h5(path: Path) -> dict:
    """Load all datasets and attrs from an HDF5 file."""
    import h5py
    with h5py.File(path, "r") as f:
        data = {}
        for k in f.keys():
            data[k] = f[k][:]
        for k, v in f.attrs.items():
            data[k] = v
        return data


def run_analysis_all_signals(
    sync: dict,
    params: AnalysisParams,
) -> tuple[dict, int, int, float, list[str]]:
    """Run analysis for all available signal types using sync.h5 data.

    Parameters
    ----------
    sync : dict
        Data loaded from sync.h5 (already aligned to imaging rate).
    params : AnalysisParams
        Analysis parameters.

    Returns:
        (results_by_signal, n_rois, n_frames, fps, signal_types_available)
    """
    dff = sync["dff"]
    n_rois, n_frames_ca = dff.shape
    fps = float(sync.get("fps_imaging", 9.8))
    deconv = sync.get("spks")
    event_masks = sync.get("event_masks")
    if event_masks is not None:
        event_masks = event_masks.astype(bool)

    # Determine available signal types
    signal_types_available = ["dff"]
    if deconv is not None:
        signal_types_available.append("deconv")
    if event_masks is not None:
        signal_types_available.append("events")

    # All kinematics are already resampled to imaging rate in sync.h5
    hd_deg = sync["hd_deg"]
    x_mm = sync["x_mm"]
    y_mm = sync["y_mm"]
    speed = sync["speed_cm_s"]
    light_on = sync["light_on"].astype(bool)
    bad_behav = sync["bad_behav"].astype(bool)
    active_mask = ~bad_behav

    x_cm = x_mm / 10.0
    y_cm = y_mm / 10.0

    # Truncate to common length
    n = min(n_frames_ca, len(hd_deg))
    dff = dff[:, :n]
    if deconv is not None:
        deconv = deconv[:, :n]
    if event_masks is not None:
        event_masks = event_masks[:, :n]
    hd_deg = hd_deg[:n]
    x_cm = x_cm[:n]
    y_cm = y_cm[:n]
    speed = speed[:n]
    light_on = light_on[:n]
    active_mask = active_mask[:n]

    results_by_signal: dict[str, list] = {}

    for signal_type in signal_types_available:
        log.info("  Running analysis with signal_type=%s", signal_type)
        p = AnalysisParams(
            signal_type=signal_type,
            speed_threshold=params.speed_threshold,
            hd_n_bins=params.hd_n_bins,
            hd_smoothing_sigma_deg=params.hd_smoothing_sigma_deg,
            place_bin_size=params.place_bin_size,
            place_smoothing_sigma=params.place_smoothing_sigma,
            place_min_occupancy_s=params.place_min_occupancy_s,
            n_shuffles=params.n_shuffles,
            alpha=params.alpha,
        )
        cell_results = []
        for i in range(n_rois):
            r = analyze_cell(
                roi_idx=i,
                dff=dff,
                deconv=deconv,
                event_masks=event_masks,
                hd_deg=hd_deg,
                x_cm=x_cm,
                y_cm=y_cm,
                speed=speed,
                light_on=light_on,
                active_mask=active_mask,
                fps=fps,
                params=p,
                seed=42,
            )
            cell_results.append(r)
        results_by_signal[signal_type] = cell_results
        n_sig = sum(
            1 for r in cell_results
            if r.hd_all and r.hd_all.get("significant", False)
        )
        log.info(
            "  %s: %d/%d significantly HD-tuned",
            signal_type, n_sig, n_rois,
        )

    return results_by_signal, n_rois, n, fps, signal_types_available


def main():
    parser = argparse.ArgumentParser(description="Stage 6: Analysis pipeline")
    parser.add_argument("--session", help="Single session exp_id to process")
    parser.add_argument("--n-shuffles", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)

    # Load experiment list
    import csv
    with open(METADATA_DIR / "experiments.csv") as f:
        experiments = list(csv.DictReader(f))

    if args.session:
        experiments = [e for e in experiments if e["exp_id"] == args.session]
        if not experiments:
            log.error("Session %s not found in experiments.csv", args.session)
            sys.exit(1)

    params = AnalysisParams(n_shuffles=args.n_shuffles)

    completed = []
    failed = []
    skipped = []

    for i, exp in enumerate(experiments, 1):
        exp_id = exp["exp_id"]
        sub, ses = parse_session_id(exp_id)
        log.info("=== [%d/%d] %s/%s (%s) ===", i, len(experiments), sub, ses, exp_id)

        # Check if already processed
        try:
            resp = s3.list_objects_v2(
                Bucket=DERIVATIVES_BUCKET,
                Prefix=f"analysis/{sub}/{ses}/analysis.h5",
                MaxKeys=1,
            )
            if resp.get("KeyCount", 0) > 0 and not args.session:
                log.info("  SKIP: already processed")
                skipped.append(exp_id)
                continue
        except Exception:
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Download sync.h5 (contains aligned neural + behavioural data)
            sync_path = tmp / "sync.h5"

            ok = download_h5(s3, DERIVATIVES_BUCKET, f"sync/{sub}/{ses}/sync.h5", sync_path)
            if not ok:
                log.warning("  SKIP: no sync.h5 (Stage 5 not done)")
                skipped.append(exp_id)
                continue

            try:
                sync = load_h5(sync_path)

                results_by_signal, n_rois, n_frames, fps, avail = run_analysis_all_signals(
                    sync, params,
                )

                if args.dry_run:
                    log.info("  DRY RUN: would save analysis.h5")
                    completed.append(exp_id)
                    continue

                # Save locally
                out_path = tmp / "analysis.h5"
                save_analysis_results(
                    out_path,
                    results_by_signal,
                    params,
                    session_id=exp_id,
                    n_rois=n_rois,
                    n_frames=n_frames,
                    fps=fps,
                    signal_types_available=avail,
                )

                # Upload to S3
                s3_key = f"analysis/{sub}/{ses}/analysis.h5"
                s3.upload_file(str(out_path), DERIVATIVES_BUCKET, s3_key)
                log.info("  Uploaded to s3://%s/%s", DERIVATIVES_BUCKET, s3_key)
                completed.append(exp_id)

            except Exception as e:
                log.exception("  ERROR: %s", e)
                failed.append(exp_id)

    # Update progress
    progress = {
        "total": len(experiments),
        "completed": len(completed),
        "failed": len(failed),
        "skipped": len(skipped),
        "completed_sessions": completed,
        "failed_sessions": failed,
        "status": "ALL DONE" if not failed else f"{len(completed)} done, {len(failed)} failed",
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    progress_json = json.dumps(progress, indent=2)
    s3.put_object(
        Bucket=DERIVATIVES_BUCKET,
        Key="analysis/_progress.json",
        Body=progress_json.encode(),
    )

    log.info("\n=== SUMMARY ===")
    log.info("Completed: %d/%d", len(completed), len(experiments))
    log.info("Failed: %d", len(failed))
    log.info("Skipped: %d", len(skipped))


if __name__ == "__main__":
    main()
