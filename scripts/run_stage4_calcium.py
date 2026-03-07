#!/usr/bin/env python3
"""Run Stage 4 (calcium processing) for all sessions.

Downloads Suite2p output + timestamps.h5 from S3, runs neuropil subtraction,
dF/F0 computation, V&H event detection, and uploads ca.h5 to S3 derivatives.

Usage:
    python scripts/run_stage4_calcium.py              # all sessions
    python scripts/run_stage4_calcium.py --session 0   # first session only
    python scripts/run_stage4_calcium.py --dry-run     # show what would be done
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

import boto3

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"


def get_sessions() -> list[dict]:
    """Read session list from metadata/experiments.csv."""
    csv_path = Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"
    sessions = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_id = row["exp_id"]
            parts = exp_id.split("_")
            animal = parts[-1]
            sub = f"sub-{animal}"
            ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
            sessions.append({"exp_id": exp_id, "sub": sub, "ses": ses})
    return sessions


def download_s3_dir(s3, bucket: str, prefix: str, local_dir: Path) -> list[str]:
    """Download all files under an S3 prefix to a local directory."""
    paginator = s3.get_paginator("list_objects_v2")
    downloaded = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel:
                continue
            local_path = local_dir / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_path))
            downloaded.append(rel)
    return downloaded


def run_session(s3, sub: str, ses: str, exp_id: str, work_dir: Path, dry_run: bool = False) -> str:
    """Run Stage 4 for a single session. Returns status string."""
    print(f"\n--- {sub}/{ses} ({exp_id}) ---")

    # Check for Suite2p output on S3
    s2p_prefix = f"ca_extraction/{sub}/{ses}/suite2p/plane0/"
    resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=s2p_prefix, MaxKeys=1)
    if resp.get("KeyCount", 0) == 0:
        print(f"  SKIP: no Suite2p output at {s2p_prefix}")
        return "skip_no_suite2p"

    # Check for timestamps.h5
    ts_key = f"movement/{sub}/{ses}/timestamps.h5"
    try:
        s3.head_object(Bucket=DERIVATIVES_BUCKET, Key=ts_key)
    except Exception:
        print(f"  SKIP: no timestamps.h5 at {ts_key}")
        return "skip_no_timestamps"

    if dry_run:
        print(f"  DRY RUN: would process and upload ca.h5")
        return "dry_run"

    session_dir = work_dir / sub / ses
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download Suite2p output
        s2p_local = session_dir / "suite2p"
        print(f"  Downloading Suite2p output...")
        files = download_s3_dir(s3, DERIVATIVES_BUCKET,
                                f"ca_extraction/{sub}/{ses}/suite2p/",
                                s2p_local)
        print(f"  Downloaded {len(files)} files")

        # Download timestamps.h5
        ts_local = session_dir / "timestamps.h5"
        print(f"  Downloading timestamps.h5...")
        s3.download_file(DERIVATIVES_BUCKET, ts_key, str(ts_local))

        # Run calcium pipeline
        print(f"  Running calcium pipeline...")
        from hm2p.calcium.run import run

        output_path = session_dir / "ca.h5"
        session_id = f"{sub}/{ses}"
        run(
            suite2p_dir=s2p_local,
            timestamps_h5=ts_local,
            session_id=session_id,
            output_path=output_path,
        )

        # Read ca.h5 to report stats
        import h5py
        with h5py.File(output_path, "r") as f:
            dff = f["dff"]
            n_rois, n_frames = dff.shape
            fps = f.attrs.get("fps_imaging", "?")
            print(f"  ROIs (cells): {n_rois}")
            print(f"  Frames: {n_frames}")
            print(f"  FPS: {fps}")
            if "event_masks" in f:
                masks = f["event_masks"][:]
                n_events = int(masks.sum())
                events_per_roi = masks.sum(axis=1)
                print(f"  Total event frames: {n_events}")
                print(f"  Mean event frames/ROI: {events_per_roi.mean():.0f}")

        # Upload to S3
        ca_key = f"calcium/{sub}/{ses}/ca.h5"
        print(f"  Uploading to s3://{DERIVATIVES_BUCKET}/{ca_key}")
        s3.upload_file(str(output_path), DERIVATIVES_BUCKET, ca_key)
        print(f"  DONE")

        return "ok"

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return f"error: {e}"

    finally:
        shutil.rmtree(session_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Run Stage 4 calcium processing")
    parser.add_argument("--session", type=int, default=None,
                        help="Process only this session index (0-based)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without processing")
    args = parser.parse_args()

    sessions = get_sessions()
    print(f"Found {len(sessions)} sessions")

    s3 = boto3.client("s3", region_name=REGION)
    work_dir = Path(tempfile.mkdtemp(prefix="hm2p-stage4-"))
    print(f"Work dir: {work_dir}")

    if args.session is not None:
        sessions = [sessions[args.session]]

    results = {}
    for i, ses in enumerate(sessions):
        status = run_session(s3, ses["sub"], ses["ses"], ses["exp_id"],
                             work_dir, dry_run=args.dry_run)
        results[ses["exp_id"]] = status

    # Summary
    print(f"\n{'='*60}")
    print(f"Stage 4 Summary:")
    ok = sum(1 for v in results.values() if v == "ok")
    skip = sum(1 for v in results.values() if v.startswith("skip"))
    err = sum(1 for v in results.values() if v.startswith("error"))
    dry = sum(1 for v in results.values() if v == "dry_run")
    print(f"  OK: {ok}, Skipped: {skip}, Errors: {err}, Dry run: {dry}")

    if err > 0:
        print(f"\nFailed sessions:")
        for exp_id, status in results.items():
            if status.startswith("error"):
                print(f"  {exp_id}: {status}")

    # Cleanup
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
