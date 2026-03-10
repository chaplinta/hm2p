#!/usr/bin/env python3
"""Run Stage 5 (sync) for all sessions.

Downloads kinematics.h5 + ca.h5 from S3, resamples kinematics to imaging rate,
merges with calcium data, and uploads sync.h5 to S3 derivatives.

Usage:
    python scripts/run_stage5_sync.py              # all sessions
    python scripts/run_stage5_sync.py --session 0   # first session only
    python scripts/run_stage5_sync.py --dry-run     # show what would be done
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
            # Skip excluded sessions
            if str(row.get("exclude", "0")).strip() == "1":
                continue
            sessions.append({"exp_id": exp_id, "sub": sub, "ses": ses})
    return sessions


def s3_key_exists(s3, bucket: str, key: str) -> bool:
    """Check whether an S3 key exists."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def run_session(
    s3, sub: str, ses: str, exp_id: str, work_dir: Path, dry_run: bool = False
) -> str:
    """Run Stage 5 for a single session. Returns status string."""
    print(f"\n--- {sub}/{ses} ({exp_id}) ---")

    kin_key = f"kinematics/{sub}/{ses}/kinematics.h5"
    ca_key = f"calcium/{sub}/{ses}/ca.h5"
    sync_key = f"sync/{sub}/{ses}/sync.h5"

    # Check if sync.h5 already exists
    if s3_key_exists(s3, DERIVATIVES_BUCKET, sync_key):
        print(f"  SKIP: sync.h5 already exists at {sync_key}")
        return "skip_exists"

    # Check for kinematics.h5
    if not s3_key_exists(s3, DERIVATIVES_BUCKET, kin_key):
        print(f"  SKIP: no kinematics.h5 at {kin_key}")
        return "skip_no_kinematics"

    # Check for ca.h5
    if not s3_key_exists(s3, DERIVATIVES_BUCKET, ca_key):
        print(f"  SKIP: no ca.h5 at {ca_key}")
        return "skip_no_ca"

    if dry_run:
        print(f"  DRY RUN: would process and upload sync.h5")
        return "dry_run"

    session_dir = work_dir / sub / ses
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download kinematics.h5
        kin_local = session_dir / "kinematics.h5"
        print(f"  Downloading kinematics.h5...")
        s3.download_file(DERIVATIVES_BUCKET, kin_key, str(kin_local))

        # Download ca.h5
        ca_local = session_dir / "ca.h5"
        print(f"  Downloading ca.h5...")
        s3.download_file(DERIVATIVES_BUCKET, ca_key, str(ca_local))

        # Report input stats
        import h5py

        with h5py.File(kin_local, "r") as f:
            kin_frames = f["frame_times"].shape[0]
            print(f"  Kinematics frames: {kin_frames}")

        with h5py.File(ca_local, "r") as f:
            ca_frame_times = f["frame_times"].shape[0]
            dff_shape = f["dff"].shape
            n_rois = dff_shape[0]
            n_imaging_frames = dff_shape[1]
            print(f"  Ca frame_times: {ca_frame_times}, dff shape: {dff_shape}")

        # Warn about frame_times / dff mismatch
        if ca_frame_times != n_imaging_frames:
            print(
                f"  WARNING: ca.h5 frame_times ({ca_frame_times}) != "
                f"dff columns ({n_imaging_frames}) — off by "
                f"{ca_frame_times - n_imaging_frames}"
            )

        # Run sync pipeline
        print(f"  Running sync pipeline...")
        from hm2p.sync.align import run

        output_path = session_dir / "sync.h5"
        session_id = f"{sub}/{ses}"
        run(
            kinematics_h5=kin_local,
            ca_h5=ca_local,
            session_id=session_id,
            output_path=output_path,
        )

        # Report output stats
        with h5py.File(output_path, "r") as f:
            keys = list(f.keys())
            ft_len = f["frame_times"].shape[0]
            dff_out = f["dff"].shape
            print(f"  sync.h5 keys: {keys}")
            print(f"  ROIs: {dff_out[0]}, imaging frames: {dff_out[1]}")
            print(f"  frame_times length: {ft_len}")

            # Check kinematics-to-imaging match
            for k in ("hd_deg", "x_mm", "y_mm", "speed_cm_s"):
                if k in f:
                    print(f"  {k} length: {f[k].shape[0]} (resampled)")

        # Upload to S3
        print(f"  Uploading to s3://{DERIVATIVES_BUCKET}/{sync_key}")
        s3.upload_file(str(output_path), DERIVATIVES_BUCKET, sync_key)
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
    parser = argparse.ArgumentParser(description="Run Stage 5 sync")
    parser.add_argument(
        "--session",
        type=int,
        default=None,
        help="Process only this session index (0-based)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without processing",
    )
    args = parser.parse_args()

    sessions = get_sessions()
    print(f"Found {len(sessions)} sessions")

    s3 = boto3.client("s3", region_name=REGION)
    work_dir = Path(tempfile.mkdtemp(prefix="hm2p-stage5-"))
    print(f"Work dir: {work_dir}")

    if args.session is not None:
        sessions = [sessions[args.session]]

    results = {}
    for i, ses in enumerate(sessions):
        status = run_session(
            s3, ses["sub"], ses["ses"], ses["exp_id"], work_dir, dry_run=args.dry_run
        )
        results[ses["exp_id"]] = status

    # Summary
    print(f"\n{'='*60}")
    print(f"Stage 5 Summary:")
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
