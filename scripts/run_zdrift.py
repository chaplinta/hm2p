#!/usr/bin/env python3
"""Run z-drift estimation for all sessions with z-stacks.

Downloads Suite2p plane0/ (ops.npy + data.bin) and the corresponding z-stack
TIFF from S3, runs compute_zdrift(), and uploads zdrift.h5 to S3 derivatives.

Usage:
    python scripts/run_zdrift.py              # all sessions with z-stacks
    python scripts/run_zdrift.py --session 0  # first matching session only
    python scripts/run_zdrift.py --dry-run    # show what would be done
    python scripts/run_zdrift.py --force      # re-run even if zdrift.h5 exists
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
RAWDATA_BUCKET = "hm2p-rawdata"
DERIVATIVES_BUCKET = "hm2p-derivatives"


def get_sessions_with_zstacks() -> list[dict]:
    """Read sessions that have a zstack_id from experiments.csv."""
    csv_path = (
        Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"
    )
    sessions = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            zstack_id = row.get("zstack_id", "").strip()
            if not zstack_id:
                continue
            exp_id = row["exp_id"]
            parts = exp_id.split("_")
            animal = parts[-1]
            sub = f"sub-{animal}"
            ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
            sessions.append({
                "exp_id": exp_id,
                "sub": sub,
                "ses": ses,
                "zstack_id": zstack_id,
                "exclude": row.get("exclude", "0").strip() == "1",
            })
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


def find_zstack_tif(s3, zstack_id: str) -> str | None:
    """Find the downsampled z-stack TIFF on S3."""
    prefix = f"sourcedata/zstacks/{zstack_id}/"
    resp = s3.list_objects_v2(Bucket=RAWDATA_BUCKET, Prefix=prefix)
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        # Prefer the downsampled zstack-*.tif, fall back to any .tif
        if "zstack-" in key and key.endswith(".tif"):
            return key
    # Fallback: any .tif that isn't a preview
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".tif") and "preview" not in key.lower():
            return key
    return None


def zdrift_exists(s3, sub: str, ses: str) -> bool:
    """Check if zdrift.h5 already exists on S3."""
    key = f"ca_extraction/{sub}/{ses}/zdrift.h5"
    try:
        s3.head_object(Bucket=DERIVATIVES_BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def run_zdrift_session(
    s3, session: dict, work_dir: Path, force: bool = False
) -> bool:
    """Run z-drift for one session. Returns True on success."""
    from hm2p.extraction.zdrift import compute_zdrift, save_zdrift

    sub, ses = session["sub"], session["ses"]
    zstack_id = session["zstack_id"]
    print(f"\n{'='*60}")
    print(f"Session: {sub}/{ses} (zstack: {zstack_id})")

    if not force and zdrift_exists(s3, sub, ses):
        print("  SKIP: zdrift.h5 already exists on S3")
        return True

    # Download Suite2p plane0
    s2p_prefix = f"ca_extraction/{sub}/{ses}/suite2p/plane0/"
    s2p_dir = work_dir / "s2p" / "plane0"
    s2p_dir.mkdir(parents=True, exist_ok=True)

    print("  Downloading Suite2p plane0...")
    # Only need ops.npy and data.bin
    for fname in ["ops.npy", "data.bin"]:
        key = f"{s2p_prefix}{fname}"
        local = s2p_dir / fname
        try:
            s3.download_file(DERIVATIVES_BUCKET, key, str(local))
        except Exception as e:
            print(f"  ERROR: Failed to download {fname}: {e}")
            return False

    # Download z-stack TIFF
    zstack_key = find_zstack_tif(s3, zstack_id)
    if zstack_key is None:
        print(f"  ERROR: No z-stack TIFF found for {zstack_id}")
        return False

    zstack_local = work_dir / "zstack.tif"
    print(f"  Downloading z-stack: {zstack_key}")
    s3.download_file(RAWDATA_BUCKET, zstack_key, str(zstack_local))

    # Run z-drift
    print("  Computing z-drift...")
    try:
        result = compute_zdrift(s2p_dir, zstack_local, batch_size=200)
    except Exception as e:
        print(f"  ERROR: compute_zdrift failed: {e}")
        return False

    n_frames = len(result["zpos"])
    z_range = result["zpos"].max() - result["zpos"].min()
    print(
        f"  Done: {n_frames} frames, {result['n_zplanes']} z-planes, "
        f"z-range={z_range} planes"
    )

    # Save and upload
    out_path = work_dir / "zdrift.h5"
    save_zdrift(result, out_path)

    s3_key = f"ca_extraction/{sub}/{ses}/zdrift.h5"
    print(f"  Uploading to s3://{DERIVATIVES_BUCKET}/{s3_key}")
    s3.upload_file(str(out_path), DERIVATIVES_BUCKET, s3_key)

    return True


def main():
    parser = argparse.ArgumentParser(description="Run z-drift estimation")
    parser.add_argument(
        "--session", type=int, default=None,
        help="Process only this session index (0-based)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without running",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if zdrift.h5 already exists",
    )
    parser.add_argument(
        "--include-excluded", action="store_true",
        help="Also process sessions marked exclude=1",
    )
    args = parser.parse_args()

    sessions = get_sessions_with_zstacks()
    if not args.include_excluded:
        sessions = [s for s in sessions if not s["exclude"]]

    print(f"Sessions with z-stacks: {len(sessions)}")

    if args.session is not None:
        if args.session >= len(sessions):
            print(f"ERROR: --session {args.session} out of range (max {len(sessions)-1})")
            sys.exit(1)
        sessions = [sessions[args.session]]

    if args.dry_run:
        for i, s in enumerate(sessions):
            print(f"  [{i}] {s['sub']}/{s['ses']} ← zstack: {s['zstack_id']}")
        return

    s3 = boto3.client("s3", region_name=REGION)
    completed = 0
    failed = 0

    for i, session in enumerate(sessions):
        print(f"\n[{i+1}/{len(sessions)}]", end="")
        with tempfile.TemporaryDirectory(prefix="hm2p-zdrift-") as tmpdir:
            ok = run_zdrift_session(
                s3, session, Path(tmpdir), force=args.force
            )
            if ok:
                completed += 1
            else:
                failed += 1

    print(f"\n{'='*60}")
    print(f"Z-drift complete: {completed}/{len(sessions)} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
