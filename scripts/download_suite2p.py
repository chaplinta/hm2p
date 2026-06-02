#!/usr/bin/env python3
"""Download Suite2p outputs from S3 for local inspection and reclassification.

Usage:
    # Download all sessions
    python scripts/download_suite2p.py

    # Download a specific session
    python scripts/download_suite2p.py --session 20210823_16_59_50_1114353

    # Download only the lightweight files (stat, iscell, ops — no F/Fneu/spks)
    python scripts/download_suite2p.py --lightweight

    # List available sessions
    python scripts/download_suite2p.py --list
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import boto3

BUCKET = "hm2p-derivatives"
REGION = "ap-southeast-2"
S3_PREFIX = "ca_extraction"
LOCAL_DIR = Path("data/suite2p")

# Files needed for ROI inspection (images + footprints + classification)
LIGHTWEIGHT_FILES = {"stat.npy", "iscell.npy", "ops.npy", "db.npy", "settings.npy"}

# Inspection set: add traces for viewing ROI activity
INSPECTION_FILES = LIGHTWEIGHT_FILES | {"F.npy", "Fneu.npy"}

# All plane0 files (includes deconvolved spikes, registration outputs)
ALL_FILES = INSPECTION_FILES | {
    "spks.npy", "detect_outputs.npy", "reg_outputs.npy", "zcorr.npy",
}


def parse_session_id(exp_id: str) -> tuple[str, str]:
    parts = exp_id.split("_")
    animal = parts[-1]
    sub = f"sub-{animal}"
    ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
    return sub, ses


def list_sessions(s3) -> list[dict]:
    """List all sessions with Suite2p outputs on S3."""
    metadata_path = Path("metadata/experiments.csv")
    with open(metadata_path) as f:
        experiments = list(csv.DictReader(f))

    sessions = []
    for exp in experiments:
        exp_id = exp["exp_id"]
        sub, ses = parse_session_id(exp_id)
        prefix = f"{S3_PREFIX}/{sub}/{ses}/suite2p/plane0/stat.npy"
        try:
            s3.head_object(Bucket=BUCKET, Key=prefix)
            sessions.append({
                "exp_id": exp_id,
                "sub": sub,
                "ses": ses,
                "exclude": exp.get("exclude", "0"),
                "primary_exp": exp.get("primary_exp", "0"),
            })
        except Exception:
            pass
    return sessions


def download_session(
    s3, exp_id: str, lightweight: bool = False, local_dir: Path = LOCAL_DIR
) -> Path:
    """Download Suite2p outputs for a session."""
    sub, ses = parse_session_id(exp_id)
    s3_prefix = f"{S3_PREFIX}/{sub}/{ses}/suite2p/"

    # List all objects under this session's suite2p dir
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=s3_prefix)
    if "Contents" not in resp:
        print(f"  No Suite2p outputs found for {exp_id}")
        return local_dir

    session_dir = local_dir / exp_id / "suite2p"
    wanted = LIGHTWEIGHT_FILES if lightweight else ALL_FILES

    total_bytes = 0
    for obj in resp["Contents"]:
        key = obj["Key"]
        filename = key.split("/")[-1]
        # Determine subdirectory (plane0 or root)
        rel = key[len(s3_prefix):]
        local_path = session_dir / rel

        if lightweight and filename not in wanted:
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists() and local_path.stat().st_size == obj["Size"]:
            continue  # already downloaded

        print(f"  {rel} ({obj['Size'] / 1e6:.1f} MB)")
        s3.download_file(BUCKET, key, str(local_path))
        total_bytes += obj["Size"]

    # Also download the root db.npy and settings.npy
    for root_file in ["db.npy", "settings.npy"]:
        root_key = f"{s3_prefix}{root_file}"
        root_local = session_dir / root_file
        try:
            if not root_local.exists():
                root_local.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(BUCKET, root_key, str(root_local))
        except Exception:
            pass

    if total_bytes > 0:
        print(f"  Downloaded {total_bytes / 1e6:.1f} MB")
    else:
        print(f"  Already up to date")

    return session_dir


def main():
    parser = argparse.ArgumentParser(description="Download Suite2p outputs from S3")
    parser.add_argument("--session", type=str, help="Specific session exp_id")
    parser.add_argument("--lightweight", action="store_true",
                        help="Download only stat/iscell/ops (no F/Fneu/spks)")
    parser.add_argument("--list", action="store_true", help="List available sessions")
    parser.add_argument("--output-dir", type=Path, default=LOCAL_DIR)
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)

    if args.list:
        sessions = list_sessions(s3)
        print(f"Sessions with Suite2p outputs: {len(sessions)}")
        for s in sessions:
            flag = "P" if s["primary_exp"] == "1" else " "
            excl = "X" if s["exclude"] == "1" else " "
            print(f"  [{flag}] [{excl}] {s['exp_id']}")
        return

    if args.session:
        exp_ids = [args.session]
    else:
        sessions = list_sessions(s3)
        exp_ids = [s["exp_id"] for s in sessions]

    print(f"Downloading {'lightweight' if args.lightweight else 'full'} "
          f"Suite2p outputs for {len(exp_ids)} sessions to {args.output_dir}/")

    for exp_id in exp_ids:
        print(f"\n{exp_id}:")
        download_session(s3, exp_id, args.lightweight, args.output_dir)

    print(f"\nDone. Load in Python with:")
    print(f"  import numpy as np")
    print(f"  stat = np.load('data/suite2p/<session>/suite2p/plane0/stat.npy', allow_pickle=True)")
    print(f"  iscell = np.load('data/suite2p/<session>/suite2p/plane0/iscell.npy')")
    print(f"  ops = np.load('data/suite2p/<session>/suite2p/plane0/ops.npy', allow_pickle=True).item()")


if __name__ == "__main__":
    main()
