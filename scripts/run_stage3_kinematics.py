#!/usr/bin/env python3
"""Run Stage 3 (kinematics) for all sessions.

Downloads DLC pose output, timestamps.h5, and meta.txt from S3,
runs the kinematics pipeline (HD, position, speed, AHV, movement state),
and uploads kinematics.h5 to S3 derivatives.

Usage:
    python scripts/run_stage3_kinematics.py              # all sessions
    python scripts/run_stage3_kinematics.py --session 0   # first session only
    python scripts/run_stage3_kinematics.py --dry-run     # show what would be done
"""

from __future__ import annotations

import argparse
import configparser
import csv
import re
import shutil
import sys
import tempfile
from pathlib import Path

import boto3
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
RAWDATA_BUCKET = "hm2p-rawdata"


def get_sessions() -> list[dict]:
    """Read session list from metadata/experiments.csv.

    Returns list of dicts with keys: exp_id, sub, ses, orientation,
    bad_behav_times, tracker.
    """
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

            orientation = float(row.get("orientation", 0) or 0)
            bad_behav_times = row.get("bad_behav_times", "")
            tracker = row.get("tracker", "dlc")

            sessions.append({
                "exp_id": exp_id,
                "sub": sub,
                "ses": ses,
                "orientation": orientation,
                "bad_behav_times": bad_behav_times,
                "tracker": tracker,
            })
    return sessions


def parse_bad_behav_times(raw: str) -> list[tuple[float, float]]:
    """Parse bad_behav_times string into list of (start_s, end_s) tuples.

    Format: semicolon-separated intervals like "11:10-11:30;13:20-21:00;27:00-end"
    - MM:SS-MM:SS pairs
    - "end" means end of session (mapped to 999999)
    - Empty string or "?" means no bad intervals
    """
    if not raw or raw.strip() in ("", "?"):
        return []

    intervals = []
    for segment in raw.split(";"):
        segment = segment.strip()
        if not segment:
            continue

        match = re.match(
            r"(\d+):(\d+)\s*-\s*(?:(\d+):(\d+)|(end))",
            segment,
        )
        if not match:
            print(f"  WARNING: could not parse bad_behav_times segment: '{segment}'")
            continue

        start_s = int(match.group(1)) * 60 + int(match.group(2))
        if match.group(5) == "end":
            end_s = 999999.0
        else:
            end_s = float(int(match.group(3)) * 60 + int(match.group(4)))

        intervals.append((float(start_s), end_s))

    return intervals


def parse_meta_txt(meta_path: Path) -> tuple[float, np.ndarray]:
    """Parse meta.txt for mm_per_pix and maze corner coordinates.

    Returns:
        (mm_per_pix, maze_corners_px) where maze_corners_px is (4, 2) array.
    """
    config = configparser.ConfigParser()
    config.read(str(meta_path))

    mm_per_pix = float(config["scale"]["mm_per_pix"])

    corners = np.array([
        [float(config["roi"]["x1"]), float(config["roi"]["y1"])],
        [float(config["roi"]["x2"]), float(config["roi"]["y2"])],
        [float(config["roi"]["x3"]), float(config["roi"]["y3"])],
        [float(config["roi"]["x4"]), float(config["roi"]["y4"])],
    ])

    return mm_per_pix, corners


def find_dlc_h5(s3, bucket: str, prefix: str) -> str | None:
    """Find the DLC .h5 file under a given S3 prefix. Returns the key or None."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".h5"):
                return key
    return None


def run_session(
    s3,
    sub: str,
    ses: str,
    exp_id: str,
    orientation: float,
    bad_behav_times: str,
    tracker: str,
    work_dir: Path,
    dry_run: bool = False,
) -> str:
    """Run Stage 3 for a single session. Returns status string."""
    print(f"\n--- {sub}/{ses} ({exp_id}) ---")

    # Check if kinematics.h5 already exists on S3
    kin_key = f"kinematics/{sub}/{ses}/kinematics.h5"
    try:
        s3.head_object(Bucket=DERIVATIVES_BUCKET, Key=kin_key)
        print(f"  SKIP: kinematics.h5 already exists at {kin_key}")
        return "skip_exists"
    except s3.exceptions.ClientError:
        pass  # Does not exist, proceed

    # Check for DLC output on S3
    pose_prefix = f"pose/{sub}/{ses}/"
    dlc_key = find_dlc_h5(s3, DERIVATIVES_BUCKET, pose_prefix)
    if dlc_key is None:
        print(f"  SKIP: no DLC .h5 file at {pose_prefix}")
        return "skip_no_dlc"

    # Check for timestamps.h5
    ts_key = f"movement/{sub}/{ses}/timestamps.h5"
    try:
        s3.head_object(Bucket=DERIVATIVES_BUCKET, Key=ts_key)
    except s3.exceptions.ClientError:
        print(f"  SKIP: no timestamps.h5 at {ts_key}")
        return "skip_no_timestamps"

    # Check for meta.txt
    meta_key = f"rawdata/{sub}/{ses}/behav/meta.txt"
    try:
        s3.head_object(Bucket=RAWDATA_BUCKET, Key=meta_key)
    except s3.exceptions.ClientError:
        print(f"  SKIP: no meta.txt at {meta_key}")
        return "skip_no_meta"

    # Parse bad behaviour intervals
    bad_intervals = parse_bad_behav_times(bad_behav_times)
    if bad_intervals:
        print(f"  Bad behaviour intervals: {bad_intervals}")

    if dry_run:
        print(f"  DRY RUN: would process and upload kinematics.h5")
        print(f"    DLC file: {dlc_key}")
        print(f"    Orientation: {orientation} deg")
        print(f"    Tracker: {tracker}")
        return "dry_run"

    session_dir = work_dir / sub / ses
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download DLC .h5 file
        dlc_filename = Path(dlc_key).name
        dlc_local = session_dir / dlc_filename
        print(f"  Downloading DLC output: {dlc_filename}...")
        s3.download_file(DERIVATIVES_BUCKET, dlc_key, str(dlc_local))

        # Download timestamps.h5
        ts_local = session_dir / "timestamps.h5"
        print(f"  Downloading timestamps.h5...")
        s3.download_file(DERIVATIVES_BUCKET, ts_key, str(ts_local))

        # Download meta.txt
        meta_local = session_dir / "meta.txt"
        print(f"  Downloading meta.txt...")
        s3.download_file(RAWDATA_BUCKET, meta_key, str(meta_local))

        # Parse meta.txt
        mm_per_pix, maze_corners_px = parse_meta_txt(meta_local)
        print(f"  Scale: {mm_per_pix:.4f} mm/px")
        print(f"  Maze corners (px): {maze_corners_px.tolist()}")

        # Run kinematics pipeline
        print(f"  Running kinematics pipeline...")
        from hm2p.kinematics.compute import run

        output_path = session_dir / "kinematics.h5"
        session_id = f"{sub}/{ses}"
        run(
            pose_path=dlc_local,
            timestamps_h5=ts_local,
            session_id=session_id,
            tracker=tracker,
            orientation_deg=orientation,
            scale_mm_per_px=mm_per_pix,
            maze_corners_px=maze_corners_px,
            bad_behav_intervals=bad_intervals,
            output_path=output_path,
        )

        # Report stats from kinematics.h5
        import h5py

        with h5py.File(output_path, "r") as f:
            n_frames = len(f["hd"]) if "hd" in f else "?"
            print(f"  Frames: {n_frames}")
            if "speed" in f:
                speed = f["speed"][:]
                print(f"  Speed: mean={np.nanmean(speed):.2f} cm/s, "
                      f"max={np.nanmax(speed):.2f} cm/s")
            if "active" in f:
                active = f["active"][:]
                pct_active = 100.0 * np.nansum(active) / len(active)
                print(f"  Active: {pct_active:.1f}%")
            if "bad_behav" in f:
                bad = f["bad_behav"][:]
                pct_bad = 100.0 * np.nansum(bad) / len(bad)
                print(f"  Bad behaviour: {pct_bad:.1f}%")

        # Upload to S3
        print(f"  Uploading to s3://{DERIVATIVES_BUCKET}/{kin_key}")
        s3.upload_file(str(output_path), DERIVATIVES_BUCKET, kin_key)
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
    parser = argparse.ArgumentParser(description="Run Stage 3 kinematics processing")
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
    work_dir = Path(tempfile.mkdtemp(prefix="hm2p-stage3-"))
    print(f"Work dir: {work_dir}")

    if args.session is not None:
        sessions = [sessions[args.session]]

    results = {}
    for i, ses in enumerate(sessions):
        status = run_session(
            s3,
            ses["sub"],
            ses["ses"],
            ses["exp_id"],
            ses["orientation"],
            ses["bad_behav_times"],
            ses["tracker"],
            work_dir,
            dry_run=args.dry_run,
        )
        results[ses["exp_id"]] = status

    # Summary
    print(f"\n{'=' * 60}")
    print("Stage 3 Summary:")
    ok = sum(1 for v in results.values() if v == "ok")
    skip = sum(1 for v in results.values() if v.startswith("skip"))
    err = sum(1 for v in results.values() if v.startswith("error"))
    dry = sum(1 for v in results.values() if v == "dry_run")
    print(f"  OK: {ok}, Skipped: {skip}, Errors: {err}, Dry run: {dry}")

    if skip > 0:
        print("\nSkipped sessions:")
        for exp_id, status in results.items():
            if status.startswith("skip"):
                print(f"  {exp_id}: {status}")

    if err > 0:
        print("\nFailed sessions:")
        for exp_id, status in results.items():
            if status.startswith("error"):
                print(f"  {exp_id}: {status}")

    # Cleanup
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
