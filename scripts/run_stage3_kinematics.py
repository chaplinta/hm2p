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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from s3_utils import s3_upload_with_verify

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


def parse_meta_txt(meta_path: Path) -> tuple[float, np.ndarray, tuple[float, float], float]:
    """Parse meta.txt for mm_per_pix, maze corners, camera centre, and maze rotation.

    Returns:
        (mm_per_pix, maze_corners_px, camera_center_px, maze_rotation_deg) where
        maze_corners_px is (4, 2) array, camera_center_px is (cx, cy)
        in cropped-frame coordinates, and maze_rotation_deg is the angle
        of the maze relative to the image axes (from corner coordinates).
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

    # Camera optical centre in cropped-frame coordinates
    # Full sensor: 1280×1024 (Basler acA1300-200um)
    crop_x = int(config["crop"]["x"])
    crop_y = int(config["crop"]["y"])
    cx = 1280.0 / 2.0 - crop_x
    cy = 1024.0 / 2.0 - crop_y

    # Maze rotation from corner geometry
    from hm2p.kinematics.perspective import compute_maze_rotation
    maze_rotation_deg = compute_maze_rotation(corners)

    return mm_per_pix, corners, (cx, cy), maze_rotation_deg


def find_dlc_h5(s3, bucket: str, prefix: str) -> str | None:
    """Find the best finetuned DLC .h5 file under a given S3 prefix.

    Delegates to :func:`hm2p.pose.select.select_best_dlc_h5_s3`, which
    checks for a ``promoted.json`` manifest first and falls back to the
    highest-snapshot heuristic.
    """
    from hm2p.pose.select import select_best_dlc_h5_s3
    return select_best_dlc_h5_s3(s3, bucket, prefix)


def _extract_dlc_provenance(dlc_filename: str) -> tuple[str, str]:
    """Extract model name and snapshot number from a DLC output filename.

    Delegates to :func:`hm2p.pose.select.extract_dlc_provenance`.
    """
    from hm2p.pose.select import extract_dlc_provenance
    return extract_dlc_provenance(dlc_filename)


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
    force: bool = False,
    champion_manifest: dict | None = None,
) -> str:
    """Run Stage 3 for a single session. Returns status string."""
    print(f"\n--- {sub}/{ses} ({exp_id}) ---")

    # Check if kinematics.h5 already exists on S3
    kin_key = f"kinematics/{sub}/{ses}/kinematics.h5"
    if not force:
        try:
            s3.head_object(Bucket=DERIVATIVES_BUCKET, Key=kin_key)
            print(f"  SKIP: kinematics.h5 already exists (use --force to re-run)")
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

    # Extract DLC model provenance from the output filename, then resolve
    # the project-wide champion id by matching the triplet against the
    # current champion manifest. "unknown" means this h5 was not produced
    # by the current champion (or no manifest exists yet) — the frontend
    # treats that as stale.
    from hm2p.pose.select import (
        extract_architecture,
        extract_dlc_provenance,
        resolve_champion_id,
    )
    dlc_filename = Path(dlc_key).name
    dlc_model_name, dlc_snapshot = extract_dlc_provenance(dlc_filename)
    dlc_architecture = extract_architecture(dlc_filename)
    dlc_champion_id = resolve_champion_id(
        dlc_model_name, dlc_architecture, dlc_snapshot, champion_manifest,
    )

    if dry_run:
        print(f"  DRY RUN: would process and upload kinematics.h5")
        print(f"    DLC file: {dlc_key}")
        print(f"    Orientation: {orientation} deg")
        print(f"    Tracker: {tracker}")
        print(f"    DLC model: {dlc_model_name}, snapshot: {dlc_snapshot}")
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
        mm_per_pix, maze_corners_px, camera_center_px, maze_rotation = parse_meta_txt(meta_local)
        print(f"  Scale: {mm_per_pix:.4f} mm/px")
        print(f"  Maze corners (px): {maze_corners_px.tolist()}")
        print(f"  Camera centre (cropped px): ({camera_center_px[0]:.1f}, {camera_center_px[1]:.1f})")
        print(f"  Maze rotation from corners: {maze_rotation:.2f}°")

        # Total orientation = CSV orientation + maze rotation from corners.
        # CSV orientation handles 90°/180° camera placement differences.
        # Maze rotation handles the small residual tilt (<1°) of the maze
        # in the camera frame, computed from the ROI corner coordinates.
        total_orientation = orientation + maze_rotation
        if abs(maze_rotation) > 0.01:
            print(f"  Total orientation: {orientation}° + {maze_rotation:.2f}° = {total_orientation:.2f}°")

        # Run kinematics pipeline (with perspective correction)
        print(f"  Running kinematics pipeline...")
        from hm2p.kinematics.compute import run

        output_path = session_dir / "kinematics.h5"
        session_id = f"{sub}/{ses}"
        print(f"  DLC model: {dlc_model_name}, snapshot: {dlc_snapshot}")
        print(f"  DLC champion id: {dlc_champion_id}")
        run(
            pose_path=dlc_local,
            timestamps_h5=ts_local,
            session_id=session_id,
            tracker=tracker,
            orientation_deg=total_orientation,
            scale_mm_per_px=mm_per_pix,
            maze_corners_px=maze_corners_px,
            bad_behav_intervals=bad_intervals,
            output_path=output_path,
            camera_center_px=camera_center_px,
            camera_height_mm=700.0,
            dlc_model_name=dlc_model_name,
            dlc_snapshot=dlc_snapshot,
            dlc_champion_id=dlc_champion_id,
        )

        # Report stats from kinematics.h5
        import h5py

        with h5py.File(output_path, "r") as f:
            n_frames = len(f["hd_deg"]) if "hd_deg" in f else "?"
            print(f"  Frames: {n_frames}")
            if "speed_cm_s" in f:
                speed = f["speed_cm_s"][:]
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

        # Upload to S3 with verify — raises RuntimeError on failure, ensuring non-zero exit
        print(f"  Uploading to s3://{DERIVATIVES_BUCKET}/{kin_key}")
        s3_upload_with_verify(s3, output_path, DERIVATIVES_BUCKET, kin_key)

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
        type=str,
        default=None,
        help="Process only this session (exp_id string or 0-based index)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without processing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if kinematics.h5 already exists on S3",
    )
    args = parser.parse_args()

    sessions = get_sessions()
    print(f"Found {len(sessions)} sessions")

    s3 = boto3.client("s3", region_name=REGION)
    work_dir = Path(tempfile.mkdtemp(prefix="hm2p-stage3-"))
    print(f"Work dir: {work_dir}")

    # Load champion manifest once. Per-session resolution then matches each
    # session's chosen DLC h5 against this manifest to decide what to stamp.
    from hm2p.pose.select import get_champion_manifest as _get_manifest
    champion_manifest = _get_manifest(s3, DERIVATIVES_BUCKET)
    if champion_manifest is None:
        print("No champion manifest found at s3://hm2p-derivatives/dlc-champion.json. "
              "All sessions will be stamped with dlc_champion_id='unknown'.")
    else:
        print(f"Champion: {champion_manifest.get('champion_id', '?')}")

    if args.session is not None:
        if args.session.isdigit():
            sessions = [sessions[int(args.session)]]
        else:
            sessions = [s for s in sessions if s["exp_id"] == args.session]
            if not sessions:
                print(f"Session {args.session} not found")
                sys.exit(1)

    results = {}
    try:
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
                force=args.force,
                champion_manifest=champion_manifest,
            )
            results[ses["exp_id"]] = status
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

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

    if err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
