#!/usr/bin/env python3
"""Upload hm2p raw data to S3 (hm2p-rawdata bucket).

Usage:
    python scripts/upload_to_s3.py [--dry-run] [--profile PROFILE] [--session EXP_ID]

What gets uploaded per session
--------------------------------
  funcimg/  ← raw session dir (01 lights-maze/{exp_id}/)
              SKIP: *_side_left.camera.mp4, *_XYT.red.tif
  behav/    ← processed video dir (hm2p/video/{exp_id}/)
              SKIP: *-undistort.mp4, *_side_left.camera.mp4

Global uploads
--------------
  sourcedata/metadata/   ← metadata/*.csv
  sourcedata/calibration/ ← sourcedata/calibration/*.npz

S3 layout (NeuroBlueprint)
--------------------------
  s3://hm2p-rawdata/
    rawdata/
      sub-{animal_id}/
        ses-{YYYYMMDD}T{HHMMSS}/
          funcimg/   ← tiff + tdms + meta.txt
          behav/     ← cropped mp4 + meta.txt + meta/
    sourcedata/
      metadata/      ← animals.csv, experiments.csv
      calibration/   ← *.npz
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
RAW_SESSIONS_ROOT = Path(
    "/Users/tristan/Library/CloudStorage/Dropbox/Neuro/Margrie"
    "/shared/lab-108/experiments/01 lights-maze"
)
VIDEO_ROOT = Path(
    "/Users/tristan/Library/CloudStorage/Dropbox/Neuro/Margrie/hm2p/video"
)
METADATA_DIR = REPO_ROOT / "metadata"
CALIB_DIR = REPO_ROOT / "sourcedata" / "calibration"

BUCKET = "hm2p-rawdata"
REGION = "ap-southeast-2"
DEFAULT_PROFILE = "hm2p-agent"

# Sync exclusions
FUNCIMG_EXCLUDES = ["*_side_left.camera.mp4", "*_XYT.red.tif"]
BEHAV_EXCLUDES = ["*-undistort.mp4", "*_side_left.camera.mp4"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def exp_id_to_nb(exp_id: str) -> tuple[str, str]:
    """Parse exp_id → (animal_id, NeuroBlueprint session name).

    '20210823_16_59_50_1114353' → ('1114353', 'ses-20210823T165950')
    """
    parts = exp_id.split("_")
    date = parts[0]
    time_str = "".join(parts[1:4])
    animal_id = parts[4]
    return animal_id, f"ses-{date}T{time_str}"


def s3_prefix(exp_id: str) -> str:
    """Full S3 prefix for this session's rawdata folder."""
    animal_id, ses_name = exp_id_to_nb(exp_id)
    return f"rawdata/sub-{animal_id}/{ses_name}"


def run_sync(
    src: Path,
    dst_s3: str,
    excludes: list[str],
    profile: str,
    dry_run: bool,
) -> int:
    """Run `aws s3 sync src dst_s3 --exclude ... [--dryrun]`.

    Returns the exit code.
    """
    cmd = [
        "aws", "s3", "sync",
        str(src),
        dst_s3,
        "--profile", profile,
        "--region", REGION,
    ]
    for pat in excludes:
        cmd += ["--exclude", pat]
    if dry_run:
        cmd.append("--dryrun")

    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def run_cp(
    src: Path,
    dst_s3: str,
    profile: str,
    dry_run: bool,
) -> int:
    """Run `aws s3 cp src dst_s3 [--dryrun]` for a single file."""
    cmd = [
        "aws", "s3", "cp",
        str(src),
        dst_s3,
        "--profile", profile,
        "--region", REGION,
    ]
    if dry_run:
        cmd.append("--dryrun")

    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


# ---------------------------------------------------------------------------
# Per-session upload
# ---------------------------------------------------------------------------
def raw_session_dir(exp_id: str) -> Path:
    """Resolve the raw session directory.

    Structure: 01 lights-maze/YYYY_MM_DD/{exp_id}/
    exp_id '20210823_16_59_50_1114353' → date_dir '2021_08_23'
    """
    date = exp_id[:8]  # '20210823'
    date_dir = f"{date[:4]}_{date[4:6]}_{date[6:]}"  # '2021_08_23'
    return RAW_SESSIONS_ROOT / date_dir / exp_id


def upload_session(exp_id: str, profile: str, dry_run: bool) -> list[int]:
    prefix = s3_prefix(exp_id)
    codes: list[int] = []

    # --- funcimg (raw session dir) ---
    raw_dir = raw_session_dir(exp_id)
    if raw_dir.exists():
        print(f"\n[funcimg] {raw_dir.name}")
        codes.append(run_sync(
            raw_dir,
            f"s3://{BUCKET}/{prefix}/funcimg/",
            FUNCIMG_EXCLUDES,
            profile,
            dry_run,
        ))
    else:
        print(f"\n[warn] raw dir not found: {raw_dir}")

    # --- behav (processed video dir) ---
    vid_dir = VIDEO_ROOT / exp_id
    if vid_dir.exists():
        print(f"\n[behav] {vid_dir.name}")
        codes.append(run_sync(
            vid_dir,
            f"s3://{BUCKET}/{prefix}/behav/",
            BEHAV_EXCLUDES,
            profile,
            dry_run,
        ))
    else:
        print(f"\n[warn] video dir not found: {vid_dir}")

    return codes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload hm2p raw data to s3://hm2p-rawdata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be uploaded without transferring anything.",
    )
    parser.add_argument(
        "--profile", default=DEFAULT_PROFILE,
        help="AWS CLI profile (default: h2mp-agent).",
    )
    parser.add_argument(
        "--session",
        metavar="EXP_ID",
        help="Upload a single session by exp_id (default: all sessions).",
    )
    args = parser.parse_args()

    experiments = pd.read_csv(METADATA_DIR / "experiments.csv")
    exp_ids: list[str] = experiments["exp_id"].tolist()
    if args.session:
        if args.session not in exp_ids:
            print(f"Error: session '{args.session}' not found in experiments.csv", file=sys.stderr)
            sys.exit(1)
        exp_ids = [args.session]

    all_codes: list[int] = []

    # Sessions
    for i, exp_id in enumerate(exp_ids, 1):
        print(f"\n{'='*60}")
        print(f"Session {i}/{len(exp_ids)}: {exp_id}")
        all_codes.extend(upload_session(exp_id, args.profile, args.dry_run))

    # Metadata CSVs
    print(f"\n{'='*60}\nMetadata CSVs")
    for f in sorted(METADATA_DIR.glob("*.csv")):
        all_codes.append(run_cp(
            f,
            f"s3://{BUCKET}/sourcedata/metadata/{f.name}",
            args.profile,
            args.dry_run,
        ))

    # Calibration .npz files
    if CALIB_DIR.exists():
        print(f"\n{'='*60}\nCalibration files")
        for f in sorted(CALIB_DIR.glob("*.npz")):
            all_codes.append(run_cp(
                f,
                f"s3://{BUCKET}/sourcedata/calibration/{f.name}",
                args.profile,
                args.dry_run,
            ))

    # Summary
    failures = [c for c in all_codes if c != 0]
    print(f"\n{'='*60}")
    if args.dry_run:
        print("Dry run complete — no files were transferred.")
    elif failures:
        print(f"Done with {len(failures)} error(s). Check output above.")
        sys.exit(1)
    else:
        print(f"Upload complete. {len(all_codes)} sync operations succeeded.")


if __name__ == "__main__":
    main()
