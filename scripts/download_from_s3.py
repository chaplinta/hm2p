#!/usr/bin/env python3
"""Download hm2p data from S3 (hm2p-rawdata bucket) to local data/.

Usage:
    python scripts/download_from_s3.py [--dry-run] [--profile PROFILE] [--session EXP_ID]

What gets downloaded per session
---------------------------------
  rawdata/sub-{animal_id}/ses-{YYYYMMDD}T{HHMMSS}/funcimg/
  rawdata/sub-{animal_id}/ses-{YYYYMMDD}T{HHMMSS}/behav/

Global downloads
----------------
  sourcedata/metadata/   ← animals.csv, experiments.csv
  sourcedata/calibration/ ← *.npz
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
METADATA_DIR = REPO_ROOT / "metadata"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"

BUCKET = "hm2p-rawdata"
REGION = "ap-southeast-2"
DEFAULT_PROFILE = "hm2p-agent"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def exp_id_to_nb(exp_id: str) -> tuple[str, str]:
    """Parse exp_id -> (animal_id, NeuroBlueprint session name).

    '20210823_16_59_50_1114353' -> ('1114353', 'ses-20210823T165950')
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


def estimate_s3_size(
    prefix: str,
    profile: str,
) -> tuple[int, int]:
    """Count objects and total bytes under an S3 prefix.

    Uses ``aws s3api list-objects-v2`` for accurate server-side sizing.

    Returns:
        (n_files, total_bytes) tuple.
    """
    n_files = 0
    total_bytes = 0
    continuation_token = None

    while True:
        cmd = [
            "aws", "s3api", "list-objects-v2",
            "--bucket", BUCKET,
            "--prefix", prefix,
            "--profile", profile,
            "--region", REGION,
            "--output", "json",
        ]
        if continuation_token:
            cmd += ["--continuation-token", continuation_token]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[warn] Failed to list {prefix}: {result.stderr.strip()}", file=sys.stderr)
            break

        data = json.loads(result.stdout)
        for obj in data.get("Contents", []):
            n_files += 1
            total_bytes += obj.get("Size", 0)

        if data.get("IsTruncated"):
            continuation_token = data.get("NextContinuationToken")
        else:
            break

    return n_files, total_bytes


def run_sync(
    src_s3: str,
    dst: Path,
    profile: str,
    dry_run: bool,
) -> int:
    """Run ``aws s3 sync src_s3 dst [--dryrun]``.

    Returns the exit code.
    """
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws", "s3", "sync",
        src_s3,
        str(dst),
        "--profile", profile,
        "--region", REGION,
    ]
    if dry_run:
        cmd.append("--dryrun")

    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


# ---------------------------------------------------------------------------
# Per-session download
# ---------------------------------------------------------------------------
def download_session(
    exp_id: str,
    data_root: Path,
    profile: str,
    dry_run: bool,
) -> list[int]:
    """Download rawdata for a single session from S3."""
    prefix = s3_prefix(exp_id)
    animal_id, ses_name = exp_id_to_nb(exp_id)
    local_ses = data_root / "rawdata" / f"sub-{animal_id}" / ses_name
    codes: list[int] = []

    # funcimg
    print(f"\n[funcimg] {exp_id}")
    codes.append(
        run_sync(
            f"s3://{BUCKET}/{prefix}/funcimg/",
            local_ses / "funcimg",
            profile,
            dry_run,
        )
    )

    # behav
    print(f"\n[behav] {exp_id}")
    codes.append(
        run_sync(
            f"s3://{BUCKET}/{prefix}/behav/",
            local_ses / "behav",
            profile,
            dry_run,
        )
    )

    return codes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Download hm2p data from s3://hm2p-rawdata to local data/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be downloaded without transferring anything.",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=f"AWS CLI profile (default: {DEFAULT_PROFILE}).",
    )
    parser.add_argument(
        "--session",
        metavar="EXP_ID",
        help="Download a single session by exp_id (default: all sessions).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Local data root directory (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip cost estimate confirmation prompt.",
    )
    args = parser.parse_args()

    experiments = pd.read_csv(METADATA_DIR / "experiments.csv")
    exp_ids: list[str] = experiments["exp_id"].tolist()
    if args.session:
        if args.session not in exp_ids:
            print(
                f"Error: session '{args.session}' not found in experiments.csv",
                file=sys.stderr,
            )
            sys.exit(1)
        exp_ids = [args.session]

    data_root: Path = args.data_root
    all_codes: list[int] = []

    # --- Cost estimation before any transfers ---
    if not args.dry_run:
        print("Estimating download size from S3...")
        total_files = 0
        total_bytes = 0
        for exp_id in exp_ids:
            prefix = s3_prefix(exp_id)
            n, b = estimate_s3_size(prefix, args.profile)
            total_files += n
            total_bytes += b
        # sourcedata
        for sd_prefix in ("sourcedata/metadata", "sourcedata/calibration"):
            n, b = estimate_s3_size(sd_prefix, args.profile)
            total_files += n
            total_bytes += b

        from hm2p.io.aws_cost import confirm_or_abort, estimate_download

        est = estimate_download(total_files, total_bytes)
        confirm_or_abort(est, yes=args.yes)

    # Sessions
    for i, exp_id in enumerate(exp_ids, 1):
        print(f"\n{'=' * 60}")
        print(f"Session {i}/{len(exp_ids)}: {exp_id}")
        all_codes.extend(
            download_session(exp_id, data_root, args.profile, args.dry_run)
        )

    # Metadata CSVs
    print(f"\n{'=' * 60}\nSourcedata (metadata + calibration)")
    all_codes.append(
        run_sync(
            f"s3://{BUCKET}/sourcedata/metadata/",
            data_root / "sourcedata" / "metadata",
            args.profile,
            args.dry_run,
        )
    )
    all_codes.append(
        run_sync(
            f"s3://{BUCKET}/sourcedata/calibration/",
            data_root / "sourcedata" / "calibration",
            args.profile,
            args.dry_run,
        )
    )

    # Summary
    failures = [c for c in all_codes if c != 0]
    print(f"\n{'=' * 60}")
    if args.dry_run:
        print("Dry run complete — no files were transferred.")
    elif failures:
        print(f"Done with {len(failures)} error(s). Check output above.")
        sys.exit(1)
    else:
        print(f"Download complete. {len(all_codes)} sync operations succeeded.")


if __name__ == "__main__":
    main()
