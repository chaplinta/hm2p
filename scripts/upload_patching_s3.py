#!/usr/bin/env python3
"""Upload patching electrophysiology + morphology data to S3.

Uploads three categories of data to s3://hm2p-derivatives/patching/:

1. **Raw ephys** — WaveSurfer H5 recordings from /data/patching/ephys/
   → s3://hm2p-derivatives/patching/ephys/{date}/{sweep_id}/...
2. **Processed** — ephys_data.mat + morph_data.mat per cell
   → s3://hm2p-derivatives/patching/processed/{cell_id}/...
3. **Analysis results** — metrics.csv, summary_stats.csv, mannwhitney.csv, PCA
   → s3://hm2p-derivatives/patching/analysis/...
4. **Metadata** — cells.csv, animals.csv (+ dated versions)
   → s3://hm2p-derivatives/patching/metadata/...

Usage:
    python scripts/upload_patching_s3.py                 # upload all
    python scripts/upload_patching_s3.py --dry-run       # show what would upload
    python scripts/upload_patching_s3.py --category ephys  # only ephys
    python scripts/upload_patching_s3.py --category processed  # only processed (morph+ephys .mat)
    python scripts/upload_patching_s3.py --category analysis   # only analysis results
    python scripts/upload_patching_s3.py --category metadata   # only metadata
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BUCKET = "hm2p-derivatives"
S3_PREFIX = "patching"

# Source directories (read-only bind mounts)
PATCHING_ROOT = Path("/data/patching")
EPHYS_DIR = PATCHING_ROOT / "ephys"
PROCESSED_DIR = PATCHING_ROOT / "processed"
METADATA_DIR = PATCHING_ROOT / "metadata"

# Analysis results (local)
ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "results" / "patching" / "analysis"


def _get_client():
    """Return a boto3 S3 client."""
    import boto3

    profile = os.environ.get("AWS_PROFILE", "hm2p-agent")
    session = boto3.Session(profile_name=profile)
    return session.client("s3")


def collect_files(category: str) -> list[tuple[Path, str]]:
    """Collect (local_path, s3_key) pairs for the given category."""
    pairs: list[tuple[Path, str]] = []

    if category in ("all", "ephys"):
        if EPHYS_DIR.exists():
            for f in sorted(EPHYS_DIR.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(PATCHING_ROOT)
                    pairs.append((f, f"{S3_PREFIX}/{rel.as_posix()}"))
        else:
            logger.warning("Ephys dir not found: %s", EPHYS_DIR)

    if category in ("all", "processed"):
        if PROCESSED_DIR.exists():
            for f in sorted(PROCESSED_DIR.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(PATCHING_ROOT)
                    pairs.append((f, f"{S3_PREFIX}/{rel.as_posix()}"))
        else:
            logger.warning("Processed dir not found: %s", PROCESSED_DIR)

    if category in ("all", "metadata"):
        if METADATA_DIR.exists():
            for f in sorted(METADATA_DIR.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(PATCHING_ROOT)
                    pairs.append((f, f"{S3_PREFIX}/{rel.as_posix()}"))
        else:
            logger.warning("Metadata dir not found: %s", METADATA_DIR)

    if category in ("all", "analysis"):
        if ANALYSIS_DIR.exists():
            for f in sorted(ANALYSIS_DIR.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(ANALYSIS_DIR)
                    pairs.append((f, f"{S3_PREFIX}/analysis/{rel.as_posix()}"))
        else:
            logger.warning("Analysis dir not found: %s", ANALYSIS_DIR)

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload patching data to S3")
    parser.add_argument(
        "--category",
        choices=["all", "ephys", "processed", "analysis", "metadata"],
        default="all",
        help="Which data to upload (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show files without uploading")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    pairs = collect_files(args.category)
    if not pairs:
        logger.info("No files to upload for category '%s'.", args.category)
        return

    total_bytes = sum(f.stat().st_size for f, _ in pairs)
    total_mb = total_bytes / (1024 * 1024)

    logger.info(
        "Found %d files (%.1f MB) for category '%s'",
        len(pairs),
        total_mb,
        args.category,
    )

    if args.dry_run:
        for local, key in pairs:
            size_kb = local.stat().st_size / 1024
            print(f"  {size_kb:8.1f} KB  s3://{BUCKET}/{key}")
        print(f"\nTotal: {len(pairs)} files, {total_mb:.1f} MB")
        return

    if not args.no_confirm:
        cost_est = total_bytes * 0.023 / (1024**3)  # S3 storage $/GB/month
        print(f"\nWill upload {len(pairs)} files ({total_mb:.1f} MB) to s3://{BUCKET}/{S3_PREFIX}/")
        print(f"Estimated storage cost: ${cost_est:.4f}/month")
        resp = input("Continue? [y/N] ")
        if resp.lower() not in ("y", "yes"):
            print("Aborted.")
            return

    client = _get_client()
    uploaded = 0
    for local, key in pairs:
        try:
            client.upload_file(str(local), BUCKET, key)
            uploaded += 1
            if uploaded % 50 == 0:
                logger.info("Uploaded %d / %d files...", uploaded, len(pairs))
        except Exception as e:
            logger.error("Failed to upload %s: %s", local, e)

    logger.info("Done. Uploaded %d / %d files to s3://%s/%s/", uploaded, len(pairs), BUCKET, S3_PREFIX)


if __name__ == "__main__":
    main()
