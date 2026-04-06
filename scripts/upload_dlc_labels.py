#!/usr/bin/env python3
"""Upload labeled DLC frames to S3 for cloud retraining.

Finds the local DLC project at sourcedata/trackers/dlc/hm2p-retrain-*/,
validates that labeling is done, and uploads labeled-data + config.yaml
to s3://hm2p-derivatives/dlc-retrain/.

Usage:
    uv run python scripts/upload_dlc_labels.py           # upload
    uv run python scripts/upload_dlc_labels.py --dry-run  # validate only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


BUCKET = "hm2p-derivatives"
S3_PREFIX = "dlc-retrain"


def find_dlc_project() -> Path | None:
    """Find the most recent local DLC retrain project."""
    base = Path("sourcedata/trackers/dlc")
    projects = sorted(base.glob("hm2p-retrain-*"))
    return projects[-1] if projects else None


def validate_labels(project_dir: Path) -> dict:
    """Check labeled-data dirs for CollectedData files."""
    labeled = project_dir / "labeled-data"
    if not labeled.exists():
        return {}

    manifest = {}
    for session_dir in sorted(labeled.iterdir()):
        if not session_dir.is_dir():
            continue
        csv_files = list(session_dir.glob("CollectedData_*.csv"))
        h5_files = list(session_dir.glob("CollectedData_*.h5"))
        png_files = list(session_dir.glob("*.png"))
        if csv_files or h5_files:
            manifest[session_dir.name] = {
                "n_frames": len(png_files),
                "has_csv": len(csv_files) > 0,
                "has_h5": len(h5_files) > 0,
            }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload DLC labels to S3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_dir = find_dlc_project()
    if not project_dir:
        print("ERROR: no DLC project found at sourcedata/trackers/dlc/hm2p-retrain-*/")
        print("Run scripts/prepare_retrain_frames.py first to create one.")
        sys.exit(1)

    print(f"Project: {project_dir}")

    config = project_dir / "config.yaml"
    if not config.exists():
        print(f"ERROR: no config.yaml in {project_dir}")
        sys.exit(1)

    manifest = validate_labels(project_dir)
    if not manifest:
        print("ERROR: no labeled data found. Label frames first:")
        print(f"  uv run python -c \"import deeplabcut; deeplabcut.label_frames('{config}')\"")
        sys.exit(1)

    total_frames = sum(v["n_frames"] for v in manifest.values())
    print(f"Sessions with labels: {len(manifest)}")
    print(f"Total labeled frames: {total_frames}")
    for name, info in manifest.items():
        print(f"  {name}: {info['n_frames']} frames")

    if args.dry_run:
        print("\nDRY RUN — would upload:")
        print(f"  {config} → s3://{BUCKET}/{S3_PREFIX}/config.yaml")
        print(f"  {project_dir}/labeled-data/ → s3://{BUCKET}/{S3_PREFIX}/labeled-data/")
        return

    # Upload config.yaml
    print(f"\nUploading config.yaml...")
    subprocess.run(
        ["aws", "s3", "cp", str(config), f"s3://{BUCKET}/{S3_PREFIX}/config.yaml"],
        check=True,
    )

    # Upload labeled-data
    print("Uploading labeled-data/...")
    subprocess.run(
        ["aws", "s3", "sync",
         str(project_dir / "labeled-data"),
         f"s3://{BUCKET}/{S3_PREFIX}/labeled-data/",
         "--exclude", "*.DS_Store"],
        check=True,
    )

    # Upload manifest
    manifest_path = project_dir / "_retrain_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    subprocess.run(
        ["aws", "s3", "cp", str(manifest_path),
         f"s3://{BUCKET}/{S3_PREFIX}/_retrain_manifest.json"],
        check=True,
    )

    print(f"\nUploaded to s3://{BUCKET}/{S3_PREFIX}/")
    print("Next: uv run python scripts/launch_dlc_finetune_ec2.py")


if __name__ == "__main__":
    main()
