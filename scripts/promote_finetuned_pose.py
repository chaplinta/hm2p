#!/usr/bin/env python3
"""Promote fine-tuned DLC results: copy pose-finetuned/ → pose/ on S3.

Only run after reviewing tracking quality in the frontend. This replaces
the SuperAnimal results with the fine-tuned model's output. Downstream
stages (kinematics, sync, analysis) must be re-run after promotion.

Usage:
    uv run python scripts/promote_finetuned_pose.py              # all sessions
    uv run python scripts/promote_finetuned_pose.py --dry-run     # preview
    uv run python scripts/promote_finetuned_pose.py --session sub-1114353/ses-20210823T165950
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import boto3

REGION = "ap-southeast-2"
BUCKET = "hm2p-derivatives"


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote fine-tuned pose results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--session", help="Promote a single session (sub-XXX/ses-YYY)")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)

    # List fine-tuned sessions
    paginator = s3.get_paginator("list_objects_v2")
    finetuned = {}
    for page in paginator.paginate(Bucket=BUCKET, Prefix="pose-finetuned/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            parts = key.split("/")
            if len(parts) >= 3:
                ses_key = f"{parts[1]}/{parts[2]}"
                finetuned.setdefault(ses_key, []).append(key)

    if not finetuned:
        print("No fine-tuned results found at pose-finetuned/")
        sys.exit(1)

    if args.session:
        if args.session not in finetuned:
            print(f"Session {args.session} not found in pose-finetuned/")
            sys.exit(1)
        finetuned = {args.session: finetuned[args.session]}

    print(f"Sessions to promote: {len(finetuned)}")

    for ses_key, files in sorted(finetuned.items()):
        src = f"s3://{BUCKET}/pose-finetuned/{ses_key}/"
        dst = f"s3://{BUCKET}/pose/{ses_key}/"
        print(f"  {src} → {dst} ({len(files)} files)")

        if not args.dry_run:
            subprocess.run(
                ["aws", "s3", "sync", src, dst, "--delete"],
                check=True,
                capture_output=True,
            )

    if args.dry_run:
        print("\nDRY RUN — no files copied.")
    else:
        # Update rerun marker to trigger downstream re-processing
        marker = {
            "rerunning": ["pose"],
            "reason": "Fine-tuned DLC model promoted — downstream stages need re-run",
        }
        s3.put_object(
            Bucket=BUCKET,
            Key="pipeline_rerun.json",
            Body=json.dumps(marker, indent=2),
            ContentType="application/json",
        )
        print(f"\nPromoted {len(finetuned)} sessions.")
        print("Downstream stages (kinematics, sync, analysis) need re-running.")
        print("Run: python scripts/run_stage3_kinematics.py --force")


if __name__ == "__main__":
    main()
