#!/usr/bin/env python3
"""Promote fine-tuned DLC results: copy pose-finetuned/ → pose/ on S3.

Only run after reviewing tracking quality in the frontend. This replaces
the SuperAnimal results with the fine-tuned model's output. Downstream
stages (kinematics, sync, analysis) must be re-run after promotion.

Uses boto3 directly — does not require awscli.

Usage:
    uv run python scripts/promote_finetuned_pose.py              # all sessions
    uv run python scripts/promote_finetuned_pose.py --dry-run     # preview
    uv run python scripts/promote_finetuned_pose.py --session sub-1114353/ses-20210823T165950
    uv run python scripts/promote_finetuned_pose.py --skip-failed  # promote completed only
"""

from __future__ import annotations

import argparse
import json
import sys

import boto3

REGION = "ap-southeast-2"
BUCKET = "hm2p-derivatives"


def _promote_session(s3, ses_key: str, files: list[str], dry_run: bool) -> int:
    """Copy all files for one session from pose-finetuned/ to pose/.

    Parameters
    ----------
    s3 : boto3 S3 client
    ses_key : str
        Session key of the form ``sub-XXX/ses-YYY``.
    files : list[str]
        Full S3 object keys under ``pose-finetuned/``.
    dry_run : bool
        If True, log actions without copying anything.

    Returns
    -------
    int
        Number of files copied (0 on dry run).
    """
    copied = 0
    for src_key in files:
        dst_key = src_key.replace("pose-finetuned/", "pose/", 1)
        if dry_run:
            print(f"    [dry-run] copy {src_key} → {dst_key}")
        else:
            s3.copy_object(
                Bucket=BUCKET,
                CopySource={"Bucket": BUCKET, "Key": src_key},
                Key=dst_key,
            )
            copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote fine-tuned pose results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview actions without copying files")
    parser.add_argument("--session",
                        help="Promote a single session (sub-XXX/ses-YYY)")
    parser.add_argument("--skip-failed", action="store_true",
                        help="Promote sessions that are present in pose-finetuned/ even "
                             "if some sessions from the training run failed. "
                             "By default all sessions must be present.")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)

    # List fine-tuned sessions via boto3 paginator (no awscli dependency)
    paginator = s3.get_paginator("list_objects_v2")
    finetuned: dict[str, list[str]] = {}
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

    print(f"Sessions available for promotion: {len(finetuned)}")

    if not (args.skip_failed or args.session):
        # Sanity check: warn if the set of available sessions looks partial
        # (i.e. the progress JSON records failures but we're not using --skip-failed)
        try:
            obj = s3.get_object(
                Bucket=BUCKET, Key="dlc-retrain/_retrain_progress.json"
            )
            progress = json.loads(obj["Body"].read())
            failed_sessions = progress.get("failed_sessions", [])
            if failed_sessions:
                print(
                    f"WARNING: _retrain_progress.json records {len(failed_sessions)} "
                    f"failed session(s): {failed_sessions}"
                )
                print(
                    "Pass --skip-failed to promote the successful sessions anyway, "
                    "or resolve the failures and re-run inference first."
                )
                sys.exit(1)
        except s3.exceptions.NoSuchKey:
            pass
        except Exception:
            pass

    promoted = []
    for ses_key, files in sorted(finetuned.items()):
        src_pfx = f"s3://{BUCKET}/pose-finetuned/{ses_key}/"
        dst_pfx = f"s3://{BUCKET}/pose/{ses_key}/"
        print(f"  {src_pfx} → {dst_pfx} ({len(files)} files)")
        _promote_session(s3, ses_key, files, dry_run=args.dry_run)
        promoted.append(ses_key)

    if args.dry_run:
        print("\nDRY RUN — no files copied.")
        return

    # Update rerun marker to trigger downstream re-processing
    marker = {
        "rerunning": ["pose"],
        "reason": "Fine-tuned DLC model promoted — downstream stages need re-run",
        "promoted_sessions": promoted,
    }
    s3.put_object(
        Bucket=BUCKET,
        Key="pipeline_rerun.json",
        Body=json.dumps(marker, indent=2),
        ContentType="application/json",
    )
    print(f"\nPromoted {len(promoted)} sessions.")
    print("Downstream stages (kinematics, sync, analysis) need re-running.")
    print("Run: python scripts/run_stage3_kinematics.py --force")


if __name__ == "__main__":
    main()
