#!/usr/bin/env python3
"""Declare a new project-wide DLC champion model.

Writes ``s3://hm2p-derivatives/dlc-champion.json`` with the fields defined
in ``docs/dlc-champion-model.md``. Archives the previous champion manifest
to ``dlc-champion-history/{old_champion_id}.json`` and clears the
``pipeline_rerun.json`` marker.

This script is normally invoked **automatically** at the end of a
successful ``run_dlc_retrain.py`` job. The ``declare_champion()`` function
is the importable entry point used by that hook. The CLI exists for manual
re-declaration (e.g. to add a ``--note`` annotation or re-write the
manifest after a metadata correction).

Usage
-----

Manual invocation::

    uv run python scripts/declare_dlc_champion.py \\
        --model-name hm2p_hrnetw32_shuffle1 \\
        --architecture HrnetW32 \\
        --snapshot 290 \\
        --training-run-id retrain-20260423T142500Z \\
        [--note "manual re-declare; ID unchanged"] \\
        [--dry-run]

Auto invocation (from ``run_dlc_retrain.py``)::

    from declare_dlc_champion import declare_champion
    declare_champion(
        model_name=..., architecture=..., snapshot=...,
        training_run_id=..., notes=...,
    )
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
import sys
import urllib.request
from pathlib import Path

import boto3

# Ensure src/ is on sys.path for hm2p imports when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hm2p.pose.select import (  # noqa: E402
    CHAMPION_MANIFEST_KEY,
    compute_champion_id,
    get_champion_manifest,
)

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
HISTORY_PREFIX = "dlc-champion-history"
PROMOTIONS_LOG_KEY = f"{HISTORY_PREFIX}/promotions.log"
PIPELINE_RERUN_KEY = "pipeline_rerun.json"
TRAINING_S3_PREFIX = "dlc-retrain/models/"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("declare_dlc_champion")


def _imds_instance_id() -> str:
    """Return the EC2 instance ID via the Instance Metadata Service.

    Returns ``"unknown"`` when not running on EC2 or when IMDS is
    unreachable.
    """
    try:
        # Try IMDSv2 first (token-based)
        token_req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        token = urllib.request.urlopen(token_req, timeout=2).read().decode().strip()
        id_req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(id_req, timeout=2).read().decode().strip()
    except Exception:
        try:
            # Fall back to IMDSv1
            resp = urllib.request.urlopen(
                "http://169.254.169.254/latest/meta-data/instance-id", timeout=2
            )
            return resp.read().decode().strip()
        except Exception:
            return "unknown"


def _git_sha(repo_root: Path) -> str:
    """Return the short git SHA of the hm2p repo, or ``"unknown"``."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def declare_champion(
    *,
    model_name: str,
    architecture: str,
    snapshot: str,
    training_run_id: str,
    notes: str = "",
    note: str = "",
    training_s3_prefix: str = TRAINING_S3_PREFIX,
    bucket: str = DERIVATIVES_BUCKET,
    dry_run: bool = False,
    s3_client: object | None = None,
    repo_root: Path | None = None,
) -> dict:
    """Promote a model to project-wide champion. Returns the new manifest dict.

    Steps:

    1. Build the new ``champion_id`` deterministically from the inputs.
    2. Read the current ``dlc-champion.json`` if present and archive it to
       ``dlc-champion-history/{old_champion_id}.json``.
    3. Write the new manifest to ``dlc-champion.json``.
    4. Append a one-line summary to ``dlc-champion-history/promotions.log``.
    5. Delete ``pipeline_rerun.json`` if present (clears in-flight marker).

    Parameters
    ----------
    model_name, architecture, snapshot:
        Identifiers extracted from the trained DLC model. ``architecture``
        must be the canonical capitalised form (e.g. ``"HrnetW32"``).
    training_run_id:
        Identifier of the EC2 retrain run. Stored as-is in the manifest.
    notes:
        Auto-generated description summarising training parameters and
        labeled-frame count. Written by ``run_dlc_retrain.py``.
    note:
        Optional free-text operator annotation (set via ``--note`` on the
        CLI). Empty by default. Distinct from ``notes`` so an automated
        declaration can record technical details and a manual operator
        edit can add a separate human-readable reason.
    training_s3_prefix:
        S3 prefix where model weights live. Stored in the manifest for
        traceability.
    bucket:
        Derivatives bucket name.
    dry_run:
        If True, build the manifest and print it but do not write to S3.
    s3_client:
        Optional injected boto3 S3 client (for tests). One is created if
        not supplied.
    repo_root:
        Path to the hm2p repo clone for the git-SHA lookup. Defaults to
        the parent of this file.
    """
    if s3_client is None:
        s3_client = boto3.client("s3", region_name=REGION)
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    promoted_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    training_date = promoted_at[:10]
    champion_id = compute_champion_id(model_name, architecture, snapshot, training_date)
    instance_id = _imds_instance_id()
    git_sha = _git_sha(repo_root)

    new_manifest = {
        "champion_id": champion_id,
        "model_name": model_name,
        "architecture": architecture,
        "snapshot": str(snapshot),
        "training_date": training_date,
        "training_run_id": training_run_id,
        "promoted_by_ec2_instance": instance_id,
        "promoted_by_git_sha": git_sha,
        "promoted_at": promoted_at,
        "training_s3_prefix": training_s3_prefix,
        "note": note,
        "notes": notes,
    }

    log.info("New champion manifest:")
    log.info(json.dumps(new_manifest, indent=2))

    if dry_run:
        log.info("[DRY RUN] would write to s3://%s/%s", bucket, CHAMPION_MANIFEST_KEY)
        log.info("[DRY RUN] would archive previous manifest (if present)")
        log.info("[DRY RUN] would append to s3://%s/%s", bucket, PROMOTIONS_LOG_KEY)
        log.info("[DRY RUN] would delete s3://%s/%s (if present)", bucket, PIPELINE_RERUN_KEY)
        return new_manifest

    # Archive previous manifest if present.
    previous = get_champion_manifest(s3_client, bucket)
    if previous is not None:
        prev_id = previous.get("champion_id", "unknown")
        archive_key = f"{HISTORY_PREFIX}/{prev_id}.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=archive_key,
            Body=json.dumps(previous, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        log.info("Archived previous champion %s to s3://%s/%s",
                 prev_id, bucket, archive_key)
    else:
        log.info("No previous champion manifest found — this is the first one.")

    # Write the new manifest.
    s3_client.put_object(
        Bucket=bucket,
        Key=CHAMPION_MANIFEST_KEY,
        Body=json.dumps(new_manifest, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    log.info("Wrote new champion manifest to s3://%s/%s", bucket, CHAMPION_MANIFEST_KEY)

    # Append to promotions log.
    log_line = f"{promoted_at}\t{champion_id}\t{instance_id}\t{git_sha}\n"
    try:
        existing = s3_client.get_object(Bucket=bucket, Key=PROMOTIONS_LOG_KEY)
        log_body = existing["Body"].read().decode("utf-8") + log_line
    except s3_client.exceptions.NoSuchKey:
        log_body = log_line
    s3_client.put_object(
        Bucket=bucket,
        Key=PROMOTIONS_LOG_KEY,
        Body=log_body.encode("utf-8"),
        ContentType="text/plain",
    )
    log.info("Appended promotion to s3://%s/%s", bucket, PROMOTIONS_LOG_KEY)

    # Delete pipeline_rerun.json if present — handing off from "in-flight"
    # to "post-run stale" is the responsibility of this script.
    try:
        s3_client.delete_object(Bucket=bucket, Key=PIPELINE_RERUN_KEY)
        log.info("Cleared in-flight marker s3://%s/%s", bucket, PIPELINE_RERUN_KEY)
    except Exception:
        log.debug("No %s to clear (or delete failed)", PIPELINE_RERUN_KEY)

    return new_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Declare a new DLC champion model.")
    parser.add_argument("--model-name", required=True,
                        help="DLC project name (e.g. hm2p_hrnetw32_shuffle1).")
    parser.add_argument("--architecture", required=True,
                        help="Architecture string (e.g. HrnetW32, Resnet50).")
    parser.add_argument("--snapshot", required=True,
                        help="Training iteration number as a string (e.g. 290).")
    parser.add_argument("--training-run-id", required=True,
                        help="Identifier from the EC2 retrain run "
                             "(matches run_id in dlc-retrain/_retrain_progress.json).")
    parser.add_argument("--notes", default="",
                        help="Auto-generated training-run description (technical details).")
    parser.add_argument("--note", default="",
                        help="Optional free-text operator annotation (separate from --notes).")
    parser.add_argument("--training-s3-prefix", default=TRAINING_S3_PREFIX)
    parser.add_argument("--bucket", default=DERIVATIVES_BUCKET)
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and print the manifest but do not write to S3.")
    args = parser.parse_args()

    declare_champion(
        model_name=args.model_name,
        architecture=args.architecture,
        snapshot=args.snapshot,
        training_run_id=args.training_run_id,
        notes=args.notes,
        note=args.note,
        training_s3_prefix=args.training_s3_prefix,
        bucket=args.bucket,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
