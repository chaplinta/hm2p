#!/usr/bin/env python3
"""Write promoted.json manifests for all sessions.

For each session that has DLC pose output on S3, this script writes a
``promoted.json`` manifest to ``pose/{sub}/{ses}/promoted.json``. Two
selection modes:

- Default (no flag): pick the .h5 with the highest ``snapshot-best-N`` number
  (i.e. ``select_best_dlc_h5_s3``).
- ``--snapshot N``: pick the .h5 whose filename contains
  ``snapshot_best-N``. Sessions that don't have an inference output for
  that snapshot are skipped with a warning.

The explicit ``--snapshot`` form is used to switch the project to a non-tip
snapshot (e.g. an earlier checkpoint that scored better on the training
metrics that matter for downstream HD analysis).

Once ``promoted.json`` is in place, :func:`hm2p.pose.select.select_best_dlc_h5_s3`
uses it as an explicit override instead of re-running the heuristic.

Usage
-----
    # Dry run — print what would be written without uploading
    uv run python scripts/promote_dlc_model.py --dry-run

    # Write manifests for all sessions, default heuristic
    uv run python scripts/promote_dlc_model.py

    # Write manifest for a specific session
    uv run python scripts/promote_dlc_model.py --session sub-1114353/ses-20210823T165950

    # Promote a specific snapshot (must already exist as a finetuned .h5
    # output for each session)
    uv run python scripts/promote_dlc_model.py --snapshot 110
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hm2p.pose.select import extract_dlc_provenance, select_best_dlc_h5_s3


def _select_snapshot_h5(
    s3_client: object, bucket: str, prefix: str, snapshot: str
) -> str | None:
    """Return the finetuned h5 key whose filename matches ``snapshot_best-{snapshot}``.

    Prefers HrnetW32 (DLC 3.x PyTorch) over Resnet50 when both are present,
    matching the project default architecture. Returns ``None`` if no
    matching file exists under *prefix*.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    matches: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".h5"):
                continue
            fname = key.split("/")[-1]
            if "_single" in fname or "_filtered" in fname:
                continue
            if f"snapshot_best-{snapshot}" not in fname:
                continue
            matches.append(key)
    if not matches:
        return None
    hrnet = [k for k in matches if "Hrnet" in k]
    return hrnet[0] if hrnet else matches[0]

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
METADATA_PATH = Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("promote_dlc")


def _load_sessions() -> list[dict]:
    """Return (sub, ses) pairs from experiments.csv."""
    sessions: list[dict] = []
    with open(METADATA_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_id = row["exp_id"]
            parts = exp_id.split("_")
            if len(parts) < 5:
                continue
            date, hh, mm, ss, animal = parts[0], parts[1], parts[2], parts[3], parts[4]
            sub = f"sub-{animal}"
            ses = f"ses-{date}T{hh}{mm}{ss}"
            sessions.append({"sub": sub, "ses": ses, "exp_id": exp_id})
    return sessions


def _write_promoted(
    s3: object,
    sub: str,
    ses: str,
    h5_key: str,
    dry_run: bool,
) -> None:
    """Build and upload promoted.json for one session."""
    h5_filename = h5_key.split("/")[-1]
    model_name, snapshot = extract_dlc_provenance(h5_filename)

    # Infer architecture from filename
    if "HrnetW32" in h5_filename or "Hrnet" in h5_filename:
        architecture = "HrnetW32"
    elif "Resnet50" in h5_filename or "Resnet" in h5_filename:
        architecture = "Resnet50"
    else:
        architecture = "superanimal"

    manifest = {
        "h5_filename": h5_filename,
        "h5_key": h5_key,
        "model_name": model_name,
        "architecture": architecture,
        "snapshot": snapshot,
        "promoted_at": datetime.datetime.now(datetime.UTC).isoformat(
            timespec="seconds"
        ),
    }

    dest_key = f"pose/{sub}/{ses}/promoted.json"
    if dry_run:
        log.info("[DRY RUN] Would write s3://%s/%s", DERIVATIVES_BUCKET, dest_key)
        log.info("          %s", json.dumps(manifest, indent=2))
        return

    s3.put_object(
        Bucket=DERIVATIVES_BUCKET,
        Key=dest_key,
        Body=json.dumps(manifest, indent=2).encode(),
        ContentType="application/json",
    )
    log.info("Wrote s3://%s/%s  (snapshot=%s)", DERIVATIVES_BUCKET, dest_key, snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without uploading to S3.",
    )
    parser.add_argument(
        "--session",
        metavar="sub-XXX/ses-YYYYMMDDTHHMMSS",
        help="Process a single session instead of all sessions.",
    )
    parser.add_argument(
        "--snapshot",
        metavar="N",
        help=(
            "Promote the finetuned .h5 with ``snapshot_best-N`` in its filename. "
            "Sessions that lack an inference output for this snapshot are "
            "skipped with a warning. Without this flag the highest-snapshot "
            "heuristic is used."
        ),
    )
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)

    if args.session:
        parts = args.session.strip("/").split("/")
        if len(parts) != 2:
            log.error("--session must be in the form sub-XXX/ses-YYYYMMDDTHHMMSS")
            sys.exit(1)
        sessions = [{"sub": parts[0], "ses": parts[1], "exp_id": args.session}]
    else:
        sessions = _load_sessions()

    log.info("Processing %d session(s)...", len(sessions))
    n_written = 0
    n_skipped = 0

    for sess in sessions:
        sub, ses = sess["sub"], sess["ses"]
        prefix = f"pose/{sub}/{ses}/"
        if args.snapshot is not None:
            h5_key = _select_snapshot_h5(s3, DERIVATIVES_BUCKET, prefix, args.snapshot)
            if h5_key is None:
                log.warning(
                    "No snapshot-best-%s .h5 found for %s/%s — skipping",
                    args.snapshot, sub, ses,
                )
                n_skipped += 1
                continue
        else:
            h5_key = select_best_dlc_h5_s3(s3, DERIVATIVES_BUCKET, prefix)
            if h5_key is None:
                log.warning("No pose .h5 found for %s/%s — skipping", sub, ses)
                n_skipped += 1
                continue
        _write_promoted(s3, sub, ses, h5_key, dry_run=args.dry_run)
        n_written += 1

    log.info(
        "Done. %d written, %d skipped (no pose data).", n_written, n_skipped
    )


if __name__ == "__main__":
    main()
