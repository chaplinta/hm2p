"""DLC model selection — single source of truth for choosing the best pose .h5.

All pipeline scripts and frontend pages must import from here. Do not
duplicate selection logic elsewhere.
"""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_best_dlc_h5(h5_keys: list[str]) -> str | None:
    """Select the best finetuned DLC .h5 from a list of S3 object keys.

    Selection rules (applied in order):

    1. Exclude ``_single`` and ``_filtered`` filename variants.
    2. Prefer finetuned models — filenames that contain ``"Hrnet"`` or
       ``"Resnet"`` (case-sensitive, matching DLC's own naming convention).
    3. Among finetuned candidates, pick the one with the highest
       ``snapshot-best-<N>`` number.
    4. If no finetuned file is found, return the first key in the filtered
       list (superanimal baseline fallback).

    Parameters
    ----------
    h5_keys:
        List of S3 object keys (full paths) to consider.

    Returns
    -------
    str | None
        The selected key, or ``None`` if *h5_keys* is empty after filtering.
    """
    filtered = [
        k for k in h5_keys
        if k.endswith(".h5")
        and "_single" not in k.split("/")[-1]
        and "_filtered" not in k.split("/")[-1]
    ]
    if not filtered:
        return None

    finetuned = [k for k in filtered if "Hrnet" in k or "Resnet" in k]
    if finetuned:
        return max(finetuned, key=_snapshot_number)
    return filtered[0]


def select_best_dlc_h5_s3(s3_client: object, bucket: str, prefix: str) -> str | None:
    """List .h5 files under an S3 prefix and select the best one.

    Checks for a ``promoted.json`` manifest first.  If that file exists and
    specifies an ``h5_filename``, the matching key is returned directly
    (explicit operator override).  If ``promoted.json`` is absent or the
    specified file is not found in the listing, falls back to
    :func:`select_best_dlc_h5`.

    Parameters
    ----------
    s3_client:
        A boto3 S3 client (``boto3.client("s3")``).
    bucket:
        S3 bucket name.
    prefix:
        S3 key prefix for the session's pose directory, e.g.
        ``"pose/sub-1114353/ses-20210823T165950/"``.

    Returns
    -------
    str | None
        The selected S3 key, or ``None`` if no usable .h5 is found.
    """
    # Collect all .h5 keys under the prefix via pagination.
    all_h5: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".h5"):
                all_h5.append(key)

    if not all_h5:
        return None

    # Check for an explicit promoted.json override.
    promoted_key = prefix.rstrip("/") + "/promoted.json"
    promoted = _load_promoted_json(s3_client, bucket, promoted_key)
    if promoted is not None:
        h5_filename = promoted.get("h5_filename")
        if h5_filename:
            # Match the filename against the listing.
            for key in all_h5:
                if key.split("/")[-1] == h5_filename:
                    log.debug("Using promoted DLC file: %s", key)
                    return key
            log.warning(
                "promoted.json specifies %r but file not found under %s; "
                "falling back to heuristic",
                h5_filename,
                prefix,
            )

    return select_best_dlc_h5(all_h5)


def extract_dlc_provenance(dlc_filename: str) -> tuple[str, str]:
    """Extract ``(model_name, snapshot)`` from a DLC output filename.

    DLC names output files using the convention::

        <video>DLC_<arch>_<project>_shuffle<N>_snapshot-best-<iter>.h5

    For fine-tuned models the filename contains an architecture string
    (``Hrnet`` or ``Resnet``) and a ``snapshot-best-<iter>`` suffix.
    SuperAnimal baseline outputs do not contain these markers.

    Parameters
    ----------
    dlc_filename:
        Bare filename (not a full path) of the DLC .h5 output.

    Returns
    -------
    tuple[str, str]
        ``(model_name, snapshot)`` where *model_name* is the DLC project
        name extracted from the filename, or ``"superanimal_topviewmouse"``
        for baseline outputs; and *snapshot* is the iteration number as a
        string, or ``"superanimal"`` for baseline outputs.
    """
    is_finetuned = "Hrnet" in dlc_filename or "Resnet" in dlc_filename
    if is_finetuned:
        snap_match = re.search(r"snapshot[_-]best[_-](\d+)", dlc_filename)
        snapshot = snap_match.group(1) if snap_match else "unknown"
        model_match = re.search(r"DLC_\w+?_(.+?)_shuffle", dlc_filename)
        model_name = model_match.group(1) if model_match else "unknown"
        return model_name, snapshot
    return "superanimal_topviewmouse", "superanimal"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _snapshot_number(key: str) -> int:
    """Return the snapshot iteration number from a DLC output key.

    Returns ``-1`` if no ``snapshot-best-<N>`` pattern is found, so that
    files without a snapshot number sort below all real snapshots.
    """
    m = re.search(r"snapshot[_-]best[_-](\d+)", key)
    return int(m.group(1)) if m else -1


def _load_promoted_json(s3_client: object, bucket: str, key: str) -> dict | None:
    """Fetch and parse promoted.json from S3, returning None on any error."""
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read())
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception:
        log.debug("Could not load %s/%s", bucket, key)
        return None
