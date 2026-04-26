"""DLC model selection — single source of truth for choosing the best pose .h5.

All pipeline scripts and frontend pages must import from here. Do not
duplicate selection logic elsewhere.

Champion model concept (see docs/dlc-champion-model.md):
    The ``dlc-champion.json`` manifest at the root of the derivatives bucket
    is the project-wide source of truth for which DLC model is current. The
    pure helpers in this module — ``compute_champion_id``,
    ``extract_architecture``, ``get_champion_manifest`` — let any caller
    derive a deterministic ``champion_id`` string from a manifest, an h5
    filename, or both. Callers stamp that string into derivative outputs;
    the frontend reads it back and refuses to display anything that does
    not match the current champion.
"""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

# S3 key of the project-wide champion manifest, relative to the derivatives
# bucket. Public constant so callers don't hardcode the path.
CHAMPION_MANIFEST_KEY = "dlc-champion.json"

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


def extract_architecture(dlc_filename: str) -> str | None:
    """Extract the DLC architecture name from an output filename.

    Recognises the two architectures the project uses:

    - ``"HrnetW32"`` — DLC 3.0 PyTorch HRNet (W32 width).
    - ``"Resnet50"`` — DLC 2.x TensorFlow ResNet-50.

    Other variants (``HrnetW48``, ``ResnetXX``) are returned in their
    canonical capitalised form. Returns ``None`` if the filename does
    not contain a recognisable architecture marker (e.g. SuperAnimal
    baseline output).

    Parameters
    ----------
    dlc_filename:
        Bare filename (not a full path) of a DLC .h5 output.

    Returns
    -------
    str | None
        The architecture string, or ``None`` if not recognisable.
    """
    m = re.search(r"DLC_(Hrnet[A-Za-z0-9]+|Resnet[0-9]+)_", dlc_filename)
    return m.group(1) if m else None


def compute_champion_id(
    model_name: str,
    architecture: str,
    snapshot: str,
    training_date: str | None = None,
) -> str:
    """Build a deterministic champion-id string.

    Format: ``dlc-{YYYYMMDD}-{arch_lower}-snap{N}``.

    The id is constructed only from the manifest fields so a caller can
    reconstruct it from HDF5 attributes alone (without having to fetch the
    manifest from S3). It is the single string compared across all
    derivatives to decide whether they were produced by the current model.

    Parameters
    ----------
    model_name:
        DLC project name. Currently unused in the id but kept in the
        signature for forward-compatibility (e.g. if two projects ever
        coexist).
    architecture:
        Architecture string from :func:`extract_architecture` (e.g.
        ``"HrnetW32"``).
    snapshot:
        Training iteration as a string (e.g. ``"50000"``).
    training_date:
        ISO date (``YYYY-MM-DD``) of training completion. If ``None``,
        today's UTC date is used. Pass an explicit date when reconstructing
        an id retroactively.

    Returns
    -------
    str
        The champion id.
    """
    import datetime

    if training_date is None:
        training_date = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    date_compact = training_date.replace("-", "")
    arch_lower = architecture.lower()
    return f"dlc-{date_compact}-{arch_lower}-snap{snapshot}"


def resolve_champion_id(
    model_name: str,
    architecture: str | None,
    snapshot: str,
    manifest: dict | None,
) -> str:
    """Return the ``dlc_champion_id`` to stamp on a derivative.

    The stamp is the manifest's ``champion_id`` only when the tuple
    ``(model_name, architecture, snapshot)`` matches the current champion
    exactly. Otherwise the stamp is ``"unknown"`` — the derivative was
    produced by a different (older or experimental) model, or the
    manifest does not exist yet.

    The frontend treats ``"unknown"`` as stale and shows a warning. There
    is intentionally no attempt to reconstruct an old champion id from the
    triplet alone, because the training date of an old model cannot be
    inferred from the h5 filename.

    Parameters
    ----------
    model_name, architecture, snapshot:
        Triplet derived from the DLC h5 filename via
        :func:`extract_dlc_provenance` and :func:`extract_architecture`.
        ``architecture`` may be ``None`` for SuperAnimal baseline files;
        in that case the function returns ``"unknown"`` directly.
    manifest:
        The current champion manifest dict (from
        :func:`get_champion_manifest`) or ``None`` when no manifest
        exists yet.

    Returns
    -------
    str
        Either the manifest's ``champion_id`` value (if matched) or
        ``"unknown"``.
    """
    if manifest is None or architecture is None:
        return "unknown"
    if (
        manifest.get("model_name") == model_name
        and manifest.get("architecture") == architecture
        and str(manifest.get("snapshot")) == str(snapshot)
    ):
        return str(manifest.get("champion_id", "unknown"))
    return "unknown"


def get_champion_manifest(
    s3_client: object,
    bucket: str,
    key: str = CHAMPION_MANIFEST_KEY,
) -> dict | None:
    """Fetch and parse the project-wide DLC champion manifest from S3.

    Returns ``None`` when the manifest is absent (e.g. before the first
    champion has been declared) or when the object cannot be parsed.
    Errors are logged but never raised.

    Parameters
    ----------
    s3_client:
        A boto3 S3 client.
    bucket:
        Derivatives bucket name (e.g. ``"hm2p-derivatives"``).
    key:
        S3 object key. Defaults to :data:`CHAMPION_MANIFEST_KEY`.

    Returns
    -------
    dict | None
        Parsed manifest dict, or ``None`` if absent / unreadable.
    """
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read())
    except s3_client.exceptions.NoSuchKey:
        log.debug("No champion manifest at s3://%s/%s", bucket, key)
        return None
    except Exception:
        log.exception("Could not load champion manifest s3://%s/%s", bucket, key)
        return None


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
