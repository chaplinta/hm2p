"""S3 upload/download helpers — used by Stage 0 ingest and CLI upload command.

All functions use the boto3 S3 client with the ``hm2p-agent`` AWS profile by
default.  The profile can be overridden via the ``AWS_PROFILE`` environment
variable or the ``profile`` parameter.

Before executing any transfer, functions estimate the AWS cost and prompt the
user for confirmation (unless ``confirm=False``).

Typical usage::

    from hm2p.io.s3 import upload_session, download_session

    # Upload raw session data to s3://hm2p-rawdata
    upload_session(
        local_dir=Path("/data/rawdata/sub-1117646/ses-20220804T135202"),
        bucket="hm2p-rawdata",
        prefix="rawdata/sub-1117646/ses-20220804T135202",
    )
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_PROFILE = "hm2p-agent"


def _client(profile: str | None = None) -> object:
    """Return a boto3 S3 client for the given AWS profile."""
    import boto3

    profile_name = profile or os.environ.get("AWS_PROFILE", _DEFAULT_PROFILE)
    session = boto3.Session(profile_name=profile_name)
    return session.client("s3")


def upload_file(
    local_path: Path,
    bucket: str,
    key: str,
    profile: str | None = None,
    confirm: bool = True,
) -> None:
    """Upload a single local file to S3.

    Args:
        local_path: Local file to upload.
        bucket: S3 bucket name (e.g. ``"hm2p-rawdata"``).
        key: S3 object key (path within the bucket).
        profile: AWS profile name (default: ``hm2p-agent``).
        confirm: If True, estimate cost and prompt before uploading.

    Raises:
        FileNotFoundError: If ``local_path`` does not exist.
        botocore.exceptions.ClientError: On S3 API errors.
    """
    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    if confirm:
        from hm2p.io.aws_cost import confirm_or_abort, estimate_upload

        confirm_or_abort(estimate_upload([local_path]))

    _client(profile).upload_file(str(local_path), bucket, key)


def download_file(
    bucket: str,
    key: str,
    local_path: Path,
    profile: str | None = None,
    confirm: bool = True,
) -> None:
    """Download a single S3 object to a local file.

    Creates parent directories as needed.

    Args:
        bucket: S3 bucket name.
        key: S3 object key.
        local_path: Destination local path.
        profile: AWS profile name (default: ``hm2p-agent``).
        confirm: If True, estimate cost and prompt before downloading.

    Raises:
        botocore.exceptions.ClientError: On S3 API errors (e.g. NoSuchKey).
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client = _client(profile)

    if confirm:
        from hm2p.io.aws_cost import confirm_or_abort, estimate_download

        head = client.head_object(Bucket=bucket, Key=key)
        size = head["ContentLength"]
        confirm_or_abort(estimate_download(n_files=1, total_bytes=size))

    client.download_file(bucket, key, str(local_path))


def upload_session(
    local_dir: Path,
    bucket: str,
    prefix: str,
    exclude_patterns: tuple[str, ...] = ("*_side_left*", "*red.tif*"),
    profile: str | None = None,
    confirm: bool = True,
) -> list[str]:
    """Upload all files in a local session directory to S3.

    Mirrors the directory tree under ``prefix`` in the bucket.  Files
    matching any pattern in ``exclude_patterns`` are skipped (e.g.
    side-camera video and red-channel TIFFs which are excluded per
    project policy).

    Args:
        local_dir: Root of the local session directory to upload.
        bucket: Destination S3 bucket name.
        prefix: Key prefix in the bucket (no trailing slash needed).
        exclude_patterns: Glob-style patterns for files to skip.
        profile: AWS profile name (default: ``hm2p-agent``).
        confirm: If True, estimate cost and prompt before uploading.

    Returns:
        List of S3 keys successfully uploaded.

    Raises:
        FileNotFoundError: If ``local_dir`` does not exist.
        botocore.exceptions.ClientError: On S3 API errors.
    """
    import fnmatch

    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    prefix = prefix.rstrip("/")

    # Collect files first so we can estimate cost before any transfer
    files_to_upload: list[tuple[Path, str]] = []
    for local_file in sorted(local_dir.rglob("*")):
        if not local_file.is_file():
            continue
        name = local_file.name
        if any(fnmatch.fnmatch(name, pat) for pat in exclude_patterns):
            continue
        relative = local_file.relative_to(local_dir)
        key = f"{prefix}/{relative.as_posix()}"
        files_to_upload.append((local_file, key))

    if confirm and files_to_upload:
        from hm2p.io.aws_cost import confirm_or_abort, estimate_upload

        confirm_or_abort(estimate_upload([f for f, _ in files_to_upload]))

    client = _client(profile)
    uploaded: list[str] = []
    for local_file, key in files_to_upload:
        client.upload_file(str(local_file), bucket, key)
        uploaded.append(key)

    return uploaded


def download_session(
    bucket: str,
    prefix: str,
    local_dir: Path,
    profile: str | None = None,
    confirm: bool = True,
) -> list[Path]:
    """Download all objects under a bucket prefix to a local directory.

    Args:
        bucket: Source S3 bucket name.
        prefix: Key prefix to download (e.g. ``"rawdata/sub-X/ses-Y"``).
        local_dir: Root local directory to download into.
        profile: AWS profile name (default: ``hm2p-agent``).
        confirm: If True, estimate cost and prompt before downloading.

    Returns:
        List of local paths written.

    Raises:
        botocore.exceptions.ClientError: On S3 API errors.
    """
    client = _client(profile)
    paginator = client.get_paginator("list_objects_v2")

    prefix = prefix.rstrip("/") + "/"

    # Collect all objects first so we can estimate cost
    all_objects: list[tuple[str, int]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(prefix) :]
            if relative:
                all_objects.append((key, obj.get("Size", 0)))

    if confirm and all_objects:
        from hm2p.io.aws_cost import confirm_or_abort, estimate_download

        total_bytes = sum(size for _, size in all_objects)
        confirm_or_abort(estimate_download(n_files=len(all_objects), total_bytes=total_bytes))

    written: list[Path] = []
    for key, _ in all_objects:
        relative = key[len(prefix) :]
        dest = local_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(dest))
        written.append(dest)

    return written


def list_sessions(
    bucket: str,
    prefix: str = "rawdata/",
    profile: str | None = None,
) -> list[str]:
    """List session prefixes under a bucket root.

    Cost: negligible (S3 LIST, < $0.001 per call). No confirmation needed.

    Args:
        bucket: S3 bucket name.
        prefix: Root prefix to list under (default ``"rawdata/"``).
        profile: AWS profile name (default: ``hm2p-agent``).

    Returns:
        Sorted list of common prefix strings (sub-level folders).
    """
    client = _client(profile)
    paginator = client.get_paginator("list_objects_v2")

    prefixes: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            prefixes.append(cp["Prefix"])

    return sorted(prefixes)
