"""S3 path resolution — transparent local vs cloud operation.

When compute_profile is 'aws-batch', data_root is an S3 URI and paths
are resolved via s3fs. When running locally, data_root is a local directory.

All pipeline stages call path_for() to get a concrete file path — they never
hard-code local or S3 paths directly.
"""

from __future__ import annotations

from pathlib import Path


def is_s3(path: str | Path) -> bool:
    """Return True if path is an S3 URI (starts with s3://)."""
    return str(path).startswith("s3://")


def path_for(
    data_root: Path | str,
    relative: str | Path,
) -> Path:
    """Resolve a relative path under data_root to a concrete local or S3 path.

    For local roots, returns a plain Path.
    For S3 roots, returns a path that can be passed to s3fs.S3FileSystem.

    Args:
        data_root: Root data directory (local path or s3:// URI).
        relative: Relative path within the data root.

    Returns:
        Resolved Path (or s3:// string cast as Path for s3fs compatibility).
    """
    if is_s3(data_root):
        return Path(str(data_root).rstrip("/") + "/" + str(relative).lstrip("/"))
    return Path(data_root) / relative
