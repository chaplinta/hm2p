"""S3 upload utility with verify-on-landing retry.

Used for critical uploads (kinematics.h5, sync.h5, analysis.h5, model weights)
where silent failure is unacceptable.  Do NOT use for best-effort uploads
(heartbeat JSON, log files, progress JSON) — those have their own try/except.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def s3_upload_with_verify(
    s3: Any,
    local_path: str | Path,
    bucket: str,
    key: str,
    *,
    retries: int = 3,
    retry_delay_s: int = 15,
) -> None:
    """Upload local_path to S3 and verify with head_object.

    Both the upload and the head_object check are retried up to ``retries``
    times before raising.  On each attempt the file is re-uploaded from scratch
    so a partial transfer is not silently treated as success.

    Parameters
    ----------
    s3 : boto3 S3 client
    local_path : str or Path
        Local file to upload.
    bucket : str
        Destination S3 bucket.
    key : str
        Destination S3 key.
    retries : int
        Maximum number of upload+verify attempts (default 3).
    retry_delay_s : int
        Seconds to wait between attempts (default 15).

    Raises
    ------
    RuntimeError
        If the file cannot be verified on S3 after all retries.
    """
    local_path = Path(local_path)
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        if attempt > 1:
            print(
                f"  s3_upload_with_verify: retry {attempt}/{retries} "
                f"for s3://{bucket}/{key} (sleeping {retry_delay_s}s)"
            )
            time.sleep(retry_delay_s)

        try:
            s3.upload_file(str(local_path), bucket, key)
        except Exception as exc:
            print(f"  s3_upload_with_verify: upload failed (attempt {attempt}): {exc}")
            last_exc = exc
            continue

        try:
            s3.head_object(Bucket=bucket, Key=key)
            # Verified — upload landed.
            return
        except Exception as exc:
            print(
                f"  s3_upload_with_verify: head_object failed after upload "
                f"(attempt {attempt}): {exc}"
            )
            last_exc = exc

    raise RuntimeError(
        f"s3_upload_with_verify: failed to confirm s3://{bucket}/{key} "
        f"after {retries} attempt(s). Last error: {last_exc}"
    ) from last_exc
