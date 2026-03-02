"""AWS cost estimation, logging, and user confirmation.

Estimates S3 operation costs before execution and optionally prompts the user
for confirmation.  All pricing is based on AWS ap-southeast-2 (Sydney) public
list prices as of March 2026.

Pricing reference:
    https://aws.amazon.com/s3/pricing/ (ap-southeast-2)
    S3 Standard storage:            $0.025 / GB / month
    PUT/COPY/POST/LIST requests:    $0.005 / 1,000
    GET/SELECT requests:            $0.0004 / 1,000
    Data transfer out (first 10TB): $0.09 / GB
    Data transfer in:               free

Snakemake AWS Batch compute costs are *not* estimated here — spot pricing is
unpredictable and Snakemake's executor handles instance provisioning outside
our code.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("hm2p.aws_cost")

# ── Pricing constants (USD, ap-southeast-2) ───────────────────────────────────
_PUT_PER_1K: float = 0.005
_GET_PER_1K: float = 0.0004
_STORAGE_PER_GB_MONTH: float = 0.025
_EGRESS_PER_GB: float = 0.09
_BYTES_PER_GB: int = 1_073_741_824  # 2**30


@dataclass(frozen=True)
class CostEstimate:
    """Estimated cost for one S3 operation."""

    operation: str  # "upload" | "download"
    n_files: int
    total_bytes: int
    request_cost_usd: float
    transfer_cost_usd: float
    storage_cost_usd: float  # one month, informational for uploads

    @property
    def total_cost_usd(self) -> float:
        """Total estimated cost (requests + transfer, excludes storage)."""
        return self.request_cost_usd + self.transfer_cost_usd

    def summary(self) -> str:
        """Human-readable multi-line summary of the estimate."""
        gb = self.total_bytes / _BYTES_PER_GB
        lines = [
            f"  Operation : {self.operation}",
            f"  Files     : {self.n_files:,}",
            f"  Size      : {gb:.3f} GB ({self.total_bytes:,} bytes)",
            f"  Requests  : ${self.request_cost_usd:.4f}",
            f"  Transfer  : ${self.transfer_cost_usd:.4f}",
            f"  Total     : ${self.total_cost_usd:.4f} USD",
        ]
        if self.storage_cost_usd > 0:
            lines.append(f"  Storage   : ${self.storage_cost_usd:.4f} USD/month (if newly stored)")
        return "\n".join(lines)


# ── Pure estimation functions ─────────────────────────────────────────────────


def estimate_upload_from_counts(n_files: int, total_bytes: int) -> CostEstimate:
    """Estimate upload cost from pre-computed file count and total size.

    Args:
        n_files: Number of files to upload.
        total_bytes: Total size in bytes.

    Returns:
        CostEstimate with operation="upload".
    """
    request_cost = (n_files / 1000) * _PUT_PER_1K
    storage_cost = (total_bytes / _BYTES_PER_GB) * _STORAGE_PER_GB_MONTH
    return CostEstimate(
        operation="upload",
        n_files=n_files,
        total_bytes=total_bytes,
        request_cost_usd=request_cost,
        transfer_cost_usd=0.0,  # ingress is free
        storage_cost_usd=storage_cost,
    )


def estimate_upload(files: list[Path]) -> CostEstimate:
    """Estimate upload cost by stat-ing local files.

    Args:
        files: Local file paths to upload. All must exist.

    Returns:
        CostEstimate with operation="upload".
    """
    total_bytes = sum(f.stat().st_size for f in files)
    return estimate_upload_from_counts(len(files), total_bytes)


def estimate_download(n_files: int, total_bytes: int) -> CostEstimate:
    """Estimate download cost from object count and total size.

    Args:
        n_files: Number of S3 objects to download.
        total_bytes: Total size in bytes (from S3 listing metadata).

    Returns:
        CostEstimate with operation="download".
    """
    request_cost = (n_files / 1000) * _GET_PER_1K
    transfer_cost = (total_bytes / _BYTES_PER_GB) * _EGRESS_PER_GB
    return CostEstimate(
        operation="download",
        n_files=n_files,
        total_bytes=total_bytes,
        request_cost_usd=request_cost,
        transfer_cost_usd=transfer_cost,
        storage_cost_usd=0.0,
    )


# ── Confirmation gate ─────────────────────────────────────────────────────────


def confirm_or_abort(
    estimate: CostEstimate,
    yes: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """Log the cost estimate and prompt for user confirmation.

    Args:
        estimate: The cost estimate to display.
        yes: If True, skip the interactive prompt and proceed.
        logger: Logger instance. Defaults to module-level ``hm2p.aws_cost``.

    Raises:
        SystemExit: If the user declines or stdin is non-interactive.
    """
    _log = logger or log
    _log.info("AWS cost estimate:\n%s", estimate.summary())

    if yes:
        _log.info("Proceeding without confirmation (--yes).")
        return

    try:
        answer = input("\nProceed? [y/N] ").strip().lower()
    except EOFError:
        answer = ""

    if answer not in ("y", "yes"):
        _log.warning("Aborted by user.")
        sys.exit(1)
