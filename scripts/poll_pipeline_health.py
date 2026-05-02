#!/usr/bin/env python3
"""Pipeline health monitor — polls S3 for heartbeat, progress, and errors.

Checks the hm2p-derivatives/dlc-retrain/ prefix for JSON files written by
the GPU and CPU EC2 instances and reports alert conditions.

Usage:
    python scripts/poll_pipeline_health.py --once
    python scripts/poll_pipeline_health.py --watch
    python scripts/poll_pipeline_health.py --watch --interval 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.exceptions import ClientError

# Import shared constants
sys.path.insert(0, __file__.rsplit("/", 1)[0])
try:
    from ec2_constants import DERIVATIVES_BUCKET, REGION, RETRAIN_PREFIX
except ImportError:
    # Fallback if running outside the scripts/ directory
    DERIVATIVES_BUCKET = "hm2p-derivatives"
    REGION = "ap-southeast-2"
    RETRAIN_PREFIX = "dlc-retrain"

# Heartbeat staleness thresholds (seconds)
HEARTBEAT_WARN_AGE_S = 3 * 60  # 3 min → warning
HEARTBEAT_CRIT_AGE_S = 5 * 60  # 5 min → critical

# Progress staleness threshold for "no sessions started" condition (seconds)
NO_PROGRESS_CRIT_AGE_S = 30 * 60  # 30 min

# Total expected sessions
TOTAL_SESSIONS = 26

# S3 keys under RETRAIN_PREFIX
_KEY_GPU_HEARTBEAT = "_heartbeat.json"
_KEY_CPU_HEARTBEAT = "_downstream_heartbeat.json"
_KEY_GPU_PROGRESS = "_retrain_progress.json"
_KEY_CPU_PROGRESS = "_downstream_progress.json"
_KEY_INFERENCE_ERRORS = "_inference_errors.json"
_KEY_DOWNSTREAM_ERRORS = "_downstream_errors.json"
_KEY_RENDER_ERRORS = "_render_errors.json"

# ANSI colour codes
_RED = "\033[31m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

# Level ordering for sorting (lower = more severe)
_LEVEL_ORDER = {"critical": 0, "warning": 1, "info": 2}


@dataclass
class Alert:
    """A single alert produced by a health check.

    Parameters
    ----------
    level:
        Severity level: ``"critical"``, ``"warning"``, or ``"info"``.
    source:
        Identifier for the check that produced this alert, e.g.
        ``"gpu_heartbeat"`` or ``"cpu_progress"``.
    message:
        Human-readable description of the alert condition.
    timestamp:
        ISO 8601 UTC timestamp at which the alert was generated.
    """

    level: str
    source: str
    message: str
    timestamp: str = field(default_factory=lambda: _now_iso())


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _age_s(iso_timestamp: str) -> float:
    """Return how many seconds have elapsed since ``iso_timestamp`` (UTC).

    Parameters
    ----------
    iso_timestamp:
        An ISO 8601 UTC timestamp string, e.g. ``"2026-04-11T12:34:56Z"``.

    Returns
    -------
    float
        Elapsed seconds.  Returns ``float("inf")`` if parsing fails.
    """
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        return (datetime.now(UTC) - dt).total_seconds()
    except (ValueError, AttributeError):
        return float("inf")


def get_s3_json(s3: Any, key: str) -> dict | None:
    """Download and parse a JSON object from S3.

    Parameters
    ----------
    s3:
        A boto3 S3 client.
    key:
        The S3 key relative to the bucket root, e.g.
        ``"dlc-retrain/_heartbeat.json"``.

    Returns
    -------
    dict or None
        Parsed JSON content, or ``None`` if the key does not exist or the
        content cannot be parsed.
    """
    try:
        resp = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key=key)
        return json.loads(resp["Body"].read())
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise
    except json.JSONDecodeError:
        return None


def check_heartbeat(s3: Any, key: str, label: str) -> list[Alert]:
    """Check a heartbeat JSON for staleness.

    Produces a ``critical`` alert when the heartbeat is older than
    :data:`HEARTBEAT_CRIT_AGE_S`, a ``warning`` when older than
    :data:`HEARTBEAT_WARN_AGE_S`, and an ``info`` alert when fresh.
    If the key is missing, returns no alerts (the instance may not have
    started yet).

    Parameters
    ----------
    s3:
        A boto3 S3 client.
    key:
        Full S3 key for the heartbeat JSON.
    label:
        Short human-readable label for this instance, e.g. ``"GPU"`` or
        ``"CPU"``.

    Returns
    -------
    list[Alert]
        Zero or more alerts.
    """
    full_key = f"{RETRAIN_PREFIX}/{key}"
    data = get_s3_json(s3, full_key)
    if data is None:
        # Missing heartbeat — instance has not started or never wrote one.
        return []

    ts = data.get("timestamp", "")
    age = _age_s(ts)
    source = f"{label.lower()}_heartbeat"

    uptime = data.get("uptime_s")
    disk = data.get("disk_free_gb")
    load = data.get("load_avg_1m")
    detail = ""
    if uptime is not None:
        mins, secs = divmod(int(uptime), 60)
        hrs, mins = divmod(mins, 60)
        detail += f"  uptime={hrs}h{mins}m{secs}s"
    if disk is not None:
        detail += f"  disk_free={disk}GB"
    if load is not None:
        detail += f"  load_1m={load}"

    if age >= HEARTBEAT_CRIT_AGE_S:
        age_min = age / 60
        return [
            Alert(
                level="critical",
                source=source,
                message=(
                    f"{label} heartbeat stale for {age_min:.1f} min "
                    f"(last seen {ts}) — instance may be dead or hung.{detail}"
                ),
            )
        ]
    if age >= HEARTBEAT_WARN_AGE_S:
        age_min = age / 60
        return [
            Alert(
                level="warning",
                source=source,
                message=(
                    f"{label} heartbeat stale for {age_min:.1f} min "
                    f"(last seen {ts}) — possible transient issue.{detail}"
                ),
            )
        ]

    return [
        Alert(
            level="info",
            source=source,
            message=f"{label} instance alive (heartbeat age {age:.0f}s).{detail}",
        )
    ]


def check_progress(s3: Any, key: str, label: str) -> list[Alert]:
    """Check a progress JSON for staleness and failure conditions.

    Raises ``critical`` alerts when:

    - The progress ``updated`` timestamp is unchanged for more than 30 min
      and no sessions have been completed yet (i.e. nothing has started).

    Raises ``info`` alerts when progress is available and sessions are
    completing normally.

    Parameters
    ----------
    s3:
        A boto3 S3 client.
    key:
        Full S3 key for the progress JSON (without the retrain prefix).
    label:
        Human-readable label, e.g. ``"GPU"`` or ``"CPU"``.

    Returns
    -------
    list[Alert]
    """
    full_key = f"{RETRAIN_PREFIX}/{key}"
    data = get_s3_json(s3, full_key)
    if data is None:
        return []

    source = f"{label.lower()}_progress"
    status = data.get("status", "unknown")
    updated = data.get("updated", "")
    completed = data.get("completed", 0)
    failed = data.get("failed", 0)
    total = data.get("total", TOTAL_SESSIONS)

    alerts: list[Alert] = []

    # Check for "no sessions started after 30 min" condition.
    # This fires when the progress file exists (instance is running) but
    # completed is still 0 and the timestamp is very old.
    age = _age_s(updated)
    if completed == 0 and age >= NO_PROGRESS_CRIT_AGE_S:
        alerts.append(
            Alert(
                level="critical",
                source=source,
                message=(
                    f"{label} progress stale for {age / 60:.1f} min with 0 sessions "
                    f"completed — instance may have hung during setup. "
                    f"Last status: '{status}'"
                ),
            )
        )
    elif status in ("Inference complete", "Downstream complete", "complete"):
        # Run has finished — report summary.
        if failed == 0:
            alerts.append(
                Alert(
                    level="info",
                    source=source,
                    message=(
                        f"{label} run complete: {completed}/{total} sessions succeeded, "
                        f"0 failures."
                    ),
                )
            )
        # Failed-session alerts are handled by check_errors; don't duplicate here.
    else:
        alerts.append(
            Alert(
                level="info",
                source=source,
                message=(
                    f"{label} progress: {status} "
                    f"({completed}/{total} done, {failed} failed, "
                    f"updated {age:.0f}s ago)"
                ),
            )
        )

    return alerts


def check_errors(s3: Any, key: str, label: str) -> list[Alert]:
    """Check an error JSON for failure counts and specific error types.

    Raises ``critical`` alerts when all sessions have failed.
    Raises ``warning`` alerts when any sessions have failed.
    Inspects error types for S3 upload failures (``warning``).

    Parameters
    ----------
    s3:
        A boto3 S3 client.
    key:
        Full S3 key for the error JSON (without the retrain prefix).
    label:
        Human-readable label, e.g. ``"Inference"`` or ``"Downstream"``.

    Returns
    -------
    list[Alert]
    """
    full_key = f"{RETRAIN_PREFIX}/{key}"
    data = get_s3_json(s3, full_key)
    if data is None:
        return []

    source = f"{label.lower()}_errors"
    errors: list[dict] = data.get("errors", [])

    if not errors:
        return []

    n_errors = len(errors)
    alerts: list[Alert] = []

    # Check all-sessions-failed (critical)
    if n_errors >= TOTAL_SESSIONS:
        alerts.append(
            Alert(
                level="critical",
                source=source,
                message=(
                    f"All {n_errors} {label} sessions failed. "
                    f"First error: [{errors[0].get('error_type', '?')}] "
                    f"{errors[0].get('error_message', '')[:120]}"
                ),
            )
        )
    else:
        # Partial failures (warning)
        sessions = [e.get("session", "?") for e in errors]
        summary = ", ".join(sessions[:5])
        if len(sessions) > 5:
            summary += f" (+{len(sessions) - 5} more)"
        alerts.append(
            Alert(
                level="warning",
                source=source,
                message=(f"{n_errors} {label} session(s) failed: {summary}"),
            )
        )

    # Check for S3 upload-related errors specifically
    upload_errors = [
        e
        for e in errors
        if "upload" in e.get("error_type", "").lower()
        or "upload" in e.get("error_message", "").lower()
        or "s3" in e.get("error_type", "").lower()
    ]
    if upload_errors:
        alerts.append(
            Alert(
                level="warning",
                source=f"{source}_upload",
                message=(
                    f"{len(upload_errors)} S3 upload failure(s) in {label} errors — "
                    f"outputs may be missing from S3."
                ),
            )
        )

    return alerts


def _check_gpu_log_for_abort(s3: Any) -> list[Alert]:
    """Check the GPU run log for watchdog abort or hard timeout strings.

    Reads the GPU log from S3 and scans for known fatal strings. This is
    a best-effort check — a missing or unreadable log produces no alerts.

    Parameters
    ----------
    s3:
        A boto3 S3 client.

    Returns
    -------
    list[Alert]
    """
    log_key = f"{RETRAIN_PREFIX}/_gpu_run_log.txt"
    try:
        resp = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key=log_key)
        log_text = resp["Body"].read().decode("utf-8", errors="replace")
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return []
        raise

    alerts: list[Alert] = []

    if "FATAL: GPU utilization 0%" in log_text and "minutes during processing" in log_text:
        alerts.append(
            Alert(
                level="critical",
                source="gpu_watchdog",
                message=(
                    "GPU watchdog abort detected in log: DLC ran on CPU (0% GPU "
                    "utilisation for 10+ min). Instance has self-terminated."
                ),
            )
        )

    if "TIMEOUT:" in log_text and "reached. Terminating." in log_text:
        alerts.append(
            Alert(
                level="critical",
                source="gpu_timeout",
                message=(
                    "Hard timeout triggered on GPU instance — instance has "
                    "self-terminated. Check _gpu_run_log.txt for details."
                ),
            )
        )

    return alerts


def _check_render_errors(s3: Any) -> list[Alert]:
    """Check render error JSON for any render failures.

    Parameters
    ----------
    s3:
        A boto3 S3 client.

    Returns
    -------
    list[Alert]
    """
    full_key = f"{RETRAIN_PREFIX}/{_KEY_RENDER_ERRORS}"
    data = get_s3_json(s3, full_key)
    if data is None:
        return []

    errors: list[dict] = data.get("errors", [])
    if not errors:
        return []

    sessions = [e.get("session", "?") for e in errors]
    summary = ", ".join(sessions[:5])
    if len(sessions) > 5:
        summary += f" (+{len(sessions) - 5} more)"

    return [
        Alert(
            level="warning",
            source="render_errors",
            message=f"{len(errors)} video render failure(s): {summary}",
        )
    ]


def check_all(s3: Any) -> list[Alert]:
    """Run all health checks and return alerts sorted by severity.

    Checks GPU heartbeat, CPU heartbeat, GPU progress, CPU progress,
    inference errors, downstream errors, render errors, and the GPU log
    for watchdog/timeout strings.

    Parameters
    ----------
    s3:
        A boto3 S3 client.

    Returns
    -------
    list[Alert]
        All alerts sorted from most to least severe.
    """
    alerts: list[Alert] = []

    alerts.extend(check_heartbeat(s3, _KEY_GPU_HEARTBEAT, "GPU"))
    alerts.extend(check_heartbeat(s3, _KEY_CPU_HEARTBEAT, "CPU"))
    alerts.extend(check_progress(s3, _KEY_GPU_PROGRESS, "GPU"))
    alerts.extend(check_progress(s3, _KEY_CPU_PROGRESS, "CPU"))
    alerts.extend(check_errors(s3, _KEY_INFERENCE_ERRORS, "Inference"))
    alerts.extend(check_errors(s3, _KEY_DOWNSTREAM_ERRORS, "Downstream"))
    alerts.extend(_check_render_errors(s3))
    alerts.extend(_check_gpu_log_for_abort(s3))

    alerts.sort(key=lambda a: _LEVEL_ORDER.get(a.level, 99))
    return alerts


def format_alert(alert: Alert) -> str:
    """Format an alert as a coloured terminal string.

    Critical alerts are formatted in bold red, warnings in yellow, info in
    green.  The level label is padded to 8 characters for alignment.

    Parameters
    ----------
    alert:
        The alert to format.

    Returns
    -------
    str
        A single-line ANSI-coloured string.
    """
    level_label = alert.level.upper().ljust(8)
    if alert.level == "critical":
        colour = _RED + _BOLD
    elif alert.level == "warning":
        colour = _YELLOW
    else:
        colour = _GREEN

    return f"{colour}[{level_label}]{_RESET} [{alert.source}] {alert.message}"


def _print_report(alerts: list[Alert]) -> None:
    """Print a full poll report to stdout."""
    now_str = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Poll time: {now_str}")
    print(f"Bucket:    s3://{DERIVATIVES_BUCKET}/{RETRAIN_PREFIX}/")
    print()

    if not alerts:
        print(f"{_GREEN}No data found — pipeline may not have started yet.{_RESET}")
        return

    for alert in alerts:
        print(format_alert(alert))

    print()
    n_crit = sum(1 for a in alerts if a.level == "critical")
    n_warn = sum(1 for a in alerts if a.level == "warning")
    n_info = sum(1 for a in alerts if a.level == "info")
    parts = []
    if n_crit:
        parts.append(f"{_RED}{_BOLD}{n_crit} critical{_RESET}")
    if n_warn:
        parts.append(f"{_YELLOW}{n_warn} warnings{_RESET}")
    parts.append(f"{_GREEN}{n_info} info{_RESET}")
    print("Summary: " + ", ".join(parts))


def main() -> None:
    """Entry point for the pipeline health monitor."""
    parser = argparse.ArgumentParser(
        description="Monitor pipeline health by polling S3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/poll_pipeline_health.py --once\n"
            "  python scripts/poll_pipeline_health.py --watch\n"
            "  python scripts/poll_pipeline_health.py --watch --interval 60\n"
        ),
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--once",
        action="store_true",
        help="Check once and exit.",
    )
    mode_group.add_argument(
        "--watch",
        action="store_true",
        help="Poll repeatedly (default interval: 120s).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=120,
        metavar="SECONDS",
        help="Polling interval for --watch mode (default: 120).",
    )
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)

    if args.once:
        alerts = check_all(s3)
        _print_report(alerts)
        # Exit with non-zero code if any critical alerts
        n_crit = sum(1 for a in alerts if a.level == "critical")
        sys.exit(1 if n_crit else 0)

    # --watch mode
    prev_critical_sources: set[str] = set()

    while True:
        # Clear screen
        print("\033[H\033[J", end="", flush=True)

        alerts = check_all(s3)
        _print_report(alerts)

        # Ring terminal bell on any new critical alerts
        current_critical_sources = {a.source for a in alerts if a.level == "critical"}
        new_criticals = current_critical_sources - prev_critical_sources
        if new_criticals:
            print("\a", end="", flush=True)
        prev_critical_sources = current_critical_sources

        print(f"\n(next poll in {args.interval}s — Ctrl-C to stop)")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
