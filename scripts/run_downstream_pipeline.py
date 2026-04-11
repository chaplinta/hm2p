#!/usr/bin/env python3
"""Run downstream pipeline (Stages 3-5-6) for sessions with completed DLC.

Checks which sessions have DLC output on S3 but are missing kinematics,
sync, or analysis, and runs those stages sequentially.

Usage:
    python scripts/run_downstream_pipeline.py              # run all pending
    python scripts/run_downstream_pipeline.py --session 0  # first session only
    python scripts/run_downstream_pipeline.py --dry-run    # show what would run
    python scripts/run_downstream_pipeline.py --watch      # poll and run as DLC completes
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
import time
import urllib.request
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
RETRAIN_PREFIX = "dlc-retrain"


def _get_instance_id() -> str:
    """Return the EC2 instance ID from the metadata service, or 'unknown'."""
    try:
        resp = urllib.request.urlopen(
            "http://169.254.169.254/latest/meta-data/instance-id", timeout=2
        )
        return resp.read().decode().strip()
    except Exception:
        return "unknown"


def update_downstream_progress(
    s3,
    session_idx: int,
    total: int,
    exp_id: str,
    stage: str,
    status: str,
    completed: int,
    failed: int,
) -> None:
    """Write per-stage downstream progress to S3.

    Writes ``dlc-retrain/_downstream_progress.json``.  Best-effort — upload
    failures are logged as warnings and do not propagate to the caller.

    Parameters
    ----------
    s3 : boto3 S3 client
    session_idx : int
        1-based index of the session being processed.
    total : int
        Total number of sessions.
    exp_id : str
        Session experiment ID.
    stage : str
        Stage name, e.g. ``"stage3"``, ``"stage5"``, ``"stage6"``.
    status : str
        Human-readable status, e.g. ``"done"`` or ``"failed"``.
    completed : int
        Number of fully completed sessions so far.
    failed : int
        Number of failed sessions so far.
    """
    payload = {
        "status": status,
        "updated": datetime.datetime.utcnow().isoformat() + "Z",
        "session": exp_id,
        "stage": stage,
        "session_idx": session_idx,
        "completed": completed,
        "failed": failed,
        "total": total,
    }
    try:
        s3.put_object(
            Bucket=DERIVATIVES_BUCKET,
            Key=f"{RETRAIN_PREFIX}/_downstream_progress.json",
            Body=json.dumps(payload, indent=2).encode(),
        )
    except Exception as e:
        print(f"  WARNING: downstream progress update failed (non-fatal): {e}")


def get_sessions() -> list[dict]:
    """Read all sessions from metadata/experiments.csv."""
    csv_path = Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"
    sessions = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_id = row["exp_id"]
            parts = exp_id.split("_")
            animal = parts[-1]
            sub = f"sub-{animal}"
            ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
            sessions.append({
                "exp_id": exp_id,
                "sub": sub,
                "ses": ses,
                "orientation": row.get("orientation", "0"),
                "bad_behav_times": row.get("bad_behav_times", ""),
            })
    return sessions


def check_stage_exists(s3, sub: str, ses: str, stage: str, file_pattern: str = "") -> bool:
    """Check if a stage output exists on S3."""
    prefix = f"{stage}/{sub}/{ses}/"
    resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=prefix, MaxKeys=10)
    if resp.get("KeyCount", 0) == 0:
        return False
    if file_pattern:
        keys = [obj["Key"] for obj in resp.get("Contents", [])]
        return any(file_pattern in k for k in keys)
    return True


def get_pipeline_status(s3, sessions: list[dict]) -> list[dict]:
    """Check which stages are done for each session."""
    statuses = []
    for ses_info in sessions:
        sub, ses = ses_info["sub"], ses_info["ses"]
        status = {
            "exp_id": ses_info["exp_id"],
            "sub": sub,
            "ses": ses,
            "pose": check_stage_exists(s3, sub, ses, "pose", ".h5"),
            "kinematics": check_stage_exists(s3, sub, ses, "kinematics", "kinematics.h5"),
            "calcium": check_stage_exists(s3, sub, ses, "calcium", "ca.h5"),
            "sync": check_stage_exists(s3, sub, ses, "sync", "sync.h5"),
            "analysis": check_stage_exists(s3, sub, ses, "analysis", "analysis.h5"),
        }
        statuses.append({**ses_info, **status})
    return statuses


def run_stage3(session: dict, dry_run: bool = False) -> tuple[bool, str]:
    """Run Stage 3 (kinematics) for a session.

    Returns
    -------
    tuple[bool, str]
        ``(success, stderr_excerpt)``
    """
    import subprocess
    cmd = [
        sys.executable, "scripts/run_stage3_kinematics.py",
        "--session", session["exp_id"],
    ]
    print(f"  Stage 3 (kinematics): {' '.join(cmd)}")
    if dry_run:
        return True, ""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr[:500]
        print(f"  Stage 3 FAILED: {stderr}")
        return False, stderr
    print(f"  Stage 3 DONE")
    return True, ""


def run_stage5(session: dict, dry_run: bool = False) -> tuple[bool, str]:
    """Run Stage 5 (sync) for a session.

    Returns
    -------
    tuple[bool, str]
        ``(success, stderr_excerpt)``
    """
    import subprocess
    cmd = [
        sys.executable, "scripts/run_stage5_sync.py",
        "--session", session["exp_id"],
    ]
    print(f"  Stage 5 (sync): {' '.join(cmd)}")
    if dry_run:
        return True, ""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr[:500]
        print(f"  Stage 5 FAILED: {stderr}")
        return False, stderr
    print(f"  Stage 5 DONE")
    return True, ""


def run_stage6(session: dict, dry_run: bool = False) -> tuple[bool, str]:
    """Run Stage 6 (analysis) for a session.

    Returns
    -------
    tuple[bool, str]
        ``(success, stderr_excerpt)``
    """
    import subprocess
    cmd = [
        sys.executable, "scripts/run_stage6_analysis.py",
        "--session", session["exp_id"],
    ]
    print(f"  Stage 6 (analysis): {' '.join(cmd)}")
    if dry_run:
        return True, ""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr[:500]
        print(f"  Stage 6 FAILED: {stderr}")
        return False, stderr
    print(f"  Stage 6 DONE")
    return True, ""


def process_session(
    session: dict,
    s3,
    session_idx: int,
    total: int,
    completed_count: int,
    failed_count: int,
    error_records: list[dict],
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Run all pending stages for a session, recording errors and progress.

    Parameters
    ----------
    session : dict
        Session dict with pipeline status flags.
    s3 : boto3 S3 client
        Used for progress and error uploads.
    session_idx : int
        1-based index of this session in the overall run.
    total : int
        Total number of sessions being processed.
    completed_count : int
        Number of sessions fully completed before this one.
    failed_count : int
        Number of sessions that failed before this one.
    error_records : list[dict]
        Mutable list; failed stages append their error record here.
    dry_run : bool
        If True, print commands without executing.
    force : bool
        If True, re-run all stages even if outputs exist.
    """
    exp_id = session["exp_id"]
    results = {"exp_id": exp_id}

    # Stage 3: Kinematics (requires pose)
    if (force or not session.get("kinematics")) and session.get("pose"):
        ok, stderr = run_stage3(session, dry_run)
        results["stage3"] = ok
        update_downstream_progress(
            s3, session_idx, total, exp_id,
            stage="stage3",
            status="done" if ok else "failed",
            completed=completed_count,
            failed=failed_count,
        )
        if not ok:
            error_records.append({
                "session": exp_id,
                "stage": "stage3",
                "error_type": "SubprocessError",
                "error_message": stderr,
                "traceback": "",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            })
    else:
        results["stage3"] = session.get("kinematics", False)

    # Stage 5: Sync (requires kinematics + calcium)
    if (force or not session.get("sync")) and results.get("stage3") and session.get("calcium"):
        ok, stderr = run_stage5(session, dry_run)
        results["stage5"] = ok
        update_downstream_progress(
            s3, session_idx, total, exp_id,
            stage="stage5",
            status="done" if ok else "failed",
            completed=completed_count,
            failed=failed_count,
        )
        if not ok:
            error_records.append({
                "session": exp_id,
                "stage": "stage5",
                "error_type": "SubprocessError",
                "error_message": stderr,
                "traceback": "",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            })
    else:
        results["stage5"] = session.get("sync", False)

    # Stage 6: Analysis (requires sync)
    if (force or not session.get("analysis")) and results.get("stage5"):
        ok, stderr = run_stage6(session, dry_run)
        results["stage6"] = ok
        update_downstream_progress(
            s3, session_idx, total, exp_id,
            stage="stage6",
            status="done" if ok else "failed",
            completed=completed_count,
            failed=failed_count,
        )
        if not ok:
            error_records.append({
                "session": exp_id,
                "stage": "stage6",
                "error_type": "SubprocessError",
                "error_message": stderr,
                "traceback": "",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            })
    else:
        results["stage6"] = session.get("analysis", False)

    return results


def _upload_error_json(s3, run_id: str, instance_id: str, error_records: list[dict]) -> None:
    """Upload _downstream_errors.json to S3.  Best-effort — logs warning on failure."""
    payload = json.dumps(
        {"run_id": run_id, "instance_id": instance_id, "errors": error_records},
        indent=2,
    ).encode()
    try:
        s3.put_object(
            Bucket=DERIVATIVES_BUCKET,
            Key=f"{RETRAIN_PREFIX}/_downstream_errors.json",
            Body=payload,
        )
        print(f"Downstream error summary uploaded ({len(error_records)} error(s))")
    except Exception as e:
        print(f"WARNING: could not upload _downstream_errors.json: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run downstream pipeline stages")
    parser.add_argument("--session", type=str, help="Process specific session (exp_id)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--force", action="store_true", help="Re-run all stages even if outputs exist")
    parser.add_argument("--watch", action="store_true", help="Poll for DLC completions")
    parser.add_argument("--watch-interval", type=int, default=300, help="Poll interval (seconds)")
    args = parser.parse_args()

    # Pre-flight: verify required stage scripts exist before processing sessions
    REQUIRED_SCRIPTS = [
        "scripts/run_stage3_kinematics.py",
        "scripts/run_stage5_sync.py",
        "scripts/run_stage6_analysis.py",
    ]
    missing = [s for s in REQUIRED_SCRIPTS if not Path(s).exists()]
    if missing:
        print(f"ERROR: required scripts not found: {missing}")
        print("Run from the repo root (cd /home/ubuntu/hm2p).")
        sys.exit(1)

    s3 = boto3.client("s3", region_name=REGION)
    sessions = get_sessions()

    run_id = datetime.datetime.utcnow().isoformat() + "Z"
    instance_id = _get_instance_id()
    error_records: list[dict] = []

    if args.session:
        sessions = [s for s in sessions if s["exp_id"] == args.session]
        if not sessions:
            print(f"Session {args.session} not found")
            sys.exit(1)

    if args.watch:
        print(f"Watching for DLC completions (polling every {args.watch_interval}s)...")
        processed = set()
        session_idx = 0
        watch_completed = 0
        while True:
            statuses = get_pipeline_status(s3, sessions)
            for status in statuses:
                exp_id = status["exp_id"]
                if exp_id in processed:
                    continue
                if status["pose"] and not status["sync"]:
                    session_idx += 1
                    print(f"\n=== Processing {exp_id} ===")
                    result = process_session(
                        status, s3,
                        session_idx=session_idx,
                        total=len(sessions),
                        completed_count=watch_completed,
                        failed_count=len(error_records),
                        error_records=error_records,
                        dry_run=args.dry_run,
                    )
                    processed.add(exp_id)
                    if result.get("stage3") and result.get("stage5") and result.get("stage6"):
                        watch_completed += 1
                    print(f"  Result: {result}")

            # Check if all done
            all_done = all(s["analysis"] or s["exp_id"] in processed for s in statuses)
            if all_done:
                print("\nAll sessions processed!")
                break

            n_pose = sum(1 for s in statuses if s["pose"])
            n_done = len(processed)
            print(f"\rDLC: {n_pose}/{len(sessions)} | Processed: {n_done}/{len(sessions)}", end="", flush=True)
            time.sleep(args.watch_interval)

        _upload_error_json(s3, run_id, instance_id, error_records)
    else:
        print("Checking pipeline status...")
        statuses = get_pipeline_status(s3, sessions)

        # Find sessions with work to do
        if args.force:
            pending = [s for s in statuses if s["pose"]]
        else:
            pending = [s for s in statuses if s["pose"] and not s["analysis"]]
        if not pending:
            print("No sessions need processing.")
            if args.dry_run:
                for s in statuses:
                    p = "Y" if s["pose"] else "N"
                    k = "Y" if s["kinematics"] else "N"
                    sy = "Y" if s["sync"] else "N"
                    a = "Y" if s["analysis"] else "N"
                    print(f"  {s['exp_id']}: pose={p} kin={k} sync={sy} analysis={a}")
            return

        print(f"\n{len(pending)} sessions to process{' (--force)' if args.force else ''}:")
        for s in pending:
            p = "Y" if s["pose"] else "N"
            k = "Y" if s["kinematics"] else "N"
            sy = "Y" if s["sync"] else "N"
            print(f"  {s['exp_id']}: pose={p} kin={k} sync={sy}")

        completed_count = 0
        for idx, s in enumerate(pending, 1):
            print(f"\n=== {s['exp_id']} ===")
            result = process_session(
                s, s3,
                session_idx=idx,
                total=len(pending),
                completed_count=completed_count,
                failed_count=len(error_records),
                error_records=error_records,
                dry_run=args.dry_run,
                force=args.force,
            )
            print(f"  Result: {result}")
            # A session is "completed" only if all three stages succeeded
            if result.get("stage3") and result.get("stage5") and result.get("stage6"):
                completed_count += 1

        _upload_error_json(s3, run_id, instance_id, error_records)


if __name__ == "__main__":
    main()
