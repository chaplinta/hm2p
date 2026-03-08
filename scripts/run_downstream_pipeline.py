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
import json
import sys
import time
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"


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


def run_stage3(session: dict, dry_run: bool = False) -> bool:
    """Run Stage 3 (kinematics) for a session."""
    import subprocess
    cmd = [
        sys.executable, "scripts/run_stage3_kinematics.py",
        "--session", session["exp_id"],
    ]
    print(f"  Stage 3 (kinematics): {' '.join(cmd)}")
    if dry_run:
        return True
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Stage 3 FAILED: {result.stderr[:500]}")
        return False
    print(f"  Stage 3 DONE")
    return True


def run_stage5(session: dict, dry_run: bool = False) -> bool:
    """Run Stage 5 (sync) for a session."""
    import subprocess
    cmd = [
        sys.executable, "scripts/run_stage5_sync.py",
        "--session", session["exp_id"],
    ]
    print(f"  Stage 5 (sync): {' '.join(cmd)}")
    if dry_run:
        return True
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Stage 5 FAILED: {result.stderr[:500]}")
        return False
    print(f"  Stage 5 DONE")
    return True


def run_stage6(session: dict, dry_run: bool = False) -> bool:
    """Run Stage 6 (analysis) for a session."""
    import subprocess
    cmd = [
        sys.executable, "scripts/run_stage6_analysis.py",
        "--session", session["exp_id"],
    ]
    print(f"  Stage 6 (analysis): {' '.join(cmd)}")
    if dry_run:
        return True
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Stage 6 FAILED: {result.stderr[:500]}")
        return False
    print(f"  Stage 6 DONE")
    return True


def process_session(session: dict, dry_run: bool = False) -> dict:
    """Run all pending stages for a session."""
    results = {"exp_id": session["exp_id"]}

    # Stage 3: Kinematics (requires pose)
    if not session.get("kinematics") and session.get("pose"):
        results["stage3"] = run_stage3(session, dry_run)
    else:
        results["stage3"] = session.get("kinematics", False)

    # Stage 5: Sync (requires kinematics + calcium)
    if not session.get("sync") and results.get("stage3") and session.get("calcium"):
        results["stage5"] = run_stage5(session, dry_run)
    else:
        results["stage5"] = session.get("sync", False)

    # Stage 6: Analysis (requires sync)
    if not session.get("analysis") and results.get("stage5"):
        results["stage6"] = run_stage6(session, dry_run)
    else:
        results["stage6"] = session.get("analysis", False)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run downstream pipeline stages")
    parser.add_argument("--session", type=str, help="Process specific session (exp_id)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--watch", action="store_true", help="Poll for DLC completions")
    parser.add_argument("--watch-interval", type=int, default=300, help="Poll interval (seconds)")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    sessions = get_sessions()

    if args.session:
        sessions = [s for s in sessions if s["exp_id"] == args.session]
        if not sessions:
            print(f"Session {args.session} not found")
            sys.exit(1)

    if args.watch:
        print(f"Watching for DLC completions (polling every {args.watch_interval}s)...")
        processed = set()
        while True:
            statuses = get_pipeline_status(s3, sessions)
            for status in statuses:
                exp_id = status["exp_id"]
                if exp_id in processed:
                    continue
                if status["pose"] and not status["sync"]:
                    print(f"\n=== Processing {exp_id} ===")
                    result = process_session(status, args.dry_run)
                    processed.add(exp_id)
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
    else:
        print("Checking pipeline status...")
        statuses = get_pipeline_status(s3, sessions)

        # Find sessions with work to do
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

        print(f"\n{len(pending)} sessions to process:")
        for s in pending:
            p = "Y" if s["pose"] else "N"
            k = "Y" if s["kinematics"] else "N"
            sy = "Y" if s["sync"] else "N"
            print(f"  {s['exp_id']}: pose={p} kin={k} sync={sy}")

        for s in pending:
            print(f"\n=== {s['exp_id']} ===")
            result = process_session(s, args.dry_run)
            print(f"  Result: {result}")


if __name__ == "__main__":
    main()
