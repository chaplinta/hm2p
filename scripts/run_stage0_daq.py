#!/usr/bin/env python3
"""Run Stage 0 (DAQ parsing) for all sessions.

Downloads TDMS + config files from S3 rawdata, parses timing pulses,
and uploads timestamps.h5 to S3 derivatives.

Usage:
    python scripts/run_stage0_daq.py              # all sessions
    python scripts/run_stage0_daq.py --session 0   # first session only
    python scripts/run_stage0_daq.py --dry-run     # show what would be done
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import boto3

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REGION = "ap-southeast-2"
RAWDATA_BUCKET = "hm2p-rawdata"
DERIVATIVES_BUCKET = "hm2p-derivatives"


def get_sessions() -> list[dict]:
    """Read session list from metadata/experiments.csv."""
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
            sessions.append({"exp_id": exp_id, "sub": sub, "ses": ses})
    return sessions


def find_tdms_files(s3, sub: str, ses: str) -> dict[str, str]:
    """Find the TDMS, meta.txt, and .ini files for a session on S3."""
    prefix = f"rawdata/{sub}/{ses}/"
    resp = s3.list_objects_v2(Bucket=RAWDATA_BUCKET, Prefix=prefix)
    files = {obj["Key"].split("/")[-1]: obj["Key"] for obj in resp.get("Contents", [])}

    tdms = None
    meta = None
    ini = None

    for name, key in files.items():
        if name.endswith("-di.tdms"):
            tdms = key
        elif name.endswith(".meta.txt") and "maze" in name.lower():
            meta = key
        elif name.endswith("_XYT.ini"):
            ini = key

    return {"tdms": tdms, "meta": meta, "ini": ini}


def download_file(s3, bucket: str, key: str, local_path: Path) -> None:
    """Download a file from S3."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(local_path))


def run_session(s3, sub: str, ses: str, exp_id: str, work_dir: Path, dry_run: bool = False) -> str:
    """Run Stage 0 for a single session. Returns status string."""
    print(f"\n--- {sub}/{ses} ({exp_id}) ---")

    # Find files on S3
    files = find_tdms_files(s3, sub, ses)
    if not files["tdms"]:
        print(f"  SKIP: no TDMS file found")
        return "skip_no_tdms"
    if not files["meta"]:
        print(f"  SKIP: no .meta.txt found")
        return "skip_no_meta"
    if not files["ini"]:
        print(f"  SKIP: no .ini found")
        return "skip_no_ini"

    print(f"  TDMS: {files['tdms'].split('/')[-1]}")
    print(f"  META: {files['meta'].split('/')[-1]}")
    print(f"  INI:  {files['ini'].split('/')[-1]}")

    if dry_run:
        print(f"  DRY RUN: would parse and upload timestamps.h5")
        return "dry_run"

    # Download files
    session_dir = work_dir / sub / ses
    session_dir.mkdir(parents=True, exist_ok=True)

    tdms_local = session_dir / files["tdms"].split("/")[-1]
    meta_local = session_dir / files["meta"].split("/")[-1]
    ini_local = session_dir / files["ini"].split("/")[-1]

    print(f"  Downloading TDMS ({files['tdms'].split('/')[-1]})...")
    download_file(s3, RAWDATA_BUCKET, files["tdms"], tdms_local)
    download_file(s3, RAWDATA_BUCKET, files["meta"], meta_local)
    download_file(s3, RAWDATA_BUCKET, files["ini"], ini_local)

    # Also download the TDMS index if it exists
    try:
        s3.download_file(RAWDATA_BUCKET, files["tdms"] + "_index",
                         str(tdms_local) + "_index")
    except Exception:
        pass

    # Parse TDMS
    print(f"  Parsing TDMS...")
    try:
        from hm2p.ingest.daq import parse_tdms, write_timestamps_h5

        arrays = parse_tdms(tdms_local)

        n_cam = len(arrays["frame_times_camera"])
        n_img = len(arrays["frame_times_imaging"])
        n_light_on = len(arrays["light_on_times"])
        print(f"  Camera frames: {n_cam}")
        print(f"  Imaging frames: {n_img}")
        print(f"  Light on pulses: {n_light_on}")
        print(f"  Camera FPS: {arrays['fps_camera']:.1f}")
        print(f"  Imaging FPS: {arrays['fps_imaging']:.1f}")
        print(f"  Duration: {arrays['frame_times_camera'][-1]:.1f}s")

        # Write timestamps.h5
        output_path = session_dir / "timestamps.h5"
        session_id = f"{sub}/{ses}"
        write_timestamps_h5(arrays, session_id, output_path)
        print(f"  Wrote {output_path}")

        # Upload to S3
        s3_key = f"movement/{sub}/{ses}/timestamps.h5"
        print(f"  Uploading to s3://{DERIVATIVES_BUCKET}/{s3_key}")
        s3.upload_file(str(output_path), DERIVATIVES_BUCKET, s3_key)
        print(f"  DONE")

        return "ok"

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return f"error: {e}"

    finally:
        # Cleanup
        shutil.rmtree(session_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Run Stage 0 DAQ parsing")
    parser.add_argument("--session", type=int, default=None,
                        help="Process only this session index (0-based)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without processing")
    args = parser.parse_args()

    sessions = get_sessions()
    print(f"Found {len(sessions)} sessions")

    s3 = boto3.client("s3", region_name=REGION)
    work_dir = Path(tempfile.mkdtemp(prefix="hm2p-stage0-"))
    print(f"Work dir: {work_dir}")

    if args.session is not None:
        sessions = [sessions[args.session]]

    results = {}
    for i, ses in enumerate(sessions):
        status = run_session(s3, ses["sub"], ses["ses"], ses["exp_id"],
                             work_dir, dry_run=args.dry_run)
        results[ses["exp_id"]] = status

    # Summary
    print(f"\n{'='*60}")
    print(f"Stage 0 Summary:")
    ok = sum(1 for v in results.values() if v == "ok")
    skip = sum(1 for v in results.values() if v.startswith("skip"))
    err = sum(1 for v in results.values() if v.startswith("error"))
    print(f"  OK: {ok}, Skipped: {skip}, Errors: {err}")

    if err > 0:
        print(f"\nFailed sessions:")
        for exp_id, status in results.items():
            if status.startswith("error"):
                print(f"  {exp_id}: {status}")

    # Cleanup
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
