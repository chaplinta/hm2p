#!/usr/bin/env python3
"""Run CASCADE spike inference on all sessions.

Downloads ca.h5 from S3, runs CASCADE to infer calibrated spike rates,
and re-uploads the updated ca.h5 with the 'spikes' dataset added.

CASCADE requires Python 3.8 + TensorFlow 2.3 — run in the CASCADE
Docker container (docker/cascade.Dockerfile).

Usage:
    python scripts/run_cascade.py              # all sessions
    python scripts/run_cascade.py --session 0  # first session only
    python scripts/run_cascade.py --dry-run    # show what would be done
    python scripts/run_cascade.py --model Global_EXC_7.5Hz_smoothing200ms

Reference:
    Rupprecht et al. 2021. "A database and deep learning toolbox for
    noise-optimized, generalized spike inference from calcium imaging."
    Nature Neuroscience 24:1324-1337. doi:10.1038/s41593-021-00895-5
"""

from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
from pathlib import Path

import boto3
import h5py
import numpy as np

REGION = "ap-southeast-2"
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


def run_session(
    s3,
    sub: str,
    ses: str,
    exp_id: str,
    model_name: str,
    work_dir: Path,
    dry_run: bool = False,
) -> str:
    """Run CASCADE for a single session."""
    print(f"\n--- {sub}/{ses} ({exp_id}) ---")

    ca_key = f"calcium/{sub}/{ses}/ca.h5"

    # Check ca.h5 exists
    try:
        s3.head_object(Bucket=DERIVATIVES_BUCKET, Key=ca_key)
    except Exception:
        print(f"  SKIP: no ca.h5 at {ca_key}")
        return "skip_no_ca"

    if dry_run:
        print(f"  DRY RUN: would run CASCADE with model {model_name}")
        return "dry_run"

    session_dir = work_dir / sub / ses
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download ca.h5
        ca_local = session_dir / "ca.h5"
        print(f"  Downloading ca.h5...")
        s3.download_file(DERIVATIVES_BUCKET, ca_key, str(ca_local))

        # Read dF/F and fps
        with h5py.File(ca_local, "r") as f:
            dff = f["dff"][:]
            fps = float(f.attrs.get("fps_imaging", 9.8))

        n_rois, n_frames = dff.shape
        print(f"  ROIs: {n_rois}, Frames: {n_frames}, FPS: {fps:.1f}")

        # Run CASCADE
        print(f"  Running CASCADE (model: {model_name})...")
        from cascade2p import cascade

        spike_prob = cascade.predict(model_name, dff)
        spikes = np.asarray(spike_prob, dtype=np.float32)
        print(f"  Spike rates: mean={spikes.mean():.4f}, max={spikes.max():.4f} spikes/s")

        # Write spikes back to ca.h5
        with h5py.File(ca_local, "a") as f:
            if "spikes" in f:
                del f["spikes"]
            f.create_dataset("spikes", data=spikes, dtype=np.float32)

        # Re-upload
        print(f"  Uploading updated ca.h5...")
        s3.upload_file(str(ca_local), DERIVATIVES_BUCKET, ca_key)
        print(f"  DONE")

        return "ok"

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return f"error: {e}"

    finally:
        shutil.rmtree(session_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Run CASCADE spike inference")
    parser.add_argument("--session", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all", action="store_true", help="Process all sessions")
    parser.add_argument(
        "--model",
        default="Global_EXC_7.5Hz_smoothing200ms",
        help="CASCADE pre-trained model name",
    )
    args = parser.parse_args()

    sessions = get_sessions()
    print(f"Found {len(sessions)} sessions")
    print(f"Model: {args.model}")

    s3 = boto3.client("s3", region_name=REGION)
    work_dir = Path(tempfile.mkdtemp(prefix="hm2p-cascade-"))

    if args.session is not None:
        sessions = [sessions[args.session]]

    results = {}
    for ses in sessions:
        status = run_session(
            s3, ses["sub"], ses["ses"], ses["exp_id"],
            args.model, work_dir, dry_run=args.dry_run,
        )
        results[ses["exp_id"]] = status

    print(f"\n{'='*60}")
    print("CASCADE Summary:")
    ok = sum(1 for v in results.values() if v == "ok")
    skip = sum(1 for v in results.values() if v.startswith("skip"))
    err = sum(1 for v in results.values() if v.startswith("error"))
    print(f"  OK: {ok}, Skipped: {skip}, Errors: {err}")

    if err > 0:
        print("\nFailed sessions:")
        for exp_id, status in results.items():
            if status.startswith("error"):
                print(f"  {exp_id}: {status}")

    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
