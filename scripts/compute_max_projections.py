#!/usr/bin/env python3
"""Compute max projections from Suite2p data.bin for all sessions.

Downloads data.bin + ops.npy from S3, computes the max projection,
saves it as ops["max_proj"], and re-uploads ops.npy.

Usage:
    python scripts/compute_max_projections.py              # all sessions
    python scripts/compute_max_projections.py --session 0  # first session only
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

import boto3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REGION = "ap-southeast-2"
BUCKET = "hm2p-derivatives"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=int, default=None)
    args = parser.parse_args()

    csv_path = Path("metadata/experiments.csv")
    sessions = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            eid = row["exp_id"]
            parts = eid.split("_")
            sub = f"sub-{parts[-1]}"
            ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
            sessions.append((eid, sub, ses))

    if args.session is not None:
        sessions = [sessions[args.session]]

    s3 = boto3.client("s3", region_name=REGION)
    work = Path(tempfile.mkdtemp(prefix="hm2p-maxproj-"))

    for eid, sub, ses in sessions:
        prefix = f"ca_extraction/{sub}/{ses}/suite2p/plane0"
        ops_key = f"{prefix}/ops.npy"
        bin_key = f"{prefix}/data.bin"

        print(f"\n--- {sub}/{ses} ---")

        # Check if max_proj already exists
        try:
            ops_data = s3.get_object(Bucket=BUCKET, Key=ops_key)["Body"].read()
            ops = np.load(__import__("io").BytesIO(ops_data), allow_pickle=True).item()
            if "max_proj" in ops:
                print("  SKIP: max_proj already in ops")
                continue
        except Exception:
            print("  SKIP: no ops.npy")
            continue

        ly = ops.get("Ly", 0)
        lx = ops.get("Lx", 0)
        if ly == 0 or lx == 0:
            print("  SKIP: no Ly/Lx in ops")
            continue

        # Download data.bin
        local_bin = work / "data.bin"
        local_ops = work / "ops.npy"
        print(f"  Downloading data.bin ({ly}x{lx})...")
        try:
            s3.download_file(BUCKET, bin_key, str(local_bin))
        except Exception:
            print("  SKIP: no data.bin on S3")
            continue

        n_frames = local_bin.stat().st_size // (ly * lx * 2)
        print(f"  {n_frames} frames, computing max projection...")

        max_proj = None
        chunk_size = 1000
        with open(local_bin, "rb") as f:
            for start in range(0, n_frames, chunk_size):
                n_read = min(chunk_size, n_frames - start)
                chunk = np.fromfile(f, dtype=np.int16, count=n_read * ly * lx)
                chunk = chunk.reshape(n_read, ly, lx).astype(np.float32)
                chunk_max = chunk.max(axis=0)
                if max_proj is None:
                    max_proj = chunk_max
                else:
                    max_proj = np.maximum(max_proj, chunk_max)

        ops["max_proj"] = max_proj
        np.save(local_ops, ops)

        print(f"  Uploading ops.npy with max_proj...")
        s3.upload_file(str(local_ops), BUCKET, ops_key)
        print(f"  DONE")

        local_bin.unlink(missing_ok=True)
        local_ops.unlink(missing_ok=True)

    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
