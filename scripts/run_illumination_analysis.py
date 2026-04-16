#!/usr/bin/env python3
"""Compute mean pixel intensity timeseries for illumination check.

For each session, downloads the raw overhead video from S3, samples
mean pixel intensity every 100 frames (~1 sample/sec at 100fps), and
uploads the result as illumination.h5 to S3.

The frontend illumination check page loads this pre-computed data
instead of downloading full videos.

Usage:
    python scripts/run_illumination_analysis.py                  # all sessions
    python scripts/run_illumination_analysis.py --session EXP_ID # one session
    python scripts/run_illumination_analysis.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import tempfile
from pathlib import Path

import boto3
import cv2
import h5py
import numpy as np

REGION = "ap-southeast-2"
RAWDATA_BUCKET = "hm2p-rawdata"
DERIVATIVES_BUCKET = "hm2p-derivatives"
METADATA_PATH = Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"

SAMPLE_EVERY = 100  # Sample every 100th frame (~1 sample/sec at 100fps)


def get_sessions() -> list[dict]:
    sessions = []
    with open(METADATA_PATH) as f:
        for row in csv.DictReader(f):
            eid = row["exp_id"]
            parts = eid.split("_")
            sub = f"sub-{parts[-1]}"
            ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
            sessions.append({"exp_id": eid, "sub": sub, "ses": ses})
    return sessions


def find_video_key(s3, sub: str, ses: str) -> str | None:
    """Find overhead video on S3."""
    prefix = f"rawdata/{sub}/{ses}/behav/"
    resp = s3.list_objects_v2(Bucket=RAWDATA_BUCKET, Prefix=prefix, MaxKeys=20)
    for obj in resp.get("Contents", []):
        fn = obj["Key"].split("/")[-1].lower()
        if fn.endswith(".mp4") and "side" not in fn and "overhead" in fn:
            return obj["Key"]
    # Fallback: any mp4 that's not side
    for obj in resp.get("Contents", []):
        fn = obj["Key"].split("/")[-1].lower()
        if fn.endswith(".mp4") and "side" not in fn:
            return obj["Key"]
    return None


def load_light_epochs(s3, sub: str, ses: str) -> dict | None:
    """Load light on/off from sync.h5."""
    key = f"sync/{sub}/{ses}/sync.h5"
    try:
        obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key=key)
        data = obj["Body"].read()
    except Exception:
        return None

    with h5py.File(io.BytesIO(data), "r") as f:
        if "light_on" not in f or "frame_times" not in f:
            return None
        light_on = f["light_on"][()].astype(bool)
        frame_times = f["frame_times"][()].astype(float)

    transitions = np.diff(light_on.astype(np.int8))
    on_idx = np.where(transitions == 1)[0] + 1
    off_idx = np.where(transitions == -1)[0] + 1

    on_times = frame_times[on_idx] if len(on_idx) > 0 else np.array([], dtype=float)
    off_times = frame_times[off_idx] if len(off_idx) > 0 else np.array([], dtype=float)

    if light_on[0]:
        on_times = np.r_[frame_times[0], on_times]

    return {
        "light_on_times": on_times,
        "light_off_times": off_times,
        "sync_frame_times": frame_times,
    }


def sample_video_intensity(video_path: str, sample_every: int = SAMPLE_EVERY) -> dict:
    """Sample mean pixel intensity from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"frame_indices": np.array([]), "intensities": np.array([])}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 100.0

    indices = []
    intensities = []

    for fi in range(0, total_frames, sample_every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        intensities.append(float(gray.mean()))
        indices.append(fi)

    cap.release()

    return {
        "frame_indices": np.array(indices, dtype=np.int64),
        "intensities": np.array(intensities, dtype=np.float32),
        "total_frames": total_frames,
        "fps": fps,
        "sample_every": sample_every,
    }


def process_session(s3, sub: str, ses: str, exp_id: str, force: bool = False) -> bool:
    """Process one session: sample video intensity, save to S3."""
    out_key = f"analysis/{sub}/{ses}/illumination.h5"

    # Check if already done
    if not force:
        try:
            s3.head_object(Bucket=DERIVATIVES_BUCKET, Key=out_key)
            print(f"  Already exists, skipping (use --force to rerun)")
            return True
        except Exception:
            pass

    # Find video
    video_key = find_video_key(s3, sub, ses)
    if not video_key:
        print(f"  No overhead video found, skipping")
        return False

    # Load light epochs
    light = load_light_epochs(s3, sub, ses)
    if light is None:
        print(f"  No sync.h5 found, skipping")
        return False

    # Download video and sample
    print(f"  Downloading video...")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        s3.download_file(RAWDATA_BUCKET, video_key, tmp.name)
        print(f"  Sampling intensity every {SAMPLE_EVERY} frames...")
        result = sample_video_intensity(tmp.name)

    if len(result["intensities"]) == 0:
        print(f"  No frames sampled, skipping")
        return False

    # Compute frame times
    frame_times = result["frame_indices"].astype(np.float64) / result["fps"]

    # Classify each sample as light-on or light-off
    on_times = light["light_on_times"]
    off_times = light["light_off_times"]
    is_light_on = np.zeros(len(frame_times), dtype=bool)
    for i, t in enumerate(frame_times):
        for j in range(len(on_times)):
            t_off = off_times[j] if j < len(off_times) else frame_times[-1] + 1
            if on_times[j] <= t < t_off:
                is_light_on[i] = True
                break

    # Compute summary stats
    on_mask = is_light_on
    off_mask = ~is_light_on
    mean_on = float(np.mean(result["intensities"][on_mask])) if on_mask.any() else np.nan
    mean_off = float(np.mean(result["intensities"][off_mask])) if off_mask.any() else np.nan

    print(f"  {len(result['intensities'])} samples, "
          f"mean_on={mean_on:.1f}, mean_off={mean_off:.1f}, "
          f"diff={mean_on - mean_off:.2f}")

    # Write to HDF5 and upload
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as tmp_h5:
        with h5py.File(tmp_h5.name, "w") as f:
            f.create_dataset("frame_indices", data=result["frame_indices"])
            f.create_dataset("intensities", data=result["intensities"])
            f.create_dataset("frame_times", data=frame_times)
            f.create_dataset("is_light_on", data=is_light_on)
            f.create_dataset("light_on_times", data=on_times)
            f.create_dataset("light_off_times", data=off_times)
            f.attrs["mean_on"] = mean_on
            f.attrs["mean_off"] = mean_off
            f.attrs["diff"] = mean_on - mean_off
            f.attrs["n_samples"] = len(result["intensities"])
            f.attrs["sample_every"] = SAMPLE_EVERY
            f.attrs["video_fps"] = result["fps"]
            f.attrs["total_video_frames"] = result["total_frames"]
            f.attrs["exp_id"] = exp_id

        s3.upload_file(tmp_h5.name, DERIVATIVES_BUCKET, out_key)
    print(f"  Uploaded → s3://{DERIVATIVES_BUCKET}/{out_key}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Compute illumination analysis for all sessions")
    parser.add_argument("--session", type=str, help="Process single session (exp_id)")
    parser.add_argument("--force", action="store_true", help="Rerun even if output exists")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    sessions = get_sessions()

    if args.session:
        sessions = [s for s in sessions if s["exp_id"] == args.session]
        if not sessions:
            print(f"Session {args.session} not found")
            sys.exit(1)

    print(f"Processing {len(sessions)} sessions...")
    success = 0
    for i, ses_info in enumerate(sessions, 1):
        exp_id = ses_info["exp_id"]
        sub, ses = ses_info["sub"], ses_info["ses"]
        print(f"\n[{i}/{len(sessions)}] {exp_id} ({sub}/{ses})")
        if args.dry_run:
            print(f"  [DRY RUN]")
            continue
        if process_session(s3, sub, ses, exp_id, force=args.force):
            success += 1

    print(f"\nDone: {success}/{len(sessions)} sessions processed")


if __name__ == "__main__":
    main()
