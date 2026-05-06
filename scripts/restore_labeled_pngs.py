#!/usr/bin/env python3
"""Restore PNGs for all labeled frames based on CollectedData H5 files.

The source of truth is CollectedData_*.h5 — it lists every labeled
frame. This script ensures every labeled frame has a corresponding
PNG, extracting from S3 videos where needed.

Does NOT touch CollectedData files. Only extracts PNGs and symlinks.

Usage:
    uv run python scripts/restore_labeled_pngs.py
"""

from __future__ import annotations

import csv
import logging
import os
import re
import sys
import tempfile
from pathlib import Path

import boto3
import cv2
import pandas as pd

REGION = "ap-southeast-2"
RAWDATA_BUCKET = "hm2p-rawdata"
REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = REPO_ROOT / "metadata" / "experiments.csv"
RETRAIN_DIR = REPO_ROOT / "retrain_frames"
LABELED_DIR = (
    REPO_ROOT
    / "sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/labeled-data"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _session_to_sub_ses(session_dir_name: str) -> tuple[str, str] | None:
    """Parse sub/ses from labeled-data directory name."""
    # e.g. 20210823_17_00_04_1114353_maze-rose_overhead.camera-cropped
    parts = session_dir_name.split("_")
    if len(parts) < 5:
        return None
    date = parts[0]
    hh, mm, ss = parts[1], parts[2], parts[3]
    animal = parts[4].split("-")[0]

    # Find matching session in experiments.csv
    with open(METADATA_PATH) as f:
        for row in csv.DictReader(f):
            eid = row["exp_id"]
            if animal in eid and date in eid:
                ep = eid.split("_")
                return f"sub-{ep[-1]}", f"ses-{ep[0]}T{ep[1]}{ep[2]}{ep[3]}"
    return None


def download_video(s3, sub: str, ses: str, dest: Path) -> Path | None:
    prefix = f"rawdata/{sub}/{ses}/behav/"
    resp = s3.list_objects_v2(Bucket=RAWDATA_BUCKET, Prefix=prefix)
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        fname = key.split("/")[-1]
        if not fname.endswith(".mp4") or "side" in fname.lower():
            continue
        local = dest / fname
        if not local.exists():
            s3.download_file(RAWDATA_BUCKET, key, str(local))
        return local
    return None


def main() -> None:
    s3 = boto3.client("s3", region_name=REGION)
    total_restored = 0
    total_linked = 0

    for session_dir in sorted(LABELED_DIR.iterdir()):
        if not session_dir.is_dir():
            continue

        h5s = list(session_dir.glob("CollectedData_*.h5"))
        if not h5s:
            continue

        df = pd.read_hdf(h5s[0])
        if len(df) == 0:
            continue

        # Get all frame filenames and their indices from CollectedData
        needed_frames: dict[str, int] = {}  # filename -> frame index
        for idx in df.index:
            fname = idx[-1] if isinstance(idx, tuple) else str(idx).split("/")[-1]
            fname = Path(fname).name
            m = re.match(r"frame_(\d+)\.png", fname)
            if m:
                needed_frames[fname] = int(m.group(1))

        # Check which are missing
        existing = set(p.name for p in session_dir.glob("frame_*.png") if p.exists())
        missing = {f: i for f, i in needed_frames.items() if f not in existing}

        if not missing:
            continue

        log.info("%s: %d labeled, %d missing PNGs",
                 session_dir.name[:50], len(needed_frames), len(missing))

        # Parse sub/ses
        sub_ses = _session_to_sub_ses(session_dir.name)
        if sub_ses is None:
            log.warning("  Can't parse sub/ses from %s", session_dir.name)
            continue
        sub, ses = sub_ses

        # Ensure retrain_frames dir exists
        tag = f"{sub}_{ses}"
        retrain_session = RETRAIN_DIR / tag
        retrain_session.mkdir(parents=True, exist_ok=True)

        # Check which PNGs are in retrain_frames already
        still_missing = {}
        for fname, idx in missing.items():
            src = retrain_session / fname
            if src.exists():
                # PNG exists in retrain_frames, just needs symlink
                dest = session_dir / fname
                if not dest.exists():
                    rel = os.path.relpath(src.resolve(), session_dir.resolve())
                    dest.symlink_to(rel)
                    total_linked += 1
            else:
                still_missing[fname] = idx

        if not still_missing:
            log.info("  All restored from existing retrain_frames/")
            continue

        # Download video and extract missing frames
        log.info("  Extracting %d frames from video...", len(still_missing))
        with tempfile.TemporaryDirectory(prefix=f"hm2p-restore-{tag[:15]}-") as tmp:
            video_path = download_video(s3, sub, ses, Path(tmp))
            if video_path is None:
                log.warning("  No video for %s", tag)
                continue

            cap = cv2.VideoCapture(str(video_path))
            for fname, idx in sorted(still_missing.items(), key=lambda x: x[1]):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Save to retrain_frames
                    out = retrain_session / fname
                    cv2.imwrite(str(out), frame)
                    # Symlink into labeled-data
                    dest = session_dir / fname
                    if not dest.exists():
                        rel = os.path.relpath(out.resolve(), session_dir.resolve())
                        dest.symlink_to(rel)
                    total_restored += 1
                    total_linked += 1
                else:
                    log.warning("  Failed to read frame %d", idx)
            cap.release()

    print(f"\nRestored {total_restored} PNGs, re-linked {total_linked}")


if __name__ == "__main__":
    main()
