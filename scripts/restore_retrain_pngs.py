#!/usr/bin/env python3
"""Restore retrain_frames PNGs from metadata JSONs + S3 videos.

Re-extracts all frames listed in metadata/retrain_frames/*.json and
re-creates symlinks into labeled-data/. Used to recover after
accidental deletion of retrain_frames/.

Usage:
    uv run python scripts/restore_retrain_pngs.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import boto3
import cv2

REGION = "ap-southeast-2"
RAWDATA_BUCKET = "hm2p-rawdata"
REPO_ROOT = Path(__file__).resolve().parent.parent
RETRAIN_DIR = REPO_ROOT / "retrain_frames"
RETRAIN_META_DIR = REPO_ROOT / "metadata" / "retrain_frames"
LABELED_DIR = (
    REPO_ROOT
    / "sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/labeled-data"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def find_labeled_data_dir(sub: str, ses: str) -> Path | None:
    if not LABELED_DIR.exists():
        return None
    ses_date = ses.replace("ses-", "").split("T")[0]
    animal = sub.replace("sub-", "")
    for ld in LABELED_DIR.iterdir():
        if ld.is_dir() and ses_date in ld.name and animal in ld.name:
            return ld
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

    for jf in sorted(RETRAIN_META_DIR.glob("*.json")):
        data = json.loads(jf.read_text())
        indices = data.get("frame_indices", [])
        session = data.get("session", "")
        tag = jf.stem

        if not indices:
            continue

        # Parse sub/ses from tag
        parts = tag.split("_ses-")
        if len(parts) != 2:
            log.warning("  Skipping %s (can't parse tag)", tag)
            continue
        sub = parts[0]
        ses = f"ses-{parts[1]}"

        retrain_session = RETRAIN_DIR / tag
        retrain_session.mkdir(parents=True, exist_ok=True)

        # Check which frames are missing
        missing = [i for i in indices if not (retrain_session / f"frame_{int(i):06d}.png").exists()]
        if not missing:
            log.info("  %s: all %d frames exist, skipping", tag, len(indices))
            continue

        log.info("  %s: restoring %d / %d frames", tag, len(missing), len(indices))

        # Download video
        with tempfile.TemporaryDirectory(prefix=f"hm2p-restore-{tag[:15]}-") as tmp:
            video_path = download_video(s3, sub, ses, Path(tmp))
            if video_path is None:
                log.warning("  No video for %s", tag)
                continue

            cap = cv2.VideoCapture(str(video_path))
            for idx in sorted(missing):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(str(retrain_session / f"frame_{int(idx):06d}.png"), frame)
                    total_restored += 1
            cap.release()

        # Re-create symlinks for ALL frames in this session (existing + restored)
        ld = find_labeled_data_dir(sub, ses)
        if ld is not None:
            linked = 0
            for idx in indices:
                png = retrain_session / f"frame_{int(idx):06d}.png"
                if not png.exists():
                    continue
                dest = ld / png.name
                if not dest.exists():
                    rel = os.path.relpath(png.resolve(), ld.resolve())
                    dest.symlink_to(rel)
                    linked += 1
            if linked:
                log.info("    Re-linked %d frames into labeled-data/", linked)

    print(f"\nRestored {total_restored} PNGs")


if __name__ == "__main__":
    main()
