#!/usr/bin/env python3
"""Preview video thumbnails at different resolutions.

Extracts a handful of frames from a session video and saves them as
thumbnails at 30, 64, and 128 px width so you can see what k-means
sees at each resolution.

Usage:
    uv run python scripts/preview_thumbnails.py --session 20210920
    uv run python scripts/preview_thumbnails.py --session 20210920 --n 20
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import boto3
import cv2
import numpy as np

REGION = "ap-southeast-2"
RAWDATA_BUCKET = "hm2p-rawdata"
REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = REPO_ROOT / "metadata" / "experiments.csv"
OUT_DIR = REPO_ROOT / "tmp_thumbnails"

WIDTHS = [30, 64, 128]


def get_sessions() -> list[dict]:
    sessions = []
    with open(METADATA_PATH) as f:
        for row in csv.DictReader(f):
            eid = row["exp_id"]
            parts = eid.split("_")
            sessions.append({
                "exp_id": eid,
                "sub": f"sub-{parts[-1]}",
                "ses": f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}",
            })
    return sessions


def find_video_local(sub: str, ses: str) -> Path | None:
    rawdata = REPO_ROOT / "rawdata" / sub / ses / "behav"
    if rawdata.exists():
        for mp4 in rawdata.glob("*.mp4"):
            if "side" not in mp4.name.lower():
                return mp4
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
    parser = argparse.ArgumentParser(description="Preview thumbnails at different resolutions.")
    parser.add_argument("--session", required=True, help="Session exp_id (partial match).")
    parser.add_argument("--n", type=int, default=10, help="Number of frames to sample (default 10).")
    args = parser.parse_args()

    sessions = get_sessions()
    matches = [s for s in sessions if args.session in s["exp_id"]]
    if not matches:
        print(f"No session matching '{args.session}'")
        sys.exit(1)
    ses_info = matches[0]
    sub, ses = ses_info["sub"], ses_info["ses"]

    s3 = boto3.client("s3", region_name=REGION)

    video_path = find_video_local(sub, ses)
    tmp_dir = None
    if video_path is None:
        tmp_dir = tempfile.mkdtemp(prefix="hm2p-thumb-")
        video_path = download_video(s3, sub, ses, Path(tmp_dir))
        if video_path is None:
            print("No video found")
            sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {video_path.name}  {w}x{h}  {n_frames} frames")

    # Pick evenly spaced frames
    indices = np.linspace(0, n_frames - 1, args.n, dtype=int)

    OUT_DIR.mkdir(exist_ok=True)
    session_dir = OUT_DIR / ses_info["exp_id"][:25]
    session_dir.mkdir(exist_ok=True)

    for width in WIDTHS:
        d = session_dir / f"{width}px"
        d.mkdir(exist_ok=True)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for width in WIDTHS:
            ratio = width / w
            h_resized = max(1, int(h * ratio))
            small = cv2.resize(gray, (width, h_resized), interpolation=cv2.INTER_NEAREST)
            out_path = session_dir / f"{width}px" / f"frame_{int(idx):06d}.png"
            cv2.imwrite(str(out_path), small)

    cap.release()

    if tmp_dir:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nSaved to {session_dir}/")
    for width in WIDTHS:
        d = session_dir / f"{width}px"
        print(f"  {width}px: {len(list(d.glob('*.png')))} thumbnails")


if __name__ == "__main__":
    main()
