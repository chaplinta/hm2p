#!/usr/bin/env python3
"""Build PCA thumbnail cache for frame selection.

For each session: reads video, downsamples every Nth frame to 64x64
grayscale, runs PCA (95% variance), saves result as .npz. The frame
selection script loads this cache instead of re-reading the video.

Usage:
    # Build cache for all sessions:
    uv run python scripts/build_pca_cache.py

    # One session:
    uv run python scripts/build_pca_cache.py --session 20210920

    # Primary only:
    uv run python scripts/build_pca_cache.py --primary-only

    # Force rebuild (ignore existing cache):
    uv run python scripts/build_pca_cache.py --force
"""

from __future__ import annotations

import argparse
import csv
import logging
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any

import boto3
import cv2
import numpy as np
from sklearn.decomposition import PCA

REGION = "ap-southeast-2"
RAWDATA_BUCKET = "hm2p-rawdata"
REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = REPO_ROOT / "metadata" / "experiments.csv"
CACHE_DIR = REPO_ROOT / "metadata" / "pca_cache"

THUMB_SIZE = 64
SAMPLE_STRIDE = 10
PCA_VARIANCE = 0.95

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


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
                "primary": row.get("primary_exp", "").lower() != "false",
                "exclude": row.get("exclude", "").lower() == "true",
            })
    return sessions


def cache_path(sub: str, ses: str) -> Path:
    return CACHE_DIR / f"{sub}_{ses}.npz"


def find_video_local(sub: str, ses: str) -> Path | None:
    rawdata = REPO_ROOT / "rawdata" / sub / ses / "behav"
    if rawdata.exists():
        for mp4 in rawdata.glob("*.mp4"):
            if "side" not in mp4.name.lower():
                return mp4
    return None


def download_video_from_s3(s3: Any, sub: str, ses: str, dest: Path) -> Path | None:
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


def build_cache(video_path: str, sub: str, ses: str) -> dict | None:
    """Read video, thumbnail, PCA, return cache dict."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning("  Cannot open video: %s", video_path)
        return None

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_indices = list(range(0, n_frames, SAMPLE_STRIDE))

    log.info("  Reading %d frames at %dx%d (stride=%d)...",
             len(all_indices), THUMB_SIZE, THUMB_SIZE, SAMPLE_STRIDE)

    thumbs = []
    valid_indices = []
    for idx in all_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (THUMB_SIZE, THUMB_SIZE),
                           interpolation=cv2.INTER_NEAREST)
        thumbs.append(small.astype(np.float64).ravel())
        valid_indices.append(idx)

    cap.release()

    if len(thumbs) < 10:
        log.warning("  Too few frames (%d)", len(thumbs))
        return None

    data = np.array(thumbs)
    valid_indices = np.array(valid_indices, dtype=np.int64)

    log.info("  PCA (%.0f%% variance)...", PCA_VARIANCE * 100)
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    data_pca = pca.fit_transform(data)
    log.info("  %d PCs explain %.1f%% variance",
             data_pca.shape[1], pca.explained_variance_ratio_.sum() * 100)

    # Save cache
    out = cache_path(sub, ses)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        data_pca=data_pca,
        frame_indices=valid_indices,
        n_frames=n_frames,
    )
    # Save PCA model separately (needed to project new frames)
    pca_path = out.with_suffix(".pca.pkl")
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)

    log.info("  Saved cache: %s (%.1f MB)",
             out.name, out.stat().st_size / 1e6)

    return {
        "data_pca": data_pca,
        "frame_indices": valid_indices,
        "n_frames": n_frames,
        "pca": pca,
    }


def load_cache(sub: str, ses: str) -> dict | None:
    """Load cached PCA data if it exists."""
    p = cache_path(sub, ses)
    pca_p = p.with_suffix(".pca.pkl")
    if not p.exists() or not pca_p.exists():
        return None
    npz = np.load(p)
    with open(pca_p, "rb") as f:
        pca = pickle.load(f)
    return {
        "data_pca": npz["data_pca"],
        "frame_indices": npz["frame_indices"],
        "n_frames": int(npz["n_frames"]),
        "pca": pca,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PCA thumbnail cache for frame selection."
    )
    parser.add_argument("--session", type=str, default=None,
                        help="Process only this session (partial match).")
    parser.add_argument("--primary-only", action="store_true",
                        help="Only process primary, non-excluded sessions.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild cache even if it exists.")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    sessions = get_sessions()

    if args.primary_only:
        sessions = [s for s in sessions if s["primary"] and not s["exclude"]]

    if args.session:
        sessions = [s for s in sessions if args.session in s["exp_id"]]
        if not sessions:
            print(f"No session matching '{args.session}'")
            sys.exit(1)

    built = 0
    skipped = 0
    for ses_info in sessions:
        sub, ses = ses_info["sub"], ses_info["ses"]
        exp_id = ses_info["exp_id"]

        if not args.force and cache_path(sub, ses).exists():
            log.info("  %s: cache exists, skipping", exp_id[:25])
            skipped += 1
            continue

        log.info("\n=== %s ===", exp_id)

        video_path = find_video_local(sub, ses)
        tmp_dir = None
        if video_path is None:
            tmp_dir = tempfile.mkdtemp(prefix=f"hm2p-pca-{exp_id[:15]}-")
            video_path = download_video_from_s3(s3, sub, ses, Path(tmp_dir))
            if video_path is None:
                log.warning("  No video for %s", exp_id)
                continue

        result = build_cache(str(video_path), sub, ses)
        if result is not None:
            built += 1

        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nBuilt {built} caches, skipped {skipped} existing")


if __name__ == "__main__":
    main()
