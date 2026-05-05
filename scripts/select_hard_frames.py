#!/usr/bin/env python3
"""Select diverse frames for DLC retraining via PCA + k-means.

For each session:
1. Download video from S3
2. Sample frames, resize to 64x64 grayscale
3. PCA to 95% variance (strips static background)
4. K-means with k=50 classes
5. Check which clusters already have a labeled frame
6. For each empty cluster, extract the frame closest to centroid

Safety: existing CollectedData_*.csv/.h5 files are never modified.

Usage:
    uv run python scripts/select_hard_frames.py --scan
    uv run python scripts/select_hard_frames.py --session 20210920
    uv run python scripts/select_hard_frames.py --min-per-session 20
    uv run python scripts/select_hard_frames.py --primary-only --min-per-session 20
    uv run python scripts/select_hard_frames.py --k 80 --session 20210920
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import boto3
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
RAWDATA_BUCKET = "hm2p-rawdata"
REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = REPO_ROOT / "metadata" / "experiments.csv"
LABELED_DIR = (
    REPO_ROOT
    / "sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/labeled-data"
)
RETRAIN_META_DIR = REPO_ROOT / "metadata" / "retrain_frames"

THUMB_SIZE = 64  # resize all frames to 64x64 square
N_CLUSTERS = 50  # k-means classes
PCA_VARIANCE = 0.95  # keep PCs explaining 95% of variance
SAMPLE_STRIDE = 10  # sample every Nth frame for clustering (speed)

sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Labeled-data helpers
# ---------------------------------------------------------------------------


def find_labeled_data_dir(sub: str, ses: str) -> Path | None:
    if not LABELED_DIR.exists():
        return None
    ses_date = ses.replace("ses-", "").split("T")[0]
    animal = sub.replace("sub-", "")
    for ld in LABELED_DIR.iterdir():
        if ld.is_dir() and ses_date in ld.name and animal in ld.name:
            return ld
    return None


def get_existing_frame_indices(sub: str, ses: str) -> set[int]:
    """Get frame indices of existing PNGs in labeled-data/."""
    ld = find_labeled_data_dir(sub, ses)
    if ld is None:
        return set()
    indices = set()
    for png in ld.glob("frame_*.png"):
        m = re.match(r"frame_(\d+)\.png", png.name)
        if m:
            indices.add(int(m.group(1)))
    return indices


def count_existing_pngs(sub: str, ses: str) -> int:
    ld = find_labeled_data_dir(sub, ses)
    return len(list(ld.glob("frame_*.png"))) if ld else 0


def count_labeled(session_dir: Path) -> int:
    for h5 in session_dir.glob("CollectedData_*.h5"):
        try:
            df = pd.read_hdf(h5)
            return int((~df.isna().all(axis=1)).sum())
        except Exception:
            pass
    return 0


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------


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


def find_video_local(sub: str, ses: str) -> Path | None:
    rawdata = REPO_ROOT / "rawdata" / sub / ses / "behav"
    if rawdata.exists():
        for mp4 in rawdata.glob("*.mp4"):
            if "side" not in mp4.name.lower():
                return mp4
    return None


def extract_frames(video_path: Path, frame_indices: list[int], dest_dir: Path) -> list[int]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    new = [i for i in frame_indices if not (dest_dir / f"frame_{int(i):06d}.png").exists()]
    if not new:
        return []
    cap = cv2.VideoCapture(str(video_path))
    written = []
    for idx in sorted(new):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(dest_dir / f"frame_{int(idx):06d}.png"), frame)
            written.append(idx)
    cap.release()
    return written


def symlink_into_labeled_data(src_dir: Path, labeled_dir: Path) -> int:
    labeled_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for png in sorted(src_dir.glob("frame_*.png")):
        dest = labeled_dir / png.name
        if not dest.exists():
            rel = os.path.relpath(png.resolve(), labeled_dir.resolve())
            dest.symlink_to(rel)
            linked += 1
    return linked


def update_meta(session_tag: str, sub: str, ses: str, new_indices: list[int],
                video_name: str | None = None) -> None:
    RETRAIN_META_DIR.mkdir(parents=True, exist_ok=True)
    meta_file = RETRAIN_META_DIR / f"{session_tag}.json"
    existing: dict = {}
    if meta_file.exists():
        existing = json.loads(meta_file.read_text())
    merged = sorted(set(existing.get("frame_indices", [])) | set(int(i) for i in new_indices))
    updated = {
        "session": f"{sub}/{ses}",
        "frame_indices": [int(i) for i in merged],
        "n_frames": len(merged),
    }
    if video_name and "video" not in existing:
        updated["video"] = video_name
    elif "video" in existing:
        updated["video"] = existing["video"]
    meta_file.write_text(json.dumps(updated, indent=2))


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def scan_sessions() -> None:
    sessions = get_sessions()
    print(f"\n{'Session':<25s}  {'PNGs':>5s}  {'Lbl':>5s}  {'Status':<8s}  {'Flags'}")
    print("-" * 65)
    total_pngs = 0
    total_labeled = 0
    for s in sessions:
        ld = find_labeled_data_dir(s["sub"], s["ses"])
        pngs = len(list(ld.glob("frame_*.png"))) if ld else 0
        labeled = count_labeled(ld) if ld else 0
        total_pngs += pngs
        total_labeled += labeled
        status = "done" if pngs == labeled and pngs > 0 else ("partial" if pngs > 0 else "empty")
        flags = []
        if s["primary"]:
            flags.append("primary")
        if s["exclude"]:
            flags.append("excl")
        print(f"{s['exp_id'][:25]:<25s}  {pngs:>5d}  {labeled:>5d}  "
              f"{status:<8s}  {' '.join(flags)}")
    print(f"\nTotal: {total_pngs} PNGs, {total_labeled} labeled")


# ---------------------------------------------------------------------------
# Core: PCA + k-means frame selection
# ---------------------------------------------------------------------------


def read_thumbnails(
    video_path: str,
    frame_indices: list[int],
) -> tuple[np.ndarray, list[int]]:
    """Read frames from video, resize to THUMB_SIZE x THUMB_SIZE grayscale.

    Returns (data, valid_indices) where data is (n, THUMB_SIZE^2).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.empty((0, THUMB_SIZE * THUMB_SIZE)), []

    thumbs = []
    valid = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (THUMB_SIZE, THUMB_SIZE),
                           interpolation=cv2.INTER_NEAREST)
        thumbs.append(small.astype(np.float64).ravel())
        valid.append(idx)

    cap.release()
    if not thumbs:
        return np.empty((0, THUMB_SIZE * THUMB_SIZE)), []
    return np.array(thumbs), valid


def process_session(
    s3: Any,
    ses_info: dict,
    n_clusters: int,
    max_new: int | None,
    dry_run: bool,
) -> int:
    """PCA + k-means selection for one session."""
    sub, ses = ses_info["sub"], ses_info["ses"]
    exp_id = ses_info["exp_id"]
    tag = f"{sub}_{ses}"

    existing_indices = get_existing_frame_indices(sub, ses)
    log.info("  %s: %d existing frames", exp_id[:25], len(existing_indices))

    # Get video
    video_path = find_video_local(sub, ses)
    tmp_dir = None
    if video_path is None:
        tmp_dir = tempfile.mkdtemp(prefix=f"hm2p-sel-{exp_id[:15]}-")
        video_path = download_video_from_s3(s3, sub, ses, Path(tmp_dir))
        if video_path is None:
            log.warning("  No video for %s", exp_id)
            return 0

    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Sample frame indices (every SAMPLE_STRIDE frames)
    all_indices = list(range(0, n_frames, SAMPLE_STRIDE))
    log.info("  %d total frames, sampling %d (stride=%d)",
             n_frames, len(all_indices), SAMPLE_STRIDE)

    # Read thumbnails for sampled frames
    log.info("  Reading thumbnails at %dx%d...", THUMB_SIZE, THUMB_SIZE)
    data, valid_indices = read_thumbnails(str(video_path), all_indices)
    if len(data) < n_clusters:
        log.warning("  Too few frames (%d) for %d clusters", len(data), n_clusters)
        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return 0

    # PCA: keep components explaining 95% of variance
    log.info("  PCA (%.0f%% variance)...", PCA_VARIANCE * 100)
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    data_pca = pca.fit_transform(data)
    log.info("  %d PCs explain %.1f%% variance (from %d pixels)",
             data_pca.shape[1], pca.explained_variance_ratio_.sum() * 100,
             THUMB_SIZE * THUMB_SIZE)

    # K-means
    k = min(n_clusters, len(data_pca))
    log.info("  K-means clustering %d frames into %d clusters...",
             len(data_pca), k)
    kmeans = MiniBatchKMeans(
        n_clusters=k, batch_size=min(100, len(data_pca)),
        max_iter=50, n_init=3,
    )
    kmeans.fit(data_pca)

    # Map existing frames to their clusters.
    # Read thumbnails of existing frames and project into PCA space.
    existing_list = sorted(existing_indices)
    occupied_clusters: set[int] = set()
    if existing_list:
        ex_data, ex_valid = read_thumbnails(str(video_path), existing_list)
        if len(ex_data) > 0:
            ex_pca = pca.transform(ex_data)
            ex_labels = kmeans.predict(ex_pca)
            occupied_clusters = set(int(l) for l in ex_labels)
            log.info("  Existing frames occupy %d / %d clusters",
                     len(occupied_clusters), k)

    # Find empty clusters and pick closest frame to centroid
    selected = []
    for cluster_id in range(k):
        if cluster_id in occupied_clusters:
            continue
        member_mask = kmeans.labels_ == cluster_id
        if not member_mask.any():
            continue
        member_local = np.where(member_mask)[0]
        centre = kmeans.cluster_centers_[cluster_id]
        dists = np.linalg.norm(data_pca[member_local] - centre, axis=1)
        best = member_local[np.argmin(dists)]
        selected.append(valid_indices[best])

    # Apply max_new limit
    if max_new is not None and len(selected) > max_new:
        selected = selected[:max_new]

    log.info("  %d empty clusters → %d new frames to extract",
             k - len(occupied_clusters), len(selected))

    if not selected:
        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return 0

    if dry_run:
        log.info("  [DRY RUN] Would extract frames: %s", selected[:10])
        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return len(selected)

    # Extract full-res PNGs
    retrain_dir = REPO_ROOT / "retrain_frames" / tag
    written = extract_frames(video_path, selected, retrain_dir)
    log.info("  Extracted %d PNGs", len(written))

    # Symlink into labeled-data
    ld = find_labeled_data_dir(sub, ses)
    if ld is None:
        clip_name = f"{exp_id}_maze-rose_overhead.camera-cropped"
        ld = LABELED_DIR / clip_name
    n_linked = symlink_into_labeled_data(retrain_dir, ld)
    log.info("  Symlinked %d new frames into %s/", n_linked, ld.name)

    # Update metadata
    update_meta(tag, sub, ses, selected,
                video_name=video_path.name if video_path else None)

    if tmp_dir:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return len(written)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select diverse frames via PCA + k-means on 64x64 thumbnails."
    )
    parser.add_argument("--scan", action="store_true",
                        help="Show labeling status for all sessions.")
    parser.add_argument("--session", type=str, default=None,
                        help="Process only this session (partial match on exp_id).")
    parser.add_argument("--per-session", type=int, default=None,
                        help="Max new frames per session.")
    parser.add_argument("--min-per-session", type=int, default=None,
                        help="Ensure each session has at least this many total frames.")
    parser.add_argument("--total", type=int, default=None,
                        help="Max total new frames across all sessions.")
    parser.add_argument("--primary-only", action="store_true",
                        help="Only process primary, non-excluded sessions.")
    parser.add_argument("--k", type=int, default=N_CLUSTERS,
                        help=f"Number of k-means clusters (default {N_CLUSTERS}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be selected without extracting.")
    args = parser.parse_args()

    if args.scan:
        scan_sessions()
        return

    s3 = boto3.client("s3", region_name=REGION)
    sessions = get_sessions()

    if args.primary_only:
        sessions = [s for s in sessions if s["primary"] and not s["exclude"]]

    if args.session:
        sessions = [s for s in sessions if args.session in s["exp_id"]]
        if not sessions:
            print(f"No session matching '{args.session}'")
            sys.exit(1)

    total_new = 0
    for ses_info in sessions:
        sub, ses = ses_info["sub"], ses_info["ses"]
        existing_count = count_existing_pngs(sub, ses)

        max_new = args.per_session

        if args.min_per_session is not None:
            need = args.min_per_session - existing_count
            if need <= 0:
                log.info("  %s: already has %d frames (>= %d), skipping",
                         ses_info["exp_id"][:25], existing_count, args.min_per_session)
                continue
            if max_new is None:
                max_new = need
            else:
                max_new = min(max_new, need)

        if args.total is not None:
            remaining = args.total - total_new
            if remaining <= 0:
                break
            if max_new is None:
                max_new = remaining
            else:
                max_new = min(max_new, remaining)

        log.info("\n=== %s (have %d%s) ===",
                 ses_info["exp_id"], existing_count,
                 f", max {max_new} new" if max_new else "")

        n = process_session(s3, ses_info, args.k, max_new, args.dry_run)
        total_new += n

    print(f"\nTotal new frames extracted: {total_new}")
    if total_new > 0 and not args.dry_run:
        print(
            "\nNext steps:\n"
            "  1. Label:   uv run python scripts/interactive_label.py\n"
            "  2. Upload:  uv run python scripts/upload_dlc_labels.py\n"
            "  3. Retrain: uv run python scripts/launch_dlc_finetune_ec2.py --sa-finetune"
        )


if __name__ == "__main__":
    main()
