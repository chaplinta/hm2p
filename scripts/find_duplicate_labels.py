#!/usr/bin/env python3
"""Find duplicate labeled frames using PCA + k-means clustering.

For each session:
1. Load PCA cache (~1800 sampled frames already in PCA space) and the
   saved PCA model.
2. Run k-means with k=--clusters (default 100) on the full cached frames
   to define an appearance space.
3. Find all labeled-frame PNGs in labeled-data for that session.
4. Download the video from S3, extract grayscale thumbnails for just
   those labeled frame indices, matching the PCA model's input dimension.
5. Project labeled thumbnails into PCA space using the saved PCA model.
6. Assign each labeled frame to its nearest cluster centroid.
7. Any cluster with >1 labeled frame contains visually similar frames.
8. For each duplicate pair, extract full-resolution frames from the video,
   stitch side-by-side horizontally, and save to
   retrain_frames/_duplicates/{session}_{frame1}_{frame2}.png.
9. Print per-session summary.

Usage:
    uv run python scripts/find_duplicate_labels.py --clusters 100
    uv run python scripts/find_duplicate_labels.py --clusters 100 \\
        --session 20210823_16_59_50_1114353
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import pickle
import re
import shutil
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any

import boto3
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REGION = "ap-southeast-2"
RAWDATA_BUCKET = "hm2p-rawdata"
REPO_ROOT = Path(__file__).resolve().parent.parent
PCA_CACHE_DIR = REPO_ROOT / "metadata" / "pca_cache"
LABELED_DIR = REPO_ROOT / "sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/labeled-data"
OUTPUT_DIR = REPO_ROOT / "retrain_frames" / "_duplicates"
METADATA_PATH = REPO_ROOT / "metadata" / "experiments.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------


def get_sessions() -> list[dict[str, str]]:
    """Return all sessions from experiments.csv with sub/ses identifiers.

    Returns
    -------
    list[dict[str, str]]
        Each dict has keys ``exp_id``, ``sub``, ``ses``.
    """
    sessions: list[dict[str, str]] = []
    with open(METADATA_PATH) as f:
        for row in csv.DictReader(f):
            eid: str = row["exp_id"]
            parts = eid.split("_")
            # parts: [date, HH, MM, SS, animal_id]
            sub = f"sub-{parts[-1]}"
            ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
            sessions.append({"exp_id": eid, "sub": sub, "ses": ses})
    return sessions


# ---------------------------------------------------------------------------
# PCA cache loading
# ---------------------------------------------------------------------------


def load_pca_cache(sub: str, ses: str) -> dict[str, Any] | None:
    """Load pre-built PCA cache and fitted PCA model for a session.

    Parameters
    ----------
    sub : str
        Subject identifier (e.g. ``"sub-1114353"``).
    ses : str
        Session identifier (e.g. ``"ses-20210823T165950"``).

    Returns
    -------
    dict or None
        Dict with ``data_pca`` (N, D), ``frame_indices`` (N,),
        ``n_frames`` (int), and ``pca`` (sklearn PCA model).
        None if cache file does not exist.
    """
    cache_file = PCA_CACHE_DIR / f"{sub}_{ses}.npz"
    pca_file = cache_file.with_suffix(".pca.pkl")
    if not cache_file.exists():
        log.warning("  PCA cache not found: %s", cache_file)
        return None
    if not pca_file.exists():
        log.warning("  PCA model not found: %s", pca_file)
        return None

    npz = np.load(cache_file)
    with open(pca_file, "rb") as f:
        pca_model = pickle.load(f)

    return {
        "data_pca": npz["data_pca"],
        "frame_indices": npz["frame_indices"],
        "n_frames": int(npz["n_frames"]),
        "pca": pca_model,
    }


# ---------------------------------------------------------------------------
# Labeled-data discovery
# ---------------------------------------------------------------------------


def clip_name_to_sub_ses(clip_name: str) -> tuple[str, str] | None:
    """Parse a labeled-data clip directory name into (sub, ses).

    Parameters
    ----------
    clip_name : str
        Directory name, e.g.
        ``"20210823_17_00_04_1114353_maze-rose_overhead.camera-cropped"``.

    Returns
    -------
    tuple[str, str] or None
        ``(sub, ses)`` e.g. ``("sub-1114353", "ses-20210823T170004")``,
        or None if the name cannot be parsed.
    """
    m = re.match(
        r"^(\d{8})_(\d{2})_(\d{2})_(\d{2})_(\d+)_",
        clip_name,
    )
    if m is None:
        return None
    date, hh, mm, ss, animal = m.groups()
    return f"sub-{animal}", f"ses-{date}T{hh}{mm}{ss}"


def find_labeled_clip_dir(sub: str, ses: str) -> Path | None:
    """Find the labeled-data clip directory for a session.

    Matching is by date + animal, since the clip timestamp can differ
    slightly from the experiment timestamp in experiments.csv.

    Parameters
    ----------
    sub : str
        Subject identifier (e.g. ``"sub-1114353"``).
    ses : str
        Session identifier (e.g. ``"ses-20210823T165950"``).

    Returns
    -------
    Path or None
        Path to the matching labeled-data subdirectory, or None.
    """
    if not LABELED_DIR.exists():
        return None

    ses_date = ses.replace("ses-", "").split("T")[0]
    animal = sub.replace("sub-", "")

    for ld in LABELED_DIR.iterdir():
        if not ld.is_dir():
            continue
        if ses_date in ld.name and animal in ld.name:
            return ld
    return None


def get_labeled_frame_indices(clip_dir: Path) -> list[int]:
    """Extract sorted frame indices from PNG filenames in a clip directory.

    Parameters
    ----------
    clip_dir : Path
        Directory containing ``frame_NNNNNN.png`` files.

    Returns
    -------
    list[int]
        Sorted frame indices.
    """
    indices: list[int] = []
    for png in clip_dir.glob("frame_*.png"):
        m = re.match(r"frame_(\d+)\.png", png.name)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


# ---------------------------------------------------------------------------
# Video download from S3
# ---------------------------------------------------------------------------


def download_video_from_s3(
    s3: Any,
    sub: str,
    ses: str,
    dest_dir: Path,
) -> Path | None:
    """Download the overhead video for a session from S3.

    Parameters
    ----------
    s3 :
        boto3 S3 client.
    sub : str
        Subject identifier.
    ses : str
        Session identifier.
    dest_dir : Path
        Directory to save the video file.

    Returns
    -------
    Path or None
        Local path to the downloaded video, or None if not found.
    """
    prefix = f"rawdata/{sub}/{ses}/behav/"
    resp = s3.list_objects_v2(Bucket=RAWDATA_BUCKET, Prefix=prefix, MaxKeys=20)
    contents = resp.get("Contents", [])

    overhead_key: str | None = None
    fallback_key: str | None = None
    for obj in contents:
        fn = obj["Key"].split("/")[-1]
        if not fn.endswith(".mp4"):
            continue
        if "side" in fn.lower():
            continue
        if "overhead" in fn.lower() or "cropped" in fn.lower():
            overhead_key = obj["Key"]
        elif fallback_key is None:
            fallback_key = obj["Key"]

    key = overhead_key or fallback_key
    if key is None:
        return None

    fn = key.split("/")[-1]
    local_path = dest_dir / fn
    if not local_path.exists():
        log.info("    Downloading %s ...", fn)
        s3.download_file(RAWDATA_BUCKET, key, str(local_path))
        log.info("    Download complete.")
    else:
        log.info("    Video already cached: %s", fn)

    return local_path


# ---------------------------------------------------------------------------
# Thumbnail extraction for specific frames
# ---------------------------------------------------------------------------


def extract_thumbnails_for_frames(
    video_path: Path,
    frame_indices: list[int],
    thumb_size: int,
) -> tuple[np.ndarray, list[int]]:
    """Extract grayscale thumbnails for specific frame indices.

    Parameters
    ----------
    video_path : Path
        Path to the video file.
    frame_indices : list[int]
        Frame indices to extract.
    thumb_size : int
        Thumbnail edge length in pixels (square).

    Returns
    -------
    tuple[np.ndarray, list[int]]
        ``(features, valid_indices)`` where ``features`` has shape
        ``(M, thumb_size * thumb_size)`` (float64) and
        ``valid_indices`` lists the frame indices that were successfully
        extracted (may be shorter than input if some frames are missing).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    features: list[np.ndarray] = []
    valid_indices: list[int] = []

    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            log.warning("    Could not read frame %d from video.", idx)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (thumb_size, thumb_size), interpolation=cv2.INTER_NEAREST)
        vec = thumb.astype(np.float64).ravel()
        features.append(vec)
        valid_indices.append(idx)

    cap.release()

    if not features:
        empty = np.empty((0, thumb_size * thumb_size), dtype=np.float64)
        return empty, []

    return np.stack(features, axis=0), valid_indices


# ---------------------------------------------------------------------------
# Full-resolution frame extraction for side-by-side output
# ---------------------------------------------------------------------------


def extract_full_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    """Extract a single full-resolution frame from a video.

    Parameters
    ----------
    video_path : Path
        Path to the video file.
    frame_idx : int
        Frame index to extract.

    Returns
    -------
    np.ndarray or None
        BGR frame array, or None if the frame could not be read.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------


def process_session(
    s3: Any,
    session_info: dict[str, str],
    n_clusters: int,
    tmp_dir: Path,
) -> dict[str, Any]:
    """Find duplicate labeled frames for one session.

    Parameters
    ----------
    s3 :
        boto3 S3 client.
    session_info : dict
        Must have ``exp_id``, ``sub``, ``ses``.
    n_clusters : int
        Number of k-means clusters.
    tmp_dir : Path
        Temporary directory for video downloads.

    Returns
    -------
    dict
        Summary with ``exp_id``, ``n_labeled``, ``n_duplicate_pairs``,
        ``duplicate_groups`` (list of lists of frame indices).
    """
    sub = session_info["sub"]
    ses = session_info["ses"]
    exp_id = session_info["exp_id"]

    result: dict[str, Any] = {
        "exp_id": exp_id,
        "n_labeled": 0,
        "n_duplicate_pairs": 0,
        "duplicate_groups": [],
    }

    # Step 1: Load PCA cache
    pca_cache = load_pca_cache(sub, ses)
    if pca_cache is None:
        log.warning("  Skipping %s — no PCA cache.", exp_id)
        return result

    data_pca = pca_cache["data_pca"]
    pca_model = pca_cache["pca"]

    # Determine thumbnail size from PCA model
    thumb_size = int(math.sqrt(pca_model.n_features_in_))
    log.info(
        "  PCA cache: %d frames, %d PCs, thumbnail size %dx%d",
        data_pca.shape[0],
        data_pca.shape[1],
        thumb_size,
        thumb_size,
    )

    # Step 2: k-means on full cached frames
    from sklearn.cluster import KMeans

    k = min(n_clusters, data_pca.shape[0])
    if k < 2:
        log.warning("  Too few cached frames for clustering, skipping.")
        return result

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(data_pca)

    # Step 3: Find labeled PNGs for this session
    clip_dir = find_labeled_clip_dir(sub, ses)
    if clip_dir is None:
        log.warning("  No labeled-data directory found for %s.", exp_id)
        return result

    labeled_indices = get_labeled_frame_indices(clip_dir)
    result["n_labeled"] = len(labeled_indices)

    if len(labeled_indices) < 2:
        log.info("  Only %d labeled frames — nothing to compare.", len(labeled_indices))
        return result

    log.info("  %d labeled frames in %s", len(labeled_indices), clip_dir.name)

    # Step 4: Download video, extract thumbnails for labeled frames
    video_path = download_video_from_s3(s3, sub, ses, tmp_dir)
    if video_path is None:
        log.warning("  No video on S3 for %s, skipping.", exp_id)
        return result

    features, valid_indices = extract_thumbnails_for_frames(
        video_path, labeled_indices, thumb_size
    )
    if len(valid_indices) < 2:
        log.info("  Could not extract enough labeled thumbnails.")
        return result

    # Step 5: Project labeled thumbnails into PCA space
    labeled_pca = pca_model.transform(features)

    # Step 6: Assign each labeled frame to nearest cluster centroid
    # Use km.predict on the PCA-projected labeled frames
    cluster_assignments = km.predict(labeled_pca)

    # Step 7: Find clusters with >1 labeled frame
    cluster_to_frames: dict[int, list[int]] = {}
    for frame_idx, cluster_id in zip(valid_indices, cluster_assignments, strict=True):
        cluster_to_frames.setdefault(int(cluster_id), []).append(frame_idx)

    duplicate_groups: list[list[int]] = []
    for _cluster_id, frames in sorted(cluster_to_frames.items()):
        if len(frames) > 1:
            duplicate_groups.append(sorted(frames))

    if not duplicate_groups:
        log.info("  No duplicate clusters found.")
        return result

    # Count pairs
    n_pairs = sum(len(list(combinations(group, 2))) for group in duplicate_groups)
    result["n_duplicate_pairs"] = n_pairs
    result["duplicate_groups"] = duplicate_groups

    log.info(
        "  Found %d duplicate groups (%d pairs) across %d clusters",
        len(duplicate_groups),
        n_pairs,
        k,
    )

    # Step 8: For each duplicate pair, stitch side-by-side and save
    session_tag = f"{sub}_{ses}"
    for group in duplicate_groups:
        for f1, f2 in combinations(group, 2):
            img1 = extract_full_frame(video_path, f1)
            img2 = extract_full_frame(video_path, f2)
            if img1 is None or img2 is None:
                log.warning("    Could not extract frames %d/%d for stitching.", f1, f2)
                continue

            # Match heights if they differ (shouldn't, but be safe)
            h1, h2 = img1.shape[0], img2.shape[0]
            if h1 != h2:
                target_h = max(h1, h2)
                if h1 < target_h:
                    scale = target_h / h1
                    img1 = cv2.resize(
                        img1,
                        (int(img1.shape[1] * scale), target_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    scale = target_h / h2
                    img2 = cv2.resize(
                        img2,
                        (int(img2.shape[1] * scale), target_h),
                        interpolation=cv2.INTER_LINEAR,
                    )

            stitched = np.concatenate([img1, img2], axis=1)
            out_path = OUTPUT_DIR / f"{session_tag}_{f1:06d}_{f2:06d}.png"
            cv2.imwrite(str(out_path), stitched)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find duplicate labeled frames via PCA + k-means clustering."
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=100,
        help="Number of k-means clusters (default: 100).",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Process a single session by exp_id.",
    )
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    all_sessions = get_sessions()

    if args.session:
        all_sessions = [s for s in all_sessions if s["exp_id"] == args.session]
        if not all_sessions:
            log.error("Session %r not found in experiments.csv.", args.session)
            raise SystemExit(1)

    # Clear output directory at the start of each run
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 62}")
    print("  Duplicate label detection (PCA + k-means)")
    print(f"  Sessions: {len(all_sessions)}   Clusters: {args.clusters}")
    print(f"{'=' * 62}\n")

    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="hm2p-dup-") as tmp_str:
        tmp_dir = Path(tmp_str)

        for i, sess in enumerate(all_sessions, 1):
            log.info("[%d/%d] %s", i, len(all_sessions), sess["exp_id"])
            result = process_session(
                s3=s3,
                session_info=sess,
                n_clusters=args.clusters,
                tmp_dir=tmp_dir,
            )
            results.append(result)

            n_labeled = result["n_labeled"]
            n_pairs = result["n_duplicate_pairs"]
            groups = result["duplicate_groups"]

            print(f"  {sess['exp_id']}:  n_labeled={n_labeled}  n_duplicate_pairs={n_pairs}")
            if groups:
                for group in groups:
                    frames_str = ", ".join(str(f) for f in group)
                    print(f"    cluster group: [{frames_str}]")
            print()

    # Summary
    total_labeled = sum(r["n_labeled"] for r in results)
    total_pairs = sum(r["n_duplicate_pairs"] for r in results)
    sessions_with_dups = sum(1 for r in results if r["n_duplicate_pairs"] > 0)

    print(f"{'=' * 62}")
    print(
        f"  Total labeled: {total_labeled}   "
        f"Total duplicate pairs: {total_pairs}   "
        f"Sessions with duplicates: {sessions_with_dups}/{len(results)}"
    )
    if total_pairs > 0:
        print(f"  Side-by-side images saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
