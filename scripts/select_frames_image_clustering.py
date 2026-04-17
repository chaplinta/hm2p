#!/usr/bin/env python3
"""Select training frames based on image appearance clustering.

For each session, downloads the overhead video from S3, extracts thumbnails
at regular intervals, clusters them with PCA + k-means, then selects frames
from underrepresented clusters weighted by DLC model uncertainty. Existing
labeled frames are taken into account to avoid redundant selections.

Usage:
    uv run python scripts/select_frames_image_clustering.py --n 120 --dry-run
    uv run python scripts/select_frames_image_clustering.py --n 120
    uv run python scripts/select_frames_image_clustering.py --n 120 \\
        --session 20210823_16_59_50_1114353
    uv run python scripts/select_frames_image_clustering.py --n 120 \\
        --per-session-max 8 --per-session-min 2
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import boto3
import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path constants (mirrors select_labelling_frames.py)
# ---------------------------------------------------------------------------
REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
RAWDATA_BUCKET = "hm2p-rawdata"
REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = REPO_ROOT / "metadata" / "experiments.csv"
RETRAIN_DIR = REPO_ROOT / "retrain_frames"
LABELED_DIR = REPO_ROOT / "sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/labeled-data"
RETRAIN_META_DIR = REPO_ROOT / "metadata" / "retrain_frames"

# Ensure src/hm2p is importable
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reused helpers (same logic as select_labelling_frames.py)
# ---------------------------------------------------------------------------


def get_sessions() -> list[dict]:
    """Return all sessions from experiments.csv with sub/ses identifiers."""
    sessions = []
    with open(METADATA_PATH) as f:
        for row in csv.DictReader(f):
            eid = row["exp_id"]
            parts = eid.split("_")
            sub = f"sub-{parts[-1]}"
            ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
            sessions.append(
                {
                    "exp_id": eid,
                    "sub": sub,
                    "ses": ses,
                    "primary": str(row.get("primary_exp", "1")).strip() == "1",
                    "exclude": str(row.get("exclude", "0")).strip() == "1",
                }
            )
    return sessions


def load_pose_from_s3(s3: Any, sub: str, ses: str) -> pd.DataFrame | None:
    """Download and load the DLC .h5 pose file for a session from S3.

    Prefers finetuned models (Resnet/Hrnet) over superanimal when multiple
    .h5 files are present.

    Parameters
    ----------
    s3 :
        boto3 S3 client.
    sub : str
        Subject identifier (e.g. ``"sub-1114353"``).
    ses : str
        Session identifier (e.g. ``"ses-20210823T165950"``).

    Returns
    -------
    pd.DataFrame or None
        DLC pose DataFrame, or None if no pose file found.
    """
    prefix = f"pose/{sub}/{ses}/"
    resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=prefix, MaxKeys=20)
    h5_keys = [
        o["Key"]
        for o in resp.get("Contents", [])
        if o["Key"].endswith(".h5") and "filtered" not in o["Key"]
    ]
    if not h5_keys:
        return None

    key = h5_keys[0]
    for k in h5_keys:
        if "Resnet" in k or "Hrnet" in k:
            key = k
            break

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        s3.download_file(DERIVATIVES_BUCKET, key, tmp.name)
        df = pd.read_hdf(tmp.name)
    return df


def already_labelled_frames(session_tag: str) -> set[int]:
    """Return frame indices already labelled for a session.

    Parameters
    ----------
    session_tag : str
        Combined tag ``"sub-XXXX_ses-YYYYMMDDTHHMMSS"``.

    Returns
    -------
    set[int]
        Set of frame indices that already exist in the metadata JSON.
    """
    json_path = RETRAIN_META_DIR / f"{session_tag}.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        return set(data.get("frame_indices", []))
    return set()


# ---------------------------------------------------------------------------
# Thumbnail extraction
# ---------------------------------------------------------------------------

THUMB_SIZE = 64  # pixels per side for clustering thumbnails
THUMB_STRIDE = 100  # extract one thumbnail every N frames (~1 s at 100 fps)


def extract_thumbnails(
    video_path: str,
    stride: int = THUMB_STRIDE,
    thumb_size: int = THUMB_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract grayscale thumbnails at regular frame intervals.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    stride : int
        Extract one frame every ``stride`` frames.
    thumb_size : int
        Thumbnail edge length in pixels (square).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(features, frame_indices)`` where ``features`` has shape
        ``(N, thumb_size * thumb_size)`` (float32, mean-subtracted per frame)
        and ``frame_indices`` is ``(N,)`` int64.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = list(range(0, total, stride))

    features: list[np.ndarray] = []
    frame_indices: list[int] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        vec = thumb.astype(np.float32).ravel()
        vec -= vec.mean()
        features.append(vec)
        frame_indices.append(idx)

    cap.release()

    if not features:
        empty_feat = np.empty((0, thumb_size * thumb_size), dtype=np.float32)
        return empty_feat, np.empty(0, dtype=np.int64)

    return np.stack(features, axis=0), np.array(frame_indices, dtype=np.int64)


# ---------------------------------------------------------------------------
# Cluster-based frame scoring
# ---------------------------------------------------------------------------


def cluster_frames(
    features: np.ndarray,
    n_clusters: int = 30,
    pca_dims: int = 50,
) -> np.ndarray:
    """Cluster frame thumbnails using PCA + k-means.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        Thumbnail feature vectors.
    n_clusters : int
        Number of k-means clusters.
    pca_dims : int
        Number of PCA components to retain before clustering.

    Returns
    -------
    np.ndarray, shape (N,) int32
        Cluster label for each frame.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    n_samples = features.shape[0]
    if n_samples < 2:
        return np.zeros(n_samples, dtype=np.int32)

    # Cap PCA dims and clusters to available data
    actual_pca = min(pca_dims, n_samples, features.shape[1])
    actual_k = min(n_clusters, n_samples)

    pca = PCA(n_components=actual_pca, random_state=42)
    reduced = pca.fit_transform(features)

    km = KMeans(n_clusters=actual_k, n_init=10, random_state=42)
    labels = km.fit_predict(reduced)
    return labels.astype(np.int32)


def extract_frame_confidences(pose_df: pd.DataFrame | None) -> np.ndarray | None:
    """Extract per-frame mean DLC likelihood as a proxy for model uncertainty.

    Parameters
    ----------
    pose_df : pd.DataFrame or None
        DLC pose DataFrame. If None, returns None.

    Returns
    -------
    np.ndarray or None
        Shape ``(N,)`` float64 with mean likelihood per frame.
        Values closer to 0 indicate higher uncertainty.
    """
    if pose_df is None:
        return None

    scorer = pose_df.columns.get_level_values(0)[0]

    if pose_df.columns.nlevels == 4:
        individuals = pose_df.columns.get_level_values(1).unique()
        ind = individuals[0]
        bodyparts = list(pose_df.columns.get_level_values(2).unique())
        lk_cols = []
        for bp in bodyparts:
            with contextlib.suppress(KeyError):
                lk_cols.append(pose_df[(scorer, ind, bp, "likelihood")].values)
    else:
        bodyparts = list(pose_df.columns.get_level_values(1).unique())
        lk_cols = []
        for bp in bodyparts:
            with contextlib.suppress(KeyError):
                lk_cols.append(pose_df[(scorer, bp, "likelihood")].values)

    if not lk_cols:
        return None

    lk = np.column_stack(lk_cols)  # (N, K)
    return np.nanmean(lk, axis=1)  # mean likelihood per frame (high = confident)


def score_clusters(
    cluster_labels: np.ndarray,
    frame_indices: np.ndarray,
    confidences: np.ndarray | None,
    pose_n_frames: int,
    existing_frame_indices: set[int],
    expected_per_cluster: float,
) -> dict[int, float]:
    """Compute a selection score for each cluster.

    Score = (1 - coverage_ratio) * mean_uncertainty_in_cluster

    Where:
    - coverage_ratio = n_existing_in_cluster / expected_per_cluster
    - uncertainty = 1 - mean_confidence (so high uncertainty = low confidence)

    Clusters already well-represented by existing labels get a lower score.

    Parameters
    ----------
    cluster_labels : np.ndarray, shape (N,)
        Cluster assignment per sampled frame.
    frame_indices : np.ndarray, shape (N,)
        Video frame index corresponding to each sampled frame.
    confidences : np.ndarray or None, shape (M,)
        Per-video-frame mean DLC likelihood. Indexed by video frame number.
        If None, uncertainty defaults to 0.5 uniformly.
    pose_n_frames : int
        Total number of frames in the pose DataFrame (for index bounds check).
    existing_frame_indices : set[int]
        Frame indices already in the labeled set.
    expected_per_cluster : float
        Expected number of existing frames per cluster under uniform coverage.

    Returns
    -------
    dict[int, float]
        Mapping from cluster label to score.
    """
    unique_clusters = np.unique(cluster_labels)
    cluster_scores: dict[int, float] = {}

    for c in unique_clusters:
        mask = cluster_labels == c
        c_frames = frame_indices[mask]

        # Coverage: how many existing labeled frames fall in this cluster
        n_existing = sum(
            1
            for f in existing_frame_indices
            if any(abs(f - cf) <= THUMB_STRIDE // 2 for cf in c_frames)
        )
        coverage_ratio = min(1.0, n_existing / max(1.0, expected_per_cluster))

        # Uncertainty: 1 - mean confidence for frames in this cluster.
        # Pose array may be shorter than video (DLC works at ~30 fps vs 100 fps
        # raw). Use direct frame index; out-of-range frames use 0.5 (neutral).
        if confidences is not None:
            valid_conf = []
            for vf in c_frames:
                pose_idx = int(round(vf))
                if 0 <= pose_idx < len(confidences):
                    valid_conf.append(1.0 - float(confidences[pose_idx]))
            mean_uncertainty = float(np.mean(valid_conf)) if valid_conf else 0.5
        else:
            mean_uncertainty = 0.5

        cluster_scores[int(c)] = (1.0 - coverage_ratio) * mean_uncertainty

    return cluster_scores


def select_from_clusters(
    cluster_labels: np.ndarray,
    frame_indices: np.ndarray,
    cluster_scores: dict[int, float],
    confidences: np.ndarray | None,
    pose_n_frames: int,
    n_select: int,
    existing_frame_indices: set[int],
    min_frame_spacing: int = 30,
) -> list[int]:
    """Select frames from clusters, prioritising high-scoring clusters.

    Within each cluster, selects frames in order of decreasing uncertainty.

    Parameters
    ----------
    cluster_labels : np.ndarray, shape (N,)
    frame_indices : np.ndarray, shape (N,)
    cluster_scores : dict[int, float]
        Score per cluster (higher = more desirable).
    confidences : np.ndarray or None
        Per-video-frame mean DLC likelihood.
    pose_n_frames : int
        Total frames in pose DataFrame.
    n_select : int
        Number of frames to select.
    existing_frame_indices : set[int]
        Frames to exclude (already labeled).
    min_frame_spacing : int
        Minimum frame gap between selected frames.

    Returns
    -------
    list[int]
        Selected video frame indices, highest priority first.
    """
    # Sort clusters by score descending, then round-robin across them
    sorted_clusters = sorted(cluster_scores, key=lambda c: -cluster_scores[c])

    # Build per-cluster candidate lists sorted by uncertainty descending
    cluster_candidates: dict[int, list[int]] = {}
    for c in sorted_clusters:
        mask = cluster_labels == c
        c_frames = frame_indices[mask].tolist()

        if confidences is not None:
            # Sort by uncertainty descending (1 - confidence)
            def _unc(f: int) -> float:
                idx = int(f)
                if 0 <= idx < len(confidences):
                    return 1.0 - float(confidences[idx])
                return 0.5

            c_frames.sort(key=_unc, reverse=True)

        cluster_candidates[c] = c_frames

    selected: list[int] = []

    # Round-robin across clusters (highest scoring first) until n_select reached
    while len(selected) < n_select:
        made_progress = False
        for c in sorted_clusters:
            if len(selected) >= n_select:
                break
            while cluster_candidates[c]:
                f = cluster_candidates[c].pop(0)
                if f in existing_frame_indices:
                    continue
                if any(abs(f - s) < min_frame_spacing for s in selected):
                    continue
                selected.append(f)
                made_progress = True
                break
        if not made_progress:
            break

    return selected


# ---------------------------------------------------------------------------
# Video download
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

    overhead_key = None
    fallback_key = None
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
# Frame extraction + symlink
# ---------------------------------------------------------------------------


def extract_frames_to_retrain_dir(
    video_path: Path,
    frame_indices: list[int],
    retrain_session_dir: Path,
) -> list[int]:
    """Extract frames from video and write to retrain_frames/<session>/.

    Skips frames that already exist on disk.

    Parameters
    ----------
    video_path : Path
        Source video file.
    frame_indices : list[int]
        Frame indices to extract.
    retrain_session_dir : Path
        Destination directory (created if absent).

    Returns
    -------
    list[int]
        Indices of newly written frames.
    """
    retrain_session_dir.mkdir(parents=True, exist_ok=True)

    new_indices = [
        idx
        for idx in frame_indices
        if not (retrain_session_dir / f"frame_{int(idx):06d}.png").exists()
    ]
    if not new_indices:
        log.info("    All %d frames already on disk, skipping extraction.", len(frame_indices))
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("    Cannot open video for extraction: %s", video_path)
        return []

    written = []
    for idx in sorted(new_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            out_path = retrain_session_dir / f"frame_{int(idx):06d}.png"
            cv2.imwrite(str(out_path), frame)
            written.append(idx)
    cap.release()

    log.info("    Extracted %d new frames to %s/", len(written), retrain_session_dir.name)
    return written


def find_labeled_data_dir(sub: str, ses: str) -> Path | None:
    """Find the labeled-data directory for a session by matching date + animal.

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


def symlink_frames_into_labeled_data(
    retrain_session_dir: Path,
    labeled_data_dir: Path,
) -> int:
    """Create relative symlinks from labeled-data/<session>/ to retrain_frames/<session>/.

    Mirrors the ``_ensure_pngs`` pattern from ``interactive_label.py``.

    Parameters
    ----------
    retrain_session_dir : Path
        Source directory containing extracted PNGs.
    labeled_data_dir : Path
        Target labeled-data session directory (created if absent).

    Returns
    -------
    int
        Number of new symlinks created.
    """
    labeled_data_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for png in sorted(retrain_session_dir.glob("frame_*.png")):
        dest = labeled_data_dir / png.name
        if not dest.exists():
            rel = os.path.relpath(png.resolve(), labeled_data_dir.resolve())
            dest.symlink_to(rel)
            linked += 1
    return linked


# ---------------------------------------------------------------------------
# Metadata update
# ---------------------------------------------------------------------------


def update_retrain_meta(
    session_tag: str,
    sub: str,
    ses: str,
    new_indices: list[int],
    video_name: str | None = None,
) -> None:
    """Merge new frame indices into the session metadata JSON.

    Parameters
    ----------
    session_tag : str
        Combined tag ``"sub-XXXX_ses-YYYYMMDDTHHMMSS"``.
    sub : str
        Subject identifier.
    ses : str
        Session identifier.
    new_indices : list[int]
        New frame indices to add.
    video_name : str or None
        Video filename to record if not already stored.
    """
    RETRAIN_META_DIR.mkdir(parents=True, exist_ok=True)
    meta_file = RETRAIN_META_DIR / f"{session_tag}.json"

    existing: dict = {}
    if meta_file.exists():
        existing = json.loads(meta_file.read_text())

    existing_indices = set(existing.get("frame_indices", []))
    merged = sorted(existing_indices | set(new_indices))

    updated = {
        "session": f"{sub}/{ses}",
        "frame_indices": merged,
        "n_frames": len(merged),
    }
    if video_name and "video" not in existing:
        updated["video"] = video_name
    elif "video" in existing:
        updated["video"] = existing["video"]

    meta_file.write_text(json.dumps(updated, indent=2))


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------


def process_session(
    s3: Any,
    session_info: dict,
    n_select: int,
    n_clusters: int,
    dry_run: bool,
) -> dict:
    """Run the full cluster-based selection pipeline for one session.

    1. Download video from S3 (into tempdir).
    2. Extract thumbnails every THUMB_STRIDE frames.
    3. PCA + k-means clustering.
    4. Load existing labeled frame indices.
    5. Load DLC confidence from S3 pose file.
    6. Score clusters by coverage + uncertainty.
    7. Select frames, then pixel dedup.
    8. Extract PNGs + create symlinks (unless dry_run).
    9. Update metadata JSON (unless dry_run).

    Parameters
    ----------
    s3 :
        boto3 S3 client.
    session_info : dict
        Keys: ``exp_id``, ``sub``, ``ses``, ``primary``, ``exclude``.
    n_select : int
        Number of new frames to select.
    n_clusters : int
        k-means cluster count.
    dry_run : bool
        If True, skip extraction and metadata updates.

    Returns
    -------
    dict
        Summary with keys: ``exp_id``, ``sub``, ``ses``, ``n_selected``,
        ``n_existing``, ``n_clusters_used``, ``cluster_scores``,
        ``selected_indices``.
    """
    sub = session_info["sub"]
    ses = session_info["ses"]
    exp_id = session_info["exp_id"]
    session_tag = f"{sub}_{ses}"

    existing = already_labelled_frames(session_tag)
    expected_per_cluster = max(1.0, len(existing) / max(1, n_clusters))

    log.info("  Loading pose confidence from S3...")
    pose_df = load_pose_from_s3(s3, sub, ses)
    confidences = extract_frame_confidences(pose_df)
    pose_n_frames = len(pose_df) if pose_df is not None else 0

    with tempfile.TemporaryDirectory(prefix=f"hm2p-{session_tag}-") as tmp_str:
        tmp = Path(tmp_str)
        video_path = download_video_from_s3(s3, sub, ses, tmp)
        if video_path is None:
            log.warning("  No video found on S3 for %s, skipping.", exp_id)
            return {
                "exp_id": exp_id,
                "sub": sub,
                "ses": ses,
                "n_selected": 0,
                "n_existing": len(existing),
                "n_clusters_used": 0,
                "cluster_scores": {},
                "selected_indices": [],
            }

        log.info("  Extracting thumbnails (stride=%d) ...", THUMB_STRIDE)
        features, frame_indices = extract_thumbnails(str(video_path), stride=THUMB_STRIDE)
        log.info("  %d thumbnails extracted.", len(features))

        if len(features) < 2:
            log.warning("  Too few frames to cluster for %s, skipping.", exp_id)
            return {
                "exp_id": exp_id,
                "sub": sub,
                "ses": ses,
                "n_selected": 0,
                "n_existing": len(existing),
                "n_clusters_used": 0,
                "cluster_scores": {},
                "selected_indices": [],
            }

        log.info("  Clustering into %d clusters (PCA+KMeans)...", n_clusters)
        actual_k = min(n_clusters, len(features))
        cluster_labels = cluster_frames(features, n_clusters=actual_k)

        cluster_scores = score_clusters(
            cluster_labels=cluster_labels,
            frame_indices=frame_indices,
            confidences=confidences,
            pose_n_frames=pose_n_frames,
            existing_frame_indices=existing,
            expected_per_cluster=expected_per_cluster,
        )

        # Select 3x candidates to give dedup room to prune
        candidates = select_from_clusters(
            cluster_labels=cluster_labels,
            frame_indices=frame_indices,
            cluster_scores=cluster_scores,
            confidences=confidences,
            pose_n_frames=pose_n_frames,
            n_select=n_select * 3,
            existing_frame_indices=existing,
        )

        if not candidates:
            log.info("  No new candidates found for %s.", exp_id)
            return {
                "exp_id": exp_id,
                "sub": sub,
                "ses": ses,
                "n_selected": 0,
                "n_existing": len(existing),
                "n_clusters_used": actual_k,
                "cluster_scores": cluster_scores,
                "selected_indices": [],
            }

        # Pixel dedup against existing PNGs
        retrain_session_dir = RETRAIN_DIR / session_tag
        retrain_session_dir.mkdir(parents=True, exist_ok=True)

        try:
            from hm2p.pose.dedup import filter_duplicates_against_existing

            n_before = len(candidates)
            candidates = filter_duplicates_against_existing(
                str(video_path), candidates, retrain_session_dir
            )
            n_removed = n_before - len(candidates)
            if n_removed > 0:
                log.info("  Dedup: removed %d/%d duplicate candidates.", n_removed, n_before)
        except Exception as exc:
            log.warning("  Dedup check failed: %s", exc)

        # Limit to requested count after dedup
        selected = candidates[:n_select]

        if dry_run:
            log.info("  [DRY RUN] Would select %d frames for %s.", len(selected), exp_id)
            return {
                "exp_id": exp_id,
                "sub": sub,
                "ses": ses,
                "n_selected": len(selected),
                "n_existing": len(existing),
                "n_clusters_used": actual_k,
                "cluster_scores": cluster_scores,
                "selected_indices": selected,
            }

        # Extract frames to retrain_frames/<session>/
        extract_frames_to_retrain_dir(video_path, selected, retrain_session_dir)

        # Symlink into labeled-data/<clip>/
        ld_dir = find_labeled_data_dir(sub, ses)
        if ld_dir is not None:
            n_linked = symlink_frames_into_labeled_data(retrain_session_dir, ld_dir)
            if n_linked:
                log.info("  Symlinked %d new frames into labeled-data/.", n_linked)
        else:
            log.info("  No labeled-data/ dir found for %s — skipping symlink.", exp_id)

        # Update metadata JSON
        video_name = video_path.name
        update_retrain_meta(session_tag, sub, ses, selected, video_name=video_name)

    return {
        "exp_id": exp_id,
        "sub": sub,
        "ses": ses,
        "n_selected": len(selected),
        "n_existing": len(existing),
        "n_clusters_used": actual_k,
        "cluster_scores": cluster_scores,
        "selected_indices": selected,
    }


# ---------------------------------------------------------------------------
# Frame budget allocation
# ---------------------------------------------------------------------------


def allocate_frames(
    sessions: list[dict],
    total_n: int,
    per_session_max: int,
    per_session_min: int,
) -> dict[str, int]:
    """Allocate frame budget across sessions.

    Primary sessions can receive up to ``per_session_max``; all others up to
    ``per_session_min``. Every session with data gets at least
    ``per_session_min`` if budget allows.

    Parameters
    ----------
    sessions : list[dict]
        Session dicts with ``sub``, ``ses``, ``primary``, ``exclude`` keys.
    total_n : int
        Total frame budget.
    per_session_max : int
        Maximum frames for a primary session.
    per_session_min : int
        Minimum frames per session (also maximum for non-primary sessions).

    Returns
    -------
    dict[str, int]
        Mapping from session_tag to frame count to select.
    """
    allocations: dict[str, int] = {}
    remaining = total_n

    # First pass: everyone gets the minimum
    for s in sessions:
        tag = f"{s['sub']}_{s['ses']}"
        alloc = min(per_session_min, remaining)
        allocations[tag] = alloc
        remaining -= alloc
        if remaining <= 0:
            break

    # Second pass: top up primary sessions to per_session_max
    for s in sessions:
        if remaining <= 0:
            break
        tag = f"{s['sub']}_{s['ses']}"
        current = allocations.get(tag, 0)
        session_max = per_session_max if s["primary"] else per_session_min
        extra = min(session_max - current, remaining)
        if extra > 0:
            allocations[tag] = current + extra
            remaining -= extra

    return allocations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select training frames by image-appearance clustering."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=120,
        help="Total frames to select across all sessions (default 120).",
    )
    parser.add_argument(
        "--per-session-max",
        type=int,
        default=8,
        help="Max frames for primary sessions (default 8).",
    )
    parser.add_argument(
        "--per-session-min",
        type=int,
        default=2,
        help="Min frames per session; max for non-primary (default 2).",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=30,
        help="k-means cluster count per session (default 30).",
    )
    parser.add_argument(
        "--session", type=str, default=None, help="Process a single session by exp_id."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be selected without extracting frames.",
    )
    parser.add_argument(
        "--label", action="store_true", help="Open interactive_label.py after extraction."
    )
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    all_sessions = get_sessions()

    if args.session:
        all_sessions = [s for s in all_sessions if s["exp_id"] == args.session]
        if not all_sessions:
            log.error("Session %r not found in experiments.csv.", args.session)
            sys.exit(1)

    # Sort: primary first, then by fewest existing labels (most room to add)
    all_sessions.sort(
        key=lambda s: (
            not s["primary"],
            len(already_labelled_frames(f"{s['sub']}_{s['ses']}")),
        )
    )

    allocations = allocate_frames(all_sessions, args.n, args.per_session_max, args.per_session_min)

    print(f"\n{'=' * 62}")
    print("  Image-clustering frame selection")
    print(
        f"  Sessions: {len(all_sessions)}   Budget: {args.n}   "
        f"Clusters: {args.n_clusters}   Dry-run: {args.dry_run}"
    )
    print(f"{'=' * 62}\n")

    results = []
    for i, sess in enumerate(all_sessions, 1):
        tag = f"{sess['sub']}_{sess['ses']}"
        n_alloc = allocations.get(tag, 0)
        if n_alloc == 0:
            log.info("[%d/%d] %s: allocation = 0, skipping.", i, len(all_sessions), sess["exp_id"])
            continue

        flag = "primary" if sess["primary"] else ("excl" if sess["exclude"] else "2nd")
        log.info(
            "[%d/%d] %s  [%s]  budget=%d", i, len(all_sessions), sess["exp_id"], flag, n_alloc
        )

        result = process_session(
            s3=s3,
            session_info=sess,
            n_select=n_alloc,
            n_clusters=args.n_clusters,
            dry_run=args.dry_run,
        )
        results.append(result)

        # Summary line per session
        n_existing = result["n_existing"]
        n_selected = result["n_selected"]
        k = result["n_clusters_used"]
        top_scores = sorted(result["cluster_scores"].values(), reverse=True)[:5]
        score_str = ", ".join(f"{v:.3f}" for v in top_scores)
        print(f"  {sess['exp_id']}")
        print(f"    clusters={k}  existing_labels={n_existing}  new_frames={n_selected}")
        print(f"    top cluster scores: [{score_str}]")
        if result["selected_indices"]:
            idx_preview = result["selected_indices"][:6]
            has_more = "..." if len(result["selected_indices"]) > 6 else ""
            print(f"    selected indices: {idx_preview}{has_more}")
        print()

    total_selected = sum(r["n_selected"] for r in results)
    print(f"{'=' * 62}")
    print(f"  Total: {total_selected}/{args.n} frames across {len(results)} sessions")
    if args.dry_run:
        print("  [DRY RUN] — no frames extracted.")
    print(f"{'=' * 62}\n")

    if not args.dry_run and args.label:
        print("Opening interactive labeller...")
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "interactive_label.py")],
            cwd=str(REPO_ROOT),
        )


if __name__ == "__main__":
    main()
