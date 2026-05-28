#!/usr/bin/env python3
"""Select training frames based on image appearance clustering.

Requires a PCA cache built by ``build_pca_cache.py``. For each session:
1. Load PCA cache (sampled frames already in PCA space).
2. Count existing labeled frames for this session.
3. Compute n_new = max(0, target - n_existing).
4. k-means with k = target clusters on all cached frames.
5. Assign existing labeled frames to nearest cluster centroid.
6. Select one frame from each unoccupied cluster.

Usage:
    uv run python scripts/select_frames_image_clustering.py --per-session 30 --dry-run
    uv run python scripts/select_frames_image_clustering.py --per-session 30
    uv run python scripts/select_frames_image_clustering.py --per-session 30 \\
        --session 20210823_16_59_50_1114353
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
PCA_CACHE_DIR = REPO_ROOT / "metadata" / "pca_cache"

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
# PCA cache (from build_pca_cache.py)
# ---------------------------------------------------------------------------


def load_pca_cache(sub: str, ses: str) -> dict | None:
    """Load pre-built PCA cache if available.

    Returns dict with ``data_pca``, ``frame_indices``, ``n_frames``
    or None.
    """
    import pickle

    cache_file = PCA_CACHE_DIR / f"{sub}_{ses}.npz"
    pca_file = cache_file.with_suffix(".pca.pkl")
    if not cache_file.exists():
        return None
    npz = np.load(cache_file)
    pca_model = None
    if pca_file.exists():
        with open(pca_file, "rb") as f:
            pca_model = pickle.load(f)
    return {
        "data_pca": npz["data_pca"],
        "frame_indices": npz["frame_indices"],
        "n_frames": int(npz["n_frames"]),
        "pca": pca_model,
    }


# ---------------------------------------------------------------------------
# Thumbnail stride used by PCA cache (for nearest-frame matching)
THUMB_STRIDE = 100


# ---------------------------------------------------------------------------
# Cluster-based frame scoring
# ---------------------------------------------------------------------------


def cluster_and_select(
    data_pca: np.ndarray,
    frame_indices: np.ndarray,
    existing_frame_indices: set[int],
    n_target: int,
    pca_model: object | None = None,
    clip_dir: "Path | None" = None,
) -> list[int]:
    """k=n_target clusters. Existing frames projected into PCA space and
    assigned to clusters. Select from unoccupied clusters.

    Algorithm:
    1. k-means with k = n_target on all cached frames.
    2. For each existing labeled frame, extract its thumbnail from the
       labeled-data PNG, project into PCA space, assign to nearest
       cluster centroid → cluster is "occupied".
    3. Pick one new frame from each unoccupied cluster (closest to centroid).
    4. Stop when existing + new = n_target.

    Parameters
    ----------
    data_pca : np.ndarray, shape (N, D)
        PCA-transformed thumbnail features (from PCA cache).
    frame_indices : np.ndarray, shape (N,)
        Video frame index for each sampled thumbnail.
    existing_frame_indices : set[int]
        Frame indices already labeled.
    n_target : int
        Desired total (existing + new) per session.
    pca_model : sklearn PCA or None
        PCA model to project existing frame thumbnails. Required for
        accurate cluster assignment.
    clip_dir : Path or None
        Path to labeled-data clip directory containing existing PNGs.
        Used to extract thumbnails for PCA projection.
    """
    from sklearn.cluster import KMeans

    n_existing = len(existing_frame_indices)
    n_new = max(0, n_target - n_existing)
    if n_new == 0:
        return []

    n_samples = data_pca.shape[0]
    k = min(n_target, n_samples)
    if k < 2:
        return []

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(data_pca)

    # Assign existing frames to clusters by projecting their actual
    # thumbnails into PCA space and predicting cluster assignment.
    cluster_occupancy: dict[int, int] = {c: 0 for c in range(k)}

    if pca_model is not None and clip_dir is not None:
        import cv2
        import math

        thumb_size = int(math.sqrt(pca_model.n_features_in_))
        for ef in existing_frame_indices:
            png_path = clip_dir / f"frame_{ef:06d}.png"
            if not png_path.exists():
                continue
            img = cv2.imread(str(png_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thumb = cv2.resize(gray, (thumb_size, thumb_size),
                               interpolation=cv2.INTER_NEAREST)
            vec = thumb.astype(np.float64).ravel().reshape(1, -1)
            vec_pca = pca_model.transform(vec)
            cluster_id = int(km.predict(vec_pca)[0])
            cluster_occupancy[cluster_id] += 1
        log.info("  Projected %d existing frames into PCA space for cluster assignment",
                 sum(cluster_occupancy.values()))
    else:
        # Fallback: nearest cached frame index (less accurate)
        log.warning("  No PCA model or clip_dir — falling back to nearest-index matching")
        fi_array = frame_indices.astype(np.int64)
        for ef in existing_frame_indices:
            dists = np.abs(fi_array - ef)
            nearest_idx = int(np.argmin(dists))
            if dists[nearest_idx] <= THUMB_STRIDE * 2:
                cluster_occupancy[int(labels[nearest_idx])] += 1

    n_occupied = sum(1 for v in cluster_occupancy.values() if v > 0)
    n_empty = k - n_occupied
    log.info("  k=%d clusters: %d occupied, %d empty, need %d new frames",
             k, n_occupied, n_empty, n_new)

    # Select from unoccupied clusters only.
    empty_clusters = [c for c in range(k) if cluster_occupancy[c] == 0]
    selected: list[int] = []

    for c in empty_clusters:
        if len(selected) >= n_new:
            break
        mask = labels == c
        c_pca = data_pca[mask]
        c_frames = frame_indices[mask]
        centroid = km.cluster_centers_[c]
        order = np.argsort(np.linalg.norm(c_pca - centroid, axis=1))
        for idx in order:
            fi = int(c_frames[idx])
            if fi not in existing_frame_indices and fi not in selected:
                selected.append(fi)
                break

    # Fallback: if not enough empty clusters, pick from least-occupied.
    if len(selected) < n_new:
        occupied_by_need = sorted(
            [c for c in range(k) if cluster_occupancy[c] > 0],
            key=lambda c: cluster_occupancy[c],
        )
        n_fallback = 0
        for c in occupied_by_need:
            if len(selected) >= n_new:
                break
            mask = labels == c
            c_pca = data_pca[mask]
            c_frames = frame_indices[mask]
            centroid = km.cluster_centers_[c]
            order = np.argsort(np.linalg.norm(c_pca - centroid, axis=1))
            for idx in order:
                fi = int(c_frames[idx])
                if fi not in existing_frame_indices and fi not in selected:
                    selected.append(fi)
                    n_fallback += 1
                    break
        if n_fallback > 0:
            log.info("  %d from empty clusters, %d from occupied (fallback)",
                     len(selected) - n_fallback, n_fallback)

    return selected


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
    n_target: int,
    dry_run: bool,
) -> dict:
    """Run the cluster-based selection pipeline for one session.

    1. Load PCA cache (or fall back to video download + thumbnail extraction).
    2. Count existing labeled frames → compute n_new = target - existing.
    3. k-means with k = existing + n_new.
    4. Mark clusters occupied by existing frames.
    5. Select one frame per unoccupied cluster.
    6. Extract PNGs + symlink (unless dry_run).

    Returns summary dict.
    """
    sub = session_info["sub"]
    ses = session_info["ses"]
    exp_id = session_info["exp_id"]
    session_tag = f"{sub}_{ses}"

    existing = already_labelled_frames(session_tag)
    n_existing = len(existing)
    n_new = max(0, n_target - n_existing)

    if n_new == 0:
        log.info("  Already have %d >= %d frames, nothing to do.", n_existing, n_target)
        return {
            "exp_id": exp_id, "sub": sub, "ses": ses,
            "n_selected": 0, "n_existing": n_existing,
            "selected_indices": [],
        }

    # Load PCA cache (required — run build_pca_cache.py first)
    pca_cache = load_pca_cache(sub, ses)
    if pca_cache is None:
        log.error("  No PCA cache for %s. Run: uv run python scripts/build_pca_cache.py", exp_id)
        return {
            "exp_id": exp_id, "sub": sub, "ses": ses,
            "n_selected": 0, "n_existing": n_existing,
            "selected_indices": [],
        }

    log.info("  PCA cache: %d frames, %d PCs. existing=%d, need=%d",
             pca_cache["data_pca"].shape[0], pca_cache["data_pca"].shape[1],
             n_existing, n_new)
    data_pca = pca_cache["data_pca"]
    frame_indices = pca_cache["frame_indices"]
    video_path = None

    with tempfile.TemporaryDirectory(prefix=f"hm2p-{session_tag}-") as tmp_str:
        tmp = Path(tmp_str)

        # Find labeled-data clip dir for PCA projection of existing frames
        clip_dir = find_labeled_data_dir(sub, ses)

        # k = n_target, project existing frames into PCA space, select
        # from unoccupied clusters
        selected = cluster_and_select(
            data_pca=data_pca,
            frame_indices=frame_indices,
            existing_frame_indices=existing,
            n_target=n_target,
            pca_model=pca_cache.get("pca"),
            clip_dir=clip_dir,
        )

        if not selected:
            log.info("  No new frames selected for %s.", exp_id)
            return {
                "exp_id": exp_id, "sub": sub, "ses": ses,
                "n_selected": 0, "n_existing": n_existing,
                "selected_indices": [],
            }

        if dry_run:
            log.info("  [DRY RUN] Would select %d new frames for %s.", len(selected), exp_id)
            return {
                "exp_id": exp_id, "sub": sub, "ses": ses,
                "n_selected": len(selected), "n_existing": n_existing,
                "selected_indices": selected,
            }

        # Download video for PNG extraction if we used cache earlier
        if video_path is None:
            video_path = download_video_from_s3(s3, sub, ses, tmp)
        if video_path is None:
            log.warning("  Cannot extract PNGs — no video on S3.")
            return {
                "exp_id": exp_id, "sub": sub, "ses": ses,
                "n_selected": 0, "n_existing": n_existing,
                "selected_indices": [],
            }

        retrain_session_dir = RETRAIN_DIR / session_tag
        extract_frames_to_retrain_dir(video_path, selected, retrain_session_dir)

        ld_dir = find_labeled_data_dir(sub, ses)
        if ld_dir is not None:
            n_linked = symlink_frames_into_labeled_data(retrain_session_dir, ld_dir)
            if n_linked:
                log.info("  Symlinked %d new frames into labeled-data/.", n_linked)
        else:
            log.info("  No labeled-data/ dir found for %s — skipping symlink.", exp_id)

        update_retrain_meta(session_tag, sub, ses, selected,
                            video_name=video_path.name if video_path else None)

    return {
        "exp_id": exp_id, "sub": sub, "ses": ses,
        "n_selected": len(selected), "n_existing": n_existing,
        "selected_indices": selected,
    }


# ---------------------------------------------------------------------------
# Frame budget allocation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select training frames by image-appearance clustering."
    )
    parser.add_argument(
        "--per-session",
        type=int,
        default=30,
        help="Target total labeled frames per session (default 30). "
             "Sessions already at or above this count are skipped.",
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

    print(f"\n{'=' * 62}")
    print("  Image-clustering frame selection")
    print(
        f"  Sessions: {len(all_sessions)}   Target: {args.per_session}/session   "
        f"Dry-run: {args.dry_run}"
    )
    print(f"{'=' * 62}\n")

    results = []
    for i, sess in enumerate(all_sessions, 1):
        flag = "primary" if sess["primary"] else ("excl" if sess["exclude"] else "2nd")
        log.info("[%d/%d] %s  [%s]", i, len(all_sessions), sess["exp_id"], flag)

        result = process_session(
            s3=s3,
            session_info=sess,
            n_target=args.per_session,
            dry_run=args.dry_run,
        )
        results.append(result)

        n_existing = result["n_existing"]
        n_selected = result["n_selected"]
        print(f"  {sess['exp_id']}:  existing={n_existing}  new={n_selected}  "
              f"total={n_existing + n_selected}")
        if result["selected_indices"]:
            idx_preview = result["selected_indices"][:6]
            has_more = "..." if len(result["selected_indices"]) > 6 else ""
            print(f"    selected: {idx_preview}{has_more}")
        print()

    total_new = sum(r["n_selected"] for r in results)
    total_existing = sum(r["n_existing"] for r in results)
    print(f"{'=' * 62}")
    print(f"  Existing: {total_existing}   New: {total_new}   "
          f"Total: {total_existing + total_new}")
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
