#!/usr/bin/env python3
"""Select outlier frames for DLC retraining.

Reimplements DLC's extract_outlier_frames logic without needing the
DLC project metadata (training-datasets/, shuffle info).

1. Load pose predictions from S3
2. Find outlier frames (jump + low confidence)
3. Download video, read outlier frames at 30px width (grayscale)
4. K-means cluster outlier frames (k = n_to_pick)
5. Pick one frame per cluster → maximally diverse outlier set
6. Extract full-res PNGs, symlink into labeled-data/

This is the same algorithm DLC uses internally (see
deeplabcut.utils.frameselectiontools.KmeansbasedFrameselectioncv2).

Safety: existing CollectedData_*.csv/.h5 files are never modified.

Usage:
    uv run python scripts/select_hard_frames.py --scan
    uv run python scripts/select_hard_frames.py --min-per-session 20
    uv run python scripts/select_hard_frames.py --per-session 8
    uv run python scripts/select_hard_frames.py --session 20220804_11_21 --per-session 8
    uv run python scripts/select_hard_frames.py --primary-only --min-per-session 20
    uv run python scripts/select_hard_frames.py --total 200
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import boto3
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

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

sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ALL_BODYPARTS = [
    "nose_tip", "left_ear", "right_ear", "head_midpoint",
    "neck", "mid_back", "mouse_center", "tail_base",
]

# Thumbnail width for k-means clustering (DLC default = 30).
# At this size the mouse is ~4-5 px, maze walls are a few px —
# k-means clusters by gross mouse position + pose + lighting.
RESIZE_WIDTH = 30


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
# Pose loading
# ---------------------------------------------------------------------------


def load_pose_from_s3(s3: Any, sub: str, ses: str) -> pd.DataFrame | None:
    from hm2p.pose.select import select_best_dlc_h5_s3

    prefix = f"pose/{sub}/{ses}/"
    h5_key = select_best_dlc_h5_s3(s3, DERIVATIVES_BUCKET, prefix)
    if h5_key is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        s3.download_file(DERIVATIVES_BUCKET, h5_key, tmp.name)
        return pd.read_hdf(tmp.name)


def _resolve_bp(df: pd.DataFrame, scorer: str, bp: str) -> str | None:
    try:
        _ = df[(scorer, bp, "x")]
        return bp
    except KeyError:
        if bp == "head_midpoint":
            try:
                _ = df[(scorer, "implant_base_rear", "x")]
                return "implant_base_rear"
            except KeyError:
                return None
        return None


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


def find_outlier_indices(
    df: pd.DataFrame,
    jump_threshold: float = 20.0,
    p_bound: float = 0.1,
) -> list[int]:
    """Find frame indices that are outliers (jump or low confidence).

    Same logic as DLC's extract_outlier_frames: a frame is an outlier if
    ANY bodypart has a jump > threshold or confidence < p_bound.
    """
    scorer = df.columns.get_level_values(0)[0]
    n = len(df)
    is_outlier = np.zeros(n, dtype=bool)

    for bp in ALL_BODYPARTS:
        col = _resolve_bp(df, scorer, bp)
        if col is None:
            continue

        x = df[(scorer, col, "x")].values.astype(np.float64)
        y = df[(scorer, col, "y")].values.astype(np.float64)
        lik = df[(scorer, col, "likelihood")].values.astype(np.float64)

        # Jump: frame-to-frame displacement > threshold
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        displacement = np.sqrt(dx**2 + dy**2)
        is_outlier |= displacement > jump_threshold

        # Uncertain: likelihood below bound
        is_outlier |= lik < p_bound

    indices = list(np.where(is_outlier)[0])

    # Exclude any frames already in the retrain metadata
    return indices


# ---------------------------------------------------------------------------
# K-means on video thumbnails (DLC's approach)
# ---------------------------------------------------------------------------


def kmeans_select_from_video(
    video_path: str,
    candidate_indices: list[int],
    n_pick: int,
    existing_indices: set[int],
) -> list[int]:
    """Read candidate frames at 30px width, k-means cluster, pick one per cluster.

    This is DLC's KmeansbasedFrameselectioncv2 algorithm. Clustering on
    tiny grayscale thumbnails naturally groups by mouse position + pose +
    lighting. Picking one per cluster gives maximally diverse frames.
    """
    # Remove already-existing frames from candidates
    candidates = [i for i in candidate_indices if i not in existing_indices]
    if not candidates:
        return []
    if len(candidates) <= n_pick:
        return candidates

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning("Cannot open video: %s", video_path)
        return candidates[:n_pick]

    # Get video dimensions for resize ratio
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ratio = RESIZE_WIDTH / w
    h_resized = max(1, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * ratio))
    w_resized = RESIZE_WIDTH

    def _read_thumb(idx: int) -> np.ndarray | None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (w_resized, h_resized),
                           interpolation=cv2.INTER_NEAREST)
        return small.astype(np.float64).ravel()

    # Read thumbnails for existing labeled frames (to include in clustering)
    existing_thumbs = []
    existing_thumb_count = 0
    for idx in sorted(existing_indices):
        thumb = _read_thumb(idx)
        if thumb is not None:
            existing_thumbs.append(thumb)
            existing_thumb_count += 1

    # Read thumbnails for candidate outlier frames
    log.info("    Reading %d outlier frames at %dpx width for clustering...",
             len(candidates), RESIZE_WIDTH)
    candidate_thumbs = []
    valid_indices = []
    for idx in candidates:
        thumb = _read_thumb(idx)
        if thumb is not None:
            candidate_thumbs.append(thumb)
            valid_indices.append(idx)

    cap.release()

    if len(valid_indices) <= n_pick:
        return valid_indices

    # Combine existing + candidate thumbnails for joint clustering.
    # Existing frames participate in clustering so they "occupy" clusters,
    # preventing new frames from being selected in the same visual region.
    all_thumbs = existing_thumbs + candidate_thumbs
    data = np.array(all_thumbs)
    data -= data.mean(axis=0)  # mean-subtract

    # More clusters than frames to pick — existing frames will occupy some,
    # leaving the rest for new picks. Use n_pick + n_existing so there are
    # enough clusters for both.
    n_total = len(data)
    k = min(n_pick + existing_thumb_count, n_total)
    log.info("    K-means clustering %d frames (%d existing + %d candidates) into %d clusters...",
             n_total, existing_thumb_count, len(candidate_thumbs), k)
    kmeans = MiniBatchKMeans(
        n_clusters=k, batch_size=min(100, n_total),
        max_iter=50, n_init=3,
    )
    kmeans.fit(data)

    # Identify which clusters already have an existing frame
    existing_labels = set(kmeans.labels_[:existing_thumb_count])

    # Pick one candidate per cluster, skipping clusters that contain existing frames
    selected = []
    for cluster_id in range(k):
        if cluster_id in existing_labels:
            continue  # this cluster looks like an already-labeled frame
        # Find candidate members (indices offset by existing_thumb_count)
        member_mask = kmeans.labels_[existing_thumb_count:] == cluster_id
        if not member_mask.any():
            continue
        member_local = np.where(member_mask)[0]
        # Pick closest to cluster centre
        centre = kmeans.cluster_centers_[cluster_id]
        candidate_data = data[existing_thumb_count:]
        dists = np.linalg.norm(candidate_data[member_local] - centre, axis=1)
        best = member_local[np.argmin(dists)]
        selected.append(valid_indices[best])
        if len(selected) >= n_pick:
            break

    return selected


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
    merged = sorted(set(existing.get("frame_indices", [])) | set(new_indices))
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
# Per-session processing
# ---------------------------------------------------------------------------


def process_session(
    s3: Any,
    ses_info: dict,
    n_extract: int,
    jump_threshold: float,
    p_bound: float,
    dry_run: bool,
) -> int:
    sub, ses = ses_info["sub"], ses_info["ses"]
    exp_id = ses_info["exp_id"]
    tag = f"{sub}_{ses}"

    # Load pose predictions
    df = load_pose_from_s3(s3, sub, ses)
    if df is None:
        log.warning("  No pose data for %s", exp_id)
        return 0

    # Find outlier frame indices
    outlier_indices = find_outlier_indices(df, jump_threshold, p_bound)
    log.info("  %d outlier frames found", len(outlier_indices))

    if not outlier_indices:
        log.info("  No outliers, skipping")
        return 0

    # Get existing frame indices
    meta_path = RETRAIN_META_DIR / f"{tag}.json"
    existing: set[int] = set()
    if meta_path.exists():
        existing = set(json.loads(meta_path.read_text()).get("frame_indices", []))

    if dry_run:
        n_avail = len([i for i in outlier_indices if i not in existing])
        log.info("  [DRY RUN] %d outliers available, would pick %d via k-means",
                 n_avail, min(n_extract, n_avail))
        return min(n_extract, n_avail)

    # Get video
    video_path = find_video_local(sub, ses)
    tmp_dir = None
    if video_path is None:
        tmp_dir = tempfile.mkdtemp(prefix=f"hm2p-outlier-{exp_id[:15]}-")
        video_path = download_video_from_s3(s3, sub, ses, Path(tmp_dir))
        if video_path is None:
            log.warning("  No video for %s", exp_id)
            return 0

    # K-means select diverse outliers from video
    selected = kmeans_select_from_video(
        str(video_path), outlier_indices, n_extract, existing,
    )
    log.info("  Selected %d diverse frames via k-means", len(selected))

    if not selected:
        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return 0

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
    log.info("  Symlinked %d into %s/", n_linked, ld.name)

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
        description="Select diverse outlier frames for DLC retraining. "
                    "Uses DLC's k-means-on-thumbnails approach for diversity."
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
    parser.add_argument("--jump-threshold", type=float, default=20,
                        help="Jump threshold in pixels (default 20).")
    parser.add_argument("--p-bound", type=float, default=0.1,
                        help="Likelihood threshold for uncertain frames (default 0.1).")
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

        n_extract = args.per_session

        if args.min_per_session is not None:
            need = args.min_per_session - existing_count
            if need <= 0:
                log.info("  %s: already has %d frames (>= %d), skipping",
                         ses_info["exp_id"][:25], existing_count, args.min_per_session)
                continue
            if n_extract is None:
                n_extract = need
            else:
                n_extract = min(n_extract, need)

        if args.total is not None:
            remaining = args.total - total_new
            if remaining <= 0:
                break
            if n_extract is None:
                n_extract = remaining
            else:
                n_extract = min(n_extract, remaining)

        if n_extract is None:
            n_extract = 10

        log.info("\n=== %s (have %d, extracting up to %d) ===",
                 ses_info["exp_id"], existing_count, n_extract)

        n = process_session(
            s3, ses_info, n_extract,
            args.jump_threshold, args.p_bound, args.dry_run,
        )
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
