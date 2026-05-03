#!/usr/bin/env python3
"""Select hard-to-track frames for targeted DLC retraining.

Identifies frames where the current DLC model performs worst (low
confidence, high uncertainty) and selects visually diverse frames
from those hard cases. Adds to existing labeled data without
overwriting anything.

Safety: existing CollectedData_*.csv/.h5 files are never modified.
New frames are extracted as PNGs and symlinked into the DLC
labeled-data directory. The user then labels them in napari-deeplabcut.

Usage:
    # Scan all sessions, show which need more labels:
    uv run python scripts/select_hard_frames.py --scan

    # Select 200 hard frames across all sessions:
    uv run python scripts/select_hard_frames.py --n 200

    # Select 20 hard frames from one session:
    uv run python scripts/select_hard_frames.py --n 20 \\
        --session 20210823_16_59_50_1114353

    # Dry run (show what would be selected, don't extract):
    uv run python scripts/select_hard_frames.py --n 200 --dry-run

    # Target specific bodyparts:
    uv run python scripts/select_hard_frames.py --n 200 \\
        --bodyparts nose_tip,tail_base
"""

from __future__ import annotations

import argparse
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
# Path constants
# ---------------------------------------------------------------------------
REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
RAWDATA_BUCKET = "hm2p-rawdata"
REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = REPO_ROOT / "metadata" / "experiments.csv"
RETRAIN_DIR = REPO_ROOT / "retrain_frames"
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


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------


def get_sessions() -> list[dict]:
    """Load session list from experiments.csv."""
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


def already_labelled_frames(session_tag: str) -> set[int]:
    """Return frame indices already selected for a session."""
    json_path = RETRAIN_META_DIR / f"{session_tag}.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        return set(data.get("frame_indices", []))
    return set()


# ---------------------------------------------------------------------------
# Pose loading and scoring
# ---------------------------------------------------------------------------


def load_pose_from_s3(
    s3: Any, sub: str, ses: str
) -> pd.DataFrame | None:
    """Download the best DLC H5 file from S3 and return as DataFrame."""
    from hm2p.pose.select import select_best_dlc_h5_s3

    prefix = f"pose/{sub}/{ses}/"
    h5_key = select_best_dlc_h5_s3(s3, DERIVATIVES_BUCKET, prefix)
    if h5_key is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        s3.download_file(DERIVATIVES_BUCKET, h5_key, tmp.name)
        return pd.read_hdf(tmp.name)


def score_frames_by_difficulty(
    df: pd.DataFrame,
    target_bodyparts: list[str] | None = None,
) -> np.ndarray:
    """Score each frame by tracking difficulty (higher = harder).

    Difficulty is based on:
    1. Low confidence on target bodyparts (primary signal)
    2. Large frame-to-frame jumps (secondary signal)

    Parameters
    ----------
    df : pd.DataFrame
        DLC pose output with multi-index columns.
    target_bodyparts : list[str] or None
        Bodyparts to target. None = all bodyparts.

    Returns
    -------
    np.ndarray
        Difficulty score per frame, shape (n_frames,).
    """
    scorer = df.columns.get_level_values(0)[0]
    bodyparts = target_bodyparts or ALL_BODYPARTS

    n = len(df)
    scores = np.zeros(n, dtype=np.float64)

    for bp in bodyparts:
        # Handle head_midpoint / implant_base_rear alias
        bp_col = bp
        try:
            lik = df[(scorer, bp_col, "likelihood")].values.astype(np.float64)
        except KeyError:
            if bp == "head_midpoint":
                try:
                    lik = df[(scorer, "implant_base_rear", "likelihood")].values.astype(np.float64)
                    bp_col = "implant_base_rear"
                except KeyError:
                    continue
            else:
                continue

        # Low confidence score: (1 - likelihood)^2 so very low confidence
        # frames score much higher
        conf_score = (1.0 - np.clip(lik, 0, 1)) ** 2
        scores += conf_score

        # Jump score: large frame-to-frame displacement
        try:
            x = df[(scorer, bp_col, "x")].values.astype(np.float64)
            y = df[(scorer, bp_col, "y")].values.astype(np.float64)
            dx = np.diff(x, prepend=x[0])
            dy = np.diff(y, prepend=y[0])
            displacement = np.sqrt(dx**2 + dy**2)
            # Normalize to [0, 1] range
            dmax = np.nanpercentile(displacement, 99) or 1.0
            jump_score = np.clip(displacement / dmax, 0, 1)
            scores += 0.3 * jump_score  # lower weight than confidence
        except KeyError:
            pass

    return scores


def pixel_dedup(
    video_path: str,
    candidate_indices: list[int],
    existing_indices: set[int],
    threshold: float = 0.98,
) -> list[int]:
    """Remove candidates that are too similar to existing or selected frames.

    Uses normalized cross-correlation between grayscale thumbnails.

    Parameters
    ----------
    video_path : str
        Path to video file.
    candidate_indices : list[int]
        Frame indices to consider (sorted by priority, best first).
    existing_indices : set[int]
        Frame indices already labeled (loaded for dedup comparison).
    threshold : float
        NCC threshold above which frames are considered duplicates.

    Returns
    -------
    list[int]
        Deduplicated frame indices.
    """
    THUMB_SIZE = 32  # smaller thumbs = less false-positive similarity

    def _read_thumb(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (THUMB_SIZE, THUMB_SIZE))
        return thumb.astype(np.float32).ravel()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning("Cannot open video for dedup: %s", video_path)
        return candidate_indices

    # Load existing frame thumbnails
    reference_thumbs: list[np.ndarray] = []
    for idx in sorted(existing_indices):
        thumb = _read_thumb(cap, idx)
        if thumb is not None:
            reference_thumbs.append(thumb)

    selected: list[int] = []
    selected_thumbs: list[np.ndarray] = []

    for idx in candidate_indices:
        thumb = _read_thumb(cap, idx)
        if thumb is None:
            continue

        # Compare against all reference + already selected
        too_similar = False
        norm = np.linalg.norm(thumb) or 1.0
        for ref in reference_thumbs + selected_thumbs:
            ref_norm = np.linalg.norm(ref) or 1.0
            ncc = float(np.dot(thumb, ref) / (norm * ref_norm))
            if ncc > threshold:
                too_similar = True
                break

        if not too_similar:
            selected.append(idx)
            selected_thumbs.append(thumb)

    cap.release()
    return selected


# ---------------------------------------------------------------------------
# Video download
# ---------------------------------------------------------------------------


def download_video_from_s3(
    s3: Any, sub: str, ses: str, dest_dir: Path
) -> Path | None:
    """Download overhead .mp4 from S3."""
    prefix = f"rawdata/{sub}/{ses}/behav/"
    resp = s3.list_objects_v2(Bucket=RAWDATA_BUCKET, Prefix=prefix)
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        fname = key.split("/")[-1]
        if not fname.endswith(".mp4") or "side" in fname.lower():
            continue
        local = dest_dir / fname
        s3.download_file(RAWDATA_BUCKET, key, str(local))
        return local
    return None


# ---------------------------------------------------------------------------
# Frame extraction and linking
# ---------------------------------------------------------------------------


def extract_frames(
    video_path: Path, frame_indices: list[int], dest_dir: Path
) -> list[int]:
    """Extract frames as PNGs. Skips already-existing files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    new = [i for i in frame_indices if not (dest_dir / f"frame_{int(i):06d}.png").exists()]
    if not new:
        log.info("    All frames already on disk.")
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
    log.info("    Extracted %d new frames.", len(written))
    return written


def symlink_into_labeled_data(
    retrain_dir: Path, labeled_dir: Path
) -> int:
    """Create relative symlinks from labeled-data/ to retrain_frames/."""
    labeled_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for png in sorted(retrain_dir.glob("frame_*.png")):
        dest = labeled_dir / png.name
        if not dest.exists():
            rel = os.path.relpath(png.resolve(), labeled_dir.resolve())
            dest.symlink_to(rel)
            linked += 1
    return linked


def update_meta(
    session_tag: str, sub: str, ses: str, new_indices: list[int],
    video_name: str | None = None,
) -> None:
    """Merge new frame indices into session metadata JSON."""
    RETRAIN_META_DIR.mkdir(parents=True, exist_ok=True)
    meta_file = RETRAIN_META_DIR / f"{session_tag}.json"

    existing: dict = {}
    if meta_file.exists():
        existing = json.loads(meta_file.read_text())

    merged = sorted(set(existing.get("frame_indices", [])) | set(new_indices))
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


def find_labeled_data_dir(sub: str, ses: str) -> Path | None:
    """Find the labeled-data directory for a session."""
    if not LABELED_DIR.exists():
        return None
    ses_date = ses.replace("ses-", "").split("T")[0]
    animal = sub.replace("sub-", "")
    for ld in LABELED_DIR.iterdir():
        if ld.is_dir() and ses_date in ld.name and animal in ld.name:
            return ld
    return None


# ---------------------------------------------------------------------------
# Scan mode
# ---------------------------------------------------------------------------


def scan_sessions(s3: Any, target_bodyparts: list[str] | None = None) -> list[dict]:
    """Scan all sessions and report labeling status + difficulty."""
    sessions = get_sessions()
    results = []

    for ses_info in sessions:
        sub, ses = ses_info["sub"], ses_info["ses"]
        tag = f"{sub}_{ses}"
        existing = already_labelled_frames(tag)

        df = load_pose_from_s3(s3, sub, ses)
        if df is None:
            results.append({
                **ses_info, "tag": tag,
                "n_existing": len(existing), "n_frames": 0,
                "mean_difficulty": 0, "pct_hard": 0,
                "ear_detection_rate": 0,
            })
            continue

        scorer = df.columns.get_level_values(0)[0]
        n_frames = len(df)
        scores = score_frames_by_difficulty(df, target_bodyparts)

        # Ear detection rate
        try:
            lik_le = df[(scorer, "left_ear", "likelihood")].values
            lik_re = df[(scorer, "right_ear", "likelihood")].values
            both_09 = float(((lik_le > 0.9) & (lik_re > 0.9)).mean() * 100)
        except KeyError:
            both_09 = 0.0

        # Fraction of frames above difficulty threshold
        hard_threshold = np.percentile(scores, 90)
        pct_hard = float((scores > hard_threshold).mean() * 100)

        results.append({
            **ses_info, "tag": tag,
            "n_existing": len(existing), "n_frames": n_frames,
            "mean_difficulty": float(np.mean(scores)),
            "pct_hard": pct_hard,
            "ear_detection_rate": both_09,
        })

    return results


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------


def process_session(
    s3: Any,
    ses_info: dict,
    n_select: int,
    target_bodyparts: list[str] | None,
    dry_run: bool,
) -> dict:
    """Select hard frames for one session."""
    sub, ses = ses_info["sub"], ses_info["ses"]
    tag = f"{sub}_{ses}"
    exp_id = ses_info["exp_id"]

    existing = already_labelled_frames(tag)
    log.info("  %s: %d existing labels", exp_id[:25], len(existing))

    df = load_pose_from_s3(s3, sub, ses)
    if df is None:
        log.warning("  No pose data for %s", exp_id)
        return {"exp_id": exp_id, "n_selected": 0, "n_existing": len(existing)}

    scores = score_frames_by_difficulty(df, target_bodyparts)

    # Rank frames by difficulty (highest first), exclude already-labeled
    ranked = np.argsort(-scores)
    candidates = [int(i) for i in ranked if int(i) not in existing]

    # Take top 3x candidates for dedup headroom
    candidates = candidates[:n_select * 3]

    if dry_run:
        selected = candidates[:n_select]
        log.info("  [DRY RUN] Would select %d frames (top difficulty scores)", len(selected))
        return {
            "exp_id": exp_id, "n_selected": len(selected),
            "n_existing": len(existing), "selected": selected,
        }

    # Download video for dedup + extraction
    with tempfile.TemporaryDirectory(prefix=f"hm2p-hard-{tag}-") as tmp_str:
        tmp = Path(tmp_str)
        video_path = download_video_from_s3(s3, sub, ses, tmp)
        if video_path is None:
            log.warning("  No video for %s", exp_id)
            return {"exp_id": exp_id, "n_selected": 0, "n_existing": len(existing)}

        # Pixel dedup against existing + each other
        selected = pixel_dedup(
            str(video_path), candidates, existing, threshold=0.92
        )
        selected = selected[:n_select]
        log.info("  Selected %d frames after dedup", len(selected))

        if not selected:
            return {"exp_id": exp_id, "n_selected": 0, "n_existing": len(existing)}

        # Extract PNGs
        retrain_session_dir = RETRAIN_DIR / tag
        extract_frames(video_path, selected, retrain_session_dir)

        # Symlink into labeled-data/
        labeled_dir = find_labeled_data_dir(sub, ses)
        if labeled_dir is None:
            # Create new labeled-data dir using the clip name convention
            clip_name = f"{exp_id.replace('_', '_')}_maze-rose_overhead.camera-cropped"
            labeled_dir = LABELED_DIR / clip_name
        n_linked = symlink_into_labeled_data(retrain_session_dir, labeled_dir)
        log.info("  Symlinked %d new frames into %s/", n_linked, labeled_dir.name)

        # Update metadata
        update_meta(tag, sub, ses, selected, video_name=video_path.name)

    return {
        "exp_id": exp_id, "n_selected": len(selected),
        "n_existing": len(existing), "selected": selected,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select hard-to-track frames for targeted DLC retraining."
    )
    parser.add_argument(
        "--n", type=int, default=200,
        help="Total frames to select (default 200).",
    )
    parser.add_argument(
        "--scan", action="store_true",
        help="Scan sessions and show labeling status, don't select frames.",
    )
    parser.add_argument(
        "--session", type=str, default=None,
        help="Process only this session (exp_id or partial match).",
    )
    parser.add_argument(
        "--bodyparts", type=str, default=None,
        help="Comma-separated bodyparts to target (default: all). "
             "E.g. nose_tip,tail_base",
    )
    parser.add_argument(
        "--per-session-max", type=int, default=15,
        help="Max frames per session (default 15).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be selected without extracting.",
    )
    args = parser.parse_args()

    target_bps = args.bodyparts.split(",") if args.bodyparts else None

    s3 = boto3.client("s3", region_name=REGION)
    sessions = get_sessions()

    if args.scan:
        print("\nScanning sessions for labeling status...\n")
        results = scan_sessions(s3, target_bps)
        results.sort(key=lambda r: r.get("mean_difficulty", 0), reverse=True)

        print(f"{'Session':<25s}  {'Labels':>6s}  {'Frames':>7s}  "
              f"{'Ears@0.9':>8s}  {'Difficulty':>10s}  {'Hard%':>5s}")
        print("-" * 75)
        for r in results:
            print(f"{r['exp_id'][:25]:<25s}  {r['n_existing']:>6d}  "
                  f"{r['n_frames']:>7d}  {r['ear_detection_rate']:>7.1f}%  "
                  f"{r['mean_difficulty']:>10.3f}  {r['pct_hard']:>4.1f}%")

        total_labels = sum(r["n_existing"] for r in results)
        print(f"\nTotal labeled frames: {total_labels}")
        print(f"Sessions with labels: {sum(1 for r in results if r['n_existing'] > 0)}")
        return

    # Filter to specific session if requested
    if args.session:
        sessions = [s for s in sessions if args.session in s["exp_id"]]
        if not sessions:
            print(f"No session matching '{args.session}'")
            sys.exit(1)

    # Allocate frames across sessions, weighted by difficulty
    log.info("Loading pose data to score sessions...")
    session_scores = {}
    for ses_info in sessions:
        df = load_pose_from_s3(s3, ses_info["sub"], ses_info["ses"])
        if df is not None:
            scores = score_frames_by_difficulty(df, target_bps)
            session_scores[ses_info["exp_id"]] = float(np.mean(scores))

    # Sort by difficulty (hardest first)
    sessions.sort(
        key=lambda s: session_scores.get(s["exp_id"], 0), reverse=True
    )

    # Allocate: harder sessions get more frames
    remaining = args.n
    allocations: dict[str, int] = {}
    for ses_info in sessions:
        if remaining <= 0:
            break
        alloc = min(args.per_session_max, remaining)
        allocations[ses_info["exp_id"]] = alloc
        remaining -= alloc

    log.info("Frame allocation across %d sessions:", len(allocations))
    for eid, n in allocations.items():
        diff = session_scores.get(eid, 0)
        log.info("  %s: %d frames (difficulty=%.3f)", eid[:25], n, diff)

    # Process sessions
    total_selected = 0
    for ses_info in sessions:
        n = allocations.get(ses_info["exp_id"], 0)
        if n == 0:
            continue
        log.info("\n=== %s ===", ses_info["exp_id"])
        result = process_session(s3, ses_info, n, target_bps, args.dry_run)
        total_selected += result["n_selected"]

    print(f"\nTotal frames selected: {total_selected}")
    if not args.dry_run:
        print(
            "\nNext steps:\n"
            "  1. Label frames:  uv run python scripts/interactive_label.py\n"
            "  2. Upload labels: uv run python scripts/upload_dlc_labels.py\n"
            "  3. Retrain:       uv run python scripts/launch_dlc_finetune_ec2.py --sa-finetune"
        )


if __name__ == "__main__":
    main()
