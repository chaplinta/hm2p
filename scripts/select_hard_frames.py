#!/usr/bin/env python3
"""Select diverse, hard-to-track frames for targeted DLC retraining.

Builds a pose-geometry feature space from the model's predictions
(body orientation, elongation, head angle, inter-ear distance, arena
position, per-bodypart confidence) and uses greedy farthest-point
sampling with difficulty weighting to select frames that maximally
cover the appearance/pose space while emphasising hard cases.

Safety: existing CollectedData_*.csv/.h5 files are never modified.
New frames are extracted as PNGs and symlinked into the DLC
labeled-data directory. The user then labels them in napari-deeplabcut.

Usage:
    # Scan all sessions, show labeling status + difficulty:
    uv run python scripts/select_hard_frames.py --scan

    # Select 200 diverse hard frames across all sessions:
    uv run python scripts/select_hard_frames.py --n 200

    # Select from one session:
    uv run python scripts/select_hard_frames.py --n 20 \\
        --session 20210823

    # Adjust difficulty vs diversity balance (0=pure diversity, 1=pure difficulty):
    uv run python scripts/select_hard_frames.py --n 200 --alpha 0.3

    # Dry run:
    uv run python scripts/select_hard_frames.py --n 200 --dry-run
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

# Minimum temporal spacing between selected frames (in pose-file frame units).
# At 30 fps this is ~1 second.
MIN_FRAME_SPACING = 30


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
# Pose loading
# ---------------------------------------------------------------------------


def load_pose_from_s3(s3: Any, sub: str, ses: str) -> pd.DataFrame | None:
    """Download the best DLC H5 file from S3 and return as DataFrame."""
    from hm2p.pose.select import select_best_dlc_h5_s3

    prefix = f"pose/{sub}/{ses}/"
    h5_key = select_best_dlc_h5_s3(s3, DERIVATIVES_BUCKET, prefix)
    if h5_key is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        s3.download_file(DERIVATIVES_BUCKET, h5_key, tmp.name)
        return pd.read_hdf(tmp.name)


def _resolve_bp(df: pd.DataFrame, scorer: str, bp: str) -> str | None:
    """Resolve bodypart name, handling head_midpoint/implant_base_rear alias."""
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
# Pose-geometry feature extraction
# ---------------------------------------------------------------------------


def build_pose_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build pose-geometry feature matrix and difficulty scores from DLC output.

    Features per frame (all position-independent except centroid):
    - body_orientation_sin, body_orientation_cos (from tail→nose axis)
    - body_elongation (spread along vs across body axis)
    - head_body_angle (head turn relative to body, radians)
    - inter_ear_distance (px, normalised)
    - centroid_x, centroid_y (normalised to [0,1])
    - per-bodypart confidence (8 values)
    - mean_brightness_proxy (mean of all x-coords modulo — crude light proxy)

    Returns (features, difficulty_scores) both shape (n_frames,).
    """
    scorer = df.columns.get_level_values(0)[0]
    n = len(df)

    # Extract raw coords and likelihoods
    coords = {}  # bp -> (x, y, lik) arrays
    for bp in ALL_BODYPARTS:
        col = _resolve_bp(df, scorer, bp)
        if col is None:
            coords[bp] = (np.full(n, np.nan), np.full(n, np.nan), np.zeros(n))
            continue
        coords[bp] = (
            df[(scorer, col, "x")].values.astype(np.float64),
            df[(scorer, col, "y")].values.astype(np.float64),
            df[(scorer, col, "likelihood")].values.astype(np.float64),
        )

    # --- Geometric features ---

    # Body axis: tail_base → nose_tip
    tx, ty, _ = coords["tail_base"]
    nx, ny, _ = coords["nose_tip"]
    body_dx = nx - tx
    body_dy = ny - ty
    body_angle = np.arctan2(body_dy, body_dx)
    body_orient_sin = np.sin(body_angle)
    body_orient_cos = np.cos(body_angle)

    # Body elongation: spread along body axis vs perpendicular
    all_x = np.column_stack([coords[bp][0] for bp in ALL_BODYPARTS])
    all_y = np.column_stack([coords[bp][1] for bp in ALL_BODYPARTS])
    centroid_x = np.nanmean(all_x, axis=1)
    centroid_y = np.nanmean(all_y, axis=1)

    # Project all keypoints onto body axis and perpendicular
    dx_from_c = all_x - centroid_x[:, None]
    dy_from_c = all_y - centroid_y[:, None]
    cos_a = np.cos(body_angle)[:, None]
    sin_a = np.sin(body_angle)[:, None]
    proj_along = dx_from_c * cos_a + dy_from_c * sin_a
    proj_perp = -dx_from_c * sin_a + dy_from_c * cos_a
    spread_along = np.nanstd(proj_along, axis=1)
    spread_perp = np.nanstd(proj_perp, axis=1)
    elongation = spread_along / (spread_perp + 1e-6)

    # Head-body angle: angle between (tail→neck) and (neck→nose)
    nkx, nky, _ = coords["neck"]
    vec_body_x = nkx - tx
    vec_body_y = nky - ty
    vec_head_x = nx - nkx
    vec_head_y = ny - nky
    body_ang = np.arctan2(vec_body_y, vec_body_x)
    head_ang = np.arctan2(vec_head_y, vec_head_x)
    head_body_angle = np.arctan2(
        np.sin(head_ang - body_ang), np.cos(head_ang - body_ang)
    )

    # Inter-ear distance
    lex, ley, _ = coords["left_ear"]
    rex, rey, _ = coords["right_ear"]
    inter_ear = np.sqrt((lex - rex) ** 2 + (ley - rey) ** 2)
    # Normalise by median (typical ear distance)
    ear_median = np.nanmedian(inter_ear)
    inter_ear_norm = inter_ear / (ear_median + 1e-6)

    # Normalise centroid to [0, 1]
    cx_min, cx_max = np.nanmin(centroid_x), np.nanmax(centroid_x)
    cy_min, cy_max = np.nanmin(centroid_y), np.nanmax(centroid_y)
    cx_norm = (centroid_x - cx_min) / (cx_max - cx_min + 1e-6)
    cy_norm = (centroid_y - cy_min) / (cy_max - cy_min + 1e-6)

    # Per-bodypart confidence
    conf_cols = []
    for bp in ALL_BODYPARTS:
        conf_cols.append(coords[bp][2])
    conf_matrix = np.column_stack(conf_cols)  # (n, 8)

    # --- Assemble feature matrix ---
    features = np.column_stack([
        body_orient_sin,       # 0
        body_orient_cos,       # 1
        elongation,            # 2
        head_body_angle,       # 3
        inter_ear_norm,        # 4
        cx_norm,               # 5
        cy_norm,               # 6
        conf_matrix,           # 7-14
    ])  # shape (n, 15)

    # Replace NaN with column median
    for col_i in range(features.shape[1]):
        col = features[:, col_i]
        nan_mask = ~np.isfinite(col)
        if nan_mask.any():
            median_val = np.nanmedian(col)
            if not np.isfinite(median_val):
                median_val = 0.0
            col[nan_mask] = median_val

    # Standardise each feature to zero mean, unit variance
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    stds[stds < 1e-8] = 1.0
    features = (features - means) / stds

    # --- Difficulty scores ---
    # Weighted confidence deficit: HD-critical bodyparts weighted higher
    weights = {
        "nose_tip": 2.0, "left_ear": 3.0, "right_ear": 3.0,
        "head_midpoint": 1.5, "neck": 1.0, "mid_back": 0.5,
        "mouse_center": 0.5, "tail_base": 1.0,
    }
    difficulty = np.zeros(n, dtype=np.float64)
    for i, bp in enumerate(ALL_BODYPARTS):
        w = weights.get(bp, 1.0)
        difficulty += w * (1.0 - np.clip(conf_matrix[:, i], 0, 1)) ** 2

    # Normalise to [0, 1]
    d_max = difficulty.max()
    if d_max > 0:
        difficulty /= d_max

    return features, difficulty


# ---------------------------------------------------------------------------
# Greedy farthest-point sampling with difficulty weighting
# ---------------------------------------------------------------------------


def greedy_diverse_select(
    features: np.ndarray,
    difficulty: np.ndarray,
    n_select: int,
    existing_indices: set[int],
    alpha: float = 0.3,
    min_spacing: int = MIN_FRAME_SPACING,
) -> list[int]:
    """Select diverse, hard frames via greedy farthest-point sampling.

    At each step, picks the frame that maximises:
        score_i = alpha * difficulty_i + (1 - alpha) * min_dist_to_selected(i)

    This balances difficulty (alpha) against diversity (1 - alpha).

    Parameters
    ----------
    features : np.ndarray, shape (n_frames, n_features)
        Standardised pose-geometry feature matrix.
    difficulty : np.ndarray, shape (n_frames,)
        Difficulty score per frame, in [0, 1].
    n_select : int
        Number of frames to select.
    existing_indices : set[int]
        Frame indices already labeled (excluded + used as initial reference).
    alpha : float
        Difficulty vs diversity trade-off. 0 = pure diversity, 1 = pure difficulty.
    min_spacing : int
        Minimum frame index spacing between selected frames.

    Returns
    -------
    list[int]
        Selected frame indices.
    """
    n = len(features)
    available = np.ones(n, dtype=bool)

    # Exclude already-labeled frames and their temporal neighbourhood
    for idx in existing_indices:
        lo = max(0, idx - min_spacing)
        hi = min(n, idx + min_spacing + 1)
        available[lo:hi] = False

    # Track minimum distance to any selected/existing frame
    min_dist = np.full(n, np.inf, dtype=np.float64)

    # Initialise distances from existing labeled frames
    for idx in existing_indices:
        if 0 <= idx < n:
            dists = np.linalg.norm(features - features[idx], axis=1)
            min_dist = np.minimum(min_dist, dists)

    # Normalise min_dist to [0, 1] for combining with difficulty
    def _norm_dists() -> np.ndarray:
        d = min_dist.copy()
        d[~available] = -1
        dmax = d[available].max() if available.any() else 1.0
        if dmax > 0:
            d /= dmax
        d[~available] = -1
        return d

    selected: list[int] = []

    for _ in range(n_select):
        if not available.any():
            break

        norm_d = _norm_dists()

        # Combined score
        scores = alpha * difficulty + (1 - alpha) * norm_d
        scores[~available] = -np.inf

        best = int(np.argmax(scores))
        if scores[best] == -np.inf:
            break

        selected.append(best)

        # Mark temporal neighbourhood as unavailable
        lo = max(0, best - min_spacing)
        hi = min(n, best + min_spacing + 1)
        available[lo:hi] = False

        # Update min distances
        dists = np.linalg.norm(features - features[best], axis=1)
        min_dist = np.minimum(min_dist, dists)

    return selected


# ---------------------------------------------------------------------------
# Session discovery & metadata (unchanged)
# ---------------------------------------------------------------------------


def already_labelled_frames(session_tag: str) -> set[int]:
    """Return frame indices already selected for a session."""
    json_path = RETRAIN_META_DIR / f"{session_tag}.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        return set(data.get("frame_indices", []))
    return set()


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


# ---------------------------------------------------------------------------
# Video download & frame extraction (unchanged)
# ---------------------------------------------------------------------------


def download_video_from_s3(s3: Any, sub: str, ses: str, dest_dir: Path) -> Path | None:
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


def extract_frames(video_path: Path, frame_indices: list[int], dest_dir: Path) -> list[int]:
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


def symlink_into_labeled_data(retrain_dir: Path, labeled_dir: Path) -> int:
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


def update_meta(session_tag: str, sub: str, ses: str, new_indices: list[int],
                video_name: str | None = None) -> None:
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


def scan_sessions(s3: Any) -> list[dict]:
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
                "mean_difficulty": 0,
            })
            continue
        _, difficulty = build_pose_features(df)
        results.append({
            **ses_info, "tag": tag,
            "n_existing": len(existing), "n_frames": len(df),
            "mean_difficulty": float(np.mean(difficulty)),
        })
    return results


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------


def process_session(
    s3: Any,
    ses_info: dict,
    n_select: int,
    alpha: float,
    dry_run: bool,
) -> dict:
    """Select diverse hard frames for one session."""
    sub, ses = ses_info["sub"], ses_info["ses"]
    tag = f"{sub}_{ses}"
    exp_id = ses_info["exp_id"]

    existing = already_labelled_frames(tag)
    log.info("  %s: %d existing labels", exp_id[:25], len(existing))

    df = load_pose_from_s3(s3, sub, ses)
    if df is None:
        log.warning("  No pose data for %s", exp_id)
        return {"exp_id": exp_id, "n_selected": 0, "n_existing": len(existing)}

    features, difficulty = build_pose_features(df)

    selected = greedy_diverse_select(
        features, difficulty, n_select, existing, alpha=alpha,
    )
    log.info("  Selected %d diverse frames (alpha=%.2f)", len(selected), alpha)

    if dry_run or not selected:
        return {
            "exp_id": exp_id, "n_selected": len(selected),
            "n_existing": len(existing), "selected": selected,
        }

    # Download video for frame extraction
    with tempfile.TemporaryDirectory(prefix=f"hm2p-hard-{tag}-") as tmp_str:
        tmp = Path(tmp_str)
        video_path = download_video_from_s3(s3, sub, ses, tmp)
        if video_path is None:
            log.warning("  No video for %s", exp_id)
            return {"exp_id": exp_id, "n_selected": 0, "n_existing": len(existing)}

        retrain_session_dir = RETRAIN_DIR / tag
        extract_frames(video_path, selected, retrain_session_dir)

        labeled_dir = find_labeled_data_dir(sub, ses)
        if labeled_dir is None:
            clip_name = f"{exp_id}_maze-rose_overhead.camera-cropped"
            labeled_dir = LABELED_DIR / clip_name
        n_linked = symlink_into_labeled_data(retrain_session_dir, labeled_dir)
        log.info("  Symlinked %d new frames into %s/", n_linked, labeled_dir.name)

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
        description="Select diverse hard-to-track frames for DLC retraining."
    )
    parser.add_argument(
        "--n", type=int, default=200,
        help="Total frames to select (default 200).",
    )
    parser.add_argument(
        "--scan", action="store_true",
        help="Scan sessions and show labeling status.",
    )
    parser.add_argument(
        "--session", type=str, default=None,
        help="Process only this session (exp_id or partial match).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.3,
        help="Difficulty vs diversity trade-off. 0=pure diversity, "
             "1=pure difficulty (default 0.3).",
    )
    parser.add_argument(
        "--per-session-max", type=int, default=12,
        help="Max frames per session (default 12).",
    )
    parser.add_argument(
        "--per-session-min", type=int, default=4,
        help="Min frames per session (default 4).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be selected without extracting.",
    )
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    sessions = get_sessions()

    if args.scan:
        print("\nScanning sessions for labeling status...\n")
        results = scan_sessions(s3)
        results.sort(key=lambda r: r.get("mean_difficulty", 0), reverse=True)

        print(f"{'Session':<25s}  {'Labels':>6s}  {'Frames':>7s}  {'Difficulty':>10s}")
        print("-" * 55)
        for r in results:
            print(f"{r['exp_id'][:25]:<25s}  {r['n_existing']:>6d}  "
                  f"{r['n_frames']:>7d}  {r['mean_difficulty']:>10.3f}")
        total_labels = sum(r["n_existing"] for r in results)
        print(f"\nTotal labeled frames: {total_labels}")
        return

    if args.session:
        sessions = [s for s in sessions if args.session in s["exp_id"]]
        if not sessions:
            print(f"No session matching '{args.session}'")
            sys.exit(1)

    # Score all sessions by difficulty for allocation
    log.info("Loading pose data to score sessions...")
    session_difficulty: dict[str, float] = {}
    for ses_info in sessions:
        df = load_pose_from_s3(s3, ses_info["sub"], ses_info["ses"])
        if df is not None:
            _, diff = build_pose_features(df)
            session_difficulty[ses_info["exp_id"]] = float(np.mean(diff))

    # Sort by difficulty
    sessions.sort(key=lambda s: session_difficulty.get(s["exp_id"], 0), reverse=True)

    # Allocate: every session gets min, then top up by difficulty
    remaining = args.n
    allocations: dict[str, int] = {}
    # First pass: everyone gets min
    for ses_info in sessions:
        if remaining <= 0:
            break
        alloc = min(args.per_session_min, remaining)
        allocations[ses_info["exp_id"]] = alloc
        remaining -= alloc
    # Second pass: top up hardest sessions
    for ses_info in sessions:
        if remaining <= 0:
            break
        eid = ses_info["exp_id"]
        current = allocations.get(eid, 0)
        extra = min(args.per_session_max - current, remaining)
        if extra > 0:
            allocations[eid] = current + extra
            remaining -= extra

    log.info("Frame allocation across %d sessions:", len(allocations))
    for eid, n in allocations.items():
        log.info("  %s: %d frames (difficulty=%.3f)", eid[:25], n,
                 session_difficulty.get(eid, 0))

    # Process sessions
    total_selected = 0
    for ses_info in sessions:
        n = allocations.get(ses_info["exp_id"], 0)
        if n == 0:
            continue
        log.info("\n=== %s ===", ses_info["exp_id"])
        result = process_session(s3, ses_info, n, args.alpha, args.dry_run)
        total_selected += result["n_selected"]

    print(f"\nTotal frames selected: {total_selected}")
    if not args.dry_run and total_selected > 0:
        print(
            "\nNext steps:\n"
            "  1. Label frames:  uv run python scripts/interactive_label.py\n"
            "  2. Upload labels: uv run python scripts/upload_dlc_labels.py\n"
            "  3. Retrain:       uv run python scripts/launch_dlc_finetune_ec2.py --sa-finetune"
        )


if __name__ == "__main__":
    main()
