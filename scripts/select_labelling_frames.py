#!/usr/bin/env python3
"""Add the N worst-tracked frames per session for DLC retraining.

For each session, downloads the pose .h5 from S3, scores every frame by
model uncertainty (weighted confidence, temporal jumps, unusual poses),
then selects the top N worst frames that are NOT similar to existing
labeled frames (image-based dedup + pose diversity).

Usage:
    uv run python scripts/select_labelling_frames.py --extra 10 --dry-run
    uv run python scripts/select_labelling_frames.py --extra 10
    uv run python scripts/select_labelling_frames.py --extra 10 --session 20210823_16_59_50_1114353
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
RAWDATA_BUCKET = "hm2p-rawdata"
METADATA_PATH = Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"
RETRAIN_DIR = Path(__file__).resolve().parent.parent / "retrain_frames"
LABELED_DIR = (
    Path(__file__).resolve().parent.parent
    / "sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/labeled-data"
)


def get_sessions() -> list[dict]:
    sessions = []
    with open(METADATA_PATH) as f:
        for row in csv.DictReader(f):
            eid = row["exp_id"]
            parts = eid.split("_")
            sub = f"sub-{parts[-1]}"
            ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
            sessions.append({"exp_id": eid, "sub": sub, "ses": ses})
    return sessions


def load_pose_from_s3(s3, sub: str, ses: str) -> pd.DataFrame | None:
    """Download and load the DLC .h5 for a session."""
    prefix = f"pose/{sub}/{ses}/"
    resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=prefix, MaxKeys=20)
    h5_keys = [
        o["Key"] for o in resp.get("Contents", [])
        if o["Key"].endswith(".h5") and "filtered" not in o["Key"]
    ]
    if not h5_keys:
        return None

    # Prefer finetuned (Resnet/HrnetW32) over superanimal
    key = h5_keys[0]
    for k in h5_keys:
        if "Resnet" in k or "Hrnet" in k:
            key = k
            break

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        s3.download_file(DERIVATIVES_BUCKET, key, tmp.name)
        df = pd.read_hdf(tmp.name)
    return df


# Default keypoint weights — uniform (1.0 for all).
# Override with --bodypart-weights to target specific bodyparts.
DEFAULT_KEYPOINT_WEIGHTS: dict[str, float] = {
    "nose_tip": 1.0,
    "nose": 1.0,
    "left_ear": 1.0,
    "right_ear": 1.0,
    "head_midpoint": 1.0,
    "implant_base_rear": 1.0,  # legacy alias
    "neck": 1.0,
    "mid_back": 1.0,
    "mouse_center": 1.0,
    "tail_base": 1.0,
}

# Active weights — set from CLI or defaults. Module-level so score_frames
# can access without threading the argument through every function.
KEYPOINT_WEIGHTS: dict[str, float] = dict(DEFAULT_KEYPOINT_WEIGHTS)


def score_frames(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Score each frame by how much the model struggles.

    Higher score = worse tracking = better candidate for labelling.
    Combines:
    1. Weighted low confidence (HD-critical keypoints weighted 2-3x)
    2. Temporal jumps (prediction inconsistency)
    3. Unusual posture (body spread deviating from median)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        (scores, positions_flat, bodypart_names)
        positions_flat is (N, K*2) for diversity filtering.
    """
    scorer = df.columns.get_level_values(0)[0]

    # Handle multi-animal format
    if df.columns.nlevels == 4:
        individuals = df.columns.get_level_values(1).unique()
        bodyparts = list(df.columns.get_level_values(2).unique())
        ind = individuals[0]
        lk_cols, x_cols, y_cols, bp_names = [], [], [], []
        for bp in bodyparts:
            try:
                lk_cols.append(df[(scorer, ind, bp, "likelihood")].values)
                x_cols.append(df[(scorer, ind, bp, "x")].values)
                y_cols.append(df[(scorer, ind, bp, "y")].values)
                bp_names.append(bp)
            except KeyError:
                pass
    else:
        bodyparts = list(df.columns.get_level_values(1).unique())
        lk_cols, x_cols, y_cols, bp_names = [], [], [], []
        for bp in bodyparts:
            try:
                lk_cols.append(df[(scorer, bp, "likelihood")].values)
                x_cols.append(df[(scorer, bp, "x")].values)
                y_cols.append(df[(scorer, bp, "y")].values)
                bp_names.append(bp)
            except KeyError:
                pass

    if not lk_cols:
        return np.zeros(len(df)), None, []

    lk = np.column_stack(lk_cols)  # (N, K)
    x = np.column_stack(x_cols)
    y = np.column_stack(y_cols)

    # Build positions matrix for diversity filtering
    pos_cols = []
    for i in range(len(bp_names)):
        pos_cols.append(x_cols[i])
        pos_cols.append(y_cols[i])
    positions = np.column_stack(pos_cols) if pos_cols else None

    n = len(df)

    # 1. Weighted confidence score — HD-critical keypoints weighted higher
    weights = np.array([KEYPOINT_WEIGHTS.get(bp, 1.0) for bp in bp_names])
    weights = weights / weights.sum()  # normalise
    weighted_lk = np.nansum(lk * weights[np.newaxis, :], axis=1)
    conf_score = 1.0 - weighted_lk

    # 2. Temporal jump score
    dx = np.diff(x, axis=0, prepend=x[:1])
    dy = np.diff(y, axis=0, prepend=y[:1])
    displacement = np.sqrt(dx**2 + dy**2)
    median_disp = np.nanmedian(displacement, axis=0, keepdims=True)
    median_disp[median_disp < 1] = 1
    jump_score = np.nanmean(displacement / median_disp, axis=1)
    jump_score = np.clip(jump_score, 0, 10) / 10.0

    # 3. Unusual posture: body spread deviating from median
    # Captures grooming (compact), rearing (extended), sharp turns
    cx = np.nanmean(x, axis=1, keepdims=True)
    cy = np.nanmean(y, axis=1, keepdims=True)
    spread = np.nanmean(np.sqrt((x - cx)**2 + (y - cy)**2), axis=1)
    spread_norm = spread / (np.nanmedian(spread) + 1e-6)
    pose_score = np.abs(spread_norm - 1.0)
    pose_score = np.clip(pose_score, 0, 3) / 3.0

    # Combined score
    score = 0.5 * conf_score + 0.3 * jump_score + 0.2 * pose_score
    return score, positions, bp_names


def already_labelled_frames(session_tag: str) -> set[int]:
    """Get frame indices already labelled for this session."""
    json_path = Path(__file__).resolve().parent.parent / "metadata" / "retrain_frames" / f"{session_tag}.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        return set(data.get("frame_indices", []))
    return set()


def image_dedup(
    video_path: str,
    candidate_indices: list[int],
    n_target: int,
    thumb_size: int = 32,
    min_similarity: float = 0.95,
) -> list[int]:
    """Select visually diverse frames from a video using image similarity.

    Downscales each candidate frame to a small grayscale thumbnail and
    greedily selects frames that are sufficiently different from all
    already-selected frames (normalised cross-correlation < min_similarity).

    Parameters
    ----------
    video_path : str
        Path to the video file.
    candidate_indices : list[int]
        Frame indices to consider (should be pre-sorted by priority).
    n_target : int
        Number of frames to select.
    thumb_size : int
        Thumbnail edge size for comparison (default 32 = 32x32 grayscale).
    min_similarity : float
        Maximum normalised cross-correlation between selected frames.
        Default 0.95 rejects near-identical frames while keeping
        frames with moderate differences.

    Returns
    -------
    list[int]
        Selected frame indices.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return candidate_indices[:n_target]

    # Extract thumbnails for all candidates
    thumbs: dict[int, np.ndarray] = {}
    for idx in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        thumbs[idx] = thumb.astype(np.float32).ravel()
        # Normalise to zero mean, unit variance for NCC
        m = thumbs[idx].mean()
        s = thumbs[idx].std()
        if s > 1e-6:
            thumbs[idx] = (thumbs[idx] - m) / s
        else:
            thumbs[idx] = thumbs[idx] - m
    cap.release()

    # Greedy selection: pick candidates in priority order, reject if
    # too similar to any already-selected frame
    selected = []
    selected_thumbs: list[np.ndarray] = []
    for idx in candidate_indices:
        if len(selected) >= n_target:
            break
        if idx not in thumbs:
            continue
        t = thumbs[idx]
        too_similar = False
        for st in selected_thumbs:
            ncc = float(np.dot(t, st) / len(t))
            if ncc > min_similarity:
                too_similar = True
                break
        if not too_similar:
            selected.append(idx)
            selected_thumbs.append(t)

    return selected


def select_diverse(
    scores: np.ndarray,
    positions: np.ndarray | None,
    n: int,
    min_spacing: int = 30,
    min_position_dist: float = 50.0,
    exclude: set[int] | None = None,
    existing_positions: np.ndarray | None = None,
) -> list[int]:
    """Select top-N frames by score with spacing and diversity constraints.

    Diversity uses two criteria so frames differ in either location or pose:
    1. Centroid distance — mean body position in the arena
    2. Pose shape distance — body configuration after subtracting centroid

    A frame is rejected only if BOTH centroid and shape are too similar
    to an already-selected frame OR to any frame in existing_positions
    (the already-labeled set).

    Parameters
    ----------
    existing_positions : (M, K*2) float, optional
        Positions of already-labeled frames. New selections must also
        be diverse relative to these.
    """
    order = np.argsort(-scores)  # highest score first
    selected = []
    exclude = exclude or set()

    # Precompute centroids and pose shapes for diversity check
    _centroids = None
    _shapes = None
    if positions is not None:
        p = np.nan_to_num(positions.astype(np.float64), nan=0.0)
        n_kp = p.shape[1] // 2
        xs = p[:, 0::2]
        ys = p[:, 1::2]
        cx = np.mean(xs, axis=1, keepdims=True)
        cy = np.mean(ys, axis=1, keepdims=True)
        _centroids = np.column_stack([cx.ravel(), cy.ravel()])
        _shapes = np.column_stack([xs - cx, ys - cy])

    # Pre-seed with existing labeled frame centroids/shapes
    _existing_centroids: list[np.ndarray] = []
    _existing_shapes: list[np.ndarray] = []
    if existing_positions is not None and len(existing_positions) > 0:
        ep = np.nan_to_num(existing_positions.astype(np.float64), nan=0.0)
        n_kp_e = ep.shape[1] // 2
        ex = ep[:, 0::2]
        ey = ep[:, 1::2]
        ecx = np.mean(ex, axis=1, keepdims=True)
        ecy = np.mean(ey, axis=1, keepdims=True)
        for i in range(len(ep)):
            _existing_centroids.append(np.array([ecx[i, 0], ecy[i, 0]]))
            _existing_shapes.append(np.concatenate([ex[i] - ecx[i, 0], ey[i] - ecy[i, 0]]))

    def _too_similar(idx: int) -> bool:
        if _centroids is None:
            return False
        c = _centroids[idx]
        s = _shapes[idx]
        # Check against newly selected
        for si in selected:
            c_dist = np.sqrt(np.sum((c - _centroids[si]) ** 2))
            s_dist = np.mean(np.abs(s - _shapes[si]))
            if c_dist < min_position_dist and s_dist < min_position_dist:
                return True
        # Check against existing labeled frames
        for ec, es in zip(_existing_centroids, _existing_shapes):
            c_dist = np.sqrt(np.sum((c - ec) ** 2))
            s_dist = np.mean(np.abs(s - es))
            if c_dist < min_position_dist and s_dist < min_position_dist:
                return True
        return False

    for idx in order:
        if len(selected) >= n:
            break
        idx = int(idx)
        if idx in exclude:
            continue
        if any(abs(idx - s) < min_spacing for s in selected):
            continue
        if _too_similar(idx):
            continue
        selected.append(idx)

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Add the N worst-tracked frames per session for DLC retraining."
    )
    parser.add_argument("--extra", type=int, default=10,
                        help="Number of extra frames to add per session (default 10).")
    parser.add_argument("--min-spacing", type=int, default=30,
                        help="Min frame spacing within a session")
    parser.add_argument("--session", type=str, default=None,
                        help="Process a single session by exp_id.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show selection without extracting frames")
    parser.add_argument("--label", action="store_true",
                        help="Extract frames AND open napari for each session")
    parser.add_argument(
        "--bodypart-weights", type=str, default=None,
        help="Override bodypart weights for scoring (comma-separated bp:weight). "
        "E.g. --bodypart-weights nose_tip:3,left_ear:2. "
        "Unspecified bodyparts default to 1.0.",
    )
    args = parser.parse_args()

    # Apply bodypart weight overrides
    global KEYPOINT_WEIGHTS
    if args.bodypart_weights:
        KEYPOINT_WEIGHTS = dict(DEFAULT_KEYPOINT_WEIGHTS)
        for item in args.bodypart_weights.split(","):
            bp, w = item.strip().split(":")
            KEYPOINT_WEIGHTS[bp.strip()] = float(w.strip())
        print(f"  Bodypart weights: {KEYPOINT_WEIGHTS}")

    s3 = boto3.client("s3", region_name=REGION)
    sessions = get_sessions()

    if args.session:
        sessions = [s for s in sessions if s["exp_id"] == args.session]
        if not sessions:
            print(f"Session {args.session!r} not found.")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Adding {args.extra} worst frames per session")
    print(f"  Sessions: {len(sessions)}   Dry-run: {args.dry_run}")
    print(f"{'='*60}\n")

    all_selected = {}
    total = 0

    for i, ses_info in enumerate(sessions, 1):
        sub, ses = ses_info["sub"], ses_info["ses"]
        session_tag = f"{sub}_{ses}"
        exp_id = ses_info["exp_id"]

        df = load_pose_from_s3(s3, sub, ses)
        if df is None:
            print(f"  [{i}/{len(sessions)}] {exp_id}: no pose data, skipping")
            continue

        scores, positions, bp_names = score_frames(df)
        already = already_labelled_frames(session_tag)

        # Load existing labeled positions for diversity check
        existing_pos = None
        import re as _re
        for ld in LABELED_DIR.iterdir():
            if not ld.is_dir():
                continue
            ses_date = ses.replace("ses-", "").split("T")[0]
            animal = sub.replace("sub-", "")
            if ses_date in ld.name and animal in ld.name:
                h5 = ld / "CollectedData_tristan.h5"
                if h5.exists():
                    try:
                        gt = pd.read_hdf(h5)
                        if len(gt) > 0 and positions is not None:
                            labeled_indices = []
                            for idx in gt.index:
                                ff = idx[2] if isinstance(idx, tuple) else str(idx).split("/")[-1]
                                m = _re.match(r"frame_(\d+)\.png", ff)
                                if m:
                                    fi = int(m.group(1))
                                    if fi < len(positions):
                                        labeled_indices.append(fi)
                            if labeled_indices:
                                existing_pos = positions[labeled_indices]
                    except Exception:
                        pass
                break

        # Get 3x candidates for dedup headroom
        candidates = select_diverse(
            scores, positions, args.extra * 3,
            min_spacing=args.min_spacing,
            exclude=already,
            existing_positions=existing_pos,
        )

        # Trim to target
        selected = candidates[:args.extra]

        score_strs = [f"{scores[i]:.3f}" for i in selected[:5]]
        print(f"  [{i}/{len(sessions)}] {exp_id}: existing={len(already)}, "
              f"adding={len(selected)}, top scores={score_strs}")

        if selected:
            all_selected[session_tag] = {
                "exp_id": exp_id, "sub": sub, "ses": ses,
                "frames": selected,
            }
            total += len(selected)

    print(f"\nTotal: {total} new frames across {len(all_selected)} sessions")

    # Save selection
    output_path = Path("retrain_frames/_next_batch.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(all_selected, indent=2, default=str))
    print(f"Saved to {output_path}")

    if args.dry_run:
        print("\n[DRY RUN] — no frames extracted.")
        return

    # Extract frames
    import subprocess
    import shutil
    import cv2

    print(f"\nExtracting {total} frames...\n")

    for i, (tag, info) in enumerate(all_selected.items(), 1):
        sub, ses, exp_id = info["sub"], info["ses"], info["exp_id"]
        print(f"[{i}/{len(all_selected)}] {exp_id}")

        rf_dir = RETRAIN_DIR / tag
        rf_dir.mkdir(parents=True, exist_ok=True)

        # Download video
        s3_prefix = f"rawdata/{sub}/{ses}/behav/"
        resp = s3.list_objects_v2(Bucket=RAWDATA_BUCKET, Prefix=s3_prefix, MaxKeys=20)
        video_path = None
        video_dir = Path(f"/tmp/{tag}")
        video_dir.mkdir(parents=True, exist_ok=True)
        for obj in resp.get("Contents", []):
            fn = obj["Key"].split("/")[-1]
            if fn.endswith(".mp4") and "side" not in fn.lower():
                local = video_dir / fn
                if not local.exists():
                    s3.download_file(RAWDATA_BUCKET, obj["Key"], str(local))
                if "overhead" in fn or "cropped" in fn:
                    video_path = local
                elif video_path is None:
                    video_path = local

        if video_path is None:
            print(f"  No video found, skipping")
            continue

        # Image dedup against existing PNGs on disk
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
            from hm2p.pose.dedup import filter_duplicates_against_existing
            before = len(info["frames"])
            info["frames"] = filter_duplicates_against_existing(
                str(video_path), info["frames"], rf_dir,
            )[:args.extra]
            removed = before - len(info["frames"])
            if removed:
                print(f"  Dedup removed {removed} similar frames")
        except Exception as e:
            print(f"  Dedup failed: {e}")
            info["frames"] = info["frames"][:args.extra]

        # Extract PNGs
        cap = cv2.VideoCapture(str(video_path))
        written = 0
        for idx in sorted(info["frames"]):
            out = rf_dir / f"frame_{int(idx):06d}.png"
            if out.exists():
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(str(out), frame)
                written += 1
        cap.release()
        if written:
            print(f"  Extracted {written} PNGs")

        # Symlink into labeled-data
        for ld in LABELED_DIR.iterdir():
            if not ld.is_dir():
                continue
            ses_date = ses.replace("ses-", "").split("T")[0]
            animal = sub.replace("sub-", "")
            if ses_date in ld.name and animal in ld.name:
                import os
                linked = 0
                for png in sorted(rf_dir.glob("frame_*.png")):
                    dest = ld / png.name
                    if not dest.exists():
                        rel = os.path.relpath(png.resolve(), ld.resolve())
                        dest.symlink_to(rel)
                        linked += 1
                if linked:
                    print(f"  Symlinked {linked} into labeled-data/")
                break

        # Update metadata
        meta_dir = Path("metadata/retrain_frames")
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_file = meta_dir / f"{tag}.json"
        existing_indices = set()
        if meta_file.exists():
            existing_data = json.loads(meta_file.read_text())
            existing_indices = set(existing_data.get("frame_indices", []))
        all_indices = sorted(existing_indices | set(info["frames"]))
        meta_file.write_text(json.dumps({
            "session": f"{sub}/{ses}",
            "frame_indices": all_indices,
            "n_frames": len(all_indices),
        }, indent=2))

        if args.label:
            cmd = [
                sys.executable, "scripts/interactive_label.py",
                "--session", exp_id,
            ]
            subprocess.run(cmd)

    print(f"\n{'='*60}")
    print(f"Done! Added {total} frames across {len(all_selected)} sessions.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
