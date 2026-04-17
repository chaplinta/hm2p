#!/usr/bin/env python3
"""Select the best frames to label across all sessions for DLC retraining.

Downloads pose .h5 files from S3, finds frames where the model struggles
(low confidence, temporal jumps, unusual poses), and selects a diverse
set across sessions ensuring no near-duplicates.

Usage:
    uv run python scripts/select_labelling_frames.py             # 60 frames
    uv run python scripts/select_labelling_frames.py --n 100     # 100 frames
    uv run python scripts/select_labelling_frames.py --dry-run   # show selection without extracting
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


def score_frames(df: pd.DataFrame) -> np.ndarray:
    """Score each frame by how much the model struggles.

    Higher score = worse tracking = better candidate for labelling.
    Combines:
    1. Low mean confidence (model uncertain)
    2. Temporal jumps (prediction inconsistency)
    3. Position variance (unusual poses tend to have scattered predictions)
    """
    scorer = df.columns.get_level_values(0)[0]

    # Handle multi-animal format
    if df.columns.nlevels == 4:
        # Pick best individual per frame
        individuals = df.columns.get_level_values(1).unique()
        bodyparts = df.columns.get_level_values(2).unique()
        # Use first individual for simplicity
        ind = individuals[0]
        lk_cols = []
        x_cols = []
        y_cols = []
        for bp in bodyparts:
            try:
                lk_cols.append(df[(scorer, ind, bp, "likelihood")].values)
                x_cols.append(df[(scorer, ind, bp, "x")].values)
                y_cols.append(df[(scorer, ind, bp, "y")].values)
            except KeyError:
                pass
    else:
        bodyparts = df.columns.get_level_values(1).unique()
        lk_cols = []
        x_cols = []
        y_cols = []
        for bp in bodyparts:
            try:
                lk_cols.append(df[(scorer, bp, "likelihood")].values)
                x_cols.append(df[(scorer, bp, "x")].values)
                y_cols.append(df[(scorer, bp, "y")].values)
            except KeyError:
                pass

    if not lk_cols:
        return np.zeros(len(df))

    lk = np.column_stack(lk_cols)  # (N, K)
    x = np.column_stack(x_cols)
    y = np.column_stack(y_cols)

    n = len(df)

    # 1. Low confidence score (inverted mean likelihood)
    mean_lk = np.nanmean(lk, axis=1)
    conf_score = 1.0 - mean_lk

    # 2. Temporal jump score: how much bodyparts move frame-to-frame
    dx = np.diff(x, axis=0, prepend=x[:1])
    dy = np.diff(y, axis=0, prepend=y[:1])
    displacement = np.sqrt(dx**2 + dy**2)
    # Normalise by median displacement (so stationary periods don't dominate)
    median_disp = np.nanmedian(displacement, axis=0, keepdims=True)
    median_disp[median_disp < 1] = 1
    jump_score = np.nanmean(displacement / median_disp, axis=1)
    # Clip extreme values
    jump_score = np.clip(jump_score, 0, 10) / 10.0

    # 3. Pose diversity: variance of bodypart positions relative to centroid
    cx = np.nanmean(x, axis=1, keepdims=True)
    cy = np.nanmean(y, axis=1, keepdims=True)
    spread = np.nanmean(np.sqrt((x - cx)**2 + (y - cy)**2), axis=1)
    # Normalise
    spread_norm = spread / (np.nanmedian(spread) + 1e-6)
    # Unusual poses have very high or very low spread
    pose_score = np.abs(spread_norm - 1.0)
    pose_score = np.clip(pose_score, 0, 3) / 3.0

    # Combined score (weighted)
    score = 0.5 * conf_score + 0.3 * jump_score + 0.2 * pose_score
    return score


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
) -> list[int]:
    """Select top-N frames by score with spacing and diversity constraints.

    Diversity uses two criteria so frames differ in either location or pose:
    1. Centroid distance — mean body position in the arena
    2. Pose shape distance — body configuration after subtracting centroid

    A frame is rejected only if BOTH centroid and shape are too similar
    to an already-selected frame.
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

    for idx in order:
        if len(selected) >= n:
            break
        idx = int(idx)
        if idx in exclude:
            continue
        # Spacing constraint
        if any(abs(idx - s) < min_spacing for s in selected):
            continue
        # Diversity constraint: reject if both location AND pose are similar
        if _centroids is not None and selected:
            too_similar = False
            for s in selected:
                c_dist = np.sqrt(np.sum((_centroids[idx] - _centroids[s]) ** 2))
                s_dist = np.mean(np.abs(_shapes[idx] - _shapes[s]))
                if c_dist < min_position_dist and s_dist < min_position_dist:
                    too_similar = True
                    break
            if too_similar:
                continue
        selected.append(idx)

    return selected


def main():
    parser = argparse.ArgumentParser(description="Select best frames for DLC labelling")
    parser.add_argument("--n", type=int, default=60, help="Total frames to select")
    parser.add_argument("--per-session-max", type=int, default=8,
                        help="Max frames per session")
    parser.add_argument("--per-session-min", type=int, default=2,
                        help="Min frames per session (if session has data)")
    parser.add_argument("--min-spacing", type=int, default=30,
                        help="Min frame spacing within a session")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show selection without extracting frames")
    parser.add_argument("--label", action="store_true",
                        help="Extract frames AND open napari for each session")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    sessions = get_sessions()
    print(f"Scanning {len(sessions)} sessions for frame selection...")

    # Score all sessions
    session_scores = []
    for ses_info in sessions:
        sub, ses = ses_info["sub"], ses_info["ses"]
        session_tag = f"{sub}_{ses}"

        df = load_pose_from_s3(s3, sub, ses)
        if df is None:
            print(f"  {ses_info['exp_id']}: no pose data, skipping")
            continue

        scores = score_frames(df)
        mean_score = float(np.nanmean(scores))
        already = already_labelled_frames(session_tag)

        session_scores.append({
            "exp_id": ses_info["exp_id"],
            "sub": sub,
            "ses": ses,
            "tag": session_tag,
            "n_frames": len(df),
            "mean_score": mean_score,
            "scores": scores,
            "n_already_labelled": len(already),
            "already_labelled": already,
            "df": df,
        })
        print(f"  {ses_info['exp_id']}: {len(df)} frames, mean_score={mean_score:.3f}, "
              f"already_labelled={len(already)}")

    # Sort sessions by mean score (worst first), then by fewest existing labels
    session_scores.sort(key=lambda s: (-s["mean_score"], s["n_already_labelled"]))

    # Allocate frames across sessions
    # First pass: give each session its minimum
    # Second pass: fill remaining budget from worst sessions
    remaining = args.n
    allocations = {}

    for s in session_scores:
        n_alloc = min(args.per_session_min, remaining)
        allocations[s["tag"]] = n_alloc
        remaining -= n_alloc
        if remaining <= 0:
            break

    # Second pass: distribute remaining to worst sessions
    for s in session_scores:
        if remaining <= 0:
            break
        current = allocations.get(s["tag"], 0)
        extra = min(args.per_session_max - current, remaining)
        if extra > 0:
            allocations[s["tag"]] = current + extra
            remaining -= extra

    # Select specific frames per session
    print(f"\n{'='*60}")
    print(f"Selecting {args.n} frames across {len(allocations)} sessions")
    print(f"{'='*60}\n")

    all_selected = {}
    total = 0

    for s in session_scores:
        tag = s["tag"]
        n_alloc = allocations.get(tag, 0)
        if n_alloc == 0:
            continue

        # Build position matrix for similarity check
        scorer = s["df"].columns.get_level_values(0)[0]
        if s["df"].columns.nlevels == 4:
            ind = s["df"].columns.get_level_values(1).unique()[0]
            bodyparts = s["df"].columns.get_level_values(2).unique()
            pos_cols = []
            for bp in bodyparts:
                try:
                    pos_cols.append(s["df"][(scorer, ind, bp, "x")].values)
                    pos_cols.append(s["df"][(scorer, ind, bp, "y")].values)
                except KeyError:
                    pass
        else:
            bodyparts = s["df"].columns.get_level_values(1).unique()
            pos_cols = []
            for bp in bodyparts:
                try:
                    pos_cols.append(s["df"][(scorer, bp, "x")].values)
                    pos_cols.append(s["df"][(scorer, bp, "y")].values)
                except KeyError:
                    pass

        positions = np.column_stack(pos_cols) if pos_cols else None

        # Get 3x candidates using pose-based diversity, then image_dedup
        # will prune to n_alloc using actual pixel similarity.
        candidates = select_diverse(
            s["scores"], positions, n_alloc * 3,
            min_spacing=args.min_spacing,
            exclude=s["already_labelled"],
        )

        all_selected[tag] = {
            "exp_id": s["exp_id"],
            "sub": s["sub"],
            "ses": s["ses"],
            "frames": candidates,  # will be pruned by image_dedup during extraction
            "n_target": n_alloc,
        }
        total += min(n_alloc, len(candidates))

        scores_at_selected = [f"{s['scores'][i]:.3f}" for i in candidates[:n_alloc]]
        print(f"{s['exp_id']}: {len(candidates)} candidates → target {n_alloc} "
              f"(already labelled: {s['n_already_labelled']})")
        print(f"  Top indices: {candidates[:n_alloc]}")
        print(f"  Top scores:  {scores_at_selected}")
        print()

    print(f"Total: {total} frames across {len(all_selected)} sessions")

    # Save selection as JSON
    output_path = Path("retrain_frames/_next_batch.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(all_selected, indent=2, default=str))
    print(f"\nSaved selection to {output_path}")

    if args.dry_run:
        print("\n[DRY RUN] — no frames extracted.")
        return

    # Extract frames for all sessions (no napari unless --label)
    import subprocess
    import shutil

    print(f"\n{'='*60}")
    if args.label:
        print(f"Extracting and labelling {total} frames across {len(all_selected)} sessions.")
        print(f"Napari will open for each session — label frames, close to continue.")
    else:
        print(f"Extracting {total} frames across {len(all_selected)} sessions.")
    print(f"{'='*60}\n")

    for i, (tag, info) in enumerate(all_selected.items(), 1):
        print(f"[{i}/{len(all_selected)}] {info['exp_id']} — {len(info['frames'])} frames")

        sub, ses = info["sub"], info["ses"]
        session_tag = f"{sub}_{ses}"

        # Download video from S3 and extract frames
        # (reuse prepare_retrain_frames logic but without napari)
        rf_dir = RETRAIN_DIR / session_tag
        rf_dir.mkdir(parents=True, exist_ok=True)

        # Check which frames already exist
        new_frames = []
        for idx in info["frames"]:
            png = rf_dir / f"frame_{int(idx):06d}.png"
            if not png.exists():
                new_frames.append(idx)

        if not new_frames:
            print(f"  All {len(info['frames'])} frames already extracted, skipping download")
        else:
            print(f"  Extracting {len(new_frames)} new frames ({len(info['frames']) - len(new_frames)} already exist)")

            # Download video
            video_dir = Path(f"/tmp/{session_tag}")
            video_dir.mkdir(parents=True, exist_ok=True)
            s3_prefix = f"rawdata/{sub}/{ses}/behav/"
            resp = s3.list_objects_v2(Bucket=RAWDATA_BUCKET, Prefix=s3_prefix, MaxKeys=20)
            video_path = None
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
                print(f"  ERROR: no video found, skipping")
                continue

            # Image-based dedup: reject candidates identical to existing
            # frames on disk or to each other (full-res pixel diff, <1% = dup)
            n_before = len(info["frames"])
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
                from hm2p.pose.dedup import filter_duplicates_against_existing
                deduped = filter_duplicates_against_existing(
                    str(video_path), info["frames"], rf_dir,
                )
                info["frames"] = deduped
                new_frames = [f for f in deduped if not (rf_dir / f"frame_{int(f):06d}.png").exists()]
                n_removed = n_before - len(deduped)
                if n_removed > 0:
                    print(f"  Dedup: removed {n_removed}/{n_before} frames identical to existing")
            except Exception as e:
                print(f"  Dedup check failed: {e}")

            # Also apply target limit after dedup
            n_target = info.get("n_target", len(info["frames"]))
            if len(info["frames"]) > n_target:
                info["frames"] = info["frames"][:n_target]
                new_frames = [f for f in info["frames"] if not (rf_dir / f"frame_{int(f):06d}.png").exists()]

            # Extract frames
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            for idx in sorted(new_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(str(rf_dir / f"frame_{int(idx):06d}.png"), frame)
            cap.release()
            print(f"  Extracted {len(new_frames)} frames to {rf_dir}/")

        # Copy frames to labeled-data dir
        # Find or create the labeled-data session folder
        if LABELED_DIR.exists():
            # Find matching video stem
            video_stem = None
            for ld in LABELED_DIR.iterdir():
                if ld.is_dir() and session_tag.split("_")[0].replace("sub-", "") in ld.name:
                    # Match by date component
                    ses_date = ses.replace("ses-", "").split("T")[0]
                    if ses_date in ld.name:
                        video_stem = ld.name
                        break

            if video_stem:
                ld_dir = LABELED_DIR / video_stem
            else:
                # Create new — use first mp4 stem
                if video_path:
                    video_stem = video_path.stem
                    ld_dir = LABELED_DIR / video_stem
                    ld_dir.mkdir(parents=True, exist_ok=True)
                else:
                    ld_dir = None

            if ld_dir and ld_dir.exists():
                copied = 0
                for png in rf_dir.glob("frame_*.png"):
                    dest = ld_dir / png.name
                    if not dest.exists():
                        shutil.copy2(png, dest)
                        copied += 1
                if copied:
                    print(f"  Copied {copied} new frames to labeled-data/")

        # Save frame indices to metadata
        meta_dir = Path("metadata/retrain_frames")
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_file = meta_dir / f"{session_tag}.json"
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

        # Open napari if --label
        if args.label:
            cmd = [
                sys.executable, "scripts/prepare_retrain_frames.py",
                f"{sub}/{ses}", *[str(f) for f in info["frames"]],
            ]
            subprocess.run(cmd)

    print(f"\n{'='*60}")
    print(f"Done! {total} frames across {len(all_selected)} sessions.")
    if args.label:
        print(f"\nNext steps:")
        print(f"  uv run python scripts/upload_dlc_labels.py")
        print(f"  uv run python scripts/launch_dlc_finetune_ec2.py")
    else:
        print(f"\nFrames extracted. To label them:")
        print(f"  uv run python scripts/select_labelling_frames.py --n {args.n} --label")
        print(f"\nOr label manually per session:")
        for tag, info in all_selected.items():
            frames_str = " ".join(str(f) for f in info["frames"])
            print(f"  uv run python scripts/prepare_retrain_frames.py {info['sub']}/{info['ses']} {frames_str}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
