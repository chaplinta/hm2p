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


def select_diverse(
    scores: np.ndarray,
    positions: np.ndarray | None,
    n: int,
    min_spacing: int = 30,
    min_position_dist: float = 5.0,
    exclude: set[int] | None = None,
) -> list[int]:
    """Select top-N frames by score with spacing and diversity constraints."""
    order = np.argsort(-scores)  # highest score first
    selected = []
    exclude = exclude or set()

    for idx in order:
        if len(selected) >= n:
            break
        idx = int(idx)
        if idx in exclude:
            continue
        # Spacing constraint
        if any(abs(idx - s) < min_spacing for s in selected):
            continue
        # Position similarity constraint
        if positions is not None and selected:
            pos = positions[idx]
            if np.isnan(pos).any():
                continue
            too_similar = False
            for s in selected:
                diff = np.nanmean(np.abs(positions[s] - pos))
                if diff < min_position_dist:
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

        selected = select_diverse(
            s["scores"], positions, n_alloc,
            min_spacing=args.min_spacing,
            exclude=s["already_labelled"],
        )

        all_selected[tag] = {
            "exp_id": s["exp_id"],
            "sub": s["sub"],
            "ses": s["ses"],
            "frames": selected,
        }
        total += len(selected)

        scores_at_selected = [f"{s['scores'][i]:.3f}" for i in selected]
        print(f"{s['exp_id']}: {len(selected)} frames "
              f"(already labelled: {s['n_already_labelled']})")
        print(f"  Indices: {selected}")
        print(f"  Scores:  {scores_at_selected}")
        print()

    print(f"Total: {total} frames across {len(all_selected)} sessions")

    if args.dry_run:
        print("\n[DRY RUN] — no frames extracted. Run without --dry-run to extract.")
        return

    # Generate prepare_retrain_frames commands
    print(f"\n{'='*60}")
    print("Run these commands to extract and label the frames:")
    print(f"{'='*60}\n")

    for tag, info in all_selected.items():
        frames_str = " ".join(str(f) for f in info["frames"])
        print(f"uv run python scripts/prepare_retrain_frames.py "
              f"{info['sub']}/{info['ses']} {frames_str}")

    # Also save as JSON for automation
    output_path = Path("retrain_frames/_next_batch.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(all_selected, indent=2, default=str))
    print(f"\nSaved selection to {output_path}")


if __name__ == "__main__":
    main()
