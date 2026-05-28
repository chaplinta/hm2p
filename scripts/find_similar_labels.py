#!/usr/bin/env python3
"""Find labeled frames with similar bodypart positions.

Compares raw (x, y) label coordinates — two frames are similar only if
the mouse is in the same position, facing the same direction, with the
same body configuration. No alignment or normalization is applied.

For each session:
1. Load CollectedData H5 from the DLC labeled-data directory.
2. Extract (x, y) for all bodyparts except head_midpoint.
3. Skip frames with >50% NaN bodyparts.
4. Compute mean per-bodypart Euclidean distance for all pairs.
5. Flag pairs below --max-dist threshold (default 20 px).
6. For flagged pairs, stitch side-by-side images from labeled-data PNGs.
7. Print histogram of pairwise distances for threshold calibration.
8. Write flagged pairs to CSV.
11. Print summary table.

Usage:
    uv run python scripts/find_similar_labels.py --max-dist 10
    uv run python scripts/find_similar_labels.py --max-dist 10 \\
        --session 20210823_16_59_50_1114353
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
LABELED_DIR = REPO_ROOT / "sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/labeled-data"
OUTPUT_DIR = REPO_ROOT / "retrain_frames" / "_similar_labels"
METADATA_PATH = REPO_ROOT / "metadata" / "experiments.csv"

# Bodyparts to compare (exclude head_midpoint -- rigidly attached to skull)
COMPARE_BODYPARTS = [
    "nose_tip",
    "left_ear",
    "right_ear",
    "neck",
    "mid_back",
    "mouse_center",
    "tail_base",
]

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
# Labeled-data discovery
# ---------------------------------------------------------------------------


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


def clip_name_to_exp_id(clip_name: str) -> str | None:
    """Parse a labeled-data clip directory name into an exp_id string.

    Parameters
    ----------
    clip_name : str
        Directory name, e.g.
        ``"20210823_17_00_04_1114353_maze-rose_overhead.camera-cropped"``.

    Returns
    -------
    str or None
        Experiment id in the form ``"20210823_17_00_04_1114353"``, or None
        if the name cannot be parsed.
    """
    m = re.match(
        r"^(\d{8}_\d{2}_\d{2}_\d{2}_\d+)_",
        clip_name,
    )
    if m is None:
        return None
    return m.group(1)


# ---------------------------------------------------------------------------
# Pose loading from CollectedData H5
# ---------------------------------------------------------------------------


def load_poses(clip_dir: Path) -> tuple[np.ndarray, list[int]] | None:
    """Load (x, y) pose coordinates for COMPARE_BODYPARTS from CollectedData.

    Parameters
    ----------
    clip_dir : Path
        Directory containing ``CollectedData_tristan.h5``.

    Returns
    -------
    tuple[np.ndarray, list[int]] or None
        ``(poses, frame_indices)`` where ``poses`` has shape
        ``(N, B, 2)`` (N frames, B bodyparts, x/y). Frames with >50%
        NaN bodyparts are excluded. Returns None if no valid frames remain
        or the H5 file does not exist.
    """
    h5_files = list(clip_dir.glob("CollectedData_*.h5"))
    if not h5_files:
        return None

    h5_path = h5_files[0]
    df = pd.read_hdf(h5_path)

    # DLC CollectedData is multi-indexed: (scorer, bodypart, coord)
    # Extract the scorer level
    if isinstance(df.columns, pd.MultiIndex):
        scorer = df.columns.get_level_values(0)[0]
    else:
        log.warning("  Unexpected column format in %s", h5_path)
        return None

    # Extract frame indices from the DataFrame index
    # Index values are strings like "labeled-data/<clip>/frame_000042.png"
    frame_indices: list[int] = []
    for idx_val in df.index:
        idx_str = str(idx_val)
        m = re.search(r"frame_(\d+)", idx_str)
        if m:
            frame_indices.append(int(m.group(1)))
        else:
            frame_indices.append(-1)

    n_frames = len(df)
    n_bodyparts = len(COMPARE_BODYPARTS)
    poses = np.full((n_frames, n_bodyparts, 2), np.nan)

    for bp_idx, bp_name in enumerate(COMPARE_BODYPARTS):
        try:
            x_col = (scorer, bp_name, "x")
            y_col = (scorer, bp_name, "y")
            poses[:, bp_idx, 0] = df[x_col].values
            poses[:, bp_idx, 1] = df[y_col].values
        except KeyError:
            # Bodypart not present in this dataset — leave as NaN
            log.debug("  Bodypart %s not found in %s", bp_name, h5_path.name)

    # Filter out frames with >50% NaN bodyparts
    valid_mask = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        n_valid_bp = np.sum(~np.isnan(poses[i, :, 0]))
        valid_mask[i] = n_valid_bp > n_bodyparts / 2

    if not np.any(valid_mask):
        return None

    filtered_poses = poses[valid_mask]
    filtered_indices = [fi for fi, keep in zip(frame_indices, valid_mask, strict=True) if keep]

    return filtered_poses, filtered_indices


# ---------------------------------------------------------------------------
# Pairwise distance computation (raw coordinates)
# ---------------------------------------------------------------------------


def compute_pose_distance(pose_a: np.ndarray, pose_b: np.ndarray) -> float | None:
    """Mean per-bodypart Euclidean distance on raw (x, y) coordinates.

    No alignment, no centroid subtraction — two frames are only similar
    if the mouse is in the same position, facing the same direction,
    with the same body configuration.

    Only bodyparts where both frames have non-NaN labels are used.
    Returns None if fewer than 3 shared bodyparts.
    """
    valid_a = ~np.isnan(pose_a[:, 0])
    valid_b = ~np.isnan(pose_b[:, 0])
    shared = valid_a & valid_b

    if np.sum(shared) < 3:
        return None

    a = pose_a[shared]
    b = pose_b[shared]

    distances = np.sqrt(np.sum((a - b) ** 2, axis=1))
    return float(np.mean(distances))


def compute_all_pairwise_distances(
    poses: np.ndarray, frame_indices: list[int]
) -> list[tuple[int, int, float]]:
    """Compute raw coordinate distances for all frame pairs.

    Parameters
    ----------
    poses : np.ndarray
        Shape ``(N, B, 2)`` — all poses for a session.
    frame_indices : list[int]
        Frame indices corresponding to rows in ``poses``.

    Returns
    -------
    list[tuple[int, int, float]]
        ``(frame_i, frame_j, distance)`` for every valid pair.
    """
    n = len(frame_indices)
    results: list[tuple[int, int, float]] = []

    for i, j in combinations(range(n), 2):
        dist = compute_pose_distance(poses[i], poses[j])
        if dist is not None:
            results.append((frame_indices[i], frame_indices[j], dist))

    return results


# ---------------------------------------------------------------------------
# Video download from S3
# ---------------------------------------------------------------------------


def extract_full_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    """Extract a single full-resolution frame from a video.

    NOTE: Currently unused — frames are read directly from labeled-data PNGs.
    Kept for potential future use with cross-session comparison.

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


def stitch_side_by_side(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Stitch two images side-by-side horizontally, matching heights.

    Parameters
    ----------
    img1 : np.ndarray
        Left image (BGR).
    img2 : np.ndarray
        Right image (BGR).

    Returns
    -------
    np.ndarray
        Horizontally concatenated image.
    """
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
    return np.concatenate([img1, img2], axis=1)


# ---------------------------------------------------------------------------
# Text histogram
# ---------------------------------------------------------------------------


def print_distance_histogram(distances: list[float], n_bins: int = 300) -> None:
    """Print a simple text histogram of pairwise distances.

    Parameters
    ----------
    distances : list[float]
        All pairwise Procrustes distances across sessions.
    n_bins : int
        Number of histogram bins.
    """
    if not distances:
        print("  No pairwise distances to display.")
        return

    arr = np.array(distances)
    min_d = float(arr.min())
    max_d = float(arr.max())
    counts, edges = np.histogram(arr, bins=n_bins)
    max_count = int(counts.max()) if counts.max() > 0 else 1
    bar_width = 40

    print(f"\n  Pairwise distance distribution ({len(distances)} pairs)")
    print(
        f"  min={min_d:.1f}  max={max_d:.1f}  "
        f"median={float(np.median(arr)):.1f}  "
        f"mean={float(np.mean(arr)):.1f}"
    )
    print()

    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        count = int(counts[i])
        bar_len = int(round(count / max_count * bar_width))
        bar = "#" * bar_len
        print(f"  [{lo:7.1f}, {hi:7.1f})  {count:5d}  {bar}")

    print()


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------


def process_session(
    session_info: dict[str, str],
    max_dist: float,
    csv_writer: csv.DictWriter,
    dry_run: bool = False,
) -> tuple[dict[str, Any], list[float]]:
    """Find similar labeled frames for one session.

    Parameters
    ----------
    s3 :
        boto3 S3 client.
    session_info : dict
        Must have ``exp_id``, ``sub``, ``ses``.
    max_dist : float
        Maximum mean per-bodypart displacement (pixels) to flag as similar.
    tmp_dir : Path
        Temporary directory for video downloads.
    csv_writer : csv.DictWriter
        Writer for the flagged-pairs CSV.

    Returns
    -------
    tuple[dict, list[float]]
        Summary dict with ``exp_id``, ``n_frames``, ``n_similar_pairs``,
        and a list of all pairwise distances for this session.
    """
    sub = session_info["sub"]
    ses = session_info["ses"]
    exp_id = session_info["exp_id"]

    result: dict[str, Any] = {
        "exp_id": exp_id,
        "n_frames": 0,
        "n_similar_pairs": 0,
    }
    all_distances: list[float] = []

    # Step 1: Find labeled-data clip directory
    clip_dir = find_labeled_clip_dir(sub, ses)
    if clip_dir is None:
        log.warning("  No labeled-data directory found for %s.", exp_id)
        return result, all_distances

    # Step 2: Load poses
    pose_result = load_poses(clip_dir)
    if pose_result is None:
        log.warning("  No valid labeled frames for %s.", exp_id)
        return result, all_distances

    poses, frame_indices = pose_result
    result["n_frames"] = len(frame_indices)

    if len(frame_indices) < 2:
        log.info("  Only %d labeled frame(s) -- nothing to compare.", len(frame_indices))
        return result, all_distances

    log.info(
        "  %d valid labeled frames in %s",
        len(frame_indices),
        clip_dir.name,
    )

    # Step 3: Compute all pairwise Procrustes distances
    pairwise = compute_all_pairwise_distances(poses, frame_indices)
    all_distances = [d for _, _, d in pairwise]

    # Step 4: Flag pairs below threshold
    similar_pairs = [(f1, f2, d) for f1, f2, d in pairwise if d < max_dist]
    result["n_similar_pairs"] = len(similar_pairs)

    if not similar_pairs:
        log.info("  No similar pairs below threshold %.1f px.", max_dist)
        return result, all_distances

    log.info(
        "  Found %d similar pair(s) below %.1f px threshold.",
        len(similar_pairs),
        max_dist,
    )

    # Write CSV entries
    for f1, f2, dist in similar_pairs:
        csv_writer.writerow(
            {
                "session": exp_id,
                "frame_1": f1,
                "frame_2": f2,
                "mean_displacement_px": f"{dist:.2f}",
            }
        )

    # Step 5: Generate side-by-side images from labeled-data PNGs
    if not dry_run:
        for f1, f2, dist in similar_pairs:
            png1 = clip_dir / f"frame_{f1:06d}.png"
            png2 = clip_dir / f"frame_{f2:06d}.png"
            if not png1.exists() or not png2.exists():
                log.warning("    Missing PNG for frames %d/%d.", f1, f2)
                continue
            img1 = cv2.imread(str(png1))
            img2 = cv2.imread(str(png2))
            if img1 is None or img2 is None:
                continue
            stitched = stitch_side_by_side(img1, img2)
            out_path = OUTPUT_DIR / f"{exp_id}_{f1:06d}_{f2:06d}.png"
            cv2.imwrite(str(out_path), stitched)

    return result, all_distances


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find labeled frames with similar bodypart configurations "
            "(Procrustes-aligned pose comparison)."
        ),
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=20.0,
        help="Maximum mean per-bodypart displacement in pixels (default: 20).",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Process a single session by exp_id.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print table and histogram only — no video download or images.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete duplicate frames (keep one per group, delete the rest). "
        "Removes PNGs, labels from CollectedData, and metadata entries. "
        "Requires confirmation.",
    )
    args = parser.parse_args()

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

    # Open CSV for flagged pairs
    csv_path = OUTPUT_DIR / "_flagged.csv"
    csv_fieldnames = ["session", "frame_1", "frame_2", "mean_displacement_px"]

    print(f"\n{'=' * 62}")
    print("  Similar label detection (Procrustes-aligned pose comparison)")
    print(f"  Sessions: {len(all_sessions)}   Max distance: {args.max_dist:.1f} px")
    print(f"{'=' * 62}\n")

    results: list[dict[str, Any]] = []
    global_distances: list[float] = []

    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        csv_writer.writeheader()

        for i, sess in enumerate(all_sessions, 1):
            log.info("[%d/%d] %s", i, len(all_sessions), sess["exp_id"])
            result, distances = process_session(
                session_info=sess,
                max_dist=args.max_dist,
                csv_writer=csv_writer,
                dry_run=args.dry_run,
            )
            results.append(result)
            global_distances.extend(distances)

    # Distance histogram
    print_distance_histogram(global_distances)

    # Summary table
    print(f"{'=' * 62}")
    print(f"  {'Session':<45s} {'Frames':>7s} {'Similar':>7s}")
    print(f"  {'-' * 45} {'-' * 7} {'-' * 7}")
    total_frames = 0
    total_similar = 0
    for r in results:
        exp = r["exp_id"][:44]
        n_f = r["n_frames"]
        n_s = r["n_similar_pairs"]
        total_frames += n_f
        total_similar += n_s
        sim_str = str(n_s) if n_s > 0 else "-"
        print(f"  {exp:<45s} {n_f:>7d} {sim_str:>7s}")
    print(f"  {'-' * 45} {'-' * 7} {'-' * 7}")
    print(f"  {'TOTAL':<45s} {total_frames:>7d} {total_similar:>7d}")
    print()
    if total_similar > 0:
        print(f"  Flagged pairs saved to: {csv_path}")
        if not args.dry_run:
            print(f"  Side-by-side images saved to: {OUTPUT_DIR}/")
    else:
        print("  No similar pairs found.")
    print(f"{'=' * 62}\n")

    # --delete: remove duplicate frames
    if args.delete and total_similar > 0:
        # Build per-session groups: for each session, collect all frames
        # that appear in a similar pair. Use a graph approach: if A~B and
        # A~C, then {A, B, C} is one group — keep A (lowest index), delete B and C.
        import pandas as _pd
        from collections import defaultdict

        flagged_df = _pd.read_csv(csv_path)
        to_delete_by_session: dict[str, set[int]] = defaultdict(set)

        for sess_name, group in flagged_df.groupby("session"):
            # Build adjacency: union-find to group connected frames
            all_frames_in_pairs: set[int] = set()
            edges: list[tuple[int, int]] = []
            for _, row in group.iterrows():
                f1, f2 = int(row["frame_1"]), int(row["frame_2"])
                all_frames_in_pairs.add(f1)
                all_frames_in_pairs.add(f2)
                edges.append((f1, f2))

            # Simple union-find
            parent: dict[int, int] = {f: f for f in all_frames_in_pairs}

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, b: int) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for f1, f2 in edges:
                union(f1, f2)

            # Group by root
            groups: dict[int, list[int]] = defaultdict(list)
            for f in all_frames_in_pairs:
                groups[find(f)].append(f)

            # For each group, keep the lowest frame index, delete the rest
            for members in groups.values():
                keep = min(members)
                for f in members:
                    if f != keep:
                        to_delete_by_session[str(sess_name)].add(f)

        total_to_delete = sum(len(v) for v in to_delete_by_session.values())

        print(f"\n{'!' * 62}")
        print(f"  WARNING: About to DELETE {total_to_delete} frames across "
              f"{len(to_delete_by_session)} sessions.")
        print(f"  This removes PNGs, labels from CollectedData, and metadata.")
        print(f"  For each group of similar frames, the lowest frame index is kept.")
        print(f"{'!' * 62}")
        for sess, frames in sorted(to_delete_by_session.items()):
            print(f"  {sess[:45]}: delete {sorted(frames)}")
        print()

        confirm = input("  Type 'yes' to confirm deletion: ").strip().lower()
        if confirm != "yes":
            print("  Aborted.")
            return

        # Perform deletion
        deleted_total = 0
        for sess_name, del_frames in sorted(to_delete_by_session.items()):
            # Find clip dir and retrain dir
            clip_dir = None
            for ld in LABELED_DIR.iterdir():
                if not ld.is_dir():
                    continue
                # Match by session name components
                if sess_name.split("_")[0] in ld.name and sess_name.split("_")[-1] in ld.name:
                    clip_dir = ld
                    break

            if clip_dir is None:
                log.warning("  Could not find labeled-data dir for %s", sess_name)
                continue

            # Find matching retrain_frames dir and metadata
            import re as _re
            m = _re.search(r"(\d{8}).*?(\d{7})", clip_dir.name)
            if not m:
                continue
            date, animal = m.group(1), m.group(2)
            rf_dir = None
            meta_path = None
            for rd in Path(REPO_ROOT / "retrain_frames").iterdir():
                if rd.is_dir() and animal in rd.name and date in rd.name:
                    rf_dir = rd
                    meta_path = Path(REPO_ROOT / "metadata" / "retrain_frames" / f"{rd.name}.json")
                    break

            # Delete PNGs
            for fi in sorted(del_frames):
                png_ld = clip_dir / f"frame_{fi:06d}.png"
                if png_ld.exists():
                    os.unlink(str(png_ld))
                if rf_dir:
                    png_rf = rf_dir / f"frame_{fi:06d}.png"
                    if png_rf.exists():
                        os.unlink(str(png_rf))

            # Remove from CollectedData
            for h5 in clip_dir.glob("CollectedData_*.h5"):
                try:
                    df = pd.read_hdf(h5)
                    before = len(df)
                    to_drop = [
                        idx for idx in df.index
                        if any(f"frame_{fi:06d}" in str(idx) for fi in del_frames)
                    ]
                    if to_drop:
                        df = df.drop(to_drop)
                        df.to_hdf(h5, key="df_with_missing", mode="w")
                        df.to_csv(h5.with_suffix(".csv"))
                        log.info("  %s: CollectedData %d -> %d", sess_name[:30], before, len(df))
                except Exception as exc:
                    log.warning("  Failed to update CollectedData for %s: %s", sess_name, exc)

            # Update metadata JSON
            if meta_path and meta_path.exists():
                import json as _json
                data = _json.loads(meta_path.read_text())
                data["frame_indices"] = [
                    i for i in data.get("frame_indices", []) if i not in del_frames
                ]
                data["n_frames"] = len(data["frame_indices"])
                meta_path.write_text(_json.dumps(data, indent=2))

            deleted_total += len(del_frames)
            print(f"  Deleted {len(del_frames)} frames from {sess_name[:45]}")

        print(f"\n  Total deleted: {deleted_total} frames")


if __name__ == "__main__":
    main()
