#!/usr/bin/env python3
"""Render crowd movies for MoSeq syllables.

For each of the top-N syllables (by usage), this script:
1. Loads syllable_id arrays from S3 (kinematics/{sub}/{ses}/syllables.npz)
2. Finds all bouts of that syllable across all sessions
3. Downloads the corresponding video frames from labelled_30fps.mp4
4. Aligns bouts to a fixed length (pad/truncate)
5. Averages frames across bouts to create a crowd movie
6. Saves as MP4 and uploads to S3

The syllable_id arrays are at 30fps (same rate as labelled videos),
so frame indices align directly.

Usage:
    python scripts/render_crowd_movies.py
    python scripts/render_crowd_movies.py --n-syllables 10 --max-bouts 30 --dry-run
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import boto3
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("crowd_movies")

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"


# ── S3 helpers ──────────────────────────────────────────────────────────────


def get_s3_client():
    return boto3.client("s3", region_name=REGION)


def list_syllable_sessions(s3) -> list[dict]:
    """List all sessions with syllables.npz on S3."""
    results = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=DERIVATIVES_BUCKET, Prefix="kinematics/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("syllables.npz"):
                parts = key.split("/")
                if len(parts) >= 3:
                    results.append({
                        "sub": parts[1],
                        "ses": parts[2],
                        "key": key,
                    })
    return results


def download_s3_bytes(s3, key: str) -> bytes | None:
    """Download bytes from S3."""
    try:
        obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key=key)
        return obj["Body"].read()
    except Exception:
        log.debug("Not found: s3://%s/%s", DERIVATIVES_BUCKET, key)
        return None


def upload_s3_file(s3, local_path: Path, key: str) -> bool:
    """Upload a local file to S3."""
    try:
        s3.upload_file(
            str(local_path), DERIVATIVES_BUCKET, key,
            ExtraArgs={"ContentType": "video/mp4"},
        )
        log.info("Uploaded s3://%s/%s", DERIVATIVES_BUCKET, key)
        return True
    except Exception:
        log.exception("Failed to upload %s", key)
        return False


# ── Video frame extraction ──────────────────────────────────────────────────


def extract_frames_from_video(
    video_path: str | Path, start_frame: int, n_frames: int
) -> np.ndarray | None:
    """Extract a clip of n_frames starting at start_frame.

    Returns array of shape (n_frames, H, W, 3) uint8, or None on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return None
    return np.array(frames, dtype=np.uint8)


# ── Bout detection ──────────────────────────────────────────────────────────


def find_bouts(syllable_ids: np.ndarray, target_syl: int) -> list[tuple[int, int]]:
    """Find all contiguous bouts of target_syl.

    Returns list of (start_frame, duration) tuples.
    """
    bouts = []
    in_bout = False
    start = 0
    for i, sid in enumerate(syllable_ids):
        if sid == target_syl:
            if not in_bout:
                start = i
                in_bout = True
        else:
            if in_bout:
                bouts.append((start, i - start))
                in_bout = False
    if in_bout:
        bouts.append((start, len(syllable_ids) - start))
    return bouts


# ── Crowd movie rendering ──────────────────────────────────────────────────


def render_crowd_movie(
    clips: list[np.ndarray],
    bout_frames: int,
) -> np.ndarray:
    """Average aligned clips into a crowd movie.

    Args:
        clips: List of (T, H, W, 3) uint8 arrays (variable T).
        bout_frames: Target clip length in frames.

    Returns:
        (bout_frames, H, W, 3) uint8 array — the averaged crowd movie.
    """
    if not clips:
        raise ValueError("No clips to average")

    h, w = clips[0].shape[1], clips[0].shape[2]
    accumulator = np.zeros((bout_frames, h, w, 3), dtype=np.float64)
    count = np.zeros((bout_frames, 1, 1, 1), dtype=np.float64)

    for clip in clips:
        # Truncate or use as-is (pad handled by count tracking)
        n = min(len(clip), bout_frames)
        accumulator[:n] += clip[:n].astype(np.float64)
        count[:n] += 1

    # Avoid division by zero
    count = np.maximum(count, 1)
    avg = (accumulator / count).astype(np.uint8)
    return avg


def save_crowd_movie_mp4(
    frames: np.ndarray, output_path: Path, fps: int = 10
) -> None:
    """Save frames (T, H, W, 3) as an MP4 file using cv2.

    Uses mp4v codec. Plays at a slower fps (10) so syllable motion is visible.
    """
    h, w = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    log.info("Saved %s (%d frames, %dx%d)", output_path.name, len(frames), w, h)


# ── Main pipeline ──────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render crowd movies for MoSeq syllables"
    )
    parser.add_argument(
        "--n-syllables", type=int, default=20,
        help="Number of top syllables to render (default: 20)",
    )
    parser.add_argument(
        "--max-bouts", type=int, default=50,
        help="Max bouts to average per syllable (default: 50)",
    )
    parser.add_argument(
        "--bout-frames", type=int, default=15,
        help="Fixed clip length in frames (default: 15 = 0.5s at 30fps)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Local output directory (default: temp dir)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without downloading video",
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Skip uploading to S3",
    )
    parser.add_argument(
        "--playback-fps", type=int, default=10,
        help="Playback FPS for output videos (default: 10, slower than real-time)",
    )
    args = parser.parse_args()

    s3 = get_s3_client()

    # ── Step 1: Load all syllable data ──────────────────────────────────
    log.info("Listing syllable sessions on S3...")
    syl_sessions = list_syllable_sessions(s3)
    if not syl_sessions:
        log.error("No syllable sessions found on S3")
        return 1
    log.info("Found %d sessions with syllable data", len(syl_sessions))

    # Load syllable arrays for all sessions
    session_data: list[dict] = []
    for ss in syl_sessions:
        raw = download_s3_bytes(s3, ss["key"])
        if raw is None:
            continue
        npz = np.load(io.BytesIO(raw))
        syl_ids = npz.get("syllable_id", npz.get("syllable_ids"))
        if syl_ids is None:
            log.warning("No syllable_id in %s", ss["key"])
            continue
        session_data.append({
            "sub": ss["sub"],
            "ses": ss["ses"],
            "syllable_id": syl_ids.astype(int),
        })
    log.info("Loaded syllable arrays for %d sessions", len(session_data))

    # ── Step 2: Compute global usage and find top syllables ─────────────
    global_counts: dict[int, int] = defaultdict(int)
    for sd in session_data:
        unique, counts = np.unique(sd["syllable_id"], return_counts=True)
        for sid, cnt in zip(unique, counts):
            global_counts[int(sid)] += int(cnt)

    sorted_syls = sorted(global_counts.keys(), key=lambda x: global_counts[x], reverse=True)
    top_syls = sorted_syls[: args.n_syllables]
    log.info(
        "Top %d syllables (of %d total): %s",
        len(top_syls), len(sorted_syls), top_syls,
    )

    # ── Step 3: Find all bouts per syllable across sessions ─────────────
    # Structure: {syl_id: [(session_idx, start_frame, duration), ...]}
    all_bouts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for sess_idx, sd in enumerate(session_data):
        for syl in top_syls:
            bouts = find_bouts(sd["syllable_id"], syl)
            for start, dur in bouts:
                all_bouts[syl].append((sess_idx, start, dur))

    for syl in top_syls:
        log.info(
            "Syllable %d: %d bouts, %d frames total",
            syl, len(all_bouts[syl]), global_counts[syl],
        )

    if args.dry_run:
        log.info("DRY RUN — would render %d crowd movies. Exiting.", len(top_syls))
        return 0

    # ── Step 4: Download videos and extract clips ───────────────────────
    # Use a temp dir for video caching and output
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        _tmpdir = tempfile.mkdtemp(prefix="crowd_movies_")
        output_dir = Path(_tmpdir)

    video_cache_dir = output_dir / "_video_cache"
    video_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_video_path(sess_idx: int) -> Path | None:
        """Download labelled_30fps.mp4 for a session (cached locally)."""
        sd = session_data[sess_idx]
        sub, ses = sd["sub"], sd["ses"]
        local_path = video_cache_dir / f"{sub}_{ses}.mp4"
        if local_path.exists():
            return local_path

        s3_key = f"pose/{sub}/{ses}/labelled_30fps.mp4"
        log.info("Downloading video: s3://%s/%s", DERIVATIVES_BUCKET, s3_key)
        try:
            s3.download_file(DERIVATIVES_BUCKET, s3_key, str(local_path))
            return local_path
        except Exception:
            log.warning("Video not found: %s", s3_key)
            return None

    # ── Step 5: Render each syllable ────────────────────────────────────
    rendered = []
    for syl in top_syls:
        bouts = all_bouts[syl]
        if not bouts:
            log.warning("Syllable %d: no bouts, skipping", syl)
            continue

        # Subsample bouts if too many (pick random subset for diversity)
        if len(bouts) > args.max_bouts:
            rng = np.random.default_rng(seed=syl)
            indices = rng.choice(len(bouts), size=args.max_bouts, replace=False)
            bouts = [bouts[i] for i in sorted(indices)]

        clips: list[np.ndarray] = []
        # Group bouts by session to minimize video re-downloads
        by_session: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for sess_idx, start, dur in bouts:
            by_session[sess_idx].append((start, dur))

        for sess_idx, bout_list in by_session.items():
            video_path = get_video_path(sess_idx)
            if video_path is None:
                continue
            for start, dur in bout_list:
                clip = extract_frames_from_video(
                    video_path, start, args.bout_frames,
                )
                if clip is not None and len(clip) >= 3:
                    clips.append(clip)

        if not clips:
            log.warning("Syllable %d: no valid clips extracted, skipping", syl)
            continue

        log.info("Syllable %d: averaging %d clips", syl, len(clips))
        crowd = render_crowd_movie(clips, args.bout_frames)
        out_path = output_dir / f"syllable_{syl}.mp4"
        save_crowd_movie_mp4(crowd, out_path, fps=args.playback_fps)
        rendered.append((syl, out_path, global_counts[syl], len(clips)))

    log.info("Rendered %d / %d crowd movies", len(rendered), len(top_syls))

    # ── Step 6: Upload to S3 ────────────────────────────────────────────
    if not args.no_upload:
        # Also upload a summary JSON for the frontend
        summary = []
        for syl, out_path, usage, n_clips in rendered:
            s3_key = f"kinematics/crowd_movies/syllable_{syl}.mp4"
            upload_s3_file(s3, out_path, s3_key)
            summary.append({
                "syllable_id": syl,
                "usage_frames": usage,
                "n_clips_averaged": n_clips,
                "s3_key": s3_key,
            })

        # Upload summary JSON
        import json
        summary_bytes = json.dumps(summary, indent=2).encode()
        summary_key = "kinematics/crowd_movies/summary.json"
        try:
            s3.put_object(
                Bucket=DERIVATIVES_BUCKET, Key=summary_key,
                Body=summary_bytes, ContentType="application/json",
            )
            log.info("Uploaded summary: s3://%s/%s", DERIVATIVES_BUCKET, summary_key)
        except Exception:
            log.exception("Failed to upload summary JSON")

        log.info("Upload complete: %d crowd movies", len(rendered))
    else:
        log.info("Skipping upload (--no-upload). Files in: %s", output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
