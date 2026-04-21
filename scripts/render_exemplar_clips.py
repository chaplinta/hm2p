#!/usr/bin/env python3
"""Render exemplar video clips for MoSeq syllables.

For each of the top-N syllables (by usage), this script:
1. Loads syllable_id arrays from S3 (kinematics/{sub}/{ses}/syllables.npz)
2. Finds all bouts of that syllable across all sessions
3. Selects the 3 most typical bouts (closest to median duration, diverse sessions)
4. Downloads the corresponding labelled_30fps.mp4 and extracts clips
5. Adds a coloured border during the active bout frames
6. Saves as MP4 and uploads to S3

"Most typical" = bouts whose duration is closest to the median duration for
that syllable, with a preference for selecting bouts from different sessions.

Usage:
    python scripts/render_exemplar_clips.py
    python scripts/render_exemplar_clips.py --n-syllables 20 --n-exemplars 3 --dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import subprocess
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
log = logging.getLogger("exemplar_clips")

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"

# Clip rendering parameters
CONTEXT_FRAMES = 15  # frames of context before/after bout (0.5s at 30fps)
MIN_CLIP_FRAMES = 30  # minimum clip length (1s at 30fps)
MAX_CLIP_FRAMES = 90  # maximum clip length (3s at 30fps)
PLAYBACK_FPS = 15  # slower than 30fps real-time for visibility
BORDER_PX = 4  # coloured border width during active bout
BORDER_COLOR_BGR = (0, 200, 0)  # green border during active bout


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
                    results.append({"sub": parts[1], "ses": parts[2], "key": key})
    return results


def download_s3_bytes(s3, key: str) -> bytes | None:
    try:
        obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key=key)
        return obj["Body"].read()
    except Exception:
        log.debug("Not found: s3://%s/%s", DERIVATIVES_BUCKET, key)
        return None


def upload_s3_file(s3, local_path: Path, key: str, content_type: str = "video/mp4") -> bool:
    try:
        s3.upload_file(
            str(local_path), DERIVATIVES_BUCKET, key,
            ExtraArgs={"ContentType": content_type},
        )
        log.info("Uploaded s3://%s/%s", DERIVATIVES_BUCKET, key)
        return True
    except Exception:
        log.exception("Failed to upload %s", key)
        return False


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


def select_exemplar_bouts(
    all_bouts: list[tuple[int, int, int]],
    n_exemplars: int = 3,
    min_duration: int = 3,
) -> list[tuple[int, int, int]]:
    """Select the most typical bouts for a syllable.

    Each bout is (session_idx, start_frame, duration).
    "Most typical" = closest to median duration, preferring different sessions.

    Args:
        all_bouts: List of (session_idx, start_frame, duration) across all sessions.
        n_exemplars: Number of exemplar bouts to select.
        min_duration: Minimum bout duration to consider (filter noise).

    Returns:
        List of (session_idx, start_frame, duration) for selected exemplars.
    """
    # Filter out very short bouts
    valid = [(s, f, d) for s, f, d in all_bouts if d >= min_duration]
    if not valid:
        return all_bouts[:n_exemplars]

    durations = np.array([d for _, _, d in valid])
    median_dur = np.median(durations)

    # Score: absolute distance from median duration
    scores = np.abs(durations - median_dur)

    # Sort by score (best first)
    order = np.argsort(scores)

    # Greedy selection: prefer different sessions for diversity
    selected = []
    used_sessions = set()
    # First pass: pick best-scoring bout from each unique session
    for idx in order:
        if len(selected) >= n_exemplars:
            break
        sess_idx = valid[idx][0]
        if sess_idx not in used_sessions:
            selected.append(valid[idx])
            used_sessions.add(sess_idx)

    # Second pass: fill remaining slots from best remaining bouts
    if len(selected) < n_exemplars:
        for idx in order:
            if len(selected) >= n_exemplars:
                break
            if valid[idx] not in selected:
                selected.append(valid[idx])

    return selected


# ── Video extraction ──────────────────────────────────────────────────────


def extract_clip_with_context(
    video_path: str | Path,
    bout_start: int,
    bout_duration: int,
    total_video_frames: int | None = None,
) -> tuple[np.ndarray | None, int, int, int]:
    """Extract a clip centered on a bout with context frames.

    Returns:
        (frames, clip_start, bout_offset, bout_duration) where:
        - frames: (T, H, W, 3) uint8 array or None
        - clip_start: absolute frame index where clip starts in video
        - bout_offset: frame offset within clip where bout starts
        - bout_duration: actual bout duration in frames
    """
    # Compute clip boundaries with context
    pre_context = CONTEXT_FRAMES
    post_context = CONTEXT_FRAMES
    clip_start = max(0, bout_start - pre_context)
    actual_pre = bout_start - clip_start
    clip_length = actual_pre + bout_duration + post_context

    # Enforce min/max clip length
    clip_length = max(clip_length, MIN_CLIP_FRAMES)
    clip_length = min(clip_length, MAX_CLIP_FRAMES)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0, 0, 0

    if total_video_frames is None:
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Clamp to video bounds
    if clip_start + clip_length > total_video_frames:
        clip_length = total_video_frames - clip_start
    if clip_length <= 0:
        cap.release()
        return None, 0, 0, 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    frames = []
    for _ in range(clip_length):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return None, 0, 0, 0

    return np.array(frames, dtype=np.uint8), clip_start, actual_pre, bout_duration


def add_bout_border(
    frames: np.ndarray,
    bout_offset: int,
    bout_duration: int,
) -> np.ndarray:
    """Add a coloured border to frames during the active bout."""
    result = frames.copy()
    b = BORDER_PX
    color = BORDER_COLOR_BGR

    for i in range(bout_offset, min(bout_offset + bout_duration, len(result))):
        f = result[i]
        # Top border
        f[:b, :] = color
        # Bottom border
        f[-b:, :] = color
        # Left border
        f[:, :b] = color
        # Right border
        f[:, -b:] = color
        result[i] = f

    return result


def save_clip_mp4(frames: np.ndarray, output_path: Path, fps: int = PLAYBACK_FPS) -> None:
    """Save frames as H.264 MP4 for browser playback."""
    h, w = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    raw_path = output_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()

    if shutil.which("ffmpeg"):
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw_path),
             "-c:v", "libx264", "-crf", "23", "-preset", "fast",
             "-movflags", "+faststart", "-pix_fmt", "yuv420p",
             str(output_path)],
            capture_output=True, text=True,
        )
        raw_path.unlink(missing_ok=True)
        if result.returncode != 0:
            log.warning("ffmpeg re-encode failed: %s", result.stderr[-200:])
            # Fall back: just rename the raw mp4v file
            if not output_path.exists():
                raw_path2 = output_path.with_suffix(".raw.mp4")
                if raw_path2.exists():
                    raw_path2.rename(output_path)
    else:
        raw_path.rename(output_path)

    log.info("Saved %s (%d frames, %dx%d)", output_path.name, len(frames), w, h)


# ── Main pipeline ──────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render exemplar video clips for MoSeq syllables"
    )
    parser.add_argument(
        "--n-syllables", type=int, default=20,
        help="Number of top syllables to render (default: 20)",
    )
    parser.add_argument(
        "--n-exemplars", type=int, default=3,
        help="Number of exemplar clips per syllable (default: 3)",
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
        "--playback-fps", type=int, default=PLAYBACK_FPS,
        help=f"Playback FPS for output videos (default: {PLAYBACK_FPS})",
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

    # ── Step 2: Global usage + top syllables ────────────────────────────
    global_counts: dict[int, int] = defaultdict(int)
    for sd in session_data:
        unique, counts = np.unique(sd["syllable_id"], return_counts=True)
        for sid, cnt in zip(unique, counts):
            global_counts[int(sid)] += int(cnt)

    sorted_syls = sorted(global_counts.keys(), key=lambda x: global_counts[x], reverse=True)
    top_syls = sorted_syls[: args.n_syllables]
    log.info("Top %d syllables: %s", len(top_syls), top_syls)

    # ── Step 3: Find all bouts and select exemplars ─────────────────────
    # {syl_id: [(session_idx, start_frame, duration), ...]}
    all_bouts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for sess_idx, sd in enumerate(session_data):
        for syl in top_syls:
            bouts = find_bouts(sd["syllable_id"], syl)
            for start, dur in bouts:
                all_bouts[syl].append((sess_idx, start, dur))

    # Select exemplars for each syllable
    exemplar_plan: dict[int, list[tuple[int, int, int]]] = {}
    for syl in top_syls:
        bouts = all_bouts[syl]
        exemplars = select_exemplar_bouts(bouts, n_exemplars=args.n_exemplars)
        exemplar_plan[syl] = exemplars
        durations = [d for _, _, d in bouts if d >= 3]
        median_dur = np.median(durations) if durations else 0
        log.info(
            "Syllable %d: %d bouts total, median dur=%.0f frames, "
            "selected %d exemplars: %s",
            syl, len(bouts), median_dur, len(exemplars),
            [(session_data[s]["sub"] + "/" + session_data[s]["ses"], f, d)
             for s, f, d in exemplars],
        )

    if args.dry_run:
        # Show which videos would need downloading
        needed_sessions = set()
        for syl, exemplars in exemplar_plan.items():
            for sess_idx, _, _ in exemplars:
                needed_sessions.add(sess_idx)
        log.info(
            "DRY RUN — would render %d syllables × %d exemplars = %d clips. "
            "Need to download %d / %d session videos.",
            len(top_syls), args.n_exemplars,
            sum(len(e) for e in exemplar_plan.values()),
            len(needed_sessions), len(session_data),
        )
        return 0

    # ── Step 4: Set up output dirs ──────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        _tmpdir = tempfile.mkdtemp(prefix="exemplar_clips_")
        output_dir = Path(_tmpdir)

    video_cache_dir = output_dir / "_video_cache"
    video_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_video_path(sess_idx: int) -> Path | None:
        """Download labelled_30fps.mp4 for a session (cached locally)."""
        sd = session_data[sess_idx]
        local_path = video_cache_dir / f"{sd['sub']}_{sd['ses']}.mp4"
        if local_path.exists():
            return local_path
        s3_key = f"pose/{sd['sub']}/{sd['ses']}/labelled_30fps.mp4"
        log.info("Downloading video: s3://%s/%s", DERIVATIVES_BUCKET, s3_key)
        try:
            s3.download_file(DERIVATIVES_BUCKET, s3_key, str(local_path))
            return local_path
        except Exception:
            log.warning("Video not found: %s", s3_key)
            return None

    # ── Step 5: Render exemplar clips ───────────────────────────────────
    # Group all needed clips by session to minimize video downloads
    session_clips: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    # session_clips[sess_idx] = [(syl_id, start_frame, duration, exemplar_idx), ...]
    for syl in top_syls:
        for ex_idx, (sess_idx, start, dur) in enumerate(exemplar_plan[syl]):
            session_clips[sess_idx].append((syl, start, dur, ex_idx))

    rendered: list[dict] = []
    for sess_idx in sorted(session_clips.keys()):
        clips_needed = session_clips[sess_idx]
        sd = session_data[sess_idx]
        log.info(
            "Processing %s/%s: %d clips to extract",
            sd["sub"], sd["ses"], len(clips_needed),
        )

        video_path = get_video_path(sess_idx)
        if video_path is None:
            log.warning("Skipping session %s/%s: video not available", sd["sub"], sd["ses"])
            continue

        for syl_id, bout_start, bout_dur, ex_idx in clips_needed:
            frames, clip_start, bout_offset, _ = extract_clip_with_context(
                video_path, bout_start, bout_dur,
            )
            if frames is None or len(frames) < 3:
                log.warning(
                    "Syllable %d exemplar %d: failed to extract clip", syl_id, ex_idx,
                )
                continue

            # Add green border during active bout
            frames = add_bout_border(frames, bout_offset, bout_dur)

            # Save
            out_name = f"syllable_{syl_id}_ex{ex_idx}.mp4"
            out_path = output_dir / out_name
            save_clip_mp4(frames, out_path, fps=args.playback_fps)

            rendered.append({
                "syllable_id": syl_id,
                "exemplar_idx": ex_idx,
                "sub": sd["sub"],
                "ses": sd["ses"],
                "bout_start_frame": int(bout_start),
                "bout_duration_frames": int(bout_dur),
                "bout_duration_sec": round(bout_dur / 30.0, 2),
                "clip_frames": len(frames),
                "clip_duration_sec": round(len(frames) / 30.0, 2),
                "s3_key": f"kinematics/exemplar_clips/{out_name}",
            })

        # Delete cached video to save disk space
        cached_video = video_cache_dir / f"{sd['sub']}_{sd['ses']}.mp4"
        if cached_video.exists():
            cached_video.unlink()
            log.info("Deleted cached video: %s", cached_video.name)

    log.info("Rendered %d exemplar clips", len(rendered))

    # ── Step 6: Build summary with per-syllable stats ───────────────────
    # Group rendered clips by syllable, add global stats
    syl_summary: dict[int, dict] = {}
    for syl in top_syls:
        bouts = all_bouts[syl]
        durations = [d for _, _, d in bouts if d >= 3]
        syl_summary[syl] = {
            "syllable_id": syl,
            "total_frames": global_counts[syl],
            "total_bouts": len(bouts),
            "median_duration_frames": int(np.median(durations)) if durations else 0,
            "median_duration_sec": round(float(np.median(durations)) / 30.0, 2) if durations else 0,
            "exemplars": [],
        }

    for clip_info in rendered:
        syl = clip_info["syllable_id"]
        if syl in syl_summary:
            syl_summary[syl]["exemplars"].append(clip_info)

    summary = {
        "n_syllables": len(top_syls),
        "n_exemplars_per_syllable": args.n_exemplars,
        "n_sessions": len(session_data),
        "total_clips_rendered": len(rendered),
        "syllables": [syl_summary[syl] for syl in top_syls if syl in syl_summary],
    }

    # Save summary locally
    summary_path = output_dir / "exemplar_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Saved summary: %s", summary_path)

    # ── Step 7: Upload to S3 ────────────────────────────────────────────
    if not args.no_upload:
        for clip_info in rendered:
            local_path = output_dir / Path(clip_info["s3_key"]).name
            if local_path.exists():
                upload_s3_file(s3, local_path, clip_info["s3_key"])

        # Upload summary JSON
        summary_key = "kinematics/exemplar_clips/exemplar_summary.json"
        try:
            s3.put_object(
                Bucket=DERIVATIVES_BUCKET, Key=summary_key,
                Body=json.dumps(summary, indent=2).encode(),
                ContentType="application/json",
            )
            log.info("Uploaded summary: s3://%s/%s", DERIVATIVES_BUCKET, summary_key)
        except Exception:
            log.exception("Failed to upload summary JSON")

        log.info("Upload complete: %d exemplar clips", len(rendered))
    else:
        log.info("Skipping upload (--no-upload). Files in: %s", output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
