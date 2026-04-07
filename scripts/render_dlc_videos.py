#!/usr/bin/env python3
"""Batch-render labelled DLC pose videos.

For each session:
1. Download raw overhead video from S3 (hm2p-rawdata).
2. Download DLC .h5 from S3 (hm2p-derivatives).
3. Convert multi-animal DLC to single-animal (best individual per frame).
4. Map DLC frames to raw video frames (100fps→30fps = every 3.33 frames).
5. Draw keypoints + skeleton on each frame.
6. Encode as H.264 MP4 at 30fps, downscaled to 416x304.
7. Upload labelled video to S3.

Usage:
    python scripts/render_dlc_videos.py --session 20210823_16_59_50_1114353
    python scripts/render_dlc_videos.py --all
    python scripts/render_dlc_videos.py --all --dry-run
    python scripts/render_dlc_videos.py --all --skip-existing
    python scripts/render_dlc_videos.py --all --no-upload
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import tempfile
from pathlib import Path

import boto3
import cv2
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAWDATA_BUCKET = "hm2p-rawdata"
DERIV_BUCKET = "hm2p-derivatives"

# Keypoint colours (BGR for OpenCV)
KEYPOINT_COLORS: dict[str, tuple[int, int, int]] = {
    "nose_tip": (0, 0, 255),            # red
    "nose": (0, 0, 255),                # red (alias for SuperAnimal output)
    "left_ear": (255, 0, 0),            # blue
    "right_ear": (255, 255, 0),         # cyan
    "implant_base_rear": (0, 165, 255), # orange
    "neck": (128, 0, 128),              # purple
    "mid_back": (0, 255, 0),            # green
    "mouse_center": (0, 255, 255),      # yellow
    "tail_base": (255, 0, 255),         # magenta
}

SKELETON: list[tuple[str, str]] = [
    ("nose_tip", "implant_base_rear"),
    ("nose_tip", "left_ear"),
    ("nose_tip", "right_ear"),
    ("nose", "left_ear"),       # fallback for SuperAnimal "nose" name
    ("nose", "right_ear"),
    ("left_ear", "implant_base_rear"),
    ("right_ear", "implant_base_rear"),
    ("left_ear", "right_ear"),
    ("implant_base_rear", "neck"),
    ("neck", "mid_back"),
    ("mid_back", "mouse_center"),
    ("mouse_center", "tail_base"),
]

# All possible bodypart names (finetuned + SuperAnimal variants)
BODYPARTS = list(KEYPOINT_COLORS.keys())

CIRCLE_RADIUS = 4
LINE_WIDTH = 2
CONFIDENCE_THRESHOLD = 0.5

OUTPUT_WIDTH = 416
OUTPUT_HEIGHT = 304
OUTPUT_FPS = 30

# Raw video is 100fps, DLC ran on 30fps subsampled video.
# ffmpeg -r 30 picks frames at evenly spaced intervals (every 3.33 frames),
# NOT every 3rd frame.  We compute the exact mapping per DLC frame.

METADATA_PATH = Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"
LOCAL_OUTPUT_DIR = Path("/tmp/dlc_labelled")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_exp_id(exp_id: str) -> tuple[str, str]:
    """Convert exp_id to (sub, ses) NeuroBlueprint names.

    '20210823_16_59_50_1114353' -> ('sub-1114353', 'ses-20210823T165950')
    """
    parts = exp_id.split("_")
    animal = parts[-1]
    sub = f"sub-{animal}"
    ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
    return sub, ses


def load_sessions(metadata_path: Path) -> list[dict]:
    """Load non-excluded sessions from experiments.csv."""
    sessions = []
    with open(metadata_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("exclude", "0").strip() == "1":
                continue
            sessions.append(row)
    return sessions


def find_s3_file(s3, bucket: str, prefix: str, pattern: str) -> str | None:
    """Find a single S3 key matching a substring pattern under prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if pattern in key:
                return key
    return None


def s3_key_exists(s3, bucket: str, key: str) -> bool:
    """Check if an S3 key exists."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def download_s3(s3, bucket: str, key: str, local_path: Path) -> None:
    """Download an S3 object to a local path."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("  Downloading s3://%s/%s", bucket, key)
    s3.download_file(bucket, key, str(local_path))


def convert_madlc_to_single(df: pd.DataFrame) -> pd.DataFrame:
    """Convert multi-animal DLC DataFrame to single-animal by picking best
    individual per frame (highest mean likelihood across bodyparts).

    If already single-animal (3-level columns), returns as-is.
    """
    if df.columns.nlevels == 3:
        log.info("  Already single-animal format")
        return df

    if df.columns.nlevels != 4:
        raise ValueError(f"Expected 3 or 4 column levels, got {df.columns.nlevels}")

    scorer = df.columns.get_level_values("scorer")[0]
    individuals = df.columns.get_level_values("individuals").unique().tolist()
    available_bps = df.columns.get_level_values("bodyparts").unique().tolist()

    use_bps = [bp for bp in BODYPARTS if bp in available_bps]
    if not use_bps:
        raise ValueError(
            f"None of {BODYPARTS} found in data. Available: {available_bps}"
        )

    n_frames = len(df)
    log.info("  Selecting best individual per frame (%d frames, %d individuals)",
             n_frames, len(individuals))

    # Build (n_frames, n_individuals) likelihood matrix
    ind_scores = np.full((n_frames, len(individuals)), -1.0)
    for j, ind in enumerate(individuals):
        lk_cols = []
        for bp in use_bps:
            try:
                lk_cols.append(df[(scorer, ind, bp, "likelihood")].values)
            except KeyError:
                pass
        if lk_cols:
            ind_scores[:, j] = np.nanmean(np.column_stack(lk_cols), axis=1)

    best_ind_idx = np.argmax(ind_scores, axis=1)

    # Build single-animal dataframe
    new_columns = pd.MultiIndex.from_tuples(
        [(scorer, bp, coord) for bp in use_bps for coord in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"],
    )
    new_data = np.empty((n_frames, len(new_columns)), dtype=np.float64)

    col_idx = 0
    for bp in use_bps:
        for coord in ("x", "y", "likelihood"):
            all_vals = np.full((n_frames, len(individuals)), np.nan)
            for j, ind in enumerate(individuals):
                try:
                    all_vals[:, j] = df[(scorer, ind, bp, coord)].values
                except KeyError:
                    pass
            new_data[:, col_idx] = all_vals[np.arange(n_frames), best_ind_idx]
            col_idx += 1

    return pd.DataFrame(new_data, index=df.index, columns=new_columns)


def extract_keypoints(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract per-bodypart arrays from single-animal DLC DataFrame.

    Returns dict mapping bodypart name to (N, 3) array of [x, y, likelihood].
    """
    scorer = df.columns.get_level_values("scorer")[0]
    available_bps = df.columns.get_level_values("bodyparts").unique().tolist()
    result = {}
    for bp in BODYPARTS:
        if bp in available_bps:
            x = df[(scorer, bp, "x")].values
            y = df[(scorer, bp, "y")].values
            lk = df[(scorer, bp, "likelihood")].values
            result[bp] = np.column_stack([x, y, lk])
    return result


def draw_frame(
    frame: np.ndarray,
    keypoints: dict[str, np.ndarray],
    frame_idx: int,
) -> np.ndarray:
    """Draw keypoints and skeleton on a single frame.

    Parameters
    ----------
    frame : BGR image array (original resolution, pre-downscale).
    keypoints : dict of bodypart -> (N, 3) arrays [x, y, likelihood].
    frame_idx : DLC frame index (after 3x subsampling).
    """
    # Draw skeleton lines first (under circles)
    for bp1, bp2 in SKELETON:
        if bp1 not in keypoints or bp2 not in keypoints:
            continue
        kp1 = keypoints[bp1][frame_idx]
        kp2 = keypoints[bp2][frame_idx]
        if np.isnan(kp1[:2]).any() or np.isnan(kp2[:2]).any():
            continue
        # Only draw line if BOTH endpoints are above confidence threshold
        if kp1[2] < CONFIDENCE_THRESHOLD or kp2[2] < CONFIDENCE_THRESHOLD:
            continue
        pt1 = (int(round(kp1[0])), int(round(kp1[1])))
        pt2 = (int(round(kp2[0])), int(round(kp2[1])))
        # Average colour of the two endpoints
        c1 = KEYPOINT_COLORS.get(bp1, (255, 255, 255))
        c2 = KEYPOINT_COLORS.get(bp2, (255, 255, 255))
        color = tuple((a + b) // 2 for a, b in zip(c1, c2))
        cv2.line(frame, pt1, pt2, color, LINE_WIDTH, cv2.LINE_AA)

    # Draw keypoint circles
    for bp, kp_array in keypoints.items():
        kp = kp_array[frame_idx]
        if np.isnan(kp[:2]).any():
            continue
        pt = (int(round(kp[0])), int(round(kp[1])))
        color = KEYPOINT_COLORS.get(bp, (255, 255, 255))
        if kp[2] >= CONFIDENCE_THRESHOLD:
            cv2.circle(frame, pt, CIRCLE_RADIUS, color, -1, cv2.LINE_AA)  # filled
        else:
            cv2.circle(frame, pt, CIRCLE_RADIUS, color, 1, cv2.LINE_AA)  # hollow

    return frame


# ---------------------------------------------------------------------------
# Main rendering
# ---------------------------------------------------------------------------

def render_session(
    s3,
    exp_id: str,
    *,
    dry_run: bool = False,
    skip_existing: bool = False,
    no_upload: bool = False,
) -> str | None:
    """Render labelled video for a single session.

    Returns the S3 key of the uploaded video, local path if no_upload, or
    "skipped"/"dry-run" for those modes. None on failure.
    """
    sub, ses = parse_exp_id(exp_id)
    log.info("Processing %s (%s/%s)", exp_id, sub, ses)

    upload_key = f"pose/{sub}/{ses}/labelled_30fps.mp4"

    # Check if already exists on S3
    if skip_existing and not no_upload:
        if s3_key_exists(s3, DERIV_BUCKET, upload_key):
            log.info("  Skipping — already exists on S3")
            return "skipped"

    # --- Find S3 keys ---
    video_prefix = f"rawdata/{sub}/{ses}/behav/"
    video_key = find_s3_file(s3, RAWDATA_BUCKET, video_prefix,
                             "overhead.camera-cropped.mp4")
    if not video_key:
        video_key = find_s3_file(s3, RAWDATA_BUCKET, video_prefix,
                                 "overhead.camera.mp4")
    if not video_key:
        log.warning("  No overhead video found for %s, skipping", exp_id)
        return None

    pose_prefix = f"pose/{sub}/{ses}/"
    pose_key = find_s3_file(s3, DERIV_BUCKET, pose_prefix, "_superanimal_")
    if not pose_key:
        pose_key = find_s3_file(s3, DERIV_BUCKET, pose_prefix, ".h5")
    if not pose_key:
        log.warning("  No DLC .h5 found for %s, skipping", exp_id)
        return None

    if dry_run:
        log.info("  [DRY RUN] Would process:")
        log.info("    Video: s3://%s/%s", RAWDATA_BUCKET, video_key)
        log.info("    Pose:  s3://%s/%s", DERIV_BUCKET, pose_key)
        log.info("    Output: s3://%s/%s", DERIV_BUCKET, upload_key)
        return "dry-run"

    log.info("  Video: %s", video_key)
    log.info("  Pose:  %s", pose_key)

    with tempfile.TemporaryDirectory(prefix=f"dlc_render_{exp_id}_") as tmpdir:
        tmp = Path(tmpdir)
        video_local = tmp / "video.mp4"
        pose_local = tmp / "pose.h5"

        # Download
        download_s3(s3, RAWDATA_BUCKET, video_key, video_local)
        download_s3(s3, DERIV_BUCKET, pose_key, pose_local)

        # Load DLC data
        log.info("  Loading DLC data...")
        df = pd.read_hdf(pose_local)
        df = convert_madlc_to_single(df)
        keypoints = extract_keypoints(df)
        n_dlc_frames = len(df)
        log.info("  DLC frames: %d, bodyparts: %s",
                 n_dlc_frames, list(keypoints.keys()))

        # Open video
        cap = cv2.VideoCapture(str(video_local))
        if not cap.isOpened():
            log.error("  Failed to open video %s", video_local)
            return None

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        log.info("  Video frames: %d (100fps), DLC frames: %d (30fps)",
                 total_video_frames, n_dlc_frames)

        # Determine output path
        if no_upload:
            out_path = LOCAL_OUTPUT_DIR / sub / ses / "labelled_30fps.mp4"
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = tmp / "labelled_30fps.mp4"

        # Set up writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, OUTPUT_FPS,
                                 (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        if not writer.isOpened():
            log.error("  Failed to create video writer")
            cap.release()
            return None

        # Process frames: for each DLC frame, seek to the correct raw video
        # frame.  ffmpeg -r 30 on 100fps picks frame at t = dlc_idx / 30,
        # which is raw_frame = round(dlc_idx * raw_fps / 30).
        raw_fps = cap.get(cv2.CAP_PROP_FPS) or 100.0
        written = 0

        for dlc_frame_idx in range(n_dlc_frames):
            target_raw = round(dlc_frame_idx * raw_fps / OUTPUT_FPS)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_raw)
            ret, frame = cap.read()
            if not ret:
                break

            frame = draw_frame(frame, keypoints, dlc_frame_idx)
            frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                               interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1

            if written % 5000 == 0:
                log.info("  Rendered %d / %d frames", written, n_dlc_frames)

        cap.release()
        writer.release()
        log.info("  Wrote %d frames to %s", written, out_path.name)

        # Compress with ffmpeg H.264 (mp4v from cv2 is huge)
        import shutil as _shutil
        if _shutil.which("ffmpeg"):
            compressed = out_path.with_name("labelled_30fps_h264.mp4")
            cmd = [
                "ffmpeg", "-y", "-i", str(out_path),
                "-c:v", "libx264", "-crf", "28", "-preset", "fast",
                "-movflags", "+faststart",
                str(compressed),
            ]
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                old_size = out_path.stat().st_size / 1024 / 1024
                new_size = compressed.stat().st_size / 1024 / 1024
                log.info("  Compressed %.0f MB → %.0f MB (H.264 CRF 28)", old_size, new_size)
                out_path.unlink()  # Remove uncompressed
                compressed.rename(out_path)  # Replace with compressed
            else:
                log.warning("  ffmpeg compression failed, using uncompressed: %s", result.stderr[-200:])

        # Upload
        if no_upload:
            log.info("  Saved locally: %s", out_path)
            return str(out_path)

        log.info("  Uploading to s3://%s/%s", DERIV_BUCKET, upload_key)
        s3.upload_file(str(out_path), DERIV_BUCKET, upload_key)
        log.info("  Upload complete")
        return upload_key


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render DLC-labelled pose videos for hm2p sessions."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", type=str,
                       help="Process a single session (exp_id)")
    group.add_argument("--all", action="store_true",
                       help="Process all non-excluded sessions")

    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without downloading or rendering")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip sessions whose labelled_30fps.mp4 already exists on S3")
    parser.add_argument("--no-upload", action="store_true",
                        help="Save locally to /tmp/dlc_labelled/ instead of uploading")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load metadata
    sessions = load_sessions(METADATA_PATH)
    log.info("Loaded %d non-excluded sessions from %s", len(sessions), METADATA_PATH)

    # Filter to requested session
    if args.session:
        sessions = [s for s in sessions if s["exp_id"] == args.session]
        if not sessions:
            log.error("Session %s not found in metadata", args.session)
            sys.exit(1)

    # Set up S3 client
    # Try hm2p-agent profile, fall back to default
    try:
        boto_session = boto3.Session(profile_name="hm2p-agent")
        s3 = boto_session.client("s3", region_name="ap-southeast-2")
        s3.list_buckets()  # test credentials
    except Exception:
        s3 = boto3.client("s3", region_name="ap-southeast-2")

    # Process sessions
    results: list[tuple[str, str | None]] = []
    for i, s in enumerate(sessions, 1):
        exp_id = s["exp_id"]
        log.info("=== Session %d/%d: %s ===", i, len(sessions), exp_id)
        try:
            result = render_session(
                s3, exp_id,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
                no_upload=args.no_upload,
            )
            results.append((exp_id, result))
        except Exception:
            log.exception("Failed to process %s", exp_id)
            results.append((exp_id, None))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    rendered = sum(1 for _, r in results if r is not None and r not in ("skipped", "dry-run"))
    skipped = sum(1 for _, r in results if r == "skipped")
    dry_run_count = sum(1 for _, r in results if r == "dry-run")
    failed = sum(1 for _, r in results if r is None)
    print(f"  Rendered: {rendered}")
    if skipped:
        print(f"  Skipped (existing): {skipped}")
    if dry_run_count:
        print(f"  Dry run: {dry_run_count}")
    if failed:
        print(f"  Failed:  {failed}")
        for exp_id, r in results:
            if r is None:
                print(f"    - {exp_id}")
    print(f"  Total:   {len(results)}")


if __name__ == "__main__":
    main()
