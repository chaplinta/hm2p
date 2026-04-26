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
import datetime
import json
import logging
import sys
import tempfile
import traceback
import urllib.request
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
    "head_midpoint": (0, 165, 255),      # orange
    "implant_base_rear": (0, 165, 255), # orange (legacy alias)
    "neck": (128, 0, 128),              # purple
    "mid_back": (0, 255, 0),            # green
    "mouse_center": (0, 255, 255),      # yellow
    "tail_base": (255, 0, 255),         # magenta
}

SKELETON: list[tuple[str, str]] = [
    ("nose_tip", "head_midpoint"),
    ("nose_tip", "left_ear"),
    ("nose_tip", "right_ear"),
    ("nose", "left_ear"),       # fallback for SuperAnimal "nose" name
    ("nose", "right_ear"),
    ("left_ear", "head_midpoint"),
    ("right_ear", "head_midpoint"),
    ("left_ear", "right_ear"),
    ("head_midpoint", "neck"),
    ("neck", "mid_back"),
    ("mid_back", "mouse_center"),
    ("mouse_center", "tail_base"),
]

# All possible bodypart names (finetuned + SuperAnimal variants)
BODYPARTS = list(KEYPOINT_COLORS.keys())

CIRCLE_RADIUS = 2
LINE_WIDTH = 1
# Always show all keypoints regardless of confidence. High-confidence
# points are filled circles, low-confidence are hollow. No threshold
# filtering — every estimated position is drawn.
CONFIDENCE_THRESHOLD = 0.0

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


def _get_instance_id() -> str:
    """Return the EC2 instance ID from the metadata service, or 'unknown'."""
    try:
        resp = urllib.request.urlopen(
            "http://169.254.169.254/latest/meta-data/instance-id", timeout=2
        )
        return resp.read().decode().strip()
    except Exception:
        return "unknown"

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
    """Load all sessions from experiments.csv.

    Per CLAUDE.md, pipeline stages must process all sessions regardless
    of the ``exclude`` flag — that flag is for analysis-time filtering
    only.
    """
    sessions = []
    with open(metadata_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    """Draw keypoints and skeleton on a single frame.

    Parameters
    ----------
    frame : BGR image array (at output resolution).
    keypoints : dict of bodypart -> (N, 3) arrays [x, y, likelihood].
        Coordinates are in original DLC resolution.
    frame_idx : DLC frame index.
    scale_x, scale_y : Scale factors from DLC coords to frame coords.
    """
    # Draw skeleton lines first (under circles)
    for bp1, bp2 in SKELETON:
        if bp1 not in keypoints or bp2 not in keypoints:
            continue
        kp1 = keypoints[bp1][frame_idx]
        kp2 = keypoints[bp2][frame_idx]
        if np.isnan(kp1[:2]).any() or np.isnan(kp2[:2]).any():
            continue
        if kp1[2] < CONFIDENCE_THRESHOLD or kp2[2] < CONFIDENCE_THRESHOLD:
            continue
        pt1 = (int(round(kp1[0] * scale_x)), int(round(kp1[1] * scale_y)))
        pt2 = (int(round(kp2[0] * scale_x)), int(round(kp2[1] * scale_y)))
        c1 = KEYPOINT_COLORS.get(bp1, (255, 255, 255))
        c2 = KEYPOINT_COLORS.get(bp2, (255, 255, 255))
        color = tuple((a + b) // 2 for a, b in zip(c1, c2))
        cv2.line(frame, pt1, pt2, color, LINE_WIDTH, cv2.LINE_AA)

    # Draw keypoint circles
    for bp, kp_array in keypoints.items():
        kp = kp_array[frame_idx]
        if np.isnan(kp[:2]).any():
            continue
        pt = (int(round(kp[0] * scale_x)), int(round(kp[1] * scale_y)))
        color = KEYPOINT_COLORS.get(bp, (255, 255, 255))
        if kp[2] >= CONFIDENCE_THRESHOLD:
            cv2.circle(frame, pt, CIRCLE_RADIUS, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, pt, CIRCLE_RADIUS, color, 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Main rendering
# ---------------------------------------------------------------------------

RENDER_MODES = {
    "raw": "labelled_30fps.mp4",
    "median": "labelled_median_30fps.mp4",
    "pipeline": "labelled_pipeline_30fps.mp4",
}


def _apply_median_filter(keypoints: dict[str, np.ndarray], window: int = 3) -> dict[str, np.ndarray]:
    """Apply rolling median filter to keypoint x/y, preserving likelihood.

    Default window of 3 frames at 30fps ≈ 100ms temporal smoothing.
    If the DLC frame rate changes, adjust to maintain ~100ms
    (window = round(0.1 * fps)).
    """
    from scipy.ndimage import median_filter

    filtered = {}
    for bp, arr in keypoints.items():
        out = arr.copy()
        for col in range(2):  # x, y only
            vals = arr[:, col]
            nan_mask = np.isnan(vals)
            if nan_mask.all():
                continue
            filled = vals.copy()
            if nan_mask.any():
                idx = np.arange(len(vals), dtype=float)
                valid = ~nan_mask
                filled[nan_mask] = np.interp(idx[nan_mask], idx[valid], vals[valid])
            out[:, col] = median_filter(filled, size=window, mode="nearest")
            # Don't restore NaN — keep interpolated values so labels
            # are always visible in the video
        filtered[bp] = out
    return filtered


def _apply_pipeline_filter(
    keypoints: dict[str, np.ndarray], conf_threshold: float = 0.05, window: int = 3, max_gap: int = 5,
) -> dict[str, np.ndarray]:
    """Apply pipeline-style filtering: confidence threshold → interpolate gaps → median."""
    filtered = {}
    for bp, arr in keypoints.items():
        out = arr.copy()
        # Set low-confidence to NaN
        low = out[:, 2] < conf_threshold
        out[low, 0] = np.nan
        out[low, 1] = np.nan
        # Interpolate short gaps
        for col in range(2):
            vals = out[:, col]
            nan_mask = np.isnan(vals)
            if nan_mask.all() or not nan_mask.any():
                continue
            # Find gap lengths
            idx = np.arange(len(vals))
            valid = ~nan_mask
            # Only interpolate gaps <= max_gap
            interped = vals.copy()
            interped[nan_mask] = np.interp(idx[nan_mask], idx[valid], vals[valid])
            # Restore long gaps
            gap_starts = np.where(np.diff(nan_mask.astype(int)) == 1)[0] + 1
            gap_ends = np.where(np.diff(nan_mask.astype(int)) == -1)[0] + 1
            if nan_mask[0]:
                gap_starts = np.r_[0, gap_starts]
            if nan_mask[-1]:
                gap_ends = np.r_[gap_ends, len(vals)]
            for gs, ge in zip(gap_starts, gap_ends):
                if ge - gs > max_gap:
                    interped[gs:ge] = np.nan
            out[:, col] = interped
        # Median filter
        from scipy.ndimage import median_filter as _mf
        for col in range(2):
            vals = out[:, col]
            nan_mask = np.isnan(vals)
            if nan_mask.all():
                continue
            filled = vals.copy()
            if nan_mask.any():
                ix = np.arange(len(vals), dtype=float)
                v = ~nan_mask
                filled[nan_mask] = np.interp(ix[nan_mask], ix[v], vals[v])
            out[:, col] = _mf(filled, size=window, mode="nearest")
            out[nan_mask, col] = np.nan
        filtered[bp] = out
    return filtered


def render_session(
    s3,
    exp_id: str,
    *,
    dry_run: bool = False,
    skip_existing: bool = False,
    no_upload: bool = False,
    modes: list[str] | None = None,
) -> str | None:
    """Render labelled video(s) for a single session.

    Parameters
    ----------
    modes : list of mode names to render. Default: all three
        ("raw", "median", "pipeline").

    Returns the S3 key of the last uploaded video, or None on failure.
    """
    if modes is None:
        modes = list(RENDER_MODES.keys())

    sub, ses = parse_exp_id(exp_id)
    log.info("Processing %s (%s/%s) modes=%s", exp_id, sub, ses, modes)

    # Check if all modes already exist
    if skip_existing and not no_upload:
        all_exist = all(
            s3_key_exists(s3, DERIV_BUCKET, f"pose/{sub}/{ses}/{RENDER_MODES[m]}")
            for m in modes
        )
        if all_exist:
            log.info("  Skipping — all modes exist on S3")
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
    from hm2p.pose.select import select_best_dlc_h5_s3
    pose_key = select_best_dlc_h5_s3(s3, DERIV_BUCKET, pose_prefix)
    if not pose_key:
        log.warning("  No DLC .h5 found for %s, skipping", exp_id)
        return None

    if dry_run:
        for m in modes:
            log.info("  [DRY RUN] %s → s3://%s/pose/%s/%s/%s",
                     m, DERIV_BUCKET, sub, ses, RENDER_MODES[m])
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
        keypoints_raw = extract_keypoints(df)
        n_dlc_frames = len(df)
        log.info("  DLC frames: %d, bodyparts: %s",
                 n_dlc_frames, list(keypoints_raw.keys()))

        # Build filtered keypoint variants
        keypoints_by_mode = {"raw": keypoints_raw}
        if "median" in modes:
            keypoints_by_mode["median"] = _apply_median_filter(keypoints_raw)
        if "pipeline" in modes:
            keypoints_by_mode["pipeline"] = _apply_pipeline_filter(keypoints_raw)

        # Open video
        cap = cv2.VideoCapture(str(video_local))
        if not cap.isOpened():
            log.error("  Failed to open video %s", video_local)
            return None

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale_x = OUTPUT_WIDTH / orig_w
        scale_y = OUTPUT_HEIGHT / orig_h
        log.info("  Video: %d frames (%dx%d), rendering %d modes",
                 total_video_frames, orig_w, orig_h, len(modes))

        # Build frame index mapping
        raw_fps = cap.get(cv2.CAP_PROP_FPS) or 100.0
        target_raw_frames = [
            round(i * raw_fps / OUTPUT_FPS) for i in range(n_dlc_frames)
        ]
        needed_raw = {}
        for dlc_idx, raw_idx in enumerate(target_raw_frames):
            needed_raw[raw_idx] = dlc_idx
        max_raw = max(needed_raw) if needed_raw else 0

        # Open one ffmpeg pipe per mode (single video read, multiple outputs)
        import shutil as _shutil
        import subprocess

        use_ffmpeg = bool(_shutil.which("ffmpeg"))
        pipes = {}  # mode → (out_path, ffproc or writer)
        for m in modes:
            fname = RENDER_MODES[m]
            if no_upload:
                out_path = LOCAL_OUTPUT_DIR / sub / ses / fname
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = tmp / fname

            if use_ffmpeg:
                # -loglevel error + -nostats suppress per-frame progress so
                # the stderr pipe stays small. Without this ffmpeg's periodic
                # encoding stats fill the 64 KB OS pipe buffer (stderr is not
                # drained until communicate() at the end), ffmpeg blocks on
                # its stderr write, stops reading stdin, and the loop's
                # stdin.write() deadlocks. Observed at ~20,000 frames in.
                ffproc = subprocess.Popen(
                    ["ffmpeg", "-y", "-loglevel", "error", "-nostats",
                     "-f", "rawvideo", "-pix_fmt", "bgr24",
                     "-s", f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}", "-r", str(OUTPUT_FPS),
                     "-i", "pipe:0", "-c:v", "libx264", "-crf", "23",
                     "-preset", "medium", "-movflags", "+faststart", str(out_path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,   # only errors now; safe to capture
                )
                pipes[m] = (out_path, ffproc, None)
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(out_path), fourcc, OUTPUT_FPS,
                    (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                )
                pipes[m] = (out_path, None, writer)

        # Single-pass render: read each frame once, draw for each mode
        written = 0
        raw_idx = 0
        while raw_idx <= max_raw:
            ret, frame_orig = cap.read()
            if not ret:
                break

            if raw_idx in needed_raw:
                dlc_frame_idx = needed_raw[raw_idx]
                frame_resized = cv2.resize(
                    frame_orig, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )

                for m in modes:
                    frame_copy = frame_resized.copy()
                    kps = keypoints_by_mode[m]
                    frame_copy = draw_frame(frame_copy, kps, dlc_frame_idx,
                                            scale_x=scale_x, scale_y=scale_y)

                    out_path, ffproc, writer = pipes[m]
                    if ffproc is not None:
                        ffproc.stdin.write(frame_copy.tobytes())
                    else:
                        writer.write(frame_copy)

                written += 1
                if written % 5000 == 0:
                    log.info("  Rendered %d / %d frames", written, n_dlc_frames)

            raw_idx += 1

        cap.release()
        for m in modes:
            out_path, ffproc, writer = pipes[m]
            if ffproc is not None:
                _, stderr_bytes = ffproc.communicate()
                rc = ffproc.returncode
                if rc != 0:
                    log.error(
                        "  ffmpeg exited with code %d for mode=%s session=%s",
                        rc, m, exp_id,
                    )
                    log.error(
                        "  ffmpeg stderr: %s",
                        stderr_bytes[-500:].decode(errors="replace"),
                    )
                    # Remove the partial output; mark mode as failed
                    out_path.unlink(missing_ok=True)
                    pipes[m] = (None, ffproc, writer)
            if writer is not None:
                writer.release()
        log.info("  Wrote %d frames × %d modes", written, len(modes))

        # Upload all modes that completed successfully (out_path not None)
        last_key = None
        for m in modes:
            out_path = pipes[m][0]
            if out_path is None:
                log.warning("  Skipping upload for mode=%s (ffmpeg failed)", m)
                continue
            upload_key = f"pose/{sub}/{ses}/{RENDER_MODES[m]}"
            if no_upload:
                log.info("  Saved locally: %s", out_path)
            else:
                log.info("  Uploading %s → s3://%s/%s", m, DERIV_BUCKET, upload_key)
                s3.upload_file(str(out_path), DERIV_BUCKET, upload_key)
                last_key = upload_key

        return last_key or str(pipes[modes[0]][0]) if pipes[modes[0]][0] else None


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
                       help="Process all sessions in experiments.csv")

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
    log.info("Loaded %d sessions from %s", len(sessions), METADATA_PATH)

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

    run_id = datetime.datetime.utcnow().isoformat() + "Z"
    instance_id = _get_instance_id()
    error_records: list[dict] = []

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
        except Exception as exc:
            tb = traceback.format_exc()
            log.exception("Failed to process %s", exp_id)
            error_records.append({
                "session": exp_id,
                "stage": "render",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": tb,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            })
            results.append((exp_id, None))

    # Upload structured error summary — always written, even if empty.
    errors_payload = json.dumps(
        {"run_id": run_id, "instance_id": instance_id, "errors": error_records},
        indent=2,
    ).encode()
    try:
        s3.put_object(
            Bucket=DERIV_BUCKET,
            Key="dlc-retrain/_render_errors.json",
            Body=errors_payload,
        )
        log.info("Render error summary uploaded (%d error(s))", len(error_records))
    except Exception as e:
        log.warning("Could not upload _render_errors.json: %s", e)

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
