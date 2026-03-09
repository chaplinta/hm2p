"""Retraining helper — extract frames and prepare labeling data for DLC.

Functions to extract poorly-tracked frames from videos and prepare them
for manual labeling in the DLC GUI. Supports both DLC 3.x project format
and standalone frame export.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hm2p.pose.quality import stratified_frame_selection, worst_frames

log = logging.getLogger(__name__)


def extract_frames_from_video(
    video_path: Path,
    frame_indices: npt.NDArray[np.intp],
    output_dir: Path,
    prefix: str = "frame",
) -> list[Path]:
    """Extract specific frames from a video file as images.

    Parameters
    ----------
    video_path : Path
        Path to the .mp4 video.
    frame_indices : (n,) int
        Frame numbers to extract (0-indexed).
    output_dir : Path
        Directory to write extracted frames.
    prefix : str
        Filename prefix for extracted images.

    Returns
    -------
    list of Path
        Paths to extracted frame images.

    Raises
    ------
    ImportError
        If cv2 (OpenCV) is not installed.
    FileNotFoundError
        If video file doesn't exist.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV required for frame extraction. "
            "Install: pip install opencv-python-headless"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    paths = []
    sorted_indices = sorted(frame_indices)

    for idx in sorted_indices:
        if idx >= total_frames:
            log.warning("Frame %d exceeds video length (%d), skipping", idx, total_frames)
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            log.warning("Failed to read frame %d", idx)
            continue

        out_path = output_dir / f"{prefix}_{idx:06d}.png"
        cv2.imwrite(str(out_path), frame)
        paths.append(out_path)

    cap.release()
    log.info("Extracted %d/%d frames to %s", len(paths), len(frame_indices), output_dir)
    return paths


def prepare_retraining_manifest(
    session_id: str,
    frame_indices: npt.NDArray[np.intp],
    frame_paths: list[Path],
    quality_bins: list[tuple[str, npt.NDArray[np.intp]]] | None = None,
    output_path: Path | None = None,
) -> dict:
    """Create a JSON manifest for retraining frame selection.

    The manifest records which frames were selected, why, and where
    the extracted images are. This helps organize labeling work.

    Parameters
    ----------
    session_id : str
        Session identifier.
    frame_indices : (n,) int
        Selected frame indices.
    frame_paths : list of Path
        Paths to extracted frame images.
    quality_bins : list of (label, indices) or None
        Quality bin assignments from stratified selection.
    output_path : Path or None
        If provided, writes manifest to this JSON file.

    Returns
    -------
    dict
        Manifest with session info, frame selections, and paths.
    """
    frames = []
    for idx, path in zip(frame_indices, frame_paths):
        bin_label = "unknown"
        if quality_bins:
            for label, bin_idx in quality_bins:
                if int(idx) in bin_idx.tolist():
                    bin_label = label
                    break
        frames.append({
            "frame_index": int(idx),
            "image_path": str(path),
            "quality_bin": bin_label,
        })

    manifest = {
        "session_id": session_id,
        "n_frames": len(frames),
        "frames": frames,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2))
        log.info("Manifest written to %s", output_path)

    return manifest


def select_retraining_frames(
    likelihood: npt.NDArray[np.floating],
    method: str = "stratified",
    n_frames: int = 20,
    min_spacing: int = 30,
) -> dict:
    """Select frames for retraining using the specified method.

    Parameters
    ----------
    likelihood : (n_frames, n_keypoints) or (n_frames,) float
        Per-frame confidence scores.
    method : str
        ``"worst"`` — pick the N worst frames.
        ``"stratified"`` — pick frames across quality bins.
    n_frames : int
        Total frames to select (approximate for stratified).
    min_spacing : int
        Minimum gap between selected frames.

    Returns
    -------
    dict
        ``"indices"`` — selected frame indices.
        ``"method"`` — method used.
        ``"bins"`` — quality bin info (stratified only).
    """
    if method == "worst":
        indices = worst_frames(likelihood, n_frames=n_frames, min_spacing=min_spacing)
        return {"indices": indices, "method": "worst", "bins": None}
    elif method == "stratified":
        n_per_bin = max(1, n_frames // 4)
        result = stratified_frame_selection(
            likelihood, n_per_bin=n_per_bin, min_spacing=min_spacing,
        )
        return {
            "indices": result["indices"],
            "method": "stratified",
            "bins": result["bins"],
        }
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'worst' or 'stratified'.")
