#!/usr/bin/env python3
"""Prepare frames for DLC retraining — downloads video, extracts frames, sets up DLC project.

Run from the repo root on your Mac:
    uv run python scripts/prepare_retrain_frames.py sub-1114353/ses-20210823T165950 606 2093 8793 14567

Or pass a JSON file of frame indices:
    uv run python scripts/prepare_retrain_frames.py sub-1114353/ses-20210823T165950 --frames-file retrain_frames.json

This script:
1. Downloads the overhead video from S3
2. Extracts the specified frames as PNG images
3. Creates a DLC project (if one doesn't exist)
4. Copies frames into the project's labeled-data folder
5. Prints the command to open the DLC labeling GUI
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download video + extract frames for DLC retraining",
    )
    parser.add_argument(
        "session",
        help="Session key: sub-XXXXX/ses-YYYYMMDDT...",
    )
    parser.add_argument(
        "frames",
        nargs="*",
        type=int,
        help="Frame indices to extract (space-separated)",
    )
    parser.add_argument(
        "--frames-file",
        type=Path,
        help="JSON file with list of frame indices",
    )
    parser.add_argument(
        "--rawdata-bucket",
        default="hm2p-rawdata",
    )
    parser.add_argument(
        "--dlc-project-name",
        default="hm2p-retrain",
    )
    parser.add_argument(
        "--experimenter",
        default="tristan",
    )
    args = parser.parse_args()

    sub, ses = args.session.split("/")
    session_tag = f"{sub}_{ses}"

    # Get frame indices
    if args.frames_file:
        frame_indices = np.array(json.loads(args.frames_file.read_text()))
    elif args.frames:
        frame_indices = np.array(args.frames)
    else:
        print("ERROR: provide frame indices as arguments or via --frames-file")
        sys.exit(1)

    print(f"Session: {sub}/{ses}")
    print(f"Frames: {len(frame_indices)}")

    # Step 1: Download video
    video_dir = Path(f"/tmp/{session_tag}")
    video_dir.mkdir(parents=True, exist_ok=True)
    s3_prefix = f"s3://{args.rawdata_bucket}/rawdata/{sub}/{ses}/behav/"

    print(f"\n--- Downloading video from {s3_prefix} ---")
    subprocess.run(
        ["aws", "s3", "sync", s3_prefix, str(video_dir),
         "--exclude", "*", "--include", "*.mp4", "--exclude", "*side*"],
        check=True,
    )

    mp4s = list(video_dir.glob("*overhead*.mp4")) + list(video_dir.glob("*cropped*.mp4"))
    if not mp4s:
        mp4s = list(video_dir.glob("*.mp4"))
    if not mp4s:
        print("ERROR: no video found")
        sys.exit(1)
    video_path = mp4s[0]
    print(f"Video: {video_path}")

    # Step 2: Extract frames
    output_dir = Path(f"retrain_frames/{session_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Extracting {len(frame_indices)} frames ---")
    try:
        from hm2p.pose.retrain import extract_frames_from_video
        extract_frames_from_video(
            video_path=video_path,
            frame_indices=frame_indices,
            output_dir=output_dir,
        )
    except ImportError:
        # Fallback: use cv2 directly
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        for idx in sorted(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                out_path = output_dir / f"img{idx:06d}.png"
                cv2.imwrite(str(out_path), frame)
        cap.release()
    print(f"Extracted to {output_dir}/")

    # Step 3: Find or create DLC project
    dlc_base = Path("sourcedata/trackers/dlc")
    dlc_base.mkdir(parents=True, exist_ok=True)
    projects = sorted(dlc_base.glob(f"{args.dlc_project_name}-{args.experimenter}-*"))

    # SuperAnimal TopViewMouse bodyparts — must match for fine-tuning
    BODYPARTS = ["left_ear", "right_ear", "mid_back", "mouse_center", "tail_base"]
    SKELETON = [
        ["left_ear", "right_ear"],
        ["right_ear", "mid_back"],
        ["mid_back", "mouse_center"],
        ["mouse_center", "tail_base"],
    ]

    if projects:
        project_dir = projects[-1]
        print(f"\n--- Using existing DLC project: {project_dir} ---")
    else:
        print(f"\n--- Creating DLC project ---")
        import deeplabcut
        config_path = deeplabcut.create_new_project(
            args.dlc_project_name,
            args.experimenter,
            [str(video_path)],
            working_directory=str(dlc_base),
            copy_videos=False,
        )
        project_dir = Path(config_path).parent
        print(f"Created: {project_dir}")

    # Fix config.yaml bodyparts to match SuperAnimal TopViewMouse
    config_path = project_dir / "config.yaml"
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if cfg.get("bodyparts") != BODYPARTS:
        print(f"  Updating bodyparts: {cfg.get('bodyparts')} → {BODYPARTS}")
        cfg["bodyparts"] = BODYPARTS
        cfg["skeleton"] = SKELETON
        # Set up SuperAnimal conversion table for fine-tuning
        # Format: {superanimal_model: {project_bp: superanimal_bp}}
        cfg["SuperAnimalConversionTables"] = {
            "superanimal_topviewmouse": {bp: bp for bp in BODYPARTS}
        }
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"  Config updated: {config_path}")

    # Step 4: Copy frames into labeled-data
    # DLC expects frames in a subdirectory named after the video (stem without extension).
    # Read config.yaml to find the expected directory name.
    video_stem = video_path.stem
    labeled_dir = project_dir / "labeled-data" / video_stem
    labeled_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    for img in output_dir.glob("*.png"):
        dest = labeled_dir / img.name
        if not dest.exists():
            shutil.copy2(img, dest)

    n_copied = len(list(labeled_dir.glob("*.png")))
    print(f"Copied {n_copied} frames to {labeled_dir}/")

    # Step 5: Save frame indices to metadata for reproducibility
    indices_dir = Path("metadata/retrain_frames")
    indices_dir.mkdir(parents=True, exist_ok=True)
    indices_file = indices_dir / f"{session_tag}.json"
    indices_file.write_text(json.dumps({
        "session": f"{sub}/{ses}",
        "video": video_path.name,
        "frame_indices": frame_indices.tolist(),
        "n_frames": len(frame_indices),
    }, indent=2))
    print(f"\nSaved frame indices to {indices_file}")

    # Step 6: Open labeling GUI
    config_path = project_dir / "config.yaml"
    print(f"\n--- Opening DLC labeling GUI ---")
    print(f"Config: {config_path}")
    print("Label all frames, then close the napari window.\n")

    import deeplabcut
    deeplabcut.label_frames(str(config_path))

    # Keep napari event loop running until the user closes the window
    try:
        import napari
        napari.run()
    except (ImportError, RuntimeError):
        pass

    print(f"\n{'='*60}")
    print("Labeling complete. Next steps:\n")
    print("1. Upload your labels to S3:")
    print("     uv run python scripts/upload_dlc_labels.py\n")
    print("2. Launch GPU training on AWS (fine-tunes SuperAnimal on your labels,")
    print("   then re-runs inference on all 26 sessions):")
    print("     uv run python scripts/launch_dlc_retrain_ec2.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
