#!/usr/bin/env python3
"""Render all labeled training frames with body part annotations overlaid.

For each labeled frame, draws colored circles at each keypoint position
and writes the annotated image to a review directory.

Output naming: {session_short}_{extracted_frame_num}_{full_session_id}_{movie_frame_num}.png
Example: 20210823_1114353_03_20210823_17_00_04_1114353_maze-rose_overhead.camera-cropped_002087.png

Usage:
    uv run python scripts/render_labeled_frames.py
    uv run python scripts/render_labeled_frames.py --output-dir /tmp/labeled_review
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LABELED_DIR = Path("sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/labeled-data")
DEFAULT_OUTPUT = Path("review_frames")

BODYPARTS = [
    "nose_tip", "left_ear", "right_ear", "head_midpoint",
    "neck", "mid_back", "mouse_center", "tail_base",
]

# BGR colors for OpenCV
BP_COLORS = {
    "nose_tip": (0, 0, 255),
    "left_ear": (255, 0, 0),
    "right_ear": (255, 255, 0),
    "head_midpoint": (0, 165, 255),
    "neck": (128, 0, 128),
    "mid_back": (0, 255, 0),
    "mouse_center": (0, 255, 255),
    "tail_base": (255, 0, 255),
}

SKELETON = [
    ("nose_tip", "head_midpoint"),
    ("nose_tip", "left_ear"),
    ("nose_tip", "right_ear"),
    ("left_ear", "head_midpoint"),
    ("right_ear", "head_midpoint"),
    ("left_ear", "right_ear"),
    ("head_midpoint", "neck"),
    ("neck", "mid_back"),
    ("mid_back", "mouse_center"),
    ("mouse_center", "tail_base"),
]


def short_session(clip_name: str) -> str:
    parts = clip_name.split("_")
    if len(parts) >= 5:
        return f"{parts[0]}_{parts[4].split('-')[0]}"
    return clip_name[:20]


def main():
    parser = argparse.ArgumentParser(description="Render labeled frames with annotations")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for d in sorted(LABELED_DIR.iterdir()):
        if not d.is_dir():
            continue

        h5 = d / "CollectedData_tristan.h5"
        if not h5.exists():
            continue

        try:
            df = pd.read_hdf(h5)
        except Exception:
            continue

        if len(df) == 0:
            continue

        scorer = df.columns.get_level_values(0)[0]
        short = short_session(d.name)
        full_id = d.name

        for i, idx in enumerate(df.index):
            frame_file = idx[2] if isinstance(idx, tuple) else str(idx).split("/")[-1]
            m = re.match(r"frame_(\d+)\.png", frame_file)
            movie_frame = m.group(1) if m else "000000"

            # Load image
            png_path = d / frame_file
            if png_path.is_symlink():
                png_path = png_path.resolve()
            if not png_path.exists():
                continue

            img = cv2.imread(str(png_path))
            if img is None:
                continue

            row = df.iloc[i]

            # Draw skeleton first (under circles)
            for bp1, bp2 in SKELETON:
                try:
                    x1 = row[(scorer, bp1, "x")]
                    y1 = row[(scorer, bp1, "y")]
                    x2 = row[(scorer, bp2, "x")]
                    y2 = row[(scorer, bp2, "y")]
                except KeyError:
                    continue
                if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                    continue
                c1 = BP_COLORS.get(bp1, (255, 255, 255))
                c2 = BP_COLORS.get(bp2, (255, 255, 255))
                color = tuple((a + b) // 2 for a, b in zip(c1, c2))
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1, cv2.LINE_AA)

            # Draw keypoints
            for bp in BODYPARTS:
                try:
                    x = row[(scorer, bp, "x")]
                    y = row[(scorer, bp, "y")]
                except KeyError:
                    continue
                if np.isnan(x) or np.isnan(y):
                    continue
                color = BP_COLORS.get(bp, (255, 255, 255))
                cv2.circle(img, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
                cv2.circle(img, (int(x), int(y)), 4, (0, 0, 0), 1, cv2.LINE_AA)

            # Add text label
            label = f"{short} #{i+1}"
            cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            # Save
            out_name = f"{short}_{i+1:02d}_{full_id}_{movie_frame}.png"
            cv2.imwrite(str(output_dir / out_name), img)
            total += 1

        print(f"  {short}: {len(df)} frames rendered")

    print(f"\n{total} frames written to {output_dir}/")


if __name__ == "__main__":
    main()
