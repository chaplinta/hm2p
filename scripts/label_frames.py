#!/usr/bin/env python3
"""Custom DLC labelling tool — all sessions in one napari window.

Bypasses DLC's broken label_frames() (rc13 only shows first session).
Loads all frames from all sessions in labeled-data/, displays existing
labels as coloured points, and saves back to CollectedData format on close.

Usage:
    uv run python scripts/label_frames.py                    # all sessions
    uv run python scripts/label_frames.py --session 1114353  # filter by animal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DLC_PROJECT = Path("sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20")
CONFIG_PATH = DLC_PROJECT / "config.yaml"
LABELED_DIR = DLC_PROJECT / "labeled-data"

BODYPARTS = [
    "nose_tip", "left_ear", "right_ear", "implant_base_rear",
    "neck", "mid_back", "mouse_center", "tail_base",
]

COLORS = {
    "nose_tip": [1, 0, 0, 1],          # red
    "left_ear": [0, 0, 1, 1],          # blue
    "right_ear": [0, 1, 1, 1],         # cyan
    "implant_base_rear": [1, 0.65, 0, 1],  # orange
    "neck": [0.5, 0, 0.5, 1],          # purple
    "mid_back": [0, 0.8, 0, 1],        # green
    "mouse_center": [1, 0.84, 0, 1],   # gold
    "tail_base": [1, 0, 1, 1],         # magenta
}

SCORER = "tristan"


def load_session_data(session_dir: Path) -> dict:
    """Load frames and existing labels for a session."""
    import cv2

    # Load images
    pngs = sorted(session_dir.glob("*.png"))
    if not pngs:
        return None

    images = []
    frame_names = []
    for png in pngs:
        img = cv2.imread(str(png))
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            frame_names.append(png.name)

    if not images:
        return None

    # Pad images to the same size (some sessions have mixed resolutions)
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    padded = []
    for img in images:
        if img.shape[0] != max_h or img.shape[1] != max_w:
            pad = np.zeros((max_h, max_w, 3), dtype=img.dtype)
            pad[:img.shape[0], :img.shape[1]] = img
            padded.append(pad)
        else:
            padded.append(img)
    images = padded

    # Load existing labels
    labels = {}  # bodypart -> list of (frame_idx, x, y) or None per frame
    csv_files = list(session_dir.glob("CollectedData_*.csv"))
    h5_files = list(session_dir.glob("CollectedData_*.h5"))

    if h5_files:
        df = pd.read_hdf(h5_files[0])
    elif csv_files:
        df = pd.read_csv(csv_files[0], header=[0, 1, 2], index_col=0)
    else:
        df = None

    for bp in BODYPARTS:
        bp_points = []
        for i, fname in enumerate(frame_names):
            if df is not None:
                # Find this frame in the dataframe
                matching = [idx for idx in df.index if fname in str(idx)]
                if matching:
                    row = df.loc[matching[0]]
                    try:
                        x = float(row[(SCORER, bp, "x")])
                        y = float(row[(SCORER, bp, "y")])
                        if not (np.isnan(x) or np.isnan(y)):
                            bp_points.append((i, y, x))  # napari uses (frame, row, col)
                            continue
                    except (KeyError, ValueError):
                        pass
                bp_points.append(None)
            else:
                bp_points.append(None)
        labels[bp] = bp_points

    return {
        "images": np.stack(images),
        "frame_names": frame_names,
        "labels": labels,
        "session_dir": session_dir,
        "session_name": session_dir.name,
    }


def save_labels(session_data: dict, point_layers: dict) -> None:
    """Save napari point layers back to CollectedData CSV + H5."""
    session_dir = session_data["session_dir"]
    frame_names = session_data["frame_names"]
    n_frames = len(frame_names)

    # Build the multi-index DataFrame
    columns = pd.MultiIndex.from_tuples(
        [(SCORER, bp, coord) for bp in BODYPARTS for coord in ("x", "y")],
        names=["scorer", "bodyparts", "coords"],
    )

    # Index: labeled-data/session_name/frame_name.png
    index = [
        f"labeled-data/{session_data['session_name']}/{fn}"
        for fn in frame_names
    ]

    data = np.full((n_frames, len(BODYPARTS) * 2), np.nan)

    for bp_idx, bp in enumerate(BODYPARTS):
        if bp not in point_layers:
            continue
        points = point_layers[bp].data  # (N, 3) — frame, row, col
        for pt in points:
            frame_idx = int(round(pt[0]))
            if 0 <= frame_idx < n_frames:
                data[frame_idx, bp_idx * 2] = pt[2]      # x = col
                data[frame_idx, bp_idx * 2 + 1] = pt[1]  # y = row

    df = pd.DataFrame(data, index=index, columns=columns)

    # Save CSV
    csv_path = session_dir / f"CollectedData_{SCORER}.csv"
    df.to_csv(csv_path)

    # Save H5
    h5_path = session_dir / f"CollectedData_{SCORER}.h5"
    df.to_hdf(h5_path, key="df_with_missing", mode="w")

    print(f"  Saved {n_frames} frames to {csv_path.name} + {h5_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Label DLC frames in napari")
    parser.add_argument("--session", type=str, help="Filter sessions by substring (e.g. animal ID)")
    args = parser.parse_args()

    import napari

    if not LABELED_DIR.exists():
        print(f"No labeled-data directory at {LABELED_DIR}")
        sys.exit(1)

    # Find all session directories
    session_dirs = sorted([
        d for d in LABELED_DIR.iterdir()
        if d.is_dir() and list(d.glob("*.png"))
    ])

    if args.session:
        session_dirs = [d for d in session_dirs if args.session in d.name]

    if not session_dirs:
        print("No sessions found with frames to label.")
        sys.exit(1)

    print(f"Found {len(session_dirs)} sessions")

    # Load all sessions
    all_sessions = []
    for sd in session_dirs:
        data = load_session_data(sd)
        if data:
            all_sessions.append(data)
            print(f"  {sd.name}: {len(data['frame_names'])} frames")

    if not all_sessions:
        print("No frames found.")
        sys.exit(1)

    # Build combined image stack with session boundaries
    # Each session is a separate "slice" in the viewer
    # We use a session selector widget to switch between them

    viewer = napari.Viewer(title="hm2p DLC Labeller")

    # Track current session
    current_session_idx = [0]
    point_layers_per_session = {}

    def load_session_into_viewer(idx: int):
        """Load a session's images and labels into the viewer."""
        # Remove old layers
        while len(viewer.layers) > 0:
            viewer.layers.pop(0)

        data = all_sessions[idx]
        viewer.add_image(data["images"], name=data["session_name"])

        # Add point layers per bodypart
        layers = {}
        for bp in BODYPARTS:
            existing_points = [p for p in data["labels"].get(bp, []) if p is not None]
            pts = np.array(existing_points) if existing_points else np.empty((0, 3))

            layer = viewer.add_points(
                pts,
                name=bp,
                face_color=COLORS.get(bp, [1, 1, 1, 1]),
                border_color="white",
                size=8,
                ndim=3,
            )
            layers[bp] = layer

        point_layers_per_session[idx] = layers
        viewer.title = f"hm2p DLC Labeller — {data['session_name']} ({idx + 1}/{len(all_sessions)})"
        viewer.dims.set_point(0, 0)  # Go to first frame

    def save_current_session():
        """Save the current session's labels."""
        idx = current_session_idx[0]
        if idx in point_layers_per_session:
            save_labels(all_sessions[idx], point_layers_per_session[idx])

    def next_session():
        save_current_session()
        idx = min(current_session_idx[0] + 1, len(all_sessions) - 1)
        current_session_idx[0] = idx
        load_session_into_viewer(idx)

    def prev_session():
        save_current_session()
        idx = max(current_session_idx[0] - 1, 0)
        current_session_idx[0] = idx
        load_session_into_viewer(idx)

    # Add keyboard shortcuts
    @viewer.bind_key("n")
    def _next(viewer):
        next_session()

    @viewer.bind_key("p")
    def _prev(viewer):
        prev_session()

    @viewer.bind_key("s")
    def _save(viewer):
        save_current_session()
        print(f"  Saved session {current_session_idx[0] + 1}/{len(all_sessions)}")

    # Load first session
    load_session_into_viewer(0)

    print(f"\nKeyboard shortcuts:")
    print(f"  n — next session")
    print(f"  p — previous session")
    print(f"  s — save current session")
    print(f"  Close window — save and exit")
    print(f"\nTo label: select a bodypart layer, click 'Add points' mode, click on image.")
    print(f"To edit: select a point and drag it.")
    print(f"To delete: select a point and press Delete.")

    napari.run()

    # Save on close
    save_current_session()
    print(f"\nDone. Labelled {len(all_sessions)} sessions.")
    print(f"\nNext steps:")
    print(f"  uv run python scripts/upload_dlc_labels.py")
    print(f"  uv run python scripts/launch_dlc_finetune_ec2.py")


if __name__ == "__main__":
    main()
