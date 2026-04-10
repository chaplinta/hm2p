#!/usr/bin/env python3
"""Interactive DLC labelling — pick sessions from a menu, label in napari.

Scans labeled-data/ for sessions with frames. Shows an interactive menu.
Pick a session → napari opens with all frames + existing labels.
Close napari → back to menu. Quit when done.

Usage:
    uv run python scripts/interactive_label.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DLC_PROJECT = Path("sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20")
LABELED_DIR = DLC_PROJECT / "labeled-data"

BODYPARTS = [
    "nose_tip", "left_ear", "right_ear", "implant_base_rear",
    "neck", "mid_back", "mouse_center", "tail_base",
]

COLORS = {
    "nose_tip": [1, 0, 0, 1],
    "left_ear": [0, 0, 1, 1],
    "right_ear": [0, 1, 1, 1],
    "implant_base_rear": [1, 0.65, 0, 1],
    "neck": [0.5, 0, 0.5, 1],
    "mid_back": [0, 0.8, 0, 1],
    "mouse_center": [1, 0.84, 0, 1],
    "tail_base": [1, 0, 1, 1],
}

SCORER = "tristan"


def scan_sessions() -> list[dict]:
    """Find all sessions with frames in labeled-data/."""
    if not LABELED_DIR.exists():
        return []

    sessions = []
    for d in sorted(LABELED_DIR.iterdir()):
        if not d.is_dir():
            continue
        pngs = sorted(d.glob("*.png"))
        if not pngs:
            continue

        # Count existing labels
        n_labelled = 0
        for h5_f in d.glob("CollectedData_*.h5"):
            try:
                df = pd.read_hdf(h5_f)
                # Count rows with at least one non-NaN coordinate
                n_labelled = int((~df.isna().all(axis=1)).sum())
                break
            except Exception:
                pass
        if n_labelled == 0:
            for csv_f in d.glob("CollectedData_*.csv"):
                try:
                    # DLC CSV: 3 header rows, first 3 columns are tuple index
                    df = pd.read_csv(csv_f, header=[0, 1, 2], index_col=[0, 1, 2])
                    n_labelled = int((~df.isna().all(axis=1)).sum())
                    break
                except Exception:
                    pass

        sessions.append({
            "dir": d,
            "name": d.name,
            "n_frames": len(pngs),
            "n_labelled": n_labelled,
        })

    return sessions


def label_session(session_dir: Path):
    """Open napari for a single session — single layer, colour-coded bodyparts."""
    import cv2
    import napari

    pngs = sorted(session_dir.glob("*.png"))
    if not pngs:
        print("  No frames found.")
        return

    # Load images
    images = []
    frame_names = []
    for png in pngs:
        img = cv2.imread(str(png))
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            frame_names.append(png.name)

    if not images:
        print("  No images loaded.")
        return

    # Pad to same size
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

    stack = np.stack(padded)

    # Load existing labels from H5 (more reliable than CSV parsing)
    df = None
    for h5_f in session_dir.glob("CollectedData_*.h5"):
        try:
            df = pd.read_hdf(h5_f)
            break
        except Exception:
            pass

    # Build single points array: (N, 3) coords + bodypart identity
    all_pts = []      # [frame, y, x]
    all_bp_ids = []   # bodypart index
    all_colors = []   # face color per point

    for bp_idx, bp in enumerate(BODYPARTS):
        for i, fname in enumerate(frame_names):
            if df is not None:
                matching = [idx for idx in df.index if fname in str(idx)]
                if matching:
                    try:
                        x = float(df.loc[matching[0], (SCORER, bp, "x")])
                        y = float(df.loc[matching[0], (SCORER, bp, "y")])
                        if not (np.isnan(x) or np.isnan(y)):
                            all_pts.append([i, y, x])
                            all_bp_ids.append(bp_idx)
                            all_colors.append(COLORS.get(bp, [1, 1, 1, 1]))
                    except (KeyError, ValueError):
                        pass

    pts_array = np.array(all_pts) if all_pts else np.empty((0, 3))
    colors_array = np.array(all_colors) if all_colors else np.empty((0, 4))
    bp_ids = np.array(all_bp_ids) if all_bp_ids else np.empty(0, dtype=int)

    # Current bodypart for new points
    current_bp = [0]

    # Open napari
    viewer = napari.Viewer(title=f"Label: {session_dir.name}")
    viewer.add_image(stack, name="frames")

    # Single points layer
    properties = {"bodypart": [BODYPARTS[i] for i in bp_ids]} if len(bp_ids) > 0 else {"bodypart": []}
    layer = viewer.add_points(
        pts_array,
        name="labels",
        properties=properties,
        face_color=colors_array if len(colors_array) > 0 else COLORS[BODYPARTS[0]],
        border_color="white",
        size=10,
        ndim=3,
    )

    # Status text showing current bodypart
    def _update_title():
        bp = BODYPARTS[current_bp[0]]
        color_name = bp
        viewer.title = f"Label: {session_dir.name} | Bodypart: {bp} ({current_bp[0]+1}/{len(BODYPARTS)}) | Press 1-8 to switch"

    _update_title()

    # When new points are added, assign current bodypart colour
    def _on_data_change(event):
        n_pts = len(layer.data)
        n_props = len(layer.properties.get("bodypart", []))
        if n_pts > n_props:
            # New point(s) added — assign current bodypart
            bp = BODYPARTS[current_bp[0]]
            new_props = list(layer.properties.get("bodypart", []))
            new_colors = list(layer.face_color) if len(layer.face_color) > 0 else []
            while len(new_props) < n_pts:
                new_props.append(bp)
                new_colors.append(COLORS.get(bp, [1, 1, 1, 1]))
            layer.properties = {"bodypart": new_props}
            layer.face_color = np.array(new_colors)

    layer.events.data.connect(_on_data_change)

    # Keyboard shortcuts: 1-8 to select bodypart
    for i in range(min(8, len(BODYPARTS))):
        bp_idx = i
        @viewer.bind_key(str(i + 1))
        def _select_bp(viewer, _idx=bp_idx):
            current_bp[0] = _idx
            _update_title()

    print(f"\n  Keyboard: 1-8 select bodypart, close window to save")
    print(f"  Current: {BODYPARTS[0]} (press 1-8 to switch)")

    napari.run()

    # Save on close — reconstruct per-bodypart DataFrame from single layer
    columns = pd.MultiIndex.from_tuples(
        [(SCORER, bp, coord) for bp in BODYPARTS for coord in ("x", "y")],
        names=["scorer", "bodyparts", "coords"],
    )
    index_tuples = [
        ("labeled-data", session_dir.name, fn)
        for fn in frame_names
    ]
    index = pd.MultiIndex.from_tuples(index_tuples)
    data = np.full((len(frame_names), len(BODYPARTS) * 2), np.nan)

    bp_names = layer.properties.get("bodypart", [])
    for pt_idx, pt in enumerate(layer.data):
        frame_idx = int(round(pt[0]))
        if 0 <= frame_idx < len(frame_names) and pt_idx < len(bp_names):
            bp = bp_names[pt_idx]
            if bp in BODYPARTS:
                bp_i = BODYPARTS.index(bp)
                data[frame_idx, bp_i * 2] = pt[2]      # x = col
                data[frame_idx, bp_i * 2 + 1] = pt[1]  # y = row

    df_out = pd.DataFrame(data, index=index, columns=columns)
    df_out.to_csv(session_dir / f"CollectedData_{SCORER}.csv")
    df_out.to_hdf(session_dir / f"CollectedData_{SCORER}.h5", key="df_with_missing", mode="w")

    labelled = int(np.any(~np.isnan(data), axis=1).sum())
    print(f"  Saved {labelled}/{len(frame_names)} labelled frames.")


def main():
    print("=" * 60)
    print("  hm2p Interactive DLC Labeller")
    print("=" * 60)

    while True:
        sessions = scan_sessions()

        if not sessions:
            print("\nNo sessions with frames found in labeled-data/.")
            print("Run scripts/select_labelling_frames.py first to extract frames.")
            break

        print(f"\n  {'#':<4} {'Session':<65} {'Frames':>6} {'Labelled':>9}")
        print("  " + "-" * 86)
        for i, s in enumerate(sessions):
            status = "✓" if s["n_labelled"] >= s["n_frames"] else " "
            print(f"  {i:<4} {s['name']:<65} {s['n_frames']:>6} {s['n_labelled']:>8} {status}")

        total_frames = sum(s["n_frames"] for s in sessions)
        total_labelled = sum(s["n_labelled"] for s in sessions)
        print(f"\n  Total: {total_frames} frames, {total_labelled} labelled "
              f"({total_labelled/total_frames*100:.0f}%)")

        print(f"\n  Enter session number (0-{len(sessions)-1}), 'a' for all, or 'q' to quit:")
        choice = input("  > ").strip()

        if choice.lower() == "q":
            break
        elif choice.lower() == "a":
            for i, s in enumerate(sessions):
                print(f"\n[{i+1}/{len(sessions)}] {s['name']}")
                label_session(s["dir"])
        elif choice.isdigit() and 0 <= int(choice) < len(sessions):
            s = sessions[int(choice)]
            print(f"\nOpening {s['name']} ({s['n_frames']} frames)...")
            label_session(s["dir"])
        else:
            print("  Invalid choice.")

    print("\nDone. Next steps:")
    print("  uv run python scripts/upload_dlc_labels.py")
    print("  uv run python scripts/launch_dlc_finetune_ec2.py")


if __name__ == "__main__":
    main()
