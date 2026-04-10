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

        # Count existing labels (rows with at least one non-NaN coordinate)
        n_labelled = 0
        for csv_f in d.glob("CollectedData_*.csv"):
            try:
                df = pd.read_csv(csv_f, header=[0, 1, 2], index_col=0)
                # Count rows that have at least one labelled bodypart
                n_labelled = int((~df.isna().all(axis=1)).sum())
            except Exception:
                pass
        if n_labelled == 0:
            for h5_f in d.glob("CollectedData_*.h5"):
                try:
                    df = pd.read_hdf(h5_f)
                    n_labelled = int((~df.isna().all(axis=1)).sum())
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
    """Open napari for a single session with all frames + existing labels."""
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

    # Load existing labels
    existing_labels = {}
    csv_files = list(session_dir.glob("CollectedData_*.csv"))
    h5_files = list(session_dir.glob("CollectedData_*.h5"))
    df = None
    if h5_files:
        try:
            df = pd.read_hdf(h5_files[0])
        except Exception:
            pass
    if df is None and csv_files:
        try:
            df = pd.read_csv(csv_files[0], header=[0, 1, 2], index_col=0)
        except Exception:
            pass

    for bp in BODYPARTS:
        pts = []
        for i, fname in enumerate(frame_names):
            if df is not None:
                matching = [idx for idx in df.index if fname in str(idx)]
                if matching:
                    try:
                        x = float(df.loc[matching[0], (SCORER, bp, "x")])
                        y = float(df.loc[matching[0], (SCORER, bp, "y")])
                        if not (np.isnan(x) or np.isnan(y)):
                            pts.append([i, y, x])
                    except (KeyError, ValueError):
                        pass
        existing_labels[bp] = np.array(pts) if pts else np.empty((0, 3))

    # Open napari
    viewer = napari.Viewer(title=f"Label: {session_dir.name}")
    viewer.add_image(stack, name="frames")

    point_layers = {}
    for bp in BODYPARTS:
        layer = viewer.add_points(
            existing_labels[bp],
            name=bp,
            face_color=COLORS.get(bp, [1, 1, 1, 1]),
            border_color="white",
            size=8,
            ndim=3,
        )
        point_layers[bp] = layer

    napari.run()

    # Save on close
    columns = pd.MultiIndex.from_tuples(
        [(SCORER, bp, coord) for bp in BODYPARTS for coord in ("x", "y")],
        names=["scorer", "bodyparts", "coords"],
    )
    index = [
        f"labeled-data/{session_dir.name}/{fn}"
        for fn in frame_names
    ]
    data = np.full((len(frame_names), len(BODYPARTS) * 2), np.nan)

    for bp_idx, bp in enumerate(BODYPARTS):
        if bp in point_layers:
            for pt in point_layers[bp].data:
                frame_idx = int(round(pt[0]))
                if 0 <= frame_idx < len(frame_names):
                    data[frame_idx, bp_idx * 2] = pt[2]      # x = col
                    data[frame_idx, bp_idx * 2 + 1] = pt[1]  # y = row

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
