#!/usr/bin/env python3
"""Patch DLC 3.0rc13/rc14 memory_replay.py to handle missing bboxes/bodyparts.

When the SA detector finds no animal in a frame, the predictions dict
is missing 'bboxes' and 'bodyparts' keys, causing KeyError in both
get_pose_predictions (line ~120) and prepare_memory_replay_dataset
(line ~217).

This script patches the installed DLC to use .get() with empty list
defaults at both locations. Safe to run multiple times (idempotent).

Run on EC2 after pip install deeplabcut, before training:
    python3 scripts/patch_dlc_memory_replay.py
"""

import inspect
import sys
from pathlib import Path


def patch() -> bool:
    """Apply the memory_replay patch. Returns True if patched."""
    try:
        import deeplabcut
    except Exception as e:
        print(f"Cannot import deeplabcut: {e}")
        return False

    mr_path = (
        Path(inspect.getfile(deeplabcut)).parent
        / "pose_estimation_pytorch"
        / "modelzoo"
        / "memory_replay.py"
    )

    if not mr_path.exists():
        print(f"memory_replay.py not found at {mr_path}")
        return False

    code = mr_path.read_text()

    if "# hm2p-patched" in code:
        print("Already patched.")
        return True

    patched = False

    # Patch 1: get_pose_predictions serialisation (line ~120)
    old1 = '"bboxes": predictions["bboxes"].tolist(),'
    new1 = '"bboxes": predictions.get("bboxes", []) if not hasattr(predictions.get("bboxes"), "tolist") else predictions["bboxes"].tolist(),  # hm2p-patched'

    # Simpler: just guard with .get() and handle both cases
    # Actually simplest: add setdefault before the dict comprehension
    old_block = "    for image, prediction in zip(images_to_process, predictions):\n        sa_predictions[image] = prediction"
    new_block = (
        "    for image, prediction in zip(images_to_process, predictions):\n"
        "        prediction.setdefault('bboxes', [])  # hm2p-patched\n"
        "        prediction.setdefault('bodyparts', [])  # hm2p-patched\n"
        "        sa_predictions[image] = prediction"
    )

    if old_block in code:
        code = code.replace(old_block, new_block)
        patched = True
        print("Patched get_pose_predictions (setdefault for bboxes/bodyparts)")

    # Patch 2: prepare_memory_replay_dataset consumption (line ~217)
    old2 = 'prediction["bboxes"]'
    new2 = 'prediction.get("bboxes", [])'

    # Only replace in the prepare_memory_replay_dataset context
    # Find the function and replace within it
    old_line2 = 'bbox_preds = [xywh2xyxy(pred) for pred in prediction["bboxes"]]'
    new_line2 = 'bbox_preds = [xywh2xyxy(pred) for pred in prediction.get("bboxes", [])]  # hm2p-patched'

    if old_line2 in code:
        code = code.replace(old_line2, new_line2)
        patched = True
        print("Patched prepare_memory_replay_dataset (bbox_preds .get())")

    # Patch 3: bodyparts access in the same function
    old_line3 = 'matched_pred = prediction["bodyparts"][optimal_index]'
    new_line3 = 'matched_pred = prediction.get("bodyparts", [None] * (optimal_index + 1))[optimal_index] if prediction.get("bodyparts") and optimal_index < len(prediction.get("bodyparts", [])) else None  # hm2p-patched'

    if old_line3 in code:
        code = code.replace(old_line3, new_line3)
        patched = True
        print("Patched prepare_memory_replay_dataset (bodyparts access)")

    if patched:
        mr_path.write_text(code)
        print(f"Wrote patched file: {mr_path}")
    else:
        print("No patchable code found (different DLC version?)")

    return patched


if __name__ == "__main__":
    success = patch()
    sys.exit(0 if success else 1)
