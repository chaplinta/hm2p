# DLC Retraining Workflow

Guide for manually labelling frames and retraining the DLC pose model.

## Overview

The hm2p project uses DeepLabCut's SuperAnimal TopViewMouse model for pose
estimation. When tracking quality is poor for certain sessions, the model can
be fine-tuned on manually labelled frames. This document covers the full
workflow from frame selection to retraining.

## Bodyparts

7 of SuperAnimal's ~13 bodyparts are labelled and tracked:

| Bodypart | Purpose |
|---|---|
| `nose_tip` | HD estimate (nose→neck midline), exploration |
| `left_ear` | Primary HD (ear vector perpendicular) |
| `right_ear` | Primary HD (ear vector perpendicular) |
| `neck` | HD fallback (nose→neck axis), head-body dissociation |
| `mid_back` | Body axis |
| `mouse_center` | Position, speed |
| `tail_base` | Body orientation |

4 head keypoints (nose, ears, neck) enable robust HD fusion with
confidence-weighted fallback when individual points are occluded
(e.g. nose behind the 2P implant). The
`SuperAnimalConversionTables` in `config.yaml` maps these 7 to the matching
SuperAnimal keypoints, so fine-tuning transfers the pre-trained backbone
weights for just these bodyparts.

## Step 1: Select frames for labelling

Use the **Tracking Quality** page in the frontend (Pipeline > Tracking QC) to
identify poorly-tracked sessions and select frames. Or run directly:

```bash
uv run python scripts/prepare_retrain_frames.py sub-XXXXX/ses-YYYYMMDDT... 606 2093 8793
```

This script:
1. Downloads the overhead video from S3
2. Extracts the specified frames as PNGs to `retrain_frames/`
3. Creates (or reuses) a DLC project at `sourcedata/trackers/dlc/hm2p-retrain-*/`
4. Copies PNGs into `labeled-data/` and opens the napari labelling GUI
5. Saves frame indices to `metadata/retrain_frames/<session>.json`

## Step 2: Label frames

The napari GUI opens automatically. Label all 5 bodyparts on each frame, then
close the window. Labels are saved as:

```
sourcedata/trackers/dlc/hm2p-retrain-<experimenter>-<date>/
  config.yaml
  labeled-data/<video_stem>/
    CollectedData_<experimenter>.csv   # <-- human annotations
    CollectedData_<experimenter>.h5    # <-- human annotations
    frame_000606.png                   # <-- regenerable, not in git
    frame_002093.png
    ...
```

## Step 3: Upload labels to S3 and retrain

```bash
# Validate and upload labels
uv run python scripts/upload_dlc_labels.py --dry-run   # check first
uv run python scripts/upload_dlc_labels.py              # upload

# Launch GPU retraining on EC2 (fine-tunes SuperAnimal, re-runs all 26 sessions)
uv run python scripts/launch_dlc_retrain_ec2.py
```

## Version control strategy

### What is tracked in git (small, irreplaceable)

| File | Why |
|---|---|
| `metadata/retrain_frames/*.json` | Frame indices + session + video name |
| `sourcedata/trackers/dlc/hm2p-retrain-*/config.yaml` | DLC project config, bodypart definitions |
| `sourcedata/trackers/dlc/hm2p-retrain-*/labeled-data/*/CollectedData_*.csv` | Human annotations (x,y coords per bodypart per frame) |
| `sourcedata/trackers/dlc/hm2p-retrain-*/labeled-data/*/CollectedData_*.h5` | Human annotations (binary format) |

### What is NOT in git (large, regenerable)

| File | How to regenerate |
|---|---|
| `retrain_frames/*.png` | `prepare_retrain_frames.py` re-extracts from S3 video + frame indices |
| `labeled-data/*/*.png` | Same PNGs, copied from `retrain_frames/` |

### Recovery procedure

If PNGs are lost (e.g. after a fresh clone), regenerate them:

```bash
# For each session in metadata/retrain_frames/:
uv run python scripts/prepare_retrain_frames.py sub-XXXXX/ses-YYYYMMDDT... \
    --frames-file metadata/retrain_frames/<session>.json
```

This downloads the video from S3, extracts the frames, and copies them into the
DLC project's `labeled-data/` directory. The `CollectedData_*.csv` labels (from
git) reference these frames by filename, so everything reconnects automatically.

**Verified:** Re-extracted PNGs are bit-for-bit identical to the originals
(same md5 hash). The frame extraction from mp4 via OpenCV is deterministic.

## Pipeline dependency

After retraining, DLC inference re-runs on all 26 sessions. This invalidates
all downstream stages:

```
DLC Training (Stage 2a) --> DLC Inference (Stage 2b) --> Kinematics (Stage 3) --> MoSeq (Stage 3b) --> Sync (Stage 5) --> Analysis (Stage 6)
```

Stage 4 (Calcium) is independent and does not need re-running.
