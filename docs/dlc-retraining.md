# DLC Retraining Workflow

Guide for manually labelling frames and retraining the DLC pose model.

## Overview

The hm2p project uses DeepLabCut's SuperAnimal TopViewMouse model for pose
estimation. When tracking quality is poor for certain sessions, the model can
be fine-tuned on manually labelled frames. This document covers the full
workflow from frame selection to retraining.

## Bodyparts

8 bodyparts are labelled and tracked:

| Bodypart | SuperAnimal | Purpose |
|---|---|---|
| `nose_tip` | `nose` | HD estimate (nose→neck midline), exploration |
| `left_ear` | `left_ear` | Primary HD (ear vector perpendicular) |
| `right_ear` | `right_ear` | Primary HD (ear vector perpendicular) |
| `head_midpoint` | *(custom)* | Skull reference (rigid, high-contrast 2P headstage) |
| `neck` | `neck` | HD fallback (nose→neck axis), head-body dissociation |
| `mid_back` | `mid_back` | Dorsal midline just behind shoulders |
| `mouse_center` | `mouse_center` | Geometric centre of body, position/speed |
| `tail_base` | `tail_base` | Body orientation |

5 head keypoints (nose, ears, head_midpoint, neck) enable robust HD fusion
with confidence-weighted fallback when individual points are occluded.

**`head_midpoint` is SuperAnimal-TopViewMouse keypoint #26.** Earlier
versions of this doc said `head_midpoint` had "no SuperAnimal equivalent" —
that was incorrect. The SA-TVM model is trained on TopViewMouse-5K which
includes the MausHaus dataset (head-mounted-cable mice), so the SA model
has seen this exact keypoint. The `SuperAnimalConversionTables` block in
`config.yaml` carries the identity-to-identity mapping for all 8 project
bodyparts:

| Project bodypart | SA-TVM index | Notes |
|---|---|---|
| `nose_tip` | 0 | |
| `left_ear` | 1 | |
| `right_ear` | 2 | |
| `head_midpoint` | 26 | Maps to SA-TVM `head_midpoint` (MausHaus origin). |
| `neck` | 7 | |
| `mid_back` | 8 | |
| `mouse_center` | 9 | |
| `tail_base` | 13 | |

This identity-mapping enables the SuperAnimal memory-replay fine-tune path
(`scripts/run_dlc_retrain.py --sa-finetune`) to warm-start from SA-TVM
weights for every bodypart, including `head_midpoint`.

## Labelling conventions

Labels follow the **SuperAnimal TopViewMouse** convention for keypoint
placement. Additional guidelines specific to this dataset:

- **Occluded body parts are labelled** if the position can be inferred
  from the visible anatomy. For example, if the nose is hidden behind
  the headstage but the ear positions and head midpoint make the nose
  location unambiguous, label it. This teaches the model to predict
  through occlusion rather than producing NaN, which improves tracking
  continuity.
- **Do not label** a body part if its position is genuinely ambiguous —
  e.g. mouse curled into a ball with multiple body parts overlapping
  and indistinguishable. Leave the keypoint as NaN for that frame.
- **Left/right ear convention**: left ear is the mouse's anatomical
  left (your right when viewing from above with the nose pointing up).
  Follow the SuperAnimal convention — do not mirror.
- **head_midpoint**: place on the rear edge of the 2P headstage base,
  centred on the midline. This is the most reliably visible keypoint
  due to the high contrast of the headstage.
- **tail_base**: where the tail meets the body, not the tip of the tail.

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
uv run python scripts/launch_dlc_finetune_ec2.py
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

## SuperAnimal fine-tuning (memory-replay path)

`scripts/run_dlc_retrain.py --sa-finetune` runs an alternative training
path that warm-starts from the SuperAnimal-TopViewMouse HRNet-W32 release
weights via DeepLabCut's
`build_weight_init` + `create_training_dataset(weight_init=...)` API.
The legacy ImageNet HRNet path (`--sa-finetune` not set) remains the
default; SA fine-tuning is opt-in.

### Method

> Ye S, Filippova A, Lauer J, Schneider S, Vidal M, Qiu T, Mathis A,
> Mathis MW. 2024. "SuperAnimal pretrained pose estimation models for
> behavioral analysis." *Nature Communications* 15:5165.
> doi:[10.1038/s41467-024-48792-2](https://doi.org/10.1038/s41467-024-48792-2).
> Code: https://github.com/DeepLabCut/DeepLabCut.
> Weights: [SuperAnimal-TopViewMouse on HuggingFace](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse).

The memory-replay protocol uses the SA model's own zero-shot predictions
(confidence > 0.7) as pseudo-labels for the SA-TVM channels we did not
label. This prevents catastrophic forgetting of those channels and lets
the encoder + decoder keep learning from the full 27-channel signal
while we train only on the 8 channels we care about (Ye 2024 Methods
§"Memory replay fine tuning"; Fig. 1d).

### When to use it

Use `--sa-finetune` when expecting better OOD generalisation (small
high-curvature keypoints like nose, ambiguous boundary keypoints like
tail). The legacy ImageNet path stays competitive once enough labels
exist; the SA path's relative gain shrinks with data per Ye 2024
Tables S3-S4. Both shuffles can coexist on the same DLC project — the
SA path lives on a parallel shuffle, so the legacy snap-110 stays in
place until the operator promotes the SA model.

### Operating procedure

```bash
# Step 0 — sanity (run once on EC2):
uv run python -c "
import dlclibrary
print(dlclibrary.list_available_models())
print(dlclibrary.list_available_detectors())
"

# Step 1 — labels already on S3 (no change to existing flow).
uv run python scripts/upload_dlc_labels.py

# Step 2 — launch SA fine-tune. Defaults: 120 epochs, batch_size 8,
# lr 5e-5, frozen BN. EBS root volume bumps to 120 GB.
uv run python scripts/launch_dlc_finetune_ec2.py --sa-finetune

# Step 3 — after the EC2 run self-terminates and pose-finetuned/
# lands on S3, run the comparison locally:
uv run python scripts/compare_models.py \
    --baseline-id  dlc-20260430-hrnetw32-snap110 \
    --candidate-id <new id; from dlc-champion.json> \
    --labels-dir   sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/labeled-data/ \
    --output       verdict.json --upload-s3

# Exit code: 0 = passes the v2 §4.6 promotion gate, keep the new
# champion. 2 = gate failed; roll back via declare_dlc_champion.py
# with the previous champion's identifiers (the prior manifest is
# archived under dlc-champion-history/).
```

### Recommended manual QC step

Before declaring a passed verdict final, render a labelled video for one
held-out dark-condition session under each model (baseline + candidate)
via `scripts/render_dlc_videos.py` and compare side-by-side. Architect
notes (open question #6) recommend this; the existing `dlc_viewer_page.py`
shows the render. There is no code-level enforcement.

### Follow-ups (not implemented this rollout)

- **Auto-archive baseline predictions** before the SA-finetune run
  overwrites `pose/{sub}/{ses}/*.h5`. Currently operator-driven:
  ```bash
  aws s3 sync s3://hm2p-derivatives/pose/ \
              s3://hm2p-derivatives/pose-archive/<baseline_champion_id>/
  ```
  Architect open-question #4: candidate for a follow-up commit so
  `compare_models.py --mode predict` can resolve the baseline prefix
  without operator action.
- **Pre-render two labelled videos** before promotion: out of scope
  for this rollout (architect open-question #2).

### Pitfalls (from architect §6 and Ye 2024 §"Discussion")

1. **256x256 input mismatch.** The SA-TVM HRNet was trained at 256x256
   crops. `_train_sa_finetune` warns (does not abort) when the SA
   shuffle's `pytorch_config.yaml` `data.train.input_size` differs.
2. **Detector name fallback.** `dlclibrary.list_available_detectors()`
   may list either `fasterrcnn_resnet50_fpn_v2` (DLC ≥ 3.0 default) or
   `fasterrcnn_resnet50_fpn`. The SA path probes both and picks the v2
   variant first.
3. **SA detector multi-animal-trained.** The SA detector was trained on
   TopViewMouse-5K which includes TriMouse (3-mouse occluded scenes).
   Top-down inference returns nothing if the detector returns no bbox.
   `probe_sa_detector_bbox_rate` (in `hm2p.pose.finetune`) flags this
   as a soft pre-flight — fewer than 90% bbox-positive frames should
   prompt re-training just the detector before the full SA fine-tune.
4. **80/20 vs 95/5 split.** The paper's tables are computed on 95/5;
   this project uses 80/20. Compare *relative gains* (SA vs ImageNet
   on the same hm2p holdout), not absolute RMSE against paper tables.
5. **Memory-replay backward cost is ~3×.** Wall-clock is still cheaper
   than 400-epoch ImageNet because epochs reduce 400 → 120, but
   per-iteration VRAM cost is higher. `g4dn.xlarge` (16 GB VRAM) at
   `batch_size=8` is sufficient. EBS root bumps to 120 GB.
6. **Stale DLC example script.**
   `examples/testscript_superanimal_transfer_learning.py` in the DLC
   repo uses pre-3.0 `superanimal_name=` / `superanimal_transfer_learning=`
   kwargs that no longer exist on `train_network`. Do not copy from it.
7. **DLC issue #2742.** `video_inference_superanimal(..., video_adapt=True)`
   fails on folder paths. The SA path uses `deeplabcut.analyze_videos`
   on individual file paths instead — same as the legacy ImageNet path.
