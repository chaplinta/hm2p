# Plan: Stratified train/test split for DLC training

## Problem

DLC's `create_training_dataset` splits train/test randomly per video folder.
With ~800 frames across 26 sessions, the random 90/10 split can put all rare
poses into train (or test), making the test set unrepresentative. Two sessions
with the mouse in the same position contribute redundant test frames. Worse,
"same pose, same context" pairs across train/test constitute data leakage.

## Approach

Two-layer similarity to decide the split:

1. **Pose-space similarity** — the primary criterion. Normalise keypoint
   labels (centre on root joint, scale by torso length), then cluster in
   normalised pose space. This answers: "does my test set contain poses
   the model has effectively already seen?"

2. **Image-space similarity** — secondary, to distinguish "same pose,
   different visual context" (acceptable) from "same pose, same context"
   (leakage). Use the PCA cache thumbnails for this.

The split ensures each pose cluster contributes to both train and test,
AND that no test frame is too similar in both pose AND image space to
any train frame.

## Location

New function `_create_stratified_split()` in `scripts/run_dlc_retrain.py`,
called after `deeplabcut.create_training_dataset()` to overwrite DLC's
random split.

## Steps

### Step 1: Build normalised pose vectors

1. After `create_training_dataset()` runs, locate the
   `Documentation_data-*.pickle` file it produced.
2. Load all CollectedData H5 files across all sessions, extract (x, y)
   for all bodyparts. Pool into one array (N_total, B, 2).
3. Handle NaNs: for frames with > 50% NaN bodyparts, mark as "sparse"
   (these go to train — they are unusual and valuable but poor test
   candidates). For remaining NaNs, fill with per-bodypart mean.
4. Normalise each frame's keypoints:
   - Centre on root joint (mouse_center or centroid of all bodyparts)
   - Scale by torso length (nose_tip to tail_base distance) so poses
     are comparable across different body sizes and camera zoom levels
   - This removes position and scale, keeping pose + orientation

### Step 2: Pose-space clustering

5. k-means on the normalised pose vectors with k = ~50 clusters.
   This groups frames by body configuration (walking, turning, grooming,
   rearing, etc.) regardless of maze position.
6. Each cluster represents a "pose type". The test set must sample from
   each pose type proportionally.

### Step 3: Stratified split with leakage check

7. For each cluster, assign 90% to train, 10% to test. Each cluster
   with >= 2 frames gets at least 1 test frame.
8. Leakage check: for each candidate test frame, compute:
   - Pose similarity to nearest train frame (Euclidean distance on
     normalised coords, or OKS)
   - Image similarity to nearest train frame (PCA-projected thumbnail
     distance, using the PCA cache)
   - If BOTH pose AND image similarity are above thresholds (same pose
     + same visual context), swap this test frame to train and pick a
     different frame from the same cluster for test.
9. OKS (Object Keypoint Similarity) can be computed as:
   `OKS = mean(exp(-d_i^2 / (2 * s^2 * k_i^2)))` where d_i is
   per-keypoint distance, s is object scale, k_i is per-keypoint
   constant. OKS > 0.9 between a test and train frame = leakage.

### Step 4: Overwrite DLC split

10. Overwrite the pickle file with the new train/test index arrays.
11. Regenerate the `.mat` training matrix file that DLC reads during
    `train_network`.

## Key details

- Normalisation: centre on root joint + scale by torso length. NOT raw
  coordinates (which conflate position with pose) and NOT Procrustes
  (which removes orientation — orientation matters for training).
- Clustering is across all sessions — avoids redundant test frames from
  different sessions where the mouse is in the same spot doing the same
  thing.
- head_midpoint excluded from pose normalisation (rigidly attached to
  skull, not informative for pose) but included in image similarity.
- Split is deterministic (fixed random_state=42 on k-means).
- k is configurable via `--split-clusters` CLI arg (default 50).
- Falls back to DLC's random split if clustering fails.

## Leakage detection output

After splitting, report:
- Number of test frames, train frames
- Per-cluster counts (train / test)
- Number of potential leakage pairs caught and swapped
- Histogram of min(pose_distance) for each test frame to its nearest
  train frame — should show no test frames with very small distances
- Histogram of min(image_distance) for same — cross-referenced

## What does NOT change

- `create_training_dataset` still runs first (builds DLC project
  structure, video links, etc.).
- Only the split indices are overwritten, not the dataset structure.
- `TrainingFraction` from config.yaml is respected (applied per-cluster
  instead of globally).

## Files affected

- `scripts/run_dlc_retrain.py` — add `_create_stratified_split()`, call
  after `create_training_dataset` in both SA and ImageNet paths.

## Risks

- DLC's `.mat` file format must match exactly or `train_network` crashes.
  Need to verify format by reading an existing one first.
- Clusters with 1 frame: goes to train (can't split).
- OKS constants (k_i per keypoint) need sensible defaults for mouse
  bodyparts — COCO values are for humans. Use uniform k_i initially.
- PCA cache may not exist for all sessions — fall back to pose-only
  leakage check if image similarity unavailable.

## Status

Plan only. Not yet implemented.
