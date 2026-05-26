# Plan: Session-level holdout with pose-cluster balancing

## Problem

DLC's `create_training_dataset` splits train/test randomly at the frame
level. With ~800 frames from 26 video sessions, this causes:

1. **Data leakage**: frames from the same session share background,
   lighting, and temporal correlation. Near-identical frames from the
   same video end up in both train and test, allowing the model to
   memorise rather than generalise (Glazner et al. 2025).
2. **Poor coverage**: random split may miss important pose types in the
   test set.
3. **Incomparable metrics**: each run gets a different random split from
   a different frame pool, making model comparison unreliable.

Frame-level approaches (k-means within-cluster split, greedy distance
maximisation) cannot fix this because the dominant leakage boundary is
the **session** — temporal context shared between frames.

## Approach: Hold out entire sessions

Hold out **3 primary non-excluded sessions** (~90 frames, ~11%) as the
test set. Select sessions whose combined pose-cluster distribution best
matches the overall dataset. No frame in the test set shares temporal
context with any training frame — zero leakage.

This simulates real deployment: applying the trained model to a session
it has never seen.

## Location

New function `_create_stratified_split()` in `scripts/run_dlc_retrain.py`,
called after `deeplabcut.create_training_dataset()` to overwrite DLC's
random split.

## Steps

### Step 1: Compute pose features

1. After `create_training_dataset()` runs, locate the
   `Documentation_data-*.pickle` file it produced.
2. Load all CollectedData H5 files across all sessions, extract (x, y)
   for all bodyparts. Pool into one array (N_total, B*2).
3. Handle NaNs: fill with per-bodypart mean.
4. Use raw coordinates (position + orientation + pose all matter for
   training diversity — translation is something the model must
   generalise to).

### Step 2: Pose-space clustering

5. k-means on raw coordinate vectors with k = 10-15 clusters. This
   groups frames into pose archetypes (walking, turning, grooming,
   near walls, etc.).
6. Assign each frame a cluster label.

### Step 3: Compute per-session cluster distributions

7. For each session, compute its cluster distribution: a vector of
   proportions showing which pose types are represented.
8. Compute the overall dataset cluster distribution (the target).

### Step 4: Select test sessions

9. From the **primary non-excluded sessions** (currently 11 sessions),
   enumerate all combinations of 3 sessions.
10. For each combination, compute the aggregate cluster distribution of
    those 3 sessions.
11. Select the combination that minimises KL divergence (or chi-squared
    distance) from the overall dataset cluster distribution. This
    ensures the test set covers all pose types proportionally.
12. All frames from the 3 selected sessions become the test set.
    All frames from the remaining 23 sessions become the train set.

### Step 5: Verify coverage

13. Check that the test set covers all K pose clusters. If any cluster
    has zero representation, try the next-best 3-session combination.
14. Report per-cluster counts (train / test).

### Step 6: Overwrite DLC split

15. Overwrite the `Documentation_data-*.pickle` file with the new
    train/test index arrays.
16. Regenerate the `.mat` training matrix file that DLC reads during
    `train_network`.

## Why primary non-excluded sessions for test?

- **Primary sessions** are the ones used for analysis. Test metrics on
  primary sessions directly predict downstream analysis quality.
- **Excluded sessions** have known issues (bad behaviour, fluctuating
  traces, sync problems). Using them as test frames would test the model
  on edge cases rather than typical deployment conditions.
- **Secondary sessions** are acceptable too, but primary sessions are
  preferred because they represent the target deployment distribution.
- With 11 primary sessions, holding out 3 leaves 8 primary + 9 secondary
  + 6 excluded = 23 sessions for training (~700 frames).

## Session candidates (primary, non-excluded)

| exp_id | animal | notes |
|--------|--------|-------|
| 20210823_16_59_50_1114353 | 1114353 | 43 frames |
| 20211028_11_25_50_1115465 | 1115465 | |
| 20211203_15_10_27_1115464 | 1115464 | |
| 20220408_15_01_57_1116663 | 1116663 | |
| 20220608_15_27_32_1117217 | 1117217 | |
| 20220608_16_22_06_1116994 | 1116994 | |
| 20220802_15_06_53_1117646 | 1117646 | |
| 20221003_14_36_54_1118020 | 1118020 | |
| 20221018_10_56_17_1117788 | 1117788 | |
| 20221115_13_27_42_1118213 | 1118213 | |
| 20221116_14_31_12_1118320 | 1118320 | |

Selecting 3 from 11 gives C(11,3) = 165 candidate combinations to
evaluate — trivial to compute.

## Fallback

If no 3-session combination from primary sessions covers all pose
clusters, relax to include secondary (non-excluded) sessions in the
candidate pool. This adds 9 more sessions (20 total candidates,
C(20,3) = 1140 combinations).

If still insufficient, use a hybrid: hold out 2 full sessions + top up
with DUPLEX-selected frames from other sessions, ensuring no two top-up
frames share a session.

## Diagnostics output

After splitting, report:
- Which 3 sessions were selected for test, and why
- Number of test frames, train frames
- Per-cluster counts (train / test)
- KL divergence between test and overall cluster distribution
- Per-session membership (train vs test)
- Histogram of min(pose_distance) for each test frame to its nearest
  train frame — confirms no leakage

## Key details

- Split is deterministic (fixed random_state=42 on k-means).
- k (pose clusters) configurable via `--split-clusters` (default 12).
- n_test_sessions configurable via `--n-test-sessions` (default 3).
- Falls back to DLC's random split if clustering or session selection
  fails.
- `TrainingFraction` from config.yaml is respected (~90% of total data
  in train, achieved by holding out 3 of 26 sessions ≈ 11.5% test).

## What does NOT change

- `create_training_dataset` still runs first (builds DLC project
  structure, video links, etc.).
- Only the split indices are overwritten, not the dataset structure.
- All 26 sessions contribute labeled data — excluded/secondary sessions
  are still trained on, just not used for test.

## Files affected

- `scripts/run_dlc_retrain.py` — add `_create_stratified_split()`, call
  after `create_training_dataset` in both SA and ImageNet paths.

## Risks

- DLC's `.mat` file format must match exactly or `train_network` crashes.
  Need to verify format by reading an existing one.
- If the 3 test sessions happen to contain a disproportionate number of
  frames (e.g. session with 43 frames), the train/test ratio shifts.
  With ~30 frames per session, 3 sessions ≈ 90 frames ≈ 11% — acceptable.
- The test set evaluates generalisation to new sessions, not new pose
  types within seen sessions. This is a stronger test than frame-level
  holdout, so test metrics may be lower than with random splitting.

## References

- Kennard & Stone 1969. "Computer aided design of experiments."
  Technometrics 11(1):137-148. (maximin distance — for training sets)
- Snee 1977. "Validation of regression models." Technometrics
  19(4):415-428. (DUPLEX algorithm)
- Glazner et al. 2025. "Find the Leak, Fix the Split." arXiv:2511.13944.
  (cluster-based splitting for video-derived data)
- Ye et al. 2024. "SuperAnimal pretrained pose estimation models."
  Nature Communications 15:5165. (leave-one-dataset-out evaluation)
- Mathis et al. 2021. "Pretraining boosts out-of-domain robustness for
  pose estimation." WACV. (Horse-10, identity-based splits)
- Yu et al. 2021. "AP-10K: A Benchmark for Animal Pose Estimation in
  the Wild." NeurIPS. (per-species random splits)

## Status

Implemented. `_create_stratified_split()` and `_clip_dir_to_exp_id()` added to
`scripts/run_dlc_retrain.py`. Called after `create_training_dataset()` in both
the SA-finetune and ImageNet HRNet paths. CLI args `--split-clusters` (default
12) and `--n-test-sessions` (default 3) control the split parameters.

Tests: `tests/scripts/test_stratified_split.py` (24 tests).
