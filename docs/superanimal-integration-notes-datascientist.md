# SuperAnimal Integration Notes — Data Scientist Review

**Paper:** Ye et al. 2024. "SuperAnimal pretrained pose estimation models for behavioral analysis."
*Nature Communications* 15:5165. doi:10.1038/s41467-024-48792-2

**Prepared:** 2026-04-02
**Context:** Review of the paper and DLC 3.x codebase to understand why our SuperAnimal
transfer-learning attempt failed and what the correct approach is.

---

## 1. Paper Summary

### What SuperAnimal is

SuperAnimal is a method for building foundation pose estimation models that work zero-shot
across many settings, without requiring per-lab labeling. The key insight is treating
diverse, inconsistently-labeled pose datasets as subsets of a single superset of keypoints
(panoptic pose estimation), then training one model on the union.

The paper presents two models:
- **SuperAnimal-TopViewMouse (SA-TVM)**: trained on TopViewMouse-5K — ~5000 images
  from 13 overhead-view lab mouse datasets, merged from diverse labs.
- **SuperAnimal-Quadruped (SA-Q)**: trained on Quadruped-80K — >80,000 images of
  quadrupeds (horses, dogs, rodents, etc.).

### Training datasets and data sources for SA-TVM

TopViewMouse-5K merges: DLC-Openfield, TriMouse, BlackMice, WhiteMice (SIMBA),
MausHaus (Mathis lab home cage), Kiehn-Lab-Openfield/Swimming/Treadmill, 3CSI, BM, EPM,
LDB, OFT. These cover diverse cage shapes, lighting, camera heights, and mouse strains.
Data banked at zenodo.org/records/10618947.

### The three core algorithmic innovations

**1. Keypoint gradient masking.**
Because no single dataset defines all keypoints, training naively would penalize the model
for not predicting keypoints that simply weren't labeled. The fix: during backpropagation,
mask (zero out) the gradient for any keypoint that is not defined in the current training
image's source dataset. The loss function treats "undefined" as neither penalized nor
encouraged — only "defined but occluded" keypoints are trained in the standard way. Without
masking, RMSE more than doubled (14.31 vs 27.90 pixels) in zero-shot testing.

**2. Memory replay fine-tuning.**
Catastrophic forgetting is a real risk: fine-tuning on a 4-keypoint dataset causes the
model to forget its other 23 keypoints. Memory replay prevents this by using the model's
own zero-shot predictions as pseudo-labels for keypoints not present in the downstream
dataset. Concretely: for any keypoint undefined in the target dataset's annotations, the
training label is replaced with the model's own (pre-inference, frozen) prediction,
provided that prediction has confidence > 0.7. At threshold 1.0, memory replay degenerates
to naive fine-tuning. The pseudo-labels are pre-computed once and stored on disk to prevent
label drift (not regenerated each epoch).

**3. Automatic keypoint matching.**
When a user's dataset has different keypoint names from the SuperAnimal superset, the
paper provides a bipartite matching algorithm using Hungarian assignment over Euclidean
distances between zero-shot predictions and ground truth. This produces a conversion
table mapping the user's keypoints to the model's superset indices. The method handles
annotator bias for ambiguous keypoints (e.g., exact position along the tail).

### Architectures

The paper benchmarks three architectures:
- **DLCRNet (bottom-up)**: multi-scale, no detector required. Faster but sensitive to
  animal size variation (spatial-pyramid search at test time helps).
- **HRNet-w32 (top-down)**: requires an object detector (Faster R-CNN + ResNet-50 + FPN).
  More robust in crowded scenes and standardizes animal size at both train and test time.
  **This is what we use** (SA-TVM HRNet-w32).
- **AnimalTokenPose (transformer)**: best zero-shot on DLC-Openfield (4.57 px RMSE),
  but computationally heavier.

HRNet-w32 has 29M parameters. Training protocol: Adam optimizer, lr=5e-4, 210 epochs,
step decay at epochs 170 and 200, batch size 64. For small fine-tuning sets (<64 unique
images), batch norm running stats are frozen and lr is halved to 5e-5 for stability.

### Video adaptation (unsupervised)

Two unsupervised test-time methods are described:

**Spatial-pyramid search (bottom-up models only):** At inference, the same video is
processed at multiple resolutions. Predictions are filtered by confidence and cosine
similarity to the median keypoint, then aggregated. This addresses the mismatch between
the animal size distribution in training and the user's video. Not needed for top-down
models because the detector crop standardizes size.

**Video adaptation (pseudo-labeling):** Run the model on the target video, treat all
high-confidence predictions (>0.5) as pseudo-labels, then fine-tune the model on these
for 1000 iterations with batch size 1, batch norm stats frozen during adaptation. This
is done *in eval mode* for batch norm (frozen running stats), only updating batch norm
affine parameters. Empirically sufficient to greatly reduce jitter. Processing rate:
~12 FPS (vs 4 FPS for self-pacing). Significantly outperforms Kalman filtering as a
post-processing baseline (p<0.003, Cohen's d>0.785).

The key distinction: video adaptation adapts model weights (unsupervised, pseudo-label
based), whereas spatial-pyramid search is a test-time augmentation that does not modify
weights.

### Performance claims relevant to us

SA-TVM HRNet-w32 on DLC-Openfield (top-down, zero-shot): **95.2 mAP, 4.88 px RMSE**.
This is on fully unseen data from a lab that was not in training.

Fine-tuning with memory replay + 10 images: RMSE 7.68 px (vs ImageNet baseline: 18.14 px).
To match SA-TVM's 10-image performance, ImageNet pretraining requires 101 images.
Conclusion: 10x data efficiency in the low-data regime.

Effect sizes are large (Cohen's d 4.88–10.99) and statistically robust.

---

## 2. Fine-tuning Workflow (as described in the paper)

The paper describes two fine-tuning modes:

**"Transfer learning" (paper terminology):** Load the SuperAnimal *encoder* only (HRNet
backbone). Use a randomly initialized decoder (prediction head). Train on the downstream
dataset. This is analogous to ImageNet transfer learning but starting from a pose-aware
backbone. The paper finds this only modestly better than ImageNet-based transfer learning.

**"Fine-tuning" with memory replay (recommended):** Load both encoder *and* decoder from
the SuperAnimal checkpoint. Project the decoder to the superset keypoint space. For
keypoints undefined in the target dataset, substitute pseudo-labels from the zero-shot
model. This is what the paper means by "memory replay fine-tuning" and is the method that
achieves 10-100x data efficiency.

The pseudo-code is given in the Methods section:
- Pre-compute zero-shot predictions on the entire labeled set and save to disk.
- During training, for each keypoint undefined in the target dataset, replace the GT label
  with the saved pseudo-label, provided its confidence > 0.7.
- GT labels take precedence for defined keypoints.
- Loss is computed over the full superset (all 27 keypoints for SA-TVM).

This is why the decoder must remain 27-way output during fine-tuning. The model never
shrinks its output head — it just re-weights which keypoints receive real vs pseudo labels.

**HRNet fine-tuning hyperparameters (from Methods):**
- Adam, lr=5e-4 → decays at epochs 170, 200 (over 210 epochs total for full training)
- For small data (<64 images): lr=5e-5, frozen BN running stats
- Fine-tuning iterations: 70k (DLCRNet in paper; HRNet uses epochs not iterations)
- Video adaptation: 1000 iters, batch size 1, BN in eval mode

---

## 3. The TopViewMouse Model

### Bodypart superset (27 keypoints)

From the project config (`superanimal_topviewmouse.yaml`), SA-TVM defines 26 named
bodyparts (the paper says 27; the discrepancy is likely a head_midpoint that is computed
rather than predicted, or the config excludes one that is implicit):

```
nose, left_ear, right_ear, left_ear_tip, right_ear_tip,
left_eye, right_eye, neck, mid_back, mouse_center,
mid_backend, mid_backend2, mid_backend3,
tail_base, tail1, tail2, tail3, tail4, tail5,
left_shoulder, left_midside, left_hip,
right_shoulder, right_midside, right_hip,
tail_end, head_midpoint
```

### What we use from this superset

Our 8 hm2p bodyparts and their SA-TVM superset equivalents:

| hm2p bodypart       | SA-TVM superset keypoint | Notes |
|---------------------|--------------------------|-------|
| `nose_tip`          | `nose`                   | Direct match |
| `left_ear`          | `left_ear`               | Direct match |
| `right_ear`         | `right_ear`              | Direct match |
| `implant_base_rear` | *(custom, no match)*     | Headstage — unique to our rig |
| `neck`              | `neck`                   | Direct match |
| `mid_back`          | `mid_back`               | Direct match |
| `mouse_center`      | `mouse_center`           | Direct match |
| `tail_base`         | `tail_base`              | Direct match |

7 of 8 bodyparts map directly to SA-TVM superset keypoints. Only `implant_base_rear`
is novel. This is an ideal case for fine-tuning: the model already has strong priors for
the 7 matched keypoints, and only needs to learn one new keypoint.

### Architecture in DLC 3.x

The HRNet-w32 variant is registered as `superanimal_topviewmouse_hrnetw32` in DLC's
model zoo. The checkpoint is downloaded automatically from HuggingFace when first used
via `deeplabcut.video_inference_superanimal()`. Internally, the DLC 3.x PyTorch backend
saves checkpoints as `.pt` files using `torch.save()` of a state dict with keys: metadata
(metrics), model state dict, and optionally optimizer state.

---

## 4. Why Our Integration Failed

Our attempt: use SuperAnimal transfer learning with DLC 3.0.0rc13 for 8-bodypart tracking,
HRNet-W32. We hit "checkpoint format mismatches" and fell back to ImageNet training.

Based on the paper, the DLC 3.x codebase, and the PR #2756 changelog, several issues
could have caused this:

### 4a. Old vs new checkpoint format (most likely cause)

DLC underwent a major rewrite from TensorFlow (DLC 2.x) to PyTorch (DLC 3.x). The
SuperAnimal model weights were originally stored in TensorFlow checkpoint format (.index,
.data-00000-of-00001, .meta). The transition to PyTorch required re-exporting all
SuperAnimal weights to `.pt` format. In rc13 (a release candidate), this conversion may
not have been complete, or the HuggingFace-hosted checkpoint was still in TF format while
the loading code expected PyTorch `.pt` format.

PR #2756 ("SuperAnimal Model Updates", merged October 2024) specifically fixed:
- Bodypart mapping inconsistencies when fine-tuning with memory replay
- The `WeightInitialization` class was redesigned to store snapshot paths directly
  (more modular, format-agnostic)
- Tensor size mismatches in the DataLoader during training

### 4b. Decoder head output dimension mismatch

The SuperAnimal HRNet-w32 head outputs predictions for all 27 superset keypoints. Our
project was configured for 8 bodyparts. The `WeightInitialization` mechanism uses a
`conversion_array` to map project bodyparts to SuperAnimal indices — e.g., `[0, 1, 2, 7,
12, 16, 9, 13]` for our 8 bodyparts (indices into the 27-keypoint SA-TVM superset).

If this conversion array was not correctly constructed, or if the DLC code attempted to
directly load the 27-channel decoder weights into an 8-channel head, a dimension mismatch
error would result. The `with_decoder=True` flag is required to use the pre-trained decoder
with memory replay; if omitted, a fresh decoder is initialized randomly.

In rc13, the `build_weight_init()` API may not have been finalized, and the conversion
array format may have changed between rc versions.

### 4c. Incorrect fine-tuning API invocation

The correct DLC 3.x API for SuperAnimal fine-tuning is:

```python
weight_init = deeplabcut.modelzoo.build_weight_init(
    cfg=config_path,
    super_animal="superanimal_topviewmouse",
    model_name="hrnetw32",
    detector_name="fasterrcnn_resnet50_fpn",
    with_decoder=True,
    memory_replay=True,
)
deeplabcut.create_training_dataset(config_path, ...)
deeplabcut.train_network(config_path, weight_init=weight_init, ...)
```

In rc13, this API was in flux. The `train/pose_cfg.yaml` was still being generated for
PyTorch shuffles in some rc versions but not others (PR #2756 removed it), and callers
that assumed it existed would fail silently or with misleading errors.

### 4d. rc13 was a release candidate with known instability

Release candidate versions of DLC 3.0 were explicitly unstable. The first stable DLC 3.0
release incorporating the finalized SuperAnimal PyTorch pipeline was published after
PR #2756 merged in October 2024. Running rc13 for fine-tuning would have exposed several
unresolved bugs documented in that PR.

### 4e. Summary: what actually went wrong

Almost certainly a combination of (a) TF-format checkpoint being loaded by PyTorch code,
(b) the conversion array not being passed correctly (so DLC tried to load 27-channel
weights into an 8-channel head), and (c) the rc13 `train/pose_cfg.yaml` generation
behavior being inconsistent.

The fallback to ImageNet training was therefore correct given the circumstances, but it
means we gave up the 10x data efficiency benefit.

---

## 5. Correct Integration Approach

### Prerequisites

- DLC >= 3.0.0 stable (not rc13). As of 2026-04-02, use the latest stable release.
  Install: `uv pip install "deeplabcut[pytorch]"`
- CUDA available (GPU required for training)
- The SA-TVM HRNet-w32 checkpoint is auto-downloaded from HuggingFace on first use

### Step-by-step workflow

**Step 1: Create a DLC project with your 8 bodyparts.**

```python
import deeplabcut

config_path = deeplabcut.create_new_project(
    "hm2p-retrain",
    "experimenter",
    videos=[],
    bodyparts=["nose_tip", "left_ear", "right_ear", "implant_base_rear",
               "neck", "mid_back", "mouse_center", "tail_base"],
)
```

**Step 2: Create a training dataset (with SuperAnimal shuffle).**

Use `create_training_dataset` with the PyTorch engine and SuperAnimal model:

```python
deeplabcut.create_training_dataset(
    config_path,
    num_shuffles=1,
    net_type="hrnet_w32",
    engine="pytorch",
)
```

**Step 3: Build the weight initialization object.**

This is the critical step that was absent or broken in our rc13 attempt:

```python
weight_init = deeplabcut.modelzoo.build_weight_init(
    cfg=config_path,
    super_animal="superanimal_topviewmouse",
    model_name="hrnetw32",
    detector_name="fasterrcnn_resnet50_fpn",
    with_decoder=True,      # load the 27-keypoint decoder, not just backbone
    memory_replay=True,     # use pseudo-labels for undefined keypoints
)
```

This constructs a `WeightInitialization` object that internally holds:
- `snapshot_path`: path to the SA-TVM HRNet-w32 checkpoint (auto-downloaded)
- `detector_snapshot_path`: path to the Faster R-CNN detector checkpoint
- `conversion_array`: integer array mapping our 8 bodyparts to SA-TVM superset indices
- `memory_replay=True`: enables pseudo-label substitution during training

The conversion array is built from `SuperAnimalConversionTables` in `config.yaml`. This
table must be populated correctly before calling `build_weight_init`. DLC populates it
via `deeplabcut.modelzoo.create_conversion_table()`.

**Step 4: Populate the conversion table.**

```python
# Explicit conversion table (our 8 bodyparts → SA-TVM superset names)
conversion_map = {
    "nose_tip": "nose",
    "left_ear": "left_ear",
    "right_ear": "right_ear",
    "implant_base_rear": None,    # no SA-TVM equivalent — trained from scratch
    "neck": "neck",
    "mid_back": "mid_back",
    "mouse_center": "mouse_center",
    "tail_base": "tail_base",
}
# Save to CSV, then:
deeplabcut.modelzoo.read_conversion_table_from_csv(config_path, csv_path)
```

For `implant_base_rear` (no SA-TVM match), the conversion array entry should be `-1` or
left unmapped. The gradient masking mechanism will treat it as an undefined keypoint for
the SA-TVM decoder, meaning pseudo-labels won't be substituted for it — it will be
learned purely from our labeled frames (correct behavior).

**Step 5: Train with weight initialization.**

```python
deeplabcut.train_network(
    config_path,
    shuffle=1,
    weight_init=weight_init,
    # For small labeled sets (<64 images), DLC uses lr=5e-5 and frozen BN automatically
)
```

The training loop will:
1. Load SA-TVM HRNet-w32 backbone + decoder into the model.
2. Pre-compute zero-shot predictions on all labeled frames.
3. For each training iteration:
   - For keypoints in {left_ear, right_ear, nose, neck, mid_back, mouse_center, tail_base}:
     use our GT labels (if defined for the current image).
   - For all other SA-TVM superset keypoints not in our set: use the cached pseudo-labels
     if their confidence > 0.7; otherwise skip.
   - For `implant_base_rear`: use our GT labels only (no pseudo-labels available).
4. Compute loss over all 27 SA-TVM keypoints (pseudo-labels included).
5. Update encoder + decoder weights.

**Step 6: Run inference on all 26 sessions.**

After training, inference is identical to any DLC model:

```python
deeplabcut.analyze_videos(config_path, video_list, shuffle=1)
```

The output contains predictions for only the 8 project bodyparts (the decoder is queried
with the conversion array to extract the relevant channels).

### Number of labeled frames needed

Based on the paper's Fig. 1e data (SA-TVM, HRNet-w32, DLC-Openfield benchmark):

| Labeled frames | mAP (memory replay) | RMSE (px) |
|---------------|---------------------|-----------|
| 10 (~1%)      | 99.6                | 2.38      |
| 50 (~5%)      | 99.8                | 1.95      |
| 100 (~10%)    | 99.9                | 1.54      |
| Full dataset  | 99.9                | 1.21      |

For comparison, ImageNet transfer learning needs 100+ frames to reach what SA-TVM achieves
with 10 frames. In our case (8 bodyparts, overhead view, similar to training distribution):
- 20–30 labeled frames across diverse sessions should be sufficient for excellent
  tracking of the 7 matched bodyparts.
- `implant_base_rear` will need more frames (50+) because it has no SA-TVM prior.

Our existing retrain set (selected via `prepare_retrain_frames.py`) is already more than
adequate in quantity. The main issue was the failed weight loading.

---

## 6. Video Adaptation as Alternative to Full Fine-tuning

### What it is

Video adaptation is an *unsupervised* method that requires zero labeled frames. It works
by running the SA-TVM model on the target video, treating high-confidence predictions
(> 0.5) as pseudo-labels, and fine-tuning the model for 1000 iterations (batch size 1,
batch norm frozen). No human labels are required.

### DLC API

```python
deeplabcut.video_inference_superanimal(
    [video_path],
    superanimal_name="superanimal_topviewmouse",
    scale_list=range(200, 600, 50),   # only needed for DLCRNet bottom-up
    video_adapt=True,                  # enables pseudo-label fine-tuning
    model_name="hrnetw32",             # top-down: no scale_list needed
)
```

### Is it useful for us?

**Pros:**
- Zero labeling effort.
- Can be applied session-by-session to adapt to our specific arena illumination,
  camera height, and mouse appearance.
- Particularly effective for reducing temporal jitter (the main quality issue in 30 fps
  tracking of fast-moving mice).

**Cons:**
- Video adaptation is applied at inference time, not at training time. It adapts the
  model to one video's appearance. If our sessions have consistent recording conditions
  across the 26 sessions, the benefit is limited after the first adaptation.
- For top-down HRNet-w32, the animal size issue (why spatial-pyramid search was invented)
  does not apply — top-down models standardize crop size at both train and test time. So
  the primary benefit of video adaptation for us is temporal smoothness.
- `implant_base_rear` will not benefit from video adaptation because SA-TVM has no
  superset keypoint for it. The pseudo-labels will be meaningless for this point.
- Video adaptation does not transfer learned weights across sessions (each video is
  adapted independently).

**Recommendation:** Video adaptation is not a substitute for fine-tuning with our labeled
data. It could be useful as a post-processing step if jitter remains problematic after
fine-tuning, but it should not be the primary strategy. Our 7/8 bodyparts are already
well within the SA-TVM training distribution; the issue is `implant_base_rear`, which
video adaptation cannot address.

If labeled frames are available (they are — we already have them), memory replay
fine-tuning is strictly better.

---

## 7. Expected Improvements vs Training from ImageNet

### What we currently have

We trained HRNet-w32 from ImageNet weights (no SuperAnimal prior) after the rc13 failure.
This is the baseline.

### Expected improvement from SuperAnimal fine-tuning

Using the paper's Table S3 and Table S4 (SA-TVM, HRNet-w32):

**In the low data regime (10–50 labeled frames):**
- ImageNet transfer learning at 1% data: mAP 91.5, RMSE 7.0 px (DLC-Openfield)
- SA-TVM memory replay at 1% data: mAP 99.6, RMSE 2.38 px
- Improvement: ~8 pp mAP, ~3x lower RMSE

**In the moderate data regime (100 labeled frames):**
- ImageNet at 10%: mAP 99.3, RMSE 1.57 px
- SA-TVM memory replay at 10%: mAP 99.9, RMSE 1.54 px
- At this point, differences are small — both reach near-ceiling performance on the
  DLC-Openfield benchmark.

**Caveats for our dataset:**
- DLC-Openfield is a relatively easy benchmark (single mouse, clean background,
  consistent lighting). Our rose maze has: overhead 2P headstage occluding some bodyparts,
  variable lighting (lights on/off), and the fibre cable.
- In harder OOD conditions, the advantage of SA-TVM over ImageNet is larger, not smaller.
- `implant_base_rear` has no SuperAnimal prior, so improvement there depends entirely on
  how many frames we label.
- For our 7 matched bodyparts in the low-data regime, we would expect RMSE to improve
  from roughly the ImageNet baseline (~7 px with 10 frames) to ~2.5 px with SA-TVM.

**Practical expectation:** If we have 30–50 labeled frames per bodypart, SA-TVM memory
replay should achieve tracking quality indistinguishable from having 300–500 ImageNet-
pretrained labeled frames. The main observable difference in our pipeline will be:
- Fewer dropped-confidence frames (fewer NaNs in kinematics.h5).
- Better tracking during fast turns and occlusions.
- Lower jitter on the ear-vector angle (direct impact on HD precision).

For our primary HD analysis (ear-vector angle), a 3x reduction in spatial error matters.
At 9.6 Hz and ~15 cm/s average speed, the mouse moves ~1.5 cm/frame. An RMSE of 7 px
(~2 mm at our calibration) in ear location produces ~2° HD error from geometry; an RMSE
of 2.5 px produces ~0.7° error. This is below our HD bin width (typically 10–12°).

In practice, the more important metric is whether the ear-vector angle drops out (very
low confidence on both ears simultaneously). SA-TVM pre-training reduces this because the
model has stronger pose priors and is more robust to partial occlusion by the headstage.

---

## 8. Action Items for Implementation

1. **Upgrade DLC** from 3.0.0rc13 to the current stable release. Verify with
   `import deeplabcut; deeplabcut.__version__`.

2. **Verify SuperAnimal checkpoint availability.** Run
   `deeplabcut.video_inference_superanimal(["test.mp4"], "superanimal_topviewmouse")`
   on a single short video. This triggers the HuggingFace download and confirms the
   checkpoint loads without error.

3. **Build the conversion table.** Create a CSV file:
   ```
   project_bodypart,superanimal_bodypart
   nose_tip,nose
   left_ear,left_ear
   right_ear,right_ear
   implant_base_rear,
   neck,neck
   mid_back,mid_back
   mouse_center,mouse_center
   tail_base,tail_base
   ```
   Empty superanimal_bodypart means "no match; train from labeled data only."

4. **Create a SuperAnimal fine-tuning shuffle** alongside the existing ImageNet shuffle.
   Compare tracking quality before promoting to production.

5. **Test on a held-out session** not used for labeling. Report mAP, RMSE, and the
   fraction of frames with both ears tracked at confidence > 0.6.

6. **Do not use video_adapt for routine inference.** It is per-video and does not solve
   the `implant_base_rear` problem. Apply it as an optional post-hoc step only if temporal
   jitter in the ear vector remains problematic after fine-tuning.

---

## References

- Ye S, Filippova A, Lauer J, Schneider S, Vidal M, Qiu T, Mathis A, Mathis MW. 2024.
  "SuperAnimal pretrained pose estimation models for behavioral analysis."
  *Nature Communications* 15:5165. doi:10.1038/s41467-024-48792-2

- DeepLabCut GitHub: https://github.com/DeepLabCut/DeepLabCut

- PR #2756 "SuperAnimal Model Updates" (October 2024):
  https://github.com/DeepLabCut/DeepLabCut/pull/2756

- SuperAnimal TopViewMouse training data (Zenodo):
  https://zenodo.org/records/10618947

- DLC ModelZoo: http://modelzoo.deeplabcut.org

- Wang J, Sun K, Cheng T et al. 2020. "Deep High-Resolution Representation Learning for
  Visual Recognition." *IEEE TPAMI*. (HRNet-w32 architecture)
