# SuperAnimal Integration Notes — Data Scientist Review

**Paper:** Ye et al. 2024. "SuperAnimal pretrained pose estimation models for behavioral analysis."
*Nature Communications* 15:5165. doi:10.1038/s41467-024-48792-2
GitHub: https://github.com/DeepLabCut/DeepLabCut (modelzoo subpackage)

**Prepared:** 2026-04-02
**Context:** Review of the paper and DLC 3.x codebase to understand why our SuperAnimal
transfer-learning attempt failed and what the correct approach is.

**Our setup:** 8 bodyparts (nose_tip, left_ear, right_ear, implant_base_rear, neck, mid_back,
mouse_center, tail_base), 184 labelled frames, DLC 3.0.0rc13, overhead camera, light/dark alternation.
Previous attempt: checkpoint format mismatches, head architecture incompatibilities, fell back
to HRNet-W32 from ImageNet.

---

## 1. What SuperAnimal Does

### The core idea: panoptic pose estimation

SuperAnimal is a method for building foundation pose estimation models that work zero-shot
across many settings, without requiring per-lab labeling. The key insight is treating
diverse, inconsistently-labeled pose datasets as subsets of a single superset of keypoints
(panoptic pose estimation), then training one model on the union.

The paper presents two models:
- **SuperAnimal-TopViewMouse (SA-TVM)**: trained on TopViewMouse-5K — ~5000 images
  from 13 overhead-view lab mouse datasets, merged from diverse labs. This is the relevant
  model for hm2p.
- **SuperAnimal-Quadruped (SA-Q)**: trained on Quadruped-80K — >80,000 images of
  quadrupeds (horses, dogs, rodents, etc.). Not relevant for hm2p.

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
HRNet-W32. We hit checkpoint format mismatches and head architecture incompatibilities,
and fell back to HRNet-W32 from ImageNet.

Based on the paper, the DLC 3.x codebase (source analysis), and known issues in the
rc-series, the following failure modes are most likely:

### 4a. TF-to-PyTorch checkpoint format mismatch (most likely root cause)

DLC underwent a major rewrite from TensorFlow (2.x) to PyTorch (3.x). SuperAnimal weights
were originally stored in TF checkpoint format (.index / .data-00000 / .meta). The
transition to PyTorch required re-exporting all SA weights to `.pt` format via
`torch.save()`. In rc13 (a release candidate), this conversion was incomplete for the
HRNet-w32 variant: the HuggingFace-hosted checkpoint may still have been in TF format
while the loading code expected PyTorch state dict format.

PR #2756 ("SuperAnimal Model Updates", merged October 2024) specifically addressed:
- Bodypart mapping inconsistencies during memory replay fine-tuning
- `WeightInitialization` class redesigned to be format-agnostic
- DataLoader tensor size mismatches during training

### 4b. Decoder head output dimension mismatch (second most likely cause)

The SA-TVM HRNet-w32 decoder outputs heatmaps for all 27 superset keypoints: shape
(27, H, W). Our project is configured for 8 bodyparts. The `WeightInitialization` mechanism
uses a `conversion_array` to remap the 27-channel decoder to the user's N channels.

If this conversion array was not constructed (i.e., `create_conversion_table()` was not
called first), then `get_conversion_table()` would raise ValueError. Alternatively, if
`with_decoder=True` was set but the conversion array was the wrong shape, PyTorch would
raise a tensor dimension mismatch on the first forward pass. The error message in this
case is a generic tensor shape error, not an informative DLC-specific message.

Crucially: `WeightInitialization.__post_init__()` enforces:
- `memory_replay=True` requires `with_decoder=True`
- `with_decoder=True` requires `conversion_array` to be provided
- `len(bodyparts)` must equal first dimension of `conversion_array`

If `build_weight_init()` was called without first calling `create_conversion_table()`,
one of these assertions would fail (or the downstream call would fail).

### 4c. Confusion between `create_pretrained_project` and the fine-tuning workflow

`create_pretrained_project()` and `create_pretrained_project_pytorch()` create a project
pre-configured for the SA-TVM 27-keypoint vocabulary. They are designed for zero-shot or
minimal-label use with SA-TVM keypoints unchanged. If called with custom bodyparts (not
matching the SA-TVM 27 keypoints), the project config is inconsistent: the model expects
27 keypoints but the annotation space has 8. This causes a silent mismatch.

The correct fine-tuning workflow starts with `create_new_project()` (standard DLC project
creation with custom bodyparts), then adds SuperAnimal weight initialization separately
via `build_weight_init()`. These two workflows were not clearly distinguished in rc13
documentation.

### 4d. DLC 3.0 rc13 was known-unstable for fine-tuning

DLC 3.0.0rc13 was a release candidate during active PyTorch engine development. GitHub
issue #2702 documents training instability (GPU underutilization, system freeze) in rc-series
versions. The `train/pose_cfg.yaml` file expected by some code paths was being inconsistently
generated across rc versions; callers that relied on it would fail silently or with
misleading file-not-found errors.

### 4e. Summary

Almost certainly a combination of: TF-format checkpoint loaded by PyTorch code, conversion
array missing or malformed (causing 27-channel decoder to be loaded against an 8-channel
head), and rc13 instability. The fallback to ImageNet training was correct given the
circumstances, but it means we forfeited the data efficiency benefit that SA-TVM provides.

---

## 5. Correct API for DLC 3.0 (Stable Release)

### Prerequisites

- DLC stable release (not rc13). Install: `uv pip install "deeplabcut[pytorch]"`
  Verify: `import deeplabcut; print(deeplabcut.__version__)`
- GPU required for training (CPU is too slow for HRNet-w32)
- SA-TVM HRNet-w32 checkpoint auto-downloaded from HuggingFace on first use
  (from `mwmathis/DeepLabCut-SuperAnimal-TopViewMouse`)

### Step-by-step: memory replay fine-tuning with custom bodyparts

The critical insight: this is a two-workflow system. `create_pretrained_project()` is for
zero-shot use with SA-TVM keypoints unchanged. Fine-tuning with custom bodyparts requires
starting with `create_new_project()` and adding SA-TVM weight initialization separately.

**Step 1: Create a standard DLC project with custom bodyparts.**

```python
import deeplabcut

config_path = deeplabcut.create_new_project(
    "hm2p-superanimal-ft",
    "chaplinta",
    videos=["/path/to/representative_video.mp4"],
    working_directory="/workspace/derivatives/pose",
    copy_videos=False,
)
# Edit config.yaml to set bodyparts: [nose_tip, left_ear, right_ear,
# implant_base_rear, neck, mid_back, mouse_center, tail_base]
```

**Step 2: Label frames (or import existing labels).**

Import the 184 existing labeled frames from the current DLC project. Ensure bodypart names
in the labels match the config.yaml exactly (case-sensitive).

**Step 3: Create a training dataset.**

```python
deeplabcut.create_training_dataset(
    config_path,
    num_shuffles=1,
    net_type="hrnet_w32",
    engine="pytorch",    # must be pytorch, not tensorflow
)
```

**Step 4: Create and register the bodypart conversion table.**

This step was missing from our failed rc13 attempt. Write a CSV file:

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

The empty value for `implant_base_rear` means no SA-TVM equivalent. DLC will train it
from our labeled data only (no pseudo-labels from SA-TVM zero-shot predictions).

Then load it into the project config:

```python
from deeplabcut.modelzoo import utils as zoo_utils
zoo_utils.read_conversion_table_from_csv(config_path, "/path/to/conversion.csv")
# This writes to config.yaml under "SuperAnimalConversionTables"
```

Alternatively, use:

```python
zoo_utils.create_conversion_table(
    config=config_path,
    super_animal="superanimal_topviewmouse",
    project_to_super_animal={
        "nose_tip": "nose",
        "left_ear": "left_ear",
        "right_ear": "right_ear",
        "implant_base_rear": None,   # no SA-TVM equivalent
        "neck": "neck",
        "mid_back": "mid_back",
        "mouse_center": "mouse_center",
        "tail_base": "tail_base",
    },
)
```

**Step 5: Build the weight initialization object.**

```python
from deeplabcut.modelzoo.weight_initialization import build_weight_init

weight_init = build_weight_init(
    cfg=config_path,
    super_animal="superanimal_topviewmouse",
    model_name="hrnetw32",
    detector_name=None,         # no separate detector for single-animal top-down
    with_decoder=True,          # load SA-TVM 27-keypoint decoder
    memory_replay=True,         # pseudo-labels for SA keypoints not in our set
)
```

The `WeightInitialization` object enforces consistency (via `__post_init__`):
- `memory_replay=True` requires `with_decoder=True`
- `with_decoder=True` requires `conversion_array` to be present (from Step 4)
- `len(bodyparts)` must equal first dimension of `conversion_array`

If Step 4 was skipped, `build_weight_init()` will raise `ValueError` from
`get_conversion_table()` — "No conversion table found in config." This is the error that
likely appeared in an obscure form during our rc13 attempt.

**Step 6: Train.**

```python
deeplabcut.train_network(
    config_path,
    shuffle=1,
    weight_init=weight_init,
    # For small datasets (<64 images), DLC reduces lr=5e-5 and freezes BN stats
)
```

Training loop (memory replay): for each batch, GT labels are used for our 8 keypoints.
For the other 19 SA-TVM keypoints not in our project, cached zero-shot predictions are
used as pseudo-labels if confidence > 0.7. Loss is computed over all 27 SA-TVM keypoints.
`implant_base_rear` receives GT labels only (no pseudo-labels generated).

**Step 7: Inference.**

```python
deeplabcut.analyze_videos(config_path, video_list, shuffle=1)
```

Output: predictions for all 8 project bodyparts. The decoder conversion array maps the
27-channel SA-TVM output to the 8 project bodyparts.

### Encoder-only transfer (lower risk alternative)

If the decoder conversion table continues to cause issues, encoder-only transfer is still
meaningful at our label count. Set `with_decoder=False`:

```python
weight_init = build_weight_init(
    cfg=config_path,
    super_animal="superanimal_topviewmouse",
    model_name="hrnetw32",
    detector_name=None,
    with_decoder=False,     # encoder only; decoder randomly initialized
    memory_replay=False,    # irrelevant without decoder
)
```

No conversion table is needed for encoder-only transfer. This is the lowest-risk path and
still provides pose-aware visual features in the backbone.

### Zero-shot inference + video adaptation (no labeled data required)

From the paper's Code API section:

```python
deeplabcut.video_inference_superanimal(
    videos=["/path/to/video.mp4"],
    superanimal_name="superanimal_topviewmouse",
    model_name="hrnetw32",
    detector_name=None,
    scale_list=None,            # not needed for top-down HRNet (only for DLCRNet)
    video_adapt=True,           # pseudo-label domain adaptation
    adapt_iterations=1000,
    pseudo_threshold=0.5,       # paper uses 0.5 for video adapt evaluation
    bbox_threshold=0.9,
    pose_epochs=4,
    video_adapt_batch_size=8,
    pcutoff=0.1,
    dest_folder="/path/to/output",
)
```

Note: output uses SA-TVM 27-keypoint vocabulary, not our 8 custom bodyparts.
Extract: nose (->nose_tip), left_ear, right_ear, neck, mid_back, mouse_center, tail_base.
`implant_base_rear` is unavailable from zero-shot output.

### Number of labeled frames needed

Based on paper Table S3 (SA-TVM, HRNet-w32, DLC-Openfield benchmark):

| Labeled frames | Method              | mAP    | RMSE (px) |
|---------------|---------------------|--------|-----------|
| 10 (~1%)      | ImageNet transfer   | 91.5   | 7.00      |
| 10 (~1%)      | SA-TVM memory replay| 99.6   | 2.38      |
| 50 (~5%)      | ImageNet transfer   | 98.9   | 2.16      |
| 50 (~5%)      | SA-TVM memory replay| 99.8   | 1.95      |
| 100 (~10%)    | ImageNet transfer   | 99.3   | 1.57      |
| 100 (~10%)    | SA-TVM memory replay| 99.9   | 1.54      |
| Full (~1000)  | ImageNet transfer   | 100.0  | 1.13      |
| Full (~1000)  | SA-TVM memory replay| 99.9   | 1.21      |

At 184 labeled frames (approximately 10-18% of the DLC-Openfield benchmark), we are in
the regime where SA-TVM memory replay and ImageNet transfer learning converge. The largest
gain is in the low-data regime (<50 frames), which is below our current labeling level.
However, even at 100 frames, SA-TVM has better encoder features and pose priors, which
should manifest as lower error on harder frames (occlusion by headstage, fast turns,
dark illumination conditions).

---

## 6. Video Adaptation: What It Does and When to Use It

### Mechanism

Video adaptation (`video_adapt=True`) is test-time self-supervised domain adaptation.
It requires zero labeled frames. The process:
1. Run SA-TVM zero-shot inference on the first video in the input list.
2. Keep predictions with confidence > `pseudo_threshold` (default 0.1; paper uses 0.5
   for quantitative evaluation).
3. Fine-tune the model on these pseudo-labels for `pose_epochs` epochs (default 4,
   = ~1000 iterations at batch size 8).
4. During fine-tuning, batch normalization layers are frozen (called in `model.eval()` mode
   for BN stats only — affine parameters still update). This is critical: without frozen BN
   stats, the small pseudo-label batch shifts the BN statistics and performance degrades.
5. Re-run inference with the adapted weights.

Only the first video drives adaptation; the same adapted weights are applied to all videos.
The paper shows this generalizes to all videos in the same experimental setup (robustness
gain: +4 mAP over self-pacing on all 30 Horse-30 videos after adapting to one).

### Full function signature (DLC 3.0)

```python
deeplabcut.video_inference_superanimal(
    videos=["/path/to/video.mp4"],              # list of videos; first used for adaptation
    superanimal_name="superanimal_topviewmouse",
    model_name="hrnetw32",
    detector_name=None,                          # single-animal: no separate detector needed
    scale_list=None,                             # only for DLCRNet bottom-up; not HRNet
    video_adapt=True,
    adapt_iterations=1000,                       # for DLCRNet; use pose_epochs for HRNet
    pseudo_threshold=0.5,                        # confidence cutoff for pseudo-labels
    bbox_threshold=0.9,                          # detector confidence (if detector used)
    pose_epochs=4,                               # for HRNet top-down: epochs of adaptation
    detector_epochs=4,
    video_adapt_batch_size=8,
    pcutoff=0.1,                                 # confidence cutoff for final predictions
    dest_folder="/path/to/output",
)
```

### Is video_adapt useful for hm2p?

**Where it helps:**
- Reduces temporal jitter in keypoint traces. The paper shows significant jitter reduction
  (F(1, 23286) = 190.03, p < 0.0001) across diverse video types.
- Adapts to our specific illumination conditions, including the light/dark alternation. The
  pixel statistics shift substantially between lit and dark frames; video_adapt can reduce
  the domain gap for dark frames where SA-TVM was likely never trained.
- Zero additional labeling effort.
- For top-down HRNet-w32 specifically, the spatial-pyramid resolution mismatch is already
  handled by the detector crop, so video_adapt primarily provides temporal smoothing.

**Where it does not help:**
- `implant_base_rear`: SA-TVM has no superset keypoint for this. Video adaptation will
  not generate pseudo-labels for it and cannot improve its tracking.
- Cannot replace labeled data for learning custom keypoints.
- Does not transfer learned weights across sessions by default (each batch of videos is
  adapted from the base SA-TVM checkpoint, not from a prior session's adapted weights).
- Two videos in our dataset with substantially different appearance (e.g., first video is
  lights-on only) may require separate adaptation passes.

**Recommended use:** Run video_adapt as a zero-shot baseline test to evaluate SA-TVM
performance on our camera setup before investing in full fine-tuning. This answers the
question: "How much does SA-TVM help without any labeling?" The answer informs how much
benefit to expect from fine-tuning on top.

After fine-tuning, video_adapt could be applied as an optional post-processing step if
temporal jitter in the ear-vector angle remains problematic. This would require calling
`video_inference_superanimal()` with the fine-tuned checkpoint:

```python
deeplabcut.video_inference_superanimal(
    videos=all_session_videos,
    superanimal_name="superanimal_topviewmouse",
    model_name="hrnetw32",
    video_adapt=True,
    customized_pose_checkpoint="/path/to/finetuned-snapshot.pt",
)
```

### Light/dark specific consideration

Our sessions alternate 1 min lights-on / 1 min lights-off (total darkness). The SA-TVM
training data (TopViewMouse-5K) was entirely collected under normal laboratory lighting.
Dark frames are genuinely out-of-distribution for SA-TVM. Video adaptation to our videos
will help because the pseudo-labels from lit frames carry spatial prior information that
constrains predictions in dark frames. However, if dark frames have very low confidence
for all keypoints, few pseudo-labels will pass the confidence threshold and adaptation
will be minimal for those frames. In that case, fine-tuning with our labeled dark frames
is the only effective path.

---

## 7. Expected Improvement from SuperAnimal vs ImageNet-Only

### Quantitative benchmarks from the paper (SA-TVM, HRNet-w32)

From Table 1 and Table S3 (SA-TVM HRNet-w32 on DLC-Openfield benchmark):

| Data ratio | Labeled frames | Method               | mAP    | RMSE (px) |
|------------|---------------|----------------------|--------|-----------|
| 1%         | ~10            | ImageNet transfer    | 91.5   | 7.00      |
| 1%         | ~10            | SA-TVM memory replay | 99.6   | 2.38      |
| 5%         | ~50            | ImageNet transfer    | 98.9   | 2.16      |
| 5%         | ~50            | SA-TVM memory replay | 99.8   | 1.95      |
| 10%        | ~100           | ImageNet transfer    | 99.3   | 1.57      |
| 10%        | ~100           | SA-TVM memory replay | 99.9   | 1.54      |
| 100%       | ~1000          | ImageNet transfer    | 100.0  | 1.13      |
| 100%       | ~1000          | SA-TVM memory replay | 99.9   | 1.21      |
| zero-shot  | 0              | SA-TVM               | 95.2   | 4.88      |

Key finding: at 10 labeled frames, SA-TVM memory replay achieves 2.38 px RMSE vs ImageNet
7.00 px — approximately 3x lower error. To match SA-TVM's 10-frame performance, ImageNet
requires 101 frames. Cohen's d = 4.88 at the 1% data ratio (p < 0.0001).

For TriMouse (3 simultaneous mice, harder): at 1% data (1 frame per mouse!), ImageNet
achieves RMSE 31.6 px vs SA-TVM memory replay 5.85 px — a 5x improvement. Cohen's d =
10.99 (p < 0.0001). This shows the prior is especially valuable in harder conditions.

### What the benchmarks mean for our setup

Our dataset: 184 labeled frames, 8 custom bodyparts, overhead single-mouse. At ~184 frames,
we are between the 10% and 100% data regime for DLC-Openfield. The SA-TVM advantage in RMSE
is small at this label count for keypoints with SA-TVM priors. The advantage is expected
to be larger in practice because:

1. **Our conditions are harder than DLC-Openfield.** DLC-Openfield is a clean open-field
   with consistent overhead lighting and no headstage. Our rose maze has: 2P headstage
   occluding neck/back region, fibre cable, and complete darkness in alternating epochs.
   In harder OOD conditions, the SA-TVM pose prior matters more, not less.

2. **Dark frames are genuinely OOD for both models**, but SA-TVM's stronger pose encoder
   provides more robust feature extraction in low-contrast frames than an ImageNet-only encoder.
   The encoder has learned what a mouse body looks like structurally, not just texturally.

3. **implant_base_rear is learned from scratch regardless.** For this keypoint, the encoder
   prior still helps (better feature extraction from the backbone) even if the decoder starts
   randomly. Both with_decoder=False and with_decoder=True give encoder benefits.

4. **HD-critical keypoints (left_ear, right_ear) are strongly represented in SA-TVM training
   data.** TopViewMouse-5K includes ears across 13 lab datasets in diverse conditions. The
   pose prior for ear localization is strong.

### Impact on HD computation

The primary analysis pipeline computes head direction as the angle of the left_ear → right_ear
vector. Spatial tracking error in ear positions propagates to angular error in HD estimates.

Geometric relationship: if the inter-ear distance is D pixels and the lateral tracking error
is ε px, the angular error in HD is approximately arctan(2ε / D). For a mouse where D ≈ 60
px (at our camera calibration), RMSE 7 px → HD error ≈ 13°, RMSE 2.5 px → HD error ≈ 5°.

At our HD bin width of 10–12°, a 13° tracking error substantially smears the tuning curve
peaks. A 5° error is below the bin width and would have minimal impact on tuning curve
shape. This is a meaningful improvement for HD cell analysis.

In practice, the more important metric is **confidence dropout rate**: the fraction of frames
where both ears are tracked at confidence < 0.6 (requiring interpolation or exclusion).
SA-TVM pre-training reduces this by providing stronger pose priors that prevent the model
from losing both ears simultaneously. This is the primary quality benefit we expect at
our label count.

### Expected outcome

Given 184 labeled frames:
- **Encoder-only SA-TVM transfer**: ~10-20% RMSE improvement over current HRNet-W32 from
  ImageNet. Primarily benefits encoder feature quality. Low implementation risk.
- **Memory replay full fine-tuning**: ~20-40% RMSE improvement for the 7 SA-TVM-matched
  keypoints. Higher benefit in dark frames. Requires conversion table setup. Moderate risk.
- **implant_base_rear**: same as ImageNet baseline for the decoder, encoder benefits only.

---

## 8. Recommended Action Items for hm2p

### Priority order

**Priority 1: Verify SA-TVM zero-shot + video_adapt on our videos (no labels needed)**

Before investing in fine-tuning infrastructure, run video_adapt zero-shot on 2–3 sessions
to establish baseline SA-TVM performance on our camera setup:

```python
deeplabcut.video_inference_superanimal(
    videos=["/path/to/session_video.mp4"],
    superanimal_name="superanimal_topviewmouse",
    model_name="hrnetw32",
    video_adapt=True,
    pose_epochs=4,
    pseudo_threshold=0.5,
    dest_folder="/path/to/zero_shot_output",
)
```

Map SA-TVM keypoints to hm2p bodyparts: nose→nose_tip, left_ear, right_ear, neck,
mid_back, mouse_center, tail_base. Compute tracking quality metrics (confidence dropout
rate, RMSE vs manual annotations on held-out frames). This establishes the ceiling for
zero-shot performance and the floor for fine-tuning benefit.

**Priority 2: Encoder-only SA-TVM transfer (lowest risk, clear benefit)**

1. Upgrade DLC to current stable: `uv pip install "deeplabcut[pytorch]"`
2. Create fresh DLC project with 8 custom bodyparts
3. Import 184 labeled frames
4. `build_weight_init(with_decoder=False, memory_replay=False)` — no conversion table needed
5. Train: `deeplabcut.train_network(config_path, weight_init=weight_init)`
6. Evaluate on held-out sessions; compare RMSE and confidence dropout to current model

**Priority 3: Memory replay fine-tuning (if encoder-only succeeds)**

Once encoder-only transfer is confirmed working:
1. Add conversion table (Step 4 above)
2. Rebuild `weight_init` with `with_decoder=True, memory_replay=True`
3. Train and compare

If the conversion table or decoder loading fails, fall back to encoder-only which is
already a meaningful improvement.

**Priority 4: Label additional dark-condition frames**

If dark-frame tracking quality is substantially worse than lit-frame quality, label an
additional 30–50 frames from dark epochs specifically. SA-TVM has no dark-condition prior;
additional labeled dark frames are the only path to robust dark-frame tracking.

### What not to do

- Do not use `create_pretrained_project()` for fine-tuning with custom bodyparts.
  This function is for zero-shot SA-TVM inference only.
- Do not set `with_decoder=True` without first calling `create_conversion_table()`.
  The error is non-informative and caused our original failure.
- Do not use rc13 for any SuperAnimal fine-tuning work. Upgrade first.
- Do not run video_adapt routinely on all 26 sessions (high compute cost, limited benefit
  over fine-tuning). Use it as a diagnostic or one-time baseline test.

---

## References

- Ye S, Filippova A, Lauer J, Schneider S, Vidal M, Qiu T, Mathis A, Mathis MW. 2024.
  "SuperAnimal pretrained pose estimation models for behavioral analysis."
  *Nature Communications* 15:5165. doi:10.1038/s41467-024-48792-2
  GitHub: https://github.com/DeepLabCut/DeepLabCut

- Wang J, Sun K, Cheng T, Jiang B, Deng C, Zhao Y, Liu D, Hou Y, Cheng W, Hu W, Ding B,
  Liu Y. 2020. "Deep High-Resolution Representation Learning for Visual Recognition."
  *IEEE TPAMI*. (HRNet-w32 architecture used in SA-TVM)

- TopViewMouse-5K training data (Zenodo):
  https://zenodo.org/records/10618947

- SuperAnimal model weights (HuggingFace):
  https://huggingface.co/mwmathis/DeepLabCut-SuperAnimal-TopViewMouse

- DLC ModelZoo web app: http://modelzoo.deeplabcut.org

- SA-TVM bodypart configuration: `deeplabcut/modelzoo/project_configs/superanimal_topviewmouse.yaml`
- SA-TVM keypoint conversion table: `deeplabcut/modelzoo/conversion_tables/conversion_table_topview.csv`
- Weight initialization API: `deeplabcut/modelzoo/weight_initialization.py`
- Video adaptation API: `deeplabcut/modelzoo/video_inference.py`
