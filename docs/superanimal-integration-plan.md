# SuperAnimal Integration Plan

**Date:** 2026-04-02  
**Status:** Planning — do not implement until reviewed  
**Authors:** Data Scientist + Lead Developer research notes synthesised by Architect  
**Related docs:**
- `docs/superanimal-integration-notes-datascientist.md` — paper analysis and SA workflow
- `docs/superanimal-integration-notes-leaddev.md` — DLC 3.x API forensics and failure analysis
- `docs/architecture-review-dlc-pipeline.md` — prior review of 20 pipeline flaws

---

## Context and Starting State

**DLC project:** `sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/`  
**Bodyparts (8):** `nose_tip`, `left_ear`, `right_ear`, `head_midpoint`, `neck`,
`mid_back`, `mouse_center`, `tail_base`  
**Labeled frames:** 183 frames across 16 sessions (confirmed from `CollectedData_tristan.h5` files)  
**Current `default_net_type`:** `resnet_50` (in `config.yaml`)  
**Current backbone pretrain status:** HRNet run from random init — `pretrained: false` in
`deeplabcut/modelzoo/model_configs/hrnet_w32.yaml` template; never overridden in training  
**Confirmed mAP:** ResNet-50 + ImageNet ~57%; HRNet-w32 + random init ~34%  
**Bodypart mapping (hm2p → SuperAnimal TopViewMouse):**

The paper states that users only need to label the bodyparts they care about.
Gradient masking prevents penalties for unlabelled SA bodyparts, and memory
replay generates pseudo-labels for the remaining SA keypoints to prevent
catastrophic forgetting. Our 8 bodyparts map to the SA-TVM 27-keypoint
superset as follows:

| # | hm2p bodypart | SA-TVM keypoint | SA index | Notes |
|---|---|---|---|---|
| 0 | `nose_tip` | `nose` | 0 | Direct match |
| 1 | `left_ear` | `left_ear` | 1 | Direct match |
| 2 | `right_ear` | `right_ear` | 2 | Direct match |
| 3 | `head_midpoint` | *(none)* | -1 | Custom — no SA equivalent. Zero-initialised, trained from our labels only |
| 4 | `neck` | `neck` | 7 | Direct match |
| 5 | `mid_back` | `mid_back` | 8 | Direct match |
| 6 | `mouse_center` | `mouse_center` | 9 | Direct match |
| 7 | `tail_base` | `tail_base` | 13 | Direct match |

**conversion_array:** `[0, 1, 2, -1, 7, 8, 9, 13]`

The SA-TVM model's full 27 keypoints (from `superanimal_topviewmouse.yaml`):
`nose(0)`, `left_ear(1)`, `right_ear(2)`, `left_ear_tip(3)`, `right_ear_tip(4)`,
`left_eye(5)`, `right_eye(6)`, `neck(7)`, `mid_back(8)`, `mouse_center(9)`,
`mid_backend(10)`, `mid_backend2(11)`, `mid_backend3(12)`, `tail_base(13)`,
`tail1-5(14-18)`, `left_shoulder(19)`, `left_midside(20)`, `left_hip(21)`,
`right_shoulder(22)`, `right_midside(23)`, `right_hip(24)`, `tail_end(25)`,
`head_midpoint(26)`.

7 of 8 bodyparts have direct SA matches. `head_midpoint` is handled
via the `-1` sentinel in the conversion array — the backbone still provides
features for it, only the head channel is trained from scratch.

**config.yaml `SuperAnimalConversionTables`** (current, needs `head_midpoint: null` added):
```yaml
SuperAnimalConversionTables:
  superanimal_topviewmouse:
    nose_tip: nose
    left_ear: left_ear
    right_ear: right_ear
    head_midpoint: null    # ← add this for SA transfer
    neck: neck
    mid_back: mid_back
    mouse_center: mouse_center
    tail_base: tail_base
```

---

## Risk Assessment Summary

| Approach | Expected mAP | Implementation effort | Risk |
|---|---|---|---|
| HRNet-w32 + ImageNet (Phase 1, immediate fix) | 55–65% | 30 min | Low |
| HRNet-w32 + SA backbone only, Mode A (Phase 2) | 65–75%+ | 2–3 hrs | Medium |
| HRNet-w32 + SA full fine-tune, Mode B (rebuild) | 75–85%+ | 8–16 hrs | High |
| Patch DLC source for channel slicing | Unpredictable | 4–8 hrs + testing | Very high |

**Decision rule:** If Phase 1 achieves >= 65% mAP, proceed to Phase 2 as an incremental
experiment on a separate shuffle (not replacing the working model). If Phase 1 achieves
< 55% mAP — i.e., worse than ResNet-50 — revert to ResNet-50 immediately.

---

## Phase 1: Immediate Fix — HRNet + ImageNet Pretraining

### Problem

`run_dlc_retrain.py` sets `pytorch_config.yaml` backbone to `hrnet_w32` directly in the
YAML dict but never sets `"pretrained": True`. The DLC HRNet template
(`deeplabcut/modelzoo/model_configs/hrnet_w32.yaml`) defaults to `pretrained: false`,
meaning training starts from random initialisation. This is why HRNet achieves only 34%
mAP — far below the ResNet-50 baseline of 57%.

### What to change in `run_dlc_retrain.py`

The `train()` function currently does two things to configure the backbone: (1) direct
YAML manipulation of `pytorch_config.yaml`, and (2) a call to `deeplabcut.train_network()`
with no `pytorch_cfg_updates`. The fix requires one addition to the `train_network` call.

**Current call (lines 160–165):**
```python
deeplabcut.train_network(
    str(config_path),
    maxiters=maxiters,
    displayiters=100,
    saveiters=5000,
)
```

**Required call:**
```python
deeplabcut.train_network(
    str(config_path),
    maxiters=maxiters,
    displayiters=100,
    saveiters=5000,
    batch_size=batch_size,          # fixes issue 19 from architecture review
    pytorch_cfg_updates={
        "model.backbone.pretrained": True,
    },
)
```

**Why `pytorch_cfg_updates` is the correct mechanism:** The YAML manipulation block
already sets `backbone.model_name = "hrnet_w32"` and `backbone.type = "HRNet"`, but does
not set `pretrained`. The `pytorch_cfg_updates` dict is applied by DLC's `train_network`
using dotted-path updates on the loaded `pytorch_config.yaml` *before* model
instantiation. Setting `"model.backbone.pretrained": True` is equivalent to adding
`pretrained: true` under the `backbone:` block in the YAML. It overrides the template
default.

Note: the current YAML manipulation block already writes `backbone.model_name` correctly
before `train_network` is called. The `pytorch_cfg_updates` then applies on top. The order
is correct.

**Also fix `default_net_type` in `config.yaml` (one-time manual change):**

The project `config.yaml` has `default_net_type: resnet_50`. When `create_training_dataset`
is called without a `weight_init` object, it reads `default_net_type` to build
`pytorch_config.yaml`. Leaving it as `resnet_50` causes `create_training_dataset` to
generate a ResNet config, which is then overridden by the YAML manipulation block. This
works but is fragile — the initial ResNet config may have incorrect head channel counts
that the manipulation block incompletely patches. The cleaner fix is:

```
default_net_type: hrnet_w32
```

in `config.yaml`. This is a one-time change to the DLC project config before uploading
updated labels to S3. After this change, `create_training_dataset` generates an HRNet
config natively and the YAML manipulation block either becomes a no-op for the backbone
section or can be simplified.

### Expected improvement

From Ye et al. 2024 Table S3 (HRNet-w32, DLC-Openfield benchmark, most comparable to our
overhead-view setup):

- ImageNet baseline, ~180 frames (comparable to our 183): mAP in the 95–99% range on an
  in-distribution benchmark. Our OOD setting (novel arena, headstage occlusion, variable
  lighting) will perform worse. Realistic expectation: **55–65% mAP**, recovering to
  approximately the ResNet-50 level or better. HRNet-w32 has 29M parameters vs ResNet-50's
  23M and stronger multi-scale feature learning; with ImageNet weights it should be at
  least as good as ResNet-50.

- The ResNet-50 baseline achieved ~57% mAP. HRNet + ImageNet should reach the same range,
  with potential upside from the stronger backbone once it is not training from scratch.

### Augmentation note

The current augmentation settings are aggressive (rotation 180°, scale 0.25–2.5x,
brightness/contrast ±60%, gaussian noise 30). These were appropriate for a dataset
training from random init (to prevent overfitting on 183 frames). With ImageNet pretraining,
strong augmentation remains appropriate — the pretrained backbone provides regularisation
at the feature level, not at the augmentation level. No augmentation changes are required
for Phase 1.

However, if mAP is still low after Phase 1, consider reducing augmentation strength as a
diagnostic experiment before Phase 2. Very aggressive geometric augmentation can hurt HRNet
specifically because its high-resolution feature maps are sensitive to large spatial
distortions.

### Launch procedure for Phase 1

No changes to `launch_dlc_finetune_ec2.py`. The existing launch workflow is unchanged.

```bash
# 1. Edit config.yaml: set default_net_type: hrnet_w32
# 2. Upload updated config + labels
uv run python scripts/upload_dlc_labels.py

# 3. Launch training + inference
uv run python scripts/launch_dlc_finetune_ec2.py --epochs 400
```

The `--epochs 400` is unchanged from the current default.

### Rollback plan

If Phase 1 HRNet + ImageNet achieves < 55% mAP (i.e., worse than ResNet-50):

1. Revert `default_net_type` to `resnet_50` in `config.yaml`.
2. Remove the `pytorch_cfg_updates` from `run_dlc_retrain.py`.
3. Re-upload labels.
4. Re-launch training. This restores the working ResNet-50 model.

The ResNet-50 model weights from the previous run are still on S3 at
`s3://hm2p-derivatives/dlc-retrain/models/`. They can be used directly with
`--infer-only` from that snapshot without re-training.

---

## Phase 2: SuperAnimal Backbone Transfer (Mode A)

**Prerequisite:** Phase 1 mAP result available. Proceed to Phase 2 if Phase 1 is below
65% mAP, or as a planned incremental improvement regardless.

### What Mode A is

Mode A loads the SA-TVM HRNet-w32 backbone (encoder only) as the starting point instead
of ImageNet. The decoder (prediction head) is randomly initialised, as in standard transfer
learning. Mode A does not require the conversion table and does not use memory replay. It
is the correct pathway for a project that was not created via `create_pretrained_project()`.

Expected gain over ImageNet backbone: approximately 5–15 mAP points in our data regime
(Ye et al. 2024, Table S3: Mode A at 1% data gives mAP 96.6 vs ImageNet 91.5 on the
DLC-Openfield in-distribution benchmark; our OOD setting scales this advantage
proportionally, not absolutely).

### Step-by-step implementation

**Step 1: Verify DLC version is stable >= 3.0.0 (not rc13)**

```python
import deeplabcut
print(deeplabcut.__version__)
# Must be >= "3.0.0" (a release, not "3.0.0rc13" or similar)
```

If the version is rc13, upgrade:
```bash
uv pip install "deeplabcut[pytorch,gui]" --upgrade
```

The SA fine-tuning API changed significantly between rc13 and 3.0.0 stable (PR #2756,
merged October 2024). The steps below are written for 3.0.0 stable.

**Step 2: Set `default_net_type: hrnet_w32` in `config.yaml`**

This is the same change as Phase 1. If Phase 1 was done, this is already in place.

```yaml
# In config.yaml
default_net_type: hrnet_w32
```

This is required because `create_training_dataset` reads `default_net_type` to decide
which backbone config to generate. The `model_name` parameter in `build_weight_init` does
not back-propagate into the project config automatically.

**Step 3: Build the `WeightInitialization` object**

This step must happen before `create_training_dataset` is called:

```python
from deeplabcut.modelzoo.weight_initialization import build_weight_init

config_path = "/path/to/hm2p-retrain-tristan-2026-03-20/config.yaml"

weight_init = build_weight_init(
    cfg=config_path,
    super_animal="superanimal_topviewmouse",
    model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2",   # top-down: detector required
    with_decoder=False,                             # Mode A: backbone only
    memory_replay=False,                            # not applicable without decoder
)
```

This triggers a HuggingFace download of `superanimal_topviewmouse_hrnet_w32.pt` on first
run (cached in DLC's local cache directory thereafter). The returned `weight_init` object
stores the checkpoint path.

`detector_name`: the lead developer's notes show `"fasterrcnn_resnet50_fpn_v2"` (v2
variant) in the Mode A example. Verify against the DLC source's
`deeplabcut/modelzoo/weight_initialization.py` — if `"fasterrcnn_resnet50_fpn"` (v1) is
the correct string for SA-TVM, use that. The checkpoint naming convention is
`{detector_name}.pt`, so the filename will confirm which is correct.

**Step 4: Create training dataset with `weight_init`**

```python
import deeplabcut

deeplabcut.create_training_dataset(
    config_path,
    weight_init=weight_init,
    # num_shuffles defaults to 1; use shuffle=2 to keep shuffle=1 as the ImageNet model
)
```

Internally, `create_training_dataset` branches on `weight_init.with_decoder`:
- `with_decoder=False` → calls `make_pytorch_pose_config()` (standard backbone config)
- This generates `pytorch_config.yaml` with the SA checkpoint path written under
  `train_settings.weight_init.snapshot_path`

The generated `pytorch_config.yaml` will use HRNet-w32 as the backbone (because
`default_net_type: hrnet_w32` is now set), with the SA checkpoint as the weight source.

**Shuffle strategy:** Create a new shuffle (shuffle=2) for the SA model, leaving
shuffle=1 intact as the ImageNet baseline. This allows direct A/B comparison without
destroying the working model.

```python
deeplabcut.create_training_dataset(
    config_path,
    weight_init=weight_init,
    num_shuffles=1,
    # This creates the next available shuffle index
)
# Note the shuffle index that was created (e.g., shuffle=2)
```

**Step 5: Train with `load_head_weights=False`**

This is the critical parameter identified in the lead developer's notes. Without it,
`load_snapshot()` in `runners/train.py` defaults to `load_head_weights=True`, which
attempts to load the 27-channel SA head into our 8-channel head with strict=True —
causing a `RuntimeError`.

```python
deeplabcut.train_network(
    config_path,
    shuffle=2,                          # the SA shuffle from Step 4
    epochs=400,
    displayiters=100,
    saveiters=5000,
    batch_size=8,
    load_head_weights=False,            # Mode A: backbone only, head random init
    pytorch_cfg_updates={
        "model.backbone.pretrained": True,  # belt-and-suspenders: SA checkpoint IS pretrained
    },
)
```

The `load_head_weights=False` flag causes `load_snapshot()` to:
1. Load the full SA checkpoint.
2. Filter keys to those with the `backbone.` prefix.
3. Call `model.backbone.load_state_dict(backbone_weights)` — shapes always match
   because the backbone architecture is identical (HRNet-w32 in both SA and our model).
4. Leave the head randomly initialised.

The `pytorch_cfg_updates` is belt-and-suspenders: the SA checkpoint contains a
pretrained backbone by definition, but this ensures any `pretrained: false` from the
template is overridden at model instantiation.

**Step 6: Evaluate and compare**

```python
deeplabcut.evaluate_network(
    config_path,
    Shuffles=[1, 2],        # compare ImageNet (shuffle=1) vs SA backbone (shuffle=2)
    plotting=True,
)
```

Evaluation metrics to compare:
- Per-bodypart RMSE (pixels)
- mAP at PCK threshold (DLC default: 0.95 of body length)
- Fraction of frames with confidence > 0.6 for both ears simultaneously (most important
  for HD computation)
- Visual inspection of labeled video overlays for 2–3 held-out sessions

**Step 7: Run inference with the SA model (if evaluation is better)**

```python
deeplabcut.analyze_videos(
    config_path,
    video_list,
    shuffle=2,               # use SA backbone shuffle
    destfolder=str(out_dir),
    batch_size=64,
)
```

This is identical to current inference, with `shuffle=2` selecting the SA model.

### Changes to `run_dlc_retrain.py` for Phase 2

The current script performs `create_training_dataset` then YAML manipulation, then
`train_network`. For SA Mode A, the workflow changes to:

1. `build_weight_init(with_decoder=False)` before `create_training_dataset`.
2. `create_training_dataset(weight_init=weight_init)` instead of the bare call.
3. Remove (or conditional-skip) the YAML backbone manipulation block — the backbone
   is already set correctly by `create_training_dataset` when `weight_init` is passed.
4. `train_network(..., load_head_weights=False)` instead of the bare call.

The `pytorch_cfg_updates` for `pretrained=True` is still needed.

These changes should be isolated behind a `--sa-backbone` flag to keep the Phase 1
(HRNet + ImageNet) path runnable without modification:

```bash
# Phase 1 (ImageNet):
python scripts/run_dlc_retrain.py --epochs 400

# Phase 2 (SA backbone):
python scripts/run_dlc_retrain.py --epochs 400 --sa-backbone
```

The `launch_dlc_finetune_ec2.py` forwards arbitrary flags to `run_dlc_retrain.py` via
`mode_flag`, so adding `--sa-backbone` there requires only one additional argument in
`build_user_data` and a new `argparse` argument in `launch_dlc_finetune_ec2.py`.

### Handling `head_midpoint`

`head_midpoint` has no SA-TVM superset equivalent. In Mode A, this is not a problem:
the head is randomly initialised regardless. All 8 bodyparts train from the same randomly
initialised head. The SA backbone provides pose-aware features to the head, which helps
the 7 matched bodyparts learn faster and generalise better. `head_midpoint` benefits
from the backbone features to the same extent as any other novel keypoint — the backbone
represents spatial relationships and texture features, not keypoint identity.

Expected behaviour for `head_midpoint`: similar training curve to Phase 1 (ImageNet).
No improvement from SA backbone is expected for this keypoint specifically, but no
degradation either.

### Fallback if SA backbone transfer fails

If `build_weight_init()` raises an error (checkpoint not found, version mismatch, etc.):

1. Run `deeplabcut.video_inference_superanimal(["test.mp4"], "superanimal_topviewmouse")`
   to trigger a clean checkpoint download and verify the HuggingFace endpoint is
   accessible and the checkpoint format is correct for the installed DLC version.
2. Check `deeplabcut.__version__` — confirm it is >= 3.0.0 stable, not an rc.
3. If the checkpoint loads for inference but not for `build_weight_init`, the API version
   is mismatched. Inspect `deeplabcut/modelzoo/weight_initialization.py` for the current
   `build_weight_init` signature.
4. If all SA-specific code fails: fall back to Phase 1 (HRNet + ImageNet). Phase 1 is
   expected to recover to ResNet-50 performance or better, which was the previous working
   state.

---

## Phase 3: Video Adaptation (Optional, Not Recommended as Primary Strategy)

### What it is

Video adaptation applies the SA-TVM model to an unlabelled target video, uses
high-confidence predictions (> 0.5) as pseudo-labels, and fine-tunes the model for 1000
iterations with batch size 1 and batch normalisation in eval mode. No human labels are
required. This is described in Ye et al. 2024 Section 2.4 ("Unsupervised Video
Adaptation").

The DLC 3.x API:

```python
deeplabcut.video_inference_superanimal(
    [video_path],
    superanimal_name="superanimal_topviewmouse",
    model_name="hrnetw32",
    video_adapt=True,           # enables pseudo-label fine-tuning for this video
    # scale_list not needed for top-down HRNet (only for DLCRNet bottom-up)
)
```

### Why it is not the primary strategy for this project

1. **`head_midpoint` is invisible to SA-TVM.** Video adaptation generates pseudo-labels
   from SA-TVM's zero-shot predictions. SA-TVM has no superset keypoint for
   `head_midpoint`, so no pseudo-labels are generated for it. The headstage is our
   most novel tracking challenge and video adaptation cannot help with it.

2. **Top-down models do not need scale adaptation.** The spatial-pyramid search benefit
   (the main motivation for video adaptation in the paper) is for bottom-up DLCRNet only.
   For HRNet-w32 top-down, the Faster R-CNN detector standardises the animal crop size at
   both train and test time. Video adaptation for HRNet primarily reduces temporal jitter,
   which is a second-order concern compared to overall mAP.

3. **We have 183 labeled frames.** Video adaptation is for the zero-label case. Memory
   replay fine-tuning with our existing labels will achieve better absolute performance
   than video adaptation, because our GT labels are more reliable than SA-TVM pseudo-labels
   for our specific rig (headstage, cable, arena shape).

4. **Per-video weights do not transfer.** Each video is adapted independently. With 26
   sessions, this means 26 separate adapted models. Managing, storing, and applying 26
   models is operationally complex with no benefit over a single fine-tuned model.

5. **API ambiguity for custom projects.** `video_inference_superanimal` runs the SA-TVM
   model directly, not our fine-tuned model. Video adaptation adapts the SA-TVM model, not
   our project model. The output bodyparts are SA-TVM's full 27-point superset, not our
   8 bodyparts. Mapping the adaptation output back to our `movement`-compatible format
   would require post-processing that is not currently in the pipeline.

### When to use video adaptation

Video adaptation could be applied as a diagnostic step if, after Phase 2:
- The ear-vector angle shows high temporal jitter (std > 5° per 100 ms window).
- The cause is traced to low-confidence frames rather than mislabeling.
- The affected sessions have consistent recording conditions different from the training set.

In this case, apply `video_inference_superanimal` with `video_adapt=True` on the
affected session, use SA-TVM's output for the 7 matched bodyparts only, and continue
using the fine-tuned model for `head_midpoint`. This is a per-session manual
intervention, not a routine pipeline step.

---

## Architecture: Pipeline Integration

### Where these phases fit in the Stage 2 pipeline

The Stage 2 pipeline has two substages:

- **Stage 2a (DLC training):** Takes labeled frames from `sourcedata/trackers/dlc/`,
  produces a trained model on EC2 GPU, uploads weights to S3.
- **Stage 2b (DLC inference):** Downloads trained model, runs inference on all 26 session
  videos, writes `.h5` pose files to `s3://hm2p-derivatives/pose/`.

Phase 1 and Phase 2 affect Stage 2a only. Inference (Stage 2b) is identical regardless
of which backbone was used — `deeplabcut.analyze_videos` is called the same way.

Downstream stages (3, 3b, 5, 6) are unaffected architecturally. However, per CLAUDE.md,
re-running Stage 2 invalidates all downstream stages:

> If Stage 2 (DLC) is re-run, all downstream stages must re-run: Stage 3 → Stage 3b
> (MoSeq) → Stage 5 → Stage 6. Stage 4 is independent.

After any Phase 1 or Phase 2 training run that results in inference over all 26 sessions,
Stages 3, 3b, 5, and 6 must be re-run. This is already handled by the existing pipeline
invalidation logic.

### Changes to `launch_dlc_finetune_ec2.py`

**Phase 1:** No changes to the launch script. The fix is entirely within
`run_dlc_retrain.py`.

**Phase 2:** Add a `--sa-backbone` flag:

```python
# In main() argparse block:
parser.add_argument(
    "--sa-backbone",
    action="store_true",
    help="Use SuperAnimal backbone transfer (Mode A) instead of ImageNet pretraining.",
)

# In build_user_data():
# Append "--sa-backbone" to mode_flag when args.sa_backbone is True.
```

The `build_user_data` function constructs the `mode_flag` string that is passed to
`run_dlc_retrain.py`. Adding `--sa-backbone` to this string is sufficient:

```python
if sa_backbone:
    mode_flag += " --sa-backbone"
```

### Changes to `run_dlc_retrain.py`

**Phase 1 (minimal change, high confidence):**

1. Add `pytorch_cfg_updates={"model.backbone.pretrained": True}` to the
   `deeplabcut.train_network()` call (line 160).
2. Add `batch_size=batch_size` to the same call (fixes architecture review Issue 19).

**Phase 2 (guarded behind `--sa-backbone` flag):**

1. Add `--sa-backbone` to `argparse` in `main()`.
2. Pass the flag through to `train()`.
3. In `train()`, if `sa_backbone=True`:
   a. Call `build_weight_init(with_decoder=False, ...)` before `create_training_dataset`.
   b. Pass `weight_init=weight_init` to `create_training_dataset`.
   c. Skip the backbone YAML manipulation block (no longer needed — backbone is set
      correctly by `create_training_dataset`).
   d. Add `load_head_weights=False` to `train_network`.
4. If `sa_backbone=False`: existing Phase 1 code path (HRNet + ImageNet).

The SA backbone shuffle index (e.g., shuffle=2) must be passed consistently between
`create_training_dataset` and `train_network`. Add a `shuffle` variable tracking this.

### Snakemake DAG

No changes to the Snakemake DAG are required for Phase 1 or Phase 2. These phases modify
how the Stage 2a rule trains the model, not the rule's inputs or outputs. The rule's output
is the trained model weights on S3, which is unchanged.

If Phase 2 produces a parallel shuffle (shuffle=2) that is evaluated before promotion, a
new intermediate Snakemake rule could gate promotion on evaluation metrics. This is
optional and deferred — for now, evaluation and promotion remain manual decisions outside
the Snakemake DAG.

---

## Testing Strategy

### Phase 1 acceptance criteria

Before promoting the Phase 1 model to production (`pose/` on S3), verify:

1. **mAP >= 57%** — must match or exceed the ResNet-50 baseline. If below, diagnose
   augmentation or learning rate before proceeding.
2. **Per-bodypart RMSE:** `left_ear` and `right_ear` RMSE <= 5 px in `evaluate_network`
   output. These are the primary HD-critical keypoints.
3. **Ear confidence dropout:** Fraction of frames where both ears simultaneously have
   confidence < 0.6 must be <= 10% on a held-out session. This is measured from the
   `.h5` pose file, not from the evaluation CSV.
4. **Visual inspection:** Review the labeled video for at least one session with lights
   off (total darkness). This is the hardest tracking condition and the most informative
   quality check.

Held-out session for evaluation: use one session NOT in the 16 labeled sessions. Identify
from `metadata/experiments.csv` any `exp_id` whose corresponding `labeled-data/` folder
does not exist.

### Phase 2 acceptance criteria

Phase 2 is accepted if:

1. SA backbone shuffle (shuffle=2) achieves > Phase 1 mAP on the held-out session.
2. `left_ear` and `right_ear` RMSE is lower than Phase 1.
3. No regressions on `head_midpoint` (expected: similar to Phase 1).

If Phase 2 does not improve over Phase 1, keep Phase 1 as production and record the
negative result in this document.

### Unit test scope

No new unit tests are added for the training scripts themselves — these are EC2 scripts
that require GPU and real DLC weights, which violate the "synthetic arrays only" rule for
`tests/`. The DLC-specific logic that can be unit-tested is:

- `parse_session_id()` in `ec2_utils.py` (covered by existing tests in `frontend/data.py`;
  once moved to `ec2_utils.py` per architecture review Issue 7, add tests there).
- Any new pure-Python logic added to `run_dlc_retrain.py` that does not depend on DLC
  (e.g., progress JSON construction, S3 path building, bodypart-to-SA-index mapping).

The SA conversion table values (7 bodypart SA indices) can be tested:

```python
# In tests/test_dlc_utils.py (new file)
def test_sa_tvm_conversion_indices():
    """SA-TVM bodypart indices for our 7 matched bodyparts."""
    # From superanimal_topviewmouse.yaml (0-indexed):
    expected = {
        "nose_tip": 0,      # nose
        "left_ear": 1,
        "right_ear": 2,
        "neck": 7,
        "mid_back": 8,
        "mouse_center": 9,
        "tail_base": 13,
    }
    # Verify against the DLC source file if importable, else assert known values
    assert expected["left_ear"] == 1
    assert expected["right_ear"] == 2
    assert expected["mid_back"] == 8
    assert expected["mouse_center"] == 9
    assert expected["tail_base"] == 13
```

---

## Rollback Plan

### If Phase 1 fails (mAP < 55%)

1. In `run_dlc_retrain.py`: remove `pytorch_cfg_updates`.
2. In `config.yaml`: revert `default_net_type` to `resnet_50`.
3. Upload updated config: `uv run python scripts/upload_dlc_labels.py`.
4. Launch with ResNet-50: the existing ResNet code path is preserved as long as
   `pytorch_cfg_updates` is not passed.

Alternatively, use the existing model weights on S3 (ResNet-50, mAP ~57%) directly with
`--infer-only`. These are at `s3://hm2p-derivatives/dlc-retrain/models/` from the last
successful training run.

### If Phase 2 fails (SA backbone fails to load or degrades mAP)

1. Do not promote shuffle=2.
2. Keep shuffle=1 (Phase 1 ImageNet model) as production.
3. Record the failure mode in this document under a "Phase 2 Results" section.
4. The `--sa-backbone` flag is a no-op in the default (no-flag) code path, so it does
   not affect any existing workflow.

### Production model identity

At all times, the model in production is identified by:
- Shuffle index (1 = ImageNet, 2 = SA backbone)
- Timestamp of the `dlc-retrain/models/` S3 objects
- `_retrain_progress.json` `updated` field

The frontend's `_get_rerun_status()` uses S3 timestamps to detect pipeline invalidation.
This is unaffected by which shuffle is in production — the S3 paths for pose outputs
(`pose/{sub}/{ses}/`) are the same regardless.

---

## Open Questions

1. **Is `load_head_weights=False` the correct parameter name in DLC 3.0 stable?**
   The lead developer's notes show this parameter based on source inspection of
   `runners/train.py`. Verify against `deeplabcut.train_network.__doc__` or the source
   in the installed DLC version before running Phase 2.

2. **Does `build_weight_init` accept `"fasterrcnn_resnet50_fpn"` or `"fasterrcnn_resnet50_fpn_v2"`?**
   The data scientist's notes say `"fasterrcnn_resnet50_fpn"`; the lead developer's notes
   say `"fasterrcnn_resnet50_fpn_v2"`. Check `deeplabcut/modelzoo/weight_initialization.py`
   for the list of valid detector names. Using the wrong string will raise a `ValueError`
   or download the wrong checkpoint.

3. **Should `head_midpoint` frames be weighted more heavily?**
   With 183 frames and only 16 sessions, `head_midpoint` has fewer clean examples
   than the 7 SA-matched bodyparts (the headstage is frequently occluded by the cable).
   Consider whether to add a per-keypoint loss weight in `pytorch_config.yaml` for
   `head_midpoint`. This is a DLC-internal config option; its availability and syntax
   should be verified from the DLC 3.x docs before attempting.

4. **What is the mAP from the most recent training run?**
   The architecture review and research notes quote mAP from earlier runs. Confirm the
   current production model's mAP from the evaluation CSV in `s3://hm2p-derivatives/dlc-retrain/`.

---

## References

Ye S, Filippova A, Lauer J, Schneider S, Vidal M, Qiu T, Mathis A, Mathis MW. 2024.
"SuperAnimal pretrained pose estimation models for behavioral analysis."
*Nature Communications* 15:5165. doi:10.1038/s41467-024-48792-2

DeepLabCut GitHub: https://github.com/DeepLabCut/DeepLabCut

PR #2756 "SuperAnimal Model Updates" (October 2024):
https://github.com/DeepLabCut/DeepLabCut/pull/2756

Wang J, Sun K, Cheng T, et al. 2020. "Deep High-Resolution Representation Learning for
Visual Recognition." *IEEE TPAMI*. (HRNet-w32 architecture)

SuperAnimal TopViewMouse training data (Zenodo): https://zenodo.org/records/10618947
