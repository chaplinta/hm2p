# SuperAnimal Integration Notes — Lead Developer

**Author:** Lead Developer agent  
**Date:** 2026-04-02  
**Status:** Analysis complete — recommendation: stay with HRNet + ImageNet

---

## Context

We attempted to use SuperAnimal-TopViewMouse (SA-TVM) transfer learning to improve pose
estimation for our 5 body-part project (`left_ear`, `right_ear`, `mid_back`,
`mouse_center`, `tail_base`). Multiple approaches failed. This document explains why each
approach failed at the code level, identifies the correct API, and makes a risk-assessed
recommendation.

**Reference:**  
Ye et al. 2024. "SuperAnimal pretrained pose estimation models for behavioral analysis."
*Nature Communications* 15:5165. doi:10.1038/s41467-024-48792-2

---

## 1. The Correct DLC 3.0 Code Path for SuperAnimal Fine-Tuning

### 1.1 Two distinct use cases

DLC 3.0 implements SuperAnimal in two fundamentally different modes that are not
interchangeable:

**Mode A: Transfer learning (backbone only, no decoder weights)**  
Load SA backbone weights into a *new* project with custom bodyparts. The decoder (head) is
randomly initialised and trains from scratch. This gives a better starting point than
ImageNet but the head does not benefit from SA priors.

**Mode B: Fine-tuning (backbone + decoder, `with_decoder=True`)**  
Load both backbone and decoder weights. The decoder maps SA bodypart indices to project
bodypart indices via a `conversion_array`. The project must declare a `SuperAnimalConversionTables` entry in `config.yaml` before this mode works. Training uses
memory replay (optional) to prevent catastrophic forgetting of unused SA bodyparts.

### 1.2 The correct API sequence

**Step 1 — Build the weight init object (done outside `create_training_dataset`):**

```python
from deeplabcut.modelzoo.weight_initialization import build_weight_init

weight_init = build_weight_init(
    cfg=config_path,              # project config.yaml
    super_animal="superanimal_topviewmouse",
    model_name="hrnet_w32",       # must match SA checkpoint architecture
    detector_name="fasterrcnn_resnet50_fpn_v2",  # for top-down; None for bottom-up
    with_decoder=False,           # Mode A: backbone only
    memory_replay=False,
)
```

For Mode B (fine-tuning with decoder), you must first create the conversion table:

```python
import deeplabcut.modelzoo.utils as mzoo_utils

mzoo_utils.create_conversion_table(
    config_path,
    super_animal="superanimal_topviewmouse",
    # mapping from project bodypart → SA bodypart name
    conversion_mapping={
        "left_ear":    "left_ear",
        "right_ear":   "right_ear",
        "mid_back":    "mid_back",
        "mouse_center":"mouse_center",
        "tail_base":   "tail_base",
    },
)
weight_init = build_weight_init(
    cfg=config_path,
    super_animal="superanimal_topviewmouse",
    model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2",
    with_decoder=True,            # Mode B: encoder + decoder
    memory_replay=False,          # True = train all 27 SA bodyparts via pseudo-labels
)
```

**Step 2 — Create training dataset with weight_init passed in:**

```python
deeplabcut.create_training_dataset(config_path, weight_init=weight_init)
```

Internally, `create_training_dataset` branches:
- `weight_init.with_decoder=False` → calls `make_pytorch_pose_config()` (standard config,
  backbone type matches `net_type` in project config)
- `weight_init.with_decoder=True` → calls `make_super_animal_finetune_config()` (uses SA
  model config as base, raises `ValueError` if `with_decoder=False`)

The `weight_init` object is serialised into `pytorch_config.yaml` under:
```yaml
train_settings:
  weight_init:
    snapshot_path: /path/to/superanimal_topviewmouse_hrnet_w32.pt
    with_decoder: false
    memory_replay: false
    dataset: superanimal_topviewmouse
    conversion_array: null
    bodyparts: null
```

**Step 3 — Train:**

```python
deeplabcut.train_network(config_path, epochs=200)
```

`train_network` reads `train_settings.weight_init` from `pytorch_config.yaml` and
reconstructs the `WeightInitialization` object via `WeightInitialization.from_dict()`. No
`superanimal_name` or `superanimal_transfer_learning` parameters exist in the current
`train_network` signature. The official example script at
`examples/testscript_superanimal_transfer_learning.py` uses these deprecated parameter
names (from a pre-3.0 API).

### 1.3 Relevant file paths in the DLC source

| Component | File |
| --- | --- |
| `build_weight_init` | `deeplabcut/modelzoo/weight_initialization.py` |
| `WeightInitialization` dataclass | `deeplabcut/core/weight_init.py` |
| `create_training_dataset` dispatch | `deeplabcut/generate_training_dataset/trainingsetmanipulation.py` |
| `make_super_animal_finetune_config` | `deeplabcut/pose_estimation_pytorch/modelzoo/config.py` |
| `make_pytorch_pose_config` | `deeplabcut/pose_estimation_pytorch/config/make_pose_config.py` |
| `get_super_animal_snapshot_path` | `deeplabcut/pose_estimation_pytorch/modelzoo/utils.py` |
| `ConversionTable` | `deeplabcut/core/conversion_table.py` |
| Backbone weight loading | `deeplabcut/pose_estimation_pytorch/runners/train.py` — `load_snapshot()` |
| SA-TVM bodypart list (27 pts) | `deeplabcut/modelzoo/project_configs/superanimal_topviewmouse.yaml` |
| Topview conversion CSV | `deeplabcut/modelzoo/conversion_tables/conversion_table_topview.csv` |
| hrnet_w32 model config template | `deeplabcut/modelzoo/model_configs/hrnet_w32.yaml` |

### 1.4 How `conversion_array` works (Mode B only)

The `ConversionTable.to_array()` method returns an integer index array of length equal to
the number of project bodyparts. Each entry is the position of the matching SA bodypart in
the SA bodypart list (0-indexed, 27 entries for SA-TVM). For example, if our 5 bodyparts
map to SA indices [1, 2, 8, 9, 13], the conversion array is `[1, 2, 8, 9, 13]`.

The `conversion_array` is then used by the training runner to slice the SA head's output
channels — selecting output channels [1, 2, 8, 9, 13] from the 27-channel head to
initialise our 5-channel head. **This is the correct mechanism for Mode B head weight
transfer.** The runner uses `load_head_weights=True` with the conversion slicing, not
strict `load_state_dict` on the full checkpoint.

### 1.5 SA checkpoint naming convention

Checkpoint files follow the pattern: `{dataset}_{model_name}.pt`  
For our case: `superanimal_topviewmouse_hrnet_w32.pt`

Downloaded from HuggingFace via `dlclibrary`. Stored locally in the DLC cache.

### 1.6 Valid SA-TVM bodyparts (the 27-point superset)

From `superanimal_topviewmouse.yaml` and the conversion CSV:

```
nose, left_ear, right_ear, left_ear_tip, right_ear_tip, left_eye, right_eye,
neck, mid_back, mouse_center, mid_backend, mid_backend2, mid_backend3,
tail_base, tail1, tail2, tail3, tail4, tail5,
left_shoulder, left_midside, left_hip, right_shoulder, right_midside, right_hip,
tail_end, head_midpoint
```

Our 5 body parts (`left_ear`, `right_ear`, `mid_back`, `mouse_center`, `tail_base`) all
appear in this list. SA indices: left_ear=1, right_ear=2, mid_back=8, mouse_center=9,
tail_base=13 (0-indexed).

---

## 2. Why Our Previous Attempts Failed

### 2.1 Attempt: `superanimal_name` parameter in `create_training_dataset`

**What we tried:** Passing `superanimal_name="superanimal_topviewmouse"` directly to
`create_training_dataset()`.

**Why it failed:** This parameter does not exist in `create_training_dataset()`. The SA
integration is handled via the `weight_init` parameter, which must be a
`WeightInitialization` object built by `build_weight_init()`. The parameter
`superanimal_name` existed in an older pre-3.0 API and was removed.

The official example script (`testscript_superanimal_transfer_learning.py`) in the repo
still uses `superanimal_name` and `superanimal_transfer_learning=True` as arguments to
`train_network()`, but inspecting the current `train_network` signature shows neither
parameter exists. **The example script is stale and documents a removed API.**

### 2.2 Attempt: `build_weight_init()` downloads weights but DLC creates ResNet config

**What we tried:** Calling `build_weight_init()` then `create_training_dataset()` with
`net_type = "hrnet_w32"` in project `config.yaml`.

**Why it failed:** When `weight_init.with_decoder=False`, `create_training_dataset` calls
`make_pytorch_pose_config()` not `make_super_animal_finetune_config()`. The standard
`make_pytorch_pose_config()` builds the model config from the project's `net_type`
setting, which in our case was probably `resnet_50` (the DLC default). Even if `net_type`
was `hrnet_w32`, the weight_init snapshot path would be written correctly — but if the
project config still specified the default `net_type`, the wrong backbone was instantiated.

**Root cause:** `make_pytorch_pose_config()` reads `net_type` from the project
`config.yaml`, which defaults to `"resnet_50"`. The `model_name` parameter in
`build_weight_init()` does not automatically update `net_type` in the project config.
These two settings are not coupled automatically.

**Fix required:** The project `config.yaml` must have `default_net_type: hrnet_w32` (or
the equivalent for the DLC field name) before `create_training_dataset` is called.
Alternatively, `pytorch_cfg_updates` can override individual model config keys during
training, but the backbone type must already be correct in the config at dataset creation
time.

### 2.3 Attempt: Manually overriding backbone to HRNet causes state_dict mismatch

**What we tried:** After DLC generated a ResNet config, we manually edited
`pytorch_config.yaml` to change the backbone type to HRNet, then called
`train_network()`.

**Why it failed:** The model was instantiated as an HRNet using `timm.create_model()` with
`pretrained=False` (as per the hrnet_w32 modelzoo config template). The
`load_snapshot()` method then attempted to load the SA checkpoint — which also contains
HRNet weights — with `load_head_weights=True` by default.

HRNet backbone key names in the SA checkpoint have the prefix `backbone.`. The runner
strips this prefix and calls `model.backbone.load_state_dict(backbone_weights)`. This
worked. But if `load_head_weights=True`, the runner also calls
`model.load_state_dict(snapshot["model"])` with **strict=True by default**. The SA
checkpoint head has 27 output channels; our model head has 5 output channels. The
`state_dict` key shapes did not match → `RuntimeError`.

**Root cause:** DLC's `load_snapshot` in `train.py` uses strict loading by default. When
loading the full model state dict (backbone + head together), any shape mismatch in the
head channels causes a hard failure. The conversion array mechanism that would correctly
slice head channels only operates within `make_super_animal_finetune_config` + Mode B
(`with_decoder=True`). Manually editing configs bypasses this mechanism.

### 2.4 Attempt: `with_decoder=True` — head architecture mismatch

**What we tried:** Setting `with_decoder=True` in `build_weight_init()`.

**Why it failed (partially):** This correctly triggers `make_super_animal_finetune_config`.
However, this function raises a `ValueError` if called without `with_decoder=True`. If we
met this requirement, the function then loads the SA model config from
`deeplabcut/modelzoo/model_configs/hrnet_w32.yaml` as the base for `pytorch_config.yaml`.
The SA HRNet head config specifies `heatmap_channels: [32, "num_bodyparts"]` where
`num_bodyparts` is a template placeholder. The function substitutes `len(converted_bodyparts)` — which would be 5 for our project.

**The remaining failure:** `make_super_animal_finetune_config` also requires a valid
`SuperAnimalConversionTables` entry in `config.yaml`. If this table was not created first
via `create_conversion_table()`, `get_conversion_table()` raises `ValueError: No
conversion table found for superanimal_topviewmouse`.

Additionally, if the conversion table exists but the conversion_array slicing logic in the
runner is not triggered (because `weight_init` was not properly serialised into
`pytorch_config.yaml`), the full 27-channel head checkpoint will still be loaded with
strict=True, producing a shape mismatch.

### 2.5 Attempt: Setting `with_decoder=True` but DLC loads head strictly

**Root cause (confirmed from source):** In `runners/train.py`, `load_snapshot()`:

```python
if self._load_head_weights:
    model.load_state_dict(snapshot["model"])  # strict=True, default PyTorch behaviour
else:
    backbone_weights = {k[len("backbone."):]: v ...}
    model.backbone.load_state_dict(backbone_weights)
```

There is no intermediate path that loads the head with channel slicing at this layer. The
`conversion_array` is stored in `WeightInitialization` and used by
`make_super_animal_finetune_config` to configure the model architecture (setting
`num_bodyparts=5` so the head outputs 5 channels), but the runner still loads the *full*
checkpoint with `model.load_state_dict(snapshot["model"])`. This works only if the model
being loaded was also trained with `num_bodyparts=5`, which the SA checkpoint was not.

**This is the fundamental problem:** The SA checkpoint head has shape `(27, 32, 1, 1)`
(27 output channels). Our fine-tuned model head has shape `(5, 32, 1, 1)`. DLC does not
implement channel slicing during `load_state_dict`. It assumes `with_decoder=True` means
the checkpoint and model have matching architecture — which is only true for projects that
were originally created *from* a pretrained SA project (not retrofitted).

---

## 3. The Correct Approach

### 3.1 What actually works

The SA fine-tuning pathway (`with_decoder=True`, Mode B) is designed for projects created
with `deeplabcut.create_pretrained_project()` — a convenience function that:

1. Creates a new DLC project starting from the SA config (27 bodyparts).
2. Downloads SA weights as the initial checkpoint.
3. Sets `resume_training_from` in `pytorch_config.yaml` pointing to the SA checkpoint.

In this workflow, the model head is initialised with 27 channels matching the SA
checkpoint. Fine-tuning then trains that 27-channel head on the user's labeled data (where
only 5 bodyparts have ground truth; the other 22 are filled with pseudo-labels from SA
zero-shot predictions if `memory_replay=True`). The output model still predicts 27
bodyparts — the user simply uses only the 5 they labeled.

**This is not how we set up our project.** We created a standard DLC project with 5
bodyparts. The SA integration pathway assumes the project was created from SA.

### 3.2 Option A: Retrofit — use backbone-only transfer (Mode A) correctly

This requires:

1. In project `config.yaml`, set `default_net_type: hrnet_w32` (if this field name is
   correct — verify with `deeplabcut.utils.auxiliaryfunctions.read_config()`).
2. Call `build_weight_init(with_decoder=False, ...)` — backbone only.
3. Call `create_training_dataset(weight_init=weight_init)`.
4. Call `train_network(config_path, load_head_weights=False)` — pass
   `load_head_weights=False` to prevent the runner from attempting to load the head from
   the SA checkpoint.

The `load_head_weights=False` is the critical parameter. With this flag, `load_snapshot`
only loads backbone weights (filtered by `backbone.` prefix), which have matching shapes
regardless of head size. The head initialises randomly as usual.

**Expected benefit from the paper:** Mode A (transfer learning, backbone only) on SA-TVM
HRNet with 1% of training data: mAP 96.6 vs 91.5 for ImageNet baseline (Table S3). With
our ~70–100 labeled frames, we are well into the low-data regime where SA backbone
transfer is most beneficial.

### 3.3 Option B: Rebuild project from SA (Mode B, full fine-tuning)

This would require:

1. Creating a new DLC project with `create_pretrained_project()` using
   `superanimal_topviewmouse` as the base model. This gives a 27-bodypart project.
2. Importing our existing labeled frames and re-labeling against the 27-bodypart
   SA schema (or running zero-shot SA inference as pseudo-labels for unlabeled bodyparts,
   then manually correcting the 5 we care about).
3. Training with memory replay. The model will output all 27 bodyparts; we use only 5.

**Estimated cost:** Re-labeling or pseudo-labeling all frames is significant work (~4–8
hours for 26 sessions of data). The downstream pipeline (`movement` library) would need to
extract only our 5 bodyparts from a 27-bodypart output. This is not supported by the
current `movement.io.load_poses` call without post-processing.

**Not recommended** for the current project phase. Revisit if mAP remains below 50%.

### 3.4 Option C: Train with ResNet-50 (current working baseline)

Our ResNet-50 baseline (previously) achieved mAP ~57%. The current HRNet-W32 with
aggressive augmentation achieves ~34%. The HRNet regression is likely due to:

- HRNet using `timm` with `pretrained=False` → random initialisation, no ImageNet weights.
- Aggressive augmentation may be too strong for small datasets with HRNet.
- HRNet requires more data than ResNet to converge from random init.

**Check:** Confirm that the HRNet config has `pretrained: false` vs `true`. In
`deeplabcut/modelzoo/model_configs/hrnet_w32.yaml`, the template sets `pretrained: false`.
This means we are training HRNet from random initialisation — not from ImageNet. This is
the main reason for mAP ~34%.

**Immediate fix (Option C):** Switch to `pretrained: true` in the HRNet backbone config by
passing via `pytorch_cfg_updates`:

```python
deeplabcut.train_network(
    config_path,
    epochs=400,
    pytorch_cfg_updates={
        "model.backbone.pretrained": True
    },
)
```

This is the simplest fix and should recover performance comparable to ResNet-50 or better,
without any SA complexity.

---

## 4. Whether to Modify DLC Source Code

**Short answer: No, not recommended.**

The head channel mismatch between SA (27 channels) and our project (5 channels) is a real
design gap in DLC 3.0. The existing code does not implement the channel-slicing path in
`load_snapshot`. Adding this would require:

1. Modifying `PoseTrainingRunner.load_snapshot()` to accept an optional
   `conversion_array` parameter and slice `snapshot["model"]` head keys accordingly.
2. Passing `conversion_array` from `WeightInitialization` through
   `build_training_runner()` to the runner.
3. Handling key name differences between checkpoint head keys and model head keys (these
   vary between HRNet heatmap head and ResNet deconv head).

**Risks:**
- DLC updates frequently. A fork that patches the runner will diverge from upstream and
  require maintenance.
- The slicing logic must correctly identify head vs backbone keys in the checkpoint. Key
  naming conventions are not documented and may change between DLC minor versions.
- Testing would require a full SA fine-tuning integration test, which takes hours to run.

**If we proceeded anyway**, the patch would be approximately:

```python
# In PoseTrainingRunner.load_snapshot(), after loading snapshot:
if weight_init is not None and weight_init.with_decoder and weight_init.conversion_array is not None:
    state_dict = snapshot["model"]
    # Remap head output channels
    for key in head_keys:  # identify by key prefix, e.g., "head."
        if state_dict[key].shape[0] == len(sa_bodyparts):
            state_dict[key] = state_dict[key][weight_init.conversion_array]
    model.load_state_dict(state_dict, strict=False)
else:
    model.load_state_dict(state_dict, strict=True)
```

This is fragile because `head_keys` must be identified dynamically and differs between
model architectures. **Not worth it for a 5-bodypart project.**

---

## 5. Risk Assessment: SA Transfer vs ImageNet HRNet

### Current state

| Config | mAP | Issue |
| --- | --- | --- |
| ResNet-50, ImageNet | ~57% | Working baseline |
| HRNet-W32, `pretrained=False` | ~34% | Not using ImageNet weights |
| HRNet-W32, SA backbone (Mode A) | not tested | Requires correct setup per §3.2 |

### Expected outcomes

| Approach | Expected mAP | Effort | Risk |
| --- | --- | --- | --- |
| HRNet-W32, `pretrained=True` (ImageNet) | 55–65% | 30 min config fix | Low |
| HRNet-W32 + SA backbone (Mode A, correct) | 65–75%+ | 2–3 hrs | Medium |
| ResNet-50 + SA backbone (Mode A) | Not meaningful | N/A | SA TVM only has HRNet checkpoint |
| Full SA fine-tune (Mode B, rebuild project) | 75–85%+ | 8–16 hrs | High |
| Patch DLC source for channel slicing | Unpredictable | 4–8 hrs + testing | Very high |

From Ye et al. 2024 Table S3 (HRNet-W32, DLC-Openfield, similar bodypart count):
- ImageNet transfer with 1% data: mAP 91.5%
- SA fine-tuning with 1% data: mAP 98.8%

Note: those benchmark numbers are on an in-distribution dataset where all 27 SA bodyparts
are labeled. Our OOD setting (rose maze, novel camera, 5 of 27 bodyparts) is harder.
Realistic gains are smaller, perhaps 5–15 mAP points improvement from SA backbone over
ImageNet.

### Recommendation

**Priority 1 (immediate):** Fix the HRNet ImageNet pretrain bug. Pass
`pytorch_cfg_updates={"model.backbone.pretrained": True}` in `train_network` and re-run.
This alone should recover to ~55–65% mAP.

**Priority 2 (if Priority 1 does not reach 60% mAP):** Implement SA backbone-only
transfer (Mode A, §3.2) correctly:
- Set `default_net_type: hrnet_w32` in `config.yaml`
- Build weight_init with `with_decoder=False`
- Pass `load_head_weights=False` to `train_network`

**Do not pursue:** Mode B (full fine-tune), DLC source modification, or rebuilding the
project from SA. These require disproportionate effort for the likely gain.

**Lower bound check:** If HRNet with ImageNet weights underperforms ResNet-50 (which can
happen — HRNet is not always better for small datasets), revert to ResNet-50. Both
architectures are supported by the SA backbone pathway if we later want SA transfer.

---

## 6. Summary Table of Failure Modes

| Attempt | Failure mechanism | Code location |
| --- | --- | --- |
| `superanimal_name` param in `create_training_dataset` | Parameter does not exist (removed pre-3.0 API) | `trainingsetmanipulation.py` |
| `superanimal_name` in `train_network` | Parameter does not exist in current signature | `apis/training.py` |
| `build_weight_init()` + default ResNet net_type | `make_pytorch_pose_config` uses project `net_type` (ResNet), not SA model_name | `make_pose_config.py` |
| Manual backbone override → state_dict mismatch | `load_snapshot` calls `model.load_state_dict` with strict=True; head channels 27≠5 | `runners/train.py:load_snapshot` |
| `with_decoder=True` without conversion table | `get_conversion_table()` raises ValueError | `modelzoo/utils.py` |
| `with_decoder=True` + conversion table but no SA project | `make_super_animal_finetune_config` still loads 27-channel head, runner loads strictly | `runners/train.py` + `modelzoo/config.py` |

---

## 7. References

Ye et al. 2024. "SuperAnimal pretrained pose estimation models for behavioral analysis."
*Nature Communications* 15:5165. doi:10.1038/s41467-024-48792-2

DLC source (inspected 2026-04-02, main branch):
- `deeplabcut/modelzoo/weight_initialization.py` — `build_weight_init`
- `deeplabcut/core/weight_init.py` — `WeightInitialization` dataclass
- `deeplabcut/pose_estimation_pytorch/runners/train.py` — `load_snapshot`
- `deeplabcut/pose_estimation_pytorch/modelzoo/config.py` — `make_super_animal_finetune_config`
- `deeplabcut/generate_training_dataset/trainingsetmanipulation.py` — `create_training_dataset`
- `deeplabcut/modelzoo/project_configs/superanimal_topviewmouse.yaml` — 27-bodypart list
- `deeplabcut/modelzoo/conversion_tables/conversion_table_topview.csv` — bodypart name mapping
- `examples/testscript_superanimal_transfer_learning.py` — stale example (uses removed API)
