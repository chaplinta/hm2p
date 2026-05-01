# SuperAnimal Fine-Tune Plan v2

**Date:** 2026-04-30
**Branch:** `feat/sync-pipeline-diagnostics`
**Status:** Plan only — no code changes pending
**Supersedes:** `docs/superanimal-integration-plan.md` (v1, retained as record)
**Authors:** Neuro Data Scientist agent, after re-reading Ye et al. 2024 in full

> Cite as: Ye S, Filippova A, Lauer J, Schneider S, Vidal M, Qiu T, Mathis A, Mathis MW.
> 2024. "SuperAnimal pretrained pose estimation models for behavioral analysis."
> *Nature Communications* 15:5165. doi:10.1038/s41467-024-48792-2.
> Code: https://github.com/DeepLabCut/DeepLabCut. Weights:
> https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse.

---

## 1. Why v1 needs replacing

Champion on `main` is `dlc-20260430-hrnetw32-snap110`: HRNet-W32 + ImageNet, trained
400 epochs on 354 frames across 26 sessions, 80/20 split. Test RMSE_pcutoff 3.77 px,
mAP ~65.6 %. Per-keypoint medians:

- `nose_tip` ~24 px, PCK@10 = 17 %.
- `tail_base` ~59 px, PCK@10 = 24 %.
- `head_midpoint` long-tailed in dark / cable-occluded sessions.
- ears, neck, mid_back, mouse_center near ~5 px.

v1 deferred SuperAnimal on the assumption that "more labels will fix nose and tail".
Re-reading Ye 2024 makes the deferral wrong on four grounds:

1. **All eight hm2p keypoints map identity-to-identity onto SA-TVM.** `head_midpoint`
   is SA-TVM keypoint #26 (`superanimal_topviewmouse.yaml`). The conversion table in
   `config.yaml` is already complete (and correct — confirmed by lead-dev §7).
2. The two broken keypoints are exactly the ones SA-TVM is built to help: small
   high-curvature features (nose) and ambiguous boundary features (tail_base). The
   ImageNet backbone has no animal-pose prior; the SA-TVM HRNet has been trained
   on TopViewMouse-5K which itself includes TriMouse (multi-mouse, occluded tails)
   and MausHaus (head-mounted-cable mice). The relevant priors already exist in the
   released weights.
3. Ye Table S3 (HRNet-W32, DLC-Openfield, in-distribution): at 1 % data
   (≈ 10 frames), memory-replay SA fine-tune reaches 2.38 px vs ImageNet 7.00 px.
   At 100 % data (≈ 1,000 frames), the gap collapses to 1.21 vs 1.13 px. We are at
   354 frames in OOD conditions — closer to the small-data side of the curve where
   SA helps most.
4. SA fine-tuning is run on a parallel shuffle (snap-110 stays as shuffle 1). There
   is no downside risk: if the SA shuffle fails the gate (§4.6), keep snap-110.

---

## 2. SuperAnimal in one page (only the parts we need)

- **Pre-training.** SA-TVM is HRNet-W32 (top-down) and DLCRNet (bottom-up) trained
  panoptically over 13 merged top-view mouse datasets (~5 K frames). Two
  algorithmic contributions matter for fine-tuning:
  - *Keypoint gradient masking* (Methods, Eq. 1–3): mask the loss at undefined
    keypoints so missing GT does not penalise. Critical for pre-training; we do not
    invoke it directly because our data is uniformly labelled.
  - *Memory replay* (Methods §"Memory replay fine tuning"; pseudocode in main
    text): use the SA model's own zero-shot predictions (confidence > 0.7) as
    pseudo-labels for the SA channels we did not label. Prevents catastrophic
    forgetting of those channels — but more importantly, lets the encoder + decoder
    keep learning from the full 27-channel signal while we train only on the 8
    channels we care about.
- **What's released and what we use.** The HuggingFace SA-TVM HRNet-W32 weights
  (`superanimal_topviewmouse_hrnet_w32.pt`) plus a Faster R-CNN bbox detector
  (`fasterrcnn_resnet50_fpn[_v2]`). Top-down means the detector crops the mouse
  before the pose head sees it, which sidesteps the spatial-pyramid search the
  paper recommends for bottom-up DLCRNet. For us, top-down HRNet-W32 + memory
  replay is the canonical recipe.
- **Paper benchmarks on TopViewMouse fine-tuning (HRNet-W32, Tables S3–S4).**
  All numbers are RMSE (px) on in-distribution test splits. SA refers to memory
  replay; ImageNet is transfer learning.

  | Data | DLC-Openfield ImageNet | DLC-Openfield SA | TriMouse ImageNet | TriMouse SA |
  | --- | --- | --- | --- | --- |
  | 1 %  | 7.00 | 2.38 | 31.56 | 5.85 |
  | 5 %  | 2.16 | 1.95 | 6.93  | 4.19 |
  | 10 % | 1.57 | 1.54 | 4.21  | 2.86 |
  | 100 %| 1.13 | 1.21 | 2.28  | 2.10 |

  Note the SA advantage is largest in the low-data and high-occlusion regimes
  (TriMouse). Our nose/tail problem maps onto the second column most cleanly.

- **Inference hooks the paper offers.**
  - *Spatial-pyramid search* (bottom-up only): irrelevant.
  - *Unsupervised video adaptation* (1 K iters, batch 1, BN frozen, conf > 0.5
    pseudo-labels): reduces jitter, not absolute RMSE on labelled frames. Optional
    post-process; not part of v2.

---

## 3. v1 plan review against the paper

| v1 statement | Verdict | v2 correction |
| --- | --- | --- |
| `head_midpoint` has no SA-TVM equivalent (Phase 1 commentary, lines 39, 472) | Wrong; lead-dev §7 already corrected mid-document, but Phase 2 narrative still hedges. | All 8 keypoints map identity-to-identity. `conversion_array = [0,1,2,26,7,8,9,13]`, no `-1` slots. |
| HRNet-W32 + ImageNet "is adequate for all bodyparts except nose and tail" | Misframes the problem. Adding labels does not produce a pose prior. | Move SA fine-tuning from Phase 2 to the next training run. |
| Phase 2 chooses Mode A (`with_decoder=False`) | Suboptimal. Picked because lead-dev §2.5 incorrectly concluded head channel slicing did not work. §7 reversed that: `HeatmapHead.convert_weights` slices the SA 27-channel head to our 8-channel head when `with_decoder=True`. | Use `with_decoder=True, memory_replay=True`. |
| `pytorch_cfg_updates={"model.backbone.pretrained": True}` on the SA path | Wrong. The SA checkpoint *is* the pretrained source; this kwarg can clobber it depending on load order. | Drop `pretrained=True` on the SA shuffle; keep it on the ImageNet shuffle. |
| `load_head_weights=False` for SA fine-tuning | Correct only for Mode A. For memory replay we want the SA head as warm start; conversion array handles slicing at model build, not at snapshot load. | Drop. |
| 400 epochs from random init | Too long once we warm-start. Paper trained SA-TVM HRNet itself for 210 epochs over 5 K frames; memory-replay fine-tune was 70 K iters (~60 epochs over 5 K frames) at batch 8. | 100–150 epochs, save every 10 — best-snap selection chooses the winner. |
| Augmentation table (rot ±45°, scale 0.7–1.4, etc.) | Mostly fine, somewhat too aggressive vs paper's HRNet recipe (random flip + half-body transform + random scale rotation). | Reduce rotation to ±30°, scale to 0.7–1.3; keep brightness/contrast (IR-camera-specific, not in paper). |
| `detector_name="fasterrcnn_resnet50_fpn_v2"` open question | Unresolved — v2 carries the same uncertainty. | Step 0 in §5 verifies via `dlclibrary.list_available_detectors()`. |
| rc13 → 3.0.0 stable warning | Still relevant; assume 3.0.0+ on EC2 (memory `project_superanimal_dlc.md` confirms). | No change. |

---

## 4. v2 plan

### 4.1 Goal

Train one parallel SA-TVM memory-replay HRNet-W32 shuffle on the same 354 labelled
frames. Promote it to champion only if it passes a non-parametric paired gate on
held-out test frames.

### 4.2 Exact API call sequence (DLC ≥ 3.0.0 stable)

Cite: Ye 2024 Fig. 1d + Methods §"Memory replay". DLC source:
`deeplabcut.modelzoo.weight_initialization.build_weight_init`,
`deeplabcut.create_training_dataset`, `deeplabcut.train_network`,
`deeplabcut/pose_estimation_pytorch/models/heads/simple_head.py:HeatmapHead.convert_weights`.

```python
import deeplabcut
from deeplabcut.modelzoo.weight_initialization import build_weight_init

config_path = "/tmp/dlc-retrain/config.yaml"   # downloaded from S3 in run_dlc_retrain.py

# Pre-conditions (assert before proceeding):
#  - default_net_type: hrnet_w32 in config.yaml.
#  - SuperAnimalConversionTables.superanimal_topviewmouse covers all 8 bodyparts
#    identity-to-identity (already true on main as of 2026-04-30).
#  - dlclibrary.list_available_models() lists 'superanimal_topviewmouse_hrnet_w32'.
#  - dlclibrary.list_available_detectors() lists the detector name passed below.

weight_init = build_weight_init(
    cfg=config_path,
    super_animal="superanimal_topviewmouse",
    model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2",   # verify via dlclibrary listing
    with_decoder=True,                              # required for memory replay
    memory_replay=True,                             # paper Methods §"Memory replay"
)

new_shuffles = deeplabcut.create_training_dataset(
    config_path,
    weight_init=weight_init,
    num_shuffles=1,
    net_type="hrnet_w32",                # explicit override of config.yaml default
)
sa_shuffle = new_shuffles[-1]            # shuffle 2 if shuffle 1 is snap-110

deeplabcut.train_network(
    config_path,
    shuffle=sa_shuffle,
    epochs=120,                          # see §4.3
    save_epochs=10,
    displayiters=100,
    batch_size=8,
    pytorch_cfg_updates={
        "train_settings.optimizer.params.lr": 5e-5,    # Methods §"HRNet-w32" small-data
        "model.backbone.freeze_bn_stats": True,        # Methods §"HRNet-w32" small-data
    },
)
```

What v2 deliberately does **not** pass:

- `load_head_weights=False` — we want the SA head as warm start; channel slicing
  happens at model build via `HeatmapHead.convert_weights`, not at snapshot load.
- `model.backbone.pretrained=True` — the SA checkpoint already carries the
  trained backbone.

### 4.3 Training schedule (calibrated to the paper)

| Item | Champion (snap-110) | v2 SA shuffle | Source |
| --- | --- | --- | --- |
| Backbone init | HRNet-W32 + ImageNet | HRNet-W32 + SA-TVM | Ye 2024 Fig. 1c–d |
| Decoder init | Random | SA-TVM 27-ch head sliced to 8 ch via `convert_weights` | Lead-dev §7.2 |
| Memory replay | n/a | True (conf threshold 0.7) | Methods, pseudocode |
| Optimiser | Adam, lr 1e-4 | Adam, lr 5e-5 | Methods §"HRNet-w32" small-data |
| Epochs | 400 | 120 | Paper used ~70 K iters memory-replay over 5 K frames; we scale by frames not epochs to match the iteration budget |
| LR step decay | — | epoch 90, 110 (×0.1) | Paper steps at 81 % / 95 % of budget |
| Save every | 10 epochs | 10 epochs | Best-snap selection |
| Batch size | 8 | 8 | Same VRAM budget on g4dn.xlarge |
| BN running stats | Trainable | Frozen | Methods §"HRNet-w32" small-data |
| Augmentation (rot / scale) | ±45° / 0.7–1.4 | ±30° / 0.7–1.3 | Paper's HRNet protocol is milder |
| Augmentation (other) | Flip H+V, motion blur, brightness ±15 %, contrast ±10 %, gauss noise 15 | Same, gauss noise 10 | IR-camera brightness/contrast patch is project-specific; keep |

### 4.4 Script changes

Three files. No new pipeline stage.

**`scripts/run_dlc_retrain.py` — modify `train()`**

1. Add `--sa-finetune` argparse flag (default off; existing path unchanged).
2. When `--sa-finetune`:
   - Skip the manual `pytorch_config.yaml` rewrite block (lines 119–193). Backbone,
     head channels, and weight init are now handled by `create_training_dataset`.
   - Insert the `build_weight_init(...)` call from §4.2 before
     `create_training_dataset`.
   - Pass `weight_init=weight_init` and `net_type="hrnet_w32"` to
     `create_training_dataset`; record the returned shuffle index.
   - Call `train_network` with the §4.2 kwargs.
   - Apply the §4.3 augmentation block to the new shuffle's `pytorch_config.yaml`
     (the augmentation patch is the only YAML edit that survives §4.4 step 2; the
     backbone block is gone).
3. Inference, evaluation, S3 upload, and champion-declare flow are unchanged. The
   `.h5` filename will encode `Hrnetw32` and the SA snapshot index, and the
   existing `extract_dlc_provenance` parser will pick that up; the new champion
   ID will look like `dlc-20260430-hrnetw32sa-snap60` (subject to actual best
   snapshot).

**`scripts/launch_dlc_finetune_ec2.py` — propagate the flag**

1. Add `--sa-finetune` to argparse.
2. In `build_user_data`, when set, append `--sa-finetune` to `mode_flag` and tag
   the launch as `mode="sa-finetune"` for cost records.

A separate `scripts/launch_sa_finetune_ec2.py` is **not** needed. The existing
launcher with one extra flag is the simplest cut.

**`scripts/compare_models.py` — new file** (lead-dev work item)

Inputs: champion ID and candidate ID. Loads the test split's `CollectedData_*.h5`
labels, runs the candidate model on those frames, loads the existing champion's
predictions on the same frames from S3, and emits the §4.5 statistics + §4.6
verdict as JSON.

### 4.5 Validation — non-parametric only (per CLAUDE.md)

Hold-out: re-use the existing 20 % test split. Per-frame errors are paired across
models because the frames are identical.

For each test frame *f* and keypoint *k*:
- `e_old(f,k)` = Euclidean distance, snap-110 prediction vs GT.
- `e_new(f,k)` = same for the SA shuffle.

**Primary test (per keypoint, paired):** Wilcoxon signed-rank,
`scipy.stats.wilcoxon(e_old[:,k], e_new[:,k], alternative="greater")`. Reject H0
→ SA error is lower for keypoint *k*. Bonferroni-correct across the 8 keypoints
(α/8 = 6.25 × 10⁻³). Effect size: matched-pair rank-biserial *r* (Kerby 2014).

NOT Mann-Whitney U (errors are paired by frame), NOT a t-test (distributions are
heavy-tailed for nose/tail).

**Descriptive per-keypoint:** Median per-frame error (px); PCK@k for k ∈ {5, 10,
20} px; bootstrapped 95 % CI on the median (10 K resamples, percentile method).

**HD-relevant downstream metric:** Compute the ear-vector head-direction angle on
test frames using each model. Per-frame absolute circular error |θ_pred − θ_gt|
wrapped to ±π. Compare paired errors with Wilcoxon signed-rank. This is the
metric that matters for the science.

### 4.6 Champion-promotion gate

The SA shuffle becomes the project-wide champion only if **all** of:

1. **`nose_tip`:** median per-frame error decreases by ≥ 30 % vs snap-110, with
   Wilcoxon `p < 6.25 × 10⁻³` (Bonferroni for 8 tests) and rank-biserial *r* > 0.3
   (medium-or-larger).
2. **`tail_base`:** same statistical thresholds, ≥ 40 % median reduction.
3. **No regression > 10 % in median error on any other keypoint**, and any
   regression that exists is non-significant (p > 0.05).
4. **`head_midpoint`:** 90th-percentile per-frame error decreases by ≥ 20 % (the
   long-tail compression is the headline metric for this keypoint).
5. **HD-angle test:** median absolute HD error decreases (no statistical
   threshold; descriptive only — the science is the constraint).
6. Visual QC on the rendered labelled video for one held-out dark-condition
   session passes the existing manual gate.

If the gate fails, log the result and keep snap-110. SA fine-tuning has zero
effect on the live pipeline until promotion is run manually.

### 4.7 Expected gains (predictions, not measurements)

#### `nose_tip`
- Current: median 24 px, PCK@10 = 17 %.
- Paper basis (Table S3, HRNet-W32 DLC-Openfield): at 1 % data, ImageNet 7.0 px
  vs SA 2.4 px (~3.0× reduction). At 5 % data, ImageNet 2.16 vs SA 1.95
  (~1.1× reduction). Relative gain shrinks fast with data.
- Our regime: 354 frames is in the high-data tail of TopViewMouse-5K but in the
  low-data regime for OOD generalisation — the headstage occludes the snout in
  ~30 % of frames, which is not represented in TopViewMouse-5K.
- **Expected: median drops to 8–12 px (40–65 % reduction); PCK@10 rises to 50–70 %.**
  Reasoning: most of the current failure mode is mis-anchoring (model "snaps"
  the nose to a non-nose feature under occlusion); the SA backbone's animal-pose
  prior should let the head extrapolate the nose much like the OOD examples in
  Fig. 3b–e. If the residual error is dominated by label noise the floor is
  closer to 12 px.

#### `tail_base`
- Current: median 59 px, PCK@10 = 24 %.
- Paper basis (Table S4, TriMouse, HRNet-W32, 1 % data): ImageNet 31.6 px vs SA
  5.85 px (~5.4× reduction). TriMouse has 3 mice with occluding tails — closest
  paper analog to our cable-occluded tail.
- **Expected: median drops to 18–30 px (50–70 % reduction); PCK@10 rises to
  45–60 %.** This is the single biggest scientific motivation for SA-TVM.
  TopViewMouse-5K *includes* TriMouse, so the SA-TVM head has seen many tails.

#### `head_midpoint`
- Current: ~good in light, long-tailed in dark/cable-occluded sessions.
- Paper basis: SA-TVM has `head_midpoint` (index 26). MausHaus
  (head-cable-mounted-camera mice) is part of TopViewMouse-5K, so SA has seen
  cable-occlusion at this exact keypoint.
- **Expected: moderate improvement.** Median similar; 90th-percentile long tail
  compresses by 20–40 %. Hard to predict numerically (paper does not separately
  benchmark this keypoint).

#### Other 5 keypoints
- Current: ~5 px median, well-behaved.
- **Expected: flat to mild improvement (0–15 % reduction).** Paper Table S3 at
  100 % DLC-Openfield in-distribution shows ImageNet 1.13 px vs SA 1.21 px — the
  rare regime where SA can be marginally worse. On OOD data SA should at worst
  tie ImageNet for these well-behaved keypoints.

---

## 5. Pitfalls flagged after re-reading the paper

Not in v1.

1. **Image resolution mismatch.** The SA-TVM HRNet was trained at 256×256
   crops. `make_super_animal_finetune_config` should write the right resolution,
   but the *crop* dimension is what matters in top-down inference. Verify the
   resolution on the SA shuffle's `pytorch_config.yaml` (`data.train.input_size`)
   and the Faster R-CNN inference resolution before promotion. A mismatch will
   look like spurious degradation.
2. **80/20 vs 95/5 split.** Paper's tables are computed on 95/5; ours is 80/20.
   Don't compare absolute RMSE to the paper — compare *relative gains* (SA vs
   ImageNet on the same hm2p holdout, which is what §4.5 does).
3. **Memory replay computes loss over 27 channels per frame.** Backward pass is
   ~3× the cost of naive fine-tune. Wall-clock for 120 epochs ≈ 50 min on
   g4dn.xlarge (vs ≈ 90 min for 400-epoch ImageNet) — net cheaper, but the
   per-iteration cost is higher. Keep batch_size=8 (no headroom on g4dn.xlarge
   for batch 16 with memory replay). USD ~0.50 (~AUD 0.75) per attempt at spot
   prices.
4. **The SA detector is multi-animal-trained.** TopViewMouse-5K includes
   TriMouse. Top-down pose returns nothing if the detector returns no bbox.
   Single-animal hm2p inference assumes 1 bbox/frame; verify the detector
   returns ≥ 1 box on a few dark sessions before launching the full 26-session
   inference. Workaround if it fails: re-train just the detector on hm2p frames
   (much smaller change than re-training the pose model).
5. **DLC issue #2742.** `video_inference_superanimal(..., video_adapt=True)`
   fails on folder paths. v2 does not call this API; flagged for any future
   iteration that adds video adaptation.
6. **`testscript_superanimal_transfer_learning.py` in DLC repo is stale.** Uses
   pre-3.0 `superanimal_name=` / `superanimal_transfer_learning=` kwargs that
   no longer exist on `train_network`. Do not copy from it.
7. **`memory_replay=True` requires the conversion table to cover every project
   bodypart.** Ours does. If a bodypart is added later without an SA-TVM match,
   set its conversion-table value to `null` (yields `-1` in the conversion array,
   which `convert_weights` zero-initialises). Audit the table before changing it.
8. **The "10–100× more data efficient" headline does not apply at 354 frames.**
   It is a low-data-regime claim. Expect 1–3× gains on the broken keypoints, not
   10×.

---

## 6. Operating procedure

```bash
# Step 0 — sanity (run once on EC2 before any full launch):
uv run python -c "
import dlclibrary
print(dlclibrary.list_available_models())     # confirm 'superanimal_topviewmouse_hrnet_w32'
print(dlclibrary.list_available_detectors())   # confirm detector_name to use in §4.2
"

# Step 1 — config.yaml must have default_net_type: hrnet_w32 and the conversion table.
# Both already true on main as of 2026-04-30. No edit needed.

# Step 2 — upload labels (no change to existing script).
uv run python scripts/upload_dlc_labels.py

# Step 3 — launch SA fine-tune.
uv run python scripts/launch_dlc_finetune_ec2.py --sa-finetune --epochs 120

# Step 4 — after pose-finetuned/ lands on S3, run the comparison locally:
uv run python scripts/compare_models.py --base snap-110 --candidate <new>

# Step 5 — if §4.6 gate passes, the EC2 run will already have auto-declared the
#   new champion. If it fails, run promote_dlc_model.py with --snapshot 110 to
#   restore snap-110 as champion and keep the SA shuffle on S3 for diagnosis.
```

---

## 7. Open questions for the architect

1. **Detector name.** `fasterrcnn_resnet50_fpn_v2` vs `fasterrcnn_resnet50_fpn`.
   Resolved by Step 0 above. If wrong, `build_weight_init` raises before training
   starts — failure is loud and fast.
2. **Half-body transform.** DLC 3.x's albumentations bridge may not expose it.
   If unavailable, drop it; flips + scale + rotation are sufficient.
3. **Parallel A/B inference.** The frontend champion-staleness machinery already
   shows the older champion until a new one is declared, so a side-by-side run
   is not technically required. Architect to decide whether to pre-render two
   labelled videos for one held-out session before promotion as a manual QC.

---

## 8. Citation (for code, docs, frontend per CLAUDE.md)

In code docstrings:

```python
"""SuperAnimal-TopViewMouse memory-replay fine-tuning.

Method: Ye et al. 2024. "SuperAnimal pretrained pose estimation models for
behavioral analysis." Nature Communications 15:5165.
doi:10.1038/s41467-024-48792-2.
Code: https://github.com/DeepLabCut/DeepLabCut.
Weights: https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse.

Memory-replay protocol: Ye 2024 Methods §"Memory replay fine tuning" + Fig. 1d.
Conversion-array channel slicing: HeatmapHead.convert_weights in
deeplabcut/pose_estimation_pytorch/models/heads/simple_head.py.
"""
```

Frontend (Methods & References expander on the DLC tracking-quality page):

> Ye S, Filippova A, Lauer J, Schneider S, Vidal M, Qiu T, Mathis A, Mathis MW.
> 2024. "SuperAnimal pretrained pose estimation models for behavioral analysis."
> *Nature Communications* 15:5165.
> doi:[10.1038/s41467-024-48792-2](https://doi.org/10.1038/s41467-024-48792-2).
> Pre-trained weights: SuperAnimal-TopViewMouse (HuggingFace). Memory-replay
> fine-tuning per Methods §"Memory replay fine tuning".
