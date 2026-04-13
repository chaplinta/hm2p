# SuperAnimal Integration Plan — Data Scientist Review

**Reviewer:** Data Scientist (neuro-data-scientist agent)  
**Date:** 2026-04-02  
**Documents reviewed:**
- `docs/superanimal-integration-plan.md` (3-phase plan)
- `docs/superanimal-integration-notes-datascientist.md` (paper analysis)
- `docs/superanimal-integration-notes-leaddev.md` (DLC API forensics)
- `docs/hrnet-diagnosis.md` (HRNet failure analysis)
- `papers/biorxiv-scans/biorxiv-scan-2026-04-02.md` (literature context)
- `docs/architecture-review-dlc-pipeline.md` (pipeline issues)

---

## Overall Assessment

The plan is well-structured, the diagnosis of the HRNet failure is correct, and the
phased approach is sound. Phase 1 (fix the `pretrained: false` bug) is the right
immediate action. However, there are several gaps and one significant disagreement
between the research notes that needs resolution before Phase 2 proceeds. I address
each of the six questions below, then list additional gaps and risks.

---

## Question 1: Is Phase 1 (53.8% mAP with HRNet) disappointing?

**The 53.8% number is from HRNet with random initialisation** — i.e., the broken
configuration. Phase 1 has not been run yet. The plan proposes to fix `pretrained: false`
to `pretrained: true`, which is expected to bring HRNet-W32 to 55-65% mAP.

The comparison to make is:
- ResNet-50 + ImageNet: **57% mAP, 9.4 px RMSE** (working baseline)
- HRNet-W32 + random init: **34% mAP** (broken — per lead dev notes; the 53.8% figure
  may be from a partially-fixed run or a different evaluation set)
- HRNet-W32 + ImageNet (Phase 1 prediction): **55-65% mAP**

The 34% mAP from random init is completely expected. Training a 29M-parameter network
from scratch on 184 frames is a textbook underconstrained optimisation problem. The
hrnet-diagnosis.md correctly identifies this as the dominant factor (~15-20 mAP points),
with aggressive augmentation as a secondary contributor (~3-8 points).

**Should we stick with ResNet-50?** Not yet. The plan's decision rule is correct: run
Phase 1 with ImageNet-pretrained HRNet-W32 first. HRNet-W32 consistently outperforms
ResNet-50 on pose benchmarks at equal data regimes when both are pretrained (Wang et al.
2020; Ye et al. 2024 Table S3). If Phase 1 HRNet + ImageNet falls below 55% mAP, then
revert to ResNet-50. The existing ResNet-50 weights on S3 provide a safe fallback.

**One concern:** the augmentation settings between the ResNet-50 and HRNet-W32 runs were
different (HRNet used much more aggressive augmentation — rotation +/-180 vs +/-30,
scale 0.25-2.5x vs 0.5-1.25x). This confounds the backbone comparison. For a clean
Phase 1 evaluation, I recommend running HRNet + ImageNet with **both** augmentation
settings (aggressive and moderate) as separate shuffles. This isolates the backbone
effect from the augmentation effect. The moderate augmentation run takes the same GPU
time and removes a confound from the interpretation.

---

## Question 2: Is Phase 2 (SA backbone transfer) likely to help given rc13 bugs?

Phase 2 should not be attempted on rc13. The plan already states this (Step 1 of
Phase 2: "Must be >= 3.0.0, not rc13"). The rc13-specific failure modes documented in
the data scientist's notes (TF-to-PyTorch checkpoint format mismatch, stale API
parameters, training instability) are real and were fixed in subsequent releases,
particularly PR #2756 (October 2024).

**Given a stable DLC release**, Phase 2 (Mode A, backbone-only transfer) is likely to
provide a meaningful improvement. The expected gain from the paper is 5-15 mAP points
over ImageNet in our data regime, though this is an in-distribution benchmark number.
Our out-of-distribution setting (headstage occlusion, dark epochs, rose maze geometry)
will reduce the absolute gain but not eliminate it. The SA-TVM backbone has seen overhead
mouse poses from 13 different labs, which provides genuinely useful pose-aware features
that ImageNet lacks.

**The key risk is not performance but implementation:** the API surface for SA transfer
in DLC 3.x is complex, the example scripts are stale (as the lead dev confirmed), and
the exact parameter names may differ between 3.0.0 and the latest stable. I recommend
verifying the API against the installed DLC version's source before writing any training
code.

---

## Question 3: Should we upgrade DLC to stable release first?

**Yes, unambiguously.** This should happen before Phase 1, not before Phase 2.

Reasons:
1. rc13 has documented training instability (GitHub #2702).
2. PR #2756 fixed SuperAnimal weight initialisation — even Mode A benefits from
   correct `WeightInitialization` serialisation into `pytorch_config.yaml`.
3. The `pretrained: false` default in the HRNet template may have been fixed in the
   stable release (worth checking — if so, Phase 1 becomes a no-op for the backbone
   config).
4. Running Phase 1 on rc13 and then upgrading for Phase 2 introduces a version
   confound. If Phase 1 numbers look marginal, we cannot tell whether it is the
   backbone, the augmentation, or rc13 instability.

**Recommendation:** Upgrade to the latest DLC stable release as the first action, before
any training runs. Then run Phase 1. This eliminates one variable from all subsequent
experiments.

The plan's Phase 1 launch procedure should be amended to include the DLC upgrade step.

---

## Question 4: Mode A vs Mode B disagreement

The two research notes disagree on which mode to recommend:

- **Data scientist notes** recommend Mode B (full fine-tuning, `with_decoder=True`,
  memory replay) as the ultimate goal, with Mode A as the fallback. Priority 3 in their
  action items is memory replay fine-tuning.
- **Lead developer notes** initially recommend Mode A only (Section 3.2) and explicitly
  say "Do not pursue Mode B" (Section 5), but then the Section 7 addendum reverses
  this after discovering that `convert_weights` in `HeatmapHead` does implement
  channel slicing correctly.
- **The integration plan** recommends Mode A for Phase 2, with Mode B deferred to a
  hypothetical Phase 3 (which it explicitly labels "not recommended as primary
  strategy" — but that Phase 3 is video adaptation, not Mode B; Mode B is never given
  its own phase).

**Who is correct?**

Both are partially correct, and the plan has a gap. Here is the resolution:

**Mode A** (backbone-only, `with_decoder=False`) is the lower-risk, faster path. The
head is randomly initialised and trains on our 8 bodyparts. No conversion table
complications. Expected gain: backbone features are pose-aware instead of
ImageNet-generic. Implementation: 2-3 hours.

**Mode B** (`with_decoder=True`, memory replay) is the higher-ceiling path. The lead
developer's Section 7.2 confirmed that `convert_weights` does handle the 27-to-8
channel remapping correctly, including zero-initialisation for `head_midpoint` at
index -1. The data scientist's notes correctly identify this as the path that achieves
10x data efficiency in the paper. The lead developer's initial "do not pursue" was
based on an incomplete source analysis that Section 7 corrected.

**My recommendation:**

1. Phase 1: HRNet + ImageNet (as planned).
2. Phase 2: Mode A (backbone-only, as planned).
3. **Phase 2b (new):** Mode B (`with_decoder=True`, memory replay), as a separate
   shuffle (shuffle=3). Run only if Mode A achieves at least 60% mAP (confirming the
   SA checkpoint loads correctly). This is an incremental experiment, not a project
   rebuild.

Mode B requires:
- Adding `head_midpoint: null` to the `SuperAnimalConversionTables` in
  `config.yaml` (the plan already notes this).
- `build_weight_init(with_decoder=True, memory_replay=True)`.
- The conversion array `[0, 1, 2, -1, 7, 8, 9, 13]` (already documented).

Mode B does **not** require rebuilding the project from `create_pretrained_project()`.
The lead developer's Section 7.4 confirms this. The model outputs 8 channels (our
bodyparts), not 27, because `make_super_animal_finetune_config` sets
`num_keypoints=8` and `convert_weights` remaps the SA decoder. This is less disruptive
than the lead developer initially feared.

The plan should add a Phase 2b for Mode B. Currently Mode B falls into a gap: it is
discussed in the notes but has no implementation plan, no acceptance criteria, and no
shuffle strategy.

---

## Question 5: Realistic mAP ceiling with 184 frames

The paper's in-distribution benchmarks at comparable label counts show:

| Frames | ImageNet mAP | SA memory replay mAP |
|--------|-------------|---------------------|
| ~10    | 91.5        | 99.6                |
| ~100   | 99.3        | 99.9                |
| ~1000  | 100.0       | 99.9                |

These are in-distribution numbers on DLC-Openfield (clean arena, no headstage, no
darkness, all 27 bodyparts labeled). Our out-of-distribution factors:

1. **Headstage and cable** — partially occlude the dorsal body surface. The SA-TVM
   training data includes no headstage. This is a hard domain shift for mid_back, neck,
   and especially head_midpoint.
2. **Total darkness** — zero visual information in ~50% of frames. SA-TVM was trained
   entirely under normal lighting. Dark frames are genuinely OOD for both ImageNet and
   SA backbones; only our labeled dark frames provide supervision for this domain.
3. **Rose maze geometry** — walls, arms, and central hub create partial occlusions and
   constrain movement. SA-TVM's training data is mostly open-field.
4. **Camera and arena variation** — different camera height, lens, and arena size from
   the TopViewMouse-5K labs.
5. **Only 8 of 27 bodyparts labeled** — Mode B with memory replay mitigates this via
   pseudo-labels, but Mode A does not.

**Realistic ceiling estimates for our setup:**

| Configuration | Estimated mAP | Confidence |
|---|---|---|
| ResNet-50 + ImageNet (current) | 57% (measured) | Known |
| HRNet-W32 + ImageNet (Phase 1) | 55-65% | High |
| HRNet-W32 + SA backbone, Mode A | 62-72% | Medium |
| HRNet-W32 + SA full, Mode B + memory replay | 68-78% | Medium-low |
| Any method + 300 labeled frames (incl. dark) | 72-82% | Medium |
| Mode B + 300 frames | 75-85% | Medium-low |

The target of >70% mAP and <7 px RMSE is achievable but not guaranteed with 184
frames alone. It will likely require either Mode B or additional labeled frames (or
both). The 85%+ range requires more labels, particularly from dark epochs and
difficult headstage-occluded poses.

**Key insight from the data:** at 184 frames, we are past the regime where SA transfer
provides dramatic gains (that regime is <50 frames). The paper shows ImageNet and SA
converge by ~100 frames on in-distribution data. Our out-of-distribution setting
extends the useful range of SA transfer somewhat, but **labeling more frames has a
higher expected return than architectural improvements at this point** (see Question 6).

---

## Question 6: Would more labeled frames help more than SA transfer?

**Yes, with one important caveat about which frames.**

The data scientist's notes correctly identify dark-frame tracking as a domain gap that
no amount of SA transfer can close — SA-TVM was trained entirely under normal lighting.
Additional labeled frames from dark epochs would directly address the hardest tracking
condition.

**Recommendation:**

1. Run Phase 1 first (HRNet + ImageNet, 30 min fix). This establishes the baseline.
2. Run Phase 2 Mode A (SA backbone, 2-3 hrs). This tests the SA benefit.
3. **In parallel with Phase 2:** label 40-60 additional frames, targeted as follows:
   - 20-30 frames from dark epochs (total darkness, mouse actively moving)
   - 10-15 frames with severe headstage/cable occlusion (any light condition)
   - 5-10 frames at high angular velocity (fast head turns — these produce the
     most tracking errors for HD computation)
   
   This brings the total to ~230-240 frames. The targeted selection matters more than
   the count — random frame selection would add frames from easy conditions that are
   already well-represented.

4. Retrain with the expanded label set + SA backbone. Compare to Phase 1 and Phase 2.

**Cost-benefit:** Labeling 40-60 targeted frames takes approximately 2-4 hours of
manual work. This is comparable to the implementation time for Phase 2 Mode B. But
labeled data compounds with every method — it improves ImageNet training, SA Mode A,
SA Mode B, and any future method. SA transfer is a one-time architectural gain. When
in doubt, label more data.

**The exception:** if the primary bottleneck is not overall mAP but specifically the
confidence dropout rate for ears during fast head turns in darkness, then labeling
more dark-epoch frames is the only solution. SA transfer cannot help with a condition
that is absent from the SA training distribution.

---

## Additional Gaps and Risks

### 1. Missing: augmentation asymmetry control

The hrnet-diagnosis.md identifies that ResNet-50 and HRNet-W32 used different
augmentation settings. The plan does not address this. Phase 1 should include a
controlled comparison: run HRNet + ImageNet with both the aggressive augmentation
(current settings) and the moderate augmentation (ResNet-50's settings). Without this,
we cannot attribute any mAP difference to the backbone vs the augmentation.

### 2. Missing: frame count discrepancy

The plan says 183 frames; the data scientist's notes say 184. The difference is one
frame, which is immaterial for training, but it suggests one document counted differently
(possibly including or excluding a problematic frame). Confirm from
`CollectedData_tristan.h5` which is correct and use that number consistently.

### 3. Missing: Phase 2b (Mode B) implementation plan

As discussed in Question 4. The plan jumps from Mode A (Phase 2) to video adaptation
(Phase 3). Mode B should be Phase 2b with its own implementation steps, acceptance
criteria, and shuffle index.

### 4. Missing: `detector_name` parameter disambiguation

The plan uses `"fasterrcnn_resnet50_fpn_v2"` (the lead developer's value). The data
scientist's notes use `detector_name=None` for single-animal top-down, with the
rationale that no separate detector is needed for single-animal videos. These are
contradictory.

For SA-TVM HRNet (top-down architecture): a detector IS required to crop the animal
before pose estimation. The DLC top-down pipeline uses Faster R-CNN internally. The
question is whether `detector_name` must be explicitly set in `build_weight_init` or
whether DLC infers it from the SA model config.

**Resolution needed:** Check the `build_weight_init` source for the installed DLC
version. If `detector_name=None` causes a fallback to the default SA-TVM detector,
then `None` is correct and simpler. If it raises an error, use the explicit string.
The lead developer's `"fasterrcnn_resnet50_fpn_v2"` vs `"fasterrcnn_resnet50_fpn"` (v1
vs v2) should be verified against the checkpoint filename on HuggingFace.

### 5. Missing: evaluation on held-out dark-epoch frames

The plan's acceptance criteria (Section: Testing Strategy) mention reviewing a labeled
video for one lights-off session, but do not specify a quantitative evaluation split
for dark frames specifically. Given that dark-epoch tracking is our hardest condition
and the one most relevant to the science (visual cue removal), the evaluation should
report per-bodypart RMSE separately for light-on and light-off frames. This would
reveal whether SA transfer specifically helps in darkness or only in well-lit
conditions.

### 6. Risk: `load_head_weights` parameter may not exist in stable DLC

The plan's Open Question 1 flags this correctly. The parameter `load_head_weights=False`
is identified from source inspection of `runners/train.py` in an unspecified DLC
version. If this parameter does not exist in the installed stable release, Phase 2
Mode A has no mechanism to prevent the runner from attempting to load the 27-channel SA
head into the 8-channel model head.

**Mitigation:** Before Phase 2, inspect `deeplabcut.train_network.__doc__` and
`deeplabcut.pose_estimation_pytorch.runners.train.PoseTrainingRunner.load_snapshot`
in the installed version. If `load_head_weights` does not exist, Mode A may require a
different approach (possibly `pytorch_cfg_updates` to null out the snapshot path for
the head, or using `with_decoder=True` + conversion table instead — i.e., Mode B
becomes the only viable SA path).

### 7. Risk: learning rate for small datasets

The data scientist's notes mention that DLC automatically reduces the learning rate
to 5e-5 and freezes BN running stats for datasets with <64 unique images. Our dataset
has 184 frames, which is above this threshold — so the standard lr of 5e-4 will be
used. With a pretrained backbone (either ImageNet or SA), 5e-4 may be too high for the
early training epochs, causing the pretrained features to be destroyed before the head
learns. Standard fine-tuning practice uses a lower lr for the backbone (1e-5 to 5e-5)
and a higher lr for the randomly-initialised head (1e-4 to 5e-4).

DLC does not expose per-layer learning rate control through `pytorch_cfg_updates`
(as far as the notes indicate). If Phase 1 shows mAP instability or degradation during
training, consider reducing the global lr to 1e-4 or 5e-5 via `pytorch_cfg_updates`.

### 8. Missing: bodypart-specific evaluation for HD computation

The plan's acceptance criteria require `left_ear` and `right_ear` RMSE <= 5 px. This
is the right focus for HD computation, but the threshold should be justified from the
HD error propagation. The data scientist's notes provide this calculation: at inter-ear
distance D ~ 60 px, RMSE 5 px gives HD angular error ~ arctan(10/60) ~ 9.5 degrees.
At a bin width of 10-12 degrees, this is borderline. A stricter target of RMSE <= 3 px
(HD error ~ 5.7 degrees, comfortably below bin width) would be more appropriate for HD
cell analysis. If 3 px is not achievable, the HD bin width should be increased to
compensate for the tracking noise.

### 9. Risk: pipeline invalidation cost

The plan correctly notes that re-running Stage 2 invalidates Stages 3, 3b, 5, and 6
for all 26 sessions. Each Phase 1/2/2b experiment that produces a new model and runs
inference triggers a full downstream re-run. With three planned experiments (Phase 1,
Mode A, Mode B) plus potentially a fourth (expanded labels), this is four full pipeline
re-runs.

This is acceptable given the importance of tracking quality for all downstream science,
but it should be budgeted in the EC2 cost estimate. The plan does not include a
compute cost estimate.

### 10. Lead developer notes: 5 vs 8 bodyparts

The lead developer's notes were written with 5 bodyparts (Sections 1-6), then corrected
to 8 in the Section 7 addendum. While the addendum resolves the discrepancy, the earlier
sections contain incorrect conversion arrays (length 5 instead of 8) and incorrect head
channel counts (5 instead of 8). These should not be used directly — only the Section 7
values (`conversion_array: [0, 1, 2, -1, 7, 8, 9, 13]`) are correct.

---

## Recommended Execution Order

1. **Upgrade DLC** to latest stable release. Verify version. Inspect
   `train_network` signature for `load_head_weights` parameter.

2. **Phase 1:** HRNet-W32 + ImageNet. Two shuffles: one with current aggressive
   augmentation, one with moderate augmentation (ResNet-50 settings). Evaluate both.
   Accept if mAP >= 57% (matching ResNet-50 baseline).

3. **Label 40-60 additional frames** (targeted: dark epochs, headstage occlusion,
   fast turns). This runs in parallel with Phase 1 GPU training.

4. **Phase 2 (Mode A):** SA backbone-only transfer. New shuffle. Compare to Phase 1
   best. Accept if mAP improves by >= 3 points over Phase 1.

5. **Phase 2b (Mode B):** SA full fine-tuning with memory replay and conversion table.
   New shuffle. Compare to Mode A. Accept if mAP improves further.

6. **Retrain best model with expanded labels** (184 + 40-60 new frames). This is the
   production model candidate.

7. **Evaluate per-condition:** report RMSE separately for light-on vs light-off frames,
   and separately for ears vs other bodyparts. The production model decision should
   weight ear tracking in darkness most heavily, because this is the condition that
   matters most for the science (HD tuning in visual cue removal).

---

## Summary Table

| Question | Answer |
|---|---|
| Is 53.8% mAP disappointing? | No — it is from the broken config (random init). Phase 1 has not run yet. |
| Stick with ResNet-50? | Not yet. Run Phase 1 with HRNet + ImageNet first. Revert if < 55%. |
| Will Phase 2 help despite rc13 bugs? | Yes, if we upgrade DLC first. rc13 must not be used for SA transfer. |
| Upgrade DLC first? | Yes, before Phase 1, not just before Phase 2. |
| Mode A vs Mode B? | Both are valid. Mode A first (lower risk), then Mode B (higher ceiling). Add Phase 2b. |
| mAP ceiling with 184 frames? | 68-78% with Mode B; 75-85% with Mode B + 50 more targeted labels. |
| More labels vs SA transfer? | More labels, targeted at dark/occluded/fast-turn conditions, has higher expected return. Do both. |

---

## References

Ye S, Filippova A, Lauer J, Schneider S, Vidal M, Qiu T, Mathis A, Mathis MW. 2024.
"SuperAnimal pretrained pose estimation models for behavioral analysis."
*Nature Communications* 15:5165. doi:10.1038/s41467-024-48792-2

Wang J, Sun K, Cheng T, et al. 2020. "Deep High-Resolution Representation Learning for
Visual Recognition." *IEEE TPAMI*. doi:10.1109/TPAMI.2020.2983686
