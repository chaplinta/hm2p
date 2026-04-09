# HRNet-W32 vs ResNet-50: Failure Diagnosis

**Date:** 2026-04-02  
**Status:** Analysis — not yet acted on  
**Related docs:**
- `docs/superanimal-integration-plan.md` — Phase 1/2 fix plan
- `docs/superanimal-integration-notes-leaddev.md` — DLC API forensics

---

## Summary

HRNet-W32 achieved 34% mAP vs ResNet-50's 57%. The gap has multiple causes, ordered here
by estimated contribution. Random initialisation is the dominant factor, but three
secondary factors likely compound it.

---

## 1. Random initialisation — primary cause (~15–20 mAP points)

The DLC HRNet model template (`deeplabcut/modelzoo/model_configs/hrnet_w32.yaml`) sets
`pretrained: false` by default. Our training script never overrode this. ResNet-50 used
the DLC default, which is ImageNet pretrained via timm.

This is not subtle. Training a 29M-parameter network from random initialisation on 184
frames is a severely under-constrained problem. ImageNet pretraining provides:

- Low-level edge/texture features that are directly useful for locating body parts in
  overhead video
- A sensible initialisation of all weight distributions (avoids dead neurons, gradient
  flow is reasonable from epoch 1)
- Effective weight regularisation — the pretrained features act as a strong prior,
  reducing the number of useful weight configurations the optimiser must search

HRNet is harder to train from scratch than ResNet-50 for a given data size because it
has more free parameters in its parallel-resolution branches. With 184 frames, ImageNet
ResNet-50 will almost always beat randomly-initialised HRNet regardless of augmentation
or training duration.

**The learning stats confirm this:** RMSE was 33 px at epoch 10, only converging to
~14–16 px by epoch 300+, and mAP plateaued at 34% by epoch 150. The model was still
learning slowly from scratch at epoch 300. ImageNet initialisation would have reached
near-asymptote within the first 50 epochs.

**Fix:** Pass `pytorch_cfg_updates={"model.backbone.pretrained": True}` to
`deeplabcut.train_network()`. This is the entirety of Phase 1.

---

## 2. Aggressive augmentation on a small dataset — secondary cause (~3–8 mAP points)

HRNet-W32 operates on high-resolution feature maps throughout (unlike ResNet which
downsamples aggressively). This makes it more sensitive to large spatial distortions. The
augmentation used was:

- Rotation ±180° (vs ±30° for ResNet)
- Scale 0.25–2.5x (vs 0.5–1.25x)
- Brightness/contrast ±60% (not used for ResNet)
- hflip + vflip (not used for ResNet)
- Gaussian noise 30 (vs 12.75 for ResNet)

With 184 frames, a ±180° rotation means the model sees roughly every possible orientation
equally often. For an overhead view of a mouse in a rose maze, many of these orientations
are physically impossible or very rare. The augmented data distribution mismatches the
true data distribution significantly.

More concretely: with 184 frames and large-range augmentations, the probability that any
training step shows the network a "natural" example (near-canonical pose, normal
lighting) is low. The network never sees enough clean examples to anchor its
representations. This is especially harmful at random initialisation because the network
cannot rely on pretrained features to ignore the distortions.

**Note:** The original logic was sound — aggressive augmentation was intended to prevent
overfitting on 184 frames from scratch. With ImageNet pretraining this reasoning partially
holds, but ±180° rotation and 0.25–2.5x scale remain unusually aggressive for a
top-down single-animal tracker where the arena and camera are fixed. Mice in the rose
maze do not appear at 0.25x scale. After Phase 1, if mAP is still below 60%, try reducing
to ±45° rotation and 0.6–1.5x scale as a diagnostic step.

**Assessment:** This likely accounts for a few mAP points of the gap, not the majority.
It would have hurt ResNet-50 too if applied equally. The main effect of the asymmetric
augmentation is that comparisons between the two runs are confounded — we cannot isolate
the backbone effect cleanly.

---

## 3. Confidence calibration failure — tertiary cause / diagnostic signal

All HRNet confidences were < 0.4, and RMSE_pcutoff was either a single outlier (28.9 px
at epoch 10) or NaN for all other epochs. This is unusual. For a well-trained model,
RMSE_pcutoff should be lower than overall RMSE because filtering to high-confidence
predictions removes hard negatives.

Getting NaN for most epochs means almost no predictions cleared the pcutoff threshold —
confirming that confidence scores never crossed the evaluation threshold across most of
training. This is consistent with a model that has not learned to be selective: it assigns
uniformly low confidence because the heatmap peaks are broad and uncertain, which is
exactly what happens when training from random init on a small dataset with strong
augmentation.

HRNet architecture note: HRNet's prediction head produces heatmaps at higher spatial
resolution than ResNet (full-resolution feature maps rather than deconvolved maps from
downsampled features). High-resolution heatmaps should in principle be sharper and produce
better-calibrated confidences. But this only holds if the backbone features are already
good. From random init, high-resolution feature maps contain noise rather than pose
information, and the resulting heatmaps have diffuse peaks — exactly the pattern seen
here.

The confidence issue is not a separate architectural problem with HRNet. It is a symptom
of the same root cause: random initialisation plus aggressive augmentation prevented the
model from learning localised heatmap peaks.

**With ImageNet pretraining:** heatmap peaks should be sharper from early in training.
Confidence calibration should recover to normal levels.

---

## 4. HRNet head channel count — not a bottleneck

The question raises whether the 32-channel intermediate representation in HRNet-W32's
head is a bottleneck. It is not.

HRNet-W32 uses 32 channels per resolution branch in its backbone, but the prediction head
(HeatmapHead in DLC) operates on the fused multi-resolution feature map, which is
significantly wider. The `32` in `W32` refers to the smallest branch width; wider
branches have 64, 128, 256 channels. The final heatmap head takes the concatenated
high-resolution features as input.

ResNet-50's deconvolutional head is simpler and has fewer parameters than HRNet's
multi-scale head. For a small dataset, this actually means HRNet has a harder
optimisation problem, not an easier one. With ImageNet pretraining, the richer HRNet
head should be an advantage. Without it, it is a liability.

**Conclusion:** Channel count is not an independent bottleneck.

---

## 5. Would ImageNet-pretrained HRNet match or beat ResNet-50?

Expected yes, based on:

- HRNet-W32 has 29M parameters vs ResNet-50's 23M and uses multi-scale parallel feature
  learning. On pose estimation benchmarks it consistently outperforms ResNet-50 at equal
  data regimes (Wang et al. 2020, TPAMI).
- Ye et al. 2024 Table S3 shows HRNet-W32 with ImageNet weights outperforms ResNet-50
  equivalents on DLC-Openfield at all data fractions.
- The key condition is ImageNet pretraining, not architecture. ResNet-50 was at an
  advantage only because it was correctly pretrained.

Realistic expectation for Phase 1 (HRNet + ImageNet, 400 epochs, same 184 frames):
55–65% mAP. This recovers the ResNet-50 baseline and likely exceeds it slightly.

If Phase 1 falls below 55%, the augmentation asymmetry is the most likely additional
factor to address (see item 2 above).

---

## 6. Other DLC config issues

One confirmed issue beyond `pretrained: false`:

**`default_net_type: resnet_50` in `config.yaml` (fragile but not broken)**  
When `create_training_dataset` is called, it reads `default_net_type` to generate the
initial `pytorch_config.yaml`. Leaving it as `resnet_50` causes a ResNet config to be
generated, which the training script then partially overrides via YAML manipulation. This
is fragile — the generated ResNet config may have incorrect head channel assumptions that
the override incompletely patches. The Phase 1 fix should include changing this to
`default_net_type: hrnet_w32`.

No other config issues are confirmed as contributing to the 34% mAP result. The
augmentation parameters and learning rate are set in the training script and were applied
consistently for 400 epochs.

---

## Decision tree for Phase 1

```
Phase 1: HRNet + ImageNet (fix pretrained=True, change default_net_type)
    ├── mAP >= 65%  → proceed to Phase 2 (SA backbone) as incremental experiment
    ├── 55–65%      → working model; Phase 2 optional
    ├── < 55%       → diagnose augmentation (reduce rotation to ±45°, scale to 0.6–1.5x)
    └── < ResNet-50 → revert to ResNet-50 weights already on S3; record failure
```

---

## References

Wang J, Sun K, Cheng T, et al. 2020. "Deep High-Resolution Representation Learning for
Visual Recognition." IEEE TPAMI. (HRNet-W32 architecture)

Ye S, Filippova A, Lauer J, et al. 2024. "SuperAnimal pretrained pose estimation models
for behavioral analysis." Nature Communications 15:5165. doi:10.1038/s41467-024-48792-2
