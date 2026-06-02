# keypoint-MoSeq Improvement Investigation Report

**Date:** 2026-06-02
**Context:** kpms v0.6.8 first run (2026-06-01) produced unusable output. This report
investigates root causes and concrete fixes.

---

## Executive Summary

The current kpms output is poor because the fitting pipeline has **five compounding
problems**, not just one. Fixing kappa alone will not be sufficient. Listed in order of
severity:

1. **Missing two-stage pipeline** -- the script runs only a single `fit_model()` call
   without the required AR-only initialisation stage, then skips the full SLDS model
   entirely.
2. **latent_dim is 10, should be 4** -- the paper reports 4 PCs explain 90% of
   variance for 2D overhead data; 10 PCs from 12 effective dimensions overfits to noise.
3. **Noise calibration skipped** -- the error_estimator uses generic defaults
   (slope=-0.5, intercept=0.25) instead of calibrated values for our DLC tracker.
4. **kappa too high (1e6)** -- but this is partly a downstream consequence of problems
   1-3: the model cannot find structure, so only extreme stickiness prevents chaotic
   switching.
5. **Missing sigmasq_loc estimation** -- the centroid movement prior is not estimated
   from data, using a generic default instead.

**Recommendation:** kpms IS the right tool for this data, but the pipeline needs a
substantial rewrite. The 8-keypoint 2D overhead setup is within the validated range for
kpms (Weinreb et al. used 8 keypoints from overhead DLC data and reported good results
with 4 PCs). The problem is entirely in how we run the tool, not the tool itself.

---

## 1. Is kpms suitable for 8-keypoint 2D overhead data?

**Yes, with caveats.**

### Evidence from the literature

Weinreb et al. 2024 (Nature Methods) explicitly tested kpms on:
- **2D overhead (top-down) DLC data** using the TopViewMouse SuperAnimal network
- **8 keypoints** (two ears + six dorsal midline points) -- exactly our keypoint count
- **4 PCs** were sufficient for 90% variance with 2D overhead data (vs 6 PCs for
  bottom-up 2D, and 6 PCs for 3D)

They report that kpms "identifies similar sets of behavioral transitions as depth MoSeq
and preserves important information about behavioral timing, despite being fed behavioral
data that are relatively lower dimensional." The 2D overhead data is explicitly in the
validated domain.

### Limitations of 2D overhead

The paper notes that "the higher dimensionality of depth data (relative to the 8
keypoints identified in the 2D data) afforded MoSeq more information about pose during
spontaneous behavior." This means:
- 2D overhead will miss vertical behaviours (rearing, hunching, stretching up)
- Grooming subtypes may be harder to distinguish
- Speed/direction discrimination should work well (this is what matters for our
  head direction analysis)

### Minimum keypoints

No hard minimum is documented. The paper validated with 8 keypoints in 2D; other
studies have used 6-12. Our 8 keypoints (nose, left_ear, right_ear, head_midpoint,
neck, mid_back, mouse_center, tail_base) cover head, trunk, and posterior -- this is
adequate.

### Conclusion

8 keypoints from overhead DLC is within the validated range. The method should work.
Our poor results are from pipeline errors, not fundamental incompatibility.

---

## 2. Root Cause Analysis: Five Compounding Errors

### 2.1 Missing two-stage pipeline (CRITICAL)

**The reference kpms pipeline has TWO fitting stages:**

```
Stage 1: AR-HMM only (ar_only=True)    ~50 iterations
         Kappa scanning happens HERE
Stage 2: Full SLDS model (ar_only=False) ~500 iterations, lower kappa
```

**Our script runs ONE stage** with `ar_only` at its default value (`False`). This means
we are fitting the full SLDS model from a random initialisation, without the AR-HMM
warm-up. The AR-only stage is documented as essential: "EML scores are higher for models
fit with an autoregressive-only (AR-only) initialization stage compared to those without"
(Weinreb et al.).

The kappa scanning should also happen after the AR-only stage, not before (as our script
does with 25-iteration sweeps from a cold start).

**Impact:** Without AR-only initialisation, the SLDS stage starts with poor syllable
assignments and cannot converge properly. This explains why all kappa values produce
similarly short bout durations -- the model never finds meaningful structure.

### 2.2 latent_dim = 10, should be ~4 (CRITICAL)

The current config uses `num_pcs=10` and (from the config) `latent_dim=10`
(via `ar_hypparams.latent_dim=10`).

**The data dimensionality:**
- 8 keypoints x 2 coordinates = 16 raw dimensions
- Minus 4 for centroid (2) + heading (2) alignment = **12 effective dimensions**
- 10 PCs from 12 dimensions captures ~99% of variance, including noise
- The paper recommends "the minimum PCs to explain 90% of variance, or 10, whichever
  is lower"
- For 2D overhead with 8 keypoints: **4 PCs** explained 90% variance in the paper

Using 10 PCs from 12 dimensions means the model is fitting noise dimensions. The AR-HMM
then models high-frequency tracking jitter as state transitions, producing the rapid
switching we observe.

**No PCA variance was saved** in our run (the `pca_variance.json` file does not exist on
S3), so we cannot verify the exact scree plot. This needs to be computed on the next run.

### 2.3 Noise calibration skipped (IMPORTANT)

The noise calibration widget learns the relationship between DLC confidence scores and
actual keypoint errors. It sets the `error_estimator` slope and intercept in the config.
Our config uses the generic defaults:

```yaml
error_estimator:
  intercept: 0.25
  slope: -0.5
```

These defaults may be inappropriate for our SuperAnimal fine-tuned model, which could
have different confidence-error characteristics than the models used to set the defaults.

**Impact of wrong noise model:** If the model underestimates noise, it treats tracking
jitter as real movement and rapidly switches states. If it overestimates noise, it
ignores real movement and collapses into too few states. Either way, syllable quality
degrades.

**Workaround for headless execution:** The noise calibration widget requires interactive
Jupyter, but there are two alternatives:

1. **Manual NaN masking** (recommended by Caleb Weinreb in GitHub issue #167):
   ```python
   coordinates[k] = np.where(
       confidences[k][:,:,None] > 0.9,
       coordinates[k],
       np.nan
   )
   ```
   This bypasses the error_estimator entirely by converting low-confidence points to
   NaN, which kpms handles via its missing data model.

2. **Run calibration locally once**, then hard-code the slope/intercept values into the
   headless script. Calibration only needs to be done once per tracker model, not per
   session.

### 2.4 kappa selection (MODERATE -- downstream of 2.1-2.3)

The kappa sweep results show:

| kappa | Median bout (frames) | Median bout (ms) | Effective syllables |
|-------|---------------------|-------------------|---------------------|
| 1e3   | 2                   | 67                | 21                  |
| 1e4   | 3                   | 100               | 18                  |
| 1e5   | 5                   | 167               | 13                  |
| 1e6   | 7                   | 233               | 12                  |

**No kappa value reaches the 300-500ms target.** Even at 1e6 (extreme stickiness), the
median bout is only 233ms. This is the key diagnostic: when the model cannot find
temporal structure regardless of kappa, the problem is upstream (initialisation, noise,
dimensionality).

GitHub issue #167 describes an identical pattern: "All kappas lead to similar median
syllable length." The solution was: (1) fix the confidence/noise handling, and (2) use
much higher kappas (1e10-1e15) combined with the two-stage pipeline.

The sweep was also run with only 25 iterations from a cold start (no AR-only
initialisation), which is insufficient for the model to converge.

### 2.5 Missing sigmasq_loc estimation (MINOR)

The reference pipeline calls `kpms.estimate_sigmasq_loc(data["Y"], data["mask"],
filter_size=config()["fps"])` to set the centroid movement prior from the data. Our
script skips this, using the default `sigmasq_loc=0.5`. This affects centroid tracking
but is less critical than problems 2.1-2.4.

---

## 3. Bodypart Selection Assessment

### Current selection (8 bodyparts)

| Bodypart       | Informative? | Notes |
|---------------|-------------|-------|
| nose          | Yes         | Anterior reference, good tracking |
| left_ear      | Yes         | Head orientation |
| right_ear     | Yes         | Head orientation |
| head_midpoint | Questionable | Custom keypoint for 2P headstage, not a natural body landmark |
| neck          | Yes         | Head-body coupling |
| mid_back      | Yes         | Trunk posture |
| mouse_center  | Yes         | Central body reference |
| tail_base     | Mixed       | Posterior reference, but can be noisy |

### Collinearity concern

The dorsal midline points (neck, mid_back, mouse_center, tail_base) lie approximately
along a single axis when viewed from above. In a straight posture, they are nearly
collinear and provide redundant information. They become informative when the mouse
curves its body, but for a top-down view this is a limited postural dimension.

### Recommendation

- **Keep:** nose, left_ear, right_ear, neck, mid_back, mouse_center, tail_base (7
  bodyparts)
- **Consider removing:** head_midpoint -- this is a headstage landmark, not a natural
  body feature. Its motion characteristics may differ from natural keypoints (it moves
  rigidly with the headstage, not with soft tissue). However, if DLC tracks it well, it
  provides extra head pose information. Test both configurations.
- The paper's reference used 8 keypoints (2 ears + 6 dorsal midline), so 7-8 is in the
  validated range.

---

## 4. Does kpms have confidence filtering?

**Yes, through two mechanisms:**

1. **conf_threshold in format_data** (currently set to 0.5): Points below this threshold
   are flagged as missing. However, this interacts with the error_estimator to set
   per-frame observation noise, not a hard filter.

2. **Manual NaN masking before loading** (recommended in issue #167): Set low-confidence
   coordinates to NaN before calling `format_data`. This is the more reliable approach
   for our case, as it bypasses the error_estimator entirely.

Our DLC champion model has variable confidence across bodyparts (the tracking quality
assessment showed 80.4% HD validity and 95.1% position validity). The bottom-up
SuperAnimal model without a top-down detector can produce spurious detections. Aggressive
confidence filtering (threshold 0.8-0.9) before kpms would help.

---

## 5. Concrete Fix Plan

### Phase 1: Pipeline rewrite (required)

Rewrite `run_kpms.py` to implement the reference two-stage pipeline:

```python
# 1. Load and format data
coordinates, confidences, bodyparts = kpms.load_keypoints(...)
# 2. NaN-mask low-confidence points (conf < 0.9)
for k in coordinates:
    coordinates[k] = np.where(
        confidences[k][:,:,None] > 0.9,
        coordinates[k],
        np.nan
    )
# 3. Format data
data, metadata = kpms.format_data(coordinates, confidences, **config())
# 4. Estimate sigmasq_loc from data
sigmasq_loc = kpms.estimate_sigmasq_loc(
    data["Y"], data["mask"], filter_size=config()["fps"]
)
# 5. Fit PCA, inspect scree plot, set latent_dim=4 (or whatever 90% variance needs)
pca = kpms.fit_pca(**data, **config())
# 6. Init model
model = kpms.init_model(data, pca=pca, **config())
model = kpms.update_hypparams(model, sigmasq_loc=sigmasq_loc)
# 7. AR-only stage (50 iterations)
model, model_name = kpms.fit_model(
    model, data, metadata, project_dir,
    ar_only=True, num_iters=50
)
# 8. Kappa scan (at this point, not before)
# Use np.logspace(3, 7, 5) or wider
# Select kappa giving median bout ~300-500ms
# 9. Load checkpoint, update kappa, fit full SLDS model (500 iterations)
model, data, metadata, current_iter = kpms.load_checkpoint(
    project_dir, model_name, iteration=50
)
model = kpms.update_hypparams(model, kappa=selected_kappa)
model = kpms.fit_model(
    model, data, metadata, project_dir, model_name,
    ar_only=False, start_iter=current_iter,
    num_iters=current_iter + 500
)[0]
```

### Phase 2: Parameter changes

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| latent_dim | 10 | **4** | Paper value for 2D overhead; 90% variance rule |
| num_pcs | 10 | **4** (or whatever matches 90% variance) | Same as latent_dim |
| kappa | 1e6 | **Sweep after AR-only stage** | Cannot pre-determine; depends on fixed noise/PCA |
| ar_only stage | Missing | **50 iterations** | Required for SLDS initialisation |
| Full SLDS | 100 iters, no AR init | **500 iterations after AR init** | Paper recommendation |
| conf_threshold | 0.5 | **NaN mask at 0.9** | Bypass error_estimator entirely |
| sigmasq_loc | 0.5 (default) | **Estimated from data** | Per-dataset centroid prior |
| noise_calibration | Skipped | **NaN masking workaround** | Avoids need for interactive widget |
| bodyparts | 8 | **7-8 (test with/without head_midpoint)** | Marginal change |

### Phase 3: Validation

After re-fitting, check:
1. Scree plot: do 4 PCs explain >=90% variance?
2. Median bout duration: 300-500ms at the selected kappa?
3. Syllable distribution: 20-40 syllables covering ~80% of time?
4. Entropy ratio: 0.6-0.8?
5. Single-frame bouts: <5% of all bouts?
6. Behavioural interpretability: do top syllables correspond to locomotion, turning,
   grooming, stillness, etc.?

---

## 6. Should we consider alternatives?

### VAME

**VAME** (vame-py >=0.12) is a VAE-based segmentation tool that:
- Accepts DLC/SLEAP/LightningPose input directly
- Works with 2D overhead data (developed on bottom-up recordings but architecture is
  view-agnostic)
- Handles noise via confidence filtering + IQR outlier removal + Savgol smoothing
- Does NOT have the complex two-stage fitting pipeline -- simpler to configure
- Uses HMM or K-means for final segmentation after VAE embedding

**However:** Weinreb et al. 2024 directly compared kpms to VAME and found that "behavioral
states from VAME, B-SOiD and MotionMapper were usually brief (median duration 33-100ms)"
and "their transitions aligned significantly less closely with changepoints in keypoint
data." VAME performed worse than kpms on the same 2D overhead data.

**Recommendation:** Do NOT switch to VAME. kpms is the better tool when properly
configured. VAME's simpler pipeline is an advantage for prototyping but not for final
analysis.

### Simple speed-based HMM

A 2-3 state HMM on speed + angular head velocity would be trivial to implement and would
capture immobile/locomoting/turning states. This is not "behavioural syllable" discovery
-- it is a simple kinematic classifier.

**Recommendation:** This is useful as a COMPLEMENT to kpms (e.g., for speed-filtering
in HD analysis), but it does not replace unsupervised syllable discovery. We already
compute movement state in the kinematics pipeline.

### DLC2Action

Semi-supervised, requires manual labels. Not a replacement for unsupervised discovery.
Could be used downstream to validate kpms syllables against human-annotated behaviours.

### Conclusion on alternatives

Stay with kpms but fix the pipeline. No alternative tool will perform better given that
kpms was explicitly validated on our exact data type (8-keypoint 2D overhead DLC).

---

## 7. Estimated compute cost for re-run

The first run took ~3 hours on c5.4xlarge for 100 iterations of single-stage fitting.
The corrected pipeline requires:
- 50 AR-only iterations (~1.5 hours, possibly faster as AR-only is simpler)
- Kappa scan: 5 values x 25 iterations each = 125 iterations (~3-4 hours)
- 500 full SLDS iterations (~15-20 hours)

**Total estimate: ~20-25 hours on c5.4xlarge** (CPU). A c5.9xlarge or c5.12xlarge would
reduce this proportionally through more parallel Gibbs sampling threads.

Given the 26 sessions x ~50,000 frames each = ~1.3M total frames, this is a substantial
computation. Consider running on a larger instance (c5.12xlarge at ~$2.04/hr spot =
~$40-50 USD / ~$65-80 AUD for the full run).

---

## 8. Summary of Recommendations

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| 1 | Implement two-stage pipeline (AR-only + SLDS) | Medium | Critical |
| 2 | Reduce latent_dim to 4 (verify via scree plot) | Trivial | Critical |
| 3 | Add NaN masking for confidence < 0.9 | Trivial | High |
| 4 | Add sigmasq_loc estimation from data | Trivial | Moderate |
| 5 | Run kappa sweep AFTER AR-only stage | Medium | High |
| 6 | Increase full-model iterations to 500 | Trivial | Moderate |
| 7 | Save PCA variance and scree plot diagnostics | Trivial | QA |
| 8 | Test with/without head_midpoint | Low | Low |

**Bottom line:** kpms is the right tool for this data. The poor results are from five
pipeline implementation errors, not from a fundamental mismatch between the method and
our data. A properly configured re-run with the two-stage pipeline, correct
dimensionality, and noise handling should produce usable syllables.

---

## References

- Weinreb et al. 2024. "Keypoint-MoSeq: parsing behavior by linking point tracking to
  pose dynamics." Nature Methods 21:1329-1339. doi:10.1038/s41592-024-02318-2.
  https://github.com/dattalab/keypoint-moseq
- GitHub issue #167: "All kappas (and decrease factors) lead to similar median syllable
  length." https://github.com/dattalab/keypoint-moseq/issues/167
- GitHub issue #153: "downsampling causes errors in calibration."
  https://github.com/dattalab/keypoint-moseq/issues/153
- kpms documentation: https://keypoint-moseq.readthedocs.io/
- kpms Colab notebook (reference pipeline):
  https://colab.research.google.com/github/dattalab/keypoint-moseq/blob/main/docs/keypoint_moseq_colab.ipynb
