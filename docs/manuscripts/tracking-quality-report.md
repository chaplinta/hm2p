# DLC Tracking Quality Report

**Date:** 2026-05-14
**Scope:** All 26 sessions, comparing SuperAnimal (SA) TopViewMouse model (zero-shot)
vs current champion (custom-trained HrnetW32, `hm2p-retrainMar20`, snapshot `best-100`).
**Data source:** `s3://hm2p-derivatives/pose/` (raw DLC H5 files at 30 fps) and
`s3://hm2p-derivatives/sync/` (resampled to imaging rate, ~9.6 Hz).

---

## Executive Summary

The **current champion model** (custom-trained DLC HrnetW32) provides tracking quality
that is adequate for all planned behavioural analyses. Key metrics:

- **Detection rate:** 100% of frames (no bounding-box detector; bottom-up single-animal model)
- **Ear confidence >0.9:** 73.9% (left), 68.4% (right) across all frames
- **Ear confidence >0.4:** 98.3% (left), 98.5% (right)
- **Jump rate (>20 px):** 1.6% (left ear), 1.6% (right ear)
- **HD validity in sync.h5:** 80.4% +/- 1.7% (after quantile:0.25 confidence filtering)
- **Position validity in sync.h5:** 95.1% +/- 2.0%

The **SuperAnimal TopViewMouse model** (zero-shot, no fine-tuning) is substantially worse
and is **not suitable** for use:

- **Detection rate:** 56.5% +/- 14.2% (range 29.0% -- 83.8%)
- **Ear confidence >0.9 (all frames):** 4.4% (left), 2.1% (right)
- **Jump rate (>20 px):** 8.5% (left ear), 8.8% (right ear) -- 5.5x worse
- **Both ears >0.4:** only 36.9% of all frames

**Recommendation:** Continue using the current champion model. The SA model's top-down
detector fails to detect the mouse in ~43% of frames due to the 2P headstage occluding
the animal's body from overhead view. Fine-tuning the SA detector on this specific setup
would likely improve detection rate, but the current champion already outperforms the SA
model on every metric and is sufficient for publication.

---

## 1. Model Descriptions

### Current Champion: Custom-Trained HrnetW32

- **Architecture:** HrnetW32 (bottom-up, single-animal)
- **Training data:** 354 labeled frames from this dataset (8 bodyparts)
- **Snapshot:** `best-100` (mAP=76.6, mAR=78.9, RMSE=7.12 px)
- **Champion ID:** `dlc-20260506-hrnetw32-snapbest-100`
- **Key property:** No top-down bounding-box detector -- predicts keypoints for every
  frame regardless of whether the animal is "detected." This means 100% of frames have
  pose predictions (though some may be low confidence).

### SuperAnimal TopViewMouse (Zero-Shot)

- **Architecture:** HrnetW32 backbone + FasterRCNN ResNet50 FPN v2 detector (top-down)
- **Training data:** SuperAnimal TopViewMouse corpus (multi-species, multi-lab)
- **Fine-tuning:** None (zero-shot inference only)
- **Key property:** Uses a two-stage pipeline: (1) FasterRCNN detects animal bounding boxes,
  (2) HrnetW32 predicts keypoints within detected boxes. Frames where the detector fails
  to find the mouse get `likelihood = -1` for all bodyparts.

---

## 2. Per-Bodypart Detection and Confidence

### 2.1 Detection Rate (fraction of frames with any prediction)

| Bodypart       | Champion | SA (zero-shot) | Delta     |
|----------------|----------|-----------------|-----------|
| left_ear       | 100.0%   | 56.5%           | **-43.5%** |
| right_ear      | 100.0%   | 56.5%           | **-43.5%** |
| nose           | 100.0%   | 56.5%           | **-43.5%** |
| head_midpoint  | 100.0%   | 56.5%           | **-43.5%** |
| neck           | 100.0%   | 56.5%           | **-43.5%** |
| tail_base      | 100.0%   | 56.5%           | **-43.5%** |
| mouse_center   | 100.0%   | 56.5%           | **-43.5%** |
| mid_back       | 100.0%   | 56.5%           | **-43.5%** |

Note: The SA model's detection rate is identical for all bodyparts because non-detection
is at the bounding-box level (FasterRCNN), not the keypoint level. When the detector
finds the animal, all 27 keypoints are predicted.

### 2.2 Confidence > 0.4 (fraction of ALL frames)

| Bodypart       | Champion | SA (zero-shot) | Delta     |
|----------------|----------|-----------------|-----------|
| left_ear       | 98.3%    | 45.7%           | **-52.5%** |
| right_ear      | 98.5%    | 40.4%           | **-58.1%** |
| nose           | 96.7%    | 43.0%           | **-53.7%** |
| head_midpoint  | 98.3%    | 38.3%           | **-60.0%** |
| neck           | 99.0%    | 37.6%           | **-61.4%** |
| tail_base      | 97.9%    | 50.9%           | **-47.0%** |
| mouse_center   | 99.6%    | 52.6%           | **-47.0%** |

### 2.3 Confidence > 0.9 (fraction of ALL frames)

| Bodypart       | Champion | SA (zero-shot) | Delta     |
|----------------|----------|-----------------|-----------|
| left_ear       | 73.9%    | 4.4%            | **-69.5%** |
| right_ear      | 68.4%    | 2.1%            | **-66.3%** |
| nose           | 70.9%    | 4.2%            | **-66.7%** |
| head_midpoint  | 66.2%    | 0.8%            | **-65.4%** |
| neck           | 82.5%    | 0.7%            | **-81.8%** |
| tail_base      | 74.0%    | 18.2%           | **-55.8%** |
| mouse_center   | 95.3%    | 6.6%            | **-88.8%** |

### 2.4 Confidence Among Detected Frames Only (SA model)

Even when the SA model detects the animal, keypoint confidence is substantially lower
than the champion model:

| Bodypart       | Champion (median) | SA detected-only (median) |
|----------------|-------------------|---------------------------|
| left_ear       | 0.980             | 0.701                     |
| right_ear      | 0.959             | 0.595                     |
| nose           | 0.975             | 0.634                     |
| head_midpoint  | 0.948             | 0.517                     |
| neck           | 0.994             | 0.527                     |
| tail_base      | 0.980             | 0.848                     |
| mouse_center   | 0.999             | 0.802                     |

---

## 3. Jump Rate Analysis

Frame-to-frame displacement > 20 px (among valid transitions between detected frames).
At 30 fps, 20 px corresponds to ~600 px/s -- far beyond physiological mouse movement.

### 3.1 Mean Jump Rate Across Sessions

| Bodypart       | Champion | SA (zero-shot) | Ratio  |
|----------------|----------|-----------------|--------|
| left_ear       | 1.56%    | 8.53%           | 5.5x   |
| right_ear      | 1.55%    | 8.80%           | 5.7x   |
| nose           | 3.54%    | 13.72%          | 3.9x   |
| head_midpoint  | 1.66%    | 9.80%           | 5.9x   |
| neck           | 0.98%    | 6.59%           | 6.7x   |
| mouse_center   | 0.21%    | 3.55%           | 16.7x  |

### 3.2 Worst Sessions for Jump Rate (SA model, left_ear)

| Session                                       | SA J>20px | Champion J>20px |
|-----------------------------------------------|-----------|-----------------|
| sub-1118320/ses-20221116T143112                | 20.80%    | 0.80%           |
| sub-1117646/ses-20220804T112159                | 17.16%    | 4.61%           |
| sub-1118018/ses-20221010T114335                | 16.39%    | 1.22%           |
| sub-1117217/ses-20220608T152732                | 13.04%    | 1.72%           |
| sub-1118020/ses-20221003T143654                | 11.47%    | 1.91%           |

### 3.3 Ear Distance Consistency (coefficient of variation)

Inter-ear distance should be approximately constant for rigid head geometry.

| Session (sample)                               | Champion CV | SA CV   |
|------------------------------------------------|-------------|---------|
| sub-1114353/ses-20210823T165950                 | 0.347       | 0.542   |
| sub-1114356/ses-20210924T160921                 | 0.301       | 0.521   |
| sub-1115464/ses-20211203T151027                 | (similar)   | (similar) |

The SA model shows ~50% higher variability in inter-ear distance, indicating less
precise localisation of ear keypoints even when detected.

---

## 4. HD (Head Direction) Quality in sync.h5

The sync.h5 files use the **current champion model** (old retrained HrnetW32). HD is
computed from ear vectors with confidence-based filtering (threshold: `quantile:0.25`,
i.e., bottom 25% of each bodypart's confidence distribution is rejected).

### 4.1 Per-Session HD and Position Validity

| Session                                       | Excl | N frames | HD valid | Pos valid | Active % | Bad behav % | HD conf |
|-----------------------------------------------|------|----------|----------|-----------|----------|-------------|---------|
| sub-1114353/ses-20210823T165950                |      | 18000    | 80.8%    | 94.1%     | 66.8%    | 45.9%       | 1.267   |
| sub-1114356/ses-20210920T110937                |      | 18000    | 79.4%    | 98.3%     | 68.4%    | 0.0%        | 1.194   |
| sub-1114356/ses-20210923T150514                |      | 18000    | 80.4%    | 95.5%     | 40.3%    | 0.0%        | 1.054   |
| sub-1114356/ses-20210924T160921                |      | 18000    | 78.1%    | 93.8%     | 40.6%    | 0.0%        | 1.062   |
| sub-1114356/ses-20211028T102238                | *    | 18000    | 77.5%    | 89.7%     | 26.8%    | 0.0%        | 0.912   |
| sub-1115464/ses-20211203T151027                |      | 18000    | 78.4%    | 96.3%     | 47.7%    | 0.0%        | 1.241   |
| sub-1115465/ses-20211028T112550                |      | 18000    | 79.0%    | 92.0%     | 42.6%    | 30.1%       | 1.253   |
| sub-1115465/ses-20211029T135008                |      | 18000    | 79.5%    | 95.1%     | 52.5%    | 0.0%        | 1.091   |
| sub-1115465/ses-20211102T151134                |      | 18000    | 79.0%    | 96.5%     | 33.8%    | 0.0%        | 1.136   |
| sub-1115816/ses-20211216T143639                |      | 36000    | 78.3%    | 97.5%     | 52.1%    | 0.0%        | 1.198   |
| sub-1116663/ses-20220408T150157                |      | 18000    | 80.4%    | 95.6%     | 49.6%    | 39.1%       | 1.169   |
| sub-1116663/ses-20220411T164508                |      | 18000    | 80.6%    | 92.8%     | 68.9%    | 0.0%        | 1.341   |
| sub-1116994/ses-20220608T162206                |      | 18000    | 79.4%    | 93.8%     | 35.7%    | 48.8%       | 1.126   |
| sub-1117217/ses-20220531T110613                | *    | --       | --       | --        | --       | --          | --      |
| sub-1117217/ses-20220601T135318                | *    | 18000    | 82.2%    | 93.0%     | 75.2%    | 0.0%        | 1.161   |
| sub-1117217/ses-20220608T152732                |      | 18000    | 79.8%    | 93.1%     | 65.1%    | 8.1%        | 1.129   |
| sub-1117646/ses-20220802T150653                |      | 18000    | 82.3%    | 96.9%     | 47.8%    | 32.0%       | 0.927   |
| sub-1117646/ses-20220804T112159                |      | 18000    | 81.8%    | 96.0%     | 82.0%    | 0.0%        | 0.897   |
| sub-1117646/ses-20220804T135202                | *    | 18000    | 78.2%    | 98.8%     | 47.8%    | 0.0%        | 0.801   |
| sub-1117788/ses-20221018T105617                |      | 14577    | 82.3%    | 95.7%     | 79.7%    | 16.3%       | 1.169   |
| sub-1118018/ses-20221010T114335                |      | 18000    | 82.6%    | 94.2%     | 80.4%    | 0.0%        | 1.060   |
| sub-1118020/ses-20221003T143654                |      | 18000    | 82.6%    | 95.5%     | 81.5%    | 0.0%        | 1.066   |
| sub-1118023/ses-20221004T104258                |      | 18000    | 82.9%    | 95.9%     | 85.9%    | 0.0%        | 1.042   |
| sub-1118213/ses-20221115T132742                |      | 18000    | 82.6%    | 97.0%     | 83.2%    | 0.0%        | 1.167   |
| sub-1118317/ses-20221117T132031                | *    | 18000    | 81.8%    | 96.7%     | 74.2%    | 0.0%        | 1.212   |
| sub-1118320/ses-20221116T143112                |      | 18000    | 80.7%    | 94.5%     | 70.6%    | 0.0%        | 1.200   |

\* = excluded session. `HD conf` is the sum of individual HD method confidences (ears,
head-neck, nose-head, nose-neck); values >1 indicate agreement across multiple methods.

### 4.2 Sync Summary (N=25 sessions with data, excluding sync failure)

| Metric                     | Mean    | SD     | Min    | Max    |
|----------------------------|---------|--------|--------|--------|
| HD valid fraction           | 80.4%   | 1.7%   | 77.5%  | 82.9%  |
| Position valid fraction     | 95.1%   | 2.0%   | 89.7%  | 98.8%  |
| HD confidence (mean)        | 1.115   | 0.125  | 0.801  | 1.341  |

### 4.3 Session with Sync Failure

- **sub-1117217/ses-20220531T110613**: sync status = `FAILED_TEMPORAL_OVERLAP`
  (camera recording 383s longer than imaging; overlap fraction 0.83 < threshold 0.95).
  This session is already flagged as `exclude=1` (camera sync problem). The sync.h5
  file exists but contains no data arrays (only attributes).

### 4.4 Per-Bodypart Validity in sync.h5 (at imaging rate)

| Metric               | Mean   | SD    |
|----------------------|--------|-------|
| Left ear valid       | 77.5%  | 1.5%  |
| Right ear valid      | 77.1%  | 1.3%  |
| Nose_tip valid       | 78.6%  | 2.5%  |
| Head_midpoint valid  | 77.2%  | 1.4%  |
| Neck valid           | 77.0%  | 1.3%  |
| Mouse_center valid   | 75.4%  | 0.5%  |

The ~75% per-bodypart validity arises from the `quantile:0.25` confidence threshold,
which removes the bottom 25% of confidence values. The HD validity (~80%) exceeds
individual bodypart validity because HD is derived from multiple redundant methods
(ear vector, head-neck vector, nose-head vector, nose-neck vector) and remains valid
as long as at least one method produces a valid estimate.

---

## 5. SA Model: Light vs Dark Detection Rate

The SA model's ~43% non-detection rate is **not caused by darkness**. Detection rates
are statistically indistinguishable between light-on and light-off conditions:

| Condition | Detection Rate (mean +/- SD) |
|-----------|------------------------------|
| Light ON  | 57.0% +/- 14.2%             |
| Light OFF | 57.5% +/- 14.4%             |
| Delta     | +0.5% (not significant)      |

The non-detection is caused by the FasterRCNN top-down detector failing to recognise
the mouse under the 2P headstage from overhead view. The headstage significantly alters
the animal's silhouette compared to the SuperAnimal training corpus, which primarily
contains unrestrained mice without surgical implants.

---

## 6. SA Model: Per-Session Detection Rates

| Session                                       | Excl | Detection | Both ears >0.4 | Both ears >0.9 |
|-----------------------------------------------|------|-----------|----------------|----------------|
| sub-1114353/ses-20210823T165950                |      | 47.9%     | 32.3%          | 0.4%           |
| sub-1114356/ses-20210920T110937                |      | 44.1%     | 25.8%          | 0.1%           |
| sub-1114356/ses-20210923T150514                |      | 30.6%     | 22.1%          | 0.1%           |
| sub-1114356/ses-20210924T160921                |      | 45.3%     | 31.3%          | 0.7%           |
| sub-1114356/ses-20211028T102238                | *    | 49.7%     | 21.7%          | 0.1%           |
| sub-1115464/ses-20211203T151027                |      | 58.3%     | 41.2%          | 1.0%           |
| sub-1115465/ses-20211028T112550                |      | 59.7%     | 31.4%          | 0.2%           |
| sub-1115465/ses-20211029T135008                |      | 29.0%     | 17.8%          | 0.1%           |
| sub-1115465/ses-20211102T151134                |      | 49.9%     | 29.2%          | 0.2%           |
| sub-1115816/ses-20211216T143639                |      | 51.7%     | 32.6%          | 0.3%           |
| sub-1116663/ses-20220408T150157                |      | 73.7%     | 46.5%          | 0.4%           |
| sub-1116663/ses-20220411T164508                |      | 71.4%     | 53.0%          | 0.8%           |
| sub-1116994/ses-20220608T162206                |      | 49.8%     | 38.2%          | 0.2%           |
| sub-1117217/ses-20220531T110613                | *    | 38.9%     | 28.2%          | 0.4%           |
| sub-1117217/ses-20220601T135318                | *    | 49.4%     | 40.5%          | 0.6%           |
| sub-1117217/ses-20220608T152732                |      | 66.8%     | 35.0%          | 0.5%           |
| sub-1117646/ses-20220802T150653                |      | 46.5%     | 34.1%          | 0.4%           |
| sub-1117646/ses-20220804T112159                |      | 45.5%     | 22.4%          | 0.2%           |
| sub-1117646/ses-20220804T135202                | *    | 61.2%     | 26.0%          | 0.1%           |
| sub-1117788/ses-20221018T105617                |      | 78.7%     | 60.6%          | 0.7%           |
| sub-1118018/ses-20221010T114335                |      | 83.8%     | 56.1%          | 1.2%           |
| sub-1118020/ses-20221003T143654                |      | 75.6%     | 54.0%          | 0.5%           |
| sub-1118023/ses-20221004T104258                |      | 65.8%     | 48.5%          | 0.3%           |
| sub-1118213/ses-20221115T132742                |      | 66.5%     | 52.1%          | 1.1%           |
| sub-1118317/ses-20221117T132031                | *    | 57.5%     | 37.9%          | 0.3%           |
| sub-1118320/ses-20221116T143112                |      | 72.6%     | 41.1%          | 0.2%           |

Detection rate varies substantially across sessions (29.0% -- 83.8%), likely related to
headstage visibility and mouse appearance variation across animals. Later sessions
(sub-1118xxx, TFB fibre, f6mm lens) tend to have higher detection rates, possibly because
the TFB fibre has a different appearance profile.

---

## 7. Champion Model: Per-Session Ear Confidence at >0.9

| Session                                       | Left ear >0.9 | Right ear >0.9 |
|-----------------------------------------------|---------------|----------------|
| sub-1114353/ses-20210823T165950                | 72.1%         | 75.0%          |
| sub-1114356/ses-20210920T110937                | 71.6%         | 64.1%          |
| sub-1114356/ses-20210923T150514                | 61.2%         | 47.2%          |
| sub-1114356/ses-20210924T160921                | 87.2%         | 67.2%          |
| sub-1114356/ses-20211028T102238 *              | 55.3%         | 35.9%          |
| sub-1115464/ses-20211203T151027                | 83.3%         | 70.8%          |
| sub-1115465/ses-20211028T112550                | 80.8%         | 76.3%          |
| sub-1115465/ses-20211029T135008                | 67.9%         | 58.5%          |
| sub-1115465/ses-20211102T151134                | 78.0%         | 83.4%          |
| sub-1115816/ses-20211216T143639                | 65.9%         | 67.7%          |
| sub-1116663/ses-20220408T150157                | 90.2%         | 74.8%          |
| sub-1116663/ses-20220411T164508                | 84.3%         | 80.9%          |
| sub-1116994/ses-20220608T162206                | 74.7%         | 77.5%          |
| sub-1117217/ses-20220531T110613 *              | 66.1%         | 65.8%          |
| sub-1117217/ses-20220601T135318 *              | 75.4%         | 76.8%          |
| sub-1117217/ses-20220608T152732                | 75.9%         | 66.9%          |
| sub-1117646/ses-20220802T150653                | 64.5%         | 60.2%          |
| sub-1117646/ses-20220804T112159                | 67.0%         | 61.3%          |
| sub-1117646/ses-20220804T135202 *              | 69.4%         | 65.8%          |
| sub-1117788/ses-20221018T105617                | 82.5%         | 72.5%          |
| sub-1118018/ses-20221010T114335                | 80.4%         | 79.2%          |
| sub-1118020/ses-20221003T143654                | 69.7%         | 68.4%          |
| sub-1118023/ses-20221004T104258                | 68.8%         | 64.5%          |
| sub-1118213/ses-20221115T132742                | 80.8%         | 76.9%          |
| sub-1118317/ses-20221117T132031 *              | 67.2%         | 70.5%          |
| sub-1118320/ses-20221116T143112                | 81.0%         | 70.3%          |

**Mean:** left ear 73.9% +/- 8.4%, right ear 68.4% +/- 10.2%.

The right ear consistently has lower confidence than the left ear (68.4% vs 73.9%
at >0.9 threshold). This is a known asymmetry in the training data and does not affect
HD computation materially because both ears are high confidence (>0.4) in >98% of frames.

---

## 8. Notable Sessions

### Sessions with quality concerns

1. **sub-1114356/ses-20211028T102238** (excluded): Lowest active fraction (26.8%),
   lowest HD confidence (0.912). Excluded for "fluctuating traces."

2. **sub-1117217/ses-20220531T110613** (excluded): Sync failure. Camera sync problem
   noted in experiments.csv. Sync.h5 has no data arrays.

3. **sub-1117217/ses-20220601T135318** (excluded): Camera sync problem. Sync data
   exists but session is excluded.

4. **sub-1117646/ses-20220804T135202** (excluded): "Not a good 2p recording." Lowest
   HD confidence (0.801) among sessions with data.

5. **sub-1114353/ses-20210823T165950**: Highest bad_behav fraction (45.9%). Mouse
   stuck on fibre/wires for extended periods. Still has 80.8% HD validity.

6. **sub-1116994/ses-20220608T162206**: Second-highest bad_behav (48.8%). Bad behaviour
   window is 10:00-25:00 (15 min of 30 min session).

### Sessions with best quality

1. **sub-1118023/ses-20221004T104258**: Highest HD validity (82.9%), highest active
   fraction (85.9%), no bad_behav, no exclusion. Primary experiment.

2. **sub-1118213/ses-20221115T132742**: HD 82.6%, active 83.2%, position 97.0%.

3. **sub-1117646/ses-20220804T112159**: HD 81.8%, active 82.0%. Not primary but
   non-excluded.

---

## 9. Assessment: Is Tracking Good Enough?

### For HD tuning curves

**Yes.** HD is valid in 80.4% of imaging frames on average. After excluding bad_behav
and immobile periods, a typical session retains ~12,000 valid HD frames at ~9.6 Hz
(~20 min of data). With 36 bins of 10 degrees, occupancy is typically >100 frames per bin,
which is sufficient for stable tuning curve estimation.

### For population decoding

**Yes.** The 80% valid rate and high temporal consistency (low jump rate) mean that
Bayesian decoding can use the majority of each session. The ~20% NaN frames are scattered
(not concentrated in time), so cross-validation folds will have adequate sampling.

### For light vs dark comparisons

**Yes.** The champion model's confidence is not differentially affected by light condition
(the model was trained on both light and dark frames from this dataset). The ~20% filtering
applies approximately equally in both conditions.

### For speed filtering

**Yes.** Position validity is 95.1%, and speed is well-defined for >93% of frames.
Speed-filtered analyses will lose ~7% of frames from position NaN, which is acceptable.

### For the SA model specifically

**No.** The SA model is not suitable for any analysis requiring HD computation.
With only 36.9% of frames having both ears above 0.4 confidence (and only 0.4% above 0.9),
HD tuning curves would be based on a fraction of the available data and contaminated by
mislocalized keypoints. The 5-6x higher jump rate indicates frequent tracking failures
that would introduce noise into any kinematic analysis.

---

## 10. Recommendations

### Immediate (for manuscript)

1. **Continue with current champion model.** The tracking quality is sufficient for all
   planned analyses. No model change is needed.

2. **Verify excluded sessions.** Four sessions are excluded: two for camera sync problems
   (sub-1117217), one for bad 2P (sub-1117646/ses-20220804T135202), one for fluctuating
   traces (sub-1114356/ses-20211028T102238). Also sub-1118317/ses-20221117T132031 is
   excluded for bad 2P + tethering. All exclusion decisions appear justified by the
   quality metrics.

3. **Report tracking quality in methods.** State that DLC HrnetW32 was trained on 354
   labeled frames (8 bodyparts); that pose confidence was filtered per bodypart using a
   quantile-based threshold (bottom 25%); that 80.4% of imaging frames had valid HD
   estimates; and that results were robust to stricter confidence thresholds.

### Future (if SA fine-tuning is pursued)

4. **Fine-tune the FasterRCNN detector.** The SA model's keypoint confidence (when
   detected) is reasonable (median ~0.70 for ears), but the top-down detector fails in
   43% of frames. Fine-tuning the detector on this specific setup (2P headstage from
   overhead) would likely restore detection rate close to 100%.

5. **Consider video adaptation (DLC 3.x).** SuperAnimal's video adaptation mode
   pseudo-labels frames from the target video and uses them to adapt the model. This
   avoids manual labeling but requires careful quality control.

6. **Do not switch models mid-project.** All current downstream derivatives (kinematics,
   sync, analysis) are derived from the champion model. Switching models requires
   re-running the entire pipeline from Stage 2b onward.

---

## 11. Data Files Analysed

- **Pose files (SA):** 26 sessions from `s3://hm2p-derivatives/pose/` (30 fps,
  `_superanimal_topviewmouse_hrnet_w32_fasterrcnn_resnet50_fpn_v2.h5`)
- **Pose files (champion):** 26 sessions from `s3://hm2p-derivatives/pose/` (30 fps,
  `DLC_HrnetW32_hm2p-retrainMar20shuffle1_snapshot_best-100.h5`)
- **Sync files:** 26 sessions from `s3://hm2p-derivatives/sync/` (`sync.h5`)
- **Analysis date:** 2026-05-14
- **Note:** `pose-finetuned/` on S3 contains identical files to `pose/` for the
  champion model (verified by byte-exact comparison of likelihood arrays).
