# Analysis Plan — Condition-dependent activity and spatial tuning

## Overview

Compare neural activity and spatial tuning between **Penk+** and **non-Penk CamKII+** RSP
neurons across four behavioural conditions (2x2: movement x light).

### Key challenge

Penk and non-Penk are **different mice** with potentially different GCaMP expression levels,
SNR, and baseline activity. Raw firing rates are not directly comparable across genotypes.
All comparisons must use within-cell normalization or mixed-effects models with mouse as
a random effect.

---

## 1. Condition-split activity analysis

### 2x2 conditions (per frame)

| | Light ON | Light OFF |
|---|---|---|
| **Moving** (speed >= 2.5 cm/s) | active + light | active + dark |
| **Stationary** (speed < 2.5 cm/s) | stationary + light | stationary + dark |

### Metrics per cell per condition

- **Event rate** (events/s) — from V&H event detection masks
- **Mean dF/F** — mean fluorescence during condition frames
- **Mean event amplitude** — mean dF/F during detected events only

### Within-cell modulation indices

- **Movement modulation**: `(rate_moving - rate_stationary) / (rate_moving + rate_stationary)`
- **Light modulation**: `(rate_light - rate_dark) / (rate_light + rate_dark)`
- These indices are bounded [-1, 1] and independent of baseline expression

### Statistical approach

- **Linear mixed-effects model** (via statsmodels or pymer4):
  - Fixed effects: `movement * light * celltype`
  - Random intercept: `mouse`
  - Accounts for nested structure (cells within mice within genotypes)
- **Per-cell tests**: Wilcoxon rank-sum between conditions (paired within cell)
- **Population-level**: Compare distributions of modulation indices between Penk vs non-Penk

### Exclusion criteria

- Minimum 30s in each of the 4 conditions
- Exclude `bad_behav` frames
- SNR > 3 (from ca.h5)

---

## 2. HD tuning analysis

### Tuning curve computation

For each cell, compute HD tuning curve = mean signal as a function of head direction.

**Configurable parameters:**

| Parameter | Options | Default |
|-----------|---------|---------|
| Signal type | `dff`, `deconv`, `events` (binary), `event_rate` | `dff` |
| Number of HD bins | 12, 18, 24, 36, 72 | 36 (10 deg) |
| Smoothing sigma | 0, 3, 6, 9, 12, 15 deg | 6 deg |
| Speed filter | 0, 1.0, 2.5, 5.0 cm/s | 2.5 cm/s |
| Condition split | all, light_only, dark_only | per-condition |

### Tuning metrics

- **Mean Vector Length (MVL)**: circular mean resultant length of the tuning curve.
  MVL = 1 means perfectly tuned; MVL = 0 means uniform.
- **Preferred Direction (PD)**: angle of the circular mean vector
- **Rayleigh vector length**: same as MVL but sometimes reported differently
- **Skaggs spatial information**: bits/event, adapted for circular variable
- **Peak-to-trough ratio**: max / min of tuning curve
- **Tuning width**: FWHM of peak in tuning curve

### Significance testing (bootstrap)

1. Circularly shift the signal relative to HD by a random offset (preserving temporal
   autocorrelation of both signal and behaviour)
2. Recompute the tuning metric (e.g. MVL)
3. Repeat N times (default 1000)
4. P-value = fraction of shuffled metrics >= observed metric
5. Significance threshold: p < 0.05 (or user-configurable)

**Why circular shift, not frame permutation:** Random frame permutation destroys temporal
autocorrelation and inflates significance. Circular shift preserves the temporal structure
of both neural and behavioural signals while breaking the specific time-alignment between them.

### Light vs dark comparison

- Compute separate tuning curves for light-on and light-off frames
- **Tuning curve correlation**: Pearson r between light and dark tuning curves
- **PD shift**: angular difference in preferred direction between conditions
- **MVL ratio**: MVL_dark / MVL_light (>1 = more tuned in dark)
- **Population-level**: compare distributions of these metrics between Penk vs non-Penk

---

## 3. Place tuning analysis

### Rate map computation

2D histogram of mean signal as a function of (x, y) position.

**Configurable parameters:**

| Parameter | Options | Default |
|-----------|---------|---------|
| Signal type | `dff`, `deconv`, `events`, `event_rate` | `dff` |
| Bin size | 1.0, 2.0, 2.5, 5.0 cm | 2.5 cm |
| Smoothing sigma | 0, 1.5, 3.0, 5.0 cm | 3.0 cm |
| Speed filter | 0, 1.0, 2.5, 5.0 cm/s | 2.5 cm/s |
| Min occupancy | 0, 0.1, 0.5 s per bin | 0.5 s |
| Condition split | all, light_only, dark_only | per-condition |

### Tuning metrics

- **Skaggs spatial information** (bits/event):
  `SI = sum_i p_i * (r_i / r_mean) * log2(r_i / r_mean)`
  where `p_i` = occupancy fraction, `r_i` = rate in bin i, `r_mean` = overall mean rate
- **Spatial coherence**: correlation between rate in each bin and mean rate of its neighbours
- **Place field count**: number of contiguous regions above threshold
- **Place field size**: fraction of bins in the largest place field
- **Peak rate**: maximum bin rate
- **Sparsity**: `(sum p_i * r_i)^2 / sum(p_i * r_i^2)` — low = sparse = place-like

### Significance testing (bootstrap)

Same circular shift method as HD tuning. Shift signal relative to position.

### Light vs dark comparison

- Rate map correlation (Pearson r between light and dark rate maps, valid bins only)
- Place field stability: overlap of place fields between conditions
- Spatial information ratio: SI_dark / SI_light

---

## 4. Robustness checking

The user wants to compute metrics under many parameter combinations to verify conclusions
are robust. The analysis module supports a `parameter_grid` approach:

```python
grid = ParameterGrid(
    signal_type=["dff", "deconv", "events"],
    n_bins=[18, 36, 72],           # HD only
    smoothing_sigma=[3.0, 6.0, 12.0],
    speed_threshold=[0.0, 2.5, 5.0],
)
```

For each combination, compute all tuning metrics. Then visualize:
- Heatmap of "fraction of cells significant" across parameter grid
- Correlation matrix of metrics across parameter choices
- Flag any parameter combination where the qualitative conclusion flips

---

## 5. DLC pose estimation pipeline decision

### Model choice

Using **SuperAnimal TopViewMouse** from DeepLabCut 3.0rc13 with the **HRNet-W32**
backbone. This is a foundation model pre-trained on a large corpus of top-view mouse
data, so it generalises to our overhead camera setup without fine-tuning.

### Detector removed for cropped videos

The default DLC 3.0 pipeline uses a FasterRCNN detector stage to crop animals before
pose inference. For our data (single mouse, already cropped to arena), the detector is
unnecessary and too slow: only ~7 it/s even with `batch_size=8`. We run **pose-only
inference** (no detection stage), which is substantially faster.

### Inference settings

- `batch_size=64` for pose inference
- GPU: NVIDIA T4 on EC2 g4dn.xlarge (ap-southeast-2)
- No fine-tuning or transfer learning -- using the foundation model weights directly

### Keypoints

The SuperAnimal TopViewMouse model outputs **27 keypoints**. For downstream kinematics
(Stage 3), we use the following subset:

| SuperAnimal keypoint | Pipeline variable |
|---|---|
| `left_ear` | ear-left |
| `right_ear` | ear-right |
| `mid_back` | back-upper |
| `mouse_center` | back-middle |
| `tail_base` | back-tail |

The remaining 22 keypoints are retained in the pose output files but not used for
head direction or position computation.

---

## 6. Data flow

```
ca.h5           →  dF/F, deconv (spks), event_masks  (n_rois × n_imaging_frames)
kinematics.h5   →  HD, x, y, speed, active, light_on (n_camera_frames)
timestamps.h5   →  frame times for both modalities

sync/align.py   →  resample kinematics to imaging frame times

analysis/       →  condition split, tuning curves, significance, comparison

Stage 6 output: analysis.h5  (per session, saved to S3)
  /{signal_type}/activity/{metric}           (n_rois,)
  /{signal_type}/hd/{condition}/tuning_curves (n_rois, n_bins)
  /{signal_type}/hd/{condition}/mvl           (n_rois,)
  /{signal_type}/hd/{condition}/p_value       (n_rois,)
  /{signal_type}/hd/{condition}/significant   (n_rois,)
  /{signal_type}/hd/comparison/{metric}       (n_rois,)
  /{signal_type}/place/{condition}/...        same structure
  /params/*                                   analysis parameters
```

### Multi-signal analysis pipeline

The analysis pipeline (`scripts/run_stage6_analysis.py`) runs every analysis metric
using **all available calcium measures**:

| Signal | Source | What it captures |
|--------|--------|-----------------|
| `dff` | dF/F0 from ca.h5 | Raw fluorescence changes |
| `deconv` | Suite2p deconvolved (spks) | Inferred spike rate |
| `events` | V&H event masks (binary) | Detected calcium transients |

Results are saved per signal type so the frontend can directly compare whether
conclusions (e.g. "ROI 5 is HD-tuned") hold across all measures. Agreement
is quantified using Jaccard similarity of the significant-ROI sets.

---

## 7. Visualization (frontend)

### Activity by condition tab
- Violin/box plots of event rate per condition, split by celltype
- Modulation index distributions (Penk vs non-Penk)
- Per-cell scatter: movement modulation vs light modulation, colored by celltype

### HD tuning tab
- Polar plot of HD tuning curve per cell (light vs dark overlaid)
- Population summary: fraction of HD cells by condition and celltype
- MVL distribution by celltype and condition
- PD shift distribution by celltype

### Place tuning tab
- Rate map heatmaps per cell (light vs dark side by side)
- Population summary: fraction of place cells by condition and celltype
- SI distribution by celltype and condition

### Robustness tab
- Parameter grid heatmap of significant cell fraction
- Metric stability across parameter choices

---

## 8. References

- HD tuning, MVL, shuffled significance: Zong et al. 2022 (Cell)
- Skaggs spatial information: Skaggs et al. 1996
- Event detection: Voigts & Harnett 2020 (Neuron)
- Speed threshold 2.5 cm/s: Zong et al. 2022
- Mixed-effects models for nested neural data: standard neuroscience practice
