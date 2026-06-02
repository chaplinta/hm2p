# 3-Way ROI Classifier: Refined Implementation Plan

## 1. Scientific Goal

Classify every Suite2p ROI as **soma**, **dendrite**, or **artefact** using a
supervised Random Forest model trained on morphological and temporal features.
Output per-ROI class probabilities, with a configurable threshold for the
soma/dendrite boundary.

---

## 2. Feature Set (14 features)

### 2.1 Morphological features from stat.npy (7 features, no extra computation)

| # | Feature | Key in stat | Expected discrimination |
|---|---------|-------------|------------------------|
| 1 | `compact` | `compact` | Soma ~1.0 (round); dendrite >1.1 (elongated); artefact variable |
| 2 | `aspect_ratio` | `aspect_ratio` | Soma ~1.0; dendrite >1.3 (thin fragments); artefact variable |
| 3 | `npix_norm` | `npix_norm` | Soma ~1.5 (large); dendrite ~0.7 (small); artefact variable |
| 4 | `radius` | `radius` | Soma ~6-8 px; dendrite ~3-5 px |
| 5 | `npix_norm_no_crop` | `npix_norm_no_crop` | Complementary to npix_norm |
| 6 | `footprint` | `footprint` | Categorical (0-3). Cellpose (1) biased toward soma shapes |
| 7 | `snr_stat` | `snr` | Suite2p's built-in normalized SNR (0-1). Higher for real neurons |

### 2.2 Derived morphological features (3 features, computed from stat.npy)

| # | Feature | Computation | Expected discrimination |
|---|---------|-------------|------------------------|
| 8 | `soma_crop_fraction` | `npix_soma / npix` | Soma ~0.8-1.0; dendrite ~0.3-0.6 |
| 9 | `npix_norm_ratio` | `npix_norm / npix_norm_no_crop` | How much soma_crop reduces the ROI. Soma ~1.0; dendrite <0.7 |
| 10 | `n_pixels_raw` | `npix` (raw, unnormalized) | Absolute size; tiny ROIs more likely artefact |

### 2.3 Temporal features (4 features, computed from F.npy / Fneu.npy)

All temporal features computed at **fs = 9.6 Hz** (not the ops.fs=29.97
which is a SciScan metadata misparse).

| # | Feature | Computation | Expected discrimination |
|---|---------|-------------|------------------------|
| 11 | `skew` | `skew` from stat.npy (already computed at correct rate by Suite2p) | Soma >2; dendrite 1-3; artefact <1 |
| 12 | `kurtosis` | `scipy.stats.kurtosis(F_corrected)` | Real neurons: high kurtosis (sharp transients); artefacts: low |
| 13 | `neuropil_corr` | `np.corrcoef(F_corrected, Fneu)[0,1]` | Artefacts: high (neuropil blobs); real neurons: low-moderate |
| 14 | `std_trace` | `std` from stat.npy | Trace variability |

### 2.4 Features considered but excluded

| Feature | Reason for exclusion |
|---------|---------------------|
| `event_rate` | Requires event detection at correct fs; circular dependency with classifier (events depend on which ROIs are valid). Defer to v2 if needed. |
| `decay_tau_median` | Same circularity as event_rate; also noisy for dendrites with fast transients. |
| `eccentricity` | Redundant with aspect_ratio; aspect_ratio is directly available in stat.npy. |
| `perimeter/area ratio` | Requires reconstructing the 2D mask; `compact` captures the same information. |
| `solidity` (area/convex hull) | Requires convex hull computation from pixel masks; marginal benefit over compact. |
| `n_connected_components` | Most ROIs are connected (Suite2p default); fragmented ROIs are rare. |

### 2.5 Feature rationale for the three morphology types

**Short thin dendrite fragments** (majority of dendrites per user):
- `aspect_ratio` is the primary discriminator -- thin fragments have AR >1.3
- `compact` is high (>1.1) because elongated shapes have high mean radial spread
- `soma_crop_fraction` is low -- soma_crop removes most pixels from a thin process
- `skew` moderate (1-3) -- real calcium transients but smaller amplitude than soma

**Amorphous blob artefacts** (majority of artefacts per user):
- `skew` is the primary discriminator -- low (<1) because no real calcium transients
- `neuropil_corr` is high -- these blobs track neuropil fluorescence
- `compact` and `aspect_ratio` are variable (blobs can be any shape)
- `snr_stat` is low

**Soma:**
- `compact` ~1.0 (round)
- `aspect_ratio` ~1.0 (circular)
- `skew` high (>2) -- strong, reliable calcium transients
- `soma_crop_fraction` ~0.8-1.0 -- soma_crop retains most pixels

---

## 3. Training Data Strategy

### 3.1 Current data situation

| Data source | Content | Spatial info? | Usable? |
|-------------|---------|---------------|---------|
| S3 `iscell.npy` | Binary soma classifier output (classifier-generated, NOT manual) | Yes (same stat.npy) | Partially -- gives soma candidates but no dendrite/artefact split |
| Old `classifier_soma.npy` | 4413 ROIs, manual binary soma labels (313 soma) | No (only 3 features stored) | Cannot map to S3 ROIs |
| Old `classifier_dend.npy` | Same 4413 ROIs, manual binary dend labels (241 dend) | No (only 3 features stored) | Cannot map to S3 ROIs; soma_crop mismatch |

**Key finding:** The old classifier files store only `(skew, compact, npix_norm)`
training data -- no spatial coordinates. The S3 Suite2p outputs are from a
different Suite2p run (v1.0.0.1 vs old version), producing different ROI sets
with different feature values. Direct spatial matching or feature matching
between old labels and new ROIs is not feasible.

### 3.2 Recommended approach: Bootstrap + Manual Review

**Phase 1 -- Bootstrap labels (automated, ~30 min dev time):**

Generate tentative 3-way labels from existing S3 data:

```python
# For each session's stat.npy + iscell.npy:
if iscell_prob > 0.7:      # soma classifier says cell with high confidence
    label = SOMA            # tentative
elif skew > 1.5 and (aspect_ratio > 1.2 or compact > 1.1):
    label = DENDRITE        # tentative: active + elongated
else:
    label = ARTEFACT        # tentative: low activity or amorphous
```

These thresholds are initial estimates from the old classifier's feature
distributions (Section 2.3 of the technical report). They will produce noisy
labels that must be corrected manually.

**Phase 2 -- Manual labeling in Streamlit (the gold standard):**

Build a Streamlit labeling page that shows:
- Mean image with ROI mask overlay (color-coded by bootstrap prediction)
- Zoomed view of selected ROI (pixel mask on mean image)
- Calcium trace (F_corrected) for the selected ROI
- Feature values: compact, aspect_ratio, skew, soma_crop_fraction, neuropil_corr
- Radio buttons: soma / dendrite / artefact
- Bootstrap prediction shown as default selection
- "Skip" button for ambiguous ROIs
- Progress bar and session selector

Label **6-8 sessions** (targeting ~1000 labeled ROIs total, balanced across
sessions and animals). Estimated effort: 3-4 hours of manual labeling. The
user has confirmed these will be high-quality labels (same standard as the
old pipeline's Suite2p GUI curation).

**Phase 3 -- Train classifier on manually corrected labels.**

### 3.3 Label file format

Per-session file: `roi_labels.npy`
- Shape: `(n_rois,)`, dtype: `int8`
- Values: `0 = artefact`, `1 = soma`, `2 = dendrite`, `-1 = unlabeled`
- Stored alongside stat.npy in the Suite2p output directory on S3

---

## 4. Model Architecture

### 4.1 Random Forest (primary)

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=5,
    class_weight='balanced',  # handle artefact class imbalance
    random_state=42,
    n_jobs=-1,
    oob_score=True,           # out-of-bag estimate for free
)
```

**Why Random Forest:**
- Native multi-class support
- Works well with small training sets (~500-1000 samples)
- Feature importance is interpretable (verify that compact, skew, aspect_ratio dominate)
- Robust to outliers and feature scale
- Out-of-bag score provides an estimate of generalization without cross-validation
- No hyperparameter tuning needed (good defaults)

**Why not XGBoost/LightGBM:** Marginal accuracy gain not worth the extra
dependency and tuning complexity for ~1000 training samples.

**Why not deep learning:** Insufficient training data. A CNN on ROI image
patches would need ~1000+ per class and add major complexity.

### 4.2 Output format

The classifier outputs a **probability vector** `[P(artefact), P(soma), P(dendrite)]`
for each ROI. The hard label is determined by a threshold-based rule:

```python
probs = clf.predict_proba(features)  # (n_rois, 3)

# Default: argmax classification
labels = np.argmax(probs, axis=1)  # 0=artefact, 1=soma, 2=dendrite

# With configurable soma threshold:
# If P(soma) >= soma_threshold -> soma
# Else if P(dendrite) >= dendrite_threshold -> dendrite
# Else -> artefact
```

### 4.3 Saved model artefact

```
sourcedata/classifiers/roi_classifier_v1.joblib   # sklearn model
sourcedata/classifiers/roi_classifier_v1.json      # metadata
```

Metadata JSON:
```json
{
    "version": "1.0",
    "n_features": 14,
    "feature_names": ["compact", "aspect_ratio", ...],
    "n_training_samples": {"soma": 200, "dendrite": 150, "artefact": 650},
    "training_sessions": ["ses-20210823T165950", ...],
    "cv_f1_macro": 0.92,
    "suite2p_version": "1.0.0.1",
    "fs_assumed_hz": 9.6,
    "date_trained": "2026-06-02",
    "wandb_run_id": "abc123"
}
```

---

## 5. Validation Strategy

### 5.1 Cross-validation

**Leave-one-session-out (LOSO):** Train on N-1 sessions, test on held-out
session. This tests generalization across imaging conditions (different
animals, days, FOV depths). Report per-class F1 and confusion matrix for each
fold.

**5-fold stratified CV** as a secondary metric (faster, more stable estimates).

Target: **F1_macro >= 0.85** across LOSO folds.

### 5.2 Feature importance

Use Random Forest `feature_importances_` (Gini importance) to verify that
biologically meaningful features dominate:
- Expect: compact, aspect_ratio, skew in top 3
- Red flag: if footprint or npix dominate, the model may be fitting session-specific
  artefacts rather than biology

Also compute permutation importance (shuffle each feature, measure accuracy drop)
as a robustness check.

### 5.3 Confusion matrix analysis

Expected confusions:
- **Soma <-> Dendrite:** Branching tree dendrites (type b from user) may overlap
  with compact soma. The threshold slider (Section 6) addresses this.
- **Dendrite <-> Artefact:** Neuropil blobs with some real signal. `neuropil_corr`
  should help separate these.
- **Soma <-> Artefact:** Rare. Somas have high skew; artefacts have low skew.

### 5.4 Comparison with old pipeline

For sessions that have old pipeline outputs (soma + dendrite cell counts known
from the published analysis), compare:
- Total soma count: new classifier vs old soma classifier
- Total dendrite count: new classifier vs old dend classifier
- Agreement on which ROIs are soma/dendrite (where spatial matching is possible
  via the S3 iscell.npy as a proxy)

---

## 6. Threshold UI Design

### 6.1 Configurable threshold in the frontend

The analysis pipeline stores **probabilities** in sync.h5, not hard labels.
The hard label is computed at analysis time using configurable thresholds.

**sync.h5 schema addition:**
```
/roi_class_prob    (n_rois, 3) float32    # [P(artefact), P(soma), P(dendrite)]
/roi_class         (n_rois,) int8          # hard label at default threshold
```

**Frontend threshold controls (in page body, not sidebar):**

```
[Soma probability threshold]  ----[slider 0.3 ... 0.9]---- [default: 0.5]
[Dendrite probability threshold]  ----[slider 0.3 ... 0.9]---- [default: 0.5]

At current thresholds:
  Soma: 42 ROIs  |  Dendrite: 28 ROIs  |  Artefact: 71 ROIs
  (changed from default: +3 soma, -2 dendrite, -1 artefact)
```

**Threshold logic:**
1. If `P(soma) >= soma_threshold` and `P(soma) > P(dendrite)`: **soma**
2. Elif `P(dendrite) >= dendrite_threshold` and `P(dendrite) > P(soma)`: **dendrite**
3. Else: **artefact**

Default thresholds: `soma_threshold = 0.5`, `dendrite_threshold = 0.5`
(equivalent to argmax when only two classes compete).

Raising `soma_threshold` from 0.5 to 0.8 makes soma classification more
conservative -- borderline round dendrites will be reclassified as dendrites.
This lets the user test whether HD tuning results are robust to the
soma/dendrite boundary.

### 6.2 Sensitivity analysis

The analysis pages (tuning curves, decoding, etc.) should include a
"ROI Classification Sensitivity" expander that shows:
- How many ROIs change class across the threshold range
- Whether the main result (e.g., mean MVL for soma vs dendrite) changes
  qualitatively when the threshold is varied from 0.3 to 0.9

---

## 7. Frame Rate Correction

**The ops.fs = 29.97 stored in Suite2p outputs is WRONG.** It comes from
a SciScan metadata misparse. The true frame rate is ~9.6 Hz.

This affects two temporal features:
- `skew` -- **not affected** (skewness is frame-rate independent)
- `kurtosis` -- **not affected** (kurtosis is frame-rate independent)
- `neuropil_corr` -- **not affected** (correlation is frame-rate independent)
- `std_trace` -- **not affected** (std is frame-rate independent)

If we later add `event_rate` or `decay_tau_median`, those **would** need
fs = 9.6 Hz. For the current 14-feature set, no features require explicit
frame rate correction because all temporal features are frame-rate invariant
statistics (skew, kurtosis, correlation, std).

However, the **feature extraction code** must document that fs = 9.6 Hz is
the correct rate, and any future temporal features (event rate, decay time)
must use this value, not ops['fs'].

---

## 8. Implementation Modules

### 8.1 Module structure

```
src/hm2p/calcium/
    roi_classifier.py       # Feature extraction + classification
    roi_labeling.py         # Bootstrap label generation

frontend/pages/
    roi_labeling.py         # Streamlit labeling interface
    roi_classification.py   # Classification results + threshold UI

tests/
    test_roi_classifier.py  # Unit tests with synthetic data
```

### 8.2 Core API

```python
# roi_classifier.py

def extract_features(
    stat: np.ndarray,       # stat.npy (array of dicts)
    F: np.ndarray,          # F.npy (n_rois, n_frames)
    Fneu: np.ndarray,       # Fneu.npy (n_rois, n_frames)
    neucoeff: float = 0.7,
) -> tuple[np.ndarray, list[str]]:
    """Extract 14 classification features from Suite2p outputs.

    Returns
    -------
    features : np.ndarray, shape (n_rois, 14)
    feature_names : list of str, length 14
    """

def train_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    session_ids: np.ndarray | None = None,   # for LOSO CV
) -> tuple[RandomForestClassifier, dict]:
    """Train 3-way ROI classifier.

    Returns
    -------
    clf : RandomForestClassifier
    metrics : dict with cv_scores, confusion_matrix, feature_importances
    """

def classify_rois(
    stat: np.ndarray,
    F: np.ndarray,
    Fneu: np.ndarray,
    clf: RandomForestClassifier,
    soma_threshold: float = 0.5,
    dendrite_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify ROIs using trained model.

    Returns
    -------
    labels : np.ndarray, shape (n_rois,), int8
        0=artefact, 1=soma, 2=dendrite
    probs : np.ndarray, shape (n_rois, 3), float32
        [P(artefact), P(soma), P(dendrite)]
    """

def apply_threshold(
    probs: np.ndarray,
    soma_threshold: float = 0.5,
    dendrite_threshold: float = 0.5,
) -> np.ndarray:
    """Apply configurable thresholds to probability vectors.

    Separated from classify_rois so the frontend can re-threshold
    without re-running the classifier.
    """
```

### 8.3 W&B integration

Project: `hm2p-roi-classifier` (same entity as `hm2p-dlc`).

Log per training run:
- Hyperparameters (n_estimators, max_depth, etc.)
- LOSO CV scores (per-fold F1, confusion matrices)
- Feature importances (bar chart)
- Overall metrics (F1_macro, accuracy, Cohen's kappa)
- Training data summary (N per class, sessions used)
- PCA/UMAP embedding of features colored by label (as W&B plot)

---

## 9. Visualization

### 9.1 Frontend: ROI Classification Results page

**Layout:**
1. Session selector (dropdown in page body)
2. Threshold sliders (soma, dendrite)
3. Mean image with ROI overlays (color: green=soma, magenta=dendrite, gray=artefact)
4. ROI count summary bar
5. Per-ROI table: ROI_id, class, P(soma), P(dendrite), P(artefact), compact, AR, skew
6. Click-to-inspect: clicking an ROI shows its trace and zoomed mask

**Embedding toggle:** PCA or UMAP of the 14-feature space, colored by
predicted class. Toggle between PCA and UMAP via radio buttons in page body.
- PCA: deterministic, fast, shows global variance structure
- UMAP: non-linear, reveals cluster structure, but stochastic

### 9.2 Frontend: ROI Labeling page

**Layout:**
1. Session selector
2. Mean image with ROI overlay (one ROI highlighted)
3. ROI navigation: prev / next / jump-to-ROI
4. Calcium trace for current ROI (F_corrected and Fneu)
5. Feature panel: compact, AR, skew, soma_crop_frac, neuropil_corr
6. Label buttons: soma / dendrite / artefact / skip
7. Bootstrap prediction shown (e.g., "Predicted: dendrite (P=0.73)")
8. Progress bar: "Labeled 42/141 ROIs in this session"
9. Save button -> uploads roi_labels.npy to S3

---

## 10. Integration with Downstream Analysis

### 10.1 sync.h5 additions

```
/roi_class_prob   (n_rois, 3) float32    # classifier probabilities
/roi_class        (n_rois,) int8          # hard labels at default threshold
```

Attributes on `/roi_class`:
- `classifier_version`: str (e.g., "1.0")
- `soma_threshold`: float (default 0.5)
- `dendrite_threshold`: float (default 0.5)

### 10.2 Analysis filtering

All downstream analyses (tuning curves, decoding, etc.) should:
1. Load `roi_class_prob` from sync.h5
2. Apply `apply_threshold()` with the user-configured thresholds
3. Filter ROIs by class before computing metrics
4. Report results separately for soma and dendrite populations

Example:
```python
probs = sync['roi_class_prob'][:]
labels = apply_threshold(probs, soma_threshold=0.5)
soma_mask = labels == 1
dendrite_mask = labels == 2

# HD tuning for soma only
tuning_soma = compute_tuning_curves(dff[soma_mask], head_direction)
tuning_dend = compute_tuning_curves(dff[dendrite_mask], head_direction)
```

---

## 11. Open Questions for the User

**Q1: Labeling session selection.** I plan to select 6-8 sessions for manual
labeling, balanced across animals and cell types (Penk+ vs Penk-CamKII+).
Should I prioritize any specific sessions (e.g., ones you know have
particularly clear or tricky ROIs)?

**Q2: Suite2p GUI vs Streamlit labeling.** I'm recommending a custom Streamlit
labeling page because the Suite2p GUI only supports binary classification.
However, if you prefer to label in the Suite2p GUI (which you're already
familiar with), we could do a two-pass approach:
- Pass 1: label everything as cell vs artefact
- Pass 2: among cells, label soma vs dendrite
This would require saving two iscell.npy files per session. Which do you
prefer?

**Q3: Branching tree dendrites (type b).** You mentioned some dendrites are
branching trees, not just thin fragments. These may overlap with soma in
compact/aspect_ratio space. Should these be labeled as "dendrite" (since
they are dendrites anatomically), or would it be useful to have a fourth
class "dendrite_branch" to separate them? My recommendation is to keep it
as 3 classes (soma/dendrite/artefact) and let the threshold slider handle
ambiguous cases, but I want to confirm.

**Q4: Cross-section blob dendrites (type c).** You also mentioned some
dendrites appear as cross-section blobs (oblique dendritic shafts). These
will look very similar to soma morphologically (round, compact). How
common are these? If rare (<5% of dendrites), the classifier may not have
enough examples to learn them, and they would be best handled by manual
correction. If common, we may need additional features (e.g., amplitude --
cross-section dendrites may have smaller transients than soma).

**Q5: Footprint as feature.** Suite2p's `footprint` field (0-3) indicates
which detection algorithm found the ROI. In your data, footprint values are
distributed: 0 (sparsery initial, 29%), 1 (cellpose, 43%), 2 (sparsery
refined, 28%), 3 (manual add, <1%). Cellpose is biased toward soma shapes,
so `footprint=1` is a weak soma prior. I plan to include this as a
categorical feature (one-hot encoded). Any concern with this?

---

## 12. Implementation Timeline

| Phase | Task | Effort | Dependency |
|-------|------|--------|------------|
| 1 | Feature extraction module + tests | 1 day | None |
| 2 | Bootstrap label generation | 0.5 day | Phase 1 |
| 3 | Streamlit labeling page | 1 day | Phase 2 |
| 4 | Manual labeling (user) | 3-4 hours | Phase 3 |
| 5 | Train classifier + validation | 1 day | Phase 4 |
| 6 | Classification results page + threshold UI | 1 day | Phase 5 |
| 7 | sync.h5 integration | 0.5 day | Phase 5 |
| 8 | W&B logging + PCA/UMAP viz | 0.5 day | Phase 5 |

Total dev time: ~5.5 days. User labeling time: ~3-4 hours.
