# Technical Report: 3-Way ROI Classification (Soma / Dendrite / Artefact)

## Context

Single-plane two-photon GCaMP calcium imaging in mouse retrosplenial cortex
(RSP), ~9.6 Hz, where soma and dendrite ROIs coexist in the same focal plane.
Suite2p detects all ROIs indiscriminately; the goal is to classify each as
**soma**, **dendrite**, or **artefact** post-hoc.

---

## 1. Suite2p's Built-in Capabilities

### 1.1 Default Classifier Architecture

Suite2p uses a **two-stage naive Bayes + logistic regression** classifier
(binary: cell / not-cell):

1. **Feature extraction:** Three features per ROI from `stat.npy`:
   - `skew` — skewness of the neuropil-corrected fluorescence trace
   - `compact` — ratio of observed mean pixel distance from ROI centre to
     expected distance for a disk of equal area (1.0 = perfect disk, >1 = less
     compact)
   - `npix_norm` — pixel count normalized by median ROI size (after soma crop)

2. **Non-parametric density estimation:** Each feature is binned into 100
   adaptive quantile bins. Within each bin, the fraction of "cell" labels is
   computed and Gaussian-smoothed (sigma=2 bins). This gives P(cell | feature
   value) for each feature independently.

3. **Log-likelihood ratio:** For each ROI, the log-odds ratio
   log[P(cell|x) / P(not-cell|x)] is computed per feature.

4. **Logistic regression:** A scikit-learn `LogisticRegression(C=100)` model
   combines the 3 log-likelihood features into a single probability.

**Source:** `suite2p/classification/classifier.py`, lines 10-163 (Suite2p
v0.14+, Stringer & Pachitariu, HHMI Janelia).

### 1.2 Can it do 3-way classification?

**No.** The classifier is strictly binary. The `iscell` array is float
{0.0, 1.0}, the logistic regression outputs a single probability, and
`classifier.run()` returns shape `(n_rois, 2)` — [binary label, probability].
There is no multi-class support in the codebase.

The Suite2p GUI also operates in binary mode only: ROIs are toggled between
"cell" and "not cell". There is no third category.

### 1.3 Can it be retrained?

Yes, for binary classification. The GUI provides:
- "Add current data to classifier" — augments training set with current
  manual labels
- "Build classifier" — creates a new classifier from multiple `iscell.npy`
  files

But always binary. To use it for dendrite detection, the old pipeline
(section 2) trained a **separate** dendrite-specific binary classifier.

### 1.4 Available ROI Features (stat.npy)

From `suite2p/detection/stats.py` (`roi_stats()` function):

| Feature | Description | Computation |
|---------|-------------|-------------|
| `npix` | Total pixel count of ROI | Direct count |
| `npix_soma` | Pixel count after soma_crop | Count of `soma_crop==True` pixels |
| `npix_norm` | Normalized pixel count (soma) | `npix_soma / median(npix_soma)` |
| `npix_norm_no_crop` | Normalized pixel count (full) | `npix / median(npix)` |
| `med` | Median centre [y, x] | Pixel closest to coordinate median |
| `compact` | Compactness | `mrs / mrs0`, clamped at min 1.0 |
| `mrs` | Mean radial spread | Mean distance of pixels from centre (normalized by diameter) |
| `mrs0` | Expected mrs for a disk | Mean distance for equal-area disk |
| `radius` | Major axis radius | From 2D Gaussian fit (via `fitMVGaus`) |
| `aspect_ratio` | Elongation | `2 * r_major / (r_major + r_minor)`, range [1.0, 2.0] |
| `footprint` | Functional spatial extent | 0 (sparsery) or 1 (cellpose) |
| `soma_crop` | Boolean mask per pixel | Radial profile thresholding in `soma_crop()` |
| `overlap` | Boolean mask per pixel | Pixels shared with other ROIs |
| `skew` | Trace skewness | Skewness of neuropil-corrected F trace |
| `std` | Trace std | Std of neuropil-corrected F trace |
| `lam` | Pixel weights | Fluorescence intensity weights |

The `soma_crop` operation (enabled by default: `ops["soma_crop"]=True`)
identifies the boundary where cumulative weighted area growth drops below
max/3. It effectively separates the soma core from dendritic extensions
**within a single ROI footprint**. This is a key feature for classification.

### 1.5 Cellpose Integration

Suite2p optionally uses Cellpose (Stringer et al. 2020) for anatomical
detection (`ops["detection"]["algorithm"] = "cellpose"`). Cellpose is trained
on somatic shapes and tends to detect round, soma-like structures. Cellpose
ROIs get `footprint=1` while sparsery ROIs get `footprint=0`. This could
serve as a very rough soma prior, but Cellpose does not classify dendrites.

### 1.6 The 2026 Suite2p Paper

Stringer & Pachitariu. "Extracting large-scale neural activity with Suite2p."
bioRxiv, 2026.02.04.703741v1, Feb 2026.

This paper describes GPU-accelerated motion correction and improved cell
detection. It does **not** introduce multi-class ROI classification or
soma/dendrite separation. The classifier remains binary.

---

## 2. Legacy Pipeline Approach

### 2.1 How the Old Pipeline Did It

The old `hm2p-analysis` pipeline (`old-pipeline/proc/proc_s2p.py`) used a
**dual-run** strategy:

1. **Run Suite2p with `classifier_soma.npy`** — detect soma ROIs
   - `ops["crop_soma"] = True` (default) — dendrites cropped before
     computing compactness
   - `ops["connected"] = True` — enforce connected masks

2. **Run Suite2p with `classifier_dend.npy`** — detect dendrite ROIs
   - Copies the registered binary from the soma run (no re-registration)
   - `ops["crop_soma"] = False` — do NOT crop dendrites
   - Uses separate classifier trained on dendrite labels

3. **Post-hoc validation** (`proc_ca.py` line 28-33): Check that no ROI is
   labeled as both soma and dendrite. If any are, raise an exception.

### 2.2 Analysis of Existing Classifiers

Both classifiers exist at `old-pipeline/s2p/classifier_soma.npy` and
`old-pipeline/s2p/classifier_dend.npy`.

**Training data:** 4413 ROIs (same set for both classifiers), same 3 features
(`skew`, `compact`, `npix_norm`).

| Category | N | Definition |
|----------|---|------------|
| Soma only | 311 | `soma_iscell=1, dend_iscell=0` |
| Dendrite only | 239 | `soma_iscell=0, dend_iscell=1` |
| Both (overlap) | 2 | `soma_iscell=1, dend_iscell=1` — classification error |
| Artefact | 3861 | `soma_iscell=0, dend_iscell=0` |

### 2.3 Feature Distributions by Category

```
               skew                compact             npix_norm
               mean   median       mean   median       mean   median
Soma (311):    4.253  4.138        1.033  1.016        1.651  1.551
Dend (239):    3.054  2.822        1.229  1.174        0.721  0.596
Artefact:      0.732  0.467        1.098  1.051        0.774  0.600
```

**Key discriminative patterns:**

- **Soma:** High skew (strong transients), very compact (~1.0), large
  normalized size (~1.5x median)
- **Dendrite:** Moderate skew, less compact (elongated), smaller normalized
  size
- **Artefact:** Low skew (no clear transients), variable compactness,
  variable size

The clearest separation is:
- **Soma vs everything:** `compact ≈ 1.0` AND `skew > ~2` AND
  `npix_norm > ~1.0`
- **Dendrite vs artefact:** `skew > ~1.5` (dendrites have genuine calcium
  transients, artefacts generally do not)
- **Soma vs dendrite:** `compact` is the primary discriminator
  (soma ≈ 1.02, dendrite ≈ 1.17)

### 2.4 Limitations of the Dual-Run Approach

1. **Redundant computation:** ROI detection runs twice on the same registered
   movie
2. **Inconsistent ROI sets:** The two runs may detect different ROIs (not the
   same set with different labels)
3. **No artefact class:** Neither classifier explicitly labels artefacts —
   they are the leftover "not cell" from both classifiers
4. **Limited features:** Only 3 features used; `aspect_ratio` is not used
   despite being informative
5. **Small training set:** Only 311 soma + 239 dendrite labels across ~4400
   ROIs

---

## 3. Community and Published Approaches

### 3.1 SUBPREP: Subcellular Preprocessing Toolbox

Jiang, Zhao & Sheffield. "A Preprocessing Toolbox for 2-Photon Subcellular
Calcium Imaging." eNeuro 12(5), 2025.
doi:10.1523/ENEURO.0565-24.2025

This toolbox addresses axon/dendrite ROI processing but focuses on:
- **Frequency-based ROI filtering** (FFT bandpass on calcium traces)
- **Motion artefact detection** (PCA + level change detection)
- **ROI grouping** (clustering subcellular ROIs from the same neuron)

It does **not** provide a soma vs dendrite classifier. It assumes ROIs are
already known to be subcellular (e.g., from a dedicated dendrite imaging
plane). Not directly applicable to our single-plane mixed-compartment problem.

### 3.2 CaImAn

CaImAn (Giovannucci et al. 2019) uses a 4-layer CNN to classify spatial
footprints as "cell" or "not cell". The CNN is trained on somatic shapes.
For dendritic/axonal data, CaImAn uses `sparse_nmf` initialization but does
not provide a soma vs dendrite classifier. CaImAn's approach to dendrites is
fundamentally different: it detects dendritic components when explicitly
configured for subcellular imaging, rather than classifying mixed-compartment
ROIs post-hoc.

### 3.3 Voelcker et al. 2023 (RSP Dendrite/Soma/Spine)

Voelcker, Bhatt & Bhatt. "Egocentric processing of items in spines, dendrites,
and somas in the retrosplenial cortex." Neuron, 2023.
doi:10.1016/j.neuron.2023.11.016

This is the most relevant published work — subcellular imaging in RSP with
Suite2p. Their approach:
- Suite2p for motion correction and ROI detection
- **Manual curation in Suite2p GUI** to classify ROIs as soma, dendrite,
  or spine
- Dendrites grouped by temporal correlation (highly correlated segments
  clustered into one dendrite)
- No automated 3-way classifier — purely manual inspection by human
  operators using the Suite2p GUI

### 3.4 GraFT: Morphology-Free Analysis

Adam, Bhatt et al. "Fast and accessible morphology-free functional
fluorescence imaging analysis." PLOS Computational Biology, 2026 /
bioRxiv 2025.04.15.648462.

GraFT uses graph-based temporal dictionary learning — identifies neural
components by shared temporal activity rather than spatial morphology.
Potentially useful for detecting dendrites that share activity with their
soma, but this is a fundamentally different detection approach, not a
classifier for existing Suite2p ROIs. Would require replacing the entire
detection pipeline.

### 3.5 Allen Institute (aind-ophys-extraction)

The Allen Institute's `aind-ophys-extraction` pipeline combines Cellpose
and Suite2p but uses the standard binary classifier. No 3-way classification.

---

## 4. Feature Engineering for Custom Classifier

### 4.1 Morphological Features (from stat.npy)

| Feature | Soma | Dendrite | Artefact |
|---------|------|----------|----------|
| `compact` | ~1.0 (round) | >1.1 (elongated) | Variable |
| `aspect_ratio` | ~1.0 (circular) | >1.3 (elongated) | Variable |
| `npix_norm` | ~1.5 (large) | ~0.7 (small) | Variable |
| `npix` | Large (~100-300 px) | Small (~30-100 px) | Variable |
| `radius` | ~6-8 px | ~3-5 px | Variable |
| `soma_crop` ratio | ~1.0 (all pixels in soma) | <0.5 (many pixels outside soma core) | Variable |

**Derived features to compute:**
- `soma_crop_fraction = npix_soma / npix` — fraction of pixels within the
  soma boundary. Soma: ~0.8-1.0, Dendrite: ~0.3-0.6
- `eccentricity` — from the 2D Gaussian fit covariance matrix
  (available from `fitMVGaus`)
- `perimeter / area ratio` — proxy for shape complexity
- `solidity` — area / convex hull area (Suite2p may compute this)

### 4.2 Temporal Features (from F.npy, Fneu.npy)

| Feature | Soma | Dendrite | Artefact |
|---------|------|----------|----------|
| `skew` | High (>2) | Moderate (1-3) | Low (<1) |
| `std` | Moderate | Moderate | Low or very high |
| `transient_amplitude` | Large | Moderate | Small/absent |
| `transient_frequency` | Regular | Regular | Irregular/absent |
| `neuropil_correlation` | Low | Low-moderate | High (artefact is neuropil) |

**Derived temporal features to compute:**
- `mean_transient_amplitude` — mean peak amplitude of detected transients
- `event_rate` — transients per minute
- `decay_time` — half-decay time of transients (soma slower due to volume)
- `neuropil_correlation` — Pearson correlation between ROI trace and its
  neuropil ring. Artefacts tend to have high neuropil correlation
- `SNR` — signal (mean transient amplitude) / noise (std of non-event periods)
- `kurtosis` — excess kurtosis of the trace

### 4.3 Spatial Features (computed from pixel masks)

- `convexity` — convex hull area / actual area
- `n_connected_components` — number of connected components in the ROI mask
  (dendrites may be fragmented)
- `longest_axis_length` — maximum Feret diameter

---

## 5. Recommended Approach

### 5.1 Primary Recommendation: Custom 3-Way Classifier

Train a lightweight supervised classifier using Suite2p features. This is the
most practical approach given:
- 26 sessions of existing data with old-pipeline soma/dendrite labels
- Feature distributions show clear separability (section 2.3)
- No existing tool provides 3-way classification out of the box

#### Architecture: Random Forest or Gradient Boosted Trees

**Why not Suite2p's naive Bayes + logistic regression?**
- Hardcoded for binary classification
- Only uses 3 features
- Modifying it for 3-way would require forking Suite2p

**Why Random Forest / XGBoost?**
- Handles multi-class natively
- Works well with small training sets (<1000 samples)
- Feature importance is interpretable
- Robust to feature scale and outliers
- scikit-learn provides well-tested implementations
- No hyperparameter tuning needed for Random Forest (good defaults)

**Why not deep learning (CNN on ROI images)?**
- Would require rendering each ROI as an image patch
- Much more training data needed (~1000+ per class)
- Less interpretable
- Overkill for a problem with good hand-crafted features

#### Feature Set (8-12 features)

**From stat.npy (no additional computation):**
1. `compact` — primary morphological discriminator
2. `aspect_ratio` — elongation measure
3. `npix_norm` — normalized size
4. `skew` — trace skewness (activity indicator)
5. `radius` — ROI radius from Gaussian fit
6. `std` — trace standard deviation

**Computed from stat.npy:**
7. `soma_crop_fraction = npix_soma / npix` — key discriminator
8. `npix_norm_ratio = npix_norm / npix_norm_no_crop` — how much does soma
   cropping change the size?

**Computed from F.npy / Fneu.npy (optional but valuable):**
9. `neuropil_correlation` — correlation(F_corrected, Fneu)
10. `snr` — signal-to-noise ratio
11. `event_rate` — transient frequency (from existing event detection code)
12. `kurtosis` — trace kurtosis

#### Training Data

**Option A: Use old-pipeline labels (recommended first step)**
- 311 soma labels + 239 dendrite labels + 3861 artefact labels
- Already available in `classifier_soma.npy` and `classifier_dend.npy`
- Caveat: labels from a single Suite2p run — feature values may differ
  slightly from current Suite2p version
- Need to re-extract features from current Suite2p stat.npy and re-label

**Option B: Manual labeling in Suite2p GUI (gold standard)**
- Load each session's Suite2p output in the GUI
- Manually classify each ROI as soma (1), dendrite (2), or artefact (0)
- Export labels as a custom `.npy` file
- 5-10 sessions (of 26) with ~100-500 ROIs each should suffice
- Estimated effort: 2-4 hours of manual labeling

**Option C: Bootstrap from Option A, refine with Option B**
- Train initial classifier on old-pipeline labels
- Apply to all sessions
- Manually review and correct misclassifications
- Retrain on corrected labels

### 5.2 Implementation Plan

```python
# Pseudocode for the 3-way classifier

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

def extract_features(stat, F, Fneu, neucoeff=0.7):
    """Extract classification features from Suite2p outputs.

    Parameters
    ----------
    stat : np.ndarray of dicts
        Suite2p stat.npy
    F : np.ndarray, shape (n_rois, n_frames)
        Raw fluorescence traces
    Fneu : np.ndarray, shape (n_rois, n_frames)
        Neuropil fluorescence traces
    neucoeff : float
        Neuropil subtraction coefficient

    Returns
    -------
    features : np.ndarray, shape (n_rois, n_features)
    feature_names : list of str
    """
    n_rois = len(stat)
    Fcorr = F - neucoeff * Fneu

    features = np.zeros((n_rois, 10))
    feature_names = [
        'compact', 'aspect_ratio', 'npix_norm', 'skew', 'radius',
        'std', 'soma_crop_fraction', 'npix_norm_ratio',
        'neuropil_correlation', 'kurtosis'
    ]

    for i in range(n_rois):
        s = stat[i]
        features[i, 0] = s.get('compact', 1.0)
        features[i, 1] = s.get('aspect_ratio', 1.0)
        features[i, 2] = s.get('npix_norm', 1.0)
        features[i, 3] = s.get('skew', 0.0)
        features[i, 4] = s.get('radius', 5.0)
        features[i, 5] = s.get('std', 0.0)

        npix = s.get('npix', 1)
        npix_soma = s.get('npix_soma', npix)
        features[i, 6] = npix_soma / max(npix, 1)  # soma_crop_fraction

        npix_norm = s.get('npix_norm', 1.0)
        npix_norm_no_crop = s.get('npix_norm_no_crop', npix_norm)
        features[i, 7] = npix_norm / max(npix_norm_no_crop, 1e-6)

        # Temporal features
        fc = Fcorr[i]
        fn = Fneu[i]
        features[i, 8] = np.corrcoef(fc, fn)[0, 1]  # neuropil corr
        from scipy.stats import kurtosis
        features[i, 9] = kurtosis(fc)

    return features, feature_names


def train_classifier(features, labels, n_splits=5):
    """Train and evaluate a 3-way ROI classifier.

    Parameters
    ----------
    features : np.ndarray, shape (n_rois, n_features)
    labels : np.ndarray, shape (n_rois,)
        0=artefact, 1=soma, 2=dendrite

    Returns
    -------
    clf : RandomForestClassifier
        Trained classifier
    """
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',  # handle class imbalance
        random_state=42,
        n_jobs=-1
    )

    # Cross-validated evaluation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, features, labels, cv=cv, scoring='f1_macro')
    print(f"CV F1 (macro): {scores.mean():.3f} +/- {scores.std():.3f}")

    # Fit on full data
    clf.fit(features, labels)

    return clf
```

### 5.3 Validation Strategy

1. **Cross-validation:** 5-fold stratified CV on labeled data. Report F1
   per class and confusion matrix.

2. **Session holdout:** Train on N-1 sessions, test on held-out session.
   This tests generalization across imaging conditions.

3. **Manual review:** For each session, visualize classifier predictions
   overlaid on the mean image. Manually review edge cases.

4. **Feature importance:** Use Random Forest `feature_importances_` to
   verify that biologically meaningful features (compact, aspect_ratio,
   skew) dominate.

5. **Comparison with old pipeline:** Run on sessions that have old-pipeline
   soma/dendrite labels. Measure agreement (Cohen's kappa).

### 5.4 Alternative: Threshold-Based Rules (Simpler but Rigid)

Given the feature distributions from the old classifiers:

```python
def classify_roi_simple(stat_entry):
    """Simple threshold-based 3-way classification.

    Based on distributions from 4413 labeled ROIs.
    """
    compact = stat_entry.get('compact', 1.0)
    skew = stat_entry.get('skew', 0.0)
    npix_norm = stat_entry.get('npix_norm', 1.0)
    aspect_ratio = stat_entry.get('aspect_ratio', 1.0)

    # Artefacts: low activity (low skew)
    if skew < 1.0:
        return 'artefact'

    # Soma: compact, active, reasonably sized
    if compact < 1.05 and skew > 2.0 and npix_norm > 0.8:
        return 'soma'

    # Dendrite: less compact, active
    if compact > 1.1 or aspect_ratio > 1.3:
        if skew > 1.0:
            return 'dendrite'

    # Ambiguous — default to artefact
    return 'artefact'
```

This is a starting point only. The thresholds need tuning on the actual data,
and a proper classifier (section 5.1) will perform much better on edge cases.

---

## 6. What NOT to Do

1. **Do not run Suite2p twice** (as the old pipeline did). This is wasteful
   and produces inconsistent ROI sets. Run detection once, then classify
   post-hoc.

2. **Do not use Suite2p's binary classifier directly.** It cannot
   distinguish soma from dendrite — only cell from not-cell.

3. **Do not use CaImAn for this.** Its CNN classifier is also binary
   (cell/not-cell) and trained on somatic shapes.

4. **Do not use Cellpose as a classifier.** Cellpose detects soma-like
   shapes but does not classify — it segments. It would miss dendrites
   entirely.

5. **Do not use deep learning without sufficient training data.** A CNN
   on ROI image patches would need ~500+ examples per class and is overkill
   for this problem.

6. **Do not ignore the `soma_crop` parameter.** Setting
   `ops["soma_crop"]=False` during detection changes how `compact` and
   `aspect_ratio` are computed. For the 3-way classifier, run detection with
   `soma_crop=True` (default) so that both the cropped and uncropped features
   are available.

---

## 7. Summary of Recommendations

| Priority | Approach | Effort | Quality |
|----------|----------|--------|---------|
| 1 (recommended) | Random Forest on Suite2p features | Medium (1-2 days) | High |
| 2 (quick start) | Threshold rules on compact + skew | Low (hours) | Medium |
| 3 (gold standard) | Manual curation + RF retraining | High (days) | Highest |
| 4 (avoid) | Dual Suite2p run (old pipeline) | High | Medium |
| 5 (avoid) | CNN on ROI patches | Very high | Unknown |

### Recommended Implementation Order

1. **Immediate:** Extract all available features from existing Suite2p stat.npy
   files. Compute derived features (soma_crop_fraction, neuropil_correlation).

2. **Day 1:** Convert old-pipeline labels into 3-way labels (soma=1, dend=2,
   artefact=0) using the two classifier files. Train Random Forest. Evaluate
   with cross-validation.

3. **Day 2:** Apply classifier to all 26 sessions. Build a review interface
   (Streamlit page) showing ROI masks colored by predicted class, with
   confidence scores. Manually review 2-3 sessions.

4. **Day 3-4:** Correct misclassifications, retrain, iterate. Aim for >90%
   agreement with manual labels.

5. **Ongoing:** Add a `roi_class` column to the sync.h5 / analysis outputs.
   Filter analyses by ROI class (e.g., HD tuning curves for soma only).

---

## 8. References

- Pachitariu M, Stringer C, et al. "Suite2p: beyond 10,000 neurons with
  standard two-photon microscopy." bioRxiv 061507, 2017.
  doi:10.1101/061507
  GitHub: https://github.com/MouseLand/suite2p

- Stringer C, Pachitariu M. "Extracting large-scale neural activity with
  Suite2p." bioRxiv 2026.02.04.703741v1, 2026.
  doi:10.64898/2026.02.04.703741

- Stringer C, Wang T, Michaelos M, Pachitariu M. "Cellpose: a generalist
  algorithm for cellular segmentation." Nature Methods 18:100-106, 2021.
  doi:10.1038/s41592-020-01018-x
  GitHub: https://github.com/MouseLand/cellpose

- Giovannucci A, et al. "CaImAn: an open source tool for scalable calcium
  imaging data analysis." eLife 8:e38173, 2019.
  doi:10.7554/eLife.38173
  GitHub: https://github.com/flatironinstitute/CaImAn

- Jiang A, Zhao C, Sheffield MEJ. "A Preprocessing Toolbox for 2-Photon
  Subcellular Calcium Imaging." eNeuro 12(5), 2025.
  doi:10.1523/ENEURO.0565-24.2025

- Voelcker B, Bhatt R, Bhatt DK. "Egocentric processing of items in
  spines, dendrites, and somas in the retrosplenial cortex." Neuron, 2023.
  doi:10.1016/j.neuron.2023.11.016

- Adam Y, Bhatt D, et al. "Fast and accessible morphology-free functional
  fluorescence imaging analysis." PLOS Computational Biology, 2026.
  bioRxiv 2025.04.15.648462.
