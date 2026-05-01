# Soma / Dendrite / Artefact Classifier

## Overview

Each Suite2p ROI in the hm2p pipeline is classified into one of three
categories:

* **soma** — somatic ROI, used for HD tuning and population analyses.
* **dend** — dendritic ROI, kept in the dataset but typically excluded
  from the primary HD analysis.
* **artefact** — too small or too diffuse to be a real ROI; merged with
  `iscell=False` ROIs into the `non-cell` category in `ca.h5`.

The classifier returns calibrated per-ROI **probabilities**
(`p_soma`, `p_dend`, `p_artefact`) alongside the hard label, so that
ambiguous ROIs can be surfaced in the curation UI rather than silently
locked in by an opaque threshold.

## Implementation

The classifier framework lives in
[`src/hm2p/extraction/soma_classifier.py`](../src/hm2p/extraction/soma_classifier.py).

Two implementations conform to the `SomaClassifier` Protocol:

### Rule-based scorer (current default — provisional)

`RuleBasedClassifier` reproduces the legacy hand-tuned thresholds:

```text
radius < 2.0 OR compact < 0.1   →  artefact
aspect_ratio > 2.5              →  dendrite
otherwise                       →  soma
```

Each decision boundary is mapped onto a steep logistic function so that
calibrated probabilities are produced near the boundary and saturated
well past it. Activity features (`autocorr_halfwidth_s`, `fneu_corr`)
contribute additional evidence for `p_dend` when they are available; ROIs
with NaN activity features fall back to shape-only scoring.

By construction, on shape inputs alone the *argmax* of the rule-based
scorer matches `classify_roi_types` for every ROI — switching to the new
framework does not change the assigned label of any existing ROI. This
is checked in `tests/extraction/test_soma_classifier.py` with both a
deterministic sweep over the boundary regions and a hypothesis property
test.

The rule-based scorer is **provisional**. It cannot improve
discrimination over the legacy thresholds — by construction it cannot.
Its purpose is to expose calibrated `p_soma` so that ambiguous ROIs can
be flagged for manual curation; the long-term replacement is a
cross-validated logistic regression.

### Logistic regression classifier

`LogisticRegressionClassifier` is a thin wrapper around an sklearn
`Pipeline([StandardScaler, LogisticRegression])` that has been trained
offline on curated labels. The wrapper enforces the canonical class
order (`("soma", "dend", "artefact")`) regardless of how the underlying
estimator stored its classes.

`load_classifier` returns the logistic-regression classifier when a
fitted pickle exists at
`sourcedata/trackers/suite2p/soma_classifier.pkl`, and the rule-based
scorer otherwise (with a single warning logged).

## Feature extraction

[`src/hm2p/extraction/soma_features.py`](../src/hm2p/extraction/soma_features.py)
builds a single `pandas.DataFrame` with one row per ROI and columns:

| Feature                | Description                                                                     |
|------------------------|---------------------------------------------------------------------------------|
| `radius`               | Equivalent disk radius (Suite2p `stat[i]['radius']`)                            |
| `compact`              | Compactness, ranges in (0, 1] (Suite2p)                                         |
| `aspect_ratio`         | Major/minor axis ratio (Suite2p)                                                |
| `npix`                 | Number of pixels in footprint                                                   |
| `npix_norm`            | Suite2p's pixel-count z-score                                                   |
| `skew`                 | Suite2p trace skewness                                                          |
| `std`                  | Suite2p trace standard deviation                                                |
| `peak_to_noise_dff`    | `max(dF/F) / (1.4826 * MAD(dF/F))`; robust SNR proxy                            |
| `autocorr_halfwidth_s` | Lag (s) at which dF/F autocorrelation drops below 0.5; wider → slower kinetics |
| `fneu_corr`            | Spearman rank correlation between ROI dF/F and the mean Fneu trace              |

The dF/F trace is computed inline via a global eighth-percentile baseline
(Jia et al. 2011) so that feature extraction can run on raw Suite2p
arrays without re-doing the full Stage 4 pipeline.

## Manual ROI curation workflow

The Streamlit page
[`frontend/pages/roi_curation_page.py`](../frontend/pages/roi_curation_page.py)
is the in-pipeline replacement for the legacy Suite2p GUI labelling
workflow. It surfaces the ROIs that the classifier is most uncertain
about (default: `0.3 < p_soma < 0.7`) and lets the curator confirm or
override the model's argmax label one ROI at a time.

### CSV schema

Every saved label is appended to `metadata/roi_curation.csv` with the
columns:

| Column       | Description                                                         |
|--------------|---------------------------------------------------------------------|
| `session_id` | Canonical session identifier `YYYYMMDD_HH_MM_SS_<animal_id>`.       |
| `roi_index`  | Zero-based ROI index within the session's `dff` array.              |
| `label`      | One of `"soma"`, `"dend"`, `"artefact"`.                            |
| `curator`    | Free-form curator name (defaults to `$HM2P_CURATOR` or `$USER`).    |
| `timestamp`  | ISO-8601 UTC timestamp (no microseconds).                            |

The first three columns match the schema consumed by
`scripts/train_soma_classifier.py`, so the same file feeds both the
runtime label resolver and offline classifier training.

The CSV is **append-only**. Re-labelling an ROI never overwrites the
previous row; instead, a new row is appended with a fresh timestamp, and
[`hm2p.extraction.curation.load_latest_labels`](../src/hm2p/extraction/curation.py)
resolves duplicates on read by taking the row with the largest
timestamp.

### Runtime resolver

[`hm2p.extraction.curation.effective_roi_label(roi_qc, n_rois)`](../src/hm2p/extraction/curation.py)
returns per-ROI string labels following this resolution order:

1. The curated label, if `roi_qc/curated_label[i]` is one of `soma`,
   `dend`, `artefact`.
2. The argmax of `p_soma`, `p_dend`, `p_artefact` when all three are
   finite.
3. `"soma"` as the conservative fallback.

Downstream code that wants the curator's verdict should call this
helper instead of recomputing argmax from probabilities.

### Persisting curated labels into ca.h5

The "Apply curation to ca.h5" button on the curation page calls
[`apply_curation_to_ca_h5`](../src/hm2p/extraction/curation.py), which
reads the latest CSV labels for the selected session and writes a
string array `roi_qc/curated_label` of length `n_rois` into the local
`ca.h5`. Un-curated ROIs receive an empty string. The function does
**not** push back to S3 — uploading the curated `ca.h5` is a separate,
deliberate operation.

## Training a real classifier

The script
[`scripts/train_soma_classifier.py`](../scripts/train_soma_classifier.py)
trains a `LogisticRegressionClassifier` from curated labels and saves it
to disk via `joblib`. It accepts either:

* `--labels labels.csv` — a plain three-column CSV (`session_id`,
  `roi_index`, `label`); for hand-curated label sets created outside
  the curation page.
* `--curation-csv metadata/roi_curation.csv` — the append-only file
  produced by the curation page. The script resolves duplicates via
  `load_latest_labels` and ignores the `curator` and `timestamp`
  columns.

### Labelling workflow

1. Run the curation page (`streamlit run frontend/app.py` → ROI
   Curation) on every session whose `ca.h5` has soma classifier
   probabilities. The default filter `0.3 < p_soma < 0.7` surfaces the
   ambiguous ROIs first; widen to review additional cases.
2. Aim for ~200 confirmed labels across 2–3 representative sessions
   before training a classifier.
3. Run the training script:

   ```bash
   python -m scripts.train_soma_classifier \
       --curation-csv metadata/roi_curation.csv \
       --output sourcedata/trackers/suite2p/soma_classifier.pkl \
       --report-dir reports/soma_classifier/
   ```

   The script performs 5-fold stratified cross-validation, writes a
   per-fold metrics table (`cv_report.csv`), an aggregate confusion
   matrix (`confusion_matrix.csv`), and per-class feature coefficients
   (`feature_coefficients.csv`) to `--report-dir`.

4. Re-run Stage 4 on every session — the runtime path will pick up the
   new pickle automatically and write classifier-derived `p_soma` /
   `p_dend` / `p_artefact` arrays into each `ca.h5` file.

### Acceptance criteria

The classifier replaces the rule-based scorer once it is demonstrably
better:

* macro-F1 ≥ 0.85 on held-out folds, with no class falling below 0.7;
* confusion matrix shows fewer than 5 % artefact-as-soma errors;
* manual review on ~20 random ambiguous ROIs (lowest `p_soma`) confirms
  the labels look right.

Until those conditions are met, the rule-based scorer is the safer
default.

## References

* Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
  two-photon microscopy." bioRxiv. [doi:10.1101/061507](https://doi.org/10.1101/061507).
  https://github.com/MouseLand/suite2p
* Pedregosa et al. 2011. "Scikit-learn: Machine Learning in Python."
  *Journal of Machine Learning Research* 12:2825–2830.
  https://scikit-learn.org
* Jia H, Rochefort NL, Chen X, Konnerth A. 2011. "In vivo two-photon
  imaging of sensory-evoked dendritic calcium signals in cortical
  neurons." *Nature Protocols* 6:28–35.
  [doi:10.1038/nprot.2010.169](https://doi.org/10.1038/nprot.2010.169)
