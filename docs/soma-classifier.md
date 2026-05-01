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

## Training a real classifier

The script
[`scripts/train_soma_classifier.py`](../scripts/train_soma_classifier.py)
trains a `LogisticRegressionClassifier` from curated labels and saves it
to disk via `joblib`.

### Labelling workflow

1. Open Suite2p's GUI on a session with `ca.h5` already produced.
2. For ~200 ROIs across 2–3 representative sessions, manually mark each
   as soma, dendrite, or artefact (Suite2p's classifier UI lets you flip
   ROIs into `iscell=False` if they are clearly artefactual; for soma vs
   dend you may need to keep notes externally).
3. Export the labels to a CSV with columns `session_id`, `roi_index`,
   `label` — for example:

   ```csv
   session_id,roi_index,label
   20220804_13_52_02_1117646,0,soma
   20220804_13_52_02_1117646,1,artefact
   20220804_13_52_02_1117646,5,dend
   ...
   ```

4. Run the training script:

   ```bash
   python -m scripts.train_soma_classifier \
       --labels labels.csv \
       --output sourcedata/trackers/suite2p/soma_classifier.pkl \
       --report-dir reports/soma_classifier/
   ```

   The script performs 5-fold stratified cross-validation, writes a
   per-fold metrics table (`cv_report.csv`), an aggregate confusion
   matrix (`confusion_matrix.csv`), and per-class feature coefficients
   (`feature_coefficients.csv`) to `--report-dir`.

5. Re-run Stage 4 on every session — the runtime path will pick up the
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
