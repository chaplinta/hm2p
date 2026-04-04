# Dipoppa et al. 2018 -- Detailed Summary

## Citation

Dipoppa M, Ranson A, Krumin M, Pachitariu M, Carandini M, Harris KD. 2018. "Vision and Locomotion Shape the Interactions between Neuron Types in Mouse Visual Cortex." *Neuron* 98(3):602-615.e8. doi:10.1016/j.neuron.2018.03.037

**Affiliations:** Institute of Neurology, UCL; Institute of Ophthalmology, UCL; Janelia Research Campus, HHMI.

---

## Overview

This study uses two-photon calcium imaging to record from four genetically identified neuron types (Pyr, Pvalb, Vip, Sst) in mouse primary visual cortex (V1) during locomotion and visual stimulation. The central finding is that the disinhibitory circuit model (Vip inhibits Sst, disinhibiting Pyr) holds for spontaneous activity in darkness but fails when visual stimuli are present -- locomotion increases Sst responses to large stimuli and Vip responses to small stimuli. A recurrent neural field model captures these effects by allowing locomotion to modulate feedforward and recurrent synaptic weights. The paper also makes an important methodological contribution by developing and validating a neuropil correction procedure for two-photon GCaMP imaging.

---

## Key Findings and Arguments

### 1. Locomotion effects on baseline activity depend on cell type and visual context

- **Pyr cells:** Weak, diverse effects of locomotion on baseline activity (mean correlation with speed ~0.03).
- **Pvalb cells:** Effects depend on cortical depth -- negative correlation in superficial L2/3, positive in deeper L2/3. This depth dependence reconciles previously conflicting reports.
- **Vip cells:** Strong positive correlation with locomotion speed (mean ~0.27), consistent across studies.
- **Sst cells:** The key reconciliation finding. Locomotion increases Sst baseline activity when mice view a grey screen (r_gray = 0.18) but decreases it in complete darkness (r_dark = -0.07). The same cell can show opposite modulation depending on visual conditions.

### 2. Locomotion modulates visual responses differently from baseline activity

- Locomotion increases visual responses across all cell types, regardless of the direction of its effect on baseline activity.
- The correlation between locomotion modulation of baseline (M_B) and evoked activity (M_R) is weak or non-significant in all cell types.
- This dissociation challenges simple gain modulation models and suggests distinct mechanisms for locomotion effects on spontaneous versus evoked activity.

### 3. Recurrent network model

- A four-population (Pyr, Pvalb, Sst, Vip) recurrent neural field model predicts each cell type's activity from the measured activity of the other types.
- Locomotion effects require modulation of both feedforward synaptic weights (increased) and recurrent weights, not a simple multiplicative gain.

### 4. Neuropil correction is essential for accurate results

The paper explicitly states that without neuropil correction, "one would observe an artifactual negative correlation of fluorescence with running speed, particularly in image regions with weak GCaMP expression." This is because hemodynamic absorption during locomotion (increased blood flow darkens the tissue at GCaMP emission wavelengths) contaminates the fluorescence signal.

---

## Neuropil Contamination and Correction

This paper provides one of the most detailed descriptions of neuropil correction methodology in the literature:

### The problem

- Out-of-focus GCaMP fluorescence from neuropil surrounding an ROI contaminates the signal attributed to individual neurons.
- This contamination is particularly problematic when the neuropil itself is modulated by stimuli or behaviour (e.g., locomotion), because it introduces a systematic bias correlated with the experimental variable of interest.
- In regions with weak GCaMP expression, the neuropil contamination can dominate the signal, producing artifactual correlations.

### The correction method (adapted from Peron et al. 2015)

1. **Neuropil mask definition:** A ring-shaped mask is defined around each ROI, extending up to 35 um from the ROI border and excluding pixels belonging to other detected cells.
2. **Correction factor estimation:** For each cell, the relationship between neuropil fluorescence N(t) and the lower envelope of somatic fluorescence F(t) is estimated. Neuropil signals are binned into 20 intervals, and the 5th percentile of somatic fluorescence is computed for each bin. Linear regression on these percentiles yields a per-cell correction coefficient a_i.
3. **Population averaging:** Because densely firing cells may have correlated somatic and neuropil signals (making the lower-envelope method unreliable), the correction factor a_exp is computed by averaging a_i only over cells with high skewness (>4, indicating sparse firing).
4. **Correction:** The corrected fluorescence is F(t) - a_exp * N(t).
5. **Default value:** In experiments where only interneurons (low skewness cells) express GCaMP, the mean correction factor from all other experiments is used: a_exp = 0.82.

### Validation

- The authors confirm that Sst cell locomotion modulation results are not due to neuropil contamination by repeating measurements in mice expressing GCaMP only in Sst cells (Sst-IRES-Cre + Cre-dependent GCaMP6m). The positive correlation with running speed persists in both cell bodies and neuropil.
- Putative Pyr cells were identified by fluorescence skewness (>2.7) rather than by genetics alone, and results were consistent across all GCaMP expression methods.

### The a_exp = 0.82 value

This is notably higher than the commonly used Suite2p default of 0.7. The Dipoppa et al. method estimates the correction factor empirically from the data using the lower-envelope approach, rather than assuming a fixed value. A factor of 0.82 implies that approximately 82% of the neuropil signal measured in the neuropil mask is present within the somatic ROI.

---

## Relevance to hm2p

### 1. Neuropil correction coefficient matters for HD tuning

The default Suite2p neuropil correction uses r = 0.7 (i.e., F_corrected = F_soma - 0.7 * F_neuropil). Dipoppa et al. estimate 0.82 empirically. In the hm2p dataset, the correct coefficient may differ from both values because:

- The head-mounted two-photon microscope has a different PSF than benchtop systems.
- GCaMP expression density in RSP may differ from V1.
- The imaging depth may differ.

The lower-envelope method described in this paper should be applied to the hm2p data to empirically estimate the appropriate correction factor, rather than relying on the Suite2p default.

### 2. Locomotion-dependent neuropil contamination

Dipoppa et al. show that without neuropil correction, locomotion produces artifactual negative fluorescence correlations due to hemodynamic absorption. In the hm2p rose maze, mice alternate between locomotion and stationary periods. If neuropil correction is inadequate, this hemodynamic effect could create apparent differences in neural activity between movement and rest that are artifactual. Since locomotion speed and head direction are correlated (mice turn their heads while moving), this could contaminate HD tuning measurements.

### 3. Cell-type-specific validation is essential

The paper demonstrates that the same neuropil correction factor cannot be blindly applied to all cell types. Densely firing cells (like some interneurons) require different treatment than sparsely firing cells. For hm2p, if Penk+ and Penk-CamKII+ populations differ in firing rate distributions, the neuropil correction quality may differ between populations. This should be tested by examining the lower-envelope relationship separately for each population.

### 4. Light vs dark neuropil changes

Dipoppa et al. show that Sst cell modulation by locomotion reverses between grey-screen and darkness conditions. This demonstrates that neuropil signals are themselves modulated by visual context. In the hm2p light/dark paradigm, neuropil signals may change between light and dark epochs (e.g., visual cortex input to RSP is present in light but absent in dark). Any change in neuropil correction quality between conditions could create artifactual differences in tuning properties.

### 5. Skewness-based cell identification

The paper uses fluorescence skewness > 2.7 to identify putative pyramidal cells among unlabeled neurons. This approach may be applicable to hm2p for quality control of ROI classification -- genuine somatic ROIs should have higher skewness than dendrite-contaminated or neuropil-dominated ROIs.

---

## Key References Cited

| Reference | Relevance |
|---|---|
| Peron et al. 2015 (Neuron) | Neuropil correction method adapted by this paper |
| Chen et al. 2013 (Nature) | GCaMP6 characterisation, neuropil correction context |
| Fu et al. 2014 (Cell) | Sst cell locomotion effects (darkness condition) |
| Polack et al. 2013 (Nat Neurosci) | Sst cell locomotion effects (grey screen condition) |
| Pakan et al. 2016 (Curr Biol) | Visual context dependence of locomotion modulation |
| Niell & Stryker 2010 (Neuron) | Locomotion modulation of V1 activity |
