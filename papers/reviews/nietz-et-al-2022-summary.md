# Nietz et al. 2022 -- Detailed Summary

## Citation

Nietz AK, Popa LS, Streng ML, Carter RE, Kodandaramaiah SB, Ebner TJ. 2022. "Wide-Field Calcium Imaging of Neuronal Network Dynamics In Vivo." *Biology* 11(11):1601. doi:10.3390/biology11111601

**Affiliations:** Department of Neuroscience, University of Minnesota; Department of Mechanical Engineering, University of Minnesota.

---

## Overview

This review surveys the use of wide-field (mesoscopic) calcium imaging for studying neuronal network dynamics in vivo in behaving mice. It covers the technical basis of single-photon epifluorescence imaging with GECIs, the analytical methods for decomposing and interpreting wide-field data (ICA, NMF, functional connectivity, GLMs), and representative findings from studies of motor control, learning, decision-making, and visual processing. The review also addresses the limitations of the approach, including neuropil contamination, hemodynamic confounds, and limited spatial and temporal resolution.

---

## Key Findings and Arguments

### 1. Wide-field imaging provides mesoscopic-scale coverage

- Wide-field calcium imaging uses single-photon excitation and CMOS cameras to record from large cortical areas (multiple mm^2) simultaneously in head-fixed mice.
- Typical frame rates are 20--40 Hz, limited by GCaMP6f kinetics.
- The signal represents summed activity from a 3D voxel including dendrites, somata, and axons from multiple neurons and cortical layers, predominantly weighted toward L2/3.

### 2. GCaMP properties and limitations

- GCaMP6f can detect single action potentials but has decay times >100 ms.
- Saturation and kinetics limit the ability to resolve high-frequency firing.
- GCaMP7 and GCaMP8 families offer improved kinetics and sensitivity.
- Red-shifted indicators (RCaMPs) allow deeper imaging and compatibility with optogenetics.

### 3. Hemodynamic contamination requires correction

- Blood flow increases with neuronal activation, and oxygenated haemoglobin absorbs at approximately 530 nm (overlapping GCaMP emission), darkening the fluorescence signal.
- Dual-wavelength imaging is the preferred correction: interleaving excitation at the GCaMP-active wavelength (~470 nm) with a calcium-independent isosbestic wavelength (~405 nm) and subtracting the hemodynamic component.
- Other methods include spatial filtering, ICA-based removal, and vasculature masking.

### 4. Analytical approaches for decomposing wide-field data

- **ROI-based:** Manual or atlas-aligned regions of interest; simple but subjective.
- **PCA/SVD:** Orthogonal decomposition; can produce spatially delocalised components that span multiple brain regions.
- **Spatial ICA (sICA):** Non-orthogonal; separates data into maximally statistically independent spatial components; better for identifying localised functional regions.
- **LocaNMF (Localised semi-NMF):** Restricts spatial components to anatomical atlas regions; improves interpretability and cross-subject comparability.
- **Functional connectivity:** Pairwise correlation matrices between ROIs; graph theory metrics (centrality, modularity) quantify network architecture.
- **GLMs:** Relate calcium signals to behavioural parameters; can quantify concurrent influences of multiple variables.

### 5. Behaviour engages widespread cortical areas

- Even simple motor tasks (e.g., unilateral reach-to-grasp) produce bilateral, cortex-wide activation.
- Locomotion modulates activity across multiple cortical areas including visual, somatosensory, and retrosplenial cortex. During locomotion, retrosplenial cortex increases in centrality while motor and somatosensory areas decrease.
- Learning compresses cortical activation sequences and increases the causal influence of premotor area M2.

---

## Neuropil Contamination

The review addresses neuropil contamination as a fundamental limitation of wide-field imaging:

### The problem at mesoscopic scale

- Wide-field single-photon imaging "captures activity primarily from layers II and III of the cerebral cortex and the signal represents the combined neuropil activity in a region."
- Unlike two-photon imaging, wide-field imaging has no optical sectioning capability. The measured fluorescence at each pixel is a summation of signals from all structures within the excitation light cone, including neuropil across multiple cortical layers.
- Each pixel is "a mixture of multisource signals from dendrites, axons, and somata from different neurons as well as different cortical layers."

### Why this matters

- The signal cannot be attributed to individual neurons without additional constraints (sparse labelling, soma-restricted expression).
- Neuropil signals include both local processing (dendrites, axon collaterals) and long-range afferent activity (axonal projections from distant areas). The contribution of each is unknown for any given pixel.
- Brain motion, baseline fluctuations, and photonic noise add further contamination.

### Comparison with two-photon imaging

- The review notes that standard two-photon imaging achieves much better axial resolution but still suffers from neuropil contamination at the single-cell level (not discussed in detail in this paper).
- It highlights that one-photon widefield imaging "can be contaminated by neuropil and hemodynamic signal" (citing Waters 2020 and Valley et al. 2021).

---

## Relevance to hm2p

### 1. Context for understanding neuropil contamination across imaging modalities

While the hm2p project uses two-photon (not wide-field) imaging, this review provides context for why neuropil contamination is a universal concern across calcium imaging methods. The physical basis differs (axial resolution vs. no optical sectioning), but the consequence is the same: signals attributed to individual ROIs contain contributions from surrounding structures.

### 2. Retrosplenial cortex during locomotion

The review reports that during locomotion, retrosplenial cortex increases in functional connectivity centrality (from the Nietz/Ebner lab's own wide-field data). This is relevant to hm2p because it suggests that RSP network state changes during movement, which could affect both neuropil signals and somatic activity in the imaging field.

### 3. Analytical methods for population-level analyses

The review's discussion of ICA, NMF, and functional connectivity methods is relevant to hm2p population-level analyses. However, these methods are designed for mesoscopic (multi-region) data and may not apply directly to single-plane two-photon imaging of a local RSP population.

### 4. Hemodynamic contamination in one-photon imaging

Although hm2p uses two-photon imaging (which is less susceptible to hemodynamic absorption because excitation and emission wavelengths differ), the review's discussion of hemodynamic confounds highlights that any component of the signal pathway that involves visible-wavelength photons (GCaMP emission at ~510 nm) can be attenuated by haemoglobin absorption. In two-photon imaging, this effect is much smaller but not zero.

### 5. Temporal resolution comparison

The review notes wide-field imaging at 20--40 Hz is limited by GCaMP kinetics. The hm2p dataset at 9.6 Hz is substantially slower than this, further limiting temporal resolution for spike inference and creating longer integration windows during which neuropil contamination accumulates.

---

## Key References Cited

| Reference | Relevance |
|---|---|
| Waters 2020 | 1P imaging contamination by neuropil and hemodynamics |
| Valley et al. 2021 | Hemodynamic contamination in widefield imaging |
| Stringer et al. 2019 | Large-scale 2P cortical recordings, behaviour-related activity |
| Musall et al. 2019 | Widespread behavioural modulation of cortical activity |
| Chen et al. 2013 (Nature) | GCaMP6 properties |
| Pachitariu et al. 2017 | Suite2p for ROI extraction and motion correction |
