# Stringer et al. 2025 -- Detailed Summary

## Citation

Stringer C, Zhong L, Syeda A, Du F, Kesa M, Pachitariu M. 2025. "Rastermap: a discovery method for neural population recordings." *Nature Neuroscience* 28(1):201-212. doi:10.1038/s41593-024-01783-4

**Affiliations:** Howard Hughes Medical Institute, Janelia Research Campus, Ashburn, VA, USA.

---

## Overview

This paper introduces Rastermap, a visualisation and sorting method for large-scale neural recordings. Rastermap arranges neurons along a one-dimensional axis such that neurons with similar activity patterns are placed nearby, then displays them as a sorted raster plot. The method is benchmarked against t-SNE, UMAP, and other embedding algorithms on realistic simulations, and then applied to recordings of tens of thousands of neurons from mouse cortex during spontaneous, stimulus-evoked, and task-evoked epochs, as well as to electrophysiological recordings across species and to artificial neural networks. The paper is primarily a methods contribution for neural data visualisation and discovery, with calcium imaging data processed using the Suite2p pipeline, which includes neuropil correction as a standard step.

---

## Key Findings and Arguments

### 1. Rastermap algorithm

- Neurons are first clustered (k-means, typically 100 clusters).
- An asymmetric similarity matrix is computed between clusters using the peak cross-correlation at non-negative time lags. The asymmetry ensures that clusters with earlier activity are placed toward the bottom of the raster.
- The similarity matrix is sorted to match a predefined target matrix that combines a global power-law structure with a local "travelling salesman" sequential structure.
- Clusters are upsampled and individual neurons are assigned to positions by correlation with upsampled cluster centres.
- "Superneurons" (averages of nearby sorted neurons) are used for visualisation when neuron counts are large.

### 2. Benchmarking

- On simulations containing sequential firing, sensory tuning, sustained responses, and power-law noise, Rastermap outperforms t-SNE, UMAP, Isomap, Laplacian eigenmaps, hierarchical clustering, and PCA in correctly ordering neurons and maintaining module integrity.
- Rastermap is more consistent across random seeds than t-SNE.
- The algorithm runs in under 2 minutes on datasets with tens of thousands of neurons.

### 3. Application to cortical recordings

- **Virtual reality task (66,318 neurons, visual cortex):** Rastermap reveals two large populations encoding different corridors (rewarding and non-rewarding), plus populations encoding the grey space between corridors and reward-related events.
- **Spontaneous activity (34,086 neurons, sensorimotor cortex):** Sorted neurons show clear behavioural receptive fields -- groups of neurons whose activity correlates with specific movements (running, eye movements, whisking, nose movements). Behavioural prediction of neural activity from these variables is substantial.
- The method also works on electrophysiological recordings (rat hippocampus place cells, monkey frontal cortex) and artificial neural networks.

### 4. Limitations

- Rastermap works best when neural activity can be summarised along a single dimension. In high-dimensional scenarios where multiple independent axes of variation exist (e.g., orientation and spatial frequency tuning in V1), the one-dimensional sorting cannot capture all structure simultaneously.
- The initial k-means clustering is the main source of run-to-run variability.

---

## Neuropil Contamination

The paper's treatment of neuropil is brief and embedded within the methods:

### Suite2p neuropil correction as standard preprocessing

- "Calcium imaging data were processed using the Suite2p toolbox, available at https://github.com/MouseLand/suite2p. Suite2p performs motion correction, region of interest (ROI) detection, cell classification, neuropil correction and spike deconvolution as described previously."
- The reference is to Pachitariu et al. 2016/2017 (Suite2p), which implements the standard F_corrected = F_soma - 0.7 * F_neuropil correction.
- No further discussion of neuropil correction methodology or validation is provided in this paper.

### Implication

The paper treats neuropil correction as a solved preprocessing step handled by Suite2p. The default correction factor (0.7) is applied without empirical validation for the specific experimental conditions. This is representative of common practice in the field but may not be optimal for all datasets.

---

## Relevance to hm2p

### 1. Rastermap as a discovery tool for RSP population structure

Rastermap could be applied to the hm2p synchronised calcium data (sync.h5) to visualise the population structure of RSP neurons. By sorting Penk+ and Penk-CamKII+ neurons based on their activity patterns and aligning the sorted raster with head direction, light/dark epochs, and movement variables, patterns of functional organisation could be identified that might not be apparent from single-neuron analyses.

Specific applications:
- **HD-tuned subpopulations:** Rastermap sorting should reveal sequential activation patterns aligned with head direction if a substantial fraction of neurons are HD-tuned.
- **Light vs dark differences:** Running Rastermap separately on light and dark epochs and comparing the sorting could reveal population-level reorganisation.
- **Behavioural receptive fields:** The approach from the spontaneous activity analysis (Fig. 3) -- computing superneuron correlations with speed, AHV, and other behavioural variables -- could be directly applied to hm2p data.

### 2. Suite2p neuropil correction as community standard

The paper's use of Suite2p's default neuropil correction (r = 0.7) without further validation reflects common practice. This is relevant as a reference point for the hm2p project: many published studies use this default, so the hm2p analysis should at minimum match this standard and ideally improve upon it with empirical estimation of the correction coefficient.

### 3. Superneuron averaging for noisy data

The superneuron concept (averaging nearby neurons in Rastermap sorting) is relevant for the hm2p dataset, which has relatively few neurons per session (~15 ROIs on average). With small populations, individual neuron noise is high. Superneuron-like averaging across sessions with similar tuning properties could improve SNR for population-level analyses.

### 4. Rastermap's time-lagged correlations for HD sequences

The asymmetric similarity matrix computed from time-lagged cross-correlations is directly relevant to temporal sequences in HD coding. If RSP neurons fire in a temporal sequence as the animal's head rotates (as predicted by attractor models), Rastermap's sequential detection module should be able to identify and sort these sequences.

---

## Key References Cited

| Reference | Relevance |
|---|---|
| Pachitariu et al. 2016/2017 (Suite2p) | Standard calcium imaging processing pipeline including neuropil correction |
| Stringer et al. 2019 (Science) | Large-scale spontaneous activity recordings; behavioural modulation of cortical activity |
| Stringer & Pachitariu 2019 (Curr Opin Neurobiol) | Computational processing of calcium imaging data |
