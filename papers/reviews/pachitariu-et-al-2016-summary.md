# Pachitariu et al. 2016 — Detailed Summary

## Citation

Pachitariu M, Stringer C, Dipoppa M, Schroeder S, Rossi LF, Dalgleish H, Carandini M, Harris KD. 2016. "Suite2p: beyond 10,000 neurons with standard two-photon microscopy." *bioRxiv* preprint. doi:10.1101/061507

**Affiliations:** UCL Institute of Neurology; UCL Department of Neuroscience, Physiology, and Pharmacology; Gatsby Computational Neuroscience Unit; UCL Institute of Ophthalmology; UCL Wolfson Institute for Biomedical Research.

---

## Overview

This preprint introduces Suite2p, an end-to-end computational pipeline for processing two-photon calcium imaging movies. The pipeline comprises four stages: (1) image registration via phase correlation, (2) ROI detection using a generative model that explicitly accounts for neuropil contamination, (3) automated ROI classification with a GUI for manual curation, and (4) signal extraction with neuropil correction and optional spike deconvolution. The paper demonstrates that Suite2p runs faster than real time on standard workstations and recovers approximately twice as many cells as the then-current CNMF-based method (CaImAn), enabling routine detection of ~10,000 neurons simultaneously with standard resonant-scanning two-photon microscopes.

---

## Key Findings and Arguments

### 1. Phase correlation for sub-pixel registration

Suite2p uses phase correlation (spatial whitening before cross-correlation) rather than standard cross-correlation for motion correction. This emphasises high-frequency image content (cellular structures) over low-frequency background. The method achieves ~0.1 pixel registration error, outperforming standard cross-correlation methods, and is >15x faster than upsampled cross-correlation. Non-rigid registration is supported by dividing the FOV into blocks and interpolating offsets with Gaussian basis functions.

### 2. Explicit neuropil model during ROI detection

The core methodological contribution is a generative model of the recorded signal that decomposes each pixel's fluorescence into (a) contributions from spatially localised ROIs, (b) a spatially smooth neuropil signal represented by raised-cosine basis functions, and (c) Gaussian measurement noise. The algorithm iterates between detecting new ROIs, re-estimating timecourses, and reassigning pixels. An initial SVD factorisation of the entire dataset reduces computational load. The explicit neuropil model prevents neuropil-driven false positives from dominating the ROI set.

### 3. Semi-automated classification

A naive Bayes classifier labels ROIs as cells or non-cells based on activity-dependent statistics (skewness, variance, pixel correlation) and shape statistics (area, aspect ratio). The classifier reduces manual curation time and can be retrained on user decisions. A GUI displays ROIs with their traces, neuropil contamination, and multiple image projections.

### 4. Neuropil-corrected signal extraction

The neuropil signal is highly correlated with somatic signals over long distances. Suite2p jointly models the neuropil and ROI signals to extract corrected fluorescence traces. Spike deconvolution is included as an optional final step.

---

## Relevance to hm2p

### 1. Suite2p is the primary calcium extraction tool in the hm2p pipeline

Suite2p is the default extractor for Stage 1 of the hm2p pipeline. Understanding the algorithm's design choices — particularly its neuropil model and ROI classification — is necessary for interpreting the calcium traces used in all downstream analyses.

### 2. Neuropil contamination is a key confound for cell-type comparisons

The hm2p project compares Penk+ and Penk-CamKII+ RSP neurons. If these populations differ in expression density, depth, or morphology, they may have systematically different neuropil contamination levels. Suite2p's neuropil subtraction coefficient (default 0.7) may not be equally appropriate for both populations. This is a confound that should be assessed explicitly — for example, by comparing the neuropil coefficient distributions between cell types or by running FISSA as a robustness check.

### 3. ROI classification affects cell yield

The automated classifier's decisions on what constitutes a soma vs a dendrite directly impact the hm2p dataset. Since hm2p sessions are single-plane with both somata and dendrites in the same plane, the classifier's performance on distinguishing these is relevant. The hm2p pipeline performs post-hoc ROI classification by shape, and understanding Suite2p's own classification statistics provides context for how many ROIs might be misclassified.

### 4. Motion correction matters for freely-moving data

Although the hm2p microscope is head-mounted (not benchtop), Suite2p's non-rigid registration is still applied to correct for brain motion during free movement. The quality of this registration directly affects trace quality, and motion artefacts are a potential confound for apparent neural activity differences between light and dark epochs (if mice move differently in darkness).

### 5. Spike deconvolution is not the primary method in hm2p

The hm2p pipeline uses CASCADE for spike inference rather than Suite2p's built-in OASIS deconvolution. However, understanding what Suite2p provides as raw output (F, Fneu, spks) is necessary for properly handling the extraction stage.
