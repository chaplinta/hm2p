# Stringer et al. 2026 — Detailed Summary

## Citation

Stringer C, Ki C, DelGrosso N, LaFosse P, Zhang Q, Pachitariu M. 2026. "Extracting large-scale neural activity with Suite2p." *bioRxiv* preprint. doi:10.64898/2026.02.04.703741

**Affiliations:** HHMI Janelia Research Campus; Carnegie Mellon University, Neuroscience Institute and Dept of Machine Learning; Universitat Bonn, Institute for Experimental Epileptology and Cognitive Sciences.

---

## Overview

This paper is the comprehensive methods paper for the current version of Suite2p, describing algorithms that have been developed and refined over 10 years since the original 2016 preprint. The major advances include GPU-accelerated non-rigid motion correction, improved cell detection using a dual-L0-penalty matrix decomposition algorithm, integration of alternative detection methods (Sourcery, Cellpose), and systematic benchmarking against CaImAn and Fiola. The paper demonstrates recording of >100,000 neurons simultaneously from mouse cortex using a standard commercial microscope (Thorlabs Bergamo2) with riboL1-based jGCaMP8s transgenic mice.

---

## Key Findings and Arguments

### 1. GPU-accelerated motion correction

The registration algorithm uses iterative reference refinement, rigid phase correlation, and non-rigid block-wise correction with adaptive smoothing. A key innovation is the adaptive smoothing step: blocks with high SNR get accurate local corrections while low-SNR blocks get more global (but still accurate) corrections. Suite2p's registration is substantially faster than both CaImAn and Fiola (233 Hz rigid, 170 Hz non-rigid vs 69/35 Hz for CaImAn and 23 Hz for Fiola) and outperforms both on quantitative registration metrics.

### 2. Registration quality control metrics

A PCA-based quality control method is introduced: after motion correction, the top principal components of the corrected movie should reflect neural activity, not residual motion. By averaging frames with high vs low projections onto each PC and registering these averages to each other, the amplitude of residual motion can be estimated per PC. This metric can detect both rigid and non-rigid residual motion, and gradual temporal drift (which was found to affect CaImAn due to its online template updating).

### 3. Improved ROI detection via dual-L0 sparsity

The detection algorithm identifies ROIs by their transient excursions above baseline, formulated as matrix decomposition with L0 penalties on both spatial and temporal factors. The algorithm convolves frames with square templates of multiple sizes, thresholds at a high value, and iteratively extracts ROIs by greedy matching pursuit. This multi-scale approach handles somas, dendrites, and other compartments. Suite2p achieves higher F1 scores than CaImAn/Fiola, with fewer false negatives and false positives on hybrid ground-truth recordings.

### 4. Hybrid ground-truth benchmarking

A novel benchmarking approach uses recordings from riboL1-based soma-localised calcium indicators (where ground truth is straightforward via Cellpose anatomical segmentation) with added simulated neuropil from real traces recorded in other planes. This produces realistic benchmarks with known ground truth.

### 5. Simplified extraction and deconvolution

Trace extraction uses simple pixel averaging (removing overlapping portions of ROIs) rather than source-deconvolution-based demixing. Spike deconvolution uses OASIS with the AR-1 model and no sparsity constraints. The authors note that newer sensors like jGCaMP8 have fluorescence kinetics closer to the AR-1 model assumption.

### 6. Scale demonstration

Over 100,000 neurons recorded simultaneously from 7 planes at 1.87 Hz using a standard Thorlabs Bergamo2 microscope with a Nikon 10x 0.5 NA objective and riboL1-based jGCaMP8s transgenic mice. Suite2p processes such datasets within 20 minutes on a GPU cluster.

---

## Relevance to hm2p

### 1. Current version of the primary extraction tool

This is the definitive reference for the version of Suite2p used in the hm2p pipeline. The registration and detection algorithms described here supersede the 2016 preprint. Methods sections of any hm2p publication should cite this paper alongside the original.

### 2. Registration quality control for freely-moving data

The PCA-based registration metrics are directly applicable to hm2p data. Since freely-moving mice produce more brain motion than head-fixed preparations, running these QC metrics on each hm2p session would provide quantitative evidence that motion correction was adequate. This is particularly important for the darkness epochs, where mice may exhibit different movement patterns.

### 3. Neuropil handling in single-plane data

Suite2p's extraction approach (pixel averaging with overlap removal) is simpler than CaImAn's deconvolution-based demixing. For hm2p single-plane data where somata and dendrites coexist, understanding how Suite2p handles overlapping ROIs is relevant. The pipeline's use of FISSA as an optional additional neuropil correction step can be justified as addressing potential residual contamination.

### 4. Cell detection parameters need validation

The paper notes that a single detection parameter (total number of ROIs) was optimised for benchmarks. For hm2p data, the detection parameters should be validated on a per-session basis to ensure consistent cell yields across sessions and cell types. Different GCaMP expression levels between Penk+ and Penk-CamKII+ populations could bias detection if not checked.

### 5. Z-drift monitoring

The paper mentions real-time Z-drift estimation integrated with ScanImage. For head-mounted two-photon data (as in hm2p), Z-drift between light and dark epochs could be a confound if the mouse's posture changes systematically. The registration metrics described here can detect such drift.
