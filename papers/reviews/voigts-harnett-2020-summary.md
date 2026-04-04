# Voigts & Harnett 2020 — Detailed Summary

## Citation

Voigts J, Harnett MT. 2020. "Somatic and Dendritic Encoding of Spatial Variables in Retrosplenial Cortex Differs during 2D Navigation." *Neuron* 105(2):237-245. doi:10.1016/j.neuron.2019.10.016

**Affiliations:** Department of Brain & Cognitive Sciences and McGovern Institute for Brain Research, Massachusetts Institute of Technology, Cambridge, MA, USA.

---

## Overview

This paper introduces a rotating headpost system that permits conventional two-photon imaging during free 2D locomotion with volitional head rotation. The key scientific finding is that local calcium transients in apical tuft dendrites of L5 pyramidal neurons in retrosplenial cortex (RSC) encode navigational variables (head direction, position) differently from their parent somata. This fulfils the theoretical requirements for active dendritic computation in RSC during navigation. The system preserves head direction coding (validated via tetrode recordings in postsubiculum) while providing the optical stability needed for sub-cellular two-photon imaging.

---

## Key Findings and Arguments

### 1. Rotating headpost preserves HD coding

Tetrode recordings in postsubiculum showed that the rotating headpost had no adverse effect on HD coding compared to free behaviour (N = 9 neurons; mean absolute difference in preferred heading = 3.49 degrees). Mice habituated rapidly and explored spontaneously within seconds.

### 2. Less Z-motion than conventional head fixation

Because the headpost system yields to lateral and rotational forces (via floor translation and bearing rotation), less torque is applied by the animal than in conventional head fixation. This results in less baseline fluorescence fluctuation (a proxy for Z-motion) compared to static head fixation (p < 0.005).

### 3. Simultaneous somatic and dendritic imaging in RSC

GCaMP6f was expressed in small populations (~50-100 neurons) of L5 pyramidal neurons in RSC. Simultaneous two-plane imaging captured somata (350-500 um depth) and distal apical tuft dendrites (20-60 um depth) at 5 Hz per plane. Soma-dendrite pairs were identified by tracing along transiently active dendrites in fast z-scans. N = 105 soma/dendrite pairs were analysed.

### 4. Local dendritic events are distinct from somatic activity

The majority of dendritic GCaMP transients coincided with somatic transients (joint events), but a significant fraction of soma-dendrite pairs exhibited local dendritic events (23.8% of dendrites had >15% independent events) and local somatic events (16.2% of somata had >15% independent events). Local events were smaller in amplitude and faster in decay than joint events.

### 5. Dendritic tuning differs from somatic tuning

The critical finding: local dendritic HD and position tuning differed from somatic tuning in the same neurons. Dendritic tuning was reliable across split halves of the experiment (control for variance), but differed more from somatic tuning than from itself (p < 0.005 for both HD and position). Different branches of the same neuron differed in position tuning (p = 0.010) but not significantly in HD tuning (p = 0.153; N = 12 cells).

### 6. RSC neurons show speed and rotation modulation

Somatic firing rates in RSC L5 increased with both head rotation speed and locomotion speed. Cells showed a variety of HD and position tuning strengths, with some cells tuned in both dimensions.

---

## Relevance to hm2p

### 1. Direct precedent for RSC imaging during navigation

This is the most directly relevant methodological precedent for hm2p. It demonstrates that two-photon calcium imaging of RSC neurons during free 2D navigation with head rotation is feasible and yields interpretable HD and spatial tuning. However, the Voigts & Harnett system uses a rotating headpost (semi-restrained) rather than a head-mounted microscope (fully free), so the hm2p dataset represents a step forward in behavioural naturalism.

### 2. Somatic vs dendritic tuning is relevant to hm2p ROI classification

The hm2p dataset contains both soma and dendrite ROIs in the same imaging plane, classified post-hoc by shape. The Voigts & Harnett finding that dendritic tuning differs from somatic tuning means that including dendrite ROIs in the analysis could introduce noise or systematic bias. If dendrite contamination differs between Penk+ and Penk-CamKII+ populations (due to morphological differences), this could confound the cell-type comparison.

### 3. Anatomical basis for HD input to RSC dendrites

The paper cites Shibata (1993) showing that anterior thalamic HD inputs synapse onto distal apical dendrites in RSC. This anatomical arrangement supports the idea that HD information arrives at dendrites and is integrated with other signals at more perisomatic synapses. If Penk+ and Penk-CamKII+ neurons differ in their dendritic input patterns, they might show different dendritic vs somatic tuning, though hm2p's single-plane imaging cannot resolve this.

### 4. Speed and rotation modulation as confounds

The finding that RSC firing rates increase with both locomotion speed and head rotation speed is directly relevant to hm2p. If mice move at different speeds in light vs dark, or if the two cell types have different speed sensitivity, this could confound the interpretation of light/dark differences in HD tuning. Speed and angular head velocity should be included as covariates in all analyses.

### 5. Sample rate considerations

The Voigts & Harnett system imaged at 5 Hz per plane, and their simulations showed >90% of GCaMP events were detectable at this rate for decay constants >0.4 s. The hm2p microscope operates at ~9.6 Hz, providing somewhat better temporal resolution. However, fast dendritic transients (decay < 0.4 s) may still be missed. This is less of a concern for hm2p since the focus is on somatic signals.

### 6. Effect sizes for HD tuning in RSC

The paper provides baseline effect sizes for HD information content in RSC L5 neurons during navigation, quantified as KL divergence between rate and occupancy distributions. These values can be compared with hm2p results to assess whether the two cell types fall within or outside the expected range.
