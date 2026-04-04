# Vickers & McCormick 2024 -- Detailed Summary

## Citation

Vickers ED, McCormick DA. 2024. "Pan-cortical 2-photon mesoscopic imaging and neurobehavioral alignment in awake, behaving mice." *bioRxiv* preprint. doi:10.1101/2023.10.19.563159

**Affiliations:** Institute of Neuroscience, University of Oregon; Department of Biology, University of Oregon.

---

## Overview

This preprint describes the development of two novel in vivo preparations for pan-cortical two-photon mesoscopic imaging at single-cell resolution in awake, behaving mice. The "dorsal mount" (modified Crystal Skull) enables bilateral dorsal cortex imaging, while the novel "side mount" (temporo-parietal) preparation extends coverage to lateral cortex including auditory cortex. Using a Thorlabs 2p-RAM mesoscope, the authors record up to approximately 7,500 neurons simultaneously at approximately 3 Hz across a 5 x 5 mm field of view, or approximately 800 neurons across four smaller FOVs at approximately 10 Hz. The paper serves as a methodological proof of principle, demonstrating alignment of widespread cortical neural activity with behavioural primitives and arousal/movement state transitions.

---

## Key Findings and Arguments

### 1. Two complementary preparations for pan-cortical access

- **Dorsal mount:** Mouse upright on running wheel, objective vertical. Covers bilateral dorsal cortex from posterior visual areas to anterior motor cortex. Based on the Crystal Skull preparation (Kim et al. 2016) with modifications including custom 3D-printed titanium headposts and mounting hardware.
- **Side mount:** Mouse head rotated 22.5 degrees, objective vertical or slightly tilted. Covers dorsal and lateral cortex across one hemisphere, including primary auditory, visual, somatosensory, and motor cortex simultaneously. This is the first preparation enabling simultaneous 2P imaging of all major primary sensory cortices plus frontal motor areas.

### 2. High axial resolution mitigates neuropil contamination

The 2p-RAM mesoscope achieves a point-spread function of 0.61 x 0.61 x 4.25 um (xyz) at 970 nm excitation. The authors explicitly highlight that this axial resolution (4.25 um) is important for "avoiding region-of-interest (ROI) contamination by neuropil and neighbouring cells." They contrast this with the Diesel 2p system, which has a broader axial PSF (approximately 8--10 um) that would increase neuropil contamination.

### 3. Suite2p processing with standard neuropil correction

- ROI extraction and classification are performed with Suite2p, including rigid and non-rigid motion correction.
- Fluorescence intensity is calculated as dF/F where F is computed pixel-by-pixel using a 30--60 s rolling 10th or 15th percentile baseline of the neuropil-subtracted mean ROI pixel intensity: **F - 0.7 * Fneu**.
- ROIs with Suite2p classifier cell probability less than 0.5 are excluded.
- This represents the standard Suite2p pipeline with the default neuropil correction coefficient of 0.7.

### 4. Widespread arousal and movement-related cortical activity

- A large percentage of cortical neural variance can be accounted for by spontaneous fluctuations in arousal and movement (consistent with Musall et al. 2019, Stringer et al. 2019).
- The "temporal filtering effect" (Shimoaka et al. 2018) means that correlations between behaviour and neural activity depend on the exact time since the behaviour began, making simultaneous multi-area recording essential for proper cross-regional comparisons.

### 5. Behavioural monitoring and alignment

- Up to 3 high-speed cameras capture body and face movements. DeepLabCut is used for pose tracking (66 labelled points). B-SOiD is used for unsupervised behavioural motif extraction.
- Detailed kinematic alignment allows relating neural activity to second-by-second movement and arousal changes.

---

## Neuropil Contamination

This paper addresses neuropil contamination in two ways: through the choice of microscope (optimised axial PSF) and through standard computational correction.

### The axial PSF as first line of defence

- The authors explicitly frame the 4.25 um axial PSF of the 2p-RAM mesoscope as a feature that reduces neuropil contamination. A thinner optical section means less out-of-focus neuropil is excited along with the soma.
- They note that the Diesel 2p system's broader axial PSF (approximately 8--10 um) would capture more neuropil signal per ROI, representing a trade-off between scan speed and contamination susceptibility.
- This highlights that neuropil contamination severity depends on the specific optical system, not just on the correction algorithm applied post-hoc.

### Computational correction: F - 0.7 * Fneu

- The paper uses the standard Suite2p neuropil correction with the default coefficient (0.7) without further discussion or validation.
- The baseline F0 is computed as the rolling 10th or 15th percentile of the neuropil-corrected trace over 30--60 s windows (100--200 frames). This sliding baseline avoids contamination by transient calcium events but may not adequately track slow drifts in neuropil contamination level (e.g., changes in brain state or arousal).

### 1P widefield imaging has worse contamination

The paper notes that "1-photon widefield imaging can be contaminated by neuropil and hemodynamic signal (Waters 2020; Valley et al. 2021) and typically does not achieve single cell resolution," positioning 2P imaging as superior for single-cell analyses precisely because of its ability to optically section and computationally correct for neuropil.

---

## Relevance to hm2p

### 1. Axial PSF comparison with head-mounted 2P

The Thorlabs 2p-RAM mesoscope has a 4.25 um axial PSF, which is relatively tight for a mesoscope. The head-mounted two-photon microscope used in hm2p likely has a broader axial PSF (miniaturised optics typically sacrifice axial resolution). This means neuropil contamination in the hm2p dataset is likely more severe than in the Vickers & McCormick recordings, making neuropil correction more critical.

### 2. Standard Suite2p pipeline as community benchmark

The paper uses the same F - 0.7 * Fneu correction as many other studies. This establishes that the hm2p analysis pipeline using Suite2p's default correction is consistent with current practice for large-scale two-photon recordings. However, the hm2p project should evaluate whether 0.7 is appropriate for a head-mounted microscope with different optical properties.

### 3. Arousal and movement contamination across cortex

The finding that arousal and movement modulate neural activity across the entire cortex, including retrosplenial cortex, is directly relevant. In the hm2p rose maze, mice experience continuous changes in arousal and movement state. If neuropil signals in RSP carry arousal/movement information (which they almost certainly do, given the pan-cortical nature of these signals), then inadequate neuropil correction could introduce movement-correlated artefacts into the somatic traces.

### 4. Simultaneous multi-area imaging context

Vickers & McCormick argue that simultaneous multi-area recording is essential because of the temporal filtering effect. The hm2p project records only from RSP, so cross-regional comparisons are not possible. However, the finding that RSP changes in functional connectivity centrality during locomotion (from Nietz et al.) suggests that RSP network state is influenced by whole-brain dynamics that cannot be captured by single-area imaging.

### 5. DeepLabCut and B-SOiD methodological parallel

The paper uses DeepLabCut for pose tracking and B-SOiD for behavioural motif extraction, both of which are tools available (though CLAUDE.md prefers keypoint-MoSeq or VAME over B-SOiD). The 66-point DLC model is more detailed than the hm2p 5-keypoint SuperAnimal model, but the general approach (DLC tracking followed by behavioural segmentation) is shared.

### 6. Rolling baseline for F0

The 30--60 s rolling 10th percentile baseline used in this paper for computing dF/F is consistent with standard practice. The hm2p pipeline should use a similar approach, but the 1-min light/dark epoch duration means that baseline estimates near epoch transitions could span both conditions, potentially introducing artefacts. A condition-aware baseline (computed separately within light and dark epochs) may be preferable.

---

## Key References Cited

| Reference | Relevance |
|---|---|
| Kim et al. 2016 | Crystal Skull dorsal cortex preparation |
| Sofroniew et al. 2016 | 2p-RAM mesoscope design |
| Pachitariu et al. 2017 (Suite2p) | Motion correction, ROI detection, neuropil correction |
| Musall et al. 2019 | Movement explains large fraction of cortical activity variance |
| Stringer et al. 2019 (Science) | Spontaneous behaviour modulates cortical activity across areas |
| Waters 2020 | Neuropil contamination in 1P imaging |
| Mathis et al. 2018 | DeepLabCut for pose estimation |
