# Zong et al. 2022 — Detailed Summary

## Citation

Zong W, Obenhaus HA, Skytoen ER, Eneqvist H, de Jong NL, Vale R, Jorge MR, Moser MB, Moser EI. 2022. "Large-scale two-photon calcium imaging in freely moving mice." *Cell* 185(7):1240-1256. doi:10.1016/j.cell.2022.02.017

**Affiliations:** Kavli Institute for Systems Neuroscience and Centre for Neural Computation, Norwegian University of Science and Technology (NTNU), Trondheim, Norway.

---

## Overview

This paper presents MINI2P, a second-generation miniaturised two-photon microscope for large-scale calcium imaging in freely moving mice. MINI2P weighs below 3 g, has a flexible 0.7-mm connection cable, and can image over 1,000 neurons simultaneously across multiple interleaved planes. The design addresses the key limitations of the first-generation system (Zong et al. 2017): excessive weight (5 g) and stiff optical cables that impeded natural mouse behaviour. The paper demonstrates that a 3 g microscope with thin cables produces running distance, speed, and turning behaviour indistinguishable from untethered control animals, and provides proof-of-principle recordings from visual cortex, medial entorhinal cortex, and hippocampus showing spatially tuned cells in all areas.

---

## Key Findings and Arguments

### 1. Weight and cable flexibility determine behavioural impact

Systematic comparison of 5 g (thick cable) vs 3 g (thin cable) vs no-microscope conditions in 10 mice showed that the 5 g system significantly reduced running distance, speed, centre exploration, and turning behaviour. The 3 g system with thin cable was statistically indistinguishable from unencumbered controls on all metrics (Friedman tests with Tukey post-hoc, all p > 0.90 for 3g-t vs control). Cable stiffness had a larger impact on movement than microscope weight.

### 2. Micro-tunable lens for lightweight z-scanning

A custom quartet micro-tunable lens (mTlens, 0.06 g, <0.4 ms response time, 240 um z-scanning range) replaced the previous 1.8 g electrically tunable lens. The mTlens uses electrostatic driving with negligible thermal effects, whereas the previous ETL showed >20 degrees C temperature rise and focal plane drift within 10 minutes at full power.

### 3. Tapered fibre bundle for efficient fluorescence collection

A tapered fibre bundle (TFB) with a 1.5-mm collection end tapering to a 0.7-mm fibre bundle maintains collection efficiency equivalent to the previous 1.5-mm SFB while enabling a thin, flexible cable assembly.

### 4. Two versions optimised for different priorities

MINI2P-L uses a large-angle MEMS scanner for a larger FOV (up to 510 x 510 um) at 15 Hz. MINI2P-F uses a fast MEMS scanner for higher speed (40 Hz) at a smaller FOV (~420 x 420 um). Both support multiple objectives for different imaging configurations (cortical surface, deep brain via GRIN lens, hippocampus via prism).

### 5. Large-scale recordings in freely moving mice

Visual cortex: 592 neurons imaged through a cranial window in transgenic GCaMP6s mice during free foraging (6 mice, 10-minute recordings). Medial entorhinal cortex: 404 neurons imaged via glass prism in 3 mice during free foraging. Hippocampus: 464 neurons imaged via cannula + GRIN lens in 3 mice. All areas showed spatially tuned cells. Field stitching across adjacent FOVs enabled recordings from >10,000 neurons in the same animal.

### 6. Stable imaging during diverse behaviours

Rigid and non-rigid motion were quantified across brain regions. Residual drift after correction was <2 um across all 30 spatial PCs in all regions. Imaging was stable during free foraging in open fields for extended sessions.

---

## Relevance to hm2p

### 1. The imaging system used in hm2p

MINI2P (or its close derivative) is the two-photon miniscope used in the hm2p experiment. The specifications described here — FOV, frame rate, z-scanning, weight, cable properties — define the technical parameters of the hm2p dataset. This paper should be cited in any hm2p publication as the primary microscope reference.

### 2. Behavioural validation

The paper's systematic demonstration that mice with 3 g microscopes behave identically to unencumbered controls is essential evidence that hm2p's rose maze behavioural data are not artefacted by the microscope. If reviewers question whether the microscope affected exploration or HD sampling, this paper provides the counter-evidence.

### 3. Frame rate and temporal resolution

The hm2p dataset uses ~9.6 Hz imaging, which falls between MINI2P-L (15 Hz) and MINI2P-F (40 Hz). At 9.6 Hz, GCaMP6 transients with decay constants >0.4 s should be well-sampled (>90% detection rate per Voigts & Harnett 2020 simulations). This frame rate is adequate for HD tuning analysis (HD changes on timescales of hundreds of ms to seconds) but limits the temporal precision of spike inference and angular head velocity correlations.

### 4. Single-plane vs multi-plane

MINI2P supports multi-plane imaging via the mTlens, but hm2p uses single-plane acquisition. This means the full z-scanning capability is not exploited, and all ROIs (somata and dendrites) come from a single optical section. The trade-off is higher per-plane frame rate (9.6 Hz vs ~3-5 Hz per plane if multi-plane were used).

### 5. Motion artefact expectations

The paper's motion correction residuals (<2 um) set the expectation for hm2p data quality. Larger residuals in specific hm2p sessions would indicate technical problems or unusually vigorous behaviour. Comparing motion metrics between light and dark epochs is a necessary control.

### 6. FOV and cell yield

The MINI2P-L FOV of up to 510 x 510 um is relevant to hm2p's cell yields. With ~15 ROIs per session on average, the hm2p dataset is operating at low density compared to the 500+ neurons per FOV demonstrated in cortex here. This likely reflects RSP's sparser labelling via viral injection rather than a microscope limitation, but it constrains per-session statistical power.
