# Zong et al. 2017 — Detailed Summary

## Citation

Zong W, Wu R, Li M, Hu Y, Li Y, Li J, Rong H, Wu H, Xu Y, Lu Y, Jia H, Fan M, Zhou Z, Zhang Y, Wang A, Chen L, Cheng H. 2017. "Fast high-resolution miniature two-photon microscopy for brain imaging in freely behaving mice." *Nature Methods* 14(7):713-719. doi:10.1038/nmeth.4305

**Affiliations:** State Key Laboratory of Membrane Biology, Peking University; Institute of Basic Medical Sciences; Suzhou Institute of Biomedical Engineering and Technology, Chinese Academy of Sciences; School of Electronics Engineering, Peking University.

---

## Overview

This paper presents FHIRM-TPM (Fast High-Resolution Miniature Two-Photon Microscope), the first head-mounted two-photon microscope capable of resolving single dendritic spines in freely behaving mice. The headpiece weighs 2.15 g and achieves 0.64 um lateral and 3.35 um axial resolution, 40 Hz frame rate at 256 x 256 pixels, and a 130 x 130 um field of view. The key enabling technology is HC-920, a custom-designed hollow-core photonic crystal fibre that delivers 920-nm femtosecond laser pulses with negligible nonlinear pulse-broadening, allowing efficient excitation of GCaMP6. The paper demonstrates stable imaging of somata, dendrites, and spines during vigorous behaviours (tail suspension, stepping down, social interaction) over multi-hour sessions.

---

## Key Findings and Arguments

### 1. Hollow-core photonic crystal fibre (HC-920)

Standard single-mode fibres cause severe pulse broadening at 920 nm due to material dispersion and nonlinear effects, making GCaMP excitation impractical. Previous HC-PCFs centred at 800 nm or 1060 nm could not efficiently excite GCaMP6. The custom HC-920 transmits 920-nm femtosecond pulses (85 fs input, ~100 fs output after 1 m) at powers up to 200 mW with negligible nonlinear broadening, independent of power level. Dispersion compensation is achieved using standard glass material of calculated length.

### 2. Miniature compound objective and MEMS scanner

A high-NA (0.8) achromatic miniature objective provides sub-micrometre lateral resolution. A 2D MEMS scanning mirror (0.8 mm diameter, ~6 kHz resonant frequency, +/-10 degree scanning angle) enables video-rate imaging. A supple fibre bundle (SFB) of 800 fused glass fibres collects emitted fluorescence.

### 3. Resolution approaches benchtop two-photon

On an integrated test platform, FHIRM-TPM resolved dendrites and spines with contrast and resolution nearly identical to a benchtop two-photon microscope. Miniature wide-field microscopy failed to resolve dendritic structures due to out-of-focus background. Somatic calcium transients from FHIRM-TPM had amplitudes (dF/F ~150%) comparable to benchtop, while wide-field amplitudes were an order of magnitude lower (~5-10%).

### 4. Stable imaging during vigorous behaviour

Frame-to-frame displacement averaged 0.19 +/- 0.09 um during the stepping-down paradigm (the most vigorous), remaining below half a pixel. Total FOV drift was <10 um over a 4-hour multi-paradigm protocol. Spine-level calcium transients were resolved even during vigorous body movements.

### 5. Movement-enhanced neural activity

A pilot experiment found increased dendritic activity in V1 during free exploration in darkness compared to head-fixed immobility, consistent with electrophysiological findings of movement-enhanced cortical activity.

---

## Relevance to hm2p

### 1. Foundational technology for hm2p imaging

The FHIRM-TPM is the first generation of the head-mounted two-photon microscope technology on which the hm2p imaging system is based. Understanding its design principles, resolution limits, and motion artefact characteristics is necessary for interpreting hm2p data and writing methods sections. The subsequent MINI2P (Zong et al. 2022) improved upon this design.

### 2. Resolution and FOV constraints

The 130 x 130 um FOV limits the number of neurons that can be imaged simultaneously in a single plane. This is directly relevant to the hm2p cell yields (~15 ROIs per session on average across 26 sessions) and constrains the statistical power available for per-session analyses.

### 3. Motion artefact baseline

The <0.5 pixel frame-to-frame displacement during vigorous movement establishes a baseline for what to expect in hm2p data. Since hm2p mice are in a rose maze (less vigorous than tail suspension), motion artefacts should be within this range. However, the total drift over long sessions (<10 um over 4 hours) is relevant for hm2p's alternating light/dark epochs — if drift accumulates preferentially during one condition, it could confound the comparison.

### 4. GCaMP excitation efficiency

The HC-920 fibre's ability to deliver undistorted 920-nm pulses is what makes GCaMP6 imaging practical in a head-mounted system. The dF/F amplitudes (~150%, comparable to benchtop) set expectations for signal quality in hm2p recordings. If hm2p sessions show substantially lower dF/F, this could indicate technical issues.

### 5. Single-plane limitation

This first-generation system could not image more than one focal plane, which is also the case for hm2p sessions (single plane per session). This means somatic and dendritic ROIs coexist in the same plane and must be separated post-hoc, as discussed in the Voigts & Harnett 2020 summary.
