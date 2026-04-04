# Helmchen & Denk 2005 -- Detailed Summary

## Citation

Helmchen F, Denk W. 2005. "Deep tissue two-photon microscopy." *Nature Methods* 2(12):932-940. doi:10.1038/nmeth818

**Affiliations:** Department of Neurophysiology, Brain Research Institute, University of Zurich; Department of Biomedical Optics, Max Planck Institute for Medical Research, Heidelberg.

---

## Overview

This review covers the fundamental physical principles of two-photon laser-scanning microscopy (2PLSM) and the technical considerations for optimising imaging depth in intact biological tissue. The paper explains why nonlinear excitation provides intrinsic optical sectioning (no pinhole needed), why near-infrared excitation penetrates deeper than visible light, and what factors limit the maximum achievable imaging depth. It also surveys in vivo labelling techniques and applications in neuroscience, including calcium imaging of neuronal networks.

---

## Key Findings and Arguments

### 1. Why two-photon excitation enables deep imaging

- In two-photon absorption, two near-infrared photons (~700--1000 nm) combine their energies to excite a fluorophore that would normally require a single visible-wavelength photon. Because the probability of two-photon absorption depends on the square of the light intensity, fluorescence is generated only at the focal point where photon density is highest.
- This spatial confinement of excitation means that even multiply scattered emission photons can be assigned to their focal origin, eliminating the need for a confocal pinhole and allowing collection of all photons regardless of scatter.
- Near-infrared excitation scatters less in tissue than visible light. The scattering mean free path in brain grey matter is approximately 200 um at 800 nm in vivo, compared to 50--100 um at 630 nm.

### 2. Signal decay with depth

- Ballistic (unscattered) excitation power decays exponentially with depth: P_ball = P_0 * exp(-z/l_s), where l_s is the scattering mean free path.
- Because two-photon fluorescence depends on the square of intensity, signal decays as exp(-2z/l_s) -- twice as fast as the power loss.
- Compensating for this loss requires exponentially increasing laser power with depth, up to the damage threshold.

### 3. Maximum imaging depth

- With a standard Ti:sapphire oscillator (~100 fs pulses, ~1 W average power), the maximum imaging depth in neocortex is approximately 600--800 um.
- With a regenerative amplifier, depths up to 1 mm have been demonstrated (Theer et al. 2003).
- The ultimate depth limit is set by out-of-focus fluorescence generated near the sample surface, which produces a background that reduces contrast. This is particularly limiting in densely labelled tissue (e.g., transgenic GFP mice).

### 4. Point-spread function and resolution

- The effective PSF for 2PLSM is the square of the illumination PSF: PSF_2p = [PSF(v/2, u/2)]^2.
- Theoretical lateral resolution is approximately 0.5 um and axial resolution approximately 2 um with a high-NA objective at 800 nm excitation.
- In practice, resolution degrades with depth due to tissue-induced wavefront aberrations and effective NA reduction from scattering of peripheral rays.
- The fill factor (beam width relative to objective back aperture) trades off resolution versus power transmission. A fill factor of approximately 0.7 transmits nearly all power with only minor resolution loss.

### 5. In vivo calcium imaging of neuronal networks

- The review highlights bulk-loading of calcium indicators (OGB-1-AM; Stosiek et al. 2003) as enabling population-level calcium imaging with cellular resolution in living animals.
- It notes that calcium signals are found in the neuropil, providing additional information about input activity in afferent pathways (citing Kerr et al. 2005).
- Because bulk loading does not discriminate between cell types, specific counterstaining (e.g., sulforhodamine 101 for astrocytes) may be needed to identify particular cell populations.

### 6. Temporal resolution limitations

- Typical frame rates with galvanometric scanners are 15--30 Hz for a single plane. Resonant scanners can achieve video-rate (30 fps) full-frame acquisition.
- Fast acquisition across multiple z-planes or volumes remains a technological challenge.

---

## Neuropil Contamination

Although this review does not focus specifically on neuropil contamination, it provides the physical framework for understanding why it occurs:

### Axial resolution and out-of-focus contributions

- The axial PSF of 2PLSM is approximately 2 um (FWHM) under ideal conditions, but in practice this broadens with depth due to aberrations and scattering. At imaging depths of 100--300 um (typical for cortical L2/3), the effective axial resolution may be 4--6 um or more.
- Because cortical neuropil is dense (axons, dendrites, boutons surround every soma), an ROI drawn around a cell body will inevitably include fluorescence from neuropil structures above and below the focal plane, as well as laterally adjacent structures within the PSF.

### Scattered fluorescence collection

- In deep tissue imaging, the strategy is to collect as many photons as possible regardless of apparent origin (whole-area detection). This maximises signal but means that scattered emission photons from adjacent structures contribute to each pixel's measured intensity.
- The review notes that fluorescence from a deep focus emerges from the tissue surface as a diffuse region with FWHM approximately 1.5 times the focal depth. This spread means that photons from neuropil structures near the soma can be scattered into the somatic ROI during detection.

### Implication: out-of-focus fluorescence is inherent to 2PLSM

- The review explains that while two-photon excitation is spatially confined, the finite axial extent of the PSF (several microns) means that structures within this volume but outside the soma will be excited. In densely labelled tissue (as with viral GCaMP expression), this produces a neuropil contamination signal that must be corrected computationally.

---

## Relevance to hm2p

### 1. Axial resolution constrains neuropil correction

The hm2p project uses a head-mounted two-photon microscope imaging at approximately 9.6 Hz. The axial PSF of such miniaturised microscopes is typically broader than that of benchtop systems (often 5--10 um), meaning that each somatic ROI samples a larger axial volume and is therefore more susceptible to neuropil contamination. This makes neuropil correction particularly important for the hm2p dataset.

### 2. Depth-dependent signal quality

The review's discussion of exponential signal decay with depth is relevant because RSP neurons are imaged at varying depths depending on the cortical layer. Neurons in deeper layers will have lower SNR and higher relative neuropil contamination, potentially creating a depth-dependent bias in tuning measurements.

### 3. Motion artefacts in freely-moving mice

The review discusses tissue pulsation as a source of imaging artefacts in anaesthetised or head-fixed preparations. In freely-moving mice (as in hm2p), brain motion is substantially greater. Motion in z causes the imaging plane to shift, changing which neuropil structures contribute to each ROI's fluorescence. This can introduce transient contamination changes correlated with movement (and therefore potentially with head direction changes), creating a particularly insidious confound for HD tuning analyses.

### 4. Dense GCaMP expression increases background

The review notes that the maximum imaging depth is limited by near-surface out-of-focus fluorescence, particularly in densely labelled tissue. In hm2p, viral GCaMP expression labels many neurons, creating a bright neuropil background. This increases the baseline fluorescence (F0) and reduces the dF/F amplitude of somatic transients, making neuropil correction more critical.

### 5. Understanding the physics helps evaluate correction methods

The physical principles in this review (PSF shape, scattering, whole-area detection) inform the choice and evaluation of neuropil correction methods. A fixed coefficient correction (F - r*Fneu) assumes a constant spatial relationship between soma and neuropil signals, which may not hold if the PSF changes with depth or if motion shifts the focal plane. FISSA (spatial ICA-based correction) may be more robust to these effects.

---

## Key References Cited

| Reference | Relevance |
|---|---|
| Denk et al. 1990 (Science) | Original two-photon microscopy paper |
| Stosiek et al. 2003 (PNAS) | Bulk loading of calcium indicators in vivo |
| Kerr et al. 2005 (PNAS) | Neuropil calcium signals as a measure of input activity |
| Theer et al. 2003 (Opt Lett) | Deep imaging with regenerative amplifier (1 mm depth) |
| Nimmerjahn et al. 2004 (Nat Methods) | Sulforhodamine 101 astrocyte marker |
| Oheim et al. 2001 (J Neurosci Methods) | Parameters influencing imaging depth in brain tissue |
