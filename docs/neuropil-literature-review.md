# Neuropil Contamination in Two-Photon Calcium Imaging: Literature Review

A structured review of 7 papers on neuropil signals, contamination, and
correction methods in two-photon calcium imaging, with specific implications
for head-direction cell recordings in retrosplenial cortex.

---

## 1. What is the neuropil signal?

### Composition of cortical neuropil

Cortical neuropil consists of approximately 50% axons and presynaptic boutons,
30% dendrites and dendritic spines, and 10% glial processes (Kerr et al. 2005,
citing Braitenberg & Schuz 1998). In two-photon imaging, the neuropil signal
refers to fluorescence collected from the tissue surrounding identified cell
bodies — a dense mesh of subcellular structures that cannot be individually
resolved at standard imaging resolution.

### The neuropil signal is primarily axonal in origin

Kerr et al. (2005) provided the definitive demonstration that neuropil
calcium fluctuations in neocortex are dominated by axonal (presynaptic)
activity, not dendritic (postsynaptic) signals. They showed this through two
complementary experiments:

1. **Selective dendritic loading.** When calcium indicator (OGB-1-AM) was
   injected deep in layer 5 to label only apical dendrites projecting to
   superficial layers, the neuropil signal was absent or markedly reduced in
   L2/3 (peak dF/F of 1.7 +/- 1.1% at 150 um depth vs 10-30% with standard
   bulk loading). The correlation between neuropil and the electrocorticogram
   (ECoG) dropped to 0.1-0.2 in superficial layers.

2. **AMPA receptor blockade.** Local application of the AMPA antagonist
   GYKI53655 abolished postsynaptic spiking (mean firing rate dropped from
   0.038 to 0.014 Hz) but did not change the amplitude or ECoG-correlation of
   neuropil fluctuations (p = 0.87). This rules out a postsynaptic origin.

Kerr et al. termed the neuropil signal the "optical encephalogram" (OEG),
reflecting bulk AP-evoked calcium transients in presynaptic boutons activated
during cortical Up states. The OEG was tightly correlated with both the ECoG
(r = 0.70 +/- 0.1) and intracellular membrane potential recordings
(r = 0.75 +/- 0.1), and did not depend on imaging depth from pial surface
when dye was loaded in L2/3.

**Key conclusion:** The neuropil signal represents volume-averaged local
input activity — the aggregate of presynaptic calcium transients from axons
projecting into the imaged region. It is a measure of afferent drive, not
local output.

### Helmchen & Denk (2005) — Physical basis

Helmchen & Denk (2005) provide the technical framework for understanding
why neuropil signals contaminate somatic recordings in two-photon imaging.
Although two-photon excitation is spatially confined to the focal volume,
scattered fluorescence photons are collected from a wider region. The point
spread function (PSF) of a typical 2P system has finite axial extent
(~2-4 um), meaning that fluorescent structures above and below the focal
plane contribute to the detected signal. In the dense neuropil of cortex,
where every voxel contains fluorescent processes, this out-of-focus
fluorescence is unavoidable.

The key physics: while excitation is localised (nonlinear dependence on
intensity means scattered excitation photons do not generate meaningful
signal), the emitted fluorescence from the focal volume is collected using
"whole-area" epi-detection without a pinhole. Any fluorescence photon
collected by the objective is counted, regardless of its apparent position.
In practice, this means the detected signal from a somatic ROI includes
contributions from neuropil structures within and around the focal volume.

### Ali & Kwan (2019) — Compartment-specific calcium dynamics

Ali & Kwan (2019) provide a systematic review of what calcium signals mean
in different neuronal compartments, with direct relevance to neuropil
interpretation:

**Somatic signals** reflect spiking output. In L2/3 pyramidal neurons, calcium
enters through high-threshold voltage-gated channels (L-, N-, P/Q-type) only
during action potentials. Subthreshold membrane potential fluctuations produce
minimal somatic calcium change. However, Ali & Kwan note that a weak
correlation between subthreshold Vm and somatic calcium signals can appear
due to neuropil contamination — the surrounding neuropil reflects local
network activity that correlates with the membrane potential of the imaged
cell (their Supplementary Fig. 1e-f, citing Kwan & Dan 2012).

**Axonal signals** reflect presynaptic depolarisation and are reliably evoked
by action potentials. Calcium influx at boutons is mediated by P/Q- and
N-type channels, with fast rise (~1 ms) and decay (~60 ms). There is
substantial variability across boutons from the same cell (>10-fold variation),
and calcium dynamics depend on the postsynaptic cell type and neuromodulatory
state.

**Dendritic signals** are more complex:
- Backpropagating action potentials (bAPs) cause calcium influx that
  attenuates with distance from soma (negligible beyond 200-250 um in L2/3
  pyramidal neurons).
- Dendritic regenerative events (NMDA spikes, calcium spikes) produce large,
  localised calcium transients.
- Synaptic inputs produce spine-localised calcium influx (primarily via NMDA
  receptors), which can be isolated by subtracting the correlated shaft signal.

**Critical point for soma-dendrite ROI classification:** When imaging a
single plane containing both soma and dendrite ROIs, somatic ROIs report
spiking output while dendritic ROIs reflect a mixture of bAPs, dendritic
spikes, and synaptic inputs. These compartments carry fundamentally different
information. Neuropil contamination affects both, but may affect dendrite ROIs
differently because dendrites are more spatially extended and their neuropil
ring may sample a different local environment.

Ali & Kwan also showed that in RSP specifically, axonal calcium signals from
RSP neurons projecting to Cg1/M2 tracked graded afferent stimulation in an
awake mouse, confirming that axonal calcium signals faithfully report
presynaptic activity in vivo.

---

## 2. Neuropil contamination as a confound

### How contamination arises

In two-photon imaging, somatic ROIs collect fluorescence not only from the
cell body but also from neuropil structures that overlap with or surround
the soma in three dimensions. Suite2p estimates this contamination using a
"neuropil ring" — the annular region surrounding each ROI, excluding other
detected cells — and subtracts a scaled version of this neuropil signal from
the somatic trace.

### Dipoppa et al. (2018) — The case study in why correction matters

Dipoppa et al. (2018) provide the most explicit demonstration of how neuropil
contamination can produce artefactual results in cell-type-specific imaging.
They recorded from Pyr, Pvalb, Vip, and Sst interneurons in mouse V1 and
found that neuropil correction was essential for interpreting locomotion
modulation of neural activity.

**The critical finding:** Without neuropil correction, one would observe an
artefactual negative correlation of fluorescence with running speed in all
cell types. This is because the neuropil signal itself is modulated by
locomotion (reflecting network-level state changes), and this modulation
contaminates the somatic signal. After correction, the true cell-type-specific
modulation patterns emerge: Vip cells are positively modulated, Sst cells
show context-dependent modulation (positive with gray screen, negative in
darkness), and Pyr cells show weak, diverse modulation.

**Neuropil correction method (Peron et al. 2015 approach):**
Dipoppa et al. adopted an empirical method to estimate the neuropil
correction coefficient per experiment:
1. Define a neuropil mask as the region up to 35 um from each ROI border,
   excluding other detected cells.
2. For each cell, bin the neuropil signal into 20 intervals and estimate the
   5th percentile of somatic fluorescence in each bin.
3. Compute the slope (alpha_i) by linear regression on these lower-envelope
   points.
4. Average alpha across cells with high skewness (>4, i.e. sparsely firing
   pyramidal cells) to get a per-experiment correction factor.
5. Correct: F_corrected(t) = F(t) - alpha_exp * N(t).

The average correction factor was alpha = 0.82 (range across experiments not
reported). They note the method works well for sparsely firing cells but can
fail for densely firing cells (e.g. interneurons), where the cell's own
activity correlates with the neuropil signal.

**Relevance to hm2p:** This method is more principled than Suite2p's default
fixed coefficient of 0.7. The empirical lower-envelope approach estimates the
true contamination fraction, which can vary by imaging conditions, expression
levels, and cortical depth. However, it requires enough sparsely-firing cells
to estimate alpha reliably.

### Vickers & McCormick (2024) — Mesoscale imaging and neuropil subtraction

Vickers & McCormick (2024) used pan-cortical two-photon mesoscopic imaging
to record from up to 7500 neurons simultaneously across the mouse dorsal
and lateral cortex. Their neuropil handling is notable for several reasons:

1. **Standard Suite2p correction:** They computed dF/F using a rolling
   10th-15th percentile baseline of the neuropil-subtracted signal
   (F - 0.7 * Fneu), the same approach used in hm2p.

2. **Z-axis resolution matters:** The 2p-RAM mesoscope achieves a 4.25 um
   axial PSF at 970 nm, providing good optical sectioning that reduces
   neuropil contamination relative to systems with larger axial PSFs
   (e.g. Diesel 2p at ~8-10 um). For the miniature 2P microscope used in
   hm2p (FHIRM-TPM), the axial resolution is typically 8-12 um, which means
   substantially more neuropil contamination than in a bench-top 2P system.

3. **Movement artefact control:** They explicitly verified that fluorescence
   transients in GCaMP-negative structures (blood vessels, inactive neuropil)
   were minimal, confirming that observed activity was neural rather than
   motion-driven. This is an important control for freely-moving preparations
   like hm2p, where brain motion relative to the microscope is more severe.

4. **Arousal and movement signals pervade cortex:** Large populations of
   neurons across all cortical areas (including retrosplenial cortex)
   coordinated their activity with spontaneous changes in movement and
   arousal. This means the neuropil signal in RSP likely carries substantial
   movement/arousal-related input, which could confound HD tuning analyses if
   not properly corrected.

### Nietz et al. (2022) — Wide-field imaging limitations

Nietz et al. (2022) review wide-field calcium imaging and note that in this
modality, the signal is explicitly a population summation within a 3D voxel
including dendrites, somata, and axons from different neurons and cortical
layers. While this is primarily about 1-photon wide-field imaging (not 2P),
their discussion of signal composition reinforces the point that any imaging
modality with limited axial resolution captures a mixture of compartmental
signals.

They note that the observed signal represents "the combined neuropil activity
in a region" and that contamination by hemodynamics, baseline fluctuations,
and photonic noise are significant confounding factors requiring correction.

### Stringer et al. (2025) — Rastermap and Suite2p neuropil correction

Stringer et al. (2025) describe Rastermap, a visualisation method for neural
population recordings. Their relevance to neuropil is procedural: they use
Suite2p for all preprocessing, which "performs motion correction, ROI
detection, cell classification, neuropil correction and spike deconvolution."
This confirms that the Suite2p pipeline with neuropil correction is the de
facto standard for large-scale 2P calcium imaging analysis. Their use of
"superneurons" (averages of 50 neurons sorted by Rastermap position) also
effectively averages out residual neuropil contamination, since contamination
is spatially correlated but functionally heterogeneous across the sorted
dimension.

---

## 3. Implications for HD tuning analysis in RSP

### Could neuropil contamination create spurious HD tuning?

**Yes, in principle.** If the neuropil signal carries HD-related information
(from axonal inputs to RSP that are themselves HD-tuned, such as projections
from ADN or PoS), then neuropil contamination would impose a common
HD-tuned signal on all somatic ROIs. This would:

- Inflate the number of apparent HD cells
- Impose a common preferred direction bias across neurons
- Artificially correlate the tuning of nearby neurons
- Reduce the apparent diversity of preferred directions

The risk is particularly acute in RSP because it receives dense HD-tuned
input from the anterodorsal thalamic nucleus (ADN) and postsubiculum (PoS).
These afferents terminate as axonal boutons in RSP neuropil — exactly the
structures that dominate the neuropil signal (Kerr et al. 2005).

### Could contamination mask real tuning?

**Also possible.** If a cell's true preferred direction differs from the
neuropil HD signal, subtraction with a coefficient that is too high would
distort the tuning curve. Conversely, if the coefficient is too low, residual
contamination would pull the tuning curve toward the neuropil's preferred
direction (the population input).

### Differential effects on soma vs dendrite ROIs

For soma ROIs:
- Contamination is primarily from neuropil structures surrounding and
  overlapping the cell body in the axial dimension
- Suite2p's neuropil ring samples the surrounding tissue at the same depth
- The 0.7 coefficient may under- or over-correct depending on the local
  neuropil density and expression level

For dendrite ROIs:
- Dendrites are spatially extended processes; their ROIs are typically more
  elongated and thinner than somatic ROIs
- The neuropil ring for a dendrite ROI samples tissue adjacent to the
  dendrite segment, which may include signal from the same neuron's other
  branches or nearby axons
- Dendrite ROIs inherently contain a mixture of bAP-related signals,
  dendritic spikes, and synaptic input (Ali & Kwan 2019)
- The neuropil contamination fraction may be higher for dendrite ROIs
  because dendrites have smaller cross-sections relative to the PSF,
  meaning a larger fraction of the focal volume contains non-dendrite tissue

### Implications for Penk+ vs Penk-CamKII+ comparisons

If the two populations differ in:
- Soma size (affecting the ROI-to-neuropil ratio)
- Cortical layer position (affecting neuropil composition)
- GCaMP expression level (affecting SNR and thus relative contamination)
- Dendritic morphology (affecting dendrite ROI properties)

...then neuropil contamination could differentially affect the two populations
and create spurious between-group differences in HD tuning metrics (MVL,
tuning width, preferred direction distribution). Any cell-type comparison must
account for potential differences in contamination levels.

---

## 4. Best practices for neuropil correction

### Recommended approaches (in order of rigour)

1. **Empirical per-experiment coefficient (Peron et al. 2015 / Dipoppa et al.
   2018).** Estimate alpha from the lower envelope of the soma-vs-neuropil
   scatter plot using sparsely-firing cells. More accurate than a fixed
   coefficient but requires enough sparse cells. The average across
   experiments was 0.82 in Dipoppa et al.

2. **FISSA — spatial ICA neuropil subtraction (Keemink et al. 2018).**
   Decomposes the signal into independent sources using non-negative matrix
   factorisation, separating the somatic signal from neuropil contributions
   without assuming a fixed contamination fraction. More principled for
   heterogeneous tissue and varying expression levels.

3. **Suite2p default (coefficient = 0.7).** The simplest approach. Pachitariu
   et al. (2017) showed this produces reasonable results on average, but
   0.7 is likely an underestimate for many preparations (Dipoppa et al. found
   0.82; other groups report values up to 0.9).

4. **Soma-targeted GCaMP expression.** Using constructs that restrict GCaMP
   to the soma (e.g. soma-targeted GCaMP8) would eliminate the axonal and
   dendritic neuropil signal at its source. Not applicable to hm2p
   retrospectively, but relevant for future experiments.

### Controls to implement

Based on the reviewed literature, the following controls should be performed:

1. **Report the neuropil coefficient used and its effect.** Show key results
   (HD tuning metrics, population decoding) with both 0.7 and the empirical
   alpha, and with FISSA correction. If results are robust across methods,
   neuropil contamination is unlikely to drive the findings.

2. **Compare neuropil HD tuning to somatic HD tuning.** If the neuropil
   signal itself shows HD tuning (expected, given ADN/PoS inputs to RSP),
   quantify its strength. Somatic HD tuning should exceed neuropil HD tuning
   for genuinely tuned cells.

3. **Examine neuropil-to-soma ratio by cell type.** If Penk+ and Penk-CamKII+
   ROIs differ systematically in contamination fraction, this must be
   accounted for before interpreting between-group differences.

4. **Test whether apparent cell-type differences survive matched neuropil
   correction.** Use the per-ROI alpha (or FISSA) rather than a global
   coefficient, and confirm that between-group effects persist.

5. **Verify that motion artefacts in darkness are not confounded with neural
   signals.** In freely-moving preparations, brain motion may increase in
   darkness (if mice move differently). Suite2p's motion correction should
   handle rigid motion, but residual non-rigid motion could contaminate
   signals.

---

## 5. Specific concerns for RSP and freely-moving recordings

### RSP receives dense HD-tuned axonal input

The neuropil in RSP is enriched with axons from the head direction circuit
(ADN, PoS, LMN). This means the neuropil signal in RSP likely carries
stronger HD information than in sensory cortices. The standard neuropil
correction (which assumes the neuropil is a nuisance signal) may therefore
remove genuine HD-related input signals. Care must be taken not to
over-correct.

### The miniature 2P microscope has poorer axial resolution

The FHIRM-TPM used in hm2p has an axial PSF of approximately 8-12 um,
compared to 2-4 um for bench-top 2P systems. This means more out-of-focus
neuropil contributes to each somatic ROI measurement. The neuropil
contamination fraction is likely higher than the 0.7-0.82 range reported
for bench-top systems.

### Freely-moving introduces additional motion artefacts

Unlike head-fixed preparations (Dipoppa et al., Vickers & McCormick,
Stringer et al.), freely-moving mice produce substantial brain motion
relative to the microscope objective. While Suite2p's rigid and non-rigid
motion correction mitigates this, residual motion can produce fluorescence
changes that correlate with movement — potentially confounding HD tuning
analyses where the animal's head direction changes with movement.

### Light vs dark condition effects on neuropil

The light-off condition removes visual input to RSP. This should reduce the
visual component of the neuropil signal but not the HD circuit input (which
is partially maintained by path integration). Comparing neuropil signal
properties between light and dark conditions provides a control for the
visual contribution to neuropil contamination.

Dipoppa et al. (2018) showed that the effect of locomotion on Sst cell
baseline activity switched sign between gray-screen and darkness conditions
in V1. Analogous condition-dependent neuropil effects may exist in RSP.

---

## 6. Summary table

| Paper | Year | Key finding for neuropil | Relevance to hm2p |
|-------|------|--------------------------|--------------------|
| Kerr et al. | 2005 | Neuropil signal is axonal, represents local input (OEG) | RSP neuropil likely reflects HD-tuned afferent input from ADN/PoS |
| Helmchen & Denk | 2005 | 2P physics: out-of-focus fluorescence unavoidable, especially with larger PSF | Mini-2P PSF (~8-12 um) means more neuropil contamination than bench-top |
| Dipoppa et al. | 2018 | Neuropil correction essential; without it, artefactual locomotion modulation | Must correct before computing HD tuning; results must survive correction |
| Ali & Kwan | 2019 | Calcium signals differ by compartment; dendrites show mixed bAP + synaptic signals | Soma vs dendrite ROIs carry different information; neuropil contamination differs |
| Nietz et al. | 2022 | Wide-field signal is neuropil-dominated population sum | Contextualises what low-resolution imaging captures |
| Vickers & McCormick | 2024 | Arousal/movement signals pervade cortex including RSP; standard 0.7 correction used | RSP neuropil carries movement/arousal signals; potential confound for HD analysis |
| Stringer et al. | 2025 | Suite2p neuropil correction is standard; superneurons average out contamination | Validates pipeline approach; Rastermap as visualisation tool |

---

## References

- Kerr JND, Greenberg D, Helmchen F. 2005. "Imaging input and output of
  neocortical networks in vivo." PNAS 102(39):14063-14068.
  doi:10.1073/pnas.0506029102

- Helmchen F, Denk W. 2005. "Deep tissue two-photon microscopy." Nature
  Methods 2(12):932-940. doi:10.1038/nmeth818

- Dipoppa M, Ranson A, Krumin M, Pachitariu M, Carandini M, Harris KD. 2018.
  "Vision and locomotion shape the interactions between neuron types in mouse
  visual cortex." Neuron 98(3):602-615. doi:10.1016/j.neuron.2018.03.037

- Ali F, Kwan AC. 2019. "Interpreting in vivo calcium signals from neuronal
  cell bodies, axons, and dendrites: a review." Neurophotonics 7(1):011402.
  doi:10.1117/1.NPh.7.1.011402

- Nietz AK, Popa LS, Streng ML, Carter RE, Kodandaramaiah SB, Ebner TJ.
  2022. "Wide-field calcium imaging of neuronal network dynamics in vivo."
  Biology 11(11):1601. doi:10.3390/biology11111601

- Vickers ED, McCormick DA. 2024. "Pan-cortical 2-photon mesoscopic imaging
  and neurobehavioral alignment in awake, behaving mice." bioRxiv.
  doi:10.1101/2023.10.19.563159

- Stringer C, Zhong L, Syeda A, Du F, Kesa M, Pachitariu M. 2025.
  "Rastermap: a discovery method for neural population recordings." Nature
  Neuroscience 28:201-212. doi:10.1038/s41593-024-01783-4

### Additional references cited

- Peron SP, Freeman J, Iyer V, Guo C, Svoboda K. 2015. "A cellular
  resolution map of barrel cortex activity during tactile behavior." Neuron
  86(3):783-799. doi:10.1016/j.neuron.2015.03.027

- Pachitariu M, Stringer C, Dipoppa M, Schroeder S, Rossi LF, Dalgleish H,
  Carandini M, Harris KD. 2017. "Suite2p: beyond 10,000 neurons with standard
  two-photon microscopy." bioRxiv 061507. doi:10.1101/061507

- Keemink SW, Lowe SC, Pakan JMP, Dylda E, van Rossum MCW, Bhatt DH. 2018.
  "FISSA: A neuropil decontamination toolbox for calcium imaging signals."
  Scientific Reports 8:3493. doi:10.1038/s41598-018-21640-2
  https://github.com/rochefort-lab/fissa

- Chen T-W, Wardill TJ, Sun Y, et al. 2013. "Ultrasensitive fluorescent
  proteins for imaging neuronal activity." Nature 499:295-300.
  doi:10.1038/nature12354

- Jia H, Rochefort NL, Chen X, Konnerth A. 2011. "In vivo two-photon imaging
  of sensory-evoked dendritic calcium signals in cortical neurons."
  Nature Protocols 6:28-35. doi:10.1038/nprot.2010.169
