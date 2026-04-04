# Ali & Kwan 2019 -- Detailed Summary

## Citation

Ali F, Kwan AC. 2019. "Interpreting in vivo calcium signals from neuronal cell bodies, axons, and dendrites: a review." *Neurophotonics* 7(1):011402. doi:10.1117/1.NPh.7.1.011402

**Affiliations:** Department of Psychiatry, Yale University School of Medicine; Department of Neuroscience, Yale University School of Medicine.

---

## Overview

This review provides a systematic primer on interpreting calcium signals recorded from different neuronal compartments (soma, axon, dendrite) with fluorescent indicators in vivo. The central organising principle is that calcium transients in each compartment reflect distinct electrical events -- somatic signals primarily reflect action potentials, axonal signals reflect presynaptic activation, and dendritic signals can arise from backpropagating action potentials (bAPs), regenerative dendritic events, or synaptic inputs. The review emphasises the importance of direct in vivo calibration (simultaneous imaging and electrophysiology) and discusses the specific limitations of each interpretation.

---

## Key Findings and Arguments

### 1. Somatic calcium signals as a proxy for spiking

- In cortical pyramidal neurons, somatic calcium transients are driven primarily by high-threshold voltage-dependent calcium channels that open only during suprathreshold depolarisation.
- Simultaneous two-photon imaging and juxtacellular recording confirms that GCaMP6s/f transients faithfully reflect action potentials (Chen et al. 2013). Single-spike detection is reliable with GCaMP6s (time to peak 179 ms, decay 550 ms) and GCaMP6f (45 ms rise, 142 ms decay).
- Between action potentials, somatic fluorescence is largely unchanged, consistent with the high-threshold nature of somatic calcium channels. Any weak correlation between subthreshold membrane potential and somatic calcium signal likely reflects contamination from surrounding neuropil fluorescence rather than true somatic calcium entry.

### 2. Variability across cell types

- The fluorescence change per action potential differs between excitatory neurons, PV interneurons, and SST interneurons, with excitatory neurons showing the largest signal and interneurons showing approximately 50% lower signal per spike (with OGB-1). This difference is attributed to endogenous calcium buffers in GABAergic interneurons.
- GCaMP6 has a nonlinear (cooperative) calcium-to-fluorescence relationship, unlike OGB-1, which complicates quantitative spike inference at high firing rates.

### 3. Detecting decreases in firing rate

- Calcium signals can report decreases in firing rate, particularly for neurons with high baseline activity. However, hyperpolarisation without change in spiking has no detectable effect on somatic calcium.

### 4. Axonal calcium signals

- Axonal bouton calcium transients reflect presynaptic depolarisation and can serve as a proxy for afferent input activity. There is substantial variability in calcium amplitude across boutons from the same cell (>10-fold), which depends on postsynaptic cell type and neuromodulatory state.
- The review includes unpublished data from axons of RSC neurons projecting to Cg1/M2, showing graded calcium responses to electrical stimulation of RSC.

### 5. Dendritic signals: multiple sources

- Backpropagating action potentials (bAPs) cause calcium influx in proximal dendrites but attenuate with distance from the soma. In L2/3 pyramidal neurons, bAP-associated calcium becomes negligible beyond 200--250 um from the soma.
- Regenerative dendritic events (NMDA spikes, calcium spikes) produce large-amplitude calcium transients that can spread across multiple tuft branches.
- Synaptic inputs produce localised calcium transients in dendritic spines, primarily through NMDA receptors. These can be isolated by subtracting the shaft signal from the spine signal, though this procedure removes synaptically evoked signals that coincide with bAPs or dendritic spikes.

---

## Neuropil Contamination

The review addresses neuropil contamination directly in the context of somatic calcium imaging:

- **Source of contamination:** Any weak correlation observed between somatic calcium signals and subthreshold membrane potential fluctuations "probably does not come from the imaged cell, but rather originates from contamination of fluorescence signals from the surrounding neuropil, which would reflect the local network activity."
- **Mechanism:** Because two-photon imaging collects fluorescence from a finite axial volume (the point-spread function extends several microns in z), out-of-focus fluorescence from neuropil structures (axons, dendrites, boutons) surrounding the soma contributes to the measured somatic signal.
- **Implication for interpretation:** Without neuropil correction, somatic calcium signals may appear to carry information about subthreshold or network-level activity that does not originate from the imaged neuron. This is a spurious correlation that can confound analyses of tuning properties.
- **Cell-type differences in susceptibility:** Neurons with sparse firing (e.g., L2/3 pyramidal cells) are more susceptible to neuropil contamination artefacts because their true somatic signal is small relative to the neuropil contribution. Interneurons with high firing rates have larger somatic signals that partially mask neuropil contamination, but their lower fluorescence change per spike (due to endogenous calcium buffers) may partially offset this advantage.

---

## Relevance to hm2p

### 1. Neuropil contamination is a primary concern for HD tuning measurements

The hm2p project measures head direction tuning from somatic GCaMP signals in RSP neurons. If neuropil signals carry HD information (from surrounding HD-tuned axons and dendrites), then uncorrected neuropil contamination could artificially inflate apparent HD tuning in weakly tuned or untuned neurons. This is particularly relevant for:

- **Penk+ vs Penk-CamKII+ comparisons:** If one population has sparser firing than the other, it will be more susceptible to neuropil contamination, potentially creating a spurious difference in tuning properties.
- **Light vs dark comparisons:** If neuropil signals change between conditions (e.g., overall activity levels differ), this could create apparent changes in tuning that reflect network-level rather than single-cell effects.

### 2. Cell-type-specific calcium dynamics must be considered

The review documents that different cell types have different fluorescence-per-spike amplitudes (excitatory > PV > SST with OGB-1). If Penk+ and Penk-CamKII+ RSP neurons differ in their endogenous calcium buffering or GCaMP expression levels, this could affect:

- **MVL comparisons:** Neurons with lower signal-per-spike will have noisier tuning curves and lower apparent MVL, even if their underlying HD selectivity is identical.
- **Spike inference:** CASCADE or other deconvolution methods may perform differently across populations if the underlying calcium dynamics differ.

### 3. RSC axon imaging data in this paper

The review includes data from RSC projection axons to Cg1/M2 expressing GCaMP6s, demonstrating that axonal calcium signals in RSP projections can be reliably measured. This is relevant context for understanding what the neuropil signal in RSP imaging fields might contain -- it includes axonal projections from thalamic HD inputs (ADN/PoS), visual cortex, and hippocampal formation.

### 4. Imaging rate limitations

At 9.6 Hz, the hm2p dataset cannot resolve individual spikes in neurons with firing rates above approximately 5 Hz (Nyquist limit). The review's discussion of linearity breakdown at high firing rates and GCaMP saturation effects should be considered when interpreting calcium signals as a proxy for firing rate, particularly for putative high-firing neurons.

---

## Key References Cited

| Reference | Relevance |
|---|---|
| Chen et al. 2013 (Nature) | GCaMP6 characterisation, single-spike calibration |
| Kwan & Dan 2012 (Curr Biol) | Simultaneous imaging and electrophysiology, cell-type differences in fluorescence per spike |
| Peron et al. 2015 (Neuron) | Neuropil correction methods for two-photon imaging |
| Kerr et al. 2005 (PNAS) | Neuropil signals represent axonal input activity |
| Helmchen et al. 1996 (Biophys J) | Calcium buffering and action potential-evoked calcium signalling |
