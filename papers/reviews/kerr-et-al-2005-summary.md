# Kerr, Greenberg & Helmchen 2005 -- Detailed Summary

## Citation

Kerr JND, Greenberg D, Helmchen F. 2005. "Imaging input and output of neocortical networks in vivo." *Proceedings of the National Academy of Sciences* 102(39):14063-14068. doi:10.1073/pnas.0506029102

**Affiliations:** Department of Cell Physiology, Max Planck Institute for Medical Research, Heidelberg, Germany.

---

## Overview

This paper demonstrates that two-photon calcium imaging of bulk-loaded neocortical tissue can simultaneously resolve both the input and output activity of local cortical circuits in vivo. The key finding is that the neuropil calcium signal -- termed the "optical encephalogram" (OEG) -- originates predominantly from axonal structures and provides a measure of local afferent input activity. Simultaneously, somatic calcium transients report neuronal output (spiking) with single-cell and single-spike resolution. Using these two signals, the authors characterise spontaneous activity during cortical Up states in anaesthetised rats, finding that spiking is sparse (<0.1 Hz), heterogeneously distributed, and directly dependent on the amplitude of afferent input as measured by the OEG.

---

## Key Findings and Arguments

### 1. Single-spike resolution in bulk-loaded tissue

Using simultaneous cell-attached recordings and two-photon imaging of OGB-1-AM loaded L2/3 neurons, the authors demonstrate that 97% of single action potentials and 100% of bursts are optically detected. Mean single-AP transient amplitude is 10.0 +/- 0.9% dF/F with a decay time constant of 0.82 +/- 0.42 s (n = 10 transients, 5 animals). Calcium transient amplitude correlates with the number of action potentials (R-squared = 0.81).

### 2. The neuropil signal (OEG) is predominantly axonal

This is the central finding relevant to neuropil contamination. The authors dissect the origin of neuropil fluorescence fluctuations using two complementary approaches:

- **Deep loading experiment:** When OGB-1 is loaded selectively into layer 5 (labelling deep neurons and their apical dendrites projecting to superficial layers), the OEG in superficial layers is absent or markedly reduced. Peak dF/F amplitudes were 1.7 +/- 1.1% at 150 um depth (compared to 10--30% with standard superficial loading). This demonstrates that dendritic calcium transients contribute minimally to the neuropil signal.
- **AMPA receptor blockade:** Local application of the AMPA antagonist GYKI53655 abolished postsynaptic spiking (mean firing rate dropped from 0.038 to 0.014 Hz) but did not change the amplitude of OEG fluctuations (P = 0.87). This confirms that the OEG is insensitive to postsynaptic activity and reflects presynaptic (axonal) calcium signals.

### 3. Composition of cortical neuropil

The paper cites anatomical data: cortical neuropil consists of approximately 50% axons and presynaptic boutons, 30% dendrites and dendritic spines, and 10% glial processes. Given that the OEG is predominantly axonal, it reflects the bulk average of action-potential-evoked calcium transients in presynaptic boutons and axons activated during Up-state periods.

### 4. OEG correlates with ongoing electrical activity

The neuropil fluorescence signal is tightly correlated with both the electrocorticogram (ECoG; mean peak correlation 0.70 +/- 0.1) and intracellular membrane potential recordings (mean peak correlation 0.75 +/- 0.1). This correlation does not depend on imaging depth when loading is in L2/3, indicating that the OEG reflects a spatially uniform measure of input activity.

### 5. Input-output relationship revealed optically

By plotting the number of postsynaptic calcium transients against OEG amplitude, the authors reveal a threshold-linear input-output function: stronger axonal activation (larger OEG) produces more neuronal spikes in the local population.

### 6. Sparse, heterogeneous spontaneous activity

On average, only 10.6 +/- 2.1% of neurons are active during each Up state. The active subpopulation changes continuously over time (not a fixed subset). Mean firing rate is 0.048 +/- 0.002 Hz (n = 212 neurons). Inter-event intervals follow an exponential distribution (Poisson-like).

---

## Neuropil Contamination

This paper is foundational for understanding neuropil contamination in two-photon calcium imaging:

### What the neuropil signal represents

- The neuropil signal is **not** a passive background or noise. It is an active, physiologically meaningful signal reflecting the aggregate presynaptic (axonal) calcium transients from afferent inputs to the local cortical region.
- It is tightly coupled to ongoing network state (Up/Down states) and to the level of excitatory drive to the local circuit.
- Neuropil fluctuations have amplitudes of 10--30% dF/F at 0.5--1 Hz frequencies, which is comparable to or larger than single-spike somatic transients (approximately 10% dF/F).

### Implications for somatic signal extraction

- Because the neuropil signal can be as large as or larger than single-spike somatic signals, any fluorescence measured within an ROI drawn around a cell soma will be contaminated by the surrounding neuropil signal.
- This contamination will impose a correlated baseline fluctuation on all neurons in the field of view, potentially creating spurious correlations between neurons and between neural activity and behavioural variables.
- The finding that the OEG is primarily axonal means that neuropil contamination introduces information about the **inputs** to the local circuit into the measured somatic signal, not about the local postsynaptic activity.

### Why dendrites contribute minimally

- The deep-loading experiment demonstrates that dendritic calcium transients (from backpropagating action potentials in apical trunks during subthreshold Up states) contribute little to the neuropil signal. This is consistent with the finding that subthreshold depolarisation does not produce detectable calcium transients in apical dendritic trunks (Waters et al. 2003).
- This means the neuropil signal is dominated by long-range and local axonal inputs, not by dendritic activity of nearby neurons.

---

## Relevance to hm2p

### 1. The neuropil in RSP imaging fields carries afferent HD information

RSP receives dense projections from the anterior dorsal thalamic nucleus (ADN) and postsubiculum (PoS) -- both carrying head direction signals. The axonal boutons from these afferents will contribute to the neuropil signal in the imaging plane. This means the neuropil in hm2p imaging fields likely carries HD-tuned axonal signals from the classical HD circuit. If this neuropil signal is not properly subtracted from somatic ROIs, it will add HD-correlated activity to every neuron's trace, inflating apparent HD tuning across the population.

### 2. The magnitude of contamination is substantial

Kerr et al. report neuropil fluctuations of 10--30% dF/F, comparable to single-spike transients. In the hm2p dataset at 9.6 Hz imaging rate, the neuropil-to-soma signal ratio could be substantial, particularly for weakly firing neurons. The standard Suite2p neuropil correction (F - 0.7 * Fneu) may not fully remove this contamination.

### 3. Light/dark condition differences in neuropil signals

If the neuropil signal in RSP reflects afferent input from visual cortex (which projects to RSP) in addition to HD circuit inputs, then neuropil signals may differ systematically between light and dark epochs. This could create a confound for light/dark comparisons of tuning properties if neuropil correction is inadequate. A change in apparent HD tuning between conditions could partially reflect changes in neuropil contamination rather than changes in somatic firing.

### 4. Penk+ vs Penk-CamKII+ vulnerability to contamination

If one population (e.g., Penk+) has lower mean firing rates than the other, its somatic signal-to-neuropil ratio will be lower, making it more susceptible to neuropil contamination artefacts. Any observed difference in HD tuning between populations must be evaluated against this potential confound.

### 5. Methodological recommendation

The OEG concept suggests that examining the neuropil signal itself in hm2p data could be informative. If the neuropil signal shows HD tuning (which it likely does, given thalamic inputs), this would quantify the contamination risk and help calibrate appropriate correction coefficients.

---

## Key References Cited

| Reference | Relevance |
|---|---|
| Stosiek et al. 2003 (PNAS) | Bulk loading of calcium indicators in vivo |
| Nimmerjahn et al. 2004 (Nat Methods) | Sulforhodamine 101 for astrocyte identification |
| Waters et al. 2003 (J Neurosci) | No detectable dendritic calcium during subthreshold Up states |
| Braitenberg & Schuz 1998 | Anatomical composition of cortical neuropil |
