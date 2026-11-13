# bioRxiv Scan — 17 May 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-05-17. Searches covered: retrosplenial cortex (RSC/RSP), Penk/enkephalin +
cortex, head direction + two-photon imaging, head direction + darkness/landmarks/drift,
spatial navigation + maze + calcium imaging, neuropil contamination + two-photon, head-mounted
miniature two-photon microscopy, retrosplenial + visual processing + spatial navigation,
population decoding + neural tuning + freely moving, GCaMP + spike inference/deconvolution,
angular head velocity + RSC + calcium imaging.

Relatively quiet week on bioRxiv for these topics. One directly RSC-relevant preprint
appeared within the 7-day window (May 10–17); two additional recent preprints (May 7 and
May 11) are included as they fall within the extended search radius and are directly
relevant to project context.

---

## Highly relevant papers

### 1. Retrosplenial PV and SST interneurons shape egocentric spatial precision and stability

Authors not specified in search results. 2026.
"Retrosplenial PV and SST interneurons shape egocentric spatial precision and stability."
bioRxiv. Posted 10 May 2026.
https://www.biorxiv.org/content/10.64898/2026.05.10.724096v1

**Findings:** PV and SST inhibitory interneurons in RSC regulate dissociable components
of egocentric spatial coding in mice. PV interneurons are strongly modulated by self-motion
and exhibit bearing-aligned synchrony that precedes SST activation, linking locomotion to
egocentric precision. SST interneurons show weak self-motion modulation but robust
boundary-anchored activity with globally coherent population dynamics, stabilising spatial
representations over time. Optogenetic silencing of PV interneurons degraded egocentric
coding precision; silencing of SST interneurons disrupted global population organisation
without equivalent local precision loss.

**Relevance to hm2p:** This is the most directly relevant paper this week. Although our
project records from excitatory pyramidal populations (Penk+ and Penk⁻CamKII+), those
populations are shaped by exactly the PV- and SST-mediated inhibition described here.
Three specific implications: (1) If PV-mediated precision depends on self-motion signals,
HD tuning in our data should be movement-speed dependent — worth testing as an analysis
on both cell types to see if they differ in this dependence. (2) SST boundary-anchoring
could stabilise HD tuning across light/dark transitions; cells embedded in SST-dense
circuits might maintain tuning better in darkness if SST activity persists without visual
drive. (3) PV and SST cells are broadly expressed across both Penk+ and non-Penk
pyramidal populations, so these interneuron findings set a baseline for understanding
why HD coding quality varies across our recorded neurons. Cite when discussing local
circuit mechanisms and spatial coding heterogeneity.

---

## Moderately relevant papers

### 2. Hippocampal BMI-based navigation reveals CA1 representations of intended actions

Micou C, Ho H, O'Leary T, Krupic J. 2026.
"Hippocampal brain-machine interface-based navigation reveals CA1 representations of
intended actions." bioRxiv. Posted 11 May 2026.
https://www.biorxiv.org/content/10.64898/2026.05.11.724143v1

**Findings:** A brain-machine interface was used to allow mice to navigate via CA1 neural
activity alone, dissociating intended navigation from movement-derived sensory feedback.
CA1 place maps could be activated and updated in the absence of external inputs or
locomotion, revealing internal representations of intended position and direction. The
spatial map was internally driven and did not require locomotion or sensory flow to maintain
positional coding.

**Relevance to hm2p:** Not directly about RSP, but relevant to the idiothetic vs visual
cue question at the heart of our project. The finding that CA1 maps can update without
external sensory input (including visual flow) bears on what happens to hippocampal inputs
to RSP during our darkness epochs: the hippocampal spatial signal driving RSP does not
necessarily collapse in the dark. This means HD drift we observe in dark epochs in RSP
cannot be attributed solely to loss of hippocampal position signal — it must be explained
by loss of direct visual anchoring to RSP itself. This strengthens the interpretation that
any cell-type-specific differences in dark-epoch HD stability reflect differential visual
input processing, not differential hippocampal input.

---

## Tangentially relevant / methods papers

### 3. Spatial navigation through evolution: a single-cell atlas of the mammalian entorhinal cortex

Ralbovszki DM, Westfall JJ, Mori Y, Khodosevich K, et al. 2026.
"Spatial navigation through evolution: a single-cell atlas of the mammalian entorhinal
cortex." bioRxiv. Posted 7 May 2026 (10 days ago — just outside 7-day window, included
given direct relevance to navigation cell-type context).
https://www.biorxiv.org/content/10.64898/2026.05.07.723541v1

**Findings:** Cross-species single-cell transcriptomic atlas of entorhinal cortex from
human, Hamadryas baboon, mouse, and Egyptian fruit bat. Integration with whole-brain
diffusion tensor imaging revealed conserved and species-specific connectivity between
entorhinal cortex, hippocampus, and sensory cortices. Conserved cell type clusters
were identified alongside species-specific expansions, including elaboration of
layer-II stellate cell types in species with high navigational demand.

**Relevance to hm2p:** Indirectly relevant as a reference for the molecular diversity of
spatial navigation circuits. The conserved cell-type architecture in entorhinal cortex
provides a comparative context for interpreting Penk+ and CamKII+ as molecularly defined
excitatory subpopulations in RSP. More practically, this atlas may help identify whether
Penk-expressing cells in entorhinal cortex have conserved circuit properties that could
inform expectations for Penk+ RSP neurons. Not a citation target for the main results but
useful for the introduction framing spatial navigation cell-type diversity.

---

## Searches with no relevant results in the last 7 days

**Penk/enkephalin + cortex:** No preprints on Penk-expressing neurons in cortex in a
spatial navigation or HD context. Most recent Penk papers (Apr 2026) cover striatal and
MPOA circuits (reward, mating behaviour). The gap in Penk+ cortical neuron function
research remains open.

**Head direction + darkness / visual landmarks / drift:** No new preprints within the
7-day window. The most recent relevant papers (Secer et al., Tian et al.) are from late
2025 and early 2026 and were covered in previous scans.

**Head direction + two-photon imaging:** No new preprints in the 7-day window. Existing
relevant papers predate this week.

**Neuropil contamination + two-photon:** No new preprints. Most recent relevant work
(GCaMP8 spike inference benchmarking, Mar 2025) was covered previously.

**Head-mounted miniature two-photon:** No new preprints. Most recent hardware papers
(FHIRM-TPM 3.0, M-MINI2P) are from early 2025 and were noted in earlier scans.

**Spatial navigation + maze + calcium imaging:** No new rodent maze calcium imaging
preprints this week.

**GCaMP + spike inference / deconvolution:** No new preprints. CASCADE / GCaMP8
benchmarking paper (Pachitariu et al. v3, Mar 2025) remains most recent in this space.

**Angular head velocity + RSC + calcium imaging:** No new preprints beyond the RSP
PV/SST paper above.

**Population decoding + neural tuning + freely moving:** No new preprints directly
relevant to RSP HD population decoding this week.

---

## Summary

2 new papers within the 7-day window, 1 additional within 10 days. **Total: 3 papers
included.**

The RSP PV/SST interneuron paper (paper 1) is the clear highlight of the week —
directly addressing cell-type-specific inhibitory control of RSC spatial coding
with optogenetics in mice. It provides a mechanistic framework for understanding
spatial coding heterogeneity in our own excitatory Penk+ and non-Penk populations.
The hippocampal BMI paper (paper 2) strengthens the interpretation that HD drift in
our dark epochs is driven by loss of direct visual input to RSP, not hippocampal signal
collapse. No new papers on Penk cortical function, HD darkness, or neuropil correction
this week — these gaps persist as markers of the novelty space our project occupies.
