# bioRxiv Scan — 7 April 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-04-07. Searches covered: retrosplenial cortex, Penk/enkephalin +
cortex, head direction + two-photon imaging, head direction + darkness/landmarks/drift,
spatial navigation + maze (rodents), visual processing in RSP, spatial navigation in RSP,
head-mounted two-photon microscopy, calcium imaging + maze navigation, neuropil
contamination + two-photon, cell-type-specific RSC mouse, spike inference + GCaMP8 +
CASCADE.

Papers already listed in the 2026-04-02 and 2026-04-04 scans are not repeated here.

**Note on scan yield:** The April 5–7 posting window contained no new preprints directly
relevant to RSP HD coding or spatial navigation in freely moving rodents. The papers below
were surfaced by the search but were posted earlier (September 2024 – March 2025) and
missed in prior scans because they appeared under less specific search terms. One version
update is also noted.

---

## Highly relevant papers

### 1. Unique transcriptomic cell types of granular RSC are conserved across mice and rats

Brooks IAW, Rybicki-Kler C, Jedrasiak-Cape I, Ahmed OJ. 2024.
"Unique Transcriptomic Cell Types of the Granular Retrosplenial Cortex are Preserved
Across Mice and Rats Despite Dramatic Changes in Key Marker Genes." bioRxiv (September
2024); now published as J. Neuroscience 45(48) e2246242025.
https://www.biorxiv.org/content/10.1101/2024.09.17.613545v1

**Findings:** Single-nucleus RNA-seq of granular RSC (RSG) in mice and rats identified
two cell types unique to RSG — L2/3 low-rheobase (LR) neurons and L5a RSG neurons —
that together constitute more than 50% of all RSG excitatory neurons in both species.
The same two subtypes are conserved despite dramatic species differences in canonical
marker genes (Scnn1a, which labels mouse L5a RSG, is absent in rat). The study provides
cross-species marker gene panels and notes that the LR subtype is the same cell class
studied in Jedrasiak-Cape et al. 2024 (already in April 4 scan) for angular velocity
coding. From Ahmed lab at University of Michigan; published in J. Neuroscience 2025.

**Relevance to hm2p:** This is the transcriptomic characterisation of the same RSG cell
types whose functional roles we are investigating. The LR subtype (>25% of RSG excitatory
neurons) and the L5a RSG subtype are genetically defined excitatory populations. Neither
paper maps these to Penk+ vs Penk⁻CamKII+ identity — that correspondence is unknown and
our study could address it. If Penk+ neurons map to one of these subtypes (LR or L5a),
this paper defines their transcriptomic identity. We should cite this alongside the
Jedrasiak-Cape et al. 2024 paper when discussing RSP excitatory cell-type diversity. The
Ahmed lab papers together establish that RSG excitatory subtypes differ in both gene
expression and functional computation — a strong precedent for our hypothesis.

---

## Moderately relevant papers

### 2. Single-cell spatial transcriptomics of RSC during memory consolidation

Bliese SR, Basu B, Beyer SE, Chatterjee S. 2025.
"Single-cell resolution spatial transcriptomic signature of the retrosplenial cortex
during memory consolidation." bioRxiv.
https://www.biorxiv.org/content/10.1101/2025.03.12.642891v1

**Findings:** Used spatial transcriptomics (Visium + single-cell resolution) to map gene
expression changes in RSC at multiple time points after a spatial memory task (Morris
water maze). Identified a time-windowed upregulation of transcription regulation, protein
folding, and MAPK pathway genes across RSC subdivisions. Distinct gene expression
signatures were found in RSC excitatory neurons vs inhibitory interneurons, with laminar
specificity in the early consolidation window.

**Relevance to hm2p:** Provides a spatial transcriptomic baseline for RSC cell-type-
specific gene expression during spatial tasks. The spatial resolution allows comparison
of gene expression patterns across RSC layers and subdivisions (granular vs dysgranular).
Not directly about HD coding or navigation in a dark/light condition, but offers
transcriptomic context for interpreting our functional calcium imaging data: the cell
types we record (Penk+ in one population, Penk⁻CamKII+ in another) undergo distinct
transcriptional responses during spatial tasks. The MAPK findings could be relevant to
activity-dependent plasticity in HD representations across sessions.

---

## Tangentially relevant / methods papers

### 3. Spike inference from calcium imaging data with GCaMP8 indicators (v3)

Berens P, Bethge M, Friedrich J, Lütcke H, et al. (CASCADE consortium). 2025.
"Spike inference from calcium imaging data acquired with GCaMP8 indicators." bioRxiv v3.
https://www.biorxiv.org/content/10.1101/2025.03.03.641129v3.full

**Findings:** Benchmarked CASCADE, OASIS, and MLSpike against ground-truth simultaneous
electrophysiology and GCaMP8 imaging. CASCADE adapted for GCaMP8 outperforms unadapted
versions. GCaMP8s and GCaMP8m (but not GCaMP8f or earlier GCaMP6/7 variants) reliably
detect isolated action potentials at realistic noise levels due to faster rise kinetics.
The authors provide pretrained CASCADE models for GCaMP8 variants and best-fit OASIS/
MLSpike parameters.

**Relevance to hm2p:** Our pipeline uses CASCADE for spike inference (Stage 4). If the
imaging data used GCaMP6 or GCaMP7 variants (as in our legacy sessions), the existing
CASCADE models remain appropriate. However, if any future sessions use GCaMP8, pretrained
models should be updated using the parameters in this paper. The paper also confirms
CASCADE as the benchmark method for spike inference — supports our choice over OASIS/V&H.
Note: the v3 revision (surfaced this week) updates supplementary benchmarks; the core
findings are unchanged from v1 (March 2025).

---

### 4. Version update: parallax error paper revised to v2

Authors not fully specified. 2025 (v2 posted 2026).
"Parallax error in the head-direction system indicates simple cue-anchoring mechanism."
bioRxiv v2.
https://www.biorxiv.org/content/10.1101/2025.04.25.650191v2.full

The April 2, 2026 scan cited v1 of this paper ("Head-Direction Cells in Postsubiculum
Show Systematic Parallax Errors During Visual Anchoring"). The title has been revised
in v2 to "Parallax error in the head-direction system indicates simple cue-anchoring
mechanism," reflecting a more mechanistic framing. The core finding is unchanged: the HD
system uses a simple visual anchoring mechanism (not explicit parallax correction), and
the residual error is explained by a combination of angular velocity integration and
visual anchoring. No change to our relevance assessment. Update citation if citing.

---

## Searches with no new relevant results

**Retrosplenial cortex (April 5–7, 2026):** No new RSP preprints posted in the past
three days. Most recent RSP preprint remains Joo et al. (March 27, 2026), listed in the
April 4 scan.

**Penk/enkephalin + cortex:** No new preprints. Recent Penk papers continue to address
striatal/subcortical contexts. The gap in Penk+ cortical neuron function remains open.

**Head direction + two-photon imaging:** No new preprints in the scan window. Searches
returned the same set of papers from prior weeks.

**Head direction + darkness/landmarks/drift:** No new experimental papers. The
computational modelling preprints (Sarramone et al. MEC oscillations; 3D HD attractor
model) were both captured in the April 4 scan.

**Spatial navigation + maze + calcium imaging:** No new preprints from April 5–7.

**Head-mounted two-photon microscopy:** No new instruments or methods papers this week.

**Neuropil contamination + two-photon:** No new methods papers. Standard approaches
(Suite2p fixed-coefficient, FISSA) remain current.

---

## Summary

**Scan yield:** Low for this specific window (April 5–7). The prior two scans (April 2
and April 4) covered a week-long burst of relevant papers; the current three-day window
produced no new directly relevant RSP/HD preprints.

**Most significant new item:** The Brooks/Ahmed RSG transcriptomics paper (now published
in J. Neuroscience) completes the picture from the April 4 scan: the Jedrasiak-Cape et al.
2024 functional paper and the Brooks et al. 2024 transcriptomic paper together establish
that the LR and L5a RSG excitatory subtypes are the dominant cell classes in RSG, are
functionally distinct (AHV coding), and are transcriptomically conserved across species.
Our study adds the question of whether these subtypes map to Penk+ vs Penk⁻CamKII+
identity and whether they differ in HD coding.

**Papers to update in citation list:**
- Brooks et al. 2024 (RSG transcriptomics) — cite alongside Jedrasiak-Cape et al. 2024
  in introduction; note it is now published in J. Neuroscience
- Bliese et al. 2025 (RSC spatial transcriptomics) — cite in methods/discussion when
  discussing RSC cell-type context
- Parallax error paper — update citation to v2 title

**Penk+ gap status:** Unchanged. No published work characterises Penk-expressing RSP
neurons in any functional context. Three consecutive scans confirm this gap is current.

**Total new papers this scan:** 3 (2 missed from prior periods; 1 version update). No
papers from the April 5–7 posting window were relevant.
