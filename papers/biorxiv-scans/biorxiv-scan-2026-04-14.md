# bioRxiv Scan — 14 April 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-04-14. Searches covered: retrosplenial cortex (RSP/RSC),
Penk/enkephalin + cortex, head direction + two-photon calcium imaging, head direction +
darkness/landmarks/drift, spatial navigation + maze (rodents), visual processing in RSP,
spatial navigation in RSP, head-mounted two-photon microscopy, calcium imaging + maze
navigation, neuropil contamination + two-photon, population decoding + head direction,
Suite2p/CASCADE/neuropil spike inference, CEBRA/keypoint-MoSeq/DeepLabCut + calcium
imaging, spatial coding + light/dark alternation, prefrontal–thalamic + hippocampus +
navigation, anterior thalamus + RSC + head direction.

Papers already listed in scans from 2026-04-02, 2026-04-04, 2026-04-05, and 2026-04-06
are not repeated here.

Note: direct programmatic access to biorxiv.org (REST API, collection pages) returned
HTTP 403 during this scan. All results are based on indexed web search results. Papers
posted in the last 1–3 days may not yet be indexed by search engines.

---

## Highly relevant papers

No new highly relevant papers were found in the April 7–14, 2026 window. Searches
covering RSP HD coding, Penk/enkephalin + cortex, head direction + calcium imaging,
and head direction + darkness/landmarks returned no new, unseen preprints directly
addressing our core topics.

---

## Moderately relevant papers

### 1. Locomotion-invariant prefrontal–thalamic goal states organise spatially aligned episode-specific hippocampal maps

Golipour Z, Yen S-F, Üstüner C, Ito HT. 2026.
"Locomotion-invariant prefrontal–thalamic goal states organize spatially aligned
episode-specific hippocampal maps." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.04.09.716651v1

Posted: April 9, 2026.

**Findings:** During maze navigation under different goal configurations, hippocampal
CA1 encoded goal state in a population subspace orthogonal to the spatial coding axis,
keeping episode-specific maps spatially aligned rather than globally remapping. The goal-
state signal was supplied by a medial prefrontal cortex–nucleus reuniens (thalamus)
pathway that maintained persistent representations across locomotion and immobility and
was reinstated when previously experienced goal configurations recurred. Silencing nucleus
reuniens selectively abolished CA1 goal-state coding while sparing spatial coding.

**Relevance to hm2p:** Relevant at two levels. First, it demonstrates that thalamic
input (nucleus reuniens) carves out a specific coding subspace in cortical/hippocampal
networks orthogonal to spatial coding. By analogy, anterior thalamic nucleus (ATN) input
to RSP may set a "context" subspace (light vs dark) orthogonal to the HD subspace — a
population geometry hypothesis worth testing on our sync.h5 data using CEBRA or PCA
subspace alignment. Second, the locomotion invariance of the prefrontal-thalamic goal
signal contrasts with the locomotion-dependent HD attractor dynamics described in the
April 6 scan. This suggests different thalamic relays impose different locomotion
dependencies on downstream cortical representations; ATN HD input to RSP may therefore
have a distinct locomotion profile from reuniens. Direct subjects are CA1 and mPFC, not
RSP — background relevance only.

---

## Tangentially relevant / methods papers

### 2. A multiplexed striatal architecture for generalised spatial goal progress

Authors not fully specified. 2026.
"A multiplexed striatal architecture for generalized spatial goal progress." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.04.08.716650v1

Posted: April 8, 2026.

**Findings:** Nucleus accumbens (NAc) populations encode a scale-invariant
distance-to-goal signal normalised by total journey length, across maze geometries and
task rules. Orthogonal NAc subspaces simultaneously encode distances to multiple
(previous and current) goals. This encoding persisted during hippocampal and MEC
silencing but required dopaminergic input from VTA.

**Relevance to hm2p:** Not directly relevant to RSP HD coding or the light/dark
manipulation. However, the demonstration of orthogonal subspace coding for distinct
spatial variables is a methodological consideration for our population analyses (CEBRA,
linear decoder). We should test whether RSP HD coding and "arm identity" coding in the
rose maze are orthogonal or overlapping — if not orthogonal, HD decoding may be
confounded by position-in-maze correlations. A statistical analysis design question
rather than a direct biological parallel.

---

### 3. Invariant activity sequences across the mouse brain

Bimbard C, Harris KD, Carandini M. 2025/2026.
"Invariant activity sequences across the mouse brain." bioRxiv.
https://www.biorxiv.org/content/10.64898/2025.12.20.695676v3

Posted: December 20, 2025 (v1); v3 updated April 8, 2026.

**Findings:** Neurons throughout the mouse brain fire with fixed latencies relative to
population mean activity, forming stereotyped sequences that are stable for weeks. These
invariant sequences appear during both stimulus-driven and spontaneous activity, and
were found in every recorded brain region in large-scale Neuropixels recordings.
RSC (retrosplenial cortex) is among the brain regions surveyed.

**Relevance to hm2p:** Invariant population sequence structure in RSC means that
single-neuron firing times within a population burst are not independent — a caveat
for population decoding analyses (CEBRA, linear decoder) that assume independent noise.
The multi-week stability of sequence structure also suggests that any Penk+ vs non-Penk
sequence differences we observe would persist across sessions. The paper does not
differentiate sequences by cell type within RSC; relevance to HD coding is indirect.

---

## Notable journal publications (not new preprints)

### Wei YT, Couto J, Kloosterman F, Bonin V. 2026. Nature Communications.

"Anterior and posterior retrosplenial cortex form distinct visuospatial circuits in
the mouse." Nature Communications.
doi:10.1038/s41467-026-70762-z

The bioRxiv preprint (June 2025) was reviewed in the 2026-04-02 scan. The paper is
now published in Nature Communications (peer-reviewed). The findings — an anterior–
posterior gradient in RSC with anterior neurons showing sharper position tuning and
preference for fast visual stimuli, and posterior neurons showing broader tuning — are
unchanged from the preprint. The journal version now provides the final citation for
the manuscript. See the April 2 scan for the full relevance assessment.

---

## Searches with no new results

**Retrosplenial cortex + spatial navigation:** No new preprints this week. Searches
returned the same 2024–2025 papers already catalogued across the four previous scans.

**Penk/enkephalin + cortex:** No new preprints on Penk-expressing neurons in cortex.
The only 2026 Penk preprint found in the scan window was on MPOA Penk neurons in mating
behaviour (April 3, 2026 — just outside this window) — not relevant to RSP or navigation.
The functional characterisation gap for Penk+ RSP neurons remains open after five
consecutive scans.

**Head direction + darkness/landmarks/drift:** No new experimental papers. Searches
returned previously catalogued papers (parallax errors in postsubiculum v4 update,
computational HD modelling, active locomotion / HD attractor from April 6 scan).

**Head direction + two-photon calcium imaging:** No new papers. Field quiet since
Tian et al. 2026 MEC miniature 2P + light-deprivation paper (April 2 scan).

**Calcium imaging + maze navigation (rodents):** No new papers this week. NaviGraph
(RSC maze calcium imaging; Koren Iton et al. 2025; April 6 scan) remains the most
recent methodological parallel.

**Neuropil contamination + two-photon:** No new methods papers. Suite2p update
(Pachitariu & Stringer 2026; April 5 scan) remains the most recent relevant paper.

**Head-mounted two-photon / miniature two-photon:** No new technology papers this
week beyond those covered in the April 2 and April 6 scans.

**Suite2p / CASCADE / roiextractors:** No new preprints on these tools.

**CEBRA / keypoint-MoSeq / DeepLabCut:** No updates this week.

**Calcium imaging + light/dark alternation:** No new papers. This experimental design
remains underrepresented in the calcium imaging literature.

---

## Total count

- Highly relevant: 0
- Moderately relevant: 1 (Golipour et al., April 9, 2026)
- Tangentially relevant / methods: 2 (NAc spatial goal paper, Bimbard et al. sequences)
- Notable journal publications: 1 (Wei et al., Nature Communications)
- Searches with no new results: all topic-specific searches

---

## Summary

A quiet week for core hm2p topics. No preprints addressed RSP HD cell-type specificity,
visual landmark anchoring in RSP, or Penk-expressing cortical neurons. The most
noteworthy new paper is Golipour et al. (April 9), which motivates testing whether a
light/dark context signal in RSP occupies a subspace orthogonal to the HD coding
subspace — directly actionable in our Stage 6 population analyses.

The Wei et al. preprint from the April 2 scan has now been published in Nature
Communications, providing the final citation for the RSC A-P gradient paper.

**Running confirmation of Penk+ cortex gap:** Five consecutive scans (April 2, 4, 5,
6, 14) have returned zero preprints characterising Penk-expressing neurons in any
cortical region in a spatial navigation, head direction, or calcium imaging context.

**Key action item from this scan:** Test whether RSP HD coding and arm-identity coding
in the rose maze occupy orthogonal population subspaces (motivated by Golipour et al.
goal-state orthogonality result). Add to Stage 6 analysis plan.
