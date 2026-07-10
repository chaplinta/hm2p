# bioRxiv Scan — 10 July 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-07-10. Primary window: July 9–10, 2026 (since the July 9 scan).
The broader 7-day window (July 3–10) overlaps substantially with prior scans; papers
already captured in the 2026-04-02 through 2026-07-09 scans are not repeated here,
with one exception: a paper posted July 3 (doi:10.64898/2026.07.03.736438) was not
found by the July 4–9 scans due to search-engine indexing lag and is captured here for
the first time.
Searches covered: retrosplenial cortex (RSP/RSC), Penk/enkephalin + cortex, head
direction + two-photon imaging, head direction + darkness/landmarks/drift, spatial
navigation + maze (rodents), visual processing in RSP, spatial navigation in RSP,
head-mounted two-photon microscopy, calcium imaging + maze navigation, neuropil
contamination + two-photon, population decoding + head direction + cortex,
calcium imaging + cell-type identification.

Papers listed in any prior scan (2026-04-02 through 2026-07-09) are not repeated here
except where indexing lag caused a genuine new capture.

Note: direct bioRxiv API and search page access (biorxiv.org) returned HTTP 403 errors
throughout this session. All searches performed via web search engine; newly posted
papers indexed with a lag of days to weeks may not be captured.

---

## Highly relevant papers

No new highly relevant preprints found for July 9–10, 2026.

---

## Moderately relevant papers

### 1. Coordinated acetylcholine release and adaptation of neuronal representations in the retrosplenial cortex during contextual uncertainty

Goodwin D, Boja A, Tessereau C, Issa JB, Li G, Li Y, Dombeck D et al. 2026.
"Coordinated acetylcholine release and adaptation of neuronal representations in the
retrosplenial cortex during contextual uncertainty." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.07.03.736438
Posted: 3 July 2026. First captured in this scan (indexing lag from prior sessions).

**Findings:** Simultaneous measurement of acetylcholine (ACh) release and 2P calcium
imaging of neuronal population activity in the dysgranular retrosplenial cortex (dRSC)
of head-fixed mice navigating a virtual environment with periodic contextual changes.
ACh release scaled with expected uncertainty (routine contextual variability) but showed
an additional phasic response to unexpected uncertainty coinciding with discrete
context-boundary transitions. Concurrent population activity adapted its representational
geometry at context shifts: individual cell tuning curves rescaled but the global
manifold topology was preserved. The dRSC was confirmed to house representations of
spatial position, boundary proximity, visual landmark identity, and reward location —
consistent with its role as a contextual spatial registry. ACh release correlated
with the magnitude of neural representational change, suggesting cholinergic drive
mediates the gain-change mechanism rather than triggering categorical remapping.

**Relevance to hm2p:** Light-off epochs in our paradigm are a discrete contextual
change: abrupt removal of visual cues while self-motion continues. By this paper's
framework, each light transition should elicit a phasic ACh response in RSC, which
in turn could transiently modulate HD tuning strength. This predicts a systematic
tuning-degradation peak immediately after light-off onset followed by partial recovery
within the epoch as path integration stabilises the representation — a testable pattern
in our existing sync.h5 data (bin HD tuning quality in 10–15 s windows within the first
30 s of each dark epoch). The confirmation that dRSC houses visual landmark
representations also supports our prediction that Penk+ RSP neurons (if enriched in
dRSC) would be more affected by cue removal than Penk⁻CamKII+ cells. Dombeck lab
(Northwestern): high-quality 2P imaging; results are likely to be rigorous and to have
influenced the field by the time hm2p is submitted.

---

## Tangentially relevant / methods papers

No new tangentially relevant or methods papers found for July 9–10, 2026.

---

## Searches with no new results (July 9–10)

**Retrosplenial cortex (RSP/RSC):** No new RSC/RSP preprints identified on spatial
coding, HD tuning, visual processing, or cell-type-specific function beyond the paper
captured above.

**Penk/enkephalin + cortex:** No new preprints. This is the fifty-seventh consecutive
scan cycle without a cortical Penk+ navigational or spatial paper. All Penk hits across
the full monitoring period remain subcortical (striatum, hypothalamus, brainstem, dorsal
raphe, enteric nervous system). The absence of any literature characterising Penk+ RSP
neurons in a spatial or HD context remains complete — confirming an open literature gap.

**Head direction + two-photon imaging (freely moving):** No new preprints.

**Head direction + darkness / visual landmarks / drift:** No new experimental papers.

**Spatial navigation + maze + calcium imaging (freely moving mouse):** No new papers
matching all criteria.

**Visual processing in RSP / spatial navigation in RSP:** No new preprints.

**Head-mounted / miniature two-photon microscopy:** No new technology papers.

**Neuropil contamination + two-photon:** No new correction methods or benchmarking papers.

**Population decoding + head direction + cortex:** No new papers.

---

## Summary

**Total new relevant preprints this scan:** 1 (moderately relevant; posted July 3,
first indexed in this session due to search-engine lag).

**Cumulative picture (April 2–July 10, 99 days, 58 scans):**

No preprint has characterised HD tuning of genetically-defined RSP excitatory neuron
populations (Penk+ or Penk⁻CamKII+) across any scan window. The substantive
preprints within the active monitoring window, ranked by recency:

- **Goodwin, Dombeck et al. 2026** (July 3): ACh release in dRSC coordinates
  representational adaptation during contextual uncertainty. Relevant for interpreting
  tuning dynamics at light-off transitions. See above.
  https://www.biorxiv.org/content/10.64898/2026.07.03.736438

- **Oh et al. 2026** (May 10): RSC PV/SST interneurons and egocentric spatial coding
  precision/stability. PV cells govern egocentric coding precision via motion-linked
  bearing-aligned synchrony; SST cells govern long-term global stability via
  boundary-anchored activity. Optogenetic silencing confirms independent roles. Informs
  Discussion framing of Penk+ vs Penk⁻CamKII+ HD differences: excitatory cell-type
  differences are likely shaped by differential PV/SST gate state.
  https://www.biorxiv.org/content/10.64898/2026.05.10.724096v1

- **Coordinated representational drift across mouse cortex** (May 5): RSC confirmed as
  highest-density spatially tuned region in dorsal cortex alongside V1; population
  manifold geometry mirrors maze structure. Coordinated multi-region drift across a
  47-day longitudinal imaging dataset; within-session stability substantially higher
  than cross-session. Relevant for reporting HD tuning stability estimates.
  https://www.biorxiv.org/content/10.64898/2026.05.05.723038v1

- **Prankerd et al. 2026** (May 15): coppaFISH 3D volumetric in situ sequencing +
  CASTalign computational pipeline for post-hoc transcriptomic cell-type assignment
  to in vivo 2P-imaged neurons. Relevant as a future validation or follow-up tool for
  confirming viral cell-type labels in our dataset.
  https://www.biorxiv.org/content/10.64898/2026.05.15.725413v1

**Persistent action items:**

1. Separate moving vs. stationary epochs in all light/dark HD tuning comparisons
   (Jayakumar et al. locomotion-dependent recalibration result).
2. Document imaging position along the RSC anterior-posterior axis; cite Wei et al.
   Nature Communications (doi:10.1038/s41467-026-70762-z).
3. CEBRA orthogonal subspace test: does light/dark context separate from the HD
   coding subspace in population activity?
4. Update Jayakumar et al. citation to Current Biology DOI (S0960-9822(26)00222-8).
5. In the Discussion, frame cell-type HD tuning differences in terms of the
   PV-precision / SST-stability dissociation from Oh et al. 2026.
6. Note coordinated representational drift result when reporting session-wise HD
   tuning stability — within-session estimates are more reliable than cross-session.
7. **New (this scan):** Bin HD tuning quality in 10–15 s windows within the first 30 s
   of each dark epoch to test for a transient tuning-degradation peak at light-off
   onset (Goodwin/Dombeck et al. ACh-mediated representational adaptation prediction).
