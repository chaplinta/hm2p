# bioRxiv Scan — 7 September 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-09-07. Searches covered: retrosplenial cortex (RSP/RSC), Penk/enkephalin
+ cortex, head direction cells + two-photon imaging, head direction + darkness/landmarks/drift,
spatial navigation + maze (rodents), visual processing in RSP, head-mounted two-photon
microscopy, calcium imaging + maze navigation, neuropil contamination + two-photon.

Note on coverage: direct access to biorxiv.org is restricted in this environment; searches
were conducted via web search engine indexing of bioRxiv. Very recent preprints (last 1–3
days) may not yet be fully indexed and could be missed. Coverage of papers from August 31 –
September 7 is best-effort.

---

## Highly relevant papers

*No papers from the last 7 days were identified as highly relevant to the core hm2p question
(cell-type-specific RSP HD tuning; visual vs idiothetic anchoring in light/dark conditions).*

---

## Moderately relevant papers

### 1. Brainwide representation of navigation

van Beest EH, Terry B, Booth G, Harris KD, Carandini M. 2026.
"Brainwide representation of navigation." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.08.29.747979v1
Posted: 3 September 2026.

**Findings:** Recorded from >20,000 neurons across multiple brain regions in mice navigating
a virtual linear corridor designed to decouple spatial position from co-varying signals
(running speed, reward, visual flow). Spatial position was encoded in every brain region
examined, but was more prevalent in neurons that were also tuned to visual landmarks.
Most neurons brainwide were modulated by running speed, likely reflecting arousal rather
than pure spatial coding. Reward signals were also widespread. The hippocampal formation
represented spatial position more uniformly across neurons than other regions, but less
precisely than visual cortex.

**Relevance to hm2p:** The finding that visual-landmark tuning predicts spatial encoding
brainwide is directly relevant to our light/dark manipulation. If RSP Penk+ neurons are
enriched in visual-landmark-coupled spatial coding, this predicts they would be disproportionately
affected by lights-off. The result that running speed modulates most neurons is a reminder
that speed must be included as a covariate when comparing HD tuning between movement states
(our `bad_behav` exclusion and movement-state classifier are relevant here). The study uses
virtual navigation (head-fixed, passive optic flow), limiting direct comparison to our
freely-moving, 3D-rotating maze paradigm, but the brainwide framing is useful context for
population decoding analyses.

---

## Tangentially relevant / methods papers

*No new methods papers specifically on neuropil subtraction, Suite2p, or DLC/pose
estimation were found in the last 7 days.*

---

## Searches with no relevant results from the last 7 days

**Retrosplenial cortex (RSP/RSC):** No new preprints in the 7-day window beyond the
van Beest et al. brainwide navigation paper above. Several older RSP preprints returned
(from April–June 2026) including the RSP PV/SST interneuron paper (May 2026) and the
RSC goal-relevant frames paper (June 2026), but these were covered in prior scans.

**Penk/enkephalin + cortex:** No relevant preprints. Returns continued to show Penk
neurons in striatum (D2-MSN cocaine abstinence), dorsal raphe (behavioural preference),
and MPOA (consummatory mating). Nothing in RSP or HD-related circuits. The gap in
characterisation of cortical Penk+ neurons in a spatial context remains open.

**Head direction cells + two-photon imaging:** No new preprints in the 7-day window.
The most recent relevant paper (Tian et al. 2026, MEC HD + miniature 2P + light deprivation)
was covered in the April 2026 scan.

**Head direction + darkness / landmarks / drift:** No new preprints from the last 7 days.
The parallax error paper (Jayakumar et al., postsubiculum) was covered in prior scans and
has since been published (PMC:13001495).

**Spatial navigation + maze (rodents) + calcium imaging:** No new preprints from the last
7 days on freely-moving maze navigation with calcium imaging.

**Visual processing in RSP:** No new preprints from the last 7 days.

**Head-mounted two-photon microscopy:** No new preprints from the last 7 days. The M-MINI2P
and miniBB2p papers from 2025 have now been published in Cell Reports Methods and Nature
Communications respectively.

**Neuropil contamination + two-photon:** No new preprints from the last 7 days.

---

## Summary

**Papers from the last 7 days:** 1 (van Beest et al., moderately relevant)

**Notable absence:** No new preprints on RSP cell-type-specific spatial coding or HD tuning
in the past week. The field continues to produce work on brainwide spatial representations
and RSP interneuron circuits (earlier in 2026), but nothing directly targeting genetically-
defined RSP subpopulations in a navigation context. The core gap that hm2p addresses
remains open.

**Trend to note:** The van Beest / Carandini group's brainwide navigation paper is the
latest in a series of large-scale, region-agnostic spatial coding studies (following the
IBL repeated-site work and similar efforts). These papers consistently find spatial signals
everywhere, which raises the bar for claiming RSP is *specially* important — the argument
will need to rest on cell-type specificity, light-dependence, or population-level decoding
accuracy rather than mere presence of HD tuning.
