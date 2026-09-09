# bioRxiv Scan — 9 September 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-09-09. Searches covered: retrosplenial cortex (RSP/RSC),
Penk/enkephalin + cortex, head direction cells + two-photon imaging, head direction +
darkness/landmarks/drift, spatial navigation + maze (rodents), visual processing in RSP,
spatial navigation in RSP, head-mounted two-photon microscopy, calcium imaging + maze
navigation, neuropil contamination + two-photon.

**Note on coverage:** Direct access to biorxiv.org is blocked by the session proxy;
searches were conducted via web search engine indexing of bioRxiv. Indexing lags
submission by 3–7 days, so preprints posted September 6–9 are unlikely to appear yet.
Papers from September 2–6 that were already covered in the September 7–8 daily scans
are noted but not re-reported in full here.

---

## Highly relevant papers

*No new preprints from the September 2–9 window were identified as highly relevant to
the core hm2p question (cell-type-specific RSP HD tuning; visual vs idiothetic anchoring
in light/dark conditions).*

---

## Moderately relevant papers

*The one paper from the last 7 days meeting this threshold — van Beest EH, Terry B,
Booth G, Harris KD, Carandini M. 2026. "Brainwide representation of navigation."
bioRxiv. https://www.biorxiv.org/content/10.64898/2026.08.29.747979v1 (posted September 3) —
was reported in full in the 2026-09-07 scan and is not duplicated here. Brief recap:
>20,000 neurons brainwide in virtual-corridor navigation; spatial position encoded
everywhere but more common in visually-landmark-tuned neurons; running speed modulates
most neurons; hippocampal formation more uniform but less precise than visual cortex.*

---

## Tangentially relevant / methods papers

*No new methods preprints on neuropil subtraction, Suite2p/CaImAn, DLC/pose estimation,
CASCADE, or miniature two-photon microscopy were identified in the September 8–9 window.*

---

## Searches with no new results (September 8–9)

All searches returned the same set as the September 8 scan. Specific outcomes:

**Retrosplenial cortex (RSP/RSC):** No new preprints. Repeated hits remain the June 2026
distance-mapping RSC paper (Naud et al.), the May 2026 RSP PV/SST interneurons paper
(Oh et al.), and the February 2026 active-pursuit / egocentric coding paper, all
previously logged.

**Penk/enkephalin + cortex:** No new preprints. The monitoring period (April 2 –
September 9, 161 days, 99 scans) has produced zero preprints characterising Penk+ neurons
in cortex in a spatial or HD context. All Penk/enkephalin hits remain subcortical
(striatum, dorsal raphe, MPOA, enteric neurons). The gap is fully open.

**Head direction cells + two-photon imaging:** No new preprints. Most recent relevant
paper remains Tian et al. 2026 (MEC HD cells, miniature 2P, light deprivation), covered
in the April 2026 scan.

**Head direction + darkness / landmarks / drift:** No new experimental preprints. The
parallax-error paper (Jayakumar et al.) and the area 29e landmark coupling paper (Secer
et al.) are both now in peer-reviewed journals (PMC:13001495 and PMC:12642567).

**Spatial navigation + maze (rodents) + calcium imaging:** No new preprints from the last
24 hours. Van Beest et al. (Sep 3) remains the only recent entry in this category within
the 7-day window.

**Visual processing in RSP:** No new preprints. Yang et al. PNAS 2026 and Wei et al.
Nature Communications 2026 remain the most recent publications on this topic.

**Spatial navigation in RSP:** No new preprints.

**Head-mounted / miniature two-photon microscopy:** No new preprints.

**Calcium imaging + maze navigation:** No new preprints.

**Neuropil contamination + two-photon:** No new preprints. The Wang et al. 2026 paper
on subthreshold GCaMP transients (August 24) remains the most recent tangentially
relevant find for our neuropil subtraction and ROI classifier pipeline.

---

## Summary

**Papers from the last 7 days:** 1 total (van Beest et al., covered in 2026-09-07 scan)

**New papers from September 8–9 specifically:** 0

**Status of the core research gap:** No preprints on cell-type-specific RSP HD tuning,
genetically-defined RSP subpopulation coding, or visual cue dependence in RSP have
appeared in 161 days of monitoring. The hm2p question remains unaddressed in the preprint
literature.

**Publication notes (carry-forward from Sep 8):**

- Jayakumar et al. postsubiculum HD / parallax paper → now published (PMC:13001495).
  Update journal citation in manuscript reference list.
- Secer et al. area 29e landmark coupling paper → now published (PMC:12642567).
  Update journal citation in manuscript reference list.
- M-MINI2P (March 2025 bioRxiv) → now published in Cell Reports Methods.
- miniBB2p (October 2024 bioRxiv) → now published in Nature Communications.

---

## Persistent action items (carried from September 8 scan)

1. Document the RSC anterior–posterior coordinate for every hm2p session FOV. Check
   whether Penk+ and non-Penk virus expression distributions differ along the A–P axis.
   Cite Yang et al. PNAS 2026 and Wei et al. Nature Communications 2026 in manuscript
   Methods. **Priority: high — analysis prerequisite.**

2. Separate moving vs. stationary epochs in all light/dark HD tuning comparisons.
   Cite Jayakumar et al. 2026 (Curr Biol doi:S0960-9822(26)00222-8).
   **Priority: high.**

3. Include running speed as a covariate in all HD tuning models; do not compare HD
   tuning quality across light conditions without controlling for speed differences.
   Cite van Beest et al. 2026 (brainwide navigation paper) for the brainwide
   speed-modulation finding. **Priority: high.**

4. Bin HD tuning quality in 10–15 s windows within the first 30 s of each dark epoch
   to test for a transient tuning-degradation peak at light-off onset.
   Cite Goodwin/Dombeck et al. 2026 (ACh-RSC contextual adaptation).
   **Priority: medium.**

5. CEBRA orthogonal subspace test: does light/dark context separate from the HD coding
   subspace? Does the RSP population manifold mirror the rose maze graph topology?
   Cite Feld/Spiers et al. 2026 and Peters et al. 2026.
   **Priority: medium.**

6. Frame excitatory cell-type HD differences using the PV-precision / SST-stability
   inhibitory dissociation (Oh et al. 2026) in the Discussion.
   **Priority: medium.**

7. Check ROI classifier performance separating soma vs. dendrite ROIs with respect to
   subthreshold GCaMP transients (Wang et al. 2026). Low-amplitude dF/F events in soma
   ROIs may reflect subthreshold dendritic HD input, especially in low-firing non-Penk
   cells. **Priority: medium.**

8. Note coordinated representational drift (Peters et al. 2026) when reporting session-
   wise HD tuning stability: within-session estimates are more reliable than cross-session
   comparisons. **Priority: low.**

9. Watch for RSP follow-up from the Giocomo lab structured navigation platform
   (Fisher et al. bioRxiv July 2026). Neuropixels RSC recordings under this paradigm
   would be a direct competitor study. **Priority: watch.**
