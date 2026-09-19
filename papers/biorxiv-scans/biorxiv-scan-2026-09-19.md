# bioRxiv Scan — 19 September 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-09-19. Searches covered: retrosplenial cortex (RSP/RSC), head
direction + two-photon imaging, head direction + darkness/landmarks/drift, spatial
navigation + maze + calcium imaging (rodents), Penk/enkephalin + cortex, neuropil
contamination + two-photon, head-mounted miniature two-photon (freely moving), visual
processing + retrosplenial cortex + spatial navigation, active locomotion + path
integration + head direction.

Target window: preprints posted 2026-09-12 to 2026-09-19.

---

## Highly relevant papers

No new highly relevant preprints identified in the 7-day window.

---

## Moderately relevant papers

No new moderately relevant preprints identified in the 7-day window.

---

## Tangentially relevant / methods papers

No new tangentially relevant or methods preprints identified in the 7-day window.

---

## Searches with no relevant results

All searches returned results, but no preprints posted within the Sept 12–19 window
were identified as relevant to the hm2p project. Specific findings per topic:

**Retrosplenial cortex (RSP/RSC):** Searches returned previously catalogued papers
(Active pursuit gates egocentric RSC coding; divergent spatial codes in RSC; RSC
context-dependent transformation). Most recent RSC preprint found was posted
2026-09-10: "Adolescent alcohol exposure disrupts RSC physiology in adult males" — not
relevant (non-spatial, pharmacological). No new RSC spatial-navigation or HD papers
this week.

**Penk / enkephalin + cortex:** No preprints found in cortex + spatial navigation
context. Results from this week were limited to enteric neurons, striatal circuits,
and MPOA Penk neurons in mating behaviour. Consistent with previous scans: Penk+
RSP neuron function remains uncharacterised in the spatial navigation literature.

**Head direction + two-photon:** No new preprints in this combination this week.

**Head direction + darkness / landmarks / drift:** No new preprints this week.
Previously catalogued papers (Parallax error / cue-anchoring; coordinated HD in ADN
and RSC) reappeared in search results.

**Spatial navigation + maze + calcium imaging:** No new rodent maze calcium imaging
preprints relevant to RSP this week. One NMDA-plasticity hippocampal paper (July
2026) and earlier spatial working memory papers appeared, but none relevant to RSP
or HD cells.

**Head-mounted miniature two-photon:** No new technology preprints this week. Searches
returned previously noted papers (miniBB2p, M-MINI2P, simultaneous 2- and 3-photon
multiplane), all predating the search window.

**Neuropil contamination / two-photon:** The most recent relevant paper found was
"High Axial Resolution Is Necessary for Quantitative Two-Photon Calcium Imaging of
Neuronal Populations" (Yoon, Afifa, Ferrer Imbert, Charles, Ji; posted 2026-07-28),
outside the search window. No new neuropil methods preprints this week.

**Visual processing in RSP / spatial navigation in RSP:** Searches returned
previously catalogued papers only.

---

## Notable recent papers just outside the 7-day window

These did not fall within Sept 12–19 but appeared in searches and may not have been
captured in prior weekly scans.

### Brainwide representation of navigation

van Beest EH, Terry B, Booth G, Harris KD, Carandini M. 2026.
"Brainwide representation of navigation." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.08.29.747979v1
Posted: 2026-09-03.

**Findings:** Recorded from >20,000 neurons across brain regions in mice navigating a
virtual corridor designed to dissociate spatial position from correlated signals (speed,
visual flow, reward proximity). Spatial position was encoded brainwide — not confined to
hippocampal formation. Position encoding was more common in neurons also tuned to visual
landmarks. The hippocampal formation represented position more uniformly than other
regions but less precisely than visual cortex.

**Relevance to hm2p:** Directly relevant to interpreting our population-level position
and HD decoding results. The finding that position encoding is brainwide (not
hippocampus-specific) is consistent with RSP being a genuine spatial encoder, not merely
a relay. The association between visual landmark tuning and spatial position encoding is
particularly important: if Penk+ RSP neurons are more landmark-responsive (our
prediction from prior scans), they should also show stronger position coding — a
testable prediction from this dataset. The Harris / Carandini lab authorship is notable;
this group has strong technical standards and the paper will likely be published in
Nature or Science.

---

### High Axial Resolution Is Necessary for Quantitative Two-Photon Calcium Imaging

Yoon HA, Afifa U, Ferrer Imbert G, Charles AS, Ji N. 2026.
"High Axial Resolution Is Necessary for Quantitative Two-Photon Calcium Imaging of
Neuronal Populations." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.07.28.741086v1
Posted: 2026-07-28.

**Findings:** Imaged cortical neurons across five axial resolutions and five analysis
pipelines (including Suite2p). Lower axial resolution corrupts neuronal tuning curves
and population correlations, with no pipeline fully correcting these artifacts. Robust
somatic/neuropil separation remains an unresolved challenge at lower axial resolutions.

**Relevance to hm2p:** Directly relevant to our Stage 1 (Suite2p) and Stage 4 (FISSA
neuropil subtraction) pipeline design. The key implication: our head-mounted 2P system
has a defined axial resolution, and we should document it and check whether it falls
within the "quantitatively reliable" range shown in this paper. If it does not, FISSA
spatial ICA neuropil subtraction (our optional Stage 4 path) becomes more important, not
less. This paper should be cited in our methods when discussing ROI extraction and
neuropil subtraction choices. Also relevant when comparing our tuning curve estimates to
those from head-fixed 2P systems with higher axial resolution.

---

## Summary

**Quiet week.** No new preprints found in the Sept 12–19 window that are relevant to
the hm2p RSP HD cell-type project.

**The literature gap remains open.** Across all scans to date, no paper has
characterised HD tuning or visual-cue dependence in genetically defined RSP
subpopulations. Penk+ RSP neurons specifically remain uncharacterised in any navigation
context.

**Two papers just outside the window worth noting:**
- van Beest et al. 2026 (brainwide position coding; landmark tuning predicts position
  coding brainwide) provides a useful population-level framework for interpreting our
  decoding results.
- Yoon et al. 2026 (axial resolution + neuropil contamination) is directly relevant to
  validating our pipeline design choices.

Total new papers this week: 0.
Papers noted from outside the window: 2.
