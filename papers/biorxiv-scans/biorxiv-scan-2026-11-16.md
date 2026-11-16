# bioRxiv Scan — 16 November 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-11-16. Search window: last 7 days (2026-11-10 to 2026-11-16). Papers
from 2026-11-10 through 2026-11-13 are covered in prior scans this week (see
2026-11-10.md through 2026-11-13.md). This scan covers new material from 2026-11-13 to
2026-11-16, plus one backfill (amyloidosis paper, 2026.11.11.766020) that does not
appear in prior weekly scans. Searches covered: retrosplenial cortex (RSC/RSP),
Penk/enkephalin + cortex, head direction + two-photon imaging, head direction +
darkness/landmarks/drift, spatial navigation + maze + calcium imaging, visual processing
in RSP, head-mounted miniature two-photon microscopy, neuropil contamination +
two-photon.

Note: direct HTTP access to biorxiv.org is not available from this cloud session;
searches rely on web search engine indexing, which lags 24–48 hours for the most
recently posted preprints. No papers with DOI prefix 2026.11.14–16 were returned by any
search; content posted in the last 48–72 hours is likely not yet indexed.

---

## Highly relevant papers

No new highly relevant bioRxiv preprints found in the Nov 13–16 window.

---

## Moderately relevant papers

No new moderately relevant preprints found in the Nov 13–16 window.

---

## Tangentially relevant / methods papers

### 1. Amyloidosis and tauopathy progressively disrupt near-critical dynamics in selectively vulnerable circuits

Authors not specified in search results. 2026.
"Amyloidosis and tauopathy progressively disrupt near-critical dynamics in selectively
vulnerable circuits." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.11.11.766020v1

Posted 2026-11-11. Not covered in prior scans.

**Findings:** Single-unit recordings in RSC, anterior cingulate cortex (ACC), dorsal CA1,
and primary visual cortex (V1) in young and aged APP/PS1 mice and wild-type controls.
RSC is identified as one of three selectively vulnerable circuits — regions where
near-critical network dynamics are most strongly disrupted by accumulating amyloid and
tau pathology. The study frames RSC vulnerability through the lens of criticality theory
(deviation from power-law scaling in population activity), not HD tuning or spatial
function.

**Relevance to hm2p:** Very indirect. This paper does not study spatial navigation, HD
cells, or any behaviour relevant to our paradigm. The electrophysiology is in a disease
model (APP/PS1), not a Penk-Cre or CamKII intersectional mouse. The sole connection is
that RSC is used as a recording site, confirming it remains an active target for single-unit
work. Not citable in our manuscript for any direct scientific point; noted here only for
completeness.

---

## Searches with no relevant results

**Retrosplenial cortex (RSC/RSP):** One new RSC preprint found (amyloidosis, above). No
new RSC preprints relevant to spatial navigation, HD tuning, or visual cue processing in
the Nov 13–16 window.

**Penk/enkephalin + cortex:** No new preprints. Consistent with all prior 2026 scans —
no group has characterised Penk+ cortical neurons in a spatial or HD context.

**Head direction + two-photon imaging:** No new preprints in the window.

**Head direction + darkness / visual landmarks / drift:** No new preprints in the window.

**Spatial navigation + maze + calcium imaging:** No new preprints directly relevant to
RSP or HD in the window.

**Head-mounted miniature two-photon microscopy:** No new technology preprints in the
window. The Mini2P protocol paper (Tello Garcia et al., 2026-11-09, covered in
2026-11-12.md) remains the week's methods contribution.

**Neuropil contamination + two-photon:** No new methods papers in the window.

---

## Summary

**New items found this scan (Nov 13–16 window, plus one Nov 11 backfill):** 1.

| # | Title (abbreviated) | Posted | Type | Relevance |
|---|---|---|---|---|
| 1 | Amyloidosis disrupts near-criticality in RSC/ACC/CA1/V1 | 2026-11-11 | bioRxiv | Tangential |

**Weekly summary (Nov 10–16, combining all scans this week):**

| # | Title (abbreviated) | Posted | Scan | Relevance |
|---|---|---|---|---|
| 1 | RSC parallel directional codes — 3D (Cell Reports) | 2026-11-11 | 2026-11-12 | High |
| 2 | Visual capture of HD — Fragile X PoSub | 2026-11-09 | 2026-11-13 | Moderate |
| 3 | Mini2P protocol — freely moving mice | 2026-11-09 | 2026-11-12 | Tangential |
| 4 | V2 L5 IT/ET delta-theta filtering | 2026-11-06 | 2026-11-10 | Tangential |
| 5 | Amyloidosis — near-criticality in RSC | 2026-11-11 | 2026-11-16 | Tangential |

The week's scientifically significant contribution was the Zhang et al. Cell Reports paper
(covered 2026-11-12.md) on RSC parallel directional codes during 3D curved-surface
navigation with 2P calcium imaging. It remains the most directly relevant new publication
this week to our project, particularly its finding that RSC activity supports MEC spatial
coding in darkness. The Duszkiewicz et al. PoSub visual capture paper (covered
2026-11-13.md) is the mechanistically most useful complement, motivating AHV tuning
analysis stratified by cell type in our dataset.

**Persistent gap:** No group has characterised genetically defined RSP subpopulations
(Penk+, CamKII intersectional, or other molecular types) in an HD or spatial navigation
context. This gap is unchanged across all 2026 scans to date.

**Active action item (from Duszkiewicz et al.):** Analyse AHV tuning strength
(mean vector length or tuning curve width) separately for Penk+ and Penk⁻CamKII+ cells.
Weaker AHV tuning in Penk+ would, under the visual-capture framework, predict stronger
visual anchoring in light and larger PD drift in dark epochs — both testable from
existing sync.h5 data.
