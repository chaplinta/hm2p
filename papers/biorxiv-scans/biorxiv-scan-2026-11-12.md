# bioRxiv Scan — 12 November 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-11-12. Search window: last 7 days (2026-11-05 to 2026-11-12). Papers
from 2026-11-01 through 2026-11-10 are covered in prior scans this week (see
2026-11-05.md through 2026-11-10.md); this scan focuses on new material from
2026-11-10 to 2026-11-12, with one backfill from Nov 9 missed by prior scans. Searches
covered: retrosplenial cortex (RSC/RSP), Penk/enkephalin + cortex, head direction +
two-photon imaging, head direction + darkness/landmarks/drift, spatial navigation +
maze + calcium imaging, visual processing in RSP, head-mounted miniature two-photon
microscopy, neuropil contamination + two-photon.

Note: direct HTTP access to biorxiv.org is not available from this cloud session;
searches rely on web search engine indexing, which lags 24–48 hours for the most
recently posted preprints. The main finding this scan is a peer-reviewed Cell Reports
publication (not a bioRxiv preprint), flagged separately at the top.

---

## ★ New peer-reviewed publication (not bioRxiv)

### RSC exhibits parallel directional codes and contributes to MEC spatial coding in 3D curved-surface navigation

Zhang X, Yuan X, et al. 2026.
"Retrosplenial cortex exhibits parallel directional codes and contributes to medial
entorhinal spatial coding in 3D curved-surface navigation." Cell Reports.
https://www.cell.com/cell-reports/fulltext/S2211-1247(26)01375-6
Analysis code: https://github.com/zx990416/RSC_MEC_curved_surface_navigation

Published approximately 2026-11-11 (confirmed via Cell Reports RSS metadata "1 day ago"
at time of scan).

**Findings:** Two-photon calcium imaging in freely moving mice navigating a cylindrical
arena (3D curved surface). RSC contains two distinct directional representations
operating in parallel: one encoding azimuthal head direction (classical HD tuning) and
one encoding direction of motion along the curved surface relative to 3D geometry.
Chemogenetic silencing of RSC impaired stable positional coding in simultaneously
recorded MEC neurons, with the effect strongest during limited-visual-cue conditions.
The RSC→MEC influence was geometry-dependent and not observed for flat-surface
navigation.

**Relevance to hm2p:** Directly relevant on two levels. First, the finding of parallel
directional codes in RSC (classical HD + geometry-referenced) predicts that our
rose-maze data could contain a mixture of cell types — standard HD cells and cells
tracking maze-arm direction geometry — that we have not yet distinguished. Second, the
visual-support dependency of the RSC→MEC coupling mirrors our light/dark manipulation:
the effect was strongest when visual cues were limited. This paper should be read
alongside Laurent et al. (2024/2025) on multidirectional cells in RSC multi-room
environments. We should check whether Penk+ and non-Penk populations are enriched in
one of the two directional codes. The analysis code is available in MATLAB and covers
full-session calcium + behaviour files.

---

## Highly relevant papers

No new highly relevant bioRxiv preprints found in the Nov 10–12 window.

---

## Moderately relevant papers

No new moderately relevant bioRxiv preprints found in the Nov 10–12 window.

---

## Tangentially relevant / methods papers

### 1. Protocol for cortical and deep-brain calcium imaging in freely moving mice with the Mini2P system

Tello Garcia IA, Alhassen S, Portera-Cailliau C, Zeiger WA. 2026.
"A protocol for cortical and deep-brain calcium imaging in freely moving mice with the
Mini2P system." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.11.09.764980v1

Posted 2026-11-09. Missed by prior scans due to search-index lag; included here as
backfill.

**Findings:** Step-by-step protocol for implementing miniaturised two-photon calcium
imaging in freely moving mice using the Mini2P system (Zong et al., Cell 2022).
Covers surgical implantation, optical alignment, motion correction, ROI extraction,
and integration with behavioural tracking. Includes decision points, validation
criteria, and troubleshooting at each stage. Protocol is stated to be adaptable
across brain regions and behavioural paradigms.

**Relevance to hm2p:** Methodologically parallel to our setup. We use a head-mounted
two-photon system; this protocol formalises best practices for the same class of
experiment. Useful for troubleshooting, as a methods citation, and for assessing
whether our preprocessing pipeline matches the community standard. Note that our
imaging system predates Mini2P v2; some steps (axial scanning, objective choice) will
differ, but the motion-correction and ROI-extraction guidance is general.

---

## Searches with no relevant results

**Retrosplenial cortex (RSC/RSP):** No new RSC bioRxiv preprints found in the Nov
10–12 window. The Cell Reports paper (above) is the week's key RSC contribution.

**Penk/enkephalin + cortex:** No preprints in any window this week. All 2026
Penk/enkephalin preprints concern striatum, dorsal raphe, amygdala, and peripheral
circuits. The absence of RSP Penk+ characterisation literature persists unchanged
across all 2026 scans — this remains an open gap.

**Head direction + two-photon imaging:** No new preprints in the Nov 10–12 window.

**Head direction + darkness / visual landmarks / drift:** No new preprints in the
Nov 10–12 window.

**Spatial navigation + maze + calcium imaging:** No new preprints directly relevant
to RSP or HD in the Nov 10–12 window.

**Head-mounted miniature two-photon microscopy:** No new preprints beyond the Mini2P
protocol backfill (Nov 9) covered above.

**Neuropil contamination + two-photon:** No new methods papers in the Nov 10–12 window.

---

## Summary

**New items found this scan (Nov 10–12, plus one Nov 9 backfill):** 2.

| # | Title (abbreviated) | Posted/Published | Type | Relevance |
|---|---|---|---|---|
| 1 | RSC parallel directional codes — 3D curved-surface | 2026-11-11 | Cell Reports | **High** |
| 2 | Mini2P protocol — freely moving mice | 2026-11-09 | bioRxiv | Tangential |

**Weekly summary (Nov 5–12, combining all scans):**

| # | Title (abbreviated) | Posted | Scan | Relevance |
|---|---|---|---|---|
| 1 | RSC parallel directional codes — 3D (Cell Reports) | 2026-11-11 | 2026-11-12 | High |
| 2 | Mini2P protocol freely moving | 2026-11-09 | 2026-11-12 | Tangential |
| 3 | V2 L5 IT/ET delta-theta filtering | 2026-11-06 | 2026-11-10 | Tangential |

The week was quiet on bioRxiv for core hm2p topics. The Cell Reports publication is
the primary output: it is the first paper to describe parallel RSC directional codes
in a geometry-dependent context, and the visual-support dependency of the RSC→MEC
influence directly maps onto our light/dark paradigm.

**Persistent gap:** No group has published a preprint or paper characterising
genetically-defined RSP subpopulations (Penk+ or any other cell type) in an HD or
navigation context. The Zhang et al. Cell Reports paper uses bulk calcium imaging
without cell-type identity — it does not distinguish Penk+ from non-Penk cells. Our
project remains the only study positioned to address this question.

**Action item from Zhang et al.:** Check whether the two RSC directional code types
(azimuthal HD vs geometry-referenced direction) are differentially distributed across
Penk+ and Penk⁻CamKII+ populations. The rose maze geometry (radial arms from a
central hub) provides a natural test, analogous to the cylindrical arena in this paper.
