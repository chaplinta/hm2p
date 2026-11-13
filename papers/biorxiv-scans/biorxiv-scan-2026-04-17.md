# bioRxiv Scan — 17 April 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-04-17. Scan window: April 15–17, 2026 (two days since the April 15
scan), plus two papers from April 10–11 that were missed in the April 15 scan due to
known indexing delays (that scan noted 403 fetch errors in its final 48–72 hours).
Searches covered: retrosplenial cortex, Penk/enkephalin + cortex, head direction +
two-photon, head direction + darkness/landmarks/drift, spatial navigation + maze
(rodents), visual processing in RSP, spatial navigation in RSP, head-mounted two-photon
microscopy, calcium imaging + maze navigation, neuropil contamination + two-photon,
path integration + visual cues + idiothetic.

Papers already listed in prior scans (2026-04-02, 2026-04-04, 2026-04-05, 2026-04-06,
2026-04-15) are not repeated here. Note: direct bioRxiv page fetches continue to return
403 errors; searches relied on web-indexed results.

---

## Highly relevant papers

No papers in this category were found in the April 15–17 window.

---

## Moderately relevant papers

No papers in this category were found in the April 15–17 window.

---

## Tangentially relevant / methods papers

### 1. Path integration and spatial updating recruit distinct cognitive-neural mechanisms

Authors not fully specified in search results. 2026.
"Path Integration and Spatial Updating Recruit Distinct Cognitive-Neural Mechanisms
in Humans." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.04.11.717901v1

Posted: April 11, 2026. (Missed in the April 15 scan due to indexing delay.)

**Findings:** Human fMRI study dissociating path integration from spatial updating.
Participants showed faster responses and distinct eye-fixation patterns during spatial
updating than during path integration, challenging the assumption that path integration
is the substrate for spatial updating. Neuroimaging showed that the precuneus and
dorsal premotor cortex were more activated during spatial updating, but the precuneus
had stronger functional connectivity with the thalamus and frontal cortex during path
integration. The authors conclude that spatial updating and path integration are
dissociable navigation processes supported by distinct behavioural and neural
mechanisms.

**Relevance to hm2p:** The distinction between path integration (internal, idiothetic)
and spatial updating (incorporating external cues) maps onto the cognitive question our
light-off condition tests. When the room lights go out, our mice lose the ability to
perform visual spatial updating and must rely on path integration alone; any HD drift
we observe in darkness reflects the limits of path integration. This paper provides
conceptual framing for that interpretation, even though it uses human fMRI rather than
mouse calcium imaging. The precuneus/thalamic connectivity result echoes the ATN–RSC
axis in rodents. Not directly actionable for our analyses but useful for the
introduction and discussion.

---

### 2. Photon-Resolved Excitation-Denoised (PRED) three-photon imaging for neural activity detection in behaving mice

Authors not fully specified in search results. 2026.
"Photon-Resolved Excitation-Denoised (PRED) Three-Photon Imaging Improves Detection
of Neuronal Activity in Awake and Behaving Mice." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.04.10.717694v1

Posted: April 10, 2026. (Missed in the April 15 scan due to indexing delay.)

**Findings:** Three-photon microscopy provides optical access to neurons 500–1500 µm
below the brain surface but imaging neuronal activity during free behaviour has
remained impractical due to signal-to-noise constraints. The PRED approach applies
photon-resolved denoising tuned to the excitation statistics of three-photon
fluorescence, achieving reliable detection of calcium transients from hippocampal
dentate gyrus neurons in behaving mice at 20–30 Hz, and identifying spatially tuned
cells at depths inaccessible to conventional two-photon microscopy.

**Relevance to hm2p:** Methods context only. Our setup is a head-mounted two-photon
system (single cortical plane, ~150–250 µm depth), so three-photon depth capability
is not directly applicable. The paper is relevant for the discussion of technical
limitations and future directions: deeper RSP layers (L5/6) and hippocampal subfields
directly below RSP are currently inaccessible to us; three-photon systems would allow
simultaneous imaging of RSP and CA1/subiculum in the same animal, which would be
informative for circuit-level HD studies.

---

## Notable publication (not a new preprint)

**Wei YT, Couto J, Kloosterman F, Bonin V. 2026. "Anterior and posterior retrosplenial
cortex form distinct visuospatial circuits in the mouse." Nature Communications.**
https://www.nature.com/articles/s41467-026-70762-z

The bioRxiv preprint of this paper (June 2025, 10.1101/2025.06.24.661247v1) was
included in the April 2 scan. It has now been formally published in Nature Communications.
The published version describes anterior RSC neurons with sharper position tuning and
preference for fast, low–spatial-frequency visual motion, and posterior RSC neurons with
broader tuning and preference for slow, high–spatial-frequency patterns. The A-P gradient
in RSC visual processing properties is relevant to interpreting our single-plane imaging
data: the position of our FOV along the A-P axis should be reported in the methods and
acknowledged as a potential confound if Penk+ cells are not uniformly distributed along
this axis. Now that this is published, it should be cited rather than the preprint.

---

## Searches with no new relevant results

**Retrosplenial cortex (spatial navigation / HD):** No new preprints in the April 15–17
window. Searches continued to return papers from previous scans.

**Penk/enkephalin + cortex:** No new preprints. This is now the fifth consecutive scan
with no Penk+ cortical navigation paper. The gap is confirmed.

**Head direction + two-photon imaging:** No new papers in this window.

**Head direction + darkness / landmarks / drift:** No new experimental papers.

**Spatial navigation + maze (rodents, calcium imaging):** No new papers.

**Head-mounted / miniature two-photon microscopy:** No new technology papers.

**Neuropil contamination + two-photon:** No new methods papers.

---

## Summary of implications for hm2p

**This was again a quiet window.** No papers directly relevant to RSP HD cell-type
comparisons, light/dark cue dependence, or calcium imaging in freely moving mice
appeared in the April 15–17 period. The two missed papers from April 10–11 are
tangentially relevant at best.

**One actionable update:** The Wei et al. 2026 paper is now published in Nature
Communications. Update any citation of the preprint in notes or draft text to the
journal version.

**Cumulative action items remain unchanged** from the April 15 scan:

1. Separate moving vs stationary epochs in all light/dark HD tuning comparisons.
2. Document imaging position along the RSC anterior-posterior axis; cite Wei et al.
   2026 (Nature Comms) when addressing A-P gradient effects.
3. Consider maze-graph (NaviGraph-style) analysis for rose maze arm-specific activity.
4. Note that CASCADE may need re-training for GCaMP8 data if applicable (Stage 4).

**Total new papers this scan: 2** (both tangentially relevant). Cumulative across all
five scans: approximately 22 papers of varying relevance. The hm2p gap (genetically
defined RSP HD cell populations, light/dark dependence, two-photon in freely moving
mice) remains unoccupied in the literature.
