# bioRxiv Scan — 24 November 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-11-24. Search window: last 18 days (2026-11-07 to 2026-11-24); the
previous scan ran on 2026-11-06, so this scan picks up from there.
Searches covered: retrosplenial cortex (RSC/RSP), Penk/enkephalin + cortex, head direction
+ two-photon imaging, head direction + darkness/landmarks/drift, spatial navigation + maze
(rodents), visual processing in RSP, head-mounted miniature two-photon microscopy, calcium
imaging + maze navigation, neuropil contamination + two-photon.

Note: direct HTTP access to biorxiv.org is not available from the cloud session, so
searches rely on web search indexing, which may lag by 24–48 hours for the most recent
posts.

---

## Highly relevant papers

No new highly relevant preprints found in the scan window.

---

## Moderately relevant papers

### 1. Visual capture of the head-direction signal in a rat model of Fragile X Syndrome

Duszkiewicz AJ, Rastedt AA, Vadher A, Wood ER, Peyrache A, Dudchenko PA. 2026.
"Visual capture of the head-direction signal in a rat model of Fragile X Syndrome."
bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.11.06.764867v1

**Findings:** Electrophysiological recordings in postsubiculum (PoSub) of wild-type (WT)
and Fmr1 knockout (KO) rats during a cue-rotation protocol. In WT rats, the HD signal
only partially followed landmark rotations, consistent with weighted integration of visual
and idiothetic cues; visual landmark control progressively weakened across repeated cue
conflicts. In Fmr1 KO rats, the HD signal showed enhanced visual capture — it tracked
landmark rotations more completely. This was accompanied by reduced angular head velocity
(AHV) tuning in PoSub neurons, suggesting that the KO animals rely more heavily on visual
landmarks and less on self-motion (path integration) signals.

**Relevance to hm2p:** Directly relevant to our light on / light off paradigm. The core
question in our dataset — how much does each RSP population rely on visual anchoring vs.
idiothetic path integration to maintain HD — is exactly what this paper operationalises,
albeit in a different brain region (PoSub), different species (rat), and a different
manipulation (genetic, not sensory). Key implications:
- The WT result (partial but not complete visual capture; weakening across repeated
  conflicts) establishes baseline weighting of visual vs. idiothetic cues in the HD
  system of a healthy animal. Our lights-off manipulation removes visual input entirely —
  a more extreme perturbation than a cue rotation.
- The AHV tuning result is a mechanistic proxy for idiothetic processing strength. We
  should compute AHV tuning in our RSP populations: if Penk⁻CamKII+ cells show stronger
  AHV modulation than Penk+ cells, that would predict greater HD stability in darkness for
  the non-Penk population.
- The progressive weakening of visual control across repeated conflicts is a form of
  prediction-error updating. Our mice experience repeated light transitions — does RSP HD
  tuning degrade progressively within sessions, and does this differ by cell type?
- Not a primary citation for our work (different region, different manipulation), but
  useful background for the introduction when framing visual vs. idiothetic weighting.

---

## Tangentially relevant / methods papers

No new tangentially relevant or methods papers found in the scan window.

---

## Searches with no relevant results

**Retrosplenial cortex (RSC/RSP):** No new RSC-specific preprints found in the Nov 7–24
window on spatial navigation, HD tuning, visual processing, or circuit anatomy. The
November 2026 bioRxiv neuroscience archive contains entries in this window, but none
found by web search indexing are topically relevant to RSP/HD/spatial coding. The most
recent relevant RSC preprints remain those from the 2026-11-05 scan (MINI2P-II,
CA1→RSC inhibitory connectivity) and the 2026-10 scans.

**Penk/enkephalin + cortex:** No preprints found in the scan window. The Penk-expressing
neurons in cortex in a spatial or HD context remain uncharacterised in the literature.
All enkephalin-related preprints returned concerned striatal circuits (cocaine, reward),
dorsal raphe (pain), brainstem, or peripheral tissues.

**Head direction + two-photon calcium imaging:** No new freely-moving 2P HD imaging
preprints in the window.

**Head direction + darkness / visual landmarks / drift:** One new paper found (Duszkiewicz
et al., above). No other preprints in the window on HD drift in darkness or landmark
anchoring in rodents.

**Spatial navigation + maze + calcium imaging:** No new preprints in the window directly
relevant to RSP or HD. Searches returned older work from the August–October 2026 period
already covered in prior scans.

**Visual processing in RSP:** No new preprints in the window.

**Head-mounted miniature two-photon microscopy:** No new technology preprints in the
window. The MINI2P-II paper (2026-10-27, covered 2026-11-05) remains the most recent
advance in this area.

**Neuropil contamination + two-photon:** No new methods papers in the window on neuropil
subtraction approaches.

---

## Summary

**Papers found (Nov 7–24, by relevance):**

| # | Title (abbreviated) | Date posted | Relevance |
|---|---|---|---|
| 1 | Duszkiewicz et al. — Visual HD capture in Fragile X rat | ~2026-11-06 | Moderate |

Total: 1 new paper found. This was a quiet period for topics directly relevant to hm2p.

**Persistent gap:** Genetically-defined RSP subpopulations (Penk+ or any equivalent
cell type) remain unstudied in HD or spatial navigation contexts across all 2026 scans.
No other group has characterised cell-type-specific HD tuning, visual cue dependence, or
population decoding in RSP using intersectional genetic approaches. This remains the
primary gap that hm2p addresses.

**New analysis consideration from Duszkiewicz et al.:** The reduced AHV tuning in animals
with enhanced visual dependence provides a potential within-dataset proxy: computing AHV
tuning separately for Penk+ and Penk⁻CamKII+ cells may predict which population is more
visually anchored without requiring any additional manipulation beyond our existing
light/dark epochs.
