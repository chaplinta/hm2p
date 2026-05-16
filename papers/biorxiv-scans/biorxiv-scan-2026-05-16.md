# bioRxiv Scan — 16 May 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-05-16. Searches covered: retrosplenial cortex (RSC/RSP), head direction
+ two-photon imaging, head direction + darkness/landmarks/drift, spatial navigation + maze
+ calcium imaging (freely moving mouse), Penk/enkephalin + cortex, neuropil contamination
+ two-photon correction, miniature head-mounted two-photon microscopy, visual processing in
RSP, genetically defined neuron populations + spatial coding + cortex.

**Note:** The 2026-05-14 scan already covered the week of May 7–14 and captured the
two highest-priority papers (Oh et al. 2026 RSC PV/SST interneurons; Ralbovszki et al.
2026 entorhinal atlas). Those papers are not repeated here. This scan focuses on anything
posted May 14–16 and highlights one overlooked paper from May 5 not captured previously.

---

## Highly relevant papers

*No new highly relevant papers posted in the May 14–16 window. The Oh et al. RSC PV/SST
interneurons paper (2026-05-10), the most directly relevant paper of the week, was
reported in the 2026-05-14 scan.*

---

## Moderately relevant papers

### 1. Coordinated representational drift across the mouse cortex

Authors not specified in search results. 2026.
"Coordinated Representational Drift Across the Mouse Cortex." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.05.05.723038v1

**Findings:** Used a robotic cranial exoskeleton for longitudinal widefield cellular-resolution
calcium imaging to track over 110,000 unique layer 2/3 neurons across retrosplenial (RSC),
visual (V1), somatosensory (S1), and motor (M1) cortices in mice navigating a figure-8 maze
over 47 days. Single-neuron spatial tuning followed a posterior-to-anterior gradient:
RSC and V1 contained the highest proportions of spatially tuned neurons. Population activity
formed a low-dimensional manifold whose geometry mirrored the maze structure. Despite regional
differences in spatial tuning prevalence, all four regions decorrelated with similar exponential
timescales. Session-by-session deviations from the decay trajectory were correlated across all
pairwise region combinations, indicating coordinated rather than independent drift across dorsal
cortex. Drift was consistent with an orthogonal transformation of the population code that
preserved geometric relationships between spatial representations across sessions.

**Relevance to hm2p:** Three direct implications. (1) RSC is confirmed as the cortical region
with the highest proportion of spatially tuned neurons alongside V1 — this strengthens the
rationale for imaging RSC. (2) The low-dimensional maze-mirroring manifold is relevant to
our CEBRA population-embedding analysis: we should check whether the geometry of our RSP
population code tracks the rose-maze structure similarly, and whether Penk+ vs Penk⁻CamKII+
populations contribute differentially to this manifold. (3) Coordinated multi-region drift
raises an important caveat for within-session stability analyses: single-session tuning
properties are more stable than cross-session ones, which is expected given our experiment
design (multiple sessions per animal). We should report session-wise HD tuning stability and
note this as a known limitation.

---

## Tangentially relevant / methods papers

### 2. Hippocampal brain-machine interface-based navigation reveals CA1 representations of intended actions

Micou C, Ho H, O'Leary T, Krupic J. 2026.
"Hippocampal brain-machine interface-based navigation reveals CA1 representations of
intended actions." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.05.11.724143v1

**Findings:** Used a brain-machine interface (BMI) in which real-time CA1 population activity
directly drove navigation in a virtual environment. In BMI-controlled conditions, CA1 place
codes rapidly reconfigured to disregard locomotion-related inputs. Comparing BMI navigation,
locomotion navigation, and passive playback identified CA1 responses specific to conditions
where animals causally influenced their travel, suggesting that agency is represented by a
distinct place cell code.

**Relevance to hm2p:** Not directly relevant to RSP HD analysis. However, the finding that
hippocampal spatial codes are modulated by volitional agency (not just sensory input) is
interesting background for interpreting why our mice show variable HD tuning across sessions
and within light/dark epochs — arousal, motivation, and agency all modulate the spatial coding
network. Noted for background reading; no direct citation planned.

---

## Searches with no new results this week (May 14–16)

**Retrosplenial cortex (general):** No new RSC/RSP preprints posted in the May 14–16 window
on spatial coding, navigation, or cell-type-specific function.

**Head direction + darkness / visual landmarks / drift:** No new preprints. The landmark-drift
and cue-conflict literature remains as summarised in the 2026-04-02 and 2026-05-14 scans.

**Head direction + two-photon imaging:** No new preprints combining HD analysis with 2P calcium
imaging in freely moving animals.

**Penk/enkephalin + cortex:** No new relevant preprints. As in all prior scans, Penk-cortex
literature is dominated by striatal and brainstem contexts; the RSP Penk+ gap persists.

**Neuropil contamination + two-photon:** No new correction methods or benchmarking papers.

**Miniature head-mounted two-photon:** No new scope technology preprints.

**Spatial navigation + maze + calcium imaging (freely moving mouse):** No new papers matching
all criteria.

---

## Summary

**Total new relevant preprints this scan:** 2 (0 highly relevant, 1 moderately relevant,
1 tangential).

The period May 14–16 was quiet for RSP/HD neuroscience. The coordinated drift paper
(May 5, missed in the May 14 scan) is the substantive addition: it provides the most
comprehensive multi-region longitudinal characterisation of RSC spatial coding to date and
has direct implications for how we report tuning stability and interpret population geometry
in our dataset. The Hippocampal BMI paper is background context only.

The consistent absence of any preprints characterising genetically-defined RSP excitatory
subpopulations across all scans this month reaffirms that our study addresses an open gap.
