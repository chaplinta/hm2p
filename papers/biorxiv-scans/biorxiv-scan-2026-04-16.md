# bioRxiv Scan — 16 April 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-04-16. Searches covered: retrosplenial cortex (RSC/RSP), Penk/enkephalin
+ cortex, head direction + two-photon calcium imaging, head direction + darkness/landmarks/
drift, spatial navigation + maze (rodents), visual processing in RSP, spatial navigation in
RSP, head-mounted two-photon microscopy, calcium imaging + maze navigation, neuropil
contamination + two-photon, head direction + attractor + visual cue, head direction + cell
type specific, locomotion + head direction.

Scan window: 2026-04-09 through 2026-04-16. Papers already reported in the 2026-04-02,
2026-04-04, 2026-04-05, or 2026-04-06 scans are not repeated.

---

## Highly relevant papers

No highly relevant papers were posted this week. Searches for retrosplenial cortex, head
direction cell-type specificity, calcium imaging of spatial cells in freely moving mice,
and light/dark HD manipulation returned no new preprints in the April 9–16 window that
had not already been covered in previous scans.

---

## Moderately relevant papers

### 1. Locomotion-invariant prefrontal–thalamic goal states organize episode-specific hippocampal maps

Authors not fully specified in search results. 2026.
"Locomotion-invariant prefrontal–thalamic goal states organize spatially aligned
episode-specific hippocampal maps." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.04.09.716651v1

Posted: 9 April 2026.

**Findings:** When mice navigate the same maze under different goal configurations,
hippocampal CA1 segregates navigation episodes by encoding goal state along a population
dimension orthogonal to the spatial coding subspace — place field maps remain spatially
aligned while the goal-state axis separates episodes. A prefrontal–thalamic pathway
(medial prefrontal cortex + nucleus reuniens) supplies persistent goal-state
representations across locomotion and immobility, reinstated when previously experienced
goal configurations recur. Silencing nucleus reuniens selectively abolishes CA1 goal-state
coding (disrupting goal-axis separation and goal-biased pre-navigation spike sequences)
while spatial coding is spared.

**Relevance to hm2p:** The demonstration that hippocampal population activity encodes
behavioural context as a dimension orthogonal to spatial coding is relevant for our
population analysis. Light vs. dark epochs represent a context switch in our paradigm:
if the same principle holds in RSP, we might find that Penk+ or non-Penk population
activity separates light and dark episodes along a dimension orthogonal to the HD coding
subspace. This is directly testable with CEBRA (contrastive population embeddings with
behavioural supervision). The involvement of nucleus reuniens is also relevant because it
is a relay for hippocampal–RSP communication; if nucleus reuniens encodes context, it
may convey contextual state to RSP as well as hippocampus. The prefrontal–thalamic
pathway finding motivates checking whether population-level RSP activity in our data
shows context-separated subspaces (light vs. dark) independently of the HD subspace,
as a secondary analysis after core HD tuning comparisons are complete.

---

## Tangentially relevant / methods papers

### 2. Path integration and spatial updating recruit distinct cognitive-neural mechanisms in humans

Authors not fully specified in search results. 2026.
"Path Integration and Spatial Updating Recruit Distinct Cognitive-Neural Mechanisms in
Humans." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.04.11.717901v1

Posted: 11 April 2026.

**Findings:** Human fMRI study examining two components of self-motion-based navigation:
path integration (integrating self-motion to estimate current position from a starting
point) and spatial updating (updating the remembered positions of other landmarks as the
observer moves). The precuneus and dorsal premotor cortex showed stronger activation
during spatial updating. The precuneus had stronger functional connectivity with thalamus
and frontal cortex during path integration. Results support a dissociation between these
two operations that are often conflated under the umbrella of "idiothetic navigation."

**Relevance to hm2p:** Our dark epochs rely on idiothetic HD maintenance — mice must
path-integrate angular self-motion in the absence of visual cues. This human study
distinguishes path integration (updating current state) from spatial updating (updating
stored landmark associations). In our light-off condition, both processes may be
relevant: path integration maintains the current HD estimate, while spatial updating
would be needed to restore the correct HD orientation when lights come back on. If Penk+
and non-Penk cells differ in how quickly they re-anchor to visual landmarks after
lights-on, this could reflect a difference in spatial updating efficiency rather than
(or in addition to) path integration accuracy during darkness. Human fMRI limits direct
translation, but the conceptual dissociation is worth keeping in mind when designing
the lights-on recovery analysis.

---

## Searches with no relevant results this week

**Retrosplenial cortex (RSC/RSP):** No new preprints posted April 9–16. Searches
returned only earlier papers already covered in previous scans (hypoglycemia/RSC from
March 2026, CortexCAM from February 2026, Laurent et al., Wei et al., etc.).

**Penk/enkephalin + cortex:** No new papers in a spatial navigation or RSP context.
Continues to return striatal (cocaine abstinence) and brainstem (pain/mating) Penk
studies. The functional characterisation of Penk-expressing RSP neurons remains an
unoccupied niche. Four consecutive scans confirm this.

**Head direction + darkness/landmarks/drift:** No new experimental papers. The "active
locomotion" / HD attractor paper (covered April 6) remains the most recent contribution
in this space.

**Calcium imaging + maze navigation:** No new papers combining calcium imaging with
maze navigation in a light-manipulation or HD-focused context.

**Neuropil contamination + two-photon:** No new methods papers this week.

**Head-mounted two-photon microscopy:** No new systems papers this week. The technology
papers covered in the April 2 scan (M-MINI2P, miniBB2p, FHIRM-TPM 3.0, simultaneous
2+3 photon multiplane) remain the most recent contributions.

**Visual processing in RSP:** No new papers. The Wei et al. A-P gradient paper (June
2025, covered April 2) and the RSC→V1 feedback paper (September 2025, covered April 4)
remain the most recent.

---

## Summary

A quiet week for hm2p-relevant topics. Two papers are worth noting:

- The prefrontal–thalamic goal-state paper (paper #1) suggests a testable secondary
  analysis: do RSP population subspaces separate light and dark contexts orthogonally
  from the HD coding dimension? This is a natural extension of the CEBRA analysis
  already planned for Stage 6.

- The path integration / spatial updating dissociation paper (paper #2) motivates
  treating the lights-on recovery period as a distinct analytical window from the
  steady-state dark epoch, since landmark re-anchoring (spatial updating) and
  path integration may operate on different timescales and may differ between cell types.

**Penk+ gap:** Four consecutive weekly scans have found no papers characterising
Penk-expressing neurons in RSP or any cortical region in a spatial/HD context.
The absence is consistent and confirms the open scientific gap that hm2p addresses.

**Papers this week: 2 total** (0 highly relevant, 1 moderately relevant, 1 tangential).
