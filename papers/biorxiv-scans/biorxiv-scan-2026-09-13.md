# bioRxiv Scan — 13 September 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-09-13. Searches covered: retrosplenial cortex (RSC/RSP), head direction
+ two-photon imaging, head direction + darkness / landmarks / drift, spatial navigation +
maze + calcium imaging, miniature two-photon in freely moving mice, Penk / enkephalin +
cortex, neuropil contamination + two-photon, RSP visual processing + spatial navigation.
Note: searches targeted the last 7 days; very recently posted preprints may not yet be
fully indexed by search engines. Papers from the period May–September 2026 not covered
in the prior scan (2026-04-02) are included where relevant.

---

## Highly relevant papers

### 1. Brainwide representation of navigation

van Beest EH, Terry B, Booth G, Harris KD, Carandini M. 2026.
"Brainwide representation of navigation." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.08.29.747979v1

**Findings:** Recorded from over 20,000 neurons spanning hippocampus, retrosplenial
cortex, visual cortex, and other areas while mice navigated a virtual corridor designed
to decouple position from correlated signals (running speed, licking, arousal). Spatial
position was encoded in every region sampled and was significantly more prevalent among
neurons tuned to visual landmarks. Hippocampus represented position slightly more
uniformly than other regions but less precisely than visual cortex. Most neurons
brainwide were also modulated by running speed (likely reflecting arousal state) and
many by reward delivery.

**Relevance to hm2p:** Directly contextualises our RSP findings. The key result —
that landmark-tuned neurons are more likely to carry spatial position information —
predicts that RSP neurons encoding visual landmarks (which should be disrupted in our
light-off epochs) would also show degraded positional coding in darkness. The finding
that spatial coding is widespread, not unique to hippocampus, supports the importance
of characterising RSP-specific (and cell-type-specific) contributions. Worth citing in
the introduction when framing why RSP spatial coding is worth studying. Harris and
Carandini's dataset is also a potential benchmark for population decoding accuracy.

Posted: September 3, 2026.

---

### 2. Distance mapping and variable-specific geometry of goal-relevant frames in the retrosplenial cortex

Authors not fully specified in search results. 2026.
"Distance Mapping and Variable-Specific Geometry of Goal-Relevant Frames in the
Retrosplenial Cortex." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.06.01.729188v1

**Findings:** Neuropixels recordings in freely moving rats during goal-directed
navigation. RSC neurons encoded Euclidean distance to the goal, with this representation
selectively biased toward the current goal location. Task engagement selectively enhanced
allocentric HD representations anchored to a visual landmark cue, while egocentric
boundary-bearing signals showed no detectable task-related enhancement. Mixed-selective
RSC population activity exhibited variable-specific geometry: distance-to-goal showed
high local smoothness and decoding performance; egocentric boundary bearing showed
stronger macro-scale separability.

**Relevance to hm2p:** The finding that allocentric HD tuning is enhanced during
goal-directed behaviour, anchored specifically to a visual landmark, is directly relevant
to how we interpret HD tuning strength in our dataset. Our rose maze is not strictly
goal-directed, but mice do show structured movement across the arms. The landmark-anchored
HD enhancement disappearing when the task removes visual feedback (analogous to our
light-off condition) is a testable prediction. The variable-specific geometry result
suggests that HD and positional signals in RSC occupy distinct population subspaces —
relevant for our CEBRA analyses.

Posted: June 3, 2026.

---

### 3. Retrosplenial PV and SST interneurons shape egocentric spatial precision and stability

Authors not fully specified in search results. 2026.
"Retrosplenial PV and SST interneurons shape egocentric spatial precision and stability."
bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.05.10.724096v1

**Findings:** PV interneurons in RSC are strongly modulated by self-motion and exhibit
bearing-aligned synchrony that precedes SST interneuron activation, linking movement
to egocentric coding precision. SST interneurons show weak self-motion modulation but
robust boundary-anchored activity with globally coherent dynamics that stabilise
representations over time. Optogenetic silencing revealed dissociable effects: PV
perturbation degraded egocentric coding precision; SST perturbation disrupted global
population organisation. Behaviourally, PV silencing impaired initial egocentric
orientation while SST silencing preserved initial orientation but impaired its sustained
update.

**Relevance to hm2p:** This is the first demonstration of cell-type-specific interneuron
contributions to spatial coding in RSC. The dissociation between PV (precision/acute)
and SST (stability/sustained) roles maps onto the idiothetic vs visual cue question in
our dataset. Our light-off condition removes the visual cues that likely support
boundary-anchored SST activity, predicting a specific loss of long-term HD stability
in darkness. For our excitatory populations (Penk+ vs non-Penk), these interneuron
subtypes are likely differentially engaged: if Penk+ neurons receive more SST
inhibition, they may depend more on sustained visual context. Worth investigating whether
our calcium traces show correlated modulation patterns consistent with SST gating.

Posted: May 11, 2026.

---

## Moderately relevant papers

### 4. High axial resolution is necessary for quantitative two-photon calcium imaging of neuronal populations

Yoon HYA, Afifa U, Ferrer Imbert G, Charles AS, Ji N. 2026.
"High Axial Resolution Is Necessary for Quantitative Two-Photon Calcium Imaging of
Neuronal Populations." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.07.28.741086v1

**Findings:** Demonstrated that lower axial resolution corrupts neuronal tuning
properties and population correlations in ways that no standard analysis pipeline can
correct. Neuropil contamination (somatic fluorescence mixed with neuropil signal)
remains an unresolved problem. Pipeline choice (Suite2p vs CaImAn vs others) yields
divergent results even at high resolution. Soma-targeted viral constructs (using soma
targeting sequences) mitigate but do not eliminate pipeline failures.

**Relevance to hm2p:** Directly relevant to our calcium processing choices (Stage 4).
The finding that neuropil contamination corrupts tuning properties is critical for HD
tuning analyses — if our neuropil subtraction (fixed coefficient or FISSA) leaves
residual contamination, HD tuning widths and preferred direction estimates will be
affected. The divergent results across pipelines supports our decision to run both
Suite2p and CaImAn as pluggable extractors and compare outputs. The soma-targeting
caveat is relevant if we ever switch viral constructs in future experiments.

Posted: July 30, 2026.

---

### 5. Coordinated representational drift across the mouse cortex

Authors not fully specified in search results. 2026.
"Coordinated Representational Drift Across the Mouse Cortex." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.05.05.723038v1

**Findings:** Used a robotic cranial exoskeleton with widefield calcium imaging at
cellular resolution to chronically track over 110,000 unique layer 2/3 neurons across
retrosplenial, visual, somatosensory, and motor cortex across 47 days in mice navigating
a figure-8 maze. Found coordinated representational drift across all cortical areas:
neurons rotated their population-level coding directions together, with the RSC showing
drift correlated with hippocampal remapping events.

**Relevance to hm2p:** Representational drift in RSC is directly relevant to our
multi-session analyses (we have up to 26 sessions across animals). If RSC HD tuning
drifts even within an environment, this is a confound for cross-session comparisons.
The correlation between RSC drift and hippocampal remapping provides a mechanistic link.
For our light/dark manipulation: if lights-off constitutes a context switch that induces
hippocampal remapping, a coordinated RSC drift could explain PD shifts rather than true
idiothetic integration failure. Worth checking whether PD drift in our dark epochs
resets on lights-on (suggesting context-linked drift) or accumulates monotonically
(suggesting idiothetic drift accumulation).

Posted: May 5, 2026.

---

### 6. Coordinated acetylcholine release and adaptation of neuronal representations in the retrosplenial cortex during contextual uncertainty

Authors not fully specified in search results. 2026.
"Coordinated acetylcholine release and adaptation of neuronal representations in the
retrosplenial cortex during contextual uncertainty." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.05.02.722331v1

**Findings:** RSC acetylcholine release is tightly correlated with movement velocity.
During contextual uncertainty (conflicting spatial cues), RSC neuronal representations
undergo rapid adaptation correlated with local ACh release. ACh modulation selectively
affected cell ensembles encoding the ambiguous cue, while ensembles encoding stable
reference cues were relatively spared.

**Relevance to hm2p:** Our light-off epochs constitute a form of contextual uncertainty
(loss of visual reference). If RSC ACh release tracks movement velocity and gates
representational adaptation, then locomotion state in darkness is a potential confound
for HD tuning stability analyses. More specifically: the ensemble-selective ACh effect
predicts that only RSC neurons encoding visual landmark information (possibly Penk+?)
would show adaptation in darkness, while neurons with stable idiothetic reference
(possibly non-Penk?) would be less affected. We cannot measure ACh with calcium imaging,
but this framing motivates checking whether HD tuning degradation in darkness correlates
with running speed.

Posted: May 2, 2026.

---

## Tangentially relevant / methods papers

### 7. Environmental novelty modulates rapid cortical plasticity during navigation

Authors not fully specified in search results. 2025.
"Environmental Novelty Modulates Rapid Cortical Plasticity During Navigation." bioRxiv.
https://www.biorxiv.org/content/10.1101/2025.10.21.683723v1

Two-photon calcium imaging combined with holographic optogenetic stimulation in mice
navigating virtual reality environments. RSC layer 2/3 neurons showed stimulation-induced
plasticity only in novel environments; RSC layer 5 neurons showed plasticity regardless
of novelty. Position-correlated spatial representations emerged rapidly in both RSC and
V1 in novel but not familiar environments. Relevant because our mice are well-trained in
the rose maze — their RSC representations should be stable and in the "familiar" regime
described here, where layer 2/3 plasticity is saturated.

---

### Searches with no relevant results this week

**Penk / enkephalin + cortex (spatial navigation context):** No new preprints. Searches
returned papers on Penk in MPOA (mating behaviour), striatum (cocaine abstinence), and
enteric nervous system. No Penk+ cortical neuron papers in a spatial or navigation
context. The gap confirmed in prior scans remains open.

**Head direction + darkness / landmarks / drift (last 7 days):** No new preprints in
the strict 7-day window. The prior scan's papers (Jayakumar et al., Tian et al.) remain
the most relevant published work.

**Head-mounted two-photon microscopy (last 7 days):** No new hardware preprints. The
prior scan (M-MINI2P, miniBB2p, simultaneous 2+3-photon) covered the current landscape.

**Miniature endoscope / miniscope + navigation:** Nothing new relevant to our paradigm.

---

## Summary

**Papers with direct implications for hm2p analyses:**

- Van Beest et al. 2026 (Carandini/Harris lab) — spatial position is brainwide but
  denser in landmark-tuned neurons; provides a strong prior for predicting light-off
  effects on RSC position coding.
- The RSC goal-distance paper (June 2026) — landmark-anchored HD enhancement under
  task engagement; relevant for interpreting HD tuning strength in structured movement.
- PV/SST interneuron paper (May 2026) — cell-type-specific contributions to spatial
  precision vs stability in RSC; provides mechanistic context for interpreting
  Penk+ vs non-Penk differences.
- Yoon et al. 2026 — neuropil contamination corrupts tuning, pipeline choice matters;
  validates our decision to use FISSA and to compare Suite2p/CaImAn extractors.

**New considerations for analysis:**

- Check whether PD drift in dark epochs resets on lights-on (context-linked drift,
  as predicted by representational drift paper) vs accumulates monotonically
  (idiothetic integration failure).
- Check whether HD tuning degradation in darkness correlates with running speed (from
  the ACh-movement velocity coupling result).
- The brainwide navigation paper suggests landmark tuning as the key predictor of
  spatial position coding — compute this for our RSP neurons.

**Papers to cite in the manuscript (additions to prior list):**

- Van Beest et al. 2026 (brainwide navigation, Harris/Carandini) — for framing RSP
  spatial coding in a brainwide context.
- Yoon et al. 2026 (axial resolution and neuropil) — for methods section, neuropil
  subtraction justification.

**Total new papers included:** 7 (6 from May–September 2026, 1 from October 2025 not
previously catalogued). The gap in Penk+ cortical neuron literature persists.
