# bioRxiv Scan — 14 September 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-09-14. Searches covered: retrosplenial cortex, head direction +
two-photon, maze navigation + calcium imaging, miniature two-photon, Penk/enkephalin +
cortex, retrosplenial + spatial navigation, head direction + darkness/landmarks/drift,
neuropil contamination + two-photon, FISSA/Suite2p/CASCADE methods.

**Note on search coverage:** Direct access to biorxiv.org is blocked in this session;
searches were conducted via web search (WebSearch tool). Papers posted within
approximately the last 3–5 days are typically not yet indexed by web search engines. No
papers from the strict 7-day window (2026-09-07 to 2026-09-14) were returned for any
search topic. Papers reported below are new since the previous scan (2026-04-02) and
were identified through targeted keyword searches.

---

## Highly relevant papers

### 1. Distance mapping and task-enhanced HD anchoring in RSC during goal navigation

Chen Y, Wei X, Tang L, Xu H. 2026.
"Distance Mapping and Variable-Specific Geometry of Goal-Relevant Frames in the
Retrosplenial Cortex." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.06.01.729188v1

**Findings:** RSC neurons in freely moving rats encoded Euclidean distance to a goal,
and this distance representation was selectively biased toward the goal during
navigation. Task engagement selectively enhanced allocentric head-direction
representations anchored to a landmark cue. Egocentric boundary-bearing signals showed
no detectable task-related enhancement — the task-dependent effect was specific to
landmark-anchored HD.

**Relevance to hm2p:** The dissociation between task-enhanced landmark-anchored HD
(allocentric) and unchanged egocentric boundary coding is a direct prediction for our
light/dark analysis. In darkness, landmark-anchored HD should be most disrupted (no
visual landmark), while egocentric boundary signals (maze walls, tactile) should
persist. If Penk+ neurons preferentially express the landmark-anchored HD component,
they should show greater tuning loss in dark epochs than non-Penk cells. The goal-
distance encoding also raises the question of whether our rose maze arms act as goals,
creating a position confound for HD tuning that varies with maze location — worth
checking whether HD tuning strength correlates with distance to arm endpoints.

---

### 2. Coordinated representational drift across RSC and dorsal cortex

Peters R, Hope J, Redish AD, Kodandaramaiah S. 2026.
"Coordinated Representational Drift Across the Mouse Cortex." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.05.05.723038v1

**Findings:** Tracked >110,000 unique L2/3 neurons across RSC, visual, somatosensory,
and motor cortices in freely navigating mice over 47 days using a robotic cranial
exoskeleton with chronic widefield calcium imaging. RSC and visual cortex had the
highest proportions of spatially tuned neurons. Single-neuron tuning drifted
continuously, but population geometry — mirroring maze structure — remained stable.
Drift was coordinated across all four regions (correlated session-to-session deviations)
and persisted after controlling for shared behavioural fluctuations.

**Relevance to hm2p:** Two implications. First, RSC is independently confirmed as the
dorsal cortical area with the highest proportion of spatially tuned neurons (alongside
V1), situating it as the appropriate area to study HD coding in our paradigm. Second,
the coordinated drift result matters for our multi-session data: within-session
transitions between light-on and light-off epochs could reflect brain-state-level
modulation shared across areas rather than RSP-specific HD anchoring loss. We should
check whether apparent HD tuning changes in dark epochs correlate with simultaneously
measured behavioural fluctuations (speed, arousal proxies), as the paper shows shared
behavioural state accounts for some but not all coordinated drift.

---

## Moderately relevant papers

### 3. RSC PV and SST interneurons dissociate spatial precision from representational stability

Authors not fully specified in search results. 2026.
"Retrosplenial PV and SST Interneurons Shape Egocentric Spatial Precision and
Stability." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.05.10.724096v1

**Findings:** Parvalbumin (PV) interneurons in RSC were strongly modulated by
self-motion, displayed bearing-aligned synchrony that preceded SST activation, and were
linked to egocentric coding precision. Somatostatin (SST) interneurons showed weak
self-motion modulation but robust boundary-anchored activity with globally coherent
dynamics, and were linked to representational stability over time.

**Relevance to hm2p:** The PV/SST dissociation maps onto our two key experimental
variables: precision (within-epoch HD tuning sharpness) vs. stability (HD anchoring
across light/dark transitions). In darkness, boundary-anchored SST-mediated stability
should persist (maze walls remain), while PV-mediated precision may degrade (no visual
motion flow for self-motion estimation). If Penk+ and non-Penk populations have
different connectivity with PV vs. SST interneurons, this could explain cell-type-
specific differences in which aspect of HD coding degrades in darkness. This paper
should be cited in the discussion of RSC circuit mechanisms.

---

### 4. Brainwide navigation representation: landmark tuning enriches spatially tuned cells

van Beest EH, Terry BS, Booth CRO, Harris KD, Carandini M. 2026.
"Brainwide Representation of Navigation." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.08.29.747979v1

**Findings:** Recorded >20,000 neurons brainwide in mice navigating a virtual corridor
designed to decorrelate position, visual landmarks, running speed, and reward. Spatial
position was encoded in every region but was enriched in neurons that were also tuned
to visual landmarks. Most neurons brainwide were modulated by running speed, likely
reflecting global arousal. The hippocampal formation encoded position more uniformly
across the environment, but less precisely than visual cortex.

**Relevance to hm2p:** The landmark-tuning enrichment of spatially tuned cells is a
direct prediction for our light/dark paradigm: removing visual landmarks in darkness
should disproportionately deplete the population of HD-tuned cells, not merely degrade
individual cells' tuning strength. This is a testable distinction — we should analyse
both the fraction of HD-tuned cells and the tuning sharpness separately for light vs.
dark epochs. The near-universal running-speed modulation (arousal-linked) is a reminder
that any between-condition comparison must control for speed; our `bad_behav` exclusion
handles stuck-mouse artefacts but not speed differences between light and dark epochs
driven by exploratory behaviour.

---

## Tangentially relevant / methods papers

### 5. Giocomo lab platform: freely moving Neuropixels + real-time DLC + visual control

Fisher TG, Sosa M, Gonzalez A, Cheng X, Giocomo LM. 2026.
"Structured Navigation in a Goal-Directed Task Reveals Flexible Spatial Coding." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.07.27.741044v1

**Findings:** New freely moving platform integrating controlled visual stimulus
projection, real-time DeepLabCut pose estimation, trajectory monitoring, and chronic
Neuropixels recordings. Applied to MEC in rats learning a multistage spatial targeting
task. MEC showed canonical spatial and HD tuning, plus population-level goal-distance
tracking. Spatial and HD coding adapted to goal-directed task demands.

**Relevance to hm2p:** Methods context: real-time DLC integration in freely moving
recording pipelines is now established, consistent with our Stage 2b/3 approach. The
MEC goal-distance encoding is upstream context for Chen et al. (paper 1 above) — if
MEC sends goal-distance signals to RSC, RSC goal-distance coding may reflect
thalamocortical relays of entorhinal signals rather than intrinsic RSC computation.
The state-dependent spatial remapping in MEC is a caveat: MEC inputs to RSC fluctuate
with task demands, so apparent RSC HD changes across light/dark epochs could partly
reflect changing MEC input statistics rather than RSC-intrinsic recalibration.

---

### 6. High axial resolution required for unbiased 2P population imaging; neuropil remains unresolved

Yoon HAY, Afifa U, Ferrer Imbert G, Charles AS, Ji N. 2026.
"High Axial Resolution Is Necessary for Quantitative Two-Photon Calcium Imaging of
Neuronal Populations." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.07.28.741086v1

**Findings:** Systematically varied optical axial resolution (3.6–21.0 μm) in L2/3 V1
neurons across five analysis pipelines and three GCaMP variants (cytosolic, transgenic,
soma-targeted). Reducing axial resolution attenuated ΔF/F₀, corrupted visual
responsiveness and orientation tuning classifications, and biased pairwise correlations.
Soma-targeted sensors reduced but did not eliminate contamination. Conclusion: robust
separation of somatic from neuropil signals remains unsolved across all tested
pipelines.

**Relevance to hm2p:** Directly validates our concern about neuropil contamination in
Stage 4. The finding that no pipeline fully resolves the problem supports the use of
FISSA (spatial ICA neuropil subtraction) as the most principled available approach,
and it should be cited in our Stage 4 documentation when explaining why FISSA is
offered as an alternative to fixed-coefficient subtraction. The orientation tuning
degradation result (low axial resolution corrupts tuning classification) is a cautionary
analogue for our HD tuning measurements: axial resolution variation across sessions or
animals could confound HD tuning comparisons. Worth documenting our optical parameters
in session metadata.

---

### 7. Updated Suite2p methods paper

Pachitariu M, Stringer C, et al. 2026.
"Extracting Large-Scale Neural Activity with Suite2p." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.02.04.703741v1

**Findings:** Updated comprehensive description of all current Suite2p algorithms. GPU-
accelerated non-rigid motion correction substantially outperforms alternatives at >5×
speed. Cell detection outperforms CNMF (CaImAn) and Fiola, finding more cells with
fewer false positives and faster runtime. Includes neuropil correction and spike
deconvolution. Demonstrates recordings of >100,000 neurons from mouse cortex with a
standard commercial microscope.

**Relevance to hm2p:** Suite2p is our default Stage 1 extractor. This is now the
current citation for Suite2p (posted February 4, 2026) and should replace the older
Pachitariu et al. 2017 bioRxiv preprint in pipeline documentation and any future
manuscript. The benchmarking against CaImAn provides citable justification for our
default extractor choice. Note: this paper predates the April 2 scan but may have been
missed by the automated search at that time.

---

## Published note

**Wei, Couto, Kloosterman, Bonin 2025 preprint** (covered in April 2 scan, paper 9)
has now been published peer-reviewed:

Wei M, Couto J, Kloosterman F, Bonin V. 2026. "Anterior and posterior retrosplenial
cortex form distinct visuospatial circuits in the mouse." Nature Communications.
https://www.nature.com/articles/s41467-026-70762-z

Content is unchanged from the preprint summary in the April 2 scan. Update citations
in docs and frontend references from the bioRxiv preprint to the published paper.

---

## Searches with no relevant new results

**Penk/enkephalin + cortex:** No new preprints. Results again confined to striatum
(D2-MSN enkephalin, cocaine abstinence), MPOA (mating behaviour), brainstem, and
enteric neurons. The absence of papers on Penk+ cortical neurons in spatial or HD
contexts continues across all scans. This confirms that our study addresses a genuine
gap in characterising Penk+ RSP function.

**Head direction + darkness / visual landmark drift:** The Brainwide navigation paper
(van Beest et al., above) is the most recent relevant paper (posted September 3),
outside the strict 7-day window. No papers from September 7–14 were identified.

**Head-mounted two-photon microscopy:** No new technology preprints beyond those
already captured in the April 2 scan (M-MINI2P, miniBB2p, FHIRM-TPM 3.0).

**7-day window caveat:** Web search does not reliably index bioRxiv papers within ~3–7
days of posting. The absence of papers from September 7–14 in these results does not
confirm a quiet week — it reflects the indexing lag. Manual checking of the bioRxiv
neuroscience new-submissions page for this date range is recommended if completeness
is required.

---

## Summary

**7 papers identified as new since last scan (2026-04-02):**
- 2 highly relevant (Chen et al. RSC goal distance + HD anchoring; Peters et al.
  coordinated RSC drift)
- 2 moderately relevant (RSC PV/SST interneurons; Carandini brainwide navigation)
- 3 tangential/methods (Giocomo freely-moving platform; Ji 2P axial resolution +
  neuropil; Suite2p update)

**Notable trend:** RSC is attracting sustained attention as a multi-variable spatial
hub encoding goal distance, egocentric precision, and representational stability, in
addition to classical allocentric HD. This supports framing Penk+ vs. non-Penk
differences in terms of which spatial variable each population specialises in — e.g.,
landmark-anchored HD precision (Penk+?) vs. boundary-stabilised HD (non-Penk?) — not
merely whether they have HD tuning.

**Papers to cite in the manuscript (newly added since April scan):**
- Chen et al. 2026 (RSC goal distance + task-enhanced HD anchoring)
- Peters et al. 2026 (coordinated representational drift; RSC highest spatially tuned)
- van Beest et al. 2026 (brainwide navigation; landmark tuning enriches spatial cells)
- Yoon et al. 2026 (neuropil contamination unresolved; axial resolution matters)
- Update Wei et al. reference to published Nature Communications version
