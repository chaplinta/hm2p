# bioRxiv Scan — 4 April 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk-CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-04-04. Searches covered: retrosplenial cortex, Penk/enkephalin +
cortex, head direction + two-photon, head direction + darkness/landmarks/drift, spatial
navigation + maze (rodents), visual processing in RSP, spatial navigation in RSP,
head-mounted two-photon microscopy, calcium imaging + maze navigation, neuropil
contamination + two-photon.

Papers already listed in the 2026-04-02 scan are not repeated here.

---

## Highly relevant papers

### 1. Cell-type-specific cholinergic control of granular RSC with implications for angular velocity coding

Jedrasiak-Cape I, Rybicki-Kler C, Brooks JM, et al. 2024.
"Cell-type-specific cholinergic control of granular retrosplenial cortex with
implications for angular velocity coding across brain states." bioRxiv.
https://www.biorxiv.org/content/10.1101/2024.06.04.597341v1

**Findings:** Identified a distinct RSG cell type (low-rheobase, LR neurons) with a
unique cholinergic receptor expression profile. Unlike other RSG excitatory subtypes, LR
neurons do not fire persistently in response to cholinergic agonists. Computational
modelling showed that this lack of persistence allows LR neurons to rapidly compute
angular head velocity (AHV), independent of cholinergic state changes during navigation.
Other excitatory subtypes showed persistent firing under cholinergic drive, suggesting
distinct functional roles across RSG cell types.

**Relevance to hm2p:** This is one of the closest existing studies to our project in
concept: cell-type-specific functional differences in RSP/RSG for navigation-relevant
computations (AHV in their case, HD in ours). The finding that genetically distinct RSG
excitatory subtypes have different cholinergic modulation profiles and different
computational roles provides a direct precedent for our hypothesis that Penk+ and
Penk-CamKII+ populations differ in HD coding properties. We should cite this paper when
motivating cell-type-specific analyses. Key question: do the LR neurons overlap with
Penk+ or Penk-CamKII+ populations? The paper does not use Penk-Cre, so the
correspondence is unknown, but it establishes the principle that RSP excitatory subtypes
are not functionally homogeneous. Not in the April 2 scan — posted June 2024 but missed
because search terms did not surface it.

---

### 2. Cholinergic disruption of state-dependent RSP layer 1 activity causes temporal associative memory deficit under stress

Tanimura A, Login H, Radulovic J, et al. 2025.
"Cholinergic disruption of state-dependent retrosplenial layer 1 activity causes
temporal associative memory deficit under stress." bioRxiv.
https://www.biorxiv.org/content/10.1101/2025.05.27.656297v1

**Findings:** Layer 1 inhibitory neurons in ventral RSP generate activity patterns
correlated with immobility and contribute specifically to temporal associative memory
formation. Stress-induced cholinergic modulation through muscarinic M1 receptors
selectively impairs temporal (but not contextual) associative memory by inhibiting these
L1 neurons and disrupting their local connectivity and afferent responsiveness.

**Relevance to hm2p:** Demonstrates that RSP contains functionally distinct
subpopulations (here, L1 interneurons) with selective roles in specific memory processes.
The immobility-correlated activity pattern is relevant: our mice alternate between active
exploration and pauses in the maze, and L1 circuit modulation could differentially affect
Penk+ vs non-Penk activity during these states. Not directly about HD coding, but
reinforces that RSP cell-type specificity matters for circuit computation. The stress
manipulation is not relevant to our paradigm, but the cholinergic angle connects to the
Jedrasiak-Cape et al. paper above.

---

## Moderately relevant papers

### 3. RSP vulnerability links severe hypoglycemia to cognitive impairment through neuron-microglia crosstalk

Joo JY, Lee S, et al. 2026.
"Retrosplenial cortex vulnerability links severe hypoglycemia to cognitive impairment
through neuron-microglia crosstalk." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.03.27.714654v1

**Findings:** Identified RSP as particularly vulnerable to hypoglycemia-induced neuronal
damage in mouse models. The injury is driven by a feedforward interaction between
neuron-specific Drp1-dependent mitochondrial fission and microglial IL-1 signalling.
Targeting either pathway rescued neuronal damage and reversed cognitive impairment.

**Relevance to hm2p:** Not directly relevant to HD coding, but noteworthy as a new RSP
paper posted March 27, 2026 (within the scan window). The finding that RSP is
selectively vulnerable to metabolic stress could be relevant if discussing RSP's
metabolic demands during active navigation, though this is speculative. More useful as
general RSP literature awareness. No cell-type specificity addressed.

---

### 4. Differential modulation of aversive signalling by expectation across the cingulate cortex

Rogers SA, Oswell CS, Ejoh LL, Corder G. 2026.
"Differential modulation of aversive signaling by expectation across the cingulate
cortex." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.01.05.697709v1

**Findings:** Used longitudinal calcium imaging to show that both anterior cingulate
cortex (ACC) and RSP contain stable shock-responsive ensembles. However, only RSP
flexibly remodelled its activity when shocks were predicted by cues. ACC encoded ongoing
nociception and immediate defensive actions; RSP transformed aversive signals into
temporally structured representations supporting learning and memory.

**Relevance to hm2p:** Demonstrates that RSP performs context-dependent flexible
remodelling of neural representations — not just in spatial tasks but also in aversive
conditioning. This is consistent with RSP as a general contextual integration hub.
Our light/dark alternation is a contextual manipulation; the finding that RSP flexibly
remodels representations across contexts supports the expectation that RSP activity
should change meaningfully between light and dark epochs. Uses calcium imaging in RSP,
which is methodologically close to our approach. Not in the April 2 scan.

---

### 5. Ultraslow entorhinal oscillations shape spatial memory through grid cell drifting

Sarramone L, Presso M, Fernandez-Leon JA. 2026.
"Ultraslow entorhinal oscillations shape spatial memory through grid cell drifting."
bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.03.13.711323v1

**Findings:** Grid cells in MEC of head-fixed mice exhibit ultraslow oscillations
(<0.01 Hz) during walking on a 1D running wheel in darkness. These oscillations produce
systematic grid cell drifting. Computational modelling suggests ultraslow oscillations
link spatiotemporal memories acquired during navigation via synaptic projections from
MEC to hippocampus.

**Relevance to hm2p:** The darkness condition is methodologically relevant — grid cell
drifting in darkness parallels HD cell drifting in our dark epochs. The ultraslow
timescale (<0.01 Hz, period >100 s) is interesting because our dark epochs are 60 s,
which is within one cycle of these oscillations. If similar ultraslow dynamics exist in
RSP HD cells, they could contribute to systematic PD drift patterns in darkness. However,
our calcium imaging at 9.6 Hz cannot resolve these oscillations directly in the neural
signal — we would need to look for drift rate modulation at this timescale. The
head-fixed preparation limits comparison with our freely moving paradigm. Computational
model, not experimental recording in RSP.

---

### 6. Toroidal topology of grid-cell activity precedes spatial navigation during development

Authors not fully specified. 2026.
"Toroidal topology of grid-cell activity precedes spatial navigation during
development." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.03.10.710908v1

**Findings:** Toroidal manifolds emerge in MEC subnetworks as early as postnatal day 10,
before eye/ear opening and before spatial exploration. Ring-like topology of head
direction representations and toroidal topology of grid cells are expressed before
spatial experience and only later become anchored to the external environment through
navigational behaviour. Ring-like HD manifolds were detectable by P9.

**Relevance to hm2p:** Establishes that HD circuit topology is intrinsic and not
dependent on visual experience, but visual anchoring develops later. This is consistent
with the prediction that removing visual cues in our paradigm should not destroy HD
tuning (the ring attractor persists) but should disrupt visual anchoring (drift from
landmarks). Provides developmental evidence for the distinction between path-integration
maintenance and visual anchoring that we test in adult animals. Not directly about RSP,
but relevant background for the HD circuit framework.

---

## Tangentially relevant / methods papers

### 7. Computational modelling of HD cells in three-dimensional space

Authors not fully specified. 2026.
"Computational modeling of head direction cells in three-dimensional space: directional
encoding and visual cue manipulation." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.02.02.703434v1

**Findings:** Developed a toroidal continuous attractor network that jointly encodes
horizontal azimuth and vertical pitch of head direction. The model reproduces
experimentally recorded HD tuning curves and simulates visual cue rotation effects in
3D space. Focused on bat HD system.

**Relevance to hm2p:** Tangentially relevant — our mice navigate in 2D (flat maze), so
the 3D extension is not directly applicable. However, the visual cue rotation modelling
could inform interpretation of what happens when visual cues are removed entirely (our
dark condition). The continuous attractor framework is the standard model for HD cells.

---

### 8. Optogenetic silencing of hippocampal inputs to RSP disrupts spatial working memory (v3 update)

Pinto-Correia B, Caldeira-Bernardo P, Remondes M. 2024/2026.
"Optogenetic silencing hippocampal inputs to the retrosplenial cortex causes a prolonged
disruption of spatial working memory." bioRxiv.
https://www.biorxiv.org/content/10.1101/2024.01.26.577365v3

**Findings:** eArch3.0-mediated silencing of hippocampal terminals in RSP during a
delayed non-match-to-place task impaired memory retrieval and hastened (less accurate)
decision-making, with effects lasting up to 3 subsequent trials. Demonstrates that
hippocampal input to RSP carries contextual information necessary for spatial working
memory.

**Relevance to hm2p:** The hippocampus-to-RSP projection is a major input pathway. This
paper shows RSP depends on hippocampal input for spatial memory, which is relevant
context for understanding what RSP computes. However, the paper addresses working memory
in a T-maze, not HD coding in a rose maze. Title changed between v2 (in April 2 scan
as a reference link) and v3, suggesting revision. Included here as a flagged update.

---

### 9. The functional organisation of retrosplenial feedback to V1

Timplalexi M, Mateos-Aparicio P, Connelly WM, Ranson A. 2025.
"The functional organisation of retrosplenial feedback to V1." bioRxiv.
https://www.biorxiv.org/content/10.1101/2025.09.25.678583v1

**Findings:** RSC-to-V1 axons relay retinotopically selective signals tuned for spatial
and temporal frequency but not orientation. Two-colour imaging revealed that RSC bouton
receptive fields in V1 are systematically offset in the nasal direction relative to local
V1 neurons, suggesting RSC provides predictive spatial information to V1.

**Relevance to hm2p:** Characterises the RSP-to-V1 feedback pathway — the circuit by
which RSP could influence early visual processing. In our paradigm, this pathway is
relevant during light-on epochs when visual processing is active. The nasal offset
of RSP feedback RFs suggests RSP sends forward-looking spatial predictions to V1. Not
directly about HD coding, but relevant for understanding RSP's role in the visual
processing hierarchy. Not in the April 2 scan.

---

## Searches with no new relevant results

**Penk/enkephalin + cortex:** Same as April 2 scan — no new preprints on Penk-expressing
neurons in cortex in a spatial navigation or HD context. Recent Penk papers address
striatal D2-MSN enkephalin gating during cocaine abstinence (March 2026) and dorsal
raphe enkephalin in reward/aversion (2025). The gap in Penk+ cortical neuron function
remains wide open.

**Neuropil contamination + two-photon:** No new methods papers on neuropil correction
in the scan window. The most recent relevant work remains the soma-targeted jGCaMP8
vectors (2021) and standard fixed-coefficient subtraction methods.

**Head-mounted two-photon microscopy:** No new systems beyond those listed in the
April 2 scan (M-MINI2P, miniBB2p, FHIRM-TPM 3.0, simultaneous 2+3 photon multiplane).

**Spatial navigation + maze (rodents):** No new maze navigation papers with calcium
imaging in the scan window. The Omniroute maze (automated configurable maze, 2025) is
a methods paper with no neural recording.

---

## Summary of implications for hm2p

**New additions to the literature landscape since April 2:**

The most significant new finding is the Jedrasiak-Cape et al. 2024 paper on cell-type-
specific cholinergic control of RSG, which was missed in the first scan. This paper
provides the strongest existing precedent for our core hypothesis: that genetically
defined RSP excitatory subtypes have distinct computational roles in navigation-relevant
signals. Their LR neurons compute AHV differently from other subtypes due to distinct
cholinergic modulation. We should cite this paper prominently when motivating our
cell-type-specific approach.

The Rogers et al. 2026 aversive signalling paper adds to the evidence that RSP flexibly
remodels representations across contexts (calcium imaging), supporting the expectation
that our light/dark manipulation should produce measurable changes in RSP population
activity.

The Sarramone et al. 2026 ultraslow oscillation paper raises an interesting possibility:
systematic drift in darkness may have an oscillatory structure at very slow timescales.
Our 60-second dark epochs are too short to fully assess this, but we could look for
non-monotonic drift patterns.

**Papers to add to the citation list:**
- Jedrasiak-Cape et al. 2024 (cell-type-specific RSG computation) — cite in introduction
  when motivating cell-type hypothesis
- Rogers et al. 2026 (RSP contextual remodelling) — cite in discussion of light/dark
  context switching
- Tanimura et al. 2025 (RSP L1 cholinergic disruption) — cite if discussing RSP
  cell-type functional diversity more broadly

**Penk+ gap confirmation:** Still no published work on Penk-expressing RSP neurons in
any functional context. Our study remains the first to characterise this population's
HD coding properties.
