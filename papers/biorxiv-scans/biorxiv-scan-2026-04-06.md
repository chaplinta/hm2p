# bioRxiv Scan — 6 April 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk-CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-04-06. Searches covered: retrosplenial cortex, Penk/enkephalin +
cortex, head direction + two-photon, head direction + darkness/landmarks/drift, spatial
navigation + maze (rodents), visual processing in RSP, spatial navigation in RSP,
head-mounted two-photon microscopy, calcium imaging + maze navigation, neuropil
contamination + two-photon, angular head velocity + retrosplenial, cell-type-specific
spatial navigation, head direction + ring attractor + visual cue removal.

Papers already listed in the 2026-04-02 or 2026-04-04 scans are not repeated here.

---

## Highly relevant papers

### 1. Active locomotion predictively rescues head direction attractor dynamics in head-fixed mice

Authors not fully specified. 2026.
"Active locomotion predictively rescues head direction attractor dynamics in head-fixed
mice." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.01.12.698940v1

**Findings:** Recorded from anterodorsal thalamic nucleus (ADN) HD cells in head-fixed
mice under light-on and light-off conditions during foraging. Complete head-fixation
disrupted both unit-level and population-level HD encoding. Constraining head-on-body
movements impaired HD population activity. However, attractor dynamics recovered several
hundred milliseconds *before* locomotion onset during head restraint. This predictive
recovery suggests that an efference copy or prediction of re-afferent signals is
necessary to maintain HD network activity. During head fixation, immobility altered the
HD ring attractor state, whereas locomotion onset predictively restored it. The HD system
operates as an active, state-dependent estimator rather than a passive integrator.

**Relevance to hm2p:** This is directly relevant to interpreting our light-off data.
The finding that HD attractor dynamics depend on locomotion state means we must account
for movement state when analysing HD tuning in darkness. Specifically:
- Stationary periods in darkness may show degraded HD representation not just because of
  missing visual cues, but because of reduced locomotor drive to the attractor.
- The predictive recovery before locomotion onset suggests motor planning signals
  contribute to HD maintenance — this is a confound if mice move differently in light
  vs dark.
- The ADN recording site is one synapse upstream of RSP. If ADN HD signals are
  state-dependent, RSP receives state-dependent HD input. Any cell-type differences in
  how Penk+ vs non-Penk cells respond to this state-dependent input would be informative.
- Key analysis: compare HD tuning stability in darkness separately for moving vs
  stationary epochs. If tuning degrades more during stationary dark periods, this is
  consistent with the efference copy mechanism described here.

---

### 2. NaviGraph: a graph-based framework for multimodal analysis of spatial decision-making

Koren Iton A, Iton E, Michaelson DM, Blinder P. 2025.
"NaviGraph: A graph-based framework for multimodal analysis of spatial decision-making."
bioRxiv.
https://www.biorxiv.org/content/10.1101/2025.05.18.654725v1

**Findings:** Developed an open-source pipeline (NaviGraph) that maps behavioural
trajectories, head orientation dynamics, and neuronal calcium imaging data onto a
graph-based representation of maze structure. Demonstrated the framework using
miniaturised microscope calcium imaging in retrosplenial cortex during maze navigation.
Identified decision-point-specific neuronal activity patterns and subpopulation dynamics
linked to path familiarity. The graph structure enables alignment of neural activity
with specific maze locations and decision points.

**Relevance to hm2p:** Directly relevant as a methodological framework. Our rose maze
has a well-defined graph structure (central hub + radial arms + dead ends), and
NaviGraph's approach of mapping neural activity onto maze topology could be applied to
our data. Specific applications:
- Map Penk+ vs non-Penk activity to decision points (central hub where arm choices are
  made) vs dead ends vs corridors.
- Test whether the two populations differ in decision-point-specific activity, which
  would suggest different roles in navigation planning vs spatial orientation.
- The path familiarity analysis is relevant: do cells change their activity as mice
  revisit arms within a session?
- The fact that they used RSP calcium imaging in a maze is the closest methodological
  parallel to our work. We should compare our maze exploration analyses with their
  approach.
- Posted May 2025 but missed in earlier scans. Not a new preprint, but newly discovered.

---

### 3. Cortex-wide cellular imaging in freely locomoting mice using CortexCAM

Cherkkil P, Viavattine G, Kota S, et al. 2026.
"Cortex-Wide Cellular Imaging in Freely Locomoting Mice Using Cortex Camera Array
Microscope (CortexCAM)." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.02.11.705445v1

**Findings:** Developed a multi-camera array microscope (CortexCAM) capable of imaging
over 9000 individual neurons across primary and secondary motor, somatosensory, visual,
retrosplenial, and association cortices simultaneously in freely locomoting mice. Used a
passive mechanical gantry system allowing volitional control of translational (x, y) and
rotational (yaw) motion. Applied to alternating choice tasks and social interactions,
enabling cortex-wide cellular dynamics during behaviours impossible under head fixation.

**Relevance to hm2p:** This is a technology paper, but the inclusion of retrosplenial
cortex in their cortex-wide imaging is noteworthy. Their approach images RSP alongside
visual cortex and motor cortex simultaneously, which could reveal inter-area dynamics
that our single-region imaging cannot capture. Relevant for discussion of limitations
and future directions: simultaneous RSP + V1 imaging during light/dark alternation
would allow direct measurement of visual input to RSP rather than inferring it from
the light condition. The gantry system for free locomotion is an alternative approach
to head-mounted 2P — different tradeoffs (lower resolution but wider coverage). Not
directly applicable to our current analysis but relevant as methods context.

---

## Moderately relevant papers

### 4. Widespread corticothalamic connectivity identifies the inferior pulvinar as a central node in visual network architecture

Kwan WC, Fan AY, Romanowski AJ, Carril-Mundinano I, de Souza MJ, Bourne JA. 2026.
"Widespread Corticothalamic Connectivity Identifies the Inferior Pulvinar as a Central
Node in Visual Network Architecture." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.03.02.709198v1

**Findings:** MRI-guided retrograde tracer injections in marmoset inferior pulvinar (PIm)
revealed cortical inputs from occipital, temporal, parietal, and cingulate cortices with
strong layer V predominance. Early and middle-tier visual areas (V1, V2, V3, V3A, V4,
V6/DM) contributed substantial input. The inferior pulvinar was identified as a central
hub in visual network architecture, with implications for visuomotor integration and
residual visual function.

**Relevance to hm2p:** The inferior pulvinar is part of the thalamic visual pathway
that is parallel to the lateral geniculate pathway. Cingulate cortex (including RSP)
projects to PIm, suggesting RSP visual feedback reaches not only V1 (as shown by
Timplalexi et al. 2025 in the April 4 scan) but also thalamic visual relays. This
is relevant for understanding the full circuit through which RSP visual signals
propagate. Marmoset anatomy, so species translation to mouse is uncertain. Background
for discussion only.

---

### 5. A plug-and-play ROI imaging module extends three-photon microscopy to 1.7 mm depth

Authors not fully specified. 2026.
"A plug-and-play ROI imaging module and deep-learning denoising framework extend
three-photon microscopy to 1.7 mm depth." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.01.02.697343v1

**Findings:** Developed a plug-and-play module for existing three-photon systems that
selectively excites only neuron-occupied regions, reducing power requirements to
approximately one-third (36.7 mW) of conventional full-field scanning while producing
a 21.6-fold increase in fluorescence intensity. Combined with CellposeSAM segmentation
and deep learning denoising. Extends functional imaging depth to 1.7 mm.

**Relevance to hm2p:** Technology paper relevant to future directions. The ROI-based
scanning approach could improve signal quality in head-mounted 2P systems where power
is limited. The deep learning denoising framework could be applicable to low-SNR
calcium imaging data. Not directly applicable to our current dataset but relevant as
methods context for the discussion of technical advances.

---

## Searches with no new relevant results

**Penk/enkephalin + cortex:** No new preprints beyond those noted in the April 2 and
April 4 scans. The same striatal and brainstem Penk papers continue to dominate search
results. No work on Penk-expressing neurons in RSP or any cortical region in a spatial
navigation context. The gap remains wide open and confirms the novelty of our study.

**Neuropil contamination + two-photon:** No new methods papers on neuropil correction
in the scan window. Search results return standard approaches (Suite2p fixed-coefficient
subtraction, CaImAn). No advances in neuropil correction methods since the last scan.

**Head-mounted two-photon microscopy:** No new systems beyond those listed in the
April 2 scan (M-MINI2P, miniBB2p, FHIRM-TPM 3.0, simultaneous 2+3 photon multiplane).
The CortexCAM (paper #3 above) is a widefield array approach rather than two-photon.

**Head direction + darkness/landmarks/drift:** No new experimental papers beyond those
in previous scans. The active locomotion paper (#1 above) was the only new finding in
this search space.

**Spatial navigation + maze (rodents):** NaviGraph (#2 above) was the only new
discovery. No other maze + calcium imaging papers not already covered.

---

## Summary of implications for hm2p

**New additions to the literature landscape since April 4:**

The most important new finding is the active locomotion / HD attractor paper (paper #1).
This has direct implications for our analysis: HD attractor dynamics are state-dependent,
with locomotion onset predictively restoring attractor function during head restraint.
While our mice are freely moving (not head-fixed), the principle applies: stationary
periods in darkness may show degraded HD representation due to reduced locomotor drive,
not just missing visual input. This means our analysis of HD tuning in darkness MUST
separate moving from stationary epochs to disentangle visual cue loss from locomotor
state effects. This was already a planned analysis (see Jayakumar et al. in the April 2
scan), but this paper provides additional mechanistic motivation.

NaviGraph (paper #2) provides a methodological framework for mapping neural activity
onto maze graph structure that could be applied to our rose maze data. It is also the
closest existing study to ours in terms of using RSP calcium imaging during maze
navigation, which makes it an important comparison point.

**New confound identified:**
- Locomotor state confound in darkness: HD attractor degradation during immobility
  (paper #1) means that if mice are more stationary in dark epochs, HD tuning
  degradation could be partly locomotor-driven rather than vision-driven. Must
  match movement speed distributions when comparing light vs dark HD tuning.

**Papers to add to the citation list:**
- Active locomotion / HD attractor paper (2026) — cite when discussing movement-state
  controls in the HD analysis
- NaviGraph (Koren Iton et al. 2025) — cite in methods comparison and maze-structure
  analysis

**Penk+ gap confirmation:** Still no published work on Penk-expressing RSP neurons in
any functional context. Three consecutive scans confirm the novelty of our study.
