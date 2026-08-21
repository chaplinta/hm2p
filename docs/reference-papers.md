# Reference Papers

Papers used in or relevant to the hm2p pipeline, with summaries of how
each contributes to the project.

All papers are stored in `/workspace/papers/`.

---

## 1. Zong et al. 2017 — FHIRM-TPM microscope

**Citation:** Zong W et al. 2017. "Fast high-resolution miniature two-photon
microscopy for brain imaging in freely behaving mice." Nature Methods
14(7):713-719. doi:10.1038/nmeth.4305

**What it describes:** Design and validation of the first-generation miniature
two-photon microscope (FHIRM-TPM) used for calcium imaging in freely
behaving mice. The headpiece weighs 2.15 g and uses a hollow-core photonic
crystal fiber to deliver 920 nm femtosecond laser pulses. Achieves 0.64 µm
lateral resolution at 40 Hz frame rate with 256×256 pixel raster scanning.

**Relevance to hm2p:** The FHIRM-TPM is the basis for the microscope used
in the hm2p experiments, with modifications (SciScan acquisition instead of
the original LabVIEW FPGA control; imaging at ~9.6 Hz rather than 40 Hz,
consistent with a larger FOV or different scan settings). Single plane imaging
per session.

**Methods used in hm2p from this paper:** None directly — the paper's
template-matching event detection (3×SD of baseline) is not used. However,
the SD-threshold event detector added to `calcium/events.py` follows a
similar principle.

---

## 2. Zong et al. 2022 — MINI2P and NATEX analysis pipeline

**Citation:** Zong W et al. 2022. "Large-scale two-photon calcium imaging in
freely moving mice." Cell 185(7):1240-1256. doi:10.1016/j.cell.2022.02.017

**What it describes:** The MINI2P is the next-generation miniature two-photon
microscope (<3 g) with z-scanning for multi-plane imaging. The paper
demonstrates recordings from >1000 neurons in visual cortex, entorhinal
cortex, and hippocampus in freely moving mice. Includes the NATEX analysis
pipeline for motion correction, cell detection, calcium signal processing,
and spatial tuning analysis.

**Relevance to hm2p:** The NATEX analysis methods are the closest published
reference for how to process freely-moving 2P calcium imaging data. Several
methods from this paper are adapted in the hm2p pipeline.

**Methods used in hm2p from this paper:**

| Method | Zong 2022 approach | hm2p implementation |
|--------|-------------------|---------------------|
| Motion correction | Suite2p rigid + non-rigid, 6×6 blocks | Same (Suite2p defaults) |
| Neuropil subtraction | Fixed 0.7 coefficient | Same (`calcium/neuropil.py`) |
| Baseline (F0) | 8th percentile in ±15s window | Suite2p method: Gaussian smooth → rolling min → rolling max. Simpler but adequate for hm2p data which shows minimal bleaching |
| dF/F | (F - F0) / F0 | Same |
| Event detection | 2×SD threshold, min 0.75s duration | SD-threshold method in `calcium/events.py` (`detect_events_sd`). V&H method also available |
| Deconvolution | Suite2p OASIS, normalized | Same: `spks.npy` → `deconv` and `deconv_norm` in ca.h5 |
| SNR | 90th percentile of event amplitudes / mean noise | Peak amplitude / std of non-event periods (similar but not identical) |
| HD computation | Ear vector, DLC with 4 body parts | Same approach, 5 body parts (SuperAnimal TopViewMouse) |
| Speed filter | Exclude < 2.5 cm/s | Same |
| Spatial tuning | Skaggs information, MVL, shuffle tests | Same (`analysis/tuning.py`, `analysis/information.py`) |

---

## 3. Voigts & Harnett 2020 — RSP imaging and V&H event detection

**Citation:** Voigts J & Harnett MT. 2020. "Somatic and dendritic encoding of
spatial variables in retrosplenial cortex differs during 2D navigation."
Neuron 105(2):237-245. doi:10.1016/j.neuron.2019.10.016

**What it describes:** Two-photon imaging in RSP (the same brain region as
hm2p) during free locomotion with volitional head rotation, using a rotating
headpost. Shows that apical tuft dendrites encode different navigational
variables than their parent somata. Introduces a percentile-based calcium
event detection algorithm.

**Relevance to hm2p:** This is the primary methodological reference for the
hm2p experiment — same brain region (RSP), same imaging modality (2P
GCaMP), similar frame rate (~9-11 Hz). The V&H event detection algorithm is
one of the two methods implemented in `calcium/events.py`.

**Methods used in hm2p from this paper:**

| Method | V&H approach | hm2p implementation |
|--------|-------------|---------------------|
| Event detection | Percentile-based Gaussian noise model: mean = 40th prc, std = 10th-90th prc range, onset at P(noise) < 0.2, offset at P(noise) > 0.7 rising | Same algorithm in `calcium/events.py` (`detect_events_single`). prob_onset changed to 0.3 to catch rises earlier |
| Joint soma-dendrite detection | Product of soma and dendrite noise probabilities | Not implemented (single-plane imaging, soma/dendrites classified post-hoc by shape) |
| Neuropil subtraction | Per-ROI coefficient from linear fit to lowest 10th percentile | Not used — hm2p uses fixed 0.7 coefficient (Suite2p/Zong convention) |
| HD computation | Ear vector from DLC | Same |
| Spatial tuning | KL divergence between rate and occupancy distributions | Skaggs information instead (more standard in HD/place cell literature) |

**Key difference:** The V&H paper images soma and dendrites in separate
planes using an ETL. hm2p images a single plane where soma and dendrite ROIs
co-exist and are classified post-hoc by aspect ratio.

---

## 4. Pachitariu et al. 2016 — Suite2p (original)

**Citation:** Pachitariu M et al. 2016. "Suite2p: beyond 10,000 neurons with
standard two-photon microscopy." bioRxiv. doi:10.1101/061507

**What it describes:** The original Suite2p pipeline for motion correction, ROI
detection, activity extraction with neuropil correction, and spike
deconvolution. Runs faster than real-time on standard workstations and
recovers ~2× more cells than previous methods.

**Relevance to hm2p:** Suite2p is the default calcium extraction backend
(Stage 1). All 26 sessions were processed with Suite2p on EC2 g4dn.xlarge.

**Methods used in hm2p from this paper:**

- **Registration:** Phase-correlation-based rigid registration, with optional
  non-rigid correction using block-wise shifts.
- **ROI detection:** Activity-based cell detection using iterative sparse
  matrix decomposition with L0 sparsity constraints.
- **Neuropil model:** Per-ROI annular neuropil region, coefficient stored in
  `ops["neucoeff"]` (typically 0.7).
- **Baseline (F0):** Rolling Gaussian smooth (σ = `sig_baseline` × fps
  frames) → rolling minimum (`win_baseline` × fps frames) → rolling maximum
  (same window). This 3-step filter is reimplemented in `calcium/dff.py`.
- **Deconvolution:** OASIS algorithm for fast online deconvolution. Output
  stored as `spks.npy`, loaded into ca.h5 as `deconv`.
- **Cell classification:** `iscell.npy` binary classifier output. hm2p adds
  post-hoc soma/dendrite/artefact classification based on aspect ratio and
  compactness from `stat.npy`.

---

## 5. Stringer et al. 2026 — Suite2p (updated)

**Citation:** Stringer C et al. 2026. "Extracting large-scale neural activity
with Suite2p." (HHMI Janelia)

**What it describes:** Updated Suite2p paper with GPU-accelerated non-rigid
motion correction, improved cell detection benchmarks (outperforms CaImAn
and Fiola), quality control methods, and demonstrations on 100,000+ neuron
recordings.

**Relevance to hm2p:** Documents the version of Suite2p (v1.0+) actually
used in the pipeline. The GPU-accelerated registration is what runs on the
EC2 g4dn instances. The quality control steps inform the QC frontend pages.

**Key updates over Pachitariu 2016:**

- GPU non-rigid registration runs 5× faster than CPU alternatives
- Cell detection finds more cells with fewer false positives than CaImAn/Fiola
- New benchmarking framework for evaluating detection performance
- Support for one-photon and voltage imaging (not used in hm2p)

---

## 6. Rosenberg et al. 2021 — Maze navigation behaviour

**Citation:** Rosenberg M et al. 2021. "Mice in a labyrinth show rapid
learning, sudden insight, and efficient exploration." eLife 10:e66175.
doi:10.7554/eLife.66175

**What it describes:** Behavioural analysis of mice navigating a binary
labyrinth (63 T-junctions, 64 dead ends). Mice learn the correct 10-bit
choice after only ~10 reward experiences — 1000× faster than 2AFC
experiments. Exploration is explained by local turning rules (forward bias +
alternation) without requiring a global cognitive map.

**Relevance to hm2p:** The maze analysis module (`src/hm2p/maze/`) adapts
Rosenberg's behavioural metrics for the hm2p q-rose maze (7×5 grid, 23
accessible cells, 7 T-junctions, 9 dead ends). The q-rose maze is much
simpler (~1/8th the size) and has no reward, but the same analytical
framework applies.

**Methods adapted in hm2p from this paper:**

| Method | Rosenberg approach | hm2p implementation |
|--------|-------------------|---------------------|
| Maze graph | Binary tree with 127 corridors | 7×5 grid graph, 23 cells (`maze/topology.py`) |
| Trajectory discretization | Node sequence from continuous position | Same approach (`maze/discretize.py`) |
| Occupancy | Time per node / total time | Same (`maze/analysis.py:cell_occupancy`) |
| Exploration efficiency | Fraction of nodes visited over time | Same (`maze/analysis.py:exploration_efficiency`) |
| Turn bias | Left/right/forward/back at each junction | Same (`maze/analysis.py:per_junction_turn_bias`) |
| Monotonic paths | Paths with monotonically decreasing distance to target | Adapted (`maze/analysis.py:find_monotonic_paths`) |
| Sequence entropy | Shannon entropy of node transition sequences | Same (`maze/analysis.py:sequence_entropy`) |
| Markov model | 1st and 2nd order transition matrices | Same (`maze/analysis.py:transition_matrix`) |

**Key difference:** Rosenberg's maze is a binary tree (each junction has
exactly 2 forward options). The hm2p q-rose maze has T-junctions with 3
options (left/right/back) and corridors with 2 options (forward/back). The
topology analysis accounts for this.

---

## 7. Zagha et al. 2022 — Movement confounds in neural recordings

**Citation:** Zagha E et al. 2022. "The Importance of Accounting for Movement
When Relating Neuronal Activity to Sensory and Cognitive Processes." J
Neurosci 42(8):1375-1382. doi:10.1523/JNEUROSCI.1919-21.2021

**What it describes:** Movement-related neural activity is widespread across
the mouse brain, including early sensory areas. Failing to account for
movement risks misattributing movement-related signals to sensory or
cognitive processes. Reviews three case studies where ignoring movement
would have led to incorrect conclusions. Argues that movement signals
should be considered first when correlating neural activity with task
variables.

**Relevance to hm2p:** Directly relevant because hm2p records from freely
moving mice. Neural activity in RSP that correlates with head direction
could partly reflect movement-related signals rather than genuine
directional tuning. The paper provides specific recommendations for
controlling for this confound.

**Implications for hm2p analysis:**

1. **Speed gating is necessary but not sufficient.** The current analysis
   excludes frames where speed < 2.5 cm/s. But Zagha et al. show that
   movement-related signals persist even after speed gating, because
   subthreshold movements, arousal changes, and preparatory postural shifts
   co-vary with task variables.

2. **GLM with movement regressors.** The NEMOS encoding model should include
   speed, AHV, and acceleration as nuisance regressors alongside HD and
   position. This isolates the variance explained by HD after accounting for
   movement. If HD tuning disappears after including movement regressors,
   the tuning was movement-related, not directional.

3. **Matched-condition comparisons.** When comparing light vs dark HD tuning,
   movement statistics (speed distribution, AHV, time active) should be
   reported for each condition to confirm they are comparable. If mice move
   differently in the dark, apparent tuning changes could reflect movement
   differences rather than visual cue dependence.

4. **Task-uninstructed movements.** The hm2p mice are freely exploring (no
   task, no reward). All movements are uninstructed. This means the full
   behavioural repertoire (grooming, rearing, whisking, postural shifts) is
   present and could generate confounding neural signals.

---

## 8. Pereira et al. 2026 — Route planning and structure learning in complex mazes

**Citation:** Pereira M, Godinho BS, Machens CK, Costa RM, Akam T. 2026.
"Flexible route planning and rapid structure learning by mice in complex
environments." bioRxiv. doi:10.64898/2026.06.02.729586

**What it describes:** A behavioural assay ("route planning task") in a
reconfigurable 6×6 grid of towers connected by removable walkways. A randomly
selected tower is cued by an LED on each trial; the mouse navigates there for a
water reward. 16 mice (8 pilot, 8 main), 59,651 trajectories, 282,855 choice
points. Maze layouts were numerically optimised (170,000 candidates) to
dissociate **structure-based** navigation (reducing shortest-path/geodesic
distance to goal) from **vector-based** navigation (reducing Euclidean distance
to goal), using two graph metrics: fraction of "informative" states where the
two strategies disagree, and flatness of the betweenness-centrality distribution.
A hierarchical-Bayesian mixture-of-strategies model (softmax over graded action
preferences) shows a large Structure weight, smaller Vector weight, and a large
Anti-backward (u-turn avoidance) weight. Structure knowledge is evident from the
first session on a newly configured maze. No neural recordings.

**Relevance to hm2p:** Moderate — methodological, not conceptual. The **critical
disanalogy** is that every headline analysis in this paper is defined relative to
a cued goal (excess steps, optimal choice rate, structure/vector indices all
require the goal location). The hm2p paradigm is free exploration with no goal,
no reward and no cue, so those analyses cannot be computed on our data.
Rosenberg et al. 2021 (entry 6) remains the closer analogue and is **not**
superseded by this paper — Rosenberg studied goal-free exploration, Pereira
studied goal-directed navigation, and the two conclusions are compatible.

What it does add: (a) a third independent demonstration that u-turn avoidance is
a dominant behavioural component across mazes and tasks, agreeing with
Rosenberg's forward bias and with the preservation of local turn rules across
light/dark in the hm2p behaviour analysis; (b) independent support for very
rapid (single-session) structure learning, alongside the hm2p single-dark-epoch
adaptation result; (c) betweenness centrality as a graded maze-structure
descriptor.

**Methods usable in hm2p from this paper:**

| Method | Pereira approach | Applicability to hm2p |
|--------|-----------------|----------------------|
| Betweenness centrality | Per-node fraction of all-pairs shortest paths passing through it; used to characterise maze structure | Directly applicable. `maze/topology.py` already computes all-pairs shortest paths. Usable as a graded per-cell covariate for the light/dark occupancy-change analysis, replacing the categorical corridor/junction/dead-end split |
| Strategy index normalisation | `I_q = (c_q − r_q)/(o_q − r_q)`, correcting a rule's hit rate for the expected rate under uniform choice among available actions | Applicable to `maze/choice_models.py:rule_accuracies`, which currently reports an uncorrected raw hit rate |
| Mixture-of-strategies model | Conditional logit / softmax over additively combined **graded** scalar action preferences, with an explicit Anti-backward component; fit to every choice | Applicable as a replacement for the deterministic winner-take-all rules in `maze/choice_models.py`. The explicit anti-backward term is the key element the hm2p implementation lacks — u-turns are ~half of junction choices in the q-rose maze and no current rule predicts them |
| Fraction of informative states | Fraction of (location, goal) pairs where structure- and vector-preferred action sets are disjoint | Not applicable — goal-dependent. Could be computed as a descriptive property of the q-rose maze only |
| Excess steps, optimal choice rate, structure/vector indices | Goal-relative navigation efficiency measures | **Not applicable** — require a cued goal |
| Hierarchical Bayesian inference (Piray et al. 2019) | Group-level model fitting and comparison | Not adopted as the inferential layer — conflicts with the project's non-parametric policy |

**Statistical caution:** All inference in this paper is parametric (rmANOVA,
one-sample and paired t-tests) with n = 8. The pilot experiment reports n = 8 but
its rmANOVA degrees of freedom, F(8,40), imply n = 6; the main experiment is
internally consistent. Hierarchical model tests are reported as t₉ from 8
subjects without explanation of the degrees of freedom. Cite the effects, not the
p-values.

**Detailed review:** `papers/reviews/pereira-et-al-2026-summary.md`

---

## Source Code and Data Repositories

| Paper | Repository |
|-------|-----------|
| Zong 2017 | [Protocol Exchange](http://dx.doi.org/10.1038/protex.2017.048) (assembly protocol only) |
| Zong 2022 | [github.com/kavli-ntnu/MINI2P_toolbox](https://github.com/kavli-ntnu/MINI2P_toolbox) (NATEX pipeline, MATLAB) |
| V&H 2020 | [github.com/jvoigts/rotating-2p-image-correction](https://github.com/jvoigts/rotating-2p-image-correction) (brightness correction; event detection in STAR Methods) |
| Pachitariu 2016 | [github.com/MouseLand/suite2p](https://github.com/MouseLand/suite2p) |
| Stringer 2026 | Same repo (Suite2p v1.0+) |
| Rosenberg 2021 | [github.com/markusmeister/Rosenberg-2021-Repository](https://github.com/markusmeister/Rosenberg-2021-Repository) |
| Zagha 2022 | No code repository (review/perspective article) |
| Pereira 2026 | [github.com/pyControl/hardware/tree/master/GridMaze](https://github.com/pyControl/hardware/tree/master/GridMaze) (apparatus design files); [github.com/michaelfsp/pycbm](https://github.com/michaelfsp/pycbm) (hierarchical Bayesian inference) |
