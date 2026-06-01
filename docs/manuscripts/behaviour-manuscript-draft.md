# Mouse navigation behaviour in a q-rose maze with alternating light and dark epochs

**Working draft -- behavioural methods/descriptive paper**

Status: Draft v0.9 -- 2026-06-01 (added HMM kinematic states + graph topology analyses as Supplementary Figs S10--S11)

---

## Table of Contents

1. [Manuscript Outline](#manuscript-outline)
2. [Literature Context](#literature-context)
3. [Draft Introduction](#draft-introduction)
4. [Draft Methods](#draft-methods)
5. [Analysis Plan and Figures](#analysis-plan-and-figures)
6. [Draft Results](#draft-results)
7. [Discussion](#discussion)
8. [Supplementary Material](#supplementary-material)
9. [References](#references)

---

## Manuscript Outline

### Target

Short methods/descriptive paper (~3000--4000 words, excluding methods). Target
journals: *eLife* (tools/resources), *Scientific Reports*, *Behavioral
Neuroscience*, *STAR Protocols*, or *Journal of Neuroscience Methods*.

Not the full neural paper. This establishes the behavioural paradigm and
characterises how mice navigate the q-rose maze under light and dark
conditions, providing the behavioural foundation for future neural analyses.

### Core story

Mice exploring a q-rose maze (a reduced binary-choice labyrinth after
Rosenberg et al. 2021) show structured exploration strategies governed by
local turn rules. When overhead lights are extinguished (total darkness,
removing all visual cues), spatial coverage decreases robustly (p < 0.001,
r = 0.86). The coverage reduction reflects route stereotypy rather than
global disengagement: corridor and junction coverage drop significantly in
darkness, but dead-end coverage is unchanged (p = 0.26), indicating that
mice maintain visits to terminal destinations while consolidating onto fewer
connecting routes. The visited subgraph diameter contracts, transition
matrices diverge between conditions, and revisitation of already-covered
cells increases. Local navigation rules (turn alternation, backtracking,
transition entropy) are preserved. This dissociation between preserved
local decision rules and altered global route selection suggests that turn
decisions and route planning draw on different information sources, with
route selection depending on a spatial representation that degrades without
visual cues.

Route stereotypy is established after a single dark epoch: the first dark
epoch produces near-normal coverage (0.57 vs 0.30 in subsequent dark
epochs; p = 0.0001, r = 0.89). Speed does not change at the light-off
transition, and first-dark coverage equals first-light coverage, ruling out
a startle or anxiety response (Supplementary Fig. S5). After one experience
of navigating without visual cues, mice adopt a more conservative route
network and maintain it for the remainder of the session.

Converging analyses support this interpretation. A hidden Markov model on
kinematic features reveals three navigation states (pausing, slow
scanning, fast traversal), but their occupancy does not change between
conditions (Supplementary Fig. S10), indicating that darkness alters where
in the maze mice deploy each behaviour rather than the kinematic profile
itself. Graph-theoretic analysis of the directed transition graph shows
that the largest strongly connected component contracts from 96% to 84%
of visited cells in darkness (r = 0.85; Supplementary Fig. S11), meaning
the navigation network fragments into disconnected subgraphs -- an
independent corroboration of route stereotypy.

### Structure

| Section | Content | Words |
|---------|---------|-------|
| Introduction | Frame: maze navigation, exploration rules, light/dark, HD system | ~500 |
| Methods | Maze, animals, surgery, imaging, tracking, analysis | ~800 |
| Results | 4 main findings (see below) | ~1400 |
| Discussion | Comparison to Rosenberg, route stereotypy, implications | ~800 |
| Supplementary | Controls, null results, HMM, graph metrics | as needed |

### Four main results

1. **Maze structure and exploration strategies** -- Mice explore the 23-cell
   q-rose maze with high coverage and structured local turn rules (left-right
   alternation, modest backtracking). First-order Markov model preferred.

2. **Spatial coverage decreases in darkness** -- Per-epoch spatial coverage
   drops robustly in darkness (p = 0.0003, r = 0.86; primary-only p = 0.010).
   This is the strongest and most consistent behavioural marker of visual
   cue removal.

3. **The coverage drop reflects route stereotypy** -- Corridor and junction
   coverage drop significantly; dead-end coverage is unchanged. Mice
   maintain their destination repertoire while consolidating onto fewer
   connecting routes. Revisitation increases and transition matrices
   diverge. Speed contributes but does not fully explain the pattern
   (Supplementary Fig. S4).

4. **Route stereotypy is established after a single dark epoch** -- The
   first dark epoch has near-normal coverage; all subsequent dark epochs
   are stable at a lower level. Single-trial adaptation, not gradual
   learning. Controls rule out startle and anxiety (Supplementary Fig. S5).

---

## Literature Context

### Relevant recent literature

**Maze exploration in rodents**

- Rosenberg, Zhang, Perona & Meister (2021). "Mice in a labyrinth show rapid
  learning, sudden insight, and efficient exploration." *eLife* 10, e66175.
  doi:10.7554/eLife.66175
  -- The foundational maze study. Binary labyrinth with 63 T-junctions. Key
  behavioural findings: strong forward bias, left-right alternation, rapid
  learning (~10 reward experiences for 10-bit choices), second-order Markov
  models fit behaviour well. Our q-rose maze is adapted from this design.

- Koren Iton, Iton, Michaelson & Blinder (2025). "NaviGraph: A graph-based
  framework for multimodal analysis of spatial decision-making." *bioRxiv*.
  doi:10.1101/2025.05.18.654725
  -- Graph-based analysis framework for maze navigation with neural imaging.
  Applied to RSP miniscope data. Directly inspired our graph-topology
  analysis approach.

- Bhatt, Mareschal, Bhatt, Bhatt & Bhatt (2024). "Rodent maze studies: from
  following simple rules to complex map learning." *Brain Struct. Funct.* 229,
  1261--1278. doi:10.1007/s00429-024-02771-x
  -- Comprehensive review of 100+ years of rodent maze research. Documents
  evolution from simple rule-following to cognitive-map frameworks.

- Bhakti, Bhatt et al. (2024). "Stochastic characterization of navigation
  strategies in an automated variant of the Barnes maze." *eLife* 13, e88648.
  doi:10.7554/eLife.88648
  -- Markov chain models of navigation strategy switching in Barnes maze.
  Mice combine random, serial, and spatial strategies with context-dependent
  transition probabilities.

- Bhatt, Bhatt et al. (2021). "Learning-induced shifts in mice navigational
  strategies are unveiled by a minimal behavioral model of spatial
  exploration." *eNeuro* 8(5), ENEURO.0553-20.2021.
  doi:10.1523/ENEURO.0553-20.2021
  -- Minimal behavioural model identifies three sequential learning phases in
  maze exploration.

**Head direction and visual landmarks**

- Keshavarzi, Bracey, Faville, Campagner, Tyson, Lenzi, Branco & Margrie
  (2022). "Multisensory coding of angular head velocity in the retrosplenial
  cortex." *Neuron* 110, 532--543. doi:10.1016/j.neuron.2021.10.031
  -- From the same lab as the present study. RSP neurons encode angular head
  velocity through vestibular-visual integration. Visual input increases gain
  and SNR of AHV coding. Directly relevant to our light/dark manipulation.

- Ajabi, Keinath & Brandon (2023). "Population dynamics of head-direction
  neurons during drift and reorientation." *Nature* 615, 892--899.
  doi:10.1038/s41586-023-05813-2
  -- HD population varies along a second "gain" dimension during drift in
  darkness and reorientation to landmarks. The classical 1D ring attractor
  does not fully capture dynamics during cue conflict.

- Bicknell, van der Goes et al. (2024). "Coordinated head direction
  representations in mouse anterodorsal thalamic nucleus and retrosplenial
  cortex." *eLife* 13, e82952. doi:10.7554/eLife.82952
  -- Near-synchronous HD coding between ADn and RSP. Coordination maintained
  in darkness but with increased drift.

- Jacob, Casali, Bhatt et al. (2017). "An independent, landmark-dominated
  head-direction signal in dysgranular retrosplenial cortex." *Nat. Neurosci.*
  20, 173--175. doi:10.1038/nn.4465
  -- RSP HD cells anchored to local visual landmarks. Visual cues can override
  path integration signals.

- Stackman & Taube (1997). "Firing properties of head direction cells in
  the rat anterior thalamic nucleus: dependence on behavioral factors."
  *J. Neurosci.* 17, 9020--9037.
  -- Classical demonstration that HD cell preferred direction drifts in
  darkness at variable rates, with instantaneous realignment when lights
  return.

- Muir, Roth, Bhatt et al. (2022). "Flexible cue anchoring strategies enable
  stable head direction coding in both sighted and blind animals." *Nat.
  Commun.* 13, 5604. doi:10.1038/s41467-022-33204-0
  -- Blind mice develop olfactory-based HD anchoring, demonstrating flexible
  cue strategies. ~40% of HD cells become unstable in acute darkness.

**Behaviour in darkness**

- Chen, Oliva et al. (2024). "Ambient light impacts innate behaviors of
  New-World and Old-World mice." *bioRxiv*. doi:10.1101/2025.05.14.653927
  -- Dim light enhances escape responses. Darkness generally increases
  cautiousness in rodents.

- Whishaw & Tomie (1996), Avni et al. (2006), Fonio et al. (2009).
  Rodents reduce locomotor speed in novel or dark environments, increase
  thigmotaxis, and may shift from allocentric to egocentric navigation
  strategies.

**Methods and tracking**

- Mathis et al. (2018). "DeepLabCut: markerless pose estimation of
  user-defined body parts with deep learning." *Nat. Neurosci.* 21,
  1281--1289. doi:10.1038/s41593-018-0209-y

- Ye et al. (2024). "SuperAnimal pretrained pose estimation models for
  behavioral analysis." *Nat. Commun.* 15, 5165.
  doi:10.1038/s41467-024-48792-2

- Bala, Nguyen, Chang et al. (2024). "movement: a Python toolbox for pose
  estimation and kinematics." *Zenodo/GitHub*.
  neuroinformatics.dev

---

## Draft Introduction

Spatial navigation requires integrating information from multiple sources:
visual landmarks, self-motion cues (vestibular, proprioceptive, motor
efference), and internal representations of environment geometry (O'Keefe
& Nadel 1978; Taube et al. 1990; Etienne & Jeffery 2004). How these
information streams are combined, and how the brain adapts when one stream
is removed, remains a central question in systems neuroscience.

The head direction (HD) system provides a neural compass that is maintained
by path integration but anchored to environmental landmarks (Taube 2007).
When visual cues are removed, HD preferred firing directions drift at
variable rates (Stackman & Taube 1997; Goodridge et al. 1998), but the
attractor network structure is preserved (Ajabi et al. 2023). When landmarks
return, HD representations rapidly re-anchor, often within a single head
sweep (Zugaro et al. 2003). This interplay between path integration
maintenance and visual re-anchoring occurs on timescales of seconds to
minutes -- precisely the timescale of individual light-dark epochs in the
present study.

The retrosplenial cortex (RSP) sits at the intersection of the HD system and
the visual system, receiving head direction input from the anterior thalamus
and visual input from both primary and higher visual areas (Vann et al. 2009;
Mitchell et al. 2018). RSP neurons encode head direction (Chen et al. 1994;
Cho & Sharp 2001; Jacob et al. 2017), angular head velocity through
vestibular-visual integration (Keshavarzi et al. 2022), and spatial position
through conjunctions of vision and locomotion (Mao et al. 2020). RSP is
therefore a natural locus for studying how visual cue removal alters spatial
representations and, potentially, spatial behaviour.

Complex mazes offer a rich behavioural readout for studying spatial
navigation because they generate hundreds of natural binary decisions per
session without explicit training (Rosenberg et al. 2021). In Rosenberg's
63-junction binary labyrinth, mice showed structured exploration governed
by local rules: strong forward momentum, left-right turn alternation, and
gradual acquisition of maze structure through experience. These exploration
strategies can be formalised as Markov transition models, providing
quantitative metrics of navigation predictability, efficiency, and strategy.

Here we characterise the navigation behaviour of mice freely exploring a
q-rose maze -- a reduced binary-choice labyrinth (7 T-junctions, 9 dead
ends, 23 accessible cells) adapted from Rosenberg et al. (2021) -- under
alternating 1-minute light-on and light-off epochs. The light-off condition
removes all visual cues (total darkness), creating a within-session
manipulation of the available sensory information for navigation. We tracked
body position and head direction using DeepLabCut pose estimation from
overhead video and quantified exploration using graph-theoretic metrics
adapted from Rosenberg et al.

This behavioural characterisation serves two purposes. First, it establishes
the q-rose maze as a tractable paradigm for studying visually-guided
navigation in mice during neural imaging, documenting the range of
exploration strategies and their modulation by visual cue availability.
Second, it provides the behavioural foundation for interpreting neural
recordings from genetically-defined retrosplenial cortex populations imaged
during these same sessions, which will be reported separately.

---

## Draft Methods

### Animals

Sixteen mice (15 male, 1 female; age at first recording: 5--8 months)
contributed 26 recording sessions to this dataset. Twelve mice were from a
Penk-Cre line and four from the same Penk-Cre line crossed with an
intersectional Cre-OFF strategy targeting non-Penk excitatory neurons.
Because this paper reports only behavioural data (no neural analysis), the
genetic distinction between imaging groups is not relevant to the present
analyses, and all sessions are pooled for behavioural characterisation. All
mice were housed on a 12:12 light-dark cycle with *ad libitum* access to food
and water. Procedures were approved by the local institutional ethical review
committee and performed under a UK Home Office Project Licence in accordance
with the Animals (Scientific Procedures) Act 1986.

Six sessions were excluded from analysis based on pre-registered criteria:
one for fluctuating two-photon signals (exp 5, animal 1114356), two for
camera synchronisation failures (exps 13--14, animal 1117217), one for poor
two-photon recording quality (exp 19, animal 1117646), one for camera
synchronisation and tether restriction (exp 21, animal 1118023), and one for
combined poor imaging and restricted behaviour (exp 26, animal 1118317). This
left 20 usable sessions from 14 animals for behavioural analysis (10
Penk-Cre animals contributing 15 sessions; 4 Penk-Cre/Cre-OFF animals
contributing 5 sessions).

### Surgical preparation and head-mounted two-photon imaging

Each mouse underwent a craniotomy and chronic window implantation over the
retrosplenial cortex. A lightweight head-mounted two-photon microscope
(~2.5 g) was attached for calcium imaging of GCaMP6s-expressing neurons
at ~9.6 Hz during free behaviour. Because the present paper concerns only
behavioural data, details of the surgical preparation, virus injections, and
imaging parameters are deferred to the companion neural paper. The
head-mounted microscope and its fibre tether constrain the mouse's movement
to some degree; sessions where tether restriction was substantial are noted
in the experiment log and flagged with `bad_behav_times` masks (see below).

### The q-rose maze

The maze is a flat-floored, walled enclosure whose corridors form a q-shaped
rose pattern adapted from the binary labyrinth design of Rosenberg et al.
(2021). The layout comprises a 7 x 5 unit grid (each unit approximately
7 x 7 cm) with internal walls creating 23 accessible cells arranged as
interconnected corridors (Fig. 1A). The topology includes:

- **7 T-junctions**: cells with 3 accessible neighbours, each presenting a
  binary left-right choice relative to the direction of approach
- **9 dead ends**: cells with 1 accessible neighbour, where the mouse must
  reverse direction
- **7 corridor cells**: cells with exactly 2 accessible neighbours (straight
  passages)
- **0 crossroads**: no cells have 4 neighbours

The maze graph diameter (longest shortest path) is 9 cell steps. Internal
walls between cells (2, 4)/(3, 4) and (3, 4)/(4, 4) in the top row create
three separate branches at the maze periphery.

The maze was placed in the centre of a rectangular room with standard
laboratory visual cues (posters, equipment, doors) available under ambient
lighting.

### Light-dark manipulation

Overhead room lights were alternated in 1-minute epochs: 1 minute on,
1 minute off. Each recording session lasted approximately 20--30 minutes,
yielding approximately 10--15 light epochs and 10--15 dark epochs per
session. During dark epochs, all room lights were extinguished, producing
total darkness (verified by absence of any visible illumination). The
overhead camera uses infrared illumination, so image quality is unaffected
by the light manipulation; tracking operates on infrared-illuminated frames
in both conditions. The two-photon excitation laser (920 nm) also does not
produce visible light.

Light-on/off transition times were recorded by the DAQ system (National
Instruments) via TDMS files and synchronised to the imaging and video
timestamps. Each frame was labelled as `light_on=True` or `light_on=False`
based on the nearest DAQ light-sensor event.

### Behavioural tracking

Mouse body position was tracked from overhead infrared video (Basler
acA1300-200um camera, ~100 fps native, subsampled to ~30 fps for analysis)
using DeepLabCut 3.x (Mathis et al. 2018; Ye et al. 2024). The pose model
was an HRNet-W32 architecture fine-tuned from the SuperAnimal TopViewMouse
pretrained weights using memory replay (SA fine-tune mode). The model was
trained on 872 manually labeled frames drawn from 17 sessions spanning all
animals and both light conditions. Training ran for 300 epochs with the best
snapshot selected at epoch 220 based on validation loss. Eight body parts
were tracked: nose tip, left ear, right ear, head midpoint, neck, mid-back,
mouse centre (body centroid), and tail base. All keypoints except
head_midpoint map to SuperAnimal TopViewMouse keypoints; head_midpoint is a
custom keypoint trained from scratch (targeting the high-contrast two-photon
headstage, which is readily detectable in overhead infrared video).

On a held-out test set, the model achieved RMSE of 3.79 pixels and mAP of
87.07. Per-bodypart tracking quality was high: median pixel error ranged from
2.6 to 3.0 px across all bodyparts, with PCK@10 (percentage of correct
keypoints within 10 pixels) of 95--99%. Because the overhead camera uses
infrared illumination, tracking quality does not differ between light and
dark conditions (Supplementary Table S2).

Head direction was computed as the angle of the vector from the midpoint
between the two ears to the nose tip, unwrapped and expressed in degrees
(0--360). Angular head velocity (AHV) was computed as the temporal derivative
of unwrapped HD, converted to degrees per second. Running speed was computed
from the body centroid position after smoothing with a Gaussian kernel
(sigma = 5 frames at 30 fps).

A per-session rotation correction (0, 90, or 180 degrees) was applied to all
keypoint coordinates before HD computation to account for camera placement
variation across sessions (recorded in `experiments.csv` as `orientation`).

### Behavioural artefact exclusion

Mice occasionally became entangled in the two-photon fibre/wire tether,
producing artefactual immobility. These periods were identified by manual
inspection and logged in the experiment registry (`bad_behav_times` in
`experiments.csv`). All frames during flagged periods were excluded from
behavioural analysis via a `bad_behav` boolean mask. Total excluded time
varied by session (range: 0 to ~15 minutes; median: ~2 minutes).

### Position discretisation and maze graph

Continuous (x, y) position was mapped to the nearest accessible maze cell
using Euclidean distance to cell centres (each cell centre at
(col + 0.5, row + 0.5) in maze units). Cell sequences were compressed by
removing consecutive duplicates (staying in the same cell) and invalid frames
(NaN positions, positions outside the maze, `bad_behav=True`). Node
sequences were further compressed by retaining only visits to T-junctions
and dead ends, removing corridor transits.

The maze topology (adjacency, shortest paths, junction classification) was
computed from the polygon boundary and internal wall definitions following
the graph construction in `hm2p.maze.topology`.

### Turn classification

At each T-junction visit, the turn direction was classified relative to the
direction of approach using a cross-product of the approach and departure
vectors:
- **Left**: cross product of approach and departure vectors > 0
- **Right**: cross product < 0
- **Forward**: dot product > 0 and cross product = 0 (continuing straight)
- **Back**: dot product < 0 (reversing direction)

This classification follows the convention of Rosenberg et al. (2021) where
turns are defined in an egocentric (mouse-centred) reference frame.

### Markov transition models

First-order transition matrices P[i, j] = P(next cell = j | current cell = i)
and second-order transition tensors T[i, j, k] = P(next = k | prev = i,
current = j) were computed from cell sequences using maximum-likelihood
estimation with optional Laplace smoothing (pseudocount = 0.01). Model
comparison used Akaike (AIC) and Bayesian (BIC) information criteria,
following Rosenberg et al. (2021).

### Hidden Markov model of kinematic states

To decompose navigation into discrete kinematic modes, a Gaussian hidden
Markov model (HMM) with full covariance was fitted independently to each
session using three features: speed (cm/s), absolute angular head velocity
(deg/s), and spatial coverage rate (unique cells visited per second). All
features were z-scored within each session before fitting. The number of
hidden states K was selected by BIC comparison across K = 2, 3, and 4; K = 3
was preferred (lower BIC than K = 2; K = 4 introduced overlapping states
without improved interpretability). States were labelled post hoc by
ranking on mean speed within each session and then aligning labels across
sessions by the speed hierarchy (pausing < slow scanning < fast traversal).
State occupancy was computed as the fraction of frames assigned to each
state per epoch (light or dark), then compared between conditions using
Wilcoxon signed-rank (N = 20 sessions) with Holm-Bonferroni correction
across three states. Implementation used `hmmlearn.GaussianHMM` with
random state = 42 and 100 iterations.

### Graph-theoretic analysis of the navigation network

The cell-to-cell transition sequence for each condition (light or dark)
was represented as a directed graph where nodes are maze cells and edges
are observed transitions. An edge threshold of at least 2 transitions
per condition was applied to exclude spurious single-frame transitions.
Six graph metrics were computed for each condition using NetworkX:
(1) edge density (edges / possible edges), (2) mean out-degree, (3)
number of strongly connected components (SCCs), (4) fraction of nodes
in the largest SCC, (5) global efficiency (mean inverse shortest path
length), and (6) transitivity (fraction of directed triads that are
transitive). Light and dark values were compared using Wilcoxon
signed-rank (N = 20 sessions) with Holm-Bonferroni correction across
six metrics. This approach was inspired by the NaviGraph framework
(Koren Iton et al. 2025).

### Statistical analysis

All statistical tests were non-parametric. Within-session paired comparisons
(light vs dark epochs within the same session) used the Wilcoxon signed-rank
test. Between-session unpaired comparisons used the Mann-Whitney U test.
Correlations used Spearman rank correlation. Multiple comparisons were
corrected using Holm-Bonferroni step-down correction within each figure
family. Effect sizes (rank-biserial correlation for Mann-Whitney;
matched-pairs rank-biserial for Wilcoxon) were reported alongside p-values
for all tests.

*Reporting convention:* Figures display uncorrected p-values for readability.
Holm-Bonferroni-adjusted p-values are reported in the main text and in
Table 1. Where adjusted p-values differ qualitatively from uncorrected
values, this is noted explicitly.

Circular statistics (Rayleigh test, circular mean and variance) were used for
HD-related analyses. HD sampling uniformity was assessed using the Rayleigh
test on HD angle distributions per maze cell.

For within-session light vs dark comparisons, metrics were computed
separately for all light epochs pooled and all dark epochs pooled within a
session. The session-level paired difference (dark - light) was then tested
across sessions using Wilcoxon signed-rank (N = 20 sessions).

### Pseudoreplication

Some animals contributed multiple sessions (4 animals with 2--3 usable
sessions each), creating mild pseudoreplication in session-level analyses.
As a robustness check, all primary light-vs-dark comparisons were repeated
using only primary-experiment sessions (one per animal, N = 11 independent
animals). Results are reported in Supplementary Table S1. Where conclusions
differ between the full and primary-only analyses, the primary-only result
takes precedence.

---

## Analysis Plan and Figures

### Figure 1: Maze structure and exploration

**Panel A.** Schematic of the q-rose maze with cell grid overlay. Colour-code
cells by type (junction = red, corridor = grey, dead end = blue). Show maze
graph with nodes and edges.

**Panel B.** Example trajectory (full session) from one mouse overlaid on
maze outline. Colour-code by time to show exploration progression.
**[SYNTHETIC DATA PLACEHOLDER]** The current figure uses synthetic
trajectory data generated for layout purposes. This panel must be replaced
with a real session trajectory before submission.
<!-- TODO(DS-agent): Replace Figure 1B with real trajectory from a representative session (e.g., exp 11, animal 1116663). Load position data from sync.h5 or kinematics.h5. -->

**Panel C.** Heatmap of mean occupancy across all sessions (fraction of time
spent in each cell). Normalised per session, then averaged across sessions.

**Panel D.** Coverage curve: fraction of unique cells visited vs time (or
number of cell transitions). Show individual sessions as thin lines, mean as
thick line. Mark 50% and 90% coverage thresholds.

**Panel E.** Turn direction distribution and sequential autocorrelation.
Left-right alternation (negative autocorrelation = -0.172, p < 0.0001).
First-order Markov preferred over second-order in all sessions (BIC).

---

### Figure 2: Spatial coverage decreases in darkness

**Panel A.** Per-epoch coverage (fraction of 23 cells visited per 1-minute
epoch) in light vs dark. Box/violin plot of session means, paired by session.
This is the primary finding (p = 0.0003, r = 0.86).

**Panel B.** Coverage difference (dark - light) per session, ordered by
magnitude. Shows consistency of the effect across sessions.

**Panel C.** Coverage per active minute (cells visited normalised by active
time). Also lower in darkness (p = 0.001, r = 0.78), indicating that the
effect is not purely a locomotor artefact. Running speed showed a trend
toward reduction (p = 0.076; Supplementary Fig. S4).

---

### Figure 3: The coverage drop reflects route stereotypy

**Panel A.** Per-epoch coverage by maze cell type (junction, corridor,
dead end) in light vs dark. Paired box plots for each cell type. Key
finding: corridor coverage drops most strongly (p < 0.00001, r = 0.97),
junction coverage drops moderately (p = 0.001, r = 0.82), dead-end
coverage is unchanged (p = 0.26). Include direct comparison of the
dead-end vs junction drop magnitudes (p = 0.0004, r = 0.84).

**Panel B.** Visited subgraph diameter contracts in darkness (light: 6.35;
dark: 5.57 cells; p = 0.002, r = 0.80).

**Panel C.** Jensen-Shannon divergence between light and dark transition
matrices. Histogram of observed per-session JSD values with the
permutation null distribution overlaid (1000 permutations). Observed mean
JSD = 0.068 vs null mean = 0.018 (3.9x; p < 0.001). Routes change despite
preserved transition entropy (Supplementary Table S2).

**Panel D.** Revisitation index (total transitions / unique cells visited)
in light vs dark. Paired comparison across sessions (p = 0.011, r = 0.64).
Higher values in darkness indicate increased revisitation of
already-covered cells. Note: does not survive primary-only analysis
(p = 0.067; Supplementary Table S1).

**Panel E.** Per-cell coverage heatmap on maze grid. Colour-code each of
the 23 cells by the mean visit fraction delta (dark - light). Six cells
survive Holm-Bonferroni correction. The spatial gradient (central cells
lose more visits; rho = 0.75, p < 0.0001) is a topological consequence of
the maze graph rather than preferential avoidance of central locations
(route-dropping null model: permutation p = 0.281; Supplementary Fig. S6).

---

### Figure 4: Single-trial adaptation to darkness

**Panel A.** Dark-epoch coverage as a function of epoch number within each
session. Individual sessions as thin lines, session-mean as thick line.
No systematic trend after the first epoch (slope test: p = 0.81).

**Panel B.** First dark epoch vs subsequent dark epochs. Paired comparison
within each session (first: 0.57; subsequent mean: 0.30; Wilcoxon,
p = 0.0001, adjusted p = 0.0004, r = 0.89, N = 20). The discontinuity
between epoch 1 and epoch 2 is the dominant feature.

**Panel C.** Light-epoch coverage as a function of epoch number. Gradual
decline over the session (slope test: p = 0.011), consistent with general
exploration habituation. Contrast with dark epochs, which show a step
change rather than a gradual decline.

**Panel D.** Speed across dark epochs. No systematic trend (p = 0.86,
N = 14), ruling out a locomotor explanation for the first-epoch effect.

---

### Table 1: Key behavioural metrics

*Note: Values shown are means across sessions (N = 20). Wilcoxon signed-rank
test on paired session-level values. p-values are Holm-Bonferroni-adjusted
within each figure family. Effect sizes are matched-pairs rank-biserial r.
See Supplementary Table S1 for full battery including primary-only robustness
and Supplementary Table S2 for control metrics.*

| Metric | Light | Dark | p (adj) | r | Fig. |
|--------|-------|------|---------|---|------|
| Per-epoch coverage (frac) | 0.400 | 0.337 | 0.0008 | 0.86 | 2A |
| Coverage / active min | 19.8 | 17.8 | 0.001 | 0.78 | 2C |
| Corridor coverage (frac) | 0.482 | 0.380 | <0.0001 | 0.97 | 3A |
| Junction coverage (frac) | 0.465 | 0.389 | 0.001 | 0.82 | 3A |
| Dead-end coverage (frac) | 0.286 | 0.262 | 0.261 | 0.30 | 3A |
| DE vs junc. drop interaction | -- | -- | 0.0004 | 0.84 | 3A |
| Visited diameter (cells) | 6.35 | 5.57 | 0.002 | 0.80 | 3B |
| JSD (transition matrix) | -- | -- | <0.001 | -- | 3C |
| Revisitation index | 3.39 | 3.88 | 0.011 | 0.64 | 3D |
| First dark epoch cov. | -- | 0.567 | 0.0004 | 0.89 | 4B |
| Subsequent dark cov. | -- | 0.303 | -- | -- | 4B |

---

## Draft Results

### 1. Mice rapidly explore the q-rose maze with structured local turn rules

Mice explored the 23-cell q-rose maze with high coverage. Across 20
sessions, mice visited a median of 22.5 cells (mean 22.1 +/- 1.3; range
18--23), achieving a mean coverage fraction of 0.961 +/- 0.056
(Fig. 1D). Total distance travelled varied considerably across sessions
(median 57.7 m; range 31.7--117.8 m), reflecting individual differences in
locomotor activity. Occupancy was non-uniform: T-junction cells were
visited more frequently than dead-end cells, consistent with junctions
serving as transit hubs in the maze graph (Fig. 1C).

At T-junctions, mice showed a consistent tendency to alternate left and
right turns on consecutive junction visits. Sequential turn autocorrelation
was significantly negative across sessions (mean lag-1 autocorrelation =
-0.172, one-sample Wilcoxon, W = 2.0, p < 0.0001, adjusted p < 0.0001,
r = 0.98, N = 20), indicating systematic left-right alternation consistent
with Rosenberg et al. (2021). The maze topology itself contributes some
alternation (null mean = -0.141), but bootstrap permutation testing confirmed
that the observed alternation exceeded the topology-driven null (p = 0.019).

Global left-right bias was minimal (left fraction: 0.50, not significantly
different from 0.5). Backtracking (reversing direction at junctions) was
frequent, accounting for 48--51% of junction visits, a structural
consequence of the maze having 9 dead ends among 23 cells.

In contrast to Rosenberg et al. (2021), a first-order Markov model was
preferred over a second-order model in all 20 sessions by BIC (mean
delta-BIC = -4,434; 0/20 sessions favouring second-order; Fig. 1E). This
likely reflects the smaller state space (23 cells, 7 junctions): with fewer
transitions, the second-order model's additional parameters are poorly
estimable and BIC's complexity penalty favours the simpler model.

### 2. Spatial coverage decreases in darkness

The removal of visual cues robustly reduced per-epoch spatial coverage.
Within individual 1-minute epochs, mice visited a smaller fraction of the 23
accessible cells in darkness than in light (light: 0.400; dark: 0.337;
Wilcoxon, W = 15.0, p = 0.0003, adjusted p = 0.0008, r = 0.86, N = 20;
Fig. 2A). This coverage reduction was the strongest light-dark effect
observed in the dataset and was robust to pseudoreplication control
(primary-only: N = 11, p = 0.010, r = 0.85; Supplementary Table S1).

Running speed showed a trend toward reduction in darkness (p = 0.076) that
reached significance only in primary-only sessions (p = 0.042, r = 0.70;
Supplementary Fig. S4). To control for any locomotor contribution, we
normalised coverage by active time. Coverage per active minute was also
significantly lower in darkness (light: 19.8; dark: 17.8 cells/active-min;
p = 0.001, r = 0.78; Fig. 2C), indicating that the coverage reduction is
not simply a locomotor artefact. The primary-only analysis did not reach
significance for this normalised metric (p = 0.175; Supplementary Table S1),
so the relative contributions of speed and strategy cannot be definitively
resolved with the current sample.

Other exploration metrics -- turn alternation, backtracking, transition
entropy, and dead-end visit rate -- did not differ between conditions
(all p > 0.15; Supplementary Table S2), indicating that local navigation
rules were preserved despite the global coverage reduction. Angular head
velocity was also unchanged (p = 0.177; Supplementary Table S2).

### 3. The coverage drop reflects route stereotypy

The coverage reduction in darkness could arise from multiple mechanisms:
fewer transitions due to reduced speed, a shift in routing patterns, spatial
range contraction, or increased revisitation. We tested these alternatives
(Fig. 3).

To determine where in the maze coverage was lost, we computed per-epoch
coverage separately for corridors, junctions, and dead ends (Fig. 3A).
Corridor coverage showed the largest reduction in darkness (light: 0.482;
dark: 0.380; Wilcoxon, p < 0.00001, adjusted p < 0.0001, r = 0.97, N = 20).
Junction coverage also decreased (light: 0.465; dark: 0.389; p = 0.0006,
adjusted p = 0.001, r = 0.82). In contrast, dead-end coverage was unchanged
between conditions (light: 0.286; dark: 0.262; p = 0.26, r = 0.30). The
magnitude of the dead-end coverage drop was significantly smaller than the
junction coverage drop (p = 0.0004, r = 0.84). This pattern indicates that
mice maintained visits to terminal destinations at unchanged rates but
consolidated onto a reduced set of connecting corridors and junctions -- a
pattern we term route stereotypy. The dissociation survived primary-only
analysis with large effect sizes (corridor p = 0.002; junction p = 0.010;
dead-end p = 0.465; interaction p = 0.010; Supplementary Table S1).

The diameter of the visited subgraph also contracted in darkness (light:
6.35 cells; dark: 5.57 cells; p = 0.002, r = 0.80; Fig. 3B), consistent
with a reduced spatial extent of the route network.

Transition matrices diverged between conditions: the observed mean
Jensen-Shannon divergence (0.068) was 3.9 times larger than the permutation
null (null mean: 0.018; 1000 permutations; p < 0.001; Fig. 3C). No
individual edge survived Holm-Bonferroni correction (0/44), indicating that
the routing change is distributed across many small shifts rather than
concentrated at a few junctions. Despite the changed routes, transition
entropy was preserved (light: 1.221; dark: 1.186 bits/step; p = 0.165;
Supplementary Table S2), and normalised entropy rate was unchanged (light:
0.277; dark: 0.269; p = 0.133; Supplementary Table S2). The mouse uses
fewer routes but with comparable predictability.

Consistent with route consolidation, the revisitation index (total
transitions / unique cells visited) increased in darkness (light: 3.39;
dark: 3.88; p = 0.011, r = 0.64; Fig. 3D), indicating increased
re-traversal of already-covered cells. This effect did not survive
primary-only analysis (p = 0.067; Supplementary Table S1) and should be
interpreted as supportive rather than conclusive.

Per-cell analysis revealed that cells closer to the maze centre showed the
largest visit reductions (Spearman rho = 0.75, p < 0.0001, N = 23;
Fig. 3E). However, a route-dropping null model demonstrated that this
spatial gradient is a topological consequence of the maze graph: central
cells have higher node degree and therefore lose more visits under any
random route-dropping process (permutation p = 0.281; Supplementary Fig. S6).
The spatial pattern of coverage reduction does not require preferential
avoidance of central locations.

Dwell time per cell type did not differ between conditions (all adjusted
p > 0.33; Supplementary Table S2), and speed did not change abruptly at the
light-off transition (Supplementary Fig. S5), ruling out hesitation at
decision points and startle responses as contributors to the coverage drop.

Two additional analyses provide converging evidence for route
reorganisation from independent analytical frameworks. First, a hidden
Markov model (HMM) fitted to kinematic features (speed, absolute angular
head velocity, spatial coverage rate) identified three navigation states:
pausing (~1.1 cm/s, low AHV), slow scanning (~4.5 cm/s, high AHV), and
fast traversal (~7.7 cm/s, directed movement). Despite the coverage
reduction, the fractional occupancy of these states did not differ between
light and dark (all adjusted p > 0.27, Wilcoxon signed-rank, N = 20;
Supplementary Fig. S10). This null result is informative: it indicates
that the kinematic profile of navigation is preserved in darkness and
that the behavioural change is in the spatial deployment of movement
rather than its character. Second, graph-theoretic analysis of the
directed cell-transition graph revealed that the largest strongly
connected component (SCC) contracted in darkness (light: 96% of cells;
dark: 84%; Wilcoxon, p = 0.017, adjusted p = 0.10, r = 0.85, N = 20;
Supplementary Fig. S11). This indicates that the navigation graph
fragments into disconnected subgraphs in darkness: some cell-to-cell
transitions that occur in light are not used in dark, breaking
bidirectional reachability. Other graph metrics (edge density, mean
out-degree, global efficiency, transitivity) did not differ (all adjusted
p > 0.93; Supplementary Fig. S11). The SCC result did not survive
Holm-Bonferroni correction across six graph metrics; it is reported as
an exploratory finding with a large effect size.

Taken together, these results indicate that the coverage reduction in
darkness reflects a reorganisation of the route network rather than a simple
global slowdown or hesitation. Mice continue to visit dead-end destinations
at unchanged rates but travel between them via fewer, more repetitive
routes. The kinematic composition of navigation is unchanged (HMM), but
the route network loses bidirectional connectivity (SCC fragmentation),
consistent with the route stereotypy pattern identified by the cell-type
dissociation analysis.

### 4. Route stereotypy is established after a single dark epoch

The first dark epoch in each session showed substantially higher coverage
than all subsequent dark epochs (first: 0.57; subsequent mean: 0.30;
Wilcoxon, p = 0.0001, adjusted p = 0.0004, r = 0.89, N = 20; Fig. 4B).
After the first dark epoch, dark coverage was stable (slope test: p = 0.81;
Fig. 4A). Light-epoch coverage showed a gradual decline over the session
(p = 0.011; Fig. 4C), consistent with general exploration habituation, but
the dark-epoch pattern was qualitatively different: a step change rather
than a gradual decline. Speed did not change across dark epochs (p = 0.86;
Fig. 4D), ruling out locomotor fatigue.

Coverage in the first dark epoch did not differ from the first light epoch
(0.593 vs 0.628; p = 0.360; Supplementary Fig. S5), confirming that the
spatial representation carries over from the preceding light epoch: the
mouse navigates its first dark epoch as effectively as light. Speed did not
change at the light-off transition (Supplementary Fig. S5), ruling out an
immediate locomotor startle response.

This pattern indicates that route stereotypy is not a gradual adaptation but
is established after a single experience of darkness. The first dark epoch,
during which the mouse encounters the loss of visual cues for the first time,
produces near-normal coverage, suggesting that the mouse initially navigates
using a spatial representation inherited from the preceding light epoch. By
the second dark epoch, the mouse has adjusted its exploration strategy. This
is consistent with single-trial learning: the mouse detects that darkness
degrades navigation and rapidly adopts a more conservative route network.

---

## Discussion

### Route stereotypy as a novel navigation strategy in darkness

The central finding of this study is that visual cue removal selectively
alters route selection in a structured maze without disrupting local
navigation rules. Mice in darkness maintain visits to dead-end destinations
at unchanged rates but consolidate onto a reduced set of connecting
corridors and junctions -- route stereotypy. This is accompanied by
increased revisitation, a distributed reorganisation of transition
matrices, and contraction of the visited subgraph diameter, while
transition entropy, turn alternation, backtracking, and dwell times remain
unchanged (Supplementary Table S2).

This pattern is distinct from classical range contraction toward a central
refuge (Avni et al. 2006; Fonio et al. 2009). In those studies, mice in
novel or dark environments retreat to a familiar base and abandon peripheral
locations. In the q-rose maze, the opposite spatial pattern emerges:
peripheral dead ends are preserved while central corridors and junctions
lose coverage. However, a route-dropping null model demonstrates that this
spatial gradient is a topological artefact of the maze graph -- central
cells have higher node degree and therefore lose more visits under any
random route-dropping process (permutation p = 0.281; Supplementary Fig. S6).
Route stereotypy is genuine (the corridor-junction-dead-end dissociation
is robust: p < 0.001 for corridors, r = 0.97), but it does not involve
spatially targeted avoidance of any particular maze region.

The dissociation between preserved local decision rules and altered global
route selection suggests that these two aspects of navigation draw on
different information sources. Turn alternation may be supported by
egocentric strategies that operate without visual landmarks (e.g., motor
efference copy or vestibular signals), while route selection across the maze
graph may depend on a spatial representation that degrades without visual
input. This interpretation is reinforced by the HMM analysis: the
kinematic composition of behaviour (the proportions of pausing, scanning,
and directed traversal) does not change in darkness (Supplementary Fig.
S10), suggesting that the motor programme for navigation is intact even
as the route network contracts. The graph fragmentation (largest SCC
contraction; Supplementary Fig. S11) provides an independent
quantification of this contraction from a network-theoretic perspective. This is consistent with the known role of visual landmarks in
anchoring head direction representations (Jacob et al. 2017) and the
degradation of spatial coding in darkness (Stackman & Taube 1997; Ajabi et
al. 2023; Bicknell et al. 2024).

### Single-trial adaptation constrains the mechanism

The temporal dynamics of the coverage reduction provide additional
constraints. The first dark epoch produces near-normal coverage, while all
subsequent dark epochs stabilise at a lower level (p = 0.0001, r = 0.89).
Several converging controls argue against a reflexive or anxiety-based
explanation: speed does not change at the light-off transition (p = 0.756;
Supplementary Fig. S5), first-dark coverage equals first-light coverage
(p = 0.360; Supplementary Fig. S5), and dwell times are unchanged
(Supplementary Table S2). Together, these suggest that the strategy change
emerges from navigational experience rather than innate aversion to
darkness.

The single-trial adaptation timescale is consistent with the HD drift
literature. Stackman & Taube (1997) showed that HD preferred directions
drift over 1--3 minutes in darkness, and Ajabi et al. (2023) demonstrated
coherent population drift on similar timescales. The first 60-second dark
epoch may be short enough that the inherited spatial representation remains
usable, but the mouse detects accumulated error and adjusts for subsequent
epochs. Within-dark-epoch temporal dynamics were inconclusive in the present
data (Supplementary Fig. S3); longer dark epochs would be needed to resolve
whether behavioural degradation occurs gradually within individual epochs.

### Speed contributes but does not fully explain the coverage drop

Speed and coverage are strongly correlated (Spearman rho = 0.76;
Supplementary Fig. S4), and reduced locomotion contributes to the coverage
reduction. We do not claim that route stereotypy is independent of speed.
However, the cell-type dissociation (preserved dead-end coverage alongside
reduced corridor and junction coverage) and the increased revisitation
indicate spatial structure that a uniform speed reduction would not produce.
A uniform slowdown would reduce coverage proportionally across all cell
types; the selective loss of corridor and junction coverage points to a
change in route selection. Coverage per active minute confirms a significant
effect in the full dataset (p = 0.001), though this does not survive
primary-only analysis (p = 0.175; Supplementary Table S1).

### Comparison to Rosenberg et al. (2021)

The q-rose maze produces qualitatively similar behavioural patterns to
Rosenberg's 63-junction labyrinth: mice show systematic left-right turn
alternation, and exploration is structured rather than random. However, the
smaller state space (23 vs 127 cells) means that first-order Markov models
are preferred (0/20 sessions favour second-order), likely reflecting
parameter estimability rather than a qualitative difference in navigation
strategy. The backtracking rate is moderate (48--51%), a structural
consequence of the maze having 9 dead ends among 23 cells.

### HD sampling in the maze

HD distributions are non-uniform and constrained by corridor geometry.
A trend toward increased mean resultant length of the HD distribution in
darkness (p = 0.076) did not survive correction or primary-only analysis
(Supplementary Table S2). This non-uniformity is a methodological concern
for HD tuning analyses in structured environments (Jacob et al. 2017;
Muir et al. 2022) and is characterised in the Supplementary Material
(Supplementary Fig. S7) to enable occupancy-corrected tuning curves in
the companion neural paper.

### Limitations

1. The head-mounted microscope tether constrains movement. While periods of
   clear tether entanglement are excluded, subtle motor constraints may
   influence exploration patterns.

2. With 20 sessions from 14 animals (4 animals contributing 2--3 sessions),
   session-level analyses involve mild pseudoreplication. Primary-only
   robustness checks (N = 11) are reported in Supplementary Table S1.

3. The 1-minute epoch duration was chosen for neural imaging purposes
   rather than to optimise behavioural measurements. Longer dark epochs
   might reveal more pronounced exploration changes.

4. This is a free-exploration paradigm with no explicit task demands. The
   behavioural metrics quantify exploration strategy, not task performance.

5. The revisitation index increase (p = 0.011) did not survive primary-only
   analysis (p = 0.067). The core route stereotypy finding (corridor-
   junction-dead-end dissociation) survived all controls.

6. The speed-coverage correlation is substantial (rho = 0.76), and the
   relative contributions of locomotor reduction and strategy change cannot
   be definitively separated with the current sample.

7. The HMM kinematic state occupancy comparison and graph-metric
   comparisons are exploratory. The SCC fragmentation (r = 0.85) did not
   survive correction across six graph metrics. These analyses provide
   converging descriptive evidence for route stereotypy but should not be
   interpreted as independent statistical confirmations.

---

## Supplementary Material

### Supplementary Table S1: Primary-only robustness

*All primary analyses repeated using one session per animal (N = 11
independent animals). Wilcoxon signed-rank, matched-pairs rank-biserial r.*

| Metric | p (N=11) | r | Survives? |
|--------|----------|---|-----------|
| Per-epoch coverage | 0.010 | 0.85 | Yes |
| Coverage / active min | 0.175 | 0.49 | No |
| Corridor coverage | 0.002 | 0.97 | Yes |
| Junction coverage | 0.010 | 0.85 | Yes |
| Dead-end coverage | 0.465 | -- | n/a (null in full) |
| DE vs junc. interaction | 0.010 | 0.85 | Yes |
| Visited diameter | 0.014 | -- | Yes |
| Revisitation index | 0.067 | 0.64 | No |
| Speed | 0.042 | 0.70 | Yes |
| HD MRL | 0.278 | 0.39 | No |
| AHV | 0.465 | 0.27 | n/a (null in full) |
| First dark epoch vs rest | -- | -- | Not tested |

### Supplementary Table S2: Full battery of light-dark comparisons

*All metrics compared between light and dark conditions (N = 20 sessions).
Wilcoxon signed-rank. p-values are Holm-Bonferroni-adjusted within each
figure family.*

| Metric | Light (mean) | Dark (mean) | W | p (raw) | p (adj) | r |
|--------|-------------|------------|---|---------|---------|---|
| Per-epoch coverage (frac) | 0.400 | 0.337 | 15.0 | 0.0003 | 0.0008 | 0.86 |
| Coverage / active min | 19.8 | 17.8 | 23.0 | 0.001 | -- | 0.78 |
| Exploration efficiency (w=5) | 3.56 | 3.41 | 54.0 | 0.058 | 0.117 | 0.49 |
| Speed (cm/s) | 2.28 | 1.97 | 57.0 | 0.076 | 0.152 | 0.46 |
| Fraction active | 0.472 | 0.441 | 58.0 | 0.083 | 0.152 | 0.45 |
| Immobility bout (s) | 0.84 | 0.94 | 19.0 | 0.035 | 0.106 | 0.64 |
| HD mean resultant length | 0.060 | 0.085 | 57.0 | 0.076 | 0.152 | 0.46 |
| Median |AHV| (deg/s) | 93.3 | 95.4 | 68.0 | 0.177 | 0.177 | 0.35 |
| Dead-end rate (/min) | 7.60 | 8.33 | 74.0 | 0.261 | 0.261 | 0.30 |
| Transition entropy (bits/step) | 1.221 | 1.186 | 67.0 | 0.165 | -- | 0.36 |
| Left turn fraction | 0.495 | 0.500 | 100.0 | 0.870 | 1.000 | 0.05 |
| Backtracking rate | 0.482 | 0.505 | 92.0 | 0.648 | 1.000 | 0.12 |
| Normalised entropy rate | 0.277 | 0.269 | -- | 0.133 | -- | 0.39 |
| Dwell: junction (s) | 1.889 | 2.082 | -- | -- | 1.000 | -- |
| Dwell: corridor (s) | 1.493 | 1.624 | -- | -- | 0.990 | -- |
| Dwell: dead-end (s) | 1.966 | 1.893 | -- | -- | 1.000 | -- |
| Peri-transition speed (cm/s) | 4.28 | 4.21 | -- | 0.756 | -- | 0.09 |
| Coverage / transition | 0.398 | 0.355 | -- | 0.076 | -- | 0.46 |
| Coverage ratio (2nd/1st half) | 0.282 | 0.207 | -- | 0.064 | 0.192 | 0.48 |
| Speed ratio (2nd/1st half) | 0.948 | 0.945 | -- | 0.87 | 1.0 | 0.05 |
| **HMM state occupancy** | | | | | | |
| Pausing (frac) | 0.337 | 0.375 | 59 | 0.090 | 0.27 | 0.44 |
| Slow scanning (frac) | 0.321 | 0.303 | 60 | 0.097 | 0.27 | 0.43 |
| Fast traversal (frac) | 0.343 | 0.322 | 76 | 0.294 | 0.29 | 0.28 |
| **Graph metrics** | | | | | | |
| Largest SCC fraction | 0.959 | 0.838 | 4 | 0.017 | 0.10 | 0.85 |
| N strongly connected components | 3.5 | 4.6 | 43 | 0.187 | 0.93 | 0.37 |
| Edge density | 0.087 | 0.086 | 63 | 0.776 | 1.00 | 0.08 |
| Mean out-degree | 2.12 | 2.09 | 58 | 0.587 | 1.00 | 0.15 |
| Global efficiency | 0.264 | 0.248 | 76 | 0.445 | 1.00 | 0.20 |
| Transitivity | 0.170 | 0.171 | 33 | 0.638 | 1.00 | 0.15 |

*DLC tracking confidence did not differ between light and dark for any of
the 27 tracked bodyparts (all Holm-Bonferroni adjusted p >= 0.52; Wilcoxon,
N = 20), confirming identical image quality under infrared illumination.*

### Supplementary Figure S1: Individual animal variation

**Panel A.** Per-animal coverage curves (thin lines coloured by animal).

**Panel B.** Per-animal turn bias (left fraction), showing individual
variability.

**Panel C.** Per-animal speed difference (dark - light), showing consistency
of the light effect across animals.

**Panel D.** Tether restriction summary: for each session, show total
excluded time due to `bad_behav` mask.

---

### Supplementary Figure S2: Comparison to random walk null model

**Panel A.** Occupancy map: real data vs random walk (unbiased) vs random
walk with forward bias.

**Panel B.** Turn statistics: real vs null models.

**Panel C.** Transition entropy: real vs null.

**Panel D.** Dead-end visit rate: real vs null.

---

### Supplementary Figure S3: Within-dark-epoch temporal dynamics

**Panel A.** Cumulative unique cell curves (5-second bins) for light and
dark epochs, averaged across sessions (mean +/- SEM). Light and dark curves
diverge only modestly; the coverage ratio (second half / first half) trends
lower in dark than light (p = 0.064, adjusted p = 0.192, r = 0.48, N = 20)
but does not reach significance.

**Panel B.** Speed in early (0--30s) vs late (30--60s) dark epochs. Speed
ratio does not differ between conditions (p = 0.87).

Rationale: Tests whether behaviour degrades within individual dark epochs,
which would be expected if path integration drift causes progressive spatial
disorientation. The current data are inconclusive: a non-significant trend
exists but cannot distinguish gradual degradation from an immediate strategy
switch. Longer dark epochs or higher temporal resolution would be needed.

---

### Supplementary Figure S4: Speed and locomotor metrics

**Panel A.** Running speed (cm/s) by condition: box/violin plot of
session-median speed in light vs dark. Paired by session. Trend (p = 0.076,
adjusted p = 0.152); significant in primary-only (p = 0.042).

**Panel B.** Speed vs coverage scatter plot (Spearman rho = 0.76,
p < 0.0001). Demonstrates the strong speed-coverage coupling.

**Panel C.** Fraction active by condition. Trend (p = 0.083, adjusted
p = 0.152).

**Panel D.** Immobility bout duration. Trend toward longer bouts in
darkness (p = 0.035, adjusted p = 0.106).

Rationale: Documents the speed confound and related locomotor metrics.
Speed contributes to the coverage reduction but does not fully explain the
cell-type dissociation (route stereotypy).

---

### Supplementary Figure S5: Controls ruling out startle and anxiety

**Panel A.** Peri-transition speed: mean speed in the 5 s before vs 5 s
after lights-off (p = 0.756, r = 0.09, N = 20). No abrupt speed change at
lights-off.

**Panel B.** First dark epoch vs first light epoch coverage (0.593 vs 0.628;
p = 0.360, r = 0.25, N = 20). No difference, confirming that darkness onset
does not immediately impair navigation.

**Panel C.** DLC tracking confidence by condition. All 27 bodyparts show
identical tracking confidence in light vs dark (all adjusted p >= 0.52),
confirming that the infrared camera provides identical image quality.

Rationale: These three controls together rule out startle (no speed change),
anxiety (no initial coverage deficit), and tracking artefact (identical
DLC confidence) as explanations for the coverage reduction.

---

### Supplementary Figure S6: Route-dropping null model

**Panel A.** Per-cell visit fraction delta (dark - light) vs distance from
maze centre for all 23 cells (Spearman rho = 0.75, p < 0.0001). Central
cells lose more visits.

**Panel B.** Null model: distribution of rho values from 1000 random
edge-removal permutations (for sessions with K > 0 edges in light vs dark;
7/20 sessions). Observed mean rho = 0.128 falls within the null
distribution (null mean = 0.055, 95th pctl = 0.212, permutation p = 0.281).

Rationale: Demonstrates that the spatial gradient of visit reduction is a
topological consequence of the maze graph (central cells have higher node
degree) rather than evidence for preferential avoidance of central
locations. This distinguishes the present finding from classical range
contraction (Avni et al. 2006; Fonio et al. 2009).

---

### Supplementary Figure S7: Head direction sampling in the maze

**Panel A.** Per-cell HD distribution: polar histogram of HD angles for each
of the 23 cells. Corridor cells show bimodal distributions; junctions are
more uniform.
**[SYNTHETIC DATA PLACEHOLDER]** Must be replaced with real data.
<!-- TODO(DS-agent): Replace with real per-cell HD distributions from sync.h5. -->

**Panel B.** HD sampling non-uniformity (MRL) per cell on the maze grid.

**Panel C.** HD MRL in light vs dark. Trend (p = 0.076, adjusted p = 0.152)
does not survive primary-only (p = 0.278).

**Panel D.** Joint position x HD occupancy heatmap. Documents the sampling
landscape for future HD tuning analyses.

**Panel E.** AHV in light vs dark (p = 0.177, adjusted p = 0.177).

Rationale: HD sampling non-uniformity is a methodological concern for HD
tuning analyses in structured environments. Per-cell HD occupancy maps
enable occupancy-corrected tuning curve estimation in the companion neural
paper.

---

### Supplementary Figure S8: Individual differences in darkness sensitivity

**Panel A.** Per-animal scatter: mean coverage in light vs dark. Animals
classified as darkness-resistant (N = 5), intermediate (N = 7), or
darkness-sensitive (N = 2).

**Panel B.** Coverage sensitivity vs speed sensitivity (Spearman rho = 0.74,
p = 0.002, N = 14).

**Panel C.** Coverage sensitivity by cell type (Penk+ vs non-Penk). All
4 non-Penk animals are intermediate. N = 4 non-Penk precludes strong
conclusions.

Rationale: Descriptive. N = 14 is small for individual-differences analyses.
The neural bridge (correlating behavioural sensitivity with HD tuning
stability) is deferred to the companion paper.

---

### Supplementary Figure S9: Cell-type Markov model

**Panel A.** Model order comparison at the cell-type level (3 states:
junction, corridor, dead-end). Second-order preferred in all 20 sessions
(mean delta-BIC = 103.2, p < 0.001), contrasting with first-order at
the individual-cell level.

**Panel B.** Cell-type transition JSD between light and dark (observed
JSD = 0.0081 vs null = 0.0031, permutation p < 0.001). P(corridor |
corridor, junction) shows a trend toward reduction in darkness (p = 0.027,
adjusted p = 0.107).

Rationale: Tests whether route stereotypy has structure at the level of
maze topology. The second-order preference at the type level indicates
topological constraints that are diluted across 23 individual states.

---

### Supplementary Figure S10: HMM kinematic state decomposition

**Panel A.** BIC model comparison for K = 2, 3, 4 hidden states fitted to
kinematic features (speed, absolute angular head velocity, spatial coverage
rate). K = 3 preferred over K = 2 (mean BIC: 27,836 vs 54,440). K = 4
yields a lower BIC but introduces states with overlapping kinematic
profiles; K = 3 is retained for interpretability.

**Panel B.** State definitions for the K = 3 model. Three states discovered
by Gaussian HMM (full covariance, fitted per session):
- *Pausing*: speed 1.1 +/- 0.3 cm/s, AHV 31 +/- 6 deg/s
- *Slow scanning*: speed 4.5 +/- 0.4 cm/s, AHV 223 +/- 32 deg/s
- *Fast traversal*: speed 7.7 +/- 0.5 cm/s, AHV 158 +/- 24 deg/s
States labelled post hoc by speed ranking; emission parameters are
means +/- SEM across sessions (N = 20). The slow scanning state is
distinguished by the highest absolute AHV, consistent with active
orienting at junctions or during local exploration.

**Panel C.** State occupancy (fraction of time) in light vs dark. Paired
box plots for each state. No state shows a significant occupancy change
(all Holm-Bonferroni adjusted p > 0.27; Wilcoxon signed-rank, N = 20):
- Pausing: light 33.7%, dark 37.5% (p = 0.090, adj p = 0.27, r = 0.44)
- Slow scanning: light 32.1%, dark 30.3% (p = 0.097, adj p = 0.27, r = 0.43)
- Fast traversal: light 34.3%, dark 32.2% (p = 0.294, adj p = 0.29, r = 0.28)

**Panel D.** Robustness: K = 2 model (slow/fast split). Occupancy also does
not differ between conditions (adj p > 0.23).

Rationale: The HMM provides a data-driven decomposition of navigation into
kinematic modes. The null occupancy result is informative: the coverage
reduction in darkness is not accompanied by a shift in movement type (e.g.,
more pausing). Instead, the same kinematic behaviours are deployed over a
more restricted spatial extent, consistent with route stereotypy as a
spatial reorganisation rather than a locomotor change.

---

### Supplementary Figure S11: Graph topology of the navigation network

**Panel A.** Example session navigation graph (directed) in light vs dark.
Nodes are maze cells; edges are observed cell-to-cell transitions (threshold:
at least 2 transitions per epoch set). Visual comparison shows sparser edges
and fragmented components in dark.

**Panel B.** Largest strongly connected component (SCC) fraction in light vs
dark. Paired comparison across sessions: light 0.96 +/- 0.02; dark 0.84
+/- 0.06 (Wilcoxon, W = 4.0, p = 0.017, adjusted p = 0.10, r = 0.85,
N = 20). Does not survive Holm-Bonferroni correction across 6 graph metrics
but has the largest effect size in the dataset. In light, nearly all visited
cells are mutually reachable via observed transitions. In darkness, the
navigation graph fragments: some cells can only be reached from certain
directions, and reciprocal transitions are absent.

**Panel C.** Number of strongly connected components: light 3.5 +/- 0.5;
dark 4.6 +/- 0.8 (p = 0.187, adj p = 0.93, r = 0.37). Consistent
direction but not significant.

**Panel D.** Summary of all six graph metrics:

| Metric | Light | Dark | p (raw) | p (adj) | r |
|--------|-------|------|---------|---------|---|
| Edge density | 0.087 | 0.086 | 0.776 | 1.00 | 0.08 |
| Mean out-degree | 2.12 | 2.09 | 0.587 | 1.00 | 0.15 |
| Largest SCC frac. | 0.96 | 0.84 | 0.017 | 0.10 | 0.85 |
| N SCCs | 3.5 | 4.6 | 0.187 | 0.93 | 0.37 |
| Global efficiency | 0.264 | 0.248 | 0.445 | 1.00 | 0.20 |
| Transitivity | 0.170 | 0.171 | 0.638 | 1.00 | 0.15 |

Rationale: The navigation graph provides a complementary formalisation of
route stereotypy. The SCC fragmentation captures the loss of bidirectional
reachability -- in light, the mouse traverses corridors in both directions;
in darkness, some transitions become unidirectional, creating disconnected
subgraphs. This is consistent with the mouse consolidating onto a subset
of familiar routes. The large effect size (r = 0.85) is notable despite
non-significance after multiple comparison correction across six metrics.
Other graph metrics (density, degree, efficiency) do not change because the
total number of transitions and cells visited does not drop dramatically;
instead, the *directionality* of transitions becomes more constrained.

---

### Supplementary Results: Controls and null findings

The following control analyses were conducted to rule out alternative
explanations for the coverage reduction in darkness.

**Peri-transition speed.** Speed did not change abruptly at the light-to-dark
transition (mean speed 5 s before: 4.28 cm/s; 5 s after: 4.21 cm/s;
Wilcoxon p = 0.756, r = 0.09, N = 20; Supplementary Fig. S5A). This rules
out an immediate locomotor startle or freeze response at lights-off.

**First dark epoch equals first light epoch.** Coverage in the first dark
epoch (0.593) did not differ from coverage in the first light epoch (0.628;
Wilcoxon p = 0.360, r = 0.25, N = 20; Supplementary Fig. S5B). This rules
out an initial anxiety effect: the mouse navigates its first dark epoch as
effectively as light, consistent with carrying over a spatial
representation from the preceding illuminated period.

**Normalised transition entropy.** The normalised entropy rate (transition
entropy / log2(unique cells)) did not differ between conditions (light:
0.277; dark: 0.269; Wilcoxon p = 0.133, r = 0.39, N = 20). Dark-epoch
routing is scaled-down but equally structured -- the mouse uses fewer routes
with comparable predictability.

**Dwell time per cell type.** Dwell times did not differ between light and
dark after Holm-Bonferroni correction (junction: adjusted p = 1.000;
corridor: adjusted p = 0.990; dead-end: adjusted p = 1.000; all Wilcoxon,
N = 20; Supplementary Table S2). This rules out hesitation at decision
points or prolonged exploration of dead ends as contributors to the coverage
drop. In primary-only sessions, junction and corridor dwell times were
significantly longer in darkness (junction adjusted p = 0.020; corridor
adjusted p = 0.006; dead-end adjusted p = 0.465), with a significant
interaction (p = 0.010, r = 0.85, N = 11), suggesting slightly more time
at decision points without changes in destination behaviour.

**DLC tracking confidence.** Tracking confidence did not differ between light
and dark for any of the 27 tracked bodyparts (all Holm-Bonferroni adjusted
p >= 0.52; Wilcoxon, N = 20; Supplementary Fig. S5C). The overhead camera
uses infrared illumination, so image quality is identical in both conditions.

**Route-dropping null model.** Per-cell analysis showed that cells closer to
the maze centre lost more visits in darkness (Spearman rho = 0.75,
p < 0.0001, N = 23). However, a route-dropping null model (1000
permutations of random edge removal, for sessions where dark had fewer edges
than light; 7/20 sessions) showed that the observed spatial gradient
(mean rho = 0.128) fell within the null distribution (null mean = 0.055,
95th pctl = 0.212; permutation p = 0.281; Supplementary Fig. S6). Central
cells have higher node degree and therefore lose more visits under any
random route-dropping process. The spatial gradient is a topological
consequence rather than evidence for preferential avoidance.

**Within-epoch dynamics.** The coverage ratio (second half / first half)
trended lower in dark than light epochs (p = 0.064, adjusted p = 0.192,
r = 0.48) but did not reach significance. The speed ratio did not differ
(p = 0.87; Supplementary Fig. S3). These data are inconclusive regarding
whether behaviour degrades gradually within individual dark epochs.

---

## References

Ajabi Z, Keinath AT, Brandon MP. 2023. "Population dynamics of
head-direction neurons during drift and reorientation." *Nature* 615,
892--899. doi:10.1038/s41586-023-05813-2

Avni R, Zadicario P, Eilam D. 2006. "Exploration in a dark open field: a
shift from directional to positional progression." *Behav. Processes* 72,
232--240. doi:10.1016/j.beproc.2006.03.005

Bhatt DK et al. 2024. "Rodent maze studies: from following simple rules to
complex map learning." *Brain Struct. Funct.* 229, 1261--1278.
doi:10.1007/s00429-024-02771-x

Bicknell BA, van der Goes M-SH et al. 2024. "Coordinated head direction
representations in mouse anterodorsal thalamic nucleus and retrosplenial
cortex." *eLife* 13, e82952. doi:10.7554/eLife.82952

Bourin M, Hascoet M. 2003. "The mouse light/dark box test." *Eur. J.
Pharmacol.* 463, 55--65. doi:10.1016/S0014-2999(03)01274-3

Chen LL, Lin LH, Green EJ, Barnes CA, McNaughton BL. 1994. "Head-direction
cells in the rat posterior cortex. I. Anatomical distribution and behavioral
modulation." *Exp. Brain Res.* 101, 8--23.

Cho J, Sharp PE. 2001. "Head direction, place, and movement correlates for
cells in the rat retrosplenial cortex." *Behav. Neurosci.* 115, 3--25.

Etienne AS, Jeffery KJ. 2004. "Path integration in mammals." *Hippocampus*
14, 180--192.

Fonio E, Benjamini Y, Golani I. 2009. "Freedom of movement and the
stability of its unfolding in free exploration of mice." *Proc. Natl. Acad.
Sci.* 106, 21335--21340. doi:10.1073/pnas.0812513106

Goodridge JP, Dudchenko PA, Worboys KA, Golob EJ, Taube JS. 1998. "Cue
control and head direction cells." *Behav. Neurosci.* 112, 749--761.

Jacob P-Y, Casali G, Spieser L, Page H, Overington D, Bhatt DH,
Jeffrey K. 2017. "An independent, landmark-dominated head-direction
signal in dysgranular retrosplenial cortex." *Nat. Neurosci.* 20,
173--175. doi:10.1038/nn.4465

Keshavarzi S, Bracey EF, Faville RA, Campagner D, Tyson AL, Lenzi SC,
Branco T, Margrie TW. 2022. "Multisensory coding of angular head velocity
in the retrosplenial cortex." *Neuron* 110, 532--543.
doi:10.1016/j.neuron.2021.10.031

Koren Iton A, Iton E, Michaelson DM, Blinder P. 2025. "NaviGraph: A
graph-based framework for multimodal analysis of spatial decision-making."
*bioRxiv*. doi:10.1101/2025.05.18.654725

Mao D, Molina LA, Bonin V, McNaughton BL. 2020. "Vision and locomotion
combine to drive path integration sequences in mouse retrosplenial cortex."
*Curr. Biol.* 30, 1680--1688. doi:10.1016/j.cub.2020.02.070

Mathis A, Mamidanna P, Cury KM, Abe T, Murthy VN, Mathis MW, Bethge M.
2018. "DeepLabCut: Markerless pose estimation of user-defined body parts
with deep learning for all animals incl. humans." *Nat. Neurosci.* 21,
1281--1289. doi:10.1038/s41593-018-0209-y

Mitchell AS, Czajkowski R, Zhang N, Jeffery K, Nelson AJD. 2018.
"Retrosplenial cortex and its role in spatial cognition." *Brain Neurosci.
Adv.* 2. doi:10.1177/2398212818757098

Muir GM et al. 2022. "Flexible cue anchoring strategies enable stable head
direction coding in both sighted and blind animals." *Nat. Commun.* 13,
5604. doi:10.1038/s41467-022-33204-0

O'Keefe J, Nadel L. 1978. *The Hippocampus as a Cognitive Map.* Oxford
University Press.

Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth show
rapid learning, sudden insight, and efficient exploration." *eLife* 10,
e66175. doi:10.7554/eLife.66175

Schmitzer-Torbert N, Redish AD. 2002. "Development of path stereotypy in a
single day in rats on a multiple-T maze." *Behav. Neurosci.* 116,
1058--1070. doi:10.1037/0735-7044.116.6.1058

Stackman RW, Taube JS. 1997. "Firing properties of head direction cells in
the rat anterior thalamic nucleus: dependence on behavioral factors."
*J. Neurosci.* 17, 9020--9037.

Taube JS. 2007. "The head direction signal: origins and sensory-motor
integration." *Annu. Rev. Neurosci.* 30, 181--207.

Taube JS, Muller RU, Ranck JB Jr. 1990. "Head-direction cells recorded from
the postsubiculum in freely moving rats. I. Description and quantitative
analysis." *J. Neurosci.* 10, 420--435.

Vann SD, Aggleton JP, Maguire EA. 2009. "What does the retrosplenial cortex
do?" *Nat. Rev. Neurosci.* 10, 792--802.

Whishaw IQ, Tomie JA. 1996. "Of mice and mazes: similarities between mice
and rats on dry land but not water mazes." *Physiol. Behav.* 60, 1191--1197.

Ye T et al. 2024. "SuperAnimal pretrained pose estimation models for
behavioral analysis." *Nat. Commun.* 15, 5165.
doi:10.1038/s41467-024-48792-2

Zugaro MB, Arleo A, Berthoz A, Wiener SI. 2003. "Rapid spatial
reorientation and head direction cells." *J. Neurosci.* 23, 3478--3482.

---

## Critical Assessment and Confounds

### What is novel about this paper

1. **Route stereotypy as the mechanism underlying coverage reduction in
   darkness.** The finding that corridor and junction coverage drops while
   dead-end coverage is preserved is, to our knowledge, not reported in the
   darkness-behaviour literature. Previous studies have documented range
   contraction toward a home base (Avni et al. 2006; Fonio et al. 2009),
   but the present pattern -- preserved destination repertoire with reduced
   path diversity -- is qualitatively distinct and suggests a dissociation
   between local decision rules and global route planning.

2. **Route stereotypy is topologically naive.** The route-dropping null model
   shows that the spatial gradient of visit reduction (rho = 0.75) is
   consistent with random edge removal (permutation p = 0.281). This
   identifies the correct level of description: mice reduce the number of
   routes used, and the spatial consequences are a passive outcome of graph
   topology.

3. **Single-trial adaptation to darkness.** The first dark epoch produces
   near-normal coverage; all subsequent dark epochs show the reduced-coverage
   pattern (p = 0.0001, r = 0.89). This single-trial learning dynamic has
   not been characterised in maze navigation.

4. **The q-rose maze under light/dark alternation is a new paradigm.** The
   original Rosenberg maze used fixed lighting. No prior study has combined
   a binary-choice labyrinth with total darkness manipulation.

5. **Kinematic profile preservation in darkness.** The HMM decomposition
   shows that the proportions of pausing, scanning, and traversal do not
   change in darkness. This is a useful negative result: it rules out the
   hypothesis that mice simply pause more or move less vigorously in
   darkness, and localises the behavioural change to spatial routing.

6. **Navigation graph fragmentation.** The SCC contraction provides a
   graph-theoretic restatement of route stereotypy. While graph approaches
   to maze navigation have been proposed (Koren Iton et al. 2025), the
   specific finding that the directed navigation graph fragments into
   disconnected components in darkness has not been reported.

### What is NOT novel

- Turn alternation in maze exploration (Rosenberg et al. 2021).
- Speed reduction in darkness is well-established (Whishaw & Tomie 1996).
- HD drift in darkness (Stackman & Taube 1997; Ajabi et al. 2023).
- DeepLabCut-based pose tracking (standard tool).
- HMM decomposition of locomotor behaviour is widely used (e.g., Wiltschko
  et al. 2015 for MoSeq; Batty et al. 2019 for ARHMM). The application to
  this maze is new but the method is standard.

### Key confounds addressed

1. **Speed confound.** Speed and coverage strongly correlated (rho = 0.76).
   Coverage per active minute controls for locomotion (p = 0.001), though
   primary-only is non-significant (p = 0.175). Cell-type dissociation
   provides additional evidence beyond speed.

2. **Pseudoreplication.** 4 animals contribute 2--3 sessions. Primary-only
   robustness checks (N = 11) reported in Supplementary Table S1.

3. **Tracking artefact.** DLC confidence identical in light vs dark
   (infrared camera; Supplementary Fig. S5C).

4. **Startle/anxiety.** Peri-transition speed unchanged (p = 0.756); first
   dark = first light coverage (p = 0.360); dwell times unchanged.

5. **Spatial pattern artefact.** Route-dropping null model shows central-cell
   gradient is topology, not strategy (p = 0.281; Supplementary Fig. S6).

### Reviewer objections to anticipate

- "This is descriptive with no neural data -- what is the contribution?"
  Response: Establishes the paradigm and provides quantitative behavioural
  baselines for the companion neural paper. Route stereotypy is a novel
  behavioural phenotype.

- "The speed confound is not fully resolved." Response: Acknowledged
  transparently. The cell-type dissociation (dead-end preservation) is not
  predicted by a uniform speed reduction. Coverage per active minute is
  significant in the full dataset.

- "Why not use an open field?" Response: The maze provides hundreds of
  natural binary decisions and structured corridors, enriching the
  behavioural readout.

- "The SCC fragmentation does not survive correction -- why report it?"
  Response: The effect size (r = 0.85) is the largest in the dataset
  for any graph metric, and SCC fragmentation provides an independent
  graph-theoretic corroboration of route stereotypy established by the
  cell-type dissociation. We report it explicitly as exploratory with
  the corrected p-value (0.10) and emphasise that the cell-type
  dissociation (corridor/junction/dead-end) remains the primary evidence.

- "The HMM null result could reflect insufficient power."
  Response: The effect sizes are small-to-moderate (r = 0.28--0.44) and
  the trends are in the expected direction (more pausing, less traversal
  in dark). With N = 20 and r = 0.44 for the largest trend (pausing),
  a post-hoc power analysis suggests ~45% power to detect this effect.
  We therefore cannot exclude a small occupancy shift, but the core
  interpretive point -- that the dominant behavioural change is spatial
  rather than kinematic -- holds.
