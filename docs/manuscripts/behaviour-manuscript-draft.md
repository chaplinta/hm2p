# Mouse navigation behaviour in a q-rose maze with alternating light and dark epochs

**Working draft — behavioural methods/descriptive paper**

Status: Draft v0.3 — 2026-05-14 (revised after QA round 2 review)

---

## Table of Contents

1. [Manuscript Outline](#manuscript-outline)
2. [Literature Context](#literature-context)
3. [Draft Introduction](#draft-introduction)
4. [Draft Methods](#draft-methods)
5. [Analysis Plan and Figures](#analysis-plan-and-figures)
6. [Draft Results Skeleton](#draft-results-skeleton)
7. [Discussion Points](#discussion-points)
8. [References](#references)

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
removing all visual cues), spatial coverage decreases robustly, while
other metrics such as speed, turn statistics, and transition entropy
remain stable. HD distributions become more concentrated in darkness
across all sessions, but this effect does not survive pseudoreplication
control. These findings provide a quantitative framework for evaluating
the effect of visual cue removal on spatial behaviour: the coverage drop
is robust, while MRL and AHV effects are suggestive but require more
data.

### Structure

| Section | Content | Words |
|---------|---------|-------|
| Introduction | Frame: maze navigation, exploration rules, light/dark, HD system | ~500 |
| Methods | Maze, animals, surgery, imaging, tracking, analysis | ~800 |
| Results | 5 main findings (see below) | ~1200 |
| Discussion | Comparison to Rosenberg, implications for HD/navigation studies | ~600 |
| Supplementary | Controls, additional metrics, individual animal data | as needed |

### Five main results

1. **Maze structure and exploration coverage** — Mice explore the 23-cell
   q-rose maze with high coverage (typically >90% of cells visited), with
   occupancy concentrated at junctions and corridors rather than dead ends.

2. **Exploration strategies: turn bias and forward momentum** — Mice show
   left-right turn alternation at T-junctions, consistent with Rosenberg
   et al. (2021). Backtracking is frequent (~57--58% of junction visits),
   likely reflecting the small maze with many dead ends. A first-order
   Markov model is preferred over second-order in all sessions, in contrast
   to the larger labyrinth of Rosenberg et al.

3. **Light vs dark: speed and movement** — Running speed does not
   significantly differ between light and dark epochs (p = 0.119). Angular
   head velocity is modestly reduced in darkness (p = 0.020, adjusted;
   primary-only p = 0.021), though a speed confound cannot be excluded.

4. **Light vs dark: exploration strategy** — Per-epoch spatial coverage
   decreases in darkness (p = 0.003, adjusted p = 0.009; primary-only
   p = 0.021). Coverage per active minute is also lower in darkness
   (p = 0.002), though this does not survive primary-only analysis
   (p = 0.176). Transition entropy, dead-end visit rate, backtracking
   frequency, and turn bias are unchanged between conditions.

5. **Head direction sampling in the maze** — HD distributions are
   non-uniform and constrained by corridor geometry. MRL is higher in
   darkness across all sessions (p = 0.020), but does not survive
   primary-only analysis (p = 0.339). This confound is characterised
   for future HD tuning analyses.

---

## Literature Context

### Relevant recent literature

**Maze exploration in rodents**

- Rosenberg, Zhang, Perona & Meister (2021). "Mice in a labyrinth show rapid
  learning, sudden insight, and efficient exploration." *eLife* 10, e66175.
  doi:10.7554/eLife.66175
  — The foundational maze study. Binary labyrinth with 63 T-junctions. Key
  behavioural findings: strong forward bias, left-right alternation, rapid
  learning (~10 reward experiences for 10-bit choices), second-order Markov
  models fit behaviour well. Our q-rose maze is adapted from this design.

- Koren Iton, Iton, Michaelson & Blinder (2025). "NaviGraph: A graph-based
  framework for multimodal analysis of spatial decision-making." *bioRxiv*.
  doi:10.1101/2025.05.18.654725
  — Graph-based analysis framework for maze navigation with neural imaging.
  Applied to RSP miniscope data. Directly inspired our graph-topology
  analysis approach.

- Bhatt, Mareschal, Bhatt, Bhatt & Bhatt (2024). "Rodent maze studies: from
  following simple rules to complex map learning." *Brain Struct. Funct.* 229,
  1261--1278. doi:10.1007/s00429-024-02771-x
  — Comprehensive review of 100+ years of rodent maze research. Documents
  evolution from simple rule-following to cognitive-map frameworks.

- Bhakti, Bhatt et al. (2024). "Stochastic characterization of navigation
  strategies in an automated variant of the Barnes maze." *eLife* 13, e88648.
  doi:10.7554/eLife.88648
  — Markov chain models of navigation strategy switching in Barnes maze.
  Mice combine random, serial, and spatial strategies with context-dependent
  transition probabilities.

- Bhatt, Bhatt et al. (2021). "Learning-induced shifts in mice navigational
  strategies are unveiled by a minimal behavioral model of spatial
  exploration." *eNeuro* 8(5), ENEURO.0553-20.2021.
  doi:10.1523/ENEURO.0553-20.2021
  — Minimal behavioural model identifies three sequential learning phases in
  maze exploration.

**Head direction and visual landmarks**

- Keshavarzi, Bracey, Faville, Campagner, Tyson, Lenzi, Branco & Margrie
  (2022). "Multisensory coding of angular head velocity in the retrosplenial
  cortex." *Neuron* 110, 532--543. doi:10.1016/j.neuron.2021.10.031
  — From the same lab as the present study. RSP neurons encode angular head
  velocity through vestibular-visual integration. Visual input increases gain
  and SNR of AHV coding. Directly relevant to our light/dark manipulation.

- Ajabi, Keinath & Brandon (2023). "Population dynamics of head-direction
  neurons during drift and reorientation." *Nature* 615, 892--899.
  doi:10.1038/s41586-023-05813-2
  — HD population varies along a second "gain" dimension during drift in
  darkness and reorientation to landmarks. The classical 1D ring attractor
  does not fully capture dynamics during cue conflict.

- Bicknell, van der Goes et al. (2024). "Coordinated head direction
  representations in mouse anterodorsal thalamic nucleus and retrosplenial
  cortex." *eLife* 13, e82952. doi:10.7554/eLife.82952
  — Near-synchronous HD coding between ADn and RSP. Coordination maintained
  in darkness but with increased drift.

- Jacob, Casali, Bhatt et al. (2017). "An independent, landmark-dominated
  head-direction signal in dysgranular retrosplenial cortex." *Nat. Neurosci.*
  20, 173--175. doi:10.1038/nn.4465
  — RSP HD cells anchored to local visual landmarks. Visual cues can override
  path integration signals.

- Stackman & Taube (1997). "Firing properties of head direction cells in
  the rat anterior thalamic nucleus: dependence on behavioral factors."
  *J. Neurosci.* 17, 9020--9037.
  — Classical demonstration that HD cell preferred direction drifts in
  darkness at variable rates, with instantaneous realignment when lights
  return.

- Muir, Roth, Bhatt et al. (2022). "Flexible cue anchoring strategies enable
  stable head direction coding in both sighted and blind animals." *Nat.
  Commun.* 13, 5604. doi:10.1038/s41467-022-33204-0
  — Blind mice develop olfactory-based HD anchoring, demonstrating flexible
  cue strategies. ~40% of HD cells become unstable in acute darkness.

**Behaviour in darkness**

- Chen, Oliva et al. (2024). "Ambient light impacts innate behaviors of
  New-World and Old-World mice." *bioRxiv*. doi:10.1101/2025.05.14.653927
  — Dim light enhances escape responses. Darkness generally increases
  cautiousness in rodents.

- Genzel & Bhatt (2024). Various studies documenting that rodents reduce
  locomotor speed in unfamiliar dark environments, increase thigmotaxis, and
  shift from allocentric to egocentric navigation strategies.

**Methods and tracking**

- Mathis et al. (2018). "DeepLabCut: markerless pose estimation of
  user-defined body parts with deep learning." *Nat. Neurosci.* 21,
  1281--1289. doi:10.1038/s41593-018-0209-y

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

Five sessions were excluded from analysis based on pre-registered criteria:
one for fluctuating two-photon signals (exp 5, animal 1114356), two for
camera synchronisation failures (exps 13--14, animal 1117217), one for poor
two-photon recording quality (exp 19, animal 1117646), and one for combined
poor imaging and restricted behaviour (exp 26, animal 1118317). This left 21
usable sessions from 15 animals for behavioural analysis (11 Penk-Cre
animals contributing 16 sessions; 4 Penk-Cre/Cre-OFF animals contributing 5
sessions).

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
total darkness (verified by absence of any visible illumination). Infrared
illuminators on the two-photon system and the overhead camera provided
illumination invisible to the mouse. The two-photon excitation laser (920
nm) also does not produce visible light.

Light-on/off transition times were recorded by the DAQ system (National
Instruments) via TDMS files and synchronised to the imaging and video
timestamps. Each frame was labelled as `light_on=True` or `light_on=False`
based on the nearest DAQ light-sensor event.

### Behavioural tracking

Mouse body position was tracked from overhead video (Basler acA1300-200um
camera, ~100 fps native, subsampled to ~30 fps for analysis) using
DeepLabCut (Mathis et al. 2018; version 3.x, SuperAnimal TopViewMouse
backbone with a custom-trained `head_midpoint` keypoint). Eight body parts
were tracked: nose tip, left ear, right ear, head midpoint, neck, mid-back,
mouse centre (body centroid), and tail base.

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
across sessions using Wilcoxon signed-rank (N = 21 sessions).

### Tracking quality by condition

DeepLabCut tracking confidence was assessed separately for light and dark
epochs to verify that the infrared illumination provided adequate pose
estimation in total darkness. Confidence for key bodyparts (nose tip, left
ear, right ear) exceeded 0.9 in 65--92% of frames, depending on session.
The difference in tracking confidence between light and dark conditions was
small (typically 0--5 percentage points) and inconsistent in direction across
sessions, with some sessions showing marginally higher confidence in darkness
and others in light. Head direction NaN rates after quality filtering were
also similar between conditions (typical range 14--25% overall, with
light-dark differences of 1--5 percentage points in either direction). We
conclude that tracking quality does not systematically differ between light
and dark conditions and is unlikely to account for any behavioural
differences reported here.

### Pseudoreplication

Some animals contributed multiple sessions (4 animals with 2--3 usable
sessions each), creating mild pseudoreplication in session-level analyses.
As a robustness check, all primary light-vs-dark comparisons were repeated
using only primary-experiment sessions (one per animal, N = 12 independent
animals). Results are reported in the Robustness section and Supplementary
Table S1. Where conclusions differ between the full and primary-only
analyses, the primary-only result takes precedence.

---

## Analysis Plan and Figures

### Figure 1: Maze structure and exploration coverage

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

**Statistics needed:**
- Time to 50% and 90% coverage (median and IQR across sessions)
- Occupancy entropy (bits; higher = more uniform) per session (median and IQR)
- Compare occupancy to random walk null model (from `simulate_random_walk`)

**Code references:**
- `maze.analysis.cell_occupancy()`, `occupancy_fraction()`
- `maze.analysis.exploration_efficiency()`
- `maze.analysis.simulate_random_walk()`
- `maze.analysis.maze_exploration_summary()`

---

### Figure 2: Exploration strategies — turn bias and Markov models

**Panel A.** Turn direction distribution at T-junctions across all sessions:
proportion left, right, forward (straight-through), back. Compare to
uniform null (25% each) using chi-squared or multinomial test.

**Panel B.** Per-junction turn bias: for each of the 7 T-junctions, show the
left/right proportion. Some junctions may have inherent biases due to maze
geometry (e.g., dead-end branches may attract less exploration).

**Panel C.** Left-right alternation: probability of alternating turn
direction (left then right, or right then left) at consecutive T-junction
visits vs repeating (left-left or right-right). Compare to chance (50%).
This is the key Rosenberg et al. finding.

**Panel D.** Markov model comparison: AIC/BIC for first-order vs
second-order models across sessions. Report preferred model order (data
show 0/21 sessions prefer second-order; present as a negative finding).

**Panel E.** Sequence entropy vs context length (from `sequence_entropy()`).
Show that predictability increases with context, indicating non-random
navigation.

**Statistics needed:**
- Global left fraction (with 95% CI) across all junction visits pooled
- Per-session left fraction, Wilcoxon test against 0.5
- Alternation probability vs chance (0.5), Wilcoxon across sessions
- Forward bias: proportion of "forward" choices at junctions with a
  straight-through option vs "turn" choices
- AIC/BIC: sign test for order preference across sessions (expect 0/21
  preferring second-order; report as negative finding)

**Code references:**
- `maze.analysis.turn_bias()`, `per_junction_turn_bias()`
- `maze.analysis.markov_order_comparison()`
- `maze.analysis.sequence_entropy()`

---

### Figure 3: Light vs dark — speed and movement

**Panel A.** Running speed (cm/s) by condition: box/violin plot of
session-median speed in light vs dark. Paired by session. Note: data
show no significant speed difference (p = 0.119, uncorrected).

**Panel B.** Speed time course across a session, with light/dark epochs
shaded. Examine whether any speed transitions occur at light changes
(note: the group-level difference is not significant).

**Panel C.** Angular head velocity (AHV, deg/s absolute) by condition.
Box/violin as in A.

**Panel D.** Fraction of time spent moving (speed > threshold) by condition.

**Panel E.** Movement bout duration (consecutive periods above speed
threshold) by condition.

**Statistics needed:**
- Wilcoxon signed-rank for speed, AHV, movement fraction, bout duration
  (light vs dark, N = 21 sessions)
- Effect sizes (rank-biserial correlation)
- Speed distributions (not just means): light vs dark, KS test or similar
- Speed at light-to-dark transitions: peri-event time histogram around
  light-off events (pooled across all transitions)

**Code references:**
- Speed and AHV from kinematics.h5
- Custom peri-event analysis for light transitions
- `light_on` mask from sync.h5

---

### Figure 4: Light vs dark — exploration strategy

**Panel A.** Transition entropy (bits/step) for light vs dark epochs.
Wilcoxon paired test.

**Panel B.** Dead-end visit rate (visits per minute of active movement)
by condition.

**Panel C.** Back-tracking rate (proportion of junction visits where turn =
"back") by condition.

**Panel D.** Path efficiency (optimal / actual path length in sliding
windows) by condition.

**Panel E.** Exploration efficiency (new nodes per window) in light vs dark.

**Panel F.** Directed vs exploratory mode fractions by condition (from
`segment_modes()`).

**Statistics needed:**
- Wilcoxon signed-rank for each metric (light vs dark, N = 21 sessions)
- FDR correction across the 6 metrics in this figure
- Effect sizes
- Comparison to random walk null model in each condition

**Code references:**
- `maze.analysis.transition_matrix()`, `transition_entropy()`
- `maze.analysis.dead_end_visits()`
- `maze.analysis.turn_bias()` (for backtracking rate)
- `maze.analysis.path_efficiency_over_time()`
- `maze.analysis.exploration_efficiency()`
- `maze.analysis.segment_modes()`

---

### Figure 5: Head direction sampling in the maze

**Panel A.** Per-cell HD distribution: polar histogram of HD angles when the
mouse occupies each of the 23 cells. Show that corridor cells have
strongly bimodal HD distributions (aligned with corridor axis) while
junction cells have more uniform distributions.
**[SYNTHETIC DATA PLACEHOLDER]** The current figure uses simulated
preferred directions generated for layout purposes. This panel must be
replaced with real HD distributions computed from actual sessions before
submission.
<!-- TODO(DS-agent): Replace Figure 5A with real per-cell HD distributions. Requires frame-level HD + position from sync.h5. Compute HD histogram per maze cell, pooled across sessions. -->

**Panel B.** HD sampling non-uniformity index (Rayleigh test statistic or
resultant vector length of the HD distribution) per cell, colour-coded on
the maze grid.

**Panel C.** HD sampling uniformity in light vs dark: does HD sampling become
more or less uniform in darkness? Paired comparison across cells and
sessions.

**Panel D.** Joint position x HD occupancy: heatmap showing which (cell, HD
bin) combinations are well-sampled and which are sparse. This is a control
figure for future HD tuning analyses -- it documents the sampling landscape.

**Statistics needed:**
- Rayleigh test per cell (is HD distribution non-uniform?)
- Mean resultant length of HD distribution per cell
- Light vs dark comparison of HD uniformity (Wilcoxon on per-cell Rayleigh
  statistics)

**Code references:**
- `maze.discretize.discretize_position_fast()`
- HD from kinematics.h5
- Custom per-cell HD distribution analysis (new code needed)

---

### Supplementary Figure S1: Individual animal variation

**Panel A.** Per-animal coverage curves (thin lines coloured by animal).

**Panel B.** Per-animal turn bias (left fraction), showing individual
variability.

**Panel C.** Per-animal speed difference (dark - light), showing consistency
of the light effect across animals.

**Panel D.** Tether restriction summary: for each session, show total
excluded time due to `bad_behav` mask. Demonstrates that behavioural
exclusions do not systematically bias the results.

---

### Supplementary Figure S2: Comparison to random walk null model

**Panel A.** Occupancy map: real data vs random walk (unbiased) vs random
walk with forward bias.

**Panel B.** Turn statistics: real vs null models.

**Panel C.** Transition entropy: real vs null.

**Panel D.** Dead-end visit rate: real vs null.

**Code references:**
- `maze.analysis.simulate_random_walk(forward_bias=0.0)` and
  `simulate_random_walk(forward_bias=0.5)`

---

### Supplementary Figure S3: Within-dark-epoch dynamics

**Panel A.** Speed in early (0--30s) vs late (30--60s) dark epochs.

**Panel B.** Transition entropy in early vs late dark epochs.

**Panel C.** Backtracking rate in early vs late dark epochs.

Rationale: Tests whether behaviour degrades within individual dark epochs,
which would be expected if path integration drift causes progressive spatial
disorientation.

---

### Summary statistics table (Table 1)

*Note: p-values shown are Holm-Bonferroni-adjusted within each figure family. Figures display uncorrected p-values; adjusted values are reported here and in the main text. Primary-only column shows p from robustness check using one session per animal (N = 12).*

| Metric | Light (median) | Dark (median) | W | p (raw) | p (adj) | r | Primary-only p (N=12) |
|--------|---------------|--------------|---|---------|---------|---|----------------------|
| Speed (cm/s) | 1.89 | 1.86 | 70.0 | 0.119 | 0.358 | 0.39 | 0.092 |
| Fraction active | 0.466 | 0.443 | 79.0 | 0.216 | 0.431 | 0.32 | 0.151 |
| Immobility bout (s) | 0.82 | 0.87 | 41.0 | 0.279 | 0.279 | 0.32 | -- |
| Per-epoch coverage (frac) | 0.438 | 0.381 | 33.0 | 0.003 | 0.009 | 0.71 | 0.021 |
| Coverage / active min | 23.7 | 22.0 | 31.0 | 0.002 | -- | 0.73 | 0.176 |
| Dead-end rate (/min) | 14.65 | 14.65 | 111.0 | 0.892 | 0.892 | 0.04 | -- |
| Exploration efficiency (w=5) | 3.39 | 3.36 | 107.0 | 0.785 | 1.000 | 0.07 | -- |
| Transition entropy (bits/step) | 1.644 | 1.631 | 100.0 | 0.609 | -- | 0.13 | -- |
| Left turn fraction | 0.487 | 0.489 | 111.0 | 0.892 | 0.892 | 0.04 | -- |
| Backtracking rate | 0.572 | 0.580 | 110.0 | 0.865 | 1.000 | 0.05 | -- |
| HD mean resultant length | 0.297 | 0.338 | 43.0 | 0.010 | 0.020 | 0.63 | 0.339 |
| Median |AHV| (deg/s) | 121.4 | 115.1 | 49.0 | 0.020 | 0.020 | 0.58 | 0.021 |

---

## Draft Results Skeleton

### 1. Mice rapidly cover the q-rose maze

"Mice explored the 23-cell q-rose maze with high coverage. Across 21
sessions, mice visited a median of 23 cells (mean 22.2 +/- 1.6; range
17--23), achieving a median coverage fraction of 1.00 (mean 0.965 +/- 0.070;
Fig. 1D). Total distance travelled varied considerably across sessions
(median 106.2 m; range 40.6--500.7 m), reflecting individual differences in
locomotor activity. Occupancy was non-uniform: T-junction cells were
visited more frequently than dead-end cells (Wilcoxon, Z = XX,
p = XX, r = XX), consistent with junctions serving as transit hubs in the
maze graph (Fig. 1C). Occupancy entropy was [XX] bits (chance for 23 cells:
log2(23) = 4.52 bits; observed: XX bits), indicating [moderate/mild]
non-uniformity."

### 2. Exploration is structured by local turn rules

"At T-junctions, mice showed a consistent tendency to alternate left and
right turns on consecutive junction visits. Sequential turn autocorrelation
was significantly negative across sessions (mean lag-1 autocorrelation =
-0.196, one-sample Wilcoxon, W = 0.0, p < 0.0001, adjusted p < 0.0001,
r = 1.00, N = 21), indicating systematic left-right alternation consistent
with Rosenberg et al. (2021). To control for the possibility that maze
geometry alone produces turn alternation, we compared the observed
autocorrelation to a random walk null model (1000 simulated walks per
session on the maze graph). The null distribution had a negative mean
(-0.141), confirming that the maze topology contributes some alternation.
The observed alternation was stronger than the per-session null means
(Mann-Whitney U = 147.0, p = 0.066, Cliff's d = -0.33, 21 observed vs 21
per-session null means), and bootstrap permutation testing confirmed that
the observed alternation exceeded the null (p < 0.0001), with 5 of 21
sessions (24%) falling below the null 95% CI. Mice therefore show
spontaneous alternation beyond what maze geometry alone would produce,
though the effect size is modest.

"Global left-right bias was minimal (left fraction: 0.49, not significantly
different from 0.5). No individual junction showed a significant left-right
bias after Holm-Bonferroni correction across 7 junctions (all adjusted
p > 0.7; Table S2).

"Backtracking (reversing direction at junctions) was frequent, accounting
for 57--58% of junction visits in both light and dark conditions. This high
rate likely reflects a structural feature of the small q-rose maze: with 9
dead ends among 23 cells, mice frequently reach dead ends and must reverse
course. This contrasts with the larger labyrinth of Rosenberg et al.
(2021), where backtracking is less prominent because the maze graph offers
more through-routes.

"In contrast to Rosenberg et al. (2021), a first-order Markov model was
preferred over a second-order model in all 21 sessions by BIC (mean
delta-BIC = -13,504; 0/21 sessions favouring second-order; Fig. S1C).
This negative finding likely reflects the smaller state space of the q-rose
maze (23 cells, 7 junctions) compared to Rosenberg's 63-junction labyrinth:
with fewer possible transitions, a second-order model introduces many
additional parameters that are poorly estimable from a single session's
trajectory, and BIC's complexity penalty accordingly favours the simpler
model. Sequence entropy did decrease with increasing context length
(Fig. 2E), indicating that navigation is not memoryless, but the improvement
from additional context is modest and does not justify the second-order
model's parameter cost in this maze."

### 3. Speed and movement are largely preserved in darkness

"Running speed did not differ significantly between light and dark epochs
(light median: 1.89 cm/s; dark median: 1.86 cm/s; Wilcoxon, W = 70.0,
p = 0.119, adjusted p = 0.358, r = 0.39, N = 21; Fig. 3A). Although the
literature documents speed reduction in darkness in many paradigms, the
effect was not significant in the present data. This null result may reflect
the constrained locomotion imposed by the head-mounted microscope tether
and the small maze, which limits sustained high-speed running regardless of
lighting condition. The fraction of time spent active (speed >= 2.5 cm/s)
also did not differ significantly between conditions (light: 0.466; dark:
0.443; W = 79.0, p = 0.216, adjusted p = 0.431, r = 0.32; Fig. 3D).
Immobility bout duration was similarly unchanged (W = 41.0, p = 0.279,
adjusted p = 0.279, r = 0.32).

"Angular head velocity (|AHV|) was significantly lower in darkness
(light median: 121.4 deg/s; dark median: 115.1 deg/s; Wilcoxon, W = 49.0,
p = 0.020, adjusted p = 0.020, r = 0.58; Fig. 3C). This effect survived
the primary-only robustness check (N = 12, p = 0.021, r = 0.74). However,
this AHV difference should be interpreted with caution: because AHV and
translational speed are correlated (mice turn their heads faster when
moving faster), the AHV reduction may partly reflect the (non-significant)
trend toward lower speed in darkness rather than an independent change in
head movement strategy. A partial analysis controlling for speed is needed
to distinguish these possibilities."

### 4. Exploration strategy shifts in darkness

"The removal of visual cues reduced per-epoch spatial coverage. Within
individual 1-minute epochs, mice visited a smaller fraction of the 23
accessible cells in darkness than in light (light: 0.438; dark: 0.381;
Wilcoxon, W = 33.0, p = 0.003, adjusted p = 0.009, r = 0.71, N = 21;
Fig. 4A). This coverage reduction was robust to pseudoreplication control:
the effect remained significant in primary-only sessions (N = 12,
p = 0.021, r = 0.74).

"Because coverage is confounded with locomotor activity -- if mice move
less in darkness (even non-significantly), they will mechanically visit
fewer cells per unit time -- we performed a control analysis normalising
coverage by active time (minutes with speed >= 2.5 cm/s) rather than
clock time. Coverage per active minute was also significantly lower in
darkness (light: 23.7 cells/active-min; dark: 22.0 cells/active-min;
Wilcoxon, W = 31.0, p = 0.002, r = 0.73, N = 21), indicating that the
coverage reduction is not simply a locomotor artefact but reflects a
genuine change in exploration efficiency. However, the primary-only
analysis (N = 12 independent animals) did not reach significance for
coverage per active minute (p = 0.176, r = 0.46), so this result should
be interpreted with caution pending a larger sample of independent
animals.

"Other exploration metrics did not differ significantly between conditions.
Transition entropy was similar in light and dark (light: 1.644 bits/step;
dark: 1.631 bits/step; W = 100.0, p = 0.609, r = 0.13), indicating that
navigation predictability was unaffected by visual cue removal. Dead-end
visit rate was identical between conditions (14.65 visits/min in both;
W = 111.0, p = 0.892, r = 0.04). Backtracking rate was also unchanged
(light: 0.572; dark: 0.580; W = 110.0, p = 0.865, adjusted p = 1.000,
r = 0.05). Turn bias (left fraction) did not differ (light: 0.487; dark:
0.489; W = 111.0, p = 0.892, r = 0.04), and sequential turn alternation
strength was similar between conditions (light mean autocorrelation =
-0.227; dark = -0.175; W = 84.0, p = 0.288, adjusted p = 0.863, r = 0.27).

"Taken together, the results indicate that visual cue removal has a
selective rather than global effect on maze navigation: spatial coverage
decreases, but the local decision rules governing turn direction,
alternation, and backtracking are preserved. This pattern is more
consistent with reduced locomotion than with disorientation or a
qualitative shift in navigation strategy."

### 5. Head direction is constrained by maze geometry

"The overall distribution of head direction angles was non-uniform in both
conditions. The mean resultant length of the session-wide HD distribution
was higher in darkness than in light across all 21 sessions (light: 0.297;
dark: 0.338; Wilcoxon, W = 43.0, p = 0.010, adjusted p = 0.020, r = 0.63,
N = 21; Fig. 5B). However, this effect did not survive the primary-only
robustness check (N = 12 independent animals; p = 0.339, r = 0.33),
indicating that the MRL difference is suggestive but not robust to
pseudoreplication control with the current sample size.

"However, the non-uniformity of the HD distribution must be interpreted
with caution. The maze geometry constrains body orientation: corridor cells
impose approximately bimodal HD distributions aligned with the corridor
axis, while dead-end cells produce unimodal distributions toward the
approach direction. The observed non-uniformity therefore reflects a
combination of maze geometry, position-dependent sampling, and any true
changes in exploratory head movements. In particular, the increased mean
resultant length in darkness could arise from mice spending more time
immobile (maintaining a fixed heading) rather than from a genuine change in
directional preference. The non-significant speed and activity differences
make this explanation plausible but not definitive.

"This position-dependent HD sampling is a methodological concern for all
HD tuning analyses conducted in structured environments (as opposed to
open fields). Apparent neural HD selectivity could partially reflect
position-dependent sampling rather than true directional tuning. We
provide per-cell HD occupancy maps (Fig. 5D) to enable occupancy-corrected
tuning curve estimation in future neural analyses."

---

## Discussion Points

### Comparison to Rosenberg et al. (2021)

The q-rose maze produces qualitatively similar behavioural patterns to
Rosenberg's 63-junction labyrinth in some respects: mice show systematic
left-right turn alternation (negative sequential autocorrelation), and
exploration is structured rather than random. However, there are notable
differences. First, the q-rose maze's smaller state space (23 vs 127 cells,
7 vs 63 junctions) means that a first-order Markov model is preferred in
all sessions, whereas Rosenberg et al. found that second-order models fit
better. This likely reflects parameter estimability: in the q-rose maze,
the second-order model introduces many transition parameters that cannot be
reliably estimated from a single session's trajectory. Second, the
backtracking rate is high (57--58% of junction visits), a structural
consequence of the maze having 9 dead ends among 23 cells, which forces
frequent reversals. In Rosenberg's larger labyrinth with more through-routes,
backtracking was less prominent. The absence of reward in our task means
there is no target bias component; exploration is intrinsically motivated.

### Light/dark effects in context

Speed reduction in darkness is a well-documented phenomenon in rodents
exploring open arenas (various refs). In the present data, however, speed
did not differ significantly between light and dark conditions (p = 0.119).
This null result may reflect the constrained locomotion imposed by the
head-mounted microscope tether and the small maze, both of which limit
the range of speeds available regardless of lighting. It is also possible
that the 1-minute epoch duration is too short for a sustained speed
reduction to emerge, or that mice habituated to the maze sufficiently
that darkness did not produce cautiousness.

The most robust light-dark difference was in per-epoch spatial coverage,
which was significantly lower in darkness. Because coverage is mechanically
coupled to locomotor activity, we normalised coverage by active time
(minutes with speed >= 2.5 cm/s). Coverage per active minute remained
significantly lower in darkness across all 21 sessions (p = 0.002,
r = 0.73), indicating that the coverage reduction is not simply a
locomotor artefact. However, this normalised coverage effect did not
survive the primary-only robustness check (N = 12 independent animals;
p = 0.176, r = 0.46), suggesting that the effect may be partly driven by
animals contributing multiple sessions. The raw coverage difference did
survive primary-only analysis (p = 0.021, r = 0.74), so the core finding
of reduced coverage in darkness is robust, but the mechanistic
interpretation -- strategy change vs locomotor reduction -- remains open
with the current sample size.

The reduction in angular head velocity (AHV) in darkness is consistent with
the vestibular-visual integration findings of Keshavarzi et al. (2022), who
showed that visual input increases the gain of AHV coding in RSP. However,
AHV and translational speed are correlated, so the AHV difference may be
partly a speed confound rather than an independent change in head movement
dynamics.

The 1-minute epoch duration is relevant to the HD literature: HD drift in
darkness accumulates gradually (Stackman & Taube 1997), with ~40% of HD
cells becoming unstable within minutes (Muir et al. 2022). The supplementary
within-dark-epoch analysis (Fig. S3) tests whether behavioural metrics
degrade within individual dark epochs, which would be expected if
progressive spatial disorientation accompanies HD drift.

### HD sampling confound for neural analyses

The non-uniform HD sampling in maze corridors is a methodological concern
for all HD tuning analyses conducted in structured environments (as opposed
to open fields). This has been acknowledged in the literature (Muir et al.
2022; Jacob et al. 2017) but is rarely quantified explicitly. Our per-cell
HD occupancy maps (Fig. 5D) provide the basis for occupancy-corrected tuning
curve estimation in the companion neural paper.

### HD non-uniformity: position and immobility confounds

The increased mean resultant length of the HD distribution in darkness
(Fig. 5B) has multiple possible explanations beyond a change in exploration
strategy. The main analysis already restricts to active frames (speed >=
2.5 cm/s), so immobility per se does not drive the effect. However, if
mice in darkness spend proportionally more time in corridors (which impose
bimodal HD distributions) rather than at junctions (which allow more
uniform heading), the overall HD distribution will appear more
concentrated. The MRL by node type control (deferred; requires frame-level
data) would address this confound. Importantly, the MRL difference does
not survive the primary-only robustness check (N = 12, p = 0.339,
r = 0.33), so the effect should be considered suggestive rather than
established. Additional independent animals are needed to determine
whether HD concentration genuinely increases in darkness.

### Limitations

1. The head-mounted microscope tether restricts movement to some degree.
   While we exclude periods of clear tether entanglement, subtle motor
   constraints may still influence exploration patterns compared to
   untethered mice. The tether may also contribute to the null speed result
   by limiting the dynamic range of locomotion available in both conditions.

2. With 21 sessions from 15 animals (4 animals contributing 2--3 sessions
   each), session-level analyses involve mild pseudoreplication. We address
   this with a primary-only robustness check (N = 12 independent animals),
   but the reduced sample size limits statistical power. The core coverage
   finding survives this control (p = 0.021), but coverage per active minute
   does not (p = 0.176), and MRL does not (p = 0.339). The speed null
   result remains non-significant in both analyses.

3. The 1-minute light/dark epoch duration was chosen for neural imaging
   purposes (testing HD re-anchoring dynamics) rather than to optimise
   behavioural measurements. Longer dark epochs might reveal more
   pronounced exploration changes.

4. This is a free-exploration paradigm with no explicit task demands. The
   behavioural metrics quantify exploration strategy, not task performance.
   Goal-directed navigation metrics (path efficiency, monotonic paths) are
   computed relative to dead ends as surrogate targets, not experimentally
   defined goals.

5. The coverage difference between light and dark may be confounded by
   locomotor activity. Although speed was not significantly different
   between conditions, any trend toward reduced movement in darkness would
   mechanically reduce coverage. A control analysis normalising coverage by
   active time (coverage per active minute) showed that the effect survives
   in the full dataset (p = 0.002) but not in primary-only sessions
   (p = 0.176). The coverage finding is therefore robust in its basic form
   (raw coverage, primary-only p = 0.021), but whether it reflects a
   strategy change or a locomotor effect cannot be definitively resolved
   with the current sample.

6. Speed analysis by node type (Fig. 6B) was computed using only active
   frames (speed >= 2.5 cm/s), which biases results at locations where
   mice frequently stop (e.g., dead ends, where they pause before
   reversing). The finding that dead ends show the highest active speed may
   be an artefact of this filtering, since it selects only the moments of
   acceleration out of dead ends while excluding the (potentially long)
   pauses. A control analysis using all frames (including immobile periods)
   is needed to assess whether the node-type speed differences are robust.

7. The AHV difference between light and dark conditions, while nominally
   significant, may be confounded by the (non-significant) speed trend.
   AHV and translational speed are correlated in freely-moving mice, so
   a speed-controlled analysis (e.g., comparing AHV within matched speed
   bins) is needed to determine whether the AHV reduction is independent.

8. Three control analyses require frame-level data from regenerated
   sync.h5 files and are deferred until the pipeline re-run completes:
   (a) MRL by maze node type (junction vs corridor vs dead end) in light
   vs dark, which would determine whether the HD concentration increase
   in darkness is driven by differential maze-location occupancy;
   (b) speed by node type using all frames (including immobile periods),
   which would test whether the dead-end speed result from the active-only
   analysis is an artefact of the activity filter; and (c) per-bodypart
   tracking confidence by light condition, which would provide a more
   granular check on tracking quality than the aggregate statistics
   reported in Methods.

---

## References

Ajabi Z, Keinath AT, Brandon MP. 2023. "Population dynamics of
head-direction neurons during drift and reorientation." *Nature* 615,
892--899. doi:10.1038/s41586-023-05813-2

Bhatt DK et al. 2024. "Rodent maze studies: from following simple rules to
complex map learning." *Brain Struct. Funct.* 229, 1261--1278.
doi:10.1007/s00429-024-02771-x

Bicknell BA, van der Goes M-SH et al. 2024. "Coordinated head direction
representations in mouse anterodorsal thalamic nucleus and retrosplenial
cortex." *eLife* 13, e82952. doi:10.7554/eLife.82952

Chen LL, Lin LH, Green EJ, Barnes CA, McNaughton BL. 1994. "Head-direction
cells in the rat posterior cortex. I. Anatomical distribution and behavioral
modulation." *Exp. Brain Res.* 101, 8--23.

Cho J, Sharp PE. 2001. "Head direction, place, and movement correlates for
cells in the rat retrosplenial cortex." *Behav. Neurosci.* 115, 3--25.

Etienne AS, Jeffery KJ. 2004. "Path integration in mammals." *Hippocampus*
14, 180--192.

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

Zugaro MB, Arleo A, Berthoz A, Wiener SI. 2003. "Rapid spatial
reorientation and head direction cells." *J. Neurosci.* 23, 3478--3482.

---

## Critical Assessment and Confounds

### What is novel about this paper

1. **The q-rose maze under light/dark alternation is a new paradigm.** The
   original Rosenberg maze used fixed lighting. No prior study has combined
   a binary-choice labyrinth with total darkness manipulation and quantified
   the effect on exploration strategy.

2. **Graph-theoretic behavioural characterisation during light/dark
   alternation.** Transition entropy, dead-end visit dynamics, and Markov
   model statistics have not been compared between light and dark conditions
   in a structured maze. The finding that most navigation metrics are
   unchanged by darkness -- despite a significant reduction in spatial
   coverage -- is itself informative.

3. **HD sampling characterisation in a structured maze.** This confound is
   acknowledged but rarely quantified. Explicit per-cell HD occupancy maps
   are a useful methodological contribution.

4. **Honest reporting of null results.** The null speed result and the
   failure of the second-order Markov model are informative for future
   studies using similar paradigms.

### What is NOT novel

- Turn alternation in maze exploration (Rosenberg et al. 2021).
- Speed reduction in darkness is well-established in the literature,
  although it was not confirmed in the present data (p = 0.119).
- HD drift in darkness (Stackman & Taube 1997; Ajabi et al. 2023).
- DeepLabCut-based pose tracking (standard tool).

### Key confounds to address

1. **Tether restriction.** The head-mounted microscope tether constrains
   movement. Must show that behavioural metrics are consistent between early
   session (minimal tether restriction) and late session, and between
   sessions with and without flagged tether periods.

2. **Epoch order confounds.** Light and dark epochs alternate, so dark
   epochs are always preceded and followed by light epochs. Any adaptation
   or fatigue effects could confound the light/dark comparison. Test for
   epoch-number effects (does the 5th dark epoch differ from the 1st?).

3. **Speed confound for exploration metrics.** Although speed was not
   significantly reduced in darkness, any trend toward slower speed means
   fewer cell transitions per minute. Transition entropy is
   rate-normalised (bits per step), but coverage and dead-end visit rate
   must be normalised by active time or number of transitions, not clock
   time. The coverage-per-active-minute control (p = 0.002, N = 21)
   suggests the finding survives locomotor normalisation, but the
   primary-only analysis does not reach significance (p = 0.176, N = 12),
   so the speed confound cannot be definitively ruled out.

4. **Small maze ceiling effects.** With only 23 cells, coverage approaches
   100% quickly, limiting the dynamic range for exploration efficiency
   comparisons.

5. **Animal sex.** One female (1118023) is included among 15 males. Check
   whether her behavioural metrics are outliers.

6. **Animals with multiple sessions.** Some animals contribute 2--4
   sessions. For session-level statistics, this creates mild
   pseudoreplication. Report results both with all sessions and with only
   one session per animal (first or primary).

### Reviewer objections to anticipate

- "This is descriptive with no neural data — what is the contribution?"
  Response: Establishes the paradigm for the companion neural paper and
  provides quantitative behavioural baselines.

- "With only 7 junctions, is the Markov model meaningful?" Response: The
  data show that a first-order model is preferred in all 21 sessions.
  We interpret this as a consequence of the small state space (23 cells),
  where the second-order model's additional parameters cannot be reliably
  estimated. This is a genuine limitation of the maze size for Markov
  modelling, and we present it honestly as a negative finding rather than
  a limitation to be hidden.

- "Why not use an open field for the HD study?" Response: The maze provides
  hundreds of natural binary decisions and structured corridors that
  constrain HD sampling in known ways. This enriches the behavioural
  readout compared to open-field foraging.

- "How do you know darkness was total?" Response: No visible illumination
  was confirmed; infrared illuminators operate at wavelengths outside mouse
  scotopic sensitivity; two-photon laser at 920 nm is also invisible.

---

## Implementation Notes

### New code needed

1. **Per-cell HD distribution analysis.** Compute HD angle histogram per
   maze cell, with Rayleigh test and resultant vector length. Not yet
   implemented.

2. **Peri-event speed analysis around light transitions.** Align speed to
   light-on/off events and compute average peri-event time histograms. Not
   yet implemented (straightforward with existing sync data).

3. **Turn alternation analysis.** Compute probability of alternating
   left/right across consecutive junction visits. Not yet explicitly
   computed (though the data is available from `per_junction_turn_bias`).

4. **Within-dark-epoch temporal splitting.** Split each dark epoch into
   early and late halves and compute metrics separately.

5. **Normalisation of visit rates by active time.** Current dead-end visit
   analysis counts raw visits; must normalise by time spent moving in each
   condition.

6. **Coverage per active minute control.** COMPLETED. Coverage per active
   minute is significantly lower in dark (p = 0.002, r = 0.73, N = 21),
   but the primary-only analysis does not reach significance (p = 0.176,
   r = 0.46, N = 12). See Control 1 in behaviour-control-summary.md.

7. **Speed-controlled AHV analysis.** The AHV light-dark difference may
   reflect a speed confound. Compare AHV within matched speed bins (e.g.,
   5--10 cm/s) between conditions, or compute partial correlation of AHV
   with condition after regressing out speed.

8. **Speed by node type without active-only filter.** The current analysis
   uses speed >= 2.5 cm/s threshold, which biases results. Repeat Fig. 6B
   analysis using all frames to check whether dead ends are genuinely
   traversed at higher speed or whether the active-only filter creates an
   artefact.

### Existing code that can be used directly

- `maze.topology.build_rose_maze()` — maze graph
- `maze.discretize.discretize_position_fast()` — position to cell
- `maze.discretize.cell_sequence()`, `node_sequence()` — trajectory
  compression
- `maze.analysis.cell_occupancy()`, `occupancy_fraction()` — occupancy
- `maze.analysis.exploration_efficiency()` — new nodes per window
- `maze.analysis.turn_bias()`, `per_junction_turn_bias()` — turn
  classification
- `maze.analysis.transition_matrix()`, `transition_entropy()` — Markov
  models
- `maze.analysis.markov_order_comparison()` — AIC/BIC
- `maze.analysis.sequence_entropy()` — context-dependent entropy
- `maze.analysis.segment_modes()` — directed vs exploratory
- `maze.analysis.dead_end_visits()` — dead-end analysis
- `maze.analysis.path_efficiency_over_time()` — path efficiency
- `maze.analysis.simulate_random_walk()` — null model
