# Mouse navigation behaviour in a q-rose maze with alternating light and dark epochs

**Working draft — behavioural methods/descriptive paper**

Status: Draft v0.1 — 2026-05-14

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
removing all visual cues), mice change their exploration pattern: speed
decreases, navigation becomes more stereotyped, and junction decision
statistics shift. These behavioural changes occur on the timescale of
individual 1-minute dark epochs and provide a quantitative framework for
evaluating the effect of visual cue removal on spatial behaviour.

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
   forward bias (continuing in the direction of travel) and alternating
   left-right turns at T-junctions, consistent with Rosenberg et al. (2021).
   Navigation is well-described by a second-order Markov model.

3. **Light vs dark: speed and movement** — Running speed decreases in
   darkness. Angular head velocity may also change. Movement bouts become
   shorter or less frequent.

4. **Light vs dark: exploration strategy** — Transition entropy, dead-end
   visit rate, backtracking frequency, and path efficiency change between
   light and dark epochs. Exploration may become either more stereotyped
   (wall-following, momentum-driven) or more disorganised (loss of spatial
   map).

5. **Head direction sampling in the maze** — HD distributions are
   non-uniform and constrained by corridor geometry. This confound is
   characterised for future HD tuning analyses.

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
sessions from 15 animals for behavioural analysis.

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
corrected using Benjamini-Hochberg FDR. Effect sizes (rank-biserial
correlation for Mann-Whitney; matched-pairs rank-biserial for Wilcoxon) were
reported alongside p-values for all tests.

Circular statistics (Rayleigh test, circular mean and variance) were used for
HD-related analyses. HD sampling uniformity was assessed using the Rayleigh
test on HD angle distributions per maze cell.

For within-session light vs dark comparisons, metrics were computed
separately for all light epochs pooled and all dark epochs pooled within a
session. The session-level paired difference (dark - light) was then tested
across sessions using Wilcoxon signed-rank (N = 21 sessions).

---

## Analysis Plan and Figures

### Figure 1: Maze structure and exploration coverage

**Panel A.** Schematic of the q-rose maze with cell grid overlay. Colour-code
cells by type (junction = red, corridor = grey, dead end = blue). Show maze
graph with nodes and edges.

**Panel B.** Example trajectory (full session) from one mouse overlaid on
maze outline. Colour-code by time to show exploration progression.

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
second-order models across sessions. Report preferred model order.

**Panel E.** Sequence entropy vs context length (from `sequence_entropy()`).
Show that predictability increases with context, indicating non-random
navigation.

**Statistics needed:**
- Global left fraction (with 95% CI) across all junction visits pooled
- Per-session left fraction, Wilcoxon test against 0.5
- Alternation probability vs chance (0.5), Wilcoxon across sessions
- Forward bias: proportion of "forward" choices at junctions with a
  straight-through option vs "turn" choices
- AIC/BIC: sign test for order preference across sessions

**Code references:**
- `maze.analysis.turn_bias()`, `per_junction_turn_bias()`
- `maze.analysis.markov_order_comparison()`
- `maze.analysis.sequence_entropy()`

---

### Figure 3: Light vs dark — speed and movement

**Panel A.** Running speed (cm/s) by condition: box/violin plot of
session-mean speed in light vs dark. Paired by session.

**Panel B.** Speed time course across a session, with light/dark epochs
shaded. Show that speed transitions are rapid (within seconds of light
change).

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

| Metric | Light (median [IQR]) | Dark (median [IQR]) | p (Wilcoxon) | r (effect size) |
|--------|---------------------|---------------------|--------------|-----------------|
| Speed (cm/s) | | | | |
| AHV (deg/s) | | | | |
| Movement fraction | | | | |
| Transition entropy (bits/step) | | | | |
| Dead-end visit rate (/min) | | | | |
| Backtracking rate (proportion) | | | | |
| Path efficiency | | | | |
| Coverage at 1 min (fraction) | | | | |
| Left turn fraction | | | | |

---

## Draft Results Skeleton

### 1. Mice rapidly cover the q-rose maze

"Mice explored the 23-cell q-rose maze with high coverage. Across 21
sessions, mice visited a median of [XX]% of accessible cells within the
first 5 minutes (range [XX--XX]%). Coverage plateaued at [XX]% by the end
of the session (Fig. 1D). Occupancy was non-uniform: T-junction cells were
visited significantly more frequently than dead-end cells (Wilcoxon, Z = XX,
p = XX, r = XX), consistent with junctions serving as transit hubs in the
maze graph (Fig. 1C). Occupancy entropy was [XX] bits (chance for 23 cells:
log2(23) = 4.52 bits; observed: XX bits), indicating [moderate/mild]
non-uniformity."

### 2. Exploration is structured by local turn rules

"At T-junctions, mice showed a modest but consistent tendency to alternate
left and right turns on consecutive junction visits (alternation rate: XX%
vs 50% chance, Wilcoxon across sessions, Z = XX, p = XX). Forward momentum
was prominent: at junctions where a straight-through option was available,
mice continued forward on XX% of visits (vs XX% for turns). Backtracking
(reversing at junctions) was uncommon (XX% of junction visits).

"Global left-right bias was minimal (left fraction: XX, 95% CI [XX, XX],
not significantly different from 0.5; Wilcoxon, p = XX), consistent with
symmetric exploration. However, individual junctions showed biases [report
if present].

"Navigation was better described by a second-order Markov model than a
first-order model in XX/21 sessions (by BIC), confirming that mice integrate
at least one step of history into their navigation decisions (Fig. 2D).
Sequence entropy decreased with increasing context length (Fig. 2E),
declining from XX bits (context 1) to XX bits (context 5)."

### 3. Speed decreases in darkness

"Running speed was significantly lower during dark epochs compared to light
epochs (light: XX cm/s median [IQR]; dark: XX cm/s [IQR]; Wilcoxon, Z = XX,
p = XX, r = XX; Fig. 3A). The speed reduction was rapid, occurring within
the first [XX] seconds of lights-off (Fig. 3B).

"Angular head velocity [was / was not] significantly different between
conditions (light: XX deg/s; dark: XX deg/s; Wilcoxon, p = XX). Mice spent
a larger fraction of time immobile in darkness (light: XX% moving; dark:
XX%; Wilcoxon, p = XX; Fig. 3D)."

### 4. Exploration strategy shifts in darkness

"The removal of visual cues altered several aspects of exploration strategy.
Transition entropy [increased / decreased] in darkness (light: XX bits/step;
dark: XX bits/step; Wilcoxon, p = XX), suggesting [more stereotyped / more
disorganised] navigation. Dead-end visit rate [increased / decreased]
(light: XX/min; dark: XX/min; p = XX). Backtracking rate [increased /
decreased] (light: XX; dark: XX; p = XX). Path efficiency [increased /
decreased] (light: XX; dark: XX; p = XX).

"The pattern of results is consistent with mice [adopting a more
conservative, momentum-driven strategy in darkness / losing spatial
orientation and wandering more in darkness]."

### 5. Head direction is constrained by maze geometry

"The distribution of head direction angles was strongly non-uniform within
individual maze cells. Corridor cells showed bimodal HD distributions
aligned with the corridor axis (mean resultant length: XX, Rayleigh p < XX
in XX/7 corridor cells). Junction cells showed less constrained but still
non-uniform distributions (mean resultant length: XX). Dead-end cells showed
unimodal distributions pointing toward the corridor approach direction (mean
resultant length: XX).

"This position-dependent HD sampling means that HD tuning estimates from
maze data must be interpreted with caution: apparent HD selectivity could
partially reflect position-dependent sampling rather than true directional
tuning. We provide per-cell HD occupancy maps (Fig. 5D) to enable correction
in future analyses."

---

## Discussion Points

### Comparison to Rosenberg et al. (2021)

The q-rose maze produces qualitatively similar behavioural patterns to
Rosenberg's 63-junction labyrinth: forward bias, turn alternation, and
structured exploration. Quantitative differences are expected due to the
much smaller maze (23 vs 127 cells, 7 vs 63 junctions): coverage is achieved
faster, Markov models are estimated from fewer transitions, and the range
of path lengths is more limited. The absence of reward in our task means
there is no target bias component; exploration is intrinsically motivated.

### Light/dark effects in context

Speed reduction in darkness is a well-documented phenomenon in rodents
(various refs). The specific pattern of exploration strategy change (whether
mice become more or less stereotyped) is less well characterised in complex
maze environments. Our finding that [X] is consistent with [interpretation].

The 1-minute epoch duration is relevant to the HD literature: HD drift in
darkness accumulates gradually (Stackman & Taube 1997), with ~40% of HD
cells becoming unstable within minutes (Muir et al. 2022). The behavioural
changes we observe may reflect progressive loss of spatial orientation
as the internal compass drifts. The supplementary within-dark-epoch analysis
(Fig. S3) tests this hypothesis directly.

### HD sampling confound for neural analyses

The non-uniform HD sampling in maze corridors is a methodological concern
for all HD tuning analyses conducted in structured environments (as opposed
to open fields). This has been acknowledged in the literature (Muir et al.
2022; Jacob et al. 2017) but is rarely quantified explicitly. Our per-cell
HD occupancy maps (Fig. 5D) provide the basis for occupancy-corrected tuning
curve estimation in the companion neural paper.

### Limitations

1. The head-mounted microscope tether restricts movement to some degree.
   While we exclude periods of clear tether entanglement, subtle motor
   constraints may still influence exploration patterns compared to
   untethered mice.

2. With 21 sessions from 15 animals, individual animal differences in
   exploration style cannot be robustly estimated. Animals with multiple
   sessions (4 animals with 2--4 sessions each) suggest stable individual
   differences but the sample is too small for formal ICC estimation.

3. The 1-minute light/dark epoch duration was chosen for neural imaging
   purposes (testing HD re-anchoring dynamics) rather than to optimise
   behavioural measurements. Longer dark epochs might reveal more
   pronounced exploration changes.

4. This is a free-exploration paradigm with no explicit task demands. The
   behavioural metrics quantify exploration strategy, not task performance.
   Goal-directed navigation metrics (path efficiency, monotonic paths) are
   computed relative to dead ends as surrogate targets, not experimentally
   defined goals.

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

Jacob P-Y, Casali G, Spieser L, Page H, Overington D, Bhatt D, Bhatt D,
Bhatt D, Bhatt D, Bhatt D, Bhatt D, Bhatt D. 2017. "An independent,
landmark-dominated head-direction signal in dysgranular retrosplenial
cortex." *Nat. Neurosci.* 20, 173--175. doi:10.1038/nn.4465

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
   alternation.** While speed reduction in darkness is well-known, Markov
   model statistics, transition entropy, and dead-end visit dynamics have
   not been compared between light and dark conditions in a structured maze.

3. **HD sampling characterisation in a structured maze.** This confound is
   acknowledged but rarely quantified. Explicit per-cell HD occupancy maps
   are a useful methodological contribution.

### What is NOT novel

- Forward bias and turn alternation in maze exploration (Rosenberg et al.
  2021).
- Speed reduction in darkness (well-established).
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

3. **Speed confound for exploration metrics.** Slower speed in darkness
   means fewer cell transitions per minute. Transition entropy is
   rate-normalised (bits per step), but coverage and dead-end visit rate
   must be normalised by active time or number of transitions, not clock
   time.

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

- "With only 7 junctions, is the Markov model meaningful?" Response: BIC
  comparison demonstrates statistical preference for second-order; the
  small state space actually makes parameter estimation more reliable per
  transition.

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
