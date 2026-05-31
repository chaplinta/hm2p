# Mouse navigation behaviour in a q-rose maze with alternating light and dark epochs

**Working draft — behavioural methods/descriptive paper**

Status: Draft v0.6.1 — 2026-05-31 (QA fixes: JSD ratio 3.5x→3.9x, cell (3,3) delta correction, Fig. S1C→2D, Holm-Bonferroni family footnotes, named all 6 significant cells)

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
removing all visual cues), spatial coverage decreases robustly (p < 0.001,
r = 0.86), providing the strongest and most consistent behavioural marker of
visual cue removal. Running speed shows a modest trend toward reduction in
darkness (p = 0.076), which reaches significance in primary-only sessions
(p = 0.042), suggesting that reduced locomotion partially contributes to
the coverage drop. Other metrics -- turn statistics, transition entropy,
backtracking, and dead-end visit rate -- remain stable between conditions.

The coverage reduction reflects route stereotypy rather than global
disengagement: corridor and junction coverage drop significantly in
darkness, but dead-end coverage is unchanged (p = 0.26), indicating that
mice maintain visits to terminal destinations while consolidating onto fewer
connecting routes. The visited subgraph diameter contracts (6.35 to 5.57
cells, p = 0.002), transition matrices diverge between conditions (JSD =
0.068, 3.9x permutation null, p < 0.001), and revisitation of
already-covered cells increases (p = 0.011). This dissociation between
preserved local navigation rules and altered global route selection suggests
that turn decisions and route planning draw on different information sources,
with route selection depending on a spatial representation that degrades
without visual cues.

The spatial structure of the coverage reduction is not uniform across the
maze. Cells closer to the maze centre show the largest visit reductions
(distance-from-centre correlation: Spearman rho = 0.75, p < 0.0001, N = 23),
inverting the expected pattern of peripheral abandonment. Central backbone
corridors and junctions lose the most visits, while peripheral dead ends are
maintained -- deepening the route stereotypy interpretation.

The coverage reduction occurs as a single-trial adaptation: the first dark
epoch has substantially higher coverage than all subsequent dark epochs
(0.57 vs 0.30; Wilcoxon, p = 0.0001, r = 0.89, N = 20), after which dark
coverage stabilises. There is no gradual improvement or worsening across
later dark epochs (slope p = 0.81). This pattern indicates that mice adjust
their exploration strategy after their first experience of darkness in the
session and then maintain this adjusted strategy for the remainder.

HD distributions become more concentrated in darkness, but this trend does
not survive multiple comparisons correction (adjusted p = 0.152) or
primary-only analysis (p = 0.278). AHV does not differ between conditions
(p = 0.177). These findings establish spatial coverage, route stereotypy,
and first-epoch adaptation as reliable behavioural readouts of visual cue
availability in the q-rose maze.

### Structure

| Section | Content | Words |
|---------|---------|-------|
| Introduction | Frame: maze navigation, exploration rules, light/dark, HD system | ~500 |
| Methods | Maze, animals, surgery, imaging, tracking, analysis | ~800 |
| Results | 6 main findings (see below) | ~1600 |
| Discussion | Comparison to Rosenberg, route stereotypy, implications for HD/navigation studies | ~800 |
| Supplementary | Controls, additional metrics, individual animal data | as needed |

### Seven main results

1. **Maze structure and exploration coverage** — Mice explore the 23-cell
   q-rose maze with high coverage (typically >90% of cells visited), with
   occupancy concentrated at junctions and corridors rather than dead ends.

2. **Exploration strategies: turn bias and forward momentum** — Mice show
   left-right turn alternation at T-junctions, consistent with Rosenberg
   et al. (2021). Backtracking is frequent (~48--51% of junction visits),
   reflecting the small maze with many dead ends. A first-order Markov model
   is preferred over second-order in all sessions, in contrast to the larger
   labyrinth of Rosenberg et al.

3. **Light vs dark: spatial coverage (primary finding)** — Per-epoch spatial
   coverage decreases in darkness (p = 0.0003, adjusted p = 0.0008,
   r = 0.86, N = 20; primary-only p = 0.010, r = 0.85, N = 11). This is
   the most robust light-dark difference and survives all robustness checks.
   Coverage per active minute is also lower in darkness (p = 0.001,
   r = 0.78), indicating that the effect is not purely a locomotor artefact,
   though the primary-only analysis does not reach significance for this
   normalised metric (p = 0.175, r = 0.49).

3b. **The coverage drop reflects route stereotypy concentrated at central
   maze locations** — Speed and coverage are strongly correlated (Spearman
   rho = 0.76, p < 0.0001), but speed does not fully account for the
   coverage reduction: coverage per transition shows a trend toward reduced
   exploration efficiency per decision (light 0.398 vs dark 0.355,
   p = 0.076, r = 0.46). Corridor coverage drops most strongly
   (p < 0.00001, r = 0.97), junction coverage drops moderately (p = 0.001,
   r = 0.82), but dead-end coverage is unchanged (p = 0.26). Mice maintain
   visits to terminal destinations while consolidating onto fewer connecting
   routes. Per-cell analysis reveals that this consolidation is
   topographically structured: cells closer to the maze centre show the
   largest visit reductions (distance-from-centre: rho = 0.75, p < 0.0001,
   N = 23), and among corridor cells, eccentricity correlates with coverage
   change (rho = 0.87, p = 0.012, N = 7). The visited subgraph diameter
   contracts (6.35 to 5.57 cells, p = 0.002, r = 0.80), transition matrices
   diverge between conditions (JSD = 0.068, 3.9x permutation null,
   p < 0.001), and revisitation increases (p = 0.011, r = 0.64).

3c. **Route stereotypy is established after a single dark epoch** — The
   first dark epoch in each session has substantially higher coverage than
   all subsequent dark epochs (0.57 vs 0.30; Wilcoxon, p = 0.0001, r = 0.89,
   N = 20). After the first dark epoch, coverage is stable across subsequent
   dark epochs (slope test: p = 0.81). Light-epoch coverage shows a gradual
   decline over the session (p = 0.011), consistent with general
   habituation, but there is no corresponding trend in dark epochs beyond
   the first. The mouse adapts after its first experience of darkness, then
   maintains that adjusted strategy.

4. **Light vs dark: speed and movement** — Running speed shows a trend
   toward reduction in darkness (p = 0.076, adjusted p = 0.152, r = 0.46,
   N = 20) that reaches significance in primary-only sessions (p = 0.042,
   r = 0.70, N = 11). Angular head velocity does not differ between
   conditions (p = 0.177, r = 0.35). Other exploration metrics (transition
   entropy, dead-end visit rate, backtracking frequency, turn bias) are
   unchanged.

5. **Head direction sampling in the maze** — HD distributions are
   non-uniform and constrained by corridor geometry. MRL shows a
   non-significant trend toward higher values in darkness (p = 0.076,
   adjusted p = 0.152, r = 0.46; primary-only p = 0.278, r = 0.39). This
   trend does not survive correction and is characterised as a confound for
   future HD tuning analyses rather than a confirmed finding.

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
infrared illumination, tracking quality does not differ systematically
between light and dark conditions.

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
across sessions using Wilcoxon signed-rank (N = 20 sessions).

### Pseudoreplication

Some animals contributed multiple sessions (4 animals with 2--3 usable
sessions each), creating mild pseudoreplication in session-level analyses.
As a robustness check, all primary light-vs-dark comparisons were repeated
using only primary-experiment sessions (one per animal, N = 11 independent
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
show 0/20 sessions prefer second-order; present as a negative finding).

**Panel E.** Sequence entropy vs context length (from `sequence_entropy()`).
Show that predictability increases with context, indicating non-random
navigation.

**Statistics needed:**
- Global left fraction (with 95% CI) across all junction visits pooled
- Per-session left fraction, Wilcoxon test against 0.5
- Alternation probability vs chance (0.5), Wilcoxon across sessions
- Forward bias: proportion of "forward" choices at junctions with a
  straight-through option vs "turn" choices
- AIC/BIC: sign test for order preference across sessions (expect 0/20
  preferring second-order; report as negative finding)

**Code references:**
- `maze.analysis.turn_bias()`, `per_junction_turn_bias()`
- `maze.analysis.markov_order_comparison()`
- `maze.analysis.sequence_entropy()`

---

### Figure 3: Light vs dark — spatial coverage (primary finding)

**Panel A.** Per-epoch coverage (fraction of 23 cells visited per 1-minute
epoch) in light vs dark. Box/violin plot of session means, paired by session.
This is the primary finding of the paper (p = 0.0003, r = 0.86).

**Panel B.** Coverage per active minute (cells visited normalised by active
time). Controls for the speed confound: even after normalisation, coverage
is lower in darkness (p = 0.001, r = 0.78).

**Panel C.** Exploration efficiency (new unique nodes per sliding window of
5 cell transitions) in light vs dark. Shows a trend (p = 0.058, r = 0.49)
consistent with the coverage finding but not independently significant.

**Panel D.** Coverage difference (dark - light) per session, ordered by
magnitude. Shows consistency of the effect across sessions.

**Statistics needed:**
- Wilcoxon signed-rank for each metric (light vs dark, N = 20 sessions)
- Holm-Bonferroni correction across the 3 metrics in this figure
- Effect sizes
- Primary-only robustness check (N = 11)

**Code references:**
- `maze.analysis.cell_occupancy()`, `occupancy_fraction()`
- `maze.analysis.exploration_efficiency()`

---

### Figure 4: Route stereotypy in darkness

**Panel A.** Speed difference (dark - light) vs coverage difference (dark -
light) per session. Scatter plot with Spearman rho = 0.76 (p < 0.0001).
Shows the strong speed-coverage coupling (H1) and highlights that speed
contributes to but does not fully explain the coverage drop.

**Panel B.** Per-epoch coverage by maze cell type (junction, corridor,
dead end) in light vs dark. Paired box plots for each cell type. Key
finding: corridor coverage drops most strongly (p < 0.00001, r = 0.97),
junction coverage drops moderately (p = 0.001, r = 0.82), dead-end
coverage is unchanged (p = 0.26). Include direct comparison of the
dead-end vs junction drop magnitudes (p = 0.0004, r = 0.84).

**Panel C.** Revisitation index (total transitions / unique cells visited)
in light vs dark. Paired comparison across sessions (p = 0.011, r = 0.64).
Higher values in darkness indicate increased revisitation of
already-covered cells.

**Panel D.** Jensen-Shannon divergence between light and dark transition
matrices. Histogram of observed per-session JSD values with the
permutation null distribution overlaid (1000 permutations). Observed mean
JSD = 0.068 vs null mean = 0.018 (3.9x). Demonstrates that routing
patterns change between conditions despite preserved transition entropy.

**Panel E.** Per-cell coverage heatmap on maze grid. Colour-code each of
the 23 cells by the mean visit fraction delta (dark - light). Central
backbone cells (especially (3,2), (4,2), (3,3)) show the largest reductions
(darkest colours); peripheral dead ends show near-zero change. Six cells
survive Holm-Bonferroni correction across 23 cells.

**Panel F.** Scatter plot: distance from maze centre (cell (3,2)) vs visit
fraction delta for all 23 cells, coloured by node type (junction, corridor,
dead end). Spearman rho = 0.75 (p < 0.0001, N = 23). Shows that the
coverage reduction is concentrated at central maze locations, inverting
the expected peripheral-abandonment pattern.

**Statistics needed:**
- Spearman correlation of speed-diff vs coverage-diff (H1)
- Wilcoxon signed-rank for coverage by cell type in light vs dark,
  Holm-Bonferroni corrected across 3 cell types (H3)
- Wilcoxon signed-rank comparing dead-end drop magnitude vs junction
  drop magnitude (H3 interaction)
- Wilcoxon signed-rank for visited subgraph diameter (H3)
- Wilcoxon signed-rank for revisitation index (H4)
- Permutation test for JSD (1000 permutations, H2)
- Per-cell Wilcoxon signed-rank (N = 20 sessions per cell),
  Holm-Bonferroni corrected across 23 cells (H6)
- Spearman: distance from centre vs delta (N = 23 cells) (H6)
- Spearman: eccentricity vs delta for corridor cells (N = 7, descriptive) (H6)
- Effect sizes (rank-biserial) for all tests

**Code references:**
- Coverage by cell type: `maze.analysis.cell_occupancy()` with
  `maze.topology.node_types()`
- Revisitation index: custom (total transitions / unique cells)
- JSD: `scipy.spatial.distance.jensenshannon` on transition matrices
- Permutation null: shuffle light/dark epoch labels within session
- Per-cell visit fraction: custom per-cell epoch-level analysis
- Distance from centre: `maze.topology` shortest-path distances

---

### Figure 5: Epoch adaptation — single-trial strategy adjustment

**Panel A.** Dark-epoch coverage as a function of epoch number within each
session. Individual sessions as thin lines, session-mean as thick line.
No systematic trend is apparent after the first epoch (slope test: p = 0.81).

**Panel B.** First dark epoch vs subsequent dark epochs. Paired comparison
within each session (first: 0.57; subsequent mean: 0.30; Wilcoxon,
p = 0.0001, adjusted p = 0.0004, r = 0.89, N = 20). The discontinuity
between epoch 1 and epoch 2 is the dominant feature of the data.

**Panel C.** Light-epoch coverage as a function of epoch number. Gradual
decline over the session (slope test: p = 0.011), consistent with general
exploration habituation. Contrast with dark epochs, which show a step
change rather than a gradual decline.

**Panel D.** Speed across dark epochs. No systematic trend (p = 0.86,
N = 14), ruling out a locomotor explanation for the first-epoch effect.

**Statistics needed:**
- Wilcoxon signed-rank: first dark epoch coverage vs mean of subsequent
  dark epochs (N = 20 sessions), Holm-Bonferroni corrected within Family 6
- Wilcoxon one-sample: within-session Spearman rho of epoch number vs
  dark coverage, testing whether median slope differs from zero (N = 20)
- Wilcoxon one-sample: light-epoch slope control (N = 20)
- Wilcoxon one-sample: speed-epoch slope control (N = 14)
- Effect sizes (rank-biserial) for all tests

**Code references:**
- Epoch-level coverage: `maze.analysis.cell_occupancy()` per epoch
- Epoch numbering from sync.h5 light_on transitions
- Speed per epoch from kinematics.h5

---

### Figure 6: Light vs dark — speed and other metrics

**Panel A.** Running speed (cm/s) by condition: box/violin plot of
session-median speed in light vs dark. Paired by session. Note: data
show a trend (p = 0.076, uncorrected; primary-only p = 0.042).

**Panel B.** Speed time course across a session, with light/dark epochs
shaded. Examine whether any speed transitions occur at light changes.

**Panel C.** Fraction of time spent moving (speed > threshold) by condition.
Trend in same direction as speed (p = 0.083).

**Panel D.** Immobility bout duration (median seconds per bout) by
condition. Trend toward longer bouts in darkness (p = 0.035, adjusted
p = 0.106, r = 0.64).

**Panel E.** Transition entropy, dead-end rate, backtracking, and turn bias
in light vs dark — summary bar/violin showing all are non-significant.

**Statistics needed:**
- Wilcoxon signed-rank for speed, fraction active, immobility bouts,
  transition entropy, dead-end rate, backtracking, turn bias
  (light vs dark, N = 20 sessions)
- Effect sizes (rank-biserial correlation)
- Speed distributions (not just means): light vs dark, KS test or similar
- Speed at light-to-dark transitions: peri-event time histogram around
  light-off events (pooled across all transitions)

**Code references:**
- Speed and AHV from kinematics.h5
- Custom peri-event analysis for light transitions
- `light_on` mask from sync.h5

---

### Figure 7: Head direction sampling in the maze

**Panel A.** Per-cell HD distribution: polar histogram of HD angles when the
mouse occupies each of the 23 cells. Show that corridor cells have
strongly bimodal HD distributions (aligned with corridor axis) while
junction cells have more uniform distributions.
**[SYNTHETIC DATA PLACEHOLDER]** The current figure uses simulated
preferred directions generated for layout purposes. This panel must be
replaced with real HD distributions computed from actual sessions before
submission.
<!-- TODO(DS-agent): Replace Figure 7A with real per-cell HD distributions. Requires frame-level HD + position from sync.h5. Compute HD histogram per maze cell, pooled across sessions. -->

**Panel B.** HD sampling non-uniformity index (Rayleigh test statistic or
resultant vector length of the HD distribution) per cell, colour-coded on
the maze grid.

**Panel C.** HD sampling uniformity in light vs dark: does HD sampling become
more or less uniform in darkness? Paired comparison across cells and
sessions. Note: MRL trend (p = 0.076) does not survive correction or
primary-only analysis.

**Panel D.** Joint position x HD occupancy: heatmap showing which (cell, HD
bin) combinations are well-sampled and which are sparse. This is a control
figure for future HD tuning analyses -- it documents the sampling landscape.

**Panel E.** Angular head velocity (|AHV|) in light vs dark. Part of the
same Holm-Bonferroni family as MRL (Family 4). AHV does not differ between
conditions (p = 0.177, adjusted p = 0.177, r = 0.35).

**Statistics needed:**
- Rayleigh test per cell (is HD distribution non-uniform?)
- Mean resultant length of HD distribution per cell
- Light vs dark comparison of HD uniformity (Wilcoxon on per-cell Rayleigh
  statistics)
- Light vs dark comparison of |AHV| (Wilcoxon, N = 20)

**Code references:**
- `maze.discretize.discretize_position_fast()`
- HD and AHV from kinematics.h5
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

### Supplementary Figure S3: Within-dark-epoch temporal dynamics (H5)

**Panel A.** Cumulative unique cell curves (5-second bins) for light and
dark epochs, averaged across sessions (mean +/- SEM). Light and dark curves
diverge only modestly; the coverage ratio (second half / first half) trends
lower in dark than light (p = 0.064, adjusted p = 0.192, r = 0.48, N = 20)
but does not reach significance. The speed ratio does not differ between
conditions (p = 0.87), indicating that within-epoch slowing does not account
for the trend.

**Panel B.** Speed in early (0--30s) vs late (30--60s) dark epochs.

**Panel C.** Transition entropy in early vs late dark epochs.

**Panel D.** Backtracking rate in early vs late dark epochs.

Rationale: Tests whether behaviour degrades within individual dark epochs,
which would be expected if path integration drift causes progressive spatial
disorientation. The current data are inconclusive: a non-significant trend
toward gradual coverage reduction exists but cannot distinguish between
gradual representational degradation (H5a) and an immediate strategy switch
(H5b). Lights-on recovery data were insufficient to compute a recovery
statistic (N = 0 usable pairs). This analysis should be revisited with
longer recording sessions.

---

### Supplementary Figure S4: Individual differences in darkness sensitivity (H9)

**Panel A.** Per-animal scatter plot: mean coverage in light vs mean
coverage in dark. Animals classified as darkness-resistant (coverage drop
< 1 cell equivalent; N = 5), intermediate (N = 7), or darkness-sensitive
(coverage drop > 3 cells; N = 2). Lines connect each animal's light and
dark means.

**Panel B.** Coverage sensitivity vs speed sensitivity per animal (Spearman
rho = 0.74, p = 0.002, N = 14). Animals that slow down more in darkness
also lose more coverage, consistent with the session-level speed-coverage
coupling (H1).

**Panel C.** Coverage sensitivity by cell type (Penk+ vs non-Penk). All
4 non-Penk animals fall in the intermediate range. Both darkness-sensitive
animals and 3 of 5 darkness-resistant animals are Penk+. No cell-type
effect is apparent (individual variation dominates group variation), but
N = 4 non-Penk precludes strong conclusions.

Rationale: Documents the range of individual differences in darkness
sensitivity. The neural bridge analysis (correlating behavioural darkness
sensitivity with HD tuning stability) is deferred until the neural analysis
pipeline produces per-animal HD stability scores. Note: N = 14 animals is
small for individual-differences analyses; these results are descriptive
and should not be over-interpreted.

---

### Supplementary Figure S5: Cell-type Markov model (H10)

**Panel A.** Model order comparison at the cell-type level (3 states:
junction, corridor, dead-end). A second-order model is preferred over
first-order in all 20 sessions (mean delta-BIC = 103.2, Wilcoxon,
p < 0.001). This contrasts with the cell-level analysis (Section 2), where
first-order was always preferred. The maze's topology creates second-order
structure when transitions are collapsed to cell types: knowing both the
current cell type and the previous cell type (e.g., corridor preceded by
junction) provides additional predictive information about the next cell
type.

**Panel B.** Cell-type transition divergence between light and dark. The
Jensen-Shannon divergence of cell-type transition matrices is significant
(observed mean JSD = 0.0081 vs permutation null mean = 0.0031, permutation
p < 0.001). Specific second-order transition P(corridor | corridor,
junction) shows a trend toward reduction in darkness (light: 0.796; dark:
0.771; p = 0.027, adjusted p = 0.107, r = 0.56, N = 20), consistent with
fewer corridor-corridor sequences on the backbone. Other specific
transitions (P(dead-end | junction, corridor), P(junction | dead-end,
corridor)) did not differ significantly between conditions after correction.

Rationale: This analysis tests whether route stereotypy has structure at
the level of maze topology (cell types) in addition to specific cell
identities. The second-order preference at the type level, combined with
the first-order preference at the cell level, suggests that the maze's
topological constraints impose sequential dependencies that are only
apparent when individual cell identities are collapsed. The trending
reduction in backbone traversal probability is consistent with the route
stereotypy narrative but requires a larger dataset to confirm.

---

### Summary statistics table (Table 1)

*Note: Values shown are means across sessions (N = 20) unless otherwise noted. For speed and immobility bout duration, each session contributes its within-session median; the table reports the cross-session mean of those per-session medians. The Wilcoxon signed-rank test is computed on the 20 paired session-level values. p-values shown are Holm-Bonferroni-adjusted within each figure family. Figures display uncorrected p-values; adjusted values are reported here and in the main text. Primary-only column shows p from robustness check using one session per animal (N = 11).*

*Coverage per active minute and transition entropy are each the sole metric in their respective analysis families (post-hoc control and supplementary, respectively), so no family-wise correction is applicable (marked --).*

| Metric | Light (mean) | Dark (mean) | W | p (raw) | p (adj) | r | Primary-only p (N=11) |
|--------|-------------|------------|---|---------|---------|---|----------------------|
| Per-epoch coverage (frac) | 0.400 | 0.337 | 15.0 | 0.0003 | 0.0008 | 0.86 | 0.010 |
| Coverage / active min | 19.8 | 17.8 | 23.0 | 0.001 | --^a^ | 0.78 | 0.175 |
| Exploration efficiency (w=5) | 3.56 | 3.41 | 54.0 | 0.058 | 0.117 | 0.49 | -- |
| Speed (cm/s) | 2.28 | 1.97 | 57.0 | 0.076 | 0.152 | 0.46 | 0.042 |
| Fraction active | 0.472 | 0.441 | 58.0 | 0.083 | 0.152 | 0.45 | 0.042 |
| Immobility bout (s) | 0.84 | 0.94 | 19.0 | 0.035 | 0.106 | 0.64 | -- |
| HD mean resultant length | 0.060 | 0.085 | 57.0 | 0.076 | 0.152 | 0.46 | 0.278 |
| Median |AHV| (deg/s) | 93.3 | 95.4 | 68.0 | 0.177 | 0.177 | 0.35 | 0.465 |
| Dead-end rate (/min) | 7.60 | 8.33 | 74.0 | 0.261 | 0.261 | 0.30 | -- |
| Transition entropy (bits/step) | 1.221 | 1.186 | 67.0 | 0.165 | --^b^ | 0.36 | -- |
| Left turn fraction | 0.495 | 0.500 | 100.0 | 0.870 | 1.000 | 0.05 | -- |
| Backtracking rate | 0.482 | 0.505 | 92.0 | 0.648 | 1.000 | 0.12 | -- |
| *Route stereotypy metrics (Fig. 4)* | | | | | | | |
| Corridor coverage (frac) | 0.482 | 0.380 | -- | <0.00001 | <0.0001^c^ | 0.97 | -- |
| Junction coverage (frac) | 0.465 | 0.389 | -- | 0.0006 | 0.001^c^ | 0.82 | -- |
| Dead-end coverage (frac) | 0.286 | 0.262 | -- | 0.261 | 0.261^c^ | 0.30 | -- |
| Visited diameter (cells) | 6.35 | 5.57 | -- | 0.002 | -- | 0.80 | -- |
| Revisitation index | 3.39 | 3.88 | -- | 0.011 | -- | 0.64 | -- |
| JSD (transition matrix) | -- | -- | -- | <0.001^d^ | -- | -- | -- |
| Coverage / transition | 0.398 | 0.355 | -- | 0.076 | -- | 0.46 | -- |
| Distance-from-centre x delta | -- | -- | -- | <0.0001^e^ | -- | rho=0.75 | -- |
| *Epoch adaptation metrics (Fig. 5)* | | | | | | | |
| First dark epoch cov. (frac) | -- | 0.567 | -- | 0.0001 | 0.0004^f^ | 0.89 | -- |
| Subsequent dark epoch cov. (frac) | -- | 0.303 | -- | -- | -- | -- | -- |
| Dark cov. slope (rho vs epoch) | -- | -- | -- | 0.81 | 0.81 | 0.06 | -- |
| Light cov. slope (rho vs epoch) | -- | -- | -- | 0.011 | -- | -- | -- |
| *Within-epoch dynamics (Fig. S3)* | | | | | | | |
| Coverage ratio (2nd/1st half) | 0.282 | 0.207 | -- | 0.064 | 0.192 | 0.48 | -- |
| Speed ratio (2nd/1st half) | 0.948 | 0.945 | -- | 0.87 | 1.0 | 0.05 | -- |
| *Cell-type Markov (Fig. S5)* | | | | | | | |
| 2nd-order preferred (N sessions) | -- | -- | -- | <0.001^g^ | -- | -- | -- |
| Type-level JSD | -- | -- | -- | <0.001^d^ | -- | -- | -- |
| P(C\|C,J) | 0.796 | 0.771 | -- | 0.027 | 0.107 | 0.56 | -- |

^a^ Coverage per active minute is a post-hoc control analysis (single test, not part of any Holm-Bonferroni family). ^b^ Transition entropy is the sole light-dark comparison in Supplementary Figure S1 and is not corrected within any family. ^c^ Holm-Bonferroni corrected within Family 5 (corridor, junction, dead-end coverage; Fig. 4). ^d^ Permutation test (1000 permutations); observed mean JSD = 0.068 vs null mean = 0.018 (cell-level) or 0.0081 vs 0.0031 (type-level). ^e^ Spearman correlation across all 23 cells: distance from maze centre (3,2) vs visit fraction delta (dark - light); single test on all cells. ^f^ Holm-Bonferroni corrected within Family 6 (first-vs-rest, slope, early-vs-late; Fig. 5). ^g^ Wilcoxon one-sample on delta-BIC values; 20/20 sessions prefer second-order at cell-type level (Fig. S5). ^h^ Holm-Bonferroni corrected across 3 planned tests within H5 (coverage ratio, speed ratio, recovery). ^i^ Holm-Bonferroni corrected across 4 transition probabilities within the cell-type Markov analysis.

*Holm-Bonferroni correction families:* Family 1 (Fig. 3: coverage, dead-end rate, exploration efficiency); Family 2 (Fig. 6: speed, fraction active, immobility bout); Family 3 (Fig. 6: left turn fraction, turn autocorrelation vs zero, autocorrelation light vs dark, backtracking rate); Family 4 (Fig. 7: MRL, AHV); Family 5 (Fig. 4: corridor coverage, junction coverage, dead-end coverage); Family 6 (Fig. 5/epoch adaptation: first-vs-rest, slope test, early-vs-late); Family 7 (H5: coverage ratio, speed ratio, recovery); Family 8 (cell-type Markov: 4 transition probabilities).

---

## Draft Results Skeleton

### 1. Mice rapidly cover the q-rose maze

"Mice explored the 23-cell q-rose maze with high coverage. Across 20
sessions, mice visited a median of 22.5 cells (mean 22.1 +/- 1.3; range
18--23), achieving a mean coverage fraction of 0.961 +/- 0.056
(Fig. 1D). Total distance travelled varied considerably across sessions
(median 57.7 m; range 31.7--117.8 m), reflecting individual differences in
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
-0.172, one-sample Wilcoxon, W = 2.0, p < 0.0001, adjusted p < 0.0001,
r = 0.98, N = 20), indicating systematic left-right alternation consistent
with Rosenberg et al. (2021). To control for the possibility that maze
geometry alone produces turn alternation, we compared the observed
autocorrelation to a random walk null model (1000 simulated walks per
session on the maze graph). The null distribution had a negative mean
(-0.141), confirming that the maze topology contributes some alternation.
The observed alternation was stronger than the per-session null means
(Mann-Whitney U = 140.0, p = 0.108, Cliff's d = -0.30, 20 observed vs 20
per-session null means). Note that the Mann-Whitney test does not reach
significance at alpha = 0.05, so the between-group comparison alone does
not establish that observed alternation exceeds the topology-driven null.
However, bootstrap permutation testing confirmed that the observed
alternation exceeded the null (p = 0.019), and 3 of 20 sessions (15%)
fell below the null 95% CI. Mice therefore show spontaneous alternation
beyond what maze geometry alone would produce, though the effect size is
modest and the conclusion rests primarily on the permutation test and the
per-session outlier analysis rather than the Mann-Whitney comparison.

"Global left-right bias was minimal (left fraction: 0.50, not significantly
different from 0.5). No individual junction showed a significant left-right
bias after Holm-Bonferroni correction across 7 junctions (all adjusted
p > 0.6; Table S2).

"Backtracking (reversing direction at junctions) was frequent, accounting
for 48--51% of junction visits in both light and dark conditions. This rate
likely reflects a structural feature of the small q-rose maze: with 9
dead ends among 23 cells, mice frequently reach dead ends and must reverse
course. This contrasts with the larger labyrinth of Rosenberg et al.
(2021), where backtracking is less prominent because the maze graph offers
more through-routes.

"In contrast to Rosenberg et al. (2021), a first-order Markov model was
preferred over a second-order model in all 20 sessions by BIC (mean
delta-BIC = -4,434; 0/20 sessions favouring second-order; Fig. 2D).
This negative finding likely reflects the smaller state space of the q-rose
maze (23 cells, 7 junctions) compared to Rosenberg's 63-junction labyrinth:
with fewer possible transitions, a second-order model introduces many
additional parameters that are poorly estimable from a single session's
trajectory, and BIC's complexity penalty accordingly favours the simpler
model. Sequence entropy did decrease with increasing context length
(Fig. 2E), indicating that navigation is not memoryless, but the improvement
from additional context is modest and does not justify the second-order
model's parameter cost in this maze."

### 3. Spatial coverage decreases in darkness

"The removal of visual cues robustly reduced per-epoch spatial coverage.
Within individual 1-minute epochs, mice visited a smaller fraction of the 23
accessible cells in darkness than in light (light: 0.400; dark: 0.337;
Wilcoxon, W = 15.0, p = 0.0003, adjusted p = 0.0008, r = 0.86, N = 20;
Fig. 3A). This coverage reduction was the strongest light-dark effect
observed in the dataset and was robust to pseudoreplication control: the
effect remained significant in primary-only sessions (N = 11, p = 0.010,
r = 0.85).

"Running speed showed a trend toward reduction in darkness that partially
accounts for the coverage drop. The speed trend was not significant in the
full dataset (light mean: 2.28 cm/s; dark mean: 1.97 cm/s; Wilcoxon,
W = 57.0, p = 0.076, adjusted p = 0.152, r = 0.46, N = 20; Fig. 6A) but
reached significance in primary-only sessions (N = 11, p = 0.042, r = 0.70).
To control for this speed confound, we normalised coverage by active time
(minutes with speed >= 2.5 cm/s). Coverage per active minute was also
significantly lower in darkness (light mean: 19.8 cells/active-min; dark
mean: 17.8 cells/active-min; Wilcoxon, W = 23.0, p = 0.001, r = 0.78,
N = 20),
indicating that the coverage reduction is not simply a locomotor artefact
but reflects a genuine change in exploration efficiency. However, the
primary-only analysis did not reach significance for this normalised metric
(N = 11, p = 0.175, r = 0.49), so the extent to which the coverage drop
reflects reduced speed versus a true strategy change cannot be definitively
resolved with the current sample.

"Exploration efficiency (unique nodes per sliding window of 5 cell
transitions) showed a consistent trend in the same direction (light: 3.56;
dark: 3.41; W = 54.0, p = 0.058, adjusted p = 0.117, r = 0.49; Fig. 3C),
providing converging evidence for reduced spatial exploration in darkness,
though this metric did not reach significance after correction.

"Immobility bout duration showed a trend toward longer bouts in darkness
(light mean: 0.84 s; dark mean: 0.94 s; W = 19.0, p = 0.035, adjusted
p = 0.106, r = 0.64), consistent with increased cautiousness or pausing behaviour
during visual cue removal, though this did not survive multiple comparisons
correction."

### 3b. The coverage drop reflects route stereotypy

"The coverage reduction in darkness could arise from multiple mechanisms:
fewer cell transitions due to reduced speed, a shift in routing patterns,
spatial range contraction, or increased revisitation of already-covered
cells. We tested these hypotheses systematically (Fig. 4).

"Speed and per-epoch coverage were strongly correlated across sessions
(Spearman rho = 0.76, p < 0.0001, N = 20; Fig. 4A), confirming that
locomotor activity contributes substantially to the coverage difference.
However, coverage per transition (unique cells visited divided by total
cell-to-cell transitions) showed a trend toward reduced exploration
efficiency per decision in darkness (light: 0.398; dark: 0.355; Wilcoxon,
p = 0.076, r = 0.46, N = 20), suggesting that the coverage drop is not
fully explained by reduced locomotion.

"To determine where in the maze coverage was lost, we computed per-epoch
coverage separately for corridors, junctions, and dead ends (Fig. 4B).
Corridor coverage showed the largest reduction in darkness (light: 0.482;
dark: 0.380; Wilcoxon, p < 0.00001, adjusted p < 0.0001, r = 0.97, N = 20).
Junction coverage also decreased (light: 0.465; dark: 0.389; p = 0.0006,
adjusted p = 0.001, r = 0.82). In contrast, dead-end coverage was unchanged
between conditions (light: 0.286; dark: 0.262; p = 0.26, r = 0.30). The
magnitude of the dead-end coverage drop was significantly smaller than the
junction coverage drop (Wilcoxon on paired differences, p = 0.0004,
r = 0.84). The diameter of the visited subgraph -- a measure of the spatial
extent of the explored region -- also contracted in darkness (light: 6.35
cells; dark: 5.57 cells; p = 0.002, r = 0.80). This pattern indicates that
mice maintained visits to terminal destinations (dead ends) at unchanged
rates but consolidated onto a reduced set of connecting corridors and
junctions -- a pattern we term route stereotypy.

"Consistent with route consolidation, the revisitation index (total
cell-to-cell transitions divided by the number of unique cells visited)
increased in darkness (light: 3.39; dark: 3.88; Wilcoxon, p = 0.011,
r = 0.64, N = 20; Fig. 4C), indicating that mice traversed the same cells
more repeatedly in darkness. The discovery AUC (area under the cumulative
unique-cell discovery curve, normalised by the number of transitions)
showed a trend in the same direction (light: 0.297; dark: 0.273; p = 0.058,
r = 0.49), converging with the revisitation finding.

"To test whether routing patterns themselves changed, we computed the
Jensen-Shannon divergence (JSD) between the light-epoch and dark-epoch
first-order transition matrices for each session (Fig. 4D). The observed
mean JSD (0.068) was 3.9 times larger than the permutation null
distribution obtained by shuffling epoch labels within sessions (null mean:
0.018; 1000 permutations; p < 0.001). However, no individual transition
(edge) survived Holm-Bonferroni correction across the 44 edges tested
(0/44 significant), indicating that the routing change is distributed
across many small shifts rather than concentrated at a few junctions. This
result resolves an apparent paradox: transition entropy is preserved between
conditions (Section 4), yet the actual routes used differ. The transition
matrix changes in its pattern but not in its predictability.

"To determine whether the coverage reduction was spatially uniform or
concentrated at specific maze locations, we computed the per-cell change in
visit fraction between light and dark epochs across all 23 cells (Fig. 4E).
Six cells showed significant visit reductions after Holm-Bonferroni
correction across 23 cells (adjusted p < 0.05):
(1,0) (adjusted p = 0.045, r = 0.77),
(2,2) (adjusted p = 0.040, r = 0.82),
(3,2) (adjusted p = 0.006, r = 0.93),
(3,3) (adjusted p = 0.006, r = 0.96),
(4,2) (adjusted p = 0.014, r = 0.87),
and (5,2) (adjusted p = 0.036, r = 0.80).
In contrast, no dead-end cell showed a significant change.
The magnitude of the visit reduction correlated with distance from the maze
centre (Spearman rho = 0.75, p < 0.0001, N = 23): cells closer to the
centre of the maze lost more visits than peripheral cells (Fig. 4F). Among
corridor cells specifically, eccentricity (maximum shortest-path distance
to any other cell in the graph) correlated with coverage change (rho = 0.87,
p = 0.012, N = 7), though this correlation should be treated as descriptive
given the small sample. This pattern inverts the expectation from classical
range contraction studies, where peripheral locations are abandoned first
(Avni et al. 2006; Fonio et al. 2009). Instead, mice in darkness
selectively reduce visits to the central backbone of the maze while
maintaining visits to peripheral dead ends, consistent with a simplification
of the route network at its most connected hub.

"Taken together, these results indicate that the coverage reduction in
darkness reflects a spatially structured reorganisation of the mouse's
route network rather than a simple global slowdown. Mice continue to visit
dead-end destinations at unchanged rates but travel between them via fewer,
more repetitive routes, with the largest reductions at central backbone
locations. This route stereotypy is accompanied by increased revisitation
and a distributed but significant change in routing patterns."

### 3c. Route stereotypy is established after a single dark epoch

"The route stereotypy finding raises a question about temporal dynamics:
does the coverage reduction develop gradually across repeated dark epochs
(suggesting learning or progressive disorientation), or is it established
rapidly?

"We examined dark-epoch coverage as a function of epoch number within each
session. There was no gradual trend in dark-epoch coverage across the
session: the median within-session Spearman correlation between epoch number
and coverage was not different from zero (median rho = -0.056, Wilcoxon,
p = 0.81, r = 0.06, N = 20), and early dark epochs (first third) did not
differ from late dark epochs (last third) in their coverage deficit relative
to the preceding light epoch (p = 0.39, r = 0.22, N = 20; Fig. 5A). Speed
likewise showed no trend across dark epochs (median speed-epoch rho = 0.075,
p = 0.86, N = 14 sessions with sufficient epoch count), ruling out a
locomotor fatigue explanation.

"However, the first dark epoch in each session showed dramatically higher
coverage than all subsequent dark epochs (first: 0.57; subsequent mean:
0.30; Wilcoxon, p = 0.0001, adjusted p = 0.0004, r = 0.89, N = 20;
Fig. 5B). After the first dark epoch, dark coverage was constant across
all remaining dark epochs. Light-epoch coverage showed a gradual decline
over the session (median rho = -0.21, Wilcoxon, p = 0.011, N = 20),
consistent with general exploration habituation, but no corresponding
trend was present in dark epochs beyond the first.

"This pattern indicates that route stereotypy is not a gradual adaptation
but is established after a single experience of darkness. The first dark
epoch, during which the mouse encounters the loss of visual cues for the
first time in the session, produces near-normal coverage, suggesting that
the mouse initially navigates using a spatial representation inherited from
the preceding light epoch. By the second dark epoch, coverage has dropped
to its sustained low level, implying that the mouse has adjusted its
exploration strategy in response to the first dark experience. This is
consistent with single-trial learning -- the mouse detects that darkness
degrades navigation and rapidly adopts a more conservative route network."

### 4. Local navigation rules are preserved in darkness

"Other exploration metrics did not differ significantly between conditions.
Transition entropy was similar in light and dark (light: 1.221 bits/step;
dark: 1.186 bits/step; W = 67.0, p = 0.165, r = 0.36), indicating that
navigation predictability was unaffected by visual cue removal. Dead-end
visit rate did not differ between conditions (light: 7.60 visits/min;
dark: 8.33 visits/min; W = 74.0, p = 0.261, r = 0.30). Backtracking rate
was also unchanged (light: 0.482; dark: 0.505; W = 92.0, p = 0.648,
adjusted p = 1.000, r = 0.12). Turn bias (left fraction) did not differ
(light: 0.495; dark: 0.500; W = 100.0, p = 0.870, r = 0.05), and
sequential turn alternation strength was similar between conditions (light
mean autocorrelation = -0.158; dark = -0.170; W = 92.0, p = 0.648,
adjusted p = 1.000, r = 0.12).

"Angular head velocity (|AHV|) did not differ between conditions (light
mean: 93.3 deg/s; dark mean: 95.4 deg/s; Wilcoxon, W = 68.0, p = 0.177,
adjusted p = 0.177, r = 0.35; Fig. 7E). The primary-only analysis also
showed no effect (N = 11, p = 0.465, r = 0.27). The absence of a
significant AHV difference in the present data contrasts with the
vestibular-visual integration findings of Keshavarzi et al. (2022), who
showed that visual input modulates AHV coding gain in RSP. This discrepancy
may reflect the different measurement scales (behavioural AHV vs neural
AHV coding), the constrained locomotion in the small maze, or insufficient
power to detect a subtle effect.

"Taken together, the preservation of local navigation rules alongside the
route stereotypy described in Section 3b indicates that visual cue removal
has a selective effect on maze navigation. The local decision rules
governing turn direction, alternation, and backtracking are maintained, but
the global route network contracts: mice visit the same destinations via
fewer connecting paths. This dissociation suggests that local turn
decisions and global route selection draw on different information sources."

### 5. Head direction is constrained by maze geometry

"The overall distribution of head direction angles was non-uniform in both
conditions. The mean resultant length of the session-wide HD distribution
showed a non-significant trend toward higher values in darkness (light:
0.060; dark: 0.085; Wilcoxon, W = 57.0, p = 0.076, adjusted p = 0.152,
r = 0.46, N = 20; Fig. 7C). This trend did not survive the primary-only
robustness check (N = 11 independent animals; p = 0.278, r = 0.39). The MRL
difference is therefore suggestive but not robust to pseudoreplication
control or multiple comparisons correction with the current sample size.

"The non-uniformity of the HD distribution must be interpreted with caution.
The maze geometry constrains body orientation: corridor cells impose
approximately bimodal HD distributions aligned with the corridor axis,
while dead-end cells produce unimodal distributions toward the approach
direction. The observed non-uniformity therefore reflects a combination of
maze geometry, position-dependent sampling, and any true changes in
exploratory head movements. In particular, the trend toward increased mean
resultant length in darkness could arise from mice spending more time in
corridors (which impose stronger directional constraints) or from increased
time spent immobile (maintaining a fixed heading) rather than from a genuine
change in directional preference. The non-significant speed and activity
trends make this explanation plausible.

"This position-dependent HD sampling is a methodological concern for all
HD tuning analyses conducted in structured environments (as opposed to
open fields). Apparent neural HD selectivity could partially reflect
position-dependent sampling rather than true directional tuning. We
provide per-cell HD occupancy maps (Fig. 7D) to enable occupancy-corrected
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
backtracking rate is moderate (48--51% of junction visits), a structural
consequence of the maze having 9 dead ends among 23 cells, which forces
frequent reversals. In Rosenberg's larger labyrinth with more through-routes,
backtracking was less prominent. The absence of reward in our task means
there is no target bias component; exploration is intrinsically motivated.

### Light/dark effects in context

The most robust light-dark difference was in per-epoch spatial coverage,
which was lower in darkness (p = 0.0003, r = 0.86, N = 20; primary-only
p = 0.010, r = 0.85, N = 11). This effect survived all robustness checks
and represents the primary behavioural finding. Coverage per active minute
was also lower in darkness (p = 0.001, r = 0.78), indicating that the
effect is not simply a locomotor artefact.

Speed reduction in darkness is a well-documented phenomenon in rodents
exploring open arenas (Whishaw & Tomie 1996; Avni et al. 2006; Fonio et al.
2009). In the present data, speed showed a
trend toward reduction (p = 0.076) that reached significance only in
primary-only sessions (N = 11, p = 0.042, r = 0.70). This partial effect
suggests that reduced locomotion contributes to the coverage drop but does
not fully account for it, since coverage per active minute also decreases.
The limited dynamic range of locomotion imposed by the head-mounted
microscope tether and the small maze may attenuate the speed difference
compared to open-field studies.

Angular head velocity did not differ significantly between conditions
(p = 0.177). This null result is notable given that Keshavarzi et al. (2022)
demonstrated that visual input modulates AHV coding gain in RSP. The
discrepancy may reflect different measurement levels (behavioural AHV vs
neural AHV coding), the constrained movement in the maze, or insufficient
power. The AHV null should be interpreted cautiously rather than taken as
evidence against visual modulation of head movements.

The 1-minute epoch duration is relevant to the HD literature: HD drift in
darkness accumulates gradually (Stackman & Taube 1997), with ~40% of HD
cells becoming unstable within minutes (Muir et al. 2022). The supplementary
within-dark-epoch analysis (Fig. S3) tests whether behavioural metrics
degrade within individual dark epochs, which would be expected if
progressive spatial disorientation accompanies HD drift.

### Route stereotypy as the mechanism underlying coverage reduction

The coverage reduction in darkness is not explained by a simple global
slowdown or by disruption of local navigation rules. Instead, mice maintain
visits to dead-end destinations at unchanged rates but consolidate onto a
reduced set of connecting corridors and junctions -- route stereotypy. This
is accompanied by increased revisitation of already-covered cells and a
distributed but significant reorganisation of the transition matrix. The
pattern is distinct from classic range contraction toward a central refuge
(Avni et al. 2006; Fonio et al. 2009): the mouse does not retreat to a safe
location but reduces path diversity while preserving its destination
repertoire. This dissociation suggests that local turn decisions and global
route selection rely on different information sources -- turn alternation may
be supported by egocentric strategies that operate without visual landmarks,
while route selection across the maze graph may depend on a spatial
representation that degrades without visual cues.

The per-cell analysis reveals that the coverage reduction is
topographically structured: cells closer to the maze centre show the
largest visit reductions (rho = 0.75, p < 0.0001), with the central
T-junction (3,2) and adjacent backbone cells showing the most significant
drops. This inverts the classical range contraction pattern (Avni et al.
2006; Fonio et al. 2009), in which peripheral locations are abandoned
first. In the q-rose maze, peripheral dead ends are maintained while the
central backbone is simplified. This pattern is consistent with the route
stereotypy interpretation: the central backbone serves as a transit hub
connecting multiple branches, so when the mouse restricts its route
network, the hub cells that served alternative routes are the ones most
affected. The remaining routes still reach the same dead-end destinations,
but via fewer central transit points. A supplementary analysis using
cell-type Markov models (collapsing the 23 individual cells to 3 types:
junction, corridor, dead-end) revealed that second-order models are
preferred at this level (mean delta-BIC = 103; 20/20 sessions), in
contrast to the first-order preference at the individual-cell level.
This indicates that the maze topology imposes sequential constraints on
transitions between cell types that are diluted across 23 individual
states. The probability of sustained backbone traversal (P(corridor |
corridor, junction)) showed a trend toward reduction in darkness
(p = 0.027, adjusted p = 0.107), consistent with fewer extended backbone
traversals, though this did not survive correction.

The speed-coverage correlation (rho = 0.76) is substantial, and reduced
locomotion undoubtedly contributes to the coverage drop. We do not claim
that route stereotypy is independent of speed. Rather, the cell-type
dissociation (preserved dead-end coverage alongside reduced corridor and
junction coverage) and the increased revisitation index indicate that the
coverage reduction has a spatial structure that a uniform speed reduction
would not produce. A uniform slowdown would reduce coverage proportionally
across all cell types; the selective loss of corridor and junction coverage
with preserved dead-end coverage points to a change in route selection
rather than a simple reduction in the number of steps taken.

The transition matrix analysis provides a complementary perspective. Despite
preserved transition entropy (the overall predictability of the next step
given the current position), the actual routes used in light and dark
conditions differ significantly (JSD = 0.068, p < 0.001 vs permutation
null). This indicates that the routing change is not a loss of structure
but a shift to a different, equally structured routing pattern --
consistent with consolidation onto a subset of familiar routes rather than
disorientation.

### Single-trial adaptation to darkness

The temporal dynamics of the coverage reduction provide additional
constraint on its mechanism. The first dark epoch in each session produces
near-normal coverage (0.57 vs 0.40 in light), while all subsequent dark
epochs stabilise at a lower level (0.30; first vs rest: p = 0.0001,
r = 0.89). There is no gradual improvement or worsening across later dark
epochs (slope p = 0.81), and speed does not change across dark epochs
(p = 0.86), ruling out locomotor fatigue. Light-epoch coverage does decline
gradually over the session (p = 0.011), consistent with general exploration
habituation, but the dark-epoch pattern is qualitatively different: a step
change rather than a gradual decline.

This single-trial adaptation is noteworthy for several reasons. First, it
suggests that the mouse navigates the first dark epoch using a spatial
representation carried over from the preceding light epoch, which has not
yet degraded sufficiently to alter behaviour. By the second dark epoch, the
mouse has learned that darkness degrades its navigation and adopts a more
conservative strategy. This is consistent with the timescale of HD drift
in darkness: Stackman & Taube (1997) showed that HD preferred directions
drift over 1--3 minutes in darkness, and Ajabi et al. (2023) demonstrated
coherent population drift on similar timescales. The first 60-second dark
epoch may therefore be short enough that the inherited representation
remains usable, but the mouse detects accumulated error and adjusts for
subsequent epochs.

Second, the single-trial nature of the adaptation argues against a purely
reflexive anxiogenic response to darkness. If the coverage reduction
reflected an immediate anxiety response to darkness onset (as in the
light-dark box paradigm; Bourin & Hascoet 2003), it should be present from
the first dark epoch. The fact that the first dark epoch produces
near-normal exploration indicates that the mouse is not immediately
cautious in darkness; rather, it discovers through experience that its
navigation is degraded and adjusts accordingly. However, we cannot exclude
the possibility that the mouse has some initial caution that is offset by
novelty-driven exploration during the first dark epoch, with the net
result being near-normal coverage.

Third, the within-dark-epoch dynamics were inconclusive. The coverage ratio
(second half / first half of each dark epoch) showed a non-significant
trend toward lower values in dark than light epochs (p = 0.064, adjusted
p = 0.192), and the speed ratio did not differ (p = 0.87). These data
cannot distinguish between gradual representational degradation within each
dark epoch and an immediate strategy switch at lights-off. Longer dark
epochs or higher temporal resolution would be needed to resolve this
question. The lights-on recovery analysis was not feasible due to
insufficient data (N = 0 usable epoch pairs with initial light coverage
data).

### HD sampling confound for neural analyses

The non-uniform HD sampling in maze corridors is a methodological concern
for all HD tuning analyses conducted in structured environments (as opposed
to open fields). This has been acknowledged in the literature (Muir et al.
2022; Jacob et al. 2017) but is rarely quantified explicitly. Our per-cell
HD occupancy maps (Fig. 7D) provide the basis for occupancy-corrected tuning
curve estimation in the companion neural paper.

### HD non-uniformity: position and immobility confounds

The trend toward increased mean resultant length of the HD distribution in
darkness (p = 0.076, r = 0.46) has multiple possible explanations beyond a
change in exploration strategy. The main analysis already restricts to
active frames (speed >= 2.5 cm/s), so immobility per se does not drive the
effect. However, if mice in darkness spend proportionally more time in
corridors (which impose bimodal HD distributions) rather than at junctions
(which allow more uniform heading), the overall HD distribution will appear
more concentrated. The MRL by node type control (deferred; requires
frame-level data) would address this confound. The MRL trend does not
survive the primary-only robustness check (N = 11, p = 0.278, r = 0.39)
or Holm-Bonferroni correction (adjusted p = 0.152), so the effect should
be considered a non-significant trend rather than an established finding.

### Limitations

1. The head-mounted microscope tether restricts movement to some degree.
   While we exclude periods of clear tether entanglement, subtle motor
   constraints may still influence exploration patterns compared to
   untethered mice. The tether may also contribute to the modest speed trend
   by limiting the dynamic range of locomotion available in both conditions.

2. With 20 sessions from 14 animals (4 animals contributing 2--3 sessions
   each), session-level analyses involve mild pseudoreplication. We address
   this with a primary-only robustness check (N = 11 independent animals),
   but the reduced sample size limits statistical power. The core coverage
   finding survives this control (p = 0.010), and speed reaches significance
   in primary-only (p = 0.042), but coverage per active minute does not
   (p = 0.175), and MRL does not (p = 0.278). AHV remains non-significant
   in both analyses.

3. The 1-minute light/dark epoch duration was chosen for neural imaging
   purposes (testing HD re-anchoring dynamics) rather than to optimise
   behavioural measurements. Longer dark epochs might reveal more
   pronounced exploration changes.

4. This is a free-exploration paradigm with no explicit task demands. The
   behavioural metrics quantify exploration strategy, not task performance.
   Goal-directed navigation metrics (path efficiency, monotonic paths) are
   computed relative to dead ends as surrogate targets, not experimentally
   defined goals.

5. The coverage difference between light and dark may be partially
   confounded by locomotor activity. Although speed showed only a
   non-significant trend in the full dataset (p = 0.076), the trend
   reached significance in primary-only sessions (p = 0.042). A control
   analysis normalising coverage by active time (coverage per active
   minute) confirmed that the effect survives locomotor normalisation in
   the full dataset (p = 0.001) but not in primary-only sessions
   (p = 0.175). The coverage finding is therefore robust in its basic form
   (raw coverage, primary-only p = 0.010), but whether it reflects a
   strategy change, a locomotor reduction, or both cannot be definitively
   resolved with the current sample.

6. Speed analysis by node type (Supplementary, not yet assigned to a
   figure) was computed using only active
   frames (speed >= 2.5 cm/s), which biases results at locations where
   mice frequently stop (e.g., dead ends, where they pause before
   reversing). The finding that dead ends show the highest active speed may
   be an artefact of this filtering, since it selects only the moments of
   acceleration out of dead ends while excluding the (potentially long)
   pauses. A control analysis using all frames (including immobile periods)
   is needed to assess whether the node-type speed differences are robust.

7. Three control analyses require frame-level data from regenerated
   sync.h5 files and are deferred until the pipeline re-run completes:
   (a) MRL by maze node type (junction vs corridor vs dead end) in light
   vs dark, which would determine whether the HD concentration trend in
   darkness is driven by differential maze-location occupancy;
   (b) speed by node type using all frames (including immobile periods),
   which would test whether the dead-end speed result from the active-only
   analysis is an artefact of the activity filter; and (c) per-bodypart
   tracking confidence by light condition, which would provide a more
   granular check on tracking quality than the aggregate statistics
   reported in Methods.

8. The route stereotypy findings (H3, H4) have not yet been subjected to
   the primary-only robustness check (N = 11 independent animals). Given
   the large effect sizes for corridor coverage (r = 0.97) and junction
   coverage (r = 0.82), these are likely to survive, but this must be
   confirmed before submission. The revisitation index (r = 0.64) and
   coverage-per-transition trend (p = 0.076) have more modest effect sizes
   and may not survive the reduced sample.

9. The total distance values reported here (median 57.7 m) are
   substantially lower than preliminary estimates from an earlier tracking
   model (median ~106 m). The earlier values were inflated by tracking
   jitter (noisy keypoint estimates producing spurious inter-frame
   displacements). The current values are derived from an improved
   DLC model (test RMSE 3.79 px vs 7.12 px) and are more reliable.

10. The per-cell analysis (H6) involves 23 simultaneous Wilcoxon tests
    corrected with Holm-Bonferroni. The eccentricity correlation among
    corridor cells (rho = 0.87, p = 0.012, N = 7) should be treated as
    descriptive given the small sample: with N = 7, Spearman requires
    rho > 0.79 for significance at alpha = 0.05 (two-tailed), and the
    correlation is not robust to removal of any single point.

11. The first-epoch finding (H8) is robust (p = 0.0001, r = 0.89) but its
    interpretation depends on the assumption that the first dark epoch is
    comparable to subsequent ones in all respects except novelty. An
    alternative explanation is that the first dark epoch benefits from
    residual spatial memory that degrades with time, not with darkness per
    se. However, the absence of a gradual decline across subsequent dark
    epochs (slope p = 0.81) argues against progressive degradation.

12. Individual differences in darkness sensitivity (H9) are documented but
    should be interpreted cautiously given the small sample (N = 14 animals,
    with 4 non-Penk). The classification into sensitivity groups (resistant/
    intermediate/sensitive) is descriptive and arbitrary. No cell-type
    effect was apparent, but this contrast is underpowered (N = 10 vs N = 4).

13. The within-epoch temporal dynamics analysis (H5) was inconclusive. The
    coverage ratio trend (p = 0.064) does not survive correction (adjusted
    p = 0.192) and the lights-on recovery control could not be computed.
    This question requires either longer dark epochs or higher temporal
    resolution to resolve.

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

2. **Central backbone simplification inverts classical range contraction.**
   The per-cell analysis shows that cells closest to the maze centre lose
   the most visits in darkness (rho = 0.75, p < 0.0001), the opposite of
   the peripheral abandonment predicted by classical range contraction
   (Avni et al. 2006). This structural specificity strengthens the route
   stereotypy interpretation: the mouse simplifies the hub of its route
   network, not its periphery.

3. **Single-trial adaptation to darkness.** The first dark epoch produces
   near-normal coverage; all subsequent dark epochs show the reduced-coverage
   pattern (first vs rest: p = 0.0001, r = 0.89). This single-trial
   learning dynamic has not been characterised in maze navigation, and it
   constrains the mechanism: the coverage reduction is not a reflexive
   response to darkness onset but a learned adjustment after the first
   experience of navigating without visual cues.

4. **The q-rose maze under light/dark alternation is a new paradigm.** The
   original Rosenberg maze used fixed lighting. No prior study has combined
   a binary-choice labyrinth with total darkness manipulation and quantified
   the effect on exploration strategy.

5. **Graph-theoretic behavioural characterisation during light/dark
   alternation.** Transition entropy, dead-end visit dynamics, and Markov
   model statistics have not been compared between light and dark conditions
   in a structured maze. The finding that most navigation metrics are
   unchanged by darkness -- despite a significant reduction in spatial
   coverage -- is itself informative.

6. **HD sampling characterisation in a structured maze.** This confound is
   acknowledged but rarely quantified. Explicit per-cell HD occupancy maps
   are a useful methodological contribution.

7. **Honest reporting of null results.** The null AHV result, the
   non-significant MRL trend, the null speed result (which becomes
   significant only in primary-only analysis), and the failure of the
   second-order Markov model are all informative for future studies using
   similar paradigms.

### What is NOT novel

- Turn alternation in maze exploration (Rosenberg et al. 2021).
- Speed reduction in darkness is well-established in the literature,
  though it was only a trend in the present data (p = 0.076).
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

3. **Speed confound for exploration metrics.** Speed showed a trend toward
   reduction in darkness (p = 0.076; primary-only p = 0.042), and speed and
   coverage are strongly correlated across sessions (Spearman rho = 0.76,
   p < 0.0001). Any trend toward slower speed means fewer cell transitions
   per minute. Transition entropy is rate-normalised (bits per step), but
   coverage and dead-end visit rate must be normalised by active time or
   number of transitions, not clock time. The coverage-per-active-minute
   control (p = 0.001, N = 20) and the coverage-per-transition trend
   (p = 0.076, r = 0.46) suggest that the coverage reduction is not fully
   explained by locomotion, and the cell-type dissociation (preserved
   dead-end coverage with reduced corridor/junction coverage) is not
   predicted by a uniform speed reduction. However, the primary-only
   analysis does not reach significance for coverage per active minute
   (p = 0.175, N = 11), so the speed confound cannot be definitively
   ruled out.

4. **Small maze ceiling effects.** With only 23 cells, coverage approaches
   100% quickly, limiting the dynamic range for exploration efficiency
   comparisons.

5. **Animal sex.** One female (1118023) was excluded in this analysis round
   (exp 21), so the remaining 14 animals are all male. This removes the
   sex confound but limits generalisability.

6. **Animals with multiple sessions.** Some animals contribute 2--3
   sessions. For session-level statistics, this creates mild
   pseudoreplication. Report results both with all sessions and with only
   one session per animal (primary).

### Reviewer objections to anticipate

- "This is descriptive with no neural data — what is the contribution?"
  Response: Establishes the paradigm for the companion neural paper and
  provides quantitative behavioural baselines.

- "With only 7 junctions, is the Markov model meaningful?" Response: The
  data show that a first-order model is preferred in all 20 sessions.
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
  was confirmed; the camera uses infrared illumination invisible to mice;
  two-photon laser at 920 nm is also invisible.

- "The speed result is only a trend — can you really claim it explains
  coverage?" Response: We present speed as a partial explanation, not a
  full one. The coverage-per-active-minute control shows the coverage drop
  persists after normalisation (p = 0.001), though this control does not
  survive primary-only analysis (p = 0.175). We are transparent about this
  ambiguity.

- "Route stereotypy could just be a speed artefact — slower mice traverse
  fewer corridors." Response: The speed-coverage correlation (rho = 0.76)
  is honestly reported. However, a uniform speed reduction would reduce
  coverage proportionally across all cell types. The selective preservation
  of dead-end coverage alongside reduced corridor and junction coverage is
  not predicted by a uniform slowdown. Coverage per transition (unique
  cells / total transitions) also shows a trend toward reduced efficiency
  per decision (p = 0.076), though this does not reach significance.

- "Is the JSD permutation test appropriate? You shuffled epoch labels, not
  transitions." Response: The permutation shuffles light/dark labels of
  entire epochs within each session (preserving the temporal structure of
  transitions within epochs), then recomputes the transition matrix for
  each permuted condition. This tests whether the observed divergence
  exceeds what would be expected from random partitioning of the same
  behavioural data.

- "The first-epoch finding could be a novelty effect, not a darkness
  effect." Response: Each session begins with a light epoch, so the first
  dark epoch is always the second epoch of the session. The mouse has
  already been in the maze for ~1 minute under illumination before
  darkness begins. The near-normal coverage in the first dark epoch
  therefore reflects continued exploration from a familiar state, not
  initial novelty. Furthermore, the first dark epoch is distinguishable
  from the first light epoch in that it represents a within-session
  condition change, not a session onset effect.

- "The per-cell analysis has massive multiple comparisons (23 tests). How
  many would you expect to be significant by chance?" Response: Under
  Holm-Bonferroni at alpha = 0.05, we expect <1 false positive. Six cells
  survived correction, with the strongest showing r = 0.93--0.97. The
  spatial gradient (distance-from-centre correlation, rho = 0.75) was
  computed on all 23 cells without multiple comparisons issues, as it is
  a single test of spatial structure.

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
   minute is significantly lower in dark (p = 0.001, r = 0.78, N = 20),
   but the primary-only analysis does not reach significance (p = 0.175,
   r = 0.49, N = 11). See Control 1 in behaviour-control-summary.md.

7. **Speed by node type without active-only filter.** The current analysis
   uses speed >= 2.5 cm/s threshold, which biases results. Repeat the
   speed-by-node-type analysis using all frames to check whether dead
   ends are genuinely
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
