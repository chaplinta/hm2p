# Maze Exploration and RSP Neural Activity: Working Notes

Working document for ideas connecting rose maze exploration behaviour to RSP
neural population activity. Scientific notes, not polished prose.

---

## 1. The Rosenberg Markov Model and How It Maps to This Maze

### The original study

Rosenberg, Zhang, Perona & Meister (2021). "Mice in a labyrinth show rapid
learning, sudden insight, and efficient exploration." eLife 10, e66175.
doi:10.7554/eLife.66175

Rosenberg et al. studied mice navigating a large binary labyrinth (63
T-junctions, 64 dead ends, 6 levels of binary branching). Key analysis
framework:

1. **Node trajectory**: Collapse continuous position to a sequence of
   junction/dead-end visits, removing corridor transits. This is already
   implemented in `maze/discretize.py::node_sequence()`.

2. **Markov transition models**: Fit first- and second-order Markov chains to
   the node visit sequence. First-order: P(next_node | current_node).
   Second-order: P(next_node | prev_node, current_node). Compare via
   AIC/BIC. Already implemented in `maze/analysis.py`.

3. **Four-bias random walk model**: Decompose navigation into four
   independent biases: (a) forward bias (momentum), (b) left-right bias,
   (c) inward-outward bias (toward/away from maze center), (d) target bias
   (toward rewarded location). The combined model explained ~66% of
   variance in port visits.

4. **Exploration efficiency**: Count new distinct nodes discovered per
   sliding window of visits (NewNodes4). Already implemented.

5. **Conditional entropy**: H(X_n | X_1, ..., X_{n-1}) for increasing
   context lengths. Lower entropy = more predictable/stereotyped
   navigation. Already implemented in `maze/analysis.py::sequence_entropy()`.

6. **Monotonic path detection**: Find runs where graph distance to a target
   decreases at every step, indicating goal-directed behaviour. Already
   implemented.

### Mapping to the rose maze

The rose maze is much smaller than Rosenberg's labyrinth: 23 accessible
cells, 7 T-junctions, 9 dead ends, 7 corridors, 0 crossroads. Maximum
graph diameter is ~9 steps. This has important consequences:

**Advantages:**
- Complete coverage is achieved quickly (most mice visit all 23 cells within
  minutes), making within-session exploration dynamics tractable.
- The small state space means transition matrices are estimable even from
  short epochs (1-minute light/dark periods yield ~50-150 node transitions
  depending on speed).
- The topology is simple enough that all paths are short, so "goal-directed"
  vs "exploratory" classification is more constrained.

**Limitations:**
- With only 7 T-junctions, junction-level Markov models have very few
  states. A 7x7 transition matrix has 42 free parameters (minus structural
  zeros from the graph) -- estimable, but with limited statistical power for
  detecting subtle strategy shifts.
- No reward is given in this task, so the "target bias" component of
  Rosenberg's four-bias model does not apply. The mice are exploring freely,
  not solving a maze for reward.
- The 1-min epoch duration limits the number of transitions per condition
  per epoch. At ~9.6 Hz imaging and typical mouse speeds, expect ~50-200
  cell transitions per minute, but only ~20-60 node (junction/dead-end)
  transitions per minute.
- Second-order models (prev, current -> next) require a 7x7x7 tensor with
  343 entries but most are structurally zero (the graph constrains which
  triplets are reachable). The effective parameter space is much smaller.

### What is already implemented

The `maze/analysis.py` module already contains:
- `transition_matrix()` — first-order Markov (cell-level and node-level)
- `transition_matrix_2nd_order()` — second-order Markov
- `transition_entropy()`, `transition_entropy_2nd_order()` — weighted
  conditional entropy
- `cross_entropy()`, `cross_entropy_2nd_order()` — model evaluation
- `markov_order_comparison()` — AIC/BIC model selection
- `stationary_distribution()` — eigenvector-based long-run distribution
- `sequence_entropy()` — context-dependent entropy (StringEntropy analogue)
- `exploration_efficiency()` — new nodes per window (NewNodes4 analogue)
- `segment_modes()` — directed vs exploratory classification
- `find_monotonic_paths()` — goal-directed run detection
- `path_efficiency_over_time()` — optimal/actual path length ratio
- `turn_bias()`, `per_junction_turn_bias()` — turn classification at junctions
- `dead_end_visits()` — per-dead-end visit counts and dwell times
- `simulate_random_walk()` — null model with optional forward bias
- `maze_exploration_summary()` — comprehensive per-session summary

This is a fairly complete implementation of the Rosenberg framework adapted
for the rose maze. The main gaps are in connecting these behavioural metrics
to neural data.

---

## 2. RSP and Exploration/Navigation: Literature Context

### RSP is more than a compass

RSP is often reduced to "the cortical HD relay," but the literature shows a
much richer role:

**Landmark anchoring and visual reference frames.**
Jacob et al. (2017) Nat Neurosci: discovered that dysgranular RSP contains
neurons tuned to head direction in a local, visually-defined reference frame
(bidirectional cells, "BD cells"), not just global HD. These cells are
dominated by visual landmarks even when landmarks conflict with the global
HD signal. This establishes RSP as a site where visual information actively
shapes the directional representation, not just passively receiving HD from
ADn.

**Visual-motor integration for path integration.**
Fischer et al. (2020) Curr Biol (actually Mao, Molina, Bonin & McNaughton):
RSP neurons show position-tracking sequences driven by the conjunction of
vision and locomotion. Optic flow can override locomotion signals, suggesting
RSP performs cue integration rather than pure path integration. This is
directly relevant — our light/dark manipulation separates these inputs.

**Coordinated HD across regions.**
Bicknell, van der Goes et al. (2024) eLife: Simultaneous ADn-RSP recordings
show nearly synchronous HD representation between thalamus and cortex during
cue rotation and in darkness. The coordination is maintained without vision,
consistent with strong feedforward drive from ADn to RSP. However, the bias
toward similar tuning between connected units suggests RSP is not just
inheriting the ADn signal but integrating it.

**RSP for spatial reasoning and hypothesis formation.**
Voigts, Kanitscheider et al. (2025) Nat Neurosci: RSP encodes mixtures of
spatial hypotheses and performs sequential hypothesis refinement via recurrent
dynamics. Mice navigating with ambiguous landmarks use RSP to resolve spatial
ambiguity over time. This positions RSP as a site for active inference about
location, not just passive sensory relay.

**Projection-specific RSP circuits.**
Han et al. (2024) Mol Psychiatry: Two RSP projection pathways have distinct
functions. M2-projecting RSP neurons (receiving dorsal subiculum, ADn, LD/LP
thalamus, somatosensory input) are required for object-location memory AND
place-action association. AD-projecting RSP neurons (receiving anterior
cingulate, medial septum input) are required only for object-location memory.
This demonstrates that RSP contains functionally distinct subpopulations
defined by their projection targets, with different roles in spatial cognition.

**Anterior vs posterior RSP.**
Recent work (2026, Nat Commun): Anterior RSP shows sharper position tuning
during navigation and sensitivity to fast, low-spatial-frequency visual
motion. Posterior RSP shows broader position selectivity and stronger
responses to slow, high-spatial-frequency patterns. This AP gradient means
imaging location within RSP matters for what signals we expect.

**RSP lesions impair navigation decisions.**
RSP-lesioned rats show impaired spatial working memory, difficulty using
landmarks for navigation, and slower alternation learning. RSP firing
patterns represent upcoming goal locations as animals approach choice points
(Mao et al. 2017 Curr Biol), suggesting RSP contributes to prospective
navigation planning. Lesioned animals use shortcuts faster (reduced
inhibition of alternative routes?) but make more errors with environmental
changes, suggesting RSP stabilises navigation strategies.

**RSP cell types: what we know.**
Brennan et al. (2020) Cell Reports: Layer 2/3 of granular RSP (RSG) is
dominated by "low-rheobase" (LR) neurons — hyperexcitable small pyramidal
cells that receive preferential input from anterior thalamus and dorsal
subiculum. These LR neurons can transform HD signals into angular velocity
signals through their biophysical properties.

Transcriptomic studies (Chen et al. 2023 Cell Reports; J Neurosci 2024):
RSG contains unique cell types not found elsewhere in cortex. Penk
(proenkephalin) is expressed in specific RSP subpopulations, but its
functional role is poorly characterised. The transcriptomic identity of
RSG cell types is preserved across mice and rats despite changes in marker
gene expression. Whether Penk+ neurons correspond to LR neurons, specific
laminar positions, or particular projection targets remains unknown.

**What is NOT known about Penk+ vs non-Penk RSP neurons:**
- Whether they differ in HD tuning properties
- Whether they differ in visual landmark dependence
- Whether they have different projection targets
- Whether they play different roles in navigation vs sensory anchoring
- Whether their activity relates differently to exploration behaviour
- Whether one population is more involved in spatial reasoning/hypothesis
  formation (Voigts et al. 2025) vs sensory anchoring (Jacob et al. 2017)

---

## 2b. The Rose Maze as a Natural Decision-Making Paradigm

### Every T-junction is a free binary choice

Rosenberg et al. (2021) make this point explicitly: a branching maze
generates dozens of binary decisions per session **without any training**.
Each T-junction visit is a natural two-alternative forced choice (2AFC).
The mouse must choose left or right, guided by whatever internal model it
has of the maze, its current goals (exploration, escape, foraging), and
available sensory cues.

In a standard trained 2AFC task, mice need weeks of shaping and thousands
of trials to reach criterion, typically generating 100–200 decisions per
session under artificial reward contingencies. In the rose maze, 7
T-junctions are visited repeatedly across a ~20-minute session, producing
**hundreds of decision points per session** — each one a free,
intrinsically motivated choice. There is no reward shaping the decisions,
so the choices reflect the animal's genuine navigation strategy.

This reframes the dataset: we are not just recording HD cells during
locomotion. We are recording RSP neurons during hundreds of natural
spatial decisions.

### Why this matters for RSP

RSP lesion studies consistently show impairments in spatial tasks that
require choosing between routes or integrating landmarks with path
information (Vann & Aggleton 2002, 2004; Pothuizen et al. 2008). RSP
receives convergent inputs from visual cortex (landmarks) and anterior
thalamus (HD signal), making it a candidate site where sensory evidence
is integrated to guide spatial decisions.

The Voigts et al. (2025) finding that RSP encodes mixtures of spatial
hypotheses through recurrent dynamics suggests RSP may actively
deliberate before navigation decisions — not merely relay HD information.
If this is true, we should see decision-related activity at T-junctions:
pre-decision ramping, choice-predictive signals, or post-decision
commitment signals.

### The light/dark manipulation as decision context

The alternating light/dark epochs create a natural manipulation of
decision-making context:

- **Light epochs**: The mouse has visual landmarks to guide decisions.
  Choices may be more consistent (lower entropy), more goal-directed
  (monotonic paths), and more influenced by allocentric spatial
  information.
- **Dark epochs**: Visual landmarks are removed. Decisions must rely on
  path integration, proprioception, and memory of the maze structure.
  We might expect higher choice entropy, more backtracking, more
  dead-end visits, and possibly a shift from allocentric to egocentric
  decision strategies (e.g. stronger forward bias, more stereotyped
  turn sequences).

This is a cleaner manipulation than most decision-making studies because
it changes the **information available for the decision** without
changing the task, the reward structure, or the motor requirements.

### Connection to cell-type-specific function

If Penk+ and Penk⁻CamKII+ neurons play different roles in the decision
process — e.g. one population encodes the sensory evidence (visual
landmarks), while the other maintains the internal model (path
integration, maze structure memory) — then:

- Their activity patterns at T-junctions should differ
- The light/dark manipulation should differentially affect their
  decision-related signals
- The population that relies on visual input should show degraded
  decision-predictive activity in darkness, while the other maintains it

The high trial count from natural maze decisions gives statistical power
to detect such differences even with small neural populations (~10–30
cells per session). This is something a trained 2AFC with 200 trials
could not achieve.

### Quantifying decisions

Each T-junction visit can be characterised by:
- **Choice** (left/right relative to approach direction)
- **Consistency** (does the mouse make the same choice on repeated visits?)
- **Latency** (time spent at the junction before committing — proxy for
  deliberation, though limited by 9.6 Hz temporal resolution)
- **Preceding trajectory** (was the approach direct or meandering?)
- **Outcome** (did the choice lead to a dead end or continue deeper?)
- **Context** (light vs dark, early vs late in session, first visit vs
  revisit)

The Markov transition model provides a formal framework: the transition
probabilities at each junction encode the animal's decision policy.
Changes in these probabilities between light and dark epochs quantify how
visual context shapes spatial decisions.

---

## 3. Novel Analysis Ideas: Maze Exploration x RSP Activity

### 3.1 Exploration strategy differences between genotypes (behavioural)

**Question:** Do Penk-Cre animals (carrying Penk+ label) and Penk-Cre
animals with Cre-OFF virus (carrying nonpenk label) explore the maze
differently?

**Approach:** Compare per-session maze metrics between the two genotype
groups:
- Occupancy entropy (spatial coverage uniformity)
- Coverage rate (time to visit all 23 cells)
- Dead-end dwell time
- Turn bias (left/right preference, back-tracking frequency)
- Transition entropy (predictability of navigation)
- Path efficiency (optimal vs actual path lengths)
- Fraction of time in directed vs exploratory mode

**Feasibility:** Already implementable with existing code. Animal-level
Mann-Whitney (N=12 vs N=4). Low statistical power due to the 4 nonpenk
animals, but large effects would still be detectable.

**Caveat:** Any behavioural difference is confounded with cohort/genotype
effects, not cell-type effects per se. The virus does not alter behaviour;
only the imaging target differs. But if the Penk-Cre line itself has
behavioural phenotypes (enkephalin is a neuromodulator), this could be
genuinely interesting. The single female Penk+ animal (1118023) should be
checked as an outlier.

**Priority: Medium.** Useful as a control/confound check. If behaviour
differs substantially between groups, all neural comparisons must account
for it. If behaviour is similar, it strengthens the interpretation that
neural differences reflect cell-type properties rather than behavioural
confounds.

**Already planned as:** H4.1, H4.2 in hypotheses.md.


### 3.2 Exploration strategy changes in darkness

**Question:** How does removal of visual cues (lights off) change maze
exploration?

**Approach:** Compare within-session, within-animal:
- Speed (already in kinematics)
- Occupancy entropy (light vs dark 1-min epochs)
- Transition entropy (light vs dark)
- Dead-end visit rate
- Back-tracking frequency (at junctions)
- Path efficiency
- Forward bias (momentum through corridors)

**Specific predictions:**
- Speed should decrease in darkness (well-established in rodents).
- Exploration may become more stereotyped (lower entropy) in darkness as
  mice rely on path integration / wall-following rather than landmark-guided
  navigation.
- Alternatively, exploration could become less efficient (higher entropy) if
  mice lose their spatial map and wander.
- Back-tracking at junctions may increase if mice lose confidence in their
  heading.
- Forward bias (momentum through corridors without turning) may increase in
  darkness as a conservative strategy.

**Feasibility:** High. Each session has alternating 1-min light/dark epochs.
Compute maze metrics per epoch, then paired Wilcoxon across epochs within
sessions. 21 non-excluded sessions give reasonable power.

**Key confound:** Speed differences between light and dark will affect the
number of transitions per epoch. Must normalize metrics by number of
transitions or time, not raw counts. Transition entropy is already
rate-normalized (bits per step). Occupancy entropy needs to be computed
over transitions, not frames (to avoid confounding slow exploration with
low entropy).

**Priority: High.** This is a clean within-animal comparison with a clear
prediction. The direction and magnitude of the exploration shift in darkness
sets the stage for asking whether RSP activity predicts or tracks that shift.

**Already planned as:** H4.3 in hypotheses.md, but without the specific
Markov model predictions.


### 3.3 Genotype x light interaction in exploration

**Question:** Does the light-to-dark shift in exploration strategy differ
between Penk+ and nonpenk animals?

**Approach:** Compute delta (light - dark) for each maze metric per session,
then compare deltas between genotypes (animal-level Mann-Whitney).

**Rationale:** If Penk+ and nonpenk RSP populations play different roles in
visual vs idiothetic navigation (the core hypothesis), and if those
populations influence behaviour, then the animals carrying different labels
might show different behavioural responses to light removal. This is a long
shot -- the virus does not alter function -- but it provides a behavioural
signature to correlate with neural findings.

**Priority: Low-medium.** Interesting but underpowered (12 vs 4) and
conceptually indirect.

**Already planned as:** H4.4 in hypotheses.md.


### 3.4 RSP population activity predicts upcoming maze transitions

**Question:** Can RSP neural activity predict which direction the mouse will
turn at an upcoming junction, before the turn occurs?

**Approach:**
1. Identify all junction approach events (mouse enters a corridor cell
   leading to a T-junction).
2. Extract neural population vectors in a window before arrival at the
   junction (e.g., 0.5-2 seconds before).
3. Train a classifier (logistic regression, SVM, or simple template
   matching) to predict turn direction (left, right, back) from the
   pre-junction neural population vector.
4. Compare classification accuracy to chance (1/3 for 3-way; or 1/2 for
   left vs right excluding backtracking).
5. Test separately for Penk+ and nonpenk populations.

**What would this show:**
- Above-chance prediction = RSP carries prospective navigation information,
  consistent with Mao et al. (2017) showing RSP represents upcoming goal
  locations at choice points.
- Cell-type difference in prediction accuracy = different populations
  contribute differently to prospective coding.

**Feasibility concerns:**
- At 9.6 Hz, a 1-second pre-junction window gives only ~10 frames. With
  ~10-30 simultaneously imaged neurons per session, the feature space is
  manageable but the temporal resolution is coarse.
- Number of junction visits per session per junction type is limited
  (~20-60 per minute total, split across 7 junctions).
- Must exclude junctions where only one turn direction is possible (dead-end
  corridors with only one exit).
- Cross-validation is essential (leave-one-out or k-fold per session).
- This analysis requires simultaneous neural + behavioural data at the
  junction-event level, which needs careful alignment.

**Expected effect size:** If this works at all in RSP with calcium imaging,
expect modest accuracy (~55-65% for binary left/right). This is not a
primary claim but would be a strong supporting finding.

**Priority: High.** This is the most novel connection between maze
behaviour and RSP activity. It goes beyond HD tuning to test whether RSP
carries prospective navigational intent, and whether the two populations
differ in this regard. Even a null result is informative (RSP in this
preparation does not carry pre-decision signals above what HD alone
predicts).

**Critical control:** HD at the time of junction approach strongly predicts
turn direction (a mouse facing left will turn left). Must control for
instantaneous HD. The key question is whether neural activity predicts
turns *above and beyond* what HD alone predicts. Use a nested model
comparison: accuracy(HD only) vs accuracy(HD + neural) vs accuracy(neural
only).


### 3.5 Dead-end visits and neural activity patterns

**Question:** Is there a characteristic RSP activity pattern associated with
dead-end visits (entering, dwelling, and exiting)?

**Approach:**
1. Identify all dead-end entry/exit events from the cell trajectory.
2. Align neural activity to dead-end entry (time zero) and extract peri-
   event traces (e.g., -2 to +5 seconds).
3. Compute average population response around dead-end events.
4. Compare dead-end-associated activity to corridor-transit activity
   (matched for speed and HD sampling).
5. Test whether activity differs between Penk+ and nonpenk populations.

**Rationale:** Dead ends are interesting because:
- The mouse must reverse direction (180-degree turn), creating a large HD
  change over a short period.
- Dead ends are spatial "error" signals in an exploration context — the
  mouse reaches a terminus and must backtrack.
- If RSP carries any "error" or "surprise" signal related to navigation,
  dead-end arrival is where it should appear.
- If RSP carries prospective goal information (3.4 above), dead-end
  arrival might show a distinctive pattern as the mouse re-plans.

**Feasibility:** Moderate. Dead-end visits are frequent enough (9 dead ends,
mice visit them regularly). The confound is that dead-end events are
correlated with specific HD ranges, speeds (slowing down), and positions.
Must carefully match control events.

**Expected result:** Given that RSP neurons are primarily HD-tuned, much of
the dead-end-associated activity will simply reflect the 180-degree head
turn. The interesting question is whether there is *residual* activity after
accounting for HD — a navigation-related signal beyond the compass.

**Priority: Medium.** Interesting but difficult to interpret cleanly due to
HD/speed confounds at dead ends.


### 3.6 Exploration entropy correlates with HD coding quality

**Question:** Does the quality of the HD representation (MVL, decoding
accuracy) correlate with how efficiently/stereotypically the mouse explores?

**Approach:**
1. Compute per-session HD decoding accuracy (from existing decoder module)
   and per-session exploration metrics (occupancy entropy, transition
   entropy, path efficiency).
2. Spearman correlation across sessions.
3. Test separately for Penk+ and nonpenk sessions.

**Predictions:**
- **Strong HD coding -> more efficient exploration?** If the HD
  representation is accurate, the mouse may navigate more efficiently
  (straighter paths, less backtracking). Expect negative correlation
  between HD decode error and path efficiency.
- **Or no relationship?** If maze exploration in this unrewarded task is
  driven by factors other than HD quality (curiosity, anxiety, motor
  patterns), no correlation is expected.

**Feasibility:** High — all metrics already exist or are trivially computed.
The question is whether there is enough variance in both HD quality and
exploration metrics across 21 sessions to detect a correlation.

**Caveat:** This is a between-session correlation, heavily confounded by
animal identity, imaging quality, and experience. With only 16 animals, a
within-animal comparison (if any animals have multiple sessions) would be
more informative. Check experiments.csv for animals with multiple sessions.

**Priority: Medium.** Easy to compute, potentially interesting, but
correlational and confounded.


### 3.7 Maze position decoding from RSP activity

**Question:** Can the mouse's position in the maze (which of 23 cells) be
decoded from RSP population activity?

**Approach:**
1. Use the discretised cell index as the dependent variable.
2. Train a Bayesian or template-matching decoder (analogous to the existing
   HD decoder but for spatial position).
3. Evaluate with cross-validation. Compare to chance (1/23).
4. Test light vs dark, Penk+ vs nonpenk.

**What would this show:**
- RSP carries spatial position information beyond HD alone.
- Comparison with HD decoding reveals whether RSP spatial coding is simply
  inherited from HD (expected in a maze where position and HD are partially
  correlated due to corridor geometry) or contains independent spatial
  information.

**Feasibility concerns:**
- 23 position classes with ~10-30 neurons per session is marginal. The
  decoder will need heavy regularization or dimensionality reduction.
- Position and HD are correlated in a maze (corridors constrain both). Must
  assess how much of decoded position accuracy is explained by HD alone.
- Occupancy is highly non-uniform — the mouse spends more time in some
  cells. Decoding accuracy must be compared to an occupancy-matched shuffle.

**Key control:** Decode position from HD alone (using the known position-HD
relationship in the maze). Then ask: does adding neural activity improve
position decoding beyond what HD gives you? If not, RSP is not carrying
independent spatial information.

**Priority: Medium-low.** Place coding in RSP is not the main story (that
is HD x cell type x visual dependence). But spatial information content
(Skaggs info for position, not just HD) could be a useful supplementary
analysis.

**Partially planned as:** H5.1, H5.2, H5.3 in hypotheses.md.


### 3.8 Transition-triggered neural activity

**Question:** Does RSP activity change systematically around cell-to-cell
transitions in the maze?

**Approach:**
1. Identify all cell transition events (mouse moves from cell A to cell B).
2. Align neural activity to transition time.
3. Compute peri-transition average for each neuron.
4. Group transitions by type: junction arrival, dead-end entry, corridor
   transit, light-on transition, light-off transition.
5. Compare Penk+ vs nonpenk population responses around transitions.

**Rationale:** Transitions are decision points. If RSP carries navigation-
related signals beyond HD, they should be visible around transitions,
particularly at junctions where the mouse must choose. This is the
event-triggered analogue of the decoding analysis in 3.4.

**Priority: Medium.** Useful for visualization and for building intuition,
but the HD confound is severe (transitions involve head movements).


### 3.9 Markov model residuals as a neural predictor

**Question:** When the mouse deviates from the Markov model prediction
(makes a "surprising" transition), is there a corresponding neural signature?

**Approach:**
1. Fit a first-order Markov model to the session's cell trajectory (or to
   light epochs only, then test on dark epochs).
2. At each transition, compute the model-predicted probability of the actual
   next cell. Low probability = surprising transition.
3. Correlate transition surprise (negative log probability) with neural
   population activity or population vector change magnitude around that
   transition.
4. Test separately for Penk+ and nonpenk.

**Rationale:** This is inspired by prediction-error frameworks. If RSP
carries a navigation prediction or expectation signal, deviations from the
animal's own typical behaviour should produce larger neural responses. This
has not been tested in RSP to my knowledge.

**Feasibility:** Moderate. Requires enough transitions to fit a stable
Markov model (probably need to pool across epochs or use a prior from all
sessions). The event-level analysis needs careful temporal alignment.

**Novelty:** High if it works. A navigation-surprise signal in
genetically-defined RSP subpopulations would be a novel finding.

**Priority: Medium-high.** Conceptually clean, but the analysis pipeline is
complex and the expected effect size is small.


### 3.10 Within-dark-epoch exploration dynamics

**Question:** Does exploration behaviour change *within* a dark epoch as
path integration drifts?

**Approach:**
1. Within each 1-minute dark epoch, split into early (0-30s) and late
   (30-60s) halves.
2. Compare speed, transition rate, backtracking frequency, occupancy
   entropy between early and late dark periods.
3. Correlate within-epoch behavioural change with within-epoch HD drift
   (from existing stability analysis).

**Prediction:** If mice lose their spatial representation as path
integration drifts in darkness, their exploration behaviour may become
progressively more cautious, slower, or more stereotyped. The rate of
this behavioural deterioration could correlate with the rate of HD drift,
linking neural representation stability to behavioural competence.

**Feasibility:** Moderate. 30-second windows give fewer transitions (~10-30
per window), limiting statistical power per epoch. But pooling across all
dark epochs (5-6 per session, ~100+ across the dataset) could provide
sufficient power.

**Priority: High.** This directly connects the HD anchoring story (core
hypothesis) to behaviour. If HD drift in darkness predicts behavioural
degradation, it demonstrates functional relevance of the HD representation
for navigation, not just a neural correlate.


### 3.11 Corridor-specific HD distributions

**Question:** Does the HD distribution depend on which corridor the mouse
occupies, and does this constrained sampling affect HD tuning estimates?

**Approach:**
1. For each maze cell, compute the distribution of HD angles when the mouse
   is in that cell.
2. Assess how non-uniform HD sampling is within corridors (corridors
   constrain body orientation along the corridor axis).
3. Determine whether HD tuning curve estimates are biased by non-uniform
   position-HD coupling.

**Why this matters:** In a maze, position and HD are not independent. A
mouse in a north-south corridor will predominantly face north or south.
This means HD tuning curves computed from maze data may be biased by
position-dependent sampling, and spatial information may be partially
confounded with HD information. This is a confound that must be
characterised before making strong claims about either HD or spatial coding.

**Approach to de-confound:** Compute HD tuning curves after marginalising
over position (only using time points where HD sampling is approximately
uniform) or using an explicit occupancy correction. The existing
`skaggs_info_rate()` already does occupancy-weighted information, but the
occupancy is over HD bins, not joint HD x position.

**Priority: High as a methods/control analysis.** Not a result in itself,
but essential for validating other results. Should go in supplementary
methods.


### 3.12 Cross-session exploration stability

**Question:** Are individual mice consistent in their exploration strategy
across sessions?

**Approach:**
1. For animals with multiple sessions, compute session-level exploration
   metrics (occupancy entropy, transition entropy, turn bias, coverage
   rate).
2. Compute ICC (intraclass correlation) across sessions within animals.
3. If stable individual differences exist, use them as a trait-like variable
   to correlate with neural properties.

**Rationale:** If exploration style is a stable individual trait (some mice
are bold explorers, others are cautious wall-followers), this provides a
between-animal dimension to relate to neural cell-type properties.

**Feasibility:** Depends on how many animals have multiple sessions. Check
experiments.csv. Even with only 2 sessions per animal, ICC can be estimated,
but precision will be poor.

**Priority: Low.** Interesting but requires multi-session data that may not
exist for most animals.


### 3.13 Population vector similarity across maze locations

**Question:** Does the RSP population vector (across all simultaneously
imaged neurons) differ between maze locations in a way that goes beyond HD?

**Approach:**
1. Compute mean population vector for each of the 23 maze cells (averaging
   neural activity when the mouse is in each cell).
2. Compute pairwise correlation/cosine similarity between all cell pairs.
3. Ask whether population vector similarity follows the maze graph distance
   (cells that are close in the maze have more similar neural
   representations).
4. Control for HD: cells in similar corridors have similar HD distributions.
   Regress out HD similarity and test whether residual population vector
   similarity still tracks maze distance.

**Rationale:** This tests whether RSP populations carry a map-like spatial
representation beyond the HD signal. If so, it connects to the Voigts et al.
(2025) finding that RSP encodes spatial hypotheses and locations in activity
space.

**Expected result:** Given that RSP is primarily HD-coding in freely moving
mice, most of the population vector structure should be explained by HD.
Finding residual spatial structure would be novel and important. Finding
none is the expected null and does not diminish the HD story.

**Priority: Medium.** Clean analysis with clear interpretation either way.


---

## 4. Feasibility Assessment

### Highly feasible with existing data and code

| Analysis | Code exists? | Data requirements met? | Statistical power |
|----------|-------------|----------------------|------------------|
| 3.1 Genotype exploration differences | Yes (maze_exploration_summary) | Yes (21 sessions) | Low (12 vs 4 animals) |
| 3.2 Light/dark exploration shift | Yes (per-epoch metrics) | Yes | High (within-session paired) |
| 3.3 Genotype x light interaction | Yes | Yes | Low |
| 3.6 HD quality vs exploration correlation | Partial (need to join metrics) | Yes | Moderate (21 sessions) |
| 3.10 Within-dark-epoch dynamics | Partial (need epoch splitting) | Yes | Moderate |
| 3.11 Corridor-HD distributions | No (new analysis) | Yes | High (descriptive) |

### Feasible but requires new code

| Analysis | Main requirement | Estimated effort |
|----------|-----------------|-----------------|
| 3.4 Junction turn prediction | Event-triggered decoder, HD control model | Medium-high |
| 3.5 Dead-end neural patterns | Peri-event averaging with HD matching | Medium |
| 3.8 Transition-triggered activity | Event alignment pipeline | Medium |
| 3.9 Markov residual x neural | Transition-level surprise + neural align | Medium-high |
| 3.12 Cross-session stability | Multi-session comparison | Low |
| 3.13 Population vector by location | PV computation + HD regression | Medium |
| 3.7 Position decoding | New decoder (23-class) | Medium |

### Key constraints

1. **Imaging rate (9.6 Hz):** Limits temporal precision for event-triggered
   analyses. A junction approach lasting ~1 second gives only ~10 frames.
   Pre-decision windows of 0.5-1 second give 5-10 frames. This is marginal
   for detecting transient signals.

2. **Neurons per session (~10-30):** Population analyses (decoding, PV
   similarity) are feasible but precision is limited. Template-matching
   decoders work with small populations; high-dimensional methods
   (deep learning decoders) are not appropriate.

3. **1-minute epochs:** Short for estimating Markov models or occupancy
   maps. Must pool across same-condition epochs within a session (typically
   5-6 light + 5-6 dark minutes per session).

4. **Cell-type imbalance (12 vs 4 animals):** Any between-genotype
   comparison has limited power. Focus on within-animal comparisons (light
   vs dark) and within-cell comparisons where possible.

---

## 5. Prioritised Analysis Plan

### Tier 1: Must-do (controls and core analyses)

1. **Corridor-HD distributions (3.11):** Essential confound characterisation.
   Must be done before interpreting any HD or spatial analysis in the maze
   context. Supplementary figure material.

2. **Light/dark exploration shift (3.2):** Clean within-animal comparison,
   high power. Establishes the behavioural context for interpreting neural
   light/dark differences. Main figure or supporting figure.

3. **Within-dark-epoch dynamics (3.10):** Links HD drift to behaviour.
   If HD drift predicts behavioural degradation, this is a key functional
   relevance argument for the HD anchoring story.

### Tier 2: High-value if they work

4. **Junction turn prediction from RSP activity (3.4):** Most novel
   analysis. If RSP activity predicts turns beyond what HD predicts, this
   positions RSP as carrying prospective navigation signals, not just a
   compass. Test Penk+ vs nonpenk for differential prospective coding.

5. **Markov surprise x neural (3.9):** Novel conceptually. If navigation
   prediction errors correlate with RSP activity, it links RSP to a
   prediction-error framework.

### Tier 3: Supportive / exploratory

6. **Genotype exploration differences (3.1):** Important as a confound
   check. If groups differ behaviourally, report it. If not, briefly note
   matched behaviour.

7. **Population vector by location (3.13):** Tests whether RSP has spatial
   structure beyond HD. Important for positioning the paper relative to the
   "RSP as spatial reasoning hub" literature.

8. **Dead-end neural patterns (3.5):** Interesting but hard to interpret
   cleanly.

9. **Position decoding (3.7):** Supplementary at best unless it reveals
   something unexpected.

---

## 6. How These Analyses Fit the Paper Narrative

The core paper story is: **Penk+ and Penk-CamKII+ RSP neurons differ in
how they anchor HD to visual landmarks (core hypothesis H3.4).**

The maze exploration analyses serve three roles:

**A. Behavioural context (Tier 1).** "Here is how the mice explore the
maze, here is how exploration changes in darkness, and here is the confound
landscape (HD sampling is non-uniform in a maze)." This goes in Methods
and a supplementary figure. It validates the experimental paradigm and
controls for behavioural confounds.

**B. Functional relevance (Tier 1-2).** "HD drift in darkness predicts
degradation of exploration behaviour, demonstrating that the HD
representation we measure is functionally relevant for navigation." This
could go in a main figure if the effect is clear.

**C. Beyond the compass (Tier 2).** "RSP activity predicts upcoming
navigational decisions beyond what HD alone predicts, and the two cell types
differ in this prospective coding." This would be a secondary main finding
if it survives controls. It elevates the paper from "cell-type-specific HD
anchoring" (one finding) to "cell-type-specific roles in spatial navigation"
(a richer story).

For a Nature Neuroscience-tier paper, A + the core HD story is sufficient.
B strengthens it. C would make it exceptional but is the riskiest.

For a J Neurosci or eLife paper, A + core HD story is the main content,
with B and C as supplementary or "data not shown."

---

## 7. Key Literature References

- Rosenberg, Zhang, Perona & Meister (2021). "Mice in a labyrinth show
  rapid learning, sudden insight, and efficient exploration." eLife 10,
  e66175. doi:10.7554/eLife.66175
- Jacob et al. (2017). "An independent, landmark-dominated head-direction
  signal in dysgranular retrosplenial cortex." Nat Neurosci 20, 173-175.
  doi:10.1038/nn.4465
- Fischer/Mao et al. (2020). "Vision and Locomotion Combine to Drive Path
  Integration Sequences in Mouse Retrosplenial Cortex." Curr Biol 30,
  1680-1688. doi:10.1016/j.cub.2020.02.070
- van der Goes, Bicknell et al. (2024). "Coordinated head direction
  representations in mouse anterodorsal thalamic nucleus and retrosplenial
  cortex." eLife 13, e82952. doi:10.7554/eLife.82952
- Voigts, Kanitscheider et al. (2025). "Spatial reasoning via recurrent
  neural dynamics in mouse retrosplenial cortex." Nat Neurosci 28,
  1293-1299. doi:10.1038/s41593-025-01944-z
- Han et al. (2024). "Projection-specific circuits of retrosplenial cortex
  with differential contributions to spatial cognition." Mol Psychiatry.
  doi:10.1038/s41380-024-02819-8
- Brennan et al. (2020). "Hyperexcitable Neurons Enable Precise and
  Persistent Information Encoding in the Superficial Retrosplenial Cortex."
  Cell Rep 30, 1067-1078.
- Alexander et al. (2023). "Coregistration of heading to visual cues in
  retrosplenial cortex." Nat Commun 14, 2184. doi:10.1038/s41467-023-37704-5
- Bicknell & Brennan (2021). "Thalamus and claustrum control parallel layer
  1 circuits in retrosplenial cortex." eLife 10, e62207.
- Mao et al. (2017). "Retrosplenial cortical representations of space and
  future goal locations develop with learning." Nat Neurosci.
- Mitchell et al. (2018). "Retrosplenial cortex and its role in spatial
  cognition." Brain Neurosci Adv 2. doi:10.1177/2398212818757098
- Chen et al. (2023). "Sharp cell-type-identity changes differentiate the
  retrosplenial cortex from the neocortex." Cell Rep 42, 112206.
