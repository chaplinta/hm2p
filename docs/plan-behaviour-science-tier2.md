# Tier-2 Behaviour Science Plan: Six Directions Beyond Route Stereotypy

Scientific hypotheses, literature grounding, feasibility assessments, and
implementation briefs for six further research directions arising from
the route stereotypy finding (H3, Tier-1 plan).

Status: 2026-05-31

---

## Context and Anchor Finding

The Tier-1 analyses (documented in `plan-behaviour-science.md` and
`behaviour-hypotheses-results.json`) established a central finding:

> **Route stereotypy.** In darkness, mice maintain dead-end destinations
> (coverage unchanged, p = 0.26) while consolidating onto fewer connecting
> routes (corridor coverage drops most, p < 0.00001, r = 0.97; junction
> coverage drops, p < 0.001, r = 0.82). The visited subgraph diameter
> contracts (6.35 to 5.57 cells, p = 0.002), transition matrices diverge
> (JSD = 0.068, 3.5x permutation null), and revisitation increases
> (p = 0.011, r = 0.64).

This finding dissociates local navigation rules (preserved) from global
route selection (disrupted). The six directions below deepen, extend,
and exploit this finding. Each is evaluated for scientific value,
feasibility, and priority.

---

## Direction 1: Within-Epoch Temporal Dynamics

### Question

Does coverage degrade immediately at lights-off, or does it accumulate
gradually over the 60-second dark epoch?

### Hypothesis

**H5a (gradual).** Coverage accumulates normally in the first 15--30
seconds of a dark epoch, then slows or stops as the path-integration
representation inherited from the preceding light epoch degrades. The
coverage deficit develops on the timescale of HD drift (~5--15 deg/min;
Stackman & Taube 1997; Peyrache et al. 2015).

**H5b (immediate).** Coverage drops from the first seconds of darkness,
indicating an immediate strategy switch (cautiousness/anxiety) rather
than gradual representational failure.

### Literature Support

- **HD drift timecourse.** Stackman & Taube (1997, J Neurosci) showed
  that HD cells in ADN maintained tuning for ~1--3 minutes in darkness
  before drifting substantially. Peyrache et al. (2015, Nat Neurosci)
  demonstrated coherent ensemble drift in darkness. Ajabi et al. (2023,
  Nature) quantified population-level drift dynamics in ADN, showing
  that drift rate varies across sessions and animals but remains
  coherent at the population level. A gradual coverage onset on the
  ~30-second timescale would be consistent with accumulated HD error.

- **Rapid re-anchoring.** Zugaro et al. (2003, J Neurosci) showed that
  HD cells re-anchor to visual landmarks within ~167 ms. This
  predicts that coverage should recover rapidly at lights-on (testable
  as a positive control).

- **Exploration in darkness.** Fonio et al. (2009, PNAS) described
  gradual unfolding of exploration in novel dark environments, with
  initial "looping" patterns near a home base. However, their animals
  were encountering the dark environment for the first time, whereas
  our mice have experienced multiple light-dark transitions. An
  immediate strategy switch would suggest the mouse detects darkness
  and preemptively restricts its range, consistent with the rodent
  "thigmotaxis" / anxiety response (Avni et al. 2006, Behav Processes).

- **Grid cell expansion.** Barry et al. (2012, PNAS) showed that grid
  cells expand in scale in novel environments and gradually compress
  back to baseline. If path-integration representations expand (become
  noisier) in darkness on a similar timescale, coverage could degrade
  in parallel.

### What This Would Mean

- **Gradual onset (H5a confirmed):** Strongest possible evidence for a
  representational mechanism. The behaviour paper can claim that the
  coverage decline develops on the timescale of spatial representation
  degradation, providing a behavioural correlate of HD drift. This
  directly bridges to the neural paper, where we can test whether the
  onset of coverage decline in individual sessions coincides with the
  onset of HD tuning instability.

- **Immediate onset (H5b confirmed):** Suggests a cognitive/affective
  mechanism (the mouse "knows" it is dark and adopts a defensive
  strategy). Less directly connected to representational mechanisms but
  still interesting: it implies the mouse can detect darkness within
  seconds and switch exploration mode. This is consistent with the
  light/dark box literature on anxiety in rodents.

- **Both are publishable.** The distinction between gradual and
  immediate onset is informative either way.

### Test Specification

1. Split each dark epoch into two halves at 30 seconds. Compute
   **new unique cells** discovered in each half (i.e., cells not yet
   visited in that epoch). Repeat for light epochs as a control.

2. Compute the **coverage ratio** = (new cells in last 30 s) /
   (new cells in first 30 s) separately for light and dark epochs.
   Compare with Wilcoxon signed-rank (N = 20 sessions, using session
   medians across epochs within each session).

3. **Critical ceiling-effect control.** In any epoch, the second half
   necessarily discovers fewer new cells because some were already
   found. The test must compare the *relative* deceleration between
   conditions. If light coverage ratio is 0.6 and dark is 0.3, the
   dark deficit accumulates in the second half.

4. **Speed control.** Compute speed in each half-epoch. If speed also
   declines within dark epochs, the coverage effect may be locomotor.
   Partial correlation: residual coverage ratio after controlling for
   speed ratio.

5. **Finer temporal resolution.** Compute a cumulative unique cell
   curve as a function of time (in 5-second bins) within each epoch
   type. Plot mean +/- SEM for light and dark. The moment at which
   the dark curve diverges from the light curve estimates the onset
   latency of the darkness effect.

6. **Lights-on recovery (positive control).** Compare coverage in the
   first 30 seconds of a light epoch that follows a dark epoch with
   the first 30 seconds of the first light epoch. If coverage
   "rebounds" at lights-on, the darkness effect is reversible within
   seconds (consistent with rapid HD re-anchoring).

### Expected Outcomes

- If gradual: dark coverage ratio < light coverage ratio (p < 0.05),
  with divergence onset at 15--30 seconds. Speed ratio shows a weaker
  or absent effect.
- If immediate: the cumulative unique cell curve for dark epochs
  diverges from light within the first 5--10 seconds.

### Confounds

- Epoch-position confound: dark epochs always follow light epochs. If
  mice are simply tiring over time, coverage would decline in any
  second epoch regardless of condition. **Control:** compare second
  half of light epochs with second half of dark epochs.
- Small epoch sample: with ~10 dark epochs per session, session-level
  median coverage ratios may be noisy. Consider pooling across sessions
  for the cumulative curve analysis, with session as random effect.

### Priority: HIGH

This is the most directly mechanistic of the six directions. It tests
whether the route stereotypy finding has a temporal signature consistent
with representational degradation. It also provides the strongest bridge
to the neural paper.

### Feasibility: STRAIGHTFORWARD

All data available in sync.h5 (position + light_on + timestamps at
~9.6 Hz). Requires only epoch splitting and cell-sequence recomputation.
No new data collection needed.

---

## Direction 2: Corridor-Specific Analysis

### Question

Which corridors are abandoned in darkness? Are they the longest
corridors, the most peripheral ones, or the ones farthest from the
mouse's current position?

### Hypothesis

**H6.** Corridors that connect junctions to peripheral dead ends
(i.e., corridors that are "approach corridors" serving a single
dead-end branch) are abandoned more than corridors that connect two
junctions (i.e., corridor segments of the maze backbone). This
predicts that route stereotypy is driven by avoidance of committed
excursions to dead-end branches, not by random corridor omission.

### Maze Topology Context

The q-rose maze has 7 corridors, 7 T-junctions, and 9 dead ends.
The corridor cells and their topological roles:

| Corridor cell | Connects | Role |
|---|---|---|
| (1,1) | (1,0) junction --- (1,2) junction | Backbone: connects bottom-left to central hub |
| (1,3) | (1,2) junction --- (1,4) junction | Backbone: connects central hub to top-left |
| (2,2) | (1,2) junction --- (3,2) junction | Backbone: connects left arm to central hub |
| (3,3) | (3,2) junction --- (3,4) dead end | Branch: approach corridor to central dead end |
| (4,2) | (3,2) junction --- (5,2) junction | Backbone: connects central hub to right arm |
| (5,1) | (5,0) junction --- (5,2) junction | Backbone: connects bottom-right to central hub |
| (5,3) | (5,2) junction --- (5,4) junction | Backbone: connects central hub to top-right |

Of the 7 corridors, 6 connect two junctions (backbone) and only 1
connects a junction to a dead end ((3,3), leading to the central
dead end (3,4)). This is an asymmetric design: most corridors are
backbone segments. The hypothesis therefore becomes less about
"branch corridors are abandoned" (only 1 exists) and more about
which backbone corridors are preferentially avoided.

**Revised hypothesis:** Corridors that are most peripheral (highest
graph eccentricity) are abandoned more. Specifically, corridors at
maximum graph distance from the maze center should show the largest
coverage drop.

### Literature Support

- **Rosenberg et al. (2021, eLife).** In their larger labyrinth, mice
  explored with a forward bias that favoured deeper penetration.
  Abandoning peripheral corridors is the opposite strategy, suggesting
  that darkness reverses or attenuates the exploratory drive.

- **Avni et al. (2006, Behav Processes).** In an open field in
  darkness, mice shifted from directional to positional progression,
  reducing the spatial extent of their trajectories. Peripheral
  corridors in a maze are the analogue of far-wall regions in an open
  field.

- **Koren Iton et al. (2025, bioRxiv).** NaviGraph framework assigns
  visit frequency to graph nodes, enabling detection of topological
  biases in navigation. Our analysis mirrors this approach.

### Test Specification

1. **Per-corridor coverage change.** For each of the 7 corridor cells,
   compute the fraction of epochs in which it was visited, separately
   for light and dark. Compute the per-corridor delta (dark - light).
   Rank corridors by their delta.

2. **Eccentricity correlation.** Compute the graph eccentricity of each
   corridor cell (maximum shortest-path distance to any other cell).
   Correlate eccentricity with coverage delta (Spearman, N = 7
   corridors). If peripheral corridors show larger drops, rho should
   be negative.

3. **Distance from center.** Define maze "center" as the junction with
   minimum eccentricity (likely (3,2)). Compute distance from center
   for each corridor. Correlate with coverage delta.

4. **Branch vs backbone classification.** Classify each corridor as
   "branch approach" or "backbone." Since there is only 1 branch
   corridor, this is descriptive rather than statistical.

5. **Length of connected dead-end branch.** For each corridor, compute
   the number of dead ends reachable from it (via the approach
   direction away from center). Corridors that serve more dead ends
   may be maintained because they gate access to multiple destinations.

### Expected Outcomes

- Peripheral corridors (e.g., (1,1), (5,1)) show the largest coverage
  drops. Central backbone corridors (e.g., (2,2), (4,2)) are maintained
  because they serve as transit points for multiple routes.

- If the opposite: central corridors drop more, suggesting the mouse
  retreats to the periphery. This would be surprising but would change
  the interpretation of route stereotypy.

### Confounds

- **Small N.** Only 7 corridor cells. Any correlation with N = 7 is
  underpowered and should be reported as descriptive. The primary value
  is the per-corridor heatmap visualisation, not the statistical test.

- **Occupancy bias.** If certain corridors are rarely visited even in
  light, their coverage delta may be ceiling-limited (already low,
  cannot drop much). Normalise by light-epoch coverage before computing
  delta.

- **Speed confound.** If speed differs by location (mice move faster in
  central corridors), corridor-specific coverage could be confounded
  by location-specific speed changes. Control: compute speed-per-frame
  within each corridor cell and test whether the coverage delta
  persists after speed matching.

### Priority: MEDIUM

Primarily descriptive and visualisation-driven. The N = 7 corridor
cells limit statistical power for corridor-level tests. The main
value is a per-cell heatmap (Figure panel) showing where in the maze
coverage drops, reinforcing the route stereotypy narrative visually.

### Feasibility: STRAIGHTFORWARD

Per-cell visit data is available from cell_occupancy(). Eccentricity
computable from the distance matrix in RoseMaze.dist.

---

## Direction 3: Individual Differences and Neural Bridge

### Question

Do animals that show the largest coverage drops in darkness also show
the most disrupted HD tuning? Can behavioural "darkness sensitivity"
predict neural HD stability?

### Hypothesis

**H7.** Animals with more stable HD tuning in darkness (better path
integration) maintain higher maze coverage in darkness. The
per-animal correlation between "darkness sensitivity" (coverage drop)
and "HD stability" (tuning curve correlation light vs dark) is
negative and meaningful (Spearman rho < -0.4).

### Literature Support

- **Fischer et al. (2020, Curr Biol).** RSP neurons integrate visual
  and self-motion cues, with cells varying in their reliance on each
  cue type. If this variation exists at the animal level (some animals'
  RSP populations are more visually anchored than others), behavioural
  consequences should follow.

- **Muir et al. (2022, Nat Commun).** Flexible cue anchoring strategies
  enable stable HD coding even in blind animals. This suggests that
  individual differences in HD stability are not purely about visual
  cue availability but also about the efficiency of alternative
  anchoring strategies (olfactory, proprioceptive). Animals with
  stronger non-visual anchoring should be behaviourally "darkness-
  resistant."

- **Gobbo et al. (2026, bioRxiv).** Navigational strategy shapes
  hippocampal representations. If animals differ in their navigational
  strategy (allocentric vs egocentric), this could produce correlated
  individual differences in both HD stability and behavioural coverage.

### Test Specification

1. **Per-animal darkness sensitivity score.** For each animal, compute
   the mean coverage difference (light - dark) across all sessions
   for that animal. For animals with multiple sessions, average to
   get one score per animal (N = 14 animals, 10 Penk+, 4 non-Penk).

2. **Per-animal HD stability score.** From the neural analysis, compute
   the mean tuning curve correlation (Pearson or vector correlation)
   between the light and first-dark-epoch tuning curves, averaged
   across all HD cells in each animal. This requires the neural
   analysis pipeline to be complete.

3. **Spearman correlation** between darkness sensitivity and HD
   stability (N = 14, or N = 11 primary-only).

4. **Cell-type interaction.** If the two cell types differ in HD
   stability (the core neural paper hypothesis), test whether the
   cell-type effect is consistent with the individual differences.
   For example, do animals with more Penk+ HD cells show different
   darkness sensitivity than animals with more non-Penk HD cells?

5. **Behavioural variability report.** Even without neural data,
   document the range of individual differences in darkness sensitivity.
   Show a scatter plot of per-animal coverage in light vs dark. Flag
   "darkness-resistant" animals (coverage drop < 1 cell) and
   "darkness-vulnerable" animals (coverage drop > 3 cells). This is
   valuable descriptive information for the behaviour paper.

### Expected Outcomes

- **If correlated (rho < -0.4):** Strong bridge finding. Animals whose
  RSP populations maintain HD in darkness also maintain exploration.
  This provides an individual-differences link between the two papers.
  Publishable as a primary finding of the neural paper, with the
  behavioural data as a covariate.

- **If uncorrelated:** The coverage drop is not driven by HD
  representational failure. Could suggest an anxiety/cautiousness
  mechanism operating independently of spatial representation quality.
  This is informative and publishable as a null result that constrains
  interpretation.

### Confounds and Power Concerns

- **N = 14.** This is very low for a correlation. With N = 14,
  Spearman requires rho > 0.54 for significance at alpha = 0.05.
  We can detect only large effects. A non-significant trend (rho in
  -0.3 to -0.5) should be reported honestly as suggestive.

- **Cell-type confound.** If Penk+ and non-Penk animals differ
  systematically in both behaviour and HD stability, the correlation
  could be driven by cell type rather than individual variation. Must
  test within cell types as well (though N = 10 and N = 4 make this
  very underpowered).

- **Session-level variability.** Some animals contribute multiple
  sessions. Averaging to one score per animal is correct for the
  correlation but discards within-animal variability. Report
  within-animal consistency (ICC) as a supplement.

### Priority: HIGH for the neural paper, MEDIUM for the behaviour paper

This is the single most important bridge between the two manuscripts.
However, it requires neural analysis to be complete, making it a
deferred analysis. For the behaviour paper, the descriptive
individual-differences report (Step 5) is immediately feasible and
valuable as a supplementary finding.

### Feasibility: DEFERRED for the correlation (needs neural data); IMMEDIATE for the individual-differences description

---

## Direction 4: Epoch-Number Adaptation

### Question

Does route stereotypy change over repeated light-off epochs within a
session? Do mice adapt (become less stereotyped) as they experience
more dark epochs, suggesting learning/confidence? Or is the stereotypy
constant, suggesting hardwired cue dependence?

### Hypothesis

**H8a (adaptation).** Route stereotypy (measured as coverage or corridor
coverage) in dark epochs improves over the course of the session. Early
dark epochs (epochs 1--3) show the largest coverage drop relative to
their preceding light epochs; later dark epochs (epochs 7--10) show
smaller drops. This would indicate that mice learn that darkness is
temporary and safe, reducing the defensive exploration restriction.

**H8b (constant).** Route stereotypy is stable across dark epochs. The
coverage drop is the same magnitude in the first and last dark epoch.
This would suggest that the coverage deficit reflects an obligatory
response to visual cue removal, not a learned strategy.

**H8c (worsening).** Route stereotypy increases over the session. Later
dark epochs show larger coverage drops. This would suggest accumulated
fatigue, habituation, or progressive disorientation from repeated
light-dark transitions.

### Literature Support

- **Schmitzer-Torbert & Redish (2002, Behav Neurosci).** Path
  stereotypy in rats on a multiple-T maze developed within a single
  session. Rats rapidly converge on stereotyped routes after initial
  variability. Our question is whether the same process occurs within
  dark epochs specifically.

- **Fonio et al. (2009, PNAS).** Exploration in novel environments
  unfolds gradually over minutes. Each dark epoch is a "re-entry"
  into an uncertain environment. If the mouse treats each darkness
  event as a mini-novelty exposure, the Fonio et al. prediction is
  that exploration should gradually expand (supporting H8a).

- **Place cell stability across sessions.** Place fields become
  increasingly stable with repeated exposure (Lever et al. 2002,
  Nature). If the mouse's internal map becomes more stable with
  repeated light-dark transitions, coverage should improve (H8a).

- **Habituation to repeated darkness.** In the light-dark box
  paradigm (standard anxiety test), mice habituate to darkness over
  repeated exposures (Bourin & Hascoet 2003). However, those
  exposures are typically across days, not within a single session.
  Within-session habituation to 1-minute dark epochs is not well
  characterised in the literature, making this a genuinely novel test.

### Test Specification

1. **Per-epoch coverage.** For each session, compute coverage in each
   dark epoch and the preceding light epoch. Compute the delta
   (light - dark) per epoch pair. Plot delta as a function of epoch
   number.

2. **Spearman correlation** between epoch number and coverage delta
   across all epoch pairs pooled within each session. Then compute the
   session-level median slope and test whether it differs from zero
   (Wilcoxon, N = 20).

3. **Grouped comparison.** Divide dark epochs into "early" (first
   third) and "late" (last third) within each session. Compare
   coverage delta between early and late (Wilcoxon paired, N = 20).

4. **Speed control.** Compute the same analysis for speed (does speed
   change across epochs?). If speed increases over the session, coverage
   increases could be locomotor rather than strategic.

5. **Light-epoch control.** Does light-epoch coverage also change over
   the session? If so, the epoch-number effect is a global session
   effect, not darkness-specific.

### Expected Outcomes

- If H8a: late dark epochs show coverage delta < early dark epochs
  (p < 0.05). The slope of delta vs epoch number is negative.
- If H8b: no significant trend. Delta is constant across epoch numbers.
- If H8c: late dark epochs show larger delta (worsening). This would
  suggest fatigue or accumulated disorientation.

### Confounds

- **Ceiling effects.** Light-epoch coverage may decrease over the
  session as the mouse tires or habituates. If both light and dark
  coverage decline together, the delta could remain constant even if
  the dark coverage is improving relative to its potential.

- **Epoch count variation.** Sessions have ~10--15 light-dark cycles
  (10 minutes total). Some sessions may have fewer usable epochs
  (due to bad_behav exclusion). Verify that all sessions have at least
  6 dark epochs for the early/late comparison.

- **Between-session noise.** Pooling epoch pairs across sessions
  introduces between-session variance. The session-level median slope
  approach (Step 2) handles this correctly by computing within-session
  slopes first.

### Priority: MEDIUM

This is a clean, targeted analysis that tests a specific prediction
about within-session learning. A positive result (adaptation) would
add a temporal dimension to the route stereotypy finding. A null
result (constant) is also informative: it strengthens the claim that
route stereotypy is an obligatory response to cue removal. H8c
(worsening) would be the most surprising and would suggest a
progressive spatial-memory failure accumulating across dark epochs.

### Feasibility: STRAIGHTFORWARD

All data available. Requires only epoch-level coverage computation
(already implemented) with epoch ordering metadata (easily derived
from epoch start times).

---

## Direction 5: Return-Path Efficiency from Dead Ends

### Question

When a mouse visits a dead end in darkness, does it retrace the same
path back to the junction, or does it take a different route? Same
path = habitual route memory. Different path = spatial flexibility
(or confusion).

### Hypothesis

**H9.** In darkness, mice show higher outbound-inbound path overlap
at dead ends (more stereotyped return paths) compared to light, where
they may take alternative return routes. This would be consistent with
route stereotypy: in darkness, the mouse relies on a single well-known
route to each destination and retraces it.

### Maze Topology Constraint

This analysis is constrained by the q-rose maze structure. Dead ends
are at the tips of branches, and in most cases there is only a single
path from the nearest junction to the dead end. The only "choice" on
the return path occurs when the mouse reaches the junction: does it
continue toward the destination it was heading for before the dead-end
visit, or does it reverse to a different branch?

**Critical reassessment:** In this maze, the path from a junction to
a dead end is typically a single corridor cell followed by the dead end
itself. There is no alternative route to most dead ends. The
outbound-inbound path is forced by the topology to be identical at
the corridor level.

**Where this analysis has power:** At the **junction** following the
dead-end return. The question becomes: after visiting a dead end and
returning to the parent junction, does the mouse make the same choice
(turn direction) as it did the last time it visited that junction? In
other words, does dead-end return behaviour at the junction level
become more stereotyped in darkness?

### Revised Test Specification

1. **Post-dead-end junction choice.** For each dead-end visit, identify
   the return path to the parent junction. Record the turn direction
   at the junction (left/right/back relative to approach from the
   dead-end branch).

2. **Post-dead-end turn consistency.** For each junction, compute the
   consistency of turn choices after dead-end returns across multiple
   visits within an epoch. Measure as the fraction of turns in the
   majority direction (1.0 = always same direction, 0.33 = random for
   3-option junction).

3. **Compare light vs dark** for turn consistency (Wilcoxon, N = 20).

4. **Alternative: post-dead-end destination.** After leaving the parent
   junction, what is the next destination (junction or dead end)?
   Compute the diversity of next-destinations after dead-end returns
   (entropy across destination identities). Lower entropy in darkness
   = more stereotyped onward navigation.

### Expected Outcomes

- If H9 confirmed: turn consistency at junctions after dead-end returns
  is higher in darkness (the mouse always takes the same exit). Post-
  dead-end destination entropy is lower in darkness.

- If H9 rejected: no difference in turn consistency or destination
  entropy. Dead-end return behaviour is equally variable in both
  conditions.

### Literature Support

- **Rosenberg et al. (2021, eLife).** "Home runs" — direct paths to
  the maze entrance without reversals — occurred at the end of
  exploration bouts. Our analysis tests whether dark-epoch dead-end
  visits similarly trigger stereotyped return paths.

- **Schmitzer-Torbert & Redish (2002).** Path stereotypy developed
  rapidly in a multiple-T maze. Post-dead-end turns may become
  stereotyped as a form of procedural learning.

- **Innate heuristics and escape route learning.** Campagner et al.
  (2022, Curr Biol) showed that mice learn escape routes rapidly
  and execute them with high fidelity. Dead-end returns to a familiar
  junction may use a similar memory system.

### Confounds

- **Small number of dead-end visits.** In a 1-minute epoch, a mouse
  may visit 2--4 dead ends. With ~10 dark epochs, there are ~20--40
  dark dead-end visits per session. This is marginal for per-junction
  consistency metrics. Pool across sessions or across dead ends.

- **Structural bias.** Some junctions have only 2 non-dead-end exits,
  so the "expected" consistency by chance is 0.5, not 0.33. Must
  compute chance level per junction based on its degree.

- **Speed of return.** If mice return faster from dead ends in darkness
  (a simple retreat), the turn at the junction may be less deliberate.
  Compare speed during the approach to the junction after dead-end
  exits.

### Priority: LOW-MEDIUM

This analysis is conceptually appealing but is constrained by the
maze topology (forced paths to most dead ends) and limited by the
small number of dead-end visits per epoch. The revised analysis
(post-dead-end junction choice) is feasible but may lack power. Best
treated as a supplementary analysis that enriches the route stereotypy
narrative rather than a standalone finding.

### Feasibility: MODERATE

Requires extracting dead-end visit events and the subsequent junction
choice. The dead_end_visits() function provides visit counts but not
the post-visit trajectory. New code needed to extract the junction
choice following each dead-end exit.

---

## Direction 6: Cell-Type Markov Model

### Question

Does modelling transitions between maze cell *types*
(junction, corridor, dead-end) rather than individual cells reveal
second-order structure that the cell-level analysis missed?

### Hypothesis

**H10.** A second-order cell-type Markov model (predicting next cell
type from previous two cell types) differs between light and dark,
even though the cell-level first-order model is preferred in both
conditions. Specifically, the transition
junction -> corridor -> dead-end should decrease in darkness (avoidance
of committed excursions), while the transition
junction -> corridor -> junction should increase (staying on the
backbone).

### Rationale

The cell-level Markov analysis (23 states) found that first-order
models are preferred over second-order in all sessions (by BIC).
However, collapsing to 3 cell types (junction, corridor, dead-end)
dramatically reduces the state space, making second-order models more
tractable and potentially revealing structure that was diluted across
23 individual cells.

The 3-state first-order model has 3 x 3 = 9 transition probabilities
(6 free parameters after row normalisation). The second-order model
has 3 x 3 x 3 = 27 probabilities (18 free parameters). With ~50--100
cell-type transitions per epoch, both models should be estimable
without overfitting.

### Test Specification

1. **Cell-type sequence.** Convert each cell sequence to a cell-type
   sequence: map each cell index to its type (junction = J, corridor
   = C, dead-end = D). Remove consecutive duplicates (since the cell
   sequence already removes within-cell duplicates, consecutive
   type-duplicates arise only when two adjacent cells share the same
   type, which is rare in this maze but should be handled).

2. **First-order cell-type transition matrix.** Compute the 3x3
   transition matrix for light and dark separately. Compare via JSD
   (as in H2).

3. **Second-order cell-type transition matrix.** Compute the 3x3x3
   second-order transition matrix (P(next type | prev type, current
   type)). Use AIC/BIC model comparison to test whether second-order
   is preferred over first-order at the cell-type level.

4. **Specific transition tests.** For the key transitions:
   - J -> C -> D (commitment to dead-end branch): compare light vs
     dark (Wilcoxon on per-session P(D | J,C), N = 20).
   - J -> C -> J (backbone traversal): compare light vs dark.
   - D -> C -> J (dead-end return): compare light vs dark.
   - C -> J -> C (passing through junction): compare light vs dark.

5. **Visualisation.** Sankey-style diagram showing cell-type transition
   flows for light and dark, with line width proportional to transition
   probability.

### Expected Outcomes

- If the second-order cell-type model is preferred: route stereotypy
  has structure beyond first-order cell-type transitions. The mouse
  is not just avoiding dead ends; it is avoiding the specific
  *sequences* that lead to dead ends.

- If H10 confirmed: P(D | J,C) decreases in darkness; P(J | C,J)
  increases. The mouse avoids committing to dead-end branches and
  instead stays on the backbone.

- If H10 rejected: cell-type transitions are similar between
  conditions. The route stereotypy operates at the individual-cell
  level (specific corridors avoided) rather than at the structural
  level (cell types avoided).

### Literature Support

- **Bhakti et al. (2024, eLife).** Used HMMs with distinct navigation
  strategies as latent states. Our cell-type Markov model is simpler
  (no latent states) but addresses a related question: whether
  transitions between topological categories (not individual locations)
  change with condition.

- **Rosenberg et al. (2021, eLife).** Their Markov analysis used
  individual junction identities. They did not collapse to cell-type
  categories. A cell-type analysis of their data would be interesting
  but has not been published.

### Confounds

- **Structural constraints.** The maze graph constrains which cell-type
  transitions are possible. For example, D -> D is impossible (no two
  dead ends are adjacent). J -> J is impossible (all junctions are
  separated by at least one corridor). These structural zeros must be
  accounted for in the model.

- **Imbalanced counts.** Corridor cells are traversed much more often
  than dead-end cells. The transition matrix will be dominated by
  J-C-J and C-J-C transitions. The D-relevant transitions may have
  few counts per epoch.

- **Type assignment ambiguity at corridor-corridor boundaries.** Two
  adjacent corridor cells produce a C -> C transition, which occurs
  when the mouse traverses a multi-cell corridor segment. These
  transitions are structurally uninformative. Consider treating
  multi-cell corridor segments as single edges.

### Priority: MEDIUM

This is methodologically clean and could produce a nice visualisation
(cell-type Sankey diagram). The second-order model comparison is the
scientifically interesting part: it tests whether the maze structure
itself (not just which cells are visited) changes in darkness.
However, the primary finding (route stereotypy) is already established
at the individual-cell level; the cell-type model provides supporting
depth rather than a new finding.

### Feasibility: STRAIGHTFORWARD

All data available. Requires only mapping cell indices to types and
recomputing transition matrices. The cell-type classification is
already in RoseMaze.node_types.

---

## Summary Table

| # | Direction | Priority | Feasibility | Novel contribution | Key statistic |
|---|---|---|---|---|---|
| 1 | Within-epoch temporal dynamics | HIGH | Straightforward | Tests representational vs affective mechanism | Coverage ratio (2nd/1st half), dark vs light |
| 2 | Corridor-specific analysis | MEDIUM | Straightforward | Per-cell heatmap; eccentricity correlation | Per-corridor coverage delta x eccentricity (N=7) |
| 3 | Individual differences (neural bridge) | HIGH (neural) / MEDIUM (behaviour) | Deferred (neural) / Immediate (descriptive) | Bridges two manuscripts | Spearman(coverage sensitivity, HD stability), N=14 |
| 4 | Epoch-number adaptation | MEDIUM | Straightforward | Tests learning vs obligatory response | Coverage delta x epoch number slope |
| 5 | Return-path efficiency | LOW-MEDIUM | Moderate | Enriches route stereotypy narrative | Post-dead-end junction turn consistency |
| 6 | Cell-type Markov model | MEDIUM | Straightforward | Tests structural-level route avoidance | P(D\|J,C) light vs dark; model order comparison |

---

## Implementation Priority Order

**For the DS agent, in order:**

### Phase A: Must-implement (before manuscript submission)

**A1. Within-epoch temporal dynamics (Direction 1).** This is the
highest-priority new analysis. It directly tests the mechanism behind
route stereotypy and provides the strongest bridge to the neural paper.

Implementation steps:
1. For each session, identify all epoch boundaries (light-on, light-off
   transitions from sync.h5 `light_on` column).
2. Split each epoch at the temporal midpoint (30 seconds).
3. For each half-epoch: compute new unique cells discovered (cells not
   yet visited in that epoch's first half, for the second half).
4. Compute per-session median coverage ratio (2nd/1st half) separately
   for light and dark.
5. Wilcoxon signed-rank: coverage ratio dark vs light (N=20).
6. Compute cumulative unique cell curves in 5-second bins, averaged
   across epochs within condition, then across sessions (mean +/- SEM).
7. Compute speed in each half-epoch as a control.
8. Lights-on recovery: compare coverage in first 30s of post-dark
   light epochs vs first 30s of session-initial light epochs.

Output: JSON results + cumulative curve data for plotting.

**A2. Epoch-number adaptation (Direction 4).** Tests whether route
stereotypy is constant or changes over the session.

Implementation steps:
1. Number each epoch pair (light_n, dark_n) within each session.
2. Compute coverage delta (light - dark) per epoch pair.
3. Spearman correlation: epoch number vs coverage delta, within each
   session. Report session-level median rho.
4. Early vs late comparison: Wilcoxon on coverage delta for first-third
   vs last-third epoch pairs (N=20).
5. Light-epoch coverage vs epoch number as control.

Output: JSON results.

### Phase B: Should-implement (enriches manuscript)

**B1. Corridor-specific analysis (Direction 2).** Primarily for
visualisation.

Implementation steps:
1. Compute per-cell visit fraction for each epoch (light and dark).
2. Aggregate to per-cell coverage delta (dark - light) across sessions.
3. Compute graph eccentricity for each cell.
4. Generate per-cell heatmap on maze layout (7x5 grid), colour-coded
   by coverage delta.
5. Spearman: eccentricity vs coverage delta for corridor cells (N=7,
   descriptive).

Output: Per-cell delta array + heatmap data.

**B2. Cell-type Markov model (Direction 6).** Tests structural-level
route avoidance.

Implementation steps:
1. Convert cell sequences to type sequences (J/C/D).
2. Compute 3x3 first-order and 3x3x3 second-order transition matrices
   for light and dark per session.
3. JSD between light and dark cell-type transition matrices.
4. Per-transition Wilcoxon tests for key triplets: P(D|J,C),
   P(J|C,J), P(J|D,C).
5. AIC/BIC comparison: first-order vs second-order at cell-type level.

Output: JSON results + transition flow data for Sankey visualisation.

### Phase C: Deferred or supplementary

**C1. Individual differences descriptive report (Direction 3).** No
neural data needed for the descriptive part.

Implementation steps:
1. Compute per-animal mean coverage delta (light - dark).
2. Report range, identify darkness-resistant and darkness-vulnerable
   animals.
3. Scatter plot: per-animal light coverage vs dark coverage.
4. Test whether cell type (Penk+ vs non-Penk) predicts darkness
   sensitivity (Mann-Whitney, N=10 vs N=4).

Output: Per-animal summary table.

**C2. Individual differences + neural correlation (Direction 3).**
Deferred until neural analysis pipeline produces per-animal HD
stability scores.

**C3. Return-path efficiency (Direction 5).** Lowest priority due to
topological constraints.

Implementation steps:
1. For each dead-end visit, identify the return to the parent junction.
2. Record the turn direction at the parent junction.
3. Compute per-junction turn consistency after dead-end returns.
4. Wilcoxon: consistency light vs dark (N=20).

Output: JSON results.

---

## Statistical Requirements

All tests are non-parametric as per project policy:

- Paired comparisons: Wilcoxon signed-rank
- Unpaired: Mann-Whitney U
- Correlations: Spearman rank
- Circular: Rayleigh test (for HD-related controls)
- Effect sizes: rank-biserial r = Z / sqrt(N) for Wilcoxon
- Multiple comparisons: Holm-Bonferroni within each analysis family
- Bootstrap CIs: 10,000 iterations for key estimates
- Report exact p-values and effect sizes for all tests

---

## How Each Direction Extends the Manuscript

### Current manuscript structure (v0.5)

1. Maze structure and coverage
2. Exploration strategies (turn rules, Markov)
3. Light vs dark: coverage (primary finding)
3b. Route stereotypy (Tier-1 mechanistic finding)
4. Light vs dark: speed and other metrics
5. HD sampling

### Proposed extensions

- **Direction 1 (temporal dynamics):** Becomes Results 3c. The
  coverage decline develops [gradually/immediately] within dark
  epochs, consistent with [representational degradation / strategy
  switch]. This adds a temporal dimension to the route stereotypy
  finding and connects to the HD drift literature.

- **Direction 4 (epoch adaptation):** Becomes a panel within Results
  3b or Supplementary. Shows whether route stereotypy is constant
  (obligatory) or changes (learned) over the session.

- **Direction 2 (corridor-specific):** Becomes a supplementary figure.
  Heatmap of per-cell coverage delta overlaid on maze layout. Visually
  reinforces the route stereotypy finding by showing which specific
  corridors are abandoned.

- **Direction 6 (cell-type Markov):** Becomes a panel within Results
  3b or Supplementary. Shows that route stereotypy operates at the
  structural level (cell-type transitions) not just at the individual-
  cell level.

- **Direction 3 (individual differences):** Descriptive part becomes
  Supplementary. Neural correlation part becomes a primary finding of
  the neural paper.

- **Direction 5 (return path):** Supplementary if significant. Dropped
  if null or underpowered.

### Impact on journal tier

With Directions 1 and 4 completed, the behaviour manuscript has:
- A robust primary finding (coverage drop)
- A mechanistic finding (route stereotypy: corridors/junctions drop,
  dead ends preserved)
- A temporal signature (gradual or immediate onset)
- A learning/adaptation test
- Rich supplementary analyses (corridor heatmap, cell-type Markov)
- A bridge to the neural paper (individual differences)

This strengthens the case for eLife or Behavioral Neuroscience over
STAR Protocols or J Neurosci Methods.

---

## References

Ajabi Z, Keinath AT, Wei XX, Brandon MP. 2023. "Population dynamics
of head-direction neurons during drift and reorientation." Nature 615,
892--899.

Avni R, Zadicario P, Eilam D. 2006. "Exploration in a dark open
field: a shift from directional to positional progression." Behav
Processes 72, 232--240.

Barry C, Ginzberg LL, O'Keefe J, Burgess N. 2012. "Grid cell firing
patterns signal environmental novelty by expansion." PNAS 109,
17687--17692.

Bhakti B et al. 2024. "Stochastic characterization of navigation
strategies in an automated variant of the Barnes maze." eLife 13,
e88648.

Bourin M, Hascoet M. 2003. "The mouse light/dark box test." Eur J
Pharmacol 463, 55--65.

Campagner D et al. 2022. "Innate heuristics and fast learning support
escape route selection in mice." Curr Biol 32, 2980--2987.

Eilam D, Golani I. 1989. "Home base behavior of rats (Rattus
norvegicus) exploring a novel environment." Behav Brain Res 34,
199--211.

Fischer LF, Mojica Soto-Albors R, Buck F, Harnett MT. 2020.
"Representation of visual landmarks in retrosplenial cortex." Curr
Biol 30, 1757--1770.

Fonio E, Benjamini Y, Golani I. 2009. "Freedom of movement and the
stability of its unfolding in free exploration of mice." PNAS 106,
21335--21340.

Gobbo F et al. 2026. "Navigational strategy dictates hippocampal
representation of space in an everyday memory task." bioRxiv.
doi:10.1101/2025.05.10.653115.

Koren Iton A et al. 2025. "NaviGraph: A graph-based framework for
multimodal analysis of spatial decision-making." bioRxiv.
doi:10.1101/2025.05.18.654725.

Lever C, Wills T, Cacucci F, Burgess N, O'Keefe J. 2002. "Long-term
plasticity in hippocampal place-cell representation of environmental
geometry." Nature 416, 90--94.

Muir GM et al. 2022. "Flexible cue anchoring strategies enable stable
head direction coding in both sighted and blind animals." Nat Commun
13, 5604.

Peyrache A, Lacroix MM, Petersen PC, Buzsaki G. 2015. "Internally
organized mechanisms of the head direction sense." Nat Neurosci 18,
569--575.

Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth
show rapid learning, sudden insight, and efficient exploration." eLife
10, e66175.

Schmitzer-Torbert N, Redish AD. 2002. "Development of path stereotypy
in a single day in rats on a multiple-T maze." Behav Neurosci 116,
1058--1070.

Stackman RW, Taube JS. 1997. "Firing properties of head direction
cells in the rat anterior thalamic nucleus: dependence on behavioral
factors." J Neurosci 17, 9020--9037.

Zugaro MB, Arleo A, Berthoz A, Wiener SI. 2003. "Rapid spatial
reorientation and head direction cells." J Neurosci 23, 3478--3482.
