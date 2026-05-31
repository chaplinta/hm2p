# Behaviour Science Plan: What Darkness Does to Navigation

Scientific hypotheses and their tests, grounded in the current findings
and literature. This is a hypothesis-driven plan, not a methods catalogue.

Status: 2026-05-30

---

## 1. The Puzzle

The current findings present a dissociation that demands explanation:

**Coverage drops in darkness** (p = 0.0003, r = 0.86) -- the strongest
behavioural effect in the dataset. Coverage per active minute also drops
(p = 0.001, r = 0.78), so this is not purely a locomotor artefact.

**But local turn rules are completely preserved.** Turn alternation (lag-1
autocorrelation: light = -0.158, dark = -0.170; p = 0.648), backtracking
rate (light: 0.482, dark: 0.505; p = 0.648), transition entropy (light:
1.221, dark: 1.186 bits/step; p = 0.165), and left-right bias (p = 0.870)
are all unchanged.

**This is paradoxical.** If the mouse is making the same types of decisions
at junctions with the same probabilities, why does it cover less of the
maze? The decision rules are the same, but the outcome is different. What
is the missing variable?

### Five candidate explanations

The coverage drop, given preserved local rules, could arise from:

A. **Fewer decisions** -- the mouse moves less, so it traverses fewer
   junctions per minute, yielding fewer opportunities to reach new cells.
   This is a speed/activity explanation.

B. **Different transition probabilities** -- the first-order Markov model
   still fits best, but the actual transition matrix changes in darkness.
   The mouse could visit the same junction types with the same turn
   statistics but route itself through the maze differently (e.g., biased
   toward a subset of junctions).

C. **Spatial contraction** -- the mouse develops a "home range" or
   preferred region in darkness and confines its exploration to a subset
   of the maze. Global turn statistics are preserved because the local
   rules operate correctly within the contracted range.

D. **Increased revisitation** -- the mouse revisits already-covered cells
   at a higher rate in darkness. This is distinct from (C): the mouse may
   not contract spatially, but may fail to track which cells it has
   visited, leading to redundant exploration.

E. **Temporal dynamics** -- the coverage drop is not uniform across the
   dark epoch. The mouse may explore normally for the first 15-30 seconds
   then "give up" or retreat, such that the coverage deficit accumulates
   late in the epoch.

These are not mutually exclusive. The science plan tests each one.

---

## 2. Key Open Questions

### Q1. Is the coverage drop a speed artefact?

**What we know:** Speed shows a trend toward reduction (p = 0.076;
primary-only p = 0.042). Coverage per active minute also drops (p = 0.001),
suggesting speed alone does not explain everything. But the coverage per
active minute result does not survive primary-only analysis (p = 0.175,
r = 0.49), leaving genuine ambiguity.

**What we need to resolve:** Are speed and coverage correlated within
sessions? If the sessions with the largest coverage drops are also the
sessions with the largest speed drops, the two effects may be
mechanistically coupled. Conversely, if coverage drops occur even in
sessions where speed does not change, speed is not the primary driver.

### Q2. Does the transition matrix change in darkness?

**What we know:** Transition entropy does not differ significantly
(p = 0.165). The same Markov order (first-order) is preferred in all
sessions. But transition entropy is a single scalar summary of a 23x23
matrix -- it is possible for the matrix to change substantially while
entropy remains similar (e.g., if the mouse trades one set of high-
probability transitions for a different set of equal probability).

**What we need to resolve:** Do specific edges in the maze graph get used
more or less in darkness? Does the mouse avoid peripheral branches?

### Q3. Does the mouse contract its range in darkness?

**What we know:** We have not tested this. The existing analyses report
only which fraction of cells are visited, not which cells are preferentially
abandoned. Rodents in darkness commonly increase thigmotaxis (wall-hugging)
and reduce excursion distance from a home base (Fonio et al. 2009; Avni
et al. 2006). In a maze with corridors rather than open space, the analogue
of thigmotaxis would be preferring central, well-connected maze regions
over peripheral dead-end branches.

### Q4. Does revisitation increase in darkness?

**What we know:** Dead-end visit rate does not differ between conditions
(p = 0.261). But this metric counts raw visits, not whether they are
first-time or repeat visits. A mouse could visit dead ends at the same
rate but visit the same dead ends repeatedly rather than new ones.

### Q5. Does the coverage deficit develop gradually or immediately?

**What we know:** We have not tested within-epoch temporal dynamics. The
draft mentions Supplementary Figure S3 (early vs late within dark epochs)
but no data have been computed. The HD drift literature suggests that
disorientation accumulates over 1-3 minutes (Stackman & Taube 1997; Muir
et al. 2022), so a gradual onset of coverage decline would be consistent
with accumulating navigational uncertainty.

### Q6. Do individual animals differ in their darkness response?

**What we know:** The coverage effect is robust (16/20 sessions show lower
coverage in dark). But we have not examined whether some animals are
"darkness-resistant" while others collapse. This matters for the neural
paper: if there are individual differences in the behavioural response to
darkness, they could correlate with individual differences in HD tuning
stability.

### Q7. Does the mouse develop a preferred location (home base) in darkness?

**What we know:** Rodents establish home bases in novel environments,
typically at enclosure corners, and make excursions of increasing distance
from this base (Fonio et al. 2009, PNAS; Eilam & Golani 1989). In
darkness, excursion range contracts. In the q-rose maze, the analogue
would be a "preferred hub" junction from which the mouse makes short
forays and returns. We have not tested this.

---

## 3. Hypotheses

### H1. Speed reduction partially but not fully explains the coverage drop

**Hypothesis:** The coverage drop in darkness is partly driven by reduced
locomotion, but a genuine exploration efficiency deficit persists after
controlling for speed. Sessions with large speed drops will show large
coverage drops, but the correlation will be imperfect.

**Rationale:** The existing data already point in this direction (coverage
per active minute drops, p = 0.001, though not in primary-only). This
hypothesis formalises that observation and adds a within-session
correlation test.

**Test:**
1. Compute the per-session difference in speed (dark - light) and the
   per-session difference in coverage (dark - light).
2. Spearman correlation between the two differences across N = 20
   sessions. If r > 0.7 and the residual coverage effect (after
   partialling out speed) is non-significant, speed is the primary
   driver. If r < 0.5 or the residual remains significant, there is an
   independent exploration effect.
3. An alternative approach: match dark and light epochs by speed (select
   light epochs where the mouse happened to move slowly, dark epochs
   where it moved fast) and test whether coverage still differs. This
   requires sufficient overlap in speed distributions.

**Expected outcome if true:** Spearman rho between speed-difference and
coverage-difference is 0.3-0.6 (moderate, not perfect). After removing
speed-matched epochs, a coverage difference of 3-5 percentage points
persists.

**Expected outcome if false:** Spearman rho > 0.8, and speed-matched
epochs show no coverage difference. This would mean the entire coverage
effect is locomotor.

**Impact on manuscript:** If true, this transforms the coverage finding
from a locomotor artefact into a genuine navigation strategy change. This
is the difference between "mice move less in the dark" (known, boring)
and "mice explore differently in the dark" (novel, interesting). It is
worth spending significant effort on this test.

**Priority: HIGH** -- this is the gatekeeper. If coverage is entirely
explained by speed, the remaining hypotheses become less interesting.

---

### H2. The transition matrix changes in darkness, even though entropy is preserved

**Hypothesis:** The first-order transition probabilities at specific
junctions shift in darkness -- the mouse redistributes its routing through
the maze graph -- but the overall entropy (predictability) of the
transitions is similar because the redistributed probabilities have
comparable variance.

**Rationale:** Entropy is a summary statistic. Two very different
transition matrices can have identical entropy. What matters for coverage
is not whether navigation is equally predictable in both conditions, but
whether specific edges are used more or less. A mouse that concentrates
its transitions on a central subgraph and avoids peripheral branches will
have lower coverage but could have similar entropy if it still makes
varied choices within that subgraph.

**Test:**
1. Compute the per-session first-order transition matrix separately for
   light and dark epochs. For each session, compute the element-wise
   difference (T_dark - T_light) for all non-zero entries.
2. Test whether specific edges (i -> j) are systematically increased or
   decreased in darkness across sessions. Use Wilcoxon signed-rank on each
   edge, with Holm-Bonferroni correction for the number of non-zero edges
   (approximately 40-50 edges in the maze graph).
3. Alternatively, compute the Jensen-Shannon divergence between the light
   and dark transition matrices per session. Test whether JSD > 0 across
   sessions (Wilcoxon on JSD values, one-sample test against zero). This
   is less powerful but avoids the multiple-comparisons problem of
   edge-by-edge tests.
4. Visualise: plot the maze graph with edge widths proportional to
   transition probability, separately for light and dark. Overlay the
   difference (dark - light) to see which routes shift.

**Expected outcome if true:** JSD is significantly > 0. A handful of
edges (especially those leading to peripheral dead-end branches) show
decreased usage in darkness. The transition matrix is more concentrated
on central hub junctions.

**Expected outcome if false:** JSD is near zero and no individual edges
survive correction. The transition matrix is genuinely stable, and the
coverage drop must be explained by reduced total movement (H1) or
revisitation patterns (H4).

**Impact on manuscript:** If true, this provides a mechanistic explanation
for the coverage drop that goes beyond speed: the mouse actively changes
its routing in darkness. This would be a substantial finding for the
behavioural paper and would connect to the neural paper (which routing
changes correlate with HD instability?).

**Literature context:** Bhakti et al. 2024 (eLife) used hidden Markov
models to detect strategy switching in Barnes maze, finding that mice
combine random, serial, and spatial strategies with context-dependent
transition probabilities. Our test is simpler (fixed-order Markov, no
HMM) but asks a related question: does the same animal switch its routing
strategy between conditions?

**Priority: HIGH** -- this is the most direct test of whether the coverage
drop reflects a strategy change.

---

### H3. The mouse contracts its spatial range in darkness

**Hypothesis:** In darkness, the mouse confines its exploration to a
subset of the maze (a "core region") and visits peripheral cells less
frequently. This spatial contraction produces the coverage drop without
requiring any change in local decision rules, because the rules operate
correctly but over a smaller portion of the maze.

**Rationale:** Range contraction in darkness is well documented in open
fields (Avni et al. 2006; Fonio et al. 2009). In a maze, the analogue
is preferring well-connected (high-degree) junctions and nearby corridors
over terminal branches that require committed excursions to dead ends.
This is functionally similar to a "home base" strategy: the mouse
establishes a familiar region and makes shorter forays from it.

**Test:**
1. **Number of unique cells visited per epoch** (already computed as
   coverage): subdivide by cell type (junction, corridor, dead end).
   Test whether dead ends specifically are visited less in darkness, or
   whether the coverage drop is uniform across cell types. Wilcoxon
   signed-rank per cell type, Holm-Bonferroni across the 3 types.
2. **Spatial extent**: Compute the graph diameter of the visited subgraph
   per epoch (the longest shortest-path between any two visited cells).
   Compare light vs dark. If the visited subgraph in darkness has a
   smaller diameter, the mouse is spatially contracted.
3. **Centroid displacement**: Compute the mean position (in graph
   coordinates) per epoch. Test whether the centroid shifts toward the
   maze centre in darkness.
4. **Per-cell visit frequency change**: For each of the 23 cells, compute
   the mean visit rate (visits per minute) in light vs dark. Test which
   cells show the largest decrease. Peripheral dead ends should show the
   largest drop if this hypothesis is correct.

**Expected outcome if true:** Dead-end coverage drops more than junction
coverage (interaction test). Visited subgraph diameter decreases by 1-2
edges in darkness. Peripheral cells (graph eccentricity > median) show
larger visit rate decreases than central cells.

**Expected outcome if false:** The coverage drop is uniform across cell
types and maze regions. This would favour H4 (revisitation) or H1 (speed)
instead.

**Impact on manuscript:** Range contraction in a maze would be the
structured-environment analogue of the well-documented open-field
darkening response. It would connect the q-rose maze findings to the
broader ethological literature on home-base behaviour and anxiety-related
exploration (Golani et al. 1999; Fonio et al. 2009). It is not novel as
a concept, but demonstrating it in a maze with quantified graph topology
is a useful contribution.

**Priority: HIGH** -- this tests a clear, falsifiable prediction about
the spatial distribution of the coverage deficit.

---

### H4. Darkness increases revisitation of already-covered cells

**Hypothesis:** In darkness, the mouse fails to efficiently track which
cells it has already visited, leading to increased redundant revisitation.
This reduces coverage because the mouse spends time re-exploring familiar
territory rather than reaching new cells.

**Rationale:** Efficient exploration requires knowing where you have and
have not been -- a spatial working memory that is plausibly dependent on
visual landmarks (or their internal representations). Without visual
cues, the mouse's representation of previously-visited locations may
degrade, leading it to revisit cells it has already explored. This is
distinct from the range contraction hypothesis (H3): H4 predicts
increased revisitation even of central cells, not just avoidance of
peripheral ones.

**Test:**
1. **Revisitation index**: For each epoch, compute the ratio
   (total cell transitions) / (unique cells visited). Higher values mean
   more revisitation per unique cell discovered. Compare light vs dark
   (Wilcoxon, N = 20).
2. **New cell discovery rate**: Compute the cumulative unique cells
   visited as a function of the number of cell transitions (not time).
   This is essentially the exploration efficiency metric at a per-
   transition resolution. Plot the mean curve for light and dark epochs.
   If the dark curve rises more slowly, the mouse is making more
   transitions before discovering each new cell.
3. **Per-cell revisitation entropy**: For each cell, compute the
   distribution of inter-visit intervals (number of transitions between
   consecutive visits). In darkness, if the mouse is "forgetting" which
   cells it has visited, inter-visit intervals should be shorter (more
   clustered revisitation to recently-visited cells rather than broad
   coverage sweeps).

**Expected outcome if true:** Revisitation index is higher in darkness
(Wilcoxon p < 0.05, r > 0.4). New cell discovery rate per transition is
lower. Inter-visit intervals are shorter (more clustered revisitation).

**Expected outcome if false:** Revisitation index is similar in light and
dark. The coverage drop is explained entirely by fewer transitions (H1)
or spatial contraction (H3) rather than by inefficient exploration.

**Impact on manuscript:** If true, this is the most interesting finding
because it implies that visual landmarks contribute to spatial working
memory for exploration tracking, not just to navigation per se. This
would connect to the cognitive map literature (O'Keefe & Nadel 1978) and
recent work on how hippocampal replay tracks exploration state. It would
also provide a clean prediction for the neural paper: if one RSP
population maintains more stable spatial representations in darkness, it
should correlate with lower revisitation.

**Priority: HIGH** -- this is the most mechanistically interesting
explanation and the most directly testable.

---

### H5. The coverage deficit develops gradually within dark epochs

**Hypothesis:** Coverage accumulates normally during the first 15-30
seconds of a dark epoch, then slows or stops. The mouse initially
navigates on inertia (using the path integration representation inherited
from the preceding light epoch), but as that representation degrades, it
becomes more cautious and reduces exploration.

**Rationale:** HD drift in darkness is gradual: Stackman & Taube (1997)
showed drift rates of roughly 5-10 deg/min in ADn, with substantial
variability. Muir et al. (2022) reported ~40% of HD cells becoming
unstable within minutes. A 1-minute dark epoch is on the edge of where
significant drift accumulates. If the coverage deficit correlates with
drift timescales, it would provide a temporal signature linking the
behavioural change to the degradation of the spatial representation.

**Test:**
1. Split each dark epoch into two halves: first 30 seconds and last 30
   seconds. Compute the number of new unique cells visited in each half.
   Compare first-half vs second-half (Wilcoxon paired, N = total dark
   epochs pooled within each session, then session-level comparison).
2. Compute speed in each half. If speed also declines within dark epochs,
   this could be a simple deceleration effect rather than an exploration
   strategy change. Both speed and unique cells must be examined.
3. As a control, perform the same split on light epochs. If the coverage
   decline is also present in light epochs (due to the ceiling effect --
   fewer new cells to discover in the second half of any epoch because
   some were already visited in the first half), the dark-specific effect
   must be assessed relative to the light baseline.
4. Compute the "coverage ratio" = (unique cells in last 30s) / (unique
   cells in first 30s) for light and dark epochs separately. Test whether
   the dark ratio is lower than the light ratio (Wilcoxon, N = 20).

**Expected outcome if true:** The dark coverage ratio is significantly
lower than the light ratio (p < 0.05). Speed may also decline within dark
epochs, but the coverage decline exceeds what speed alone predicts.

**Expected outcome if false:** Coverage accumulates at the same rate
throughout both light and dark epochs (adjusting for ceiling effects). The
coverage deficit is present from the very first seconds of darkness,
suggesting an immediate behavioural switch rather than gradual
degradation.

**Impact on manuscript:** A gradual onset would link the behavioural
change to the timescale of HD drift and path integration degradation,
providing a behavioural correlate of neural dynamics. An immediate onset
would suggest the mouse detects darkness immediately and switches to a
defensive strategy, which is also interesting but implies a different
mechanism (anxiety/cautiousness rather than representational failure).

**Priority: MEDIUM** -- important for mechanism, but the result is
interpretable either way.

---

### H6. The transition matrix shows increased hub-centrality in darkness

**Hypothesis:** In darkness, the mouse's transitions become more
concentrated around high-degree junctions (the "hubs" of the maze graph).
The stationary distribution of the dark transition matrix is more peaked
on hubs than the light transition matrix. This produces the coverage drop
because the mouse spends its time cycling through a central subgraph
rather than reaching peripheral cells.

**Rationale:** This is a more specific version of H3, grounded in graph
theory. In a graph with heterogeneous degree distribution, a random walker
with a forward bias will tend to visit high-degree nodes more often
(preferential attachment analogue). If darkness reduces the forward bias
(even subtly -- the effect does not have to show up as a significant
change in the average forward fraction), the walker will be more
"trapped" near hubs.

**Test:**
1. Compute the stationary distribution of the light and dark transition
   matrices per session (already available via
   `maze.analysis.stationary_distribution()`).
2. Compute the KL divergence between the stationary distribution and the
   uniform distribution (1/23 per cell). Test whether the dark
   distribution is further from uniform than the light distribution
   (Wilcoxon on KL divergence, N = 20).
3. Compute the Gini coefficient of the stationary distribution (a single
   measure of inequality). Higher Gini = more concentrated on fewer
   cells. Compare light vs dark.
4. Correlate the change in Gini (dark - light) with the change in
   coverage (dark - light). If they are negatively correlated (more
   concentrated stationary distribution = less coverage), this supports
   the mechanism.

**Expected outcome if true:** Dark Gini > light Gini (p < 0.05). The
stationary distribution in darkness is more concentrated on junctions
(1,0), (1,2), and (3,2) -- the three junctions with highest degree in
the q-rose maze.

**Expected outcome if false:** Stationary distributions are similar, and
the coverage drop is not driven by routing concentration.

**Impact on manuscript:** A graph-theoretic characterisation of the
navigation change would be methodologically clean and novel. It would
connect to the NaviGraph framework (Koren Iton et al. 2025) and position
the findings within graph-theoretic analysis of spatial navigation.

**Priority: MEDIUM** -- provides mechanistic depth if H3 is confirmed,
but is somewhat redundant if H3 is already strong.

---

### H7. Individual differences in darkness response correlate with neural
HD stability (cross-paper bridge)

**Hypothesis:** Animals that show the largest coverage drops in darkness
are the same animals whose RSP HD tuning is most disrupted by darkness.
The behavioural response to visual cue removal scales with the neural
response.

**Rationale:** If the coverage drop reflects degradation of the internal
spatial representation, then animals with more stable HD representations
in darkness (better path integration) should maintain better coverage.
This directly bridges the behavioural and neural papers.

**Test:**
1. Compute a per-animal "darkness sensitivity" score: the mean
   (coverage_light - coverage_dark) across sessions for each animal.
   For animals with multiple sessions, average to get one score per
   animal.
2. From the neural paper, compute a per-animal "HD stability" score:
   e.g., mean tuning curve correlation between light and first-dark-epoch
   across all HD cells in that animal.
3. Spearman correlation between darkness sensitivity and HD stability
   (N = 14 animals, or N = 11 primary-only).

**Expected outcome if true:** Negative correlation (r < -0.5):
animals with more stable HD representations show smaller coverage drops.

**Expected outcome if false:** No correlation: the behavioural and neural
responses to darkness are dissociated. This would be informative in its
own right -- it would suggest that the coverage drop is driven by
anxiety/cautiousness rather than representational failure.

**Impact on manuscript:** This is the key bridge finding between the
behaviour paper and the neural paper. It is too underpowered for the
behaviour paper alone (N = 11-14) but should be tested and reported as an
exploratory finding if it goes in the right direction. If it is strong,
it could be a primary finding of the neural paper.

**Priority: LOW for the behaviour paper, HIGH for the neural paper.**

---

### H8. Dark epochs that follow long light epochs show better coverage
than dark epochs that follow short light epochs (path integration
initialisation)

**Hypothesis:** The quality of path integration in darkness depends on
how well the spatial representation was "charged" during the preceding
light epoch. Dark epochs that were preceded by a long light epoch (or a
light epoch with extensive exploration) should show better coverage,
because the mouse enters darkness with a fresher, more accurate spatial
representation.

**Rationale:** This follows from the known dynamics of HD re-anchoring:
visual landmarks re-anchor HD rapidly (Zugaro et al. 2003, within a
single head sweep), so a long light epoch should produce a well-anchored
representation at the moment of lights-off. However, all light epochs in
this study are 1 minute, so the "preceding epoch" variation is limited.
A more tractable version of this hypothesis is: does coverage in a dark
epoch correlate with coverage in the immediately preceding light epoch?

**Test:**
1. Pair each dark epoch with its immediately preceding light epoch.
   Compute coverage for both. Spearman correlation between light-epoch
   coverage and the following dark-epoch coverage across all epoch pairs
   (pooled across sessions).
2. This tests "spatial momentum" -- whether a mouse that was actively
   exploring in light continues to explore in dark.
3. As a control, correlate dark-epoch coverage with the light epoch
   *two epochs prior* (should be weaker if there is a local carry-over
   effect).

**Expected outcome if true:** Positive correlation (r > 0.3) between
light coverage and subsequent dark coverage. The lag-1 correlation is
stronger than the lag-2 correlation.

**Expected outcome if false:** No correlation: each dark epoch is
independent of the preceding light epoch. This would suggest the
coverage drop is an immediate response to darkness, not a carry-over
from prior spatial representation quality.

**Impact on manuscript:** Modest. This is primarily a methodological
observation about epoch ordering effects. It could strengthen the
Discussion by showing that the coverage drop has temporal structure beyond
the simple light/dark contrast.

**Priority: LOW** -- interesting but not central.

---

### H9. The mouse develops a transient "home base" junction in darkness

**Hypothesis:** In darkness, the mouse selects one junction as a
temporary hub and makes short excursions from it, returning to that hub
more frequently than any single junction in light epochs. This is the
maze analogue of the "home base" behaviour described by Eilam & Golani
(1989) and Fonio et al. (2009) in open fields.

**Rationale:** Home-base behaviour is a well-documented rodent strategy
in novel or threatening environments. The mouse establishes a "safe"
location and incrementally extends its excursions. In an open field, this
manifests as a preferred corner. In a maze, the analogue is a preferred
junction. If the mouse perceives darkness as threatening (which is
plausible, given the increased immobility bouts), it might develop
home-base-like behaviour.

**Test:**
1. For each epoch (light and dark), identify the most-visited junction
   and compute the fraction of all junction visits that go to it (the
   "hub concentration ratio"). A ratio of 1/7 = 0.143 is uniform; values
   above 0.25-0.30 indicate a strong hub preference.
2. Compare the hub concentration ratio between light and dark (Wilcoxon,
   N = 20).
3. Compute the "excursion depth" from the hub: for each visit to the
   hub junction, measure the graph distance of the furthest cell visited
   before the next return to the hub. Compare mean excursion depth
   between light and dark.
4. Test whether the identity of the hub junction is consistent across
   dark epochs within a session (does the mouse return to the same
   junction each time the lights go off?).

**Expected outcome if true:** Hub concentration ratio is higher in
darkness (Wilcoxon p < 0.05). Excursion depth from the hub is shorter in
darkness. The same junction serves as hub across multiple dark epochs.

**Expected outcome if false:** Junction visit distributions are equally
dispersed in light and dark. The mouse does not develop a preferred hub.

**Impact on manuscript:** A maze analogue of home-base behaviour would be
a novel finding that connects ethological exploration literature to maze
navigation. It would also provide a concrete mechanistic explanation for
the coverage drop: the mouse reduces its range because it is operating
from a fixed base rather than ranging freely.

**Literature context:** Fonio et al. 2009 (PNAS) described how mouse
exploration in open fields unfolds as incrementally extending excursions
from a home base, with the home base established early and maintained
throughout the session. Tchernichovski et al. 1998 showed that home base
location is influenced by landmarks. Removing landmarks (darkness) could
cause the mouse to rely more heavily on a single familiar location.

**Priority: MEDIUM-HIGH** -- this is the most ethologically grounded
hypothesis and connects to a well-characterised behaviour. It is also
testable entirely within the existing behavioural data.

---

### H10. The maze wall geometry creates directional "traps" that are more
costly in darkness

**Hypothesis:** Certain maze corridors act as directional traps -- once
the mouse enters, it must traverse to a dead end before returning. In
light, the mouse can use visual landmarks to assess whether a branch is
worth entering. In darkness, it enters dead-end branches at similar rates
but takes longer to exit them (or makes more back-and-forth movements
within them), wasting time that would otherwise be spent reaching new
cells.

**Rationale:** The dead-end visit rate does not differ between conditions
(p = 0.261), but this measures only how often dead ends are visited, not
how efficiently they are traversed. If the mouse spends more time per
dead-end visit in darkness (e.g., pausing at the dead end, making
hesitant movements), this would reduce coverage without changing the visit
rate.

**Test:**
1. **Dead-end dwell time**: Compute the mean time (seconds) spent at each
   dead-end visit (from arrival to departure) in light vs dark. Compare
   (Wilcoxon, N = 20).
2. **Dead-end traversal efficiency**: Compute the number of cell
   transitions within a dead-end branch per visit (approach corridor +
   dead-end cell + return). In light, this should be close to 2 (enter,
   reverse). In darkness, if the mouse hesitates, it may be > 2.
3. **Speed within dead-end branches**: Compare the speed in dead-end
   branch corridors between light and dark (already partially computed
   as speed-by-node-type, but with the active-only filter caveat).

**Expected outcome if true:** Dead-end dwell time is longer in darkness
(> 1 second increase, Wilcoxon p < 0.05). Dead-end traversal requires
more transitions in darkness. This time cost reduces the mouse's ability
to reach other cells within the 1-minute epoch.

**Expected outcome if false:** Dead-end dwell time is similar. The mouse
traverses dead ends with equal efficiency regardless of lighting.

**Impact on manuscript:** A modest finding that adds mechanistic detail
to the coverage result. Not a standalone finding, but a useful supporting
analysis.

**Priority: LOW-MEDIUM** -- worth testing but unlikely to be a primary
finding.

---

## 4. Priority Ranking

### Tier 1: Must-test (primary findings)

| Hypothesis | Question | Scientific payoff |
|---|---|---|
| H1 | Speed vs coverage partial correlation | Gatekeeper: determines if the rest matters |
| H2 | Transition matrix changes | Direct test of strategy shift |
| H3 | Spatial range contraction | Directly explains coverage drop |
| H4 | Increased revisitation | Most mechanistically interesting |

**Order of operations:** H1 first (it determines the interpretation of
everything else), then H3 and H4 in parallel (they provide complementary
explanations), then H2 (confirms at the graph level).

### Tier 2: Should-test (supporting findings)

| Hypothesis | Question | Scientific payoff |
|---|---|---|
| H5 | Within-epoch temporal dynamics | Links to HD drift timescale |
| H9 | Home base in darkness | Ethological connection |
| H6 | Hub centrality increase | Graph-theoretic mechanistic depth |

### Tier 3: Exploratory / cross-paper

| Hypothesis | Question | Scientific payoff |
|---|---|---|
| H7 | Individual differences x neural | Bridge to neural paper |
| H8 | Epoch carry-over | Methodological observation |
| H10 | Dead-end traversal cost | Supporting detail |

---

## 5. How Each Finding Extends the Manuscript

### Current manuscript structure (5 results)

1. Maze structure and coverage
2. Exploration strategies (turn rules, Markov)
3. Light vs dark: coverage (primary finding)
4. Light vs dark: speed and other metrics
5. HD sampling

### Proposed extension

**Result 3 (coverage) should be expanded into a mechanistic story.**
Instead of simply reporting that coverage drops, the manuscript should
explain *why* it drops. The current version gestures at this (coverage
per active minute control, speed trend) but does not resolve the question.

The expanded version would read:

**Result 3a.** Coverage drops in darkness (existing finding, primary
result).

**Result 3b.** The coverage drop is partially but not fully explained by
reduced locomotion (H1). Sessions with larger speed drops show larger
coverage drops (Spearman r = X), but speed-matched comparisons still show
a coverage deficit of Y percentage points.

**Result 3c.** In darkness, the mouse contracts its spatial range toward
central maze regions (H3). Dead-end coverage drops more than junction
coverage. The visited subgraph diameter decreases by Z edges.
Alternatively: the mouse increases revisitation of central cells (H4),
as measured by a higher revisitation index in darkness.

**Result 3d.** (Supplementary) The coverage deficit develops gradually
within dark epochs (H5), consistent with the timescale of path
integration degradation. OR: The deficit is immediate, suggesting an
anxiety/cautiousness response rather than gradual representational
failure.

This transforms the coverage finding from a single statistical test into
a mechanistic narrative. The narrative structure is:

> Visual cue removal reduces maze coverage. This is partly locomotor, but
> a genuine exploration strategy change persists after controlling for
> speed. The strategy change manifests as [range contraction / increased
> revisitation / hub concentration], developing [gradually / immediately]
> within dark epochs. Local decision rules (turn alternation, Markov
> transition probabilities) are preserved, indicating that the
> exploration deficit operates at the level of route selection rather
> than individual turn decisions.

### Impact on publication

Without the mechanistic extension, the behaviour paper is a thin
descriptive report: "mice cover less of the maze in darkness." This is
publishable (as acknowledged in the draft: STAR Protocols, J Neurosci
Methods, Sci Reports) but is not particularly compelling.

With the mechanistic extension, the paper becomes: "Visual cue removal
selectively disrupts spatial exploration strategy while preserving local
navigation rules, revealing a dissociation between route-level and
decision-level spatial computation." This is a more interesting story and
could support a mid-tier journal (eLife, J Neuroscience).

The key is that the *dissociation* between local rule preservation and
global coverage reduction is the novel finding. The mechanistic
explanation (H1-H4) clarifies what that dissociation means.

---

## 6. Specific Data/Analysis Requirements per Hypothesis

### H1: Speed-coverage correlation

**Existing data:** Session-level speed and coverage are in
`behaviour-results.json`. All data required.
**New computation:** Spearman correlation, speed-matched epoch
subsampling.
**Feasibility:** Immediate -- no new data needed.

### H2: Transition matrix comparison

**Existing data:** Light/dark transition matrices computable from node
sequences in `behaviour-results.json` or from raw position data.
**New computation:** Per-edge Wilcoxon tests, JSD computation,
visualisation of difference matrix on maze graph.
**Feasibility:** Immediate -- `transition_matrix()` already implemented,
just needs to be called separately for light and dark within each
session.

### H3: Spatial range contraction

**Existing data:** Per-cell visit counts are derivable from occupancy.
Requires frame-level position + light condition.
**New computation:** Per-cell-type coverage (junction/corridor/dead end),
visited subgraph diameter, centroid displacement.
**Feasibility:** Requires frame-level data from sync.h5. The cell-type
coverage test is straightforward. Visited subgraph diameter requires
adding a function to `maze.topology` or `maze.analysis`.

### H4: Revisitation index

**Existing data:** Cell transition sequences per epoch.
**New computation:** Revisitation index (transitions / unique cells),
per-transition unique cell discovery curve.
**Feasibility:** Straightforward from existing cell sequences.

### H5: Within-epoch dynamics

**Existing data:** Frame-level position + light condition + timestamps.
**New computation:** Split epochs at midpoint, compute coverage in each
half.
**Feasibility:** Requires frame-level data. Straightforward.

### H6: Hub centrality

**Existing data:** Transition matrices (from H2).
**New computation:** Stationary distribution, Gini coefficient, KL
divergence from uniform.
**Feasibility:** `stationary_distribution()` already implemented.

### H7: Individual differences (cross-paper)

**Existing data:** Per-animal coverage difference.
**New computation:** Per-animal HD stability score (from neural analysis).
**Feasibility:** Deferred until neural analysis is complete.

### H9: Home base

**Existing data:** Per-junction visit counts per epoch.
**New computation:** Hub concentration ratio, excursion depth from hub.
**Feasibility:** Requires defining "excursion" from the node sequence --
new analysis, moderate complexity.

### H10: Dead-end traversal cost

**Existing data:** Dead-end visit data (from `dead_end_visits()`).
**New computation:** Per-visit dwell time, within-branch transition count.
**Feasibility:** `dead_end_visits()` returns visit counts; dwell time
requires frame-level timestamps per visit. Moderate.

---

## 7. Confounds and Controls for Each Hypothesis

### H1: Speed confound for coverage

**Confound:** Speed is correlated with coverage mechanically (faster
mouse = more transitions = more cells visited). The correlation test (H1)
may simply recover this mechanical relationship rather than revealing a
neural/cognitive mechanism.
**Control:** Compute coverage per transition (unique cells / total cell
transitions) as well as coverage per minute. If coverage per transition
also drops, speed is not the explanation -- the mouse is making
transitions at normal rate but covering less ground per transition.

### H2-H3: Epoch duration confound

**Confound:** Light and dark epochs are fixed at 1 minute, but usable
time within an epoch varies (due to bad_behav exclusion, edge effects).
If dark epochs have systematically less usable time (e.g., because the
mouse is immobile more often), coverage will drop mechanically.
**Control:** Normalise all epoch-level metrics by usable seconds within
each epoch. Report metrics as "per usable minute" or "per transition"
rather than per epoch.

### H3: Position tracking reliability in darkness

**Confound:** If position tracking is noisier in darkness (despite
infrared illumination), discretised cell assignments may be less accurate,
producing apparent changes in spatial distribution that are actually
tracking artefacts.
**Control:** The DLC model uses infrared, and the manuscript states
tracking quality does not differ. But confirm this by comparing
per-keypoint likelihood scores in light vs dark. If mean likelihood
drops in darkness, the tracking confound must be addressed. (This is
already flagged as a deferred control analysis in the manuscript.)

### H4: Ceiling effect for revisitation

**Confound:** In light epochs with high coverage, there are few
unvisited cells left, so any subsequent transition is likely a revisit.
In dark epochs with lower coverage, more cells remain unvisited, so any
subsequent transition has a higher chance of being a new visit. This
mechanical ceiling effect biases against finding increased revisitation
in darkness.
**Control:** Compute revisitation at matched coverage levels (compare
light and dark epochs with similar unique cell counts). Or compute
coverage per transition rather than raw revisitation index.

### H5: First/second-half ceiling in both conditions

**Confound:** In any 1-minute epoch, coverage necessarily decelerates as
more cells are visited (fewer new cells remain). The first half will
always have more new discoveries than the second half. The question is
whether this deceleration is greater in darkness than in light.
**Control:** Compute the ratio (second-half unique cells) / (first-half
unique cells) and compare this ratio between conditions.

### H9: Session duration confound for home base

**Confound:** Home base behaviour may develop over the course of the
session as the mouse becomes familiar with the maze. If dark epochs
occur later in the session (which they do, alternating with light), any
home-base effect could be a time-on-task effect rather than a
darkness-specific effect.
**Control:** Compare hub concentration ratio in early dark epochs
(epochs 1-5) vs late dark epochs (epochs 6+). If hub concentration
increases with session time equally in light and dark, it is a
familiarity effect, not a darkness effect.

---

## 8. What We Explicitly Do NOT Test (and Why)

### Cognitive map formation

We do not test whether mice are building or using a cognitive map of the
q-rose maze. The maze is too small and the sessions too short for
meaningful map-learning dynamics. Rosenberg et al. (2021) found learning
effects in their 63-junction maze over hundreds of reward experiences;
our 7-junction maze with no reward is unlikely to show the same
trajectory. The first-order Markov result already suggests that mice do
not develop complex sequential strategies in this maze.

### Goal-directed navigation

There is no reward in this task. Metrics like "path optimality" require a
defined goal. We compute path efficiency relative to dead ends as a
proxy, but this is a weak measure. The free-exploration nature of the
task means goal-direction metrics are not meaningful.

### Allocentric vs egocentric strategy classification

The maze is too small and the tracking too coarse (cell-level
discretisation) to reliably classify individual traversals as allocentric
vs egocentric. This distinction is best tested with the neural data
(population decoding of position vs HD) rather than from behaviour alone.

### Multi-session learning

Most animals have only 1-2 sessions. We cannot meaningfully track
cross-session learning or strategy evolution with this dataset.

---

## 9. Connections to the Neural Paper

The hypotheses above are designed to serve the behaviour paper
independently, but each has implications for the neural analysis:

| Behaviour hypothesis | Neural prediction |
|---|---|
| H1: Speed drives coverage | Speed modulation of Penk+ vs non-Penk firing rates may account for cell-type coverage differences |
| H2: Transition matrix changes | Specific edge changes may correlate with specific HD angle instabilities |
| H3: Range contraction | Penk+ cells in peripheral maze regions may show greater tuning instability (less sampling) |
| H4: Increased revisitation | The population with more stable HD in darkness should correlate with less revisitation |
| H5: Gradual onset | Onset of coverage decline should coincide with onset of HD drift in the neural data |
| H7: Individual differences | The bridge hypothesis -- correlate behavioural darkness sensitivity with neural HD stability |
| H9: Home base | The "home junction" may be the junction where HD decoding accuracy is highest in darkness |

The strongest bridge is H7, but it requires both papers' data. H5 is the
most actionable bridge: if coverage decline onset matches HD drift onset,
it provides a clean within-session temporal correspondence.

---

## References

Ajabi Z, Keinath AT, Brandon MP. 2023. "Population dynamics of
head-direction neurons during drift and reorientation." Nature 615,
892--899.

Avni R, Zadicario P, Eilam D. 2006. "Exploration in a dark open field:
a shift from directional to positional progression." Behav Processes 72,
232--240.

Bhakti B et al. 2024. "Stochastic characterization of navigation
strategies in an automated variant of the Barnes maze." eLife 13, e88648.

Eilam D, Golani I. 1989. "Home base behavior of rats (Rattus norvegicus)
exploring a novel environment." Behav Brain Res 34, 199--211.

Fonio E, Benjamini Y, Golani I. 2009. "Freedom of movement and the
stability of its unfolding in free exploration of mice." PNAS 106,
21335--21340.

Gobbo F et al. 2026. "Navigational strategy dictates hippocampal
representation of space in an everyday memory task." bioRxiv.

Golani I, Benjamini Y, Eilam D. 1993. "Stopping behavior: constraints on
exploration in rats (Rattus norvegicus)." Behav Brain Res 53, 21--33.

Koren Iton A et al. 2025. "NaviGraph: A graph-based framework for
multimodal analysis of spatial decision-making." bioRxiv.

Muir GM et al. 2022. "Flexible cue anchoring strategies enable stable
head direction coding in both sighted and blind animals." Nat Commun 13,
5604.

O'Keefe J, Nadel L. 1978. The Hippocampus as a Cognitive Map. Oxford
University Press.

Rosenberg M et al. 2021. "Mice in a labyrinth show rapid learning,
sudden insight, and efficient exploration." eLife 10, e66175.

Stackman RW, Taube JS. 1997. "Firing properties of head direction cells
in the rat anterior thalamic nucleus: dependence on behavioral factors."
J Neurosci 17, 9020--9037.

Tchernichovski O, Benjamini Y, Golani I. 1998. "The dynamics of
long-term exploration in the rat." Biol Cybern 78, 423--432.

Zugaro MB, Arleo A, Berthoz A, Wiener SI. 2003. "Rapid spatial
reorientation and head direction cells." J Neurosci 23, 3478--3482.
