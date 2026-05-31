# Advanced Behavioural Analysis Plan for the q-Rose Maze Manuscript

Plan for analyses beyond the current v0.4.1 manuscript. Organised by theme,
prioritised, with effort estimates. All analyses are behavioural only (no
neural data). Non-parametric statistics throughout.

**Dataset:** 20 usable sessions from 14 animals; 11 primary-only (1 per
animal). 23 cells, 7 T-junctions, 9 dead ends, 7 corridors. Alternating
1-min light/dark epochs (~10-15 per condition per session). Camera is
infrared; tracking quality identical in light and dark.

**Current manuscript state:** Five main results: (1) maze coverage and
occupancy, (2) turn bias and first-order Markov, (3) coverage drop in
darkness (primary finding, p=0.0003, r=0.86), (4) speed trend in darkness
(p=0.076), (5) HD sampling non-uniformity. First-order Markov preferred
over second-order in all 20 sessions.

---

## 1. Deeper Markov Models

### 1.1 Higher-Order Markov Models (3rd, 4th order)

**What it tests:** Whether navigation in the q-rose maze contains structure
beyond immediate pairwise transitions. The current analysis shows 0/20
sessions prefer second-order by BIC. This analysis asks whether the negative
finding extends to all higher orders, or whether some intermediate order
captures structure that second-order misses.

**Light/dark:** Compare preferred order between conditions. If darkness
degrades spatial memory, navigation should become more memoryless (lower
preferred order). Conversely, if mice rely on stereotyped wall-following in
darkness, higher-order structure could increase.

**Method:** Fit Markov chains of orders 1-5 to the cell sequence. Use BIC
for model selection (AIC tends to over-fit with small sample sizes). For
orders 3+, the parameter space grows as n_cells^k, which with 23 cells
becomes enormous -- but the graph constrains most entries to zero. Count
only reachable k-tuples as free parameters. Compute cross-entropy on
held-out data (even-odd epoch split) as a validation check independent of
information criteria.

**Tools:** Extend `maze.analysis.markov_order_comparison()` to accept
arbitrary order. Use the existing `cross_entropy()` framework. numpy only.

**Priority: Low.**

**Rationale:** The negative finding for second-order is already interpretable
and is a consequence of the small state space (23 cells, 7 junctions).
Higher orders will face the same problem but worse -- the parameter count
explodes while the amount of data per epoch is limited (~50-200 cell
transitions per minute). The current BIC analysis is already clean and
interpretable. Going higher risks over-fitting on small data and producing
results that are hard to communicate. Report the 1st-vs-2nd comparison and
note that higher orders were not pursued due to parameter estimability.

**Effort:** Low (2-3 hours). The code generalisation is straightforward.

**Expected outcome:** All orders 2+ will be disfavoured by BIC. This is a
predictable null that does not add scientific value.

---

### 1.2 Hidden Markov Models (HMMs) for Latent Behavioural States

**What it tests:** Whether mice switch between discrete latent exploration
strategies (e.g., "directed exploration," "local search," "wall-following,"
"resting") during maze navigation, and whether the probability of being in
each state changes between light and dark.

**Light/dark:** Compare the posterior state occupancy probabilities between
conditions. If darkness shifts mice toward a "cautious" or "local" state,
this would explain the coverage drop without requiring a global speed
reduction.

**Method:**

1. **Observation model.** Define observable features per time step (at the
   imaging frame rate, ~9.6 Hz):
   - Speed (cm/s)
   - Angular head velocity (deg/s)
   - Maze node type (junction/corridor/dead-end, one-hot or categorical)
   - Time since last cell transition (frames)
   - Optionally: instantaneous turn angle at last junction

2. **Model fitting.** Fit Gaussian HMMs with K = 2, 3, 4, 5 hidden states
   using hmmlearn (sklearn-compatible). Select K by BIC across the model
   family. Fit separately for each session, then compare state properties
   across sessions.

3. **State characterisation.** For each inferred state, compute the emission
   distribution over the observables. Interpret states by their speed,
   AHV, and maze-location profiles (e.g., "fast corridor traversal," "slow
   junction dwelling," "immobile").

4. **Light/dark comparison.** Compute fraction of time in each state during
   light vs dark epochs (Wilcoxon signed-rank, N=20 sessions). Test
   whether darkness shifts the state occupancy distribution.

5. **Temporal dynamics.** Examine state transition probabilities: does the
   probability of entering a "cautious" state increase at light-off
   transitions? Compute peri-event state probability around light
   transitions.

**Literature precedent:**
- Bhakti et al. (2024, eLife) used a Markov chain with four navigation
  strategies (random, serial-CW, serial-CCW, spatial) in Barnes maze,
  showing strategy switching every ~6 visits.
  Reference: Bhakti, Bhatt et al. 2024. "Stochastic characterization of
  navigation strategies in an automated variant of the Barnes maze." eLife
  13, e88648. doi:10.7554/eLife.88648
- HMMs have been applied to rodent locomotion in visual cliff tests
  (Aoyama et al. 2025, PLOS ONE) and depth-related behaviour, identifying
  3-state models (resting, exploring, navigating).
- Mixtures of learning strategies in rodent reversal learning have been
  modelled with block-HMMs (Ashwood et al. 2022, Nat Neurosci).

**Confounds:**
- Speed and maze location are correlated (mice slow at dead ends, speed in
  corridors). The HMM may simply rediscover the speed/location structure
  rather than revealing genuine latent states.
- With only 20 sessions, cross-session consistency of state definitions is
  a concern. States fit independently per session may not be comparable.
  Mitigation: fit a single HMM to concatenated data from all sessions with
  session boundaries masked, or use a hierarchical HMM.
- 1-minute epochs are short for HMM inference (60s x 9.6 Hz = ~576
  frames). With K=3 states, the model may not have enough transitions to
  reliably estimate transition probabilities within a single epoch.

**Tools:** `hmmlearn` (pip installable). Feature extraction from existing
sync.h5 fields. scipy for statistics.

**Priority: Medium.**

**Rationale:** This is the most promising of the Markov extensions. It goes
beyond the transition-matrix approach by allowing latent states that
integrate multiple observables. If the HMM reveals a "cautious exploration"
state that becomes more prevalent in darkness, this provides a mechanistic
interpretation of the coverage drop: mice are not simply slower, they are in
a different behavioural mode. However, the risk is that the HMM just
rediscovers speed (fast state vs slow state), which adds complexity without
insight. The key test is whether the HMM provides explanatory power beyond
a simple speed threshold.

**Effort:** Medium (1-2 days). Feature extraction is straightforward; model
fitting with hmmlearn is fast; the main work is in validation, state
interpretation, and light/dark comparison.

**Expected outcome:** 2-3 states will be preferred by BIC. At least one
state will be a "moving" state and one an "immobile" state. The question is
whether a third state emerges that maps to something behaviourally
meaningful beyond speed (e.g., junction deliberation, dead-end dwelling).

---

### 1.3 Transition Matrix Properties

**What it tests:** Whether the Markov transition structure differs between
light and dark beyond what transition entropy captures.

**Light/dark:** Compare matrix properties between conditions.

**Method:**

a. **Stationary distribution.** Already implemented
   (`maze.analysis.stationary_distribution()`). Compare the stationary
   distribution between light and dark transition matrices. If darkness
   shifts the stationary distribution toward dead ends, it implies the
   transition structure itself (not just speed) drives increased dead-end
   dwelling. Test: Wilcoxon on per-cell stationary probabilities, or
   Jensen-Shannon divergence between light and dark stationary distributions.

b. **Transition matrix similarity.** Compute the Frobenius norm or
   Jensen-Shannon divergence between the light and dark transition matrices
   for each session. Test whether this distance is greater than expected
   from random splits of same-condition data (bootstrap null: split light
   epochs randomly into two halves and compute the within-light matrix
   distance).

c. **Entropy rate over time.** Compute transition entropy in sliding windows
   across the session (e.g., 3-minute windows, 1-minute steps). Test
   whether entropy changes systematically across the session (fatigue,
   learning) or between conditions (Spearman correlation of entropy with
   epoch number).

d. **Mixing time.** Compute the second-largest eigenvalue of the transition
   matrix (modulus). Smaller values mean faster mixing (the Markov chain
   "forgets" its history faster). If the dark transition matrix has faster
   mixing, navigation in darkness is more "random" -- the mouse's next
   position is less predictable from its current position. If slower
   mixing, the mouse is trapped in local cycles (e.g., shuttling between
   two cells).

**Confounds:**
- Transition matrices estimated from short epochs (1 min) are noisy. Pool
  all light epochs and all dark epochs within a session before computing
  the matrix.
- Structural zeros in the matrix (impossible transitions due to walls)
  affect the eigenvalue spectrum. Restrict analysis to the submatrix of
  cells actually visited.

**Tools:** numpy (eigendecomposition), scipy (entropy). Extend existing
`transition_matrix()` and `stationary_distribution()`.

**Priority: Medium.**

**Rationale:** The stationary distribution comparison (a) is quick and
directly interpretable. If the Markov-predicted long-run occupancy shifts in
darkness, it complements the observed occupancy data. The mixing time (d) is
a compact summary of how "structured" the navigation is. These are useful
supplementary metrics that enrich the Markov analysis without requiring a
new modelling framework.

**Effort:** Low-medium (half day). Most of the infrastructure exists.

---

## 2. Graph-Theoretic Analyses (NaviGraph-Inspired)

### 2.1 What is implemented

The existing `maze.topology` and `maze.analysis` modules already provide:
- Graph construction from polygon boundary (adjacency, shortest paths)
- Junction/corridor/dead-end classification
- Node-level transition matrices
- Exploration efficiency (new nodes per window)
- Dead-end visit analysis
- Path efficiency (actual vs optimal path length)
- Monotonic path detection (goal-directed runs)
- Random walk simulation (with optional forward bias)

### 2.2 Missing: Graph Centrality Metrics

**What it tests:** Whether mice preferentially visit structurally central
cells (high betweenness or degree centrality) and whether this preference
changes in darkness.

**Method:**
a. Compute betweenness centrality and closeness centrality for each cell in
   the maze graph (networkx or manual computation -- the graph is small
   enough for brute-force).
b. Correlate centrality with observed occupancy (Spearman). High centrality
   cells are transit hubs; mice should visit them more often.
c. Compare occupancy-centrality correlation between light and dark.
   Prediction: if mice in darkness fall back on "through-routes" (high
   centrality corridors), the correlation should increase.

**Tools:** networkx (already in CLAUDE.md skills), or manual computation.

**Priority: Low.**

**Rationale:** The q-rose maze is small (23 cells) and the centrality
structure is fixed and obvious (the central junction cluster has highest
betweenness). This analysis will mostly confirm the obvious: junctions are
visited more because they are on every path. The light/dark comparison is
more interesting but unlikely to survive the noise of 1-minute epochs.

**Effort:** Low (2-3 hours). The graph is tiny.

---

### 2.3 Missing: Graph-Theoretic Path Optimality

**What it tests:** Whether mice take efficient paths (close to shortest
graph path) between revisits to the same cell, and whether path optimality
differs between light and dark.

**Method:**
a. For every pair of consecutive visits to the same cell (or same junction),
   compute the number of cell transitions between visits (actual path
   length) and the shortest graph path.
b. Path optimality = shortest_path / actual_path (1.0 = perfectly
   efficient; lower = more meandering).
c. Compare path optimality between light and dark epochs (Wilcoxon).

**Literature:** Rosenberg et al. (2021) used this approach for reward-
directed paths. In our free-exploration task, there is no explicit goal,
but revisit efficiency provides a proxy for spatial memory quality. If mice
remember the maze structure, they should take efficient return paths. In
darkness, if spatial memory degrades, return paths should become less
efficient.

**Confounds:**
- In a small maze, many revisits are incidental (the mouse wanders back
  through the same junction without intending to revisit). Restrict to
  revisits of dead ends (the mouse must intentionally navigate to a dead
  end -- it cannot be "on the way" to somewhere else since dead ends have
  only one neighbour).
- Speed differences confound path length in time units but not in graph
  steps. Use graph-step count, not time.

**Priority: Medium.**

**Rationale:** Dead-end revisit efficiency is a clean metric that
specifically tests spatial memory. If the mouse remembers where a dead end
is and can navigate back to it efficiently even in darkness, this suggests
intact spatial representation. If efficiency drops in darkness, it suggests
spatial memory loss. This connects directly to the HD drift story: if the
internal compass drifts, return paths should degrade. This could be a
supporting finding for the coverage drop (Result 3 in the manuscript).

**Effort:** Medium (half day). Requires identifying revisit events and
computing intervening path lengths.

---

### 2.4 Missing: Decision-Point Dwell Time

**What it tests:** Whether mice pause at junctions (decision points) longer
than at corridor cells, and whether junction dwell time increases in
darkness (suggesting deliberation under uncertainty).

**Method:**
a. For each cell visit, compute dwell time (consecutive frames in that
   cell).
b. Group by node type (junction, corridor, dead end).
c. Compare dwell times across types (Friedman test, N=20).
d. Compare junction dwell time in light vs dark (Wilcoxon).

**Literature:**
- Decision-related pausing at choice points is well-documented in spatial
  navigation (Redish 2016; Johnson & Redish 2007 "vicarious trial and
  error"). If mice pause longer at junctions in darkness, this suggests
  increased uncertainty or deliberation when visual cues are absent.

**Confounds:**
- Dwell time at dead ends will be long (mice must reverse), which is
  structural, not deliberation. The junction vs corridor comparison is the
  key contrast.
- Speed differences between light and dark will affect dwell time
  mechanically (slower mice spend more time in every cell). Normalise by
  median speed per epoch, or use number of cell transitions rather than
  time.

**Priority: Medium.**

**Rationale:** This adds a qualitative dimension to the current analysis.
The manuscript currently reports transition entropy (unchanged), turn bias
(unchanged), and backtracking (unchanged) in darkness. Junction dwell time
adds a temporal dimension: even if the mouse makes the same decisions, does
it take longer to decide? This is a more sensitive measure of uncertainty
than choice outcomes.

**Effort:** Low (3-4 hours). Frame-level data is available in sync.h5.

---

## 3. Sequence Analysis Beyond Markov

### 3.1 Lempel-Ziv Complexity of Navigation Sequences

**What it tests:** Whether navigation sequences have higher or lower
algorithmic complexity in darkness, as a measure of sequence
predictability/stereotypy that is independent of any Markov assumption.

**Light/dark:** Compare LZC between conditions. If mice follow more
stereotyped routes in darkness (wall-following, repeated circuits), LZC
should decrease. If mice explore more randomly (loss of spatial map), LZC
should increase.

**Method:**
a. Convert the cell visit sequence to a symbol string (23 possible symbols).
b. Compute the normalised Lempel-Ziv complexity (LZC / LZC_of_random)
   for light and dark epochs separately.
c. Compare with Wilcoxon (N=20 sessions).

**Literature:**
- LZC has been used for neural complexity (EEG, LFP) during sleep/wake
  states in rats (Abásolo et al. 2015, J Neurophysiol). Its application to
  behavioural sequences is less common but conceptually clean: LZC counts
  the number of distinct "words" in the sequence, which captures structure
  that Markov models may miss (e.g., long-range repeats, periodic
  patterns).
- Normalisation against a random sequence of the same length and alphabet
  size controls for sequence length effects.

**Confounds:**
- Short sequences (1-minute epochs produce ~50-200 cell transitions) have
  high LZC variance. Pool all same-condition epochs within a session.
- LZC is sensitive to alphabet size. Using 23 cell symbols may produce
  near-maximal LZC because most transitions are unique. Consider reducing
  the alphabet to node types (junction/corridor/dead-end = 3 symbols) or
  to junction identities only (7 symbols).

**Priority: Medium.**

**Rationale:** LZC complements transition entropy by capturing non-Markov
structure. Transition entropy measures local predictability (given the
current cell, how predictable is the next). LZC measures global sequence
complexity (how compressible is the entire route). If transition entropy is
unchanged but LZC changes, it implies structure at longer timescales (e.g.,
repeated circuits that are not captured by pairwise transitions). This is a
compact, non-parametric metric with a clean interpretation.

**Effort:** Low (3-4 hours). LZC implementation is ~20 lines of code.

---

### 3.2 Run-Length Distributions

**What it tests:** Whether the distribution of consecutive visits to the
same corridor branch changes between light and dark.

**Method:**
a. Partition the maze into branches (subgraphs rooted at each junction).
   The q-rose maze has a natural branch structure: the central corridor
   complex and several peripheral dead-end branches.
b. Compute "run lengths": the number of consecutive cell transitions spent
   within each branch before switching to a different branch.
c. Compare run-length distributions between light and dark.

**Light/dark:** If mice in darkness engage in more local exploration (staying
within one branch), run lengths should increase. If they shuttle between
branches more randomly, run lengths should decrease.

**Confounds:**
- Branch definition is somewhat arbitrary in this maze. Use the natural
  hierarchy: each dead-end branch radiates from a junction.
- Short epochs limit the number of branch switches per epoch.

**Priority: Low.**

**Rationale:** This is a specialised metric that adds limited insight beyond
what transition entropy and coverage already capture. The branch structure
of the q-rose maze is shallow (most branches are 1-2 cells deep), so run
lengths will be very short.

**Effort:** Low (3-4 hours).

---

### 3.3 Mutual Information Between Successive Choices

**What it tests:** Whether consecutive junction choices carry statistical
dependencies beyond lag-1 (which is captured by the sequential turn
autocorrelation).

**Method:**
a. Extract the sequence of left/right choices at junctions.
b. Compute mutual information I(choice_t; choice_{t+k}) for lags
   k = 1, 2, ..., 10.
c. Compare the MI decay profile between light and dark.
d. A faster MI decay in darkness would indicate shorter-range choice
   dependencies (more memoryless navigation).

**Confounds:**
- MI estimation from small samples is biased upward (Panzeri & Treves 1996).
  Use a bias correction (Panzeri-Treves or jackknife) or compare to
  shuffled null.
- The number of junction choices per epoch is limited (~20-60). MI at long
  lags will be noisy.

**Priority: Low.**

**Rationale:** The sequential turn autocorrelation (lag-1) is already
reported and is significant (negative, indicating alternation). Higher-lag
MI is a natural extension but unlikely to reveal anything new in this small
maze. The decay should be fast because the maze is small (most information
about maze structure can be captured in 1-2 steps).

**Effort:** Low (3-4 hours).

---

## 4. Temporal Dynamics

### 4.1 Epoch-Number Effects (Adaptation to Repeated Darkness)

**What it tests:** Whether mice adapt to repeated dark epochs across a
session. Does the coverage drop diminish by the 5th dark epoch compared to
the 1st? Does speed recover?

**Light/dark:** This is inherently about light/dark temporal dynamics.

**Method:**
a. Number each light epoch (1, 2, 3, ...) and each dark epoch (1, 2, ...)
   within each session.
b. Compute per-epoch metrics: coverage, speed, transition entropy.
c. Fit Spearman correlation between epoch number and each metric,
   separately for light and dark.
d. Compare early (epochs 1-3) vs late (epochs 5+) within each condition
   (Wilcoxon).

**Literature:**
- HD cells show variable drift rates in darkness (Stackman & Taube 1997)
  but the drift does not systematically change with repeated dark exposures
  in the same session. Behavioural adaptation to repeated darkness is less
  studied.
- In open fields, speed reduction in darkness typically persists across
  repeated dark exposures within a session (Avni et al. 2006), but
  adaptation has been reported with longer inter-exposure intervals.

**Confounds:**
- General fatigue/satiation confounds epoch-number effects. Mice may slow
  down throughout the session regardless of light condition. Must test
  epoch-number effects within each condition separately AND the interaction
  (does the light-dark difference change with epoch number?).
- Tether effects may worsen over time, producing artefactual late-session
  speed reduction. The bad_behav mask partially controls this but may not
  capture subtle tether drag.

**Priority: High.**

**Rationale:** This analysis directly addresses a reviewer concern
anticipated in the manuscript (Discussion, Confound 2: "epoch order
confounds"). It is already mentioned as Supplementary Figure S3 ("within-
dark-epoch dynamics") but has not been computed. This is essential for
establishing that the light/dark differences are not simply epoch-order
artefacts. If coverage drops equally in the 1st and 10th dark epoch, the
effect is robust. If it diminishes, there is adaptation.

**Effort:** Low-medium (half day). Epoch numbering is trivial from the
existing `detect_epochs()` output. Per-epoch metrics are already computed
for the coverage analysis.

---

### 4.2 Within-Dark-Epoch Dynamics (Early vs Late)

**What it tests:** Whether behaviour degrades within individual 1-minute
dark epochs, as expected if path integration drift causes progressive
spatial disorientation.

**Light/dark:** Inherently about within-dark-epoch dynamics.

**Method:**
a. Split each dark epoch into early (0-30s) and late (30-60s) halves.
b. Compute speed, transition rate (cell transitions per second),
   backtracking rate, and coverage within each half.
c. Paired Wilcoxon across sessions (early-half vs late-half of dark epochs).
d. Repeat for light epochs as a control (there should be no within-epoch
   degradation in light).

**Literature:**
- HD drift in darkness accumulates at ~0.1-1.0 deg/s (Stackman & Taube
  1997), meaning 1 minute of darkness produces ~6-60 degrees of drift.
  This is enough to impair landmark-based navigation but may not
  dramatically affect local wall-following.
- Muir et al. (2022, Nat Commun) report that ~40% of HD cells become
  unstable within minutes of darkness onset.

**Confounds:**
- 30 seconds at 9.6 Hz gives ~288 frames -- adequate for speed but
  marginal for transition counts (~15-50 cell transitions in 30s).
- Splitting light epochs provides the crucial control: if early-vs-late
  differences appear in light as well as dark, they reflect general
  within-epoch dynamics (fatigue, settling in) rather than darkness-
  specific degradation.

**Priority: High.**

**Rationale:** This is already planned as Supplementary Figure S3 in the
manuscript. It directly tests the functional relevance of the 1-minute
dark-epoch design: does 1 minute of darkness produce measurable behavioural
degradation? If yes, it strengthens the argument that the HD system is
actively used during navigation. If no, it suggests that the maze's
tactile/proprioceptive cues are sufficient for 1-minute navigation
independent of vision.

**Effort:** Low (3-4 hours). Simple splitting of existing epoch data.

---

### 4.3 Peri-Transition Speed Dynamics at Light Changes

**What it tests:** Whether mice show an immediate speed change at light-off
and light-on transitions, and how quickly the behavioural response to
darkness develops.

**Method:**
a. Identify all light-to-dark and dark-to-light transition frames.
b. Extract speed in a peri-event window (e.g., -10s to +10s around each
   transition).
c. Compute the average peri-transition speed profile (pooled across all
   transitions within a session, then across sessions).
d. Measure latency to speed change (time from light transition to 50% of
   the steady-state speed difference).

**Literature:**
- Instant (within-second) speed reduction at lights-off is common in open
  fields (Avni et al. 2006). The latency characterises how rapidly mice
  respond to darkness.
- In the HD literature, Zugaro et al. (2003) showed that HD cells
  re-anchor within a single head sweep (~300 ms) when lights return. The
  behavioural analogue (speed recovery at lights-on) has not been
  characterised in mazes.

**Confounds:**
- Light transitions happen at fixed 1-minute intervals, so the
  peri-transition window captures edge effects from the preceding epoch.
  The transition itself is sharp (DAQ-controlled).
- Individual transitions may not show a clean speed change if the mouse
  is already immobile or in the middle of a dead-end reversal.

**Priority: Medium.**

**Rationale:** This adds temporal resolution to the speed analysis. The
current manuscript reports average speed in light vs dark but does not show
the dynamics. A peri-event speed plot would be a visually compelling panel
showing the time course of the behavioural response to darkness.

**Effort:** Low (3-4 hours). Straightforward peri-event averaging.

---

### 4.4 Session-Level Temporal Trajectory of Exploration

**What it tests:** Whether exploration strategy evolves across the session
(learning the maze, fatigue, habituation) and whether light/dark
differences persist or change across the full session.

**Method:**
a. Compute per-epoch coverage, speed, transition entropy across all epochs
   in order.
b. Plot as time series with light/dark shading.
c. Fit a linear trend (Spearman correlation with epoch index) to each
   metric, separately for light and dark epochs.
d. Test whether the light-dark coverage gap changes across the session.

**Priority: Medium.**

**Rationale:** Distinguishes fatigue/habituation effects from light/dark
effects. If coverage drops across the session in both conditions, but the
light-dark gap remains constant, the gap is robust. If the gap narrows
late in the session (mice habituate to darkness), this limits the
generalisability of the finding to early-session data.

**Effort:** Low (3-4 hours).

---

## 5. Spatial Patterns

### 5.1 Per-Cell Occupancy Maps (Light vs Dark)

**What it tests:** Whether mice change their spatial distribution in the
maze during darkness (e.g., concentrating near the entry point, avoiding
dead ends, preferring corridors).

**Method:**
a. Compute occupancy fraction per cell in light vs dark epochs (existing
   `cell_occupancy()` function, applied to condition-specific subsets).
b. Visualise as paired heatmaps on the maze grid.
c. Compute the Jensen-Shannon divergence between light and dark occupancy
   distributions for each session. Test whether JS divergence is greater
   than a within-condition null (bootstrap: split light epochs randomly).

**Priority: Medium.**

**Rationale:** Currently, occupancy is reported only as whole-session
aggregate (Fig. 1C). Showing how occupancy shifts between light and dark
adds spatial specificity to the coverage finding: is the coverage drop
uniform across the maze, or concentrated in peripheral dead-end branches?
If mice in darkness avoid peripheral branches (which require longer
traversals and more directional changes), this supports a spatial memory
interpretation.

**Effort:** Low (3-4 hours). Existing code can be applied per condition.

---

### 5.2 Home Base Detection

**What it tests:** Whether mice establish a "home base" in the maze (a
cell they return to repeatedly, with excursion-return dynamics) and whether
the home base location differs between light and dark.

**Method:**
a. For each session, identify the cell with the highest number of visits
   (or highest occupancy), excluding junctions that are high by structural
   necessity.
b. Test whether mice return to the home base at a rate above chance
   (compare to random walk null model).
c. Test whether the home base changes between light and dark epochs.

**Literature:**
- Home base behaviour is well-documented in open-field exploration
  (Eilam & Golani 1989; Fonio et al. 2009). Rodents establish a home base
  near a salient landmark and make excursions of increasing length. In
  the q-rose maze, "home base" may map to a preferred junction or the
  entry corridor.
- In darkness, rodents may shift their home base or increase home-base
  dwell time as a cautious strategy.

**Confounds:**
- In a small maze, the "most visited cell" may simply be the most central
  junction (highest betweenness centrality), which is structural rather
  than behavioural. Control by comparing to the random walk stationary
  distribution.
- Mice are placed in the maze at a specific entry point. Early-session
  high occupancy near the entry is expected and does not reflect a
  behaviourally established home base.

**Priority: Low.**

**Rationale:** Home base behaviour is better studied in open fields where
there are no structural constraints on movement. In the q-rose maze, the
topology strongly constrains which cells are visited most, making it
difficult to distinguish genuine home base behaviour from graph-structural
effects. This analysis is included for completeness but is unlikely to
yield a publishable finding.

**Effort:** Low (3-4 hours).

---

### 5.3 Spatial Heatmaps of Speed (Light vs Dark)

**What it tests:** Whether speed reduction in darkness is spatially uniform
or concentrated at specific maze locations (e.g., junctions, dead ends).

**Method:**
a. Compute mean speed per cell in light and dark conditions.
b. Visualise as paired heatmaps.
c. Compute per-cell speed difference (dark - light) and display as a
   difference map.

**Priority: Medium.**

**Rationale:** The current analysis reports speed by node type (Figure 6 /
supplementary) but not spatially. If speed reduction in darkness is
concentrated at junctions (suggesting deliberation) rather than uniform,
this adds interpretive value.

**Effort:** Low (2-3 hours). Existing frame-level speed and position data.

---

## 6. Cross-Session Consistency and Individual Differences

### 6.1 Individual Animal Exploration Strategies

**What it tests:** Whether individual mice have consistent exploration
strategies across sessions (for animals with multiple sessions), and
whether there are stable inter-individual differences.

**Method:**
a. For the 4 animals with multiple sessions (2-3 sessions each), compute
   ICC (intraclass correlation) for each exploration metric (coverage rate,
   transition entropy, turn bias, speed, coverage drop in darkness).
b. For all 14 animals, compute per-animal means (averaging across sessions)
   and examine the distribution. Are there clearly "bold" vs "cautious"
   explorers?
c. Cluster animals by their exploration profile (hierarchical clustering
   on standardised metrics). Test whether clusters correlate with celltype
   group (Penk+ vs nonpenk) -- this is a genotype confound check.

**Confounds:**
- With only 4 multi-session animals and 2-3 sessions each, ICC estimates
  will have wide confidence intervals. This is exploratory.
- Session order confounds: later sessions may show habituation effects
  regardless of individual strategy.

**Priority: Low-medium.**

**Rationale:** Individual differences in exploration are a known source of
variance. Documenting them supports the claim that the light/dark effects
are consistent across individuals (which is already shown by the paired
Wilcoxon analysis). The ICC analysis would strengthen this by quantifying
within-animal stability. However, with only 4 multi-session animals, the
statistical power is very limited.

**Effort:** Low (half day).

---

### 6.2 Celltype Group Behaviour Comparison

**What it tests:** Whether Penk+ and nonpenk animals differ in baseline
exploration behaviour, independent of the light/dark manipulation.

**Method:** Already computed in the current analysis script (Supplementary
S2). Extend with additional metrics (HMM state occupancy, LZC, junction
dwell time) if those analyses are implemented.

**Status:** Already implemented. Report as negative finding if no
differences (strengthens the "same behaviour, different neural
representation" interpretation).

**Priority: Already done.** No additional effort needed.

---

## 7. Methodological Additions

### 7.1 Bootstrap Confidence Intervals for Key Metrics

**What it tests:** Provides precision estimates beyond p-values.

**Method:** For each key metric (coverage, speed, transition entropy in
light vs dark), compute 95% bootstrap CI on the paired difference
(dark - light). Use the hierarchical bootstrap (resample sessions within
animals, then animals) to account for pseudoreplication.

**Literature:** Saravanan et al. (2020), as used in Gobbo et al. (2026),
provides a concrete implementation reference.

**Priority: High.**

**Rationale:** Reviewers will want confidence intervals, not just p-values.
The hierarchical bootstrap also addresses the pseudoreplication concern
more rigorously than the primary-only robustness check.

**Effort:** Low-medium (half day). Standard bootstrap implementation.

---

### 7.2 Speed-Matched Coverage Comparison

**What it tests:** Whether the coverage drop in darkness persists after
explicitly matching speed distributions.

**Method:**
a. For each session, subselect dark-epoch frames to match the speed
   distribution of light-epoch frames (propensity score matching or
   simple quantile matching).
b. Recompute per-epoch coverage on the speed-matched subset.
c. Test light vs speed-matched-dark coverage (Wilcoxon).

**Priority: High.**

**Rationale:** The current "coverage per active minute" control is an
indirect normalisation. Direct speed matching is a stronger control that
eliminates the speed confound entirely. If coverage is still lower in
darkness after speed matching, the finding is bulletproof. This addresses
the most likely reviewer objection ("the coverage drop is just because mice
are slower").

**Effort:** Medium (half day). Requires frame-level speed and position data.

---

## Summary Priority Table

| # | Analysis | Priority | Effort | Adds to manuscript |
|---|----------|----------|--------|-------------------|
| 4.1 | Epoch-number effects | **High** | Low-Med | Essential control (S3) |
| 4.2 | Within-dark-epoch dynamics | **High** | Low | Already planned (S3) |
| 7.1 | Bootstrap CIs | **High** | Low-Med | Statistical rigour |
| 7.2 | Speed-matched coverage | **High** | Med | Bulletproof primary finding |
| 1.2 | HMM latent states | **Medium** | Med | New result if non-trivial |
| 1.3 | Transition matrix properties | **Medium** | Low-Med | Enriches Markov analysis |
| 2.3 | Path optimality (revisits) | **Medium** | Med | Supports coverage story |
| 2.4 | Junction dwell time | **Medium** | Low | New behavioural dimension |
| 3.1 | Lempel-Ziv complexity | **Medium** | Low | Non-Markov complexity |
| 4.3 | Peri-transition speed dynamics | **Medium** | Low | Temporal resolution |
| 4.4 | Session temporal trajectory | **Medium** | Low | Context for Fig S3 |
| 5.1 | Per-cell occupancy (L vs D) | **Medium** | Low | Spatial specificity |
| 5.3 | Speed heatmaps (L vs D) | **Medium** | Low | Spatial specificity |
| 6.1 | Individual animal strategies | **Low-Med** | Low | ICC/consistency |
| 1.1 | Higher-order Markov | **Low** | Low | Predictable null |
| 2.2 | Graph centrality | **Low** | Low | Structural description |
| 3.2 | Run-length distributions | **Low** | Low | Specialised metric |
| 3.3 | Mutual info (higher lags) | **Low** | Low | Extension of autocorr |
| 5.2 | Home base detection | **Low** | Low | Not suited to maze |

---

## Recommended Implementation Order

**Phase 1 (Essential controls, 2-3 days):**
1. Epoch-number effects (4.1)
2. Within-dark-epoch dynamics (4.2)
3. Bootstrap confidence intervals (7.1)
4. Speed-matched coverage (7.2)

These four analyses close the gaps identified in the manuscript Discussion
(Confounds 2, S3, Limitations 5) and make the primary finding (coverage
drop) robust against the most likely reviewer objections.

**Phase 2 (Enrichment, 2-3 days):**
5. Junction dwell time (2.4)
6. Peri-transition speed dynamics (4.3)
7. Transition matrix properties (1.3) -- stationary distribution, mixing
   time, Jensen-Shannon divergence
8. Per-cell occupancy maps, light vs dark (5.1)

These add depth to the existing story without introducing new modelling
frameworks. Each is a low-effort extension of existing code.

**Phase 3 (Advanced methods, 2-3 days):**
9. HMM latent states (1.2) -- the main "new method" addition
10. Lempel-Ziv complexity (3.1)
11. Path optimality for dead-end revisits (2.3)
12. Session-level temporal trajectory (4.4)

These require new code but offer the highest novelty. The HMM analysis is
the most promising: if it reveals a darkness-specific latent state beyond
"slow," it provides a mechanistic interpretation of the coverage drop that
the current analysis lacks.

**Phase 4 (If time permits):**
13-19. Remaining low-priority analyses.

---

## Key References (New)

Bhakti, Bhatt et al. 2024. "Stochastic characterization of navigation
strategies in an automated variant of the Barnes maze." eLife 13, e88648.
doi:10.7554/eLife.88648

Ashwood ZC et al. 2022. "Mice alternate between discrete strategies during
perceptual decision-making." Nat Neurosci 25, 201-212.
doi:10.1038/s41593-021-01007-z

Aoyama M et al. 2025. "Hidden Markov models reveal behavioral state
dynamics in depth-related locomotion in mice." PLOS ONE 20(8), e0329367.
doi:10.1371/journal.pone.0329367

Singer AC, Carr MF, Karlsson MP, Frank LM. 2013. "Hippocampal SWR
activity predicts correct decisions during the initial learning of an
alternation task." Neuron 77, 1163-1173. (Changepoint detection reference)

Saravanan V, Berman GJ, Sober SJ. 2020. "Application of the hierarchical
bootstrap to multi-level data in neuroscience." Neuron Behav Data Anal
Theory 3, 1-13.
