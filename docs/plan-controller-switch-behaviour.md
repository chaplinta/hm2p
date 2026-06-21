# Plan: does the navigation controller switch in the dark? (behaviour only)

Status: **plan only — nothing implemented.** Behaviour-only analysis. No neural
data, and deliberately no assumption about what RSP does. Tests whether the
*decision rule* the mouse uses to choose its path changes when the lights go off.

## Question

At each junction the mouse makes a choice (which arm to take). What determines
that choice, and is the determinant the same in light and dark?

Two families of decision variable:

- **Allocentric / world-based:** the choice depends on position in the maze and
  memory of where it has been — bias toward arms leading to unexplored or
  less-recently-visited cells. Needs location + history; vision helps anchor it.
- **Egocentric / body-based:** the choice depends only on body-relative state —
  the last turn, an alternation tendency, keep-going / momentum, current heading
  relative to the available arms. No map, no vision needed.

The behaviour summary motivates this: in light, coverage beats a random walk
(directed); in dark it falls to near-random, yet turn balance and the
left/right **alternation** habit (lag-1 turn autocorrelation ≈ -0.17) are
**preserved in both conditions**. So a body-based default appears always on; what
may change is whether a world-based process is layered on top.

### Three outcomes the analysis must distinguish

1. **Switch** — world-based variables predict choices in light; body-based
   variables predict them in dark (a crossover).
2. **Degrade** — world-based variables predict in both, but more weakly in dark
   (same rule, noisier).
3. **No change of rule** — world-based variables predict equally in both (vision
   loss changes the inputs, not the decision rule).

Refined model: dark choices ≈ the body-based default alone; light choices = the
same default **plus** a vision-gated world-based correction. Test = does adding
world-based features improve choice prediction in light but not dark?

## Data and existing infrastructure (no new upstream computation)

From `sync.h5`: `x_maze`, `y_maze`, `hd_deg`, `light_on`, `bad_behav`,
`speed_cm_s`, `frame_times`. Reuse:

- `hm2p.maze.topology.build_rose_maze` — graph, junctions, dead-ends, adjacency,
  shortest-path distances.
- `hm2p.maze.discretize.discretize_position_fast`, `cell_sequence`.
- `hm2p.maze.neural.extract_junction_events` — per-junction events with
  `prev_cell`, `junction`, `next_cell`, `turn` (egocentric left/right/back via
  `hm2p.maze.analysis.classify_turn`).
- `hm2p.maze.analysis.turn_bias`, `per_junction_turn_bias`.

## Unit of analysis

One **junction-choice event** = arrival at a junction with a defined approach
arm, and the chosen departure arm. Pool events within a session; condition =
light/dark of the junction frame. Session is the statistical unit for paired
light/dark tests (per-animal sensitivity reported). Choices are categorical over
the available arms (exclude the arm just came from for the "explore" framing, but
also model `back` explicitly as an option, since backtracking is the key signature).

## Choice-predictor features (per event)

Computed at the moment of arrival, before the departure is known.

**Egocentric / body-based**
- last turn direction (left/right/forward/back at the previous junction)
- alternation predictor: turn that would continue the L/R alternation pattern
- momentum: arm most aligned with current heading / continuing straight
- egocentric arm angles relative to approach heading

**Allocentric / world-based**
- per available arm: visited vs not (this epoch), and time/steps since last visit
  to the cell that arm leads to (recency)
- per arm: graph distance to the nearest unvisited cell (does this arm head toward
  unexplored maze?)
- per arm: novelty = whether it leads to a less-recently-visited region

## Models and comparison

Predict the chosen arm from features, separately per condition. Two nested
comparisons:

1. **Predictive-accuracy comparison.** Fit an egocentric-only model and a
   full (egocentric + allocentric) model; compare cross-validated choice
   prediction (held-out log-likelihood / accuracy above the per-junction base
   rate). The world-based contribution = full minus egocentric. Test whether that
   contribution is positive in light and ≈ 0 in dark (the gating prediction).
   - Keep the model interpretable and non-parametric in the testing layer: fit
     per session, summarise to one number per session (e.g. ΔCV-loglik from adding
     allocentric features), then paired Wilcoxon light vs dark across sessions.
   - A multinomial logistic / simple decision-tree is fine as the per-session
     fitter; the *inference* is the non-parametric paired test across sessions,
     not the model's own p-values.

2. **Conflict-trial analysis (the cleaner, assumption-light test).** Restrict to
   junctions where the egocentric default and the world-based rule **disagree**
   (e.g. alternation says "right" but the only unvisited arm is "left"). On these
   conflict trials, which rule does the animal follow, and does the fraction
   following the world-based rule drop from light to dark? This needs no model
   fitting — it is a direct paired proportion (McNemar / paired Wilcoxon on
   per-session follow-rates) and is the headline test.

## Confounds (ranked)

1. **Feature collinearity.** Heading, position, and arm geometry are correlated
   through the maze structure, so the two feature families share variance. The
   conflict-trial test sidesteps this by construction (only disagreement cases).
   For the model comparison, report unique variance (nested ΔCV), not raw fits.
2. **Unequal junction sampling light vs dark.** Dark has fewer, more repetitive
   junction visits; match the number of events and the junction identity
   distribution across conditions before comparing (subsample / stratify by
   junction).
3. **Recency distribution differs by condition** (they revisit more in dark) — the
   same matching trap seen elsewhere. Stratify recency when it enters a feature.
4. **Speed / pausing** at junctions may differ; include or match, since it can
   correlate with deliberation and with which arm is chosen.
5. **Base-rate / geometry per junction.** Some junctions have intrinsic biases
   (`per_junction_turn_bias`); evaluate prediction *above each junction's own base
   rate*, and treat junction as a grouping factor.
6. **First dark epoch is near-normal** (the switch is learned over ~1-2 min).
   Run with and without the first light+dark epoch; optionally test the rule
   crossover *within* the dark epochs as adaptation proceeds.

## What each result would mean

- **Crossover (switch):** the animal runs a genuinely different decision rule
  without vision — a behavioural controller switch. Strongest result.
- **Gating (allocentric term present in light only):** a vision-dependent
  world-based module on top of an always-on body-based default. Also strong,
  and the most specific to the data.
- **Degrade only:** same rule throughout, weaker in dark — less interesting but
  honest.
- **No allocentric contribution in either condition:** directed coverage is
  driven by something the chosen features don't capture (e.g. boundary/thigmotaxis
  heuristics) — would send us to redefine the world-based features, not abandon
  the question.

## Constraints

Non-parametric inference (paired Wilcoxon / McNemar across sessions; per-animal
sensitivity). Body-centroid position only. Real data only; synthetic only in unit
tests. New computation lives in a tested module (e.g.
`src/hm2p/maze/choice_models.py`) with a runner; outputs to
`results/controller_switch/`. Any decision-model method taken from a paper cited
per the citation policy. **No assumption about RSP function enters this analysis** —
it is behaviour only; the neural follow-up (what, if anything, these cells covary
with under each policy) is a separate, deliberately open question.
