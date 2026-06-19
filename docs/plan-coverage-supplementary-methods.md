# Supplementary methods for maze coverage / exploration

Status: **plan only — nothing implemented.** Scopes more sophisticated measures to
supplement the primary "per-epoch coverage" finding (unique maze cells visited per
1-min epoch / 23, light vs dark; W p=0.0003, r=0.857).

## Why supplement

The primary measure is deliberately simple and that is fine as the headline. Its
limits, and what a supplement should add:

- **Presence/absence only.** A cell entered for one frame counts the same as a cell
  occupied for 30 s — it ignores the *time distribution* over cells.
- **Endpoint only.** It is the count at end of epoch; it ignores the *rate* at which
  new cells are discovered.
- **Order-blind.** It ignores the *sequence* / route structure — stereotyped routes
  vs varied routes can give the same unique-cell count.
- **Coarse.** 23 discrete cells; sub-cell structure of the trajectory is discarded.
- **Confounded with locomotion.** Fewer cells in dark could be "explores less" or
  simply "moves less". The primary measure cannot separate these — and this is the
  same confound that undermined the neural HD result, so it matters most.

## Proposed supplements (priority order)

### Tier 1 — separate exploration from locomotion (the key confound)

1. **New cells per unit distance travelled.** First-visit rate per metre of body-
   centroid path (not per unit time). Directly asks: holding distance fixed, do they
   discover new ground less in dark? Method: cumulative unique cells vs cumulative
   distance; slope over the epoch, or unique cells at a fixed distance budget.
   Stat: per-session paired Wilcoxon light vs dark. New computation; cheap (reuses
   the per-epoch distance already added).

2. **Null-model benchmarking.** Compare observed coverage (and the entropy measures
   below) to a null of random walks on the maze graph (`hm2p.maze.topology`),
   matched to the observed number of steps or distance, under either uniform or the
   empirical transition probabilities. Yields a coverage *z-score / percentile*
   against "what you'd get from just moving this much on this maze". Answers whether
   reduced dark coverage exceeds the locomotion + topology expectation. New
   computation; moderate (needs a graph random-walk simulator + per-session matching).

### Tier 2 — time-weighted occupancy (beyond presence/absence)

3. **Spatial occupancy entropy.** Shannon entropy of the time-per-cell distribution
   (uniform exploration → high; concentrated → low). More sensitive than a unique
   count because it weights by dwell time. Stat: paired light vs dark. New; trivial
   (histogram + entropy on `x_maze/y_maze`).

4. **Continuous (non-discretised) coverage.** Remove the 23-cell coarseness:
   occupancy-map entropy on a fine grid (or 2-D KDE), explored-area / convex-hull
   area, or fraction of a fine grid visited. Captures sub-cell structure. New; cheap.

### Tier 3 — route structure / stereotypy (beyond coverage)

5. **Sequence compressibility (Lempel–Ziv).** LZ complexity of the discrete cell-visit
   sequence: stereotyped, repeated routes compress more → lower LZ. Predicts lower LZ
   in dark (route consolidation) even when coverage is matched. Complements coverage
   with an order-sensitive measure. New; cheap (LZ76 on the cell sequence).

6. **Graph-edge coverage.** Fraction of maze *corridors/edges* traversed rather than
   *nodes* visited, and junction- vs corridor- vs dead-end-resolved coverage. Edge
   coverage may be more sensitive to route narrowing. Partially reuses the existing
   topology + route-stereotypy machinery.

### Tier 4 — exploration dynamics / recurrence

7. **Discovery-rate curve.** Fit cumulative-unique-cells vs *time* to a saturating
   curve; report the rate constant τ and asymptote. Light vs dark τ. New; cheap.

8. **Revisitation / return time / MSD.** Mean first-return time to a cell, recurrence
   rate, and the mean-squared-displacement diffusion exponent (sub-diffusive = more
   confined in dark). Revisitation index already exists
   (`behaviour-first-session-results.json`); the rest is new.

## Already computed (reuse, do not rebuild)

- Transition entropy of the cell Markov chain (light 1.221 vs dark 1.186; S1).
- Markov order (1st-order preferred, 0/20 sessions 2nd-order).
- Transition-matrix JSD vs a stable-behaviour null.
- Revisitation index; HMM/graph state occupancy.

## Recommended minimum supplement

Tiers 1 + 2 (new-cells-per-distance, null-model, occupancy entropy, continuous
coverage). They are the ones that (a) separate exploration from locomotion — the
confound that matters here and for the neural story — and (b) are cheap and reuse
existing position data. LZ compressibility (Tier 3.5) is the best single addition for
route stereotypy if one more is wanted.

## Constraints

Non-parametric throughout; per-session unit, paired light vs dark; primary-only
(one-session-per-animal) sensitivity reported alongside the full set, as in the main
behaviour analysis. Body-centroid position only (not head). Any paper-derived method
(LZ76, recurrence quantification) cited per the citation policy.
