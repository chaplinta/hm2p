# Behavioural Manuscript — Analysis Plan

## Scope and Rationale

This document specifies the analysis plan for a short manuscript describing
mouse navigation behaviour in the Rosenberg maze (q-rose maze) under
alternating 1-minute light/dark epochs. The manuscript is purely behavioural
— no neural data. It serves two purposes:

1. **Stand-alone behavioural characterisation** of freely-moving mice
   navigating a structured maze with visual cue removal.
2. **Establish the behavioural foundation** for the primary neural manuscript
   (cell-type-specific HD tuning and visual anchoring in RSP).

The analysis addresses hypotheses H4.1–H4.4 from `docs/hypotheses.md` and
ideas from `docs/maze-exploration-ideas.md` (sections 3.1, 3.2, 3.3).

---

## Dataset

### Sessions

26 sessions from 16 animals. Each session lasts ~15–30 minutes, with
alternating 1-minute light-on / 1-minute dark epochs (~5–10 light epochs and
~5–10 dark epochs per session, depending on duration).

**Exclusions:** 5 sessions are flagged `exclude=1` in `experiments.csv`:

| exp_index | exp_id | Reason |
|-----------|--------|--------|
| 5 | 20211028_10_22_38_1114356 | Fluctuating traces |
| 13 | 20220531_11_06_13_1117217 | Camera sync problem |
| 14 | 20220601_13_53_18_1117217 | Camera sync problem |
| 19 | 20220804_13_52_02_1117646 | Bad 2P recording |
| 26 | 20221117_13_20_31_1118317 | Bad 2P, tethering issue |

Sessions 13 and 14 have camera sync failures and will lack reliable
behavioural data. Session 26 has severe tethering artefacts. These 5 are
excluded from all analyses unless stated otherwise, leaving **21 usable
sessions** from **16 animals**.

All sessions are processed through the pipeline (per CLAUDE.md) and sync.h5
files are generated for all 26. The exclusion applies only at analysis time.

### Animals

| Celltype | Animals | Sessions |
|----------|---------|----------|
| Penk+ | 12 (11 male, 1 female: 1118023) | 18 sessions |
| Penk⁻CamKII+ | 4 (all male) | 8 sessions |

The celltype label reflects the viral construct, not a behavioural
manipulation — the virus does not alter neural function, only the imaging
target. Behavioural differences between groups, if any, reflect genotype or
cohort effects.

Note: animal 1118023 is the only female. Report sex breakdown; check whether
excluding this animal changes any result.

### Animals with multiple sessions

| animal_id | n_sessions (usable) | Notes |
|-----------|---------------------|-------|
| 1114356 | 3 (exp 2,3,4; exp 5 excluded) | Penk+ |
| 1115465 | 3 (exp 6,7,8) | Penk+ |
| 1116663 | 2 (exp 11,12) | Penk+ |
| 1117217 | 1 usable (exp 15; exp 13,14 excluded) | Penk⁻CamKII+ |
| 1117646 | 2 (exp 17,18; exp 19 excluded) | Penk⁻CamKII+ |

For between-animal comparisons, use the `primary_exp=1` session where
available. For within-animal repeated-measures analyses (e.g., cross-session
exploration consistency), use all usable sessions.

### sync.h5 fields used

All behavioural data comes from `sync.h5`, which contains neural and
behavioural signals resampled to the imaging rate (~9.6 Hz). Key fields:

| Field | dtype | Description |
|-------|-------|-------------|
| `position_x` | float64 (N,) | X position in mm |
| `position_y` | float64 (N,) | Y position in mm |
| `hd_deg` | float64 (N,) | Head direction in degrees (unwrapped) |
| `speed` | float64 (N,) | Running speed in cm/s |
| `light_on` | bool (N,) | True when lights are on |
| `bad_behav` | bool (N,) | True during bad behaviour periods |
| `time` | float64 (N,) | Time in seconds |
| Attrs: `fps_imaging` | float | Imaging frame rate (~9.6 Hz) |

Derived fields to compute from these:

- Angular head velocity (AHV) from `hd_deg` via `analysis/ahv.py::compute_ahv()`
- Maze cell index from `position_x/y` via `maze/discretize.py::discretize_position()`
- Cell/node sequences via `maze/discretize.py::cell_sequence()`, `node_sequence()`
- Movement state: `speed >= 2.5` cm/s (standard threshold from `analysis-plan.md`)

### Exclusion masks

Every analysis must apply:
1. `~bad_behav` — exclude tethering artefact periods
2. `speed >= 2.5` — for movement-dependent metrics (not for immobility analysis)
3. Valid position — `np.isfinite(position_x) & np.isfinite(position_y)`

---

## Analysis 1: Summary Statistics per Session

### 1.1 Total distance travelled (m)

**Computation:** Cumulative Euclidean displacement between consecutive frames,
excluding `bad_behav` frames:
```
dx = diff(position_x)
dy = diff(position_y)
distance = sum(sqrt(dx^2 + dy^2)) / 1000  # mm -> m
```

**sync.h5 fields:** `position_x`, `position_y`, `bad_behav`

**Existing function:** None — new code needed. Simple cumulative sum.

**Expected values:** 30–150 m per session (20 min at ~5 cm/s average).

**Report:** Per-session values; grand mean +/- SD across sessions.

### 1.2 Mean speed (cm/s) — light vs dark

**Computation:** Mean of `speed` array for frames where `light_on=True` vs
`light_on=False`, excluding `bad_behav` and `speed < 0.5` cm/s (noise floor).
Report both with and without speed floor.

**sync.h5 fields:** `speed`, `light_on`, `bad_behav`

**Existing function:** Speed distributions are computed ad hoc; no dedicated
summary function. Use numpy directly.

**Expected values:** 3–8 cm/s in light, 2–6 cm/s in dark. Literature predicts
a speed decrease in darkness.

**Statistical test:** Wilcoxon signed-rank (paired within session, light vs
dark). N = 21 sessions. Report median difference and effect size
(rank-biserial correlation).

**Caveats:** Speed at ~9.6 Hz sampling is already smoothed. The kinematic
smoothing applied upstream (movement library Gaussian filter) further smooths
instantaneous speed. Report the smoothing parameters.

### 1.3 Fraction of time active vs inactive

**Computation:** Fraction of valid (non-`bad_behav`) frames where
`speed >= 2.5` cm/s. Report separately for light and dark.

**sync.h5 fields:** `speed`, `light_on`, `bad_behav`

**Existing function:** None dedicated. Simple masking.

**Expected values:** 40–80% active time. May decrease in darkness.

**Statistical test:** Wilcoxon signed-rank (paired, light vs dark). N = 21.

### 1.4 Session duration

**Computation:** `time[-1] - time[0]` from sync.h5. Also report usable
duration after excluding `bad_behav` frames.

**sync.h5 fields:** `time`, `bad_behav`

**Expected values:** 600–1800 s (10–30 min).

### 1.5 Number of maze cells visited

**Computation:** Discretize position to maze cell indices using
`maze/discretize.py::discretize_position()`, then count unique cells visited.
Total accessible cells = 23.

**sync.h5 fields:** `position_x`, `position_y`

**Existing function:** `maze/analysis.py::maze_exploration_summary()` returns
`unique_cells_visited` and `coverage_frac`.

**Expected values:** 20–23 cells (near-complete coverage for most sessions).

---

## Analysis 2: Exploration Analysis

### 2.1 Exploration efficiency by condition

**Computation:** Exploration efficiency (unique nodes per sliding window of
visits), computed separately for light and dark epochs.

1. Discretize position to cell indices
2. Extract cell sequence (consecutive duplicates removed)
3. Split by `light_on` condition
4. Compute `maze/analysis.py::exploration_efficiency()` for each condition

**sync.h5 fields:** `position_x`, `position_y`, `light_on`, `bad_behav`

**Existing function:** `maze/analysis.py::exploration_efficiency()`

**Statistical test:** Wilcoxon signed-rank comparing efficiency at matched
window sizes between light and dark. N = 21 sessions.

**Caveats:** The number of node transitions per 1-minute epoch is limited
(~20–60). Pool across same-condition epochs within a session to get adequate
samples. Window sizes must be adapted to epoch length.

### 2.2 Dead-end visit rate — light vs dark

**Computation:** Number of dead-end visits per minute in light vs dark epochs.
Dead-end visit = entry into a cell classified as `dead_end` in the maze
topology (6 dead ends in the q-rose maze).

**sync.h5 fields:** `position_x`, `position_y`, `light_on`, `bad_behav`, `time`

**Existing function:** `maze/analysis.py::dead_end_visits()` — returns
per-dead-end `{"visits": int, "total_frames": int, "mean_dwell": float}`.
Needs to be called separately for light and dark subsets.

**Maze topology:** 6 dead ends:
`(0, 0), (2, 0), (4, 0), (6, 0), (0, 4), (6, 4)` — confirmed from
`topology.py::classify_nodes()` where `len(neighbours) == 1`.

**Statistical test:** Wilcoxon signed-rank (paired, visits/min in light vs
dark). N = 21.

**Expected result:** Dead-end visit rate may increase in darkness if mice
explore less efficiently, or decrease if mice become more conservative
(wall-following in corridors).

### 2.3 Coverage — fraction of 23 cells visited per epoch

**Computation:** For each 1-minute epoch (light or dark), count unique cells
visited / 23.

**sync.h5 fields:** `position_x`, `position_y`, `light_on`, `bad_behav`, `time`

**Existing function:** Not directly. Use `discretize_position()` +
`np.unique()` within each epoch.

**Statistical test:** Wilcoxon signed-rank (paired, mean coverage per light
epoch vs mean coverage per dark epoch within each session). N = 21.

**Expected values:** 30–80% coverage per 1-minute epoch. Coverage should be
lower in dark if mice move slower and cover less ground.

**Caveat:** Coverage depends heavily on speed. Must report speed alongside
coverage to disentangle reduced exploration from reduced locomotion.

### 2.4 Time to first visit of each cell

**Computation:** For each of the 23 cells, record the first frame at which
the mouse enters that cell. Convert to time (seconds from session start).
Report as a cumulative coverage curve (number of unique cells vs time).

**sync.h5 fields:** `position_x`, `position_y`, `time`, `bad_behav`

**Existing function:** Partially in `maze_exploration_summary()` (coverage
array). Need to extract the exact first-visit time per cell.

**Statistical test:** Not directly testable between light/dark (this is a
cumulative session-level measure). Report descriptively. Compare between
genotype groups: Kruskal-Wallis or Mann-Whitney on median time-to-full-coverage.

---

## Analysis 3: Turn Bias at Junctions

The q-rose maze has 7 T-junctions (cells with 3 neighbours): `(1, 2)`,
`(1, 4)`, `(3, 2)`, `(5, 2)`, `(5, 4)`, `(2, 2)`, `(4, 2)`.

(Note: verify the exact junction set programmatically from
`maze.junctions` after `build_rose_maze()`.)

### 3.1 Per-junction left/right bias

**Computation:** For each junction, count the number of left, right, back,
and forward traversals using `maze/analysis.py::per_junction_turn_bias()`.
Report the left fraction: `left / (left + right)`. Values near 0.5 indicate
no bias; deviations indicate lateralised turning preference.

**sync.h5 fields:** `position_x`, `position_y`, `bad_behav`

**Existing function:** `maze/analysis.py::per_junction_turn_bias()`

**Statistical test:** For each junction, binomial test against 0.5 (or
Wilcoxon signed-rank across sessions for the same junction). Apply
Holm-Bonferroni correction across 7 junctions.

**Expected result:** Mice often show a lateralised turn preference (Rosenberg
et al. 2021 found left/right biases in their labyrinth). The Rosenberg maze
had 63 junctions; we have 7, so per-junction power is lower.

### 3.2 Turn bias — light vs dark

**Computation:** Compute global `left_frac` from
`maze/analysis.py::turn_bias()` separately for light and dark epochs within
each session.

**sync.h5 fields:** `position_x`, `position_y`, `light_on`, `bad_behav`

**Existing function:** `maze/analysis.py::turn_bias()` — called on light and
dark subsets.

**Statistical test:** Wilcoxon signed-rank (paired, left_frac_light vs
left_frac_dark). N = 21.

**Expected result:** Turn bias may change in darkness if visual landmarks
anchor directional preferences. Loss of visual cues might increase
stochasticity (left_frac closer to 0.5) or shift the bias (if path
integration favours a different strategy).

### 3.3 Sequential turn correlation

**Computation:** Does the previous turn predict the next? For consecutive
junction visits, classify each as left/right. Compute the autocorrelation of
the turn sequence at lag 1.

Specifically:
1. Extract the sequence of left/right turns at all junctions (dropping back
   and forward traversals)
2. Code left=0, right=1
3. Compute Spearman rank correlation at lag 1 (turn_i vs turn_{i+1})
4. Positive correlation = tendency to repeat; negative = tendency to alternate

**sync.h5 fields:** `position_x`, `position_y`, `bad_behav`

**Existing function:** `classify_turn()` and `per_junction_turn_bias()` exist.
Need new code to extract the sequential turn series and compute
autocorrelation.

**Statistical test:** One-sample Wilcoxon signed-rank testing whether the
per-session lag-1 correlation differs from 0. N = 21. Compare light vs dark
with paired Wilcoxon.

**Expected result:** Rosenberg et al. (2021) found that mice show turn
alternation in their labyrinth (negative autocorrelation). This is a
well-known rodent behaviour ("spontaneous alternation"). May weaken in
darkness if path integration cannot support alternation.

### 3.4 Turn bias by genotype (Penk+ vs Penk⁻CamKII+ animals)

**Computation:** Compare the global `left_frac` and the sequential
turn-correlation between genotype groups.

**Statistical test:** Mann-Whitney U on animal-level means (N = 12 Penk+ vs
N = 4 Penk⁻CamKII+). Very low power — report as exploratory.

**Caveat:** Underpowered. Only interpretable if the effect size is large.
The celltype label does not predict behaviour a priori (the virus does not
alter function). Any difference is a genotype/cohort confound.

---

## Analysis 4: Speed and Movement

### 4.1 Speed distributions — light vs dark

**Computation:** For each session, compute the speed distribution (histogram
or kernel density estimate) separately for light and dark epochs. Summarise
with median speed, 25th/75th percentiles, and 95th percentile.

**sync.h5 fields:** `speed`, `light_on`, `bad_behav`

**Existing function:** `analysis/speed.py::speed_tuning_curve()` computes
mean signal vs speed, not speed distributions per se. Use numpy histogram.

**Statistical test:** Wilcoxon signed-rank on per-session median speed
(light vs dark). N = 21. Also compare the fraction of time at high speed
(>10 cm/s) between conditions.

**Expected values:** Median speed ~4–6 cm/s in light, ~3–5 cm/s in dark.
Literature consistently shows reduced locomotion speed in darkness.

### 4.2 Speed at junctions vs corridors vs dead ends

**Computation:** Classify each frame by maze-node type using
`maze/neural.py::classify_frames_by_node_type()`, then compute mean speed
within each node type.

**sync.h5 fields:** `position_x`, `position_y`, `speed`, `bad_behav`

**Existing function:**
`maze/neural.py::classify_frames_by_node_type()` — returns bool masks for
junction, corridor, dead_end, invalid.

**Statistical test:** Friedman test (3 repeated measures: junction, corridor,
dead_end) across sessions. Post-hoc: Wilcoxon signed-rank with
Holm-Bonferroni. N = 21 sessions.

**Expected result:** Speed should be lowest at dead ends (reversals), highest
in corridors (straight runs), and intermediate at junctions (decision
points). This is a basic sanity check.

### 4.3 Acceleration/deceleration at junction approaches

**Computation:** For each junction approach event (extracted via
`maze/neural.py::extract_junction_events()`), compute speed in a window
before and after junction entry:
- Pre-junction speed: mean speed in 0.5–1.0 s before junction entry
- At-junction speed: mean speed while at the junction
- Post-junction speed: mean speed in 0.5–1.0 s after junction exit

**sync.h5 fields:** `position_x`, `position_y`, `speed`, `bad_behav`

**Existing function:** `maze/neural.py::extract_junction_events()` returns
junction approach events with frame indices. Speed profiling around these
events needs new code.

**Statistical test:** Wilcoxon signed-rank comparing pre-junction vs
at-junction speed (paired within event). N = total junction events across
sessions. Apply Holm-Bonferroni for 3 pairwise comparisons.

**Expected result:** Mice should decelerate before junctions (decision
points) and re-accelerate after choosing a direction. This is a locomotor
correlate of decision-making.

### 4.4 Immobility bout duration by condition

**Computation:** Identify contiguous bouts of immobility (speed < 2.5 cm/s
for >= 0.5 s), excluding `bad_behav` frames. Compute bout duration
distribution in light vs dark.

**sync.h5 fields:** `speed`, `light_on`, `bad_behav`, `time`

**Existing function:** None. New code needed — detect threshold crossings
and measure bout lengths.

**Statistical test:** Wilcoxon signed-rank on per-session median immobility
bout duration (light vs dark). N = 21.

**Expected result:** Immobility bouts may be longer in darkness (mice freeze
or rest more when unable to see) or shorter (mice make shorter pauses before
moving again in an uncertain environment). Direction of effect is empirical.

---

## Analysis 5: Head Direction

### 5.1 HD distribution — light vs dark

**Computation:** For each session, compute the circular histogram of
`hd_deg % 360` in 36 bins (10-degree bins), separately for light and dark
epochs (excluding `bad_behav`, speed > 2.5 cm/s). Assess uniformity with the
Rayleigh test (from `astropy.stats.rayleightest` or `pycircstat`).

**sync.h5 fields:** `hd_deg`, `speed`, `light_on`, `bad_behav`

**Existing function:** None for behavioural HD distribution (as opposed to
neural tuning). Straightforward circular histogram.

**Statistical test:**
- Rayleigh test for each session in each condition (test uniformity)
- Wilcoxon signed-rank on per-session mean resultant length of the HD
  distribution (light vs dark). N = 21.

**Expected result:** HD distribution in a maze is likely non-uniform because
corridor geometry constrains body orientation. Light vs dark should show
similar distributions if the mouse traverses the same corridors, but dark
distributions may be broader (less constrained heading if the mouse pauses
or turns more). This is the behavioural analogue of the corridor-HD
confound identified in `docs/maze-exploration-ideas.md` (section 3.11).

**Caveat:** This is a behavioural HD distribution (body heading angle), not a
neural tuning analysis. Non-uniformity here reflects maze geometry, not
neural selectivity.

### 5.2 HD stability across light epochs

**Computation:** For each pair of consecutive light epochs within a session,
compute the circular mean heading in each epoch, then the angular difference.
Assess whether preferred corridor direction is stable across light epochs.

**sync.h5 fields:** `hd_deg`, `light_on`, `bad_behav`, `speed`, `time`

**Existing function:** `analysis/stability.py::drift_per_epoch()` does
something analogous for neural HD tuning. For behavioural HD, need to compute
the circular mean of the raw HD distribution per epoch.

**Statistical test:** Rayleigh test on the distribution of inter-epoch PD
shifts (testing whether shifts are non-uniformly distributed — i.e., do
epochs have consistent directional bias?).

**Expected result:** Light epochs should show stable mean heading (reflecting
consistent maze traversal patterns). Dark epochs may show higher
inter-epoch variability.

### 5.3 Angular head velocity distribution by condition

**Computation:** Compute AHV using `analysis/ahv.py::compute_ahv()`. Plot
the distribution of |AHV| (unsigned angular head velocity) for light vs dark.
Summarise with median |AHV| and 95th percentile.

**sync.h5 fields:** `hd_deg`, `light_on`, `bad_behav`, `speed`

**Existing function:** `analysis/ahv.py::compute_ahv()` — takes `hd_deg` and
`fps`, returns AHV in deg/s.

**Statistical test:** Wilcoxon signed-rank on per-session median |AHV|
(light vs dark). N = 21.

**Expected result:** |AHV| may decrease in darkness along with reduced
locomotion speed, or increase if mice scan their heads more in an uncertain
environment (head scanning is a well-documented behaviour in rodents during
spatial uncertainty).

---

## Analysis 6: Supplementary Maze Analyses

### 6.1 Transition entropy — light vs dark

**Computation:** Fit a first-order Markov transition matrix to the cell
sequence in light and dark epochs separately (pooling same-condition epochs
within each session). Compute transition entropy.

**sync.h5 fields:** `position_x`, `position_y`, `light_on`, `bad_behav`

**Existing function:** `maze/analysis.py::transition_matrix()`,
`transition_entropy()`

**Statistical test:** Wilcoxon signed-rank (entropy_light vs entropy_dark).
N = 21.

**Expected result:** Transition entropy should change in darkness — either
decrease (more stereotyped, wall-following navigation) or increase (more
random). Rosenberg et al. (2021) found that exploration entropy decreased
with learning; here we ask whether it changes with visual cue availability.

### 6.2 Markov model order selection

**Computation:** For each session, compare first-order and second-order
Markov models using BIC (already implemented in
`maze/analysis.py::markov_order_comparison()`).

**Expected result:** A second-order model (history-dependent navigation)
should fit better than first-order, indicating that the previous cell
predicts the next turn beyond what the current cell alone predicts
(Rosenberg et al. 2021). This tests whether mice show momentum/alternation
in their navigation.

### 6.3 Path efficiency over session

**Computation:** `maze/analysis.py::path_efficiency_over_time()` — windowed
path efficiency (actual steps / optimal steps between trajectory endpoints).

**Expected result:** Path efficiency may increase over the session as the
mouse learns the maze layout (if the session is the first exposure) or
remain stable (if habituated).

---

## Figure List

### Figure 1: Maze schematic and session summary

**Panels:**

- **A**: Q-rose maze topology diagram (7x5 grid, accessible cells shaded,
  junctions and dead ends marked). Schematic of the light/dark alternation
  protocol (1 min on / 1 min off).
- **B**: Example trajectory from one session overlaid on maze, colour-coded
  by light (yellow) vs dark (grey). Position in maze coordinates.
- **C**: Summary table or bar plot of per-session statistics: duration,
  distance, mean speed, fraction active, cells visited. N = 21 sessions.

**Existing functions:** Maze topology from `topology.py`. Trajectory
visualisation needs new plotting code.

**Statistical tests:** Descriptive only.

### Figure 2: Speed and locomotion in light vs dark

**Panels:**

- **A**: Speed distributions (overlaid histograms or violin plots) for light
  (yellow fill) vs dark (grey fill). Pool across sessions or show per-session
  paired lines.
- **B**: Paired-dot plot of median speed per session: light vs dark. Connected
  lines between paired conditions. Wilcoxon signed-rank test statistic and
  p-value annotated. Effect size (rank-biserial r).
- **C**: Fraction of time active (speed >= 2.5 cm/s) in light vs dark.
  Same paired-dot format.
- **D**: Immobility bout duration distributions, light vs dark.

**Existing functions:** Speed from sync.h5; bout detection needs new code.

**Statistical tests:**
- Panel B: Wilcoxon signed-rank, N = 21, report W, p, rank-biserial r.
- Panel C: Wilcoxon signed-rank, N = 21.
- Panel D: Wilcoxon signed-rank on per-session median bout duration, N = 21.

### Figure 3: Maze exploration and coverage

**Panels:**

- **A**: Cumulative unique cells visited vs time (individual session traces,
  mean overlaid). Colour: light epochs yellow background, dark epochs grey
  background.
- **B**: Per-epoch coverage (fraction of 23 cells visited in each 1-minute
  epoch). Paired box plots: light vs dark. Wilcoxon signed-rank.
- **C**: Dead-end visit rate (visits/min) in light vs dark. Paired-dot plot.
- **D**: Exploration efficiency curves (unique nodes per window) — light vs
  dark, mean +/- SEM across sessions.

**Existing functions:**
- Coverage: `maze_exploration_summary()` for session-level; per-epoch needs
  new epoch-splitting code.
- Dead-end visits: `dead_end_visits()` — split by condition.
- Exploration efficiency: `exploration_efficiency()` — split by condition.

**Statistical tests:**
- Panel B: Wilcoxon signed-rank (per-session mean light coverage vs dark
  coverage), N = 21.
- Panel C: Wilcoxon signed-rank, N = 21.
- Panel D: Wilcoxon signed-rank at each window size, Holm-Bonferroni across
  window sizes.

### Figure 4: Turn behaviour at junctions

**Panels:**

- **A**: Per-junction turn bias heatmap. 7 junctions x 4 turn types
  (left, right, back, forward). Colour intensity = fraction of total
  traversals. One panel for light, one for dark.
- **B**: Global left_frac (left / (left+right)) per session: light vs dark.
  Paired-dot plot with Wilcoxon signed-rank.
- **C**: Sequential turn autocorrelation at lag 1 per session: light vs
  dark. Paired-dot plot. One-sample Wilcoxon testing whether the lag-1
  correlation differs from 0.
- **D**: Back-tracking rate (fraction of junction visits classified as
  "back") in light vs dark. Paired-dot plot.

**Existing functions:**
- `turn_bias()`, `per_junction_turn_bias()`, `classify_turn()` — all exist.
- Sequential autocorrelation and back-tracking rate need new code.

**Statistical tests:**
- Panel A: Per-junction binomial test against 0.5 for left vs right.
  Holm-Bonferroni across 7 junctions. N = pooled turns per junction across
  sessions.
- Panel B: Wilcoxon signed-rank, N = 21.
- Panel C: One-sample Wilcoxon on lag-1 correlation (vs 0), N = 21. Paired
  Wilcoxon for light vs dark.
- Panel D: Wilcoxon signed-rank, N = 21.

### Figure 5: Head direction and angular head velocity

**Panels:**

- **A**: Circular histograms of HD distribution for an example session.
  Left: light epochs. Right: dark epochs. Overlaid Rayleigh vector.
- **B**: Mean resultant length of behavioural HD distribution per session:
  light vs dark. Paired-dot plot.
- **C**: |AHV| distributions, light vs dark (violin or histogram).
- **D**: Median |AHV| per session: light vs dark. Paired-dot plot.

**Existing functions:**
- HD distribution: numpy circular histogram on `hd_deg`.
- AHV: `ahv.py::compute_ahv()`.

**Statistical tests:**
- Panel B: Wilcoxon signed-rank, N = 21.
- Panel D: Wilcoxon signed-rank, N = 21.

### Figure 6: Speed at maze locations

**Panels:**

- **A**: Mean speed per maze cell, displayed on the maze topology. Colour
  map = speed. One panel light, one panel dark.
- **B**: Speed by node type (junction, corridor, dead end). Box/strip plots
  with individual session dots. Friedman test + post-hoc Wilcoxon.
- **C**: Speed profile around junction approaches: mean speed aligned to
  junction entry (time 0), -1 s to +1 s. Light (yellow line) vs dark
  (grey line).

**Existing functions:**
- Node type classification: `classify_frames_by_node_type()`.
- Junction events: `extract_junction_events()`.
- Speed mapping: new code needed.

**Statistical tests:**
- Panel B: Friedman test (3 node types), N = 21 sessions. Post-hoc Wilcoxon
  with Holm-Bonferroni (3 pairs).
- Panel C: Wilcoxon signed-rank on per-session pre-junction vs at-junction
  speed difference, N = 21.

### Supplementary Figure S1: Markov models and entropy

**Panels:**

- **A**: Transition matrix (first-order) for an example session, light vs dark.
  Heatmap on maze graph.
- **B**: Transition entropy per session: light vs dark. Paired-dot plot.
- **C**: First-order vs second-order model comparison (delta BIC) per session.
  Bar chart or dot plot.
- **D**: Sequence entropy (conditional entropy by context length). Mean +/-
  SEM across sessions.

**Existing functions:** All implemented in `maze/analysis.py`.

**Statistical tests:**
- Panel B: Wilcoxon signed-rank, N = 21.
- Panel C: One-sample Wilcoxon (testing whether delta_BIC > 0, favouring
  second-order). N = 21.

### Supplementary Figure S2: Genotype comparison (exploratory)

**Panels:**

- **A**: Median speed by genotype. Strip/bee-swarm plot with animal means.
  Penk+ (N = 12 animals) vs Penk⁻CamKII+ (N = 4 animals).
- **B**: Fraction active by genotype.
- **C**: Coverage by genotype.
- **D**: Left_frac by genotype.
- **E**: Light-dark speed difference by genotype.

**Statistical tests:** Mann-Whitney U at the animal level for all panels.
N = 12 vs 4. Report as exploratory — severely underpowered. Always report
effect size (rank-biserial r or Cliff's delta) alongside p-values.

---

## Statistical Approach

### General principles

All tests are non-parametric per `CLAUDE.md` and `docs/stats-strategy.md`.

| Comparison type | Test | Conditions |
|----------------|------|------------|
| Paired within-session (light vs dark) | Wilcoxon signed-rank | N = 21 sessions |
| Unpaired between genotypes | Mann-Whitney U | N = 12 vs 4 animals |
| Multiple related measures | Friedman test | N = 21 sessions |
| Post-hoc pairwise | Wilcoxon signed-rank | Holm-Bonferroni correction |
| Association between continuous variables | Spearman rank correlation | |
| Proportions (e.g., turn bias) | Binomial test (exact) | |
| Circular uniformity | Rayleigh test | |

### Multiple comparisons

Within each figure, apply Holm-Bonferroni correction across the number of
independent tests in that figure.

- Figure 2: 3 tests (speed, fraction active, bout duration) — correct across 3.
- Figure 3: 3 tests (coverage, dead-end rate, efficiency) — correct across 3.
- Figure 4: 4 tests (left_frac, autocorrelation, autocorrelation vs 0,
  back-tracking) — correct across 4. Per-junction tests in panel A corrected
  separately (7 junctions).
- Figure 5: 2 tests (HD resultant length, AHV) — correct across 2.
- Figure 6: 3 tests (node types) — correct across 3.
- Supplementary S2: 5 tests (genotype comparisons) — correct across 5.

### Effect sizes

Report for every test:
- **Rank-biserial correlation (r)** for Wilcoxon signed-rank tests:
  `r = 1 - (2W / (n(n+1)/2))` where W is the test statistic.
- **Cliff's delta** for Mann-Whitney U tests.
- **Sample size** (N sessions, N animals, N junction events).

### Sample size and power

- Within-session (light vs dark): 21 sessions. Adequate for detecting
  medium-to-large effects (d > 0.5) with 80% power.
- Between genotypes: 12 vs 4 animals. Only large effects (d > 1.5)
  detectable. Report all genotype comparisons as exploratory.

### Handling of animals with multiple sessions

For paired within-session analyses (light vs dark), each session is one
observation. This may inflate N for animals with multiple sessions
(3 animals contribute 2–3 sessions each). As a robustness check:

1. Repeat all paired analyses using only `primary_exp=1` sessions
   (13 sessions, 13 animals — fully independent).
2. Report both results. If conclusions differ, the multi-session analysis
   is flagged as potentially inflated.

For between-genotype analyses, always collapse to one value per animal
(mean across sessions for animals with multiple sessions).

---

## Practical Notes

### sync.h5 position coordinates

The position coordinates in sync.h5 are in mm. The maze discretization
functions in `maze/discretize.py` expect maze coordinates (0–7 x 0–5 in
maze cell units). The conversion from mm to maze coordinates requires the
maze calibration (scale and offset). This mapping should be stored in
`kinematics.h5` or `sync.h5` attributes. Verify this before running
maze-discretization analyses.

### Epoch boundary detection

To identify individual light/dark epochs:
```python
changes = np.where(np.diff(light_on.astype(int)) != 0)[0] + 1
boundaries = np.concatenate([[0], changes, [len(light_on)]])
```
This gives epoch start/end frame indices. Each epoch is approximately 1
minute (~576 frames at 9.6 Hz). Filter out partial epochs at session start
or end (< 30 s).

### Speed floor

Many analyses require excluding immobile periods (speed < 2.5 cm/s). Some
analyses (immobility bout analysis) specifically examine the immobile
periods. Always state which speed threshold is applied.

### Circular statistics

Head direction is a circular variable. Use circular mean
(`np.arctan2(np.sum(sin), np.sum(cos))`), circular variance, and the
Rayleigh test for uniformity. Do not compute arithmetic mean of angles.

### Code organisation

New functions needed for this manuscript:

| Function | Module | Description |
|----------|--------|-------------|
| `total_distance()` | `maze/analysis.py` | Cumulative Euclidean distance |
| `detect_immobility_bouts()` | `analysis/speed.py` | Find contiguous bouts below speed threshold |
| `per_epoch_coverage()` | `maze/analysis.py` | Unique cells per light/dark epoch |
| `sequential_turn_autocorrelation()` | `maze/analysis.py` | Lag-1 autocorrelation of left/right turns |
| `speed_by_node_type()` | `maze/analysis.py` | Mean speed per node type |
| `junction_approach_speed_profile()` | `maze/analysis.py` | Speed aligned to junction entry events |
| `behavioural_hd_distribution()` | `analysis/ahv.py` | Circular histogram + Rayleigh test on raw HD |

All functions must be pure numpy (no I/O), unit-tested with synthetic data
(hypothesis for numerical functions), and documented with citations where
methods derive from published work.

---

## References

- Rosenberg, Zhang, Perona & Meister (2021). "Mice in a labyrinth show
  rapid learning, sudden insight, and efficient exploration." eLife 10,
  e66175. doi:10.7554/eLife.66175

- Taube, Muller & Ranck (1990). "Head-direction cells recorded from the
  postsubiculum in freely moving rats. I. Description and quantitative
  analysis." J Neurosci 10(2):420-435. doi:10.1523/JNEUROSCI.10-02-00420.1990

- Skaggs, McNaughton, Wilson & Barnes (1996). "Theta phase precession in
  hippocampal neuronal populations and the compression of temporal
  sequences." Hippocampus 6(2):149-172.
  doi:10.1002/(SICI)1098-1063(1996)6:2<149::AID-HIPO6>3.0.CO;2-K

- Mardia & Jupp (2000). "Directional Statistics." Wiley.
  (For Rayleigh test, circular statistics methodology.)
