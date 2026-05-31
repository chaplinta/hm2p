# Tier-2 Behaviour Analysis — Implementation Plan

Concrete specifications for adding Directions 1, 2, 4, and 6 to
`scripts/run_behaviour_hypotheses.py`. Each section gives the function
name, signature, inputs, algorithm, statistical tests, JSON output
structure, and print summary format. A developer can implement directly
from this plan.

Status: 2026-05-31

---

## Prerequisites and Shared Infrastructure

### Existing code reused without modification

| Component | Source | Purpose |
|---|---|---|
| `load_session_data()` | `run_behaviour_hypotheses.py` | Download sync.h5, extract arrays |
| `detect_epochs()` | `run_behaviour_hypotheses.py` | Detect contiguous light/dark epochs |
| `wilcoxon_test()` | `run_behaviour_hypotheses.py` | Wilcoxon signed-rank + rank-biserial |
| `holm_bonferroni()` | `run_behaviour_hypotheses.py` | Multiple comparisons correction |
| `_make_serializable()` | `run_behaviour_hypotheses.py` | JSON conversion |
| `discretize_position_fast()` | `hm2p.maze.discretize` | Continuous position -> cell index |
| `cell_sequence()` | `hm2p.maze.discretize` | Cell index array -> transition sequence |
| `transition_matrix()` | `hm2p.maze.analysis` | First-order Markov transition matrix |
| `cell_occupancy()` | `hm2p.maze.analysis` | Frame counts per cell |
| `RoseMaze` | `hm2p.maze.topology` | Maze graph, node types, distances |
| `MAZE = build_rose_maze()` | Global constant | 23 cells, 7 junctions, 7 corridors, 9 dead ends |

### New shared helper needed

```python
def spearman_test(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman rank correlation with n and CI.

    Returns dict with keys: rho, p, n, test.
    """
```

Signature and logic: filter NaN pairs, require n >= 6, call
`scipy.stats.spearmanr`, return `{"rho": float, "p": float, "n": int,
"test": "spearman_rank"}`. On insufficient data, return
`{"rho": None, "p": None, "n": n, "test": "spearman_rank"}`.

---

## Phase A — Must-Implement

### Direction 1: Within-Epoch Temporal Dynamics (H5)

#### Purpose

Distinguish gradual representational degradation (H5a) from immediate
strategy switch (H5b) by examining how coverage accumulates within
60-second epochs.

#### Constants

```python
HALF_EPOCH_S = 30.0       # Split point within each epoch
CUMULATIVE_BIN_S = 5.0    # Bin width for cumulative coverage curves
```

---

#### Function 1a: `compute_h5_per_session`

```python
def compute_h5_per_session(data: dict, maze: RoseMaze) -> dict:
```

**Inputs from `data`:**
- `x_maze`, `y_maze`: float64 arrays (maze coordinates)
- `light_on`: bool array
- `bad_behav`: bool array
- `speed_cm_s`: float64 array
- `fps`: float (from `data["fps"]`, typically ~9.6)
- `frame_times`: float64 array (absolute timestamps)

**Algorithm:**

1. Compute `valid`, `cell_indices` as in existing H1-H4 functions.

2. Call `detect_epochs(light_on, fps)`. Filter to epochs with
   `duration_s >= MIN_EPOCH_DURATION_S` (30 s).

3. For each epoch, compute the midpoint frame index:
   ```python
   mid = ep["start"] + int(round(HALF_EPOCH_S * fps))
   mid = min(mid, ep["end"])  # safety clamp
   ```

4. For each epoch, compute:

   a. **First-half unique cells:** unique valid cell indices in
      `[ep["start"], mid)`.

   b. **Second-half NEW cells:** unique valid cell indices in
      `[mid, ep["end"])` that were NOT visited in the first half
      of this epoch.

   c. **Coverage ratio:** `new_cells_2nd / unique_cells_1st` if
      `unique_cells_1st > 0`, else `NaN`. This measures how much
      *new* coverage the second half adds relative to the first half.

   d. **Speed first half:** `nanmean(speed[start:mid][valid])`.

   e. **Speed second half:** `nanmean(speed[mid:end][valid])`.

   f. **Speed ratio:** `speed_2nd / speed_1st` if `speed_1st > 0`.

5. Collect per-epoch metrics into lists by condition:
   ```
   coverage_ratio_light, coverage_ratio_dark
   speed_ratio_light, speed_ratio_dark
   unique_cells_1st_light, unique_cells_1st_dark
   new_cells_2nd_light, new_cells_2nd_dark
   ```

6. **Cumulative coverage curve** (5-second bins, per condition):

   For each epoch, compute `n_bins = int(ep_duration_s / CUMULATIVE_BIN_S)`.
   For each bin `b` (0..n_bins-1), count cumulative unique cells visited
   from epoch start through the end of bin `b`. Store as a 2D array
   `(n_epochs, n_bins)` for each condition. Normalize to fraction of 23
   cells.

   To handle variable epoch lengths, truncate all epochs to the minimum
   number of bins across epochs in that condition (typically 12 bins for
   60-second epochs at 5-second resolution). If some epochs are shorter,
   pad with the last cumulative value (monotonic by definition).

   Compute across-epoch mean and SEM for each bin, separately for light
   and dark.

7. **Lights-on recovery** (positive control):

   For each session, identify:
   - "recovery epochs": light epochs that immediately follow a dark epoch.
   - "initial epoch": the first light epoch of the session.

   For each recovery epoch, compute unique cells in first 30 seconds.
   For the initial epoch, compute unique cells in first 30 seconds.
   Store as `recovery_cov_first_half` (median across recovery epochs)
   and `initial_cov_first_half` (single value per session).

**Returns:**

```python
{
    "median_coverage_ratio_light": float,   # session median across light epochs
    "median_coverage_ratio_dark": float,    # session median across dark epochs
    "median_speed_ratio_light": float,
    "median_speed_ratio_dark": float,
    "mean_unique_1st_light": float,         # mean unique cells in 1st half, light
    "mean_unique_1st_dark": float,
    "mean_new_2nd_light": float,            # mean new cells in 2nd half, light
    "mean_new_2nd_dark": float,
    "cumulative_curve_light": list[float],  # mean cumulative coverage per 5s bin
    "cumulative_curve_dark": list[float],
    "cumulative_sem_light": list[float],
    "cumulative_sem_dark": list[float],
    "n_bins": int,                          # number of 5s bins
    "recovery_cov_first_half": float,       # median recovery epoch 1st-half coverage
    "initial_cov_first_half": float,        # initial light epoch 1st-half coverage
    "n_light_epochs": int,
    "n_dark_epochs": int,
}
```

---

#### Function 1b: `test_h5`

```python
def test_h5(session_results: list[dict]) -> dict:
```

**Inputs:** List of per-session dicts from `compute_h5_per_session`.

**Tests (all N = 20 sessions, using session-level medians):**

1. **Coverage ratio: dark vs light (primary test).**
   - Extract `median_coverage_ratio_light` and `median_coverage_ratio_dark`
     per session.
   - `wilcoxon_test(coverage_ratio_light, coverage_ratio_dark)`.
   - If dark ratio < light ratio: second half of dark epochs adds fewer
     new cells (gradual degradation, H5a).
   - If no difference: immediate onset or no within-epoch dynamics.

2. **Speed ratio: dark vs light (control).**
   - `wilcoxon_test(speed_ratio_light, speed_ratio_dark)`.
   - Tests whether speed also declines within dark epochs (locomotor
     confound).

3. **Coverage ratio dark vs light, after speed-ratio residualisation
   (partial correlation control).**
   - Compute `coverage_ratio_diff = dark - light` and
     `speed_ratio_diff = dark - light` per session.
   - `spearman_test(coverage_ratio_diff, speed_ratio_diff)`.
   - If the coverage effect persists after partialling out speed: the
     coverage decline is not purely locomotor.

4. **Lights-on recovery (positive control).**
   - Extract `recovery_cov_first_half` and `initial_cov_first_half`.
   - `wilcoxon_test(recovery, initial)`.
   - If no difference: coverage rebounds immediately at lights-on,
     confirming rapid re-anchoring.

5. **Holm-Bonferroni across the 3 primary tests** (coverage ratio,
   speed ratio, recovery). The Spearman correlation is a control and
   reported separately.

6. **Grand-mean cumulative curves** (for plotting, not statistical
   testing):
   - Average `cumulative_curve_light` and `cumulative_curve_dark`
     across sessions (element-wise mean +/- SEM).
   - Store as arrays in the output.

**Returns:**

```python
{
    "coverage_ratio_test": {
        "mean_light": float, "mean_dark": float,
        "p": float, "p_adj": float, "r": float, "n": int,
        "test": "wilcoxon_signed_rank",
    },
    "speed_ratio_test": {
        "mean_light": float, "mean_dark": float,
        "p": float, "p_adj": float, "r": float, "n": int,
    },
    "speed_coverage_correlation": {
        "rho": float, "p": float, "n": int,
        "test": "spearman_rank",
    },
    "recovery_test": {
        "mean_recovery": float, "mean_initial": float,
        "p": float, "p_adj": float, "r": float, "n": int,
    },
    "grand_cumulative_light": list[float],  # 12 bins, mean across sessions
    "grand_cumulative_dark": list[float],
    "grand_cumulative_sem_light": list[float],
    "grand_cumulative_sem_dark": list[float],
    "n_bins": int,
    "bin_width_s": float,  # = 5.0
}
```

---

#### Print summary format (H5)

```
--- H5: Within-epoch temporal dynamics ---
  Coverage ratio (2nd/1st half):
    Light: {mean_light:.3f}, Dark: {mean_dark:.3f}
    p = {p:.4f}, p_adj = {p_adj:.4f}, r = {r:.3f}, N = {n}
  Speed ratio (2nd/1st half):
    Light: {mean_light:.3f}, Dark: {mean_dark:.3f}
    p = {p:.4f}, p_adj = {p_adj:.4f}, r = {r:.3f}, N = {n}
  Speed-coverage correlation: rho = {rho:.3f}, p = {p:.4f}
  Lights-on recovery:
    Recovery: {mean_recovery:.3f}, Initial: {mean_initial:.3f}
    p = {p:.4f}, p_adj = {p_adj:.4f}, r = {r:.3f}, N = {n}
  Interpretation: {gradual|immediate|inconclusive}
```

Interpretation logic:
- "gradual" if `coverage_ratio_test.p_adj < 0.05` AND dark ratio < light ratio
- "immediate" if `coverage_ratio_test.p_adj >= 0.05` AND the cumulative
  curves diverge in bin 0 or 1 (visual inspection flag; no automated test)
- "inconclusive" otherwise

---

### Direction 4: Epoch-Number Adaptation (H8)

#### Purpose

Test whether route stereotypy changes across repeated dark epochs within
a session, distinguishing adaptation (H8a), constancy (H8b), or
worsening (H8c).

---

#### Function 4a: `compute_h8_per_session`

```python
def compute_h8_per_session(data: dict, maze: RoseMaze) -> dict:
```

**Inputs:** Same as H5 (`x_maze`, `y_maze`, `light_on`, `bad_behav`,
`speed_cm_s`, `fps`).

**Algorithm:**

1. Compute `valid`, `cell_indices` as standard.

2. Call `detect_epochs(light_on, fps)`. Filter to `duration_s >= MIN_EPOCH_DURATION_S`.

3. **Pair dark epochs with their preceding light epochs:**

   Iterate through filtered epochs in order. For each dark epoch, look
   back for the most recent light epoch. Form epoch pairs
   `(light_n, dark_n)` where `n` is the pair index (1-based).

   Skip dark epochs that have no preceding light epoch (e.g., if the
   session starts dark, or if a light epoch was too short and got
   filtered out). Also skip light epochs not followed by a dark epoch.

4. For each epoch pair `n`:

   a. **Coverage delta:** `coverage_light_n - coverage_dark_n` (positive
      = drop in dark). Coverage = unique cells / 23.

   b. **Speed delta:** `mean_speed_light_n - mean_speed_dark_n`.

   c. **Dark coverage (absolute):** unique cells in dark epoch / 23.

   d. **Light coverage (absolute):** unique cells in light epoch / 23.

5. For each standalone dark epoch (numbered):

   a. **Coverage:** unique cells / 23.

   b. **Speed:** mean speed.

6. Store arrays:
   ```
   epoch_numbers: [1, 2, 3, ..., K]  # K = number of epoch pairs
   coverage_deltas: [delta_1, delta_2, ...]
   dark_coverages: [cov_1, cov_2, ...]
   light_coverages: [cov_1, cov_2, ...]
   speed_deltas: [delta_1, delta_2, ...]
   dark_speeds: [speed_1, speed_2, ...]
   ```

7. **Within-session Spearman correlation:** epoch_number vs coverage_delta.
   Store `rho` and `p`.

8. **Early vs late classification:**
   - Early = first third of epoch pairs: indices `[0, K//3)`.
   - Late = last third: indices `[K - K//3, K)`.
   - Store early/late mean coverage_delta and dark_coverage.

9. **First dark epoch vs rest:**
   - `first_dark_coverage = dark_coverages[0]`
   - `rest_dark_coverage = mean(dark_coverages[1:])`
   - Same for speed.

**Returns:**

```python
{
    "n_epoch_pairs": int,
    "epoch_numbers": list[int],
    "coverage_deltas": list[float],      # light - dark per pair
    "dark_coverages": list[float],
    "light_coverages": list[float],
    "speed_deltas": list[float],
    "dark_speeds": list[float],
    "within_session_rho": float | None,  # Spearman: epoch_number vs coverage_delta
    "within_session_p": float | None,
    "early_mean_delta": float,           # mean coverage delta, first third
    "late_mean_delta": float,            # mean coverage delta, last third
    "early_mean_dark_cov": float,
    "late_mean_dark_cov": float,
    "first_dark_coverage": float,
    "rest_mean_dark_coverage": float,
    "first_dark_speed": float,
    "rest_mean_dark_speed": float,
}
```

---

#### Function 4b: `test_h8`

```python
def test_h8(session_results: list[dict]) -> dict:
```

**Tests:**

1. **Session-level slope direction (primary).**
   - Extract `within_session_rho` per session (discard None).
   - One-sample Wilcoxon: are session-level rho values different from 0?
     `scipy.stats.wilcoxon(rhos, alternative="two-sided")`.
   - If median rho < 0: adaptation (coverage delta decreases with epoch
     number, meaning the dark-light gap shrinks). H8a.
   - If median rho > 0: worsening. H8c.
   - If no significant effect: constant. H8b.

2. **Early vs late coverage delta (paired).**
   - Extract per-session `early_mean_delta` and `late_mean_delta`.
   - `wilcoxon_test(early, late)`.
   - Direction: if early > late, the dark-light gap is shrinking (H8a).

3. **First dark epoch vs rest (paired).**
   - Extract `first_dark_coverage` and `rest_mean_dark_coverage`.
   - `wilcoxon_test(first, rest)`.
   - If first < rest: first dark epoch is worse (mouse adapts). H8a.
   - If first > rest: first dark epoch is better (novelty advantage, then
     fatigue). H8c.

4. **Speed control: early vs late speed delta (paired).**
   - Extract per-session early/late mean speed delta.
   - Actually simpler: test within_session_rho for speed_delta vs
     epoch_number per session, then one-sample Wilcoxon on those rhos.
   - If speed also changes with epoch number, the coverage effect could
     be locomotor.

5. **Light-epoch coverage vs epoch number (control).**
   - Spearman: epoch_number vs light_coverage, within each session.
   - One-sample Wilcoxon on session-level rhos.
   - If light coverage also declines: global session effect, not
     darkness-specific.

6. **Holm-Bonferroni across 3 primary tests:** slope direction, early vs
   late, first vs rest.

**Returns:**

```python
{
    "slope_direction_test": {
        "median_rho": float,
        "mean_rho": float,
        "p": float,
        "p_adj": float,
        "r": float,    # rank-biserial for the one-sample Wilcoxon
        "n": int,
        "test": "wilcoxon_one_sample",
    },
    "early_vs_late_test": {
        "mean_early_delta": float,
        "mean_late_delta": float,
        "p": float,
        "p_adj": float,
        "r": float,
        "n": int,
    },
    "first_vs_rest_test": {
        "mean_first_cov": float,
        "mean_rest_cov": float,
        "p": float,
        "p_adj": float,
        "r": float,
        "n": int,
    },
    "speed_slope_control": {
        "median_rho": float,
        "p": float,
        "n": int,
        "test": "wilcoxon_one_sample",
    },
    "light_coverage_slope_control": {
        "median_rho": float,
        "p": float,
        "n": int,
        "test": "wilcoxon_one_sample",
    },
    "interpretation": "adaptation|constant|worsening",
}
```

Interpretation logic:
- "adaptation" if slope_direction p_adj < 0.05 AND median_rho < 0
- "worsening" if slope_direction p_adj < 0.05 AND median_rho > 0
- "constant" otherwise

---

#### Print summary format (H8)

```
--- H8: Epoch-number adaptation ---
  Within-session slope (epoch# vs coverage delta):
    Median rho = {median_rho:.3f}, p = {p:.4f}, p_adj = {p_adj:.4f}, N = {n}
  Early vs late coverage delta:
    Early: {mean_early:.3f}, Late: {mean_late:.3f}
    p = {p:.4f}, p_adj = {p_adj:.4f}, r = {r:.3f}, N = {n}
  First dark epoch vs rest:
    First: {mean_first:.3f}, Rest: {mean_rest:.3f}
    p = {p:.4f}, p_adj = {p_adj:.4f}, r = {r:.3f}, N = {n}
  Speed slope control: median rho = {rho:.3f}, p = {p:.4f}
  Light coverage slope control: median rho = {rho:.3f}, p = {p:.4f}
  Interpretation: {interpretation}
```

---

## Phase B — Should-Implement

### Direction 2: Corridor Heatmap (H6)

#### Purpose

Identify which specific maze cells show the largest light-dark coverage
change, providing a spatial map of route stereotypy.

---

#### Function 2a: `compute_h6_per_session`

```python
def compute_h6_per_session(data: dict, maze: RoseMaze) -> dict:
```

**Algorithm:**

1. Compute `valid`, `cell_indices` as standard.

2. Detect epochs, filter to `duration_s >= MIN_EPOCH_DURATION_S`.

3. For each cell index `c` (0..22):

   a. **Light visit fraction:** fraction of light epochs in which cell
      `c` was visited at least once. I.e., for each light epoch, check
      if any valid frame has `cell_indices == c`. Count epochs where
      this is true, divide by total light epochs.

   b. **Dark visit fraction:** same for dark epochs.

4. Compute per-cell **visit rate** (visits/min) in each condition:
   - For each epoch, build cell sequence (transitions). Count how many
     times cell `c` appears in the cell sequence (entries, not frames).
   - Sum entries across epochs in each condition, divide by total
     condition duration in minutes.

5. Store two arrays of length 23: `visit_frac_light[c]`,
   `visit_frac_dark[c]`.

**Returns:**

```python
{
    "visit_frac_light": list[float],    # length 23, fraction of epochs visited
    "visit_frac_dark": list[float],
    "visit_rate_light": list[float],    # length 23, entries/min
    "visit_rate_dark": list[float],
}
```

---

#### Function 2b: `test_h6`

```python
def test_h6(session_results: list[dict], maze: RoseMaze) -> dict:
```

**Algorithm:**

1. For each cell `c` (0..22), compute the per-session visit fraction
   delta: `dark - light`. Take session-level mean delta for each cell.
   This gives one delta per cell across sessions.

2. **Per-cell visit fraction in light vs dark (paired across sessions):**
   For each cell, extract `visit_frac_light[c]` and `visit_frac_dark[c]`
   across all sessions (N = 20 paired values per cell). Run
   `wilcoxon_test` for each cell. Apply Holm-Bonferroni across 23 cells.

   Note: many cells will have ceiling effects (visited in all epochs).
   Report all 23 tests but flag that the family-wise correction makes
   individual cell significance conservative. The primary output is the
   delta heatmap, not the per-cell p-values.

3. **Eccentricity correlation (corridors only, descriptive):**
   - Compute eccentricity for each cell: `max(maze.dist[c, :])`.
   - Restrict to the 7 corridor cells.
   - Spearman: eccentricity vs mean delta across sessions.
   - N = 7 so this is descriptive; report rho but note underpowered.

4. **Distance from center correlation (all cells, descriptive):**
   - Center = cell with minimum eccentricity: `(3, 2)` (eccentricity 5).
   - Distance from center for each cell: `maze.dist[c, center_idx]`.
   - Spearman: distance vs mean delta across sessions (N = 23).
   - This has better power (N = 23) than corridor-only.

5. **Per-cell heatmap data for frontend visualisation:**
   - For each cell, output `(col, row, mean_delta, node_type)`.
   - This JSON structure is consumed by the frontend to render a
     colour-coded 7x5 grid.

**Returns:**

```python
{
    "per_cell": [
        {
            "cell": [col, row],
            "cell_idx": int,
            "node_type": str,  # "dead_end", "corridor", "t_junction"
            "mean_visit_frac_light": float,
            "mean_visit_frac_dark": float,
            "mean_delta": float,  # dark - light (negative = less visited in dark)
            "p_raw": float | None,
            "p_adj": float | None,
            "r": float | None,
            "n": int,
        },
        ...  # 23 entries
    ],
    "eccentricity_correlation_corridors": {
        "rho": float | None,
        "p": float | None,
        "n": 7,
        "test": "spearman_rank",
        "note": "N=7, descriptive only",
    },
    "distance_from_center_correlation": {
        "rho": float | None,
        "p": float | None,
        "n": 23,
        "test": "spearman_rank",
    },
    "heatmap_data": {
        "cells": [[col, row], ...],     # 23 entries
        "deltas": [float, ...],          # 23 entries, dark - light
        "node_types": [str, ...],        # 23 entries
    },
}
```

---

#### Print summary format (H6)

```
--- H6: Corridor-specific coverage ---
  Cells with significant delta (p_adj < 0.05): {list}
  Top 5 cells by |delta|:
    {cell}: delta={delta:.3f}, type={type}, p_adj={p_adj:.4f}
    ...
  Eccentricity correlation (corridors, N=7): rho={rho:.3f}, p={p:.3f}
  Distance-from-center correlation (all, N=23): rho={rho:.3f}, p={p:.4f}
```

---

### Direction 6: Cell-Type Markov Model (H10)

#### Purpose

Test whether collapsing the 23-cell Markov model to 3 cell types
(junction=J, corridor=C, dead-end=D) reveals second-order structure that
differs between light and dark.

---

#### Function 6a: `compute_h10_per_session`

```python
def compute_h10_per_session(data: dict, maze: RoseMaze) -> dict:
```

**Algorithm:**

1. Compute `valid`, `cell_indices`, cell sequences for light/dark as in
   H2.

2. **Map cell indices to type indices:**
   ```python
   TYPE_MAP = {}  # cell_idx -> type_idx
   for c in maze.junctions:
       TYPE_MAP[maze.cell_to_idx[c]] = 0  # J
   for c in maze.corridors:
       TYPE_MAP[maze.cell_to_idx[c]] = 1  # C
   for c in maze.dead_ends:
       TYPE_MAP[maze.cell_to_idx[c]] = 2  # D
   TYPE_NAMES = ["J", "C", "D"]
   N_TYPES = 3
   ```

3. **Convert cell sequence to type sequence:**
   ```python
   def cell_seq_to_type_seq(cs: np.ndarray) -> np.ndarray:
       ts = np.array([TYPE_MAP.get(c, -1) for c in cs])
       # Remove consecutive duplicates
       mask = np.concatenate([[True], ts[1:] != ts[:-1]])
       return ts[mask]
   ```

   Rationale for removing consecutive type duplicates: two adjacent
   corridor cells (e.g., traversing (2,2) -> (4,2) via (3,2), which is
   a junction, so C->J->C) produce a C->C transition only if two
   adjacent cells share the same type. In this maze, that happens rarely
   (e.g., entering a dead-end branch: junction (1,2) -> corridor (1,1)
   -> junction (1,0) has no consecutive same-type). However, the
   filtering handles any edge case.

4. **First-order type transition matrix (3x3):**
   - Count transitions in the type sequence, build `tm_1st[3, 3]`.
   - Row-normalize.

5. **Second-order type transition matrix (3x3x3):**
   - Count triplets `(type[t-1], type[t], type[t+1])`.
   - Build `tm_2nd[3, 3, 3]`. Normalise `tm_2nd[i, j, :].sum() = 1`.

6. **JSD between light and dark first-order type transition matrices:**
   - Same weighted-JSD as in H2 but on 3x3 matrices.

7. **Extract key second-order transition probabilities:**
   - `P(D | J, C)` = `tm_2nd[0, 1, 2]` — probability of reaching
     dead-end after junction -> corridor.
   - `P(J | C, J)` = `tm_2nd[1, 0, 1]` — wait, this is wrong.
     Need to think about indices carefully.

   Actually, `tm_2nd[i, j, k]` = P(next_type = k | prev_type = i,
   curr_type = j). So:
   - `P(D | J->C)` = `tm_2nd[J, C, D]` = `tm_2nd[0, 1, 2]`
   - `P(J | C->J)` = `tm_2nd[C, J, J]` — but this is J->J which
     cannot happen (no adjacent junctions). Actually wait — after
     removing type-duplicates, J->J transitions are impossible. So
     `P(J | C, J)` means the probability that after C->J, the next
     type is J. But J->J is impossible in this maze (all junctions are
     separated by corridors), so `tm_2nd[1, 0, 0]` should be ~0.

   Revised key transitions:
   - `P(D | J, C)` = `tm_2nd[0, 1, 2]` — commitment to dead end
   - `P(J | J, C)` = `tm_2nd[0, 1, 0]` — backbone traversal (J->C->J)
   - `P(J | D, C)` = `tm_2nd[2, 1, 0]` — dead-end return to junction
   - `P(C | C, J)` = `tm_2nd[1, 0, 1]` — passing through junction
     (C->J->C)
   - `P(D | C, J)` = `tm_2nd[1, 0, 2]` — dead end after passing
     through junction

8. **BIC comparison: 1st vs 2nd order at type level:**
   - Use existing `markov_order_comparison` logic adapted for the
     type sequence. Or reimplement inline since the type sequence
     uses a different state space (N_TYPES=3 instead of n_cells=23).
   - Free parameters: 1st order = 3 * (3-1) = 6 (but structurally
     zero transitions reduce this). 2nd order = 9 * (3-1) = 18
     (minus structural zeros).
   - Actually, easier to just count observed states/pairs and use
     the same formula as `markov_order_comparison`.
   - Compute on the FULL type sequence (light + dark combined), since
     model order is a structural property, not a condition comparison.

**Returns:**

```python
{
    "tm_1st_light": list[list[float]],   # 3x3
    "tm_1st_dark": list[list[float]],    # 3x3
    "tm_2nd_light": list[list[list[float]]],  # 3x3x3
    "tm_2nd_dark": list[list[list[float]]],   # 3x3x3
    "jsd_type_level": float,             # JSD between 1st-order light/dark
    "n_type_transitions_light": int,
    "n_type_transitions_dark": int,
    # Key second-order probabilities
    "p_D_given_JC_light": float,         # P(D | J->C), light
    "p_D_given_JC_dark": float,
    "p_J_given_JC_light": float,         # P(J | J->C), light (backbone)
    "p_J_given_JC_dark": float,
    "p_J_given_DC_light": float,         # P(J | D->C), light (DE return)
    "p_J_given_DC_dark": float,
    "p_C_given_CJ_light": float,         # P(C | C->J), light (pass-through)
    "p_C_given_CJ_dark": float,
    # BIC comparison (computed on full session, not per-condition)
    "type_markov_order": {
        "delta_bic": float,
        "preferred_order": int,          # 1 or 2
    },
}
```

---

#### Function 6b: `test_h10`

```python
def test_h10(session_results: list[dict], n_permutations: int = 1000) -> dict:
```

**Tests:**

1. **JSD: type-level light vs dark.**
   - Same permutation test as H2 but on type-level transition matrices.
   - Observed mean JSD vs permutation null (shuffle epoch labels).

2. **P(D | J,C): light vs dark (primary).**
   - Wilcoxon paired on per-session `p_D_given_JC_light` vs
     `p_D_given_JC_dark` (N = 20).
   - If dark < light: mice avoid committing to dead-end branches. H10
     confirmed.

3. **P(J | J,C): light vs dark.**
   - Wilcoxon paired (N = 20).
   - If dark > light: mice preferentially traverse backbone (J->C->J).

4. **P(C | C,J): light vs dark.**
   - Wilcoxon paired (N = 20).
   - Pass-through junction behaviour.

5. **P(J | D,C): light vs dark (dead-end return).**
   - Wilcoxon paired (N = 20).

6. **BIC model order: 2nd vs 1st.**
   - One-sample Wilcoxon on per-session `delta_bic` (test > 0 means
     2nd order preferred).

7. **Holm-Bonferroni across the 4 transition tests** (tests 2-5).

**Returns:**

```python
{
    "jsd_type_test": {
        "observed_mean_jsd": float,
        "permutation_p": float,
        "null_mean": float,
        "null_95_pct": float,
        "n": int,
    },
    "p_D_given_JC": {
        "mean_light": float, "mean_dark": float,
        "p": float, "p_adj": float, "r": float, "n": int,
    },
    "p_J_given_JC": {
        "mean_light": float, "mean_dark": float,
        "p": float, "p_adj": float, "r": float, "n": int,
    },
    "p_C_given_CJ": {
        "mean_light": float, "mean_dark": float,
        "p": float, "p_adj": float, "r": float, "n": int,
    },
    "p_J_given_DC": {
        "mean_light": float, "mean_dark": float,
        "p": float, "p_adj": float, "r": float, "n": int,
    },
    "model_order": {
        "mean_delta_bic": float,
        "n_prefer_2nd": int,
        "n_total": int,
        "p": float,
        "test": "wilcoxon_one_sample",
    },
}
```

---

#### Print summary format (H10)

```
--- H10: Cell-type Markov model ---
  Type-level JSD (light vs dark):
    Observed: {obs:.4f}, Null: {null:.4f}, p_perm = {p:.4f}, N = {n}
  Key transitions (2nd-order, light vs dark):
    P(D|J,C): light={light:.3f}, dark={dark:.3f}, p={p:.4f}, p_adj={p_adj:.4f}, r={r:.3f}
    P(J|J,C): light={light:.3f}, dark={dark:.3f}, p={p:.4f}, p_adj={p_adj:.4f}, r={r:.3f}
    P(C|C,J): light={light:.3f}, dark={dark:.3f}, p={p:.4f}, p_adj={p_adj:.4f}, r={r:.3f}
    P(J|D,C): light={light:.3f}, dark={dark:.3f}, p={p:.4f}, p_adj={p_adj:.4f}, r={r:.3f}
  Model order (type-level): {n_prefer_2nd}/{n_total} prefer 2nd order, delta_BIC = {dBIC:.1f}, p = {p:.4f}
```

---

## Integration into `run_behaviour_hypotheses.py`

### Structure changes

1. **Add new constants** at the top:
   ```python
   HALF_EPOCH_S = 30.0
   CUMULATIVE_BIN_S = 5.0
   ```

2. **Add new functions** in order: `spearman_test`,
   `compute_h5_per_session`, `test_h5`, `compute_h6_per_session`,
   `test_h6`, `compute_h8_per_session`, `test_h8`,
   `compute_h10_per_session`, `test_h10`.

3. **In `main()`**, after the existing per-session loop, add:
   ```python
   # H5
   r5 = compute_h5_per_session(data, MAZE)
   h5_results.append(r5)
   print(f"  H5: cov_ratio L/D={r5['median_coverage_ratio_light']:.3f}/{r5['median_coverage_ratio_dark']:.3f}")

   # H6
   r6 = compute_h6_per_session(data, MAZE)
   h6_results.append(r6)

   # H8
   r8 = compute_h8_per_session(data, MAZE)
   h8_results.append(r8)
   print(f"  H8: epoch-cov slope rho={r8['within_session_rho']}")

   # H10
   r10 = compute_h10_per_session(data, MAZE)
   h10_results.append(r10)
   print(f"  H10: P(D|J,C) L/D={r10['p_D_given_JC_light']:.3f}/{r10['p_D_given_JC_dark']:.3f}")
   ```

4. **Cross-session tests:**
   ```python
   h5_stats = test_h5(h5_results)
   h6_stats = test_h6(h6_results, MAZE)
   h8_stats = test_h8(h8_results)
   h10_stats = test_h10(h10_results)
   ```

5. **Update the output JSON** to include:
   ```python
   output["h5_temporal_dynamics"] = h5_stats
   output["h6_corridor_heatmap"] = h6_stats
   output["h8_epoch_adaptation"] = h8_stats
   output["h10_cell_type_markov"] = h10_stats
   ```

6. **Update OUTPUT_JSON** path: Consider either extending the existing
   `behaviour-hypotheses-results.json` or writing a separate
   `behaviour-hypotheses-tier2-results.json`. Recommendation: write to a
   **separate file** to avoid breaking any code that reads the Tier-1
   JSON:
   ```python
   OUTPUT_JSON_T2 = (
       Path(__file__).resolve().parent.parent
       / "docs"
       / "manuscripts"
       / "behaviour-hypotheses-tier2-results.json"
   )
   ```

### Imports to add

```python
from hm2p.maze.analysis import cell_occupancy  # for H6
```

No other new external imports needed. `scipy.stats.spearmanr` is already
imported via `sp_stats`.

---

## Confound Checklist

| Confound | Controlled by | Applied in |
|---|---|---|
| Bad behaviour frames | `bad_behav` mask | All functions |
| Short epochs | `MIN_EPOCH_DURATION_S = 30` filter | All functions |
| Speed confound | Speed ratio test (H5), speed slope (H8) | H5, H8 |
| Ceiling effect in coverage ratio | Relative comparison (dark vs light ratio), not absolute | H5 |
| Epoch-position confound | Lights-on recovery control (H5), light-epoch slope control (H8) | H5, H8 |
| Multiple comparisons | Holm-Bonferroni within each analysis family | All tests |
| Low N per corridor cell (H6) | Reported as descriptive, N=7 flagged | H6 |
| Structural zeros in type transitions | Verified against maze adjacency | H10 |

---

## Testing Strategy

Each function gets unit tests with synthetic data in
`tests/test_behaviour_hypotheses_tier2.py`.

### Synthetic data construction

```python
def _make_synthetic_session(
    n_frames: int = 5760,   # 10 min at 9.6 Hz
    fps: float = 9.6,
    n_light_dark_cycles: int = 5,
    seed: int = 42,
) -> dict:
    """Create a synthetic session dict matching load_session_data output."""
```

Generate: random walk on maze graph producing `x_maze`, `y_maze`;
alternating light/dark epochs of 60 seconds; zero `bad_behav`; constant
speed.

### Test cases per function

| Function | Test | What it verifies |
|---|---|---|
| `compute_h5_per_session` | Constant random walk | coverage ratio ~same for light/dark |
| `compute_h5_per_session` | Dark epochs restricted to 3 cells in 2nd half | dark coverage ratio < light |
| `compute_h5_per_session` | Zero valid frames | Returns NaN gracefully |
| `test_h5` | 20 identical sessions | p > 0.05 (no effect) |
| `compute_h8_per_session` | Constant walk | within_session_rho ~0 |
| `compute_h8_per_session` | Dark coverage improves with epoch# | rho < 0 |
| `compute_h6_per_session` | All cells visited every epoch | visit_frac = 1.0 |
| `compute_h10_per_session` | Simple J-C-D sequence | Correct transition probs |
| `test_h10` | Permutation null with shuffled data | p not significant |
| `spearman_test` | Known ranked data | Correct rho |
| `spearman_test` | N < 6 | Returns None gracefully |

---

## Runtime Estimate

Each new per-session function adds O(N_frames) work (one pass through
the frame arrays). With ~50,000 frames per session and 20 sessions:

- H5: ~1 second total (simple array slicing)
- H6: ~1 second total (cell occupancy per epoch)
- H8: ~0.5 seconds total (epoch pairing + correlation)
- H10: ~2 seconds total (type sequence construction + 2nd-order matrix)

The H10 permutation test (1000 permutations) adds ~30 seconds. Total
additional runtime: ~35 seconds on top of the existing ~2-minute Tier-1
run.

---

## References

Stackman RW, Taube JS. 1997. "Firing properties of head direction cells
in the rat anterior thalamic nucleus." J Neurosci 17, 9020-9037.

Peyrache A et al. 2015. "Internally organized mechanisms of the head
direction sense." Nat Neurosci 18, 569-575.

Zugaro MB et al. 2003. "Rapid spatial reorientation and head direction
cells." J Neurosci 23, 3478-3482.

Fonio E et al. 2009. "Freedom of movement and the stability of its
unfolding in free exploration of mice." PNAS 106, 21335-21340.

Rosenberg M et al. 2021. "Mice in a labyrinth show rapid learning,
sudden insight, and efficient exploration." eLife 10, e66175.

Schmitzer-Torbert N, Redish AD. 2002. "Development of path stereotypy
in a single day in rats on a multiple-T maze." Behav Neurosci 116,
1058-1070.

Bhakti B et al. 2024. "Stochastic characterization of navigation
strategies in an automated variant of the Barnes maze." eLife 13,
e88648.
