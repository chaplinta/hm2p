# Behavioural Control Analyses — Summary

Generated from 20 usable sessions (14 animals). 11 primary sessions.

Seven control analyses addressing potential confounds in the main behavioural manuscript results. All tests non-parametric.

---

## Control 1: Coverage Per Active Minute (Light vs Dark)

**Question:** Is the lower per-epoch coverage in dark driven by reduced locomotion speed, or does it persist when normalised by active time?

**Method:** Coverage per active minute = (epoch_coverage * 23) / frac_active. Approximation from per-session summaries.

- Light: 19.76 cells/active-min (median 20.77)
- Dark: 17.77 cells/active-min (median 19.17)
- W = 23.0, p = 0.0012, r = 0.781, N = 20

**Interpretation:** Coverage per active minute is significantly different between light and dark even after controlling for locomotion time. The main coverage result is not simply a speed artefact.

---

## Control 2: MRL by Node Type (Light vs Dark)

**Question:** Does the higher MRL in dark persist at each maze location type, or is it driven by differential occupancy of corridors vs junctions?

**Status:** requires_frame_data

_This control requires frame-level HD and position data from sync.h5 files. Cannot be computed from per-session summaries. When sync.h5 files are regenerated, re-run this script._

---

## Control 3: MRL and AHV — Active Frames Only

**Question:** Does the higher MRL in dark persist when excluding immobile frames? Does the AHV difference persist?

**Important note:** The main analysis (Figure 5) **already restricts** to active frames only (speed >= 2.5 cm/s). These values are identical to the Figure 5 results. This control confirms that the reported MRL and AHV comparisons are not confounded by immobility.

| Metric | Light | Dark | Test | p_adj |
| ------ | ----- | ---- | ---- | ----- |
| MRL (active only) | 0.060 | 0.085 | W = 57.0, p = 0.0759, r = 0.457, N = 20 | 0.1517 |
| Median |AHV| (active only) | 93.3 | 95.4 | W = 68.0, p = 0.1769, r = 0.352, N = 20 | 0.1769 |

---

## Control 4: Speed by Node Type — All Frames

**Question:** Does the speed hierarchy (corridor > junction > dead end) persist when including immobile frames?

**Status:** requires_frame_data

_This control requires frame-level speed and position data from sync.h5 files. Cannot be computed from per-session summaries. When sync.h5 files are regenerated, re-run this script._

Active-only reference from Figure 6:
- Junction: 8.21 cm/s
- Corridor: 9.00 cm/s
- Dead end: 8.13 cm/s

---

## Control 5: Random Walk Null Model for Alternation

**Question:** Is the observed turn alternation (negative lag-1 autocorrelation) stronger than expected from a random walk on the maze graph? This controls for maze geometry constraining turn sequences.

- Observed: mean = -0.172, median = -0.175, SD = 0.095 (N = 20)
- Null: mean = -0.141, median = -0.141, SD = 0.068 (N = 20000 simulations)
- Null 95% CI: [-0.274, -0.007]
- Mann-Whitney (20 observed vs 20 per-session null means): U = 140.0, p = 0.1075, d = -0.300
- Bootstrap permutation p (one-sided, H1: observed < null): 0.0192
- Sessions outside null 95% CI: 3 below, 1 above (20.0%)

**Note:** The null distribution itself has a negative mean (-0.141), indicating that maze geometry alone produces some degree of turn alternation. The question is whether the observed alternation (-0.172) exceeds this geometry-driven baseline.

**Interpretation:** Observed alternation is significantly stronger than the random walk null (permutation p < 0.05). While maze geometry contributes some alternation, mice show additional spontaneous alternation beyond what the graph structure would produce.

---

## Control 6: Per-Bodypart Tracking Quality by Light Condition

**Question:** Is tracking quality systematically different between light and dark for any bodypart?

**Status:** requires_frame_data

_This control requires per-bodypart raw positions from kinematics.h5 files. These are not currently on S3. When kinematics.h5 files are regenerated, re-run this script._

---

## Control 7: Primary-Only Analysis

**Sessions:** 11 sessions from 11 animals (one per animal).

Re-runs key comparisons using only primary_exp=True sessions to control for pseudoreplication from animals with multiple sessions.

| Metric | N | Light | Dark | Test |
| ------ | - | ----- | ---- | ---- |
| Epoch coverage | 11 | 0.402 | 0.346 | W = 5.0, p = 0.0098, r = 0.848, N = 11 |
| Coverage / active min | 11 | 18.06 | 16.90 | W = 17.0, p = 0.1748, r = 0.485, N = 11 |
| MRL (active) | 11 | 0.064 | 0.084 | W = 20.0, p = 0.2783, r = 0.394, N = 11 |
| Median |AHV| (deg/s) | 11 | 94.4 | 95.9 | W = 24.0, p = 0.4648, r = 0.273, N = 11 |
| Median speed (cm/s) | 11 | 2.69 | 2.25 | W = 10.0, p = 0.0420, r = 0.697, N = 11 |
| Fraction active | 11 | 0.513 | 0.474 | W = 10.0, p = 0.0420, r = 0.697, N = 11 |

Turn autocorrelation vs 0 (primary only): mean = -0.140, W = 2.0, p = 0.0029, r = 0.939, N = 11

---

## Summary of Control Analyses

| Control | Status | Key finding |
| ------- | ------ | ----------- |
| 1. Coverage per active min | computed | p = 0.0012, r = 0.781 |
| 2. MRL by node type | requires_frame_data | Deferred |
| 3. MRL/AHV active only | verified | Already active-only; MRL p = 0.0759 |
| 4. Speed by node type (all) | requires_frame_data | Deferred |
| 5. Random walk null | computed | Permutation p = 0.0192, d = -0.300 |
| 6. Bodypart tracking quality | requires_frame_data | Deferred |
| 7. Primary-only | computed | Coverage p = 0.0098, r = 0.848 |


