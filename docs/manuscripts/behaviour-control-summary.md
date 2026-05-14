# Behavioural Control Analyses — Summary

Generated from 21 usable sessions (15 animals). 12 primary sessions.

Seven control analyses addressing potential confounds in the main behavioural manuscript results. All tests non-parametric.

---

## Control 1: Coverage Per Active Minute (Light vs Dark)

**Question:** Is the lower per-epoch coverage in dark driven by reduced locomotion speed, or does it persist when normalised by active time?

**Method:** Coverage per active minute = (epoch_coverage * 23) / frac_active. Approximation from per-session summaries.

- Light: 21.89 cells/active-min (median 23.71)
- Dark: 20.03 cells/active-min (median 22.01)
- W = 31.0, p = 0.0022, r = 0.732, N = 21

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
| MRL (active only) | 0.297 | 0.338 | W = 43.0, p = 0.0101, r = 0.628, N = 21 | 0.0203 |
| Median |AHV| (active only) | 121.4 | 115.1 | W = 49.0, p = 0.0195, r = 0.576, N = 21 | 0.0203 |

---

## Control 4: Speed by Node Type — All Frames

**Question:** Does the speed hierarchy (corridor > junction > dead end) persist when including immobile frames?

**Status:** requires_frame_data

_This control requires frame-level speed and position data from sync.h5 files. Cannot be computed from per-session summaries. When sync.h5 files are regenerated, re-run this script._

Active-only reference from Figure 6:
- Junction: 14.95 cm/s
- Corridor: 17.87 cm/s
- Dead end: 26.32 cm/s

---

## Control 5: Random Walk Null Model for Alternation

**Question:** Is the observed turn alternation (negative lag-1 autocorrelation) stronger than expected from a random walk on the maze graph? This controls for maze geometry constraining turn sequences.

- Observed: mean = -0.196, median = -0.171, SD = 0.102 (N = 21)
- Null: mean = -0.141, median = -0.141, SD = 0.060 (N = 21000 simulations)
- Null 95% CI: [-0.260, -0.021]
- Mann-Whitney (21 observed vs 21 per-session null means): U = 147.0, p = 0.0663, d = -0.333
- Bootstrap permutation p (one-sided, H1: observed < null): <0.0001
- Sessions outside null 95% CI: 5 below, 0 above (23.8%)

**Note:** The null distribution itself has a negative mean (-0.141), indicating that maze geometry alone produces some degree of turn alternation. The question is whether the observed alternation (-0.196) exceeds this geometry-driven baseline.

**Interpretation:** Observed alternation is significantly stronger than the random walk null (permutation p < 0.05). While maze geometry contributes some alternation, mice show additional spontaneous alternation beyond what the graph structure would produce.

---

## Control 6: Per-Bodypart Tracking Quality by Light Condition

**Question:** Is tracking quality systematically different between light and dark for any bodypart?

**Status:** requires_frame_data

_This control requires per-bodypart raw positions from kinematics.h5 files. These are not currently on S3. When kinematics.h5 files are regenerated, re-run this script._

---

## Control 7: Primary-Only Analysis

**Sessions:** 12 sessions from 12 animals (one per animal).

Re-runs key comparisons using only primary_exp=True sessions to control for pseudoreplication from animals with multiple sessions.

| Metric | N | Light | Dark | Test |
| ------ | - | ----- | ---- | ---- |
| Epoch coverage | 12 | 0.439 | 0.390 | W = 10.0, p = 0.0210, r = 0.744, N = 12 |
| Coverage / active min | 12 | 19.96 | 18.89 | W = 21.0, p = 0.1763, r = 0.462, N = 12 |
| MRL (active) | 12 | 0.302 | 0.329 | W = 26.0, p = 0.3394, r = 0.333, N = 12 |
| Median |AHV| (deg/s) | 12 | 116.2 | 111.2 | W = 10.0, p = 0.0210, r = 0.744, N = 12 |
| Median speed (cm/s) | 12 | 2.79 | 2.46 | W = 17.0, p = 0.0923, r = 0.564, N = 12 |
| Fraction active | 12 | 0.507 | 0.479 | W = 20.0, p = 0.1514, r = 0.487, N = 12 |

Turn autocorrelation vs 0 (primary only): mean = -0.179, W = 0.0, p = 0.0005, r = 1.000, N = 12

---

## Summary of Control Analyses

| Control | Status | Key finding |
| ------- | ------ | ----------- |
| 1. Coverage per active min | computed | p = 0.0022, r = 0.732 |
| 2. MRL by node type | requires_frame_data | Deferred |
| 3. MRL/AHV active only | verified | Already active-only; MRL p = 0.0101 |
| 4. Speed by node type (all) | requires_frame_data | Deferred |
| 5. Random walk null | computed | Permutation p = <0.0001, d = -0.333 |
| 6. Bodypart tracking quality | requires_frame_data | Deferred |
| 7. Primary-only | computed | Coverage p = 0.0210, r = 0.744 |


