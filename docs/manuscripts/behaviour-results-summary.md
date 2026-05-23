# Behavioural Analysis Results — Summary

Generated from 21 usable sessions (15 animals: 11 Penk+, 4 Penk-CamKII+).

All tests are non-parametric. Effect sizes: rank-biserial r (Wilcoxon), Cliff's delta (Mann-Whitney). Multiple comparisons: Holm-Bonferroni within each figure.

---

## Session Summary

| Metric | Mean +/- SD | Median | Range |
| ------ | ----------- | ------ | ----- |
| Total distance (m) | 129.3 +/- 110.8 | 106.2 | 40.6 - 500.7 |
| Duration (s) | 1917 +/- 413 | 1843 | |
| Usable duration (s) | 1726 +/- 555 | | |
| Mean speed (cm/s) | 8.84 +/- 6.49 | | |
| Fraction active | 0.455 +/- 0.146 | | |
| Cells visited | 22.2 +/- 1.6 | 23 | |
| Coverage fraction | 0.965 +/- 0.070 | | |

---

## Figure 2: Speed and Locomotion — Light vs Dark

| Comparison | Light | Dark | Test |
| ---------- | ----- | ---- | ---- |
| Median speed (cm/s) | 1.89 | 1.86 | W = 70.0, p = 0.1193, p_adj = 0.3580, r = 0.394, N = 21 |
| Fraction active | 0.466 | 0.443 | W = 79.0, p = 0.2157, p_adj = 0.4314, r = 0.316, N = 21 |
| Median immobility bout (s) | 0.82 | 0.87 | W = 41.0, p = 0.2792, p_adj = 0.4314, r = 0.317, N = 21 |

---

## Figure 3: Maze Exploration — Light vs Dark (PRIORITY)

| Comparison | Light | Dark | Test |
| ---------- | ----- | ---- | ---- |
| Per-epoch coverage | 0.438 | 0.381 | W = 33.0, p = 0.0029, p_adj = 0.0086, r = 0.714, N = 21 |
| Dead-end rate (visits/min) | 14.65 | 14.65 | W = 111.0, p = 0.8917, p_adj = 1.0000, r = 0.039, N = 21 |
| Exploration efficiency (w=5) | 3.39 | 3.36 | W = 107.0, p = 0.7854, p_adj = 1.0000, r = 0.074, N = 21 |

---

## Figure 4: Turn Behaviour — Light vs Dark (PRIORITY)

| Comparison | Light | Dark | Test |
| ---------- | ----- | ---- | ---- |
| Left fraction | 0.487 | 0.489 | W = 111.0, p = 0.8917, p_adj = 1.0000, r = 0.039, N = 21 |
| Back-tracking rate | 0.572 | 0.580 | W = 110.0, p = 0.8649, p_adj = 1.0000, r = 0.048, N = 21 |

### Sequential turn autocorrelation

- **Overall autocorrelation vs 0:** mean = -0.196, W = 0.0, p = 0.0000, p_adj = 0.0000, r = 1.000, N = 21
- **Light vs dark autocorrelation:** light mean = -0.227, dark mean = -0.175, W = 84.0, p = 0.2877, p_adj = 0.8632, r = 0.273, N = 21

### Per-junction turn bias (pooled across sessions)

| Junction | Left | Right | Total | Left frac | Binomial p | p_adj |
| -------- | ---- | ----- | ----- | --------- | ---------- | ----- |
| (1, 0) | 282 | 291 | 573 | 0.492 | 0.7383 | 1.0000 |
| (1, 2) | 245 | 260 | 505 | 0.485 | 0.5333 | 1.0000 |
| (1, 4) | 241 | 230 | 471 | 0.512 | 0.6450 | 1.0000 |
| (3, 2) | 244 | 267 | 511 | 0.477 | 0.3304 | 1.0000 |
| (5, 0) | 257 | 224 | 481 | 0.534 | 0.1445 | 1.0000 |
| (5, 2) | 210 | 228 | 438 | 0.479 | 0.4167 | 1.0000 |
| (5, 4) | 170 | 186 | 356 | 0.478 | 0.4267 | 1.0000 |

---

## Figure 5: Head Direction and AHV

| Comparison | Light | Dark | Test |
| ---------- | ----- | ---- | ---- |
| HD mean resultant length | 0.297 | 0.338 | W = 43.0, p = 0.0101, p_adj = 0.0203, r = 0.628, N = 21 |
| Median |AHV| (deg/s) | 121.4 | 115.1 | W = 49.0, p = 0.0195, p_adj = 0.0203, r = 0.576, N = 21 |

---

## Figure 6: Speed at Maze Locations

- **Mean speed (cm/s):** Junction = 14.95, Corridor = 17.87, Dead end = 26.32
- **Friedman test:** W = 15.0, p = 0.0006, N = 21
- **Post-hoc (Holm-Bonferroni):** J vs C p_adj = 0.00015735626220703125, J vs DE p_adj = 0.0003948211669921875, C vs DE p_adj = 0.0032787322998046875
- **Junction approach:** pre = 14.43, at = 21.64 cm/s, W = 7.0, p = 0.0000, r = 0.939, N = 21

---

## Supplementary S1: Markov Models

- **Transition entropy:** Light = 1.644, Dark = 1.631, W = 100.0, p = 0.6091, r = 0.134, N = 21
- **Markov order:** mean delta_BIC = -13504.4, 0/21 sessions prefer 2nd order, W = 0.0, p = 1.0000, r = 1.000, N = 21

---

## Robustness: Primary-Only Sessions

| Comparison | N | p | r |
| ---------- | - | - | - |
| Speed L vs D | 12 | 0.0923 | 0.564 |
| Frac active L vs D | 12 | 0.1514 | 0.487 |
| Epoch coverage L vs D | 12 | 0.0210 | 0.744 |

---

## Per-Session Data

| Exp | Animal | Type | Excl | Dur(s) | Dist(m) | Speed L | Speed D | Frac Act L | Frac Act D | Cells | Cov |
| --- | ------ | ---- | ---- | ------ | ------- | ------- | ------- | ---------- | ---------- | ----- | --- |
| 1 | 1114353 | penk |  | 1866 | 73.8 | 4.62 | 5.40 | 0.653 | 0.699 | 23 | 1.00 |
| 2 | 1114356 | penk |  | 1843 | 110.9 | 1.65 | 2.36 | 0.427 | 0.487 | 22 | 0.96 |
| 3 | 1114356 | penk |  | 1860 | 55.7 | 0.33 | 0.27 | 0.277 | 0.256 | 23 | 1.00 |
| 4 | 1114356 | penk |  | 1860 | 57.3 | 0.14 | 0.48 | 0.232 | 0.307 | 20 | 0.87 |
| 5 | 1114356 | penk | Y | 1843 | 32.7 | 0.10 | 0.11 | 0.165 | 0.181 | 20 | 0.87 |
| 6 | 1115465 | penk |  | 1843 | 44.3 | 1.52 | 0.55 | 0.398 | 0.303 | 22 | 0.96 |
| 7 | 1115465 | penk |  | 1843 | 120.9 | 0.64 | 1.27 | 0.315 | 0.397 | 23 | 1.00 |
| 8 | 1115465 | penk |  | 1843 | 40.6 | 0.32 | 0.07 | 0.288 | 0.125 | 23 | 1.00 |
| 9 | 1115464 | penk |  | 1843 | 57.1 | 0.53 | 0.59 | 0.267 | 0.292 | 23 | 1.00 |
| 10 | 1115816 | penk |  | 3686 | 330.0 | 1.35 | 0.41 | 0.382 | 0.256 | 23 | 1.00 |
| 11 | 1116663 | penk |  | 1843 | 53.2 | 1.89 | 1.86 | 0.444 | 0.440 | 23 | 1.00 |
| 12 | 1116663 | penk |  | 1843 | 115.7 | 2.41 | 2.61 | 0.485 | 0.506 | 23 | 1.00 |
| 13 | 1117217 | nonpenk | Y | - | - | - | - | - | - | - | - |
| 14 | 1117217 | nonpenk | Y | 1866 | 253.3 | 3.62 | 3.05 | 0.588 | 0.543 | 23 | 1.00 |
| 15 | 1117217 | nonpenk |  | 1843 | 174.4 | 3.28 | 2.42 | 0.551 | 0.491 | 23 | 1.00 |
| 16 | 1116994 | penk |  | 1843 | 106.2 | 1.85 | 1.28 | 0.442 | 0.404 | 23 | 1.00 |
| 17 | 1117646 | nonpenk |  | 1843 | 76.2 | 1.86 | 0.65 | 0.440 | 0.327 | 23 | 1.00 |
| 18 | 1117646 | nonpenk |  | 1843 | 500.7 | 6.64 | 4.98 | 0.694 | 0.645 | 23 | 1.00 |
| 19 | 1117646 | nonpenk | Y | 1843 | 287.0 | 0.39 | 0.74 | 0.230 | 0.321 | 23 | 1.00 |
| 20 | 1118020 | penk |  | 1843 | 130.6 | 4.64 | 3.97 | 0.647 | 0.610 | 22 | 0.96 |
| 21 | 1118023 | penk |  | 1843 | 121.4 | 3.58 | 4.04 | 0.593 | 0.636 | 17 | 0.74 |
| 22 | 1118018 | penk |  | 1843 | 109.4 | 4.03 | 3.50 | 0.615 | 0.581 | 19 | 0.83 |
| 23 | 1117788 | nonpenk |  | 1493 | 91.1 | 3.17 | 3.11 | 0.560 | 0.564 | 22 | 0.96 |
| 24 | 1118213 | nonpenk |  | 1843 | 262.2 | 4.36 | 4.04 | 0.628 | 0.608 | 23 | 1.00 |
| 25 | 1118320 | penk |  | 1843 | 84.1 | 2.15 | 1.56 | 0.456 | 0.379 | 23 | 1.00 |
| 26 | 1118317 | penk | Y | 1843 | 209.0 | 2.95 | 2.23 | 0.532 | 0.476 | 17 | 0.74 |
