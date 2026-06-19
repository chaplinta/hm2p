# Behavioural Analysis Results — Summary

Generated from 23 usable sessions (15 animals: 11 Penk+, 4 Penk-CamKII+).

All tests are non-parametric. Effect sizes: rank-biserial r (Wilcoxon), Cliff's delta (Mann-Whitney). Multiple comparisons: Holm-Bonferroni within each figure.

---

## Session Summary

| Metric | Mean +/- SD | Median | Range |
| ------ | ----------- | ------ | ----- |
| Total distance (m) | 65.2 +/- 25.1 | 62.1 | 31.7 - 117.8 |
| Duration (s) | 1912 +/- 394 | 1843 | |
| Usable duration (s) | 1737 +/- 531 | | |
| Mean speed (cm/s) | 4.35 +/- 1.44 | | |
| Fraction active | 0.474 +/- 0.140 | | |
| Cells visited | 21.9 +/- 1.8 | 23 | |
| Coverage fraction | 0.953 +/- 0.077 | | |

---

## Figure 2: Speed and Locomotion — Light vs Dark

| Comparison | Light | Dark | Test |
| ---------- | ----- | ---- | ---- |
| Median speed (cm/s) | 2.41 | 2.12 | W = 78.0, p = 0.0698, p_adj = 0.1396, r = 0.435, N = 23 |
| Fraction active | 0.489 | 0.459 | W = 78.0, p = 0.0698, p_adj = 0.1396, r = 0.435, N = 23 |
| Median immobility bout (s) | 0.73 | 0.92 | W = 21.5, p = 0.0161, p_adj = 0.0643, r = 0.684, N = 23 |
| Distance per epoch (m, body) | 2.26 | 2.03 | W = 68.0, p = 0.0327, p_adj = 0.0980, r = 0.507, N = 23 |

---

## Figure 3: Maze Exploration — Light vs Dark (PRIORITY)

| Comparison | Light | Dark | Test |
| ---------- | ----- | ---- | ---- |
| Per-epoch coverage | 0.410 | 0.349 | W = 23.0, p = 0.0002, p_adj = 0.0008, r = 0.833, N = 23 |
| Dead-end rate (visits/min) | 7.56 | 8.35 | W = 87.0, p = 0.1262, p_adj = 0.1262, r = 0.370, N = 23 |
| Exploration efficiency (w=5) | 3.54 | 3.40 | W = 68.0, p = 0.0327, p_adj = 0.1017, r = 0.507, N = 23 |
| New cells per metre (body) | 4.436 | 4.148 | W = 71.0, p = 0.0415, p_adj = 0.1017, r = 0.486, N = 23 |
| Revisitation (entries/cell) | 3.450 | 3.897 | W = 65.0, p = 0.0254, p_adj = 0.1017, r = 0.529, N = 23 |

---

## Figure 4: Turn Behaviour — Light vs Dark (PRIORITY)

| Comparison | Light | Dark | Test |
| ---------- | ----- | ---- | ---- |
| Left fraction | 0.497 | 0.500 | W = 137.0, p = 0.9881, p_adj = 1.0000, r = 0.007, N = 23 |
| Back-tracking rate | 0.474 | 0.496 | W = 115.0, p = 0.5009, p_adj = 1.0000, r = 0.167, N = 23 |

### Sequential turn autocorrelation

- **Overall autocorrelation vs 0:** mean = -0.169, W = 2.0, p = 0.0000, p_adj = 0.0000, r = 0.986, N = 23
- **Light vs dark autocorrelation:** light mean = -0.163, dark mean = -0.179, W = 117.0, p = 0.5399, p_adj = 1.0000, r = 0.152, N = 23

### Per-junction turn bias (pooled across sessions)

| Junction | Left | Right | Total | Left frac | Binomial p | p_adj |
| -------- | ---- | ----- | ----- | --------- | ---------- | ----- |
| (1, 0) | 303 | 296 | 599 | 0.506 | 0.8064 | 1.0000 |
| (1, 2) | 272 | 243 | 515 | 0.528 | 0.2172 | 1.0000 |
| (1, 4) | 246 | 214 | 460 | 0.535 | 0.1483 | 0.8896 |
| (3, 2) | 202 | 228 | 430 | 0.470 | 0.2279 | 1.0000 |
| (5, 0) | 281 | 239 | 520 | 0.540 | 0.0721 | 0.5046 |
| (5, 2) | 247 | 237 | 484 | 0.510 | 0.6825 | 1.0000 |
| (5, 4) | 202 | 205 | 407 | 0.496 | 0.9210 | 1.0000 |

---

## Figure 5: Head Direction and AHV

| Comparison | Light | Dark | Test |
| ---------- | ----- | ---- | ---- |
| HD mean resultant length | 0.063 | 0.084 | W = 87.0, p = 0.1262, p_adj = 0.2524, r = 0.370, N = 23 |
| Median |AHV| (deg/s) | 94.4 | 96.2 | W = 95.0, p = 0.2002, p_adj = 0.2524, r = 0.312, N = 23 |

---

## Figure 6: Speed at Maze Locations

- **Mean speed (cm/s):** Junction = 8.18, Corridor = 9.02, Dead end = 8.15
- **Friedman test:** W = 15.9, p = 0.0004, N = 23
- **Post-hoc (Holm-Bonferroni):** J vs C p_adj = 0.0007603168487548828, J vs DE p_adj = 0.20015525817871094, C vs DE p_adj = 0.0019516944885253906
- **Junction approach:** pre = 7.19, at = 7.30 cm/s, W = 115.0, p = 0.5009, r = 0.167, N = 23

---

## Supplementary S1: Markov Models

- **Transition entropy:** Light = 1.238, Dark = 1.198, W = 76.0, p = 0.0605, r = 0.449, N = 23
- **Markov order:** mean delta_BIC = -4558.7, 0/23 sessions prefer 2nd order, W = 0.0, p = 1.0000, r = 1.000, N = 23

---

## Robustness: Primary-Only Sessions

| Comparison | N | p | r |
| ---------- | - | - | - |
| Speed L vs D | 12 | 0.1294 | 0.513 |
| Frac active L vs D | 12 | 0.1099 | 0.538 |
| Epoch coverage L vs D | 12 | 0.0210 | 0.744 |

---

## Per-Session Data

| Exp | Animal | Type | Excl | Dur(s) | Dist(m) | Speed L | Speed D | Frac Act L | Frac Act D | Cells | Cov |
| --- | ------ | ---- | ---- | ------ | ------- | ------- | ------- | ---------- | ---------- | ----- | --- |
| 1 | 1114353 | penk |  | 1866 | 62.1 | 4.66 | 5.23 | 0.675 | 0.727 | 23 | 1.00 |
| 2 | 1114356 | penk |  | 1843 | 71.0 | 1.77 | 2.12 | 0.412 | 0.453 | 21 | 0.91 |
| 3 | 1114356 | penk |  | 1860 | 40.7 | 0.14 | 0.07 | 0.278 | 0.256 | 22 | 0.96 |
| 4 | 1114356 | penk |  | 1860 | 40.4 | 0.04 | 0.14 | 0.240 | 0.291 | 20 | 0.87 |
| 5 | 1114356 | penk | Y | 1843 | 27.4 | 0.05 | 0.09 | 0.164 | 0.190 | 20 | 0.87 |
| 6 | 1115465 | penk |  | 1843 | 34.5 | 1.83 | 0.31 | 0.429 | 0.315 | 22 | 0.96 |
| 7 | 1115465 | penk |  | 1843 | 54.8 | 0.87 | 1.82 | 0.340 | 0.429 | 22 | 0.96 |
| 8 | 1115465 | penk |  | 1843 | 31.7 | 0.05 | 0.01 | 0.302 | 0.131 | 23 | 1.00 |
| 9 | 1115464 | penk |  | 1843 | 44.6 | 0.52 | 0.70 | 0.286 | 0.310 | 23 | 1.00 |
| 10 | 1115816 | penk |  | 3686 | 91.9 | 1.70 | 0.15 | 0.407 | 0.265 | 23 | 1.00 |
| 11 | 1116663 | penk |  | 1843 | 44.2 | 2.24 | 2.04 | 0.479 | 0.463 | 23 | 1.00 |
| 12 | 1116663 | penk |  | 1843 | 76.5 | 2.71 | 2.91 | 0.520 | 0.539 | 23 | 1.00 |
| 13 | 1117217 | nonpenk |  | 1843 | 86.8 | 3.37 | 3.02 | 0.598 | 0.558 | 23 | 1.00 |
| 14 | 1117217 | nonpenk |  | 1866 | 76.0 | 3.19 | 2.83 | 0.576 | 0.532 | 23 | 1.00 |
| 15 | 1117217 | nonpenk |  | 1843 | 70.5 | 3.17 | 2.13 | 0.559 | 0.469 | 23 | 1.00 |
| 16 | 1116994 | penk |  | 1843 | 32.1 | 2.25 | 1.24 | 0.468 | 0.398 | 23 | 1.00 |
| 17 | 1117646 | nonpenk |  | 1843 | 37.0 | 1.98 | 1.00 | 0.433 | 0.356 | 23 | 1.00 |
| 18 | 1117646 | nonpenk |  | 1843 | 117.8 | 4.88 | 3.92 | 0.671 | 0.629 | 23 | 1.00 |
| 19 | 1117646 | nonpenk | Y | 1843 | 132.0 | 0.06 | 1.01 | 0.253 | 0.340 | 23 | 1.00 |
| 20 | 1118020 | penk |  | 1843 | 100.8 | 4.06 | 3.90 | 0.652 | 0.627 | 22 | 0.96 |
| 21 | 1118023 | penk |  | 1843 | 94.5 | 3.59 | 4.01 | 0.624 | 0.655 | 16 | 0.70 |
| 22 | 1118018 | penk |  | 1843 | 93.1 | 3.85 | 3.56 | 0.629 | 0.602 | 18 | 0.78 |
| 23 | 1117788 | nonpenk |  | 1493 | 54.9 | 3.17 | 3.08 | 0.582 | 0.564 | 21 | 0.91 |
| 24 | 1118213 | nonpenk |  | 1843 | 84.2 | 3.34 | 3.17 | 0.595 | 0.578 | 22 | 0.96 |
| 25 | 1118320 | penk |  | 1843 | 60.5 | 2.41 | 1.93 | 0.488 | 0.412 | 22 | 0.96 |
| 26 | 1118317 | penk | Y | 1843 | 67.8 | 2.65 | 2.29 | 0.516 | 0.473 | 9 | 0.39 |
