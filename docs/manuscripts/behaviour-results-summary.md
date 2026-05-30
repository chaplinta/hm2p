# Behavioural Analysis Results — Summary

Generated from 20 usable sessions (14 animals: 10 Penk+, 4 Penk-CamKII+).

All tests are non-parametric. Effect sizes: rank-biserial r (Wilcoxon), Cliff's delta (Mann-Whitney). Multiple comparisons: Holm-Bonferroni within each figure.

---

## Session Summary

| Metric | Mean +/- SD | Median | Range |
| ------ | ----------- | ------ | ----- |
| Total distance (m) | 62.2 +/- 25.3 | 57.7 | 31.7 - 117.8 |
| Duration (s) | 1921 +/- 423 | 1843 | |
| Usable duration (s) | 1720 +/- 569 | | |
| Mean speed (cm/s) | 4.20 +/- 1.48 | | |
| Fraction active | 0.456 +/- 0.142 | | |
| Cells visited | 22.1 +/- 1.3 | 22 | |
| Coverage fraction | 0.961 +/- 0.056 | | |

---

## Figure 2: Speed and Locomotion — Light vs Dark

| Comparison | Light | Dark | Test |
| ---------- | ----- | ---- | ---- |
| Median speed (cm/s) | 2.25 | 1.99 | W = 57.0, p = 0.0759, p_adj = 0.1517, r = 0.457, N = 20 |
| Fraction active | 0.472 | 0.441 | W = 58.0, p = 0.0826, p_adj = 0.1517, r = 0.448, N = 20 |
| Median immobility bout (s) | 0.77 | 0.92 | W = 19.0, p = 0.0352, p_adj = 0.1056, r = 0.638, N = 20 |

---

## Figure 3: Maze Exploration — Light vs Dark (PRIORITY)

| Comparison | Light | Dark | Test |
| ---------- | ----- | ---- | ---- |
| Per-epoch coverage | 0.400 | 0.337 | W = 15.0, p = 0.0003, p_adj = 0.0008, r = 0.857, N = 20 |
| Dead-end rate (visits/min) | 7.60 | 8.33 | W = 74.0, p = 0.2611, p_adj = 0.2611, r = 0.295, N = 20 |
| Exploration efficiency (w=5) | 3.56 | 3.41 | W = 54.0, p = 0.0583, p_adj = 0.1165, r = 0.486, N = 20 |

---

## Figure 4: Turn Behaviour — Light vs Dark (PRIORITY)

| Comparison | Light | Dark | Test |
| ---------- | ----- | ---- | ---- |
| Left fraction | 0.495 | 0.500 | W = 100.0, p = 0.8695, p_adj = 1.0000, r = 0.048, N = 20 |
| Back-tracking rate | 0.482 | 0.505 | W = 92.0, p = 0.6477, p_adj = 1.0000, r = 0.124, N = 20 |

### Sequential turn autocorrelation

- **Overall autocorrelation vs 0:** mean = -0.172, W = 2.0, p = 0.0000, p_adj = 0.0000, r = 0.981, N = 20
- **Light vs dark autocorrelation:** light mean = -0.158, dark mean = -0.170, W = 92.0, p = 0.6477, p_adj = 1.0000, r = 0.124, N = 20

### Per-junction turn bias (pooled across sessions)

| Junction | Left | Right | Total | Left frac | Binomial p | p_adj |
| -------- | ---- | ----- | ----- | --------- | ---------- | ----- |
| (1, 0) | 249 | 249 | 498 | 0.500 | 1.0000 | 1.0000 |
| (1, 2) | 205 | 191 | 396 | 0.518 | 0.5136 | 1.0000 |
| (1, 4) | 201 | 181 | 382 | 0.526 | 0.3310 | 1.0000 |
| (3, 2) | 187 | 219 | 406 | 0.461 | 0.1238 | 0.7429 |
| (5, 0) | 230 | 194 | 424 | 0.542 | 0.0891 | 0.6234 |
| (5, 2) | 188 | 177 | 365 | 0.515 | 0.6007 | 1.0000 |
| (5, 4) | 175 | 176 | 351 | 0.499 | 1.0000 | 1.0000 |

---

## Figure 5: Head Direction and AHV

| Comparison | Light | Dark | Test |
| ---------- | ----- | ---- | ---- |
| HD mean resultant length | 0.060 | 0.085 | W = 57.0, p = 0.0759, p_adj = 0.1517, r = 0.457, N = 20 |
| Median |AHV| (deg/s) | 93.3 | 95.4 | W = 68.0, p = 0.1769, p_adj = 0.1769, r = 0.352, N = 20 |

---

## Figure 6: Speed at Maze Locations

- **Mean speed (cm/s):** Junction = 8.21, Corridor = 9.00, Dead end = 8.13
- **Friedman test:** W = 15.7, p = 0.0004, N = 20
- **Post-hoc (Holm-Bonferroni):** J vs C p_adj = 0.002552032470703125, J vs DE p_adj = 0.13272666931152344, C vs DE p_adj = 0.004650115966796875
- **Junction approach:** pre = 7.09, at = 7.23 cm/s, W = 79.0, p = 0.3488, r = 0.248, N = 20

---

## Supplementary S1: Markov Models

- **Transition entropy:** Light = 1.221, Dark = 1.186, W = 67.0, p = 0.1650, r = 0.362, N = 20
- **Markov order:** mean delta_BIC = -4433.9, 0/20 sessions prefer 2nd order, W = 0.0, p = 1.0000, r = 1.000, N = 20

---

## Robustness: Primary-Only Sessions

| Comparison | N | p | r |
| ---------- | - | - | - |
| Speed L vs D | 11 | 0.0420 | 0.697 |
| Frac active L vs D | 11 | 0.0420 | 0.697 |
| Epoch coverage L vs D | 11 | 0.0098 | 0.848 |

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
| 13 | 1117217 | nonpenk | Y | 1843 | 86.8 | 3.37 | 3.02 | 0.598 | 0.558 | 23 | 1.00 |
| 14 | 1117217 | nonpenk | Y | 1866 | 76.0 | 3.19 | 2.83 | 0.576 | 0.532 | 23 | 1.00 |
| 15 | 1117217 | nonpenk |  | 1843 | 70.5 | 3.17 | 2.13 | 0.559 | 0.469 | 23 | 1.00 |
| 16 | 1116994 | penk |  | 1843 | 32.1 | 2.25 | 1.24 | 0.468 | 0.398 | 23 | 1.00 |
| 17 | 1117646 | nonpenk |  | 1843 | 37.0 | 1.98 | 1.00 | 0.433 | 0.356 | 23 | 1.00 |
| 18 | 1117646 | nonpenk |  | 1843 | 117.8 | 4.88 | 3.92 | 0.671 | 0.629 | 23 | 1.00 |
| 19 | 1117646 | nonpenk | Y | 1843 | 132.0 | 0.06 | 1.01 | 0.253 | 0.340 | 23 | 1.00 |
| 20 | 1118020 | penk |  | 1843 | 100.8 | 4.06 | 3.90 | 0.652 | 0.627 | 22 | 0.96 |
| 21 | 1118023 | penk | Y | 1843 | 94.5 | 3.59 | 4.01 | 0.624 | 0.655 | 16 | 0.70 |
| 22 | 1118018 | penk |  | 1843 | 93.1 | 3.85 | 3.56 | 0.629 | 0.602 | 18 | 0.78 |
| 23 | 1117788 | nonpenk |  | 1493 | 54.9 | 3.17 | 3.08 | 0.582 | 0.564 | 21 | 0.91 |
| 24 | 1118213 | nonpenk |  | 1843 | 84.2 | 3.34 | 3.17 | 0.595 | 0.578 | 22 | 0.96 |
| 25 | 1118320 | penk |  | 1843 | 60.5 | 2.41 | 1.93 | 0.488 | 0.412 | 22 | 0.96 |
| 26 | 1118317 | penk | Y | 1843 | 67.8 | 2.65 | 2.29 | 0.516 | 0.473 | 9 | 0.39 |
