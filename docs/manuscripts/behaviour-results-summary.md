# Behavioural Analysis Results — Summary

Generated from 23 usable sessions (15 animals: 11 Penk+, 4 Penk-CamKII+).

All tests are non-parametric. Effect sizes: rank-biserial r (Wilcoxon), Cliff's delta (Mann-Whitney). Multiple comparisons: Holm-Bonferroni within each figure.

---

## Session Summary

| Metric | Mean +/- SD | Median | Range |
| ------ | ----------- | ------ | ----- |
| Total distance (m) | 70.1 +/- 40.7 | 67.1 | 26.4 - 217.1 |
| Duration (s) | 1912 +/- 394 | 1843 | |
| Usable duration (s) | 1737 +/- 531 | | |
| Mean speed (cm/s) | 4.58 +/- 2.55 | | |
| Fraction active | 0.460 +/- 0.144 | | |
| Cells visited | 21.7 +/- 2.1 | 22 | |
| Coverage fraction | 0.945 +/- 0.093 | | |

---

## Figure 2: Speed and Locomotion — Light vs Dark

| Comparison | Light | Dark | Test |
| ---------- | ----- | ---- | ---- |
| Median speed (cm/s) | 2.22 | 2.09 | W = 76.0, p = 0.0605, p_adj = 0.1125, r = 0.449, N = 23 |
| Fraction active | 0.476 | 0.444 | W = 75.0, p = 0.0563, p_adj = 0.1125, r = 0.457, N = 23 |
| Median immobility bout (s) | 0.72 | 0.83 | W = 9.0, p = 0.0013, p_adj = 0.0053, r = 0.882, N = 23 |
| Distance per epoch (m, body) | 2.02 | 2.11 | W = 57.0, p = 0.0123, p_adj = 0.0369, r = 0.587, N = 23 |

---

## Figure 3: Maze Exploration — Light vs Dark (PRIORITY)

| Comparison | Light | Dark | Test |
| ---------- | ----- | ---- | ---- |
| Per-epoch coverage | 0.410 | 0.354 | W = 32.0, p = 0.0006, p_adj = 0.0051, r = 0.768, N = 23 |
| Dead-end rate (visits/min) | 7.58 | 8.13 | W = 107.0, p = 0.3604, p_adj = 0.7208, r = 0.225, N = 23 |
| Exploration efficiency (w=5) | 3.56 | 3.48 | W = 83.0, p = 0.0980, p_adj = 0.3920, r = 0.399, N = 23 |
| New cells per metre (body) | 4.540 | 4.336 | W = 94.0, p = 0.1895, p_adj = 0.5686, r = 0.319, N = 23 |
| Revisitation (entries/cell) | 3.410 | 3.910 | W = 78.0, p = 0.0698, p_adj = 0.3491, r = 0.435, N = 23 |
| Occupancy entropy (bits) | 2.226 | 1.951 | W = 36.0, p = 0.0011, p_adj = 0.0078, r = 0.739, N = 23 |
| Normalised LZ complexity | 0.739 | 0.750 | W = 115.0, p = 0.5009, p_adj = 0.7208, r = 0.167, N = 23 |
| Coverage z vs random-walk null | 0.757 | 0.438 | W = 75.0, p = 0.0563, p_adj = 0.3375, r = 0.457, N = 23 |

---

## Figure 4: Turn Behaviour — Light vs Dark (PRIORITY)

| Comparison | Light | Dark | Test |
| ---------- | ----- | ---- | ---- |
| Left fraction | 0.499 | 0.493 | W = 127.0, p = 0.7540, p_adj = 1.0000, r = 0.080, N = 23 |
| Back-tracking rate | 0.468 | 0.481 | W = 122.0, p = 0.6434, p_adj = 1.0000, r = 0.116, N = 23 |

### Sequential turn autocorrelation

- **Overall autocorrelation vs 0:** mean = -0.166, W = 0.0, p = 0.0000, p_adj = 0.0000, r = 1.000, N = 23
- **Light vs dark autocorrelation:** light mean = -0.187, dark mean = -0.159, W = 124.0, p = 0.6869, p_adj = 1.0000, r = 0.101, N = 23

### Per-junction turn bias (pooled across sessions)

| Junction | Left | Right | Total | Left frac | Binomial p | p_adj |
| -------- | ---- | ----- | ----- | --------- | ---------- | ----- |
| (1, 0) | 281 | 279 | 560 | 0.502 | 0.9663 | 1.0000 |
| (1, 2) | 271 | 248 | 519 | 0.522 | 0.3342 | 1.0000 |
| (1, 4) | 240 | 224 | 464 | 0.517 | 0.4862 | 1.0000 |
| (3, 2) | 199 | 230 | 429 | 0.464 | 0.1474 | 0.8845 |
| (5, 0) | 292 | 243 | 535 | 0.546 | 0.0379 | 0.2651 |
| (5, 2) | 250 | 243 | 493 | 0.507 | 0.7870 | 1.0000 |
| (5, 4) | 200 | 204 | 404 | 0.495 | 0.8814 | 1.0000 |

---

## Figure 5: Head Direction and AHV

| Comparison | Light | Dark | Test |
| ---------- | ----- | ---- | ---- |
| HD mean resultant length | 0.068 | 0.093 | W = 88.0, p = 0.1341, p_adj = 0.2683, r = 0.362, N = 23 |
| Median |AHV| (deg/s) | 63.3 | 63.6 | W = 130.0, p = 0.8229, p_adj = 0.8229, r = 0.058, N = 23 |

---

## Figure 6: Speed at Maze Locations

- **Mean speed (cm/s):** Junction = 8.04, Corridor = 9.63, Dead end = 9.32
- **Friedman test:** W = 9.5, p = 0.0087, N = 23
- **Post-hoc (Holm-Bonferroni):** J vs C p_adj = 0.0007603168487548828, J vs DE p_adj = 0.8462316989898682, C vs DE p_adj = 0.22285175323486328
- **Junction approach:** pre = 7.82, at = 8.45 cm/s, W = 87.0, p = 0.1262, r = 0.370, N = 23

---

## Supplementary S1: Markov Models

- **Transition entropy:** Light = 1.315, Dark = 1.258, W = 54.0, p = 0.0091, r = 0.609, N = 23
- **Markov order:** mean delta_BIC = -5651.5, 0/23 sessions prefer 2nd order, W = 0.0, p = 1.0000, r = 1.000, N = 23

---

## Robustness: Primary-Only Sessions

| Comparison | N | p | r |
| ---------- | - | - | - |
| Speed L vs D | 12 | 0.1514 | 0.487 |
| Frac active L vs D | 12 | 0.1099 | 0.538 |
| Epoch coverage L vs D | 12 | 0.0425 | 0.667 |

---

## Per-Session Data

| Exp | Animal | Type | Excl | Dur(s) | Dist(m) | Speed L | Speed D | Frac Act L | Frac Act D | Cells | Cov |
| --- | ------ | ---- | ---- | ------ | ------- | ------- | ------- | ---------- | ---------- | ----- | --- |
| 1 | 1114353 | penk |  | 1866 | 60.3 | 4.31 | 5.06 | 0.665 | 0.725 | 23 | 1.00 |
| 2 | 1114356 | penk |  | 1843 | 67.1 | 1.81 | 2.09 | 0.402 | 0.444 | 22 | 0.96 |
| 3 | 1114356 | penk |  | 1860 | 38.2 | 0.51 | 0.52 | 0.269 | 0.253 | 22 | 0.96 |
| 4 | 1114356 | penk |  | 1860 | 34.9 | 0.13 | 0.44 | 0.217 | 0.276 | 20 | 0.87 |
| 5 | 1114356 | penk | Y | 1843 | 22.3 | 0.10 | 0.16 | 0.151 | 0.162 | 20 | 0.87 |
| 6 | 1115465 | penk |  | 1843 | 32.2 | 1.83 | 0.70 | 0.408 | 0.301 | 22 | 0.96 |
| 7 | 1115465 | penk |  | 1843 | 52.0 | 1.04 | 1.81 | 0.325 | 0.413 | 21 | 0.91 |
| 8 | 1115465 | penk |  | 1843 | 26.4 | 0.35 | 0.02 | 0.284 | 0.101 | 23 | 1.00 |
| 9 | 1115464 | penk |  | 1843 | 38.6 | 0.78 | 1.01 | 0.261 | 0.289 | 23 | 1.00 |
| 10 | 1115816 | penk |  | 3686 | 80.0 | 1.65 | 0.44 | 0.378 | 0.236 | 22 | 0.96 |
| 11 | 1116663 | penk |  | 1843 | 82.7 | 2.14 | 1.96 | 0.465 | 0.437 | 23 | 1.00 |
| 12 | 1116663 | penk |  | 1843 | 100.2 | 2.55 | 2.73 | 0.507 | 0.523 | 23 | 1.00 |
| 13 | 1117217 | nonpenk |  | 1843 | 110.2 | 3.41 | 2.91 | 0.604 | 0.548 | 23 | 1.00 |
| 14 | 1117217 | nonpenk |  | 1866 | 77.7 | 3.03 | 2.68 | 0.562 | 0.524 | 23 | 1.00 |
| 15 | 1117217 | nonpenk |  | 1843 | 76.9 | 3.06 | 2.09 | 0.557 | 0.459 | 23 | 1.00 |
| 16 | 1116994 | penk |  | 1843 | 29.9 | 2.18 | 1.50 | 0.459 | 0.396 | 23 | 1.00 |
| 17 | 1117646 | nonpenk |  | 1843 | 35.4 | 2.03 | 1.21 | 0.442 | 0.343 | 23 | 1.00 |
| 18 | 1117646 | nonpenk |  | 1843 | 217.1 | 5.21 | 4.03 | 0.693 | 0.633 | 23 | 1.00 |
| 19 | 1117646 | nonpenk | Y | 1843 | 161.8 | 0.56 | 1.41 | 0.292 | 0.375 | 23 | 1.00 |
| 20 | 1118020 | penk |  | 1843 | 94.8 | 3.85 | 3.62 | 0.641 | 0.610 | 22 | 0.96 |
| 21 | 1118023 | penk |  | 1843 | 90.4 | 3.46 | 3.74 | 0.610 | 0.645 | 15 | 0.65 |
| 22 | 1118018 | penk |  | 1843 | 86.6 | 3.53 | 3.27 | 0.608 | 0.576 | 16 | 0.70 |
| 23 | 1117788 | nonpenk |  | 1493 | 49.3 | 2.90 | 2.78 | 0.552 | 0.538 | 21 | 0.91 |
| 24 | 1118213 | nonpenk |  | 1843 | 79.0 | 3.10 | 2.96 | 0.573 | 0.555 | 22 | 0.96 |
| 25 | 1118320 | penk |  | 1843 | 53.3 | 2.22 | 1.90 | 0.460 | 0.386 | 22 | 0.96 |
| 26 | 1118317 | penk | Y | 1843 | 63.0 | 2.44 | 2.29 | 0.492 | 0.472 | 9 | 0.39 |
