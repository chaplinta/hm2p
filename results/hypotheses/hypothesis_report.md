# Hypothesis Test Report

**Signal:** dff
**Cells:** 347 soma ROIs from 15 animals (21 sessions)
**Groups:** 11 Penk+ animals, 4 Penk⁻CamKII+ animals

**Significant results (p < 0.05):** 3

## H1.1: Movement increases activity

- **Wilcoxon signed-rank:** p = 0.0000 **\*** (mean diff=0.0115, N=347 cells)

## H1.2: Light increases activity

- **Wilcoxon signed-rank:** p = 0.0000 **\*** (mean diff=0.0090, N=347 cells)

## H1.3: Movement x light interaction

- **Wilcoxon signed-rank:** p = 0.4635 (mean diff=0.0009, N=347 cells)

## H1.4: Baseline activity differs

- **Animal-level Mann-Whitney U:** p = 0.7531 (Penk+ mean=0.0401, Penk⁻CamKII+ mean=0.0403, N=11 vs 4 animals, CLES=0.43)
- **Cluster permutation:** p = 0.9780 (observed diff=0.0003, null std=0.0075)

## H1.5: Movement modulation differs

- **Animal-level Mann-Whitney U:** p = 0.7531 (Penk+ mean=0.0830, Penk⁻CamKII+ mean=0.0923, N=11 vs 4 animals, CLES=0.57)
- **Cluster permutation:** p = 0.5509 (observed diff=0.0790, null std=0.1145)

## H1.6: Movement x light interaction differs

- **Animal-level Mann-Whitney U:** p = 0.7531 (Penk+ mean=0.0024, Penk⁻CamKII+ mean=-0.0003, N=11 vs 4 animals, CLES=0.57)
- **Cluster permutation:** p = 0.9681 (observed diff=-0.0002, null std=0.0059)

## H2.1: RSP has HD cells

- **Descriptive:** 8/347 (2.3%) cells significant

## H2.2: HD strength differs

- **Animal-level Mann-Whitney U:** p = 0.1377 (Penk+ mean=0.4905, Penk⁻CamKII+ mean=1.4726, N=11 vs 4 animals, CLES=0.23)
- **Cluster permutation:** p = 0.0040 **\*** (observed diff=-1.0763, null std=0.3906)

## H2.3: Tuning width differs

- **Animal-level Mann-Whitney U:** p = 0.0777 (Penk+ mean=18.3029, Penk⁻CamKII+ mean=25.3303, N=11 vs 4 animals, CLES=0.18)
- **Cluster permutation:** p = 0.2295 (observed diff=-5.9619, null std=4.6567)

## H3.1: Darkness degrades tuning

- **Wilcoxon signed-rank:** p = 0.1314 (mean diff=0.1309, N=347 cells)

## H3.2: PD drifts in darkness

- **Wilcoxon (vs 0):** p = 0.2003 (mean=7.3960, median=8.8482, N=347 cells)

## H3.4: Visual cue dependence differs (KEY)

- **Animal-level Mann-Whitney U:** p = 0.7531 (Penk+ mean=5.2640, Penk⁻CamKII+ mean=0.9501, N=11 vs 4 animals, CLES=0.57)
- **Cluster permutation:** p = 0.4850 (observed diff=0.8607, null std=1.2414)

## H3.4b: PD shift differs between types

- **Animal-level Mann-Whitney U:** p = 0.4894 (Penk+ mean=14.0831, Penk⁻CamKII+ mean=-5.3952, N=11 vs 4 animals, CLES=0.64)
- **Cluster permutation:** p = 0.2715 (observed diff=19.0218, null std=17.4495)

## H3.5: Light modulation differs

- **Animal-level Mann-Whitney U:** p = 0.3429 (Penk+ mean=0.1410, Penk⁻CamKII+ mean=0.1653, N=11 vs 4 animals, CLES=0.32)
- **Cluster permutation:** p = 0.7345 (observed diff=-0.0292, null std=0.0704)

## H5.1: RSP has spatial info

- **Descriptive:** 30/347 (8.6%) cells significant

## H5.2: Spatial info differs

- **Animal-level Mann-Whitney U:** p = 0.3429 (Penk+ mean=0.0658, Penk⁻CamKII+ mean=0.0476, N=11 vs 4 animals, CLES=0.68)
- **Cluster permutation:** p = 0.1617 (observed diff=0.0286, null std=0.0207)

## H5.3: Spatial info drops in dark

- **Wilcoxon signed-rank:** p = 0.3548 (mean diff=0.0346, N=347 cells)
