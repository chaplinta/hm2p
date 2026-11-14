# Hypothesis Test Report

**Signal:** dff
**Cells:** 469 soma ROIs from 15 animals (23 sessions)
**Groups:** 11 Penk+ animals, 4 CamKII+ animals
**Sessions with behaviour:** 23

**Significant (uncorrected p < 0.05):** 10/25
**Significant (FDR-corrected):** 8/25

## H1.1: Movement increases activity

- **Wilcoxon signed-rank:** p = 0.0000 * (mean diff=0.0153, N=469) [moving_light_event_rate > stationary_light_event_rate] (FDR p = 0.0000**)

## H1.2: Light increases activity

- **Wilcoxon signed-rank:** p = 0.0000 * (mean diff=0.0078, N=469) [moving_light_event_rate > moving_dark_event_rate] (FDR p = 0.0000**)

## H1.3: Movement x light interaction

- **Wilcoxon signed-rank:** p = 0.0062 * (mean diff=0.0023, N=469) (FDR p = 0.0222**)

## H1.4: Baseline activity differs

- **Animal-level Mann-Whitney:** p = 1.0000 (Penk+ mean=0.0349, CamKII+ mean=0.0358, N=11 vs 4 animals, CLES=0.50)
- **Cluster permutation:** p = 0.9103 (observed diff=0.0007, null std=0.0052)
- **Verdict:** not_supported (FDR p = 0.9482)

## H1.5: Movement modulation differs

- **Animal-level Mann-Whitney:** p = 0.1040 (Penk+ mean=0.2969, CamKII+ mean=0.1994, N=11 vs 4 animals, CLES=0.80)
- **Cluster permutation:** p = 0.1926 (observed diff=0.1053, null std=0.0774)
- **Verdict:** not_supported (FDR p = 0.3439)

## H1.6: Movement x light interaction differs

- **Animal-level Mann-Whitney:** p = 0.8513 (Penk+ mean=0.0008, CamKII+ mean=0.0010, N=11 vs 4 animals, CLES=0.55)
- **Cluster permutation:** p = 0.5475 (observed diff=0.0015, null std=0.0024)
- **Verdict:** not_supported (FDR p = 0.6844)

## H2.1: RSP has HD cells

- **Descriptive:** 60/469 (12.8%) cells significant

## H2.2: HD strength differs

- **Animal-level Mann-Whitney:** p = 0.5714 (Penk+ mean=0.3010, CamKII+ mean=0.2948, N=11 vs 4 animals, CLES=0.61)
- **Cluster permutation:** p = 0.6656 (observed diff=0.0118, null std=0.0260)
- **Verdict:** not_supported (FDR p = 0.7734)

## H2.3: Tuning width differs

- **Animal-level Mann-Whitney:** p = 0.4894 (Penk+ mean=64.4228, CamKII+ mean=56.7838, N=11 vs 4 animals, CLES=0.64)
- **Cluster permutation:** p = 0.1512 (observed diff=7.4299, null std=5.1367)
- **Verdict:** not_supported (FDR p = 0.2932)

## H3.1: HD tuning (MVL): light vs dark

- **Wilcoxon signed-rank:** p = 0.0120 * (mean diff=-0.0371, N=469) [hd_dark_mvl > hd_light_mvl] (FDR p = 0.0375**)

## H3.2: PD drifts in darkness

- **Wilcoxon (vs 0):** p = 0.3848 (mean=4.6682, median=0.3735, N=469) (FDR p = 0.5440)

## H3.4: Visual cue dependence differs (KEY)

- **Animal-level Mann-Whitney:** p = 0.7531 (Penk+ mean=1.5667, CamKII+ mean=1.6275, N=11 vs 4 animals, CLES=0.43)
- **Cluster permutation:** p = 0.6806 (observed diff=0.0779, null std=0.2008)
- **Verdict:** not_supported (FDR p = 0.7734)

## H3.4b: PD shift differs between types

- **Animal-level Mann-Whitney:** p = 0.8513 (Penk+ mean=2.8129, CamKII+ mean=7.4333, N=11 vs 4 animals, CLES=0.45)
- **Cluster permutation:** p = 0.5216 (observed diff=-8.1017, null std=11.6729)
- **Verdict:** not_supported (FDR p = 0.6844)

## H3.5: Light modulation differs

- **Animal-level Mann-Whitney:** p = 0.2256 (Penk+ mean=0.0943, CamKII+ mean=0.1678, N=11 vs 4 animals, CLES=0.27)
- **Cluster permutation:** p = 0.3917 (observed diff=-0.0501, null std=0.0552)
- **Verdict:** not_supported (FDR p = 0.5440)

## H4.1a: Movement speed differs between types

- **Animal-level Mann-Whitney:** p = n/a (Penk+ mean=nan, CamKII+ mean=nan, N=11 vs 4 animals, CLES=nan)
- **Cluster permutation:** p = 0.9552 (observed diff=0.0961, null std=1.5486)
- **Verdict:** not_supported [session] (FDR p = 0.9552)

## H4.1b: Fraction active differs between types

- **Animal-level Mann-Whitney:** p = 0.4117 (Penk+ mean=0.6827, CamKII+ mean=0.7829, N=11 vs 4 animals, CLES=0.34)
- **Cluster permutation:** p = 0.1525 (observed diff=-0.1292, null std=0.0890)
- **Verdict:** not_supported [session] (FDR p = 0.2932)

## H4.3: Speed changes in darkness (within session)

- **Wilcoxon signed-rank:** p = 0.0233 * (mean diff=0.3278, N=23) [mean_speed_light > mean_speed_dark] [session] (FDR p = 0.0648)

## H4.4: Activity fraction changes in darkness (within session)

- **Wilcoxon signed-rank:** p = 0.0449 * (mean diff=0.0361, N=23) [frac_active_light > frac_active_dark] [session] (FDR p = 0.1122)

## H4.5: Light-dark speed change differs between types

- **Animal-level Mann-Whitney:** p = 0.4117 (Penk+ mean=0.1525, CamKII+ mean=0.5592, N=11 vs 4 animals, CLES=0.34)
- **Cluster permutation:** p = 0.1058 (observed diff=-0.5217, null std=0.3199)
- **Verdict:** not_supported [session] (FDR p = 0.2404)

## H5.1: RSP has spatial info

- **Descriptive:** 55/469 (11.7%) cells significant

## H5.2: Spatial info differs

- **Animal-level Mann-Whitney:** p = 0.2256 (Penk+ mean=1.3483, CamKII+ mean=0.3115, N=11 vs 4 animals, CLES=0.73)
- **Cluster permutation:** p = 0.3065 (observed diff=2.1748, null std=2.5337)
- **Verdict:** not_supported (FDR p = 0.5108)

## H5.3: Spatial info: light vs dark

- **Wilcoxon signed-rank:** p = 0.0000 * (mean diff=1.5779, N=469) [place_light_spatial_info > place_dark_spatial_info] (FDR p = 0.0002**)

## H6.1: HD tuning peak amplitude: light vs dark

- **Wilcoxon signed-rank:** p = 0.0000 * (mean diff=0.0119, N=469) [gain_peak_light > gain_peak_dark] (FDR p = 0.0000**)

## H6.2: Gain modulation differs between types

- **Animal-level Mann-Whitney:** p = 0.2799 (Penk+ mean=0.0774, CamKII+ mean=0.1531, N=11 vs 4 animals, CLES=0.30)
- **Cluster permutation:** p = 0.3884 (observed diff=-0.0777, null std=0.0820)
- **Verdict:** not_supported (FDR p = 0.5440)

## H7.1: Activity higher at junctions than corridors (light)

- **Wilcoxon signed-rank:** p = 0.7349 (mean diff=0.0010, N=469) [junction_act_junction_light > junction_act_corridor_light] (FDR p = 0.7988)

## H7.2: HD tuning (MVL) junctions vs corridors (light)

- **Wilcoxon signed-rank:** p = 0.0000 * (mean diff=-0.0523, N=469) [junction_mvl_corridor_light > junction_mvl_junction_light] (FDR p = 0.0000**)

## H7.3: Junction HD tuning (MVL): light vs dark

- **Wilcoxon signed-rank:** p = 0.0014 * (mean diff=-0.0411, N=469) [junction_mvl_junction_dark > junction_mvl_junction_light] (FDR p = 0.0059**)
