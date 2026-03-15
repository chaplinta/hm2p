# Hypotheses

Testable predictions for the hm2p project. Each hypothesis maps to specific
metrics in `analysis.h5` and a statistical test from `docs/stats-strategy.md`.

All between-group tests use the non-parametric framework: animal-level summary
(Mann-Whitney U) + cluster permutation test. Within-cell tests use Wilcoxon
signed-rank. See `docs/stats-strategy.md` for details.

## Confound Controls (apply to ALL tests)

Every between-group comparison must account for potential confounds beyond
cell type. The cluster permutation test (shuffling at animal level) handles
the primary confound (cells nested within animals), but the following
additional covariates must be checked for each significant result:

| Confound | Source | How to control |
|----------|--------|---------------|
| **Animal** | Cells from same mouse are not independent | Cluster permutation at animal level (primary test) |
| **Injection site** | AP/ML/DV coordinates vary between animals | Report injection coordinates; stratify or include as covariate if sites differ systematically between groups |
| **Hemisphere** | Left vs right RSP may differ | Include hemisphere as a covariate or verify balanced across groups |
| **Sex** | Male vs female physiology/behaviour differences | Include sex as a covariate or verify balanced; report sex breakdown per group |
| **Imaging depth** | Deeper cells may have worse SNR | Report distribution; exclude if confounded |
| **Session order** | Repeated sessions — habituation, learning | Include session number as covariate or use first session only |
| **FOV quality** | Motion artefacts, expression level | Exclude low-quality sessions (existing QC pipeline) |

**Metadata sources:**
- `animals.csv`: animal_id, celltype, sex
- `experiments.csv`: exp_id, session info
- Injection sites: `anatomy/injection.py` → AP/ML/DV per animal
- Hemisphere: `patching/metadata/cells.csv` (patching); needs adding to `animals.csv` for imaging

**Reporting requirement:** For every significant between-group result,
report the distribution of each confound variable across the two groups.
If a confound is unbalanced (e.g. all Penk⁻CamKII+ animals are male),
flag the result as potentially confounded.

---

## 1. Activity: Movement × Light (2×2 factorial)

Activity is measured in 4 conditions: moving-light, moving-dark,
stationary-light, stationary-dark. The analysis.h5 stores all four.
Hypotheses test main effects (movement, light) and their interaction.

### H1.1 Movement increases activity (main effect)
**Prediction:** Event rate is higher during movement than stationary,
averaging across light conditions.
**Metric:** Mean of (`moving_light` + `moving_dark`) vs mean of
(`stationary_light` + `stationary_dark`) per cell.
**Test:** Within-cell paired Wilcoxon.
**Also test separately in light and dark** to check for interaction.

### H1.2 Light increases activity (main effect)
**Prediction:** Event rate is higher in light than dark, averaging across
movement states.
**Metric:** Mean of (`moving_light` + `stationary_light`) vs mean of
(`moving_dark` + `stationary_dark`) per cell.
**Test:** Within-cell paired Wilcoxon.
**Also test separately when moving and stationary.**

### H1.3 Movement × light interaction
**Prediction:** The effect of movement on activity differs between light
and dark conditions (or equivalently, the light effect differs between
moving and stationary states). For example, movement may drive stronger
activity increases in light (when visual flow is available) than in dark.
**Metric:** Interaction contrast per cell:
(`moving_light` − `stationary_light`) − (`moving_dark` − `stationary_dark`)
**Test:** One-sample Wilcoxon on interaction contrast (test ≠ 0).

### H1.4 Penk+ and Penk⁻CamKII+ differ in baseline activity
**Prediction:** The two populations have different mean event rates or
dF/F amplitudes (in the moving-light condition, the cleanest baseline).
**Metric:** Between-group comparison of `moving_light_event_rate`,
`moving_light_mean_signal`, `moving_light_mean_amplitude`.
**Test:** Animal-level summary + cluster permutation.

### H1.5 Movement modulation differs between cell types
**Prediction:** One population is more strongly modulated by movement state.
**Metric:** Between-group comparison of `movement_modulation`.
**Test:** Animal-level summary + cluster permutation.

### H1.6 The movement × light interaction differs between cell types
**Prediction:** The 2×2 interaction contrast (H1.3) differs between Penk+
and Penk⁻CamKII+ — i.e. one population's movement sensitivity depends more
on visual context.
**Metric:** Between-group comparison of the per-cell interaction contrast.
**Test:** Animal-level summary + cluster permutation.

---

## 2. Head Direction Tuning

### H2.1 RSP contains head direction cells
**Prediction:** A significant fraction of neurons show non-uniform HD tuning
(circular shuffle test, p < 0.05).
**Metric:** `hd/all/significant` — fraction of cells with MVL above shuffle.

### H2.2 HD tuning strength differs between cell types
**Prediction:** Penk+ and Penk⁻CamKII+ populations differ in the proportion
of HD cells or in the strength of HD tuning (MVL).
**Metric:** Between-group comparison of `hd/all/mvl` and fraction of
`hd/all/significant` cells.
**Test:** Animal-level summary + cluster permutation.

### H2.3 HD tuning width differs between cell types
**Prediction:** One population has broader or narrower tuning curves.
**Metric:** Between-group comparison of `hd/all/tuning_width`.
**Test:** Animal-level summary + cluster permutation.

### H2.4 Preferred direction distribution is uniform
**Prediction:** HD cells in RSP cover all 360° without bias toward a
particular direction (Rayleigh test on PD distribution, expect p > 0.05).
**Metric:** `hd/all/preferred_direction` — test uniformity per population.

---

## 3. Visual Cue Dependence (Light vs Dark)

### H3.1 HD tuning degrades in darkness
**Prediction:** MVL decreases when lights are off (visual cues removed),
reflecting reliance on visual landmarks for HD anchoring.
**Metric:** `hd/light/mvl` vs `hd/dark/mvl` (within-cell, paired Wilcoxon).
Also `hd/comparison/mvl_ratio` < 1.

### H3.2 Preferred direction drifts in darkness
**Prediction:** The preferred direction shifts between light and dark epochs,
indicating visual cue anchoring.
**Metric:** `hd/comparison/pd_shift` — mean absolute shift > 0 (one-sample
Wilcoxon vs 0).

### H3.3 Tuning curve shape is preserved in darkness
**Prediction:** Despite drift, the overall shape of the tuning curve is
maintained (high light-dark correlation), suggesting the cell retains HD
selectivity but loses visual anchor.
**Metric:** `hd/comparison/correlation` — expect high values (> 0.5) even
when MVL drops.

### H3.4 Visual cue dependence differs between cell types (KEY HYPOTHESIS)
**Prediction:** Penk+ and Penk⁻CamKII+ populations differ in how much HD
tuning degrades in darkness. One population may rely more on visual cues
(larger MVL drop, larger PD shift) while the other relies more on
path-integration (smaller change in dark).
**Metric:** Between-group comparison of `hd/comparison/mvl_ratio`,
`hd/comparison/pd_shift`, `hd/comparison/correlation`.
**Test:** Animal-level summary + cluster permutation.
**This is the central hypothesis of the project.**

### H3.5 Light modulation of activity differs between cell types
**Prediction:** One population shows stronger light/dark modulation of
overall firing rate (not just tuning).
**Metric:** Between-group comparison of `light_modulation`.
**Test:** Animal-level summary + cluster permutation.

---

## 4. Behaviour and Maze Exploration

### H4.1 Movement patterns differ between genotypes
**Prediction:** Penk-Cre mice (Penk+ labelled) and Penk-Cre mice with
Cre-OFF virus (Penk⁻CamKII+ labelled) differ in locomotion — speed
distributions, fraction of time moving, or angular head velocity profiles.
**Metric:** Per-session mean speed, fraction active, mean AHV, grouped by
celltype (which here is really genotype × virus).
**Test:** Animal-level Mann-Whitney (N=12 vs N=4 animals).
**Caveat:** Celltype label reflects the virus, not a behavioural
manipulation — behavioural differences may be genotype-driven or cohort
effects rather than cell-type specific.

### H4.2 Maze exploration strategy differs between genotypes
**Prediction:** Animals with different labelled populations explore the
rose maze differently — occupancy distribution, coverage rate, time to
explore all arms, dead-end dwell time, turn bias.
**Metric:** From `maze/analysis.py`: `occupancy_entropy`, `coverage_rate`,
`dead_end_visits`, `turn_bias`, `exploration_efficiency`.
**Test:** Animal-level Mann-Whitney.
**Note:** Only meaningful if maze sessions are matched for duration and
experience (first vs repeated exposure).

### H4.3 Exploration changes between light and dark
**Prediction:** Mice alter their movement strategy when lights go off —
slower speed, more wall-following, reduced exploration, increased
pausing — reflecting reliance on visual cues for navigation.
**Metric:** Speed, fraction active, occupancy entropy — compared between
light and dark epochs within each session.
**Test:** Within-session paired Wilcoxon.

### H4.4 Light-induced behavioural changes differ between genotypes
**Prediction:** The behavioural shift from light to dark (speed change,
exploration change) differs between Penk+ and Penk⁻CamKII+ animals,
suggesting the two populations play different roles in visually-guided
navigation.
**Metric:** Delta (light − dark) for speed, occupancy entropy, coverage
rate — compared between genotypes.
**Test:** Animal-level summary + cluster permutation.

### H4.5 Penk+ and Penk⁻CamKII+ neurons integrate visual inputs differently
**Prediction:** Beyond HD tuning (H3.4), the two populations differ in how
light/dark transitions affect their overall activity patterns — not just
tuning curve parameters but also population correlation structure,
co-activity patterns, or event dynamics.
**Metric:** Between-group comparison of:
- `light_modulation` (overall rate change)
- Population vector correlation (light vs dark epochs)
- Event amplitude and duration changes across light transitions
- CEBRA embedding distance between light and dark states (if computed)
**Test:** Animal-level summary + cluster permutation.
**This extends H3.4 beyond HD tuning to general visual integration.**

---

## 5. Spatial Coding

### H5.1 RSP neurons carry spatial information
**Prediction:** A subset of neurons has significant spatial information
(place-like or conjunctive HD × position coding).
**Metric:** `place/all/significant`, `place/all/spatial_info`.

### H5.2 Spatial coding differs between cell types
**Prediction:** Penk+ and Penk⁻CamKII+ differ in spatial information
content or place map quality.
**Metric:** Between-group comparison of `place/all/spatial_info`,
`place/all/spatial_coherence`.
**Test:** Animal-level summary + cluster permutation.

### H5.3 Spatial coding degrades in darkness
**Prediction:** Spatial information drops in dark epochs if position coding
relies on visual landmarks.
**Metric:** `place/light/spatial_info` vs `place/dark/spatial_info`
(within-cell, paired Wilcoxon).

---

## 6. Population Decoding

### H6.1 HD can be decoded from population activity
**Prediction:** A PVA decoder trained on light epochs reconstructs head
direction with low error (< 30° mean absolute error).
**Metric:** Cross-validated decode error from `decoder.py`.

### H6.2 Decoding accuracy degrades in darkness
**Prediction:** Decoder error increases in dark epochs, reflecting reduced
HD signal fidelity.
**Metric:** Light vs dark decode error (paired comparison across sessions).

### H6.3 Decoding accuracy differs between cell types
**Prediction:** One population supports better HD decoding (lower error,
higher R²).
**Metric:** Per-session decode error, grouped by celltype.
**Test:** Animal-level summary + cluster permutation.

---

## 7. Signal Robustness

### H7.1 Results are consistent across signal types
**Prediction:** Key findings (HD tuning, cell-type differences, light/dark
effects) replicate across dF/F, deconvolved, and event-based signals.
**Metric:** Compare all above metrics computed from `dff/`, `events/`,
and (if available) `deconv/` groups in analysis.h5.
**Criterion:** A result is robust if direction and significance agree
across at least 2 of 3 signal types.

---

## Summary Table

| # | Hypothesis | Type | Key Metric | Primary Test |
|---|-----------|------|-----------|-------------|
| H1.1 | Movement increases activity | Within-cell | event_rate moving vs stationary | Wilcoxon |
| H1.2 | Light increases activity | Within-cell | event_rate light vs dark | Wilcoxon |
| **H1.3** | **Movement × light interaction** | **Within-cell** | **interaction contrast** | **Wilcoxon vs 0** |
| H1.4 | Baseline activity differs | Between-group | event_rate | Animal summary + permutation |
| H1.5 | Movement modulation differs | Between-group | movement_modulation | Animal summary + permutation |
| **H1.6** | **Interaction differs by type** | **Between-group** | **interaction contrast** | **Animal summary + permutation** |
| H2.1 | RSP has HD cells | Descriptive | fraction significant | — |
| H2.2 | HD strength differs | Between-group | MVL | Animal summary + permutation |
| H2.3 | Tuning width differs | Between-group | tuning_width | Animal summary + permutation |
| H2.4 | PD is uniform | Distribution | preferred_direction | Rayleigh |
| H3.1 | Darkness degrades tuning | Within-cell | MVL light vs dark | Wilcoxon |
| H3.2 | PD drifts in dark | Within-cell | pd_shift | Wilcoxon vs 0 |
| H3.3 | Tuning shape preserved | Within-cell | light-dark correlation | Descriptive |
| **H3.4** | **Cue dependence differs** | **Between-group** | **mvl_ratio, pd_shift** | **Animal summary + permutation** |
| H3.5 | Light modulation differs | Between-group | light_modulation | Animal summary + permutation |
| H4.1 | Movement patterns differ | Between-group (animal) | speed, frac active, AHV | Animal Mann-Whitney |
| H4.2 | Maze exploration differs | Between-group (animal) | occupancy entropy, coverage | Animal Mann-Whitney |
| H4.3 | Exploration changes in dark | Within-session | speed, entropy light vs dark | Wilcoxon |
| **H4.4** | **Behavioural light shift differs** | **Between-group** | **delta speed, delta entropy** | **Animal summary + permutation** |
| **H4.5** | **Visual integration differs** | **Between-group** | **light_mod, pop corr, events** | **Animal summary + permutation** |
| H5.1 | Spatial information | Descriptive | fraction significant | — |
| H5.2 | Spatial coding differs | Between-group | spatial_info | Animal summary + permutation |
| H5.3 | Spatial coding drops in dark | Within-cell | SI light vs dark | Wilcoxon |
| H6.1 | HD decodable | Descriptive | decode error | — |
| H6.2 | Decoding drops in dark | Within-session | decode error | Wilcoxon |
| H6.3 | Decoding differs by type | Between-group | decode error | Animal summary + permutation |
| H7.1 | Signal robustness | Replication | all metrics × 3 signals | Agreement |
