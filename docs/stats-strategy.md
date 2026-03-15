# Statistical Strategy: Accounting for Non-Independence

## The Problem

Cells are nested within animals. Standard tests (Mann-Whitney U, t-test) treat
each cell as independent, but cells from the same animal share:

- Surgery quality, imaging depth, GCaMP expression level
- Behavioural state, arousal, running speed distribution
- Optical clarity, motion artefacts, neuropil contamination

Treating N=391 cells as independent when they come from N=16 animals inflates
test statistics and produces false positives. This is **pseudoreplication**.

### Data Structure

```
Celltype (fixed, 2 levels: penk, nonpenk)
  └─ Animal (random, 16 levels: 12 penk, 4 nonpenk)
       └─ Session (random, 26 levels: 1-4 per animal)
            └─ Cell/ROI (observation, 391 total)
```

Some animals have multiple sessions (e.g. 1114356 has 4 sessions, 1117217 has 3).
ROIs within a session share the same behavioural epoch, light sequence, and
imaging conditions. ROIs across sessions of the same animal share expression
level, surgical window, and cortical region.

### Current Approach (Naive)

Pool all cells, run Mann-Whitney U. This gives p-values that are **too small**
because the effective sample size is closer to 16 (animals) than 391 (cells).

---

## Proposed Strategy

Three complementary approaches, from simplest to most rigorous. All three
should agree for a result to be considered robust.

### Approach 1: Animal-Level Summary Statistics (Conservative, Primary)

**The simplest correct approach.** Collapse to one value per animal, then
compare celltypes.

1. For each metric (MVL, SI, event rate, etc.), compute the **mean across
   cells** within each animal (or animal-session if treating sessions
   separately).
2. Compare the two groups of animal-level means using a standard unpaired
   test (Mann-Whitney U or permutation test).

**Advantages:**
- Correct by construction — each observation is independent
- No distributional assumptions beyond the test used
- Easy to understand and present

**Disadvantages:**
- Discards within-animal variance (loses power)
- With 12 vs 4 animals, statistical power is limited
- Cannot include covariates (session effects, ROI type)

**Implementation:**

```python
def animal_summary_test(
    df: pd.DataFrame,          # columns: animal_id, celltype, metric
    metric_col: str,
    group_col: str = "celltype",
    animal_col: str = "animal_id",
) -> dict:
    """Collapse to animal means, then Mann-Whitney U."""
    animal_means = df.groupby([animal_col, group_col])[metric_col].mean().reset_index()
    penk = animal_means.loc[animal_means[group_col] == "penk", metric_col]
    nonpenk = animal_means.loc[animal_means[group_col] == "nonpenk", metric_col]
    stat, p = mannwhitneyu(penk, nonpenk, alternative="two-sided")
    return {"statistic": stat, "p_value": p, "n_penk_animals": len(penk),
            "n_nonpenk_animals": len(nonpenk)}
```

**Where to apply:** Every between-celltype comparison in the pipeline —
MVL, spatial information, event rate, speed modulation, tuning width,
light/dark gain, decoder accuracy, etc.

---

### Approach 2: Linear Mixed Model (LMM, Supplementary Only)

**Parametric — use only as a supplementary check, never as the primary test.**
All primary tests must be non-parametric (Approaches 1 and 3). LMM is included
for completeness and because reviewers may expect it, but it assumes normality
of residuals which neural data rarely satisfies.

```
metric ~ celltype + (1 | animal_id)
```

Or with session nested within animal:

```
metric ~ celltype + (1 | animal_id / session_id)
```

Or with covariates:

```
metric ~ celltype * condition + roi_type + (1 | animal_id)
```

**Advantages:**
- Uses all cell-level data (preserves power)
- Properly partitions variance into animal-level and cell-level
- Can include covariates (light condition, ROI type, speed)
- Reports ICC — the fraction of variance due to animal
- Can test interactions (celltype x light condition)

**Disadvantages:**
- Requires normality of residuals (can be relaxed with GLMM)
- With only 4 nonpenk animals, random effect estimates are unstable
- Convergence can fail with too few groups

**Implementation plan:**

Module: `src/hm2p/analysis/mixed_stats.py`

```python
def lmm_celltype_test(
    df: pd.DataFrame,
    metric_col: str,
    fixed: str = "celltype",
    random: str = "animal_id",
    covariates: list[str] | None = None,
) -> dict:
    """LMM: metric ~ celltype [+ covariates] + (1|animal_id)."""
    import statsmodels.formula.api as smf

    formula = f"{metric_col} ~ C({fixed})"
    if covariates:
        formula += " + " + " + ".join(covariates)

    model = smf.mixedlm(formula, data=df, groups=df[random])
    result = model.fit(reml=True)

    # Extract celltype effect
    fe_name = [n for n in result.fe_params.index if fixed in n][0]
    beta = result.fe_params[fe_name]
    p = result.pvalues[fe_name]

    # ICC
    var_animal = result.cov_re.iloc[0, 0]
    var_resid = result.scale
    icc = var_animal / (var_animal + var_resid)

    return {"beta": beta, "se": result.bse_fe[fe_name], "z": result.tvalues[fe_name],
            "p_value": p, "icc": icc, "converged": result.converged}
```

**Key models to fit:**

| Comparison | Formula | Purpose |
|---|---|---|
| HD tuning | `mvl ~ celltype + (1\|animal)` | Do celltypes differ in HD selectivity? |
| Activity | `event_rate ~ celltype + (1\|animal)` | Do celltypes differ in firing rate? |
| Light effect | `mvl ~ celltype * light + (1\|animal)` | Does light removal affect celltypes differently? |
| Speed modulation | `speed_score ~ celltype + (1\|animal)` | Do celltypes differ in speed coding? |

**When LMM fails (convergence):** Fall back to Approach 1.

---

### Approach 3: Cluster Bootstrap Permutation Test (Primary for Nested Data)

**Non-parametric and assumption-free.** Resample at the animal level to
build a null distribution.

1. Compute the observed test statistic (e.g., difference in group means of
   a metric, using all cells).
2. For B=10000 iterations:
   a. Randomly **permute celltype labels at the animal level** (not cell level).
      All cells from animal X stay together; only the animal's celltype assignment
      is shuffled.
   b. Recompute the test statistic.
3. P-value = fraction of permuted statistics ≥ observed.

**Advantages:**
- No distributional assumptions
- Respects the clustering structure exactly
- Works with any test statistic (difference in means, medians, ratios)
- Valid even with very few clusters

**Disadvantages:**
- Computationally expensive (but feasible — 16 animals x 10000 perms is fast)
- With 12 vs 4 animals, the number of unique permutations is C(16,4) = 1820,
  so the minimum achievable p-value is ~1/1820 ≈ 0.0005
- Cannot easily include covariates

**Implementation plan:**

```python
def cluster_permutation_test(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str = "celltype",
    cluster_col: str = "animal_id",
    n_perms: int = 10000,
    stat_func: Callable = lambda a, b: np.mean(a) - np.mean(b),
) -> dict:
    """Permutation test that shuffles group labels at the cluster level."""
    # Observed statistic (pool cells within group)
    penk_vals = df.loc[df[group_col] == "penk", metric_col].dropna().values
    nonpenk_vals = df.loc[df[group_col] == "nonpenk", metric_col].dropna().values
    observed = stat_func(penk_vals, nonpenk_vals)

    # Get unique clusters and their group assignments
    cluster_groups = df.groupby(cluster_col)[group_col].first()
    cluster_ids = cluster_groups.index.values
    n_nonpenk = (cluster_groups == "nonpenk").sum()

    # Permute at cluster level
    null_stats = np.empty(n_perms)
    for i in range(n_perms):
        perm_nonpenk = np.random.choice(cluster_ids, size=n_nonpenk, replace=False)
        perm_penk = np.setdiff1d(cluster_ids, perm_nonpenk)

        a = df.loc[df[cluster_col].isin(perm_penk), metric_col].dropna().values
        b = df.loc[df[cluster_col].isin(perm_nonpenk), metric_col].dropna().values
        null_stats[i] = stat_func(a, b)

    p_value = (np.sum(np.abs(null_stats) >= np.abs(observed)) + 1) / (n_perms + 1)

    return {"observed": observed, "p_value": p_value,
            "null_mean": np.mean(null_stats), "null_std": np.std(null_stats)}
```

---

## Reporting Framework

For each between-celltype comparison, report all three:

| Level | Test | Role |
|---|---|---|
| Cell-level (naive) | Mann-Whitney U | Descriptive — "are the distributions different if we ignore nesting?" |
| Animal-level | Animal-mean Mann-Whitney | **Primary (simple)** — "are animal averages different?" |
| Cluster permutation | Permutation at animal level | **Primary (nested)** — "is the effect robust to non-parametric resampling?" |
| Mixed model | LMM with animal random intercept | Supplementary — parametric check, reports ICC |

**All primary tests must be non-parametric.** LMM is supplementary only.

**Decision rule:** A result is considered robust if:
- Cluster permutation p < 0.05 (FDR-corrected across metrics), **AND**
- Animal-level summary test shows the same direction of effect

LMM results are reported alongside for completeness (ICC is informative)
but do not determine significance.

If only the naive (cell-level) test is significant, the result is flagged as
**potentially confounded by animal** and reported as exploratory.

---

## Light/Dark Within-Cell Comparisons

For paired within-cell comparisons (e.g., "does MVL change from light to dark?"),
the nesting structure is different:

```
metric_light - metric_dark ~ celltype + (1 | animal_id)
```

Or equivalently:

```
metric ~ celltype * condition + (1 | animal_id / cell_id)
```

Here `cell_id` is nested within `animal_id`, and `condition` (light/dark) is
a within-cell repeated measure.

**Approach 1 (simple):** Compute per-cell difference (dark - light), then
use the between-celltype LMM on the differences.

**Approach 2 (full):** Fit a crossed random-effects model with animal and
cell as random intercepts, celltype x condition as fixed effects.

---

## Multiple Comparisons

Apply Benjamini-Hochberg FDR correction across all metrics tested within
each comparison family:

- **Family 1:** HD tuning metrics (MVL, tuning width, PD, SI, Rayleigh p) — 5 tests
- **Family 2:** Activity metrics (event rate, mean dF/F, event amplitude) — 3 tests
- **Family 3:** Behavioural modulation (speed score, AHV score) — 2 tests
- **Family 4:** Light/dark metrics (gain, PD shift, tuning correlation) — 3 tests
- **Family 5:** Morphology metrics (all morph_* columns) — 35 tests

Report both uncorrected and FDR-corrected p-values.

---

## Effect Sizes

Always report effect sizes alongside p-values:

- **Cohen's d** (or Hedges' g for unequal groups): standardized mean difference
- **Common language effect size (CLES)**: probability that a random Penk+ cell
  has a higher value than a random Penk⁻CamKII+ cell
- **ICC**: intraclass correlation from LMM — quantifies the animal confound

---

## Implementation Plan

1. **New module:** `src/hm2p/analysis/mixed_stats.py`
   - `animal_summary_test()` — Approach 1
   - `lmm_celltype_test()` — Approach 2
   - `cluster_permutation_test()` — Approach 3
   - `run_all_comparisons()` — orchestrator that runs all three for a metric list
   - `comparison_summary_table()` — DataFrame with all results side by side

2. **Integration:** `src/hm2p/analysis/run.py`
   - After per-cell analysis, build a pooled DataFrame with columns:
     `animal_id, session_id, celltype, roi_idx, roi_type, mvl, si, event_rate, ...`
   - Call `run_all_comparisons()` on this DataFrame
   - Save results to `analysis/population_stats.h5` or `.csv`

3. **Frontend:** Update comparison pages to show:
   - Naive p-value (current, kept for reference)
   - LMM p-value + ICC (primary)
   - Cluster permutation p-value (confirmatory)
   - Animal-level dot plots overlaid on cell-level box plots

4. **Tests:** `tests/analysis/test_mixed_stats.py`
   - Synthetic data with known ICC to verify LMM recovers it
   - Permutation test with shuffled labels gives p ≈ 0.5
   - Permutation test with real effect gives p < 0.05
   - Edge cases: single animal per group, all cells from one animal

---

## Power Considerations

With 12 Penk+ animals and 4 Penk⁻CamKII+ animals:
- Animal-level tests have effective N = 16 (severely limited)
- Cluster permutation: minimum p ≈ 0.0005 (C(16,4) = 1820 unique permutations)
- LMM gains power by using cell-level variance, but random effect estimates
  are unstable with only 4 nonpenk animals

**Interpretation guidance:** Null results may reflect low power rather than
absence of effect. Report confidence intervals alongside p-values.
