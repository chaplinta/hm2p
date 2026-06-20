# Plan: map-engagement (representational consistency) as a neural correlate of directed exploration

Status: **plan only — nothing implemented.** Design for a population measure that
tests whether the RSP spatial representation is *used* during exploration, and
whether that use weakens in darkness — the neural correlate of the behavioural
finding that mice cover the maze like a directed searcher in light and like a
random walker in dark (see `docs/manuscripts/behaviour-results-summary.md`,
Figure 3; coverage vs random-walk null z = 0.69 light → 0.23 dark, p_adj=0.018).

## Rationale

Tuning curves and population decoding measure the *instantaneous* representation
and need many cells. The behavioural effect is about *memory used over a
trajectory* (seek new ground, avoid recently-visited), and the imaging yields few
HD cells (~2–4 per session), so decoding is not viable. Instead, ask a question
that needs no decoding and degrades gracefully with cell count: **is the same
physical location re-instantiated as the same population state each time it is
visited?** High consistency = the spatial map is engaged; low = activity has
decoupled from position. Prediction: consistency is higher in light than dark.

## Data (all in `sync.h5`, no new upstream computation)

- `dff` (or `events`) with `roi_types` → soma signal, (n_soma, n_frames).
- `x_maze`, `y_maze` → maze cell via `hm2p.maze.discretize.discretize_position_fast`.
- `light_on`, `bad_behav`, `speed_cm_s`, `frame_times`, `hd_deg`.

Existing infra: `hm2p.analysis.population.population_vector_correlation`;
`hm2p.maze` discretisation.

## Measure

Per session:

1. Soma signal `dff[roi_types==0]`; z-score each ROI over the session.
2. Restrict to valid moving frames (`~bad_behav` and `speed_cm_s >= 2.5`).
3. A *visit* = a contiguous run of frames in one maze cell. Per visit, take the
   mean population vector across its frames → one (n_soma,) vector per visit.
4. For each cell with >= K visits in a condition (start K = 3), compute the mean
   pairwise Pearson correlation across its visit-vectors → within-cell consistency.
5. **Debias** against global drift / arousal: subtract the mean across-cell
   correlation (pairs of visits to *different* cells), or a position-label shuffle
   null. The reported quantity is `within - across` (or z vs the shuffle).
6. Per-session summary: mean debiased within-cell consistency, separately for
   light and dark.

Statistic: paired Wilcoxon signed-rank, light vs dark, across sessions
(unit = session; per-animal-median sensitivity reported alongside). Effect size:
matched-pairs rank-biserial. Prediction: light > dark.

## Confounds (in priority order — these decide whether the result is real)

1. **Unequal sampling (the critical one).** The dark coverage drop *is* the
   finding, so dark has fewer cells and fewer visits per cell, which changes the
   correlation estimate on its own. Before comparing, subsample light and dark to
   an equal number of cells and an equal number of visits per cell (bootstrap the
   subsampling, average). Without this the comparison is circular.
2. **Global drift / arousal.** Handled by the within-minus-across debiasing
   (step 5); also report the raw across-cell correlation per condition.
3. **Head-direction confound.** Within-cell consistency partly reflects entering
   a cell at a consistent *heading*, not place. To isolate the place-map: restrict
   to non-HD soma cells, or match the per-visit HD distribution, or regress HD out
   of the visit-vectors. Report with and without HD cells.
4. **Speed / immobility.** Activity scales with movement; using only moving frames
   (step 2) controls the bulk; optionally match the per-visit speed distribution.
5. **Bleaching / time.** Visits closer in time are more correlated; light/dark
   epochs interleave so this is largely balanced, and the across-cell debiasing
   removes the slow component. Optionally restrict pairwise correlations to visits
   within a bounded time separation.

## What would support vs kill it

- **Support:** debiased within-cell consistency significantly higher in light than
  dark, surviving the sampling-match and with non-HD cells included.
- **Kill / null:** no light-dark difference after sampling-match → the spatial
  code is equally (dis)engaged in both, and the behavioural effect has no
  population-consistency correlate at this yield. Honest outcome: the behaviour is
  the lead result and the neural data is corroborative at best.
- **Artefact flag:** an effect that vanishes once cells/visits are matched, or that
  is carried entirely by HD cells, is not map-engagement.

## Feasibility

~20 ROIs make each population vector noisy, but the estimate averages over many
visit-pairs and fails gracefully (a noisy correlation, not a chance-level
decoder). The real limit is the sampling-match step: low-coverage dark epochs
leave some cells with 1–2 visits that drop out. Likely estimable only in
higher-activity sessions — report per-session visit counts so it is clear where
the estimate is thin.

## Constraints

Non-parametric throughout; session unit, paired light vs dark; per-animal
sensitivity; real data only (synthetic only in unit tests). Reuse `population.py`
rather than reimplementing vector correlation. Place-coding / representational-
consistency framing cited per the citation policy when written up.
