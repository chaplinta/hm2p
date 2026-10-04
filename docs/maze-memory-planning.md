# Population memory and planning in the maze (`mazemem`)

Question: does pooled RSP population activity carry information about where
the animal has come from, where it is going, how far it still has to go, and
which dead end it will visit next, and does any of this differ between light
and dark?

All soma ROIs of a session are pooled. The Penk+ / Penk⁻CamKII+ label is kept
as a column and summarised per type for reference only; the main result is
the pooled one.

Code: `src/hm2p/analysis/maze_memory.py` (analysis),
`scripts/celltype_extra_mazemem.py` (runner, key `mazemem`, registered in
`scripts/run_celltype_programme.py`).

Run on EC2 (both signals in parallel on one c5.2xlarge):

```bash
python scripts/launch_celltype_runner_ec2.py mazemem --signals spikes dff --wait
```

Command-line options (through `--extra-args`): `--mazemem-shuffles` (default
200), `--mazemem-recent-k` (3), `--mazemem-min-n` (5), `--n-jobs` (1).

## Preprocessing

- Position discretised onto the 23-cell maze graph; behavioural-artefact
  frames unassigned; cell occupancies shorter than 2 frames dropped (same as
  `popdec`). Neural activity and behaviour are NaN on artefact frames, so
  window means use valid frames only.
- **Visits**: consecutive frames in one cell, merged across gaps of at most
  1 s. A longer gap, or a jump of more than 2 graph steps, starts a new
  segment; 2-step jumps are filled with the (unique, tree) intermediate cell.
- **Trips**: the visit sequence between two consecutive dead-end visits in
  one segment (A may equal B).
- **Light condition** of a trip: light if at least 90 % of its frames have the
  lights on, dark if at most 10 %, otherwise mixed (excluded).

## Analyses

| Analysis | Samples | Target | Metric |
| --- | --- | --- | --- |
| `splitter` (retrospective / prospective) | pass-through visits to non-dead-end cells (arrival neighbour differs from departure neighbour); window = the visit | `origin` and `destination` dead end of the trip; strata = (cell, arrival neighbour, departure neighbour); labels with < 5 samples and strata with < 2 labels dropped | balanced accuracy per stratum, weighted by samples; `prospective_minus_retrospective` = destination minus origin at every shift |
| `distance` | same pass-through visits | `remaining` = graph distance to the destination; `elapsed` = transitions since leaving the origin | Spearman and partial Spearman between the cross-validated ridge prediction and the target, partialling the other step count and visit speed; variants `pooled` and `place_controlled` (target and features demeaned per maze cell with training-fold means) |
| `planning` | dead-end visits followed by a trip, dwell >= 0.5 s; window `pre` = 1 s before leaving, `post` = 1 s after | `novel`: destination not among the 3 most recently visited distinct dead ends; `lru`: destination is a least recently visited dead end (other than the origin) | balanced accuracy |

Because the maze is a tree, the arrival neighbour fixes the subtree the
animal came from; within a splitter stratum the origin is decoded only among
the dead ends of that subtree (and the destination among those of the
departure subtree), so place and travel direction cannot contribute.

**Behaviour** (`behaviour.csv`): per condition, the fraction of trips whose
destination is `novel` / `lru`, against the exact expectation of two random
walkers leaving the same dead end (absorbing Markov chain): `uniform`
(backtracking allowed, as in the coverage-vs-random-walk analysis) and
`nonbacktracking` (no reversals). Forward bias alone raises the novel
fraction above the uniform walk on a tree, so the non-backtracking walk is
the stricter baseline for memory-guided choice.

## Decoders, controls and null

- Classifier: L2 logistic regression (C = 1) with balanced class weights,
  one-vs-rest above two classes. Regression: ridge (alpha = 10). Features
  z-scored on the training fold.
- Cross-validation: 5 contiguous time blocks; training samples within 5 s of
  the test block dropped; run within each stratum.
- Feature sets: `neural`; `behaviour` (sine/cosine of circular-mean HD,
  speed, |AHV|, time in session, time since leaving the last dead end, time
  since entering the last dead end, number of distinct dead ends visited so
  far — the same eight channels for all analyses); `neural_hd_removed`
  (residualised on window HD, training fold only).
- Null: circular shift of the predictors by >= 30 s, 200 shifts shared by all
  analyses, conditions and feature sets; `p = (1 + #null at least as good) /
  (1 + n)`; `excess` = observed minus null mean.
- Light vs dark: models trained and tested within a condition; light and dark
  sample sets equalised by subsampling (per stratum x label, per cell x
  remaining steps, per class). Condition `all` pools light and dark without
  equalisation.
- Cell-number control: neural scores with 10 random cells (5 draws, 10-shift
  null), `subset_curve.csv`.

Implementation note: samples, labels and folds are the same at every shift,
so the logistic (Newton's method) and ridge fits are solved for all shifts
at once. The Newton solution matches scikit-learn's `LogisticRegression`
(lbfgs) to 1e-6 (unit test). Runtime on a synthetic 18 000-frame session with
20 cells and 200 shifts: about 35 s on one core (about 100 s with 50 cells).

## Outputs

`sessions.csv`, `behaviour.csv`, `sample_counts.csv`, `subset_curve.csv`,
`subset_curve_summary.csv`, `summary.csv` (Wilcoxon signed-rank of excess vs 0
across sessions and across animal medians; pooled and per cell type,
descriptive), `comparisons.csv` (paired Wilcoxon: neural vs behaviour, HD
removed vs behaviour, neural vs HD removed; light minus dark; destination
minus origin), `run_info.json`. Rows that could not be computed have NaN
values and a `reason`.

## Confounds not controlled

- Origin / destination labels are correlated with time in session (the animal
  explores different parts of the maze at different times). Time-block CV and
  the behaviour baseline (time in session) address this only partly; slow
  drift in neural activity that tracks the explored region could still be
  decoded as origin or destination.
- Within a splitter stratum, visits from different origins differ in elapsed
  time since leaving the dead end and in the preceding path; the behaviour
  baseline contains time since trip start but not the path itself.
- Destination decoding at a pass-through visit is partly retrospective in a
  tree: the destination subtree is fixed by the departure neighbour, but which
  dead end within it is reached can depend on choices already being made
  (speed, head direction), which are only linearly controlled.
- `remaining` and `elapsed` are negatively correlated on direct trips; partial
  Spearman removes the linear rank association, not nonlinear dependence.
- Novelty labels depend on the full dead-end history, including dead-end
  visits during mixed-light or artefact-adjacent periods.
- Light and dark epochs alternate every minute; sample-count equalisation
  matches counts but not the spatial distribution of visits beyond the
  stratum definitions.

## References

- Frank LM, Brown EN, Wilson M. 2000. "Trajectory encoding in the hippocampus
  and entorhinal cortex." Neuron 27:169-178. doi:10.1016/S0896-6273(00)00018-0
- Wood ER, Dudchenko PA, Robitsek RJ, Eichenbaum H. 2000. "Hippocampal neurons
  encode information about different types of memory episodes occurring in the
  same location." Neuron 27:623-633. doi:10.1016/S0896-6273(00)00071-4
- Ferbinteanu J, Shapiro ML. 2003. "Prospective and retrospective memory coding
  in the hippocampus." Neuron 40:1227-1239. doi:10.1016/S0896-6273(03)00752-9
- Alexander AS, Nitz DA. 2015. "Retrosplenial cortex maps the conjunction of
  internal and external spaces." Nature Neuroscience 18:1143-1151.
  doi:10.1038/nn.4058
- Vedder LC, Miller AMP, Harrison MB, Smith DM. 2017. "Retrosplenial cortical
  neurons encode navigational cues, trajectories and reward locations during
  goal directed navigation." Cerebral Cortex 27:3713-3723.
  doi:10.1093/cercor/bhw192
- Howard LR, Javadi AH, Yu Y, et al. 2014. "The hippocampus and entorhinal
  cortex encode the path and Euclidean distances to goals during navigation."
  Current Biology 24:1331-1340. doi:10.1016/j.cub.2014.05.001
- Pfeiffer BE, Foster DJ. 2013. "Hippocampal place-cell sequences depict future
  paths to remembered goals." Nature 497:74-79. doi:10.1038/nature12112
- Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth show
  rapid learning, sudden insight, and efficient exploration." eLife 10:e66175.
  doi:10.7554/eLife.66175
- Kemeny JG, Snell JL. 1960. "Finite Markov Chains." Van Nostrand.
- Hoerl AE, Kennard RW. 1970. "Ridge regression: biased estimation for
  nonorthogonal problems." Technometrics 12:55-67.
  doi:10.1080/00401706.1970.10488634
- Kim S. 2015. "ppcor: an R package for a fast calculation to semi-partial
  correlation coefficients." Communications for Statistical Applications and
  Methods 22:665-674. doi:10.5351/CSAM.2015.22.6.665
- Roberts DR, Bahn V, Ciuti S, et al. 2017. "Cross-validation strategies for
  data with temporal, spatial, hierarchical, or phylogenetic structure."
  Ecography 40:913-929. doi:10.1111/ecog.02881
- Brodersen KH, Ong CS, Stephan KE, Buhmann JM. 2010. "The balanced accuracy
  and its posterior distribution." ICPR 2010, 3121-3124.
  doi:10.1109/ICPR.2010.764
- Pedregosa F, et al. 2011. "Scikit-learn: Machine Learning in Python." Journal
  of Machine Learning Research 12:2825-2830.
  https://github.com/scikit-learn/scikit-learn
