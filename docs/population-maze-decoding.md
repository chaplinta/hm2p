# Pooled-population maze decoding (`popdec`)

Question: with all soma ROIs of a session pooled (Penk+ / Penk⁻CamKII+ label
ignored for the main result), does RSP population activity carry information
about maze behaviour beyond running?

Code: `src/hm2p/analysis/population_maze.py` (analysis),
`scripts/celltype_extra_popdec.py` (runner, key `popdec`, registered in
`scripts/run_celltype_programme.py`).

Run on EC2 (both signals in parallel on one c5.2xlarge):

```bash
python scripts/launch_celltype_runner_ec2.py popdec --signals spikes dff --wait
```

## Decoders

| Decoder | Samples | Label | Metric |
| --- | --- | --- | --- |
| `inout` | running bouts into / out of dead ends, class-balanced and matched on duration and mean speed (3 x 3 quantile bins) | direction relative to dead end | balanced accuracy |
| `position` | contiguous stays in one maze cell during running (>= 3 frames; cells with >= 5 visits) | maze cell | class-balanced mean graph distance between decoded and true cell (headline); balanced accuracy |
| `edge_direction` | traversals of a maze edge (two consecutive running visits to adjacent cells); edges with >= 5 traversals each way | travel direction | balanced accuracy per edge, averaged weighted by samples |
| `junction_exit` | T-junction passes, U-turns excluded; stratum = (junction, arrival arm); strata with >= 5 of each exit | exit arm (binary within stratum) | balanced accuracy per stratum, weighted average |

Features: per-cell mean activity over the sample window, z-scored on the
training fold. Classifier: L2 logistic regression with balanced class weights
(scikit-learn). Cross-validation: 5 contiguous time blocks; training samples
within 5 s of the test block are dropped.

## Controls

- **Behaviour only**: sine/cosine of the circular mean head direction, mean
  speed and mean |AHV| in the window. The maze arms have fixed orientations,
  so head direction alone is expected to decode `edge_direction`,
  `junction_exit` and `inout` well. Neural decoding of those labels is
  interpretable only relative to this baseline.
- **Neural with HD removed**: each cell's sample activity residualised on
  sine/cosine of window HD, regression fitted on the training fold only.
- **Circular-shift null** for every decoder and feature set: predictors
  shifted relative to the labels by >= 30 s (200 shifts); features and the full
  CV are recomputed. `p = (1 + #null at least as good) / (1 + n)`.
- **Cell-number curve**: neural decoding with random subsets of 5 and 10 cells
  (10 draws, 10-shift null) and the full population, so sessions with different
  cell counts are comparable.

## Across-session statistics

- Excess over the null mean (positive = better than null), Wilcoxon signed-rank
  against 0 across sessions and across per-animal medians (`summary.csv`).
- Paired Wilcoxon of excess between feature sets: neural vs behaviour, neural
  with HD removed vs behaviour, neural vs neural with HD removed
  (`comparisons.csv`).
- Per cell type: descriptive only.

## References

- Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth show
  rapid learning, sudden insight, and efficient exploration." eLife 10:e66175.
  doi:10.7554/eLife.66175
- Alexander AS, Nitz DA. 2015. "Retrosplenial cortex maps the conjunction of
  internal and external spaces." Nature Neuroscience 18:1143-1151.
  doi:10.1038/nn.4058
- Brodersen KH, Ong CS, Stephan KE, Buhmann JM. 2010. "The balanced accuracy and
  its posterior distribution." Proceedings of the 20th International
  Conference on Pattern Recognition, 3121-3124. doi:10.1109/ICPR.2010.764
- Roberts DR, Bahn V, Ciuti S, et al. 2017. "Cross-validation strategies for
  data with temporal, spatial, hierarchical, or phylogenetic structure."
  Ecography 40:913-929. doi:10.1111/ecog.02881
- Pedregosa F, Varoquaux G, Gramfort A, et al. 2011. "Scikit-learn: Machine
  Learning in Python." Journal of Machine Learning Research 12:2825-2830.
  https://github.com/scikit-learn/scikit-learn

No frontend page uses this analysis yet; a "Methods & References" expander is
required when one does.
