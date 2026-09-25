# Plan: What differs between Penk+ and Penk⁻CamKII+ RSP neurons

Status: research plan written 2026-09-23. Code skeletons and tests are in place
(see §8); no analysis has been run on real data yet.

Companion documents: [neural-hypotheses.md](neural-hypotheses.md) (H-N series),
[stats-strategy.md](stats-strategy.md), [patching-port-plan.md](patching-port-plan.md),
[reference-papers.md](reference-papers.md).

---

## 1. Question and framing

The experiment labels two non-overlapping excitatory populations in the same
region of retrosplenial cortex (RSP) in different mice:

| Population | Genetics | `celltype` | Nature |
| --- | --- | --- | --- |
| Penk+ | Penk-Cre × CamKII-driven Cre-ON GCaMP7f (ADD3 family) | `penk` | one transcriptomically defined type |
| Penk⁻CamKII+ | Penk-Cre × CamKII-driven Cre-OFF GCaMP7f (virus 344) | `nonpenk` | the complement: a mixture of all other excitatory types |

The question is not "are they different" (the prior is that they are) but
*along which axis* the difference is largest and detectable with this dataset.
The asymmetry matters: Penk+ is expected to be compact in any feature space,
Penk⁻CamKII+ is expected to be dispersed and possibly multimodal. Tests of
central tendency are therefore not the only, or even the best, tests.

## 2. What is already known

### 2.1 Dataset

- 12 Penk+ animals (four Cre-ON constructs: ADD3 ×6, ADD3.1 ×2, A160.1+A83 ×2
  Flp-dependent, A122 ×1; one GCaMP8f; one female) and 4 Penk⁻CamKII+ animals
  (all virus 344; one is a Penk×Rbp4 cross whose Cre-OFF label also excludes
  Rbp4+ L5 cells).
- 23 non-excluded sessions (16 vs 7); 12 primary sessions (8 vs 4).
- Soma ROIs: 309 Penk+ (11 animals) vs 141 Penk⁻CamKII+ (4 animals). The
  Penk⁻CamKII+ group is animal-poor, not cell-poor.
- Equipment (SFB/f4mm vs TFB/f6mm) partially co-varies with cell type; the
  equipment-matched subset is 5 Penk+ and 2 Penk⁻CamKII+ animals and has never
  been analysed.
- Imaging depth is not recorded in metadata; injections target superficial
  RSP (DV 0.28–0.57 mm), consistent with layer 2/3. The ex vivo patching cells
  are RSPd layer 2/3.
- The two populations are never in the same animal. Every comparison is
  between-animal; the unit of inference is the animal.

### 2.2 Between-group results to date (FISSA-processed data)

All tests use animal-level Mann-Whitney U and animal-level cluster permutation
(`hm2p.analysis.mixed_stats`). Results from `results/hypothesis_report_fissa/`
and `results/bayes/`:

| Measure | Penk+ | Penk⁻CamKII+ | p (MWU / perm) | CLES | Status |
| --- | --- | --- | --- | --- | --- |
| Movement modulation index (H1.5) | 0.275 | 0.187 | 0.040 / 0.167 | 0.86 | only nominal hit; FDR ns |
| Light modulation index (H3.5) | 0.097 | 0.195 | 0.056 / 0.274 | 0.16 | leans Penk⁻CamKII+ |
| Spatial information, events (H5.2) | 0.133 | 0.080 | 0.138 / 0.217 | 0.77 | leans Penk+ |
| HD-cell fraction per animal | 0.0 % | 10.7 % | 0.177 | BF10 1.24 | only BF10 > 1 |
| HD MVL, tuning width, visual dependence, baseline rate | — | — | all > 0.3 | — | null |
| Behaviour (18 tests) | — | — | none < 0.10 | — | Penk⁻CamKII+ animals explore slightly more |

Minimum detectable effect with 11 vs 4 animals is Cohen's d ≈ 1.5 at 80 %
power; the permutation floor is p ≈ 0.00055. Nulls are bounded, not evidence of
equivalence (BF10 0.5–1.2).

### 2.3 Ex vivo patching (different mice)

37 cells from 6 animals (23 `penkpos`, 14 `penkneg`), RSPd layer 2/3, 50
electrophysiological and morphological metrics
(`results/patching/analysis/metrics.csv`). Cell-level Mann-Whitney with FDR:
spike half-width 1.90 vs 2.52 ms (Penk+ narrower), maximum spike count 22 vs
14 (Penk+ higher), half-Vm shifted, input capacitance about half, fewer basal
trees. Nominal (FDR ns): RMP −73 vs −69 mV, input resistance 160 vs 118 MΩ,
rheobase 80 vs 92 pA, smaller and shallower dendritic trees. With animal as a
random effect nothing survives (`lmm_animal.csv`): the cell-level analysis
pseudoreplicates across six mice. PCA was never run and the morphology is
unused.

### 2.4 Literature

**Granular RSP layer 2/3 is dominated by the low-rheobase (LR) neuron.**
A small pyramidal cell with high input resistance, low rheobase, little
spike-frequency adaptation, spike width intermediate between fast-spiking
interneurons and regular-spiking pyramidal cells, and Cxcl14 expression;
together with a unique L5a type it makes up more than half of granular RSP
neurons, conserved between mouse and rat (Brennan et al. 2020; Jedrasiak-Cape
et al. 2025; Sullivan et al. 2023; bioRxiv 2024.09.17.613545). LR neurons
receive preferential anterior-thalamic and subicular head-direction input and
are biophysically suited to convert sustained HD input into angular head
velocity (Brennan et al. 2020, 2021). They do not fire persistently under
cholinergic drive, unlike neighbouring regular-spiking cells, which predicts
state-independent, transient coding (Jedrasiak-Cape et al. 2025).

**The Penk+ patching phenotype matches the LR description on every measured
axis** (narrow spikes, high maximal rate, high input resistance, low rheobase,
small dendritic trees). Whether Penk marks LR cells has not been established;
the Penk-IRES2-Cre line labels a sparse subset of layer 2 and layer 6
excitatory cells in cortex.

**Penk expression and enkephalin action.** In cortex, Penk mRNA is highest in
VIP interneurons and lower in SST, PV and intratelencephalic pyramidal cells
(Smith et al. 2019). With a CamKII-driven Cre-ON construct the Penk+
population here is excitatory. Enkephalin acts on δ and μ opioid receptors
that are dense on SST and PV interneurons and suppresses their GABA release,
i.e. activity-dependent disinhibition of nearby pyramidal cells.

**Cell-type signatures in calcium imaging.** High-rate, non-adapting cells
produce frequent events and a fluorescence distribution with low skewness;
sparse pyramidal cells produce rare large events and high skewness (Niethard
et al. 2021). Locomotion modulation is cell-type specific and depends on
visual context (Dipoppa et al. 2018).

**RSP directional coding.** Dysgranular RSP carries a landmark-dominated,
sometimes bidirectional HD signal (Jacob et al. 2017); granular RSP carries
global HD. Layer 5 apical tufts encode HD and position differently from their
somata, and L5 somatic rate scales with locomotion and head-rotation speed
(Voigts and Harnett 2020). Anterior thalamic input reaches all RSP pyramidal
cells (Margetts-Smith et al. 2025), so differences between populations must
arise from intrinsic properties or local circuitry. PV and SST interneurons
shape egocentric precision and stability respectively (Oh et al. 2026).

**Model-free population tools.** Persistent homology detects the ring
topology of an HD population (Chaudhuri et al. 2019); CEBRA produces
behaviour-contrastive embeddings (Schneider et al. 2023); catch22 provides a
canonical assumption-free time-series feature set (Lubba et al. 2019);
population coupling quantifies each cell's relation to the ensemble (Okun et
al. 2015).

bioRxiv monitoring (140 daily scans, April–September 2026) found no preprint
on the function of cortical Penk+ neurons.

## 3. Design principles

1. Unit of inference is the animal. Every between-group result is reported at
   four levels: naive cell-level (descriptive only), animal-level
   Mann-Whitney, animal-level cluster permutation, and a linear mixed model
   with ICC as a supplementary check. Direction must survive leave-one-animal-out
   (LOAO) removal of each Penk⁻CamKII+ animal. Effect sizes (CLES, Cliff's
   delta) and confidence intervals are always reported. Nulls are framed as
   bounded.
2. Prefer designs with many within-animal replicates (light transitions,
   movement onsets, junction passes, cross-validated decoders) so that the
   between-group test operates on precise per-animal estimates.
3. Exploit the mixture structure: test dispersion, multimodality and
   classifier separability alongside medians.
4. Amplitude-invariant features first (skewness, inter-event-interval CV,
   autocorrelation time, tuning shape, decoding at matched cell count) because
   expression level and SNR differ between mice. Repeat every result on the
   equipment-matched subset and with virus variant as a covariate within
   Penk+.
5. Match behaviour. Penk⁻CamKII+ animals explore slightly more; tuning
   comparisons use occupancy-, speed- and AHV-matched frames
   (`hm2p.analysis.matched_tuning`).
6. Non-parametric tests only. FDR within pre-declared families.

## 4. Top three candidate differences

These are predictions to be tested, not findings.

1. **Penk+ RSP neurons are the low-rheobase cell type.** The patching
   phenotype matches LR neurons point for point, and LR cells dominate the
   layer being imaged. In vivo prediction: Penk+ cells show higher event
   rate, lower dF/F skewness, more sustained events and stronger AHV coding
   than Penk⁻CamKII+ cells.
2. **A self-motion versus visual coupling dissociation.** Penk+ cells are
   more movement- and AHV-coupled (movement modulation CLES 0.86);
   Penk⁻CamKII+ cells are more light-coupled (light modulation CLES 0.16).
   Prediction: in an encoding model, AHV and speed regressors carry more
   deviance in Penk+ cells and light/visual regressors more in Penk⁻CamKII+
   cells; light-transition transients differ in sign and size.
3. **Classic HD cells live mostly in Penk⁻CamKII+; Penk+ is a compact,
   weakly HD, AHV-conjunctive cluster.** HD-cell fraction per animal is 0 %
   versus 10.7 % (the only measure with BF10 > 1), and the mixture should be
   more dispersed. Prediction: greater dispersion and multimodality in
   Penk⁻CamKII+; a leave-one-animal-out classifier separates the groups above
   chance using kinetics, AHV and HD features.

## 5. Hypotheses

Each hypothesis lists the measure, the method (with the module that
implements it), the statistics, and the follow-up analyses or experiments.

### H1 — Intrinsic identity: Penk+ = low-rheobase neurons (ex vivo)

- Measure: the 50 existing patching metrics plus adaptation index,
  first-spike latency and spike-frequency adaptation ratio.
- Method: `hm2p.patching.lr_classify` classifies each cell as LR, RS or
  ambiguous by Brennan 2020 criteria and tests Penk+ enrichment with a
  within-animal label permutation; `data_driven_lr_axis` runs the never-run
  PCA and tests bimodality of PC1. `hm2p.patching.statistics` gains
  within-animal cluster permutation and a per-animal paired comparison.
  Runner: `scripts/run_patching_celltype.py`.
- Follow-ups: check Penk against Cxcl14 in the Allen Brain Cell Atlas RSP
  layer 2/3 clusters (desk); Penk-Cre × Cxcl14 RNAscope (wet); verify the
  reference thresholds against the published tables before any claim.

### H2 — Calcium activity signature

- Measure per cell: event rate, amplitude, rise/decay/duration, AUC,
  inter-event interval mean and CV, fraction active, plateau fraction, dF/F
  skewness and kurtosis, autocorrelation time, catch22 features.
- Prediction: Penk+ higher rate, lower skewness, longer plateau events,
  shorter inter-event intervals.
- Method: `hm2p.analysis.cell_features.session_feature_table` builds one
  per-cell table with animal, session, cell type, equipment, virus and SNR
  columns (`run_celltype_programme.py h2`). Stats:
  `mixed_stats.run_between_group_test` per feature within its FDR family;
  SNR-matched subsampling; CASCADE spike branch if available.
- Follow-ups: equipment-matched subset; Kruskal-Wallis across virus variants
  within Penk+; dendrite ROI × cell type interaction.

### H3 — Angular head velocity coding

- Measure: AHV tuning curve and modulation index, anticipatory time delay,
  conjunctive HD × AHV, light versus dark (`hm2p.analysis.ahv`).
- Prediction: Penk+ stronger AHV tuning, preserved in darkness.
- Method: persist AHV metrics per cell in the feature table; speed-matched
  comparison via `matched_tuning.match_indices_1d`; animal-level stats.
- Follow-ups: head-turn onset aligned responses; GLM partial deviance for
  AHV (H8); split by turn direction relative to the imaged hemisphere.

### H4 — State dependence: transient versus sustained

- Measure: movement-onset and offset aligned responses (transient index),
  activity decay during immobility, activity versus bout duration,
  syllable-conditioned activity (`hm2p.analysis.state_dynamics`).
- Prediction: Penk+ show onset transients and faster immobility decay;
  Penk⁻CamKII+ sustained.
- Follow-ups: MoSeq syllable information; cholinergic manipulation (wet);
  relation to the movement-modulation lean (H1.5).

### H5 — Light→dark transition dynamics

- Measure: transition-aligned single-cell and population responses, early
  and late amplitude, sign, tuning-recovery time course after dark→light
  (`hm2p.analysis.transitions`). 10–15 transitions per session give precise
  per-animal estimates; this is the best-powered between-group design in the
  dataset and has not been run.
- Prediction: Penk⁻CamKII+ larger light-off transient and slower recovery.
- Follow-ups: dark→light asymmetry; first versus later dark epochs;
  occupancy-matched recovery.

### H6 — Heterogeneity rather than central tendency

- Measure: dispersion around the group centroid, GMM/BIC component count,
  energy distance, LOAO classifier balanced accuracy with permutation
  importance, Penk-like fraction of Penk⁻CamKII+ cells
  (`hm2p.analysis.heterogeneity`).
- Prediction: Penk⁻CamKII+ more dispersed and multimodal; classifier above
  chance; Penk+ overlaps one Penk⁻CamKII+ sub-mode.
- Nulls: cell-type labels permuted at the animal level.
- Follow-ups: unsupervised clustering of pooled cells with composition test;
  whether "Penk-like" Penk⁻CamKII+ cells share HD/AHV properties.

### H7 — Population HD code: decoding and topology

- Measure: cross-validated HD decoding error at matched cell count,
  population-vector correlation, PCA ring score (angle uniformity, angle–HD
  circular correlation), persistent-homology ring score, CEBRA embedding
  consistency, light versus dark (`hm2p.analysis.topology`,
  `hm2p.analysis.decoder`).
- Prediction: Penk⁻CamKII+ sessions decode HD better; Penk+ populations lack
  a clean ring.
- Follow-ups: internal-frame drift in darkness; decode AHV rather than HD in
  Penk+.

### H8 — Encoding models

- Measure: Poisson GLM with circular HD basis, AHV, speed, 2-D position,
  light and syllable regressors; drop-one partial deviance; forward selection
  with paired Wilcoxon across folds (`hm2p.analysis.encoding`; sklearn
  backend now, NEMOS backend optional).
- Prediction: AHV and speed dominate Penk+ profiles; HD and light dominate
  Penk⁻CamKII+ profiles.
- Follow-ups: clustering of encoding profiles (feeds H6); light × variable
  interactions.

### H9 — Network coupling

- Measure: population coupling, noise correlations versus distance, lagged
  cross-correlation with the population (`hm2p.analysis.coupling`).
- Prediction: Penk+ cells are more coupled to the population and lead it
  (positive lag asymmetry), consistent with enkephalin-mediated
  disinhibition.
- Follow-ups: within versus across ROI type; opioid antagonist imaging (wet).

### H10 — Behaviour-coupled navigational coding

- Measure: junction-choice decoding, place information, syllable
  information, familiarity effect (`hm2p.maze.neural`,
  `hm2p.analysis.state_dynamics.syllable_information`).
- Prediction: Penk+ carry more route and self-motion structure;
  Penk⁻CamKII+ more allocentric and visual.
- Follow-ups: light versus dark; per-animal behavioural darkness sensitivity
  against neural measures.

## 6. Model- and assumption-free track

- Omnibus separability: LOAO classifier and energy-distance test on catch22
  plus kinetics features (H6) answer "are they distinguishable at all"
  without choosing an axis.
- Unsupervised structure: rastermap, PCA and CEBRA-time embeddings per
  session; persistent homology for ring topology (H7); GMM component counts.
- Distribution-shape comparisons (variance-ratio permutation) next to every
  location test.

## 7. Confounds and how each is handled

| Confound | Handling |
| --- | --- |
| Different mice, expression, SNR | amplitude-invariant features; SNR-matched subsampling; SNR as covariate |
| Equipment (fibre/lens) | repeat on matched subset (`mixed_stats.equipment_matched_subset`) |
| Four Penk+ virus constructs | Kruskal-Wallis across variants; variant as covariate |
| Behavioural differences | occupancy/speed/AHV matching |
| Animal-level pseudoreplication | animal-level tests, cluster permutation, LOAO |
| Multiple comparisons | BH-FDR within declared families |
| Imaging depth / A–P position unknown | record per session from serial2P where possible; report as limitation |

## 8. Implementation

New modules (all pure functions, no I/O, synthetic-data unit tests):

| Module | Hypotheses |
| --- | --- |
| `src/hm2p/analysis/cell_features.py` | H2, H3, feature table for H6 |
| `src/hm2p/analysis/heterogeneity.py` | H6, omnibus track |
| `src/hm2p/analysis/state_dynamics.py` | H4, H10 (syllables) |
| `src/hm2p/analysis/transitions.py` | H5 |
| `src/hm2p/analysis/coupling.py` | H9 |
| `src/hm2p/analysis/topology.py` | H7 |
| `src/hm2p/analysis/encoding.py` | H8 |
| `src/hm2p/analysis/mixed_stats.py` (extended) | LMM, LOAO, equipment matching |
| `src/hm2p/patching/lr_classify.py`, `patching/statistics.py` (extended) | H1 |

Runners (in `scripts/`, not executed yet): `run_celltype_programme.py`
(CLI, S3 loading, four-level reporting; one subcommand per hypothesis with
outputs under `results/celltype_programme/<hn>/`; `h2` writes the per-cell
feature table that `h6 --features` reuses), `run_celltype_hypotheses.py`
(the per-hypothesis bodies, unit-tested on synthetic sessions), and
`run_patching_celltype.py` (H1). Every subcommand supports `--dry-run` and
`--sessions N` for a smoke run. Optional
dependencies `ripser` and `pycatch22` are in the `analysis` extra; `cebra`
remains a manual install (numpy < 2 pin).

Sequencing: (1) feature table, heterogeneity and omnibus classifier;
(2) patching re-analysis; (3) AHV, state dynamics, transitions;
(4) population, coupling, navigation; (5) encoding models.

## 9. Open items

- Field-of-view A–P position and imaging depth per session.
- Availability of Suite2p ROI centroids for distance-dependent coupling.
- Whether to run CASCADE before H2/H8 so a spike-rate branch exists.

## 10. References

- Brennan EKW, Sudhakar SK, Jedrasiak-Cape I, John TT, Ahmed OJ. 2020.
  "Hyperexcitable Neurons Enable Precise and Persistent Information Encoding
  in the Superficial Retrosplenial Cortex." Cell Reports 30:1598–1612.
  doi:10.1016/j.celrep.2019.12.093
- Brennan EKW, Jedrasiak-Cape I, Kailasa S, Rice SP, Sudhakar SK, Ahmed OJ.
  2021. "Thalamus and claustrum control parallel layer 1 circuits in
  retrosplenial cortex." eLife 10:e62207. doi:10.7554/eLife.62207
- Jedrasiak-Cape I, Rybicki-Kler C, Brooks I, et al. 2025. "Cell-type-specific
  cholinergic control of granular retrosplenial cortex with implications for
  angular velocity coding across brain states." Progress in Neurobiology.
  doi:10.1016/j.pneurobio.2025.102790 (bioRxiv doi:10.1101/2024.06.04.597341)
- Yousuf H, Nye AN, Moyer JR. 2020. "Heterogeneity of Neuronal Firing Type
  and Morphology in Retrosplenial Cortex of Male F344 Rats." Journal of
  Neurophysiology 123:1849–1863. doi:10.1152/jn.00577.2019
- Sullivan KE, Kraus L, Wang L, et al. 2023. "Sharp cell-type-identity
  changes differentiate the retrosplenial cortex from the neocortex." Cell
  Reports 42:112206. doi:10.1016/j.celrep.2023.112206
- "Unique Transcriptomic Cell Types of the Granular Retrosplenial Cortex Are
  Preserved across Mice and Rats despite Dramatic Changes in Key Marker
  Genes." 2025. J Neuroscience (bioRxiv doi:10.1101/2024.09.17.613545)
- Smith SJ, Sümbül U, Graybuck LT, et al. 2019. "Single-cell transcriptomic
  evidence for dense intracortical neuropeptide networks." eLife 8:e47889.
  doi:10.7554/eLife.47889
- Niethard N, Brodt S, Born J. 2021. "Cell-Type-Specific Dynamics of Calcium
  Activity in Cortical Circuits over the Course of Slow-Wave Sleep and Rapid
  Eye Movement Sleep." Journal of Neuroscience 41:4212–4222.
  doi:10.1523/JNEUROSCI.1957-20.2021
- Dipoppa M, Ranson A, Krumin M, Pachitariu M, Carandini M, Harris KD. 2018.
  "Vision and Locomotion Shape the Interactions between Neuron Types in Mouse
  Visual Cortex." Neuron 98:602–615. doi:10.1016/j.neuron.2018.03.037
- Jacob PY, Casali G, Spieser L, Page H, Overington D, Jeffery K. 2017. "An
  independent, landmark-dominated head-direction signal in dysgranular
  retrosplenial cortex." Nature Neuroscience 20:173–175. doi:10.1038/nn.4465
- Voigts J, Harnett MT. 2020. "Somatic and Dendritic Encoding of Spatial
  Variables in Retrosplenial Cortex Indicates Distinct Roles for Dendrites in
  Cortical Computation." Neuron 105:237–245. doi:10.1016/j.neuron.2019.10.016
- Okun M, Steinmetz NA, Cossell L, et al. 2015. "Diverse coupling of neurons
  to populations in sensory cortex." Nature 521:511–515.
  doi:10.1038/nature14273
- Chaudhuri R, Gerçek B, Pandey B, Peyrache A, Fiete I. 2019. "The intrinsic
  attractor manifold and population dynamics of a canonical cognitive circuit
  across waking and sleep." Nature Neuroscience 22:1512–1520.
  doi:10.1038/s41593-019-0460-x
- Schneider S, Lee JH, Mathis MW. 2023. "Learnable latent embeddings for
  joint behavioural and neural analysis." Nature 617:360–368.
  doi:10.1038/s41586-023-06031-6. https://github.com/AdaptiveMotorControlLab/CEBRA
- Lubba CH, Sethi SS, Knaute P, Schultz SR, Fulcher BD, Jones NS. 2019.
  "catch22: CAnonical Time-series CHaracteristics." Data Mining and Knowledge
  Discovery 33:1821–1852. doi:10.1007/s10618-019-00647-x.
  https://github.com/DynamicsAndNeuralSystems/pycatch22
- Hardcastle K, Maheswaranathan N, Ganguli S, Giocomo LM. 2017. "A
  Multiplexed, Heterogeneous, and Adaptive Code for Navigation in Medial
  Entorhinal Cortex." Neuron 94:375–387. doi:10.1016/j.neuron.2017.03.025
- Ojala M, Garriga GC. 2010. "Permutation Tests for Studying Classifier
  Performance." Journal of Machine Learning Research 11:1833–1863.
- Székely GJ, Rizzo ML. 2013. "Energy statistics: A class of statistics based
  on distances." Journal of Statistical Planning and Inference 143:1249–1272.
  doi:10.1016/j.jspi.2013.03.018
- Tralie C, Saul N, Bar-On R. 2018. "Ripser.py: A Lean Persistent Homology
  Library for Python." Journal of Open Source Software 3:925.
  doi:10.21105/joss.00925. https://github.com/scikit-tda/ripser.py
- Margetts-Smith G, et al. 2025. "Dissection of retrosplenial cortex inputs:
  ubiquitous drive from anterior thalamus." bioRxiv
  doi:10.1101/2025.02.06.636939
- Oh D, Yang J, Shin J, Kwag J. 2026. "Retrosplenial PV and SST interneurons
  shape egocentric spatial precision and stability." bioRxiv
  doi:10.1101/2026.05.10.724096

---

## 11. Results of the first full run (2026-09-25)

All 23 non-excluded sessions (11 Penk+ animals, 4 Penk⁻CamKII+ animals; 450
soma ROIs, 309 vs 141), `dff` signal, 10 000 animal-level permutations,
BH-FDR within family. "Animal p" is the animal-level Mann-Whitney U,
"perm p" the animal-level cluster permutation, CLES the common-language
effect size for Penk+ > Penk⁻CamKII+ (0.5 = no difference), "LOAO" whether
the sign survives dropping each Penk⁻CamKII+ animal. Raw tables are under
`results/celltype_programme/<hn>/` (gitignored). Three code defects were
found and fixed during the run (NaN speed frames collapsing the speed index,
an O(n²) autocorrelation, and mixed-length transition time courses); the
reported numbers come from the corrected code.

### Summary

| H | Measure (group means: Penk+ vs Penk⁻CamKII+) | Animal p | Perm p | FDR | CLES | LOAO |
| --- | --- | --- | --- | --- | --- | --- |
| H2 | Inter-event interval 30.9 vs 25.4 s | 0.040 | 0.007 | 0.061 | 0.86 | stable |
| H2 | Event duration 3.27 vs 2.63 s | 0.078 | 0.010 | 0.061 | 0.82 | stable |
| H2 | Event decay time 2.43 vs 2.00 s | 0.078 | 0.011 | 0.061 | 0.82 | stable |
| H2 | Event rise time 0.84 vs 0.63 s | 0.056 | 0.035 | 0.106 | 0.84 | stable |
| H2 | Event rate in light 2.33 vs 2.85 per min | 0.026 | 0.041 | 0.106 | 0.11 | stable |
| H2 | Event amplitude 4.6 vs 7.8 (dF/F units) | 0.026 | 0.044 | 0.106 | 0.11 | stable |
| H2 | Stationary-light mean signal 0.085 vs 0.188 | 0.010 | 0.014 | 0.144 | 0.07 | stable |
| H2 | Event SNR 11.5 vs 11.0 | 0.571 | 0.808 | 0.81 | 0.39 | — |
| H3 | AHV modulation depth 0.19 vs 0.36 | 0.040 | 0.074 | 0.103 | 0.14 | stable |
| H3 | Speed-matched AHV depth, dark − light: −0.03 vs −0.18 | 0.018 | 0.058 | 0.103 | 0.91 | stable |
| H4 | Onset transient, immobility decay, sustained ratio | > 0.49 | > 0.06 | > 0.32 | — | — |
| H5 | Light→dark early transient −0.031 vs −0.096 | 0.078 | 0.135 | 0.20 | 0.82 | stable |
| H5 | Dark→light late response 0.053 vs 0.155 | 0.056 | 0.094 | 0.16 | 0.16 | stable |
| H5 | Tuning recovery times | > 0.33 | > 0.29 | 0.39 | — | — |
| H7 | Matched-N (8 cells) HD decode error 89° vs 87° | 0.41 | 0.28 | 0.79 | — | — |
| H7 | PCA ring angle–HD correlation ≈ 0 in both | > 0.22 | > 0.45 | 0.79 | — | — |
| H9 | Population coupling 0.10 vs 0.05 | 0.226 | 0.105 | 0.21 | 0.73 | stable |
| H9 | Mean noise correlation 0.052 vs 0.008 | 0.138 | 0.094 | 0.21 | 0.77 | stable |
| H10 | Junction-choice decoding above chance 0.08 vs 0.13 | 0.66 | 0.74 | 0.74 | — | — |
| H10 | Place information, familiarity coding | > 0.34 | > 0.46 | 0.74 | — | — |

### H1 (ex vivo patching, `scripts/run_patching_celltype.py`)

The patched cell types segregate by mouse: four mice contributed only Penk+
cells, one only Penk⁻ cells, one both (6 Penk⁻ and 3 Penk+). The comparison
is therefore between animals with effectively 5 versus 2 mice, and the
attainable animal-level permutation p is about 0.13 whatever the effect size.
Cell-level effects remain large and in the LR direction (Cliff's δ: spike
half-width −0.78, maximal spike count +0.69, input capacitance −0.66,
half-Vm −0.68, input resistance +0.37, rheobase −0.3), but none can be
separated from animal identity with these data. LR/RS classification by
Brennan 2020 criteria (three of five met) calls 12 of 23 Penk+ cells LR versus
1 of 14 Penk⁻ cells (Fisher p = 0.011, descriptive; the animal-level
permutation is uninformative for the same reason), and PC1 of the five LR
metrics is bimodal (ΔBIC = 7) with Penk+ and Penk⁻ medians on opposite sides.
Conclusion: the ex vivo phenotype is consistent with Penk+ = LR but the
existing recordings cannot test it at the animal level; the decisive
experiment is paired recording of both cell types in the same slices.

### H6 (heterogeneity and omnibus separability, on the H2 feature table)

447 cells, 55 features (kinetics, trace shape, tuning, catch22), standardised.
Pooled: Penk⁻CamKII+ is *less* dispersed than Penk+ (dispersion ratio
0.77, animal-level permutation p = 0.002), the two distributions differ
(energy distance p = 0.037), a leave-one-animal-out logistic classifier
reaches balanced accuracy 0.69 (permutation p = 0.055; per-animal accuracy
ranges from 0 to 1), and only 9 % of Penk⁻CamKII+ cells have Penk-majority
neighbourhoods against a 69 % baseline. Animal-centred control (each
feature minus its animal mean): the dispersion difference persists (ratio
0.80, p = 0.003) and the energy distance shrinks but stays nominal (0.19,
p = 0.043), while classifier accuracy falls to chance (0.45, p = 0.89). So
the between-group separability in pooled space was carried by animal-level
offsets, whereas the higher within-animal cell-to-cell dispersion of Penk+
is not. This is the opposite of candidate difference 3: the single
transcriptomic type is the more heterogeneous population in activity space,
and the CamKII+ mixture is the more compact one. GMM/BIC gives one component
for Penk+ and five for Penk⁻CamKII+ in pooled space, which is consistent
with a compact but multi-modal mixture; the neighbourhood measure may still
carry session-level structure after centring and is descriptive.

### Reading

1. **The clearest signal is event kinetics (H2).** Penk+ calcium events
   are sparser, longer (slower rise and decay) and smaller than Penk⁻CamKII+
   events, and the light-period event rate is lower. Three kinetics metrics
   reach FDR 0.06 within a 17-metric family, every direction survives
   leave-one-animal-out, and event SNR does not differ between groups.
   SNR-matched subsampling (141 vs 141 cells, matched SNR medians) keeps the
   same pattern (perm p 0.05–0.10, FDR 0.20 with the reduced set). Within
   Penk+, no kinetics metric differs across the four virus constructs at
   FDR level (Kruskal-Wallis p ≥ 0.06). Caveats: decay time and amplitude
   correlate with SNR within cells (rho 0.3–0.4), and dF/F kinetics are
   shaped by indicator expression as well as firing; the CASCADE spike
   branch has not been run, so this remains a calcium-event statement.
2. **Direction of the LR prediction is mixed.** Longer, plateau-like events
   fit a non-adapting cell, but the predicted higher event rate and stronger
   AHV coding in Penk+ are not seen: AHV modulation depth is *higher* in
   Penk⁻CamKII+ (H3), and it falls more in darkness there under speed
   matching (CLES 0.91). The amplitude-normalised AHV index does not differ
   (p = 0.76), so the raw-depth difference partly tracks the amplitude
   difference in H2. H4 state-dependence measures are null.
3. **Light coupling leans Penk⁻CamKII+ consistently** (H2 stationary-light
   signal and light event rate, H3 dark drop, H5 larger transients in both
   directions, and the earlier H3.5 light-modulation lean). None is
   significant alone at the animal level after FDR; the direction is the
   same in every measure.
4. **Population-level measures are null and, for HD, at chance (H7).**
   With eight cells per session neither group decodes HD above chance
   (median error ≈ 88°) and neither shows an HD ring. Population coupling
   and noise correlations lean Penk+ (H9, CLES 0.73–0.77) but are not
   significant at the animal level.
5. **Navigational coding (H10) is null** in both groups; junction-choice
   decoding is barely above chance.

### What this changes

- Candidate difference 1 (Penk+ = LR neuron) is supported by the ex vivo
  phenotype and by event shape, not by rate or AHV coding. The next test is
  H1 (patching re-analysis, `scripts/run_patching_celltype.py`) and the
  CASCADE spike branch for H2.
- Candidate difference 2 (self-motion vs visual coupling) is half
  supported: the visual/light side leans Penk⁻CamKII+ across four
  independent measures; the self-motion side does not lean Penk+.
- Candidate difference 3 (HD cells in Penk⁻CamKII+, Penk+ compact) is
  contradicted on the compactness half: Penk+ is the more dispersed
  population within animals (H6), and the population HD code is at chance
  in both groups at matched N (H7). Whether the Penk+ dispersion reflects
  genuine functional diversity within one type, or the four Cre-ON virus
  constructs, is testable by re-running H6 within the ADD3-only animals.
