# Neural Hypotheses for RSP Calcium Imaging

The behavioural characterisation of the q-rose maze is complete (manuscript v1.0,
June 2026). This document lays out the hypotheses, analysis plans, and priority
ranking for the neural imaging paper: two-photon calcium imaging of Penk+ and
Penk-CamKII+ RSP neurons in freely moving mice under alternating light/dark
epochs.

Date: 2026-06-05
Revised: 2026-06-05 (post-review v2)

---

## 0. Statistical Power and Design Constraints

### 0.1 The 12-vs-4 animal problem

The dataset contains 12 Penk+ and 4 Penk-CamKII+ animals. This imbalance is the
single most important constraint on the neural paper. All between-group (cell type)
comparisons are fundamentally limited by having only 4 animals in the smaller group.

**Power analysis (two-sample Mann-Whitney, alpha = 0.05, two-tailed):**

| N_penk | N_nonpenk | Min detectable d (80% power) | Min detectable d (60% power) |
|--------|-----------|------------------------------|------------------------------|
| 12     | 4         | ~1.5                         | ~1.1                         |
| 11     | 4         | ~1.6                         | ~1.2                         |

These are very large effect sizes. For comparison, Cohen's d = 0.8 is conventionally
"large." The design can only detect effects roughly twice that size. In practical
terms: if Penk+ and Penk-CamKII+ populations have a moderate difference (d ~ 0.5-0.8)
in any HD tuning metric, the study is very unlikely to detect it.

**Implication:** All between-group tests (H-N2, H-N4, H-N5, H-N6, H-N7, H-N8, H-N9)
must be framed as **hypothesis-generating**, not hypothesis-testing. Significant results
are informative (the effect is very large), but non-significant results are
uninformative (the study cannot distinguish a true null from a moderate effect).

**Cluster permutation ceiling:** The maximum number of unique animal-label permutations
is C(16,4) = 1820. The minimum achievable permutation p-value is 1/1820 = 0.00055.
This provides adequate resolution for detecting large effects but no capacity for
multiple comparisons correction across many tests.

**Bayesian supplement:** For all between-group comparisons that yield non-significant
results, report Bayes factors (BF10) computed via the non-parametric Bayesian
Mann-Whitney test (van Doorn et al. 2020; JASP implementation). BF10 < 1/3 provides
evidence for the null; 1/3 < BF10 < 3 is inconclusive; BF10 > 3 supports an effect.
With N = 4, many comparisons will fall in the inconclusive range, and this should
be stated explicitly.

### 0.2 Session selection

Unless stated otherwise, all analyses use **primary, non-excluded sessions only**
(primary_exp = 1 AND exclude = 0). This yields approximately 11 sessions from 11
distinct animals (one session per animal). Sensitivity checks using all non-excluded
sessions (20 sessions, some animals contributing multiple sessions) are reported in
supplementary material with mixed-effects accounting for repeated measures.

### 0.3 Animal-level summaries

All between-group comparisons use **animal-level medians** (not means) as the unit
of analysis. Cell-level distributions are shown for visualisation and within-group
characterisation, but between-group statistical tests are always on animal-level
summaries to avoid pseudoreplication.

### 0.4 Leave-one-animal-out sensitivity

All between-group tests must include leave-one-animal-out (LOAO) sensitivity
analysis. For each animal, recompute the test statistic with that animal removed.
Report the range of p-values and effect sizes. Any result that depends on a single
animal (especially among the 4 Penk-CamKII+ animals) is flagged as fragile.

### 0.5 Lens, fibre, and virus confounds

The imaging hardware evolved over the course of the experiment:

| Equipment | Animals (celltype) |
|-----------|-------------------|
| SFB fibre + f4mm lens | 1114353 (penk), 1114356 (penk), 1115464 (penk), 1115465 (penk), 1115816 (penk), 1116663 (penk), 1116994 (penk), 1117217 (nonpenk), 1117646* (nonpenk) |
| TFB fibre + f4mm lens | 1117646* (nonpenk, later sessions only) |
| TFB fibre + f6mm lens | 1117788 (nonpenk), 1118018 (penk), 1118020 (penk), 1118023 (penk), 1118213 (nonpenk), 1118317 (penk), 1118320 (penk) |

*1117646 straddles the fibre transition (SFB in exp 17, TFB in exps 18-19).

**The confound:** The SFB-to-TFB fibre and f4mm-to-f6mm lens transitions overlap
partially with the cell type grouping. Of the 4 nonpenk animals, 1 is pure SFB/f4mm
(1117217), 1 straddles fibre types (1117646), and 2 are TFB/f6mm (1117788, 1118213).
Meanwhile, 7 of 12 penk animals are SFB/f4mm and 5 are TFB/f6mm.

This means any cell-type difference could be confounded with equipment differences
(different PSF, different FOV size, different neuropil contamination profile). The
f6mm lens has a larger field of view and potentially different optical aberrations
than f4mm. The TFB fibre may transmit differently than SFB.

**Mitigation strategy:**
1. Report all key metrics (MVL, tuning width, SNR, neuropil coefficient) separately
   for each equipment configuration (SFB/f4mm vs TFB/f6mm).
2. Test for equipment main effects on signal quality metrics.
3. For cell-type comparisons, report results from the TFB/f6mm subset separately
   (5 penk + 2 nonpenk animals), which controls for equipment.
4. If cell-type differences are present only in the mixed-equipment analysis but
   absent in the equipment-matched subset, they are likely equipment-driven.

**Within-Penk virus heterogeneity:** The 12 Penk+ animals used 4 different virus
variants: ADD3 (6 animals: 1114353, 1114356, 1115464, 1115465, 1116663, 1118020),
A122 (1 animal: 1115816), A160.1+A83 (2 animals: 1116994, 1118018), and ADD3.1
(2 animals: 1118317, 1118320). All non-Penk animals used virus 344.

While all Penk+ virus variants target the same Cre-ON strategy, expression levels,
onset dynamics, and titre may differ between preparations. This is an additional
source of within-group variance that could mask real effects or, if correlated with
equipment generation, create spurious trends.

**Mitigation:** Report per-animal SNR and event rate, colour-coded by virus variant.
Test for virus variant effects on signal quality within the Penk+ group
(Kruskal-Wallis). If significant, include virus variant as a covariate in
sensitivity analyses.

---

## 1. Behaviour Summary

The behavioural manuscript (v1.0) established four core findings about mouse
navigation in the q-rose maze under light/dark alternation:

1. **Structured local exploration.** Mice explore all 23 cells with systematic
   left-right turn alternation at T-junctions (lag-1 autocorrelation = -0.172,
   p < 0.0001) and a first-order Markov model is preferred over second-order
   in all sessions (0/23).

2. **Coverage drops in darkness.** Per-epoch spatial coverage decreases robustly
   when lights are extinguished (light: 0.400; dark: 0.337; p = 0.0003, r = 0.86,
   N = 23). The effect survives normalisation by active time (p = 0.001, r = 0.78),
   confirming it is not purely locomotor.

3. **Route stereotypy, not global disengagement.** Corridor coverage drops
   strongly (p < 0.0001, r = 0.97), junction coverage drops moderately
   (p = 0.001, r = 0.82), but dead-end coverage is unchanged (p = 0.26).
   Mice maintain their destination repertoire while consolidating onto fewer
   connecting routes. Transition matrices diverge (JSD = 0.068 vs null 0.018,
   p < 0.001), revisitation increases (p = 0.011, r = 0.64), and the visited
   subgraph diameter contracts (p = 0.002, r = 0.80).

4. **Single-trial adaptation.** Route stereotypy appears after a single dark
   epoch: the first dark epoch produces near-normal coverage (0.57 vs subsequent
   mean 0.30; p = 0.0001, r = 0.89). Speed does not change at the light-off
   transition, and first-dark coverage equals first-light coverage, ruling out
   startle or anxiety.

**Implications for the neural paper.** The behavioural findings constrain neural
interpretations in several ways:

- Mice do not freeze or disengage in darkness. Neural changes cannot be
  attributed to gross behavioural shutdown.
- The coverage reduction means mice visit fewer cells in darkness. HD sampling
  may become more constrained in some maze regions, requiring occupancy-corrected
  tuning curves.
- The single-trial adaptation timescale (within 1-2 minutes) is consistent with
  the known timescale of HD drift in darkness (Stackman & Taube 1997; Ajabi et
  al. 2023). Neural and behavioural changes may be causally linked.
- Route stereotypy (preserved destinations, fewer connecting routes) suggests
  the global spatial representation degrades without visual cues while local
  egocentric strategies are intact. This maps onto the distinction between
  visual landmark anchoring and path integration.
- Local turn rules are preserved in darkness, suggesting the vestibular and
  proprioceptive signals supporting egocentric turns remain functional. Any
  RSP subpopulation involved in these local decisions should maintain its
  coding.

---

## 2. Literature Context

### 2.1 RSP anatomy and the HD circuit

The retrosplenial cortex (RSP) lies at the intersection of the head direction
system and the visual system. It receives HD input from the anterior thalamic
nuclei (ADN) via the postsubiculum (PoS) through the canonical circuit
DTN -> LMN -> ADN -> PoS -> RSP (Cullen & Taube 2017; Chaplin & Margrie 2020).
RSP also receives direct visual input from V1 and higher visual areas
(bidirectionally), grid/place cell information from the hippocampal formation
via MEC, and motor planning signals from ACC/M2 (Vann et al. 2009; Mitchell
et al. 2018).

RSP neurons encode head direction (Chen et al. 1994; Cho & Sharp 2001; Jacob
et al. 2017), angular head velocity through vestibular-visual integration
(Keshavarzi et al. 2022), and spatial position through conjunctions of vision
and locomotion (Mao et al. 2020; Fischer et al. 2020). RSP HD cells are
anchored to local visual landmarks (Jacob et al. 2017), and this anchoring
can override path-integration signals.

**A critical anatomical constraint:** Margetts-Smith et al. (2025; bioRxiv
2025.02.06.636939v1) demonstrated that anterior thalamic input to RSP is
ubiquitous across all layers and both granular (RSA) and dysgranular (RSG)
divisions. This means both Penk+ and Penk-CamKII+ populations almost certainly
receive HD-tuned thalamic drive. Cell-type-specific differences in HD tuning,
if they exist, must arise from differential local processing, differential
integration of non-thalamic inputs (visual cortex, subiculum, MEC), or
differential neuromodulatory sensitivity -- not from selective thalamic
connectivity.

### 2.2 HD cells in darkness

When visual cues are removed, HD preferred firing directions drift at variable
rates (Stackman & Taube 1997; Goodridge et al. 1998). The attractor network
structure is preserved -- HD cells maintain their relative phase relationships
even as the entire representation drifts coherently (Ajabi et al. 2023). When
landmarks return, HD representations rapidly re-anchor, often within a single
head sweep (Zugaro et al. 2003). Bicknell et al. (2024) confirmed that ADN-RSP
coordination is maintained in darkness but with increased drift.

Ajabi et al. (2023) demonstrated that the HD population traverses a "second
dimension" (gain modulation) during drift in darkness and reorientation to
landmarks, beyond the classical 1D ring attractor model. Approximately 40% of
HD cells become unstable in acute darkness in the absence of non-visual
anchoring cues (Muir et al. 2022).

Tian et al. (2026) showed degradation of MEC HD cells and border cells under
light deprivation using miniature 2P in freely moving mice, providing a direct
comparison point for RSP data. Jayakumar et al. (2025) demonstrated that HD
cell recalibration during visual-motion conflict occurs only during forward
locomotion, not during head-scanning at rest, implying that PD drift rate
should be analysed conditional on movement state.

### 2.3 RSP L2/3 pyramidal neurons and functional subtypes

The imaging dataset targets superficial (L2/3) excitatory neurons. Two
genetically defined non-overlapping populations are labelled:

- **Penk+ (enkephalin-expressing):** Targeted via Penk-Cre + Cre-ON AAV.
  Penk encodes proenkephalin, the precursor of met-enkephalin and leu-enkephalin.
  In cortex, Penk+ neurons are a subset of excitatory pyramidal cells. Their
  functional role in RSP is completely uncharacterised. Across 33 consecutive
  bioRxiv scan cycles (April-June 2026), no preprint has reported on Penk+ RSP
  neurons in a spatial or HD context.

- **Penk-CamKII+ (non-Penk excitatory):** Targeted via Cre-OFF intersectional
  strategy -- Cre in Penk+ cells blocks expression, labelling only non-Penk
  CamKII+ excitatory neurons. This is the complement of the Penk+ population
  within the CamKII+ (excitatory) class.

The closest precedent for cell-type-specific functional differences in RSP
excitatory neurons comes from Jedrasiak-Cape et al. (2024), who identified LR
(late-spiking regular) neurons in granular RSP with unique cholinergic
modulation and specialised angular head velocity computation. This demonstrates
that molecularly defined RSP excitatory subtypes can have distinct computational
roles. However, Jedrasiak-Cape et al. worked in slice preparations (in vitro);
no study has compared HD coding between molecularly defined RSP excitatory
subtypes in vivo.

Oh et al. (2026) showed that RSP PV and SST interneurons have dissociable roles
in spatial coding: PV cells govern egocentric coding precision via motion-linked
bearing-aligned synchrony, while SST cells govern long-term global stability via
boundary-anchored activity. While this concerns interneurons rather than
excitatory subtypes, it establishes that genetically defined RSP subpopulations
have distinct roles in spatial coding -- and that excitatory cell-type differences
may be shaped by differential inhibitory gating.

### 2.4 Penk+ neurons: what is known elsewhere

Penk+ neurons are better characterised in other brain regions:
- In striatum, Penk+ medium spiny neurons (D2-MSNs) are the "indirect pathway"
  neurons, functionally opposed to D1-MSNs.
- In hippocampus, Penk expression marks a subset of CA1 pyramidal neurons
  (Cembrowski et al. 2016).
- In cortex, Penk is expressed in a subset of deep-layer excitatory neurons
  in some areas (Tasic et al. 2018 Allen Brain Atlas).

The relevance of subcortical Penk function to cortical Penk+ RSP neurons is
unclear. The enkephalin peptide itself modulates mu and delta opioid receptors,
suggesting a neuromodulatory output function. Whether Penk+ RSP neurons are
functionally defined by their opioid peptide release, or whether Penk
expression simply marks a transcriptomically defined subtype, is unknown.

### 2.5 Visual-vestibular integration and optic flow

Chaplin & Margrie (2020) reviewed RSP as a multimodal hub for visual and
self-motion integration. Visual-vestibular integration in V1 (via RSP) is
additive at the membrane potential level (Velez-Fort et al. 2018). This raises
the question of whether RSP integration is also additive -- if so, gain
modulation between light and dark may reflect a simple additive/subtractive
visual contribution rather than complex gating.

The lights-off manipulation removes both visual landmarks and optic flow. This
is important: it eliminates not only the static cues for landmark anchoring but
also the dynamic visual motion signals that contribute to self-motion estimation.
Any neuron that uses optic flow for angular velocity or speed estimation will
be affected, not just landmark-anchored HD cells.

### 2.6 Movement confounds

Movement explains more neural variance than sensory or cognitive variables
across the entire mouse cortex (Musall et al. 2019; Steinmetz et al. 2019;
Stringer et al. 2019; Zagha et al. 2022). RSP is no exception -- during
locomotion, RSP increases in functional connectivity centrality (Nietz et al.
2022). Any comparison between light and dark, or between cell types, must
control for speed, angular head velocity, and movement state. Voigts & Harnett
(2020) confirmed that RSP L5 somatic firing rates increase with both locomotion
speed and head rotation speed.

### 2.7 The dendrite question

Voigts & Harnett (2020) demonstrated that apical tuft dendrites of RSP L5
neurons encode head direction and position differently from their parent
somata. Local dendritic events had distinct HD tuning that was reliable
across split halves but differed from somatic tuning. The hm2p dataset
contains both soma and dendrite ROIs in the same imaging plane, now separable
via a trained XGBoost classifier. If Penk+ and Penk-CamKII+ neurons differ
in dendritic morphology or dendritic integration, soma-dendrite tuning
differences might manifest differently between populations.

### 2.8 Technical context: head-mounted 2P imaging

The data were acquired with a MINI2P-derived head-mounted two-photon microscope
(Zong et al. 2017, 2022). The ~3g headpiece was validated to produce behaviour
indistinguishable from unencumbered controls (Zong et al. 2022). Imaging at
~9.6 Hz with GCaMP6s provides adequate temporal resolution for HD tuning analysis
(HD changes on timescales of hundreds of ms to seconds) but limits spike
inference precision and fast AHV correlation. The axial PSF of miniaturised
optics (~8-12 um) is broader than benchtop systems (~4 um), making neuropil
contamination more severe and neuropil correction more critical (Vickers &
McCormick 2024; Ali & Kwan 2019).

---

## 3. Hypotheses

### Signal type conventions

Unless otherwise specified:
- **HD tuning curves** are computed from **dF/F** (neuropil-corrected, baseline
  as rolling 8th percentile). dF/F is preferred over CASCADE spike rates for
  tuning curves because it preserves graded rate information and avoids CASCADE's
  calibration uncertainties at low SNR.
- **Event rate** analyses (H-N9 factorial, transition responses in H-N6) use
  **CASCADE spike inference** (spikes/s), which is more appropriate for counting
  discrete events and comparing activity magnitudes across cells with different
  baseline fluorescence.
- **Decoding** (H-N7) uses **dF/F**, following the convention of population vector
  analysis on rate maps.
- **Robustness checks** for all key findings repeat the analysis with the
  alternative signal type (dF/F vs CASCADE) and report concordance.

### H-N1: RSP Penk+ and Penk-CamKII+ neurons contain HD-tuned cells

**Statement:** A significant fraction of both Penk+ and Penk-CamKII+ neurons
show non-uniform HD tuning in the q-rose maze.

**Reasoning:** RSP contains HD cells (Chen et al. 1994; Cho & Sharp 2001; Jacob
et al. 2017). Both populations receive ATN input (Margetts-Smith et al. 2025).
We expect to find HD cells in both.

**Prediction:** At least 15-30% of neurons in each population pass the circular
shuffle significance test (p < 0.05, 1000 shuffles). This is the floor for RSP
based on prior reports. If fewer than 15% are HD-tuned, the population may not
be in the canonical HD circuit or the imaging plane may be in a non-HD layer.

**Analysis plan:**
1. Compute HD tuning curves (36 bins, 10 degrees each) for each ROI using
   **dF/F** during active movement (speed > 2.5 cm/s). Occupancy-corrected
   (divide dF/F in each bin by time spent in that bin).
2. Compute mean vector length (MVL) for each tuning curve.
3. Circular shuffle test: circularly shift the dF/F trace by a random offset
   (uniform between 30s and T-30s to avoid edge effects, 1000 shuffles),
   recompute MVL. Cell is HD-tuned if observed MVL > 95th percentile of
   shuffle distribution.
4. Report fraction significant per population (Penk+ vs Penk-CamKII+) at
   the animal level: for each animal, compute fraction of HD-significant cells,
   then report median fraction per cell type.
5. Report MVL distributions for both populations (all cells and significant
   cells separately).
6. **Split-half stability:** Compute tuning curves from interleaved epochs
   (odd-numbered epochs vs even-numbered epochs). Correlation (Spearman) between
   the two tuning curves for each cell provides a stability metric. Cells with
   split-half rho < 0.3 are flagged as unreliable.

**Novelty:** Confirmatory. RSP contains HD cells is well established. But
demonstrating it in these specific genetically defined populations using
head-mounted 2P in a maze (rather than an open field) is a necessary foundation.

**Confounds:** Occupancy bias in the maze (non-uniform HD sampling due to corridor
geometry). Apply occupancy correction (divide dF/F in each bin by time spent in
that bin). Report HD sampling distributions per session and per light condition.

---

### H-N2: HD tuning properties differ between Penk+ and Penk-CamKII+

**Statement:** The two populations differ in one or more of: proportion of
HD-tuned cells, tuning strength (MVL), tuning width (full width at half max),
or preferred direction distribution.

**Reasoning:** If the populations have distinct computational roles, their
tuning statistics should differ. The ATN input is shared (Margetts-Smith et al.
2025), so differences would reflect local processing or differential visual/MEC
inputs.

**Prediction:** Plausible directions include: one population could have sharper
tuning (higher MVL, narrower width) if it is more strongly driven by the
thalamic HD signal and less modulated by other inputs. Alternatively, a population
with more visual integration might show broader tuning due to conjunctive coding.
No strong a priori prediction for direction -- this is genuinely exploratory for
two uncharacterised populations.

**Analysis plan:**
1. Compare MVL distributions: Mann-Whitney U on **animal-level medians** (one
   median MVL per animal). Report rank-biserial r and bootstrap 95% CI.
2. Compare tuning width (FWHM): same test structure.
3. Compare fraction of significant HD cells: compute fraction per animal, then
   Mann-Whitney U on animal-level fractions. (Do NOT use Fisher exact test on
   pooled cell counts -- this is pseudoreplication.)
4. Compare PD distributions: Kuiper's test for circular uniformity within
   each population; Watson's U^2 test for between-population comparison.
5. **LOAO sensitivity:** repeat all comparisons dropping each animal in turn.
6. **Equipment subset:** repeat on TFB/f6mm animals only (5 penk + 2 nonpenk).

**Novelty:** HIGH. No study has compared HD tuning between molecularly defined
RSP excitatory subtypes. A significant difference here is the headline finding.
However, given the power constraints (Section 0), this comparison is
hypothesis-generating: a significant result implies a very large effect, but a
null result is uninformative.

**Confounds:**
- SNR differences: if GCaMP expression differs between viral constructs, one
  population may have systematically higher/lower SNR, inflating or deflating
  apparent tuning. Must report SNR distributions (peak dF/F / baseline std)
  and repeat on SNR-matched subsets.
- Event rate differences: cells with fewer events have noisier tuning estimates.
  Report event rate per population; exclude cells below minimum threshold
  (e.g., < 50 events per session).
- Animal-level confound: 12 Penk+ vs 4 Penk-CamKII+ animals. This is the
  dominant statistical constraint. Animal-level cluster permutation has maximum
  resolution of C(16,4) = 1820 permutations, minimum p ~ 0.0005.
- Session-level confound: cannot compare within-session because populations
  come from different animals (different viral constructs).
- **Equipment confound:** see Section 0.5. Report results from the TFB/f6mm
  equipment-matched subset alongside the full-dataset comparison.
- **Virus heterogeneity within Penk+:** see Section 0.5. Test whether ADD3 vs
  non-ADD3 Penk+ animals differ in tuning metrics (Kruskal-Wallis).

---

### H-N3: HD tuning degrades in darkness (within-cell comparison)

**Statement:** MVL decreases and preferred direction shifts when lights are
extinguished, reflecting loss of visual landmark anchoring.

> **RESULT (2026-06-16) — the stated direction is wrong, and the reverse effect
> does not survive confound controls.**
>
> On the FISSA-uniform, MVL-bug-fixed data, the *raw* within-cell effect runs
> **opposite** to the prediction: MVL and spatial information are **higher in
> dark** than light (dF/F Wilcoxon p=0.0010; events p=0.0003), with PD showing
> no systematic drift (preserved). This surprise prompted a within-session
> paired confound gauntlet (`scripts/run_dark_hypotheses.py`,
> `src/hm2p/analysis/matched_tuning.py`; results in `results/dark_hypotheses/`,
> 23 sessions).
>
> **The raw dark>light MVL effect does not survive.** Equalising the light/dark
> HD-occupancy distribution (A1) or the joint speed×|AHV| distribution (A2)
> between conditions collapses it: A1 p=0.16 (FDR 0.25), A2 p=0.19 (FDR 0.25),
> both NS; median(dark−light) shrinks from the raw value to ~0.036–0.040 (still
> positive, 15/23 sessions dark>light, but underpowered). Sanity checks passed
> (match=none reproduces raw MVL exactly; occupancy matching equalises the HD
> histograms to zero difference). **Interpretation: most of the raw dark>light
> MVL is differential HD *sampling* in darkness, not a coding gain.** The
> behavioural and bleaching confounds listed below are therefore not hypothetical
> — A1/A2 show they account for nearly the whole effect.
>
> Secondary signals also fail once recomputed on the *same* occupancy-matched
> frames (B2/B3 tightened in `run_dark_hypotheses.py`). The apparent **gain**
> does not survive: stored-MVL dark>light (p=0.01) becomes matched-MVL
> median(dark−light) = −0.119, NS (p=0.21) — if anything slightly negative,
> with flat width. The apparent **dark recruitment reverses**: unmatched gave
> more dark-only than light-only HD cells (49 vs 25, McNemar p=0.007), but with
> matched per-condition significance it flips to *more light-only* (60 vs 39,
> p=0.044). So both "surviving" signals were sampling artefacts. Controls that
> held: `hd_confidence` is null light-vs-dark (IR-camera assumption upheld);
> soma↔neuropil ΔMVL not coupled (rho=0.087, p=0.065). **Implication for the
> paper: do not frame darkness as enhancing HD coding — none of the enhancement
> signals survive occupancy/kinematics matching.** See the v3 revision-log entry.

**Reasoning:** This is the expected effect from the entire HD literature
(Stackman & Taube 1997; Jacob et al. 2017; Bicknell et al. 2024). The key
question is whether it occurs in our specific setting (L2/3 RSP, GCaMP, 1-min
epochs, maze).

**Prediction:** MVL drops by 10-30% in darkness. PD shifts by 10-30 degrees
on average across cells. Tuning curve shape (correlation between light and dark
tuning curves) remains moderately high (rho > 0.3), indicating preserved
selectivity despite drift.

**Analysis plan:**
1. Compute tuning curves separately for all light epochs pooled and all dark
   epochs pooled within each session, using **dF/F** with occupancy correction.
2. Within-cell paired Wilcoxon on MVL_light vs MVL_dark (N = number of
   HD-significant cells, pooled across sessions with session as blocking factor).
3. PD shift analysis: compute circular distance between light PD and dark PD
   for each HD-significant cell.
   - **The correct null:** Under random remapping, |PD shift| has a mean of 90
     degrees (uniform circular distribution), not 0 degrees. Therefore, testing
     |PD shift| > 0 is incorrect. Instead:
   - **V-test** (Rayleigh test with specified mean direction = 0): tests whether
     PD shifts cluster around 0 (i.e., preserved anchoring despite noise). A
     significant V-test indicates PD shifts are smaller than expected from random
     remapping.
   - Report the circular mean and mean resultant length of PD shifts. If the
     mean resultant length is high and centred near 0, tuning is preserved.
   - Compare the PD shift distribution against a shuffle null: for each cell,
     compute PD shift from shuffled light/dark epoch assignments (1000 shuffles)
     to generate the expected distribution under no anchoring.
4. Compute tuning curve correlation (**Spearman** on 36-bin tuning vectors)
   between light and dark for each cell. Report distribution.
5. **Interleaved split-half stability control:** compute within-condition
   split-half correlation using odd vs even epochs (e.g., light epoch 1,3,5,...
   vs light epoch 2,4,6,...). This establishes baseline stability within a
   condition. The light-dark correlation should be lower than the within-light
   interleaved split-half.

**Novelty:** Confirmatory for the general effect. But documenting it specifically
in L2/3 RSP with calcium imaging in a complex maze (rather than electrophysiology
in an open field or cylinder) adds value. No prior study has quantified this in
RSP with head-mounted 2P.

**Confounds:**
- Behavioural confound: mice visit fewer cells in darkness (route stereotypy).
  HD sampling distribution may differ. Use occupancy-corrected tuning curves
  and verify that HD coverage is sufficient in both conditions.
- Speed confound: if mice move slower in darkness, fewer high-quality HD samples
  are available. Restrict to speed > 2.5 cm/s in both conditions.
- Bleaching: if fluorescence decays over the session and light epochs are
  systematically earlier, apparent light > dark could be an artefact. Compute
  per-ROI bleaching slope; verify no systematic light/dark bias.
- Z-drift between epochs: the mouse's posture may differ in light vs dark,
  shifting the focal plane. Monitor rigid motion correction residuals across
  epochs.

---

### H-N4: Visual cue dependence differs between Penk+ and Penk-CamKII+ (CORE HYPOTHESIS)

**Statement:** The two populations differ in how much HD tuning degrades in
darkness. One population relies more on visual landmarks (larger MVL drop, larger
PD shift, lower tuning curve correlation) while the other relies more on path
integration (smaller change in dark).

**Reasoning:** This is the central hypothesis of the project. If genetically
defined RSP subpopulations have distinct computational roles, the most likely
dissociation is between visual landmark anchoring and idiothetic (path
integration) HD maintenance. The q-rose maze light/dark paradigm directly
tests this.

**Prediction:** Two plausible scenarios:
- *Penk+ visually anchored:* Penk+ neurons show larger MVL drops and PD
  shifts in darkness than Penk-CamKII+. This would suggest Penk+ neurons
  are more dependent on visual input for HD anchoring.
- *Penk-CamKII+ visually anchored:* The reverse pattern. Less likely a priori
  because CamKII is a broad marker, but cannot be excluded.

A meaningful MVL ratio difference would be > 0.1 between populations (e.g.,
Penk+ MVL_dark/MVL_light = 0.65 vs Penk-CamKII+ = 0.85).

**Analysis plan:**
1. Compute per-cell: MVL ratio (dark/light), PD shift (V-test resultant
   length per animal), tuning curve correlation (light vs dark, **Spearman**).
2. Between-group comparison (Mann-Whitney U) on **animal-level medians** for
   each metric.
3. Cluster permutation test (shuffle animal labels, 1820 permutations).
4. Effect size: rank-biserial r + bootstrap 95% CI.
5. **Composite visual dependence index (VDI):** To reduce multiple comparisons
   across the three correlated metrics (MVL ratio, PD shift resultant length,
   tuning correlation), compute a composite score per cell:
   VDI = z(1 - MVL_ratio) + z(1 - PD_shift_resultant) + z(1 - tuning_corr)
   where z() is a rank-based standardisation (rank / N). Higher VDI = more
   visually dependent. Test cell-type difference on animal-level median VDI
   as the primary test; report individual metrics as decomposition.
   This reduces the between-group comparison to one primary test + three
   supporting decompositions, mitigating multiple comparisons.
6. **LOAO sensitivity:** repeat dropping each animal in turn.
7. **Equipment subset:** repeat on TFB/f6mm animals only.
8. **Bayesian supplement:** report BF10 for the VDI comparison.

**Novelty:** HIGH. This is the most novel hypothesis. No study has tested
whether genetically defined RSP excitatory subpopulations differ in visual
landmark dependence. However, all between-group results are
hypothesis-generating given the power constraints (Section 0).

**Confounds:**
- The 12 vs 4 animal imbalance is the dominant limitation. With only 4
  Penk-CamKII+ animals, the test has limited power. A null result requires
  Bayesian follow-up to assess evidence for equivalence.
- Behavioural differences between animal cohorts: if the 4 Penk-CamKII+
  animals happen to be faster, slower, or explore differently, this confounds
  the neural comparison. Report and compare all behavioural metrics between
  cohorts.
- GCaMP expression: different viral constructs (Cre-ON vs Cre-OFF) may produce
  different expression levels, affecting SNR and apparent tuning. Must compare
  baseline fluorescence and SNR between populations.
- **Equipment confound:** see Section 0.5.

---

### H-N5: PD drift dynamics differ by cell type

**Statement:** Within dark epochs, Penk+ neurons begin drifting from their
light-epoch PD sooner and drift at a faster rate than Penk-CamKII+ neurons
(or vice versa).

**Reasoning:** If one population is more visually anchored, it should lose its
anchor faster when lights are removed. Secer et al. (2025) showed that area 29e
(RSP-adjacent) maintained landmark coupling even when MEC decoupled, suggesting
RSP subpopulations may have heterogeneous drift rates. Jayakumar et al. (2025)
showed drift is locomotion-dependent, so analysis must condition on movement.

**Prediction:** The visually anchored population shows drift onset < 15 seconds
and drift rate > 3 degrees/second. The idiothetic population shows drift onset
> 30 seconds and drift rate < 1.5 degrees/second. (Calibrate against Jacob et al.
2017, who saw ~10 degrees drift in subiculum over comparable timescales.)

**Analysis plan:**
1. For each dark epoch, compute instantaneous PD in sliding 10-second windows
   (1-second step) relative to preceding light-epoch PD.
2. Define drift onset: first window where |PD_dark - PD_light| > 15 degrees
   (threshold calibrated from light-epoch interleaved split-half PD variability).
3. Drift rate: slope of cumulative angular drift over 60-second dark epoch
   (circular regression).
4. Compare drift onset and drift rate between populations (Mann-Whitney U on
   **animal-level medians**).
5. Condition on movement: compute drift rate only during frames with speed
   > 2.5 cm/s (Jayakumar et al. 2025).
6. **Signal type:** Use **dF/F** for sliding-window PD estimation.
7. **LOAO sensitivity.**

**Novelty:** HIGH. Drift dynamics have been studied in ADN and postsubiculum but
not compared between molecularly defined RSP subpopulations.

**Confounds:**
- 10-second windows with ~9.6 Hz imaging and ~15 ROIs per session: very noisy
  single-cell estimates. Pool across all dark epochs within a session for each
  cell. Still may be underpowered for single-cell drift rate.
- The 60-second dark epoch may be too short for slow drift. Use cumulative
  drift across consecutive dark epochs.
- Coordinated population drift (Ajabi et al. 2023) means cells within a
  session drift together. The meaningful comparison is between populations
  across animals, not between individual cells within a session.

---

### H-N6: Light-to-dark transition evokes a cell-type-specific response

**Statement:** The moment lights are extinguished produces a transient response
in RSP neurons (change in activity within the first 5-10 seconds), and this
response differs between Penk+ and Penk-CamKII+ populations.

**Reasoning:** The lights-off event is a salient sensory transition. If Penk+
neurons receive more visual drive, they should show a larger transient
suppression (loss of visual input) or enhancement (prediction error signal).
Dipoppa et al. (2018) showed that context-dependent modulation differs by cell
type in V1. The dark-to-light transition tests the converse: re-anchoring.

**Prediction:** Penk+ neurons show > 20% change in mean activity within 5
seconds of lights-off. Penk-CamKII+ show < 10% change. Asymmetry reverses at
lights-on.

**Analysis plan:**
1. Align all light-to-dark transitions across sessions. Compute peri-transition
   activity (1-second bins, -10s to +30s relative to transition) using
   **CASCADE spike rate** (spikes/s per cell, then averaged across cells per
   animal).
2. Transition response index = median spike rate [0, 5s] / median spike rate
   [-10, -5s].
3. Compare index between populations (Mann-Whitney U on **animal-level medians**).
4. Repeat for dark-to-light transitions separately.
5. Control: compute same analysis on neuropil traces (**dF/F**) to detect visual
   cortex input changes.
6. Control: compare mouse speed in the peri-transition window; include speed as
   covariate.
7. **LOAO sensitivity.**

**Novelty:** HIGH. Transition dynamics at the cell-type level in RSP are
unstudied. The repeated transitions (10-15 per session) provide statistical
power for within-animal estimates.

**Confounds:**
- Mouse may change posture/speed at transition (even though the manuscript showed
  no speed change at the transition, finer-grained analysis of the first few
  seconds may reveal subtle effects).
- Neuropil signal itself changes at the transition (visual cortex afferents to
  RSP stop carrying visual information). Inadequate neuropil correction could
  create artefactual soma-level changes.
- GCaMP dynamics: a 5-second window is only ~48 imaging frames. With GCaMP6s
  decay of ~0.5-1s, fast transient responses may be smeared.

---

### H-N7: Population decoding of HD from RSP neurons

**Statement:** Head direction can be decoded from population activity with
meaningful accuracy, and decoding degrades in darkness.

**Reasoning:** Population decoding is more robust than single-cell analysis for
detecting subtle changes in the HD code (Jacob et al. 2017). If RSP carries a
population-level HD representation, a decoder trained on light-epoch data should
generalise to light test data with < 30 degrees mean absolute error.

**Prediction:** Mean absolute decoding error < 30 degrees in light (for sessions
with > 10 HD-tuned cells). Error increases by 30-50% in darkness. Decoder
trained on light generalises less well to dark, reflecting drift and/or gain
changes.

**Analysis plan:**
1. Population vector analysis (PVA) decoder: for each time bin, decode HD from
   the population **dF/F** vector using the light-epoch tuning curves as
   templates.
2. Cross-validated (8-fold) decoding error in light epochs.
3. Test generalisation: decode dark-epoch HD using light-trained templates.
4. Compute decoding error increase (dark - light) per session.
5. Compare decoding error between Penk+ and Penk-CamKII+ sessions
   (Mann-Whitney U on **animal-level** errors).
6. Also try template-matching decoder and Bayesian decoder as robustness checks.
7. **Cell count matching:** if mean cell count differs between populations,
   subsample to matched counts before comparing decoding accuracy.
8. **LOAO sensitivity.**

**Novelty:** MODERATE. Population HD decoding from RSP has been done (Bicknell et
al. 2024). The cell-type comparison is novel; the light/dark decoding comparison
from freely-moving maze data is novel.

**Confounds:**
- Cell count per session (~15 ROIs average) severely limits decoding accuracy.
  Some sessions may have < 5 HD-tuned cells. Report minimum cell count required
  for meaningful decoding (likely > 8).
- Decoder performance depends on cell count. If Penk+ sessions have more cells
  on average, their decoding will be better regardless of cell properties.
  Must subsample to matched counts.

---

### H-N8: Speed and AHV modulation of RSP neurons

**Statement:** RSP neurons show speed modulation and angular head velocity (AHV)
modulation, and these properties differ between cell types and/or between light
and dark conditions.

**Reasoning:** Voigts & Harnett (2020) showed RSP neurons are modulated by
both locomotion speed and rotation speed. Keshavarzi et al. (2022, same lab)
showed RSP AHV coding through vestibular-visual integration, with visual input
increasing AHV coding gain and SNR. Jedrasiak-Cape et al. (2024) showed
cell-type-specific AHV computation in RSP. If one population receives more
visual-motion (optic flow) input, its AHV coding should degrade more in
darkness.

**Prediction:** Both populations show AHV modulation in light. In darkness, the
visually dependent population loses 30-50% of its AHV modulation depth while
the idiothetic population maintains it (vestibular + proprioceptive sources
persist in darkness).

**Analysis plan:**
1. AHV tuning: compute mean **dF/F** as a function of absolute AHV
   (10 deg/s bins, 0-200 deg/s). Also test signed AHV (CW vs CCW).
2. Speed tuning: mean **dF/F** as a function of speed (1 cm/s bins, 0-20 cm/s).
3. For each cell, compute AHV modulation depth (peak / baseline) and speed
   modulation depth, separately for light and dark.
4. Between-group comparison of modulation depths and their light-dark change
   (Mann-Whitney U on **animal-level medians**).
5. Control: AHV and speed are correlated. Use GLM (NEMOS) with both as
   predictors to isolate independent contributions.
6. **LOAO sensitivity.**

**Novelty:** MODERATE-HIGH. Cell-type-specific AHV coding in RSP in vivo is
novel. The light/dark comparison for AHV by cell type extends Keshavarzi et al.
2022.

**Confounds:**
- Calcium imaging at 9.6 Hz limits AHV resolution. GCaMP6s decay (~0.5-1s)
  smooths fast AHV signals. This biases toward detecting slow AHV modulation.
- AHV sampling may differ between light and dark (the behavioural manuscript
  showed AHV was unchanged, p = 0.177, but this is at the session level).
- Speed and AHV are correlated with HD change rate -- must disentangle.

---

### H-N9: Activity differs by movement state and light condition (2x2 factorial)

**Statement:** RSP neuronal activity (event rate or mean dF/F) shows main
effects of movement (active vs immobile) and light (on vs off), and their
interaction may differ between cell types.

**Reasoning:** Movement modulation is ubiquitous in cortex (Zagha et al. 2022).
Visual input modulates RSP (Chaplin & Margrie 2020). The interaction tests
whether visual input modulates the gain of movement-related activity differently
in each population.

**Prediction:** Movement increases activity (main effect). Light increases
activity (main effect, driven by visual input). The interaction: Penk+ neurons
may show a larger light-dependent gain of movement modulation (movement has
a stronger effect in light than dark) if they are more visual. Penk-CamKII+
should show a smaller light x movement interaction.

**Analysis plan:**
1. Four conditions per cell: moving-light, moving-dark, stationary-light,
   stationary-dark. Compute **CASCADE spike rate** (mean spikes/s) in each.
2. Main effects: paired Wilcoxon on movement-pooled-over-light and
   light-pooled-over-movement.
3. Interaction contrast per cell: (moving_light - stationary_light) -
   (moving_dark - stationary_dark).
4. Test interaction contrast != 0 (one-sample Wilcoxon).
5. Between-group comparison of interaction contrast (Mann-Whitney U on
   **animal-level medians** + cluster permutation).
6. **LOAO sensitivity.**

**Novelty:** LOW for main effects (expected). MODERATE for cell-type x movement
x light interaction -- this specific three-way interaction in RSP is untested.

**Confounds:**
- Definition of "immobile" matters. Threshold of 2.5 cm/s is standard but
  arbitrary. Test robustness at 1, 2.5, and 5 cm/s.
- Immobile periods may be too short in an actively exploring mouse to get
  stable estimates. Report distribution of immobile bout duration.

---

### H-N10: Neuropil HD tuning as control and signal

**Statement:** The neuropil signal in RSP imaging fields carries HD information
(reflecting HD-tuned afferent axons from ADN/PoS), this tuning degrades in
darkness, and the soma-neuropil relationship differs between populations.

**Reasoning:** Kerr et al. (2005) showed neuropil signal is predominantly
axonal. Margetts-Smith et al. (2025) showed ubiquitous ATN input. RSP neuropil
should contain HD-tuned axonal signals. This serves dual purposes: (a) as a
confound control (soma tuning > neuropil tuning confirms genuine somatic HD
coding), and (b) as an independent readout of afferent HD input.

**Prediction:** Neuropil shows significant HD tuning (MVL > shuffle). Neuropil
MVL drops in darkness (reflecting visual input loss in the afferent mix). Soma
MVL exceeds neuropil MVL for genuine HD cells (the soma signal is not just
contamination). The neuropil light/dark ratio may differ between Penk+ and
Penk-CamKII+ fields if the afferent environment differs.

**Analysis plan:**
1. Compute HD tuning curves from the mean neuropil trace (Fneu from Suite2p,
   **dF/F** computed from the neuropil ring) per session, separately for light
   and dark.
2. Neuropil MVL in light vs dark (Wilcoxon).
3. Per-ROI: compare soma MVL vs its local neuropil MVL. The difference quantifies
   genuine somatic HD coding above contamination.
4. Between-group comparison of neuropil tuning properties (Mann-Whitney U on
   **animal-level medians**).
5. Cross-validate: if neuropil-corrected traces show the same cell-type
   differences as raw traces, the result is robust to contamination.
6. **Equipment subset:** Repeat neuropil analysis separately for SFB/f4mm and
   TFB/f6mm sessions, since the different optics produce different PSFs and
   thus different neuropil contamination profiles.

**Novelty:** HIGH as a scientific finding (RSP neuropil HD tuning has not been
characterised). ESSENTIAL as a confound control -- reviewers will ask about
neuropil.

**Confounds:**
- The "neuropil" is Suite2p's annular surround estimate. It is an approximation,
  not a precise anatomical measurement. Different ROI sizes produce different
  neuropil masks.
- Neuropil correction coefficient (0.7 default) may not be optimal for the
  head-mounted 2P PSF. Use Dipoppa et al. (2018) lower-envelope method to
  estimate empirically; compare results at 0.5, 0.7, 0.82, and empirical.
- **Equipment confound:** The f4mm and f6mm lenses have different axial PSFs,
  which directly affects how much neuropil signal contaminates somatic ROIs.
  If the equipment transition correlates with cell type (Section 0.5), neuropil
  contamination differences could masquerade as cell-type differences.

---

### H-N11: Maze exploration correlates with population neural state

**Statement:** Sessions or epochs with higher spatial coverage (from the
behavioural manuscript) show stronger population-level HD coding (higher
mean MVL, lower decoding error).

**Reasoning:** The behavioural manuscript showed that coverage drops in
darkness. If the coverage drop is caused by degradation of the spatial
representation (as hypothesised), then the neural measure of that representation
(HD decoding accuracy) should correlate with the behavioural measure (coverage).

**Prediction:** Spearman correlation between per-epoch coverage and per-epoch
decoding error < -0.3 (more coverage = lower error). The correlation may be
stronger for one cell type than the other.

**Analysis plan:**
1. Compute per-epoch (1-minute) population decoding error and median MVL
   (using **dF/F**-based tuning curves).
2. Compute per-epoch spatial coverage (from behavioural analysis).
3. Spearman correlation between coverage and decoding error across all epochs
   (pool across sessions).
4. Mixed model: epoch-level correlation with session as random effect
   (supplementary only, not primary test).
5. Compare correlation strength between Penk+ and Penk-CamKII+ sessions.

**Novelty:** MODERATE. Linking maze behaviour to neural coding is the
integrative story. This bridges the behaviour and neural papers.

**Confounds:**
- Both coverage and decoding error depend on speed. A correlation could be
  driven entirely by speed as a shared covariate. Partial Spearman correlation
  controlling for speed.
- Epoch-level analysis has high autocorrelation. Use block bootstrap (resample
  epochs in blocks of 3-5) for significance testing.

---

### H-N12: Gain modulation between light and dark

**Statement:** The overall amplitude (gain) of HD tuning curves changes between
light and dark, separately from drift. One population may show gain reduction
(peak firing drops in dark) while the other maintains gain.

**Reasoning:** Ajabi et al. (2023) showed that HD populations vary along a
"second dimension" (gain) during drift. Keshavarzi et al. (2022) showed visual
input increases AHV coding gain. If visual input provides a gain signal to
RSP, its removal should reduce gain.

**Prediction:** Peak dF/F at the preferred direction is 10-30% lower in dark
than light for the visually dependent population. The idiothetic population
maintains peak amplitude.

**Analysis plan:**
1. For each HD-tuned cell, compute peak **dF/F** at PD in light and dark.
2. Gain ratio = peak_dark / peak_light.
3. Between-group comparison of gain ratio (Mann-Whitney U on **animal-level
   medians**).
4. Control for overall activity: normalise by median dF/F across all directions.
5. **LOAO sensitivity.**

**Novelty:** MODERATE. Gain modulation has been described at the population
level (Ajabi et al. 2023) but not compared between molecularly defined subtypes.

---

### H-N13: Junction-specific neural activity

**Statement:** RSP neurons show elevated activity or altered HD tuning at
T-junction decision points compared to straight corridors.

**Reasoning:** Koren Iton et al. (2025, NaviGraph) found that RSP activity
varies systematically across maze decision points. Alexander & Nitz (2015)
showed RSP neurons encode routes and prospective spatial information. If RSP
contributes to navigation decisions, activity at junctions (where a turn
decision is made) may differ from activity in corridors (straight traversal).

**Prediction:** Median activity is 10-20% higher at junctions than in corridors.
HD tuning may be temporarily disrupted at junctions (lower MVL during the
dwell) as the mouse deliberates. One population may show a larger junction
effect if it is more involved in route planning.

**Analysis plan:**
1. Classify each frame as junction, corridor, or dead-end (from the
   discretised maze position in the behavioural pipeline).
2. Compare median **CASCADE spike rate** at junctions vs corridors per cell
   (within-cell Wilcoxon).
3. Compare MVL computed only during junction dwells vs corridor traversals
   (using **dF/F** tuning curves).
4. Between-group comparison of the junction-corridor difference (Mann-Whitney U
   on **animal-level medians**).

**Novelty:** MODERATE. Novel for RSP with cell-type specificity in a maze.

**Confounds:**
- Dwell time at junctions may be very short (~5 frames at 9.6 Hz for a
  0.5s pause). Tuning curves from junction-only frames will be noisy. Must
  aggregate across all junction visits per session.
- Speed differs between junctions (slowing) and corridors (fast traversal).
  Activity differences could be speed-driven. Include speed as covariate.

---

## 4. Penk+ vs Penk-CamKII+ Specific Predictions

### What each population might do, and why

The core question is whether these two excitatory RSP subpopulations occupy
different functional niches in the HD circuit. Given the constraint that both
receive ATN HD input (Margetts-Smith et al. 2025), differences must arise from:

**Differential visual cortex input:**
If Penk+ neurons receive stronger projections from V1/HVAs (or are more
responsive to visual input via their dendritic integration), they would be:
- More strongly anchored to visual landmarks in light
- More disrupted in darkness (larger MVL drop, faster PD drift)
- More responsive at light transitions (larger transient response)
- More modulated by optic flow (larger AHV coding that degrades in dark)

**Differential hippocampal/subicular input:**
If one population receives stronger input from the hippocampal formation (HPF),
it might carry more allocentric spatial information (place-like or contextual
coding) in addition to HD. The Gobbo et al. (2026) framework predicts that
allocentric-supporting neurons are more vulnerable to landmark removal.

**Differential neuromodulatory sensitivity:**
Penk+ neurons express proenkephalin, the precursor to met- and leu-enkephalin.
These endogenous opioid peptides act on mu and delta opioid receptors.
Jedrasiak-Cape et al. (2024) showed that cholinergic modulation differentiates
RSP excitatory subtypes. If Penk+ neurons are differentially sensitive to
opioidergic or cholinergic modulation, state-dependent changes (arousal,
attention, stress from darkness) could produce cell-type-specific effects
independent of visual cue availability.

### The "null hypothesis" must be taken seriously

Given that:
- Both populations receive ATN input
- Both are excitatory L2/3 RSP neurons
- The 12 vs 4 animal design can only detect effects at d > 1.5 (Section 0)

The most likely outcome is that the two populations are similar in their basic
HD tuning properties, with any differences being subtle. A clean null result
(no difference in HD tuning, no difference in visual dependence) is still
publishable if:
1. Both populations are well-characterised (HD fraction, MVL, tuning width, etc.)
2. The light/dark effect is documented at the cell-type level
3. The neuropil control is clean
4. Bayesian analysis (BF10) demonstrates that the data are at least
   inconclusive rather than supportive of the null (BF10 for VDI and MVL
   comparisons)
5. The design constraints are reported transparently: "With 4 nonpenk animals,
   we could only have detected effects of d > 1.5 at 80% power."

### Predictions ranked by testability

| Prediction | Testable? | Power concern |
|---|---|---|
| Both populations contain HD cells | Yes, descriptive | None (within-population) |
| MVL differs between populations | Yes | Severe (12 vs 4 animals, d > 1.5 required) |
| MVL drops more in dark for one type | Yes | Severe (between-group interaction) |
| PD drift rate differs | Marginal | Severe (noisy per-cell, between-group) |
| Transition response differs | Yes | Moderate (many transitions, between-group) |
| AHV modulation differs | Yes | Moderate-severe |
| Junction activity differs | Marginal | Severe (few frames per junction) |

---

## 5. Dendrite Analysis Opportunities

The ROI classifier (XGBoost, trained on Suite2p stat features: aspect_ratio,
compact, skew, npix_norm, radius) separates soma from dendrite ROIs in the
same imaging plane. This enables several analyses:

### 5.1 Soma vs dendrite HD tuning comparison

**Hypothesis:** Following Voigts & Harnett (2020), dendritic ROIs show HD
tuning that differs from somatic tuning in the same imaging plane.

**Analysis:** Compare MVL, PD, and tuning width between soma and dendrite ROIs.
Dendrite ROIs should show: (a) lower MVL on average (more neuropil contamination
in elongated ROIs, and dendritic transients include local events with different
tuning), (b) possibly different PD than nearby soma ROIs.

**Caveat:** We cannot identify soma-dendrite pairs from the same neuron in
single-plane imaging. Dendrite ROIs likely come from multiple different neurons,
including neurons whose somata are in different planes. The comparison is
population-level (soma ROIs vs dendrite ROIs), not within-neuron.

### 5.2 Dendrite signals in light vs dark

**Hypothesis:** Dendritic HD tuning degrades differently from somatic tuning in
darkness. If dendrites carry more afferent (visual cortex) input, dendritic
tuning may be more visually dependent.

**Analysis:** Compare light-dark MVL ratio for soma vs dendrite ROIs. If
dendritic tuning drops more in dark, this suggests the dendritic compartment
receives visual input that is lost in darkness while the soma maintains HD
coding through thalamic input.

**Caveat:** Dendritic ROIs have lower SNR (smaller, elongated, more neuropil
contamination). Apparent differences in light/dark stability could reflect SNR
differences. Must match soma and dendrite ROIs by SNR.

### 5.3 Cell-type x compartment interaction

**Hypothesis:** The soma-dendrite tuning difference is larger for one cell type
than the other, suggesting different dendritic integration.

**Analysis:** 2x2 comparison: (Penk+ soma vs Penk+ dendrite) vs (Penk-CamKII+
soma vs Penk-CamKII+ dendrite). Test the interaction.

**Caveat:** Very small N. With ~15 ROIs per session and a soma/dendrite split,
dendrite ROI counts per session may be < 5. Pool across sessions per cell type.

### 5.4 Dendrite ROIs as quality control

**Analysis:** Verify that the XGBoost classifier assignment is sensible:
- Dendrite ROIs should have higher aspect ratio, lower compactness, lower skew
- Soma ROIs should have higher skew (more burst-like transients)
- If any "soma" ROI has dendrite-like morphology but high HD tuning, verify
  manually in the Suite2p GUI

### 5.5 Dendritic neuropil contamination

**Analysis:** Dendrite ROIs are more susceptible to neuropil contamination
(elongated shapes mean more neuropil within the ROI footprint). Compare the
neuropil correction coefficient (Dipoppa et al. 2018 lower-envelope method)
between soma and dendrite ROIs. If dendrite ROIs have a higher contamination
fraction, this informs the appropriate correction.

### Priority assessment

Dendrite analyses are **Tier 2 (supporting)** -- interesting and enabled by the
classifier, but not the core story. They strengthen the paper by:
1. Demonstrating that the classifier works and produces sensible results
2. Providing a control (soma > dendrite tuning confirms genuine somatic coding)
3. Adding a Voigts & Harnett replication angle

They should appear as one supplementary figure with 3-4 panels, not as a main
figure.

---

## 6. Priority Analysis Order

Ranked by (a) novelty, (b) feasibility with current data, (c) impact.

### Tier 1: Must-have for the paper (primary figures)

| Priority | Hypothesis | Novel? | Feasible? | Impact |
|---|---|---|---|---|
| 1 | **H-N1: HD cells in both populations** | Confirmatory | High | Foundation |
| 2 | **H-N3: HD tuning degrades in dark** | Confirmatory | High | Foundation |
| 3 | **H-N4: Visual dependence differs by cell type** | HIGH | Moderate (power) | Core finding |
| 4 | **H-N10: Neuropil control** | HIGH | High | Essential control |
| 5 | **H-N9: 2x2 factorial (movement x light x type)** | Moderate | High | Contextualises |

These form the core of the paper: establish HD coding, show light/dark effect,
test cell-type difference, control for neuropil, characterise basic activity.

### Tier 2: Strengthens the paper (supporting figures or panels)

| Priority | Hypothesis | Novel? | Feasible? | Impact |
|---|---|---|---|---|
| 6 | **H-N6: Transition response by type** | HIGH | High | Strong test |
| 7 | **H-N7: Population decoding** | Moderate | Moderate (cell count) | Population-level |
| 8 | **H-N5: PD drift dynamics** | HIGH | Low (noisy) | Deep mechanistic |
| 9 | **H-N8: Speed/AHV modulation** | Moderate-high | High | Completes picture |
| 10 | **H-N12: Gain modulation** | Moderate | High | Ajabi connection |

### Tier 3: Exploratory / supplementary

| Priority | Hypothesis | Novel? | Feasible? | Impact |
|---|---|---|---|---|
| 11 | **H-N2: Tuning properties differ** | High | Moderate (power) | Descriptive |
| 12 | **H-N11: Coverage-neural correlation** | Moderate | Moderate | Integrative |
| 13 | **H-N13: Junction activity** | Moderate | Low (sparse data) | Exploratory |
| 14 | **Dendrite analyses (Sec. 5)** | Moderate | Moderate | Supporting |

### Recommended figure plan

| Figure | Content | Hypotheses |
|---|---|---|
| **Fig. 1** | HD cells in RSP: tuning curves, MVL distributions, PD distributions, example cells | H-N1, H-N2 |
| **Fig. 2** | Light/dark comparison: MVL drop, PD shift, tuning correlation, split-half control | H-N3 |
| **Fig. 3** | Cell-type comparison: VDI by population, decomposed into MVL ratio, PD shift, corr | H-N4 (core) |
| **Fig. 4** | Transition dynamics: peri-transition activity by cell type | H-N6 |
| **Fig. 5** | Population decoding: light vs dark, by cell type | H-N7, H-N12 |
| **Fig. S1** | Neuropil control: HD tuning in neuropil, soma-neuropil comparison | H-N10 |
| **Fig. S2** | Activity factorial: movement x light x cell type | H-N9 |
| **Fig. S3** | Speed and AHV modulation by cell type and condition | H-N8 |
| **Fig. S4** | Dendrite analysis: soma vs dendrite tuning, classifier validation | Sec. 5 |
| **Fig. S5** | PD drift dynamics (exploratory) | H-N5 |
| **Fig. S6** | Signal quality and equipment controls: SNR, event rate, bleaching, lens/fibre | Confound checklist |
| **Fig. S7** | Coverage-neural correlation, junction activity | H-N11, H-N13 |
| **Fig. S8** | LOAO sensitivity and Bayesian supplements | Section 0 |

### Analysis execution order

Start with analyses that provide the foundation and controls before testing
the core hypothesis:

1. **Signal quality characterisation** -- SNR, event rate, peak dF/F, bleaching
   slope by population AND by equipment configuration (SFB/f4mm vs TFB/f6mm).
   This is needed before any comparison can be interpreted. Include virus variant
   comparison within Penk+ (Kruskal-Wallis on SNR by virus).
2. **H-N1** -- Establish HD cells exist. Use dF/F, 1000 shuffles, interleaved
   split-half.
3. **H-N2** -- Basic tuning property comparison between cell types (animal-level
   medians, Mann-Whitney U). This provides the descriptive foundation before
   testing visual dependence. Run early to determine whether populations differ
   at baseline before dark-epoch analysis.
4. **H-N10** -- Neuropil control. Do this before H-N3/H-N4 because it determines
   whether subsequent analyses are trustworthy. Report separately for equipment
   configurations.
5. **H-N3** -- Light/dark HD tuning change. Confirm the basic effect. Use V-test
   for PD shift, Spearman for tuning correlation, interleaved split-half.
6. **H-N9** -- 2x2 activity factorial. Basic activity characterisation using
   CASCADE spike rate.
7. **H-N4** -- Core cell-type comparison. Compute VDI, Mann-Whitney U on
   animal-level medians, cluster permutation, LOAO, equipment subset, BF10.
8. **H-N6** -- Transition responses using CASCADE spike rate. Quick to compute,
   high novelty.
9. **H-N7** -- Population decoding with dF/F.
10. **H-N8** -- Speed/AHV modulation with dF/F.
11. **H-N5** -- Drift dynamics (if feasible after Steps 1-10).
12. **H-N12** -- Gain modulation.
13. **H-N11, H-N13, dendrite analyses** -- Exploratory.

---

## 7. Key Confound Checklist (applies to all analyses)

Every finding must be evaluated against these confounds before publication:

### Signal quality confounds

| Confound | Why | Control |
|---|---|---|
| SNR | Low-SNR cells have noisier tuning -> lower apparent MVL | Report distributions; repeat on SNR-matched subsets |
| Event rate | Cells with few events have unreliable tuning | Report distributions; exclude < 50 events; rate-match |
| Peak dF/F | Expression differences between viral constructs | Report distributions; z-score dF/F within session |
| Bleaching | Fluorescence decay biases late (dark) epochs | Compute per-ROI slope; check light/dark epoch ordering |
| Neuropil coefficient | 0.7 default may not suit head-mounted 2P | Empirical estimation; test sensitivity at 0.5, 0.7, 0.82 |
| Motion artefact | Residual motion after Suite2p correction | Check dF/F-displacement correlation; compare epochs |

### Behavioural confounds

| Confound | Why | Control |
|---|---|---|
| Speed | Mice move differently in light/dark | Restrict to speed > 2.5 cm/s; include speed covariate |
| HD sampling | Non-uniform in maze; may differ by condition | Occupancy-corrected tuning curves; report distributions |
| Route stereotypy | Fewer cells visited in dark -> fewer HD samples | Report HD bin occupancy per condition |
| Bad behaviour | Tether entanglement -> artefactual immobility | Exclude bad_behav frames |
| Movement state | Moving vs immobile affects coding | Analyse separately; report fraction active |

### Design confounds

| Confound | Why | Control |
|---|---|---|
| Animal imbalance | 12 Penk+ vs 4 Penk-CamKII+ animals | Animal-level tests; power analysis (Sec 0); BF10; LOAO |
| Between-animal | Populations come from different animals | Cannot separate cell-type from animal effects |
| Sex | 15 male, 1 female (Penk+) | Report; exclude the female in sensitivity analysis |
| Injection site | AP/ML/DV variation | Report coordinates; check for systematic differences |
| Session order | Multiple sessions per animal | First-session independence check; primary sessions only as default |
| Z-drift | Focal plane may shift between epochs | Monitor motion correction residuals |

### Equipment and virus confounds (new)

| Confound | Why | Control |
|---|---|---|
| Fibre type (SFB vs TFB) | Different transmission, partially confounded with cell type | Report by equipment group; TFB/f6mm subset analysis |
| Lens (f4mm vs f6mm) | Different FOV, PSF, aberrations; f6mm may have different neuropil profile | Report separately; equipment-matched subset |
| Fibre-lens-celltype confound | 2 of 4 nonpenk are TFB/f6mm; 7 of 12 penk are SFB/f4mm | Equipment-matched subset (TFB/f6mm: 5 penk + 2 nonpenk) |
| Virus heterogeneity (Penk+) | ADD3, A122, A160.1+A83, ADD3.1 may differ in expression | Report SNR by virus; Kruskal-Wallis; exclude outlier variants |
| Virus confound (between groups) | All nonpenk use virus 344; Penk+ use 4 variants | Cannot disentangle from cell type; acknowledge as limitation |

---

## 8. References

Ajabi Z, Keinath AT, Brandon MP. 2023. "Population dynamics of head-direction
neurons during drift and reorientation." *Nature* 615, 892-899.

Ali F, Kwan AC. 2019. "Interpreting in vivo calcium signals from neuronal cell
bodies, axons, and dendrites: a review." *Neurophotonics* 7(1), 011402.

Alexander AS, Nitz DA. 2015. "Retrosplenial cortex maps the conjunction of
internal and external spaces." *Nat. Neurosci.* 18, 1143-1151.

Bicknell BA, van der Goes MSH, et al. 2024. "Coordinated head direction
representations in mouse anterodorsal thalamic nucleus and retrosplenial
cortex." *eLife* 13, e82952.

Cembrowski MS, Bachman JL, Wang L, et al. 2016. "Spatial gene-expression
gradients underlie prominent heterogeneity of CA1 pyramidal neurons." *Neuron*
89, 351-368.

Chaplin TA, Margrie TW. 2020. "Cortical circuits for integration of self-motion
and visual-motion signals." *Curr. Opin. Neurobiol.* 60, 122-128.

Chen LL, Lin LH, Green EJ, et al. 1994. "Head-direction cells in the rat
posterior cortex." *Exp. Brain Res.* 101, 8-23.

Cho J, Sharp PE. 2001. "Head direction, place, and movement correlates for
cells in the rat retrosplenial cortex." *Behav. Neurosci.* 115, 3-25.

Cullen KE, Taube JS. 2017. "Our sense of direction: progress, controversies
and challenges." *Nat. Neurosci.* 20, 1465-1473.

Dipoppa M, et al. 2018. "Vision and locomotion shape the interactions between
neuron types in mouse visual cortex." *Neuron* 98, 602-615.

Fischer LF, Mojica Soto-Albors R, Buck F, Harnett MT. 2020. "Representation
of visual landmarks in retrosplenial cortex." *eLife* 9, e51458.

Gobbo F, et al. 2026. "Navigational strategy dictates hippocampal representation
of space in an everyday memory task." *bioRxiv* 2025.05.10.653115.

Goodridge JP, Dudchenko PA, Worboys KA, et al. 1998. "Cue control and head
direction cells." *Behav. Neurosci.* 112, 749-761.

Jacob P-Y, Casali G, et al. 2017. "An independent, landmark-dominated
head-direction signal in dysgranular retrosplenial cortex." *Nat. Neurosci.*
20, 173-175.

Jayakumar RP, et al. 2025. "Path-integration recalibration is
locomotion-dependent." *Current Biology* (in press).

Jedrasiak-Cape I, et al. 2024. "Cell-type-specific angular head velocity
computation in retrosplenial cortex." *J. Neurosci.* (in press).

Kerr JND, Greenberg D, Helmchen F. 2005. "Imaging input and output of
neocortical networks in vivo." *PNAS* 102, 14063-14068.

Keshavarzi S, et al. 2022. "Multisensory coding of angular head velocity in
the retrosplenial cortex." *Neuron* 110, 532-543.

Koren Iton A, et al. 2025. "NaviGraph: A graph-based framework for multimodal
analysis of spatial decision-making." *bioRxiv* 2025.05.18.654725.

Mao D, et al. 2020. "Sparse orthogonal population representation of spatial
context in the retrosplenial cortex." *Nat. Commun.* 11, 3110.

Margetts-Smith G, Andrianova L, Kohli S, Randall AD, Aggleton JP, Witton J,
Craig MT. 2025. "Dissection of retrosplenial cortex inputs: ubiquitous drive
from anterior thalamus." *bioRxiv* 2025.02.06.636939v1.

Mitchell AS, et al. 2018. "Retrosplenial cortex and its role in spatial
cognition." *Brain Neurosci. Adv.* 2, 2398212818757098.

Muir GM, et al. 2022. "Flexible cue anchoring strategies enable stable head
direction coding in both sighted and blind animals." *Nat. Commun.* 13, 5604.

Musall S, Kaufman MT, Juavinett AL, et al. 2019. "Single-trial neural dynamics
are dominated by richly varied movements." *Nat. Neurosci.* 22, 1677-1686.

Oh K, et al. 2026. "PV/SST interneuron dissociation in RSC egocentric spatial
coding." *bioRxiv* 2026.05.10.724096.

Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth show
rapid learning, sudden insight, and efficient exploration." *eLife* 10, e66175.

Secer M, et al. 2025. "Coherence-based coupling between entorhinal cortex and
retrosplenial area 29e during spatial navigation." *bioRxiv* (preprint, details
from bioRxiv scan 2026-04-02; exact DOI to be confirmed from scan archive).

Stackman RW, Taube JS. 1997. "Firing properties of head direction cells in
the rat anterior thalamic nucleus." *J. Neurosci.* 17, 9020-9037.

Stringer C, et al. 2019. "Spontaneous behaviors drive multidimensional,
brainwide activity." *Science* 364, eaav7893.

Tian FF, et al. 2026. "Degradation of MEC spatial codes under light
deprivation." *bioRxiv* (preprint, details from bioRxiv scan; exact DOI to be
confirmed from scan archive).

van Doorn J, Ly A, Marsman M, Wagenmakers E-J. 2020. "Bayesian rank-based
hypothesis testing for the rank sum test, the signed rank test, and Spearman's
rho." *J. Appl. Stat.* 47, 2984-3006.

Vann SD, Aggleton JP, Maguire EA. 2009. "What does the retrosplenial cortex
do?" *Nat. Rev. Neurosci.* 10, 792-802.

Velez-Fort M, et al. 2018. "A circuit for integration of head- and
visual-motion signals in layer 6 of mouse primary visual cortex." *Neuron*
98, 179-191.

Vickers E, McCormick DA. 2024. "Neuropil contamination in two-photon
imaging." *Neurophotonics* (in press; exact citation to be confirmed).

Voigts J, Harnett MT. 2020. "Somatic and dendritic encoding of spatial
variables in retrosplenial cortex differs during 2D navigation." *Neuron*
105, 237-245.

Wei Z, et al. 2025. "Anterior-posterior functional gradient in retrosplenial
cortex." *Nat. Commun.* (doi:10.1038/s41467-026-70762-z).

Zagha E, et al. 2022. "The importance of accounting for movement when relating
neuronal activity to sensory and cognitive processes." *J. Neurosci.* 42,
1375-1382.

Zong W, et al. 2017. "Fast high-resolution miniature two-photon microscopy for
brain imaging in freely behaving mice." *Nat. Methods* 14, 713-719.

Zong W, et al. 2022. "Large-scale two-photon calcium imaging in freely moving
mice." *Cell* 185, 1240-1256.

Zugaro MB, et al. 2003. "Rapid spatial reorientation and head direction cells."
*J. Neurosci.* 23, 3478-3482.

---

## 9. Revision Log

### v3 (2026-06-16, post-results)

First pass with results on the FISSA-uniform, MVL-bug-fixed dataset (all 26
sessions reprocessed; Stage 6 re-run).

1. **H-N3 sign reversed and headline retired.** The raw within-cell effect is
   dark > light for both MVL and spatial information (not the predicted
   degradation). A within-session paired confound gauntlet
   (`scripts/run_dark_hypotheses.py`, `src/hm2p/analysis/matched_tuning.py`,
   23 sessions) shows the raw effect **does not survive** occupancy-matching
   (A1, p=0.15) or speed×|AHV|-matching (A2, p=0.20) — it is largely differential
   HD sampling in darkness, not a coding gain. The secondary signals fail too
   when recomputed on matched curves: the "gain" disappears (matched MVL NS,
   slightly negative) and the apparent dark recruitment reverses to favour light
   (B3 matched, McNemar p=0.044). Do not frame darkness as enhancing HD coding.
   See the result box under H-N3.

2. **Between-group cell-type story remains underpowered/null.** H-N2, H-N4 and
   the other Penk+ vs Penk-CamKII+ contrasts are null at the animal level
   (N=11 vs 4), consistent with the Section 0.1 power limit. The paper's spine
   shifts toward a single-population, within-session light/dark account.

3. **New within-session paired hypotheses proposed** (occupancy/kinematics-matched
   MVL, gain-vs-sharpening, recruitment, soma↔neuropil coupling, maze-position
   specificity). These supplement Section 3 and are the basis of the v3 gauntlet;
   they are not yet folded into the numbered H-N list here.

4. **Controls that held:** `hd_confidence` does not differ light vs dark
   (IR-camera assumption upheld); any residual MVL enhancement is gain, not
   tuning sharpening (width flat).

### v2 (2026-06-05, post-review)

Changes made in response to data scientist and QA reviews:

**Accepted and incorporated:**

1. **Power analysis (Section 0.1).** Added formal power framing: d > 1.5
   required for 80% power. All between-group tests framed as
   hypothesis-generating, not hypothesis-testing. (Data scientist point 1)

2. **PD shift null corrected (H-N3 step 3).** The null for random remapping is
   90 degrees, not 0. Replaced incorrect one-sample Wilcoxon on |PD shift| > 0
   with V-test (mean direction = 0) and shuffle comparison. (Data scientist
   point 2)

3. **Fisher exact test replaced (H-N2 step 3).** Pooling cells across animals
   is pseudoreplication. Replaced with animal-level fractions + Mann-Whitney U.
   Applied same fix throughout document (all between-group comparisons now
   explicitly on animal-level medians). (Data scientist point 3)

4. **Lens/fibre/virus confounds (Section 0.5, Section 7).** Added full
   equipment mapping table, documented the partial confound with cell type,
   and specified TFB/f6mm equipment-matched subset analysis. Added within-Penk
   virus heterogeneity (ADD3, A122, A160.1+A83, ADD3.1) as a confound.
   (Data scientist points 4, 13; QA point 2)

5. **Signal type specified per analysis (Section 3 preamble and each H-N).**
   dF/F for tuning curves and decoding; CASCADE spike rate for event rates and
   transitions. Robustness checks with alternate signal. (Data scientist point 5)

6. **H-N2 moved earlier in execution order (Section 6).** Now step 3, after
   H-N1 and before neuropil control. Rationale: need to know baseline tuning
   differences before interpreting light/dark effects. (Data scientist point 6)

7. **Composite visual dependence index (H-N4 step 5).** Added VDI combining
   MVL ratio + PD shift resultant + tuning correlation into one rank-normalised
   score. Reduces the primary between-group test to one composite + three
   decompositions. (Data scientist point 7)

8. **Shuffle count increased to 1000** throughout (was 500). (Data scientist
   point 8)

9. **Split-half changed to interleaved** (odd vs even epochs, not sequential
   halves). Applied to H-N1 step 6 and H-N3 step 5. (Data scientist point 9)

10. **Bayesian supplement added (Section 0.1, H-N4 step 8).** BF10 via
    non-parametric Bayesian Mann-Whitney for all between-group null results.
    (Data scientist point 10)

11. **LOAO sensitivity added (Section 0.4, every between-group hypothesis).**
    Required for all between-group tests. (Data scientist point 11)

12. **Session selection specified (Section 0.2).** Default is primary +
    non-excluded sessions; all non-excluded in supplementary. (Data scientist
    point 12)

13. **Median not mean** for all animal-level summaries. Changed throughout
    document. (Data scientist point 14)

14. **Pearson replaced with Spearman** for tuning curve correlation (H-N3
    step 4, H-N4 step 1). Non-parametric mandate. (QA point 3)

15. **"Inferred from synthesis" references fixed.** Margetts-Smith et al. (2025)
    verified as real (bioRxiv 2025.02.06.636939v1) and given full citation.
    Secer, Tian, and other inferred references replaced with best-available
    citation information or marked as requiring DOI confirmation from scan
    archives. Laurent et al. and Rogers et al. removed (not cited in this
    document). (QA points 1, 4)

**Note on framing:** The overall document now reflects the reality that
between-group comparisons are hypothesis-generating given the sample size.
Within-group characterisation (H-N1, H-N3 within each type) remains
well-powered and hypothesis-testing.
