# New Hypotheses from Literature Synthesis

Literature-driven hypothesis generation for the hm2p project, based on
15 paper summaries, 2 bioRxiv scans (2026-04-02, 2026-04-04), the neuropil
literature review, and the existing analysis pipeline status.

Date: 2026-04-02

---

## Part 1: Synthesis — State of Knowledge and Gaps

### What we know

**RSP contains HD cells and integrates visual and self-motion cues.** This is
well-established across species (Cho & Sharp 2001; Chen et al. 1994; Jacob
et al. 2017; Fischer et al. 2020). RSP receives converging inputs from the
HD circuit (ADN/PoS), visual cortex, hippocampal formation, and anterior
cingulate cortex (Chaplin & Margrie 2020). The circuit pathway DTN -> LMN ->
ADN -> PoS -> RSP is canonical (Cullen & Taube 2017).

**All RSP pyramidal neurons receive anterior thalamic input.** Margetts-Smith
et al. (2025) showed that ATN input is ubiquitous across granular and
dysgranular RSP, regardless of layer. This means both Penk+ and Penk-CamKII+
populations likely receive HD-tuned thalamic drive. Cell-type-specific
differences in HD tuning, if they exist, must arise from differential local
processing or differential integration of non-thalamic inputs (visual cortex,
subiculum), not from selective ATN connectivity.

**RSP has an anterior-posterior functional gradient.** Wei et al. (2025) showed
that anterior RSP neurons have sharper position tuning and prefer fast visual
stimuli, while posterior RSP neurons are broader and prefer slower motion. The
A-P position of our imaging plane constrains interpretation.

**RSP contains multidirectional cells.** Laurent et al. (2025), from Jacob's
group, identified RSP cells with room-specific directional tuning alongside
classical HD cells with stable preferred direction across environments. The
q-rose maze geometry (radial arms from a central hub) could support similar
heterogeneity.

**Visual landmark anchoring is a learned, stable RSP function.** The bioRxiv
landmark learning paper showed that RSP landmark-referenced activity
increases with learning and remains stable once established. Our mice are
well-trained, so landmark representations should be mature.

**HD circuit topology is intrinsic; visual anchoring is layered on later.**
The developmental toroidal topology paper (2026 bioRxiv) showed HD ring
manifolds emerge by P9, before eye opening, and visual anchoring develops
with navigational experience. This predicts that lights-off should not
destroy HD tuning but should disrupt its visual anchoring (drift from
landmarks).

**MEC HD cells are disrupted by darkness.** Tian et al. (2026) showed that
MEC HD cells and border cells degrade with light deprivation (miniature 2P
in freely-moving mice). This provides a direct comparison point for our RSP
data.

**Path-integration recalibration is locomotion-dependent.** Jayakumar et al.
(2025) showed that HD cell recalibration during visual-motion conflict occurs
only during forward locomotion, not during head-scanning at rest. PD drift
rate in darkness should correlate with movement state.

**Genetically-defined RSP excitatory subtypes can differ in computational
roles.** Jedrasiak-Cape et al. (2024) identified a distinct excitatory cell
type in granular RSC (LR neurons) with unique cholinergic modulation and
specialised angular head velocity computation. This is the strongest existing
precedent for our core hypothesis.

**RSP is a contextual integration hub, not purely spatial.** Rogers et al.
(2026) showed RSP flexibly remodels aversive representations across contexts.
Bech et al. (2026) showed RSP discriminates context during sensorimotor tasks.
Our light/dark alternation is itself a contextual manipulation.

**Movement-related neural activity is ubiquitous.** Zagha et al. (2022),
Stringer et al. (2019), and Musall et al. (2019) showed that movement
explains more neural variance than sensory or cognitive variables across
cortex. Every light/dark or cell-type comparison must account for movement
confounds.

### What we do not know

**Whether genetically-defined RSP excitatory subpopulations differ in HD
coding properties.** No published study has compared HD tuning between
molecularly-defined RSP excitatory subtypes. The Penk+ population is entirely
uncharacterised functionally. The bioRxiv scans confirm this gap remains open
as of April 2026.

**Whether RSP subpopulations differ in visual landmark dependence.** The
cell-type-specific contribution to visual anchoring versus path integration
in RSP has not been dissected. Jedrasiak-Cape et al. (2024) showed AHV
computation differs by cell type; the analogous question for HD visual
anchoring is untested.

**How neuropil contamination interacts with cell-type comparisons in RSP.**
The neuropil in RSP carries HD-tuned afferent input from ADN/PoS (Kerr et al.
2005). The miniature 2P axial PSF (~8-12 um) makes contamination more severe
than in bench-top systems (Helmchen & Denk 2005; Vickers & McCormick 2024).
No study has assessed whether neuropil correction adequacy differs between
genetically-defined subpopulations in RSP.

**Whether RSP contains cells with arm-specific or maze-location-dependent HD
tuning.** Laurent et al. (2025) found multidirectional cells in multi-room
RSC, but the q-rose maze is a continuous structure with multiple radiating arms
rather than discrete rooms. Whether similar heterogeneity emerges in a radial
maze is unknown.

**What role RSP plays in navigation decisions at choice points.** The
egocentric pursuit paper (2026 bioRxiv) showed that RSP allocentric HD tuning
decreases during goal-directed behaviour while egocentric target coding
increases. Whether RSP activity predicts turning decisions in a maze, and
whether this differs by cell type, is untested.

### Where the hm2p dataset is uniquely positioned

1. **Cell-type specificity + light/dark manipulation.** No other dataset
   combines genetically-identified RSP subpopulations with systematic visual
   cue removal in freely-moving mice. This directly tests whether
   subpopulations have distinct roles in visual versus idiothetic HD anchoring.

2. **Freely-moving in a structured maze.** The q-rose maze provides natural
   choice points, dead ends, and forced turns — richer spatial structure
   than open fields or linear tracks. This enables navigation-related analyses
   not possible in simpler environments.

3. **Rapid, repeated light/dark transitions.** The 1-min alternation provides
   multiple within-session replicates of the visual cue removal, enabling
   drift dynamics analysis and light-to-dark transition time courses.

4. **Two non-overlapping populations from the same brain region.** Both
   Penk+ and Penk-CamKII+ are excitatory RSP neurons, enabling controlled
   comparison within the same circuit. This is more specific than
   excitatory vs inhibitory comparisons.

---

## Part 2: New Hypothesis Ideas

The hypotheses below go beyond what is already implemented in the hypotheses
page (which covers: Penk+ vs Penk-CamKII+ overall activity, light vs dark
signal level, moving vs stationary signal level, and modulation indices for
these comparisons). They also go beyond the existing analysis modules (HD
tuning, decoding, stability, AHV, speed, gain, anchoring, information,
MoSeq syllables).

### H-NEW-1: Penk+ neurons show faster PD drift onset in darkness than Penk-CamKII+ neurons

**Hypothesis:** When visual cues are removed, Penk+ neurons begin drifting
from their light-epoch preferred direction sooner (shorter latency to
drift onset) and drift at a higher rate (degrees per second) than
Penk-CamKII+ neurons. This would indicate that Penk+ neurons are more
dependent on visual landmarks for HD anchoring.

**Motivating literature:**
- Secer et al. (2025): Area 29e (RSP-adjacent) maintained landmark coupling
  even when MEC decoupled. If Penk+ neurons correspond to a landmark-
  anchoring subpopulation, they should be most affected by landmark removal.
- Jayakumar et al. (2025): Path-integration recalibration is locomotion-
  dependent. Drift rate should be analysed conditional on movement state.
- Tian et al. (2026): MEC HD cells degrade in darkness. RSP subpopulations
  may show differential degradation.

**Analysis:**
1. For each dark epoch, compute the instantaneous PD in sliding 10-second
   windows (with 1-second step) relative to the preceding light-epoch PD.
2. Define drift onset as the first window where |PD_dark - PD_light| exceeds
   15 degrees (approximately 1.5x the jitter expected from noise — calibrate
   this threshold using light-epoch split-half PD variability).
3. Compute drift rate as the slope of cumulative angular drift over the
   60-second dark epoch (circular regression).
4. Compare drift onset latency and drift rate between Penk+ and Penk-CamKII+
   using Mann-Whitney U at the session level.
5. Control for movement speed by computing drift rate conditional on the
   animal being in motion (speed > 2.5 cm/s).

**Expected result if true:** Penk+ drift onset < 15 seconds; Penk-CamKII+
drift onset > 30 seconds. Penk+ drift rate > 3 degrees/second; Penk-CamKII+
drift rate < 1.5 degrees/second.

**Key confounds:**
- Neuropil contamination: neuropil HD tuning may drift differently from
  somatic tuning. Must verify with neuropil-only analysis.
- Low cell counts per session (~15 ROIs average). Drift rate estimation
  on single cells is noisy. Pool across dark epochs within session.
- Movement speed differences between conditions: if mice slow down in
  darkness, reduced angular sampling could appear as drift.
- The 60-second dark epoch may be too short for slow drift to reach
  detectable levels. Use cumulative drift across consecutive dark epochs.

**Priority: HIGH.** This is the most direct test of the core hypothesis and
would be the headline finding if confirmed. Novel because no study has
compared drift dynamics between genetically-defined RSP subpopulations.

---

### H-NEW-2: Population-level HD decoding from Penk-CamKII+ neurons is more robust to darkness than decoding from Penk+ neurons

**Hypothesis:** A Bayesian or template-matching HD decoder trained on
Penk-CamKII+ population activity maintains higher accuracy in dark epochs
(relative to its light-epoch performance) than a decoder trained on Penk+
activity. This tests whether one population better maintains the population
HD code via path integration.

**Motivating literature:**
- Jacob et al. (2017): Population decoding of HD is more robust than
  single-cell analysis for detecting subtle changes in tuning.
- Gonzalez et al. (2026): CA1-RSC subspaces are stable across brain states,
  suggesting population-level structure is maintained.
- Developmental toroidal topology paper (2026): The ring attractor is
  intrinsic; population decoding should detect its maintenance even when
  single cells drift.

**Analysis:**
1. Train separate decoders on light-epoch data for Penk+ and Penk-CamKII+
   populations (where both exist in the same session, this is a direct
   within-session comparison).
2. Test on dark-epoch data. Compute decoding error (mean absolute circular
   error) for each population.
3. Compute a "darkness robustness index" = 1 - (error_dark / error_light).
   Values near 1 indicate maintained decoding; near 0 indicates collapse.
4. Compare robustness indices between populations with Mann-Whitney U.

**Expected result if true:** Penk-CamKII+ robustness index > 0.6;
Penk+ robustness index < 0.4.

**Key confounds:**
- Cell count asymmetry: 12 Penk+ animals vs 4 Penk-CamKII+. Decoder
  performance depends on cell count. Must match cell numbers (subsample
  the larger population) or use per-session within-animal comparisons
  where both types are present.
  CRITICAL: Penk+ and Penk-CamKII+ are non-overlapping populations from
  DIFFERENT animals (different viral constructs). They cannot be compared
  within-session. This limits the analysis to between-animal comparisons,
  which are confounded by animal-level differences.
- Behavioural differences: mice may behave differently across animals,
  and the 4 nonpenk animals are a small sample.

**Priority: MEDIUM.** The between-animal design and 12 vs 4 imbalance
severely constrain statistical power. This analysis is worth doing but
will likely be underpowered for a definitive conclusion. Report as
supporting/suggestive rather than primary evidence.

---

### H-NEW-3: RSP neuropil HD tuning degrades more in darkness than somatic HD tuning, and this degradation differs between Penk+ and Penk-CamKII+ imaging fields

**Hypothesis:** The neuropil signal in RSP imaging fields shows HD tuning
(reflecting afferent HD input from ADN/PoS), and this tuning degrades more
in darkness than somatic tuning. Furthermore, the neuropil degradation
pattern differs between Penk+ and Penk-CamKII+ imaging fields, reflecting
different afferent input environments.

**Motivating literature:**
- Kerr et al. (2005): Neuropil signal is predominantly axonal, reflecting
  local afferent input.
- Margetts-Smith et al. (2025): ATN input is ubiquitous to all RSP pyramids.
  Neuropil HD tuning should be present in all fields.
- Secer et al. (2025): Area 29e maintained landmark coupling — the neuropil
  may reflect whether the local afferent population is more landmark- or
  path-integration-dependent.

**Analysis:**
1. Compute HD tuning curves for the mean neuropil signal (Fneu from Suite2p)
   in each session, separately for light and dark epochs.
2. Quantify neuropil HD tuning strength (MVL of neuropil tuning curve).
3. Compare neuropil MVL_light vs MVL_dark with Wilcoxon signed-rank.
4. Compare the neuropil tuning degradation (MVL_dark / MVL_light) between
   Penk+ and Penk-CamKII+ fields with Mann-Whitney U.
5. Cross-reference: for each soma ROI, compare its HD tuning with its
   local neuropil tuning. The difference (soma MVL - neuropil MVL) quantifies
   how much somatic tuning exceeds what could be explained by contamination.

**Expected result if true:** Neuropil MVL_light > neuropil MVL_dark
(confirming visual input contributes to neuropil HD signal). Soma MVL -
neuropil MVL > 0 for genuine HD cells. The neuropil degradation may differ
between Penk+ and Penk-CamKII+ fields if the local afferent composition
differs.

**Key confounds:**
- The "neuropil signal" is field-averaged Fneu from Suite2p, which reflects
  the annular surround of all ROIs. It is not a precise anatomical measure.
- The neuropil signal changes between conditions could reflect overall
  activity changes (arousal, movement) rather than HD-specific changes.
  Control by examining neuropil signal variance and mean level across
  conditions.
- Different FOV locations across sessions could mean different neuropil
  composition, adding between-session variance.

**Priority: HIGH.** This is primarily a control analysis, but it has
independent scientific value. Demonstrating that RSP neuropil carries
HD-tuned afferent input and that this input degrades with visual cue removal
would be a novel observation. It directly addresses the neuropil
contamination confound that reviewers will raise, turning a weakness into
a strength.

---

### H-NEW-4: Penk+ and Penk-CamKII+ neurons differ in angular head velocity tuning, with one population better encoding AHV in darkness

**Hypothesis:** Drawing on Jedrasiak-Cape et al. (2024), who showed that
genetically-distinct RSP excitatory subtypes differ in AHV computation,
Penk+ and Penk-CamKII+ neurons differ in their AHV tuning profiles.
Specifically, one population may show AHV tuning that is robust to
darkness (maintained by vestibular/proprioceptive input) while the other
degrades (if its AHV tuning depends on optic flow cues).

**Motivating literature:**
- Jedrasiak-Cape et al. (2024): LR neurons in granular RSC compute AHV
  differently from other excitatory subtypes due to cholinergic modulation.
- Chaplin & Margrie (2020): RSP receives vestibular signals via anterior
  thalamus and motor efference from ACC/M2.
- Voigts & Harnett (2020): RSP L5 neurons show speed and rotation modulation.

**Analysis:**
1. Compute AHV tuning curves for each neuron in light and dark epochs.
   Use the absolute AHV (unsigned) binned into 10-degree/s bins from 0
   to 200 degrees/s.
2. For each neuron, compute AHV modulation depth (peak rate / baseline rate)
   in light and dark.
3. Compare AHV modulation depth between Penk+ and Penk-CamKII+ with
   Mann-Whitney U.
4. Compare the light-to-dark change in AHV modulation between populations.
5. Additionally, examine signed AHV (clockwise vs counterclockwise) for
   asymmetric AHV tuning — some HD cells show stronger responses to one
   rotation direction.

**Expected result if true:** Both populations show AHV modulation in
light. In darkness, one population (possibly Penk-CamKII+, if more
vestibular-dependent) maintains AHV tuning while the other (Penk+, if
more visual-motion-dependent) loses it.

**Key confounds:**
- AHV is correlated with locomotion speed and with HD change rate.
  Use partial correlation or GLM to isolate AHV effects from speed.
- Calcium imaging at 9.6 Hz limits temporal resolution for fast AHV
  events. The actual AHV signal is smoothed by GCaMP kinetics (~0.5 s
  decay), making it difficult to resolve AHV tuning for rapid head turns.
- AHV sampling may differ between light and dark if mice turn at different
  rates.

**Priority: MEDIUM.** AHV tuning analysis is already implemented but the
cell-type x light/dark interaction for AHV is not explicitly tested. This
is testable with existing code and would complement the HD tuning story.
The Jedrasiak-Cape et al. precedent makes it well-motivated but the calcium
imaging temporal resolution limits sensitivity.

---

### H-NEW-5: Multidirectional tuning in the q-rose maze differs between Penk+ and Penk-CamKII+ populations

**Hypothesis:** Following Laurent et al. (2025), some RSP neurons show
arm-specific preferred directions in the q-rose maze rather than a single
global PD. Penk+ and Penk-CamKII+ populations differ in the proportion
of globally-tuned versus arm-specific HD cells.

**Motivating literature:**
- Laurent et al. (2025): RSC contains both classical HD cells and
  multidirectional cells with environment-specific PD. The q-rose maze
  arms could function as distinct spatial contexts.
- Active pursuit paper (2026 bioRxiv): RSP HD tuning decreases during
  goal-directed movement — suggesting contextual modulation.

**Analysis:**
1. Segment the q-rose maze into discrete zones: central hub, each radiating
   arm, and dead ends.
2. For each neuron, compute HD tuning curves separately within each zone
   (requires sufficient occupancy per zone per direction bin — likely
   only the central hub and the 2-3 most-visited arms will have adequate
   sampling).
3. Compute PD for each zone. Define "multidirectional" as cells where
   zone-specific PDs differ by >45 degrees from the global PD.
4. Compare the proportion of multidirectional cells between Penk+ and
   Penk-CamKII+ with a Fisher exact test or chi-squared test.

**Expected result if true:** 10-30% of RSP neurons are multidirectional
(following Laurent et al. 2025). If Penk+ neurons are enriched for
landmark-referenced coding (as hypothesised), they may show more
arm-specific tuning because each arm has different visual landmarks.
Penk-CamKII+ neurons, if more path-integration-dependent, should maintain
a stable global PD regardless of maze arm.

**Key confounds:**
- Occupancy per zone per direction bin will be very sparse. The rose
  maze has 23 accessible cells, but the animal's time is not uniformly
  distributed. Many arm-direction combinations will have too few samples.
- The analysis requires sufficient heading variability within each maze
  zone. In narrow arms, heading is constrained to two directions (inbound
  and outbound), limiting the ability to construct full tuning curves.
- With ~15 ROIs per session, per-session chi-squared tests will be
  underpowered. Must pool across sessions by cell type.

**Priority: LOW.** Conceptually interesting but likely underpowered with
the current dataset. The q-rose maze arms are narrow, constraining heading
variability within each arm. This analysis would work better in an
open field or multi-room environment. Report as exploratory if anything
emerges, but do not build the paper narrative around it.

---

### H-NEW-6: The light-to-dark transition evokes a cell-type-specific transient response in RSP, with Penk+ neurons showing a larger gain change in the first 5 seconds

**Hypothesis:** The moment lights turn off is a salient sensory event. RSP
neurons should show a transient response to this transition (analogous to
Dipoppa et al. 2018 showing context-dependent modulation). Penk+ neurons,
if more visually-driven, show a larger transient response (either
suppression or enhancement) in the first 5 seconds after lights-off
compared to Penk-CamKII+ neurons.

**Motivating literature:**
- Dipoppa et al. (2018): Locomotion effects on neural activity depend on
  visual context — Sst cells reverse modulation between grey screen and
  darkness.
- Rogers et al. (2026): RSP flexibly remodels representations across
  contexts.
- Chaplin & Margrie (2020): RSP receives direct visual input and sends
  feedback to V1.

**Analysis:**
1. Align all light-to-dark transitions (and dark-to-light) across sessions.
2. For each neuron, compute peri-transition activity (mean dF/F in 1-second
   bins from -10 to +30 seconds relative to transition).
3. Define a "transition response index" = mean activity [0, 5s] / mean
   activity [-10, -5s] (ratio of post-transition to pre-transition baseline).
4. Compare transition response indices between Penk+ and Penk-CamKII+ with
   Mann-Whitney U.
5. Separately analyse light-on transitions (dark-to-light) — these test
   whether Penk+ neurons show a larger re-anchoring signal when visual cues
   reappear.

**Expected result if true:** Penk+ neurons show a >20% change in activity
within 5 seconds of lights-off (either suppression due to lost visual
drive, or enhancement from prediction error). Penk-CamKII+ neurons show
<10% change. The asymmetry reverses at lights-on transitions.

**Key confounds:**
- Mice may startle or freeze when lights change, producing a movement
  artefact. Must compare mouse speed in the peri-transition window and
  include speed as a covariate.
- The transition itself may cause a change in neuropil signal (visual
  cortex input to RSP drops), which could contaminate somatic signals.
  Control by computing the same analysis on neuropil traces.
- Pupil dilation in darkness could affect fluorescence collection (wider
  pupil = more ambient light for the 2P detector? Unlikely with 920 nm
  excitation but should be considered).

**Priority: HIGH.** This analysis is straightforward to implement using
existing data, has clear predictions, and exploits the repeated light/dark
transitions. If Penk+ neurons show a distinct transition signature, this
is strong evidence for differential visual dependence. The transition time
course is a natural figure panel.

---

### H-NEW-7: Penk+ neurons show tighter coupling between HD tuning and locomotion speed than Penk-CamKII+ neurons

**Hypothesis:** If Penk+ neurons integrate visual and self-motion cues
(Chaplin & Margrie 2020), their HD tuning should be modulated by locomotion
speed (higher speed = more optic flow = stronger visual anchoring = sharper
tuning). Penk-CamKII+ neurons, if more reliant on path integration, should
show less speed-dependent HD tuning because vestibular signals are less
speed-dependent.

**Motivating literature:**
- Voigts & Harnett (2020): RSP L5 soma firing rates increase with both
  speed and rotation speed.
- Zagha et al. (2022): Movement dominates neural variance; must model
  speed effects explicitly.
- Chaplin & Margrie (2020): Visual-vestibular integration in V1 is additive;
  analogous gain effects may exist in RSP.

**Analysis:**
1. Bin each session's data by locomotion speed quartile (Q1: <2 cm/s,
   Q2: 2-5, Q3: 5-10, Q4: >10 cm/s).
2. Compute HD tuning curves within each speed bin, for light epochs only.
3. Compute MVL per speed bin. Plot MVL vs speed for each neuron.
4. Fit a Spearman correlation between speed and MVL for each neuron.
5. Compare the distribution of speed-MVL correlations between Penk+ and
   Penk-CamKII+ with Mann-Whitney U.
6. Repeat for dark epochs: the prediction is that the speed-MVL relationship
   weakens in darkness for Penk+ (because optic flow is absent) but
   persists for Penk-CamKII+ (if vestibular signals drive their speed
   modulation).

**Expected result if true:** Penk+ speed-MVL correlation > 0.3 in light,
drops to ~0 in dark. Penk-CamKII+ speed-MVL correlation ~0.2 in both
conditions.

**Key confounds:**
- Speed bins must have sufficient heading coverage. At very low speeds,
  mice may not sample all directions, inflating apparent MVL.
- Speed and HD are not independent — mice tend to turn while moving.
  Partial out AHV when computing speed effects.
- Occupancy correction is essential: normalise tuning curves by occupancy
  within each speed bin.

**Priority: MEDIUM.** Speed modulation analysis is partially implemented
(speed.py). The cell-type x light/dark interaction for speed-dependent
HD tuning is novel and directly tests the visual integration hypothesis.
Feasibility depends on having enough data per speed x direction bin.

---

### H-NEW-8: The neuropil-to-soma HD information ratio is higher for Penk+ neurons, indicating greater susceptibility to afferent contamination

**Hypothesis:** If Penk+ neurons have lower endogenous firing rates, sparser
activity, or smaller somata than Penk-CamKII+ neurons, their somatic
signals will have a lower signal-to-neuropil ratio, making their apparent
HD tuning more contaminated by neuropil-derived HD information.

**Motivating literature:**
- Ali & Kwan (2019): Neurons with sparse firing are more susceptible to
  neuropil contamination because their true somatic signal is small relative
  to the neuropil contribution.
- Dipoppa et al. (2018): Neuropil correction coefficient varies across
  experiments; cells with low skewness are more contaminated.
- Kerr et al. (2005): Neuropil fluctuations are 10-30% dF/F, comparable to
  single-spike transients.

**Analysis:**
1. For each soma ROI, compute the HD mutual information in the raw trace
   (before neuropil subtraction), the neuropil trace, and the corrected
   trace.
2. Define a contamination ratio: MI_neuropil / MI_corrected. Values near 1
   indicate the somatic HD information is entirely explainable by neuropil.
   Values near 0 indicate genuine somatic HD tuning.
3. Compare contamination ratios between Penk+ and Penk-CamKII+ with
   Mann-Whitney U.
4. Also compare fluorescence skewness between populations (higher skewness
   = sparser firing = more susceptible to contamination per Dipoppa et al.
   2018).

**Expected result if true:** If Penk+ neurons have higher contamination
ratios, any apparent HD tuning advantage for Penk+ must be interpreted
cautiously. If Penk-CamKII+ neurons have higher contamination ratios, the
reverse applies.

**Key confounds:**
- The neuropil ring in Suite2p samples different tissue for each ROI. ROIs
  near the edge of the FOV have truncated neuropil rings.
- Mutual information estimation is noisy with low cell counts and low frame
  rates. Use bias-corrected MI estimators (Skaggs correction already
  implemented).
- This analysis does not directly prove contamination — a high neuropil MI
  could reflect genuine afferent input that drives the cell's HD tuning
  through synaptic transmission, not contamination.

**Priority: HIGH.** This is an essential confound control that should be
included regardless of other findings. Reviewer 2 will ask whether
cell-type differences survive neuropil correction. Having this analysis
ready preempts the objection. It also has independent value: demonstrating
that RSP neuropil carries HD information from the thalamic input pathway
is novel and connects to Kerr et al. (2005) and Margetts-Smith et al.
(2025).

---

### H-NEW-9: HD coding stability within dark epochs follows a non-monotonic pattern, with an initial drift followed by partial re-anchoring

**Hypothesis:** Drawing on the ultraslow oscillation result from Sarramone
et al. (2026), PD drift in darkness may not be monotonically increasing.
Instead, the HD system may show periods of faster and slower drift, or
even partial re-anchoring via non-visual cues (maze wall contact,
proprioception). If so, the drift trajectory within 60-second dark epochs
should be non-monotonic.

**Motivating literature:**
- Sarramone et al. (2026): Grid cell drifting in darkness has an oscillatory
  structure at ultraslow timescales (<0.01 Hz, period >100 s).
- Jayakumar et al. (2025): HD recalibration depends on locomotion state.
  Pauses with head scanning may produce re-anchoring from non-visual cues.
- Tian et al. (2026): Whisker trimming disrupts MEC HD cells, suggesting
  tactile cues contribute to HD stability. Our q-rose maze walls provide
  tactile landmarks even in darkness.

**Analysis:**
1. For each dark epoch and each significantly HD-tuned neuron, compute the
   PD in sliding 5-second windows (1-second step).
2. Plot the cumulative PD drift trajectory for each dark epoch.
3. Test for non-monotonicity: compute the number of sign changes in the
   drift derivative (d(PD)/dt). Compare observed sign changes to a
   monotonic drift null model (permutation test).
4. If non-monotonic drift is detected, correlate re-anchoring events with
   behavioural events (wall contact, turns at junctions, speed changes).
5. Compare drift trajectory shape between Penk+ and Penk-CamKII+.

**Expected result if true:** PD drift shows 1-2 sign reversals per 60-second
dark epoch, coinciding with maze wall contacts or major turns. Penk-CamKII+
neurons may show more re-anchoring events if they are more responsive to
non-visual spatial cues.

**Key confounds:**
- 60-second epochs are too short for one full cycle of the ultraslow
  oscillations described by Sarramone et al. (period >100 s). We can
  detect non-monotonicity but cannot confirm oscillatory structure.
- With 9.6 Hz imaging and 5-second windows (~48 frames per window), PD
  estimation will be noisy. Pool across simultaneously recorded neurons
  (population PD) to improve SNR.
- Non-monotonic drift could reflect noise rather than re-anchoring.
  Statistical comparison against monotonic null model is essential.

**Priority: LOW-MEDIUM.** Interesting and testable, but the short dark
epochs and noisy single-cell PD estimates limit sensitivity. Best attempted
as an exploratory analysis using population-level PD estimates rather than
single-cell drift trajectories.

---

### H-NEW-10: RSP population state at the moment of lights-off predicts the subsequent drift magnitude, and this prediction differs by cell type

**Hypothesis:** The population activity pattern at the instant visual cues
disappear determines how well the HD representation is maintained in the
subsequent dark epoch. If the population is in a "strong HD state"
(coherent population vector with high MVL), drift will be slower. If in
a "weak HD state" (low population coherence), drift will be faster. This
relationship may differ between Penk+ and Penk-CamKII+ populations.

**Motivating literature:**
- Gonzalez et al. (2026): Neuronal subspaces in the CA1-RSC axis support
  distinct interactions. The initial population state at a transition
  could determine subsequent dynamics.
- Developmental toroidal topology paper (2026): The ring attractor's state
  at the moment of visual cue loss determines the starting point for
  path integration.

**Analysis:**
1. At each light-to-dark transition, compute the population vector length
   (PVL) in a 5-second window immediately before lights-off.
2. Compute the total PD drift during the subsequent dark epoch.
3. Correlate pre-transition PVL with dark-epoch drift magnitude using
   Spearman rank correlation.
4. Compare this correlation between Penk+ and Penk-CamKII+ sessions.

**Expected result if true:** Negative correlation: higher pre-transition
PVL predicts less drift. Stronger correlation for the population that
is more visually-dependent (because its HD stability depends more on the
quality of the visual representation at the moment of removal).

**Key confounds:**
- PVL depends on cell count and HD tuning distribution. Sessions with
  more cells will have higher PVL regardless of coding quality.
  Normalise by expected PVL under uniform tuning.
- The pre-transition state confounds the animal's behaviour with the neural
  state. If the mouse is running quickly before lights-off, both PVL and
  subsequent drift may be affected.

**Priority: LOW.** Conceptually interesting but requires enough
light-to-dark transitions per session (typically 5-7 per session given
the alternating protocol) and enough simultaneously-recorded cells to
compute a meaningful population vector. With ~15 ROIs per session,
this is marginal.

---

## Part 3: Cross-Cutting Themes

### Theme 1: Ubiquitous ATN input constrains the interpretation of cell-type differences

Margetts-Smith et al. (2025) showed that all RSP pyramidal neurons receive
anterior thalamic input. This fundamentally constrains the interpretation
of Penk+ vs Penk-CamKII+ HD differences: both populations receive the same
thalamic HD drive. Cell-type differences, if they exist, must arise from:

(a) Differential weighting of non-thalamic inputs (visual cortex feedback,
hippocampal input, ACC/M2 motor signals). Penk+ neurons may receive
proportionally more visual cortex input; Penk-CamKII+ may integrate more
hippocampal or vestibular signals.

(b) Differential local circuit processing. Penk expression is associated
with specific neuromodulatory profiles (enkephalin is an endogenous opioid).
Cholinergic or opioidergic modulation may differentially affect the two
populations, as suggested by the Jedrasiak-Cape et al. (2024) finding that
cholinergic modulation differs across RSG excitatory subtypes.

(c) Differential intrinsic properties. If Penk+ and Penk-CamKII+ neurons
differ in membrane time constant, spike threshold, or adaptation, they
could transform the same thalamic HD input differently.

**Implication for the paper:** The discussion must explicitly address that
differential thalamic input is unlikely to explain cell-type differences,
and propose specific alternative mechanisms. This strengthens the story
because it points toward a local computation mechanism rather than simple
connectivity differences.

### Theme 2: Visual-motion coherence as the unifying framework

Multiple papers converge on the idea that RSP integrates visual motion with
self-motion, and that the coherence between these signals matters:

- Chaplin & Margrie (2020): Visual-vestibular integration is additive in V1;
  RSP is positioned to perform a similar computation.
- The landmark learning paper (2024 bioRxiv): Uncoupling treadmill motion
  from visual feedback altered RSP responses.
- Secer et al. (2025): Area 29e coupling to MEC depended on gamma-band
  visual-to-spatial transformation.
- Jayakumar et al. (2025): Path-integration recalibration requires
  locomotion (i.e., self-motion signals).

Our lights-off condition creates a specific visual-motion coherence
violation: self-motion continues but visual motion ceases. This is not
the same as the animal being stationary in darkness. The prediction is
that RSP neurons (especially a visually-dependent subpopulation) should
show a mismatch response when the expected visual flow from locomotion
is absent.

**Testable analysis:** Compute a "mismatch index" for each neuron by
comparing activity during active locomotion in darkness (visual-motion
mismatch) vs active locomotion in light (coherent) vs stationary in
darkness (no mismatch). If a cell responds to the mismatch, its activity
during dark-locomotion should differ from both dark-stationary and
light-locomotion in a non-additive way.

### Theme 3: The neuropil is not just a confound — it is a readout of afferent drive

Across the reviewed papers, the neuropil signal emerges as a scientifically
informative signal in its own right, not merely a contamination artefact:

- Kerr et al. (2005): Neuropil = optical encephalogram, reflecting aggregate
  presynaptic input.
- Margetts-Smith et al. (2025): ATN input is ubiquitous → neuropil in RSP
  carries the thalamic HD representation.
- Dipoppa et al. (2018): Neuropil signals change with behavioural and visual
  context.

For the hm2p dataset, the neuropil signal in RSP imaging fields likely
reflects the HD-tuned thalamic/postsubicular input. Analysing how this
neuropil HD signal changes between light and dark provides a proxy for
how the upstream HD circuit is affected by visual cue removal, independent
of what the recorded RSP neurons do. This is a unique readout of circuit
input that most studies discard.

**Recommendation:** Elevate the neuropil analysis from a control to a
secondary finding. Show: (1) RSP neuropil carries HD information,
(2) this information partially degrades in darkness, (3) the somatic signal
exceeds the neuropil signal in HD information, confirming genuine somatic
tuning.

### Theme 4: The q-rose maze enables navigation decision analysis that open fields cannot

Several papers point toward RSP's role in navigation decisions beyond
simple HD coding:

- Rosenberg et al. (2021): Maze exploration follows local turning rules
  (forward bias, turn alternation).
- Active pursuit paper (2026 bioRxiv): Allocentric HD tuning decreases
  during goal-directed movement.
- Bech et al. (2026): RSP enables context-dependent sensorimotor
  transformation.

The q-rose maze, with its forced choice points and dead ends, provides
something open fields do not: discrete navigational decisions where the
animal must choose a direction. Whether RSP activity at junctions predicts
the upcoming turn — and whether this prospective coding differs by cell
type — is one of the most novel analyses possible with this dataset (as
identified in the maze exploration ideas document, Tier 2 priority). This
analysis connects HD coding to navigation function, going beyond
correlational tuning curve analysis.

### Theme 5: Cholinergic modulation as a mechanistic link between cell types and function

Two recent papers (Jedrasiak-Cape et al. 2024; Tanimura et al. 2025)
highlight cholinergic modulation as a key determinant of RSP cell-type
function. Enkephalin (Penk) is itself a neuromodulatory peptide that acts
on opioid receptors. The interaction between cholinergic and opioidergic
modulation in RSP is unexplored.

While the hm2p dataset cannot directly measure cholinergic tone, the
light/dark manipulation may indirectly modulate it (arousal and cholinergic
state are coupled; darkness may alter cholinergic drive). If Penk+ neurons
respond differently to darkness partly because of their opioidergic
properties (e.g., tonic enkephalin release modulating local circuits), this
would connect the cell-type-specific functional difference to a specific
neuromodulatory mechanism.

**Implication for the discussion:** Frame the Penk+/non-Penk distinction
not just as a genetic marker but as a neuromodulatory axis. Penk expression
defines a population that produces enkephalin, which could modulate local
circuit dynamics. Cite Jedrasiak-Cape et al. (2024) for the precedent that
RSP excitatory subtypes differ in neuromodulatory properties. This provides
a mechanistic hypothesis beyond pure description.

---

## Summary of Priorities

| Hypothesis | Priority | Novel? | Feasible with current data? |
|------------|----------|--------|----------------------------|
| H-NEW-1: PD drift onset/rate by cell type | High | Yes | Yes (needs drift computation) |
| H-NEW-3: Neuropil HD tuning analysis | High | Yes (as a finding, not just control) | Yes (Fneu available) |
| H-NEW-6: Transition response by cell type | High | Yes | Yes (straightforward alignment) |
| H-NEW-8: Neuropil-to-soma contamination ratio | High | Yes (as confound control) | Yes |
| H-NEW-4: AHV tuning x cell type x light/dark | Medium | Partly (extends Jedrasiak-Cape) | Yes (ahv.py exists) |
| H-NEW-7: Speed-dependent MVL by cell type | Medium | Yes | Needs occupancy-controlled reanalysis |
| H-NEW-2: Population decoding robustness | Medium | Yes | Limited by 12 vs 4 animal design |
| H-NEW-9: Non-monotonic drift in darkness | Low-Medium | Partly (extends Sarramone) | Marginal (short epochs, noisy) |
| H-NEW-5: Multidirectional tuning in maze arms | Low | Yes | Severely limited by occupancy |
| H-NEW-10: Pre-transition state predicts drift | Low | Yes | Marginal (few cells, few transitions) |

**Recommended for the primary paper:** H-NEW-1, H-NEW-3, H-NEW-6, H-NEW-8.
These four address the core question (cell-type-specific visual dependence),
provide essential controls (neuropil), and exploit the unique features of the
dataset (rapid transitions, repeated light/dark epochs).

**Recommended for supplementary analyses:** H-NEW-2, H-NEW-4, H-NEW-7.
These strengthen the story but are limited by statistical power or
temporal resolution.

**Recommended for exploratory/future work:** H-NEW-5, H-NEW-9, H-NEW-10.
Interesting but underpowered or requiring additional data collection.
