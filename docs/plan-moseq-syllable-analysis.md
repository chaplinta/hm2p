# keypoint-MoSeq Syllable Analysis Plan for the Behaviour Manuscript

**Date:** 2026-06-01
**Status:** Pre-analysis plan — awaiting kpms pipeline completion on EC2

---

## 1. What MoSeq captures that the HMM does not

The existing HMM decomposes behaviour into kinematic states defined by three
scalar features: speed, absolute AHV, and spatial coverage rate. It discovers
*how fast* and *how directionally* the mouse is moving, but it is blind to
*posture* and *postural dynamics*. The three HMM states (pausing, slow
scanning, fast traversal) are kinematic categories — they could describe a
car as easily as a mouse.

keypoint-MoSeq (Wiltschko et al. 2015, Weinreb et al. 2024 Nature Methods)
discovers postural syllables from the temporal dynamics of multi-keypoint
body shape. It captures:

- **Body posture sequences**: how the spine curves, head orientation relative
  to body axis, tail curvature, limb asymmetry
- **Temporal microstructure**: sub-second movement modules (typical syllable
  duration ~300-500 ms) vs the HMM's much longer state durations
- **Ethologically meaningful actions**: rearing, grooming, turning-in-place,
  sniffing, wall-following, etc. — actions defined by coordinated multi-joint
  movement, not by speed alone
- **Transition grammar**: stereotyped sequences of syllables form a behavioural
  grammar; the transition structure itself carries information

In summary: the HMM tells us *how much* of each speed regime the mouse uses;
MoSeq tells us *what the mouse is doing* at sub-second timescale. They operate
at different levels of description and should be complementary rather than
redundant.

**Critical caveat:** Our DLC model tracks 8 keypoints from overhead video.
This is fewer and lower-dimensional than the typical MoSeq input (depth video
with full 3D body contour, or dense pose models with 20+ keypoints). With 8
overhead keypoints, MoSeq will primarily capture gross body shape (elongation,
curvature, head-body angle) and head movements, but will miss fine postural
details (e.g., forepaw placement, ear position dynamics, rearing vs sitting
posture). This limits the granularity of syllable discovery. The resulting
syllables will likely be dominated by locomotor patterns (turning, straight
running, pausing postures) rather than subtle ethological behaviours.

This is not necessarily a problem for the manuscript's story — locomotor
syllable changes at maze decision points are scientifically interesting. But
it means MoSeq may not discover dramatically more than the HMM, because with
8 keypoints the postural dynamics accessible to MoSeq substantially overlap
with the kinematic features already captured by the HMM.

---

## 2. Hypotheses: prioritised by expected scientific value

### Tier 1: Genuinely informative for the manuscript narrative

#### M1. Syllable diversity is reduced in darkness

**Prediction:** The number of unique syllables used per epoch, or Shannon
entropy of syllable usage distributions, is lower in dark than light epochs.

**Rationale:** If route stereotypy reflects a simplification of the
behavioural programme (not just spatial coverage), we would expect fewer
distinct postural actions in darkness. This would extend the manuscript
beyond "mice use fewer routes" to "mice use fewer behavioural building
blocks." Conversely, if syllable diversity is preserved (like HMM occupancy),
it would strengthen the interpretation that the motor programme is intact and
only spatial routing changes.

**Statistical test:** Wilcoxon signed-rank on per-session syllable entropy
(light vs dark), N = 20.

**What the result means:**
- *Reduced diversity in dark:* Adds a new dimension to route stereotypy —
  the behavioural repertoire narrows, not just the spatial repertoire. This
  would be genuinely novel and would warrant promotion from supplementary to
  main text.
- *No change:* Reinforces the HMM null. The full motor repertoire is deployed
  but over a restricted spatial domain. Consistent with route stereotypy being
  purely spatial. Informative null — strengthens the existing narrative.

**Expected scientific value:** HIGH. Either outcome is interpretively useful.
The syllable diversity metric captures something the HMM cannot (how many
distinct postural actions vs how much time in each speed regime).

**Confound:** Fewer cell visits in darkness mechanically means fewer
opportunity-contexts for diverse syllables. Must normalise: compute entropy
per unit time, or match on number of syllable instances. A null could just
mean "same behaviours repeated in the same few corridors."

---

#### M2. Syllable transition structure becomes more repetitive in darkness

**Prediction:** The syllable-to-syllable transition entropy decreases in
darkness, or the number of unique syllable bigrams (2-grams) decreases.

**Rationale:** If mice are running the same routes repeatedly in darkness, the
*sequence* of postural actions should become more stereotyped even if the
*vocabulary* size is unchanged. This is a higher-order test of behavioural
stereotypy — not just fewer syllables, but more predictable sequences.

**Statistical test:** Wilcoxon signed-rank on per-session syllable transition
entropy (light vs dark), N = 20. Also compute the number of unique bigrams
and the Lempel-Ziv complexity of the syllable sequence as convergent metrics.

**What the result means:**
- *More repetitive transitions:* Route stereotypy penetrates to the level of
  action sequences. Mice don't just use fewer corridors — they execute the
  same motor programmes in the same order. This would be a strong finding.
- *No change:* Syllable transitions are equally variable in light and dark,
  meaning the motor grammar is preserved even though spatial behaviour is
  more constrained.

**Expected scientific value:** MODERATE-HIGH. This is a cleaner test than M1
because transition structure is less confounded by the mechanical reduction in
opportunity. However, interpretation requires care: more repetitive routes
mechanically produce more repetitive syllable sequences if syllables are
location-specific.

**Confound:** If syllable usage is partially determined by maze location
(e.g., specific syllables at junctions), then route stereotypy will
mechanically produce more repetitive syllable sequences. This is a
confound-or-feature ambiguity: it could mean "same routes produce same
syllable sequences" (uninteresting) or "the mouse's postural programme
becomes simpler" (interesting). To distinguish, compute syllable transition
entropy *within* a single location type (e.g., only at junctions, where the
mouse visits similar locations in both conditions).

---

#### M3. Junction-approach syllables predict upcoming turn direction

**Prediction:** In the ~1 second before arriving at a T-junction, specific
syllables predict whether the mouse will turn left or right. The syllable
→ turn association is stronger in light than dark.

**Rationale:** This tests whether MoSeq captures preparatory motor actions
(postural adjustments, head turns, weight shifts) that precede a turn
decision. If such preparatory syllables exist, they suggest motor planning is
embedded in postural dynamics. If they degrade in darkness, it implies that
visual information contributes to turn preparation.

**Statistical test:** For each junction approach, extract the syllable in the
1 s before junction entry. Compute mutual information between pre-junction
syllable identity and subsequent turn direction (left/right). Compare MI
between light and dark using Wilcoxon signed-rank (N = 20 sessions). Also
test with a logistic classifier: syllable → left/right accuracy.

**What the result means:**
- *Syllables predict turns, less so in dark:* MoSeq reveals preparatory motor
  planning that degrades without visual input. This is a strong, novel finding
  connecting postural dynamics to spatial decision-making.
- *Syllables predict turns equally in light and dark:* Turn preparation is
  internally generated (vestibular/proprioceptive) and does not require vision.
  Also interesting — consistent with preserved turn alternation.
- *No prediction:* Pre-junction syllables don't carry turn information. This
  would mean either MoSeq granularity is insufficient (8 keypoints) or turn
  decisions are not manifest in pre-junction posture.

**Expected scientific value:** HIGH if positive; still informative if null.
This is the most mechanistically interesting hypothesis because it links
postural dynamics to spatial decision-making. However, with 8 keypoints from
overhead video, the ability to detect subtle preparatory weight shifts may be
limited.

**Confound:** If the mouse typically approaches junctions from the same
direction, approach direction will predict turn direction (via turn
alternation), and approach syllables will be confounded with approach
direction. Must condition on approach direction or match across conditions.

---

### Tier 2: Useful but likely redundant with existing findings

#### M4. Syllable usage differs by maze location type
(junction vs corridor vs dead-end)

**Prediction:** Junctions, corridors, and dead ends have different syllable
distributions (e.g., more scanning syllables at junctions, more fast-
locomotion syllables in corridors, more rearing/sniffing at dead ends).

**Rationale:** If syllable usage is location-type-specific, it validates that
MoSeq is capturing meaningful behavioural structure in the maze context.

**Statistical test:** Kruskal-Wallis on syllable entropy or per-syllable
frequency across three location types. Chi-squared on syllable count matrices.

**What the result means:**
- *Different distributions:* Expected and confirmatory. Mice do different
  things at different maze locations. This is not novel (anyone observing a
  mouse in a maze would report this) but validates the MoSeq output.
- *No difference:* Would suggest MoSeq is not capturing location-relevant
  behaviour, which would undermine the utility of further MoSeq analyses.

**Expected scientific value:** LOW as a standalone finding (descriptive,
expected). HIGH as a validation step before testing M1-M3. Should be computed
first as a sanity check but not reported as a main result.

---

#### M5. Specific syllable frequencies change in darkness

**Prediction:** Some syllables (e.g., scanning, rearing) decrease in
darkness; others (e.g., wall-following, thigmotactic patterns) increase.

**Rationale:** If visual cue removal changes the behavioural repertoire at
the level of individual actions, specific syllables should have condition-
dependent frequencies.

**Statistical test:** For each syllable, Wilcoxon signed-rank on per-session
usage frequency (light vs dark), with Holm-Bonferroni correction across K
syllables (typically K = 30-50 for kpms).

**What the result means:**
- *Specific syllables change:* Identifies which behavioural building blocks
  are affected by darkness. Potentially interpretable if syllables can be
  labelled (e.g., "scanning syllable decreases" → "mice scan less without
  visual input").
- *No individual syllable changes:* Even if M1 (overall diversity) is null,
  this would confirm it at the individual-syllable level.

**Expected scientific value:** MODERATE. Interesting if particular syllables
can be clearly labelled and the direction of change is interpretable.
However, with K = 30-50 syllables and N = 20 sessions, statistical power
after multiple comparison correction is very low for any individual syllable.
Likely to produce a null result even if real effects exist.

**Confound:** Multiple comparisons are severe. With ~40 syllables tested,
a Bonferroni-corrected alpha of 0.05/40 = 0.00125 is demanding for N = 20.
Consider a global test (e.g., permutation test on the multivariate syllable
usage vector) rather than per-syllable tests.

---

#### M6. Darkness-specific syllables emerge

**Prediction:** Some syllables appear only (or predominantly) in dark epochs,
representing darkness-specific behaviours not seen in light.

**Rationale:** Novel behavioural strategies in darkness (e.g., increased
whisking-like head movements, wall-following with specific body posture)
might manifest as unique syllable types.

**Statistical test:** For each syllable, compute the light/dark usage ratio.
Identify syllables with > 80% of instances in dark. Test whether the count
of such syllables exceeds a permutation null (shuffle condition labels).

**What the result means:**
- *Darkness-specific syllables exist:* Novel behavioural actions emerge when
  visual input is removed. Strong finding.
- *No darkness-specific syllables:* The same repertoire is used in both
  conditions. Expected given the HMM null.

**Expected scientific value:** LOW. Very unlikely to find truly novel
syllables. More likely, the same syllables are used with modestly different
frequencies (M5). Also, with 1-minute epochs, any "darkness-specific"
syllable could simply be rare and stochastically absent from the shorter
total dark time.

---

#### M7. MoSeq syllables cluster within HMM states

**Prediction:** Each HMM state (pausing, slow scanning, fast traversal)
contains a characteristic subset of MoSeq syllables. The MoSeq syllable
provides a finer-grained decomposition within each HMM state.

**Rationale:** If HMM and MoSeq are truly complementary (kinematics vs
posture), MoSeq should subdivide HMM states into postural subtypes (e.g.,
"pausing" contains grooming-pause, scanning-pause, freezing-pause).

**Statistical test:** Compute a syllable × HMM-state co-occurrence matrix.
Test for non-uniform association with chi-squared (or permutation-based MI).
Visualise as a heatmap.

**What the result means:**
- *Strong clustering:* MoSeq syllables nest within HMM states. Each pausing
  bout has a specific postural signature. Expected and validates both
  decompositions.
- *Weak clustering:* Syllables span HMM states, meaning MoSeq captures
  dynamics orthogonal to speed/AHV. More interesting scientifically but
  harder to interpret.

**Expected scientific value:** LOW-MODERATE. Descriptive comparison of two
decompositions. Useful for understanding what each method captures but not a
publication-worthy finding on its own. Better suited as a supplementary
panel.

---

### Tier 3: Exploratory / low expected yield

#### M8. Syllable duration changes in darkness

**Prediction:** Mean syllable duration is longer in darkness (syllables
become more sustained, less frequent switching).

**Rationale:** Slower transitions between postural actions might reflect
reduced sensory drive or increased behavioural inertia without visual input.

**Statistical test:** Wilcoxon signed-rank on per-session mean syllable
duration (light vs dark), N = 20.

**Expected scientific value:** LOW. Hard to interpret even if significant.
Could reflect speed differences (slower movement → longer syllables
mechanically). Adds little beyond the speed comparison already in the
manuscript.

---

#### M9. First dark epoch syllable repertoire differs from subsequent dark epochs

**Prediction:** Paralleling the coverage finding (H8), syllable diversity in
the first dark epoch matches light epochs, while subsequent dark epochs show
reduced diversity.

**Rationale:** If the single-trial adaptation to darkness is accompanied by
a behavioural repertoire shift, syllable diversity should show the same
step-change as spatial coverage.

**Statistical test:** Wilcoxon signed-rank on first-dark-epoch syllable
entropy vs subsequent-dark-epoch entropy, N = 20. Same test as the coverage
first-epoch analysis.

**What the result means:**
- *Matches the coverage pattern:* Route stereotypy and behavioural
  simplification co-emerge after one dark epoch. Strengthens the single-trial
  adaptation narrative.
- *No difference:* The behavioural repertoire doesn't show the same step
  change, suggesting coverage reduction and repertoire are dissociable.

**Expected scientific value:** MODERATE. Directly parallels a key finding
(H8). But with only ~5-6 dark epochs per session and one "first" epoch,
statistical power is limited.

---

#### M10. Syllable usage correlates with individual differences in darkness sensitivity

**Prediction:** Animals that show larger coverage drops in darkness also show
larger syllable diversity drops.

**Rationale:** If route stereotypy and behavioural simplification are coupled,
individual differences should co-vary.

**Statistical test:** Spearman correlation between per-animal coverage
sensitivity (dark-light coverage difference) and syllable entropy sensitivity
(dark-light entropy difference), N = 14 animals (or 20 sessions).

**Expected scientific value:** LOW. N = 14 animals is very small for a
correlation. Likely underpowered. Supplementary at best.

---

## 3. Analysis priority order

| Priority | Hypothesis | Rationale |
|----------|-----------|-----------|
| 1 | M4 | Validation: do syllables vary by location? Sanity check before anything else. |
| 2 | M7 | HMM × MoSeq correspondence. Understanding what MoSeq adds. |
| 3 | M1 | Syllable diversity in light vs dark. Core question. |
| 4 | M2 | Transition structure repetitiveness. Higher-order complement to M1. |
| 5 | M3 | Junction-approach prediction. Highest novelty if positive. |
| 6 | M5 | Per-syllable frequency changes. Exploratory. |
| 7 | M9 | First-epoch parallels. Natural extension of H8. |
| 8 | M8 | Duration changes. Low priority. |
| 9 | M6 | Darkness-specific syllables. Very unlikely. |
| 10 | M10 | Individual differences correlation. Underpowered. |

**Recommended stopping rule:** Run M4 and M7 first. If MoSeq syllables do
not show location-type-specific usage (M4 null) or cluster trivially within
HMM states (M7 shows 1:1 mapping), then MoSeq is not adding information
beyond the HMM for this dataset, and further hypotheses (M1-M3, M5-M10) are
unlikely to yield novel findings. In that case, defer MoSeq to the neural
paper.

---

## 4. Critical evaluation: does MoSeq belong in this manuscript?

### Arguments FOR including MoSeq

1. **Complementary decomposition.** MoSeq captures postural dynamics that
   the HMM cannot. If syllable diversity changes in darkness (M1), it adds a
   genuinely new dimension to route stereotypy.

2. **Junction-approach prediction (M3).** If syllables predict turn direction,
   this connects sub-second body dynamics to spatial decisions — a finding
   that would be novel and citable independent of the route stereotypy story.

3. **Methodological completeness.** Reviewers increasingly expect
   unsupervised behavioural decomposition. Including MoSeq signals
   methodological rigour.

4. **kpms is the gold standard** (Weinreb et al. 2024 Nature Methods). Using
   it positions the paper methodologically.

### Arguments AGAINST including MoSeq

1. **Diminishing returns.** The manuscript already has 4 main results, 11
   supplementary figures, and 2 supplementary tables. Adding MoSeq syllables
   as S12 or S13 risks bloating a short methods paper.

2. **The HMM already makes the key point.** The HMM null (kinematic profile
   preserved in darkness) is clean and sufficient. MoSeq may just confirm the
   same finding at higher resolution without changing the interpretation.

3. **8-keypoint limitation.** With only 8 overhead keypoints, MoSeq syllables
   will be dominated by locomotor dynamics (turning, speed changes, pauses)
   — substantially overlapping with what the HMM already captures. The
   postural richness that makes MoSeq powerful in open-field depth-camera
   settings is not fully available here.

4. **The real payoff is neural.** Syllable-neural correlations (which RSP
   neurons fire during which syllables? Do Penk+ and non-Penk neurons differ
   in syllable selectivity?) are far more interesting than syllable-behaviour
   correlations alone. MoSeq will be a central analysis in the neural paper.
   Introducing it in the behaviour paper uses up the "novelty" of the method
   without delivering the biggest insight.

5. **Power concerns.** With N = 20 sessions and ~40 syllables, per-syllable
   analyses (M5) are badly underpowered after multiple comparison correction.
   Global tests (M1, M2) have adequate power but may not reveal anything the
   HMM hasn't already shown.

### Verdict

**MoSeq belongs in the neural paper, not the behaviour paper.** The
strongest reason is point 4: the real scientific payoff of syllable analysis
is correlating syllables with neural population dynamics (which RSP cells
are active during junction-approach? do syllable transitions drive or follow
HD state changes?). Including syllable-only analyses in the behaviour paper
expends the methodological novelty without delivering the key insight.

**Exception:** If M3 (junction-approach syllables predict turn direction)
produces a clean positive result, that would be genuinely novel and worth
including as a supplementary figure. It connects sub-second body dynamics
to spatial decision-making in a way that is independent of the route
stereotypy story and would strengthen the paper. But this should be tested
and evaluated before deciding — do not pre-commit to including it.

**Recommended approach:**

1. Run the kpms pipeline. Get syllable assignments for all 26 sessions.
2. Run M4 (location validation) and M7 (HMM correspondence) as sanity
   checks. If MoSeq is adding nothing beyond the HMM, stop here.
3. Run M1 (syllable diversity light vs dark). If null, this confirms the HMM
   result and is noted in a single sentence ("kpms syllable diversity was
   also unchanged; data not shown").
4. Run M3 (junction-approach prediction). If positive, write up as
   Supplementary Fig S12 with 2-3 panels.
5. Save M2, M5, M8-M10 for the neural paper where they can be paired with
   neural data.

**Time budget:** Steps 1-3 should take ~2 hours of analysis time. Step 4
adds ~2 hours if pursued. Steps M2/M5/M8-M10 are deferred.

---

## 5. MoSeq output format and analysis integration

### Expected output from kpms pipeline

Per session, kpms produces:
- `syllable_id`: (N_frames,) int16 — per-frame syllable assignment
- `syllable_confidence`: (N_frames,) float32 — per-frame confidence
- `model.p`: fitted AR-HMM model parameters

These will be stored in `derivatives/kpms/` following NeuroBlueprint
convention and synced to `kinematics.h5` as `/syllable_id`.

### Integration with existing data

All syllable analyses require aligning kpms output with:
- `kinematics.h5`: position, HD, speed, AHV, cell_id, light_on, bad_behav
- `sync.h5`: neural data alignment (for neural paper only)

The alignment is by frame index — kpms operates on the same DLC output that
produces kinematics, so frames are 1:1 aligned.

### Key analysis parameters

- **Syllable count K:** kpms auto-selects K via the model's nonparametric
  prior (sticky HDP-HMM). Typical range: 30-60 syllables for mouse
  open-field. With 8 keypoints in a maze, expect 20-40.
- **Minimum syllable duration:** kpms default is ~100 ms (3 frames at 30 fps).
  This is appropriate for our data.
- **Rare syllable threshold:** For statistical analyses, exclude syllables
  with < 5 instances per session to avoid sparse-count artefacts.

---

## References

Wiltschko AB, Johnson MJ, Iurilli G, et al. 2015. "Mapping sub-second
structure in mouse behavior." *Neuron* 88, 1121-1135.
doi:10.1016/j.neuron.2015.11.031

Weinreb C, Pearl JE, Lin S, et al. 2024. "Keypoint-MoSeq: parsing behavior
by linking point tracking to pose dynamics." *Nature Methods* 21, 1329-1339.
doi:10.1038/s41592-024-02318-2

Markowitz JE, Gillis WF, Beron CC, et al. 2018. "The striatum organizes 3D
behavior via moment-to-moment action selection." *Cell* 174, 44-58.
doi:10.1016/j.cell.2018.04.019

Batty E, Whiteway M, Saxena S, et al. 2019. "BehaveNet: nonlinear embedding
and Bayesian neural decoding of behavioral videos." *NeurIPS*.
