# Paper Summary: Gobbo et al. 2026

**Full citation:** Gobbo F, Mitchell-Heggs R, Duszkiewicz AJ, Faillace E, Tse D, Garcia-Font N, Hazon O, Clarke K, Spooner PA, Keshishian A, Schnitzer M, Schultz SR, Morris RGM. 2026. "Navigational strategy dictates hippocampal representation of space in an everyday memory task." bioRxiv. doi:10.1101/2025.05.10.653115

**Status:** bioRxiv preprint, posted 2026-05-22 (CC-BY-NC-ND 4.0)

**Species/region:** Rats, dorsal hippocampal CA1 (miniscope calcium imaging + optogenetics)

---

## 1. Paper Overview

The study investigates how the navigational strategy an animal employs (allocentric vs egocentric) shapes hippocampal CA1 representations during spatial memory performance. Two key experimental approaches are used:

**Experiment 1 (Imaging):** One-photon calcium imaging (Inscopix miniscope, 20 fps, GCaMP6f under CaMKII promoter) of CA1 pyramidal cells in freely-moving rats performing allocentric and egocentric variants of an "everyday memory" task in a 1.6 m square event arena with 6 sandwells. Longitudinal tracking across 7 recording sessions (S28-S34). n = 7-9 rats per group. Within-subject crossover design.

**Experiment 2 (Optogenetics):** Bilateral JAWS-mediated optogenetic inhibition of dorsal CA1 during the startbox planning window (10 s before trial start) or during the entire trial. n = 7-8 rats per group per virus condition.

**Key findings:**
1. Allocentric-trained animals had a higher proportion of place cells and cells with single place fields, while egocentric-trained animals had more multi-field cells.
2. Pre-trial startbox activity in allocentric-trained animals showed focused, non-local representations of specific goal sandwells. In egocentric-trained animals, this activity was more diffuse, representing multiple locations.
3. In allocentric animals, startbox representations of the correct destination increased as trial start approached, predicting correct performance.
4. Optogenetic silencing of CA1 during the startbox planning window impaired allocentric but not egocentric task performance, demonstrating a causal role for hippocampal pre-navigational planning specifically under allocentric strategy.

---

## 2. Relevance to RSP Head Direction Cell Research

**Moderate relevance.** This paper does not study the retrosplenial cortex directly, nor does it focus on head direction cells. However, it establishes several conceptual frameworks relevant to the hm2p project:

- **Strategy-dependent neural coding:** The central finding --- that the same brain region (CA1) encodes space differently depending on which navigational strategy the animal employs --- raises the question of whether RSP neurons similarly modulate their functional properties based on navigational context. In the hm2p project, the q-rose maze constrains navigational choices differently from an open arena, and the light/dark manipulation changes which cues are available to guide strategy. If Penk+ and Penk-CamKII+ populations support different navigational strategies (e.g., one anchored to landmarks, one to path integration), then Gobbo et al.'s framework predicts they would show different representational properties even in the same maze.

- **Allocentric vs egocentric representations in hippocampal-RSP circuits:** The RSP is a critical relay between hippocampus and sensory cortex. The finding that allocentric navigation requires intact hippocampal planning activity, while egocentric does not, implies that RSP-hippocampal interactions may be differentially engaged depending on navigational strategy. This is relevant because Penk+ RSP neurons project heavily to hippocampus (via subiculum), while CamKII+ neurons may project more to thalamic HD nuclei --- suggesting the two populations could be differentially recruited during allocentric vs egocentric modes.

- **Directionality index:** The paper computes a directionality index for CA1 place cells using head direction derived from DLC ear positions (same bodyparts as our project), finding no significant difference between allocentric and egocentric groups. Their formula (Rubin et al. 2014) uses 4 cardinal direction bins. Our HD tuning analyses use continuous tuning curves with MVL, which provides finer resolution, but it is worth noting that even hippocampal place cells show some directional modulation in their task.

---

## 3. Relevance to the Light/Dark Manipulation

**Low-to-moderate relevance.** The paper does not use a lights-off manipulation. However, the curtain manipulation (occluding extra-arena visual cues) serves a conceptually parallel function:

- Allocentric-trained animals fell to chance when curtains obscured extra-arena cues, confirming their dependence on visual landmarks for navigation.
- Egocentric-trained animals were unaffected by curtain occlusion.

This parallels our expectation that RSP neurons anchored to visual landmarks will lose HD tuning stability in darkness, while those relying on path integration signals will maintain it. The key difference is that our manipulation is more complete (total darkness removes ALL visual input including intra-maze cues and self-motion-related optic flow), whereas curtains only remove extra-maze cues. This distinction matters: our dark condition eliminates both landmark information AND optic flow, making it a stronger test of idiothetic self-motion signals (vestibular + proprioceptive) alone.

---

## 4. Methodological Insights

Several analysis approaches in this paper are relevant to the hm2p pipeline:

### 4.1 Hierarchical Bootstrap (Saravanan et al. 2020)
The authors use hierarchical bootstrap to handle the nested structure of cells within recordings within animals. This is directly applicable to our Penk+ vs Penk-CamKII+ comparisons, where we have cells nested within sessions nested within animals. We already plan to use bootstrap-based approaches, but the specific procedure described here (resampling at cell and animal levels, standardizing to median cell count per recording, 10,000 iterations) provides a concrete implementation reference.

### 4.2 Head Direction from DLC Ear Positions
The paper computes head direction from DLC-tracked left and right ear coordinates using atan2(dy, dx) --- identical to our approach. They use this to compute a directionality index based on firing rates across 4 cardinal direction bins. Our tuning curve approach (angular bins, Rayleigh test for non-uniformity, mean vector length) is more standard for HD cell characterization, but the consistency of tracking methodology is reassuring.

### 4.3 Population Decoding in Pre-Task Windows
The multinomial logistic regression decoder trained on arena activity and applied to startbox activity is conceptually analogous to our planned HD population decoder. We train decoders on HD during light epochs and test generalization to dark epochs. The Gobbo et al. approach of testing on temporally and spatially separated data (arena activity decoded from startbox) provides a useful template for cross-condition decoding.

### 4.4 Neural Trajectory Correlation via CCA
Canonical correlation analysis (CCA) to compare neural dynamics across trials is potentially useful for our project. We could use CCA to compare population-level HD representations between light and dark epochs, or between Penk+ and Penk-CamKII+ populations. This is more principled than simple pairwise correlations when cell identities differ across sessions.

### 4.5 Calcium Event Detection
They use OASIS deconvolution (Friedrich et al. 2017) with s_min adjusted between 0.2-0.3 depending on baseline noise. Our pipeline uses CASCADE for spike inference, which is calibrated against ground-truth electrophysiology. CASCADE is generally considered more accurate than OASIS for calcium-to-spike conversion, so this is not something to adopt, but it is useful context for understanding their event detection sensitivity.

### 4.6 Statistical Approach
They use a mix of parametric (ANOVA, LMM) and non-parametric (Mann-Whitney) tests. Their LMMs for nested data are appropriate but we should note that our project commits to non-parametric tests as primary analyses, with LMMs only as supplementary checks. The hierarchical bootstrap approach is actually preferable to LMMs for our purposes as it makes fewer distributional assumptions.

---

## 5. Relevance to Penk+ vs Penk-CamKII+ Comparison

**Low direct relevance, moderate conceptual relevance.** The paper does not study genetically-defined cell types within CA1 --- they image all CaMKII+ pyramidal cells together. However:

- **Cell-type-specific strategy encoding:** The paper cites Esparza et al. 2025 and Danielson et al. 2016, noting that deep and superficial CA1 pyramidal cells differentially encode local vs global cues and are differentially modulated by goal-directed learning. This establishes precedent for the idea that genetically-defined subpopulations within the same region can have distinct functional roles in spatial navigation. Our hypothesis that Penk+ and Penk-CamKII+ RSP neurons differ in visual vs idiothetic HD anchoring follows the same logic.

- **Nonlocal representations by cell type:** If RSP Penk+ neurons project preferentially to hippocampus and carry more allocentric/landmark-anchored HD signals, while Penk-CamKII+ neurons carry more idiothetic signals to thalamic nuclei, then the Gobbo et al. finding that allocentric performance requires hippocampal planning activity would predict that Penk+ neurons are more critical for landmark-anchored navigation, and their signal would degrade more in darkness.

- **No Penk-specific data:** The paper provides no data on enkephalin-expressing neurons in any region. The relevance to our specific cell type comparison is therefore conceptual rather than empirical.

---

## 6. Key Figures and Results to Cite

If citing this paper in the hm2p manuscript, the most relevant findings are:

- **Figure 1f,g:** Cue removal (curtains) selectively impairs allocentric but not egocentric performance. Relevant as conceptual parallel to our light-off condition eliminating landmark cues.
- **Figure 2d,e:** Different navigational strategies produce different proportions of spatially tuned cells and different field structures within the same brain region. Supports the principle that cognitive strategy shapes neural representation.
- **Figure 5h,i:** Causal demonstration (optogenetics) that hippocampal activity during decision-making is necessary for allocentric but not egocentric navigation. This is the strongest result and provides mechanistic grounding for the idea that allocentric/landmark-dependent representations are computationally distinct from egocentric/idiothetic ones.

**Suggested citation context:** In the Introduction, when motivating why Penk+ and Penk-CamKII+ RSP populations might support different navigational computations: "Recent work has demonstrated that navigational strategy fundamentally shapes hippocampal spatial representations, with allocentric navigation relying causally on hippocampal planning activity in ways that egocentric navigation does not (Gobbo et al. 2026). Whether upstream regions such as RSP contain genetically-defined subpopulations that differentially support allocentric vs egocentric navigation remains unknown."

---

## 7. Implications for the hm2p Analysis Pipeline

### 7.1 Analyses to Consider Adopting
- **Hierarchical bootstrap for cell-type comparisons:** Adopt the Saravanan et al. 2020 procedure for comparing Penk+ vs Penk-CamKII+ populations. Resample at the cell level within sessions and at the animal level across animals. This addresses the non-independence of cells within a session and the unequal N across animals (12 Penk+ animals vs 4 nonpenk animals --- a substantial imbalance).
- **CCA for cross-condition neural dynamics:** Consider CCA to compare population-level HD representations between light and dark epochs. This is more principled than comparing individual cell tuning curves and could reveal population-level geometric transformations (rotation, gain changes) that single-cell analyses miss.

### 7.2 Conceptual Framing
- The Gobbo et al. framework of allocentric-dependent vs allocentric-independent navigation maps onto our visual-landmark-anchored vs idiothetic-anchored HD signals. We can frame our findings as testing whether the RSP contains genetically-defined subpopulations that differentially contribute to these two modes of spatial representation. This connects our work to the broader allocentric/egocentric literature rather than positioning it narrowly within the HD cell field.

### 7.3 Limitations and Caveats
- **Different brain region:** CA1 is upstream of RSP in many circuits. Findings in hippocampus do not necessarily transfer to RSP.
- **Different species:** Rats vs mice. HD cell properties are broadly similar across rodent species, but there may be species-specific differences in RSP circuitry.
- **Different imaging modality:** One-photon miniscope (Inscopix) vs two-photon head-mounted microscope. One-photon imaging has higher background fluorescence and poorer optical sectioning, but samples more cells. Our two-photon data has better single-cell resolution and less neuropil contamination.
- **Different task:** Everyday memory task with learned goal locations vs continuous free navigation in a maze with alternating light/dark. The cognitive demands are quite different.
- **Statistical approach:** Their primary use of parametric tests (ANOVA, LMM) with hierarchical bootstrap as supplementary is the reverse of our approach (non-parametric primary, LMM supplementary). Not a problem for citing their findings, but their reported p-values should be interpreted with this in mind.

---

## Summary Assessment

**Overall relevance to hm2p: MODERATE**

This paper is not directly about RSP, HD cells, or genetically-defined cell types. Its primary value to the hm2p project is conceptual and methodological:

1. **Conceptual:** It establishes that navigational strategy (allocentric vs egocentric) fundamentally shapes neural representations in the spatial navigation circuit. This motivates the hypothesis that RSP subpopulations supporting different modes of HD anchoring (visual vs idiothetic) would show distinct functional properties.

2. **Methodological:** The hierarchical bootstrap procedure and CCA-based population analysis are directly applicable to our dataset.

3. **Citation value:** Useful for the Introduction when motivating cell-type-specific contributions to spatial navigation, and for the Discussion when interpreting differences between Penk+ and Penk-CamKII+ populations in the context of allocentric vs idiothetic processing.

The paper should be cited but is not a primary reference for our specific findings. It belongs in the "broader context" section of the Introduction or Discussion rather than in the direct literature comparison.
