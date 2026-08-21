# Paper Summary: Pereira et al. 2026

**Full citation:** Pereira M, Godinho BS, Machens CK, Costa RM, Akam T. 2026. "Flexible route planning and rapid structure learning by mice in complex environments." bioRxiv. doi:10.64898/2026.06.02.729586

**Status:** bioRxiv preprint, posted 2026-06-05 (CC-BY 4.0). 32 pages including 4 supplementary figures and 1 supplementary table.

**Affiliations:** Champalimaud Foundation (Lisbon); Department of Experimental Psychology, University of Oxford; Allen Institute; Sainsbury Wellcome Centre, UCL.

**Species/region:** C57BL/6 male mice. **No neural recordings.** This is a purely behavioural methods/assay paper.

**Hardware:** [github.com/pyControl/hardware/tree/master/GridMaze](https://github.com/pyControl/hardware/tree/master/GridMaze)
**HBI implementation:** [github.com/michaelfsp/pycbm](https://github.com/michaelfsp/pycbm)

---

## 1. Paper Overview

The authors develop and computationally optimise a behavioural assay ("the route planning task") intended to dissociate **structure-based navigation** (moving along the shortest path through the environment graph) from **vector-based navigation** (moving toward the goal in Euclidean space) and from habitual route selection, while generating large trial counts suitable for circuit-level recording.

**Apparatus (the "Grid-maze"):** a 6×6 array of hexagonal acrylic towers (11 cm edge-to-edge, 18 cm centre-to-centre, 1 m across the full grid) connected by removable 7 cm walkways, so the maze topology is reconfigurable between sessions. Each tower carries an LED-illuminated frosted acrylic rod (visible across the maze) and a floor-mounted nose-poke reward port. Position tracked by overhead camera. Original control was Arduino-based; the released design uses pyControl (Akam et al. 2022). Tower tops are laser-engraved with distinct textures as local cues; back-lit extra-maze cues sit 30 cm above tower level.

**Task:** a random tower is cued by its LED; the mouse navigates there and pokes for water; short ITI; new random goal. Goals sampled without replacement until exhausted, then resampled. Goals are **visually cued, not remembered** — deliberately, to separate the cost of navigating a complex environment from spatial memory for the goal.

**Two experiments:**
- **Pilot (n = 8):** 11 days pre-training on small T- and H-shaped configurations, then 9 days on one hand-designed 6×6 layout.
- **Main (n = 8):** 17 days pre-training on a 6-node H, then four maze topologies presented **sequentially** — fully connected, the hand-designed "original", "Maze A", "Maze B" (~10–12 days each).

**Dataset:** 59,651 navigation trajectories, 282,855 choice points.

**Key findings:**
1. Mice learn to navigate complex mazes efficiently — trials/session rise, excess steps per trial fall.
2. At choice points where the two strategies disagree, mice choose the **structure**-optimal option above chance; the effect strengthens with experience.
3. Maze layouts can be numerically optimised to discriminate the strategies, with an explicit trade-off between discriminability and structural diversity.
4. A hierarchical-Bayesian mixture-of-strategies model shows a large positive Structure weight, a smaller but significant Vector weight, a large Anti-backward (u-turn avoidance) weight, and a positive Structure × experience interaction.
5. Structure knowledge is evident from the **first session** on a newly configured maze.

---

## 2. Methods in Detail

### 2.1 Graph formalism
Maze layouts are undirected graphs: towers = nodes, walkways = edges. Two distances are defined per node pair: **geodesic** d_G (shortest path on the graph) and **Euclidean** d_E (straight-line on the grid). Every analysis in the paper is built from the difference between these two.

### 2.2 Excess steps per trial
`n_t = T_t − d_G(l_t, g_t)` — steps actually taken minus the shortest available route from the location at cue onset to the goal. Simple, interpretable, and the main learning-curve measure.

### 2.3 Optimal choice rate at "informative" choice points
For current location *l*, goal *g*, and candidate next locations *N*:

- structure-preferred set `P_S(l,g) = argmax_{l'∈N} [d_G(l,g) − d_G(l',g)]`
- vector-preferred set `P_V(l,g) = argmax_{l'∈N} [d_E(l,g) − d_E(l',g)]`

A choice point is **informative** when `P_S ∩ P_V = ∅`. Restrictions: only nodes with 3–4 available actions; **the u-turn action back to the previous location is excluded**; only the **first visit to each node within a trial** is scored (to avoid contamination by within-trial error correction). Chance is `|P_S|/|N|` per choice, pooled over analysed choices to give a maze-specific chance line.

### 2.4 Vector and structure indices
For each strategy *q*, count opportunities `o_q` (decisions where the remaining actions differ in the relevant distance), consistent choices `c_q`, and expected consistent choices under uniform random choice `r_q`:

`I_q = (c_q − r_q) / (o_q − r_q)`

0 = random, 1 = always follows the strategy when it makes a discriminable prediction. This normalisation is the paper's cleanest methodological contribution for the exploration case — it corrects for the fact that a strategy's apparent hit rate depends on how many options happen to be consistent with it. Data are plotted in the (I_S, I_V) plane against simulated single-strategy agents at 100 log-spaced stochasticity levels, with 95% kernel-density contours.

### 2.5 Maze structure optimisation
170,000 connected 6×6 mazes generated (10,000 each at 17 edge counts, 35–50 and 55). Generation: start from fully connected, remove edges at random to the target count, then swap present/absent edges until the graph is connected.

Two maze-level scores:
- **Fraction of informative states** `F_inf` — fraction of ordered (location, goal) pairs where `P_S ∩ P_V = ∅`, evaluated only at nodes with ≥3 neighbours, computed **before** removing backward actions.
- **Centrality-distribution flatness** `µ/σ²` of the node **betweenness-centrality** distribution (betweenness = fraction of all-pairs shortest paths passing through a node/edge).

Mazes maximising `F_inf` alone degenerate into a single long linear corridor folded into the grid (Fig. S2), where the decision collapses to "which way along the line". The flatness term penalises this. Maze A and Maze B were selected from the empirical Pareto front; the hand-designed original maze sits well inside it.

### 2.6 Mixture-of-strategies model
A conditional-logit / softmax action policy (McFadden 1974; Train 2009), fit **step-by-step** to every choice, not only to informative ones:

`A_net(a) = Σ_q β_q A_q(a)`, `P(a) = exp(A_net(a)) / Σ_{a'} exp(A_net(a'))`

Components:
- **Vector:** `A_V = d_E(l,g) − d_E(l'_a, g)` (graded, not winner-take-all)
- **Structure:** `A_S = d_G(l,g) − d_G(l'_a, g)` (graded)
- **Anti-backward:** 0 if the action is a u-turn, 1 otherwise
- **Vector × experience** and **Structure × experience**, experience coded as `log(maze step count + 1)`

Continuous components scaled by twice their empirical SD (Gelman 2008); the binary anti-backward component left on a 0/1 scale; interactions formed from already-standardised inputs. Fitting by hierarchical Bayesian inference (Piray et al. 2019), jointly inferring group-level weight distributions and subject-level posteriors.

**Model comparison** over 7 nested variants (Table S1): the winning model (model 6) has estimated population frequency 1.000 and protected exceedance probability 0.989 — a decisive margin, though all six alternatives sit at exactly 0.000/0.002, which is a suspiciously flat comparison landscape.

**Weights (Fig. 5b):** Anti-backward t₉ = 8.39, p < 0.0001; Vector t₉ = 3.23, p = 0.010; Vector × experience t₉ = 1.52, p = 0.164 (ns); Structure t₉ = 8.18, p < 0.0001; Structure × experience t₉ = 7.45, p < 0.0001.

### 2.7 Posterior predictive simulation
16 replicate draw sets; one parameter vector per subject drawn from its approximate posterior; simulated behaviour generated over that subject's actual trial and goal sequence; summarised with the same excess-steps calculation. Simulated learning curves closely track the data (Fig. 5c) — a genuine posterior-predictive check rather than a re-plot of the fit.

### 2.8 Egocentric action bias (Fig. S3)
Rate of straight/left/right/u-turn across the four mazes, split into the first step after cue onset vs all subsequent steps, restricted to locations with >1 available action. This is what motivates the Anti-backward component: u-turns are very rare once navigation to the goal has commenced.

---

## 3. Strength of Evidence and Weaknesses

**Strengths:**
- Very large choice count (282,855) for a rodent behavioural study.
- The maze-optimisation step is a real methodological advance: it makes explicit, and then quantitatively manages, the fact that in most mazes the two candidate strategies agree, so most choice points are uninformative.
- The mixture model uses **every** choice with **graded** action preferences, rather than restricting to a hand-picked disagreement subset with winner-take-all rules. Posterior-predictive learning curves are shown.
- Four topologies including a fully connected control where the strategies provably cannot be dissociated — a useful internal negative control (and indeed the indices show no separation there).
- Design files and the HBI implementation are released.

**Weaknesses, in rough order of importance:**

1. **All inferential statistics are parametric with n = 8:** repeated-measures ANOVA, one-sample t-tests, paired t-tests. No non-parametric confirmation anywhere. With eight subjects, ANOVA sphericity and normality are untestable in practice.

2. **Reported n does not match the degrees of freedom in the pilot experiment.** The text and Fig. 2 legend state n = 8, but the pilot rmANOVAs are reported as F(8,40), which for a 9-level within-subject factor implies **n = 6** (df₂ = 8 × (n−1) = 40). The paired t-test in the same figure is t₇, implying n = 8. Either two subjects were dropped from the ANOVAs without comment, or a df is misreported. The main experiment is internally consistent (F(8,56), F(9,63) → n = 8).

3. **Hierarchical test df is unexplained.** Group-level HBI weights are tested as t₉ with 8 subjects. The HBI framework's effective df is not the naive n−1, but the paper does not state what generates 9.

4. **The dissociation rests on a minority of choice points.** Even on the Pareto-optimal mazes, the fraction of informative states tops out around 0.2 (Fig. 3a). ~80% of location–goal pairs cannot distinguish the strategies. This is honestly reported, but it means the headline structure-vs-vector claim rests on a fifth of the state space.

5. **Maze order is confounded with experience.** The four topologies are presented in a fixed sequence (fully → original → A → B) with no counterbalancing, so "learning is faster on later mazes" and "later mazes are better designed" cannot be separated. The authors identify this themselves when explaining why behaviour on the *original* maze differed between experiments 1 and 2, but the confound is not controlled — it is inherent to the design.

6. **Minor internal inconsistency in the informative-state definition.** Maze-optimisation computes `F_inf` **before** removing backward actions; the behavioural analysis computes optimal-choice-rate **after** excluding them. The mazes were therefore optimised against a slightly different quantity from the one measured.

7. **Visually cued goals limit the interpretation and constrain transferability.** The authors acknowledge that visual acuity may cap the accuracy of goal-location estimates at distance. More fundamentally, an LED beacon visible across the maze means the "vector" strategy is partly a beacon-approach behaviour rather than a purely internal Euclidean computation. It also means the whole assay is unavailable in darkness.

8. **The learning model is descriptive, not mechanistic.** Learning is captured by strategy × log(steps) interaction terms. This fits the curves well but is a parameterisation of learning, not a model of what is learnt or how.

9. **"Structure-based" does not entail planning.** The authors are appropriately careful about this — they note that a cached policy over the (location × goal) product space could in principle solve the task and argue against it on normative grounds (sample inefficiency, no generalisation across goals) rather than on evidence. Choosing shortest-path-reducing options is equally consistent with a successor-representation-like cached geodesic policy as with sequential forward planning. The paper does not, and does not claim to, adjudicate between these.

10. **No neural data.** The framing throughout is "we anticipate the assay will be useful". This is an assay-development paper.

---

## 4. Relevance to hm2p

**Overall relevance: MODERATE — high methodologically, near-zero for the neural paper.**

### 4.1 The critical disanalogy: there is no goal in our task

Every headline analysis in Pereira et al. is **defined relative to a cued goal**. Excess steps, optimal choice rate, `P_S`, `P_V`, the informative-state criterion, and both the Vector and Structure model components all require `g`. The hm2p paradigm is free exploration: no reward, no cue, no goal, no trials. **The structure index, the vector index and the optimal-choice-rate analysis cannot be computed on our data at all.** Any attempt to import the headline framework directly would require inventing a surrogate goal, which would be unfalsifiable.

This must be stated plainly in any use we make of the paper, because the surface similarity ("complex maze, graph analysis, mice, route selection") invites exactly that mistake.

### 4.2 Does it supersede or contradict Rosenberg et al. 2021? No — it extends sideways.

| | Rosenberg 2021 | Pereira 2026 | hm2p |
|---|---|---|---|
| Maze | Binary tree, 127 corridors, fixed | 6×6 grid, reconfigurable | q-rose maze, 23 cells, fixed |
| Goal | Hidden water port (or none) | Randomised, LED-cued each trial | **None** |
| Question | How is *exploration* structured? | How are *goal-directed routes* selected? | How does exploration change without vision? |
| Answer | Local turning rules (forward bias + L/R alternation) explain efficiency without a global map | Choices track shortest-path-to-goal → structure knowledge | Coverage becomes less directed; local rules preserved |

The two are **not in conflict**: Rosenberg studied exploration in the absence of a goal and concluded a global map was not *required* to explain exploration efficiency; Pereira studied goal-directed navigation and concluded structure knowledge *is* used when there is a goal to be structured toward. Both can be true.

**Rosenberg remains the closer analogue for hm2p**, because like ours it is a goal-free (or goal-sparse) exploration paradigm. The framework the behaviour manuscript leans on is not superseded.

What Pereira **adds** to the lineage, for us:
- A third independent demonstration, in a completely different maze and a completely different task, that **u-turn avoidance is a dominant behavioural component** (Anti-backward is among the largest weights, t₉ = 8.39; Fig. S3 shows u-turns are rare once navigation commences). This now agrees with Rosenberg's forward bias and with our own observation that local turn rules (alternation, backtracking rate 48–51%) are **preserved across light and dark**. Three studies converge on a body-based anti-backtracking default. This materially strengthens our claim that what darkness changes is not the local decision rule.
- Independent support for **very rapid structure learning** (knowledge evident on the first session in a new environment), which sits alongside Rosenberg's one-shot homing and our own single-trial dark adaptation (first dark epoch near-normal, 0.57 vs 0.30, p = 0.0001, r = 0.89). Three timescale-convergent results.

### 4.3 What it gives us that is directly actionable

**(a) Betweenness centrality as a graded per-cell covariate — the cheapest and most concrete gain.**

Our route-stereotypy result is currently a **categorical** three-way split: corridor coverage drops (p = 0.0004), junction coverage drops (p = 0.003), dead-end coverage does not (p = 0.169). Betweenness centrality is the principled **continuous** version of exactly that split, and it makes a sharper prediction than the categorical test can.

If darkness collapses navigation onto a smaller route network, the cells that lose occupancy should be those of **intermediate** betweenness — the optional connecting routes — rather than the high-betweenness bottlenecks (which every surviving route must still traverse) or the dead-ends (which are destinations, and which we already know are preserved). A monotone decrease would instead indicate simple network contraction; an inverted-U would indicate selective pruning of redundant routes.

Concretely: per session, compute per-cell (light − dark) occupancy change, Spearman-correlate against per-cell betweenness centrality, take one rho per session, then paired/one-sample Wilcoxon across the 23 sessions (and the 15-session first-session independence check). This is non-parametric throughout, needs no new data, and `maze/topology.py` already provides the all-pairs shortest-path matrix `dist` that betweenness is computed from — betweenness itself is a `networkx` one-liner (networkx is already a project dependency per the installed skills list) or ~20 lines from the existing BFS.

**(b) Characterising the q-rose maze on Pereira's axes — a reviewer shield.**

We can compute `F_inf`-analogues and the betweenness-centrality distribution for the q-rose maze and state quantitatively where it sits relative to the Grid-maze and the Rosenberg labyrinth. The behaviour manuscript's limitation list already concedes the small state space (23 cells); a reviewer asking "is 23 cells enough structure to support claims about route selection?" is better answered with a graph-theoretic characterisation than with an assertion. Note the caveat that `F_inf` itself is goal-dependent and so is a description of the maze's *potential* to dissociate strategies, not of our animals' behaviour.

**(c) A better-specified model for the controller-switch question — and a specific reason the previous null may be a diluted-statistic null.**

This is the part worth taking seriously. Our `scripts/run_controller_switch.py` came back null (allo-follow light 0.253 vs dark 0.228, p = 0.62; ego accuracy p = 0.99), and the conclusion recorded at the time was "the effect is a weak distributed bias, too small to localise per-choice". Pereira's model differs from ours in three ways, and one of them points at a probable methodological problem in ours rather than a true null:

1. **Graded vs winner-take-all preferences.** Our `egocentric_choice` and `allocentric_choice` return a single predicted arm and return `None` on ties. Pereira assigns every candidate a **scalar** preference and lets the softmax weigh them. A rule that is 60% right about which arm is *better* contributes nothing to a winner-take-all match but contributes real likelihood to a graded model.
2. **All choices vs the disagreement subset.** Pereira fits every step; we scored only conflict trials. In our case this is a smaller loss than it sounds — conflict trials are 5,307 of 7,613 events (70%) — so this is not the main problem.
3. **U-turns modelled explicitly vs absorbed into the denominator.** *This is the likely problem.* In our implementation `neighbours()` includes the arm the animal arrived from, `classify_turn` labels it `"back"`, and **neither rule ever predicts it**: `egocentric_choice` only ever returns a `left`/`right`/`forward` arm, and `allocentric_choice` picks maximum recency, which the just-visited arm never has. Yet `conflict_follow_rate` deliberately counts backtracks in the **denominator**. Given a backtracking rate of 48–51% at junctions in this maze (behaviour manuscript, Results §1), a large share of every denominator is a u-turn that no candidate rule can ever win. That caps both follow rates near 0.5 and is almost certainly why the observed allo-follow (0.25) and ego accuracy (0.28) both sit *below* the naive two-way chance of 0.5 — a pattern that should have been read as a specification problem, not as "no rule predicts choices".

   Pereira's treatment is the correct fix and is exactly what our implementation lacks: **make anti-backward its own model component** so that u-turn variance is absorbed by a dedicated weight instead of diluting the strategy comparison, and/or condition the conflict test on non-u-turn choices.

**Recommendation on re-opening the controller-switch line: yes, but as one bounded re-analysis, not a new campaign.** The prior null was measured with a statistic that structurally could not exceed ~0.5 and that spent roughly half its denominator on an action no rule modelled. That is a good enough reason to run it once more properly. The re-analysis is a conditional-logit fit per session with components (i) anti-backward, (ii) forward/momentum, (iii) turn-alternation, (iv) graded recency of each candidate's target cell, (v) graded frontier proximity (negative graph distance from the candidate to the nearest unvisited cell), each with a light/dark interaction; headline = the light − dark difference in the recency/frontier weight, one number per session, paired Wilcoxon across 23 sessions. That is the design already written in `docs/plan-controller-switch-behaviour.md` §"Models and comparison" item 1, which was specified but never run — only the item-2 conflict-proportion was. Statistical inference stays non-parametric at the session level; the model is only the per-session summariser.

**Honest probability assessment:** I would put maybe 35–45% on this converting the null into a detectable light/dark difference. The u-turn dilution argument is a real and specific defect, which is why this is not the 10–15% I would have given before reading the paper. But 7,613 junction events over 23 sessions (median 311/session, ~100 conflict trials per condition per session) is two orders of magnitude below Pereira's 282,855, and our sessions are ~10 minutes against their 40–45. A graded model extracts more per event; it cannot manufacture events.

The second, independent reason to run it: the fitted per-epoch **weight** is a continuous covariate, where the conflict-follow rate was a noisy binomial proportion. That makes it usable for the neural×behaviour coupling analysis (H-N11) in a way the current statistic is not — it can be regressed against per-epoch population state. Given the neural side currently has no positive result, a continuous behavioural covariate is worth more than another behavioural p-value.

### 4.4 What it does NOT give us

- **Nothing for the neural paper.** No RSP, no HD cells, no darkness, no recordings of any kind. Its only legitimate use in the neural manuscript is a one-line Introduction citation supporting "mice acquire and use knowledge of environment structure rapidly". It must not be allowed to pull the neural narrative toward planning or route-selection: we have no goal, no reward, and therefore no plan to detect. Given the neural story is currently "representation preserved in darkness across every measure, behaviour changes", the temptation to import a planning frame is real and should be resisted.
- **No help with the cell-type comparison** (Penk+ vs Penk⁻CamKII+), which remains underpowered at 11 vs 4 animals for reasons this paper does not address.
- **No help with the darkness manipulation itself.** The task depends on a visible LED beacon; it cannot be run in darkness. There is no dark condition anywhere in the paper.
- **The maze-reconfiguration extension** (adding/removing one link per session to create shortcut and detour problems) is a genuinely good design suggestion, but it requires a new experiment, not a re-analysis of existing data. Worth noting for future work; not actionable now.

---

## 5. Key Figures and Results to Cite

- **Figure 5b** — mixture-of-strategies component weights. The load-bearing quantitative result: Structure ≫ Vector, and Anti-backward large. Cite for the claim that structure knowledge dominates goal-directed choice, and for the u-turn-avoidance convergence.
- **Figure 4c** — optimal choice rate above chance on the **first day** of each new maze. Cite for rapid structure learning.
- **Figure 3a,b** — maze optimisation space and the four selected topologies coloured by betweenness centrality. Cite when introducing betweenness centrality as a maze-characterisation metric.
- **Figure S3** — egocentric action bias; u-turns rare once navigation commences. Cite alongside Rosenberg's forward bias when arguing that local anti-backtracking defaults are conserved across mazes and tasks.

**Suggested citation contexts:**

*Behaviour manuscript, Discussion (rapid structure learning):* "Rapid acquisition of environment structure appears to be a general feature of rodent maze behaviour: mice learn the location of a hidden reward within ~10 experiences and home correctly on their first excursion (Rosenberg et al. 2021), and show choice behaviour reflecting knowledge of maze topology from the first session in a newly configured environment (Pereira et al. 2026). The single-epoch adaptation to darkness reported here operates on a comparably fast timescale."

*Behaviour manuscript, Discussion (preserved local rules):* "A tendency to avoid reversing direction is a robust feature of rodent navigation across maze designs and task demands, appearing as a forward bias during free exploration of a binary labyrinth (Rosenberg et al. 2021) and as a large anti-backward component in models of goal-directed route selection in a reconfigurable grid maze (Pereira et al. 2026). The preservation of backtracking rate and turn alternation across light and dark epochs here is consistent with this default operating independently of visual input."

*Behaviour manuscript, Methods (if betweenness analysis is added):* cite for the definition and for its use as a maze-structure descriptor.

---

## 6. Implications for the hm2p Pipeline

### 6.1 Worth adopting
- **Betweenness centrality** as a per-cell graph metric in `src/hm2p/maze/topology.py`, and as a covariate for the coverage-change analysis (§4.3a). Highest value-per-effort item in this paper.
- **Graded, additively-combined choice components with an explicit anti-backward term**, fit per session as a conditional logit, replacing the deterministic winner-take-all rules in `src/hm2p/maze/choice_models.py` (§4.3c). The existing `extract_choice_events` already produces the right event structure and can be reused unchanged.
- **The index normalisation `I_q = (c_q − r_q)/(o_q − r_q)`** — correcting a rule's hit rate for the expected hit rate under uniform choice among available actions. Our current `rule_accuracies` reports a raw hit rate with no such correction, which is part of why the numbers are hard to interpret against chance.

### 6.2 Not applicable
- Excess steps, optimal choice rate, structure index, vector index — all require a goal.
- Maze optimisation by `F_inf` — our maze is fixed and already built.
- Hierarchical Bayesian inference (Piray et al. 2019) as the *inferential* layer would conflict with the project's non-parametric policy. It could be used as a per-session/per-subject fitter with non-parametric testing across sessions on top, but the simpler per-session conditional logit + paired Wilcoxon already specified in `docs/plan-controller-switch-behaviour.md` is adequate and avoids the extra dependency.

### 6.3 Statistical note
Adopt none of this paper's inferential practice. It uses rmANOVA and t-tests throughout on n = 8, has an unresolved n = 6 vs n = 8 discrepancy in the pilot degrees of freedom, and reports hierarchical tests at t₉ from 8 subjects without explaining the df. When citing its findings, cite the effects, not the p-values.

---

## Summary Assessment

**Overall relevance to hm2p: MODERATE.**

A well-executed behavioural assay-development paper with a genuinely useful methodological core (graph-based maze optimisation; graded mixture-of-strategies choice modelling) and one clear scientific claim (mice select routes using shortest-path structure, learnt within a session). Its weaknesses are a small n analysed with parametric tests, an unexplained df discrepancy, a fixed maze order confounded with experience, and no neural data.

Its value to us is **methodological and lateral, not conceptual**. It does not supersede or contradict Rosenberg et al. 2021, which remains the closer analogue because our paradigm, like Rosenberg's, has no goal — and every headline analysis in Pereira is goal-dependent and therefore uncomputable on our data.

Three concrete takeaways:
1. **Betweenness centrality** turns our categorical corridor/junction/dead-end route-stereotypy result into a graded, better-powered, non-parametric test. Cheapest and most likely to pay off.
2. Reading Pereira's model exposed a probable **specification defect in our own controller-switch analysis** — u-turns occupy roughly half the denominator while no candidate rule can ever predict them, which is why both rule accuracies sat below naive chance. That justifies one bounded re-analysis with a graded, anti-backward-aware conditional logit, at perhaps 35–45% odds of changing the answer, plus a side benefit (a continuous per-epoch covariate for neural coupling) that holds even if the light/dark comparison stays null.
3. A third independent demonstration that **u-turn avoidance and rapid structure learning** are general, strengthening two Discussion claims in the behaviour manuscript.

For the neural paper it is close to irrelevant beyond a single Introduction citation, and should be actively kept from reframing that manuscript around planning.
