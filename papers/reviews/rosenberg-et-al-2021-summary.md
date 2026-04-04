# Rosenberg et al. 2021 — Detailed Summary

## Citation

Rosenberg M, Zhang T, Perona P, Meister M. 2021. "Mice in a labyrinth show rapid learning, sudden insight, and efficient exploration." *eLife* 10:e66175. doi:10.7554/eLife.66175

**Affiliations:** Division of Biology and Biological Engineering, and Division of Engineering and Applied Science, California Institute of Technology, Pasadena, USA.

---

## Overview

This paper studies unconstrained mouse behaviour in a complex binary-tree labyrinth (63 T-junctions, 64 endpoints) using automated video tracking. Mice make ~2,000 navigation decisions per hour, learn the location of a hidden water reward after only ~10 reward experiences, and in some cases show discontinuous improvements suggestive of "sudden insight." The underlying exploration strategy is largely explained by local turning rules rather than a global cognitive map, and mice demonstrate one-shot learning of the homing path on their first ever excursion into the maze.

---

## Key Findings and Arguments

### 1. Few-shot reward learning

Water-deprived mice discovered the reward port in <2,000 seconds and <17 bouts, then learned the optimal 6-step path (10-bit choice) after approximately 10 reward experiences. This is ~1,000-fold faster than learning rates in typical two-alternative forced-choice tasks. Even late in the session, mice continued to execute some long exploratory paths interspersed with optimal reward runs.

### 2. Discontinuous learning ("sudden insight")

In at least 5 of 10 rewarded mice, behavioural performance changed discontinuously, identifiable to ~200-second precision. Two types of steps were observed: (a) a sudden increase in reward collection rate, and (b) a later, independent increase in the frequency of long direct paths to the reward from distant locations. These steps were temporally separated, suggesting multiple discrete insights about the maze structure.

### 3. One-shot learning of the home path

On their very first excursion into the maze, mice navigated to the deepest endpoints and then returned home directly — making correct choices at six successive three-way junctions. This homing path was not a retrace of the outbound path (minimal overlap). Mice did not build up the home path gradually; instead they appeared to acquire a homing strategy within a single bout.

### 4. Navigation is robust to physical cue removal

Rotation of the maze by 180 degrees (which reverses any deposited odour cues) did not prevent experienced mice from navigating to the reward. Three of four mice went directly to the correct location on their first post-rotation entry. However, subtle behavioural effects persisted for over an hour, indicating that physical cues are used but are not strictly required.

### 5. Local turning rules explain exploration efficiency

Mice explored roughly twice as efficiently as a random walker (E = 0.39 vs 0.23). This efficiency was explained by three local rules at T-junctions: (a) proceed forward rather than reverse (strong bias), (b) alternate left-right turns rather than repeat the same direction, and (c) mild preference for branching off the main corridor. These rules require no global memory of places visited.

### 6. Exploration dominates behaviour

Even water-deprived mice spent ~84% of their maze time exploring rather than executing reward runs or leaving. Unrewarded mice explored at 95%. Exploration efficiency declined modestly (~23%) over the night. Both rewarded and unrewarded groups explored at statistically indistinguishable rates.

---

## Relevance to hm2p

### 1. Rose maze vs binary labyrinth — shared structure

The hm2p rose maze and the Rosenberg labyrinth share a branching structure with dead ends and forced choice points. The local turning rules identified here (forward bias, turn alternation) may apply to mouse behaviour in the rose maze and could be tested directly against the hm2p DLC tracking data. Comparing exploration efficiency metrics between the two maze designs would contextualise hm2p behavioural findings.

### 2. Exploration strategy analysis for hm2p

The paper provides a framework for quantifying maze exploration: node sequences, bout segmentation, exploration efficiency (N32), and turning bias at junctions. Several of these metrics could be adapted for the rose maze to characterise how mice distribute their time and heading samples across the maze, which is relevant for ensuring uniform HD sampling across all directions.

### 3. Behavioural state segmentation

The three-state model (explore, drink/seek, leave) provides a template for segmenting rose maze behaviour into exploration vs goal-directed movement. If RSP HD tuning properties differ between exploratory and goal-directed states, this could interact with the light/dark comparison.

### 4. Implications for the darkness condition

The paper shows that mice can navigate effectively using internal representations rather than external cues (post-rotation performance). This is directly relevant to the hm2p darkness epochs: mice may maintain effective navigation in total darkness, and their exploration strategy may not change dramatically, which would help isolate neural effects of light removal from behavioural confounds.

### 5. Learning dynamics within sessions

The finding that behavioural strategy evolves within a single session (few-shot learning, sudden insight) raises a potential confound for hm2p: neural activity patterns early in a session may differ from late patterns due to learning, independent of the light/dark manipulation. This should be controlled for by examining whether HD tuning properties change systematically over the session duration.
