# Chaplin & Margrie 2020 — Detailed Summary

## Citation

Chaplin TA, Margrie TW. 2020. "Cortical circuits for integration of self-motion and visual-motion signals." *Current Opinion in Neurobiology* 60:122-128. doi:10.1016/j.conb.2019.11.013

**Affiliations:** Sainsbury Wellcome Centre for Neural Circuits and Behaviour, UCL; Monash University / ARC Centre of Excellence for Integrative Brain Function.

---

## Overview

This review examines cortical circuits that mediate the integration of self-motion cues (vestibular, motor/locomotion) with visual-motion signals in the mouse visual cortex. The central argument is that visual cortex is not a unimodal processing region — it receives substantial non-visual self-motion input, including vestibular rotation signals and locomotion-related activity. The review maps out the known circuitry that delivers these signals to V1 and higher visual areas (HVAs), with a particular focus on retrosplenial cortex (RSP) as a key intermediary node. The paper concludes by arguing that freely-moving paradigms with cell-type-specific recording are needed to understand the full function of these circuits under naturalistic conditions.

---

## Key Findings and Arguments

### 1. Visual cortex is fundamentally multisensory for self-motion

The classical hierarchical model of visual processing (V1 receives visual input from dLGN, feeds forward to HVAs, then to association cortex) is incomplete. Mouse V1 receives non-visual self-motion signals from multiple sources:

- **Vestibular signals** arrive in V1 layer 6, relayed at least in part via RSP (Velez-Fort et al. 2018). Passive whole-body rotation evokes direction-selective spiking and membrane potential changes in V1 neurons in complete darkness.
- **Locomotion signals** reach V1 through the mesencephalic locomotor region and basal forebrain (via VIP interneuron disinhibition of SOM neurons, increasing L2/3 activity), and through ACC/M2 projections (to L1, L2/3, and L6). Thalamic locomotion signals also arrive via dLGN and LP.
- Locomotion responses in V1 include speed tuning in both calcium signals and spiking activity.

### 2. Visual-vestibular integration in V1 is additive, not subtractive

A key finding from Velez-Fort et al. 2018: visual-vestibular membrane potential responses to passive rotation are an arithmetic sum of the visual and vestibular responses. Self-motion information is not simply subtracted from visual input to isolate external object motion. Similarly, spiking responses to locomotion plus visual stimulation are a weighted sum (Saleem et al. 2013). This has implications for models of optic flow compensation and spatial coding.

### 3. Most prior studies used incomplete stimulus conditions

The review makes a methodological critique: most self-motion studies use only 2 of 3 self-motion cues (visual, vestibular, motor). The Venn diagram (Figure 2a) illustrates that:

- Passive movement = visual + vestibular (no motor command) — rare in nature
- Head-fixed treadmill = motor + visual (no vestibular) — never occurs naturally
- Darkness with movement = motor + vestibular (no visual) — natural but limited context for visual cortex

Natural locomotion in daylight engages all three simultaneously. Studying pairwise combinations may reveal special cases rather than the default operating mode of these circuits.

### 4. Motor commands alone are unreliable predictors of optic flow

The review questions whether locomotion speed signals in V1 can serve as reliable predictors of expected optic flow. The argument: running biomechanics are heavily modulated by surface conditions (slope, substrate integrity, etc.), so a motor efference copy of running speed does not straightforwardly predict the resulting head motion or retinal image flow. The vestibular system — which directly senses head acceleration and is anatomically adjacent to the visual organs — is a more reliable signal for moment-to-moment self-motion estimation.

### 5. Spatial navigation signals co-exist with self-motion signals in visual cortex

Place-field-like spatial signals have been found in V1 (Saleem et al. 2018), HVAs (Minderer et al. 2019), and RSP (Cho & Sharp 2001; Mao et al. 2017). It is unknown whether the same cells encode both allocentric spatial information (e.g. head direction, place) and self-motion signals (e.g. vestibular or locomotion responses), or whether these are supported by separate circuits.

### 6. Freely-moving paradigms are necessary

The review argues that head-fixed experiments, while valuable for stimulus control, fundamentally limit the range of naturalistic self-motion. Key points:

- Head-fixed movement is restricted to one dimension (rotation OR translation, not both)
- Movement kinematics under head-fixation differ from natural movements
- Fully naturalistic stimulus statistics are difficult to reproduce experimentally

Advances enabling freely-moving experiments include:
- Head-mounted gyroscopes/accelerometers light enough for mice (Wilson et al. 2018)
- Deep learning pose estimation — specifically DeepLabCut (Mathis et al. 2018)
- Head-mounted two-photon microscopes (Zong et al. 2017)
- Statistical methods for continuous, non-trial-based behaviour analysis (Wiltschko et al. 2015)

---

## RSP-Specific Content

This is the most relevant section for the hm2p project. The review identifies RSP as a central node in the self-motion integration circuit:

### RSP as a multimodal hub

RSP neurons show vestibular, motor, and visual responses (Velez-Fort et al. 2018; Cho & Sharp 2001; Murakami et al. 2015; Zhuang et al. 2017). The review describes RSP as having "the characteristics of a multimodal association area."

### RSP inputs

RSP receives converging inputs from multiple systems (Figure 1):

1. **Head direction signals** from the postsubiculum (PoS), which itself receives input from the anterior dorsal thalamic nucleus (ADN) via the classical HD circuit (DTN -> LMN -> ADN -> PoS -> RSP)
2. **Vestibular signals** via the anterior thalamic nuclei (part of the anterior ascending vestibular pathway)
3. **Grid/place cell signals** from the medial entorhinal cortex (MEC) via the hippocampal formation (HPF)
4. **Motor planning signals** from ACC/M2
5. **Visual input** from V1 and HVAs (bidirectional connectivity)

### RSP outputs to V1

RSP sends abundant projections to V1 (Velez-Fort et al. 2014; Makino & Komiyama 2015). The review argues RSP contributes more than just vestibular signals to V1 — it may act as a "task-dependent, selective gateway for non-visual signals such as motor planning or spatial navigation" (citing Saleem et al. 2018).

### RSP vestibular-to-V1 pathway

A specific RSP population projects to V1 layer 6, carrying vestibular rotation signals. At least a fraction of these RSP neurons receive input from the anterior thalamic nuclei (Velez-Fort et al. 2018). Rotation modulates activity in V1 L5 and L6 with little or no activation of superficial layers.

### Open questions about RSP raised in the review

1. Does RSP perform multisensory integration itself, or does it relay non-visual signals to visual cortex and receive integrated signals in return?
2. If RSP performs integration, how do its integrative operations differ from those in visual cortex?
3. What is the layer-specific input circuitry in RSP for self-motion signals? (The review notes: "no study to our knowledge has characterised the layer-specific inputs and circuitry in HVAs, or RSP for self-motion signals")
4. Are the cells encoding allocentric spatial information (HD, place) in RSP the same cells encoding self-motion signals, or are these supported by separate circuits?

### RSP spatial signals

The review cites Cho & Sharp 2001 and Mao et al. 2017 for head direction and spatial context signals in RSP. RSP is described as "well known to encode spatial signals" and "ideally situated as a gateway between visual cortex and hippocampal formation" (citing Wyass & van Groen 1992).

---

## Visual-Vestibular Integration

### Pathway: vestibular organs to visual cortex

The review outlines a specific anatomical pathway:

**Vestibular nuclei -> DTN -> LMN -> ADN -> PoS -> RSP -> V1 (layer 6)**

This is the anterior ascending vestibular pathway (Cullen & Taube 2017). The RSP-to-V1 projection is the final cortical relay. At least some RSP neurons projecting to V1 receive anterior thalamic input (Velez-Fort et al. 2018).

### Integration mode

In V1: additive (arithmetic sum of visual + vestibular at the membrane potential level). Spiking output: weighted sum of locomotion + visual stimulation. Other reported effects include gain modulation, increased response reliability, surround suppression changes, and mismatch signals.

### Locomotion pathway

Separate from vestibular signals: mesencephalic locomotor region / basal forebrain -> VIP interneurons -> SOM disinhibition -> L2/3 excitation (in darkness). Context-dependent: during visual stimulation, locomotion increases both VIP and SOM activity, suggesting the mechanism changes with behavioural state.

---

## Open Questions Raised

1. **RSP integration vs relay:** Does RSP integrate multimodal self-motion cues or simply relay them to visual cortex?
2. **Layer-specific circuitry in RSP and HVAs:** Unstudied at the time of writing.
3. **Cell-type identity of self-motion circuits:** Which specific cell types in RSP carry vestibular vs HD vs spatial vs motor signals?
4. **Overlap of spatial and self-motion coding:** Are HD/place cells in visual cortex and RSP the same neurons that respond to vestibular/locomotion stimulation?
5. **Active vs passive vestibular processing:** Whether vestibular modulation of V1 persists during active movement (as opposed to passive rotation) remains unclear.
6. **Role of visual cortex in darkness:** What function does visual cortex self-motion activity serve when there is no visual input? Lesion studies suggest V1 may be important for spatial learning in darkness (Whishaw 2004).
7. **Complete trimodal integration:** How do visual, vestibular, and motor signals combine under naturalistic conditions with all three present?

---

## Relevance to hm2p

This review is directly foundational to the hm2p project — it was written by the same first author (TAC) and articulates the scientific framework that motivates the current experiment. Several specific connections:

### 1. The experiment fulfils the review's call for freely-moving, cell-type-specific recording

The review explicitly calls for (a) freely-moving paradigms, (b) cell-type-specific recording using head-mounted two-photon microscopy, and (c) studies in RSP. The hm2p dataset — freely-moving mice in a rose maze, head-mounted 2P calcium imaging of genetically defined RSP populations — directly addresses all three.

### 2. Light/dark manipulation maps onto the visual vs idiothetic separation

The review discusses the darkness condition as the one freely-moving paradigm that isolates non-visual self-motion (motor + vestibular) from visual input. The hm2p 1-min light-on / light-off epochs implement exactly this manipulation. The review's question — "what role does visual cortex activity play in self-motion perception in darkness?" — extends to RSP: how does each RSP subpopulation maintain or lose HD tuning when visual landmarks are removed?

### 3. Cell-type specificity in RSP is the key gap

The review identifies the cell-type composition of RSP self-motion circuits as unknown. The hm2p project's comparison of Penk+ vs Penk-CamKII+ RSP neurons directly addresses this. The review's open question — "are the cells encoding allocentric spatial information the same cells encoding self-motion signals, or separate circuits?" — can be partially answered by determining whether these two genetically defined populations have distinct roles in visual anchoring vs path integration.

### 4. RSP as gateway vs integrator

The review poses the question of whether RSP integrates or relays. If Penk+ and Penk-CamKII+ neurons show different patterns of visual landmark dependence (e.g. one population drifts in darkness while the other maintains stable HD tuning), this would support the integrator model with distinct subpopulation roles. If both populations behave identically, it would suggest the integration occurs elsewhere.

### 5. HD circuit context

The review maps the HD circuit: DTN -> LMN -> ADN -> PoS -> RSP. The hm2p project records from RSP HD cells at the end of this circuit. The review's framework predicts that RSP HD cells should receive both path-integration-based HD signals (via the thalamic/vestibular pathway) and visual landmark information (via visual cortex feedback or direct visual input). Differentiating these contributions is the core hypothesis of the hm2p project.

### 6. Specific methodological tools mentioned

The review highlights DeepLabCut for pose estimation and head-mounted 2P microscopy — both used in hm2p. It also mentions the importance of analysing continuous, naturalistic behaviour rather than trial-based paradigms, which applies directly to the rose maze exploration data.

### 7. The additive integration finding constrains expectations

The review's finding that V1 visual-vestibular integration is additive (not subtractive) raises the question: is RSP integration also additive? If so, gain modulation between light and dark epochs in RSP may reflect a simple additive/subtractive visual contribution rather than a more complex gating mechanism. This should be considered when interpreting changes in tuning curve amplitude or width between conditions.

---

## Key References Cited

| Reference | Relevance |
|---|---|
| Velez-Fort et al. 2018 (Neuron) | RSP-to-V1 L6 vestibular pathway; anterior thalamic input to RSP |
| Cho & Sharp 2001 (Behav Neurosci) | HD, place, and movement correlates in rat RSP |
| Mao et al. 2017 (Nat Commun) | Spatial context signals in RSP |
| Saleem et al. 2013 (Nat Neurosci) | Weighted-sum integration of visual + locomotion in V1 |
| Saleem et al. 2018 (Nature) | Spatial position coding in visual cortex and hippocampus |
| Mathis et al. 2018 (Nat Neurosci) | DeepLabCut for pose estimation |
| Zong et al. 2017 (Nat Methods) | Head-mounted 2P microscopy |
| Cullen & Taube 2017 (Nat Neurosci) | HD circuit and vestibular pathway review |
| Wyass & van Groen 1992 (Hippocampus) | RSP-hippocampal connectivity |
| Vann et al. 2009 (Nat Rev Neurosci) | RSP function review |
