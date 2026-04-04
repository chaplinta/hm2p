# Zagha et al. 2022 — Detailed Summary

## Citation

Zagha E, Erlich JC, Lee S, Lur G, O'Connor DH, Steinmetz NA, Stringer C, Yang H. 2022. "The Importance of Accounting for Movement When Relating Neuronal Activity to Sensory and Cognitive Processes." *The Journal of Neuroscience* 42(8):1375-1382. doi:10.1523/JNEUROSCI.1919-21.2021

**Affiliations:** University of California Riverside; NYU Shanghai; NIH/NIMH; UC Irvine; Johns Hopkins University; University of Washington; HHMI Janelia Research Campus.

---

## Overview

This TechSights review argues that movement-related neural activity is ubiquitous throughout the mouse brain and must be accounted for before attributing neural signals to sensory or cognitive processes. Drawing on recent findings that movement is the dominant source of variance in neural activity across cortex (including primary sensory areas), the paper reviews task designs for separating movement-related from sensory/cognitive activity, presents three case studies where failing to account for movement would have led to incorrect conclusions, and discusses fundamental couplings between sensory, motor, and cognitive processes that may prevent complete dissociation.

---

## Key Findings and Arguments

### 1. Movement dominates neural variance brain-wide

Recent studies (Musall et al. 2019; Steinmetz et al. 2019; Stringer et al. 2019) demonstrated that task-uninstructed movements explain more variance in cortical activity than sensory stimuli, choices, or rewards — even in primary sensory cortex. Movement-related signals are multidimensional (encoding multiple aspects of behaviour), precede overt movement by at least 50 ms, and are observed in cortex, thalamus, basal ganglia, and midbrain. This is not a rodent-specific phenomenon; similar findings exist in zebrafish, Drosophila, and C. elegans.

### 2. Task design matters

Three task designs are compared: (a) Go/NoGo without delay — sensory, choice, motor, and reward signals overlap temporally, making attribution impossible; (b) Go/NoGo with delay — separates sensory from instructed motor signals, but uninstructed movements persist during the delay; (c) 2AFC — dissociates choice from non-selective movement initiation and reward expectation. The paper recommends 2AFC designs with delays, but notes that no design fully eliminates uninstructed movement confounds.

### 3. Case studies

Three examples illustrate how movement confounds lead to misinterpretation: (a) In a whisker discrimination task, mice developed a sampling strategy (retracting whiskers to only sample the target location), converting a discrimination task into a detection task — apparent "filtering" of distractor signals was actually reduced sensory drive (O'Connor et al. 2010). (b) In a memory-guided orienting task, rats made subtle postural movements toward the expected reward port during the delay period, confounding "working memory" signals with overt motor preparation (Erlich et al. 2011). (c) In a category learning task, PFC neurons appeared to remap category representations after a rule switch, but this activity could not be separated from the Go response due to the Go/NoGo design without delay (Reinert et al. 2021).

### 4. Sensory-motor coupling is intrinsic

The paper argues that movement confounds are not merely experimental limitations but reflect fundamental features of neural organisation. Primary sensory cortices receive reafference (sensory consequences of self-generated movement), efference copy (internal copies of motor commands), and top-down signals (attention, expectation, arousal). Primary somatosensory cortex can even drive whisker movements. Complete dissociation of sensory, motor, and cognitive signals may be impossible in principle.

### 5. Recommendations

Record as many behavioural features as possible. Build "null" models relating neural activity to behaviour in the absence of a goal-directed task (Stringer et al. 2019). Use perturbation experiments to test causal contributions. Create baseline behavioural models using neural activity recorded before learning.

---

## Relevance to hm2p

### 1. Movement confounds in the light/dark comparison

This is the most critical concern for hm2p. If mice move differently in light vs dark epochs (different speeds, angular velocities, exploration patterns), and movement-related signals dominate RSP neural variance, then any apparent difference in HD tuning between conditions could be a movement confound rather than a visual-anchoring effect. The paper's framework demands that speed, angular head velocity, and all tracked behavioural variables be compared between light and dark before interpreting neural differences.

### 2. Movement as a confound for cell-type differences

If Penk+ and Penk-CamKII+ neurons differ in their sensitivity to movement variables (speed, angular velocity), observed differences in HD tuning could reflect different movement encoding rather than different HD anchoring strategies. Including movement variables in encoding models (e.g., NEMOS GLMs) is necessary to control for this.

### 3. The "null model" approach applies directly

The paper recommends building a null model of neural activity explained by behaviour alone. For hm2p, this means fitting GLMs with speed, angular velocity, position, and other movement variables before adding HD as a predictor. Only variance explained by HD beyond what movement variables explain should be attributed to HD tuning.

### 4. Uninstructed movements during darkness

When lights turn off, mice may exhibit orienting responses, freezing, whisking, or other uninstructed behavioural changes that are not captured by DLC tracking of ears and body. These unseen movements could drive neural activity changes that are misattributed to loss of visual input. The paper motivates comprehensive behavioural monitoring — the hm2p project should consider whether the five tracked body parts (ears, mid-back, center, tail base) are sufficient.

### 5. Reafference in RSP

RSP receives both visual and self-motion signals (Chaplin & Margrie 2020). During free movement, RSP neurons receive reafferent visual signals (optic flow from self-motion) and efference copies. In darkness, the reafferent visual component disappears while the efference copy may persist. This asymmetry could account for apparent changes in tuning that are not related to the HD signal per se but to the loss of expected visual reafference.
