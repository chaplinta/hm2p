# bioRxiv Scan — 5 November 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-11-05. Search window: last 7 days (2026-10-29 to 2026-11-05).
Searches covered: retrosplenial cortex (RSC/RSP), Penk/enkephalin + cortex, head direction
+ two-photon imaging, head direction + darkness/landmarks/drift, spatial navigation + maze
+ calcium imaging, visual processing in RSP, head-mounted miniature two-photon microscopy,
neuropil contamination + two-photon.

---

## Highly relevant papers

### 1. MINI2P-II: simultaneous multi-region two-photon calcium imaging in freely moving mice

Kjellevold-Skallerud E, Zong W, et al. 2026.
"Multi-region two-photon calcium imaging in freely moving animals." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.10.27.760324v2

**Findings:** Introduces MINI2P-II, a 1.1 g head-mounted two-photon miniscope with high
spatial resolution, improved imaging depth, and dual-colour capability. Paired with a
dual-scope system and an implantable periscope lens assembly, the platform enables
simultaneous 2P calcium imaging from two widely separated brain regions in freely moving
mice. The authors demonstrate simultaneous recording from retrosplenial cortex (RSC) and
posterior parietal cortex (PPC), including egocentric boundary-coding neurons in both
regions. Code and hardware designs are released for non-commercial academic use.

**Relevance to hm2p:** Direct methodological context for our own head-mounted 2P
recordings. The simultaneous RSC + PPC recording capability is directly relevant: our
current dataset is single-plane RSC only, and understanding how RSC signals relate to
concurrent PPC activity during navigation is a natural extension. The egocentric
boundary-coding result in RSC adds to the repertoire of spatial signals in this region
alongside HD tuning. The 1.1 g weight (down from earlier MINI2P generations) is relevant
for animal welfare and locomotion quality — worth noting in our methods as a limitation
of our earlier-generation system. This paper should be cited in the methods discussion of
head-mounted two-photon imaging.

---

## Moderately relevant papers

### 2. Intersectional genetics reveal CA1 border inhibitory neuron diversity and postnatal RSC connectivity

Martínez Vizcaíno N, De Wit J, et al. 2026.
"Multimodal profiling and intersectional genetics reveal CA1 border inhibitory neuron
diversity and postnatal emergence of retrosplenial connectivity." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.10.29.762784v1

**Findings:** Transcriptomic profiling and intersectional genetic tools (Cxcl14-Cre
combined with an inhibitory-neuron-specific viral enhancer) characterise a population
of GABAergic neurons at the CA1 stratum radiatum/lacunosum-moleculare border that
project long-range to RSC. These CA1 border neurons show distinct cell-surface-protein
expression profiles compared with other CA1 inhibitory neuron subtypes. Retrograde
tracing reveals that RSC connectivity from this CA1 inhibitory population emerges
postnatally (between ~P3 and P5), suggesting experience-independent developmental
specification. The authors provide genetic access tools enabling future functional
manipulation of this CA1→RSC inhibitory pathway.

**Relevance to hm2p:** CA1 sends long-range GABAergic input to RSC pyramidal neurons.
If this inhibitory input differentially targets Penk+ vs Penk⁻CamKII+ RSP cells, it
could contribute to cell-type-specific differences in HD tuning or visual cue
dependence. Methodologically, the intersectional Cre + viral enhancer strategy is
closely analogous to our own intersectional approach (Penk-Cre + Cre-ON / Cre-OFF
viruses), making this a useful reference for the methods section. The postnatal
developmental timeline is worth noting: RSC connectivity is not purely experience-
dependent, constraining how much plasticity is expected in well-trained adult mice.

---

## Tangentially relevant / methods papers

### 3. Rapid switching of in-phase and anti-phase interhemispheric rhythms in granular RSC

Authors not fully specified in search results. 2026.
"Rapid switching of in-phase and anti-phase interhemispheric rhythms within a single
theta cycle: a mechanistic explanation." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.10.23.761406v1

**Findings:** In vivo LFP recordings from bilateral granular RSC show that gamma
(30–80 Hz) oscillations are in-phase across hemispheres, while higher-frequency spline
(100–160 Hz) oscillations are anti-phase, with both coupled to distinct phases of theta
(~8 Hz). A biophysical network model driven by theta-frequency external input to
inhibitory neurons reproduces this switching. Running speed modulates the strength and
precision of anti-phase spline synchrony.

**Relevance to hm2p:** We use calcium imaging, not LFP, so we cannot directly measure
these oscillatory dynamics. However, the finding that bilateral RSC shows running-speed-
dependent interhemispheric coupling is indirectly relevant: our HD tuning analyses should
control for movement speed, and the theta-coupled organisation of RSC activity provides
mechanistic background for why HD representations might be more stable during active
locomotion. The in-phase gamma / anti-phase spline dissociation could reflect distinct
Penk+ and non-Penk circuit contributions — speculative, but worth flagging as a future
direction.

---

## Searches with no relevant results

**Penk/enkephalin + cortex:** No preprints from the past week (or any recent 2026
preprint) address Penk-expressing neurons in cortex in a spatial or HD context. Results
from this search covered enkephalin in the lateral amygdala (fear learning), ventral
pallidum/striatum (cocaine abstinence), hypothalamus (consummatory behaviour), and
enteric nervous system. This gap remains unchanged from previous scans — RSP Penk+ neuron
function is essentially uncharacterised in the literature.

**Head direction + darkness / visual landmarks / drift:** No preprints from the last
7 days. The most recent relevant work in this area remains the January 2026 paper on
active locomotion rescuing HD attractor dynamics in head-fixed mice
(10.64898/2026.01.12.698940).

**Head direction + two-photon calcium imaging:** No preprints from the last 7 days. No
new freely-moving 2P HD imaging work appeared in this window.

**Spatial navigation + maze + calcium imaging:** No preprints from the last 7 days with
direct relevance to RSP or HD. The search returned older work on mesoscale cortical
dynamics and CA1 cognitive maps.

**Neuropil contamination / correction + two-photon:** No new methods papers from the last
7 days on neuropil subtraction approaches. The fixed-coefficient (0.7×) Suite2p default
remains the dominant practice in 2026.

---

## Summary

**Papers found (last 7 days, by relevance):**

| # | Title (abbreviated) | Date | Relevance |
|---|---|---|
---|
| 1 | MINI2P-II multi-region 2P imaging | 2026-10-27 | High |
| 2 | CA1 border neurons + RSC connectivity | 2026-10-29 | Moderate |
| 3 | Bilateral RSC interhemispheric oscillations | 2026-10-30 | Tangential |

Total: 3 preprints found, all RSC-relevant.

**Persistent gap:** Penk+ neurons in RSP remain uncharacterised in any navigation or HD
context — this has been true for all 2026 scans. No other group has studied genetically
defined RSP subpopulations in a spatial navigation paradigm.

**Notable development:** MINI2P-II (paper 1) establishes simultaneous multi-region 2P
recording as a practical tool and demonstrates it in RSC+PPC simultaneously. This
positions simultaneous RSC cell-type imaging across regions as the natural next step
beyond our current single-plane approach.
