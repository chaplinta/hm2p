# bioRxiv Scan — 6 August 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-08-06. Searches covered: retrosplenial cortex (RSP/RSC), head
direction + two-photon imaging, head direction + darkness/landmarks/drift, spatial
navigation + maze + calcium imaging (rodents), Penk/enkephalin + cortex, visual
processing in RSP, head-mounted two-photon microscopy, neuropil contamination +
two-photon.

**Search note:** Direct access to biorxiv.org returned HTTP 403 throughout this scan.
The API at api.biorxiv.org was similarly blocked. Searches relied on web-indexed
results via the search proxy; papers posted within the last 1–2 days may not yet be
indexed. Results from the final 7-day window (2026-07-30 to 2026-08-06) are therefore
potentially incomplete. One highly relevant PNAS paper published August 4, 2026 was
identified via web search and is included below; it is flagged as a journal publication
rather than a preprint.

---

## Highly relevant papers

### 1. Anterior and posterior retrosplenial cortex employ distinct strategies for egocentric–allocentric transformation in spatial coding

Yang Y, Zhang X, Yuan X, Cai H, Cao Q, Miao C. 2026.
"Anterior and posterior retrosplenial cortex employ distinct strategies for
egocentric–allocentric transformation in spatial coding."
*PNAS* 123(31). doi:10.1073/pnas.2600565123
Published: 4 August 2026 (**journal paper, not a preprint**)

**Findings:** Two-photon calcium imaging in freely navigating mice dissected spatial
coding along the anterior–posterior RSC axis. Anterior RSC (aRSC) neurons are
predominantly tuned for egocentric boundary vectors (EBV cells): they encode the
angle and distance of environmental boundaries relative to the animal's own body axis.
Posterior RSC (pRSC) neurons exhibit enhanced allocentric boundary representation
and tighter integration with allocentric head direction signals. A specialised
aRSC→MEC projection is enriched for conjunctive cells with global, world-referenced
tuning, providing MEC with a pre-computed allocentric spatial signal.

**Relevance to hm2p:** Directly relevant and methodologically parallel — freely
navigating mice, two-photon calcium imaging, RSC spatial coding. Two implications:
(1) The A-P axis effect means the position of our imaging FOV within RSC is not just
a nuisance variable but potentially a key determinant of what coding we observe. We
should record and report the A-P coordinate of every session FOV. (2) If aRSC is
predominantly egocentric and pRSC is predominantly allocentric, and if Penk+ and
non-Penk cells differ in laminar or A-P distribution, this could explain cell-type
HD differences without invoking cell-type-intrinsic circuit differences. The finding
that pRSC HD integration is allocentric and tighter fits our prediction that
visual-landmark-dependent HD tuning (which requires allocentric coding) should be
enriched in a specific subpopulation. Worth checking whether our virus expression
distributions differ along A-P.

---

### 2. Retrosplenial PV and SST interneurons shape egocentric spatial precision and stability

Authors not recovered from search results. 2026.
"Retrosplenial PV and SST interneurons shape egocentric spatial precision and
stability." *bioRxiv*.
https://www.biorxiv.org/content/10.64898/2026.05.10.724096v1
Posted: 10–11 May 2026

**Findings:** PV interneurons in RSC are strongly modulated by self-motion and
exhibit bearing-aligned synchrony that precedes SST activation; PV silencing degraded
egocentric coding precision. SST interneurons show weak self-motion modulation but
robust boundary-anchored activity with globally coherent dynamics; SST silencing
disrupted global population organisation without affecting initial coding precision.
The two interneuron types support dissociable components: PV gates movement-linked
precision, SST maintains long-term representational stability.

**Relevance to hm2p:** This paper characterises a functional dissociation within
RSC inhibitory circuits that is orthogonal to our excitatory (Penk+ vs. non-Penk)
cell-type distinction. Penk is an opioid peptide expressed in a subset of
excitatory neurons; it is not a marker for PV or SST interneurons. However, if PV
interneurons drive egocentric coding precision in excitatory output, then both Penk+
and non-Penk excitatory populations receive this PV-mediated sharpening. Any HD
tuning difference between the two excitatory populations would therefore need to be
explained downstream of shared PV inhibition. This constrains our interpretation:
excitatory cell-type differences in HD tuning likely arise from differences in
excitatory inputs (visual cortex, anterior thalamus, subiculum) or intrinsic
biophysical properties, not from differential interneuron targeting. Also useful for
methods: the paper likely reports egocentric coding metrics (angular precision, drift
rate, stability) with and without optogenetic perturbation — these will be useful
benchmarks for our own HD tuning quality metrics.

---

## Moderately relevant papers

### 3. Retrosplenial cortical reorganization during late adolescence introduces instability of contextual memory circuits

Full authors not recovered. 2026.
"Retrosplenial cortical reorganization during late adolescence introduces instability
of contextual memory circuits."
*PLOS Biology*. doi:10.1371/journal.pbio.3003908
Published: 17 July 2026 (**journal paper, not a preprint**)

**Findings:** Parvalbumin interneuron density and perineuronal net (PNN) expression
in mouse RSC peak in early adolescence then substantially decline in late
adolescence, causing a period of instability in contextual memory circuits. Memories
acquired during early adolescence become transiently inaccessible. PNN stabilisation
pharmacologically rescued memory retrieval during this window; normal PNN build-up in
adulthood restores expression spontaneously.

**Relevance to hm2p:** Our animals are all adults, so the adolescent instability
period itself is not directly applicable. However, the paper reinforces that PV
interneurons and their PNN environment are key regulators of RSC circuit stability —
relevant background for interpreting the PV/SST interneuron paper above. If adult
RSC PV function is mature and stable, the assumption that both Penk+ and non-Penk
excitatory neurons receive equivalent PV-mediated inhibitory input is more defensible.

---

## Tangentially relevant / methods papers

### 4. Large-scale volumetric two-photon calcium imaging (methods paper)

Authors not recovered from search results. 2026.
"Large-scale volumetric two-photon calcium imaging [...]"
*bioRxiv*.
https://www.biorxiv.org/content/10.64898/2026.05.28.728595v1
Posted: 28–29 May 2026

**Findings:** Describes a large-scale 2P microscope system with reduced neuropil
contamination using soma-targeted GCaMP expression. Notes that cytosolically
expressed calcium indicators with uncorrected neuropil contamination can mask
single-cell tuning heterogeneity, and demonstrates that soma-targeted expression
reveals cleaner single-cell directional tuning.

**Relevance to hm2p:** We use standard GCaMP with a fixed-coefficient neuropil
subtraction (Suite2p default, coefficient 0.7; FISSA as optional alternative). The
finding that neuropil contamination masks heterogeneity is directly relevant: if our
fixed-coefficient subtraction is insufficient, we may underestimate cell-to-cell
HD tuning variability within each population. This strengthens the case for using
FISSA in Stage 4 rather than the fixed-coefficient approach — FISSA performs spatial
ICA to estimate per-cell neuropil profiles rather than applying a uniform scaling
factor. Worth checking our neuropil subtraction residuals explicitly.

---

## Searches with no new relevant results

**Penk/enkephalin + cortex + navigation (last 7 days):** All results were
striatal/brainstem — dopamine system, cocaine abstinence, pain modulation. No new
preprints on Penk-expressing cortical neurons in a spatial or HD context. The gap
identified in the April scan remains open.

**Head direction + darkness / visual cue removal (last 7 days):** No new
biorxiv-indexed preprints from the past week on this specific topic. The closest
existing papers (Tian et al. 2026 MEC light deprivation; Jayakumar et al. 2025
path integration recalibration) remain the primary references.

**Head-mounted two-photon microscopy, freely moving (last 7 days):** No new
hardware preprints found within the past week. The 2025 wave of systems
(M-MINI2P, FHIRM-TPM 3.0, miniBB2p) from the April scan covers the current
state of the art.

**Neuropil contamination / correction (last 7 days):** No methodologically
focused new preprints found this week.

---

## Total: 4 papers (2 preprints, 2 journal publications)

**Highly relevant:** 2 (Yang et al. PNAS 2026; PV/SST RSC interneurons bioRxiv)
**Moderately relevant:** 1 (RSC adolescence PLOS Biology)
**Methods/tangential:** 1 (volumetric 2P soma-targeting bioRxiv)

**Notable trends:**

The Yang et al. PNAS paper is the most important finding this week. The A-P
gradient in RSC spatial coding (egocentric anterior, allocentric posterior) has now
been directly demonstrated with two-photon calcium imaging in freely navigating mice
— the same modality and preparation as hm2p. This is a major result that will need
to be addressed in our manuscript: either by reporting our imaging positions (if they
are consistent A-P) or by acknowledging the gradient as a potential confound.

The PV/SST interneuron paper from May provides important circuit context: the
inhibitory architecture of RSC segregates movement-linked precision (PV) from
long-term stability (SST). This is a useful framework for interpreting any temporal
dynamics we observe in HD tuning across the light/dark transition.

Penk+ RSP neurons remain uncharacterised in the literature. No new papers this week.
