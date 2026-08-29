# bioRxiv Scan — 29 August 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-08-29. Searches covered: retrosplenial cortex (RSC/RSP), Penk/enkephalin
+ cortex, head direction cells + two-photon imaging, head direction + darkness/landmarks/drift,
spatial navigation + maze (rodents), visual processing in RSP, spatial navigation in RSP,
head-mounted two-photon microscopy, calcium imaging + maze navigation, neuropil contamination
+ two-photon.

**Note on coverage:** Web search indexing of bioRxiv continues to lag 3–7 days behind
submission. No preprints posted August 22–29 appeared in any search on these topics,
consistent with the pattern documented in all scans from 2026-08-21 onward. The one
exception is a calcium-indicator methods paper (Wang et al., ~August 25, below), which
surfaced via a news article cross-reference rather than direct topic search.

---

## Highly relevant papers

No new bioRxiv preprints from August 22–29, 2026 were found across any of the searched
topics.

---

## Moderately relevant papers

No new bioRxiv preprints from August 22–29, 2026 were found.

---

## Tangentially relevant / methods papers

### 1. Subthreshold membrane depolarization engages GCaMP in the awake brain

Wang Y, Tseng H-a, Xiao S, Bortz E, Zhou Y, Martin A, Man H, Mertz J, Han X. 2026.
"Subthreshold membrane depolarization powerfully engages intracellular calcium dynamics
in the awake mammalian brain." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.08.24.746661

**Findings:** In awake mice, subthreshold membrane depolarisation (EPSPs below spike
threshold) generates substantial GCaMP transients in dendrites and, to a smaller degree,
in somas. Using two-photon imaging combined with whole-cell patch clamp, the authors
demonstrate that GCaMP6f soma fluorescence changes of 1–5% ΔF/F can arise purely from
subthreshold input without any action potentials. Transients were substantially larger in
proximal dendrites. The contribution scaled with depolarisation magnitude and input rate.

**Relevance to hm2p:** Our Stage 4 pipeline applies CASCADE to soma dF/F to infer spikes.
If GCaMP soma signals in RSP are contaminated by subthreshold synaptic drive — as this
paper shows they can be — then low-amplitude dF/F events below our spike-detection
threshold may still carry HD-correlated information. This is especially pertinent for
dendritic ROIs (which coexist in our single plane alongside soma ROIs): dendritic
transients may reflect subthreshold HD-correlated input from ATN even in cells that do
not spike. We should check whether the ROI classifier separates soma and dendrite ROIs
cleanly; if not, dendrite contamination may inflate apparent HD tuning for low-firing
cells. Cite in Methods when discussing the limits of spike inference.

---

## Searches with no relevant results

All ten search topics returned no confirmed preprints from the August 22–29 window:

- **Retrosplenial cortex (RSC/RSP):** No new preprints. Searches continue to surface
  Yang et al. PNAS 2026 (published August 4, A-P egocentric/allocentric gradient) and
  the July 27 Giocomo-lab structured-navigation platform preprint.

- **Penk/enkephalin + cortex:** No new preprints. The full monitoring period (April 2 –
  August 29, 149 days, 94 scans) has produced zero preprints characterising Penk+ neurons
  in cortex in a spatial or HD context. All Penk/enkephalin hits remain subcortical
  (striatum, dorsal raphe, MPOA, enteric neurons). The gap is fully open.

- **Head direction cells + two-photon imaging:** No new preprints.

- **Head direction + darkness / landmarks / drift:** No new experimental preprints.
  Results stable since the parallax error paper (v4, March 2026) and Jayakumar et al.
  (Curr Biol, 2026).

- **Spatial navigation + maze (rodents):** No new preprints from this window.

- **Visual processing in RSP:** No new preprints. Yang et al. PNAS 2026 and Wei et al.
  Nature Communications 2026 remain the most recent publications.

- **Spatial navigation in RSP:** No new preprints.

- **Head-mounted / miniature two-photon microscopy:** No new methods preprints. An
  August 2026 news release (EurekAlert, medicalxpress) covers the Opto2P-FCM (CU Anschutz
  / CU Boulder), a 5-g head-mounted device combining high-resolution two-photon imaging
  with single-neuron optogenetic control in freely moving mice, published in Optica. No
  corresponding bioRxiv preprint identified.

- **Calcium imaging + maze navigation:** No new preprints.

- **Neuropil contamination + two-photon:** No new correction or benchmarking papers.

---

## Summary

**New directly relevant preprints this week:** 0  
**New tangentially relevant / methods preprints this week:** 1 (Wang et al. 2026 — subthreshold GCaMP)

**Reason for absent coverage:** Persistent web search indexing lag (3–7 days behind
bioRxiv submission). The August 22–29 window cannot be confirmed as fully empty; this
reflects best available coverage from current search indices.

**Cumulative picture (April 2 – August 29, 149 days, 94 scans):**

No preprint has characterised HD tuning of genetically-defined RSP excitatory neuron
populations (Penk+ or Penk⁻CamKII+). The gap is fully open.

Most substantive papers within the monitoring window, ranked by recency:

- **Yang et al. PNAS 2026** (published August 4): Anterior RSC = egocentric coding;
  posterior RSC = allocentric coding; aRSC→MEC projection pre-computes allocentric signal.
  Two-photon calcium imaging, freely navigating mice. doi:10.1073/pnas.2600565123

- **Feld, Spiers et al. 2026** (July 24): RSC encodes global graph topology of relational
  knowledge; HPC encodes local node-distance. Human fMRI.
  https://www.biorxiv.org/content/10.64898/2026.07.24.740522v1

- **Goodwin, Dombeck et al. 2026** (July 3): ACh in dRSC coordinates representational
  adaptation during contextual uncertainty. Relevant for interpreting tuning dynamics at
  light-off transitions.
  https://www.biorxiv.org/content/10.64898/2026.07.03.736438

- **Oh et al. 2026** (May 10): RSC PV/SST interneurons dissociate egocentric coding
  precision (PV, movement-linked) from long-term population stability (SST,
  boundary-anchored).
  https://www.biorxiv.org/content/10.64898/2026.05.10.724096v1

- **Peters, Redish, Kodandaramaiah et al. 2026** (May 5): Coordinated representational
  drift across mouse cortex. RSC highest-density spatially tuned region; drift geometrically
  coordinated across dorsal cortex.
  https://www.biorxiv.org/content/10.64898/2026.05.05.723038v1

---

## Persistent action items (carried from August 24 scan)

1. Document the RSC anterior–posterior coordinate for every hm2p session FOV. Check
   whether Penk+ and non-Penk virus expression distributions differ along the A–P axis.
   Cite Yang et al. PNAS 2026 and Wei et al. Nature Communications 2026 in manuscript
   Methods. **Priority: high — analysis prerequisite.**

2. Separate moving vs. stationary epochs in all light/dark HD tuning comparisons.
   Cite Jayakumar et al. 2026 (Curr Biol doi:S0960-9822(26)00222-8).
   **Priority: high.**

3. Bin HD tuning quality in 10–15 s windows within the first 30 s of each dark epoch to
   test for a transient tuning-degradation peak at light-off onset. Theoretical basis:
   ACh-mediated representational adaptation (Goodwin/Dombeck et al. 2026).
   **Priority: medium.**

4. CEBRA orthogonal subspace test: does light/dark context separate from the HD coding
   subspace? Does the RSP population manifold mirror the rose maze graph topology?
   Cite Feld/Spiers et al. 2026 and Peters et al. 2026.
   **Priority: medium.**

5. Frame excitatory cell-type HD differences in the Discussion using the PV-precision /
   SST-stability inhibitory dissociation (Oh et al. 2026): differences in Penk+ vs.
   non-Penk HD tuning downstream of shared PV sharpening imply excitatory input differences
   or intrinsic biophysical properties, not differential inhibition.
   **Priority: medium.**

6. Check ROI classifier performance separating soma vs. dendrite ROIs. If separation is
   imperfect, low-amplitude dF/F events in apparent soma ROIs may reflect subthreshold
   dendritic HD input rather than somatic spiking (Wang et al. 2026). This matters most
   for low-firing non-Penk cells where the HD tuning is weaker.
   **Priority: medium — new item from this scan.**

7. Note coordinated representational drift result (Peters et al. 2026) when reporting
   session-wise HD tuning stability — within-session estimates are more reliable than
   cross-session comparisons.
   **Priority: low.**

8. Watch for RSP follow-up from the Giocomo lab structured navigation platform
   (July 2026). Neuropixels RSC recordings under this paradigm would be a direct
   competitor study.
   **Priority: watch.**
