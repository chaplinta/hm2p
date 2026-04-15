# bioRxiv Scan — 15 April 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-04-15. Scan window: April 7–15, 2026 (nine days since the April 6
scan). Searches covered: retrosplenial cortex, Penk/enkephalin + cortex, head direction
+ two-photon, head direction + darkness/landmarks/drift, spatial navigation + maze
(rodents), visual processing in RSP, spatial navigation in RSP, head-mounted two-photon
microscopy, calcium imaging + maze navigation, neuropil contamination + two-photon,
spike inference + CASCADE + GCaMP, population decoding + RSC + thalamus, cell-type-
specific calcium imaging + freely moving.

Papers already listed in the 2026-04-02, 2026-04-04, 2026-04-05, or 2026-04-06 scans
are not repeated here. Note: direct access to bioRxiv pages returned 403 errors
throughout this scan; searches relied on web-indexed results, which may not fully
capture papers posted in the final 48–72 hours before the scan date.

---

## Highly relevant papers

No papers in this category were found in the April 7–15 window.

---

## Moderately relevant papers

No papers in this category were found in the April 7–15 window.

---

## Tangentially relevant / methods papers

### 1. Single-cell perturbations reveal selective modulation of causal connectivity in RSC during decision-making

Authors not fully specified in search results. 2026.
"Single-Cell Perturbations Reveal Selective Modulation of Causal Connectivity During
Decision-Making." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.04.07.716761v1

Posted: April 8, 2026.

**Findings:** Used an all-optical approach (holographic stimulation combined with
two-photon calcium imaging) to probe causal connectivity among excitatory neurons in
layer 2/3 of mouse retrosplenial cortex during different epochs of a navigation-based
decision-making task. In-task connectivity differed significantly from no-task baseline
connectivity. Crucially, these differences were selective to the cue/decision phase of
the trial and attenuated in later task stages. The authors propose that fast, behavioural-
state-dependent modulation of local connectivity is a general mechanism in cortical
circuit function.

**Relevance to hm2p:** This is an RSC paper using calcium imaging in mice during a
maze-like navigation task, placing it in our domain. The finding of task-phase-selective
connectivity modulation is tangentially relevant in two ways. First, it demonstrates
that RSC L2/3 excitatory neurons form a dynamically reconfigurable local circuit — our
Penk+ and Penk⁻CamKII+ populations both consist of L2/3 excitatory neurons, so
task-state connectivity changes could differ between them. Second, the all-optical
approach (holographic 2P stimulation + imaging) is a methodological advance for RSC
circuit studies; we cannot apply it to our current dataset but it is relevant for
future directions. The study does not address HD tuning, light/dark alternation, or
cell-type-specific coding, so it does not directly inform our analyses.

---

## Searches with no new relevant results

**Retrosplenial cortex (spatial navigation / HD):** No new preprints in the April 7–15
window beyond the causal connectivity paper above. Search results continued to return
papers from the April 2 scan (Laurent/Jacob et al., Wei et al., Margetts-Smith et al.)
and the April 6 scan (NaviGraph, CortexCAM, active locomotion HD paper).

**Penk/enkephalin + cortex:** No new preprints on Penk-expressing neurons in any
cortical region in a spatial navigation or HD context. Results returned MPOA Penk
neurons (mating behaviour, April 3, 2026 — already outside the current window),
striatal D2-MSN enkephalin (cocaine abstinence, March 2026), and dorsal raphe
enkephalin (pain/reward). No work on Penk+ neurons in RSP or any dorsal cortical area.
This is now the fourth consecutive scan with no Penk+ cortical navigation paper,
confirming that our study addresses an uncharacterised population.

**Head direction + two-photon imaging:** No new papers. The Tian et al. 2026 MEC miniature
2P paper (April 2 scan) remains the closest methodological parallel.

**Head direction + darkness / landmarks / drift:** No new experimental papers. The
active locomotion HD attractor paper (April 6 scan) remains the most recent relevant
finding on HD dynamics without visual input.

**Spatial navigation + maze (rodents, calcium imaging):** No new papers. NaviGraph
(April 6 scan) remains the closest methodological comparison.

**Head-mounted / miniature two-photon microscopy:** No new technology papers in this
window.

**Neuropil contamination + two-photon:** No new methods papers.

**Spike inference / CASCADE / GCaMP8:** No new papers in this window. The GCaMP8
spike inference paper (2025.03.03.641129v3) was updated in March 2025 and remains
the most recent relevant methods paper; its findings (CASCADE requires re-training for
GCaMP8 data) are noted for Stage 4 implementation.

**Population decoding + RSC / thalamus:** No new papers in this window.

**Cell-type-specific calcium imaging + freely moving mouse:** No new papers directly
relevant to genetically-defined RSP populations.

---

## Summary of implications for hm2p

**This was a quiet nine-day window** for our core research areas. Only one paper touching
RSC appeared (causal connectivity during decision-making), and it is tangentially relevant
at best.

**Cumulative picture after four consecutive scans (April 2–15):**

- The Penk+ RSP literature gap is confirmed across four searches: no published study has
  characterised HD properties of Penk-expressing RSP neurons. Our study remains novel.
- The active locomotion / HD attractor paper (April 6 scan) is the most actionable recent
  finding: HD attractor dynamics are locomotion-state-dependent, so our dark-epoch analyses
  must separate moving from stationary periods.
- NaviGraph (April 6 scan) provides a methodological comparison for maze-structure-mapped
  neural activity analysis.

**No changes to analysis strategy.** Previously identified action items remain:

1. Separate moving vs stationary epochs in all light/dark HD tuning comparisons (motivated
   by the April 6 active locomotion paper and the April 2 Jayakumar et al. paper).
2. Document imaging position along the RSC anterior-posterior axis (motivated by Wei et al.
   from the April 2 scan).
3. Consider NaviGraph-style maze-graph analysis for rose maze arm-specific activity.
4. Cite the updated Suite2p paper (Pachitariu & Stringer 2026) in our Stage 1 methods.

**Total new papers this scan: 1** (tangentially relevant). Cumulative across all scans:
approximately 20 papers of varying relevance.
