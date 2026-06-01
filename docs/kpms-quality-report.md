# keypoint-MoSeq (kpms) Output Quality Report

**Date:** 2026-06-01
**Run date:** 2026-06-01 01:39-04:43 UTC (~3 hours)
**EC2 instance:** c5.4xlarge (CPU-only, JAX on CPU)
**Software:** keypoint-moseq 0.6.8, jax-moseq 0.3.3, JAX 0.6.2

---

## Check 1: DLC Model Provenance — PASS

**Question:** Did kpms consume pose data from the current champion DLC model?

**Result:** Yes. All 26 DLC .h5 files downloaded by `run_kpms.py` contain
`HrnetW32_hm2p-retrainMar20shuffle1_snapshot_best-100` in their filenames, matching
the champion ID `dlc-20260529-hrnetw32-snap100` declared in
`s3://hm2p-derivatives/dlc-champion.json`. The log confirms all 26 sessions were
downloaded and processed with zero failures.

---

## Check 2: Syllable File Structure — PASS (with minor gap)

**npz file contents:**

| Key | dtype | Shape | Notes |
|-----|-------|-------|-------|
| `syllable_id` | int16 | (N_frames,) | One value per video frame |

- All values are non-negative integers in range [0, 99].
- 100% of frames assigned to a syllable (no -1 / unassigned values).
- Frame counts match expected session lengths at 30 fps.

**Minor gap:** `syllable_prob` (posterior probabilities per syllable per frame) is
**not saved**. The `run_kpms.py` script checks for `syllable_probability` in the
kpms results dict and would save it if present, but kpms `extract_results` apparently
did not return it. This means we have hard assignments only -- no uncertainty
estimates. This is acceptable for initial syllable-conditioned analyses but limits
our ability to assess model confidence or use soft assignments.

---

## Check 3: Syllable Distribution — FAIL (extreme non-uniformity)

This is the most serious quality concern.

**Per-session syllable counts (26 sessions):**

| Statistic | Value |
|-----------|-------|
| Global unique syllables | 100 |
| Per-session: min / max | 39 / 92 |
| Per-session: median | 56 |

**Distribution analysis (5 representative sessions):**

| Session | Unique syls | Top-2 coverage | Entropy ratio | Median bout (ms) |
|---------|------------|----------------|---------------|------------------|
| 1117646_135202 | 90 | 67.1% | 0.45 | 133 |
| 1114356_110937 | 80 | 38.1% | 0.61 | 133 |
| 1115816_143639 | 84 | 73.9% | 0.39 | 267 |
| 1116663_150157 | 39 | 88.3% | 0.34 | 733 |
| 1118320_143112 | 40 | 81.4% | 0.34 | 800 |

**Global syllable frequency (pooled across 5 sessions, 332,545 frames):**

| Syllable | Frames | Percentage |
|----------|--------|------------|
| #52 | 121,016 | 36.4% |
| #86 | 113,380 | 34.1% |
| #95 | 26,830 | 8.1% |
| All others (94 syllables) | 71,319 | 21.4% |

**Assessment:** Two syllables (#52 and #86) account for ~70% of all time. The
distribution is extreme: top-5 syllables cover 82-95% of frames in most sessions.
Many syllables appear only 1-3 times across an entire 30-minute session. This is
far more concentrated than the heavy-tailed but relatively uniform distributions
reported in Weinreb et al. 2024, where 20-40 syllables typically account for ~80%
of time.

**Likely cause:** kappa = 10^6, which is 100x higher than the kpms default of 10^4.
Higher kappa enforces stronger "stickiness" (the HMM penalty for state transitions),
producing fewer, longer syllable bouts. The two dominant syllables likely represent
generic "stationary" and "slow locomotion" states that the model collapses into when
transitions are penalized this heavily.

---

## Check 4: Bout Duration — CONDITIONAL PASS

**Pooled statistics (5 sessions, 12,536 bouts):**

| Percentile | Frames | Duration (ms) |
|------------|--------|---------------|
| P5 | 1 | 33 |
| P10 | 1 | 33 |
| P25 | 2 | 67 |
| P50 | 6 | 200 |
| P75 | 21 | 700 |
| P90 | 57 | 1,900 |
| P95 | 103 | 3,433 |
| P99 | 330 | 11,010 |

**Single-frame bouts:** 14.9% of all bouts last only 1 frame (33 ms). These are
likely noise or brief transitions rather than meaningful behaviours. In sessions
with high syllable counts, single-frame bouts reach 17-21%.

**Long bouts:** Sessions with low syllable counts (1116663, 1118320) have median
bout durations of 700-800 ms with P95 reaching 7-11 seconds. Some bouts extend to
many tens of seconds, which represent extended periods in the dominant syllable
states.

**Assessment:** The median bout duration (200 ms pooled; 133-800 ms per session) is
within the plausible range for mouse behaviour (Weinreb et al. report ~300-1000 ms
typical). However, the bimodal distribution -- many single-frame bouts AND many
very long bouts -- is concerning. Healthy kpms output should show a narrower,
approximately log-normal distribution of bout durations.

---

## Check 5: Cross-Session Consistency — PASS

Since kpms fits a single AR-HMM model across all sessions simultaneously (26 sessions
pooled during fitting), syllable IDs have the same meaning across sessions by
construction.

**Verification:** The same syllable IDs dominate across all 5 inspected sessions:

- Syllable #52 and #86 are top-2 in all 5 sessions (just swapping rank order).
- Syllable #95 is consistently 3rd.
- Top-10 pairwise overlap: 6-8 out of 10 syllables shared between any session pair.

This confirms the joint fitting worked correctly and syllable semantics are
globally consistent.

---

## Check 6: Parameter Sanity — FAIL (kappa too high)

| Parameter | Our value | kpms default | Paper recommendation | Verdict |
|-----------|-----------|-------------|---------------------|---------|
| kappa | 10^6 | 10^4 | Sweep [10^2 - 10^6], choose via median duration | **100x above default** |
| num_pcs | 10 | 10 | Auto-select via scree plot | OK |
| num_iters | 50 | 50 | Default | OK (but see convergence) |
| bodyparts | 8 | varies | 5-10 recommended, exclude tail | OK |
| noise_calibration | skipped | required | "Essential" per Weinreb et al. | **Skipped** |

**kappa = 10^6:** This is the maximum value in the recommended sweep range from
Weinreb et al. 2024. The default is 10^4. Using 10^6 without first running a kappa
sweep and evaluating median syllable duration is methodologically incorrect. The
extreme syllable distribution (Check 3) is the direct consequence.

**Noise calibration skipped:** Weinreb et al. describe noise calibration as essential
for reliable syllable discovery. The `run_kpms.py` script explicitly skips it because
it requires an interactive Jupyter widget to display video frames, which is not
available on a headless EC2 instance. Without noise calibration, the observation noise
prior is set to a generic default that may not match the actual DLC tracking noise in
our data. Combined with high kappa, this likely worsened under-segmentation.

---

## Check 7: Bodypart Coverage — PASS

**8 bodyparts used:**

| Bodypart | Source | Notes |
|----------|--------|-------|
| nose | SuperAnimal | Maps to nose_tip in our convention |
| left_ear | SuperAnimal + custom | |
| right_ear | SuperAnimal + custom | |
| neck | SuperAnimal + custom | |
| mid_back | SuperAnimal + custom | |
| mouse_center | SuperAnimal + custom | |
| mid_backend | SuperAnimal | Posterior back |
| mid_backend2 | SuperAnimal | Further posterior |

**DLC output bodypart names:** The DLC .h5 files use SuperAnimal TopViewMouse
keypoint names (27 bodyparts total), not the 8 custom names listed in CLAUDE.md.
This is because the fine-tuned model outputs all SuperAnimal keypoints plus the
custom `head_midpoint`. The kpms script correctly selected the 8 requested bodyparts
from the full 27.

**Assessment:** 8 bodyparts covering head, neck, and trunk is appropriate for
top-view mouse pose. The Weinreb et al. recommendations are 5-10 keypoints excluding
tail (which adds noise). `head_midpoint` and `tail_base` were excluded from the
kpms bodypart list, which is reasonable -- `head_midpoint` is a headstage feature,
not a natural body landmark, and `tail_base` is often noisy.

---

## Check 8: Setup Log — PASS (no errors, incomplete logging)

**No errors or warnings** in the kpms fitting process. The only warning in the
entire log is a benign pip warning about running as root.

**Missing diagnostic logging:** The Python-level INFO messages from `run_kpms.py`
(bodypart validation, PCA details, config verification, per-session results) are
**not present** in the log. The log captures Docker container stdout but apparently
drops the Python logging output between "Starting keypoint-MoSeq fitting" and the
kpms progress bar. This is a logging configuration issue in the Docker setup
(Python logging may be going to stderr while only stdout is captured, or buffering
is preventing output).

**Convergence:** No ELBO, log-likelihood, or convergence diagnostic is available.
kpms uses Gibbs sampling and does not print convergence metrics by default. The
iteration time was stable (~208-209 s/iter), suggesting the sampler ran without
computational issues. However, without convergence diagnostics, we cannot confirm
the model converged in 50 iterations.

---

## Overall Verdict: NOT USABLE as-is — needs re-fitting

### Passes (5/8)

1. **DLC provenance** — correct champion model used
2. **File structure** — clean npz files, correct format
3. **Cross-session consistency** — syllable IDs are globally meaningful
4. **Bodypart coverage** — appropriate selection for top-view mouse
5. **Setup log** — no errors during fitting

### Fails (2/8)

6. **Syllable distribution** — extreme non-uniformity; 2 syllables cover 70% of time
7. **Parameter sanity** — kappa 100x above default; noise calibration skipped

### Conditional pass (1/8)

8. **Bout duration** — median is plausible but bimodal distribution is unhealthy

### Recommendations for re-run

1. **Run a kappa sweep.** Fit with kappa in [10^2, 10^3, 10^4, 10^5, 10^6] and
   select the value that gives median syllable durations of ~300-500 ms (per
   Weinreb et al. 2024 recommendation). The current data suggest kappa ~10^4
   (the default) would be more appropriate.

2. **Implement noise calibration.** Either:
   - Run an interactive notebook locally with a few representative session videos
     to calibrate noise parameters, then hard-code them in the headless run script.
   - Use the kpms `noise_calibration` function programmatically by extracting
     frames from video files on EC2 and computing the calibration non-interactively.

3. **Save syllable_prob.** Modify `run_kpms.py` to ensure posterior probabilities
   are extracted and saved. This may require calling `kpms.extract_results` with
   specific parameters to get the syllable probability matrix.

4. **Fix Docker logging.** Ensure Python logging from `run_kpms.py` is captured
   in the setup log (e.g., redirect stderr to stdout in the Docker entrypoint, or
   configure Python logging to use stdout).

5. **Increase iterations.** Consider 100 iterations for 1.5M frames across 26
   sessions. Monitor convergence by examining the `results.h5` checkpoint on S3
   to see if syllable assignments stabilize.

6. **Add convergence monitoring.** After fitting, load the results.h5 and check
   that the number of occupied syllable states stabilized over the last 10
   iterations of the Gibbs sampler.
