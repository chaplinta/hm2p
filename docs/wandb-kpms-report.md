# W&B Integration with keypoint-MoSeq: Technical Feasibility Report

**Date:** 2026-06-02
**Author:** Agent (Claude)

---

## Executive Summary

keypoint-MoSeq (kpms) has **no built-in W&B, TensorBoard, or any experiment tracking
integration**. Neither kpms nor its backend library jax-moseq contains any callback,
hook, or logger mechanism. No one in the community appears to have done this --
there are zero GitHub issues, discussions, or public examples of W&B + kpms.

However, integration is **straightforward to build** because:

1. The fitting loop in `fit_model()` is simple and modifiable (a plain `for` loop
   over Gibbs sampling iterations with tqdm).
2. `model_likelihood()` in jax-moseq computes the full log-joint probability
   decomposed by component -- this is the key diagnostic metric, and it is
   already implemented but simply never called during fitting.
3. Syllable statistics (count, durations, frequencies) are trivially computable
   from `model["states"]["z"]` at each iteration.

The recommended approach is a thin wrapper around `fit_model` that calls
`model_likelihood()` and logs metrics to W&B at each iteration.

---

## 1. Does kpms have built-in W&B support?

**No.** Exhaustive search of the installed source code (v0.6.8 / jax-moseq v0.3.3):

```
grep -rn "wandb\|tensorboard\|callback\|hook\|logger" keypoint_moseq/ → (no results)
grep -rn "wandb\|tensorboard\|callback\|hook\|logger" jax_moseq/     → (no results)
```

There are zero mentions of any experiment tracking library, callback mechanism, or
logging infrastructure in either package. The only output mechanism is:
- `tqdm` progress bar (iteration count only, no metrics)
- `print()` statements (file paths, warnings)
- HDF5 checkpoint saves (at intervals set by `save_every_n_iters`)
- `plot_progress()` generates a static PDF with 4 panels

## 2. Has anyone else done this?

**No evidence found.** Searched:
- GitHub Issues (dattalab/keypoint-moseq): 0 mentions of wandb, W&B, experiment
  tracking, TensorBoard, or monitoring across all 262+ issues.
- GitHub Discussions: The repo does not have Discussions enabled (404).
- Web search for "keypoint-moseq wandb", "kpms wandb", "MoSeq wandb": zero results.
- The kpms community uses a Slack channel (not publicly searchable), so there may
  be informal discussion there, but nothing has surfaced publicly.
- No papers, blog posts, or tutorials mention W&B + kpms.

For comparison, **DeepLabCut 3.x has native W&B support** (`pip install deeplabcut[wandb]`)
via a WandbLogger integration. VAME has no W&B integration either.

## 3. What does the fitting loop expose?

### 3.1 `fit_model()` internals

The core loop (from `keypoint_moseq/fitting.py`, lines 250-286) is:

```python
with tqdm.trange(start_iter, num_iters + 1, ncols=72) as pbar:
    for iteration in pbar:
        model = _wrapped_resample(resample_func, data, model, ...)
        # Checkpoint save + plot_progress at intervals
```

**What is available per iteration:**
- `model` dict containing `states`, `params`, `hypparams`, `noise_prior`, `seed`
- `model["states"]["z"]` — discrete syllable assignments (shape: `(N_batch, T)`)
- `model["states"]["x"]` — continuous latent trajectories
- `model["states"]["s"]` — noise scales
- `model["states"]["h"]` — heading angles
- `model["states"]["v"]` — centroid positions
- `model["params"]` — AR parameters (Ab, Q), transition matrix (pi), observation (Cd, sigmasq)
- `data` dict with observations Y and mask

**What is NOT computed per iteration:**
- No ELBO (this is Gibbs sampling, not variational inference -- there is no ELBO)
- No log-joint probability
- No convergence diagnostic
- No syllable statistics

### 3.2 `model_likelihood()` — the key diagnostic function

**This exists but is never called during fitting.** Located in
`jax_moseq/models/keypoint_slds/log_prob.py`:

```python
def model_likelihood(data, states, params, hypparams, noise_prior, **kwargs):
    """Convenience class that invokes `log_joint_likelihood`."""
    return log_joint_likelihood(
        **data, **states, **params,
        **hypparams["obs_hypparams"],
        **hypparams["cen_hypparams"],
        s_0=noise_prior
    )
```

Returns a dict mapping component names to total log probabilities:
- `"x"` — AR latent trajectory log-prob
- `"z"` — discrete state log-prob
- `"Y"` — observation (keypoint) log-prob
- `"s"` — noise scale log-prob
- `"v"` — centroid location log-prob

**Sum of all values = log-joint probability of the model.** This is the primary
convergence diagnostic for Gibbs sampling (not ELBO, since this is not variational
inference).

### 3.3 `plot_progress()` — what kpms currently tracks

The built-in `plot_progress()` function (called at checkpoint intervals) generates
4 panels:

1. **Frequency distribution** — syllable rank vs probability (current iteration)
2. **Duration distribution** — histogram of syllable durations (current iteration)
3. **Median duration** — median syllable duration across saved iterations
4. **State sequence history** — syllable assignments in a random time window across
   iterations

These are computed from the HDF5 checkpoint file (reads `z` from each saved snapshot)
and rendered as a static PDF. There is no streaming, no interactive dashboard, and
no numerical output -- only the saved figure.

### 3.4 `expected_marginal_likelihoods()` — post-hoc model comparison

This function (lines 615-678 of fitting.py) computes cross-validated marginal
log-likelihoods across multiple independent model runs. It is designed for selecting
the best run out of N parallel fits. It is a post-hoc tool, not a per-iteration
diagnostic.

## 4. Clarification: ELBO vs log-joint

keypoint-MoSeq uses **Gibbs sampling** (MCMC), not variational inference. Therefore:

- There is **no ELBO** (Evidence Lower BOund) to track. ELBO is a variational
  inference concept.
- The appropriate convergence diagnostic is the **log-joint probability**
  (`log p(data, latent | params)`) evaluated at each iteration, which should
  stabilize as the chain mixes.
- The Weinreb et al. 2024 paper reports that "500 fitting iterations (~30 min on
  GPU for ~5h of data) are sufficient" and that "the log joint probability appeared
  to have stabilized" -- but this computation is not exposed in the public API
  during fitting.

## 5. What metrics would be most useful to log?

### Per-iteration metrics (computed from model state):

| Metric | Source | Cost | Value |
|--------|--------|------|-------|
| Log-joint (total) | `sum(model_likelihood(...).values())` | ~1 JAX JIT call | Primary convergence diagnostic |
| Log-joint (by component) | `model_likelihood(...)` | Same call | Identifies which component is still moving |
| Num active syllables | `len(np.unique(z[mask > 0]))` | Trivial | Key hyperparameter diagnostic |
| Median syllable duration | `np.median(get_durations(z, mask))` | Trivial | kappa sensitivity |
| Mean syllable duration | `np.mean(get_durations(z, mask))` | Trivial | Complements median |
| Syllable frequency entropy | `scipy.stats.entropy(frequencies)` | Trivial | Distribution uniformity |
| Top-5 syllable coverage | `sum(sorted(freq)[-5:])` | Trivial | Dominance of a few syllables |
| NaN count | Already in `check_for_nans()` | Free | Numerical stability |

### Per-checkpoint metrics (more expensive, logged less frequently):

| Metric | Source | Cost | Value |
|--------|--------|------|-------|
| Syllable frequency distribution | `get_frequencies(z, mask)` | Low | Full distribution snapshot |
| Duration distribution | `get_durations(z, mask)` | Low | `wandb.Histogram` |
| State sequence image | Rasterplot of z | Moderate | `wandb.Image` |
| Transition matrix | `model["params"]["pi"]` | Low | `wandb.Image` of heatmap |

### Run-level config (logged once at init):

| Parameter | Where |
|-----------|-------|
| kappa | `hypparams["trans_hypparams"]["kappa"]` |
| num_states | `hypparams["trans_hypparams"]["num_states"]` |
| latent_dim | inferred from Ab shape |
| num_iters | fit_model arg |
| ar_only | fit_model arg |
| jitter | fit_model arg |
| num_keypoints | data shape |
| num_sessions | metadata |
| total_frames | mask.sum() |

## 6. Implementation approach

### Option A: Wrapper function (non-invasive, recommended)

Write a `fit_model_with_wandb()` that reimplements the iteration loop from
`fit_model()` while adding W&B logging. This avoids monkey-patching or modifying
the installed package.

```python
import wandb
from jax_moseq.models.keypoint_slds.log_prob import model_likelihood
from keypoint_moseq.viz import get_durations, get_frequencies

def fit_model_with_wandb(model, data, metadata, wandb_config=None, **kwargs):
    """Wrapper around kpms fit_model with W&B logging."""
    wandb.init(project="hm2p-kpms", config=wandb_config or {})

    # Log hyperparameters
    wandb.config.update({
        "kappa": model["hypparams"]["trans_hypparams"]["kappa"],
        "num_states": model["hypparams"]["trans_hypparams"]["num_states"],
        "num_iters": kwargs.get("num_iters", 50),
        ...
    })

    # Reimplement the iteration loop with logging
    for iteration in tqdm.trange(num_iters):
        model = _wrapped_resample(resample_func, data, model, ...)

        # Compute log-joint (the key metric)
        ll = model_likelihood(data, model["states"], model["params"],
                              model["hypparams"], model["noise_prior"])
        log_joint = sum(v.item() for v in ll.values())

        # Compute syllable stats
        z = np.array(model["states"]["z"])
        mask = np.array(data["mask"])
        durations = get_durations(z, mask)
        n_active = len(np.unique(z[mask > 0]))

        wandb.log({
            "log_joint": log_joint,
            "log_joint/Y": ll["Y"].item(),
            "log_joint/x": ll["x"].item(),
            "log_joint/z": ll["z"].item(),
            "syllables/n_active": n_active,
            "syllables/median_duration": np.median(durations),
            "syllables/mean_duration": np.mean(durations),
            "iteration": iteration,
        })

    wandb.finish()
    return model, model_name
```

### Option B: Monkey-patch `_wrapped_resample` (fragile, not recommended)

Could intercept the resample call to inject logging, but this is brittle across
kpms version updates.

### Option C: Post-hoc from checkpoints (limited)

Parse the HDF5 checkpoint after fitting is complete, reconstruct per-iteration
metrics from saved snapshots, and log them to W&B retroactively. Limited to
snapshot intervals (default every 25 iterations) and cannot compute log-joint
without JAX.

## 7. Computational cost of per-iteration logging

The `model_likelihood()` function is JIT-compiled (`@jax.jit` on `log_joint_likelihood`).
After the first call, it should add negligible overhead (~10-100ms per iteration
depending on data size, compared to ~3-10s for the Gibbs resample step itself).

Syllable statistics (`get_durations`, `get_frequencies`, `np.unique`) operate on
numpy arrays and are also fast (<10ms).

**Estimated overhead: <5% of total fitting time** for per-iteration logging.

## 8. Comparison with other behavioural analysis tools

| Tool | Built-in W&B | Built-in TensorBoard | Any experiment tracking |
|------|-------------|---------------------|----------------------|
| keypoint-MoSeq | No | No | No (only HDF5 checkpoints + PDF plots) |
| VAME | No | No | No |
| DLC2Action | No | No | No |
| DeepLabCut 3.x | **Yes** (`pip install deeplabcut[wandb]`) | **Yes** | Yes |
| SLEAP | No | **Yes** (via TF) | Via TensorFlow |
| LightningPose | No | **Yes** (via PL) | Via PyTorch Lightning |

DLC is the only behavioural neuroscience tool with native W&B support. Pose
estimation tools built on PyTorch Lightning or TensorFlow inherit their logging
ecosystems. Unsupervised behavioural segmentation tools (kpms, VAME, DLC2Action)
universally lack experiment tracking.

## 9. Recommendations

1. **Build a wrapper function** (Option A above) rather than modifying kpms source.
   This is robust to kpms version updates and requires ~50 lines of code.

2. **Log the log-joint probability** as the primary convergence metric. This is
   what the Weinreb et al. 2024 paper uses internally but does not expose.

3. **Log syllable count, median duration, and frequency entropy** as secondary
   diagnostics -- these are what `plot_progress()` already visualizes, just in
   a non-interactive format.

4. **Log the kappa sweep** as a W&B sweep if running multiple kappa values. This
   would replace the current ad-hoc approach of manually comparing
   `plot_progress()` PDFs across kappa values.

5. **Consider whether it is worth it** for this project. kpms fitting takes ~30 min
   on GPU for our data volume. The main diagnostic question is "did it converge?"
   which can be answered by the log-joint trace. If we are only running a handful
   of fits (kappa sweep with 3-5 values, maybe 3-5 seeds each), that is 9-25
   runs total. W&B adds value for comparing these runs on a single dashboard.
   For a single run, the built-in `plot_progress()` PDF may be sufficient.

---

## Sources

- [keypoint-MoSeq GitHub](https://github.com/dattalab/keypoint-moseq) — v0.6.8
- [jax-moseq GitHub](https://github.com/dattalab/jax-moseq) — v0.3.3
- [kpms fitting documentation](https://keypoint-moseq.readthedocs.io/en/latest/fitting.html)
- [kpms visualization documentation](https://keypoint-moseq.readthedocs.io/en/latest/viz.html)
- Weinreb et al. 2024. "Keypoint-MoSeq: parsing behavior by linking point tracking
  to pose dynamics." Nature Methods. doi:10.1038/s41592-024-02318-2
- [DeepLabCut wandb integration](https://pypi.org/project/deeplabcut/3.0.0rc4/) — listed as optional extra
