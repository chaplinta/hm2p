# SuperAnimal Fine-Tune — Implementation Design

_Status: design — not yet implemented._
_Last updated: 2026-04-30._
_Branch: `feat/sync-pipeline-diagnostics`._
_Spec: `docs/superanimal-fine-tune-plan-v2.md` (commit `04fb800`). The v2 plan is
authoritative for scientific decisions; this document translates it into a
module/file/test layout._

> Method: Ye S, Filippova A, Lauer J, Schneider S, Vidal M, Qiu T, Mathis A,
> Mathis MW. 2024. "SuperAnimal pretrained pose estimation models for
> behavioral analysis." *Nature Communications* 15:5165.
> doi:10.1038/s41467-024-48792-2.
> Code: https://github.com/DeepLabCut/DeepLabCut.
> Weights: https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse.

This design covers four deliverables:

1. A pure-compute helpers module `src/hm2p/pose/finetune.py` for the paired
   non-parametric statistics + verdict serialiser used by the comparison.
2. A new comparison CLI `scripts/compare_models.py` that runs the v2 §4.6
   champion-promotion gate against per-frame errors.
3. A `--sa-finetune` flag added to `scripts/run_dlc_retrain.py` (training side)
   and to `scripts/launch_dlc_finetune_ec2.py` (launcher passthrough).
4. Documentation, citation, and a frontend Methods & References expander.

It is deliberately additive. Mode A (the current ImageNet-only HRNet path)
remains the default. SA fine-tuning is opt-in via `--sa-finetune`. The
champion-promotion gate is run manually after training so the operator
chooses which model wins.

---

## 1. Module / file plan

| Status | File | Responsibility |
| --- | --- | --- |
| **new** | `src/hm2p/pose/finetune.py` | Pure-compute helpers (no I/O): paired Wilcoxon, rank-biserial *r*, bootstrap CI on median, circular HD-error helper, verdict assembly + JSON (de)serialiser, ear-vector HD computation reused from kinematics. |
| **new** | `scripts/compare_models.py` | CLI that loads two sets of per-frame predictions + GT, runs the §4.6 gate, and writes `verdict.json`. Calls into `finetune.py` for all statistics. |
| modify | `scripts/run_dlc_retrain.py` | Add `--sa-finetune` flag. When set, build `WeightInitialization` via `dlc.modelzoo.weight_initialization.build_weight_init(...)`, skip the manual backbone YAML rewrite, pass `weight_init=` and `net_type="hrnet_w32"` into `create_training_dataset`, and call `train_network` with the v2 §4.3 hyperparameters (Adam lr 5e-5, freeze BN running stats, step LR decay, ±30°/[0.7,1.3] augmentation). Also enforce that `default_net_type: hrnet_w32` is set in `config.yaml` before calling DLC. |
| modify | `scripts/launch_dlc_finetune_ec2.py` | Add `--sa-finetune` to argparse and propagate it through `build_user_data` into the `mode_flag` passed to `run_dlc_retrain.py`. Tag launches `mode="sa-finetune"` for cost records. Bump default disk to 120 GB (memory-replay extra optimiser state) but keep instance type `g4dn.xlarge` per v2 §5.3. |
| modify | `src/hm2p/pose/select.py` | One-line update: extend the architecture regex to recognise `Hrnetw32Sa` (DLC encodes the run name as the base architecture token, so the SA-finetuned shuffle's filename will read e.g. `..._DLC_HrnetW32_hm2p-retrain_..._snapshot-best-60.h5` — same `HrnetW32`, different `model_name` from the project file; no regex change needed in practice). The change is to **document** that the canonical champion-id format extends to `dlc-{YYYYMMDD}-hrnetw32-snap{N}` regardless of init source — the init source is captured in `notes`. |
| modify | `scripts/declare_dlc_champion.py` | No code change required. The auto-generated `notes` string written by `run_dlc_retrain.py` includes `init: superanimal_topviewmouse` (vs `init: imagenet`). The comparison verdict (verdict.json) is uploaded to `s3://hm2p-derivatives/dlc-retrain/models/_compare_verdict.json` and referenced from `notes`. |
| modify | `docs/dlc-retraining.md` | Replace the "no SuperAnimal equivalent for `head_midpoint`" claim with the corrected SA-TVM index 26 mapping. Add a "SuperAnimal fine-tuning" section pointing to the v2 plan + this design. |
| modify | `docs/dlc-champion-model.md` | Add a one-paragraph note that the champion gate (verdict.json) is consulted before manual promotion. No schema change. |
| modify | `frontend/pages/tracking_quality_page.py` | Add a "Methods & References" expander citing Ye 2024 and link to verdict.json (when present) per CLAUDE.md citation policy. |
| **new** | `tests/pose/test_finetune.py` | Unit + property tests on the pure-compute helpers. |
| **new** | `tests/scripts/test_compare_models.py` | CLI tests with synthetic two-model fixtures. |
| modify | `tests/scripts/test_declare_dlc_champion.py` | Add a parametric case verifying the `notes` field accepts the v2 `init=...` annotation without truncation. |
| modify | `tests/pose/test_select.py` | Parametric test: an SA-init filename and an ImageNet-init filename both yield the same `HrnetW32` architecture (the regex is init-source agnostic, by design). |
| modify | `tests/scripts/` (new file) `test_run_dlc_retrain.py` | Smoke tests on the `--sa-finetune` argparse plumbing — DLC itself is mocked. |
| modify | `tests/scripts/` (new file) `test_launch_dlc_finetune_ec2.py` | Smoke test that the launcher's user-data string contains `--sa-finetune` when the flag is set, and tags the cost record with `mode="sa-finetune"`. |

### 1.1 `src/hm2p/pose/finetune.py` — exported symbols

```python
"""SuperAnimal fine-tune comparison helpers (pure compute, no I/O).

Method: Ye et al. 2024. "SuperAnimal pretrained pose estimation models for
behavioral analysis." Nature Communications 15:5165.
doi:10.1038/s41467-024-48792-2.
Code: https://github.com/DeepLabCut/DeepLabCut.

These helpers implement the v2 §4.5 paired non-parametric comparison and
§4.6 promotion gate. All tests are non-parametric per CLAUDE.md.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Per-keypoint paired test
def paired_wilcoxon_per_keypoint(
    e_baseline: np.ndarray,   # (n_frames, n_keypoints) float64
    e_candidate: np.ndarray,  # same shape; NaN treated as missing
    *, alternative: str = "greater",
) -> np.ndarray:              # (n_keypoints,) p-values; NaN if too few pairs
    """Per-keypoint paired Wilcoxon signed-rank.

    H1 (alternative='greater'): baseline error > candidate error.
    Pairs with a NaN in either side are dropped. Returns NaN p-value when
    fewer than min_pairs (default 10) non-zero pairs are available.
    """

# Effect size — Kerby 2014 matched-pair rank-biserial
def rank_biserial_paired(e_baseline: np.ndarray, e_candidate: np.ndarray) -> float:
    """Matched-pair rank-biserial r. Range [-1, +1].

    r > 0 means candidate < baseline (candidate is better).
    Cite: Kerby DS. 2014. "The simple difference formula: an approach to
    teaching nonparametric correlation." Comprehensive Psychology 3:1.
    doi:10.2466/11.IT.3.1.
    """

# Bootstrap CI on the median of a 1-D array (percentile method, no parametric assumption)
def bootstrap_median_ci(
    x: np.ndarray, *, n_resamples: int = 10_000, ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:   # (median, ci_low, ci_high)

# Bonferroni correction
def bonferroni_alpha(alpha: float, n_tests: int) -> float:    # alpha / n_tests

# Per-frame Euclidean error helper
def per_frame_euclidean_error(
    pred_xy: np.ndarray,   # (n_frames, n_keypoints, 2) float64; NaN allowed
    gt_xy: np.ndarray,     # same shape
) -> np.ndarray:           # (n_frames, n_keypoints) float64; NaN where any input is NaN

# PCK at threshold
def pck_at(errors: np.ndarray, threshold_px: float) -> float

# Circular HD error from ear vector — paired comparison metric in v2 §4.5
def hd_from_ear_vector(
    left_ear: np.ndarray,   # (n_frames, 2)
    right_ear: np.ndarray,  # (n_frames, 2)
) -> np.ndarray:            # (n_frames,) radians, wrapped to (-pi, pi]
def circular_abs_error(theta_pred: np.ndarray, theta_gt: np.ndarray) -> np.ndarray

# Verdict dataclass + JSON IO
@dataclass(frozen=True)
class KeypointVerdict:
    keypoint: str
    n_pairs: int
    median_baseline_px: float
    median_candidate_px: float
    pct_change_median: float            # (median_baseline - median_candidate) / median_baseline
    p_value_wilcoxon: float
    rank_biserial_r: float
    bootstrap_ci_baseline: tuple[float, float, float]
    bootstrap_ci_candidate: tuple[float, float, float]
    pck_5_baseline: float
    pck_10_baseline: float
    pck_20_baseline: float
    pck_5_candidate: float
    pck_10_candidate: float
    pck_20_candidate: float
    p90_baseline: float
    p90_candidate: float
    pct_change_p90: float

@dataclass(frozen=True)
class GateConfig:
    alpha: float                              # 6.25e-3 (= 0.05 / 8) by default
    nose_required_pct_reduction: float        # 0.30
    tail_required_pct_reduction: float        # 0.40
    head_p90_required_pct_reduction: float    # 0.20
    rank_biserial_min: float                  # 0.30 for the gate; 0.10 floor (v2 §4.6 vs prompt — see §3.3)
    other_keypoint_max_regression_pct: float  # 0.10
    other_keypoint_regression_p_max: float    # 0.05

@dataclass(frozen=True)
class Verdict:
    schema_version: str                  # "1.0"
    baseline_id: str                     # e.g. "dlc-20260430-hrnetw32-snap110"
    candidate_id: str                    # e.g. "dlc-20260501-hrnetw32sa-snap60"
    n_frames_compared: int
    keypoints: list[KeypointVerdict]
    hd: dict                             # {"median_abs_error_baseline_rad": ..., "median_abs_error_candidate_rad": ..., "p_value_wilcoxon": ..., "rank_biserial_r": ..., "n_frames": ...}
    gate: GateConfig                     # echoed for traceability
    gate_pass_per_keypoint: dict[str, dict]  # see §3.1
    overall_pass: bool                   # all gate criteria met
    fail_reasons: list[str]              # short codes; empty when overall_pass
    generated_at: str                    # ISO 8601 UTC

def verdict_to_json(v: Verdict) -> str
def verdict_from_json(s: str) -> Verdict

# Top-level entry point used by compare_models.py
def evaluate_promotion_gate(
    e_baseline: np.ndarray,         # (n_frames, n_keypoints)
    e_candidate: np.ndarray,        # same
    keypoint_names: list[str],
    hd_baseline_rad: np.ndarray | None,
    hd_candidate_rad: np.ndarray | None,
    hd_gt_rad: np.ndarray | None,
    *, baseline_id: str, candidate_id: str, gate: GateConfig | None = None,
    rng: np.random.Generator | None = None,
) -> Verdict
```

`finetune.py` depends on numpy, scipy.stats. **No** boto3, h5py, pandas, or
DLC imports. The top-level entry point is the one function `compare_models.py`
calls — keeps the I/O layer thin.

### 1.2 `scripts/compare_models.py`

Inputs (one of two modes):

**Mode "rmse-json" (default):** read two pre-computed
`_bodypart_rmse.json` files. The existing `scripts/compute_bodypart_rmse.py`
already produces this format via `--pose-prefix pose-finetuned`; this is the
fast path because per-frame errors do not need to be recomputed. Limitation:
the JSON only stores per-bodypart aggregate stats, not per-frame errors, so
in this mode the script falls back to descriptive comparisons (median delta,
PCK delta) and **cannot** run Wilcoxon. This mode is for quick triage.

**Mode "predict" (full gate, the one the v2 spec requires):** the script
re-runs both models on the held-out test split and writes per-frame errors
to a temp file. This is the canonical mode and runs the full §4.5 paired
test. Inputs:

- `--baseline-h5-prefix s3://hm2p-derivatives/pose/` — keys with `*_DLC_*_*.h5`
  predictions from the baseline (snap-110); selected via the existing
  `select_best_dlc_h5_s3` helper.
- `--candidate-h5-prefix s3://hm2p-derivatives/pose-finetuned/` — same shape
  but for the SA-finetuned shuffle. Both prefixes contain one `.h5` per
  session under `{sub}/{ses}/`.
- `--labels-dir sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/labeled-data/`
  — local `CollectedData_*.h5` files.
- `--baseline-id dlc-20260430-hrnetw32-snap110` (champion id of baseline; for
  verdict provenance).
- `--candidate-id dlc-20260501-hrnetw32sa-snap60` (champion id of candidate).
- `--output verdict.json` (default `dlc-retrain/_compare_verdict.json`).
- `--alpha 0.00625` (Bonferroni-corrected per-keypoint α; default 0.05 / 8).
- `--upload-s3` (optional; uploads verdict to
  `s3://hm2p-derivatives/dlc-retrain/models/_compare_verdict.json`).
- `--seed 42` (RNG seed for the bootstrap; default fixed for reproducibility).

The script's body:

1. Walk `--labels-dir` for `CollectedData_*.h5` per session.
2. For each session, look up the matching prediction h5 via
   `select_best_dlc_h5_s3` under both prefixes. Drop sessions with no match
   in either prefix; record dropped sessions in `verdict.json` under
   `meta.skipped_sessions`.
3. Match GT frames to predictions by frame index (re-uses
   `compute_bodypart_rmse.py`'s `RAW_FPS=100`/`DLC_FPS=30` mapping —
   factored into a helper in `finetune.py` for reuse).
4. Build `(n_frames, n_keypoints, 2)` arrays for GT, baseline-pred,
   candidate-pred. Compute per-frame Euclidean errors using
   `per_frame_euclidean_error`.
5. Build the HD signal from `left_ear`, `right_ear` for each model and GT
   using `hd_from_ear_vector`. Compute paired absolute circular errors.
6. Call `evaluate_promotion_gate(...)` → `Verdict`.
7. Write `verdict.json` locally; upload to S3 if `--upload-s3`.
8. Print a one-page summary to stdout (exit 0 iff `overall_pass`; exit 2
   otherwise — the lead-dev's promotion script can chain on the exit code).

### 1.3 `scripts/run_dlc_retrain.py` — `--sa-finetune` wiring

The current `train()` function does the hand-rolled HRNet + ImageNet path
inline. The SA-finetune path is a separate code path called from `train()`
when `args.sa_finetune` is True. Concretely:

```python
def train(s3, *, epochs, batch_size, sa_finetune):
    ...
    if sa_finetune:
        config_path = _train_sa_finetune(s3, work, cfg, epochs, batch_size)
    else:
        config_path = _train_imagenet_hrnet(s3, work, cfg, epochs, batch_size)
    ...
```

`_train_sa_finetune` does only the things the SA path needs:

1. **Pre-condition checks** (fail loud, fail early — pitfall #1, #2, #7):
   - `default_net_type == "hrnet_w32"` in `config.yaml`. If not, set it
     in-place and print a warning.
   - `SuperAnimalConversionTables.superanimal_topviewmouse` includes all 8
     bodyparts identity-to-identity (project's current config already does
     this; just assert).
   - `dlclibrary.list_available_models()` lists
     `superanimal_topviewmouse_hrnet_w32`. If not, raise.
   - Probe `dlclibrary.list_available_detectors()` for both
     `fasterrcnn_resnet50_fpn_v2` and `fasterrcnn_resnet50_fpn`. Pass the
     first available; raise if neither is. (Pitfall #2 mitigation.)
   - Assert that the resolved SA snapshot's `data.train.input_size` matches
     a 256×256 crop after `make_super_animal_finetune_config` writes the
     pytorch_config.yaml; warn but do not fail if mismatched (pitfall #1
     mitigation — runtime assert with a printed warning).
2. **Build `WeightInitialization` via `build_weight_init(...)`** with the
   exact kwargs from v2 §4.2. Do **not** pass `load_head_weights=False` or
   `model.backbone.pretrained=True`.
3. **Call `create_training_dataset(weight_init=weight_init, num_shuffles=1,
   net_type="hrnet_w32")`** and capture the returned shuffle index.
4. **Apply only the augmentation patch** (v2 §4.3 right column) to the new
   shuffle's `pytorch_config.yaml`. Specifically: `affine.rotation=30`,
   `affine.scaling=[0.7, 1.3]`, `gaussian_noise=10`, brightness/contrast
   block unchanged from the existing IR-camera patch. Do **not** override
   `model.backbone` or head channels — `make_super_animal_finetune_config`
   writes those correctly.
5. **Call `deeplabcut.train_network`** with:
   ```python
   epochs=120,                                     # v2 §4.3
   save_epochs=10,
   displayiters=100,
   batch_size=8,
   pytorch_cfg_updates={
       "train_settings.optimizer.params.lr": 5e-5,
       "model.backbone.freeze_bn_stats": True,
       "train_settings.scheduler.type": "MultiStepLR",
       "train_settings.scheduler.params.milestones": [90, 110],
       "train_settings.scheduler.params.gamma": 0.1,
   }
   ```
   (Step decay at 90/110 is paper §"HRNet-w32" small-data; v2 §4.3.)
6. **Best-snapshot selection** is the same as the ImageNet path: `evaluate_network`
   followed by reading `evaluation-results/.../*.csv` for the best snap.
7. **Upload + champion declaration** unchanged. The auto-generated `notes`
   string for `declare_champion()` adds:
   ```
   init: superanimal_topviewmouse_hrnet_w32 (memory replay)
   conversion_array: [0,1,2,26,7,8,9,13]
   detector: <resolved name>
   epochs: 120; lr: 5e-5; bs: 8; freeze_bn_stats: True
   ```

`maxiters` becomes a no-op for the SA path (DLC PyTorch ignores it). Keep
the flag for back-compat; the help text now says `(legacy; ignored under
--sa-finetune)`.

### 1.4 `scripts/launch_dlc_finetune_ec2.py` — passthrough only

Single argparse addition:

```python
parser.add_argument("--sa-finetune", action="store_true",
                    help="Use SuperAnimal-TopViewMouse memory-replay fine-tune "
                         "instead of ImageNet HRNet. Adds ~20% wall-clock vs "
                         "the ImageNet path; cost ~USD 0.75 (AUD ~1.10).")
```

In `build_user_data`:

```python
if sa_finetune:
    mode_flag = mode_flag + " --sa-finetune"
    mode = mode + "+sa"   # cost record tag
```

No instance-type bump (v2 §5.3 keeps `g4dn.xlarge`). EBS root volume bumped
from 100 → 120 GB to absorb the SA snapshot download (~600 MB) plus the
extra training-dataset memory-replay pseudo-label cache. This is a
conservative bump, not strictly required.

`--infer-only` continues to work unchanged — it loads whichever model is on
S3 under `dlc-retrain/models/`, regardless of which path produced it.

### 1.5 `src/hm2p/pose/select.py` — no code change, but a comment

The architecture regex `re.search(r"DLC_(Hrnet[A-Za-z0-9]+|Resnet[0-9]+)_", ...)`
already matches the SA-fine-tuned filename (the architecture token is still
`HrnetW32`; the SA init source is captured separately in `notes`). Add a
docstring line under `extract_architecture` clarifying that the architecture
token is init-source agnostic. Add a parametric test (see §4) to lock that
behaviour.

### 1.6 `frontend/pages/tracking_quality_page.py` — Methods expander

Add a "Methods & References" expander in the existing per-keypoint section.
Per CLAUDE.md citation policy:

```python
with st.expander("Methods & References"):
    st.markdown("""
    Per-keypoint comparison uses paired Wilcoxon signed-rank tests on
    Euclidean errors, with rank-biserial *r* effect size (Kerby 2014) and
    Bonferroni correction across the 8 keypoints (α = 6.25e-3). Bootstrap
    CIs on the median (10 000 resamples, percentile method).

    SuperAnimal fine-tuning uses memory-replay transfer learning per:

    Ye S, Filippova A, Lauer J, Schneider S, Vidal M, Qiu T, Mathis A,
    Mathis MW. 2024. "SuperAnimal pretrained pose estimation models for
    behavioral analysis." *Nature Communications* 15:5165.
    doi:[10.1038/s41467-024-48792-2](https://doi.org/10.1038/s41467-024-48792-2).

    Pre-trained weights:
    [SuperAnimal-TopViewMouse on HuggingFace](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse).
    """)
```

When a `verdict.json` is available on S3 (under
`dlc-retrain/models/_compare_verdict.json`), the page also renders a small
table of the per-keypoint gate outcome. This is read-only display; the gate
itself runs locally via the CLI.

---

## 2. CLI surface

### 2.1 `scripts/run_dlc_retrain.py`

```
python scripts/run_dlc_retrain.py [--train-only|--infer-only]
                                  [--sa-finetune]
                                  [--epochs N]            (default 400 ImageNet, 120 SA)
                                  [--batch-size N]        (default 8)
                                  [--maxiters N]          (legacy; ignored)
                                  [--skip-failed]
```

- `--sa-finetune`: opt-in to the SuperAnimal memory-replay path. Mutually
  exclusive with the legacy ImageNet HRNet path; if set, the manual
  backbone YAML rewrite is skipped (the `make_super_animal_finetune_config`
  path writes the correct pytorch_config.yaml).
- `--epochs`: default depends on the path. If `--sa-finetune` and `--epochs`
  unset, the script uses 120. If unset without `--sa-finetune`, uses 400.
  When the user passes `--epochs` explicitly, that value is honoured for
  both paths.
- `--batch-size`: default 8 in both paths. Memory replay uses ~3× backward
  cost (pitfall #5), but at batch 8 on g4dn.xlarge there is still headroom.

### 2.2 `scripts/launch_dlc_finetune_ec2.py`

```
uv run python scripts/launch_dlc_finetune_ec2.py [--sa-finetune]
                                                 [--epochs N]
                                                 [--infer-only|--train-only]
                                                 [--dry-run]
                                                 [--status|--progress|--terminate]
```

- `--sa-finetune` is propagated to the user-data; it is **not** a
  client-side action.
- `--dry-run` continues to print the user-data without launching — useful
  to diff the pre-flag and post-flag user-data.

### 2.3 `scripts/compare_models.py`

```
uv run python scripts/compare_models.py
    --mode {predict|rmse-json}
    [--baseline-h5-prefix s3://hm2p-derivatives/pose/]
    [--candidate-h5-prefix s3://hm2p-derivatives/pose-finetuned/]
    [--baseline-rmse-json s3://hm2p-derivatives/dlc-retrain/models/_bodypart_rmse_baseline.json]
    [--candidate-rmse-json s3://hm2p-derivatives/dlc-retrain/models/_bodypart_rmse_candidate.json]
    --labels-dir sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/labeled-data/
    --baseline-id dlc-20260430-hrnetw32-snap110
    --candidate-id dlc-YYYYMMDD-hrnetw32-snap{N}
    [--output ./verdict.json]
    [--upload-s3]
    [--alpha 0.00625]
    [--seed 42]
```

Exit codes:
- `0` — overall_pass=True (all gate criteria met)
- `2` — overall_pass=False (gate failed; see verdict.json `fail_reasons`)
- `3` — comparison could not be performed (no overlapping sessions, missing
  labels, etc.); `verdict.json` still written with `meta.error`

### 2.4 `verdict.json` schema

Stable, versioned, written by `verdict_to_json` in `finetune.py`:

```json
{
  "schema_version": "1.0",
  "baseline_id": "dlc-20260430-hrnetw32-snap110",
  "candidate_id": "dlc-20260501-hrnetw32sa-snap60",
  "generated_at": "2026-05-01T12:34:56Z",
  "n_frames_compared": 71,
  "keypoints": [
    {
      "keypoint": "nose_tip",
      "n_pairs": 71,
      "median_baseline_px": 24.1,
      "median_candidate_px": 9.6,
      "pct_change_median": 0.602,
      "p_value_wilcoxon": 1.7e-9,
      "rank_biserial_r": 0.78,
      "bootstrap_ci_baseline": [24.1, 18.4, 31.2],
      "bootstrap_ci_candidate": [9.6, 7.3, 12.1],
      "pck_5_baseline": 0.06, "pck_10_baseline": 0.17, "pck_20_baseline": 0.44,
      "pck_5_candidate": 0.31, "pck_10_candidate": 0.62, "pck_20_candidate": 0.89,
      "p90_baseline": 78.2, "p90_candidate": 25.1, "pct_change_p90": 0.679
    },
    ...
  ],
  "hd": {
    "median_abs_error_baseline_rad": 0.21,
    "median_abs_error_candidate_rad": 0.12,
    "p_value_wilcoxon": 4.4e-7,
    "rank_biserial_r": 0.43,
    "n_frames": 71
  },
  "gate": {
    "alpha": 6.25e-3,
    "nose_required_pct_reduction": 0.30,
    "tail_required_pct_reduction": 0.40,
    "head_p90_required_pct_reduction": 0.20,
    "rank_biserial_min": 0.30,
    "other_keypoint_max_regression_pct": 0.10,
    "other_keypoint_regression_p_max": 0.05
  },
  "gate_pass_per_keypoint": {
    "nose_tip":      {"pass": true,  "checks": {"pct_reduction": true, "p_value": true, "rank_biserial": true}},
    "left_ear":      {"pass": true,  "checks": {"no_regression": true}},
    ...
    "tail_base":     {"pass": true,  "checks": {"pct_reduction": true, "p_value": true, "rank_biserial": true}},
    "head_midpoint": {"pass": true,  "checks": {"p90_reduction": true}}
  },
  "overall_pass": true,
  "fail_reasons": [],
  "meta": {
    "skipped_sessions": [],
    "labels_dir": "sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/labeled-data/",
    "baseline_h5_prefix": "...",
    "candidate_h5_prefix": "...",
    "rng_seed": 42
  }
}
```

`meta` is the only field outside `Verdict`'s strict dataclass; it carries
context that is useful to humans but does not gate the verdict.

---

## 3. Champion-promotion gate

### 3.1 Gate logic — exact predicates

Per v2 §4.6, the gate is a conjunction of six checks. The gate config
(`GateConfig`) carries the thresholds; defaults below match v2.

| Check | Predicate (per keypoint) | Default |
| --- | --- | --- |
| `nose_pct_reduction` | `pct_change_median >= 0.30` for `nose_tip` | 0.30 |
| `nose_significance` | `p_value_wilcoxon < 6.25e-3` AND `rank_biserial_r >= 0.30` for `nose_tip` | α/8, r ≥ 0.30 |
| `tail_pct_reduction` | `pct_change_median >= 0.40` for `tail_base` | 0.40 |
| `tail_significance` | as nose_significance, for `tail_base` | α/8, r ≥ 0.30 |
| `head_p90_reduction` | `pct_change_p90 >= 0.20` for `head_midpoint` | 0.20 |
| `no_regression` | for every other keypoint: `pct_change_median > -0.10` AND (p_value > 0.05 OR rank_biserial_r > 0) | 10 % regression band, p > 0.05 |

The HD circular-error check is **descriptive only** per v2 §4.5 — it is
recorded in the verdict but does not gate. The visual-QC step (v2 §4.6 #6)
is operator-driven and not gated by this script.

The prompt asks for `r > 0.1` as a floor for the gate; v2 §4.6 says
`r > 0.30` (medium-or-larger). I have followed v2's number. The `0.1`
threshold from the prompt is treated as a **secondary diagnostic**: the
verdict reports `rank_biserial_r > 0.1` as a separate boolean per
keypoint for any keypoint whose absolute error decreases on average. This
catches the "large but very-low-effect-size" case the prompt is worried
about. I do not gate on it because gating on two effect-size thresholds
duplicates v2's single-threshold design.

### 3.2 Manual-promotion flow

The gate is intentionally **decoupled** from automatic promotion.
`run_dlc_retrain.py` does *not* call `compare_models.py`. The reason
(consistent with v1's `dlc-champion-model.md` §4.1): champion declaration
remains the responsibility of `declare_dlc_champion.py`, and that script
already runs at the end of every retrain — including SA-finetune retrains.
The SA-finetuned shuffle becomes the champion the moment training
completes, regardless of the gate.

**This is intentional.** It keeps the EC2-side pipeline simple. The gate is
the operator's tool for deciding whether to *keep* the new champion or to
roll back. Concretely:

```bash
# After the SA-finetune EC2 run completes (auto-declared as champion):
uv run python scripts/compare_models.py \
    --mode predict \
    --baseline-h5-prefix s3://hm2p-derivatives/pose-archive/dlc-20260430-hrnetw32-snap110/ \
    --candidate-h5-prefix s3://hm2p-derivatives/pose/ \
    --labels-dir sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/labeled-data/ \
    --baseline-id dlc-20260430-hrnetw32-snap110 \
    --candidate-id dlc-20260501-hrnetw32sa-snap60 \
    --output ./verdict.json --upload-s3

# If overall_pass=true: keep the new champion; nothing else to do.
# If overall_pass=false: roll back manually by re-running declare_dlc_champion
# with the previous champion's identifiers (the prior manifest is archived in
# dlc-champion-history/{old_champion_id}.json):
uv run python scripts/declare_dlc_champion.py \
    --model-name <old> --architecture HrnetW32 --snapshot 110 \
    --training-run-id <old run_id> \
    --note "Rolled back; SA-finetune failed promotion gate; verdict at ..."
```

Two operational details:

1. **Archive of baseline predictions.** Before the SA-finetune EC2 run
   overwrites `pose/{sub}/{ses}/*.h5`, the operator should run a one-shot
   archive copy:
   ```bash
   aws s3 sync s3://hm2p-derivatives/pose/ \
              s3://hm2p-derivatives/pose-archive/dlc-20260430-hrnetw32-snap110/
   ```
   The architect would prefer this be automated as a step of
   `run_dlc_retrain.py` (see Open Question 4). For now it stays operator-
   driven, with a check in `compare_models.py` that prints a helpful error
   if the baseline prefix is empty.

2. **Streamlit display.** `tracking_quality_page.py` reads the verdict
   when present and renders a simple per-keypoint table with green/red
   chips. Adding a "Re-run gate" button is in scope for the optional commit
   in §5.

### 3.3 Staleness contract — unchanged

After SA-finetune is auto-declared as champion, every kinematics.h5 / sync.h5
/ analysis.h5 / rendered video produced from the previous champion is
stale. The existing `is_session_current()` machinery flags this. The
operator runs `scripts/run_downstream_pipeline.py` to re-build downstream
derivatives, exactly as documented in `docs/dlc-champion-model.md` §4.1.

If the operator decides to roll back (gate failed), `declare_dlc_champion.py`
restores the old champion id; downstream derivatives that were not yet
re-built remain valid, and any that were re-built become stale and must be
re-built again. Rolling back has a cost. Argument for accepting it: the
sample size for promotion decisions is *one* (the operator looks at the
verdict once and decides), and adding an automatic rollback would
encourage non-deterministic post-hoc gate tweaking.

---

## 4. Validation strategy (test-engineer-actionable)

All tests use small synthetic numpy arrays. None reads real session data.
Coverage target: ≥ 90 % per-line on `finetune.py` and the comparison script.

### 4.1 `tests/pose/test_finetune.py` — pure-compute helpers

| Group | Tests |
| --- | --- |
| `paired_wilcoxon_per_keypoint` | (a) baseline > candidate everywhere → all p < 0.001; (b) baseline ≈ candidate → p ≥ 0.5; (c) baseline < candidate → p ≥ 0.5 with `alternative="greater"`; (d) NaN handling — pairs with NaN dropped, returns NaN p when n_pairs < min_pairs; (e) hypothesis property: arrays where baseline = candidate + ε for ε > 0 always pass at α 0.001 with n ≥ 50; (f) shape validation. |
| `rank_biserial_paired` | (a) hand-computed example matching Kerby 2014 worked example; (b) range bound [-1, 1]; (c) `r > 0` iff candidate < baseline on average; (d) hypothesis property: invariant under monotone transform of inputs (rank-based). |
| `bootstrap_median_ci` | (a) deterministic with seeded RNG; (b) CI brackets the true median for known distributions (bootstrap is NOT exact, so check coverage at 95 % over 200 trials with hypothesis); (c) `n_resamples=1` returns degenerate CI; (d) NaN-only input raises. |
| `bonferroni_alpha` | trivial. |
| `per_frame_euclidean_error` | (a) zero error on identical inputs; (b) NaN propagates only to the affected frame/keypoint; (c) shape mismatch raises. |
| `pck_at` | (a) all-zero errors → 1.0; (b) all errors above threshold → 0.0; (c) NaN ignored. |
| `hd_from_ear_vector` | (a) both ears at same y → angle 0; (b) right ear above left → angle π/2 (consistent with the project's HD convention — verify against `hm2p.kinematics.compute` if it exists, see §7); (c) returns wrapped to (-π, π]. |
| `circular_abs_error` | (a) zero on identical; (b) wraps |Δ| at π. |
| `Verdict` round-trip | `verdict_from_json(verdict_to_json(v)) == v`; schema_version is preserved; missing optional fields raise on parse. |
| `evaluate_promotion_gate` | (a) synthetic two-model fixture where candidate is uniformly better → `overall_pass=True`; (b) candidate worse on `mid_back` → `overall_pass=False`, `fail_reasons=["regression_mid_back"]`; (c) candidate better on nose only → fails the tail_pct_reduction; (d) HD field always populated when both HD inputs are non-None; (e) gate echoed verbatim. |

Use `numpy.random.default_rng(42)` for any randomness in fixtures so tests
are deterministic.

### 4.2 `tests/scripts/test_compare_models.py`

| Group | Tests |
| --- | --- |
| Argparse | required args present; mutually-exclusive modes; default alpha = 0.00625. |
| Mode `predict` end-to-end | mock `select_best_dlc_h5_s3` and h5 reads; load synthetic GT + two prediction sets from temp dir; assert `verdict.json` is written with expected schema_version, baseline_id, candidate_id, n_frames_compared. |
| Exit codes | passes → 0; fails → 2; no overlap → 3. |
| `--upload-s3` | with mock S3 client (botocore stub), assert `put_object` is called with the right key and content. |
| Fixture: clear winner | construct a 50-frame test set where candidate beats baseline by 60 % on nose/tail, ties on others → `overall_pass=True`. |
| Fixture: clear loser | candidate worse by 20 % on nose → `overall_pass=False`. |
| `meta.skipped_sessions` | sessions present in labels but missing from one prefix are listed and not aborted. |

The test suite uses the `_FakeS3` pattern from
`tests/scripts/test_declare_dlc_champion.py` for symmetry.

### 4.3 `tests/scripts/test_run_dlc_retrain.py`

DLC is heavy; mock it. Tests focus on the argparse plumbing and the
config-rewrite logic.

| Test | Setup | Assert |
| --- | --- | --- |
| `--sa-finetune` parses | argparse only | `args.sa_finetune is True` |
| `--sa-finetune` default epochs | argparse + branch | epochs=120 when `--sa-finetune` and `--epochs` unset |
| ImageNet default epochs | argparse only | epochs=400 when neither flag set |
| `_train_sa_finetune` calls `build_weight_init` | mock `deeplabcut.modelzoo.weight_initialization.build_weight_init` | called with `super_animal="superanimal_topviewmouse"`, `model_name="hrnet_w32"`, `with_decoder=True`, `memory_replay=True` |
| `_train_sa_finetune` skips manual backbone rewrite | mock + temp pytorch_config.yaml | the SA path does NOT touch `model.backbone` keys |
| `_train_sa_finetune` applies augmentation patch | temp pytorch_config.yaml | `data.train.affine.rotation == 30`, `affine.scaling == [0.7, 1.3]`, `gaussian_noise == 10` |
| Pre-condition assert: missing conversion table | temp config.yaml without conversion entry for one bodypart | raises `ValueError` |
| Pre-condition assert: `default_net_type` rewrite | temp config.yaml with `default_net_type: resnet_50` and `--sa-finetune` set | the script writes `hrnet_w32` back to disk |
| Detector resolution | mock `dlclibrary.list_available_detectors()` returning only `fasterrcnn_resnet50_fpn` (not `_v2`) | the script picks `fasterrcnn_resnet50_fpn` and proceeds |
| Detector resolution failure | mock `dlclibrary.list_available_detectors()` returning neither | raises with a clear error |

### 4.4 `tests/scripts/test_launch_dlc_finetune_ec2.py`

Pure user-data string assembly tests — no AWS calls.

| Test | Assert |
| --- | --- |
| `--sa-finetune` propagates | `build_user_data(sa_finetune=True)` contains `--sa-finetune` in the `python3 scripts/run_dlc_retrain.py` line |
| Cost record tag | `mode` field includes `+sa` when `sa_finetune=True` |
| Disk size | `BlockDeviceMappings[0].VolumeSize == 120` (or unchanged if we keep 100; design choice is 120 — see §6) |
| Default arg | `sa_finetune=False` produces user-data identical to current main, modulo the EBS bump |
| Mutual-exclusion safety | no error from passing `--sa-finetune` with `--infer-only` (this is a valid combination — re-running inference of an SA-finetuned model) |

### 4.5 `tests/pose/test_select.py` — additions

Two parametric test cases:

```python
@pytest.mark.parametrize("filename, expected_arch", [
    ("video_DLC_HrnetW32_hm2p-retrain_2026-03-20_shuffle1_snapshot-best-110.h5", "HrnetW32"),
    ("video_DLC_HrnetW32_hm2p-retrain_2026-03-20_shuffle2_snapshot-best-60.h5",  "HrnetW32"),
    ("video_DLC_Resnet50_hm2p-retrain_shuffle1_snapshot-50000.h5",               "Resnet50"),
])
def test_extract_architecture_init_source_agnostic(filename, expected_arch):
    assert extract_architecture(filename) == expected_arch
```

This locks the design decision that init source (ImageNet vs SuperAnimal)
is captured in `notes`, not in the architecture token.

### 4.6 End-to-end "would-this-promote-correctly" test

In `tests/pose/test_finetune.py`:

```python
def test_evaluate_promotion_gate_clear_winner():
    rng = np.random.default_rng(0)
    n = 200
    keypoint_names = ["nose_tip", "left_ear", "right_ear", "head_midpoint",
                      "neck", "mid_back", "mouse_center", "tail_base"]
    # baseline: nose ~24 px, tail ~59 px, head_midpoint long-tailed; rest ~5 px
    # candidate: 60 % reduction on nose/tail, 30 % p90 reduction on head_midpoint,
    #            10 % improvement on rest (within no-regression band)
    e_baseline = _build_synthetic_errors(rng, ...)
    e_candidate = _build_synthetic_errors(rng, ...)
    hd_b = rng.normal(0, 0.3, n);  hd_c = rng.normal(0, 0.15, n);  hd_g = np.zeros(n)
    v = evaluate_promotion_gate(
        e_baseline, e_candidate, keypoint_names, hd_b, hd_c, hd_g,
        baseline_id="b", candidate_id="c", rng=rng,
    )
    assert v.overall_pass is True
    assert v.fail_reasons == []
```

A second test (`test_evaluate_promotion_gate_nose_regresses`) constructs
candidate errors that are 20 % *worse* on nose; asserts
`overall_pass=False` and `"regression_nose_tip"` in `fail_reasons`.

---

## 5. Rollout plan (commit sequence)

Six commits, each test-clean, each independently reviewable. All on
`feat/sync-pipeline-diagnostics`. PR opens after commit 5; the optional
commit 6 lands on the same branch if there is time.

| # | Title | Files |
| --- | --- | --- |
| 1 | `feat: pose.finetune compute helpers + tests` | `src/hm2p/pose/finetune.py`, `tests/pose/test_finetune.py` |
| 2 | `feat: scripts/compare_models.py + tests` | `scripts/compare_models.py`, `tests/scripts/test_compare_models.py` |
| 3 | `feat: run_dlc_retrain.py --sa-finetune wiring + tests` | `scripts/run_dlc_retrain.py`, `tests/scripts/test_run_dlc_retrain.py`, `tests/pose/test_select.py` (parametric additions) |
| 4 | `feat: launch_dlc_finetune_ec2.py --sa-finetune passthrough + tests` | `scripts/launch_dlc_finetune_ec2.py`, `tests/scripts/test_launch_dlc_finetune_ec2.py` |
| 5 | `docs: SuperAnimal fine-tune retraining + champion notes + Methods expander` | `docs/dlc-retraining.md`, `docs/dlc-champion-model.md`, `frontend/pages/tracking_quality_page.py` |
| 6 (opt.) | `feat: tracking-quality page consumes verdict.json` | `frontend/pages/tracking_quality_page.py`, `frontend/data.py` (cached loader for verdict.json) |

Commit 1 is the foundation — every later commit imports from it. Commits
2–4 can be reordered freely (no inter-dependencies). Commit 5 is the
documentation pass that satisfies CLAUDE.md citation policy (3 places:
code docstrings already in commits 1+2, docs in 5, frontend in 5). Commit
6 is optional; it adds the in-page verdict viewer and is purely additive.

Do not bundle commits 3 and 4 — the retrain logic and launcher passthrough
should move independently so `--sa-finetune` can land on EC2 first
(commit 4) for testing while commit 3 is still in review (the local-only
training case stays unchanged either way).

---

## 6. Pitfall mitigations

Each numbered against v2 §5.

**Pitfall 1 — 256×256 image-resolution mismatch.**
*Resolution: runtime assert in `_train_sa_finetune`.* After
`create_training_dataset` writes pytorch_config.yaml, parse it and check
`data.train.input_size`. If it is not 256×256, log a warning and write the
discrepancy to `_retrain_progress.json`. Do **not** abort — DLC may pick a
different size for newer SA snapshot versions and the training would
still converge; the promotion gate will catch any regression.

**Pitfall 2 — detector name unresolved.**
*Resolution: explicit Step-0 probe in `_train_sa_finetune`.* The script
calls `dlclibrary.list_available_detectors()` and selects the first match
from `["fasterrcnn_resnet50_fpn_v2", "fasterrcnn_resnet50_fpn"]`. If
neither is present, it raises with the full list of available detectors so
the operator can update the candidate list. Tested in §4.3.

**Pitfall 3 — SA detector multi-animal-trained (may return zero bboxes
on dark hm2p frames).**
*Resolution: pre-flight inference probe.* Add a step in
`_train_sa_finetune` that, before launching the full retrain, runs the SA
zero-shot detector on 5–10 random frames from one dark-condition session
and asserts that ≥ 90 % of frames return at least one bbox. The frames are
sampled from the same labeled-data set used for training. If the assertion
fails, raise with a clear message: "SA detector returned 0 bboxes on N/M
test frames. Re-train just the detector before launching SA fine-tune."
This is implemented as a small helper in `finetune.py`
(`probe_sa_detector_bbox_rate(...)`) so it is unit-testable without AWS.

**Pitfall 4 — 80/20 vs 95/5 split absolute-RMSE comparison.**
*Resolution: documentation only.* The verdict JSON's docstring and the
`docs/superanimal-fine-tune-design.md` (this file) explicitly state: do
not compare absolute RMSE to paper Tables S3/S4. The gate is built on
**relative gain on the hm2p test set** — paired Wilcoxon on per-frame
errors — and is unaffected by the split fraction.

**Pitfall 5 — 3× backward cost of memory replay.**
*Resolution: accept and tune `batch_size`.* g4dn.xlarge has 16 GB VRAM. At
`batch_size=8` with HRNet-W32 + memory replay, peak VRAM is ~10 GB
(empirical from Ye 2024 benchmarks; conservative). No instance-type bump.
EBS root volume bumped to 120 GB to absorb extra training-dataset cache.
Wall-clock 50–60 min vs 90 min for ImageNet 400-epoch (cheaper per run
despite higher per-iteration cost). Cost ~USD 0.50 (~AUD 0.75) at spot
prices.

**Pitfall 6 — stale DLC example script `testscript_superanimal_transfer_learning.py`.**
*Resolution: documentation only.* The retraining doc explicitly says do
not copy from `examples/testscript_superanimal_transfer_learning.py` —
its `superanimal_name=` / `superanimal_transfer_learning=` kwargs were
removed pre-3.0. The correct API is `build_weight_init(...)` →
`create_training_dataset(weight_init=...)` → `train_network(...)`.

**Pitfall 7 — DLC issue #2742 (`video_adapt=True` folder-path failure).**
*Resolution: documentation only — this design does not call
`video_inference_superanimal(..., video_adapt=True)`.* Inference uses
`deeplabcut.analyze_videos` on individual file paths exactly as the
current pipeline does.

---

## 7. Open questions

1. **Architect-resolved: gate effect-size threshold.**
   v2 §4.6 says `r > 0.30`. The prompt says `r > 0.1`. I have followed v2
   for the gate decision and added `r > 0.1` as a *secondary diagnostic*
   in the verdict (not used for gating). If the user disagrees, change
   `GateConfig.rank_biserial_min` from 0.30 to 0.10 — one constant, no
   code restructuring.

2. **Architect-resolved: gate-as-CI-step vs operator-driven.**
   `compare_models.py` is operator-driven, run locally after the EC2
   retrain self-terminates. It is **not** invoked from `run_dlc_retrain.py`
   on EC2. Rationale: the EC2 instance lacks the labelled-data directory
   without an extra `aws s3 sync sourcedata/`, and the gate decision is
   sample-size-of-one anyway. If we ever automate the gate, the right
   place is a separate `--gate` flag on `declare_dlc_champion.py` that
   reads verdict.json from S3 and rolls back if `overall_pass=False`. Out
   of scope here.

3. **Architect-resolved: detector name fallback order.**
   `["fasterrcnn_resnet50_fpn_v2", "fasterrcnn_resnet50_fpn"]` per v2
   §4.2. `_v2` first because it is the documented default in DLC ≥ 3.0
   docs.

4. **Architect-resolved: archive-baseline-predictions step.**
   The architect's preferred design is for `run_dlc_retrain.py` to
   `aws s3 sync` the current `pose/` to
   `pose-archive/{baseline_champion_id}/` *before* overwriting with new
   inference results. This is a one-liner addition (~5 lines) gated on
   the existence of a current champion manifest. **Recommendation:**
   include it as a follow-up commit *after* commit 4 (so the gate flow is
   self-contained without operator-driven archival). It is not in the
   six-commit rollout above to keep the SA-finetune work focused. Ask the
   user before adding.

5. **Open — needs user input: instance-type budget.**
   v2 §5.3 says `g4dn.xlarge` is sufficient. I have followed that. If the
   memory-replay backward pass exceeds 16 GB on real data (untestable
   without a real run), the fallback is `g4dn.2xlarge` (32 GB VRAM, ~2×
   cost). The launcher's instance type is hardcoded as a module-level
   constant (`INSTANCE_TYPE = "g4dn.xlarge"`); if a fallback is needed
   the lead-dev should add `--instance-type` to the launcher's argparse
   in a follow-up commit. The first SA-finetune run will reveal whether
   this is needed.

6. **Open — needs user input: pre-render two labelled videos for QC?**
   v2 §7.3 leaves this to the architect. **Recommendation:** yes, render
   one held-out dark-condition session under both models before
   declaring the gate "passed". The existing `render_dlc_videos.py`
   already supports `--session` filtering; the operator runs it once
   pre-promotion and views the side-by-side in `dlc_viewer_page.py`.
   Add a one-paragraph note in `docs/dlc-retraining.md` to that effect.
   No code change.

---

## 8. Citations

Per CLAUDE.md citation policy — three places.

**1. Code (in `src/hm2p/pose/finetune.py` module docstring + the SA-finetune
branch in `run_dlc_retrain.py`):**

```python
"""SuperAnimal-TopViewMouse memory-replay fine-tuning.

Method: Ye et al. 2024. "SuperAnimal pretrained pose estimation models for
behavioral analysis." Nature Communications 15:5165.
doi:10.1038/s41467-024-48792-2.
Code: https://github.com/DeepLabCut/DeepLabCut.
Weights: https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse.

Memory-replay protocol: Ye 2024 Methods §"Memory replay fine tuning" + Fig. 1d.
Channel slicing: HeatmapHead.convert_weights in
deeplabcut/pose_estimation_pytorch/models/heads/simple_head.py.
"""
```

The rank-biserial helper carries Kerby 2014's citation:

```python
"""
Cite: Kerby DS. 2014. "The simple difference formula: an approach to
teaching nonparametric correlation." Comprehensive Psychology 3:1.
doi:10.2466/11.IT.3.1.
"""
```

**2. Docs:** updates to `docs/dlc-retraining.md` (a new "SuperAnimal
fine-tuning" section linking to v2 plan + this design), and a paragraph in
`docs/dlc-champion-model.md` noting the gate decision sequence.

**3. Frontend:** Methods & References expander in
`frontend/pages/tracking_quality_page.py` (text in §1.6 above).

---

## 9. What this design does not do

- It does not refactor the existing ImageNet HRNet path. Mode A continues
  to work; SA fine-tuning is opt-in.
- It does not change the champion manifest schema. Init source
  (ImageNet vs SA) is captured in the existing `notes` field — adding a
  dedicated field would be cosmetic and would force a manifest version
  bump.
- It does not implement automatic rollback. Rollback is operator-driven
  via `declare_dlc_champion.py` — same pattern as the existing manual
  re-declaration path.
- It does not add a new pipeline stage. SA fine-tuning is a parallel
  shuffle within Stage 2a (DLC Training). The Snakemake DAG is unchanged.
- It does not run the gate inside the EC2 user-data. The gate is
  operator-driven via the local `compare_models.py` CLI after the EC2
  run completes (Open Question 2).

---
